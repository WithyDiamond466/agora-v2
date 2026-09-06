"""Skill CRUD, knowledge documents, and ``.agoraskill`` bundle export/import.

A Skill is the unit of sharing in Agora (SPEC principle 3): a system prompt
(grading voice + standards), provider/model settings, and knowledge documents
(syllabus, rubric conventions, worked examples).

Bundle format (``.agoraskill`` — a plain zip)::

    manifest.json      {format_version, name, description, provider, model, settings}
    system_prompt.md   the raw system prompt
    docs/<file>        knowledge documents, flat, sanitized filenames

Security posture for import: the manifest is validated, and **no path inside the
zip is ever trusted**. Member names are checked (no absolute paths, no ``..``,
no symlinks, no nested dirs under ``docs/``) *and* every resolved destination is
re-checked to be inside the target directory before a byte is written (zip-slip
defence, belt and braces).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Optional, Union

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app import config
from app.models import KnowledgeDoc, Skill
from app.storage import ReversibleDelete, discard_all, reraise_delete_failure, reversible_delete

log = logging.getLogger("agora.skills")

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

#: Bundle format version written into manifest.json. Importers accept <= this.
FORMAT_VERSION = 1

BUNDLE_SUFFIX = ".agoraskill"
MANIFEST_NAME = "manifest.json"
SYSTEM_PROMPT_NAME = "system_prompt.md"
DOCS_PREFIX = "docs/"

#: Knowledge docs are small by design (SPEC: no vector DB, docs attach directly
#: to requests), so the cap is deliberately tight.
MAX_DOC_BYTES = 10 * 1024 * 1024
#: Total uncompressed payload accepted from a bundle (zip-bomb guard).
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_DOCS_PER_SKILL = 50

#: extension -> canonical mime type. Extension is authoritative; the browser's
#: content-type header is advisory only.
#:
#: Only types the grading engine can actually put in a request are accepted:
#: text/markdown are inlined into the system prompt and PDFs ride along as
#: document blocks. ``.docx`` was accepted here but silently dropped from every
#: grading request (no extraction path, python-docx is not a dependency), so a
#: professor could attach a rubric-conventions doc that was never used. It is
#: rejected at upload instead, matching the file picker in skill_detail.html.
ALLOWED_DOC_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

#: Text-ish docs whose contents can be inlined into a prompt.
INLINE_TEXT_EXTENSIONS = {".txt", ".md"}

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._ -]+")
_MULTI_DOT_RE = re.compile(r"\.{2,}")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

MAX_FILENAME_LEN = 120


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class SkillError(Exception):
    """Base class — routers map these to HTTP 400."""


class SkillNotFound(SkillError):
    """No such skill / knowledge doc."""


class UnsupportedDocument(SkillError):
    """Wrong file type or too large."""


class BundleError(SkillError):
    """Malformed, unsafe, or unsupported ``.agoraskill`` bundle."""


# --------------------------------------------------------------------------
# paths & filenames
# --------------------------------------------------------------------------


def skills_root() -> Path:
    """Root for knowledge-doc storage: ``data/skills/``.

    Resolved lazily so tests (and ``AGORA_DATA_DIR``) can relocate DATA_DIR.
    """
    return Path(config.DATA_DIR) / "skills"


def skill_dir(skill_id: int) -> Path:
    """Per-skill document directory: ``data/skills/{skill_id}/``."""
    return skills_root() / str(int(skill_id))


def _ensure_skill_dir(skill_id: int) -> Path:
    path = skill_dir(skill_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_root() -> Path:
    path = Path(config.EXPORT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str | None, default: str = "document") -> str:
    """Reduce arbitrary user/zip input to a safe, flat filename.

    Strips every directory component (both separators), unicode-normalizes,
    drops anything outside ``[A-Za-z0-9._- ]``, kills leading dots and ``..``
    runs, and truncates while preserving the extension.
    """
    raw = (name or "").replace("\\", "/")
    raw = raw.split("/")[-1]
    raw = unicodedata.normalize("NFKD", raw)
    raw = raw.replace("\x00", "")
    raw = _SAFE_CHARS_RE.sub("_", raw).strip()
    raw = _MULTI_DOT_RE.sub(".", raw).lstrip(". ")
    if not raw or raw in {".", ".."}:
        return default

    stem, dot, ext = raw.rpartition(".")
    if not dot:
        stem, ext = raw, ""
    stem = stem.strip() or default
    if ext:
        ext = ext[:12]
        keep = max(1, MAX_FILENAME_LEN - len(ext) - 1)
        return f"{stem[:keep]}.{ext}"
    return stem[:MAX_FILENAME_LEN]


def _slugify(value: str, default: str = "skill") -> str:
    slug = _SLUG_RE.sub("-", (value or "").lower()).strip("-")
    return (slug or default)[:60]


def _unique_path(directory: Path, filename: str) -> Path:
    """Never clobber an existing doc: ``notes.pdf`` -> ``notes-1.pdf``."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        stem, ext = filename, ""
    for n in range(1, 1000):
        name = f"{stem}-{n}.{ext}" if ext else f"{stem}-{n}"
        candidate = directory / name
        if not candidate.exists():
            return candidate
    raise SkillError("Could not find a free filename for the upload")


def _check_doc(filename: str, data: bytes) -> tuple[str, str]:
    """Validate an incoming document. Returns (safe_filename, mime_type)."""
    safe = sanitize_filename(filename)
    ext = os.path.splitext(safe)[1].lower()
    if ext not in ALLOWED_DOC_TYPES:
        raise UnsupportedDocument(
            f"Unsupported document type {ext or '(none)'!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_DOC_TYPES))}"
        )
    if not data:
        raise UnsupportedDocument(f"{safe} is empty")
    if len(data) > MAX_DOC_BYTES:
        raise UnsupportedDocument(
            f"{safe} is {len(data) // 1024} KB; the limit is {MAX_DOC_BYTES // (1024 * 1024)} MB"
        )
    return safe, ALLOWED_DOC_TYPES[ext]


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------


def doc_dict(doc: KnowledgeDoc) -> dict[str, Any]:
    return {
        "id": doc.id,
        "skill_id": doc.skill_id,
        "title": doc.title,
        "filename": doc.filename,
        "mime_type": doc.mime_type,
        "size_bytes": doc.size_bytes,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


def skill_dict(skill: Skill, *, with_docs: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "system_prompt": skill.system_prompt,
        "provider": skill.provider,
        "model": skill.model,
        "max_tokens": skill.max_tokens,
        "mode": skill.mode or "grade",
        "mode_label": _mode_label(skill.mode),
        "created_at": skill.created_at.isoformat() if skill.created_at else None,
        "updated_at": skill.updated_at.isoformat() if skill.updated_at else None,
    }
    if with_docs:
        data["docs"] = [doc_dict(d) for d in skill.knowledge_docs]
        data["doc_count"] = len(data["docs"])
    return data


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------


def list_skills(db: Session) -> list[Skill]:
    return list(
        db.scalars(
            select(Skill)
            .options(selectinload(Skill.knowledge_docs))
            .order_by(Skill.name, Skill.id)
        ).all()
    )


def get_skill(db: Session, skill_id: int) -> Skill:
    skill = db.get(Skill, skill_id)
    if skill is None:
        raise SkillNotFound(f"Skill {skill_id} not found")
    return skill


def _clean_provider(provider: str | None) -> str:
    provider = (provider or "").strip().lower()
    if provider in config.PROVIDERS or provider in (config.MOCK_PROVIDER, config.AUTO_PROVIDER):
        return provider
    return config.DEFAULT_SKILL_PROVIDER


#: Public alias — routers normalize provider names through this.
normalize_provider = _clean_provider


def _clean_model(provider: str, model: str | None) -> str:
    model = (model or "").strip()
    if provider == config.AUTO_PROVIDER:
        # The model is chosen at grade time from the resolved provider's default.
        return config.AUTO_MODEL
    if model:
        return model[:100]
    if provider == config.MOCK_PROVIDER:
        return config.MOCK_MODEL
    return config.default_model_for(provider)


def _mode_label(mode_id: str | None) -> str:
    from app.ai import modes as modes_mod  # local import: keeps this module light

    return modes_mod.get_mode(mode_id).label


def _clean_mode(mode_id: Any) -> str:
    """A known mode id (built-in or professor-added), else ``SkillError``."""
    from app.ai import modes as modes_mod

    key = str(mode_id or modes_mod.DEFAULT_MODE).strip().lower()
    if not modes_mod.is_known_mode(key):
        known = ", ".join(m.id for m in modes_mod.list_modes())
        raise SkillError(f"Unknown skill mode {key!r}. Known modes: {known}")
    return key


def _clean_max_tokens(value: Any) -> int:
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return config.DEFAULT_MAX_TOKENS
    return max(256, min(tokens, 64000))


def _new_skill(
    db: Session,
    *,
    name: str,
    description: str | None = None,
    system_prompt: str = "",
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    mode: str | None = None,
) -> Skill:
    clean_name = (name or "").strip()
    if not clean_name:
        raise SkillError("Skill name is required")
    provider = _clean_provider(provider)
    if provider == config.AUTO_PROVIDER and model and model != config.AUTO_MODEL:
        # A concrete model implies its provider (auto + "claude-sonnet-5" → anthropic).
        provider = config.provider_for_model(model.strip()) or provider
    skill = Skill(
        name=clean_name[:200],
        description=(description or "").strip() or None,
        system_prompt=system_prompt or "",
        provider=provider,
        model=_clean_model(provider, model),
        max_tokens=_clean_max_tokens(
            max_tokens if max_tokens is not None else config.DEFAULT_MAX_TOKENS
        ),
        mode=_clean_mode(mode),
    )
    db.add(skill)
    return skill


def create_skill(
    db: Session,
    *,
    name: str,
    description: str | None = None,
    system_prompt: str = "",
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    mode: str | None = None,
) -> Skill:
    skill = _new_skill(
        db,
        name=name,
        description=description,
        system_prompt=system_prompt,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        mode=mode,
    )
    db.commit()
    db.refresh(skill)
    return skill


def update_skill(db: Session, skill_id: int, **fields: Any) -> Skill:
    skill = get_skill(db, skill_id)

    if "name" in fields and fields["name"] is not None:
        clean = str(fields["name"]).strip()
        if not clean:
            raise SkillError("Skill name cannot be empty")
        skill.name = clean[:200]
    if "description" in fields and fields["description"] is not None:
        skill.description = str(fields["description"]).strip() or None
    if "system_prompt" in fields and fields["system_prompt"] is not None:
        skill.system_prompt = str(fields["system_prompt"])
    if fields.get("provider"):
        skill.provider = _clean_provider(fields["provider"])
        # Keep the model coherent with the provider unless one was passed too.
        if skill.provider == config.AUTO_PROVIDER or (
            not fields.get("model") and not config.is_known_model(skill.provider, skill.model)
        ):
            skill.model = _clean_model(skill.provider, None)
    if fields.get("model"):
        wanted = str(fields["model"]).strip()
        if skill.provider == config.AUTO_PROVIDER and wanted and wanted != config.AUTO_MODEL:
            owner = config.provider_for_model(wanted)
            if owner:
                skill.provider = owner
        skill.model = _clean_model(skill.provider, wanted)
    if fields.get("max_tokens") is not None:
        skill.max_tokens = _clean_max_tokens(fields["max_tokens"])
    if fields.get("mode"):
        skill.mode = _clean_mode(fields["mode"])

    db.commit()
    db.refresh(skill)
    return skill


def delete_skill(db: Session, skill_id: int) -> int:
    skill = get_skill(db, skill_id)
    root = skills_root().resolve()
    directory = skill_dir(skill.id).resolve()
    tokens: list[ReversibleDelete] = []
    try:
        token = reversible_delete(directory, root, direct_child=True)
        if token is not None:
            tokens.append(token)
        db.delete(skill)
        db.commit()
    except Exception as exc:
        reraise_delete_failure(exc, db.rollback, tokens)
    discard_all(tokens)
    return skill_id


# --------------------------------------------------------------------------
# knowledge documents
# --------------------------------------------------------------------------


def list_docs(db: Session, skill_id: int) -> list[KnowledgeDoc]:
    get_skill(db, skill_id)
    return list(
        db.scalars(
            select(KnowledgeDoc)
            .where(KnowledgeDoc.skill_id == skill_id)
            .order_by(KnowledgeDoc.id)
        ).all()
    )


def add_knowledge_doc(
    db: Session,
    skill_id: int,
    filename: str,
    data: bytes,
    *,
    title: str | None = None,
) -> KnowledgeDoc:
    """Store an uploaded knowledge doc under ``data/skills/{skill_id}/``."""
    skill = get_skill(db, skill_id)
    if len(skill.knowledge_docs) >= MAX_DOCS_PER_SKILL:
        raise UnsupportedDocument(
            f"A skill can hold at most {MAX_DOCS_PER_SKILL} knowledge documents"
        )

    safe_name, mime = _check_doc(filename, data)
    directory = _ensure_skill_dir(skill.id)
    path = _unique_path(directory, safe_name)

    try:
        path.write_bytes(data)
        doc = KnowledgeDoc(
            skill_id=skill.id,
            title=(title or "").strip() or os.path.splitext(path.name)[0],
            filename=path.name,
            file_path=str(path),
            mime_type=mime,
            size_bytes=len(data),
        )
        db.add(doc)
        db.flush()
        # Sessions run with expire_on_commit=False, so drop the stale collection.
        db.expire(skill, ["knowledge_docs"])
        db.commit()
    except Exception:
        db.rollback()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.exception("Could not remove failed knowledge document %s", path)
        raise
    return doc


def get_doc(db: Session, doc_id: int, skill_id: int | None = None) -> KnowledgeDoc:
    doc = db.get(KnowledgeDoc, doc_id)
    if doc is None or (skill_id is not None and doc.skill_id != skill_id):
        raise SkillNotFound(f"Knowledge document {doc_id} not found")
    return doc


def delete_knowledge_doc(db: Session, doc_id: int, skill_id: int | None = None) -> int:
    doc = get_doc(db, doc_id, skill_id)
    parent = db.get(Skill, doc.skill_id)
    tokens: list[ReversibleDelete] = []
    try:
        if doc.file_path:
            token = reversible_delete(doc.file_path, skills_root())
            if token is not None:
                tokens.append(token)
        db.delete(doc)
        if parent is not None:
            db.expire(parent, ["knowledge_docs"])
        db.commit()
    except Exception as exc:
        reraise_delete_failure(exc, db.rollback, tokens)
    discard_all(tokens)
    return doc_id


def read_doc_text(doc: KnowledgeDoc, limit: int = 20000) -> str:
    """Best-effort text extraction for inlineable docs (txt/md only)."""
    ext = os.path.splitext(doc.filename or "")[1].lower()
    if ext not in INLINE_TEXT_EXTENSIONS or not doc.file_path:
        return ""
    try:
        raw = Path(doc.file_path).read_bytes()[: limit * 4]
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")[:limit]


def knowledge_context(skill: Skill, limit_per_doc: int = 4000) -> str:
    """Render inlineable knowledge docs as text for a prompt preamble."""
    chunks: list[str] = []
    for doc in skill.knowledge_docs:
        text = read_doc_text(doc, limit=limit_per_doc)
        if text.strip():
            chunks.append(f"--- knowledge document: {doc.title} ({doc.filename}) ---\n{text}")
        else:
            chunks.append(f"--- knowledge document: {doc.title} ({doc.filename}) [binary] ---")
    return "\n\n".join(chunks)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def build_manifest(skill: Skill) -> dict[str, Any]:
    """The manifest written into a bundle.

    ``settings`` deliberately carries no sampling params — current Anthropic
    models reject temperature/top_p (docs/AI_NOTES.md).
    """
    from app.ai import modes as modes_mod

    mode = modes_mod.get_mode(skill.mode)
    return {
        "format_version": FORMAT_VERSION,
        "name": skill.name,
        "description": skill.description or "",
        "provider": skill.provider,
        "model": skill.model,
        "settings": {"max_tokens": skill.max_tokens},
        "mode": mode.id,
        # Professor-added modes travel with the skill so a colleague's Agora
        # learns the mode on import; built-ins are known everywhere.
        "mode_spec": None if mode.builtin else modes_mod.mode_spec(mode),
        "docs": [
            {"filename": d.filename, "title": d.title, "mime_type": d.mime_type}
            for d in skill.knowledge_docs
        ],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "app": config.APP_NAME,
        "app_version": config.APP_VERSION,
    }


def export_skill(db: Session, skill_id: int, dest: str | Path | None = None) -> Path:
    """Write ``<slug>.agoraskill`` (a zip) and return its path."""
    skill = get_skill(db, skill_id)
    path = Path(dest) if dest else export_root() / f"{_slugify(skill.name)}{BUNDLE_SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(skill)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        zf.writestr(SYSTEM_PROMPT_NAME, skill.system_prompt or "")
        used: set[str] = set()
        for doc in skill.knowledge_docs:
            source = Path(doc.file_path) if doc.file_path else None
            if not source or not source.is_file():
                continue
            name = sanitize_filename(doc.filename)
            while name.lower() in used:
                stem, dot, ext = name.rpartition(".")
                name = f"{stem}-copy.{ext}" if dot else f"{name}-copy"
            used.add(name.lower())
            zf.write(source, f"{DOCS_PREFIX}{name}")
    return path


def export_skill_bytes(db: Session, skill_id: int) -> tuple[str, bytes]:
    """In-memory export — returns (download_filename, zip bytes)."""
    skill = get_skill(db, skill_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(build_manifest(skill), indent=2))
        zf.writestr(SYSTEM_PROMPT_NAME, skill.system_prompt or "")
        for doc in skill.knowledge_docs:
            source = Path(doc.file_path) if doc.file_path else None
            if source and source.is_file():
                zf.writestr(f"{DOCS_PREFIX}{sanitize_filename(doc.filename)}", source.read_bytes())
    return f"{_slugify(skill.name)}{BUNDLE_SUFFIX}", buffer.getvalue()


# --------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------


def _reject_unsafe_member(info: zipfile.ZipInfo) -> None:
    """Zip-slip / symlink guard. Raises BundleError for anything suspicious."""
    name = info.filename
    if not name or name.endswith("/"):
        return  # directory entries are ignored entirely
    if "\x00" in name:
        raise BundleError("Bundle contains an invalid entry name")

    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise BundleError(f"Bundle contains an absolute path: {name!r}")
    parts = PurePosixPath(normalized).parts
    if any(part == ".." for part in parts):
        raise BundleError(f"Bundle contains a path traversal entry: {name!r}")

    # Symlinks would let an attacker escape the extraction dir after the fact.
    if (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise BundleError(f"Bundle contains a symlink: {name!r}")


def _open_bundle(source: Union[str, Path, bytes, bytearray, BinaryIO]) -> zipfile.ZipFile:
    if isinstance(source, (bytes, bytearray)):
        handle: Any = io.BytesIO(bytes(source))
    elif isinstance(source, (str, Path)):
        handle = str(source)
    else:
        handle = source
    try:
        return zipfile.ZipFile(handle)
    except (zipfile.BadZipFile, OSError) as exc:
        raise BundleError("That file is not a valid .agoraskill bundle (bad zip)") from exc


def _read_member(zf: zipfile.ZipFile, info: Any) -> bytes:
    """Read one archive member, mapping corrupt data to a 400-able BundleError.

    A truncated member or a CRC mismatch raises ``BadZipFile`` from inside the
    read, not from ``ZipFile(...)`` — without this it would surface as a 500.
    """
    try:
        return zf.read(info)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError) as exc:
        name = getattr(info, "filename", info)
        raise BundleError(f"That bundle is corrupt (could not read {name!r})") from exc


def validate_manifest(raw: Any) -> dict[str, Any]:
    """Validate + normalize a bundle manifest. Raises BundleError."""
    if not isinstance(raw, dict):
        raise BundleError("manifest.json must contain a JSON object")

    version = raw.get("format_version")
    try:
        version_int = int(version)
    except (TypeError, ValueError):
        raise BundleError("manifest.json is missing a numeric 'format_version'") from None
    if version_int < 1 or version_int > FORMAT_VERSION:
        raise BundleError(
            f"Unsupported bundle format_version {version_int} "
            f"(this build understands 1..{FORMAT_VERSION})"
        )

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise BundleError("manifest.json is missing a non-empty 'name'")

    description = raw.get("description") or ""
    if not isinstance(description, str):
        raise BundleError("manifest.json 'description' must be a string")

    provider = _clean_provider(raw.get("provider"))
    model_raw = raw.get("model")
    if model_raw is not None and not isinstance(model_raw, str):
        raise BundleError("manifest.json 'model' must be a string")

    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        raise BundleError("manifest.json 'settings' must be an object")

    from app.ai import modes as modes_mod

    mode_raw = raw.get("mode")
    if mode_raw is not None and not isinstance(mode_raw, str):
        raise BundleError("manifest.json 'mode' must be a string")
    mode_spec_raw = raw.get("mode_spec")
    mode_spec: dict[str, Any] | None = None
    if mode_spec_raw is not None:
        try:
            mode_spec = modes_mod.mode_spec(modes_mod.validate_mode_spec(mode_spec_raw))
        except modes_mod.ModeError as exc:
            raise BundleError(f"manifest.json 'mode_spec' rejected: {exc}") from exc
    mode_id = str(mode_raw or modes_mod.DEFAULT_MODE).strip().lower()
    if mode_spec is not None and mode_spec["id"] != mode_id:
        raise BundleError("manifest.json 'mode_spec.id' must match 'mode'")
    if mode_spec is None and not modes_mod.is_known_mode(mode_id):
        # A mode this Agora does not know and the bundle did not define.
        mode_id = modes_mod.DEFAULT_MODE

    return {
        "format_version": version_int,
        "name": name.strip()[:200],
        "description": description.strip()[:2000],
        "provider": provider,
        "model": _clean_model(provider, model_raw),
        "settings": {"max_tokens": _clean_max_tokens(settings.get("max_tokens"))},
        "mode": mode_id,
        "mode_spec": mode_spec,
        "doc_titles": _manifest_doc_titles(raw.get("docs")),
    }


def _manifest_doc_titles(docs: Any) -> dict[str, str]:
    titles: dict[str, str] = {}
    if isinstance(docs, list):
        for entry in docs:
            if isinstance(entry, dict) and isinstance(entry.get("filename"), str):
                title = entry.get("title")
                titles[sanitize_filename(entry["filename"]).lower()] = (
                    title if isinstance(title, str) and title.strip() else ""
                )
    return titles


def _unique_skill_name(db: Session, name: str) -> str:
    existing = {n.lower() for n in db.scalars(select(Skill.name)).all()}
    if name.lower() not in existing:
        return name
    base = f"{name} (imported)"
    if base.lower() not in existing:
        return base[:200]
    for n in range(2, 500):
        candidate = f"{name} (imported {n})"
        if candidate.lower() not in existing:
            return candidate[:200]
    raise SkillError("Too many imported copies of this skill")  # pragma: no cover


def import_skill(
    db: Session,
    source: Union[str, Path, bytes, bytearray, BinaryIO],
    *,
    name_override: str | None = None,
) -> Skill:
    """Validate a ``.agoraskill`` bundle and create a new Skill + docs.

    Never trusts a path from the archive: entries are rejected up front and
    every write destination is re-verified to live inside the skill directory.
    """
    zf = _open_bundle(source)
    with zf:
        infos = zf.infolist()
        for info in infos:
            _reject_unsafe_member(info)

        total = sum(max(0, info.file_size) for info in infos)
        if total > MAX_BUNDLE_BYTES:
            raise BundleError("Bundle is too large to import")

        by_name = {
            info.filename.replace("\\", "/"): info
            for info in infos
            if not info.filename.endswith("/")
        }
        manifest_info = by_name.get(MANIFEST_NAME)
        if manifest_info is None:
            raise BundleError("Bundle is missing manifest.json")
        try:
            manifest_raw = json.loads(_read_member(zf, manifest_info).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BundleError("manifest.json is not valid JSON") from exc
        manifest = validate_manifest(manifest_raw)

        system_prompt = ""
        prompt_info = by_name.get(SYSTEM_PROMPT_NAME)
        if prompt_info is not None:
            system_prompt = _read_member(zf, prompt_info).decode("utf-8", errors="replace")

        # Collect docs *before* touching the DB so a bad member aborts cleanly.
        pending: list[tuple[str, bytes]] = []
        for name, info in by_name.items():
            if not name.startswith(DOCS_PREFIX):
                continue
            relative = name[len(DOCS_PREFIX) :]
            if not relative or "/" in relative:
                # Nested directories under docs/ are not part of the format.
                raise BundleError(f"Bundle contains an unexpected nested entry: {name!r}")
            base = relative.rsplit("/", 1)[-1]
            if base.startswith(".") or name.startswith("__MACOSX"):
                continue
            if info.file_size > MAX_DOC_BYTES:
                raise BundleError(f"{base} exceeds the {MAX_DOC_BYTES // (1024 * 1024)} MB limit")
            data = _read_member(zf, info)
            safe_name, _mime = _check_doc(base, data)
            pending.append((safe_name, data))

        if len(pending) > MAX_DOCS_PER_SKILL:
            raise BundleError(f"Bundle holds more than {MAX_DOCS_PER_SKILL} documents")

    if manifest.get("mode_spec"):
        # Learn the professor-added mode (keeps an existing definition as is).
        from app.ai import modes as modes_mod

        modes_mod.save_custom_mode(manifest["mode_spec"], replace=False)

    created_directory: Path | None = None
    try:
        skill = _new_skill(
            db,
            name=_unique_skill_name(
                db, name_override.strip() if name_override else manifest["name"]
            ),
            description=manifest["description"],
            system_prompt=system_prompt,
            provider=manifest["provider"],
            model=manifest["model"],
            max_tokens=manifest["settings"]["max_tokens"],
            mode=manifest["mode"],
        )
        db.flush()
        directory = skill_dir(skill.id)
        skills_root().mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise SkillError(f"Skill directory {directory} already exists") from exc
        created_directory = directory
        root = directory.resolve()
        titles = manifest["doc_titles"]
        for safe_name, data in pending:
            target = _unique_path(directory, safe_name)
            # Final zip-slip check: the resolved destination must stay inside.
            if (root / target.name).resolve().parent != root:
                raise BundleError(
                    f"Refusing to write outside the skill directory: {safe_name!r}"
                )
            target.write_bytes(data)
            db.add(
                KnowledgeDoc(
                    skill_id=skill.id,
                    title=titles.get(safe_name.lower()) or os.path.splitext(target.name)[0],
                    filename=target.name,
                    file_path=str(target),
                    mime_type=ALLOWED_DOC_TYPES[os.path.splitext(target.name)[1].lower()],
                    size_bytes=len(data),
                )
            )
        db.expire(skill, ["knowledge_docs"])
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            log.exception("Could not roll back failed skill import")
        if created_directory is not None:
            try:
                shutil.rmtree(created_directory)
            except FileNotFoundError:
                pass
            except Exception:
                log.exception(
                    "Could not remove failed skill import directory %s", created_directory
                )
        raise
    return skill


def inspect_bundle(source: Union[str, Path, bytes, bytearray, BinaryIO]) -> dict[str, Any]:
    """Read + validate a bundle without importing it (UI preview)."""
    zf = _open_bundle(source)
    with zf:
        for info in zf.infolist():
            _reject_unsafe_member(info)
        names = [i.filename for i in zf.infolist() if not i.filename.endswith("/")]
        if MANIFEST_NAME not in names:
            raise BundleError("Bundle is missing manifest.json")
        try:
            manifest = validate_manifest(
                json.loads(_read_member(zf, MANIFEST_NAME).decode("utf-8"))
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise BundleError("manifest.json is not valid JSON") from exc
        docs = [n[len(DOCS_PREFIX) :] for n in names if n.startswith(DOCS_PREFIX)]
    manifest.pop("doc_titles", None)
    manifest["documents"] = docs
    return manifest


def iter_skill_files(skill_id: int) -> Iterable[Path]:  # pragma: no cover - helper
    directory = skill_dir(skill_id)
    if directory.is_dir():
        yield from sorted(p for p in directory.iterdir() if p.is_file())


__all__ = [
    "FORMAT_VERSION",
    "BUNDLE_SUFFIX",
    "ALLOWED_DOC_TYPES",
    "MAX_DOC_BYTES",
    "SkillError",
    "SkillNotFound",
    "UnsupportedDocument",
    "BundleError",
    "skills_root",
    "skill_dir",
    "sanitize_filename",
    "normalize_provider",
    "skill_dict",
    "doc_dict",
    "list_skills",
    "get_skill",
    "create_skill",
    "update_skill",
    "delete_skill",
    "list_docs",
    "add_knowledge_doc",
    "get_doc",
    "delete_knowledge_doc",
    "read_doc_text",
    "knowledge_context",
    "build_manifest",
    "validate_manifest",
    "export_skill",
    "export_skill_bytes",
    "import_skill",
    "inspect_bundle",
]
