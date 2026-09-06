"""Skill builder: JSON API, page routes, bundle export/import, and test drive.

Owned by the *skills* module. The provider layer (``app.ai.providers``) is owned
by the engine module and is imported lazily + defensively: this router must keep
working (falling back to a deterministic mock reply) whether or not the engine
is present, and whether or not an API key has been configured.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app import config, security
from app import skills_service as svc
from app.db import get_db
from app.models import Skill

log = logging.getLogger("agora.skills")

router = APIRouter()
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class SkillIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    system_prompt: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    #: grade | feedback | selective | a professor-added mode id
    mode: Optional[str] = None


class SkillUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    mode: Optional[str] = None


class ModeIn(BaseModel):
    """A professor-added mode (see app.ai.modes)."""

    id: str = Field(min_length=2, max_length=40)
    label: str = Field(min_length=1, max_length=80)
    description: Optional[str] = Field(default=None, max_length=400)
    scored: bool = True
    selective: bool = False
    instructions: str = Field(min_length=20, max_length=20000)


class DocUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=300)


class TryIn(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    provider: Optional[str] = None
    model: Optional[str] = None


def _handle(exc: svc.SkillError) -> HTTPException:
    status = 404 if isinstance(exc, svc.SkillNotFound) else 400
    return HTTPException(status_code=status, detail=str(exc))


# --------------------------------------------------------------------------
# skill CRUD
# --------------------------------------------------------------------------


@router.get("/api/skills")
def list_skills(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [svc.skill_dict(s) for s in svc.list_skills(db)]


@router.post("/api/skills", status_code=201)
def create_skill(payload: SkillIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        skill = svc.create_skill(db, **payload.model_dump())
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    return svc.skill_dict(skill)


@router.get("/api/skills/{skill_id}")
def get_skill(skill_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return svc.skill_dict(svc.get_skill(db, skill_id))
    except svc.SkillError as exc:
        raise _handle(exc) from exc


@router.patch("/api/skills/{skill_id}")
def update_skill(
    skill_id: int, payload: SkillUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    fields = payload.model_dump(exclude_unset=True)
    try:
        return svc.skill_dict(svc.update_skill(db, skill_id, **fields))
    except svc.SkillError as exc:
        raise _handle(exc) from exc


@router.delete("/api/skills/{skill_id}")
def delete_skill(skill_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return {"deleted": svc.delete_skill(db, skill_id)}
    except svc.SkillError as exc:
        raise _handle(exc) from exc


# --------------------------------------------------------------------------
# modes (built-in + professor-added)
# --------------------------------------------------------------------------


def _modes():
    from app.ai import modes as modes_mod  # local import: keeps the router light

    return modes_mod


def _candidates(db: Session) -> list[dict[str, Any]]:
    """Provider/model options for the compare panel (empty if the engine is absent)."""
    try:
        from app.routers.compare import candidate_options  # noqa: PLC0415 - optional engine

        return candidate_options(db)
    except Exception:  # noqa: BLE001 - the builder must render without the engine
        return []


@router.get("/api/modes")
def list_modes() -> dict[str, Any]:
    modes_mod = _modes()
    return {
        "default": modes_mod.DEFAULT_MODE,
        "modes": [m.to_dict() for m in modes_mod.list_modes()],
        "custom_dir": str(modes_mod.custom_modes_dir()),
    }


@router.post("/api/modes", status_code=201)
def create_mode(payload: ModeIn) -> dict[str, Any]:
    """Add (or replace) a professor-defined mode. Built-in ids are refused."""
    modes_mod = _modes()
    try:
        mode = modes_mod.save_custom_mode(payload.model_dump())
    except modes_mod.ModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = mode.to_dict()
    data["message"] = f"Mode “{mode.label}” saved. Pick it on any skill."
    return data


@router.delete("/api/modes/{mode_id}")
def delete_mode(mode_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    modes_mod = _modes()
    in_use = [s.name for s in svc.list_skills(db) if (s.mode or "") == mode_id.lower()]
    if in_use:
        raise HTTPException(
            status_code=409,
            detail=f"Mode {mode_id!r} is used by: {', '.join(in_use)}. Switch those skills first.",
        )
    try:
        removed = modes_mod.delete_custom_mode(mode_id)
    except modes_mod.ModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail=f"No custom mode {mode_id!r}")
    return {"deleted": mode_id.lower()}


# --------------------------------------------------------------------------
# knowledge documents
# --------------------------------------------------------------------------


@router.get("/api/skills/{skill_id}/docs")
def list_docs(skill_id: int, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    try:
        return [svc.doc_dict(d) for d in svc.list_docs(db, skill_id)]
    except svc.SkillError as exc:
        raise _handle(exc) from exc


@router.post("/api/skills/{skill_id}/docs", status_code=201)
async def upload_doc(
    skill_id: int,
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    raw = await file.read()
    if len(raw) > svc.MAX_DOC_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Document exceeds the {svc.MAX_DOC_BYTES // (1024 * 1024)} MB limit",
        )
    try:
        doc = svc.add_knowledge_doc(db, skill_id, file.filename or "", raw, title=title)
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    return svc.doc_dict(doc)


@router.patch("/api/skills/{skill_id}/docs/{doc_id}")
def rename_doc(
    skill_id: int, doc_id: int, payload: DocUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    try:
        doc = svc.get_doc(db, doc_id, skill_id)
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    doc.title = payload.title.strip()
    db.commit()
    db.refresh(doc)
    return svc.doc_dict(doc)


@router.get("/api/skills/{skill_id}/docs/{doc_id}/download")
def download_doc(skill_id: int, doc_id: int, db: Session = Depends(get_db)) -> Response:
    try:
        doc = svc.get_doc(db, doc_id, skill_id)
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    if not doc.file_path:
        raise HTTPException(status_code=404, detail="Document file is missing on disk")
    # Defence in depth: only ever serve a file that lives under data/skills/,
    # and stream it rather than buffering the whole doc in memory.
    path = Path(doc.file_path).resolve()
    root = svc.skills_root().resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Document file is missing on disk")
    return FileResponse(
        path,
        media_type=doc.mime_type or "application/octet-stream",
        filename=doc.filename or path.name,
    )


@router.delete("/api/skills/{skill_id}/docs/{doc_id}")
def delete_doc(skill_id: int, doc_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return {"deleted": svc.delete_knowledge_doc(db, doc_id, skill_id)}
    except svc.SkillError as exc:
        raise _handle(exc) from exc


# --------------------------------------------------------------------------
# export / import
# --------------------------------------------------------------------------


@router.get("/api/skills/{skill_id}/export")
def export_skill(skill_id: int, db: Session = Depends(get_db)) -> Response:
    """Download the skill as a ``.agoraskill`` bundle."""
    try:
        filename, payload = svc.export_skill_bytes(db, skill_id)
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/skills/import", status_code=201)
async def import_skill(
    file: UploadFile = File(...),
    name: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Import a ``.agoraskill`` bundle as a brand-new skill."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded bundle was empty")
    if len(raw) > svc.MAX_BUNDLE_BYTES:
        raise HTTPException(status_code=413, detail="Bundle is too large")
    try:
        skill = svc.import_skill(db, raw, name_override=name)
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    return svc.skill_dict(skill)


@router.post("/api/skills/import/preview")
async def preview_bundle(file: UploadFile = File(...)) -> dict[str, Any]:
    """Validate a bundle and describe it without creating anything."""
    raw = await file.read()
    try:
        return svc.inspect_bundle(raw)
    except svc.SkillError as exc:
        raise _handle(exc) from exc


# --------------------------------------------------------------------------
# test drive — one-shot chat through the engine's provider layer
# --------------------------------------------------------------------------


def _load_provider_factory() -> Any:
    """``app.ai.providers.get_provider`` if the engine module is available."""
    try:
        from app.ai import providers as providers_mod  # noqa: PLC0415 - lazy by design
    except Exception as exc:  # noqa: BLE001 - engine may be mid-build
        log.debug("provider layer unavailable: %s", exc)
        return None
    return getattr(providers_mod, "get_provider", None)


def _kwargs_for(fn: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs the callee actually declares."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return {}
    allowed = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    return {
        key: value
        for key, value in candidates.items()
        if key in params and params[key].kind in allowed
    }


def _instantiate(factory: Any, provider_name: str, model: str, api_key: str | None, db: Session):
    """Build a provider through the engine's factory.

    The engine contract is ``get_provider(skill_or_settings, db=None)`` where the
    first argument may be a settings dict; the extra attempts below keep this
    working against simpler signatures too.
    """
    settings = {"provider": provider_name, "model": model}
    candidates = {
        "skill_or_settings": settings,
        "settings": settings,
        "skill": settings,
        "provider": provider_name,
        "name": provider_name,
        "provider_name": provider_name,
        "model": model,
        "api_key": api_key,
        "key": api_key,
        "db": db,
        "session": db,
    }
    attempts: list[tuple[tuple, dict]] = []
    filtered = _kwargs_for(factory, candidates)
    if filtered:
        attempts.append(((), filtered))
    attempts.append(((settings,), {"db": db}))
    attempts.append(((provider_name,), {}))
    attempts.append(((provider_name, model), {}))
    attempts.append(((), {}))

    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            provider = factory(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001 - bad key, missing SDK, ...
            last_error = exc
            break
        if provider is not None and hasattr(provider, "chat"):
            return provider
    if last_error is not None:
        log.info("get_provider() unusable (%s) — using the built-in mock reply", last_error)
    return None


def _extract_text(result: Any) -> str:
    """Normalize whatever a provider's ``chat()`` returns into display text."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        parts = [_extract_text(item) for item in result]
        return "\n".join(p for p in parts if p)
    if isinstance(result, dict):
        for key in ("text", "reply", "output_text", "message", "content"):
            if key in result:
                value = result[key]
                if isinstance(value, str):
                    return value
                extracted = _extract_text(value)
                if extracted:
                    return extracted
        if result.get("type") == "text":  # pragma: no cover - covered above
            return str(result.get("text", ""))
        return ""
    for attr in ("text", "reply", "output_text", "content"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, str):
                return value
            extracted = _extract_text(value)
            if extracted:
                return extracted
    return ""


def _mock_reply(skill: Skill, message: str) -> str:
    """Deterministic stand-in used when no provider layer/key is available.

    Mirrors MockProvider's contract (no network, stable output) so the skill
    builder's test drive always demonstrates the persona + doc wiring.
    """
    digest = hashlib.sha256(f"{skill.id}:{message}".encode("utf-8")).hexdigest()[:8]
    docs = [d.title for d in skill.knowledge_docs]
    doc_line = (
        f"I have {len(docs)} knowledge document(s) attached: {', '.join(docs)}."
        if docs
        else "No knowledge documents are attached yet."
    )
    prompt_line = (
        (skill.system_prompt or "").strip().splitlines()[0][:160]
        if (skill.system_prompt or "").strip()
        else "(no system prompt set)"
    )
    return (
        f"“{skill.name}” is wired and ready, but no API key is configured, so "
        f"this is a local stand-in rather than the model's answer.\n\n"
        f"Your question: {message.strip()[:400]}\n\n"
        f"This skill would answer in the voice you set: “{prompt_line}”\n"
        f"{doc_line}\n\n"
        f"Add a key in Settings → API keys to run this against the real model. "
        f"[{digest}]"
    )


def _system_prompt_for(skill: Skill) -> str:
    parts = [(skill.system_prompt or "").strip()]
    knowledge = svc.knowledge_context(skill)
    if knowledge:
        parts.append("Course knowledge documents follow.\n\n" + knowledge)
    return "\n\n".join(p for p in parts if p) or "You are a helpful teaching assistant."


def _run_chat(provider: Any, system: str, message: str, model: str, max_tokens: int) -> str:
    """Invoke the provider's ``chat`` tolerantly and return its text.

    Engine contract: ``chat(system_prompt, messages, tools=None)`` -> ChatTurn.
    """
    chat = getattr(provider, "chat", None)
    if chat is None:  # pragma: no cover - guarded by caller
        return ""
    messages = [{"role": "user", "content": message}]
    kwargs = _kwargs_for(
        chat,
        {
            "system": system,
            "system_prompt": system,
            "messages": messages,
            "tools": None,
            "model": model,
            "max_tokens": max_tokens,
        },
    )
    if "messages" in kwargs:
        result = chat(**kwargs)
    else:
        if "system" not in kwargs and "system_prompt" not in kwargs:
            messages = [{"role": "system", "content": system}, *messages]
        result = chat(messages, **kwargs)
    if inspect.isawaitable(result):
        # We are inside a worker thread, so there is no running loop to clash with.
        result = asyncio.run(result)  # type: ignore[arg-type]
    return _extract_text(result)


@router.post("/api/skills/{skill_id}/try")
async def try_skill(
    skill_id: int, payload: TryIn, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """One-shot chat with a skill (the builder's "test drive").

    Uses ``app.ai.providers.get_provider`` with the skill's provider/model when
    a key is configured; otherwise falls back to the mock provider so the
    builder is fully usable with no credentials.
    """
    try:
        skill = svc.get_skill(db, skill_id)
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    return await _drive(db, skill, payload.message, payload.provider, payload.model)


async def _drive(
    db: Session, skill: Skill, message: str, provider_req: str | None, model_req: str | None
) -> dict[str, Any]:
    """One test-drive turn: resolve the provider, ask, fall back to the mock."""
    requested_provider = svc.normalize_provider(provider_req or skill.provider)
    requested_model = (model_req or "").strip() or skill.model

    # No fixed default model: resolve "auto" (or a provider with no key) to
    # whatever the professor can actually run right now; mock is the last resort.
    provider_name, model, api_key, note = _resolve_for_try(db, requested_provider, requested_model)
    used_mock = provider_name == config.MOCK_PROVIDER or (
        config.requires_api_key(provider_name) and not api_key
    )
    if used_mock:
        provider_name, model = config.MOCK_PROVIDER, config.MOCK_MODEL

    system = _system_prompt_for(skill)
    if config.is_cloud_provider(provider_name) and config.privacy_mode() == config.PRIVACY_MODE_SWAP:
        from app.ai.privacy import protect_unscoped_text
        try:
            system = await run_in_threadpool(protect_unscoped_text, db, system)
            message = await run_in_threadpool(protect_unscoped_text, db, message)
        except Exception as exc:
            raise HTTPException(status_code=409, detail="Privacy scan failed; the skill trial was not sent.") from exc
    factory = _load_provider_factory()
    reply = ""
    error: str | None = None

    if factory is not None and not used_mock:
        provider = await run_in_threadpool(
            _instantiate, factory, provider_name, model, api_key, db
        )
        if provider is not None:
            try:
                reply = await run_in_threadpool(
                    _run_chat, provider, system, message, model, skill.max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - surface, never 500 the builder
                log.warning("test drive failed on %s: %s", provider_name, exc)
                error = f"{type(exc).__name__}: {exc}"

    if not reply:
        reply = _mock_reply(skill, message)
        used_mock = True

    return {
        "skill_id": skill.id,
        "provider": provider_name,
        "model": model,
        "requested_provider": requested_provider,
        "requested_model": requested_model,
        "note": note,
        "used_mock": used_mock,
        "engine_available": factory is not None,
        "reply": reply,
        "error": error,
    }


class CompareCandidate(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None


class CompareIn(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    candidates: list[CompareCandidate] = Field(min_length=1, max_length=4)


@router.post("/api/skills/{skill_id}/compare")
async def compare_skill(
    skill_id: int, payload: CompareIn, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """The same question to several models, side by side (nothing is saved).

    This is how a professor decides which model to pin, on their own material,
    instead of trusting a default.
    """
    import time

    try:
        skill = svc.get_skill(db, skill_id)
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    results: list[dict[str, Any]] = []
    for cand in payload.candidates:
        started = time.perf_counter()
        result = await _drive(db, skill, payload.message, cand.provider, cand.model)
        result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        results.append(result)
    return {"skill_id": skill.id, "message": payload.message, "results": results}


def _resolve_for_try(
    db: Session, requested_provider: str, requested_model: str | None
) -> tuple[str, str, str | None, str | None]:
    """(provider, model, api_key, note) for a test drive — never raises."""
    if requested_provider == config.MOCK_PROVIDER:
        return config.MOCK_PROVIDER, config.MOCK_MODEL, None, None
    note: str | None = None
    provider_name, model = requested_provider, requested_model or ""
    try:
        from app.ai.providers import resolve_provider  # noqa: PLC0415 - engine is optional

        resolution = resolve_provider(db, requested_provider, requested_model)
        if not resolution.available:
            return config.MOCK_PROVIDER, config.MOCK_MODEL, None, resolution.note
        provider_name, model, note = resolution.provider, resolution.model, resolution.note
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - unknown provider etc. → mock, with the reason
        return config.MOCK_PROVIDER, config.MOCK_MODEL, None, str(exc)
    api_key: str | None = None
    if config.requires_api_key(provider_name):
        try:
            api_key = security.get_api_key(db, provider_name)
        except security.SecretError:
            api_key = None
    return provider_name, model or config.default_model_for(provider_name), api_key, note


# --------------------------------------------------------------------------
# page routes
# --------------------------------------------------------------------------


@router.get("/skills", response_class=HTMLResponse)
def page_skills(request: Request, db: Session = Depends(get_db)) -> Any:
    skills = svc.list_skills(db)
    return templates.TemplateResponse(
        request,
        "skills.html",
        {
            "skills": skills,
            "skill_count": len(skills),
            "providers": list(config.PROVIDERS),
            "model_registry": config.load_model_registry(),
            "bundle_suffix": svc.BUNDLE_SUFFIX,
            "modes": {m.id: m for m in _modes().list_modes()},
        },
    )


@router.get("/skills/{skill_id}", response_class=HTMLResponse)
def page_skill_detail(skill_id: int, request: Request, db: Session = Depends(get_db)) -> Any:
    try:
        skill = svc.get_skill(db, skill_id)
    except svc.SkillError as exc:
        raise _handle(exc) from exc
    registry = config.load_model_registry()
    return templates.TemplateResponse(
        request,
        "skill_detail.html",
        {
            "skill": skill,
            "docs": list(skill.knowledge_docs),
            "providers": list(config.PROVIDERS),
            "model_registry": registry,
            "models": registry.get(skill.provider, {}).get("models", []),
            "allowed_doc_types": sorted(svc.ALLOWED_DOC_TYPES),
            "max_doc_mb": svc.MAX_DOC_BYTES // (1024 * 1024),
            "modes": _modes().list_modes(),
            "candidates": _candidates(db),
            "has_key": security.has_api_key(db, skill.provider),
            # The "test drive" panel talks to this skill's one-shot endpoint,
            # which answers with the same {"reply": ...} shape as /api/chat.
            "chat_api": f"/api/skills/{skill.id}/try",
            "try_api": f"/api/skills/{skill.id}/try",
        },
    )


__all__ = ["router"]
