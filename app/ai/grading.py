"""Grading engine: schema, request building, validation, persistence.

This module is the single source of truth for the grade output schema
(``GRADE_SCHEMA``) and for how a grading request is assembled:

    system prompt = skill.system_prompt
                  + rubric rendered as text
                  + small text knowledge docs inline
    user content  = knowledge PDFs/images as blocks
                  + the submission as a document/image block
                  + an anonymized header ("Student #14, Assignment: ...")

Anonymization is a product invariant: nothing leaving this machine contains a
student's name or email — only their per-course number.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from sqlalchemy.orm import Session

from app import config
from app.ai.providers import (
    Provider,
    ProviderError,
    ProviderResponseError,
    block_for_file,
    document_block,
    image_block,
    text_block,
)
from app.models import Assignment, GradeResult, Rubric, Skill, Student, Submission

log = logging.getLogger("agora.ai.grading")


# --------------------------------------------------------------------------
# schema — single source of truth (docs/AI_NOTES.md)
# --------------------------------------------------------------------------

GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "score": {"type": "number"},
                    "comment": {"type": "string"},
                },
                "required": ["key", "score", "comment"],
                "additionalProperties": False,
            },
        },
        "summary_feedback": {"type": "string"},
        "misconceptions": {"type": "array", "items": {"type": "string"}},
        "strengths": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criteria", "summary_feedback", "misconceptions", "strengths"],
    "additionalProperties": False,
}

#: Text knowledge docs larger than this are truncated rather than inlined whole.
MAX_INLINE_DOC_CHARS = 40_000
#: Hard ceiling on how many knowledge docs ride along with one grading request.
MAX_KNOWLEDGE_DOCS = 12
MAX_MISCONCEPTIONS = 12
MAX_TAG_CHARS = 120

TEXT_MIME_PREFIXES = ("text/",)
TEXT_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/x-latex",
    "application/rtf",
}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".rst", ".tex", ".html"}
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}

DEFAULT_GRADING_INSTRUCTIONS = """
You are grading a single anonymized student submission against the rubric below.

Rules:
- Score every rubric criterion, using its exact `key`. Never invent a criterion.
- A score must be between 0 and that criterion's maximum points.
- Comments address the student in the second person and point at specific moves
  in their work, not generic praise.
- `misconceptions` are short, reusable, lower-case tags naming a *conceptual*
  error (e.g. "conflates legality with morality"), not a description of this
  one submission. Reuse the same wording across students so the class-level
  analytics can group them. Return an empty list if there are none.
- `strengths` are short phrases, same style.
- The student is identified only by number. Do not speculate about identity.
""".strip()


# --------------------------------------------------------------------------
# rubric rendering
# --------------------------------------------------------------------------


def rubric_criteria(rubric: Rubric | dict[str, Any] | Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalise a rubric (ORM row, dict, or raw list) into criteria dicts."""
    if rubric is None:
        return []
    if isinstance(rubric, dict):
        raw = rubric.get("criteria") or []
    elif isinstance(rubric, (list, tuple)):
        raw = list(rubric)
    else:
        raw = getattr(rubric, "criteria", None) or []

    out: list[dict[str, Any]] = []
    for index, crit in enumerate(raw):
        if not isinstance(crit, dict):
            continue
        key = str(crit.get("key") or crit.get("title") or f"criterion_{index + 1}").strip()
        if not key:
            continue
        try:
            max_points = float(crit.get("max_points") or 0)
        except (TypeError, ValueError):
            max_points = 0.0
        out.append(
            {
                "key": key,
                "title": str(crit.get("title") or key),
                "description": str(crit.get("description") or ""),
                "max_points": max_points,
            }
        )
    return out


def render_rubric_text(rubric: Any, *, scored: bool = True) -> str:
    """Render the rubric as text for the system prompt.

    The ``- [key] Title (max N points)`` line shape is a contract: MockProvider
    parses it back out so offline grading uses the real rubric.
    """
    criteria = rubric_criteria(rubric)
    if not criteria:
        return "RUBRIC\n(No rubric criteria were configured for this assignment.)"

    name = getattr(rubric, "name", None) or (
        rubric.get("name") if isinstance(rubric, dict) else None
    )
    lines = [f"RUBRIC — {name}" if name else "RUBRIC"]
    if scored:
        lines.append("Score every criterion below, using the exact key in brackets.")
    else:
        lines.append(
            "Comment on every criterion below, using the exact key in brackets. "
            "Do not assign scores."
        )
    for crit in criteria:
        max_points = crit["max_points"]
        pretty = int(max_points) if float(max_points).is_integer() else max_points
        lines.append(f"- [{crit['key']}] {crit['title']} (max {pretty} points)")
        if crit["description"]:
            lines.append(f"    {crit['description']}")
    total = sum(c["max_points"] for c in criteria)
    pretty_total = int(total) if float(total).is_integer() else total
    lines.append(f"TOTAL: {pretty_total} points")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# request building
# --------------------------------------------------------------------------


@dataclass
class GradeRequest:
    """Everything a provider needs, plus what validation needs afterwards."""

    system_prompt: str
    content_blocks: list[dict[str, Any]]
    schema: dict[str, Any]
    #: The criteria the model is asked about (all of them, or the assignment's
    #: selection in selective mode).
    criteria: list[dict[str, Any]] = field(default_factory=list)
    provider: str = config.DEFAULT_PROVIDER
    model: str = config.DEFAULT_MODEL
    max_tokens: int = config.DEFAULT_MAX_TOKENS
    anon_label: str = ""
    filename: str = ""
    notes: list[str] = field(default_factory=list)
    #: Skill mode (see app.ai.modes): id, whether criteria carry scores, and the
    #: criteria left for the professor in selective mode.
    mode: str = "grade"
    scored: bool = True
    manual_criteria: list[dict[str, Any]] = field(default_factory=list)
    #: Increment 1 · privacy module — the Privacy Guard report for this request
    #: (``None`` when the guard did not run: off mode, or an on-device provider).
    privacy: Optional[dict[str, Any]] = None

    @property
    def max_score(self) -> float:
        return round(sum(float(c.get("max_points") or 0) for c in self.criteria), 2)


def split_criteria_for_mode(
    criteria: Sequence[dict[str, Any]], mode: Any, ai_keys: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(criteria the AI handles, criteria left to the professor).

    Only selective modes split; ``ai_keys`` is the assignment's ``ai_criteria``
    (``None`` or empty = every criterion).
    """
    criteria = list(criteria)
    if not getattr(mode, "selective", False) or not ai_keys:
        return criteria, []
    wanted = {str(k).strip() for k in ai_keys if str(k).strip()}
    picked = [c for c in criteria if c["key"] in wanted]
    if not picked:
        return criteria, []
    manual = [c for c in criteria if c["key"] not in wanted]
    return picked, manual


def _is_text_doc(mime_type: str | None, path: Path) -> bool:
    mime = (mime_type or "").lower()
    if mime.startswith(TEXT_MIME_PREFIXES) or mime in TEXT_MIME_TYPES:
        return True
    if mime:
        return False
    return path.suffix.lower() in TEXT_SUFFIXES


def extract_pdf_text(data: bytes, *, max_chars: int = MAX_INLINE_DOC_CHARS) -> str:
    """Best-effort PDF → text, for providers that cannot read PDFs natively.

    Anthropic gets the PDF as a native ``document`` block and never needs this
    (SPEC principle 5: no OCR pipeline). It exists so an OpenAI skill degrades
    to the extracted text instead of failing outright (docs/AI_NOTES.md:
    "degrade gracefully ... rather than guessing an API shape"). Scanned PDFs
    with no text layer yield "" — the caller then surfaces a clear error.
    """
    try:
        import pypdf
    except ImportError:  # pragma: no cover - pypdf is in requirements.txt
        log.warning("pypdf is not installed — cannot extract PDF text for degraded grading")
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
            if sum(len(p) for p in pages) >= max_chars:
                break
    except Exception as exc:  # noqa: BLE001 - a malformed PDF is not fatal here
        log.warning("PDF text extraction failed: %s", exc)
        return ""
    text = "\n\n".join(p.strip() for p in pages if p.strip()).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated ...]"
    return text


def _read_file(path_str: str | None) -> Optional[bytes]:
    if not path_str:
        return None
    path = Path(path_str)
    try:
        return path.read_bytes()
    except OSError as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


def build_knowledge_context(
    docs: Sequence[Any],
    *,
    db: Session | None = None,
    course_id: int | None = None,
    settings: dict[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Split knowledge docs into inline text and attachment blocks.

    Returns ``(inline_sections, blocks, notes)``. Small text docs are inlined in
    the system prompt (cheap, cacheable); PDFs and images become content blocks.
    """
    inline: list[str] = []
    blocks: list[dict[str, Any]] = []
    notes: list[str] = []
    swap_mode = db is not None and settings is not None
    if swap_mode:
        from app.ai import privacy as privacy_guard  # local import: cycle-safe

    for doc in list(docs)[:MAX_KNOWLEDGE_DOCS]:
        title = getattr(doc, "title", None) or getattr(doc, "filename", None) or "Document"
        filename = getattr(doc, "filename", None) or ""
        mime = getattr(doc, "mime_type", None)
        raw = _read_file(getattr(doc, "file_path", None))
        if raw is None:
            if swap_mode:
                raise ProviderError(
                    f"Knowledge document {title!r} is missing on disk, so it was not sent "
                    "to the provider."
                )
            notes.append(f"Knowledge doc {title!r} is missing on disk and was skipped.")
            continue

        path = Path(filename or getattr(doc, "file_path", "") or "")
        if _is_text_doc(mime, path):
            text = raw.decode("utf-8", "replace").strip()
            if len(text) > MAX_INLINE_DOC_CHARS:
                text = text[:MAX_INLINE_DOC_CHARS] + "\n[... truncated ...]"
                notes.append(f"Knowledge doc {title!r} was truncated for the request.")
            if swap_mode:
                text, _ = privacy_guard.protect_cloud_text(
                    db,
                    text,
                    course_id=course_id,
                    settings=settings,
                    subject="knowledge document",
                )
                inline.append(text)
            else:
                inline.append(f"### {title}\n{text}")
        elif (mime or "").lower() == "application/pdf" or path.suffix.lower() == ".pdf":
            if swap_mode:
                extracted = extract_pdf_text(raw)
                if not extracted.strip():
                    raise ProviderError(
                        f"Knowledge document {title!r} has no text layer, so it was not sent "
                        "to the provider."
                    )
                cleaned, _ = privacy_guard.protect_cloud_text(
                    db,
                    extracted,
                    course_id=course_id,
                    settings=settings,
                    subject="knowledge document",
                )
                blocks.append(text_block(cleaned))
            else:
                blocks.append(document_block(raw, "application/pdf", filename or title))
        elif (mime or "").lower() in IMAGE_MIME_TYPES:
            if swap_mode:
                raise ProviderError(
                    f"Knowledge document {title!r} is an image that cannot be pseudonymized, "
                    "so it was not sent to the provider."
                )
            else:
                blocks.append(image_block(raw, (mime or "image/png").lower(), filename or title))
        else:
            if swap_mode:
                raise ProviderError(
                    f"Knowledge document {title!r} has an unsupported type "
                    f"({mime or 'unknown'}), so it was not sent to the provider."
                )
            notes.append(
                f"Knowledge doc {title!r} has an unsupported type ({mime or 'unknown'}) "
                "and was skipped."
            )
    return inline, blocks, notes


def build_system_prompt(
    skill: Skill | None,
    rubric: Any,
    knowledge_sections: Sequence[str] = (),
    mode: Any = None,
) -> str:
    parts: list[str] = []
    voice = (getattr(skill, "system_prompt", "") or "").strip()
    if voice:
        parts.append(voice)
    instructions = (getattr(mode, "instructions", "") or "").strip() or DEFAULT_GRADING_INSTRUCTIONS
    parts.append(instructions)
    parts.append(render_rubric_text(rubric, scored=bool(getattr(mode, "scored", True))))
    if knowledge_sections:
        parts.append(
            "COURSE KNOWLEDGE (provided by the professor)\n" + "\n\n".join(knowledge_sections)
        )
    return "\n\n".join(parts)


def submission_header(
    student: Student | None, assignment: Assignment | None, course_name: str = ""
) -> str:
    """The only identity string that may leave the machine."""
    label = student.anon_label if student is not None else "Student (unassigned)"
    assignment_name = getattr(assignment, "name", "") or "Assignment"
    header = f"{label} · Assignment: {assignment_name}"
    if course_name:
        header += f" · Course: {course_name}"
    description = (getattr(assignment, "description", "") or "").strip()
    lines = [header]
    if description:
        lines.append(f"Assignment brief: {description}")
    lines.append(
        "Grade the attached submission against the rubric and return the required JSON object."
    )
    return "\n".join(lines)


def build_grade_request(
    db: Session,
    submission: Submission,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> GradeRequest:
    """Assemble the full grading request for one submission.

    ``provider``/``model`` override the skill's choice (the model comparison
    feature); the privacy guard then runs for *that* provider, so a cloud
    candidate never receives what only the mock was meant to see.
    """
    assignment = submission.assignment or db.get(Assignment, submission.assignment_id)
    if assignment is None:
        raise ProviderError(f"Submission {submission.id} has no assignment.")

    rubric = db.get(Rubric, assignment.rubric_id) if assignment.rubric_id else None
    skill = db.get(Skill, assignment.skill_id) if assignment.skill_id else None
    student = db.get(Student, submission.student_id) if submission.student_id else None
    course = assignment.course

    # What the skill produces (grade / feedback / selective / professor-added).
    from app.ai import modes as modes_mod  # local import: keeps the module light

    mode = modes_mod.get_mode(getattr(skill, "mode", None))
    all_criteria = rubric_criteria(rubric)
    criteria, manual_criteria = split_criteria_for_mode(
        all_criteria, mode, getattr(assignment, "ai_criteria", None)
    )
    if mode.selective and not manual_criteria and all_criteria:
        log_note = (
            "Selective grading: no criteria are ticked for this assignment, so every "
            "criterion was sent to the model."
        )
    else:
        log_note = None
    rubric_for_prompt: Any = rubric
    if manual_criteria:
        rubric_for_prompt = {"name": getattr(rubric, "name", None), "criteria": criteria}

    # No fixed default model: the skill's request is resolved against whatever
    # the professor has a key for right now (see providers.resolve_provider).
    from app.ai.providers import resolve_provider  # local import: avoids a cycle

    if provider:
        resolution = resolve_provider(db, provider, model)
    else:
        resolution = resolve_provider(
            db,
            getattr(skill, "provider", None) or config.DEFAULT_SKILL_PROVIDER,
            getattr(skill, "model", None),
        )
    provider_name = resolution.provider
    model = resolution.model
    max_tokens = int(
        getattr(skill, "max_tokens", None) or config.max_tokens_for(provider_name, model)
    )

    docs = list(getattr(skill, "knowledge_docs", []) or []) if skill is not None else []
    privacy_settings = config.load_privacy_settings()
    if config.is_cloud_provider(provider_name) and privacy_settings["mode"] == config.PRIVACY_MODE_SWAP:
        inline_sections, knowledge_blocks, notes = build_knowledge_context(
            docs,
            db=db,
            course_id=getattr(course, "id", None),
            settings=privacy_settings,
        )
    else:
        inline_sections, knowledge_blocks, notes = build_knowledge_context(docs)

    system_prompt = build_system_prompt(skill, rubric_for_prompt, inline_sections, mode=mode)
    if log_note:
        notes.append(log_note)

    blocks: list[dict[str, Any]] = list(knowledge_blocks)
    raw = _read_file(submission.file_path)
    filename = submission.original_filename or Path(submission.file_path or "").name
    if raw is None:
        notes.append("The submission file is missing on disk.")
        raise ProviderError(
            f"Submission file is missing on disk: {submission.file_path or '(no path)'}"
        )

    mime = (submission.mime_type or "application/pdf").lower()
    block = block_for_file(raw, mime, filename)
    if block.get("type") == "document" and not config.model_supports_pdf(provider_name, model):
        # This model cannot ingest a PDF natively: attach extracted text so the
        # run degrades instead of raising ProviderUnsupportedError.
        extracted = extract_pdf_text(raw)
        if extracted:
            block["_text"] = extracted
            notes.append(
                f"{model} cannot read PDFs natively; graded from text extracted locally."
            )
        else:
            notes.append(
                f"{model} cannot read PDFs natively and no text could be extracted from "
                "this file (it may be a scan). Grade it with an Anthropic skill."
            )
    # --- Increment 1 · privacy module ------------------------------------
    # Last stop before the submission joins the request: in `swap` mode the
    # Privacy Guard replaces the native file with pseudonymized extracted text
    # for CLOUD providers only (local/mock never leave the machine).
    from app.ai import privacy as privacy_guard  # local import: cycle-safe

    block, privacy_report = privacy_guard.protect_submission_block(
        db,
        submission,
        block,
        provider=provider_name,
        course_id=getattr(course, "id", None),
        raw=raw,
        notes=notes,
    )
    # --- end privacy module ----------------------------------------------

    # AI_NOTES: the document/image block goes BEFORE the text block.
    blocks.append(block)
    header = submission_header(student, assignment, getattr(course, "name", "") or "")
    if config.is_cloud_provider(provider_name) and privacy_settings["mode"] == config.PRIVACY_MODE_SWAP:
        system_prompt, _ = privacy_guard.protect_cloud_text(
            db, system_prompt, course_id=getattr(course, "id", None),
            settings=privacy_settings, subject="grading instructions",
        )
        header, _ = privacy_guard.protect_cloud_text(
            db, header, course_id=getattr(course, "id", None),
            settings=privacy_settings, subject="assignment header",
        )
    blocks.append(text_block(header))

    if skill is None:
        notes.append("No grading skill is attached to this assignment; using defaults.")
    if not criteria:
        notes.append("No rubric criteria are configured; scores cannot be validated.")
    if resolution.note:
        notes.append(resolution.note)

    return GradeRequest(
        system_prompt=system_prompt,
        content_blocks=blocks,
        schema=mode.schema(),
        criteria=criteria,
        provider=provider_name,
        model=model,
        max_tokens=max_tokens,
        anon_label=student.anon_label if student is not None else "",
        filename=filename or "",
        notes=notes,
        privacy=privacy_report,  # Increment 1 · privacy module
        mode=mode.id,
        scored=mode.scored,
        manual_criteria=manual_criteria,
    )


# --------------------------------------------------------------------------
# response validation
# --------------------------------------------------------------------------


@dataclass
class ValidatedGrade:
    criteria: list[dict[str, Any]] = field(default_factory=list)
    overall_score: float = 0.0
    max_score: float = 0.0
    summary_feedback: str = ""
    misconceptions: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    mode: str = "grade"

    @property
    def percentage(self) -> float:
        if not self.max_score:
            return 0.0
        return round(100.0 * self.overall_score / self.max_score, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": self.criteria,
            "overall_score": self.overall_score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "summary_feedback": self.summary_feedback,
            "misconceptions": self.misconceptions,
            "strengths": self.strengths,
            "anomalies": self.anomalies,
            "mode": self.mode,
        }


_WS_RE = re.compile(r"\s+")


def normalize_tags(values: Any, *, limit: int = MAX_MISCONCEPTIONS) -> list[str]:
    """Lower-case, whitespace-collapse, dedupe — analytics groups by exact tag."""
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(values, (str, bytes)):
        values = [values]
    for value in values or []:
        if isinstance(value, dict):
            value = value.get("tag") or value.get("name") or ""
        if value is None:
            continue
        tag = _WS_RE.sub(" ", str(value)).strip().strip(".;,").lower()
        if not tag:
            continue
        tag = tag[:MAX_TAG_CHARS]
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= limit:
            break
    return out


def normalize_phrases(values: Any, *, limit: int = MAX_MISCONCEPTIONS) -> list[str]:
    """Like ``normalize_tags`` but keeps the model's capitalisation (strengths)."""
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(values, (str, bytes)):
        values = [values]
    for value in values or []:
        if value is None:
            continue
        phrase = _WS_RE.sub(" ", str(value)).strip()
        if not phrase:
            continue
        phrase = phrase[:240]
        if phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        out.append(phrase)
        if len(out) >= limit:
            break
    return out


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def validate_grade_payload(
    payload: Any,
    criteria: Sequence[dict[str, Any]] | Rubric | None,
    *,
    label: str = "",
    scored: bool = True,
    manual_criteria: Sequence[dict[str, Any]] = (),
    mode: str = "grade",
) -> ValidatedGrade:
    """Turn a raw model payload into something safe to persist.

    Clamps every score into ``[0, max_points]``, drops criteria the rubric does
    not define, fills in criteria the model skipped, and records every such
    correction in ``anomalies`` so the UI can flag the result for review.

    ``scored=False`` (feedback mode) keeps comments and drops scores: every
    criterion is stored with ``score: None`` and the totals are 0. Criteria in
    ``manual_criteria`` (selective mode) are appended unscored and marked
    ``manual`` so the professor fills them in.
    """
    if not isinstance(payload, dict):
        raise ProviderResponseError("Grade payload was not a JSON object.")

    scored_mode = bool(scored)
    rubric_list = rubric_criteria(criteria)
    by_key = {c["key"]: c for c in rubric_list}
    by_casefold = {c["key"].casefold(): c for c in rubric_list}
    by_title = {c["title"].casefold(): c for c in rubric_list if c.get("title")}

    anomalies: list[str] = []
    scored: dict[str, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []

    raw_criteria = payload.get("criteria")
    if not isinstance(raw_criteria, list):
        raise ProviderResponseError("Grade payload is missing the `criteria` array.")

    for entry in raw_criteria:
        if not isinstance(entry, dict):
            anomalies.append("Dropped a criterion that was not an object.")
            continue
        raw_key = str(entry.get("key") or "").strip()
        comment = str(entry.get("comment") or "").strip()
        score = _coerce_score(entry.get("score"))

        match = (
            by_key.get(raw_key)
            or by_casefold.get(raw_key.casefold())
            or by_title.get(raw_key.casefold())
        )
        if match is None:
            if rubric_list:
                anomalies.append(
                    f"Model returned criterion {raw_key or '(blank)'!r}, which is not in the "
                    "rubric — dropped."
                )
                continue
            # No rubric configured: keep what the model produced, unscored.
            extras.append(
                {
                    "key": raw_key or f"criterion_{len(extras) + 1}",
                    "score": round(score or 0.0, 2),
                    "max_points": 0.0,
                    "comment": comment,
                }
            )
            continue

        key = match["key"]
        if key in scored:
            anomalies.append(f"Model returned criterion {key!r} twice — kept the first.")
            continue

        max_points = float(match.get("max_points") or 0)
        if not scored_mode:
            # Feedback mode: the comment is the product; any score is discarded.
            if not comment:
                anomalies.append(f"Criterion {key!r} came back without a comment.")
            scored[key] = {"key": key, "score": None, "max_points": max_points, "comment": comment}
            continue
        if score is None:
            anomalies.append(f"Criterion {key!r} had a non-numeric score — recorded as 0.")
            score = 0.0
        if score < 0:
            anomalies.append(f"Criterion {key!r} scored {score} — clamped to 0.")
            score = 0.0
        elif max_points and score > max_points:
            anomalies.append(
                f"Criterion {key!r} scored {score} above the {max_points:g}-point maximum "
                "— clamped."
            )
            score = max_points
        if not comment:
            anomalies.append(f"Criterion {key!r} came back without a comment.")

        scored[key] = {
            "key": key,
            "score": round(float(score), 2),
            "max_points": max_points,
            "comment": comment,
        }

    ordered: list[dict[str, Any]] = []
    for crit in rubric_list:
        entry = scored.get(crit["key"])
        if entry is None:
            if scored_mode:
                anomalies.append(
                    f"Model skipped criterion {crit['key']!r} — recorded as 0, review manually."
                )
            else:
                anomalies.append(f"Model skipped criterion {crit['key']!r} — no comment recorded.")
            entry = {
                "key": crit["key"],
                "score": 0.0 if scored_mode else None,
                "max_points": float(crit.get("max_points") or 0),
                "comment": "No response from the model for this criterion.",
            }
        ordered.append(entry)
    ordered.extend(extras)
    # Selective mode: the professor's criteria ride along, unscored, so the
    # result shows the whole rubric and the grading pane can take the scores.
    for crit in rubric_criteria(list(manual_criteria)):
        ordered.append(
            {
                "key": crit["key"],
                "score": None,
                "max_points": float(crit.get("max_points") or 0),
                "comment": "",
                "manual": True,
            }
        )

    summary = str(payload.get("summary_feedback") or "").strip()
    if not summary:
        anomalies.append("Model returned no summary feedback.")

    if scored_mode:
        overall = round(sum(float(c["score"] or 0) for c in ordered), 2)
        max_score = round(sum(float(c["max_points"]) for c in ordered), 2)
    else:
        overall, max_score = 0.0, 0.0

    validated = ValidatedGrade(
        criteria=ordered,
        overall_score=overall,
        max_score=max_score,
        summary_feedback=summary,
        misconceptions=normalize_tags(payload.get("misconceptions")),
        strengths=normalize_phrases(payload.get("strengths")),
        anomalies=anomalies,
        mode=mode,
    )
    if anomalies:
        log.warning(
            "Grade validation flagged %d anomal%s%s: %s",
            len(anomalies),
            "y" if len(anomalies) == 1 else "ies",
            f" for {label}" if label else "",
            "; ".join(anomalies[:5]),
        )
    return validated


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def persist_grade_result(
    db: Session, submission: Submission, validated: ValidatedGrade, model: str
) -> GradeResult:
    """Write (or replace) the GradeResult and mark the submission graded."""
    existing = submission.grade_result
    if existing is not None:  # regrade replaces the old result
        from app import review  # local import: cycle-safe

        # The old review state becomes history; the new text is unreviewed.
        review.record_regrade(db, submission, existing)
        db.delete(existing)
        db.flush()

    result = GradeResult(
        submission_id=submission.id,
        overall_score=validated.overall_score,
        max_score=validated.max_score,
        summary_feedback=validated.summary_feedback,
        criteria=validated.criteria,
        misconceptions=validated.misconceptions,
        strengths=validated.strengths,
        model=model,
        mode=validated.mode or "grade",
    )
    db.add(result)
    submission.status = "graded"
    submission.error = None
    db.commit()
    db.refresh(result)
    db.refresh(submission)

    # --- Increment 1 · insight module ------------------------------------
    # The single post-grading hook: derive this submission's observations,
    # refresh the student's card and look for a class-wide nudge. It swallows
    # its own errors, so grading can never fail because of the insight layer.
    from app.insight import on_submission_graded  # local import: cycle-safe

    on_submission_graded(db, submission, result)
    # --- end insight module ----------------------------------------------

    return result


def grade_submission(
    db: Session,
    submission: Submission,
    provider: Provider | None = None,
) -> tuple[GradeResult, ValidatedGrade]:
    """Build → call → validate → persist. Raises ``ProviderError`` on failure."""
    from app.ai.providers import get_provider  # local import: avoids a cycle at import time

    request = build_grade_request(db, submission)
    if provider is None:
        provider = get_provider(
            {
                "provider": request.provider,
                "model": request.model,
                "max_tokens": request.max_tokens,
            },
            db=db,
        )

    payload = provider.grade(request.system_prompt, request.content_blocks, request.schema)
    validated = validate_grade_payload(
        payload,
        request.criteria,
        label=request.anon_label or request.filename,
        scored=request.scored,
        manual_criteria=request.manual_criteria,
        mode=request.mode,
    )
    # A provider may report the model that actually answered (Anthropic's
    # server-side refusal fallback can substitute one); prefer that.
    served = getattr(provider, "served_model", None) or provider.model
    result = persist_grade_result(db, submission, validated, served)
    return result, validated


__all__ = [
    "GRADE_SCHEMA",
    "GradeRequest",
    "ValidatedGrade",
    "rubric_criteria",
    "render_rubric_text",
    "split_criteria_for_mode",
    "build_knowledge_context",
    "extract_pdf_text",
    "build_system_prompt",
    "submission_header",
    "build_grade_request",
    "validate_grade_payload",
    "normalize_tags",
    "normalize_phrases",
    "persist_grade_result",
    "grade_submission",
]
