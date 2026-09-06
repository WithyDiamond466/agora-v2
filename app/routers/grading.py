"""Grading workflow: submission upload, student mapping, grading runs, results.

Flow (SPEC "Grading"):
1. ``POST /api/assignments/{id}/submissions`` — multi-file upload. Files are
   type-checked (PDF/PNG/JPEG, by magic bytes), name-sanitised, and stored under
   ``data/submissions/{assignment_id}/``. Filename heuristics *prefill* a
   student mapping.
2. ``POST /api/assignments/{id}/submissions/mapping`` — the professor confirms
   (or corrects) that mapping.
3. ``POST /api/submissions/{id}/grade`` / ``POST /api/assignments/{id}/grade-all``
   — status goes ``pending -> grading`` synchronously, then a BackgroundTask
   drives ``grading -> graded | failed``.
4. ``GET /api/assignments/{id}/grading/status`` — polling for the UI.

Assignment and rubric CRUD live here too: they are the unit of grading and no
other module owns them.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app import config, review, terms
from app.ai import grading as grading_engine
from app.ai.providers import ProviderError
from app.db import SessionLocal, get_db
from app.models import Assignment, Course, GradeResult, Rubric, Skill, Student, Submission
from app.storage import ReversibleDelete, discard_all, reraise_delete_failure, reversible_delete

log = logging.getLogger("agora.grading")

router = APIRouter()
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

#: Rebound by tests; background tasks must not reuse the request-scoped session.
session_factory = SessionLocal

STATUS_PENDING = "pending"
STATUS_GRADING = "grading"
STATUS_GRADED = "graded"
STATUS_FAILED = "failed"
ALL_STATUSES = (STATUS_PENDING, STATUS_GRADING, STATUS_GRADED, STATUS_FAILED)

#: Filename magic bytes -> media type. Declared content-types are not trusted.
_MAGIC = (
    (b"%PDF", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)

#: Prefill the mapping only when the heuristic is confident; below this the UI
#: shows a suggestion the professor must pick.
AUTO_ASSIGN_CONFIDENCE = 0.85
SUGGEST_CONFIDENCE = 0.45


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class RubricIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    criteria: list[dict[str, Any]] = Field(default_factory=list)


class RubricUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    criteria: Optional[list[dict[str, Any]]] = None


class SetupCriterion(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10000)
    max_points: float = Field(gt=0, allow_inf_nan=False)


class AssignmentSetup(BaseModel):
    skill_id: int
    rubric_id: Optional[int] = None
    rubric_name: str = Field(default="", max_length=200)
    criteria: list[SetupCriterion] = Field(default_factory=list, max_length=100)


class AssignmentIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    skill_id: Optional[int] = None
    rubric_id: Optional[int] = None
    #: Selective grading: rubric criterion keys the AI scores (None = all).
    ai_criteria: Optional[list[str]] = None


class AssignmentUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    skill_id: Optional[int] = None
    rubric_id: Optional[int] = None
    #: ``[]`` clears the selection (every criterion goes to the AI again).
    ai_criteria: Optional[list[str]] = None


def _clean_ai_criteria(db: Session, rubric_id: Optional[int], keys: Optional[list[str]]) -> Optional[list[str]]:
    """Keep only keys the rubric defines; an empty selection is stored as None."""
    if keys is None:
        return None
    wanted = [str(k).strip() for k in keys if str(k).strip()]
    if not wanted:
        return None
    rubric = db.get(Rubric, rubric_id) if rubric_id else None
    known = {c["key"] for c in grading_engine.rubric_criteria(rubric)}
    if known:
        unknown = [k for k in wanted if k not in known]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown rubric criteria: {', '.join(unknown)}. Known: {', '.join(sorted(known))}",
            )
    return list(dict.fromkeys(wanted))


class MappingEntry(BaseModel):
    submission_id: int
    student_id: Optional[int] = None


class MappingIn(BaseModel):
    mapping: list[MappingEntry] = Field(default_factory=list)


class GradeAllIn(BaseModel):
    #: Re-grade submissions that already have a result.
    force: bool = False
    #: Restrict the run to specific submissions.
    submission_ids: Optional[list[int]] = None


class GradeResultUpdate(BaseModel):
    """The professor's edits to a returned grade (grading.html feedback editor)."""

    summary_feedback: Optional[str] = None
    #: ``[{key, score?, comment?}]`` — only listed criteria are touched.
    criteria: Optional[list[dict[str, Any]]] = None
    misconceptions: Optional[list[str]] = None
    strengths: Optional[list[str]] = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _get_assignment(db: Session, assignment_id: int) -> Assignment:
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise HTTPException(status_code=404, detail=f"Assignment {assignment_id} not found")
    return assignment


def _get_submission(db: Session, submission_id: int) -> Submission:
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail=f"Submission {submission_id} not found")
    return submission


def rubric_dict(rubric: Rubric | None) -> Optional[dict[str, Any]]:
    if rubric is None:
        return None
    return {
        "id": rubric.id,
        "name": rubric.name,
        "criteria": grading_engine.rubric_criteria(rubric),
        "total_points": rubric.total_points,
    }


def assignment_dict(assignment: Assignment) -> dict[str, Any]:
    return {
        "id": assignment.id,
        "course_id": assignment.course_id,
        "name": assignment.name,
        "description": assignment.description,
        "due_date": assignment.due_date.isoformat() if assignment.due_date else None,
        "skill_id": assignment.skill_id,
        "rubric_id": assignment.rubric_id,
        "ai_criteria": list(assignment.ai_criteria) if assignment.ai_criteria else None,
    }


def result_dict(result: GradeResult | None) -> Optional[dict[str, Any]]:
    if result is None:
        return None
    return {
        "id": result.id,
        "submission_id": result.submission_id,
        "overall_score": result.overall_score,
        "max_score": result.max_score,
        "percentage": result.percentage,
        "summary_feedback": result.summary_feedback,
        "criteria": result.criteria,
        "misconceptions": result.misconceptions,
        "strengths": result.strengths,
        "model": result.model,
        "mode": result.mode or "grade",
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "review": review.review_dict(result),
    }


def submission_dict(submission: Submission, *, with_result: bool = True) -> dict[str, Any]:
    student = submission.student
    result = submission.grade_result
    data: dict[str, Any] = {
        "id": submission.id,
        "submission_id": submission.id,
        "assignment_id": submission.assignment_id,
        "student_id": submission.student_id,
        "student_name": student.name if student else None,
        "student_number": student.student_number if student else None,
        "anon_label": student.anon_label if student else None,
        "filename": submission.original_filename,
        "mime_type": submission.mime_type,
        "status": submission.status,
        "error": submission.error,
        "created_at": submission.created_at.isoformat() if submission.created_at else None,
        # Flattened score fields: the queue UI reads these straight off the row.
        "overall_score": result.overall_score if result else None,
        "max_score": result.max_score if result else None,
        "percentage": result.percentage if result else None,
        "incomplete": result.incomplete if result else False,
        "review_state": result.review_state if result else None,
    }
    if with_result:
        data["result"] = result_dict(result)
    return data


def _require_terms_for(db: Session, assignment: Assignment, action: str) -> None:
    """Cloud grading is gated on the terms of use; mock/local are not.

    ``auto`` skills may resolve to a cloud provider, so they count as cloud.
    """
    skill = db.get(Skill, assignment.skill_id) if assignment.skill_id else None
    provider = (getattr(skill, "provider", None) or config.DEFAULT_PROVIDER).strip().lower()
    if config.is_cloud_provider(provider) or provider == config.AUTO_PROVIDER:
        try:
            terms.require_accepted(db, action)
        except terms.TermsNotAccepted as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


# --------------------------------------------------------------------------
# uploads: sanitising and type checking
# --------------------------------------------------------------------------

_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str | None, default: str = "submission") -> str:
    """Strip every path component and anything but ``[A-Za-z0-9._-]``.

    ``../../etc/passwd`` -> ``passwd``; ``C:\\x\\y.pdf`` -> ``y.pdf``.
    """
    raw = (name or "").replace("\\", "/").replace("\x00", "")
    raw = os.path.basename(raw).strip()
    cleaned = _UNSAFE_RE.sub("_", raw)
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.strip("._-")
    if not cleaned:
        return default
    return cleaned[:120]


def detect_media_type(raw: bytes, declared: str | None = None) -> Optional[str]:
    """Sniff the real type from magic bytes.

    ``declared`` (the browser's Content-Type) is accepted for symmetry but is
    deliberately *not* trusted: a .pdf-named text file must be rejected.
    """
    for magic, media_type in _MAGIC:
        if raw.startswith(magic):
            return media_type
    return None


def _unique_path(directory: Path, filename: str) -> Path:
    stem, suffix = os.path.splitext(filename)
    candidate = directory / filename
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def store_submission_file(assignment_id: int, filename: str, raw: bytes, media_type: str) -> Path:
    """Write the upload under ``data/submissions/{assignment_id}/`` safely."""
    directory = (Path(config.UPLOAD_DIR) / str(int(assignment_id))).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    safe = sanitize_filename(filename)
    expected_ext = config.ALLOWED_SUBMISSION_TYPES[media_type]
    if not safe.lower().endswith(expected_ext):
        safe = f"{os.path.splitext(safe)[0] or 'submission'}{expected_ext}"

    path = _unique_path(directory, safe)
    # Defence in depth: never write outside the assignment directory.
    if directory not in path.resolve().parents:
        raise HTTPException(status_code=400, detail="Rejected an unsafe upload path")
    path_existed = path.exists()
    try:
        path.write_bytes(raw)
    except Exception:
        if not path_existed:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("Could not remove partial submission upload %s", path)
        raise
    return path


# --------------------------------------------------------------------------
# filename -> student heuristics
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^a-z0-9]+")
#: "essay1_student_07", "sub-#7", "no 7" — an explicit student-number marker.
_NUMBER_HINT_RE = re.compile(r"(?:student|stud|std|#|no|num|nr)[ _\-.]*0*(\d{1,4})", re.IGNORECASE)
#: Weaker convention: a bare "s07" segment. Tried only after the strong marker.
_SHORT_NUMBER_HINT_RE = re.compile(r"(?:^|[ _\-.])s[ _\-.]*0*(\d{1,4})\b", re.IGNORECASE)


def _tokens(value: str) -> list[str]:
    return [t for t in _TOKEN_RE.split(value.lower()) if t]


def suggest_student(
    filename: str, students: list[Student]
) -> tuple[Optional[Student], float, str]:
    """Best-guess student for an uploaded file: (student, confidence, reason)."""
    if not students:
        return None, 0.0, "no students in this course"

    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    tokens = _tokens(stem)
    compact = "".join(tokens)

    by_number = {s.student_number: s for s in students}
    for pattern, confidence in ((_NUMBER_HINT_RE, 0.95), (_SHORT_NUMBER_HINT_RE, 0.9)):
        match = pattern.search(stem)
        if match:
            number = int(match.group(1))
            if number in by_number:
                return by_number[number], confidence, f"filename names student #{number}"

    best: Optional[Student] = None
    best_score = 0.0
    best_reason = ""
    for student in students:
        name_tokens = _tokens(student.name or "")
        if not name_tokens:
            continue
        hits = sum(1 for token in name_tokens if token and token in tokens)
        token_score = hits / len(name_tokens)
        ratio = difflib.SequenceMatcher(None, compact, "".join(name_tokens)).ratio()
        score = max(token_score, ratio)
        if hits and len(name_tokens) > 1 and hits == len(name_tokens):
            score = max(score, 0.95)
        if score > best_score:
            best, best_score = student, score
            best_reason = (
                f"filename matches {hits}/{len(name_tokens)} name parts"
                if hits
                else f"filename resembles the student's name ({ratio:.0%})"
            )

    # A bare number in the filename is a weaker but common convention.
    if best_score < SUGGEST_CONFIDENCE:
        for token in tokens:
            if token.isdigit() and int(token) in by_number:
                return by_number[int(token)], 0.6, f"filename contains the number {int(token)}"

    if best is None or best_score < SUGGEST_CONFIDENCE:
        return None, round(best_score, 2), "no confident match"
    return best, round(best_score, 2), best_reason


def build_mapping_suggestions(
    filenames: Iterable[str], students: list[Student]
) -> list[dict[str, Any]]:
    """Suggest one student per filename, never reusing a student twice."""
    taken: set[int] = set()
    out: list[dict[str, Any]] = []
    for filename in filenames:
        student, confidence, reason = suggest_student(filename, students)
        if student is not None and student.id in taken:
            student, confidence, reason = None, 0.0, "student already matched to another file"
        if student is not None:
            taken.add(student.id)
        out.append(
            {
                "filename": filename,
                "student_id": student.id if student else None,
                "student_number": student.student_number if student else None,
                "student_name": student.name if student else None,
                "confidence": confidence,
                "reason": reason,
            }
        )
    return out


# --------------------------------------------------------------------------
# rubric CRUD
# --------------------------------------------------------------------------


@router.get("/api/rubrics")
def list_rubrics(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rubrics = db.scalars(select(Rubric).order_by(Rubric.id.desc())).all()
    return [rubric_dict(r) for r in rubrics]  # type: ignore[misc]


@router.post("/api/rubrics", status_code=201)
def create_rubric(payload: RubricIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    criteria = grading_engine.rubric_criteria({"criteria": payload.criteria})
    rubric = Rubric(name=payload.name.strip(), criteria=criteria)
    db.add(rubric)
    db.commit()
    db.refresh(rubric)
    return rubric_dict(rubric)  # type: ignore[return-value]


@router.get("/api/rubrics/{rubric_id}")
def get_rubric(rubric_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    rubric = db.get(Rubric, rubric_id)
    if rubric is None:
        raise HTTPException(status_code=404, detail=f"Rubric {rubric_id} not found")
    return rubric_dict(rubric)  # type: ignore[return-value]


@router.patch("/api/rubrics/{rubric_id}")
def update_rubric(
    rubric_id: int, payload: RubricUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    rubric = db.get(Rubric, rubric_id)
    if rubric is None:
        raise HTTPException(status_code=404, detail=f"Rubric {rubric_id} not found")
    if payload.name is not None:
        rubric.name = payload.name.strip()
    if payload.criteria is not None:
        rubric.criteria = grading_engine.rubric_criteria({"criteria": payload.criteria})
    db.commit()
    db.refresh(rubric)
    return rubric_dict(rubric)  # type: ignore[return-value]


@router.delete("/api/rubrics/{rubric_id}")
def delete_rubric(rubric_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    rubric = db.get(Rubric, rubric_id)
    if rubric is None:
        raise HTTPException(status_code=404, detail=f"Rubric {rubric_id} not found")
    db.delete(rubric)
    db.commit()
    return {"deleted": rubric_id}


# --------------------------------------------------------------------------
# assignment CRUD
# --------------------------------------------------------------------------


@router.get("/api/courses/{course_id}/assignments")
def list_assignments(course_id: int, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    if db.get(Course, course_id) is None:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    assignments = db.scalars(
        select(Assignment).where(Assignment.course_id == course_id).order_by(Assignment.id)
    ).all()
    counts_by_assignment = {
        assignment.id: {status: 0 for status in ALL_STATUSES} for assignment in assignments
    }
    status_rows = db.execute(
        select(Submission.assignment_id, Submission.status, func.count(Submission.id))
        .join(Assignment, Assignment.id == Submission.assignment_id)
        .where(Assignment.course_id == course_id)
        .group_by(Submission.assignment_id, Submission.status)
    ).all()
    for assignment_id, status, count in status_rows:
        counts_by_assignment[assignment_id][status] = count
    out = []
    for assignment in assignments:
        data = assignment_dict(assignment)
        counts = counts_by_assignment[assignment.id]
        data["submission_counts"] = counts
        data["submission_count"] = sum(counts.values())
        out.append(data)
    return out


@router.post("/api/courses/{course_id}/assignments", status_code=201)
def create_assignment(
    course_id: int, payload: AssignmentIn, db: Session = Depends(get_db)
) -> dict[str, Any]:
    if db.get(Course, course_id) is None:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    if payload.rubric_id is not None and db.get(Rubric, payload.rubric_id) is None:
        raise HTTPException(status_code=400, detail=f"Rubric {payload.rubric_id} not found")
    if payload.skill_id is not None and db.get(Skill, payload.skill_id) is None:
        raise HTTPException(status_code=400, detail=f"Skill {payload.skill_id} not found")

    assignment = Assignment(
        course_id=course_id,
        name=payload.name.strip(),
        description=(payload.description or "").strip() or None,
        due_date=payload.due_date,
        skill_id=payload.skill_id,
        rubric_id=payload.rubric_id,
        ai_criteria=_clean_ai_criteria(db, payload.rubric_id, payload.ai_criteria),
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment_dict(assignment)


@router.get("/api/assignments/{assignment_id}")
def get_assignment(assignment_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    assignment = _get_assignment(db, assignment_id)
    data = assignment_dict(assignment)
    data["rubric"] = rubric_dict(db.get(Rubric, assignment.rubric_id) if assignment.rubric_id else None)
    data["submission_counts"] = _status_counts(db, assignment_id)
    return data


@router.patch("/api/assignments/{assignment_id}")
def update_assignment(
    assignment_id: int, payload: AssignmentUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    assignment = _get_assignment(db, assignment_id)
    if payload.name is not None:
        assignment.name = payload.name.strip()
    if payload.description is not None:
        assignment.description = payload.description.strip() or None
    if payload.due_date is not None:
        assignment.due_date = payload.due_date
    if payload.ai_criteria is not None:
        rubric_id = payload.rubric_id if payload.rubric_id is not None else assignment.rubric_id
        assignment.ai_criteria = _clean_ai_criteria(db, rubric_id, payload.ai_criteria)
    if payload.skill_id is not None:
        if db.get(Skill, payload.skill_id) is None:
            raise HTTPException(status_code=400, detail=f"Skill {payload.skill_id} not found")
        assignment.skill_id = payload.skill_id
    if payload.rubric_id is not None:
        if db.get(Rubric, payload.rubric_id) is None:
            raise HTTPException(status_code=400, detail=f"Rubric {payload.rubric_id} not found")
        assignment.rubric_id = payload.rubric_id
    db.commit()
    db.refresh(assignment)
    return assignment_dict(assignment)


@router.post("/api/assignments/{assignment_id}/setup")
def setup_assignment(assignment_id: int, payload: AssignmentSetup, db: Session = Depends(get_db)) -> dict[str, Any]:
    assignment = _get_assignment(db, assignment_id)
    if any(sub.grade_result is not None or sub.status == STATUS_GRADING for sub in assignment.submissions):
        raise HTTPException(status_code=409, detail="This assignment already has grading work. Create a new assignment to use a different rubric or Skill.")
    if db.get(Skill, payload.skill_id) is None:
        raise HTTPException(status_code=400, detail="Choose an available Skill first.")
    if payload.rubric_id is not None:
        rubric = db.get(Rubric, payload.rubric_id)
        if rubric is None or not rubric.criteria:
            raise HTTPException(status_code=400, detail="Choose a rubric with at least one criterion.")
    else:
        if not payload.rubric_name.strip() or not payload.criteria or any(not c.title.strip() for c in payload.criteria):
            raise HTTPException(status_code=400, detail="Name the rubric and add at least one named criterion.")
        criteria = [{"key": f"criterion_{i}", "title": c.title.strip(), "description": c.description.strip(), "max_points": c.max_points} for i, c in enumerate(payload.criteria, 1)]
        rubric = Rubric(name=payload.rubric_name.strip(), criteria=criteria)
        db.add(rubric)
        db.flush()
    assignment.rubric_id = rubric.id
    assignment.skill_id = payload.skill_id
    assignment.ai_criteria = None
    db.commit()
    db.refresh(assignment)
    return assignment_dict(assignment)


@router.delete("/api/assignments/{assignment_id}")
def delete_assignment(assignment_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    assignment = _get_assignment(db, assignment_id)
    tokens: list[ReversibleDelete] = []
    try:
        token = reversible_delete(
            Path(config.UPLOAD_DIR) / str(assignment_id),
            config.UPLOAD_DIR,
            direct_child=True,
        )
        if token is not None:
            tokens.append(token)
        db.delete(assignment)
        db.commit()
    except Exception as exc:
        reraise_delete_failure(exc, db.rollback, tokens)
    discard_all(tokens)
    return {"deleted": assignment_id}


# --------------------------------------------------------------------------
# submission upload + mapping
# --------------------------------------------------------------------------


def _parse_upload_mapping(mapping: str | None) -> dict[str, Optional[int]]:
    """``[{"filename": "a.pdf", "student_id": 3}]`` -> ``{"a.pdf": 3}``."""
    if not mapping:
        return {}
    try:
        parsed = json.loads(mapping)
    except (TypeError, ValueError):
        log.warning("Ignoring an unparseable upload mapping payload")
        return {}
    out: dict[str, Optional[int]] = {}
    for entry in parsed if isinstance(parsed, list) else []:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        if not filename:
            continue
        try:
            student_id = int(entry["student_id"]) if entry.get("student_id") else None
        except (TypeError, ValueError):
            student_id = None
        out[str(filename)] = student_id
    return out


@router.post("/api/assignments/{assignment_id}/submissions", status_code=201)
async def upload_submissions(
    assignment_id: int,
    files: list[UploadFile] = File(...),
    #: Parallel to `files`; empty string = "leave unassigned". Sent by the
    #: grading UI once the professor has confirmed the mapping.
    student_ids: list[str] = Form(default=[]),
    #: Alternative form of the same thing: [{"filename", "student_id"}] as JSON.
    mapping: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Multi-file upload. Creates one pending Submission per accepted file.

    Student assignment, in precedence order: the professor-confirmed mapping
    (``student_ids`` / ``mapping``), then a filename heuristic — and only when
    that heuristic is confident. Anything else comes back flagged for
    confirmation via ``POST .../submissions/mapping``.
    """
    assignment = _get_assignment(db, assignment_id)
    students = db.scalars(
        select(Student)
        .where(Student.course_id == assignment.course_id)
        .order_by(Student.student_number)
    ).all()
    already_mapped = {
        s.student_id
        for s in db.scalars(
            select(Submission).where(Submission.assignment_id == assignment_id)
        ).all()
        if s.student_id
    }
    available = [s for s in students if s.id not in already_mapped]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings: list[str] = []
    #: (index into `files`, filename, bytes, media type)
    payloads: list[tuple[int, str, bytes, str]] = []

    from app.ai.providers import resolve_provider
    skill = db.get(Skill, assignment.skill_id) if assignment.skill_id else None
    resolution = resolve_provider(db, getattr(skill, "provider", None), getattr(skill, "model", None))
    cloud_swap = config.is_cloud_provider(resolution.provider) and config.privacy_mode() == config.PRIVACY_MODE_SWAP
    for index, upload in enumerate(files):
        raw = await upload.read(config.MAX_UPLOAD_BYTES + 1)
        name = upload.filename or "submission"
        if not raw:
            rejected.append({"filename": name, "reason": "the file was empty"})
            continue
        if len(raw) > config.MAX_UPLOAD_BYTES:
            rejected.append(
                {
                    "filename": name,
                    "reason": f"larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                }
            )
            continue
        media_type = detect_media_type(raw, upload.content_type)
        if media_type not in config.ALLOWED_SUBMISSION_TYPES:
            rejected.append(
                {
                    "filename": name,
                    "reason": "unsupported file type — submissions must be PDF, PNG or JPEG",
                }
            )
            continue
        if cloud_swap and (media_type != "application/pdf" or not grading_engine.extract_pdf_text(raw).strip()):
            rejected.append({"filename": name, "reason": "Cloud privacy mode needs a PDF with selectable text; this file cannot be safely graded."})
            continue
        payloads.append((index, name, raw, media_type))

    suggestions = build_mapping_suggestions([p[1] for p in payloads], available)
    by_filename = _parse_upload_mapping(mapping)
    by_id = {s.id: s for s in students}
    taken = set(already_mapped)

    created_paths: list[Path] = []
    try:
        for (index, name, raw, media_type), suggestion in zip(payloads, suggestions):
            path = store_submission_file(assignment_id, name, raw, media_type)
            created_paths.append(path)

            confirmed: Optional[int] = None
            raw_choice = student_ids[index].strip() if index < len(student_ids) else ""
            if raw_choice:
                try:
                    confirmed = int(raw_choice)
                except ValueError:
                    warnings.append(f"{name}: ignored an unreadable student id {raw_choice!r}.")
            elif name in by_filename and by_filename[name]:
                confirmed = by_filename[name]

            if confirmed is not None:
                student = by_id.get(confirmed)
                if student is None:
                    warnings.append(f"{name}: student {confirmed} is not in this course.")
                    confirmed = None
                elif confirmed in taken:
                    warnings.append(
                        f"{name}: {student.name} already has a submission for this assignment."
                    )
                    confirmed = None

            student_id = confirmed
            if student_id is None and (
                suggestion["student_id"]
                and suggestion["confidence"] >= AUTO_ASSIGN_CONFIDENCE
                and suggestion["student_id"] not in taken
            ):
                student_id = suggestion["student_id"]
            if student_id is not None:
                taken.add(student_id)

            submission = Submission(
                assignment_id=assignment_id,
                student_id=student_id,
                file_path=str(path),
                original_filename=sanitize_filename(name),
                mime_type=media_type,
                status=STATUS_PENDING,
            )
            db.add(submission)
            db.flush()
            accepted.append(
                {
                    "submission_id": submission.id,
                    "filename": submission.original_filename,
                    "mime_type": media_type,
                    "stored_path": str(path),
                    "student_id": student_id,
                    "suggested_student_id": suggestion["student_id"],
                    "suggested_student_name": suggestion["student_name"],
                    "suggested_student_number": suggestion["student_number"],
                    "confidence": suggestion["confidence"],
                    "reason": suggestion["reason"],
                    "confirmed": confirmed is not None,
                    "auto_assigned": student_id is not None and confirmed is None,
                    "needs_confirmation": student_id is None,
                }
            )
        db.commit()
    except Exception:
        db.rollback()
        for path in created_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("Could not remove failed submission upload %s", path)
        raise

    return {
        "assignment_id": assignment_id,
        "uploaded": accepted,
        "rejected": rejected,
        "warnings": warnings,
        "needs_confirmation": sum(1 for a in accepted if a["needs_confirmation"]),
        "students": [
            {"id": s.id, "name": s.name, "student_number": s.student_number} for s in students
        ],
    }


@router.get("/api/assignments/{assignment_id}/submissions")
def list_submissions(assignment_id: int, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    _get_assignment(db, assignment_id)
    submissions = db.scalars(
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.grade_result))
        .where(Submission.assignment_id == assignment_id)
        .order_by(Submission.id)
    ).all()
    return [submission_dict(s) for s in submissions]


@router.post("/api/assignments/{assignment_id}/submissions/mapping")
def confirm_mapping(
    assignment_id: int, payload: MappingIn, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Professor-confirmed filename -> student mapping."""
    assignment = _get_assignment(db, assignment_id)
    submissions = {
        s.id: s
        for s in db.scalars(
            select(Submission).where(Submission.assignment_id == assignment_id)
        ).all()
    }
    student_ids = {entry.student_id for entry in payload.mapping if entry.student_id is not None}
    students = {
        student.id: student
        for student in db.scalars(select(Student).where(Student.id.in_(student_ids))).all()
    }

    seen_students: dict[int, int] = {
        s.student_id: s.id
        for s in submissions.values()
        if s.student_id and s.id not in {e.submission_id for e in payload.mapping}
    }
    updated: list[dict[str, Any]] = []

    for entry in payload.mapping:
        submission = submissions.get(entry.submission_id)
        if submission is None:
            raise HTTPException(
                status_code=404,
                detail=f"Submission {entry.submission_id} is not part of this assignment",
            )
        if entry.student_id is None:
            submission.student_id = None
            updated.append({"submission_id": submission.id, "student_id": None})
            continue

        student = students.get(entry.student_id)
        if student is None or student.course_id != assignment.course_id:
            raise HTTPException(
                status_code=400,
                detail=f"Student {entry.student_id} is not enrolled in this course",
            )
        clash = seen_students.get(student.id)
        if clash is not None and clash != submission.id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{student.name} is already mapped to submission {clash}. "
                    "Each student may have one submission per assignment."
                ),
            )
        seen_students[student.id] = submission.id
        submission.student_id = student.id
        updated.append({"submission_id": submission.id, "student_id": student.id})

    db.commit()
    unmapped = sum(1 for s in submissions.values() if s.student_id is None)
    return {"assignment_id": assignment_id, "updated": updated, "unmapped": unmapped}


@router.get("/api/submissions/{submission_id}/file")
def download_submission(submission_id: int, db: Session = Depends(get_db)) -> FileResponse:
    submission = _get_submission(db, submission_id)
    if not submission.file_path:
        raise HTTPException(status_code=404, detail="This submission has no stored file")
    path = Path(submission.file_path).resolve()
    upload_root = Path(config.UPLOAD_DIR).resolve()
    if upload_root not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Submission file is missing")
    return FileResponse(
        path,
        media_type=submission.mime_type or "application/octet-stream",
        filename=submission.original_filename or path.name,
    )


@router.delete("/api/submissions/{submission_id}")
def delete_submission(submission_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    submission = _get_submission(db, submission_id)
    tokens: list[ReversibleDelete] = []
    try:
        if submission.file_path:
            token = reversible_delete(submission.file_path, config.UPLOAD_DIR)
            if token is not None:
                tokens.append(token)
        db.delete(submission)
        db.commit()
    except Exception as exc:
        reraise_delete_failure(exc, db.rollback, tokens)
    discard_all(tokens)
    return {"deleted": submission_id}


# --------------------------------------------------------------------------
# grading runs
# --------------------------------------------------------------------------


def _status_counts(db: Session, assignment_id: int) -> dict[str, int]:
    counts = {status: 0 for status in ALL_STATUSES}
    for submission in db.scalars(
        select(Submission).where(Submission.assignment_id == assignment_id)
    ).all():
        counts[submission.status] = counts.get(submission.status, 0) + 1
    return counts


def _mark_failed(db: Session, submission: Submission, message: str) -> None:
    submission.status = STATUS_FAILED
    submission.error = message[:2000]
    db.commit()


def grade_one(db: Session, submission_id: int) -> None:
    """Grade a single submission in its own session. Never raises."""
    submission = db.get(Submission, submission_id)
    if submission is None:
        log.warning("Submission %s vanished before grading", submission_id)
        return
    try:
        result, validated = grading_engine.grade_submission(db, submission)
        if validated.anomalies:
            log.info(
                "Submission %s graded with %d anomal%s",
                submission_id,
                len(validated.anomalies),
                "y" if len(validated.anomalies) == 1 else "ies",
            )
        log.info(
            "Submission %s graded: %.2f/%.2f", submission_id, result.overall_score, result.max_score
        )
    except ProviderError as exc:
        log.warning("Grading submission %s failed: %s", submission_id, exc)
        db.rollback()
        _mark_failed(db, submission, str(exc))
    except Exception as exc:  # noqa: BLE001 - a failed item must not kill the run
        log.exception("Unexpected error grading submission %s", submission_id)
        db.rollback()
        _mark_failed(db, submission, f"Unexpected error: {exc}")


def run_grading_job(submission_ids: list[int]) -> None:
    """BackgroundTask entry point — sequential, one fresh session per item."""
    for submission_id in submission_ids:
        session = session_factory()
        try:
            grade_one(session, submission_id)
        finally:
            session.close()


def _queue(db: Session, submissions: list[Submission]) -> list[int]:
    """Flip ``pending|failed -> grading`` synchronously so polling sees it."""
    ids: list[int] = []
    for submission in submissions:
        submission.status = STATUS_GRADING
        submission.error = None
        ids.append(submission.id)
    db.commit()
    return ids


@router.post("/api/submissions/{submission_id}/grade")
def grade_submission_endpoint(
    submission_id: int,
    background_tasks: BackgroundTasks,
    force: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    submission = _get_submission(db, submission_id)
    if submission.status == STATUS_GRADING and not force:
        raise HTTPException(
            status_code=409,
            detail="This submission is already being graded (retry with ?force=true if the "
            "app was interrupted mid-run).",
        )
    if submission.student_id is None:
        raise HTTPException(
            status_code=409,
            detail="Assign this file to a student before grading (submissions are anonymized "
            "by student number).",
        )
    _require_terms_for(db, _get_assignment(db, submission.assignment_id), "grading with a cloud model")
    ids = _queue(db, [submission])
    background_tasks.add_task(run_grading_job, ids)
    return {"submission_id": submission_id, "status": STATUS_GRADING, "queued": 1}


@router.post("/api/assignments/{assignment_id}/grade-all")
def grade_all(
    assignment_id: int,
    background_tasks: BackgroundTasks,
    payload: GradeAllIn | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Queue every gradable submission for this assignment."""
    assignment = _get_assignment(db, assignment_id)
    _require_terms_for(db, assignment, "grading with a cloud model")
    options = payload or GradeAllIn()

    query = select(Submission).where(Submission.assignment_id == assignment_id)
    if options.submission_ids:
        query = query.where(Submission.id.in_(options.submission_ids))
    submissions = db.scalars(query.order_by(Submission.id)).all()

    queue: list[Submission] = []
    skipped: list[dict[str, Any]] = []
    for submission in submissions:
        if submission.student_id is None:
            skipped.append({"submission_id": submission.id, "reason": "no student mapped"})
            continue
        if submission.status == STATUS_GRADING and not options.force:
            # `force` also rescues rows stranded in `grading` by an interrupted run.
            skipped.append({"submission_id": submission.id, "reason": "already grading"})
            continue
        if submission.status == STATUS_GRADED and not options.force:
            skipped.append({"submission_id": submission.id, "reason": "already graded"})
            continue
        queue.append(submission)

    ids = _queue(db, queue)
    if ids:
        background_tasks.add_task(run_grading_job, ids)
    return {
        "assignment_id": assignment_id,
        "queued": len(ids),
        "submission_ids": ids,
        "skipped": skipped,
        "status": STATUS_GRADING if ids else "idle",
    }


@router.get("/api/assignments/{assignment_id}/grading/status")
def grading_status(assignment_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Polling endpoint for the grading queue UI."""
    _get_assignment(db, assignment_id)
    submissions = db.scalars(
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.grade_result))
        .where(Submission.assignment_id == assignment_id)
        .order_by(Submission.id)
    ).all()

    counts = {status: 0 for status in ALL_STATUSES}
    rows: list[dict[str, Any]] = []
    for submission in submissions:
        counts[submission.status] = counts.get(submission.status, 0) + 1
        result = submission.grade_result
        rows.append(
            {
                "id": submission.id,
                "submission_id": submission.id,
                "student_id": submission.student_id,
                "student_number": submission.student.student_number if submission.student else None,
                "student_name": submission.student.name if submission.student else None,
                "filename": submission.original_filename,
                "status": submission.status,
                "error": submission.error,
                "overall_score": result.overall_score if result else None,
                "max_score": result.max_score if result else None,
                "percentage": result.percentage if result else None,
        "incomplete": result.incomplete if result else False,
                "review_state": result.review_state if result else None,
                "mode": (result.mode or "grade") if result else None,
            }
        )
    total = len(submissions)
    done = counts[STATUS_GRADED] + counts[STATUS_FAILED]
    return {
        "assignment_id": assignment_id,
        "total": total,
        "counts": counts,
        "done": done,
        "in_progress": counts[STATUS_GRADING] > 0,
        "percent_complete": round(100.0 * done / total, 1) if total else 0.0,
        "submissions": rows,
    }


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@router.get("/api/assignments/{assignment_id}/results")
def assignment_results(assignment_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    assignment = _get_assignment(db, assignment_id)
    submissions = db.scalars(
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.grade_result))
        .where(Submission.assignment_id == assignment_id)
        .order_by(Submission.id)
    ).all()
    graded = [s for s in submissions if s.grade_result is not None]
    percents = [s.grade_result.percentage for s in graded if s.grade_result.percentage is not None]
    return {
        "assignment": assignment_dict(assignment),
        "rubric": rubric_dict(
            db.get(Rubric, assignment.rubric_id) if assignment.rubric_id else None
        ),
        "counts": _status_counts(db, assignment_id),
        "average_percent": round(sum(percents) / len(percents), 1) if percents else None,
        "results": [submission_dict(s) for s in submissions],
    }


@router.get("/api/submissions/{submission_id}/result")
def submission_result(submission_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    return submission_dict(_get_submission(db, submission_id))


@router.patch("/api/submissions/{submission_id}/result")
def update_submission_result(
    submission_id: int, payload: GradeResultUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Save the professor's edits to a grade (feedback text, per-criterion edits).

    The professor always has the last word on a grade, so edits are stored as
    given — only scores are clamped to the criterion maximum, and the overall
    score is recomputed from the edited criteria.
    """
    submission = _get_submission(db, submission_id)
    result = submission.grade_result
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Submission {submission_id} has no grade result yet"
        )

    touched: list[str] = []
    if payload.summary_feedback is not None:
        if (payload.summary_feedback.strip()) != (result.summary_feedback or ""):
            touched.append("summary_feedback")
        result.summary_feedback = payload.summary_feedback.strip()

    if payload.criteria is not None:
        edits = {
            str(entry.get("key")): entry
            for entry in payload.criteria
            if isinstance(entry, dict) and entry.get("key")
        }
        rubric = db.get(Rubric, submission.assignment.rubric_id) if submission.assignment.rubric_id else None
        maxima = {c["key"]: c["max_points"] for c in grading_engine.rubric_criteria(rubric)}
        merged: list[dict[str, Any]] = []
        for crit in result.criteria or []:
            crit = dict(crit)
            if "max_points" not in crit:
                crit["max_points"] = maxima.get(str(crit.get("key")), 0)
            edit = edits.get(str(crit.get("key")))
            if edit is not None:
                if "comment" in edit:
                    crit["comment"] = str(edit.get("comment") or "")
                    touched.append(f"criteria.{crit.get('key')}.comment")
                if "score" in edit:
                    try:
                        score = float(edit.get("score"))
                    except (TypeError, ValueError):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Criterion {crit.get('key')!r} needs a numeric score",
                        ) from None
                    if not math.isfinite(score):
                        raise HTTPException(status_code=400, detail="Scores must be finite numbers")
                    max_points = float(crit.get("max_points") or 0)
                    score = max(0.0, min(score, max_points) if max_points else max(0.0, score))
                    crit["score"] = round(float(score), 2)
                    touched.append(f"criteria.{crit.get('key')}.score")
            merged.append(crit)
        result.criteria = merged
        if (result.mode or "grade") == "feedback":
            # Feedback-only results never carry totals, whatever was edited.
            result.overall_score, result.max_score = 0.0, 0.0
        else:
            result.overall_score = round(sum(float(c.get("score") or 0) for c in merged), 2)
            result.max_score = round(sum(float(c.get("max_points") or 0) for c in merged), 2)

    if payload.misconceptions is not None:
        result.misconceptions = grading_engine.normalize_tags(payload.misconceptions)
        touched.append("misconceptions")
    if payload.strengths is not None:
        result.strengths = grading_engine.normalize_phrases(payload.strengths)
        touched.append("strengths")

    db.add(result)
    if touched:
        review.record_edit(db, submission, touched)
    db.commit()
    db.refresh(result)
    db.refresh(submission)
    if touched:
        # Corrections must replace derived evidence, even when no observations
        # remain. Use deterministic cards here; saving edits never buys AI calls.
        from app import insight
        insight.derive_observations(db, submission, result)
        insight.detect_nudges(db, submission.assignment_id)
        if submission.student_id:
            insight.refresh_card(db, submission.student_id, allow_llm=False)
    return submission_dict(submission)


@router.post("/api/assignments/{assignment_id}/grading/reset")
def reset_grading(assignment_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return submissions stranded in ``grading`` (crash mid-run) to ``pending``."""
    _get_assignment(db, assignment_id)
    stuck = db.scalars(
        select(Submission).where(
            Submission.assignment_id == assignment_id, Submission.status == STATUS_GRADING
        )
    ).all()
    for submission in stuck:
        submission.status = STATUS_PENDING
        submission.error = None
    db.commit()
    return {"assignment_id": assignment_id, "reset": [s.id for s in stuck]}


# --------------------------------------------------------------------------
# page route
# --------------------------------------------------------------------------


@router.get("/grading/{assignment_id}", response_class=HTMLResponse)
def page_grading(assignment_id: int, request: Request, db: Session = Depends(get_db)) -> Any:
    """Template contract: assignment, course, rubric, submissions (+ student, result)."""
    assignment = _get_assignment(db, assignment_id)
    course = db.get(Course, assignment.course_id)
    rubric = db.get(Rubric, assignment.rubric_id) if assignment.rubric_id else None
    skill = db.get(Skill, assignment.skill_id) if assignment.skill_id else None
    submissions = db.scalars(
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.grade_result))
        .where(Submission.assignment_id == assignment_id)
        .order_by(Submission.id)
    ).all()
    students = db.scalars(
        select(Student)
        .where(Student.course_id == assignment.course_id)
        .order_by(Student.student_number)
    ).all()
    from app.ai import modes as modes_mod  # local import: keeps the router light
    from app.routers.compare import candidate_options  # local import: avoids a cycle

    mode = modes_mod.get_mode(getattr(skill, "mode", None)) if skill else None
    return templates.TemplateResponse(
        request,
        "grading.html",
        {
            "assignment": assignment,
            "course": course,
            "rubric": rubric,
            "skill": skill,
            "mode": mode,
            "candidates": candidate_options(db),
            "available_skills": db.scalars(select(Skill).order_by(Skill.name)).all(),
            "available_rubrics": db.scalars(select(Rubric).order_by(Rubric.name)).all(),
            "setup_locked": any(sub.grade_result is not None or sub.status == STATUS_GRADING for sub in submissions),
            "submissions": submissions,
            "students": students,
            "counts": _status_counts(db, assignment_id),
            # Endpoint map consumed by templates/grading.html + static/js/grading.js.
            "api": {
                "submissions": f"/api/assignments/{assignment_id}/grading/status",
                "upload": f"/api/assignments/{assignment_id}/submissions",
                "grade_all": f"/api/assignments/{assignment_id}/grade-all",
                "grade_one": "/api/submissions/{id}/grade",
                "result": "/api/submissions/{id}/result",
                "mapping": f"/api/assignments/{assignment_id}/submissions/mapping",
                "file": "/api/submissions/{id}/file",
            },
        },
    )


@router.get("/assignments/{assignment_id}", response_class=HTMLResponse, include_in_schema=False)
def page_assignment(assignment_id: int, request: Request, db: Session = Depends(get_db)) -> Any:
    return page_grading(assignment_id, request, db)


__all__ = [
    "router",
    "sanitize_filename",
    "detect_media_type",
    "store_submission_file",
    "suggest_student",
    "build_mapping_suggestions",
    "run_grading_job",
    "grade_one",
]


# ==========================================================================
# Increment 1 · PRIVACY MODULE endpoints (appended — nothing above changed)
#
# The scan report shown next to a graded submission ("3 identifiers swapped —
# view"), an on-demand scan the professor can run before grading, and the
# display-layer pass that swaps codes back to real names for the UI.
#
# Everything here is local-only: originals and the code map never go anywhere
# near a provider.
# ==========================================================================

from pydantic import ConfigDict  # noqa: E402 - appended module section


class PrivacyDisplayIn(BaseModel):
    """Text written by a model + the course whose codes it uses."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(default="", max_length=200_000)
    course_id: Optional[int] = None


def _privacy():
    from app.ai import privacy as privacy_guard  # noqa: PLC0415 - lazy import

    return privacy_guard


def _submission_course_id(db: Session, submission: Submission) -> Optional[int]:
    assignment = submission.assignment or db.get(Assignment, submission.assignment_id)
    return getattr(assignment, "course_id", None) if assignment is not None else None


@router.get("/api/submissions/{submission_id}/privacy")
def submission_privacy_scan(
    submission_id: int, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """The Privacy Guard report recorded for this submission's last grading run."""
    privacy_guard = _privacy()
    submission = _get_submission(db, submission_id)
    scan = privacy_guard.scan_dict(privacy_guard.latest_scan(db, submission.id))
    return {
        "submission_id": submission.id,
        "mode": config.privacy_mode(),
        "scan": scan,
        "headline": (scan or {}).get("headline", "Not scanned yet"),
    }


@router.post("/api/submissions/{submission_id}/privacy/scan")
def run_privacy_scan(submission_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Scan a submission now, without grading it (the ``warn`` preview).

    Detection only: the stored file is never rewritten, and no codes are
    allocated — this answers "what would leave this machine?".
    """
    privacy_guard = _privacy()
    submission = _get_submission(db, submission_id)
    raw = None
    if submission.file_path:
        try:
            raw = Path(submission.file_path).read_bytes()
        except OSError as exc:
            raise HTTPException(
                status_code=404, detail=f"The submission file is missing on disk ({exc})."
            ) from exc

    mime = (submission.mime_type or "").lower()
    if mime.startswith("image/"):
        report = privacy_guard.empty_report(privacy_guard.MODE_WARN)
        report["warnings"].append(
            "Cloud grading stops, and the image or document with no extractable text is "
            "not sent. Use the local model or explicitly change privacy mode to continue."
        )
    else:
        text = privacy_guard.submission_text_for_scan(
            {"type": "document", "_text": None}, raw
        )
        if not text.strip():
            report = privacy_guard.empty_report(privacy_guard.MODE_WARN)
            report["warnings"].append(
                "No text layer could be extracted from this submission (it may be a scan)."
            )
        else:
            _, report = privacy_guard.pseudonymize_text(
                db,
                text,
                course_id=_submission_course_id(db, submission),
                mode=privacy_guard.MODE_WARN,
            )
    scan = privacy_guard.record_scan(db, submission.id, report)
    return {
        "submission_id": submission.id,
        "mode": config.privacy_mode(),
        "scan": privacy_guard.scan_dict(scan),
    }


@router.post("/api/privacy/display")
def privacy_display(payload: PrivacyDisplayIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Re-substitute real names into model-written text for the professor.

    Returns both forms: ``raw`` (what the AI saw and wrote — codes intact) and
    ``display`` (names restored), so the UI can show one and put the other on
    the title attribute behind a toggle.
    """
    return _privacy().render_for_display(db, payload.text, payload.course_id)


@router.get("/api/submissions/{submission_id}/result/display")
def submission_result_display(
    submission_id: int, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """One graded submission's feedback with codes swapped back to real names."""
    privacy_guard = _privacy()
    submission = _get_submission(db, submission_id)
    result = submission.grade_result
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Submission {submission_id} has no grade result yet"
        )
    course_id = _submission_course_id(db, submission)
    rendered = privacy_guard.display_result(db, result, course_id)
    rendered["submission_id"] = submission.id
    rendered["course_id"] = course_id
    return rendered


__all__ += [
    "submission_privacy_scan",
    "run_privacy_scan",
    "privacy_display",
    "submission_result_display",
]
