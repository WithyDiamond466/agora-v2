"""Review record, per-assignment release, and export — the JSON API.

Everything the grading page needs to show a human reviewed AI output before
a student saw it. Release and export are gated on the terms of use.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select

from app import release as release_mod
from app import review, terms
from app.db import get_db
from app.models import Assignment, Submission

router = APIRouter()


class WithholdIn(BaseModel):
    withheld: bool = True


class ReleaseIn(BaseModel):
    #: Also release results the professor never opened (recorded as such).
    include_unseen: bool = False
    #: Narrow the release to these submissions.
    submission_ids: Optional[list[int]] = Field(default=None)


def _submission(db: Session, submission_id: int) -> Submission:
    sub = db.scalars(
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.grade_result))
        .where(Submission.id == submission_id)
    ).first()
    if sub is None:
        raise HTTPException(status_code=404, detail=f"Submission {submission_id} not found")
    return sub


def _assignment(db: Session, assignment_id: int) -> Assignment:
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise HTTPException(status_code=404, detail=f"Assignment {assignment_id} not found")
    return assignment


def _terms_or_409(db: Session, action: str) -> None:
    try:
        terms.require_accepted(db, action)
    except terms.TermsNotAccepted as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _review_payload(sub: Submission) -> dict[str, Any]:
    return {
        "submission_id": sub.id,
        "assignment_id": sub.assignment_id,
        "review": review.review_dict(sub.grade_result),
    }


# --------------------------------------------------------------------------
# per submission
# --------------------------------------------------------------------------


@router.post("/api/submissions/{submission_id}/review/seen")
def mark_seen(submission_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """The professor opened this result. Idempotent."""
    sub = _submission(db, submission_id)
    try:
        review.mark_seen(db, sub)
    except review.ReviewError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _review_payload(sub)


@router.post("/api/submissions/{submission_id}/review/approve")
def approve(submission_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    sub = _submission(db, submission_id)
    try:
        review.approve(db, sub)
    except review.ReviewError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _review_payload(sub)


@router.post("/api/submissions/{submission_id}/review/withhold")
def withhold(
    submission_id: int, payload: WithholdIn | None = None, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Withhold (``{"withheld": true}``) or lift a withhold on one result."""
    sub = _submission(db, submission_id)
    payload = payload or WithholdIn()
    try:
        review.set_withheld(db, sub, payload.withheld)
    except review.ReviewError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    data = _review_payload(sub)
    data["message"] = "Withheld — this result will not be released." if payload.withheld else (
        "Withhold lifted."
    )
    return data


@router.get("/api/submissions/{submission_id}/review")
def review_state(submission_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    sub = _submission(db, submission_id)
    data = _review_payload(sub)
    data["events"] = review.events_for(db, sub.id)
    return data


# --------------------------------------------------------------------------
# per assignment
# --------------------------------------------------------------------------


@router.get("/api/assignments/{assignment_id}/release")
def release_summary(assignment_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    assignment = _assignment(db, assignment_id)
    data = review.release_summary(db, assignment)
    data["terms"] = terms.status(db)
    data["exporters"] = release_mod.list_exporters()
    return data


@router.post("/api/assignments/{assignment_id}/release")
def release_assignment(
    assignment_id: int, payload: ReleaseIn | None = None, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """One click: release every reviewed result (unopened ones held unless asked)."""
    assignment = _assignment(db, assignment_id)
    _terms_or_409(db, "releasing feedback to students")
    payload = payload or ReleaseIn()
    data = review.release(
        db,
        assignment,
        include_unseen=payload.include_unseen,
        submission_ids=payload.submission_ids,
    )
    released = len(data["released"])
    held = len(data["held"])
    data["message"] = (
        f"Released {released} result{'' if released == 1 else 's'}"
        + (f", held {held}" if held else "")
        + "."
    )
    return data


@router.get("/api/assignments/{assignment_id}/release/events")
def release_events(assignment_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    assignment = _assignment(db, assignment_id)
    return {"assignment_id": assignment.id, "events": review.events_for_assignment(db, assignment.id)}


@router.get("/api/exporters")
def exporters() -> dict[str, Any]:
    return {"exporters": release_mod.list_exporters()}


@router.get("/api/assignments/{assignment_id}/release/record")
def release_record(assignment_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """The canonical record every exporter renders (released rows only)."""
    assignment = _assignment(db, assignment_id)
    _terms_or_409(db, "exporting feedback")
    return release_mod.release_record(db, assignment)


@router.get("/api/assignments/{assignment_id}/export")
def export_assignment(
    assignment_id: int,
    format: str = Query(default="csv", max_length=40),
    db: Session = Depends(get_db),
) -> Response:
    assignment = _assignment(db, assignment_id)
    _terms_or_409(db, "exporting feedback")
    try:
        filename, payload, mime = release_mod.export(db, assignment, format)
    except release_mod.ExportError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(
        content=payload,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = ["router"]
