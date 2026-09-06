"""JSON endpoints for the Student Insight layer (Increment 1, Feature A).

    GET  /api/students/{id}/card            the maintained card (built on demand)
    POST /api/students/{id}/card/refresh    "Refresh card" button
    GET  /api/students/{id}/observations     the evidence feed under the card
    GET  /api/courses/{id}/nudges           "Worth a look" panel
    POST /api/nudges/{id}/dismiss           dismiss one nudge

Page routes stay where they already are (the student page belongs to courses,
the course dashboard to courses/analytics); this router is data only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import insight
from app.db import get_db
from app.models import Course, Student

router = APIRouter()


def _get_student(db: Session, student_id: int) -> Student:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    return student


@router.get("/api/students/{student_id}/card")
def get_card(
    student_id: int,
    auto: bool = Query(default=True, description="Build the card if it is missing or stale."),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """The student's card. Rebuilt automatically when observations moved on."""
    student = _get_student(db, student_id)
    card = insight.get_card(db, student_id, auto=auto)
    return {
        "student_id": student.id,
        "label": student.anon_label,
        "card": insight.card_dict(card),
        "stale": insight.card_is_stale(db, student_id, card) if card else True,
        "observation_count": len(insight.observations_for(db, student_id)),
    }


@router.post("/api/students/{student_id}/card/refresh")
def refresh_card(student_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Rebuild the card now (LLM if configured, templates if that fails)."""
    student = _get_student(db, student_id)
    card = insight.refresh_card(db, student_id)
    return {
        "student_id": student.id,
        "label": student.anon_label,
        "card": insight.card_dict(card),
        "stale": False,
    }


@router.get("/api/students/{student_id}/observations")
def list_observations(
    student_id: int,
    limit: int | None = Query(default=None, ge=1, le=500),
    kind: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """The evidence feed: every deterministic fact behind the card."""
    student = _get_student(db, student_id)
    if kind is not None and kind not in insight.KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown observation kind {kind!r}. Known: {', '.join(insight.KINDS)}",
        )
    rows = insight.observations_for(db, student_id, limit=limit)
    if kind:
        rows = [row for row in rows if row.kind == kind]
    return {
        "student_id": student.id,
        "label": student.anon_label,
        "count": len(rows),
        "observations": [insight.observation_dict(row) for row in rows],
        "misconception_state": insight.misconception_states(db, student_id),
    }


@router.get("/api/courses/{course_id}/nudges")
def list_nudges(
    course_id: int,
    include_dismissed: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """"Worth a look": misconceptions several students share on one assignment."""
    course = db.get(Course, course_id)
    if course is None:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    nudges = insight.nudges_for_course(db, course_id, include_dismissed=include_dismissed)
    return {
        "course_id": course.id,
        "count": len(nudges),
        "nudges": [insight.nudge_dict(nudge) for nudge in nudges],
    }


@router.post("/api/nudges/{nudge_id}/dismiss")
def dismiss_nudge(nudge_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    nudge = insight.dismiss_nudge(db, nudge_id)
    if nudge is None:
        raise HTTPException(status_code=404, detail=f"Nudge {nudge_id} not found")
    return {"nudge": insight.nudge_dict(nudge)}


__all__ = ["router"]
