"""JSON endpoints for the analytics dashboards + the analytics page route.

All chart data is fetched by the frontend from these endpoints:
    GET /api/courses/{course_id}/analytics/overview
    GET /api/courses/{course_id}/analytics/criteria
    GET /api/courses/{course_id}/analytics/misconceptions
    GET /api/courses/{course_id}/analytics/students/{student_id}/timeline
    GET /api/students/{student_id}/analytics/timeline   (same payload, alias)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import analytics as analytics_mod
from app import config
from app.db import get_db
from app.models import Assignment, Course, Student

router = APIRouter()
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


def _call(fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/courses/{course_id}/analytics/overview")
def overview(course_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Per-assignment score distributions and averages."""
    return _call(analytics_mod.course_overview, db, course_id)


@router.get("/api/courses/{course_id}/analytics/criteria")
def criteria(course_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Per-criterion averages across every assignment in the course."""
    return _call(analytics_mod.criteria_breakdown, db, course_id)


@router.get("/api/courses/{course_id}/analytics/misconceptions")
def misconceptions(
    course_id: int,
    # `Counter.most_common(0)` returns nothing and a negative count raises, so
    # 0/negative are rejected at validation time; omit `limit` for all tags.
    limit: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Misconception tag counts, most frequent first."""
    return _call(analytics_mod.misconception_counts, db, course_id, limit)


@router.get("/api/courses/{course_id}/analytics/students/{student_id}/timeline")
def student_timeline(
    course_id: int, student_id: int, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Scores over time, weak criteria and recurring misconceptions."""
    student = db.get(Student, student_id)
    if student is None or student.course_id != course_id:
        raise HTTPException(
            status_code=404, detail=f"Student {student_id} not found in course {course_id}"
        )
    return _call(analytics_mod.student_timeline, db, student_id)


@router.get("/api/students/{student_id}/analytics/timeline")
def student_timeline_alias(student_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _call(analytics_mod.student_timeline, db, student_id)


@router.get("/api/courses/{course_id}/analytics/summary")
def summary(course_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Compact roll-up (used by course pages and the chat assistant)."""
    return _call(analytics_mod.course_quick_stats, db, course_id)


@router.get("/courses/{course_id}/analytics", response_class=HTMLResponse)
def page_analytics(course_id: int, request: Request, db: Session = Depends(get_db)) -> Any:
    course = db.get(Course, course_id)
    if course is None:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    assignments = db.scalars(
        select(Assignment).where(Assignment.course_id == course_id).order_by(Assignment.id)
    ).all()
    students = db.scalars(
        select(Student).where(Student.course_id == course_id).order_by(Student.student_number)
    ).all()
    return templates.TemplateResponse(
        request,
        "analytics.html",
        {
            "course": course,
            "assignments": assignments,
            "students": students,
            # Charts fetch their data from these endpoints client-side.
            "api": {
                "overview": f"/api/courses/{course_id}/analytics/overview",
                "criteria": f"/api/courses/{course_id}/analytics/criteria",
                "misconceptions": f"/api/courses/{course_id}/analytics/misconceptions",
                "timeline": f"/api/courses/{course_id}/analytics/students",
            },
        },
    )


__all__ = ["router"]
