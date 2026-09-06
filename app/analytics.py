"""Aggregation queries behind the analytics dashboards.

Everything here returns plain JSON-safe dicts so the router can hand them
straight to the frontend charts. Rows come from `GradeResult`, which stores
per-criterion scores and misconception tags as JSON — SQL does the joining and
filtering, Python does the JSON-shaped roll-ups.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models import (
    Assignment,
    Course,
    GradeResult,
    Rubric,
    Student,
    Submission,
)

#: Score buckets used by every distribution chart (percent of max).
BUCKETS: list[tuple[str, float, float]] = [
    ("0-59", 0.0, 59.999999),
    ("60-69", 60.0, 69.999999),
    ("70-79", 70.0, 79.999999),
    ("80-89", 80.0, 89.999999),
    ("90-100", 90.0, 100.0),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _pct(score: float, max_score: float) -> Optional[float]:
    if not max_score:
        return None
    return round(100.0 * float(score) / float(max_score), 1)


def _round(value: Optional[float], places: int = 1) -> Optional[float]:
    return None if value is None else round(float(value), places)


def _bucketize(percents: Iterable[float]) -> list[dict[str, Any]]:
    counts = {label: 0 for label, _, _ in BUCKETS}
    for pct in percents:
        for label, low, high in BUCKETS:
            if low <= pct <= high:
                counts[label] += 1
                break
    return [
        {"label": label, "min": low, "max": round(high), "count": counts[label]}
        for label, low, high in BUCKETS
    ]


def graded_rows(db: Session, course_id: int, assignment_id: int | None = None):
    """Every graded result for a course, with submission/assignment loaded."""
    stmt = (
        select(GradeResult)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .join(Assignment, Submission.assignment_id == Assignment.id)
        .where(Assignment.course_id == course_id)
        .options(selectinload(GradeResult.submission).selectinload(Submission.assignment))
        .order_by(Assignment.id, GradeResult.id)
    )
    if assignment_id is not None:
        stmt = stmt.where(Assignment.id == assignment_id)
    return list(db.scalars(stmt).all())


def _rubric_titles(db: Session, course_id: int) -> dict[str, dict[str, Any]]:
    """key -> {title, max_points} across every rubric used by the course."""
    rubrics = db.scalars(
        select(Rubric)
        .join(Assignment, Assignment.rubric_id == Rubric.id)
        .where(Assignment.course_id == course_id)
        .distinct()
    ).all()
    out: dict[str, dict[str, Any]] = {}
    for rubric in rubrics:
        for crit in rubric.criteria or []:
            key = str(crit.get("key") or "").strip()
            if not key:
                continue
            out.setdefault(
                key,
                {
                    "title": crit.get("title") or key.replace("_", " ").title(),
                    "max_points": float(crit.get("max_points") or 0),
                },
            )
    return out


def _criterion_max(entry: dict[str, Any], fallback: dict[str, Any]) -> float:
    for candidate in (entry.get("max_points"), fallback.get("max_points")):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def normalize_tag(tag: Any) -> str:
    return str(tag or "").strip().lower()


# --------------------------------------------------------------------------
# course overview
# --------------------------------------------------------------------------


def course_overview(
    db: Session, course_id: int, rows: list[GradeResult] | None = None
) -> dict[str, Any]:
    """Per-assignment score distributions + averages, plus course totals.

    ``rows`` lets a caller that already loaded the course's graded results
    (see ``course_quick_stats``) reuse them instead of re-running the join.
    """
    course = db.get(Course, course_id)
    if course is None:
        raise LookupError(f"course {course_id} not found")

    assignments = db.scalars(
        select(Assignment).where(Assignment.course_id == course_id).order_by(Assignment.id)
    ).all()
    student_count = (
        db.scalar(select(func.count(Student.id)).where(Student.course_id == course_id)) or 0
    )

    if rows is None:
        rows = graded_rows(db, course_id)
    by_assignment: dict[int, list[GradeResult]] = defaultdict(list)
    for row in rows:
        by_assignment[row.submission.assignment_id].append(row)

    submission_totals: dict[int, int] = dict(
        db.execute(
            select(Submission.assignment_id, func.count(Submission.id))
            .join(Assignment, Submission.assignment_id == Assignment.id)
            .where(Assignment.course_id == course_id)
            .group_by(Submission.assignment_id)
        ).all()
    )

    assignment_blocks: list[dict[str, Any]] = []
    all_percents: list[float] = []

    for assignment in assignments:
        results = by_assignment.get(assignment.id, [])
        percents = [
            p
            for p in (r.percentage for r in results)
            if p is not None
        ]
        all_percents.extend(percents)
        scores = [float(r.overall_score) for r in results if r.percentage is not None]
        max_score = max((float(r.max_score) for r in results), default=0.0)

        assignment_blocks.append(
            {
                "assignment_id": assignment.id,
                "name": assignment.name,
                "due_date": assignment.due_date.isoformat() if assignment.due_date else None,
                "submission_count": submission_totals.get(assignment.id, 0),
                "graded_count": len(results),
                "max_score": max_score,
                "average_score": _round(mean(scores)) if scores else None,
                "average_percent": _round(mean(percents)) if percents else None,
                "median_percent": _round(median(percents)) if percents else None,
                "high_percent": _round(max(percents)) if percents else None,
                "low_percent": _round(min(percents)) if percents else None,
                "distribution": _bucketize(percents),
            }
        )

    return {
        "course": {"id": course.id, "name": course.name, "term": course.term},
        "totals": {
            "students": student_count,
            "assignments": len(assignments),
            "submissions": int(sum(submission_totals.values())),
            "graded": len(rows),
            "average_percent": _round(mean(all_percents)) if all_percents else None,
        },
        "assignments": assignment_blocks,
        # Convenience series for the trend chart (assignment order = x axis).
        "trend": [
            {
                "assignment_id": block["assignment_id"],
                "name": block["name"],
                "average_percent": block["average_percent"],
            }
            for block in assignment_blocks
        ],
        "distribution_buckets": [label for label, _, _ in BUCKETS],
    }


# --------------------------------------------------------------------------
# criteria
# --------------------------------------------------------------------------


def criteria_breakdown(
    db: Session, course_id: int, rows: list[GradeResult] | None = None
) -> dict[str, Any]:
    """Per-criterion averages across every graded assignment in the course."""
    course = db.get(Course, course_id)
    if course is None:
        raise LookupError(f"course {course_id} not found")

    titles = _rubric_titles(db, course_id)
    if rows is None:
        rows = graded_rows(db, course_id)

    scores: dict[str, list[float]] = defaultdict(list)
    percents: dict[str, list[float]] = defaultdict(list)
    maxes: dict[str, float] = {}
    per_assignment: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    assignment_names: dict[int, str] = {}

    for row in rows:
        assignment = row.submission.assignment
        assignment_names[assignment.id] = assignment.name
        for entry in row.criteria or []:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key") or "").strip()
            if not key:
                continue
            try:
                score = float(entry.get("score"))
            except (TypeError, ValueError):
                continue
            fallback = titles.get(key, {})
            max_points = _criterion_max(entry, fallback)
            scores[key].append(score)
            maxes[key] = max(maxes.get(key, 0.0), max_points)
            if max_points:
                pct = 100.0 * score / max_points
                percents[key].append(pct)
                per_assignment[key][assignment.id].append(pct)

    criteria = []
    for key, values in scores.items():
        meta = titles.get(key, {})
        criteria.append(
            {
                "key": key,
                "title": meta.get("title") or key.replace("_", " ").title(),
                "max_points": maxes.get(key, 0.0),
                "count": len(values),
                "average_score": _round(mean(values)),
                "average_percent": _round(mean(percents[key])) if percents.get(key) else None,
                "by_assignment": [
                    {
                        "assignment_id": assignment_id,
                        "name": assignment_names.get(assignment_id, ""),
                        "average_percent": _round(mean(vals)),
                    }
                    for assignment_id, vals in sorted(per_assignment[key].items())
                ],
            }
        )

    # Weakest first — that is what a professor is scanning for.
    criteria.sort(key=lambda c: (c["average_percent"] is None, c["average_percent"]))

    return {
        "course": {"id": course.id, "name": course.name},
        "criteria": criteria,
        "weakest": criteria[0]["key"] if criteria else None,
    }


# --------------------------------------------------------------------------
# misconceptions
# --------------------------------------------------------------------------


def misconception_counts(
    db: Session,
    course_id: int,
    limit: int | None = None,
    rows: list[GradeResult] | None = None,
) -> dict[str, Any]:
    """Misconception tag frequency across the course (tags grouped exactly)."""
    course = db.get(Course, course_id)
    if course is None:
        raise LookupError(f"course {course_id} not found")

    if rows is None:
        rows = graded_rows(db, course_id)
    counts: Counter[str] = Counter()
    students: dict[str, set[int]] = defaultdict(set)
    assignments: dict[str, set[int]] = defaultdict(set)

    for row in rows:
        submission = row.submission
        seen: set[str] = set()
        for raw in row.misconceptions or []:
            tag = normalize_tag(raw)
            if not tag or tag in seen:
                continue
            seen.add(tag)
            counts[tag] += 1
            if submission.student_id:
                students[tag].add(submission.student_id)
            assignments[tag].add(submission.assignment_id)

    items = [
        {
            "tag": tag,
            "count": count,
            "student_count": len(students[tag]),
            "assignment_count": len(assignments[tag]),
        }
        for tag, count in counts.most_common(limit)
    ]

    return {
        "course": {"id": course.id, "name": course.name},
        "misconceptions": items,
        "total_tags": len(counts),
        "total_mentions": int(sum(counts.values())),
        "graded_results": len(rows),
    }


# --------------------------------------------------------------------------
# per student
# --------------------------------------------------------------------------


def student_timeline(db: Session, student_id: int) -> dict[str, Any]:
    """Scores over time, weak criteria, and recurring misconceptions."""
    student = db.get(Student, student_id)
    if student is None:
        raise LookupError(f"student {student_id} not found")
    course = db.get(Course, student.course_id)

    rows = db.scalars(
        select(GradeResult)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .join(Assignment, Submission.assignment_id == Assignment.id)
        .where(Submission.student_id == student_id)
        .options(selectinload(GradeResult.submission).selectinload(Submission.assignment))
        .order_by(Assignment.id, GradeResult.id)
    ).all()

    titles = _rubric_titles(db, student.course_id)
    points: list[dict[str, Any]] = []
    crit_percents: dict[str, list[float]] = defaultdict(list)
    tags: Counter[str] = Counter()

    for row in rows:
        assignment = row.submission.assignment
        points.append(
            {
                "assignment_id": assignment.id,
                "name": assignment.name,
                "graded_at": row.created_at.isoformat() if row.created_at else None,
                "score": float(row.overall_score),
                "max_score": float(row.max_score),
                "percent": row.percentage,
                "incomplete": row.incomplete,
                "misconceptions": [normalize_tag(t) for t in (row.misconceptions or [])],
            }
        )
        for entry in row.criteria or []:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key") or "").strip()
            if not key:
                continue
            try:
                score = float(entry.get("score"))
            except (TypeError, ValueError):
                continue
            max_points = _criterion_max(entry, titles.get(key, {}))
            if max_points:
                crit_percents[key].append(100.0 * score / max_points)
        for raw in row.misconceptions or []:
            tag = normalize_tag(raw)
            if tag:
                tags[tag] += 1

    percents = [p["percent"] for p in points if p["percent"] is not None]
    criteria = sorted(
        (
            {
                "key": key,
                "title": titles.get(key, {}).get("title") or key.replace("_", " ").title(),
                "average_percent": _round(mean(vals)),
                "count": len(vals),
            }
            for key, vals in crit_percents.items()
        ),
        key=lambda c: c["average_percent"],
    )

    return {
        "student": {
            "id": student.id,
            "name": student.name,
            "student_number": student.student_number,
            "course_id": student.course_id,
        },
        "course": {"id": course.id, "name": course.name} if course else None,
        "timeline": points,
        "graded_count": len(points),
        "average_percent": _round(mean(percents)) if percents else None,
        "best_percent": _round(max(percents)) if percents else None,
        "worst_percent": _round(min(percents)) if percents else None,
        "criteria": criteria,
        "weak_criteria": [c for c in criteria if (c["average_percent"] or 0) < 75][:5],
        "misconceptions": [
            {"tag": tag, "count": count} for tag, count in tags.most_common()
        ],
    }


def course_quick_stats(db: Session, course_id: int) -> dict[str, Any]:
    """Small summary used by course pages and the chat assistant.

    The three roll-ups read the same result set, so it is loaded once here.
    """
    if db.get(Course, course_id) is None:
        raise LookupError(f"course {course_id} not found")
    rows = graded_rows(db, course_id)
    overview = course_overview(db, course_id, rows)
    misc = misconception_counts(db, course_id, limit=3, rows=rows)
    crits = criteria_breakdown(db, course_id, rows)
    return {
        "course": overview["course"],
        "totals": overview["totals"],
        "top_misconceptions": misc["misconceptions"],
        "weakest_criteria": crits["criteria"][:3],
    }


__all__ = [
    "BUCKETS",
    "graded_rows",
    "course_overview",
    "criteria_breakdown",
    "misconception_counts",
    "student_timeline",
    "course_quick_stats",
    "normalize_tag",
]
