"""Release record + export adapters — how reviewed feedback leaves Agora.

One canonical, per-student record is built from *released* results only
(``app.review``); everything that hands feedback onward is an adapter over
that record. CSV ships first. An LMS-specific sheet, a per-student PDF or an
email adapter can be registered later without touching review or release.

Every row carries the AI disclosure line (``app.terms.AI_DISCLOSURE``) and the
feedback text is rendered through the privacy display layer, so codes such
as ``Person-A`` are turned back into the names the student would recognise.
The code book itself never leaves the machine — only the rendered text does,
and only for the student it belongs to.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app import review, terms
from app.models import Assignment, Course, GradeResult, Rubric, Submission


class ExportError(ValueError):
    """Nothing to export, or an unknown format."""


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def _display(db: Session, text: str, course_id: Optional[int]) -> str:
    """Names restored for the student; falls back to the raw text if the guard is off."""
    if not text:
        return ""
    try:
        from app.ai import privacy as privacy_guard  # lazy: keeps this module light

        return privacy_guard.render_for_display(db, text, course_id)["display"]
    except Exception:  # noqa: BLE001 - never let display polish block an export
        return text


def _criterion_titles(rubric: Rubric | None) -> dict[str, str]:
    titles: dict[str, str] = {}
    for crit in (getattr(rubric, "criteria", None) or []):
        if isinstance(crit, dict) and crit.get("key"):
            titles[str(crit["key"])] = str(crit.get("title") or crit["key"])
    return titles


def _row(
    db: Session,
    submission: Submission,
    result: GradeResult,
    titles: dict[str, str],
    course_id: Optional[int],
) -> dict[str, Any]:
    student = submission.student
    criteria: list[dict[str, Any]] = []
    for crit in result.criteria or []:
        if not isinstance(crit, dict):
            continue
        key = str(crit.get("key") or "")
        criteria.append(
            {
                "key": key,
                "title": titles.get(key, key.replace("_", " ").title()),
                "score": crit.get("score"),
                "max_points": crit.get("max_points"),
                "comment": _display(db, str(crit.get("comment") or ""), course_id),
                "manual": bool(crit.get("manual")),
            }
        )
    scored = (result.mode or "grade") != "feedback" and bool(result.max_score)
    return {
        "submission_id": submission.id,
        "student_id": submission.student_id,
        "student_number": student.student_number if student else None,
        "student_name": student.name if student else "",
        "email": (student.email if student else "") or "",
        "mode": result.mode or "grade",
        "scored": scored,
        "score": result.overall_score if scored else None,
        "max_score": result.max_score if scored else None,
        "percent": result.percentage if scored else None,
        "feedback": _display(db, result.summary_feedback or "", course_id),
        "criteria": criteria,
        "strengths": list(result.strengths or []),
        "misconceptions": list(result.misconceptions or []),
        "model": result.model,
        "reviewed_at": review.review_dict(result)["seen_at"],
        "edits": len(result.edit_log or []),
        "released_at": review.review_dict(result)["released_at"],
        "disclosure": terms.AI_DISCLOSURE,
    }


def release_record(db: Session, assignment: Assignment) -> dict[str, Any]:
    """Every *released* result on the assignment, in student-number order."""
    course = db.get(Course, assignment.course_id)
    rubric = db.get(Rubric, assignment.rubric_id) if assignment.rubric_id else None
    titles = _criterion_titles(rubric)
    subs = db.scalars(
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.grade_result))
        .where(Submission.assignment_id == assignment.id)
    ).all()
    rows: list[dict[str, Any]] = []
    for s in subs:
        r = s.grade_result
        if r is None or r.released_at is None or r.withheld_at is not None:
            continue
        if review.manual_scores_missing(r) or r.approved_at is None or s.status != "graded":
            continue
        rows.append(_row(db, s, r, titles, assignment.course_id))
    rows.sort(key=lambda row: (row["student_number"] is None, row["student_number"] or 0))
    return {
        "assignment": {"id": assignment.id, "name": assignment.name},
        "course": {"id": course.id, "name": course.name, "term": course.term} if course else None,
        "rubric": {"id": rubric.id, "name": rubric.name} if rubric else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclosure": terms.AI_DISCLOSURE,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Exporter:
    id: str
    label: str
    description: str
    mime: str
    suffix: str
    render: Callable[[dict[str, Any]], bytes]


EXPORTERS: dict[str, Exporter] = {}


def register_exporter(exporter: Exporter) -> Exporter:
    EXPORTERS[exporter.id] = exporter
    return exporter


def list_exporters() -> list[dict[str, str]]:
    return [
        {"id": e.id, "label": e.label, "description": e.description, "mime": e.mime}
        for e in EXPORTERS.values()
    ]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value or "").strip("-").lower()
    return slug or "assignment"


def _fmt_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}"


def _criteria_text(criteria: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for crit in criteria:
        head = crit["title"]
        if crit.get("max_points"):
            head += f" {_fmt_number(crit.get('score'))}/{_fmt_number(crit.get('max_points'))}"
        comment = (crit.get("comment") or "").strip()
        parts.append(f"{head}: {comment}" if comment else head)
    return " | ".join(parts)


def _csv_text(value: Any) -> str:
    text = str(value or "")
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text


def _render_csv(record: dict[str, Any]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "student_number",
            "student_name",
            "email",
            "assignment",
            "score",
            "max_score",
            "percent",
            "feedback",
            "criteria",
            "released_at",
            "ai_disclosure",
        ]
    )
    for row in record["rows"]:
        feedback = (row["feedback"] or "").strip()
        feedback = f"{feedback}\n\n{row['disclosure']}" if feedback else row["disclosure"]
        writer.writerow(
            [
                row["student_number"] if row["student_number"] is not None else "",
                _csv_text(row["student_name"]),
                _csv_text(row["email"]),
                _csv_text(record["assignment"]["name"]),
                _fmt_number(row["score"]),
                _fmt_number(row["max_score"]),
                _fmt_number(row["percent"]),
                _csv_text(feedback),
                _csv_text(_criteria_text(row["criteria"])),
                row["released_at"] or "",
                row["disclosure"],
            ]
        )
    return buffer.getvalue().encode("utf-8-sig")


register_exporter(
    Exporter(
        id="csv",
        label="CSV",
        description="One row per student: score, feedback, per-criterion comments, disclosure. "
        "Opens in any spreadsheet and imports into most gradebooks.",
        mime="text/csv; charset=utf-8",
        suffix=".csv",
        render=_render_csv,
    )
)


def export(db: Session, assignment: Assignment, fmt: str = "csv") -> tuple[str, bytes, str]:
    """Render the release record with one adapter → (filename, bytes, mime)."""
    exporter = EXPORTERS.get((fmt or "csv").strip().lower())
    if exporter is None:
        raise ExportError(
            f"Unknown export format {fmt!r}. Known: {', '.join(sorted(EXPORTERS))}"
        )
    record = release_record(db, assignment)
    if not record["rows"]:
        raise ExportError("Nothing has been released for this assignment yet.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"{_slug(assignment.name)}-feedback-{stamp}{exporter.suffix}"
    return filename, exporter.render(record), exporter.mime


__all__ = [
    "ExportError",
    "Exporter",
    "EXPORTERS",
    "register_exporter",
    "list_exporters",
    "release_record",
    "export",
]
