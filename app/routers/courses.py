"""Courses, students, roster CSV import, and the page routes that render them."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app import analytics as analytics_mod
from app import config
from app import insight as insight_mod
from app.db import get_db
from app.models import Assignment, Course, GradeResult, Observation, Student, Submission
from app.storage import ReversibleDelete, discard_all, reraise_delete_failure, reversible_delete

router = APIRouter()
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class CourseIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    term: Optional[str] = None


class CourseUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    term: Optional[str] = None


class StudentIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: Optional[str] = None
    student_number: Optional[int] = None


class StudentUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    email: Optional[str] = None
    student_number: Optional[int] = None


def course_dict(course: Course) -> dict[str, Any]:
    return {
        "id": course.id,
        "name": course.name,
        "term": course.term,
        "created_at": course.created_at.isoformat() if course.created_at else None,
    }


def student_dict(student: Student) -> dict[str, Any]:
    return {
        "id": student.id,
        "course_id": student.course_id,
        "name": student.name,
        "student_number": student.student_number,
        "email": student.email,
        "anon_label": student.anon_label,
    }


def _get_course(db: Session, course_id: int) -> Course:
    course = db.get(Course, course_id)
    if course is None:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    return course


def _get_student(db: Session, student_id: int) -> Student:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    return student


def next_student_number(db: Session, course_id: int) -> int:
    current = db.scalar(
        select(func.max(Student.student_number)).where(Student.course_id == course_id)
    )
    return int(current or 0) + 1


# --------------------------------------------------------------------------
# course API
# --------------------------------------------------------------------------


@router.get("/api/courses")
def list_courses(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    student_counts = (
        select(Student.course_id, func.count(Student.id).label("student_count"))
        .group_by(Student.course_id)
        .subquery()
    )
    assignment_counts = (
        select(Assignment.course_id, func.count(Assignment.id).label("assignment_count"))
        .group_by(Assignment.course_id)
        .subquery()
    )
    rows = db.execute(
        select(
            Course,
            func.coalesce(student_counts.c.student_count, 0),
            func.coalesce(assignment_counts.c.assignment_count, 0),
        )
        .outerjoin(student_counts, student_counts.c.course_id == Course.id)
        .outerjoin(assignment_counts, assignment_counts.c.course_id == Course.id)
        .order_by(Course.created_at.desc(), Course.id.desc())
    ).all()
    out = []
    for course, student_count, assignment_count in rows:
        data = course_dict(course)
        data["student_count"] = student_count
        data["assignment_count"] = assignment_count
        out.append(data)
    return out


@router.post("/api/courses", status_code=201)
def create_course(payload: CourseIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    course = Course(name=payload.name.strip(), term=(payload.term or "").strip() or None)
    db.add(course)
    db.commit()
    db.refresh(course)
    return course_dict(course)


@router.get("/api/courses/{course_id}")
def get_course(course_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    course = _get_course(db, course_id)
    data = course_dict(course)
    data["students"] = [student_dict(s) for s in course.students]
    data["assignments"] = [
        {"id": a.id, "name": a.name, "due_date": a.due_date.isoformat() if a.due_date else None}
        for a in course.assignments
    ]
    return data


@router.patch("/api/courses/{course_id}")
def update_course(
    course_id: int, payload: CourseUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    course = _get_course(db, course_id)
    if payload.name is not None:
        course.name = payload.name.strip()
    if payload.term is not None:
        course.term = payload.term.strip() or None
    db.commit()
    db.refresh(course)
    return course_dict(course)


@router.delete("/api/courses/{course_id}")
def delete_course(course_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    course = _get_course(db, course_id)
    assignment_ids = db.scalars(
        select(Assignment.id).where(Assignment.course_id == course_id)
    ).all()
    tokens: list[ReversibleDelete] = []
    try:
        for assignment_id in assignment_ids:
            token = reversible_delete(
                Path(config.UPLOAD_DIR) / str(assignment_id),
                config.UPLOAD_DIR,
                direct_child=True,
            )
            if token is not None:
                tokens.append(token)
        db.delete(course)
        db.commit()
    except Exception as exc:
        reraise_delete_failure(exc, db.rollback, tokens)
    discard_all(tokens)
    return {"deleted": course_id}


# --------------------------------------------------------------------------
# student API
# --------------------------------------------------------------------------


@router.get("/api/courses/{course_id}/students")
def list_students(course_id: int, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    _get_course(db, course_id)
    students = db.scalars(
        select(Student).where(Student.course_id == course_id).order_by(Student.student_number)
    ).all()
    return [student_dict(s) for s in students]


@router.post("/api/courses/{course_id}/students", status_code=201)
def create_student(
    course_id: int, payload: StudentIn, db: Session = Depends(get_db)
) -> dict[str, Any]:
    _get_course(db, course_id)
    number = payload.student_number or next_student_number(db, course_id)
    clash = db.scalars(
        select(Student).where(
            Student.course_id == course_id, Student.student_number == number
        )
    ).first()
    if clash is not None:
        raise HTTPException(
            status_code=409, detail=f"Student number {number} already used in this course"
        )
    student = Student(
        course_id=course_id,
        name=payload.name.strip(),
        email=(payload.email or "").strip() or None,
        student_number=number,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student_dict(student)


@router.get("/api/students/{student_id}")
def get_student(student_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    return student_dict(_get_student(db, student_id))


@router.patch("/api/students/{student_id}")
def update_student(
    student_id: int, payload: StudentUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    student = _get_student(db, student_id)
    if payload.name is not None:
        student.name = payload.name.strip()
    if payload.email is not None:
        student.email = payload.email.strip() or None
    if payload.student_number is not None and payload.student_number != student.student_number:
        clash = db.scalars(
            select(Student).where(
                Student.course_id == student.course_id,
                Student.student_number == payload.student_number,
                Student.id != student.id,
            )
        ).first()
        if clash is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Student number {payload.student_number} already used in this course",
            )
        student.student_number = payload.student_number
    db.commit()
    db.refresh(student)
    return student_dict(student)


@router.delete("/api/students/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    student = _get_student(db, student_id)
    assignment_ids = db.scalars(
        select(Observation.assignment_id)
        .where(
            Observation.student_id == student_id,
            Observation.assignment_id.is_not(None),
        )
        .distinct()
    ).all()
    db.execute(
        update(Submission).where(Submission.student_id == student_id).values(student_id=None)
    )
    db.flush()
    db.expire_all()
    student = db.get(Student, student_id)
    db.delete(student)
    db.flush()
    for assignment_id in assignment_ids:
        insight_mod.detect_nudges(db, assignment_id, commit=False)
    db.commit()
    return {"deleted": student_id}


# --------------------------------------------------------------------------
# roster CSV import (really parses the file — see SPEC key flows)
# --------------------------------------------------------------------------

_NAME_HEADERS = {"name", "student", "student name", "full name", "fullname"}
_EMAIL_HEADERS = {"email", "e-mail", "email address", "mail"}
_FIRST_HEADERS = {"first", "first name", "firstname", "given name"}
_LAST_HEADERS = {"last", "last name", "lastname", "surname", "family name"}
_NUMBER_HEADERS = {"number", "student number", "student #", "num", "id", "student id"}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _norm(value: str) -> str:
    return (value or "").strip().strip("﻿").lower()


def parse_roster_csv(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a roster CSV into ``[{name, email, student_number}]`` rows.

    Accepts, in order of preference:
      * a header row with ``name`` (+ optional ``email`` / ``student number``)
      * a header row with ``first``/``last`` name columns
      * no header at all — first column is the name, an @-looking column is email

    Returns (rows, warnings). Blank lines and rows without a name are skipped.
    """
    warnings: list[str] = []
    text = text.lstrip("﻿")
    if not text.strip():
        return [], ["The CSV file was empty."]

    try:
        dialect: Any = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(text), dialect)
    records = [row for row in reader if any((cell or "").strip() for cell in row)]
    if not records:
        return [], ["The CSV file had no usable rows."]

    header = [_norm(cell) for cell in records[0]]
    has_header = bool(
        set(header) & (_NAME_HEADERS | _EMAIL_HEADERS | _FIRST_HEADERS | _LAST_HEADERS)
    )

    name_idx = email_idx = first_idx = last_idx = number_idx = None
    if has_header:
        for idx, cell in enumerate(header):
            if name_idx is None and cell in _NAME_HEADERS:
                name_idx = idx
            elif email_idx is None and cell in _EMAIL_HEADERS:
                email_idx = idx
            elif first_idx is None and cell in _FIRST_HEADERS:
                first_idx = idx
            elif last_idx is None and cell in _LAST_HEADERS:
                last_idx = idx
            elif number_idx is None and cell in _NUMBER_HEADERS:
                number_idx = idx
        body = records[1:]
        if name_idx is None and first_idx is None and last_idx is None:
            name_idx = 0
            warnings.append("No name column found; using the first column as the name.")
    else:
        name_idx = 0
        body = records
        warnings.append("No header row detected; treating column 1 as name.")

    def cell(row: list[str], idx: Optional[int]) -> str:
        if idx is None or idx >= len(row):
            return ""
        return (row[idx] or "").strip()

    rows: list[dict[str, Any]] = []
    for line_no, row in enumerate(body, start=2 if has_header else 1):
        if name_idx is not None:
            name = cell(row, name_idx)
        else:
            name = " ".join(p for p in (cell(row, first_idx), cell(row, last_idx)) if p)
        if not name and first_idx is not None:
            name = " ".join(p for p in (cell(row, first_idx), cell(row, last_idx)) if p)

        email = cell(row, email_idx)
        if not email:
            # Header-less files: pick up anything that looks like an address.
            for value in row:
                candidate = (value or "").strip()
                if _EMAIL_RE.match(candidate):
                    email = candidate
                    break
        if email and not _EMAIL_RE.match(email):
            warnings.append(f"Row {line_no}: ignoring malformed email {email!r}.")
            email = ""

        number: Optional[int] = None
        raw_number = cell(row, number_idx)
        if raw_number:
            try:
                number = int(float(raw_number))
            except ValueError:
                warnings.append(f"Row {line_no}: ignoring non-numeric student number.")

        if not name:
            warnings.append(f"Row {line_no}: skipped (no name).")
            continue

        rows.append({"name": name, "email": email or None, "student_number": number})

    return rows, warnings


def import_roster_rows(
    db: Session, course_id: int, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Insert parsed roster rows, auto-assigning student numbers."""
    existing = db.scalars(select(Student).where(Student.course_id == course_id)).all()
    by_name = {s.name.strip().lower(): s for s in existing}
    used_numbers = {s.student_number for s in existing}
    next_number = max(used_numbers, default=0) + 1

    created: list[Student] = []
    updated = 0
    skipped = 0

    for row in rows:
        key = row["name"].strip().lower()
        if key in by_name:
            student = by_name[key]
            if row.get("email") and not student.email:
                student.email = row["email"]
                updated += 1
            else:
                skipped += 1
            continue

        number = row.get("student_number")
        if number is None or number in used_numbers:
            while next_number in used_numbers:
                next_number += 1
            number = next_number
        used_numbers.add(number)

        student = Student(
            course_id=course_id,
            name=row["name"],
            email=row.get("email"),
            student_number=number,
        )
        db.add(student)
        by_name[key] = student
        created.append(student)

    db.commit()
    for student in created:
        db.refresh(student)

    return {
        "created": len(created),
        "updated": updated,
        "skipped": skipped,
        "students": [student_dict(s) for s in created],
    }


@router.post("/api/courses/{course_id}/roster/import")
async def import_roster(
    course_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Upload a roster CSV (columns: name, optional email) and create students."""
    _get_course(db, course_id)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file was empty")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Roster file is too large")

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails
        raise HTTPException(status_code=400, detail="Could not decode the CSV file")

    rows, warnings = parse_roster_csv(text)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No students found in the CSV. Expected a 'name' column.",
        )

    result = import_roster_rows(db, course_id, rows)
    result["warnings"] = warnings
    result["filename"] = file.filename
    result["parsed"] = len(rows)
    return result


# Alias — some UI code posts to /students/import.
@router.post("/api/courses/{course_id}/students/import", include_in_schema=False)
async def import_roster_alias(
    course_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return await import_roster(course_id=course_id, file=file, db=db)


# --------------------------------------------------------------------------
# page routes
# --------------------------------------------------------------------------


def _quick_stats(db: Session) -> dict[str, Any]:
    row = db.execute(
        select(
            select(func.count(Course.id)).scalar_subquery().label("courses"),
            select(func.count(Student.id)).scalar_subquery().label("students"),
            select(func.count(Assignment.id)).scalar_subquery().label("assignments"),
            select(func.count(Submission.id)).scalar_subquery().label("submissions"),
            select(func.count(GradeResult.id)).scalar_subquery().label("graded"),
            select(func.count(Submission.id))
            .where(Submission.status == "pending")
            .scalar_subquery()
            .label("pending"),
            select(func.avg(100.0 * GradeResult.overall_score / GradeResult.max_score))
            .where(GradeResult.max_score > 0)
            .scalar_subquery()
            .label("average_percent"),
        )
    ).one()
    return {
        "courses": int(row.courses or 0),
        "students": int(row.students or 0),
        "assignments": int(row.assignments or 0),
        "submissions": int(row.submissions or 0),
        "graded": int(row.graded or 0),
        "pending": int(row.pending or 0),
        "average_percent": (
            round(float(row.average_percent), 1) if row.average_percent is not None else None
        ),
    }


@router.get("/", response_class=HTMLResponse)
def page_index(request: Request, db: Session = Depends(get_db)) -> Any:
    courses = db.scalars(
        select(Course)
        .options(
            selectinload(Course.students),
            selectinload(Course.assignments),
        )
        .order_by(Course.created_at.desc(), Course.id.desc())
    ).all()
    assignment_ids = [assignment.id for course in courses for assignment in course.assignments]
    assignment_progress = {
        assignment_id: {"submissions": 0, "graded": 0, "ungraded": 0}
        for assignment_id in assignment_ids
    }
    if assignment_ids:
        status_counts = db.execute(
            select(Assignment.id, Submission.status, func.count(Submission.id))
            .outerjoin(Submission, Submission.assignment_id == Assignment.id)
            .where(Assignment.id.in_(assignment_ids))
            .group_by(Assignment.id, Submission.status)
        ).all()
        for assignment_id, status, count in status_counts:
            count = int(count or 0)
            assignment_progress[assignment_id]["submissions"] += count
            if status == "graded":
                assignment_progress[assignment_id]["graded"] += count
            else:
                assignment_progress[assignment_id]["ungraded"] += count
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "courses": courses,
            "stats": _quick_stats(db),
            "assignment_progress": assignment_progress,
        },
    )


@router.get("/courses/{course_id}", response_class=HTMLResponse)
def page_course_detail(
    course_id: int, request: Request, db: Session = Depends(get_db)
) -> Any:
    course = _get_course(db, course_id)
    students = db.scalars(
        select(Student)
        .options(selectinload(Student.submissions).selectinload(Submission.grade_result))
        .where(Student.course_id == course_id)
        .order_by(Student.student_number)
    ).all()
    assignments = db.scalars(
        select(Assignment)
        .options(
            selectinload(Assignment.submissions).selectinload(Submission.grade_result)
        )
        .where(Assignment.course_id == course_id)
        .order_by(Assignment.id)
    ).all()
    return templates.TemplateResponse(
        request,
        "course_detail.html",
        {"course": course, "students": students, "assignments": assignments},
    )


@router.get("/students/{student_id}", response_class=HTMLResponse)
def page_student_detail(
    student_id: int, request: Request, db: Session = Depends(get_db)
) -> Any:
    student = _get_student(db, student_id)
    course = db.get(Course, student.course_id)
    results = db.scalars(
        select(GradeResult)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .join(Assignment, Submission.assignment_id == Assignment.id)
        .where(Submission.student_id == student_id)
        .options(selectinload(GradeResult.submission).selectinload(Submission.assignment))
        .order_by(Assignment.id)
    ).all()
    return templates.TemplateResponse(
        request,
        "student_detail.html",
        {
            "student": student,
            "course": course,
            "results": results,
            "analytics": analytics_mod.student_timeline(db, student_id),
        },
    )


__all__ = ["router", "parse_roster_csv", "import_roster_rows", "next_student_number"]


@router.get("/analytics", response_class=RedirectResponse, include_in_schema=False)
def analytics_redirect(db: Session = Depends(get_db)) -> RedirectResponse:
    """Sidebar Analytics target. Resolves to the newest course, so the link
    works from pages that carry no course in their template context."""
    course = db.execute(select(Course).order_by(Course.id.desc())).scalars().first()
    target = f"/courses/{course.id}/analytics" if course else "/"
    return RedirectResponse(target, status_code=307)
