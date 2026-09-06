"""Demo data seeding — `python run.py --demo`.

Creates one fully-graded course so the UI (and especially analytics) is
explorable with zero API keys configured. Grades come from the engine's
MockProvider when it is importable; otherwise from an equivalent deterministic
local fallback, so seeding never depends on another module being finished.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.ai.mock_content import MOCK_CRITERION_COMMENTS
from app.db import init_db, session_scope
from app.models import (
    Assignment,
    Course,
    GradeResult,
    Rubric,
    Skill,
    Student,
    Submission,
)

log = logging.getLogger("agora.seed")

DEMO_COURSE_NAME = "PHIL 210 · Ethics of Emerging Technology"
DEMO_TERM = "Fall 2026"

DEMO_STUDENTS: list[tuple[str, str]] = [
    ("Amara Osei", "aosei@example.edu"),
    ("Ben Whitaker", "bwhitaker@example.edu"),
    ("Claudia Moreno", "cmoreno@example.edu"),
    ("Daniel Park", "dpark@example.edu"),
    ("Elena Vasquez", "evasquez@example.edu"),
    ("Farid Haddad", "fhaddad@example.edu"),
    ("Grace Lindqvist", "glindqvist@example.edu"),
    ("Hugo Brandt", "hbrandt@example.edu"),
    ("Imani Clarke", "iclarke@example.edu"),
    ("Jonas Meyer", "jmeyer@example.edu"),
    ("Keiko Tanaka", "ktanaka@example.edu"),
    ("Liam O'Donnell", "lodonnell@example.edu"),
]

RUBRIC_CRITERIA: list[dict[str, Any]] = [
    {
        "key": "thesis",
        "title": "Thesis & Argument",
        "description": "States a defensible thesis and sustains it throughout.",
        "max_points": 10,
    },
    {
        "key": "evidence",
        "title": "Use of Evidence",
        "description": "Supports claims with course readings and concrete cases.",
        "max_points": 10,
    },
    {
        "key": "counterargument",
        "title": "Counterargument",
        "description": "Engages the strongest opposing view honestly.",
        "max_points": 10,
    },
    {
        "key": "mechanics",
        "title": "Clarity & Mechanics",
        "description": "Organization, citation format, prose quality.",
        "max_points": 5,
    },
]

SYSTEM_PROMPT = """You are grading undergraduate philosophy essays for PHIL 210.

Grade strictly against the rubric, criterion by criterion. Reward students who engage the strongest version of an opposing view; penalise summary that never becomes argument. Write comments to the student, in second person, naming the specific move in their text you are reacting to. Never guess at the student's identity — submissions are anonymised by number.
"""

MISCONCEPTIONS: dict[str, list[str]] = {
    "ps1": [
        "conflates legality with morality",
        "misapplies utilitarian calculus",
        "treats correlation as causation",
        "ignores stakeholder scope",
    ],
    "essay1": [
        "conflates legality with morality",
        "strawmans the opposing view",
        "cites source without engaging it",
        "assumes technological inevitability",
    ],
}

STRENGTHS = [
    "clear thesis statement",
    "strong use of the Nissenbaum reading",
    "well-chosen concrete case",
    "honest engagement with objections",
    "tight, readable prose",
]



# --------------------------------------------------------------------------
# deterministic scoring
# --------------------------------------------------------------------------


def _hash_fraction(*parts: Any) -> float:
    """Stable 0..1 value derived from the submission identity (mirrors MockProvider)."""
    digest = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _band(pct: float) -> str:
    if pct >= 0.9:
        return "strong"
    if pct >= 0.78:
        return "solid"
    if pct >= 0.62:
        return "developing"
    return "weak"


def _fallback_grade(
    filename: str,
    criteria: list[dict[str, Any]],
    tag_pool: list[str],
    ability_key: str = "",
    drift: float = 0.0,
) -> dict:
    """Schema-shaped grade result with realistic spread (no network, no engine).

    `ability_key` is stable per student so a student's timeline reads as a
    person rather than noise; `drift` nudges later assignments upward so the
    trend chart shows the class improving.
    """
    ability = 0.55 + 0.40 * _hash_fraction(ability_key or filename, "ability") + drift
    result_criteria: list[dict[str, Any]] = []
    weak_keys: list[str] = []

    for crit in criteria:
        max_points = float(crit["max_points"])
        noise = (_hash_fraction(filename, crit["key"]) - 0.5) * 0.30
        fraction = min(1.0, max(0.28, ability + noise))
        score = round(fraction * max_points * 2) / 2  # half-point granularity
        band = _band(score / max_points if max_points else 0)
        if band in ("developing", "weak"):
            weak_keys.append(crit["key"])
        result_criteria.append(
            {
                "key": crit["key"],
                "score": score,
                "max_points": max_points,
                "comment": MOCK_CRITERION_COMMENTS[crit["key"]][band],
            }
        )

    overall = sum(c["score"] for c in result_criteria)
    total = sum(float(c["max_points"]) for c in criteria)
    pct = overall / total if total else 0.0

    # Weaker work collects more misconception tags; tag choice is stable per file.
    tag_count = 0 if pct >= 0.9 else 1 if pct >= 0.78 else 2 if pct >= 0.62 else 3
    start = int(_hash_fraction(filename, "tags") * len(tag_pool))
    tags = [tag_pool[(start + i) % len(tag_pool)] for i in range(tag_count)]

    strength_count = 3 if pct >= 0.9 else 2 if pct >= 0.7 else 1
    s_start = int(_hash_fraction(filename, "strengths") * len(STRENGTHS))
    strengths = [STRENGTHS[(s_start + i) % len(STRENGTHS)] for i in range(strength_count)]

    if pct >= 0.9:
        summary = "Excellent work. The argument is disciplined and the objection is met head-on."
    elif pct >= 0.78:
        summary = "A good essay with a clear line of argument; evidence could do more work."
    elif pct >= 0.62:
        summary = "You have the beginnings of a real argument, but it stays at the level of summary."
    else:
        summary = "This submission does not yet meet the rubric; see the per-criterion notes."

    return {
        "criteria": result_criteria,
        "summary_feedback": summary,
        "misconceptions": tags,
        "strengths": strengths,
        "weak_keys": weak_keys,
    }


def _mock_grade(
    filename: str,
    criteria: list[dict[str, Any]],
    tag_pool: list[str],
    ability_key: str = "",
    drift: float = 0.0,
) -> tuple[dict, str]:
    """Grade via the engine's MockProvider if available, else the local fallback.

    The engine module is being written concurrently, so every failure mode
    (missing module, different signature, different return shape) degrades to
    the deterministic fallback rather than breaking `run.py --demo`.
    """
    try:
        from app.ai.providers import MockProvider  # noqa: PLC0415 - lazy on purpose

        provider = MockProvider()
        result = provider.grade(
            system_prompt=SYSTEM_PROMPT,
            rubric={"criteria": criteria},
            submission={"filename": filename, "mime_type": "application/pdf"},
        )
        if isinstance(result, dict) and result.get("criteria"):
            model = getattr(provider, "model", None) or config.MOCK_MODEL
            result.setdefault("misconceptions", [])
            result.setdefault("strengths", [])
            result.setdefault("summary_feedback", "")
            return result, str(model)
    except Exception as exc:  # noqa: BLE001 - engine may be mid-build
        log.debug("MockProvider unavailable (%s); using seed fallback grader", exc)
    return _fallback_grade(filename, criteria, tag_pool, ability_key, drift), config.MOCK_MODEL


# --------------------------------------------------------------------------
# demo submission files
# --------------------------------------------------------------------------


def _mini_pdf(lines: list[str]) -> bytes:
    """A tiny but structurally valid single-page PDF (demo stand-in submission)."""
    escaped = [line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)") for line in lines]
    text_ops = "\n".join(f"({line}) Tj T*" for line in escaped)
    stream = f"BT /F1 12 Tf 54 720 Td 16 TL\n{text_ops}\nET".encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def _write_submission_file(directory: Path, filename: str, student_number: int, title: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if not path.exists():
        path.write_bytes(
            _mini_pdf(
                [
                    f"{title} — Student #{student_number}",
                    "",
                    "Demo submission generated by `python run.py --demo`.",
                    "Graded locally with the MockProvider; no network calls were made.",
                ]
            )
        )
    return str(path)


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------


def _existing_demo(db: Session) -> Optional[Course]:
    return db.scalars(select(Course).where(Course.name == DEMO_COURSE_NAME)).first()


def seed_demo(
    db: Session | None = None,
    reset: bool = True,
    insight_history: bool | None = None,
) -> dict[str, Any]:
    """Create the demo course, roster, rubric, skill, assignments and grades.

    Idempotent: by default an existing demo course is removed and rebuilt so the
    data is always the same deterministic set.

    ``insight_history`` adds a third graded assignment so a misconception can
    decay all the way to ``resolved`` on the student cards (see
    ``INSIGHT_HISTORY_ENV``). Off by default: the two-assignment demo is what
    the rest of the suite pins.
    """
    if db is None:
        init_db()
        with session_scope() as session:
            return _seed(session, reset=reset, insight_history=insight_history)
    return _seed(db, reset=reset, insight_history=insight_history)


def _seed(db: Session, reset: bool, insight_history: bool | None = None) -> dict[str, Any]:
    existing = _existing_demo(db)
    if existing is not None:
        if not reset:
            log.info("Demo course already present (id=%s)", existing.id)
            return {"course_id": existing.id, "created": False}
        db.delete(existing)
        # The demo skill and rubric are not course-scoped; delete them too so
        # repeated `run.py --demo` runs do not accumulate duplicates.
        for old_skill in db.scalars(
            select(Skill).where(Skill.name == "PHIL 210 Essay Grader")
        ).all():
            db.delete(old_skill)
        for old_rubric in db.scalars(
            select(Rubric).where(Rubric.name == "PHIL 210 Standard Essay Rubric")
        ).all():
            db.delete(old_rubric)
        db.commit()

    now = datetime.now(timezone.utc)

    course = Course(name=DEMO_COURSE_NAME, term=DEMO_TERM, created_at=now - timedelta(days=60))
    db.add(course)
    db.flush()

    students = [
        Student(
            course_id=course.id,
            name=name,
            email=email,
            student_number=idx,
            created_at=now - timedelta(days=59),
        )
        for idx, (name, email) in enumerate(DEMO_STUDENTS, start=1)
    ]
    db.add_all(students)

    rubric = Rubric(name="PHIL 210 Standard Essay Rubric", criteria=RUBRIC_CRITERIA)
    skill = Skill(
        name="PHIL 210 Essay Grader",
        description="Strict-but-humane grader for undergraduate applied-ethics writing.",
        system_prompt=SYSTEM_PROMPT,
        provider=config.DEFAULT_SKILL_PROVIDER,
        model=config.AUTO_MODEL,
        max_tokens=config.DEFAULT_MAX_TOKENS,
    )
    db.add_all([rubric, skill])
    db.flush()

    assignment_specs = [
        (
            "ps1",
            "Problem Set 1 · Moral Frameworks",
            "Apply consequentialist and deontological analysis to two short cases.",
            40,
        ),
        (
            "essay1",
            "Essay 1 · Autonomy and Algorithms",
            "Argue for or against algorithmic recommendation as a threat to autonomy.",
            18,
        ),
    ]
    # Increment 1 · insight module: a third graded assignment is what lets a
    # misconception decay past `resolving` into `resolved` on the student card.
    if _want_insight_history(insight_history):
        assignment_specs.append(INSIGHT_HISTORY_ASSIGNMENT)

    assignments: list[Assignment] = []
    for slug, name, description, days_ago in assignment_specs:
        assignment = Assignment(
            course_id=course.id,
            name=name,
            description=description,
            due_date=now - timedelta(days=days_ago),
            skill_id=skill.id,
            rubric_id=rubric.id,
            created_at=now - timedelta(days=days_ago + 14),
        )
        assignments.append(assignment)
    db.add_all(assignments)
    db.flush()

    total_points = sum(float(c["max_points"]) for c in RUBRIC_CRITERIA)
    submission_dir = config.UPLOAD_DIR / "demo"
    graded = 0

    for order, (assignment, (slug, name, _desc, days_ago)) in enumerate(
        zip(assignments, assignment_specs)
    ):
        tag_pool = MISCONCEPTIONS[slug]
        drift = 0.04 * order  # the class gets a little better as the term goes on
        for student in students:
            filename = f"{slug}_student_{student.student_number:02d}.pdf"
            file_path = _write_submission_file(
                submission_dir, filename, student.student_number, name
            )
            submission = Submission(
                assignment_id=assignment.id,
                student_id=student.id,
                file_path=file_path,
                original_filename=filename,
                mime_type="application/pdf",
                status="graded",
                created_at=now - timedelta(days=days_ago - 1),
            )
            db.add(submission)
            db.flush()

            result, model = _mock_grade(
                filename,
                RUBRIC_CRITERIA,
                tag_pool,
                ability_key=f"student-{student.student_number}",
                drift=drift,
            )
            criteria = result["criteria"]
            overall = sum(float(c.get("score", 0)) for c in criteria)
            db.add(
                GradeResult(
                    submission_id=submission.id,
                    overall_score=round(overall, 2),
                    max_score=total_points,
                    summary_feedback=result.get("summary_feedback", ""),
                    criteria=criteria,
                    misconceptions=sorted(
                        {str(t).strip().lower() for t in result.get("misconceptions", []) if t}
                    ),
                    strengths=list(result.get("strengths", [])),
                    model=model,
                    created_at=now - timedelta(days=days_ago - 2),
                )
            )
            graded += 1

    db.commit()

    insight_summary = _seed_insight(db, course, students, assignments)

    log.info(
        "Seeded demo course %r: %d students, %d assignments, %d graded submissions",
        DEMO_COURSE_NAME,
        len(students),
        len(assignments),
        graded,
    )
    return {
        "course_id": course.id,
        "created": True,
        "students": len(students),
        "assignments": len(assignments),
        "graded": graded,
        "rubric_id": rubric.id,
        "skill_id": skill.id,
        "insight": insight_summary,
    }


# ==========================================================================
# Increment 1 · INSIGHT MODULE (appended — the demo must exercise the feature)
#
# The demo grades are written straight into the DB rather than through the
# engine, so the post-grading hook never fires for them. This pass does what
# the hook would have done: derive every observation, look for course nudges,
# and consolidate one card per student with the MockProvider (offline,
# deterministic, exactly what `--demo` promises).
# ==========================================================================

#: `run.py --demo` takes no arguments, so the third assignment (needed for a
#: `resolved` misconception chip) is opt-in through the environment:
#:     AGORA_DEMO_INSIGHT_HISTORY=1 python run.py --demo
INSIGHT_HISTORY_ENV = "AGORA_DEMO_INSIGHT_HISTORY"

INSIGHT_HISTORY_ASSIGNMENT = (
    "ps2",
    "Problem Set 2 · Duty, Harm and Consent",
    "Two short cases on consent and downstream harm; cite one reading each.",
    5,
)

#: Tag pool for the third assignment, used by the local fallback grader (the
#: MockProvider path picks from its own pool). It drops the tags Problem Set 1
#: leaned on so those decay to `resolved` two assignments later.
MISCONCEPTIONS[INSIGHT_HISTORY_ASSIGNMENT[0]] = [
    "strawmans the opposing view",
    "assumes technological inevitability",
    "confuses consent with compliance",
    "cites source without engaging it",
]


def _want_insight_history(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    return os.environ.get(INSIGHT_HISTORY_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _seed_insight(
    db: Session,
    course: Course,
    students: list[Student],
    assignments: list[Assignment],
) -> dict[str, Any]:
    """Observations, nudges and cards for the freshly seeded demo grades."""
    try:
        from app import insight
        from app.ai.providers import MockProvider
    except Exception as exc:  # noqa: BLE001 - the demo must seed regardless
        log.debug("Insight layer unavailable (%s); skipping demo insight pass", exc)
        return {"available": False}

    try:
        observations = 0
        submissions = db.scalars(
            select(Submission)
            .join(Assignment, Submission.assignment_id == Assignment.id)
            .where(Assignment.course_id == course.id)
            .order_by(Submission.id)
        ).all()
        for submission in submissions:
            observations += len(insight.derive_observations(db, submission, commit=False))
        db.commit()

        nudges = 0
        for assignment in assignments:
            nudges += len(insight.detect_nudges(db, assignment.id, commit=False))
        db.commit()

        provider = MockProvider()
        states: dict[str, int] = {}
        for student in students:
            card = insight.refresh_card(db, student.id, provider=provider, commit=False)
            for entry in card.misconception_state or []:
                status = str(entry.get("status") or "")
                states[status] = states.get(status, 0) + 1
        db.commit()
    except Exception as exc:  # noqa: BLE001 - a demo must never fail to seed
        log.warning("Demo insight pass failed (%s); demo data is still usable", exc)
        db.rollback()
        return {"available": False, "error": str(exc)}

    log.info(
        "Seeded insight layer: %d observations, %d nudges, %d cards (states: %s)",
        observations,
        nudges,
        len(students),
        states or "none",
    )
    return {
        "available": True,
        "observations": observations,
        "nudges": nudges,
        "cards": len(students),
        "misconception_states": states,
    }


__all__ = [
    "seed_demo",
    "DEMO_COURSE_NAME",
    "RUBRIC_CRITERIA",
    "INSIGHT_HISTORY_ENV",
]
