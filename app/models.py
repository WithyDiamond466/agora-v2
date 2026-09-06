"""SQLAlchemy 2.x ORM models — the whole Agora schema.

Naming note: `Course` (not `Class`) avoids the Python keyword collision, per SPEC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Float,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app import config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base. `JSON` maps to SQLite's JSON1-backed TEXT column."""

    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


# --------------------------------------------------------------------------
# Courses & students
# --------------------------------------------------------------------------


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    term: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    students: Mapped[list["Student"]] = relationship(
        back_populates="course",
        cascade="all, delete-orphan",
        order_by="Student.student_number",
    )
    assignments: Mapped[list["Assignment"]] = relationship(
        back_populates="course",
        cascade="all, delete-orphan",
        order_by="Assignment.id",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Course {self.id} {self.name!r}>"


class Student(Base):
    __tablename__ = "students"
    __table_args__ = (
        UniqueConstraint("course_id", "student_number", name="uq_student_number_per_course"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    #: Per-course anonymization handle. Submissions sent to AI use this, never `name`.
    student_number: Mapped[int] = mapped_column(Integer, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    course: Mapped["Course"] = relationship(back_populates="students")
    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )

    @property
    def anon_label(self) -> str:
        """The only identifier that may leave the machine."""
        return f"Student #{self.student_number}"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Student {self.id} #{self.student_number} {self.name!r}>"


# --------------------------------------------------------------------------
# Skills & knowledge
# --------------------------------------------------------------------------


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    #: ``auto`` = whatever provider the professor has a key for at grade time.
    provider: Mapped[str] = mapped_column(String(50), default=config.DEFAULT_SKILL_PROVIDER)
    model: Mapped[str] = mapped_column(String(100), default=config.AUTO_MODEL)
    # NOTE: deliberately no temperature/top_p — current Anthropic models 400 on them.
    max_tokens: Mapped[int] = mapped_column(Integer, default=config.DEFAULT_MAX_TOKENS)
    #: What the skill produces: ``grade`` (score every criterion), ``feedback``
    #: (comments only, no scores) or ``selective`` (score only the criteria the
    #: assignment picks). Professor-added modes live in ``app.ai.modes``.
    mode: Mapped[str] = mapped_column(String(40), default="grade")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    knowledge_docs: Mapped[list["KnowledgeDoc"]] = relationship(
        back_populates="skill", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Skill {self.id} {self.name!r}>"


class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    filename: Mapped[str] = mapped_column(String(300), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    skill: Mapped["Skill"] = relationship(back_populates="knowledge_docs")


# --------------------------------------------------------------------------
# Rubrics, assignments, submissions, grades
# --------------------------------------------------------------------------


class Rubric(Base):
    __tablename__ = "rubrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    #: [{key, title, description, max_points}]
    criteria: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    assignments: Mapped[list["Assignment"]] = relationship(back_populates="rubric")

    @property
    def total_points(self) -> float:
        total = 0.0
        for crit in self.criteria or []:
            try:
                total += float(crit.get("max_points", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                continue
        return total


class Assignment(Base):
    __tablename__ = "assignments"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    skill_id: Mapped[Optional[int]] = mapped_column(ForeignKey("skills.id", ondelete="SET NULL"))
    rubric_id: Mapped[Optional[int]] = mapped_column(ForeignKey("rubrics.id", ondelete="SET NULL"))
    #: Selective grading: the rubric criterion keys the AI scores on this
    #: assignment. ``None`` means every criterion. Ignored unless the skill's
    #: mode is selective.
    ai_criteria: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    course: Mapped["Course"] = relationship(back_populates="assignments")
    skill: Mapped[Optional["Skill"]] = relationship()
    rubric: Mapped[Optional["Rubric"]] = relationship(back_populates="assignments")
    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="assignment", cascade="all, delete-orphan"
    )


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("assignments.id", ondelete="CASCADE"))
    student_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE")
    )
    file_path: Mapped[Optional[str]] = mapped_column(String(1000))
    original_filename: Mapped[Optional[str]] = mapped_column(String(300))
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    #: pending | grading | graded | failed
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    assignment: Mapped["Assignment"] = relationship(back_populates="submissions")
    student: Mapped[Optional["Student"]] = relationship(back_populates="submissions")
    grade_result: Mapped[Optional["GradeResult"]] = relationship(
        back_populates="submission", cascade="all, delete-orphan", uselist=False
    )


class GradeResult(Base):
    __tablename__ = "grade_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), unique=True
    )
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    max_score: Mapped[float] = mapped_column(Float, default=0.0)
    summary_feedback: Mapped[Optional[str]] = mapped_column(Text)
    #: [{key, score, max_points, comment}]
    criteria: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: [str] — lowercased/deduped tags; analytics groups by exact tag
    misconceptions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: [str]
    strengths: Mapped[list[Any]] = mapped_column(JSON, default=list)
    model: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    #: Skill mode that produced this result (grade | feedback | selective | custom).
    mode: Mapped[str] = mapped_column(String(40), default="grade")

    # -- human review record (see app.review) ------------------------------
    #: First time the professor opened this result in the grading pane.
    seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    #: Explicit approval of the current version, cleared by edits/regrading.
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    #: When the professor released it to students (per-assignment release).
    released_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    #: The professor said "no" to this one — it is never released or exported.
    withheld_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    #: [{at, fields: [..]}] — every professor edit, newest last.
    edit_log: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)

    submission: Mapped["Submission"] = relationship(back_populates="grade_result")

    @property
    def incomplete(self) -> bool:
        return (self.mode or "grade") != "feedback" and any(
            isinstance(c, dict) and c.get("score") is None for c in self.criteria or []
        )

    @property
    def percentage(self) -> Optional[float]:
        if self.incomplete or self.mode == "feedback":
            return None
        if not self.max_score:
            return 0.0
        return round(100.0 * float(self.overall_score) / float(self.max_score), 1)

    @property
    def review_state(self) -> str:
        """withheld | released | seen | unreviewed — for pills and filters."""
        if self.withheld_at is not None:
            return "withheld"
        if self.released_at is not None and self.approved_at is not None:
            return "released"
        if self.approved_at is not None:
            return "approved"
        if self.seen_at is not None:
            return "seen"
        return "unreviewed"


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------


class ApiCredential(Base):
    __tablename__ = "api_credentials"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    #: Fernet ciphertext. Plaintext keys never touch the DB.
    key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(200))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String(300))
    course_id: Mapped[Optional[int]] = mapped_column(ForeignKey("courses.id", ondelete="SET NULL"))
    skill_id: Mapped[Optional[int]] = mapped_column(ForeignKey("skills.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    course: Mapped[Optional["Course"]] = relationship()
    skill: Mapped[Optional["Skill"]] = relationship()
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_id_id", "session_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    course_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("courses.id", ondelete="SET NULL")
    )
    #: user | assistant | tool
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    #: Raw tool-call / tool-result payloads (parsed JSON, never string-matched).
    tool_calls: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


__all__ = [
    "Base",
    "Course",
    "Student",
    "Skill",
    "KnowledgeDoc",
    "Rubric",
    "Assignment",
    "Submission",
    "GradeResult",
    "ApiCredential",
    "ChatSession",
    "ChatMessage",
    "utcnow",
]


# ==========================================================================
# Increment 1 · PRIVACY MODULE tables (appended — nothing above was touched)
#
# The pseudonym map and the per-submission scan report. Both are strictly
# local: the map is what lets the professor ask "who is Person-A?" and lets
# the UI swap codes back to real names, and it must never leave this machine.
# ==========================================================================


class PseudonymMap(Base):
    """Stable code for one real identity, scoped to a course.

    Roster students do not need a row here — their code is derived from their
    per-course number (``Student-07``). This table covers everyone/everything
    else the Privacy Guard finds: third-party names, emails, phones, ids.
    """

    __tablename__ = "pseudonym_map"
    __table_args__ = (
        UniqueConstraint(
            "course_id", "kind", "original_text_normalized", name="uq_pseudonym_original"
        ),
        UniqueConstraint("course_id", "code", name="uq_pseudonym_code"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), index=True
    )
    #: person | email | phone | id_number
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="person")
    #: Case/whitespace-folded form used for lookups.
    original_text_normalized: Mapped[str] = mapped_column(String(300), nullable=False)
    #: As first seen — this is what the professor-facing display layer shows.
    original_text: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    #: Person-A, [EMAIL-1], [PHONE-2], [ID-1], ...
    code: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    #: Set when the professor forgets a mapping. The row stays behind as a
    #: tombstone so its code can never be reissued to a different person —
    #: every stored piece of feedback that says "Person-A" has to keep meaning
    #: what it meant when the model wrote it.
    retired_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    course: Mapped["Course"] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<PseudonymMap {self.code} course={self.course_id}>"


class PrivacyScan(Base):
    """What the Privacy Guard found (and swapped) for one submission."""

    __tablename__ = "privacy_scans"

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), index=True
    )
    #: off | warn | swap — the mode the run actually used.
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="swap")
    #: The full report dict from ``app.ai.privacy.pseudonymize_text``.
    findings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    submission: Mapped["Submission"] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<PrivacyScan submission={self.submission_id} mode={self.mode}>"


__all__ += ["PseudonymMap", "PrivacyScan"]


# ==========================================================================
# Increment 1 · INSIGHT MODULE tables (appended — nothing above was touched)
#
# The grading event stream becomes maintained per-student understanding:
#   Observation  — one deterministic, evidence-grade fact derived from a
#                  GradeResult (no LLM ever writes these).
#   StudentCard  — the consolidated narrative built FROM those observations
#                  (LLM, with a deterministic template fallback).
#   CourseNudge  — "worth a look": a misconception several students share on
#                  the same assignment.
# Every claim on a card carries the ids of the observations behind it, so the
# UI can link a sentence back to the graded work that produced it.
# ==========================================================================


class Observation(Base):
    """One factual sentence about a student, derived from one graded submission."""

    __tablename__ = "observations"

    #: criterion_low | criterion_high | misconception | strength
    KINDS = ("criterion_low", "criterion_high", "misconception", "strength")

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), index=True
    )
    assignment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("assignments.id", ondelete="CASCADE"), index=True
    )
    submission_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    #: One sentence, ≤25 words, grounded in numbers the professor can verify.
    text: Mapped[str] = mapped_column(Text, default="")
    #: {criterion_key, score, max_points} or {tag} or {phrase}
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    student: Mapped["Student"] = relationship()
    assignment: Mapped[Optional["Assignment"]] = relationship()
    submission: Mapped[Optional["Submission"]] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Observation {self.id} {self.kind} student={self.student_id}>"


class StudentCard(Base):
    """The consolidated, always-current understanding of one student."""

    __tablename__ = "student_cards"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), unique=True, index=True
    )
    summary: Mapped[str] = mapped_column(Text, default="")
    #: [{text, evidence: [observation_id]}]
    strengths: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: [{text, evidence: [observation_id]}]
    weaknesses: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: [{tag, status: active|resolving|resolved, evidence: [observation_id]}]
    misconception_state: Mapped[list[Any]] = mapped_column(JSON, default=list)
    trajectory: Mapped[str] = mapped_column(Text, default="")
    #: The model that wrote this card, or "template" when the fallback built it.
    model: Mapped[Optional[str]] = mapped_column(String(100))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    student: Mapped["Student"] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<StudentCard student={self.student_id} model={self.model!r}>"


class CourseNudge(Base):
    """A class-level pattern worth the professor's attention."""

    __tablename__ = "course_nudges"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), index=True
    )
    assignment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("assignments.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text, default="")
    #: {tag, student_ids, observation_ids}
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    dismissed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    course: Mapped["Course"] = relationship()
    assignment: Mapped[Optional["Assignment"]] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<CourseNudge {self.id} course={self.course_id}>"


__all__ += ["Observation", "StudentCard", "CourseNudge"]


# ==========================================================================
# Fall readiness · REVIEW RECORD + TERMS (appended — nothing above was touched)
#
# ReviewEvent  — append-only trail of what the professor did with an AI
#                result: opened it, edited it, withheld it, released it.
#                This is the record that shows a human reviewed AI output
#                before it reached a student.
# Acceptance   — the professor's acceptance of the terms of use (versioned),
#                which is what unlocks cloud grading and release.
# ==========================================================================


class ReviewEvent(Base):
    __tablename__ = "review_events"
    __table_args__ = (Index("ix_review_events_submission_id_id", "submission_id", "id"),)

    KINDS = ("seen", "approved", "edited", "withheld", "unwithheld", "released", "regraded")

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), index=True
    )
    assignment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("assignments.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    #: Free-form context: edited field names, release batch id, model, ...
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    submission: Mapped["Submission"] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ReviewEvent {self.kind} submission={self.submission_id}>"


class Acceptance(Base):
    __tablename__ = "acceptances"

    id: Mapped[int] = mapped_column(primary_key=True)
    #: What was accepted — only ``terms`` today.
    kind: Mapped[str] = mapped_column(String(30), nullable=False, default="terms")
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    #: Optional name the professor typed as a signature.
    signed_by: Mapped[Optional[str]] = mapped_column(String(200))
    accepted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Acceptance {self.kind} v{self.version}>"


__all__ += ["ReviewEvent", "Acceptance"]
