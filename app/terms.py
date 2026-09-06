"""Terms of use — the professor's acceptance, versioned, and the disclosure text.

The second half of what Anthropic's usage policy asks of a grading assistant:
the people affected by an AI-assisted decision are told AI was involved. Agora
cannot reach students, so this module gives the professor the words (a
syllabus statement and a per-feedback disclosure line) and records that they
accepted the terms that make them responsible for using them.

Acceptance is required before anything leaves the machine for a cloud model
and before results are released or exported. Local and mock providers are not
gated, so the demo and the tests run without a signature.

Bump ``TERMS_VERSION`` when the wording changes materially; a professor on an
older version is asked to accept again.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Acceptance

TERMS_VERSION = "2026-09-02"

TERMS_TITLE = "Using Agora to grade with AI"

#: What the professor is agreeing to. Rendered on /terms and in the bundle.
TERMS_POINTS: tuple[tuple[str, str], ...] = (
    (
        "You review before anything reaches a student.",
        "Agora drafts scores and feedback; you are the grader of record. Opening a "
        "result in the grading pane records that you looked at it. Nothing is "
        "released or exported until you release it, and results you never opened "
        "are held back unless you choose to include them.",
    ),
    (
        "You tell students that AI helped.",
        "The AI provider's terms require that people affected by an AI-assisted "
        "decision are told. Agora puts a disclosure line on every exported piece "
        "of feedback and gives you a syllabus statement below. Use them.",
    ),
    (
        "Student work goes to the provider you configure, under your account.",
        "With Privacy Guard in swap mode, names and identifiers are replaced with "
        "codes before text leaves this machine, and the code book stays here. You "
        "are responsible for confirming that sending coursework to that provider "
        "is allowed under your institution's policies.",
    ),
    (
        "You can say no to any result.",
        "Withhold a result and it is never released or exported. Edit it and "
        "your edits are what the student sees. Regrade it and it returns to "
        "unreviewed until you open it again.",
    ),
    (
        "The record stays with you.",
        "Every open, edit, withhold and release is written to the local database "
        "so you can show, later, that a person reviewed the AI's work.",
    ),
)

#: Copy-paste text for a syllabus or assignment sheet.
SYLLABUS_STATEMENT = (
    "Assessment in this course uses AI-assisted grading. An AI system drafts "
    "scores and written feedback against the published rubric; the instructor "
    "reviews every result before it is released and may change any score or "
    "comment. Student names and identifying details are replaced with codes "
    "before any text is sent to the AI provider. Feedback that was drafted with "
    "AI assistance is labelled as such. If you have questions or concerns about "
    "how your work is assessed, contact the instructor."
)

#: Appended to every piece of feedback that leaves Agora for a student.
AI_DISCLOSURE = (
    "This feedback was drafted with AI assistance and reviewed by your instructor."
)


class TermsNotAccepted(RuntimeError):
    """Raised by gated actions; routers turn it into a 409 with a link to /terms."""

    def __init__(self, action: str = "this action") -> None:
        super().__init__(
            f"Accept the terms of use before {action}. Open /terms to read and accept them."
        )
        self.action = action


def current(db: Session) -> Optional[Acceptance]:
    """The acceptance of the *current* version, or None."""
    return db.scalars(
        select(Acceptance)
        .where(Acceptance.kind == "terms", Acceptance.version == TERMS_VERSION)
        .order_by(Acceptance.id.desc())
    ).first()


def latest(db: Session) -> Optional[Acceptance]:
    """The most recent acceptance of any version (to say 'terms changed')."""
    return db.scalars(
        select(Acceptance).where(Acceptance.kind == "terms").order_by(Acceptance.id.desc())
    ).first()


def is_accepted(db: Session) -> bool:
    return current(db) is not None


def accept(db: Session, signed_by: str | None = None) -> Acceptance:
    existing = current(db)
    if existing is not None:
        return existing
    row = Acceptance(
        kind="terms", version=TERMS_VERSION, signed_by=(signed_by or "").strip()[:200] or None
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def require_accepted(db: Session, action: str = "this action") -> None:
    if not is_accepted(db):
        raise TermsNotAccepted(action)


def status(db: Session) -> dict[str, Any]:
    now = current(db)
    old = latest(db) if now is None else None
    return {
        "accepted": now is not None,
        "version": TERMS_VERSION,
        "accepted_at": now.accepted_at.isoformat() if now and now.accepted_at else None,
        "signed_by": now.signed_by if now else None,
        "previous_version": old.version if old else None,
        "terms_url": "/terms",
        "required_for": ["cloud grading", "release", "export"],
    }


__all__ = [
    "TERMS_VERSION",
    "TERMS_TITLE",
    "TERMS_POINTS",
    "SYLLABUS_STATEMENT",
    "AI_DISCLOSURE",
    "TermsNotAccepted",
    "current",
    "latest",
    "is_accepted",
    "accept",
    "require_accepted",
    "status",
]
