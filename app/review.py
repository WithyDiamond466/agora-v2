"""Record opening, edits, approval, withholding, regrading, and release.

Only explicitly approved, complete results can be released. Editing a result
clears its approval and release; regrading creates a new unreviewed version.
All actions are recorded in the append-only review event trail.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Assignment, GradeResult, ReviewEvent, Submission


def utcnow() -> datetime:
    """Naive UTC, matching what SQLite hands back, so comparisons never mix kinds."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

REVIEW_STATES = ("unreviewed", "seen", "approved", "released", "withheld")


class ReviewError(ValueError):
    """A review action that cannot be applied (no result yet, etc.)."""


def _iso(value: Optional[datetime]) -> Optional[str]:
    """SQLite hands back naive datetimes; render in-process ones the same way."""
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.isoformat()


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


def _event(db: Session, submission: Submission, kind: str, **detail: Any) -> ReviewEvent:
    if kind not in ReviewEvent.KINDS:
        raise ReviewError(f"Unknown review event kind {kind!r}")
    event = ReviewEvent(
        submission_id=submission.id,
        assignment_id=submission.assignment_id,
        kind=kind,
        detail={k: v for k, v in detail.items() if v is not None},
    )
    db.add(event)
    return event


def event_dict(event: ReviewEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "submission_id": event.submission_id,
        "assignment_id": event.assignment_id,
        "kind": event.kind,
        "detail": dict(event.detail or {}),
        "at": _iso(event.created_at),
    }


def events_for(db: Session, submission_id: int) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(ReviewEvent)
        .where(ReviewEvent.submission_id == submission_id)
        .order_by(ReviewEvent.id)
    ).all()
    return [event_dict(e) for e in rows]


def events_for_assignment(db: Session, assignment_id: int) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(ReviewEvent)
        .where(ReviewEvent.assignment_id == assignment_id)
        .order_by(ReviewEvent.id)
    ).all()
    return [event_dict(e) for e in rows]


# --------------------------------------------------------------------------
# per-result state
# --------------------------------------------------------------------------


def review_dict(result: GradeResult | None) -> Optional[dict[str, Any]]:
    """The review fields the UI renders next to a result."""
    if result is None:
        return None
    edits = list(result.edit_log or [])
    return {
        "state": result.review_state,
        "seen_at": _iso(result.seen_at),
        "approved_at": _iso(result.approved_at),
        "released_at": _iso(result.released_at),
        "withheld_at": _iso(result.withheld_at),
        "edit_count": len(edits),
        "last_edit_at": edits[-1].get("at") if edits else None,
        "needs_manual_scores": manual_scores_missing(result),
    }


def manual_scores_missing(result: GradeResult | None) -> int:
    """Criteria left for the professor (selective mode) that still have no score."""
    if result is None or result.mode == "feedback":
        return 0
    missing = 0
    for crit in result.criteria or []:
        if isinstance(crit, dict) and crit.get("score") is None:
            missing += 1
    return missing


def _result_or_raise(submission: Submission) -> GradeResult:
    result = submission.grade_result
    if result is None:
        raise ReviewError(f"Submission {submission.id} has no grade result yet")
    return result


def mark_seen(db: Session, submission: Submission, *, commit: bool = True) -> GradeResult:
    """The professor opened this result. Idempotent: the first open wins."""
    result = _result_or_raise(submission)
    if result.seen_at is None:
        result.seen_at = utcnow()
        _event(db, submission, "seen", model=result.model)
        if commit:
            db.commit()
    return result


def approve(db: Session, submission: Submission) -> GradeResult:
    """Approve this complete version. Merely opening or editing never approves."""
    result = _result_or_raise(submission)
    if submission.status != "graded":
        raise ReviewError("Wait until this submission finishes grading before approving it.")
    if result.withheld_at:
        raise ReviewError("Lift the withhold before approving this result.")
    if manual_scores_missing(result):
        raise ReviewError("Fill in all remaining criterion scores before approving.")
    if result.approved_at is None:
        mark_seen(db, submission, commit=False)
        result.approved_at = utcnow()
        result.released_at = None
        _event(db, submission, "approved", model=result.model)
        db.commit()
    return result


def record_edit(
    db: Session, submission: Submission, fields: Iterable[str], *, commit: bool = False
) -> GradeResult:
    """The professor changed something. Editing counts as having seen it."""
    result = _result_or_raise(submission)
    touched = sorted({str(f) for f in fields if f})
    now = utcnow()
    if result.seen_at is None:
        result.seen_at = now
        _event(db, submission, "seen", via="edit", model=result.model)
    log = list(result.edit_log or [])
    log.append({"at": _iso(now), "fields": touched})
    result.edit_log = log
    result.approved_at = None
    result.released_at = None
    _event(db, submission, "edited", fields=touched)
    if commit:
        db.commit()
    return result


def set_withheld(
    db: Session, submission: Submission, withheld: bool, *, commit: bool = True
) -> GradeResult:
    """Withhold (the professor says no) or lift a withhold."""
    result = _result_or_raise(submission)
    if withheld:
        if result.withheld_at is None:
            result.withheld_at = utcnow()
            result.approved_at = None
            was_released = result.released_at is not None
            result.released_at = None
            if result.seen_at is None:
                result.seen_at = result.withheld_at
            _event(db, submission, "withheld", was_released=was_released)
    else:
        if result.withheld_at is not None:
            result.withheld_at = None
            _event(db, submission, "unwithheld")
    if commit:
        db.commit()
    return result


def record_regrade(
    db: Session, submission: Submission, previous: GradeResult | None
) -> None:
    """Called by the engine just before a result is replaced.

    A regraded result starts unreviewed again — the professor has not seen the
    new text — so the previous review state is only kept as history.
    """
    if previous is None:
        return
    _event(
        db,
        submission,
        "regraded",
        previous_model=previous.model,
        previous_score=previous.overall_score,
        previous_state=previous.review_state,
        previous_edits=len(previous.edit_log or []),
    )


# --------------------------------------------------------------------------
# per-assignment release
# --------------------------------------------------------------------------


def _graded_submissions(db: Session, assignment_id: int) -> list[Submission]:
    return list(
        db.scalars(
            select(Submission)
            .options(selectinload(Submission.student), selectinload(Submission.grade_result))
            .where(Submission.assignment_id == assignment_id)
            .order_by(Submission.id)
        ).all()
    )


def _row(submission: Submission) -> dict[str, Any]:
    student = submission.student
    result = submission.grade_result
    return {
        "submission_id": submission.id,
        "student_id": submission.student_id,
        "student_name": student.name if student else None,
        "student_number": student.student_number if student else None,
        "state": result.review_state if result else None,
        "needs_manual_scores": manual_scores_missing(result),
    }


def release_summary(db: Session, assignment: Assignment) -> dict[str, Any]:
    """What one click would release, and what it would hold back."""
    subs = _graded_submissions(db, assignment.id)
    counts = {"graded": 0, **dict.fromkeys(REVIEW_STATES, 0)}
    releasable: list[dict[str, Any]] = []
    unseen: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    needs_manual = 0
    last_release: Optional[datetime] = None
    for s in subs:
        r = s.grade_result
        if r is None:
            continue
        counts["graded"] += 1
        counts[r.review_state] += 1
        needs_manual += manual_scores_missing(r)
        if r.released_at and (last_release is None or r.released_at > last_release):
            last_release = r.released_at
        if r.review_state == "approved" and not manual_scores_missing(r) and s.status == "graded":
            releasable.append(_row(s))
        elif r.review_state in {"unreviewed", "seen"}:
            unseen.append(_row(s))
        elif r.review_state == "withheld":
            withheld.append(_row(s))
    return {
        "assignment_id": assignment.id,
        "counts": counts,
        "releasable": releasable,
        "unseen": unseen,
        "withheld": withheld,
        "needs_manual_scores": needs_manual,
        "last_release_at": _iso(last_release),
    }


def release(
    db: Session,
    assignment: Assignment,
    *,
    include_unseen: bool = False,
    submission_ids: Optional[Iterable[int]] = None,
) -> dict[str, Any]:
    """Release every reviewed result on this assignment (one click).

    * withheld results are never released;
    * results never opened are held unless ``include_unseen`` is set, in which
      case the event records ``opened: false``;
    * ``submission_ids`` narrows the release to a subset.
    """
    wanted = set(int(i) for i in submission_ids) if submission_ids else None
    now = utcnow()
    batch = now.strftime("%Y%m%dT%H%M%S")
    released: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    already: list[int] = []
    for s in _graded_submissions(db, assignment.id):
        r = s.grade_result
        if r is None:
            continue
        if wanted is not None and s.id not in wanted:
            continue
        if manual_scores_missing(r):
            held.append({**_row(s), "reason": "incomplete scores"})
            continue
        if r.withheld_at is not None:
            held.append({**_row(s), "reason": "withheld"})
            continue
        if r.approved_at is None or s.status != "graded":
            held.append({**_row(s), "reason": "not approved"})
            continue
        if r.released_at is not None:
            already.append(s.id)
            continue
        opened = r.seen_at is not None
        r.released_at = now
        if not opened:
            r.seen_at = now
        _event(
            db,
            s,
            "released",
            batch=batch,
            opened=opened,
            edits=len(r.edit_log or []),
            model=r.model,
        )
        released.append({**_row(s), "state": "released", "opened_before_release": opened})
    db.commit()
    return {
        "assignment_id": assignment.id,
        "batch": batch,
        "released_at": _iso(now),
        "released": released,
        "held": held,
        "already_released": already,
        "counts": release_summary(db, assignment)["counts"],
    }


__all__ = [
    "REVIEW_STATES",
    "ReviewError",
    "review_dict",
    "manual_scores_missing",
    "mark_seen",
    "record_edit",
    "set_withheld",
    "record_regrade",
    "release_summary",
    "release",
    "events_for",
    "events_for_assignment",
    "event_dict",
]
