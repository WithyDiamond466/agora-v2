"""Student Insight layer — the grading event stream becomes understanding.

Increment 1, Feature A (docs/INCREMENT_1.md). Four stages, in order of trust:

1. **Deriver** (pure Python) — every graded submission emits ``Observation``
   rows: a criterion scored at or below half marks, a criterion at full marks,
   each misconception tag, each strength. No model writes these, so every
   downstream claim can be traced to a number on a real rubric. Idempotent per
   submission: re-grading replaces that submission's observations.
2. **Misconception decay** (pure Python) — per student+tag, ``active`` →
   ``resolving`` → ``resolved`` as the tag stops showing up in newer work.
3. **Card consolidation** (LLM through the existing ``Provider`` interface,
   with a deterministic template fallback) — the narrative the professor reads.
   The LLM only *phrases*; it cannot invent evidence (ids are validated against
   the student's real observations) and it cannot override a decay status.
4. **Nudge detection** (pure Python) — when three or more students share a
   misconception on one assignment, the course gets a "worth a look" nudge.

Privacy: observations are derived from grade results and rubric titles, which
never contain a student's name, so a card is safe to hand to the chat assistant
(students stay ``Student #7``) exactly like a graded submission is.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.models import (
    Assignment,
    Course,
    CourseNudge,
    GradeResult,
    Observation,
    Rubric,
    Skill,
    Student,
    StudentCard,
    Submission,
    utcnow,
)

log = logging.getLogger("agora.insight")


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

KIND_CRITERION_LOW = "criterion_low"
KIND_CRITERION_HIGH = "criterion_high"
KIND_MISCONCEPTION = "misconception"
KIND_STRENGTH = "strength"
KINDS = (KIND_CRITERION_LOW, KIND_CRITERION_HIGH, KIND_MISCONCEPTION, KIND_STRENGTH)

STATUS_ACTIVE = "active"
STATUS_RESOLVING = "resolving"
STATUS_RESOLVED = "resolved"
STATUSES = (STATUS_ACTIVE, STATUS_RESOLVING, STATUS_RESOLVED)

#: A criterion at or below this fraction of its maximum is a `criterion_low`.
LOW_RATIO = 0.5

#: How many students must share a tag on one assignment before it is a nudge.
NUDGE_MIN_STUDENTS = 3

MAX_STRENGTHS = 4
MAX_WEAKNESSES = 4
MAX_OBSERVATION_CHARS = 240


# --------------------------------------------------------------------------
# voice rules (enforced in the prompt AND on every template/LLM statement)
# --------------------------------------------------------------------------

#: docs/INCREMENT_1.md: "Every statement ≤25 words, grounded in numbers".
MAX_STATEMENT_WORDS = 25

#: Clinical / metric jargon a professor should never have to decode, plus the
#: preachy register the voice explicitly bans.
BANNED_PHRASES = (
    "brier",
    "z-score",
    "z score",
    "zscore",
    "p-value",
    "p value",
    "standard deviation",
    "confidence interval",
    "percentile",
    "sigma",
    "we recommend",
    "i recommend",
    "it is recommended",
    "you should",
    "make sure to",
    "be sure to",
    "needs to be reminded",
)

_BANNED_RE = re.compile("|".join(re.escape(p) for p in BANNED_PHRASES), re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def voice_violations(text: str) -> list[str]:
    """Every way ``text`` breaks the calibration voice (empty list = clean)."""
    problems: list[str] = []
    value = (text or "").strip()
    words = value.split()
    if len(words) > MAX_STATEMENT_WORDS:
        problems.append(f"{len(words)} words (limit {MAX_STATEMENT_WORDS})")
    for match in _BANNED_RE.finditer(value):
        problems.append(f"banned phrase {match.group(0)!r}")
    return problems


def enforce_voice(text: Any, *, limit: int = MAX_STATEMENT_WORDS) -> str:
    """Return ``text`` rewritten until ``voice_violations`` is empty.

    Banned phrasing is cut rather than paraphrased — a model that reaches for
    "we recommend" loses the words, not the fact underneath them.
    """
    value = _WS_RE.sub(" ", str(text or "")).strip()
    if not value:
        return ""
    value = _BANNED_RE.sub(" ", value)
    value = _WS_RE.sub(" ", value).strip(" ,;:-—")
    words = value.split()
    if len(words) > limit:
        value = " ".join(words[:limit]).rstrip(" ,;:-—")
        if not value.endswith("."):
            value += "."
    if value and value[0].islower() and not value[0].isdigit():
        value = value[0].upper() + value[1:]
    return value


# --------------------------------------------------------------------------
# settings ("Insight" section — defaults to the grading provider)
# --------------------------------------------------------------------------

INSIGHT_SETTINGS_FILENAME = "insight_settings.json"

DEFAULT_INSIGHT_SETTINGS: dict[str, Any] = {
    #: null → use the provider of the skill that graded this student's work.
    "provider": None,
    "model": None,
    #: Consolidate a card automatically when a student gains observations.
    "auto_refresh": True,
    #: Turn the LLM off entirely and always use the deterministic templates.
    "llm_enabled": True,
}


def insight_settings_path():
    """Resolved at call time so tests can repoint ``config.DATA_DIR``."""
    return config.DATA_DIR / INSIGHT_SETTINGS_FILENAME


def load_insight_settings() -> dict[str, Any]:
    """Insight settings with defaults filled in (never raises)."""
    settings = dict(DEFAULT_INSIGHT_SETTINGS)
    try:
        with open(insight_settings_path(), "r", encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        stored = {}
    if isinstance(stored, dict):
        for key in ("provider", "model"):
            value = stored.get(key)
            if isinstance(value, str) and value.strip():
                settings[key] = value.strip()
        for key in ("auto_refresh", "llm_enabled"):
            if key in stored:
                settings[key] = bool(stored.get(key))
    return settings


def save_insight_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Merge + persist the Insight settings; returns the stored result."""
    current = load_insight_settings()
    for key in ("provider", "model"):
        if key in settings:
            value = settings.get(key)
            current[key] = str(value).strip() if isinstance(value, str) and value.strip() else None
    for key in ("auto_refresh", "llm_enabled"):
        if key in settings:
            current[key] = bool(settings.get(key))
    config.ensure_dirs()
    with open(insight_settings_path(), "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2)
    return current


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pretty(value: float) -> str:
    """``10.0`` → ``10``; ``7.5`` → ``7.5``. Professors read points, not floats."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _short(text: Any, limit: int = 48) -> str:
    value = _WS_RE.sub(" ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _naive(value: Any) -> Optional[datetime]:
    """Compare DB timestamps without tripping over tz-awareness."""
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _rubric_titles(db: Session, assignment: Assignment | None) -> dict[str, str]:
    if assignment is None or not assignment.rubric_id:
        return {}
    rubric = db.get(Rubric, assignment.rubric_id)
    titles: dict[str, str] = {}
    for crit in (getattr(rubric, "criteria", None) or []):
        if isinstance(crit, dict) and crit.get("key"):
            titles[str(crit["key"])] = str(crit.get("title") or crit["key"])
    return titles


def _criterion_title(key: str, titles: dict[str, str]) -> str:
    return titles.get(key) or str(key).replace("_", " ").title()


def _tag_of(obs: Observation) -> str:
    data = obs.data if isinstance(obs.data, dict) else {}
    return str(data.get("tag") or "").strip().lower()


# --------------------------------------------------------------------------
# 1 · deriver (pure Python — no LLM ever writes an Observation)
# --------------------------------------------------------------------------


def derive_observations(
    db: Session,
    submission: Submission,
    result: GradeResult | None = None,
    *,
    commit: bool = True,
) -> list[Observation]:
    """Emit this submission's observations, replacing any it emitted before.

    Idempotent by construction: the submission's existing rows are deleted
    first, so re-grading a submission updates its evidence instead of doubling
    it.
    """
    if submission is None or submission.student_id is None:
        return []
    if result is None:
        result = submission.grade_result or db.scalars(
            select(GradeResult).where(GradeResult.submission_id == submission.id)
        ).first()

    existing = db.scalars(
        select(Observation).where(Observation.submission_id == submission.id)
    ).all()
    for row in existing:
        db.delete(row)
    db.flush()

    if result is None:
        if commit:
            db.commit()
        return []

    assignment = submission.assignment or (
        db.get(Assignment, submission.assignment_id) if submission.assignment_id else None
    )
    assignment_name = _short(getattr(assignment, "name", "") or "this assignment")
    titles = _rubric_titles(db, assignment)
    created_at = _naive(result.created_at) or _naive(utcnow())

    made: list[Observation] = []

    def _add(kind: str, text: str, data: dict[str, Any]) -> None:
        made.append(
            Observation(
                student_id=submission.student_id,
                assignment_id=submission.assignment_id,
                submission_id=submission.id,
                kind=kind,
                text=enforce_voice(text)[:MAX_OBSERVATION_CHARS],
                data=data,
                created_at=created_at,
            )
        )

    for entry in result.criteria or []:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "").strip()
        if not key:
            continue
        max_points = _number(entry.get("max_points"))
        if max_points <= 0:
            continue
        if entry.get("score") is None:
            # Feedback-only results and criteria left to the professor carry
            # no score yet; a missing number is not a zero.
            continue
        score = _number(entry.get("score"))
        title = _criterion_title(key, titles)
        payload = {
            "criterion_key": key,
            "criterion_title": title,
            "score": round(score, 2),
            "max_points": round(max_points, 2),
        }
        if score >= max_points:
            _add(
                KIND_CRITERION_HIGH,
                f"Full marks on {title}, {_pretty(score)} of {_pretty(max_points)}, "
                f"in {assignment_name}.",
                payload,
            )
        elif score <= LOW_RATIO * max_points:
            _add(
                KIND_CRITERION_LOW,
                f"Scored {_pretty(score)} of {_pretty(max_points)} on {title} "
                f"in {assignment_name}.",
                payload,
            )

    for raw in result.misconceptions or []:
        tag = str(raw or "").strip().lower()
        if not tag:
            continue
        _add(
            KIND_MISCONCEPTION,
            f"Tagged “{_short(tag, 70)}” in {assignment_name}.",
            {"tag": tag},
        )

    for raw in result.strengths or []:
        phrase = _WS_RE.sub(" ", str(raw or "")).strip()
        if not phrase:
            continue
        _add(
            KIND_STRENGTH,
            f"Credited for {_short(phrase, 70)} in {assignment_name}.",
            {"phrase": phrase},
        )

    db.add_all(made)
    if commit:
        db.commit()
    else:
        db.flush()
    return made


def observations_for(
    db: Session, student_id: int, *, limit: int | None = None, newest_first: bool = True
) -> list[Observation]:
    query = select(Observation).where(Observation.student_id == student_id)
    order = (
        (Observation.created_at.desc(), Observation.id.desc())
        if newest_first
        else (Observation.created_at.asc(), Observation.id.asc())
    )
    query = query.order_by(*order)
    if limit:
        query = query.limit(int(limit))
    return list(db.scalars(query).all())


def observation_dict(obs: Observation) -> dict[str, Any]:
    # `assignment` is a relationship load served from the session identity map
    # after the first row of a feed, so this is one query per assignment.
    assignment = obs.assignment if obs.assignment_id else None
    return {
        "id": obs.id,
        "student_id": obs.student_id,
        "assignment_id": obs.assignment_id,
        "assignment_name": getattr(assignment, "name", None),
        "submission_id": obs.submission_id,
        "kind": obs.kind,
        "text": obs.text,
        "data": obs.data or {},
        "created_at": obs.created_at.isoformat() if obs.created_at else None,
    }


# --------------------------------------------------------------------------
# 2 · misconception decay (pure Python)
# --------------------------------------------------------------------------


def graded_assignment_order(db: Session, student_id: int) -> list[int]:
    """The student's graded assignments, oldest → newest (course order)."""
    rows = db.execute(
        select(Assignment.id)
        .join(Submission, Submission.assignment_id == Assignment.id)
        .join(GradeResult, GradeResult.submission_id == Submission.id)
        .where(Submission.student_id == student_id)
        .order_by(Assignment.id)
    ).all()
    seen: list[int] = []
    for (assignment_id,) in rows:
        if assignment_id is not None and assignment_id not in seen:
            seen.append(assignment_id)
    return seen


def decay_status(recency: int) -> str:
    """Map "assignments since the tag was last seen" onto a decay state.

    ``recency`` is 0 when the tag is in the most recent graded assignment.

    docs/INCREMENT_1.md rule 1 is the explicit one and it wins: a tag in
    *either* of the two most recent graded assignments is ``active``, so
    ``recency <= 1``. ``resolving`` is then the first assignment past that
    window (``recency == 2``), and ``resolved`` is anything beyond it — which
    also satisfies rule 3's "≥2 assignments graded since it was last seen".
    """
    if recency <= 1:
        return STATUS_ACTIVE
    if recency == 2:
        return STATUS_RESOLVING
    return STATUS_RESOLVED


def misconception_states(db: Session, student_id: int) -> list[dict[str, Any]]:
    """Per-tag decay state for one student, most-recently-seen first."""
    order = graded_assignment_order(db, student_id)
    if not order:
        # Nothing has been graded, so "assignments since last seen" has no
        # meaning; reporting every stale tag as active would be a lie.
        return []
    position = {assignment_id: index for index, assignment_id in enumerate(order)}
    newest = len(order) - 1

    rows = db.scalars(
        select(Observation)
        .where(
            Observation.student_id == student_id,
            Observation.kind == KIND_MISCONCEPTION,
        )
        .order_by(Observation.id)
    ).all()

    by_tag: dict[str, dict[str, Any]] = {}
    for obs in rows:
        tag = _tag_of(obs)
        if not tag:
            continue
        if obs.assignment_id not in position:
            # An observation whose assignment is not in the graded order
            # (ungraded now, deleted, orphaned) says nothing about recency;
            # defaulting it to the newest slot pinned the tag to `active`
            # forever.
            continue
        entry = by_tag.setdefault(
            tag,
            {
                "tag": tag,
                "evidence": [],
                "assignment_ids": [],
                "last_index": -1,
                "count": 0,
            },
        )
        entry["evidence"].append(obs.id)
        entry["count"] += 1
        index = position[obs.assignment_id]
        if obs.assignment_id is not None and obs.assignment_id not in entry["assignment_ids"]:
            entry["assignment_ids"].append(obs.assignment_id)
        entry["last_index"] = max(entry["last_index"], index)

    states: list[dict[str, Any]] = []
    for entry in by_tag.values():
        recency = max(0, newest - entry["last_index"])
        states.append(
            {
                "tag": entry["tag"],
                "status": decay_status(recency),
                "evidence": entry["evidence"],
                "assignment_ids": entry["assignment_ids"],
                "seen_count": entry["count"],
                "assignments_since_last_seen": recency,
                "graded_assignments": len(order),
            }
        )
    states.sort(key=lambda s: (s["assignments_since_last_seen"], -s["seen_count"], s["tag"]))
    return states


# --------------------------------------------------------------------------
# card input (what the consolidator reasons over — anonymized)
# --------------------------------------------------------------------------


def score_timeline(db: Session, student_id: int) -> list[dict[str, Any]]:
    """Score per graded assignment, oldest → newest. No names, ever."""
    rows = db.execute(
        select(GradeResult, Assignment)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .join(Assignment, Submission.assignment_id == Assignment.id)
        .where(Submission.student_id == student_id)
        .order_by(Assignment.id, GradeResult.id)
    ).all()
    out: list[dict[str, Any]] = []
    for result, assignment in rows:
        max_score = _number(result.max_score)
        out.append(
            {
                "assignment_id": assignment.id,
                "assignment": assignment.name,
                "score": round(_number(result.overall_score), 2),
                "max_score": round(max_score, 2),
                "percent": (
                    round(100.0 * _number(result.overall_score) / max_score, 1)
                    if max_score
                    else None
                ),
            }
        )
    return out


def build_card_input(db: Session, student_id: int) -> dict[str, Any]:
    """Everything the consolidator gets: observations, decay, score timeline."""
    student = db.get(Student, student_id)
    if student is None:
        raise LookupError(f"student {student_id} not found")
    course = db.get(Course, student.course_id) if student.course_id else None

    observations = observations_for(db, student_id, newest_first=False)
    timeline = score_timeline(db, student_id)
    percents = [p["percent"] for p in timeline if p["percent"] is not None]

    criteria: dict[str, dict[str, Any]] = {}
    for obs in observations:
        if obs.kind not in (KIND_CRITERION_LOW, KIND_CRITERION_HIGH):
            continue
        data = obs.data if isinstance(obs.data, dict) else {}
        key = str(data.get("criterion_key") or "")
        if not key:
            continue
        entry = criteria.setdefault(
            key,
            {
                "key": key,
                "title": str(data.get("criterion_title") or key),
                "low_count": 0,
                "high_count": 0,
                "evidence_low": [],
                "evidence_high": [],
                "last_score": None,
                "max_points": _number(data.get("max_points")),
            },
        )
        if obs.kind == KIND_CRITERION_LOW:
            entry["low_count"] += 1
            entry["evidence_low"].append(obs.id)
        else:
            entry["high_count"] += 1
            entry["evidence_high"].append(obs.id)
        entry["last_score"] = _number(data.get("score"))

    strengths: dict[str, dict[str, Any]] = {}
    for obs in observations:
        if obs.kind != KIND_STRENGTH:
            continue
        data = obs.data if isinstance(obs.data, dict) else {}
        phrase = str(data.get("phrase") or "").strip()
        if not phrase:
            continue
        entry = strengths.setdefault(
            phrase.lower(), {"phrase": phrase, "count": 0, "evidence": []}
        )
        entry["count"] += 1
        entry["evidence"].append(obs.id)

    return {
        "student": {
            # Privacy: the per-course number is the only identifier that may
            # leave the machine (SPEC principle 2).
            "label": student.anon_label,
            "student_number": student.student_number,
        },
        "course": {"name": getattr(course, "name", None)},
        "graded_assignments": len(timeline),
        "average_percent": (round(sum(percents) / len(percents), 1) if percents else None),
        "timeline": timeline,
        "criteria": sorted(
            criteria.values(), key=lambda c: (-c["low_count"], c["high_count"], c["key"])
        ),
        "repeated_strengths": sorted(
            strengths.values(), key=lambda s: (-s["count"], s["phrase"])
        ),
        "misconception_state": misconception_states(db, student_id),
        "observations": [
            {
                "id": obs.id,
                "kind": obs.kind,
                "text": obs.text,
                "assignment_id": obs.assignment_id,
                "data": obs.data or {},
            }
            for obs in observations
        ],
    }


# --------------------------------------------------------------------------
# 3a · template fallback (deterministic — the feature degrades, never breaks)
# --------------------------------------------------------------------------


def template_card(card_input: dict[str, Any]) -> dict[str, Any]:
    """Build a schema-valid card from templates only.

    Same voice rules as the prompt: one short sentence per statement, numbers
    the professor can check, evidence ids on every claim.
    """
    label = card_input.get("student", {}).get("label") or "This student"
    graded = int(card_input.get("graded_assignments") or 0)
    average = card_input.get("average_percent")
    timeline = card_input.get("timeline") or []
    criteria = card_input.get("criteria") or []
    states = card_input.get("misconception_state") or []

    weaknesses: list[dict[str, Any]] = []
    for entry in criteria:
        if not entry.get("low_count"):
            continue
        weaknesses.append(
            {
                "text": enforce_voice(
                    f"{entry['title']} came in at or below half marks on "
                    f"{entry['low_count']} of {graded or entry['low_count']} graded "
                    "assignments."
                ),
                "evidence": list(entry.get("evidence_low") or []),
            }
        )
        if len(weaknesses) >= MAX_WEAKNESSES:
            break

    strengths: list[dict[str, Any]] = []
    for entry in criteria:
        if not entry.get("high_count"):
            continue
        strengths.append(
            {
                "text": enforce_voice(
                    f"Full marks on {entry['title']} in {entry['high_count']} of "
                    f"{graded or entry['high_count']} graded assignments."
                ),
                "evidence": list(entry.get("evidence_high") or []),
            }
        )
        if len(strengths) >= MAX_STRENGTHS:
            break
    for entry in card_input.get("repeated_strengths") or []:
        if len(strengths) >= MAX_STRENGTHS:
            break
        if entry.get("count", 0) < 2:
            continue
        strengths.append(
            {
                "text": enforce_voice(
                    f"Graders credited {entry['phrase']} on {entry['count']} assignments."
                ),
                "evidence": list(entry.get("evidence") or []),
            }
        )

    if not strengths and not weaknesses and not states:
        summary = enforce_voice(f"{label} has no graded work yet, so there is nothing to report.")
    elif average is not None:
        weakest = weaknesses[0]["text"] if weaknesses else ""
        summary = enforce_voice(
            f"{label} averages {average:g}% across {graded} graded "
            f"{'assignment' if graded == 1 else 'assignments'}."
            + (f" {weakest}" if weakest else "")
        )
    else:
        summary = enforce_voice(f"{label} has {graded} graded assignments and no scored rubric yet.")

    if len(timeline) >= 2:
        first = timeline[0].get("percent")
        last = timeline[-1].get("percent")
        if first is not None and last is not None:
            delta = round(last - first, 1)
            direction = "up" if delta > 0 else "down" if delta < 0 else "flat at"
            trajectory = enforce_voice(
                f"{direction.capitalize()} {abs(delta):g} points from {first:g}% to {last:g}% "
                f"across {len(timeline)} assignments."
                if delta
                else f"Flat at {last:g}% across {len(timeline)} assignments."
            )
        else:
            trajectory = enforce_voice(f"{len(timeline)} assignments graded; scores are missing.")
    elif len(timeline) == 1:
        percent = timeline[0].get("percent")
        trajectory = enforce_voice(
            f"One graded assignment so far, at {percent:g}%." if percent is not None
            else "One graded assignment so far."
        )
    else:
        trajectory = enforce_voice("No graded work yet.")

    return {
        "summary": summary,
        "trajectory": trajectory,
        "strengths": strengths[:MAX_STRENGTHS],
        "weaknesses": weaknesses[:MAX_WEAKNESSES],
        "misconception_state": [
            {"tag": s["tag"], "status": s["status"], "evidence": list(s.get("evidence") or [])}
            for s in states
        ],
    }


# --------------------------------------------------------------------------
# 3b · LLM consolidation (Provider interface + strict validation)
# --------------------------------------------------------------------------

CARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "trajectory": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "evidence"],
                "additionalProperties": False,
            },
        },
        "weaknesses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "evidence"],
                "additionalProperties": False,
            },
        },
        "misconception_state": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "evidence": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["tag", "status", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "trajectory", "strengths", "weaknesses", "misconception_state"],
    "additionalProperties": False,
}

CARD_SYSTEM_PROMPT = f"""
You maintain one professor's running understanding of one anonymous student.
You are given DATA: deterministic observations already derived from graded
rubrics, the decay state of each misconception, and the student's scores.

Write the card. You do not grade, and you do not add facts: every sentence must
restate something visible in DATA.

Voice rules (hard requirements):
- Every statement is one sentence of at most {MAX_STATEMENT_WORDS} words.
- Ground every statement in numbers from DATA
  ("Missed citation format on 2 of 3 essays — was every essay before").
- No clinical or statistical jargon: never write "Brier", "z-score",
  "standard deviation", "percentile" or similar.
- Never preach and never recommend. No "we recommend", no "you should".
  Plain, confident prose written for a colleague.
- Every strengths, weaknesses and misconception entry carries `evidence`: the
  ids of the DATA observations it rests on. Use only ids present in DATA; an
  entry you cannot evidence must be dropped.
- Keep every misconception `status` exactly as DATA reports it. The decay state
  is computed, not judged.
- The student is identified only by number. Never speculate about identity.

Return only the JSON object described by the schema.
""".strip()

#: The user block carries the input as JSON. MockProvider parses it straight
#: back out (same contract as the rubric line format in the grading prompt).
CARD_INPUT_MARKER = "DATA (JSON)"


def card_user_text(card_input: dict[str, Any]) -> str:
    return f"{CARD_INPUT_MARKER}\n{json.dumps(card_input, indent=2, sort_keys=True)}"


def _coerce_entries(
    raw: Any, allowed: set[int], *, limit: int
) -> list[dict[str, Any]]:
    """Sanitize LLM strength/weakness entries; drop anything unevidenced."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        text = enforce_voice(item.get("text"))
        if not text or text.lower() in seen:
            continue
        evidence: list[int] = []
        for value in item.get("evidence") or []:
            try:
                obs_id = int(value)
            except (TypeError, ValueError):
                continue
            # A model may not cite evidence that does not exist (same rule the
            # Privacy Guard applies to hallucinated spans).
            if obs_id in allowed and obs_id not in evidence:
                evidence.append(obs_id)
        if not evidence:
            continue
        seen.add(text.lower())
        out.append({"text": text, "evidence": evidence})
        if len(out) >= limit:
            break
    return out


def validate_card_payload(payload: Any, card_input: dict[str, Any]) -> dict[str, Any]:
    """Turn a model payload into a card that obeys the contract.

    Raises ``ValueError`` when the payload is not usable at all — the caller
    then falls back to the templates.
    """
    if not isinstance(payload, dict):
        raise ValueError("card payload was not a JSON object")

    allowed = {int(obs["id"]) for obs in card_input.get("observations") or [] if obs.get("id")}
    states = {s["tag"]: s for s in card_input.get("misconception_state") or []}

    summary = enforce_voice(payload.get("summary"))
    trajectory = enforce_voice(payload.get("trajectory"))
    if not summary:
        raise ValueError("card payload had no usable summary")

    # The decay state is computed, never judged: the model may only rephrase
    # tags it was given, and the status/evidence come from our own pass.
    misconceptions: list[dict[str, Any]] = []
    for item in payload.get("misconception_state") or []:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "").strip().lower()
        state = states.get(tag)
        if state is None:
            continue
        misconceptions.append(
            {
                "tag": tag,
                "status": state["status"],
                "evidence": list(state.get("evidence") or []),
            }
        )
    known = {m["tag"] for m in misconceptions}
    for tag, state in states.items():
        if tag not in known:
            misconceptions.append(
                {
                    "tag": tag,
                    "status": state["status"],
                    "evidence": list(state.get("evidence") or []),
                }
            )

    return {
        "summary": summary,
        "trajectory": trajectory or template_card(card_input)["trajectory"],
        "strengths": _coerce_entries(payload.get("strengths"), allowed, limit=MAX_STRENGTHS),
        "weaknesses": _coerce_entries(payload.get("weaknesses"), allowed, limit=MAX_WEAKNESSES),
        "misconception_state": misconceptions,
    }


def grading_provider_settings(db: Session, student_id: int) -> dict[str, Any]:
    """Default the Insight model to whatever graded this student's work."""
    skill = db.scalars(
        select(Skill)
        .join(Assignment, Assignment.skill_id == Skill.id)
        .join(Submission, Submission.assignment_id == Assignment.id)
        .where(Submission.student_id == student_id)
        .order_by(Assignment.id.desc())
    ).first()
    if skill is not None:
        return {
            "provider": skill.provider,
            "model": skill.model,
            "max_tokens": skill.max_tokens,
        }
    return {"provider": config.DEFAULT_PROVIDER}


def get_card_provider(db: Session, student_id: int) -> Any:
    """Build the provider that writes cards (Settings → Insight, or grading)."""
    from app.ai.providers import get_provider  # local import: cycle-safe

    settings = load_insight_settings()
    request = grading_provider_settings(db, student_id)
    if settings.get("provider"):
        request = {"provider": settings["provider"]}
    if settings.get("model"):
        request["model"] = settings["model"]
    return get_provider(request, db=db)


def call_card_provider(provider: Any, card_input: dict[str, Any], *, protected_text: str | None = None) -> dict[str, Any]:
    """One structured-JSON call through the Provider interface.

    ``structured_json`` is MockProvider's deterministic schema-shaped path
    (providers.py, Increment 1 insight section); every other provider answers
    arbitrary schemas through ``grade``, which takes the schema as an argument.
    """
    from app.ai.providers import text_block  # local import: cycle-safe

    blocks = [text_block(protected_text if protected_text is not None else card_user_text(card_input))]
    call = getattr(provider, "structured_json", None)
    if not callable(call):
        call = provider.grade
    return call(CARD_SYSTEM_PROMPT, blocks, CARD_SCHEMA)


def consolidate_card(
    db: Session,
    student_id: int,
    *,
    provider: Any = None,
    allow_llm: bool = True,
    card_input: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, Optional[str]]:
    """Return ``(card_payload, model, warning)``.

    Never raises for provider reasons: a failed call degrades to the templates
    with the reason recorded, because a professor mid-grading-week would rather
    read a plainer card than an error.
    """
    data = card_input if card_input is not None else build_card_input(db, student_id)
    settings = load_insight_settings()
    if not allow_llm or not settings.get("llm_enabled", True):
        return template_card(data), "template", None

    try:
        if provider is None:
            provider = get_card_provider(db, student_id)
        if config.is_cloud_provider(getattr(provider, "name", "")) and config.privacy_mode() == config.PRIVACY_MODE_SWAP:
            from app.ai.privacy import protect_cloud_text
            student = db.get(Student, student_id)
            cleaned, _ = protect_cloud_text(db, card_user_text(data),
                course_id=student.course_id if student else None, subject="student insights")
            payload = call_card_provider(provider, data, protected_text=cleaned)
        else:
            payload = call_card_provider(provider, data)
        card = validate_card_payload(payload, data)
        return card, str(getattr(provider, "model", "") or "llm"), None
    except Exception as exc:  # noqa: BLE001 - degrade, never break
        log.info("Insight card fell back to templates for student %s: %s", student_id, exc)
        return template_card(data), "template", str(exc)


# --------------------------------------------------------------------------
# card persistence
# --------------------------------------------------------------------------


def get_stored_card(db: Session, student_id: int) -> Optional[StudentCard]:
    return db.scalars(
        select(StudentCard).where(StudentCard.student_id == student_id)
    ).first()


def card_is_stale(db: Session, student_id: int, card: StudentCard | None = None) -> bool:
    """True when observations have moved on since the card was written."""
    card = card if card is not None else get_stored_card(db, student_id)
    if card is None:
        return True
    newest = db.scalars(
        select(Observation.created_at)
        .where(Observation.student_id == student_id)
        .order_by(Observation.created_at.desc(), Observation.id.desc())
        .limit(1)
    ).first()
    if newest is None:
        return False
    updated = _naive(card.updated_at)
    newest = _naive(newest)
    if updated is None or newest is None:
        return True
    return newest > updated


def refresh_card(
    db: Session,
    student_id: int,
    *,
    provider: Any = None,
    allow_llm: bool = True,
    commit: bool = True,
) -> StudentCard:
    """(Re)build and persist the card for one student."""
    payload, model, warning = consolidate_card(
        db, student_id, provider=provider, allow_llm=allow_llm
    )
    card = get_stored_card(db, student_id)
    if card is None:
        card = StudentCard(student_id=student_id)
        db.add(card)
    card.summary = payload["summary"]
    card.trajectory = payload["trajectory"]
    card.strengths = payload["strengths"]
    card.weaknesses = payload["weaknesses"]
    card.misconception_state = payload["misconception_state"]
    card.model = model
    card.updated_at = utcnow()
    if commit:
        db.commit()
        db.refresh(card)
    else:
        db.flush()
    if warning:
        log.debug("Card for student %s built from templates (%s)", student_id, warning)
    return card


def card_dict(card: StudentCard | None) -> Optional[dict[str, Any]]:
    if card is None:
        return None
    return {
        "student_id": card.student_id,
        "summary": card.summary,
        "trajectory": card.trajectory,
        "strengths": card.strengths or [],
        "weaknesses": card.weaknesses or [],
        "misconception_state": card.misconception_state or [],
        "model": card.model,
        "updated_at": card.updated_at.isoformat() if card.updated_at else None,
    }


def get_card(
    db: Session, student_id: int, *, auto: bool = True, provider: Any = None
) -> Optional[StudentCard]:
    """The student's card, rebuilt on demand when it is missing or stale."""
    card = get_stored_card(db, student_id)
    if not auto:
        return card
    if card is None or card_is_stale(db, student_id, card):
        if not observations_for(db, student_id, limit=1) and card is None:
            return None
        card = refresh_card(db, student_id, provider=provider)
    return card


def card_for_tool(db: Session, student_id: int) -> Optional[dict[str, Any]]:
    """The card as the chat assistant should see it (no names, no ids to leak)."""
    card = get_stored_card(db, student_id)
    if card is None:
        return None
    return {
        "summary": card.summary,
        "trajectory": card.trajectory,
        "strengths": [s.get("text") for s in (card.strengths or []) if isinstance(s, dict)],
        "weaknesses": [w.get("text") for w in (card.weaknesses or []) if isinstance(w, dict)],
        "misconceptions": [
            {"tag": m.get("tag"), "status": m.get("status")}
            for m in (card.misconception_state or [])
            if isinstance(m, dict)
        ],
        "updated_at": card.updated_at.isoformat() if card.updated_at else None,
        "source": card.model,
    }


# --------------------------------------------------------------------------
# 4 · nudge detection (pure Python trigger, templated phrasing)
# --------------------------------------------------------------------------


def _nudge_text(tag: str, student_count: int, assignment_name: str) -> str:
    return enforce_voice(
        f"{student_count} students were tagged “{_short(tag, 60)}” on "
        f"{_short(assignment_name, 40)} — worth a look."
    )


def detect_nudges(
    db: Session, assignment_id: int, *, commit: bool = True
) -> list[CourseNudge]:
    """Create a nudge per tag ≥3 students share on this assignment.

    Deduped on (course, assignment, tag): an existing nudge has its evidence
    refreshed instead of being duplicated, and a dismissed one stays dismissed.
    """
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        return []

    rows = db.scalars(
        select(Observation)
        .where(
            Observation.assignment_id == assignment_id,
            Observation.kind == KIND_MISCONCEPTION,
        )
        .order_by(Observation.id)
    ).all()

    by_tag: dict[str, dict[str, Any]] = {}
    for obs in rows:
        tag = _tag_of(obs)
        if not tag:
            continue
        entry = by_tag.setdefault(tag, {"students": [], "observations": []})
        if obs.student_id not in entry["students"]:
            entry["students"].append(obs.student_id)
        entry["observations"].append(obs.id)

    existing_rows = db.scalars(
        select(CourseNudge)
        .where(CourseNudge.assignment_id == assignment_id)
        .order_by(CourseNudge.id)
    ).all()
    qualifying_tags = {
        tag
        for tag, entry in by_tag.items()
        if len(entry["students"]) >= NUDGE_MIN_STUDENTS
    }
    existing: dict[str, CourseNudge] = {}
    for nudge in existing_rows:
        tag = nudge.evidence.get("tag") if isinstance(nudge.evidence, dict) else None
        if tag not in qualifying_tags:
            db.delete(nudge)
            continue
        survivor = existing.get(tag)
        if survivor is None:
            existing[tag] = nudge
            continue
        if nudge.dismissed_at is not None and (
            survivor.dismissed_at is None or nudge.dismissed_at < survivor.dismissed_at
        ):
            survivor.dismissed_at = nudge.dismissed_at
        db.delete(nudge)

    touched: list[CourseNudge] = []
    for tag in sorted(qualifying_tags):
        entry = by_tag[tag]
        evidence = {
            "tag": tag,
            "student_ids": sorted(entry["students"]),
            "observation_ids": sorted(entry["observations"]),
        }
        text = _nudge_text(tag, len(entry["students"]), assignment.name)
        nudge = existing.get(tag)
        if nudge is None:
            nudge = CourseNudge(
                course_id=assignment.course_id,
                assignment_id=assignment_id,
                text=text,
                evidence=evidence,
            )
            db.add(nudge)
        else:
            # Same pattern, more evidence — refresh in place, keep dismissal.
            nudge.text = text
            nudge.evidence = evidence
        touched.append(nudge)

    if commit:
        db.commit()
    else:
        db.flush()
    return touched


def nudges_for_course(
    db: Session, course_id: int, *, include_dismissed: bool = False
) -> list[CourseNudge]:
    query = select(CourseNudge).where(CourseNudge.course_id == course_id)
    if not include_dismissed:
        query = query.where(CourseNudge.dismissed_at.is_(None))
    return list(db.scalars(query.order_by(CourseNudge.id.desc())).all())


def dismiss_nudge(db: Session, nudge_id: int) -> Optional[CourseNudge]:
    nudge = db.get(CourseNudge, nudge_id)
    if nudge is None:
        return None
    if nudge.dismissed_at is None:
        nudge.dismissed_at = utcnow()
        db.commit()
        db.refresh(nudge)
    return nudge


def nudge_dict(nudge: CourseNudge) -> dict[str, Any]:
    return {
        "id": nudge.id,
        "course_id": nudge.course_id,
        "assignment_id": nudge.assignment_id,
        "text": nudge.text,
        "evidence": nudge.evidence or {},
        "created_at": nudge.created_at.isoformat() if nudge.created_at else None,
        "dismissed_at": nudge.dismissed_at.isoformat() if nudge.dismissed_at else None,
        "dismissed": nudge.dismissed_at is not None,
    }


# --------------------------------------------------------------------------
# the post-grading hook (called once from app/ai/grading.py)
# --------------------------------------------------------------------------


def on_submission_graded(
    db: Session,
    submission: Submission,
    result: GradeResult | None = None,
    *,
    consolidate: bool | None = None,
    provider: Any = None,
) -> dict[str, Any]:
    """Everything the insight layer does when one submission is graded.

    Deterministic work first (observations, nudges) so the evidence exists even
    if the model call is impossible; the card is consolidated once per graded
    submission — not once per observation — and any failure degrades to the
    templates. This function never raises: grading must not fail because the
    insight layer had a bad day.
    """
    summary: dict[str, Any] = {"observations": 0, "nudges": 0, "card": None}
    try:
        observations = derive_observations(db, submission, result)
        summary["observations"] = len(observations)
        if submission.assignment_id:
            summary["nudges"] = len(detect_nudges(db, submission.assignment_id))
        if consolidate is None:
            consolidate = bool(load_insight_settings().get("auto_refresh", True))
        if consolidate and submission.student_id and observations:
            card = refresh_card(db, submission.student_id, provider=provider)
            summary["card"] = card.model
    except Exception as exc:  # noqa: BLE001 - grading must never fail from here
        log.warning("Insight hook failed for submission %s: %s", getattr(submission, "id", "?"), exc)
        try:
            db.rollback()
        except Exception:  # pragma: no cover - defensive
            pass
    return summary


def rebuild_student(
    db: Session, student_id: int, *, provider: Any = None, allow_llm: bool = True
) -> StudentCard:
    """Re-derive every observation for a student, then rebuild their card."""
    submissions = db.scalars(
        select(Submission)
        .join(GradeResult, GradeResult.submission_id == Submission.id)
        .where(Submission.student_id == student_id)
        .order_by(Submission.id)
    ).all()
    for submission in submissions:
        derive_observations(db, submission, commit=False)
    db.commit()
    return refresh_card(db, student_id, provider=provider, allow_llm=allow_llm)


__all__ = [
    "KINDS",
    "STATUSES",
    "STATUS_ACTIVE",
    "STATUS_RESOLVING",
    "STATUS_RESOLVED",
    "CARD_SCHEMA",
    "CARD_SYSTEM_PROMPT",
    "CARD_INPUT_MARKER",
    "MAX_STATEMENT_WORDS",
    "enforce_voice",
    "voice_violations",
    "load_insight_settings",
    "save_insight_settings",
    "derive_observations",
    "observations_for",
    "observation_dict",
    "graded_assignment_order",
    "decay_status",
    "misconception_states",
    "score_timeline",
    "build_card_input",
    "template_card",
    "card_user_text",
    "validate_card_payload",
    "consolidate_card",
    "get_stored_card",
    "card_is_stale",
    "refresh_card",
    "card_dict",
    "get_card",
    "card_for_tool",
    "detect_nudges",
    "nudges_for_course",
    "dismiss_nudge",
    "nudge_dict",
    "on_submission_graded",
    "rebuild_student",
]
