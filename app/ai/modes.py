"""Skill modes — what a skill produces when it reads a submission.

Three built-in modes cover what a professor reaches for first:

* ``grade``      — score every rubric criterion and write feedback (the original).
* ``feedback``   — comments only: per-criterion notes and a summary, no scores.
* ``selective``  — score only the criteria the assignment picks; the rest are
                   left for the professor, marked ``manual`` on the result.

Anything else is a professor-added mode: a small JSON file in
``data/modes/`` (or a ``mode_spec`` carried in an ``.agoraskill`` bundle) with
the same five fields. The app never needs to change to accept one — the
registry below is the whole contract, the way an editor's package format is.

A mode decides three things the engine reads: the instructions that replace
the default grading brief, whether criteria carry scores, and whether only
the assignment's selected criteria are sent.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from app import config

log = logging.getLogger("agora.ai.modes")

MODE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,39}$")

GRADE_INSTRUCTIONS = """
You are grading a single anonymized student submission against the rubric below.

Rules:
- Score every rubric criterion, using its exact `key`. Never invent a criterion.
- A score must be between 0 and that criterion's maximum points.
- Comments address the student in the second person and point at specific moves
  in their work, not generic praise.
- `misconceptions` are short, reusable, lower-case tags naming a *conceptual*
  error (e.g. "conflates legality with morality"), not a description of this
  one submission. Reuse the same wording across students so the class-level
  analytics can group them. Return an empty list if there are none.
- `strengths` are short phrases, same style.
- The student is identified only by number. Do not speculate about identity.
""".strip()

FEEDBACK_INSTRUCTIONS = """
You are writing feedback on a single anonymized student submission, using the
rubric below as the lens. You do not assign scores — the instructor does.

Rules:
- Comment on every rubric criterion, using its exact `key`. Never invent a criterion.
- Each comment addresses the student in the second person, names one specific
  thing they did and one specific thing to do next time. No numbers, no grades,
  no "this would earn".
- `summary_feedback` is a short paragraph the student can act on.
- `misconceptions` are short, reusable, lower-case tags naming a *conceptual*
  error. Reuse the same wording across students. Empty list if none.
- `strengths` are short phrases, same style.
- The student is identified only by number. Do not speculate about identity.
""".strip()

SELECTIVE_INSTRUCTIONS = """
You are grading part of a single anonymized student submission. The rubric
below lists only the criteria you are responsible for; the instructor grades
the others by hand and you must not mention or score them.

Rules:
- Score every listed criterion, using its exact `key`. Never invent a criterion.
- A score must be between 0 and that criterion's maximum points.
- Comments address the student in the second person and point at specific moves
  in their work, not generic praise.
- `summary_feedback` covers only the listed criteria.
- `misconceptions` are short, reusable, lower-case tags naming a *conceptual*
  error. Reuse the same wording across students. Empty list if none.
- `strengths` are short phrases, same style.
- The student is identified only by number. Do not speculate about identity.
""".strip()


@dataclass(frozen=True)
class Mode:
    id: str
    label: str
    description: str
    #: Criteria carry numeric scores (feedback-only modes set this False).
    scored: bool
    #: Only the assignment's ``ai_criteria`` are sent; the rest are manual.
    selective: bool
    #: Replaces the default grading brief in the system prompt.
    instructions: str
    builtin: bool = False

    def schema(self) -> dict[str, Any]:
        """Structured-output schema for this mode (single source of truth)."""
        item_props: dict[str, Any] = {"key": {"type": "string"}}
        required = ["key"]
        if self.scored:
            item_props["score"] = {"type": "number"}
            required.append("score")
        item_props["comment"] = {"type": "string"}
        required.append("comment")
        return {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": item_props,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
                "summary_feedback": {"type": "string"},
                "misconceptions": {"type": "array", "items": {"type": "string"}},
                "strengths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["criteria", "summary_feedback", "misconceptions", "strengths"],
            "additionalProperties": False,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = self.schema()
        return data


BUILTIN_MODES: dict[str, Mode] = {
    "grade": Mode(
        id="grade",
        label="AI grading",
        description="Scores every rubric criterion and writes feedback. You review and release.",
        scored=True,
        selective=False,
        instructions=GRADE_INSTRUCTIONS,
        builtin=True,
    ),
    "feedback": Mode(
        id="feedback",
        label="AI feedback",
        description="Comments only — no scores. You grade; the AI drafts what the student reads.",
        scored=False,
        selective=False,
        instructions=FEEDBACK_INSTRUCTIONS,
        builtin=True,
    ),
    "selective": Mode(
        id="selective",
        label="AI selective grading",
        description="Scores only the criteria you tick on each assignment; the rest stay yours.",
        scored=True,
        selective=True,
        instructions=SELECTIVE_INSTRUCTIONS,
        builtin=True,
    ),
}

DEFAULT_MODE = "grade"


class ModeError(ValueError):
    """A custom mode definition that cannot be accepted."""


# --------------------------------------------------------------------------
# professor-added modes
# --------------------------------------------------------------------------


def custom_modes_dir() -> Path:
    """Resolved at call time so tests can repoint ``DATA_DIR``."""
    return config.DATA_DIR / "modes"


def validate_mode_spec(raw: Any) -> Mode:
    """Turn a JSON object into a Mode, or raise ``ModeError`` saying why."""
    if not isinstance(raw, dict):
        raise ModeError("A mode definition must be a JSON object")
    mode_id = str(raw.get("id") or "").strip().lower()
    if not MODE_ID_RE.match(mode_id):
        raise ModeError(
            "Mode 'id' must be 2-40 characters: lower-case letters, digits, '-' or '_', "
            "starting with a letter"
        )
    if mode_id in BUILTIN_MODES:
        raise ModeError(f"'{mode_id}' is a built-in mode and cannot be redefined")
    label = str(raw.get("label") or "").strip()
    if not label:
        raise ModeError("Mode 'label' is required")
    instructions = str(raw.get("instructions") or "").strip()
    if len(instructions) < 20:
        raise ModeError("Mode 'instructions' must be at least 20 characters — this is the brief the model grades by")
    if len(instructions) > 20_000:
        raise ModeError("Mode 'instructions' is too long (20,000 characters max)")
    for flag in ("scored", "selective"):
        if flag in raw and not isinstance(raw[flag], bool):
            raise ModeError(f"Mode '{flag}' must be true or false")
    return Mode(
        id=mode_id,
        label=label[:80],
        description=str(raw.get("description") or "").strip()[:400],
        scored=bool(raw.get("scored", True)),
        selective=bool(raw.get("selective", False)),
        instructions=instructions,
        builtin=False,
    )


def load_custom_modes() -> dict[str, Mode]:
    modes: dict[str, Mode] = {}
    directory = custom_modes_dir()
    if not directory.is_dir():
        return modes
    for path in sorted(directory.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                mode = validate_mode_spec(json.load(fh))
        except (OSError, ValueError, ModeError) as exc:
            log.warning("Skipping custom mode %s: %s", path.name, exc)
            continue
        modes[mode.id] = mode
    return modes


def save_custom_mode(raw: Any, *, replace: bool = True) -> Mode:
    """Write a professor-added mode to ``data/modes/<id>.json``."""
    mode = validate_mode_spec(raw)
    directory = custom_modes_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{mode.id}.json"
    if path.exists() and not replace:
        return load_custom_modes().get(mode.id, mode)
    spec = {k: v for k, v in asdict(mode).items() if k != "builtin"}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    return mode


def delete_custom_mode(mode_id: str) -> bool:
    mode_id = str(mode_id or "").strip().lower()
    if mode_id in BUILTIN_MODES:
        raise ModeError(f"'{mode_id}' is built in and cannot be removed")
    path = custom_modes_dir() / f"{mode_id}.json"
    if not path.is_file():
        return False
    path.unlink()
    return True


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def all_modes() -> dict[str, Mode]:
    modes = dict(BUILTIN_MODES)
    modes.update(load_custom_modes())
    return modes


def list_modes() -> list[Mode]:
    return list(all_modes().values())


def get_mode(mode_id: Optional[str]) -> Mode:
    """The mode for an id; unknown ids fall back to ``grade`` with a warning."""
    key = str(mode_id or DEFAULT_MODE).strip().lower()
    modes = all_modes()
    if key in modes:
        return modes[key]
    log.warning("Unknown skill mode %r — grading with the default mode", key)
    return modes[DEFAULT_MODE]


def is_known_mode(mode_id: Optional[str]) -> bool:
    return str(mode_id or "").strip().lower() in all_modes()


def mode_spec(mode: Mode) -> dict[str, Any]:
    """The portable definition written into a skill bundle for custom modes."""
    return {k: v for k, v in asdict(mode).items() if k != "builtin"}


__all__ = [
    "Mode",
    "ModeError",
    "BUILTIN_MODES",
    "DEFAULT_MODE",
    "GRADE_INSTRUCTIONS",
    "FEEDBACK_INSTRUCTIONS",
    "SELECTIVE_INSTRUCTIONS",
    "custom_modes_dir",
    "validate_mode_spec",
    "load_custom_modes",
    "save_custom_mode",
    "delete_custom_mode",
    "all_modes",
    "list_modes",
    "get_mode",
    "is_known_mode",
    "mode_spec",
]
