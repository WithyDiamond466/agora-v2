"""Tool-calling chat assistant (chat module).

The assistant is a real provider tool-use loop — the model decides which tool to
call and with what arguments; nothing here keyword-matches the user's text.
Tool inputs arrive as parsed JSON objects (``docs/AI_NOTES.md``: "Never
string-match tool inputs; parse JSON").

Tools
-----
``get_course_summary(course_id)``     read-only DB query
``get_student_summary(course_id, student_number)``  read-only DB query (anonymized)
``get_assignment_results(assignment_id)``  read-only DB query
``navigate_to(page, ...)``            returns an ``{"type": "navigate"}`` action
``start_grading(assignment_id)``      returns an ``{"type": "confirm_grading"}`` action

``navigate_to`` and ``start_grading`` perform **no** server-side action. They
return structured action payloads in the API response; the frontend performs the
navigation, and grading only starts after the professor confirms in the UI and
the frontend POSTs to the engine's real grading endpoint.

Privacy: tool results never contain student names. Students are identified by
their per-course number (``Student #7``) exactly like graded submissions are,
so a chat turn can never leak a roster to a provider. The chat panel resolves
numbers back to names locally when it renders.

Provider contract (owned by the engine module, ``app/ai/providers.py``): a
``Provider`` exposes ``chat(messages, tools)`` returning text or tool calls.
Because that module is built in parallel, everything here is tolerant: the call
is adapted to the provider's actual signature and the response is normalized
from any of the plausible shapes (Anthropic-style content blocks, a dict, or an
object with ``.text`` / ``.tool_calls``).
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import analytics as analytics_mod
from app import config, security
from app.models import (
    Assignment,
    Course,
    GradeResult,
    Skill,
    Student,
    Submission,
)

log = logging.getLogger("agora.chat")

#: Hard cap on tool-executing rounds per user message (SPEC: max 6). One extra
#: provider call is allowed after the last round so the model can write its
#: final answer from the tool results.
MAX_TOOL_ITERATIONS = 6

#: Pages the assistant may navigate to. These match the template contract.
PAGES = (
    "home",
    "course_detail",
    "student_detail",
    "grading",
    "skills",
    "skill_detail",
    "settings",
    "analytics",
)

#: Which id each page needs. The frontend ultimately owns routing; `url` below
#: is a convenience hint built from the routes core/skills/engine expose.
PAGE_URLS: dict[str, str] = {
    "home": "/",
    "course_detail": "/courses/{course_id}",
    "student_detail": "/students/{student_id}",
    "grading": "/grading/{assignment_id}",
    "skills": "/skills",
    "skill_detail": "/skills/{skill_id}",
    "settings": "/settings",
    "analytics": "/courses/{course_id}/analytics",
}

PAGE_REQUIRED_ID: dict[str, str] = {
    "course_detail": "course_id",
    "student_detail": "student_id",
    "grading": "assignment_id",
    "skill_detail": "skill_id",
    "analytics": "course_id",
}

#: Chat is short-turn and latency-sensitive; config's registry flags Haiku as
#: the "chat assistant and quick passes" model. Falls back to the provider
#: default when the preferred id is not in the (user-editable) registry.
CHAT_MODEL_PREFERENCE = {"anthropic": "claude-haiku-4-5"}

CHAT_MAX_TOKENS = 1500


class AssistantError(RuntimeError):
    """The provider call failed (network, auth, bad response shape)."""


class AssistantRefusal(Exception):
    """The model declined to answer.

    A refusal is a normal outcome, not a crash (docs/AI_NOTES.md): the chat
    endpoint returns it as an ordinary assistant reply, so it never surfaces as
    an HTTP 502 "AI provider error".
    """

    def __init__(self, message: str, *, category: str = ""):
        super().__init__(message)
        self.category = category


# --------------------------------------------------------------------------
# tool definitions
# --------------------------------------------------------------------------


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_course_summary",
        "description": (
            "Facts about one course: term, roster size, assignments with their "
            "grading progress, average scores, weakest rubric criteria and the "
            "most common misconception tags. Use this before answering anything "
            "about how a class is doing."
        ),
        "input_schema": _obj(
            {"course_id": {"type": "integer", "description": "Course id."}},
            ["course_id"],
        ),
    },
    {
        "name": "get_student_summary",
        "description": (
            "Performance of one student: score per assignment, average, weakest "
            "rubric criteria and recurring misconceptions. Students are "
            "identified by their per-course number, never by name."
        ),
        "input_schema": _obj(
            {
                "course_id": {"type": "integer", "description": "Course id."},
                "student_number": {
                    "type": "integer",
                    "description": "Per-course student number.",
                },
            },
            ["course_id", "student_number"],
        ),
    },
    {
        "name": "get_assignment_results",
        "description": (
            "Grading results for one assignment: submission counts by status, "
            "score distribution, per-criterion averages, misconception tags and "
            "the per-student (numbered) scores."
        ),
        "input_schema": _obj(
            {"assignment_id": {"type": "integer", "description": "Assignment id."}},
            ["assignment_id"],
        ),
    },
    {
        "name": "navigate_to",
        "description": (
            "Ask the app to open a page for the professor. Does not change any "
            "data. Supply the id the page needs: course_detail/analytics need "
            "course_id, student_detail needs student_id, grading needs "
            "assignment_id, skill_detail needs skill_id."
        ),
        "input_schema": _obj(
            {
                "page": {"type": "string", "enum": list(PAGES)},
                "course_id": {"type": "integer"},
                "student_id": {"type": "integer"},
                "assignment_id": {"type": "integer"},
                "skill_id": {"type": "integer"},
            },
            ["page"],
        ),
    },
    {
        "name": "start_grading",
        "description": (
            "Propose grading every ungraded submission for an assignment. This "
            "does NOT start grading: it asks the professor to confirm in the UI "
            "first. Tell them what will run and that they must confirm."
        ),
        "input_schema": _obj(
            {"assignment_id": {"type": "integer", "description": "Assignment id."}},
            ["assignment_id"],
        ),
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOLS)


# --------------------------------------------------------------------------
# normalized provider turn
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ProviderTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Optional[str] = None
    #: Provider-native assistant content, replayed verbatim in the next request.
    native_content: Any = None
    raw: Any = None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict or an attribute off an SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_tool_input(value: Any) -> dict[str, Any]:
    """Tool inputs are JSON. Parse them; never string-match the serialization."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_call_from(obj: Any, index: int) -> Optional[ToolCall]:
    name = _get(obj, "name")
    raw_input = _get(obj, "input", None)
    if raw_input is None:
        raw_input = _get(obj, "arguments", None)
    if raw_input is None:
        raw_input = _get(obj, "args", None)

    # OpenAI-style {"type": "function", "function": {"name", "arguments"}}
    fn = _get(obj, "function")
    if fn is not None and not name:
        name = _get(fn, "name")
        raw_input = _get(fn, "arguments")

    if not name:
        return None
    call_id = _get(obj, "id") or _get(obj, "tool_use_id") or f"call_{index}"
    return ToolCall(id=str(call_id), name=str(name), input=_parse_tool_input(raw_input))


def _blocks_to_turn(
    blocks: Iterable[Any], turn: ProviderTurn, *, collect_tools: bool = True
) -> ProviderTurn:
    texts: list[str] = []
    for i, block in enumerate(blocks):
        if isinstance(block, str):
            texts.append(block)
            continue
        btype = _get(block, "type")
        if btype in ("text", "output_text", None) and _get(block, "text"):
            texts.append(str(_get(block, "text")))
        elif collect_tools and btype in ("tool_use", "tool_call", "function_call", "function"):
            call = _tool_call_from(block, i)
            if call is not None:
                turn.tool_calls.append(call)
    if texts and not turn.text:
        turn.text = "\n".join(t for t in texts if t).strip()
    return turn


def _text_attr(raw: Any) -> str:
    for key in ("text", "reply", "message", "output_text"):
        value = _get(raw, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_turn(raw: Any) -> ProviderTurn:
    """Normalize whatever `Provider.chat` returned into a `ProviderTurn`.

    The engine returns a ``ChatTurn`` (``text`` / ``tool_calls`` with
    ``arguments`` / ``stop_reason`` / native ``content``); dicts, plain strings
    and raw Anthropic content blocks are accepted too.
    """
    turn = ProviderTurn(raw=raw)
    if raw is None:
        return turn
    if isinstance(raw, str):
        turn.text = raw
        return turn
    if isinstance(raw, (list, tuple)):
        return _blocks_to_turn(raw, turn)

    turn.stop_reason = _get(raw, "stop_reason") or _get(raw, "finish_reason")
    turn.native_content = _get(raw, "content")

    calls = _get(raw, "tool_calls")
    if calls is None:
        calls = _get(raw, "tool_uses")
    explicit = isinstance(calls, (list, tuple))
    if explicit:
        for i, item in enumerate(calls):
            call = _tool_call_from(item, i)
            if call is not None:
                turn.tool_calls.append(call)

    turn.text = _text_attr(raw)

    content = turn.native_content
    if isinstance(content, (list, tuple)):
        # With an explicit tool-call list, blocks are only a source of text.
        _blocks_to_turn(content, turn, collect_tools=not explicit)
    elif isinstance(content, str) and content.strip() and not turn.text:
        turn.text = content.strip()

    return turn


def _assistant_message(turn: ProviderTurn) -> dict[str, Any]:
    """The assistant turn, replayed for the next request.

    Provider-native content is preferred (the engine's ``ChatTurn.content``
    exists precisely so a tool-use loop can replay it verbatim); otherwise the
    turn is rebuilt as text + ``tool_use`` blocks.
    """
    if isinstance(turn.native_content, (list, tuple)) and turn.native_content:
        return {"role": "assistant", "content": list(turn.native_content)}

    blocks: list[dict[str, Any]] = []
    if turn.text:
        blocks.append({"type": "text", "text": turn.text})
    for call in turn.tool_calls:
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
        )
    return {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}


def _tool_results_message(results: list[dict[str, Any]]) -> dict[str, Any]:
    """One message carrying every tool result of a round (engine contract)."""
    mod = _providers_module()
    builder = getattr(mod, "tool_result_message", None) if mod is not None else None
    if callable(builder):
        try:
            return builder(results)
        except Exception:  # noqa: BLE001 - fall back to the documented shape
            pass
    return {"role": "tool_results", "results": results}


# --------------------------------------------------------------------------
# provider plumbing
# --------------------------------------------------------------------------


def _providers_module():
    try:
        return importlib.import_module("app.ai.providers")
    except Exception as exc:  # noqa: BLE001 - engine module may be mid-build
        log.warning("app.ai.providers unavailable (%s) — using built-in mock", exc)
        return None


def _accepted_kwargs(fn: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
    """Filter `candidates` down to parameters `fn` actually accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(candidates)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in params}


class _FallbackMockProvider:
    """Stand-in used only when `app/ai/providers.py` is missing or unusable.

    Keeps the app (and this module's tests) working if the engine module is
    absent. Deterministic, no network, no tool calls. Signature matches the
    engine's ``Provider.chat(system_prompt, messages, tools)``.
    """

    name = config.MOCK_PROVIDER
    model = config.MOCK_MODEL

    def chat(
        self,
        system_prompt: str = "",
        messages: Optional[list[dict[str, Any]]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        last = ""
        for msg in reversed(messages or []):
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str) and content.strip():
                last = content.strip()
                break
        text = "The AI provider is not available, so I cannot answer that yet."
        if last:
            text += f" (You asked: {last[:200]})"
        return {"text": text, "tool_calls": [], "stop_reason": "end_turn"}


def _mock_provider(mod: Any) -> Any:
    """The engine's MockProvider, or the local stand-in."""
    if mod is not None:
        cls = getattr(mod, "MockProvider", None)
        if cls is not None:
            for kwargs in ({"model": config.MOCK_MODEL}, {}):
                try:
                    obj = cls(**kwargs)
                except Exception:  # noqa: BLE001 - try a simpler constructor
                    continue
                if callable(getattr(obj, "chat", None)):
                    return obj
    return _FallbackMockProvider()


def _live_provider(mod: Any, provider_name: str, model: str, db: Session) -> Any:
    """Build a keyed provider through the engine factory, or None."""
    if mod is None:
        return None
    settings = {
        "provider": provider_name,
        "model": model,
        "max_tokens": CHAT_MAX_TOKENS,
    }
    factory = getattr(mod, "get_provider", None)
    if callable(factory):
        try:
            obj = factory(settings, db=db)
        except TypeError:
            try:
                obj = factory(settings)
            except Exception:  # noqa: BLE001
                return None
        except Exception as exc:  # noqa: BLE001 - no key / bad config
            log.info("Live chat provider unavailable (%s) — using mock", exc)
            return None
        if obj is not None and callable(getattr(obj, "chat", None)):
            return obj

    classes = getattr(mod, "PROVIDER_CLASSES", None) or {}
    cls = classes.get(provider_name)
    if cls is None:
        return None
    try:
        api_key = security.get_api_key(db, provider_name)
    except Exception:  # noqa: BLE001
        return None
    if not api_key:
        return None
    try:
        obj = cls(model=model, api_key=api_key, max_tokens=CHAT_MAX_TOKENS)
    except Exception:  # noqa: BLE001
        return None
    return obj if callable(getattr(obj, "chat", None)) else None


def configured_provider_name() -> str:
    """Provider the app would use for chat, honouring the engine's env override."""
    mod = _providers_module()
    env_name = getattr(mod, "FORCE_PROVIDER_ENV", "AGORA_AI_PROVIDER") if mod else None
    forced = os.environ.get(env_name or "AGORA_AI_PROVIDER", "").strip().lower()
    # No fixed default: the professor's preferred provider (Settings) wins,
    # then the app-wide fallback. A missing key still degrades to the mock.
    return forced or config.preferred_provider() or config.DEFAULT_PROVIDER


def chat_model_for(provider_name: str) -> str:
    """Preferred chat model for a provider, honouring the config registry."""
    if provider_name == config.MOCK_PROVIDER:
        return config.MOCK_MODEL
    preferred = CHAT_MODEL_PREFERENCE.get(provider_name)
    if preferred and config.is_known_model(provider_name, preferred):
        return preferred
    return config.default_model_for(provider_name)


def build_provider(
    db: Session, *, provider_name: Optional[str] = None, model: Optional[str] = None
) -> tuple[Any, str, str]:
    """Resolve the provider for chat: configured provider, else MockProvider.

    Returns ``(provider, provider_name, model)``.
    """
    name = provider_name or config.DEFAULT_PROVIDER
    chosen_model = model or chat_model_for(name)
    mod = _providers_module()

    if name != config.MOCK_PROVIDER:
        live = _live_provider(mod, name, chosen_model, db)
        if live is not None:
            return live, name, getattr(live, "model", chosen_model)

    provider = _mock_provider(mod)
    return provider, config.MOCK_PROVIDER, getattr(provider, "model", config.MOCK_MODEL)


def _binds(fn: Callable[..., Any], args: tuple, kwargs: dict) -> bool:
    """Would ``fn(*args, **kwargs)`` be a legal call for this signature?"""
    try:
        inspect.signature(fn).bind(*args, **kwargs)
    except TypeError:
        return False
    except (ValueError, AttributeError):
        # Signature-less callable (a C builtin, an odd mock): just try it.
        return True
    return True


def _as_refusal(exc: Exception) -> Optional[AssistantRefusal]:
    """Translate the engine's ``ProviderRefusalError`` into a normal outcome."""
    mod = _providers_module()
    refusal_cls = getattr(mod, "ProviderRefusalError", None) if mod is not None else None
    if refusal_cls is not None and isinstance(exc, refusal_cls):
        return AssistantRefusal(str(exc), category=getattr(exc, "category", "") or "")
    return None


def _positional_style(fn: Callable[..., Any]) -> str:
    """Is this ``chat(system_prompt, messages, ...)`` or ``chat(messages, ...)``?"""
    try:
        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return "system_first"
    if params and params[0].name in ("system_prompt", "system"):
        return "system_first"
    return "messages_first"


def call_provider_chat(
    provider: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    system: str,
    model: Optional[str] = None,
    max_tokens: int = CHAT_MAX_TOKENS,
) -> Any:
    """Call the provider's ``chat``, adapting to its real signature.

    Engine contract (``app/ai/providers.py``):
    ``chat(system_prompt, messages, tools=None) -> ChatTurn``. A provider that
    instead takes ``chat(messages, tools=...)`` is also supported.
    """
    fn = getattr(provider, "chat", None)
    if not callable(fn):
        raise AssistantError("Configured provider does not implement chat()")

    if _positional_style(fn) == "system_first":
        attempts: list[tuple[tuple, dict]] = [
            ((system, messages), {"tools": tools}),
            ((system, messages), {}),
        ]
    else:
        extras = {
            "tools": tools,
            "system": system,
            "system_prompt": system,
            "model": model,
            "max_tokens": max_tokens,
        }
        kwargs = _accepted_kwargs(fn, extras)
        if "system" in kwargs and "system_prompt" in kwargs:
            kwargs.pop("system_prompt")
        attempts = [((messages,), kwargs), ((messages,), {"tools": tools}), ((messages,), {})]

    # The signature is probed, never discovered by catching TypeError from the
    # call: a TypeError raised *inside* the SDK would otherwise silently retry
    # a tool-less request and answer with no tool ever running.
    chosen: Optional[tuple[tuple, dict]] = None
    bind_error: Optional[Exception] = None
    for args, kwargs in attempts:
        if _binds(fn, args, kwargs):
            chosen = (args, kwargs)
            break
    if chosen is None:
        try:
            inspect.signature(fn).bind(*attempts[0][0], **attempts[0][1])
        except TypeError as exc:
            bind_error = exc
        raise AssistantError(f"Provider.chat signature not understood: {bind_error}")

    args, kwargs = chosen
    if tools and not kwargs.get("tools"):
        log.warning(
            "Provider.chat(%s) does not accept `tools` — the assistant is answering "
            "without tool calling.",
            type(provider).__name__,
        )
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - provider/network failure
        refusal = _as_refusal(exc)
        if refusal is not None:
            raise refusal from exc
        raise AssistantError(str(exc) or exc.__class__.__name__) from exc


# --------------------------------------------------------------------------
# tool handlers (read-only DB queries)
# --------------------------------------------------------------------------


class ToolError(Exception):
    """Returned to the model as an `is_error` tool_result."""


def _round(value: Optional[float], places: int = 1) -> Optional[float]:
    return None if value is None else round(float(value), places)


def _course_or_error(db: Session, course_id: Any) -> Course:
    course = db.get(Course, _as_int(course_id, "course_id"))
    if course is None:
        raise ToolError(f"No course with id {course_id}. Call it with a valid course id.")
    return course


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolError(f"{field_name} must be an integer, got {value!r}") from None


def _assignment_progress(
    db: Session,
    assignment: Assignment,
    counts: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    if counts is None:
        rows = db.execute(
            select(Submission.status, func.count(Submission.id))
            .where(Submission.assignment_id == assignment.id)
            .group_by(Submission.status)
        ).all()
        counts = {status: count for status, count in rows}
    by_status = counts
    total = sum(by_status.values())
    return {
        "assignment_id": assignment.id,
        "name": assignment.name,
        "submissions": total,
        "graded": by_status.get("graded", 0),
        "pending": by_status.get("pending", 0),
        "grading": by_status.get("grading", 0),
        "failed": by_status.get("failed", 0),
        "ungraded": total - by_status.get("graded", 0),
        "has_rubric": assignment.rubric_id is not None,
        "has_skill": assignment.skill_id is not None,
    }


def tool_get_course_summary(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    course = _course_or_error(db, args.get("course_id"))
    assignments = db.scalars(
        select(Assignment).where(Assignment.course_id == course.id).order_by(Assignment.id)
    ).all()
    counts_by_assignment: dict[int, dict[str, int]] = {
        assignment.id: {} for assignment in assignments
    }
    status_rows = db.execute(
        select(Submission.assignment_id, Submission.status, func.count(Submission.id))
        .join(Assignment, Assignment.id == Submission.assignment_id)
        .where(Assignment.course_id == course.id)
        .group_by(Submission.assignment_id, Submission.status)
    ).all()
    for assignment_id, status, count in status_rows:
        counts_by_assignment[assignment_id][status] = count
    students = db.scalars(
        select(Student).where(Student.course_id == course.id).order_by(Student.student_number)
    ).all()

    payload: dict[str, Any] = {
        "course_id": course.id,
        "name": course.name,
        "term": course.term,
        "student_count": len(students),
        # Anonymized roster: ids + numbers only, so names never reach a provider.
        "roster": [
            {"student_id": s.id, "student_number": s.student_number, "label": s.anon_label}
            for s in students
        ],
        "assignments": [
            _assignment_progress(db, a, counts_by_assignment[a.id]) for a in assignments
        ],
    }

    # One load of the course's graded results feeds all three roll-ups.
    graded_rows = analytics_mod.graded_rows(db, course.id)
    students_by_id = {student.id: student for student in students}
    percentages_by_student: dict[int, list[float]] = {}
    for row in graded_rows:
        student_id = row.submission.student_id
        if student_id not in students_by_id or not row.max_score or row.percentage is None:
            continue
        percentages_by_student.setdefault(student_id, []).append(row.percentage)
    attention = []
    for student_id, percentages in percentages_by_student.items():
        student = students_by_id[student_id]
        attention.append(
            {
                "student_number": student.student_number,
                "label": student.anon_label,
                "average_percent": round(sum(percentages) / len(percentages), 1),
                "graded_count": len(percentages),
            }
        )
    attention.sort(key=lambda row: (row["average_percent"], row["student_number"]))
    payload["students_needing_attention"] = attention[:5]
    try:
        overview = analytics_mod.course_overview(db, course.id, graded_rows)
    except LookupError:
        overview = None
    if overview:
        payload["totals"] = overview.get("totals")
        payload["assignment_scores"] = [
            {
                "assignment_id": block.get("assignment_id"),
                "name": block.get("name"),
                "average_percent": block.get("average_percent"),
                "low_percent": block.get("low_percent"),
                "high_percent": block.get("high_percent"),
                "graded_count": block.get("graded_count"),
            }
            for block in overview.get("assignments", [])
        ]
    try:
        crits = analytics_mod.criteria_breakdown(db, course.id, graded_rows)
        payload["weakest_criteria"] = [
            {
                "key": c.get("key"),
                "title": c.get("title"),
                "average_percent": c.get("average_percent"),
            }
            for c in (crits.get("criteria") or [])[:3]
        ]
    except LookupError:
        pass
    try:
        misc = analytics_mod.misconception_counts(db, course.id, limit=5, rows=graded_rows)
        payload["top_misconceptions"] = misc.get("misconceptions")
    except (LookupError, TypeError):
        pass
    return payload


def tool_get_student_summary(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    course_id = _as_int(args.get("course_id"), "course_id")
    student_number = _as_int(args.get("student_number"), "student_number")
    student = db.scalar(
        select(Student).where(
            Student.course_id == course_id,
            Student.student_number == student_number,
        )
    )
    if student is None:
        raise ToolError(f"No Student #{student_number} in course {course_id}.")
    course = db.get(Course, student.course_id)

    payload: dict[str, Any] = {
        "student_id": student.id,
        # Privacy: the number is the only identifier that may leave the machine.
        "label": student.anon_label,
        "student_number": student.student_number,
        "course_id": student.course_id,
        "course_name": course.name if course else None,
    }
    try:
        timeline = analytics_mod.student_timeline(db, student.id)
    except LookupError:
        timeline = None
    if timeline:
        payload.update(
            {
                "graded_count": timeline.get("graded_count"),
                "average_percent": timeline.get("average_percent"),
                "best_percent": timeline.get("best_percent"),
                "worst_percent": timeline.get("worst_percent"),
                "timeline": timeline.get("timeline"),
                "criteria": timeline.get("criteria"),
                "misconceptions": timeline.get("misconceptions"),
            }
        )
    else:
        payload["graded_count"] = 0
        payload["note"] = "No graded work for this student yet."

    # --- Increment 1 · insight module ------------------------------------
    # The maintained card is the headline answer when one exists; the raw
    # aggregate above stays in the payload as the fallback (and as the numbers
    # the card's sentences are grounded in). Still anonymized: cards are built
    # from rubric scores and tags, never from names.
    try:
        from app.insight import card_for_tool  # local import: cycle-safe

        card = card_for_tool(db, student.id)
    except Exception as exc:  # noqa: BLE001 - a missing card must not fail the tool
        log.debug("Insight card unavailable for student %s: %s", student.id, exc)
        card = None
    if card:
        payload["card"] = card
    # --- end insight module ----------------------------------------------

    return payload


def tool_get_assignment_results(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    assignment_id = _as_int(args.get("assignment_id"), "assignment_id")
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise ToolError(f"No assignment with id {assignment_id}.")
    course = db.get(Course, assignment.course_id)
    skill = db.get(Skill, assignment.skill_id) if assignment.skill_id else None

    rows = db.execute(
        select(Submission, GradeResult, Student)
        .join(GradeResult, GradeResult.submission_id == Submission.id, isouter=True)
        .join(Student, Student.id == Submission.student_id, isouter=True)
        .where(Submission.assignment_id == assignment.id)
        .order_by(Submission.id)
    ).all()

    results = []
    percents: list[float] = []
    criteria_totals: dict[str, list[float]] = {}
    tags: dict[str, int] = {}
    for submission, grade, student in rows:
        entry: dict[str, Any] = {
            "submission_id": submission.id,
            "student_id": student.id if student else None,
            "label": student.anon_label if student else "Unassigned submission",
            "status": submission.status,
        }
        if grade is not None:
            pct = None
            if grade.max_score:
                pct = round(100.0 * float(grade.overall_score) / float(grade.max_score), 1)
                percents.append(pct)
            entry.update(
                {
                    "score": grade.overall_score,
                    "max_score": grade.max_score,
                    "percent": pct,
                    "misconceptions": grade.misconceptions or [],
                }
            )
            for crit in grade.criteria or []:
                key = str(crit.get("key", ""))
                max_points = float(crit.get("max_points") or 0)
                if key and max_points:
                    criteria_totals.setdefault(key, []).append(
                        100.0 * float(crit.get("score") or 0) / max_points
                    )
            for tag in grade.misconceptions or []:
                normalized = analytics_mod.normalize_tag(tag)
                if normalized:
                    tags[normalized] = tags.get(normalized, 0) + 1
        results.append(entry)

    progress = _assignment_progress(db, assignment)
    return {
        "assignment_id": assignment.id,
        "name": assignment.name,
        "course_id": assignment.course_id,
        "course_name": course.name if course else None,
        "skill": skill.name if skill else None,
        "progress": progress,
        "average_percent": _round(sum(percents) / len(percents)) if percents else None,
        "low_percent": min(percents) if percents else None,
        "high_percent": max(percents) if percents else None,
        "criteria_averages": [
            {"key": key, "average_percent": _round(sum(vals) / len(vals))}
            for key, vals in sorted(criteria_totals.items())
        ],
        "misconceptions": [
            {"tag": tag, "count": count}
            for tag, count in sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "results": results,
    }


def _page_url(page: str, ids: dict[str, Any]) -> Optional[str]:
    template = PAGE_URLS.get(page)
    if template is None:
        return None
    try:
        return template.format(**{k: v for k, v in ids.items() if v is not None})
    except (KeyError, IndexError):
        return None


def tool_navigate_to(db: Session, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (tool_result_payload, action). Nothing happens server-side."""
    page = args.get("page")
    if page not in PAGES:
        raise ToolError(f"Unknown page {page!r}. Valid pages: {', '.join(PAGES)}")

    ids = {
        key: (_as_int(args[key], key) if args.get(key) is not None else None)
        for key in ("course_id", "student_id", "assignment_id", "skill_id")
    }

    required = PAGE_REQUIRED_ID.get(page)
    if required and ids.get(required) is None:
        raise ToolError(f"Page {page!r} needs {required}. Look it up first, then call again.")

    # Validate the target exists so the UI never navigates into a 404.
    checks = {
        "course_id": Course,
        "student_id": Student,
        "assignment_id": Assignment,
        "skill_id": Skill,
    }
    for key, model in checks.items():
        value = ids.get(key)
        if value is not None and db.get(model, value) is None:
            raise ToolError(f"No {model.__name__.lower()} with id {value}.")

    action = {
        "type": "navigate",
        "page": page,
        "url": _page_url(page, ids),
        **{k: v for k, v in ids.items() if v is not None},
    }
    payload = {
        "ok": True,
        "navigating_to": page,
        "url": action["url"],
        "note": "The app is opening this page for the professor.",
    }
    return payload, action


def tool_start_grading(db: Session, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (tool_result_payload, action). Grading is NOT started here."""
    assignment_id = _as_int(args.get("assignment_id"), "assignment_id")
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise ToolError(f"No assignment with id {assignment_id}.")
    course = db.get(Course, assignment.course_id)
    progress = _assignment_progress(db, assignment)

    if progress["submissions"] == 0:
        raise ToolError(
            f"Assignment {assignment.name!r} has no uploaded submissions yet, so there "
            "is nothing to grade. Tell the professor to upload submissions first."
        )

    gradable_count = db.scalar(
        select(func.count(Submission.id)).where(
            Submission.assignment_id == assignment.id,
            Submission.student_id.is_not(None),
            Submission.status.in_(("pending", "failed")),
        )
    ) or 0
    if gradable_count == 0:
        raise ToolError(
            f"Assignment {assignment.name!r} has no mapped pending or failed "
            "submissions to grade."
        )

    action = {
        "type": "confirm_grading",
        "assignment_id": assignment.id,
        "assignment_name": assignment.name,
        "course_id": assignment.course_id,
        "course_name": course.name if course else None,
        "submission_count": progress["submissions"],
        "ungraded_count": gradable_count,
        "has_rubric": progress["has_rubric"],
        "has_skill": progress["has_skill"],
        "message": (
            f"Grade {gradable_count} ungraded submission(s) for {assignment.name}?"
        ),
        # The UI must confirm before POSTing to the engine's grading endpoint.
        "requires_confirmation": True,
    }
    payload = {
        "ok": True,
        "status": "awaiting_user_confirmation",
        "assignment": assignment.name,
        "ungraded_count": gradable_count,
        "note": (
            "Grading has NOT started. A confirmation prompt is now shown to the "
            "professor; they must approve it before anything is graded."
        ),
    }
    if not progress["has_rubric"]:
        payload["warning"] = "This assignment has no rubric attached."
    return payload, action


#: name -> handler. Handlers returning a tuple also emit a frontend action.
TOOL_HANDLERS: dict[str, Callable[[Session, dict[str, Any]], Any]] = {
    "get_course_summary": tool_get_course_summary,
    "get_student_summary": tool_get_student_summary,
    "get_assignment_results": tool_get_assignment_results,
    "navigate_to": tool_navigate_to,
    "start_grading": tool_start_grading,
}


def dispatch_tool(db: Session, call: ToolCall) -> tuple[dict[str, Any], bool, Optional[dict]]:
    """Run one tool call. Returns (payload, is_error, action)."""
    handler = TOOL_HANDLERS.get(call.name)
    if handler is None:
        return ({"error": f"Unknown tool {call.name!r}."}, True, None)
    try:
        outcome = handler(db, call.input or {})
    except ToolError as exc:
        return ({"error": str(exc)}, True, None)
    except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the chat
        log.exception("Tool %s failed", call.name)
        return ({"error": f"{call.name} failed: {exc}"}, True, None)

    if isinstance(outcome, tuple):
        payload, action = outcome
        return payload, False, action
    return outcome, False, None


# --------------------------------------------------------------------------
# system prompt
# --------------------------------------------------------------------------


def build_system_prompt(db: Session, context: Optional[dict[str, Any]] = None) -> str:
    context = context or {}
    page = context.get("page") or "unknown"
    course_id = context.get("course_id")
    skill_id = context.get("skill_id")

    lines = [
        "You are the Agora assistant, embedded in a local desktop app a professor "
        "uses to run their courses and grade student work with AI.",
        "",
        "Rules:",
        "- Answer from tool results, never from guesses. If you need a number, call a tool.",
        "- Students are identified by their per-course number (e.g. 'Student #7'). The "
        "roster of names stays on the professor's machine and is never sent to you, so "
        "never guess a name; if the professor names a student, ask for the number or "
        "have them open that student's page.",
        "- Use navigate_to when the professor asks to see or open something.",
        "- Use start_grading only to *propose* a grading run: it asks the professor to "
        "confirm in the UI, and nothing is graded until they do. Say so.",
        "- Be brief and concrete. Percentages to one decimal. No filler.",
        "",
        f"Current page: {page}",
    ]
    if course_id is not None:
        course = db.get(Course, course_id) if isinstance(course_id, int) else None
        if course is not None:
            lines.append(
                f"Course in context: id={course.id}, name={course.name!r}, "
                f"term={course.term!r}"
            )
            assignments = db.scalars(
                select(Assignment)
                .where(Assignment.course_id == course.id)
                .order_by(Assignment.id)
            ).all()
            if assignments:
                listed = ", ".join(f"id={a.id} {a.name!r}" for a in assignments[:15])
                lines.append(f"Assignments in this course: {listed}")
        else:
            lines.append(f"Course in context: id={course_id} (not found in the database)")
    if skill_id is not None:
        skill = db.get(Skill, skill_id) if isinstance(skill_id, int) else None
        if skill is not None:
            lines.append(f"Skill in context: id={skill.id}, name={skill.name!r}")
        else:
            lines.append(f"Skill in context: id={skill_id} (not found in the database)")
    if context.get("student_id") is not None:
        student_id = context["student_id"]
        student = db.get(Student, student_id) if isinstance(student_id, int) else None
        if student is not None:
            lines.append(
                f"Student in context: course_id={student.course_id}, "
                f"student_number={student.student_number}"
            )
        else:
            lines.append(f"Student in context: id={student_id} (not found in the database)")
    if context.get("assignment_id") is not None:
        lines.append(f"Assignment in context: id={context['assignment_id']}")

    courses = db.scalars(select(Course).order_by(Course.id)).all()
    if courses:
        for course in courses:
            lines.append(f"Known course: id={course.id}, name={json.dumps(course.name)}")
    else:
        lines.append("There are no courses in the database yet.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


@dataclass
class AssistantResult:
    reply: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    truncated: bool = False
    #: The model declined to answer — a normal reply, not an error.
    refused: bool = False
    provider: str = config.MOCK_PROVIDER
    model: str = config.MOCK_MODEL


def _refusal_reply(refusal: AssistantRefusal) -> str:
    reason = str(refusal).strip() or "The model declined to answer that."
    text = f"The AI provider declined to answer that request. {reason}"
    if refusal.category:
        text += f" (category: {refusal.category})"
    return text + " Try rephrasing, or ask about the course data directly."


def run_assistant(
    db: Session,
    message: str,
    *,
    provider: Any,
    context: Optional[dict[str, Any]] = None,
    history: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    max_tokens: int = CHAT_MAX_TOKENS,
    outbound_text_guard: Optional[Callable[[str], str]] = None,
    history_is_guarded: bool = False,
) -> AssistantResult:
    """Run the tool-calling loop for one user message.

    At most `max_iterations` tool rounds are executed; one further provider call
    is made afterwards so the model can answer from the results.
    """
    system = build_system_prompt(db, context)
    if outbound_text_guard is not None:
        system = outbound_text_guard(system)
    messages: list[dict[str, Any]] = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            if outbound_text_guard is not None and not history_is_guarded:
                content = outbound_text_guard(content)
            messages.append({"role": role, "content": content})
    outbound_message = outbound_text_guard(message) if outbound_text_guard is not None else message
    messages.append({"role": "user", "content": outbound_message})

    result = AssistantResult(reply="")
    texts: list[str] = []

    for round_index in range(max_iterations + 1):
        try:
            raw = call_provider_chat(
                provider,
                messages,
                tools=TOOLS,
                system=system,
                model=model,
                max_tokens=max_tokens,
            )
        except AssistantRefusal as refusal:
            # A refusal is surfaced as a normal reply, never as an error page.
            result.refused = True
            result.reply = _refusal_reply(refusal)
            return result
        turn = normalize_turn(raw)
        if turn.text:
            texts.append(turn.text)

        if not turn.tool_calls:
            break

        if round_index >= max_iterations:
            result.truncated = True
            log.warning("Chat hit the %d tool-iteration cap", max_iterations)
            break

        result.iterations = round_index + 1
        messages.append(_assistant_message(turn))

        tool_results = []
        for call in turn.tool_calls:
            payload, is_error, action = dispatch_tool(db, call)
            if action is not None:
                result.actions.append(action)
            result.tool_calls.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "input": call.input,
                    "ok": not is_error,
                    "result": payload,
                }
            )
            serialized = json.dumps(payload, default=str)
            if outbound_text_guard is not None:
                serialized = outbound_text_guard(serialized)
            tool_results.append(
                {
                    "tool_use_id": call.id,
                    "content": serialized,
                    "is_error": is_error,
                }
            )
        # All results of a round go back in ONE message (AI_NOTES).
        messages.append(_tool_results_message(tool_results))

    result.reply = "\n\n".join(t for t in texts if t).strip()
    if not result.reply:
        result.reply = (
            "I gathered the data but ran out of tool steps before answering. "
            "Try asking something narrower."
            if result.truncated
            else "I do not have an answer for that."
        )
    return result


__all__ = [
    "MAX_TOOL_ITERATIONS",
    "PAGES",
    "PAGE_URLS",
    "TOOLS",
    "TOOL_NAMES",
    "TOOL_HANDLERS",
    "AssistantError",
    "AssistantRefusal",
    "AssistantResult",
    "ProviderTurn",
    "ToolCall",
    "ToolError",
    "build_provider",
    "build_system_prompt",
    "call_provider_chat",
    "chat_model_for",
    "dispatch_tool",
    "normalize_turn",
    "run_assistant",
]
