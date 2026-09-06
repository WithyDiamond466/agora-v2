"""Chat API: the floating assistant panel on every page.

``POST /api/chat`` runs one turn of the tool-calling loop in
``app/ai/assistant.py`` and returns ``{reply, actions, session_id}``. Actions are
instructions *for the frontend* — the server never navigates and never starts a
grading run on its own; ``confirm_grading`` must be approved by the professor in
the UI before the frontend POSTs to the engine's grading endpoint.

Sessions and messages persist to ``ChatSession`` / ``ChatMessage`` so the panel
can be reopened with its history intact.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config, security
from app.ai import assistant as assistant_mod
from app.ai import privacy as privacy_guard
from app.db import get_db
from app.models import Assignment, ChatMessage, ChatSession, Course, Skill, Student, utcnow

log = logging.getLogger("agora.chat")

router = APIRouter()

#: How many prior turns to replay to the provider.
HISTORY_TURNS = 12


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class ChatContext(BaseModel):
    page: Optional[str] = None
    course_id: Optional[int] = None
    skill_id: Optional[int] = None
    student_id: Optional[int] = None
    assignment_id: Optional[int] = None


class ChatIn(BaseModel):
    message: str = Field(min_length=1)
    session_id: Optional[int] = None
    context: ChatContext = Field(default_factory=ChatContext)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _title_from(message: str) -> str:
    text = " ".join(message.split())
    return text[:60] + ("…" if len(text) > 60 else "")


def _session_dict(session: ChatSession) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title,
        "course_id": session.course_id,
        "skill_id": session.skill_id,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


def _message_dict(msg: ChatMessage) -> dict[str, Any]:
    records = [r for r in (msg.tool_calls or []) if isinstance(r, dict)]
    payload = {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "tool_calls": [r for r in records if r.get("kind") == "tool_call"],
        "actions": [r["action"] for r in records if r.get("kind") == "action"],
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }
    if msg.role == "assistant":
        payload["provider"] = next(
            (r.get("provider") for r in records if r.get("kind") == "provider"),
            None,
        )
    return payload


def _get_session(db: Session, session_id: int) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found")
    return session


def _resolve_session(
    db: Session, payload: ChatIn
) -> tuple[Optional[ChatSession], Optional[int], Optional[int]]:
    context = payload.context
    course_id = context.course_id
    if course_id is not None and db.get(Course, course_id) is None:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    skill_id = context.skill_id
    if skill_id is not None and db.get(Skill, skill_id) is None:
        raise HTTPException(status_code=404, detail=f"Skill {skill_id} not found")
    student = None
    if context.student_id is not None:
        student = db.get(Student, context.student_id)
        if student is None:
            raise HTTPException(
                status_code=404, detail=f"Student {context.student_id} not found"
            )
    assignment = None
    if context.assignment_id is not None:
        assignment = db.get(Assignment, context.assignment_id)
        if assignment is None:
            raise HTTPException(
                status_code=404,
                detail=f"Assignment {context.assignment_id} not found",
            )

    if payload.session_id is not None:
        session = _get_session(db, payload.session_id)
        resolved_course_id = course_id if course_id is not None else session.course_id
        resolved_skill_id = skill_id if skill_id is not None else session.skill_id
    else:
        session = None
        resolved_course_id = course_id
        resolved_skill_id = skill_id

    if resolved_course_id is not None:
        if student is not None and student.course_id != resolved_course_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Student {student.id} does not belong to course "
                    f"{resolved_course_id}"
                ),
            )
        if assignment is not None and assignment.course_id != resolved_course_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Assignment {assignment.id} does not belong to course "
                    f"{resolved_course_id}"
                ),
            )

    return session, resolved_course_id, resolved_skill_id


def _history_for(db: Session, session: ChatSession) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id, ChatMessage.role.in_(("user", "assistant")))
        .order_by(ChatMessage.id.desc())
        .limit(HISTORY_TURNS * 2)
    ).all()
    history = [
        {"role": m.role, "content": m.content, "course_id": m.course_id}
        for m in reversed(rows)
        if (m.content or "").strip()
    ]
    # The window is taken from the tail, so it can start mid-exchange (or on an
    # assistant turn whose user message was lost when a call failed). The
    # Anthropic Messages API wants the first message to be a user turn and the
    # roles to alternate, so normalise: start on a user turn, keep strict
    # alternation, and end on an assistant turn (the caller appends the new
    # user message).
    alternating: list[dict[str, Any]] = []
    expected = "user"
    for item in history:
        if item["role"] != expected:
            continue
        alternating.append(item)
        expected = "assistant" if expected == "user" else "user"
    if alternating and alternating[-1]["role"] == "user":
        alternating.pop()
    return alternating


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@router.post("/api/chat")
def chat(payload: ChatIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message was empty")

    session, resolved_course_id, resolved_skill_id = _resolve_session(db, payload)
    history = _history_for(db, session) if session is not None else []

    context = payload.context.model_dump()
    context["course_id"] = resolved_course_id
    context["skill_id"] = resolved_skill_id

    effective_course_id = resolved_course_id

    provider, provider_name, model = assistant_mod.build_provider(db)

    outbound_text_guard = None
    history_is_guarded = False
    if config.is_cloud_provider(provider_name):
        privacy_settings = config.load_privacy_settings()
        if privacy_settings["mode"] == config.PRIVACY_MODE_SWAP:
            course_ids = db.scalars(select(Course.id).order_by(Course.id)).all()
            roster_students = db.scalars(
                select(Student).order_by(Student.course_id, Student.student_number)
            ).all()
            course_id_set = set(course_ids)
            if effective_course_id not in course_id_set:
                raise HTTPException(
                    status_code=409,
                    detail="The Privacy Guard could not pseudonymize this chat without a course, so it was not sent to the provider.",
                )

            def guard_text(
                text: str,
                *,
                subject: str = "chat text",
                course_id: int = effective_course_id,
            ) -> str:
                try:
                    cleaned, _ = privacy_guard.protect_cloud_text(
                        db,
                        text,
                        course_id=course_id,
                        settings=privacy_settings,
                        subject=subject,
                        students=roster_students,
                    )
                except Exception as exc:  # noqa: BLE001 - swap mode must fail closed
                    raise HTTPException(
                        status_code=409,
                        detail=f"The Privacy Guard failed, so this chat was not sent to the provider: {exc}",
                    ) from exc
                return cleaned

            for item in history:
                history_course_id = item.get("course_id")
                if history_course_id not in course_id_set:
                    raise HTTPException(
                        status_code=409,
                        detail="The Privacy Guard could not pseudonymize unscoped chat history, so this chat was not sent to the provider.",
                    )
                item["content"] = guard_text(
                    item["content"],
                    subject="chat history",
                    course_id=history_course_id,
                )

            outbound_text_guard = guard_text
            history_is_guarded = True

    try:
        result = assistant_mod.run_assistant(
            db,
            message,
            provider=provider,
            context=context,
            history=history,
            model=model,
            outbound_text_guard=outbound_text_guard,
            history_is_guarded=history_is_guarded,
        )
    except assistant_mod.AssistantError as exc:
        log.warning("Assistant call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc

    records: list[dict[str, Any]] = [
        {"kind": "tool_call", "name": call["name"], "input": call["input"], "ok": call["ok"]}
        for call in result.tool_calls
    ]
    records += [{"kind": "action", "action": action} for action in result.actions]
    records.append({"kind": "provider", "provider": provider_name})

    if session is None:
        session = ChatSession(
            title=_title_from(message),
            course_id=resolved_course_id,
            skill_id=resolved_skill_id,
        )
        db.add(session)
        db.flush()
    else:
        session.course_id = resolved_course_id
        session.skill_id = resolved_skill_id

    db.add_all(
        [
            ChatMessage(
                session_id=session.id,
                course_id=effective_course_id,
                role="user",
                content=message,
            ),
            ChatMessage(
                session_id=session.id,
                course_id=effective_course_id,
                role="assistant",
                content=result.reply,
                tool_calls=records or None,
            ),
        ]
    )
    session.updated_at = utcnow()
    db.add(session)
    db.commit()

    return {
        "reply": result.reply,
        "actions": result.actions,
        "session_id": session.id,
        "tool_calls": [
            {"name": c["name"], "input": c["input"], "ok": c["ok"]} for c in result.tool_calls
        ],
        "provider": provider_name,
        "model": model,
        "truncated": result.truncated,
        # A refusal is a normal 200 reply carrying the explanation (AI_NOTES).
        "refused": result.refused,
    }


@router.get("/api/chat/history")
def history(
    session_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """History for a session. Without `session_id`, the most recent session."""
    if session_id is None:
        session = db.scalars(
            select(ChatSession).order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        ).first()
        if session is None:
            return {"session_id": None, "session": None, "messages": []}
    else:
        session = _get_session(db, session_id)

    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.id)
    ).all()
    return {
        "session_id": session.id,
        "session": _session_dict(session),
        "messages": [_message_dict(m) for m in messages],
    }


@router.get("/api/chat/sessions")
def list_sessions(
    limit: int = Query(default=20, ge=1, le=200), db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    sessions = db.scalars(
        select(ChatSession)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .limit(limit)
    ).all()
    return [_session_dict(s) for s in sessions]


@router.get("/api/chat/sessions/{session_id}/messages")
def session_messages(session_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    return history(session_id=session_id, db=db)


@router.delete("/api/chat/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    session = _get_session(db, session_id)
    db.delete(session)
    db.commit()
    return {"deleted": session_id}


@router.get("/api/chat/status")
def status(db: Session = Depends(get_db)) -> dict[str, Any]:
    """What the panel shows in its header: live provider or mock fallback."""
    provider_name = assistant_mod.configured_provider_name()
    has_key = provider_name != config.MOCK_PROVIDER
    if has_key:
        try:
            has_key = security.has_api_key(db, provider_name)
        except Exception:  # noqa: BLE001 - status must never 500
            has_key = False
    effective = provider_name if has_key else config.MOCK_PROVIDER
    return {
        "provider": effective,
        "model": assistant_mod.chat_model_for(effective),
        "using_mock": not has_key,
        "tools": list(assistant_mod.TOOL_NAMES),
        "pages": list(assistant_mod.PAGES),
    }


__all__ = ["router"]
