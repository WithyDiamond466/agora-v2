"""Chat module tests: the tool-calling loop, the API, and action payloads.

No network: every provider here is a mock. The `ScriptedProvider` below
implements the engine's `Provider.chat(messages, tools)` contract and replays a
canned sequence of turns, which is what lets us assert on a *real* tool round
(model asks for a tool -> handler queries SQLite -> result goes back -> model
answers) rather than on keyword matching.

If the engine module's `MockProvider` is already on disk, it is exercised too,
so a drift in the shared contract shows up here.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import analytics, config
from app.ai import assistant as A
from app.db import get_db
from app.main import app
from app.models import (
    Assignment,
    Base,
    ChatMessage,
    ChatSession,
    Course,
    GradeResult,
    Rubric,
    Skill,
    Student,
    Submission,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'chat.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def course(db):
    """A small graded course: 2 students, 1 assignment, 1 graded submission."""
    course = Course(name="PHIL 210", term="Fall 2026")
    db.add(course)
    db.commit()

    rubric = Rubric(
        name="Essay",
        criteria=[
            {"key": "thesis", "title": "Thesis", "description": "", "max_points": 10},
            {"key": "evidence", "title": "Evidence", "description": "", "max_points": 10},
        ],
    )
    skill = Skill(name="Strict Grader", system_prompt="Be strict.")
    db.add_all([rubric, skill])
    db.commit()

    assignment = Assignment(
        course_id=course.id, name="Problem Set 1", rubric_id=rubric.id, skill_id=skill.id
    )
    db.add(assignment)
    db.commit()

    students = [
        Student(course_id=course.id, name="Amara Osei", student_number=1),
        Student(course_id=course.id, name="Ben Whitaker", student_number=2),
    ]
    db.add_all(students)
    db.commit()

    graded = Submission(
        assignment_id=assignment.id,
        student_id=students[0].id,
        file_path="/tmp/a.pdf",
        mime_type="application/pdf",
        status="graded",
    )
    pending = Submission(
        assignment_id=assignment.id,
        student_id=students[1].id,
        file_path="/tmp/b.pdf",
        mime_type="application/pdf",
        status="pending",
    )
    db.add_all([graded, pending])
    db.commit()

    db.add(
        GradeResult(
            submission_id=graded.id,
            overall_score=15.0,
            max_score=20.0,
            summary_feedback="Solid thesis, thin evidence.",
            criteria=[
                {"key": "thesis", "score": 9, "max_points": 10, "comment": "Clear."},
                {"key": "evidence", "score": 6, "max_points": 10, "comment": "Thin."},
            ],
            misconceptions=["conflates legality with morality"],
            strengths=["clear thesis"],
            model=config.MOCK_MODEL,
        )
    )
    db.commit()
    return {
        "course": course,
        "assignment": assignment,
        "students": students,
        "rubric": rubric,
        "skill": skill,
    }


class ScriptedProvider:
    """Mock provider implementing the engine contract.

    ``chat(system_prompt, messages, tools=None)`` returns a ``ChatTurn``-shaped
    object; each scripted turn is either ``{"text": ...}`` or
    ``{"tool_calls": [{"name": ..., "input": {...}}]}``.
    """

    name = config.MOCK_PROVIDER
    model = config.MOCK_MODEL

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []

    def chat(self, system_prompt, messages, tools=None):
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "system": system_prompt}
        )
        if not self.turns:
            return {"text": "(no more scripted turns)", "tool_calls": []}
        turn = self.turns.pop(0)
        return {
            "text": turn.get("text", ""),
            # `arguments` is the engine's ToolCall field name.
            "tool_calls": [
                {"id": f"toolu_{i}", "name": c["name"], "arguments": c["input"]}
                for i, c in enumerate(turn.get("tool_calls", []))
            ],
            "stop_reason": "tool_use" if turn.get("tool_calls") else "end_turn",
        }


def install_provider(monkeypatch, provider):
    """Force the router to use `provider` (bypasses key lookup)."""
    monkeypatch.setattr(
        A, "build_provider", lambda db, **kw: (provider, config.MOCK_PROVIDER, config.MOCK_MODEL)
    )


# --------------------------------------------------------------------------
# tool definitions
# --------------------------------------------------------------------------


def test_tool_schemas_are_strict_and_complete():
    assert set(A.TOOL_NAMES) == {
        "get_course_summary",
        "get_student_summary",
        "get_assignment_results",
        "navigate_to",
        "start_grading",
    }
    assert set(A.TOOL_HANDLERS) == set(A.TOOL_NAMES)
    for tool in A.TOOLS:
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]
        assert tool["description"].strip()
        for key in schema["required"]:
            assert key in schema["properties"]

    nav = next(t for t in A.TOOLS if t["name"] == "navigate_to")
    assert nav["input_schema"]["properties"]["page"]["enum"] == list(A.PAGES)
    # Every page in the enum is routable.
    assert set(A.PAGES) == set(A.PAGE_URLS)


def test_navigate_urls_are_real_app_routes(client):
    """A navigate action must never send the UI to a URL nobody serves."""
    registered = set(client.get("/openapi.json").json()["paths"])
    for page, template in A.PAGE_URLS.items():
        assert template in registered, f"navigate_to({page!r}) -> {template} is not a route"


# --------------------------------------------------------------------------
# response normalization (never string-match tool inputs)
# --------------------------------------------------------------------------


def test_normalize_turn_handles_content_blocks_and_json_arguments():
    class Block:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    raw = Block(
        stop_reason="tool_use",
        content=[
            Block(type="text", text="Let me look."),
            Block(type="tool_use", id="tu_1", name="get_course_summary", input={"course_id": 3}),
        ],
    )
    turn = A.normalize_turn(raw)
    assert turn.text == "Let me look."
    assert turn.stop_reason == "tool_use"
    assert [(c.name, c.input) for c in turn.tool_calls] == [
        ("get_course_summary", {"course_id": 3})
    ]

    # Serialized arguments are parsed as JSON, never string-matched.
    dict_shape = A.normalize_turn(
        {
            "text": "ok",
            "tool_calls": [
                {"id": "1", "name": "navigate_to", "arguments": json.dumps({"page": "home"})}
            ],
        }
    )
    assert dict_shape.tool_calls[0].input == {"page": "home"}

    # Junk arguments degrade to an empty dict instead of exploding.
    assert A.normalize_turn(
        {"tool_calls": [{"id": "1", "name": "navigate_to", "input": "not json"}]}
    ).tool_calls[0].input == {}

    assert A.normalize_turn("plain string").text == "plain string"
    assert A.normalize_turn(None).text == ""


# --------------------------------------------------------------------------
# tool handlers hit the DB
# --------------------------------------------------------------------------


def test_get_course_summary_reads_real_data(db, course):
    second_submission = db.query(Submission).filter_by(
        student_id=course["students"][1].id
    ).one()
    second_submission.status = "graded"
    db.add(
        GradeResult(
            submission_id=second_submission.id,
            overall_score=10.0,
            max_score=20.0,
            summary_feedback="Needs revision.",
            criteria=[],
            misconceptions=[],
            strengths=[],
            model=config.MOCK_MODEL,
        )
    )
    db.commit()
    payload = A.tool_get_course_summary(db, {"course_id": course["course"].id})
    assert payload["name"] == "PHIL 210"
    assert payload["student_count"] == 2
    assert payload["assignments"][0]["submissions"] == 2
    assert payload["assignments"][0]["graded"] == 2
    assert payload["assignments"][0]["ungraded"] == 0
    # Privacy: the roster leaves the machine as numbers, never names.
    blob = json.dumps(payload)
    assert "Amara" not in blob and "Whitaker" not in blob
    assert payload["roster"][0]["label"] == "Student #1"
    assert payload["students_needing_attention"] == [
        {
            "student_number": 2,
            "label": "Student #2",
            "average_percent": 50.0,
            "graded_count": 1,
        },
        {
            "student_number": 1,
            "label": "Student #1",
            "average_percent": 75.0,
            "graded_count": 1,
        },
    ]


def test_course_summary_averages_match_the_roster_column(db, course):
    rounding_assignment = Assignment(
        course_id=course["course"].id, name="Rounding check"
    )
    db.add(rounding_assignment)
    db.flush()
    rounding_submission = Submission(
        assignment_id=rounding_assignment.id,
        student_id=course["students"][0].id,
        status="graded",
    )
    db.add(rounding_submission)
    db.flush()
    db.add(
        GradeResult(
            submission_id=rounding_submission.id,
            overall_score=1.0,
            max_score=3.0,
        )
    )

    pending = db.query(Submission).filter_by(
        student_id=course["students"][1].id
    ).one()
    pending.status = "graded"
    db.add(
        GradeResult(
            submission_id=pending.id,
            overall_score=1.0,
            max_score=2.0,
        )
    )
    db.commit()

    payload = A.tool_get_course_summary(db, {"course_id": course["course"].id})
    averages = {
        row["student_number"]: row["average_percent"]
        for row in payload["students_needing_attention"]
    }
    assert set(averages) == {student.student_number for student in course["students"]}
    for student in course["students"]:
        roster_percentages = [
            submission.grade_result.percentage
            for submission in student.submissions
            if submission.grade_result is not None
        ]
        assert averages[student.student_number] == round(mean(roster_percentages), 1)


def test_course_summary_student_average_uses_displayed_grade_percentages(db):
    course = Course(name="Rounding course")
    db.add(course)
    db.flush()
    student = Student(course_id=course.id, name="Ada Lovelace", student_number=1)
    assignments = [
        Assignment(course_id=course.id, name="Half"),
        Assignment(course_id=course.id, name="Third"),
    ]
    db.add_all([student, *assignments])
    db.flush()
    submissions = [
        Submission(
            assignment_id=assignment.id,
            student_id=student.id,
            status="graded",
        )
        for assignment in assignments
    ]
    db.add_all(submissions)
    db.flush()
    results = [
        GradeResult(submission_id=submissions[0].id, overall_score=1, max_score=2),
        GradeResult(submission_id=submissions[1].id, overall_score=1, max_score=3),
    ]
    db.add_all(results)
    db.commit()

    assert [result.percentage for result in results] == [50.0, 33.3]
    summary = A.tool_get_course_summary(db, {"course_id": course.id})
    average = summary["students_needing_attention"][0]["average_percent"]
    assert average == 41.6
    assert average != 41.7
    assert average == analytics.student_timeline(db, student.id)["average_percent"]


def test_get_student_and_assignment_tools(db, course):
    student = course["students"][0]
    summary = A.tool_get_student_summary(
        db,
        {
            "course_id": course["course"].id,
            "student_number": student.student_number,
        },
    )
    assert summary["label"] == "Student #1"
    assert "Amara" not in json.dumps(summary)
    assert summary["graded_count"] == 1
    assert summary["average_percent"] == pytest.approx(75.0)

    results = A.tool_get_assignment_results(db, {"assignment_id": course["assignment"].id})
    assert results["name"] == "Problem Set 1"
    assert results["progress"]["graded"] == 1
    assert results["average_percent"] == pytest.approx(75.0)
    assert {c["key"] for c in results["criteria_averages"]} == {"thesis", "evidence"}
    assert results["misconceptions"][0]["tag"] == "conflates legality with morality"
    assert "Amara" not in json.dumps(results)


def test_student_summary_tool_resolves_per_course_number(db):
    course_a = Course(name="Course A", term="Fall 2026")
    course_b = Course(name="Course B", term="Fall 2026")
    db.add_all([course_a, course_b])
    db.commit()

    db.add_all(
        [
            Student(course_id=course_a.id, name=f"A Student {number}", student_number=number)
            for number in range(1, 5)
        ]
    )
    db.commit()
    course_a_five = Student(
        course_id=course_a.id, name="A Student 5", student_number=5
    )
    db.add(course_a_five)
    db.commit()
    assert course_a_five.id == 5

    course_b_five = Student(
        course_id=course_b.id, name="B Student 5", student_number=5
    )
    db.add(course_b_five)
    db.commit()
    assert course_b_five.id != 5

    payload = A.tool_get_student_summary(
        db, {"course_id": course_b.id, "student_number": 5}
    )
    assert payload["student_id"] == course_b_five.id
    assert payload["course_id"] == course_b.id
    assert payload["student_number"] == 5
    assert "A Student" not in json.dumps(payload)

    error, is_error, action = A.dispatch_tool(
        db,
        A.ToolCall(
            id="missing",
            name="get_student_summary",
            input={"course_id": course_b.id, "student_number": 99},
        ),
    )
    assert is_error is True
    assert action is None
    assert "Student #99" in error["error"]


def test_tool_errors_are_reported_not_raised(db, course):
    payload, is_error, action = A.dispatch_tool(
        db, A.ToolCall(id="1", name="get_course_summary", input={"course_id": 9999})
    )
    assert is_error is True and action is None and "9999" in payload["error"]

    payload, is_error, _ = A.dispatch_tool(
        db, A.ToolCall(id="2", name="get_course_summary", input={"course_id": "banana"})
    )
    assert is_error is True

    payload, is_error, _ = A.dispatch_tool(
        db, A.ToolCall(id="3", name="does_not_exist", input={})
    )
    assert is_error is True and "Unknown tool" in payload["error"]

    # navigate_to must not send the UI to a page it cannot render.
    payload, is_error, _ = A.dispatch_tool(
        db, A.ToolCall(id="4", name="navigate_to", input={"page": "course_detail"})
    )
    assert is_error is True and "course_id" in payload["error"]

    payload, is_error, _ = A.dispatch_tool(
        db, A.ToolCall(id="5", name="navigate_to", input={"page": "atlantis"})
    )
    assert is_error is True


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def test_loop_runs_a_tool_round_and_answers(db, course):
    provider = ScriptedProvider(
        [
            {"tool_calls": [{"name": "get_course_summary", "input": {"course_id": course["course"].id}}]},
            {"text": "PHIL 210 has 2 students; 1 of 2 submissions on Problem Set 1 is graded."},
        ]
    )
    result = A.run_assistant(
        db, "How is PHIL 210 doing?", provider=provider, context={"page": "home"}
    )

    assert result.iterations == 1
    assert result.truncated is False
    assert "PHIL 210" in result.reply
    assert result.actions == []
    assert [c["name"] for c in result.tool_calls] == ["get_course_summary"]
    assert result.tool_calls[0]["ok"] is True

    # Two provider calls; the second carried the tool result back.
    assert len(provider.calls) == 2
    second = provider.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert any(b["type"] == "tool_use" for b in second[-2]["content"])
    results_msg = second[-1]
    assert results_msg["role"] in ("tool_results", "tool")
    tool_result = results_msg["results"][0]
    assert tool_result["tool_use_id"] == "toolu_0"
    assert tool_result["is_error"] is False
    assert json.loads(tool_result["content"])["name"] == "PHIL 210"

    # Tools and the page-context system prompt were sent on every call.
    for call in provider.calls:
        assert call["tools"] == A.TOOLS
        assert "Current page: home" in call["system"]


def test_error_tool_result_is_flagged_for_the_model(db, course):
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    {
                        "name": "get_student_summary",
                        "input": {
                            "course_id": course["course"].id,
                            "student_number": 4242,
                        },
                    }
                ]
            },
            {"text": "I could not find that student."},
        ]
    )
    result = A.run_assistant(db, "How is student 4242?", provider=provider)
    assert result.tool_calls[0]["ok"] is False
    sent = provider.calls[1]["messages"][-1]["results"][0]
    assert sent["is_error"] is True
    assert "4242" in json.loads(sent["content"])["error"]


def test_system_prompt_includes_page_context(db, course):
    prompt = A.build_system_prompt(
        db,
        {
            "page": "course_detail",
            "course_id": course["course"].id,
            "skill_id": course["skill"].id,
            "student_id": course["students"][0].id,
        },
    )
    assert "Current page: course_detail" in prompt
    assert f"id={course['course'].id}" in prompt
    assert "'Strict Grader'" in prompt
    assert "Problem Set 1" in prompt
    assert "Student in context: course_id=1, student_number=1" in prompt
    # Roster names never go into the prompt.
    assert "Amara" not in prompt


def test_tool_iteration_cap(db, course):
    """The model looping forever must be cut off at MAX_TOOL_ITERATIONS."""
    forever = [
        {"tool_calls": [{"name": "get_course_summary", "input": {"course_id": course["course"].id}}]}
        for _ in range(50)
    ]
    provider = ScriptedProvider(forever)
    result = A.run_assistant(db, "loop forever", provider=provider)

    assert A.MAX_TOOL_ITERATIONS == 6
    assert result.iterations == A.MAX_TOOL_ITERATIONS
    assert len(result.tool_calls) == A.MAX_TOOL_ITERATIONS
    # 6 tool rounds + 1 final wrap-up call.
    assert len(provider.calls) == A.MAX_TOOL_ITERATIONS + 1
    assert result.truncated is True
    assert result.reply  # never empty

    lower = A.run_assistant(db, "again", provider=ScriptedProvider(forever), max_iterations=2)
    assert lower.iterations == 2


def test_provider_failure_becomes_assistant_error(db):
    class Broken:
        def chat(self, system_prompt, messages, tools=None):
            raise RuntimeError("connection refused")

    with pytest.raises(A.AssistantError):
        A.run_assistant(db, "hi", provider=Broken())


def test_minimal_provider_signature_is_supported(db):
    """A provider that only accepts (messages, tools) still works."""

    class Minimal:
        def chat(self, messages, tools=None):
            return "hello from a minimal provider"

    assert A.run_assistant(db, "hi", provider=Minimal()).reply == (
        "hello from a minimal provider"
    )


def test_native_chat_turn_object_is_understood(db, course):
    """The engine's real ChatTurn/ToolCall dataclasses drive the loop."""
    from app.ai import providers

    class NativeProvider:
        def __init__(self):
            self.turns = [
                providers.ChatTurn(
                    text="",
                    tool_calls=[
                        providers.ToolCall(
                            id="toolu_native",
                            name="get_course_summary",
                            arguments={"course_id": course["course"].id},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                providers.ChatTurn(text="PHIL 210 looks fine.", stop_reason="end_turn"),
            ]

        def chat(self, system_prompt, messages, tools=None):
            return self.turns.pop(0)

    result = A.run_assistant(db, "how is the course?", provider=NativeProvider())
    assert result.reply == "PHIL 210 looks fine."
    assert [c["name"] for c in result.tool_calls] == ["get_course_summary"]
    assert result.tool_calls[0]["input"] == {"course_id": course["course"].id}
    assert result.tool_calls[0]["ok"] is True


# --------------------------------------------------------------------------
# action payloads
# --------------------------------------------------------------------------


def test_navigate_action_payload(db, course):
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    {
                        "name": "navigate_to",
                        "input": {"page": "analytics", "course_id": course["course"].id},
                    }
                ]
            },
            {"text": "Opening the analytics dashboard."},
        ]
    )
    result = A.run_assistant(db, "show me the analytics", provider=provider)
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action["type"] == "navigate"
    assert action["page"] == "analytics"
    assert action["course_id"] == course["course"].id
    assert action["url"] == f"/courses/{course['course'].id}/analytics"


def test_start_grading_only_asks_for_confirmation(db, course):
    assignment = course["assignment"]
    provider = ScriptedProvider(
        [
            {"tool_calls": [{"name": "start_grading", "input": {"assignment_id": assignment.id}}]},
            {"text": "Confirm in the panel and I will grade the 1 ungraded submission."},
        ]
    )
    before = db.query(GradeResult).count()
    result = A.run_assistant(db, "grade problem set 1", provider=provider)

    action = result.actions[0]
    assert action["type"] == "confirm_grading"
    assert action["assignment_id"] == assignment.id
    assert action["ungraded_count"] == 1
    assert action["requires_confirmation"] is True
    # Nothing was graded server-side — that is the frontend's job, post-confirm.
    assert db.query(GradeResult).count() == before
    assert db.query(Submission).filter_by(status="grading").count() == 0
    assert result.tool_calls[0]["result"]["status"] == "awaiting_user_confirmation"


def test_start_grading_only_counts_rows_grade_all_can_queue(db, course):
    assignment = Assignment(course_id=course["course"].id, name="Mixed statuses")
    db.add(assignment)
    db.flush()
    mapped_student = course["students"][0]
    db.add_all(
        [
            Submission(
                assignment_id=assignment.id,
                student_id=mapped_student.id,
                status="failed",
            ),
            Submission(assignment_id=assignment.id, student_id=None, status="pending"),
            Submission(
                assignment_id=assignment.id,
                student_id=mapped_student.id,
                status="grading",
            ),
            Submission(
                assignment_id=assignment.id,
                student_id=mapped_student.id,
                status="graded",
            ),
        ]
    )
    db.commit()

    payload, is_error, action = A.dispatch_tool(
        db, A.ToolCall(id="1", name="start_grading", input={"assignment_id": assignment.id})
    )

    assert is_error is False
    assert payload["ungraded_count"] == 1
    assert action["submission_count"] == 4
    assert action["ungraded_count"] == 1
    assert "Grade 1" in action["message"]


def test_start_grading_refuses_when_no_submission_is_queueable(db, course):
    assignment = Assignment(course_id=course["course"].id, name="Nothing queueable")
    db.add(assignment)
    db.flush()
    mapped_student = course["students"][0]
    db.add_all(
        [
            Submission(
                assignment_id=assignment.id,
                student_id=mapped_student.id,
                status="graded",
            ),
            Submission(
                assignment_id=assignment.id,
                student_id=mapped_student.id,
                status="grading",
            ),
            Submission(assignment_id=assignment.id, student_id=None, status="pending"),
        ]
    )
    db.commit()

    payload, is_error, action = A.dispatch_tool(
        db, A.ToolCall(id="1", name="start_grading", input={"assignment_id": assignment.id})
    )

    assert is_error is True
    assert action is None
    assert (
        "Assignment 'Nothing queueable' has no mapped pending or failed submissions to grade."
        in payload["error"]
    )


def test_start_grading_refuses_empty_assignment(db, course):
    empty = Assignment(course_id=course["course"].id, name="PS2")
    db.add(empty)
    db.commit()
    payload, is_error, action = A.dispatch_tool(
        db, A.ToolCall(id="1", name="start_grading", input={"assignment_id": empty.id})
    )
    assert is_error is True and action is None
    assert "no uploaded submissions" in payload["error"]


def test_chat_grading_confirmation_only_opens_privacy_gated_queue():
    source = Path("static/js/chat.js").read_text(encoding="utf-8")
    start = source.index("Chat.prototype.confirmCard")
    handler = source[start : source.index("/* ---- panel wiring", start)]

    assert '"Review privacy & grade"' in handler
    assert '"→ opening the privacy-gated grading queue"' in handler
    assert 'window.location.assign("/grading/" + assignmentId)' in handler
    assert "fetch(endpoint" not in handler
    assert "grade-all" not in handler


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def test_chat_endpoint_full_round_trip(client, db, course, monkeypatch):
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    {"name": "get_assignment_results", "input": {"assignment_id": course["assignment"].id}}
                ]
            },
            {
                "text": "Problem Set 1 averages 75.0%. Opening it.",
                "tool_calls": [
                    {
                        "name": "navigate_to",
                        "input": {"page": "grading", "assignment_id": course["assignment"].id},
                    }
                ],
            },
            {"text": "Problem Set 1 averages 75.0%. Opening the grading page."},
        ]
    )
    install_provider(monkeypatch, provider)

    resp = client.post(
        "/api/chat",
        json={
            "message": "How did Problem Set 1 go? Open it.",
            "context": {"page": "course_detail", "course_id": course["course"].id},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(["reply", "actions", "session_id"]).issubset(body)
    assert "75.0%" in body["reply"]
    assert body["session_id"] is not None
    assert [a["type"] for a in body["actions"]] == ["navigate"]
    assert body["actions"][0]["url"] == f"/grading/{course['assignment'].id}"
    assert [c["name"] for c in body["tool_calls"]] == [
        "get_assignment_results",
        "navigate_to",
    ]
    assert body["provider"] == config.MOCK_PROVIDER

    # Persistence: session + both messages, with the tool/action record attached.
    session = db.get(ChatSession, body["session_id"])
    assert session is not None and session.course_id == course["course"].id
    stored = db.query(ChatMessage).filter_by(session_id=session.id).order_by(ChatMessage.id).all()
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[0].content == "How did Problem Set 1 go? Open it."
    kinds = {r["kind"] for r in stored[1].tool_calls}
    assert kinds == {"tool_call", "action", "provider"}


def test_mock_chat_endpoint_answers_from_student_tool_result(
    client, course, monkeypatch
):
    from app.ai.providers import MockProvider

    install_provider(monkeypatch, MockProvider())
    response = client.post(
        "/api/chat",
        json={
            "message": "Show me student #1",
            "context": {
                "page": "course_detail",
                "course_id": course["course"].id,
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "Student #1" in body["reply"]
    assert "75.0%" in body["reply"]


def test_history_includes_mock_provider_on_assistant_message(
    client, course, monkeypatch
):
    from app.ai.providers import MockProvider

    install_provider(monkeypatch, MockProvider())
    response = client.post(
        "/api/chat",
        json={
            "message": "Show me student #1",
            "context": {
                "page": "course_detail",
                "course_id": course["course"].id,
            },
        },
    )
    assert response.status_code == 200, response.text

    payload = client.get(
        f"/api/chat/history?session_id={response.json()['session_id']}"
    ).json()
    assistant_message = next(m for m in payload["messages"] if m["role"] == "assistant")
    assert assistant_message["provider"] == config.MOCK_PROVIDER


def test_mock_chat_endpoint_names_lowest_students_without_native_names(
    client, db, course, monkeypatch
):
    from app.ai.providers import MockProvider

    second_submission = db.query(Submission).filter_by(
        student_id=course["students"][1].id
    ).one()
    second_submission.status = "graded"
    db.add(
        GradeResult(
            submission_id=second_submission.id,
            overall_score=10.0,
            max_score=20.0,
            summary_feedback="Needs revision.",
            criteria=[],
            misconceptions=[],
            strengths=[],
            model=config.MOCK_MODEL,
        )
    )
    db.commit()
    install_provider(monkeypatch, MockProvider())
    response = client.post(
        "/api/chat",
        json={
            "message": "Which students are struggling most in PHIL 210?",
            "context": {"page": "home"},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "Student #2" in body["reply"]
    assert "50.0%" in body["reply"]
    assert "Amara Osei" not in body["reply"]
    assert "Ben Whitaker" not in body["reply"]
    assert body["tool_calls"][0]["name"] == "get_course_summary"
    assert body["tool_calls"][0]["input"] == {"course_id": course["course"].id}
    assert body["tool_calls"][0]["ok"] is True


def test_mock_chat_explicit_course_overrides_page_context(
    client, db, course, monkeypatch
):
    from app.ai.providers import MockProvider

    history = Course(name="HIST 101", term="Fall 2026")
    db.add(history)
    db.commit()
    install_provider(monkeypatch, MockProvider())

    response = client.post(
        "/api/chat",
        json={
            "message": "How is HIST 101 doing?",
            "context": {
                "page": "course_detail",
                "course_id": course["course"].id,
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tool_calls"][0]["name"] == "get_course_summary"
    assert body["tool_calls"][0]["input"] == {"course_id": history.id}
    assert body["tool_calls"][0]["ok"] is True


def test_mock_chat_endpoint_resolves_course_after_first_twenty(
    client, db, monkeypatch
):
    from app.ai.providers import MockProvider

    class RecordingMockProvider(MockProvider):
        def __init__(self):
            super().__init__()
            self.system_prompts = []

        def chat(self, system_prompt, messages, tools=None):
            self.system_prompts.append(system_prompt)
            return super().chat(system_prompt, messages, tools)

    courses = [Course(name=f"C{i}", term="Fall 2026") for i in range(1, 22)]
    db.add_all(courses)
    db.commit()
    provider = RecordingMockProvider()
    install_provider(monkeypatch, provider)

    response = client.post(
        "/api/chat",
        json={"message": "How is C21 doing?", "context": {"page": "home"}},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tool_calls"][0]["name"] == "get_course_summary"
    assert body["tool_calls"][0]["input"] == {"course_id": courses[20].id}
    assert body["tool_calls"][0]["ok"] is True
    assert (
        f'Known course: id={courses[20].id}, name="C21"'
        in provider.system_prompts[0]
    )


def test_chat_endpoint_continues_a_session_with_history(client, db, course, monkeypatch):
    provider = ScriptedProvider([{"text": "first"}, {"text": "second"}])
    install_provider(monkeypatch, provider)

    first = client.post("/api/chat", json={"message": "hello", "context": {"page": "home"}}).json()
    second = client.post(
        "/api/chat",
        json={"message": "and now?", "session_id": first["session_id"], "context": {"page": "home"}},
    ).json()

    assert second["session_id"] == first["session_id"]
    # The second provider call replayed the first exchange.
    replayed = provider.calls[1]["messages"]
    assert [m["content"] for m in replayed] == ["hello", "first", "and now?"]
    assert db.query(ChatMessage).filter_by(session_id=first["session_id"]).count() == 4


def test_chat_invalid_course_context_returns_404_without_writes(
    client, db, course, monkeypatch
):
    provider = ScriptedProvider([{"text": "must not run"}])
    install_provider(monkeypatch, provider)

    response = client.post(
        "/api/chat",
        json={"message": "hello", "context": {"course_id": 999999}},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Course 999999 not found"
    assert db.query(ChatSession).count() == 0
    assert db.query(ChatMessage).count() == 0

    session = ChatSession(title="existing", course_id=course["course"].id)
    db.add(session)
    db.commit()
    db.add(ChatMessage(session_id=session.id, role="user", content="earlier"))
    db.commit()
    message_count = db.query(ChatMessage).count()

    response = client.post(
        "/api/chat",
        json={
            "message": "hello again",
            "session_id": session.id,
            "context": {"course_id": 999999},
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Course 999999 not found"
    db.expire_all()
    assert db.get(ChatSession, session.id).course_id == course["course"].id
    assert db.query(ChatMessage).count() == message_count
    assert provider.calls == []


def test_chat_invalid_skill_context_returns_404_without_writes(
    client, db, course, monkeypatch
):
    provider = ScriptedProvider([{"text": "must not run"}])
    install_provider(monkeypatch, provider)

    response = client.post(
        "/api/chat",
        json={"message": "hello", "context": {"skill_id": 999999}},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Skill 999999 not found"
    assert db.query(ChatSession).count() == 0
    assert db.query(ChatMessage).count() == 0

    session = ChatSession(title="existing", skill_id=course["skill"].id)
    db.add(session)
    db.commit()
    db.add(ChatMessage(session_id=session.id, role="user", content="earlier"))
    db.commit()
    message_count = db.query(ChatMessage).count()

    response = client.post(
        "/api/chat",
        json={
            "message": "hello again",
            "session_id": session.id,
            "context": {"skill_id": 999999},
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Skill 999999 not found"
    db.expire_all()
    assert db.get(ChatSession, session.id).skill_id == course["skill"].id
    assert db.query(ChatMessage).count() == message_count
    assert provider.calls == []


def test_chat_rejects_invalid_or_cross_course_record_context_without_writes(
    client, db, course, monkeypatch
):
    provider = ScriptedProvider([{"text": "valid"}])
    install_provider(monkeypatch, provider)

    other_course = Course(name="HIST 101", term="Fall 2026")
    db.add(other_course)
    db.flush()
    other_student = Student(
        course_id=other_course.id, name="Other Student", student_number=1
    )
    other_assignment = Assignment(course_id=other_course.id, name="Other Assignment")
    db.add_all([other_student, other_assignment])

    session = ChatSession(title="existing", course_id=course["course"].id)
    db.add(session)
    db.flush()
    db.add(ChatMessage(session_id=session.id, role="user", content="earlier"))
    db.commit()

    session_id = session.id
    db.expire_all()
    session = db.get(ChatSession, session_id)
    session_count = db.query(ChatSession).count()
    message_count = db.query(ChatMessage).count()
    session_state = (session.title, session.course_id, session.skill_id, session.updated_at)
    cases = [
        (
            {"context": {"student_id": 999999}},
            404,
            "Student 999999 not found",
        ),
        (
            {"context": {"assignment_id": 999999}},
            404,
            "Assignment 999999 not found",
        ),
        (
            {
                "context": {
                    "course_id": course["course"].id,
                    "student_id": other_student.id,
                }
            },
            409,
            f"Student {other_student.id} does not belong to course {course['course'].id}",
        ),
        (
            {
                "context": {
                    "course_id": course["course"].id,
                    "assignment_id": other_assignment.id,
                }
            },
            409,
            f"Assignment {other_assignment.id} does not belong to course {course['course'].id}",
        ),
        (
            {
                "session_id": session.id,
                "context": {"student_id": other_student.id},
            },
            409,
            f"Student {other_student.id} does not belong to course {course['course'].id}",
        ),
    ]

    for request, status_code, detail in cases:
        response = client.post("/api/chat", json={"message": "hello", **request})
        assert response.status_code == status_code
        assert response.json()["detail"] == detail

    assert provider.calls == []
    assert db.query(ChatSession).count() == session_count
    assert db.query(ChatMessage).count() == message_count
    db.expire_all()
    unchanged_session = db.get(ChatSession, session.id)
    assert unchanged_session is not None
    assert (
        unchanged_session.title,
        unchanged_session.course_id,
        unchanged_session.skill_id,
        unchanged_session.updated_at,
    ) == session_state

    response = client.post(
        "/api/chat",
        json={
            "message": "valid context",
            "context": {
                "course_id": course["course"].id,
                "student_id": course["students"][0].id,
                "assignment_id": course["assignment"].id,
            },
        },
    )
    assert response.status_code == 200
    assert len(provider.calls) == 1


def test_history_and_session_endpoints(client, db, course, monkeypatch):
    install_provider(monkeypatch, ScriptedProvider([{"text": "hi there"}]))
    created = client.post(
        "/api/chat", json={"message": "hello", "context": {"page": "home"}}
    ).json()
    session_id = created["session_id"]

    history = client.get(f"/api/chat/history?session_id={session_id}")
    assert history.status_code == 200
    payload = history.json()
    assert payload["session_id"] == session_id
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant"]
    assert payload["messages"][1]["content"] == "hi there"
    assert payload["session"]["title"] == "hello"

    assert client.get("/api/chat/history").json()["session_id"] == session_id
    alias = client.get(f"/api/chat/sessions/{session_id}/messages").json()
    assert alias == payload
    assert [s["id"] for s in client.get("/api/chat/sessions").json()] == [session_id]

    assert client.get("/api/chat/history?session_id=999999").status_code == 404
    assert client.delete(f"/api/chat/sessions/{session_id}").status_code == 200
    assert db.query(ChatSession).count() == 0
    assert db.query(ChatMessage).count() == 0
    assert client.get("/api/chat/history").json()["messages"] == []


def test_chat_endpoint_validation_and_provider_error(client, db, monkeypatch):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert (
        client.post("/api/chat", json={"message": "hi", "session_id": 4242}).status_code == 404
    )

    class Broken:
        def chat(self, system_prompt, messages, tools=None):
            raise RuntimeError("no route to host")

    install_provider(monkeypatch, Broken())
    resp = client.post("/api/chat", json={"message": "hi", "context": {"page": "home"}})
    assert resp.status_code == 502
    assert "no route to host" in resp.json()["detail"]


def test_history_window_always_starts_on_a_user_turn(client, db, course, monkeypatch):
    """The Messages API rejects a history whose first message is an assistant turn.

    That happens whenever the stored rows have odd parity — e.g. an assistant
    reply was lost to a provider failure, or the window boundary lands
    mid-exchange.
    """
    session = ChatSession(title="stranded")
    db.add(session)
    db.commit()
    db.add_all(
        [
            # An assistant turn whose user message is gone, then a full exchange,
            # then a user turn whose reply never arrived.
            ChatMessage(session_id=session.id, role="assistant", content="orphaned reply"),
            ChatMessage(session_id=session.id, role="user", content="first question"),
            ChatMessage(session_id=session.id, role="assistant", content="first answer"),
            ChatMessage(session_id=session.id, role="user", content="lost question"),
        ]
    )
    db.commit()

    provider = ScriptedProvider([{"text": "ok"}])
    install_provider(monkeypatch, provider)
    resp = client.post("/api/chat", json={"message": "next question", "session_id": session.id})
    assert resp.status_code == 200

    sent = provider.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert [m["content"] for m in sent] == ["first question", "first answer", "next question"]


def test_refusal_is_answered_not_a_502(client, db, course, monkeypatch):
    """AI_NOTES: a refusal is surfaced to the professor, never a crash."""
    from app.ai import providers

    class Refusing:
        def chat(self, system_prompt, messages, tools=None):
            raise providers.ProviderRefusalError(
                "Claude declined to process this submission.", category="jailbreak"
            )

    install_provider(monkeypatch, Refusing())
    resp = client.post("/api/chat", json={"message": "hi", "context": {"page": "home"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert "declined" in body["reply"].lower()
    assert "jailbreak" in body["reply"]
    # The refusal is stored like any other reply.
    assert db.query(ChatMessage).filter_by(role="assistant").count() == 1


def test_sdk_type_errors_do_not_silently_drop_tools(db):
    """A TypeError from inside the SDK must not degrade to a tool-less retry."""

    class Exploding:
        def chat(self, system_prompt, messages, tools=None):
            raise TypeError("unsupported operand type(s) inside the SDK")

    with pytest.raises(A.AssistantError):
        A.run_assistant(db, "hi", provider=Exploding())


def test_status_endpoint_falls_back_to_mock(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = client.get("/api/chat/status").json()
    assert body["using_mock"] is True
    assert body["provider"] == config.MOCK_PROVIDER
    assert set(body["tools"]) == set(A.TOOL_NAMES)


def test_build_provider_uses_mock_without_a_key(db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider, name, model = A.build_provider(db)
    assert name == config.MOCK_PROVIDER
    assert callable(getattr(provider, "chat", None))
    assert model == config.MOCK_MODEL
    # A live provider is never constructed without a key.
    assert A.run_assistant(db, "hello", provider=provider).reply


# --------------------------------------------------------------------------
# engine contract (only once app/ai/providers.py exists)
# --------------------------------------------------------------------------


def test_engine_mock_provider_drives_the_loop(db, course):
    from app.ai import providers

    mock_cls = providers.MockProvider
    try:
        provider = mock_cls()
    except TypeError:
        provider = mock_cls(model=config.MOCK_MODEL)

    result = A.run_assistant(
        db,
        "How is this course doing?",
        provider=provider,
        context={"page": "course_detail", "course_id": course["course"].id},
    )
    assert isinstance(result.reply, str) and result.reply
    assert isinstance(result.actions, list)
    assert result.iterations <= A.MAX_TOOL_ITERATIONS
    for call in result.tool_calls:
        assert call["name"] in A.TOOL_NAMES
        assert isinstance(call["input"], dict)


def test_tool_get_course_summary_query_count_is_constant(db):
    one_course = Course(name="One assignment", term="Fall 2026")
    four_course = Course(name="Four assignments", term="Fall 2026")
    db.add_all([one_course, four_course])
    db.flush()

    assignments_by_course = {
        one_course.id: [
            Assignment(course_id=one_course.id, name="Assignment 1"),
        ],
        four_course.id: [
            Assignment(course_id=four_course.id, name=f"Assignment {index}")
            for index in range(1, 5)
        ],
    }
    db.add_all(
        [
            assignment
            for assignments in assignments_by_course.values()
            for assignment in assignments
        ]
    )
    db.flush()
    for assignments in assignments_by_course.values():
        for assignment in assignments:
            db.add_all(
                [
                    Submission(assignment_id=assignment.id, status="pending"),
                    Submission(assignment_id=assignment.id, status="grading"),
                    Submission(assignment_id=assignment.id, status="graded"),
                    Submission(assignment_id=assignment.id, status="failed"),
                ]
            )
    db.commit()

    def measure(course_id):
        db.expire_all()
        selects = []

        def count_selects(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                selects.append(statement)

        engine_ = db.get_bind()
        event.listen(engine_, "before_cursor_execute", count_selects)
        try:
            payload = A.tool_get_course_summary(db, {"course_id": course_id})
        finally:
            event.remove(engine_, "before_cursor_execute", count_selects)
        return payload, selects

    def expected_progress(assignments):
        return [
            {
                "assignment_id": assignment.id,
                "name": assignment.name,
                "submissions": 4,
                "graded": 1,
                "pending": 1,
                "grading": 1,
                "failed": 1,
                "ungraded": 3,
                "has_rubric": False,
                "has_skill": False,
            }
            for assignment in assignments
        ]

    one_payload, one_selects = measure(one_course.id)
    four_payload, four_selects = measure(four_course.id)

    assert one_payload["assignments"] == expected_progress(
        assignments_by_course[one_course.id]
    )
    assert four_payload["assignments"] == expected_progress(
        assignments_by_course[four_course.id]
    )

    def grouped_status_queries(selects):
        normalized = [" ".join(statement.lower().split()) for statement in selects]
        return [
            statement
            for statement in normalized
            if "count(submissions.id)" in statement
            and "group by submissions.assignment_id, submissions.status" in statement
        ]

    assert len(grouped_status_queries(one_selects)) == 1
    assert len(grouped_status_queries(four_selects)) == 1
    assert len(one_selects) == len(four_selects)


def test_chat_uses_configured_provider_preference(db, monkeypatch):
    monkeypatch.setattr(A, 'configured_provider_name', lambda: 'local')
    captured = []
    sentinel = object()
    def build(mod, name, model, session):
        captured.append(name)
        return sentinel
    monkeypatch.setattr(A, '_live_provider', build)
    provider, name, model = A.build_provider(db)
    assert name == 'local'
    assert captured == ['local']
    assert provider is sentinel
