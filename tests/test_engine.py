"""Engine tests: providers, grade validation, and the full MockProvider flow.

Nothing here touches a network. Anthropic/OpenAI providers are exercised with
injected fake clients so the request shape (structured outputs, refusal
handling, no sampling params) is pinned down without an API key.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app import config
from app.ai import grading as engine
from app.ai import providers as providers_mod
from app.ai.providers import (
    AnthropicProvider,
    ChatTurn,
    MockProvider,
    OpenAIProvider,
    ProviderConfigError,
    ProviderRefusalError,
    ProviderResponseError,
    ProviderUnsupportedError,
    get_provider,
)
from app.db import get_db
from app.main import app
from app.models import Assignment, Base, Course, GradeResult, Skill, Student, Submission
from app.routers import courses as courses_router
from app.routers import grading as grading_router

PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< >>\nendobj\ntrailer\n<< >>\n%%EOF\n"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

RUBRIC_CRITERIA = [
    {"key": "thesis", "title": "Thesis", "description": "States a thesis.", "max_points": 10},
    {"key": "evidence", "title": "Evidence", "description": "Uses sources.", "max_points": 10},
    {"key": "mechanics", "title": "Mechanics", "description": "Prose quality.", "max_points": 5},
]

ROSTER = ["Amara Osei", "Ben Whitaker", "Claudia Moreno"]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def Session(tmp_path):
    engine_ = create_engine(
        f"sqlite:///{tmp_path / 'engine.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine_)
    factory = sessionmaker(bind=engine_, autoflush=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine_.dispose()


@pytest.fixture()
def db(Session):
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(Session, tmp_path, monkeypatch):
    """App wired to the tmp DB, with background jobs using the same engine."""
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "submissions")
    monkeypatch.setattr(grading_router, "session_factory", Session)

    def override_get_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def course_setup(client, db):
    """A course with a roster, a mock-provider skill, a rubric and an assignment."""
    course = client.post("/api/courses", json={"name": "PHIL 210", "term": "Fall 2026"}).json()
    students = [
        client.post(f"/api/courses/{course['id']}/students", json={"name": name}).json()
        for name in ROSTER
    ]
    rubric = client.post(
        "/api/rubrics", json={"name": "Essay Rubric", "criteria": RUBRIC_CRITERIA}
    ).json()

    skill = Skill(
        name="Mock Grader",
        system_prompt="You are grading PHIL 210 essays. Be exacting but humane.",
        provider=config.MOCK_PROVIDER,
        model=config.MOCK_MODEL,
        max_tokens=4000,
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)

    assignment = client.post(
        f"/api/courses/{course['id']}/assignments",
        json={
            "name": "Essay 1",
            "description": "Argue for or against algorithmic recommendation.",
            "rubric_id": rubric["id"],
            "skill_id": skill.id,
        },
    ).json()
    return {
        "course": course,
        "students": students,
        "rubric": rubric,
        "skill_id": skill.id,
        "assignment": assignment,
    }


def upload(client, assignment_id, files):
    return client.post(
        f"/api/assignments/{assignment_id}/submissions",
        files=[("files", (name, io.BytesIO(data), mime)) for name, data, mime in files],
    )


def assert_no_delete_backups(root):
    if root.exists():
        assert not [path for path in root.rglob("*") if ".agora-delete-" in path.name]


def test_list_courses_uses_one_select(client, db):
    courses = [Course(name=f"Course {index}") for index in range(3)]
    db.add_all(courses)
    db.flush()
    db.add(Student(course_id=courses[1].id, name="One", student_number=1))
    db.add_all(
        [
            Student(course_id=courses[2].id, name="Two", student_number=1),
            Student(course_id=courses[2].id, name="Three", student_number=2),
            Assignment(course_id=courses[1].id, name="First"),
            Assignment(course_id=courses[2].id, name="Second"),
            Assignment(course_id=courses[2].id, name="Third"),
        ]
    )
    db.commit()

    selects = []

    def count_selects(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    engine_ = db.get_bind()
    event.listen(engine_, "before_cursor_execute", count_selects)
    try:
        response = client.get("/api/courses")
    finally:
        event.remove(engine_, "before_cursor_execute", count_selects)

    assert response.status_code == 200
    counts = {
        row["name"]: (row["student_count"], row["assignment_count"])
        for row in response.json()
    }
    assert counts == {
        "Course 0": (0, 0),
        "Course 1": (1, 1),
        "Course 2": (2, 2),
    }
    assert len(selects) == 1


def test_list_assignments_query_count_is_constant(client, db):
    course = Course(name="Query course")
    db.add(course)
    db.flush()
    assignments = [
        Assignment(course_id=course.id, name=f"Assignment {index}") for index in range(3)
    ]
    db.add_all(assignments)
    db.flush()
    db.add_all(
        [
            Submission(assignment_id=assignments[0].id, status="pending"),
            Submission(assignment_id=assignments[0].id, status="graded"),
            Submission(assignment_id=assignments[1].id, status="grading"),
            Submission(assignment_id=assignments[2].id, status="failed"),
            Submission(assignment_id=assignments[2].id, status="failed"),
        ]
    )
    db.commit()

    selects = []

    def count_selects(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    engine_ = db.get_bind()
    event.listen(engine_, "before_cursor_execute", count_selects)
    try:
        response = client.get(f"/api/courses/{course.id}/assignments")
    finally:
        event.remove(engine_, "before_cursor_execute", count_selects)

    assert response.status_code == 200
    assert [row["submission_counts"] for row in response.json()] == [
        {"pending": 1, "grading": 0, "graded": 1, "failed": 0},
        {"pending": 0, "grading": 1, "graded": 0, "failed": 0},
        {"pending": 0, "grading": 0, "graded": 0, "failed": 2},
    ]
    assert [row["submission_count"] for row in response.json()] == [2, 1, 2]
    assert len(selects) == 3


def test_confirm_mapping_query_count_is_constant(client, db):
    course = Course(name="Mapping course")
    db.add(course)
    db.flush()
    assignment = Assignment(course_id=course.id, name="Mapping assignment")
    students = [
        Student(course_id=course.id, name=f"Student {index}", student_number=index)
        for index in range(1, 5)
    ]
    db.add(assignment)
    db.add_all(students)
    db.flush()
    submissions = [Submission(assignment_id=assignment.id) for _ in students]
    db.add_all(submissions)
    db.commit()

    selects = []

    def count_selects(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    engine_ = db.get_bind()
    event.listen(engine_, "before_cursor_execute", count_selects)
    try:
        response = client.post(
            f"/api/assignments/{assignment.id}/submissions/mapping",
            json={
                "mapping": [
                    {"submission_id": submission.id, "student_id": student.id}
                    for submission, student in zip(submissions, students)
                ]
            },
        )
    finally:
        event.remove(engine_, "before_cursor_execute", count_selects)

    assert response.status_code == 200
    assert len(selects) == 3
    db.expire_all()
    persisted = db.scalars(select(Submission).order_by(Submission.id)).all()
    assert [submission.student_id for submission in persisted] == [
        student.id for student in students
    ]


def test_quick_stats_uses_one_select(db):
    def measured_stats():
        selects = []

        def count_selects(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                selects.append(statement)

        engine_ = db.get_bind()
        event.listen(engine_, "before_cursor_execute", count_selects)
        try:
            stats = courses_router._quick_stats(db)
        finally:
            event.remove(engine_, "before_cursor_execute", count_selects)
        assert len(selects) == 1
        return stats

    assert measured_stats() == {
        "courses": 0,
        "students": 0,
        "assignments": 0,
        "submissions": 0,
        "graded": 0,
        "pending": 0,
        "average_percent": None,
    }

    course = Course(name="Stats course")
    db.add(course)
    db.flush()
    db.add_all(
        [
            Student(course_id=course.id, name="One", student_number=1),
            Student(course_id=course.id, name="Two", student_number=2),
        ]
    )
    assignments = [
        Assignment(course_id=course.id, name="First"),
        Assignment(course_id=course.id, name="Second"),
    ]
    db.add_all(assignments)
    db.flush()
    submissions = [
        Submission(assignment_id=assignments[0].id, status="pending"),
        Submission(assignment_id=assignments[0].id, status="graded"),
        Submission(assignment_id=assignments[1].id, status="failed"),
        Submission(assignment_id=assignments[1].id, status="grading"),
    ]
    db.add_all(submissions)
    db.flush()
    db.add_all(
        [
            GradeResult(submission_id=submissions[1].id, overall_score=8, max_score=10),
            GradeResult(submission_id=submissions[2].id, overall_score=20, max_score=20),
            GradeResult(submission_id=submissions[3].id, overall_score=99, max_score=0),
        ]
    )
    db.commit()

    assert measured_stats() == {
        "courses": 1,
        "students": 2,
        "assignments": 2,
        "submissions": 4,
        "graded": 3,
        "pending": 1,
        "average_percent": 90.0,
    }


# --------------------------------------------------------------------------
# schema + rubric rendering
# --------------------------------------------------------------------------


def test_grade_schema_is_strict_and_complete():
    schema = engine.GRADE_SCHEMA
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "criteria",
        "summary_feedback",
        "misconceptions",
        "strengths",
    }
    item = schema["properties"]["criteria"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"key", "score", "comment"}
    # Structured outputs reject unknown keys, so max_points must NOT be asked for.
    assert set(item["properties"]) == {"key", "score", "comment"}


def test_rubric_text_round_trips_into_the_mock_provider():
    text = engine.render_rubric_text({"name": "Essay Rubric", "criteria": RUBRIC_CRITERIA})
    assert "- [thesis] Thesis (max 10 points)" in text
    assert "TOTAL: 25 points" in text

    parsed = MockProvider._criteria_from_prompt(text)
    assert [c["key"] for c in parsed] == ["thesis", "evidence", "mechanics"]
    assert [c["max_points"] for c in parsed] == [10.0, 10.0, 5.0]


# --------------------------------------------------------------------------
# MockProvider
# --------------------------------------------------------------------------


def test_mock_provider_is_deterministic_and_schema_valid():
    system = engine.render_rubric_text({"criteria": RUBRIC_CRITERIA})
    blocks = [providers_mod.document_block(PDF_BYTES, filename="essay_student_01.pdf")]

    first = MockProvider().grade(system, blocks, engine.GRADE_SCHEMA)
    again = MockProvider().grade(system, blocks, engine.GRADE_SCHEMA)
    assert first == again

    assert set(first) == {"criteria", "summary_feedback", "misconceptions", "strengths"}
    assert [c["key"] for c in first["criteria"]] == ["thesis", "evidence", "mechanics"]
    for crit, spec in zip(first["criteria"], RUBRIC_CRITERIA):
        assert set(crit) == {"key", "score", "comment"}
        assert 0 <= crit["score"] <= spec["max_points"]
        assert crit["comment"]
    assert isinstance(first["summary_feedback"], str) and first["summary_feedback"]
    assert all(isinstance(t, str) for t in first["misconceptions"])
    assert all(t == t.lower() for t in first["misconceptions"])

    other = MockProvider().grade(
        system,
        [providers_mod.document_block(PDF_BYTES, filename="essay_student_09.pdf")],
        engine.GRADE_SCHEMA,
    )
    assert other["criteria"] != first["criteria"], "scores must vary with the filename"


def test_mock_provider_accepts_the_seed_helper_signature():
    """app.seed calls MockProvider.grade(system_prompt=, rubric=, submission=)."""
    result = MockProvider().grade(
        system_prompt="Grade this.",
        rubric={"criteria": RUBRIC_CRITERIA},
        submission={"filename": "ps1_student_04.pdf", "mime_type": "application/pdf"},
    )
    assert [c["key"] for c in result["criteria"]] == ["thesis", "evidence", "mechanics"]
    assert result["summary_feedback"]


def test_mock_and_seed_fallback_share_criterion_comment_catalog(monkeypatch):
    from app import seed as seed_mod
    from app.ai.mock_content import MOCK_CRITERION_COMMENTS

    fractions = {
        "strong": 1.0,
        "solid": 0.6,
        "developing": 0.45,
        "weak": 0.0,
    }
    for band, fraction in fractions.items():
        monkeypatch.setattr(
            providers_mod, "_fraction", lambda *_args, value=fraction: value
        )
        monkeypatch.setattr(
            seed_mod, "_hash_fraction", lambda *_args, value=fraction: value
        )
        for key in MOCK_CRITERION_COMMENTS:
            criterion = {
                "key": key,
                "title": key.title(),
                "description": "",
                "max_points": 10,
            }
            provider_grade = MockProvider().grade(
                system_prompt="Grade this.",
                rubric={"criteria": [criterion]},
                submission={"filename": "essay.pdf"},
            )
            seed_grade = seed_mod._fallback_grade("essay.pdf", [criterion], ["tag"])

            expected = MOCK_CRITERION_COMMENTS[key][band]
            assert provider_grade["criteria"][0]["comment"] == expected
            assert seed_grade["criteria"][0]["comment"] == expected


def test_mock_provider_chat_emits_parsed_tool_calls():
    tools = [{"name": name, "description": "", "input_schema": {}} for name in
             providers_mod.MOCK_CHAT_TOOLS]

    turn = MockProvider().chat("You are Agora.", [{"role": "user", "content": "grade assignment 3"}], tools)
    assert isinstance(turn, ChatTurn)
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].name == "start_grading"
    assert turn.tool_calls[0].arguments == {"assignment_id": 3}

    nav = MockProvider().chat("", [{"role": "user", "content": "go to the analytics page"}], tools)
    assert nav.tool_calls[0].name == "navigate_to"
    assert nav.tool_calls[0].arguments["page"] == "analytics"

    followup = MockProvider().chat(
        "",
        [
            {"role": "user", "content": "grade assignment 3"},
            providers_mod.tool_result_message(
                [{"tool_use_id": "mock_tool_1", "content": "{\"queued\": 3}"}]
            ),
        ],
        tools,
    )
    assert followup.stop_reason == "end_turn"
    assert followup.tool_calls == []
    assert followup.text


def test_mock_chat_uses_context_course_and_roster_number():
    tools = [
        {"name": name, "description": "", "input_schema": {}}
        for name in providers_mod.MOCK_CHAT_TOOLS
    ]
    messages = [{"role": "user", "content": "Show me student #5"}]

    turn = MockProvider().chat(
        'Course in context: id=22, name="Course B"', messages, tools
    )
    assert turn.tool_calls[0].name == "get_student_summary"
    assert turn.tool_calls[0].arguments == {"course_id": 22, "student_number": 5}

    no_context = MockProvider().chat("Current page: home", messages, tools)
    assert no_context.text == "Open a course before asking about Student #5."
    assert no_context.tool_calls == []


def test_mock_chat_resolves_named_course_without_page_context():
    tools = [
        {"name": name, "description": "", "input_schema": {}}
        for name in providers_mod.MOCK_CHAT_TOOLS
    ]
    prompt = "\n".join(
        [
            'Known course: id=7, name="PHIL 210 · Ethics of Emerging Technology"',
            'Known course: id=8, name="HIST 101 · World History"',
            'Known course: id=10, name="ART 1 · Drawing Foundations"',
            'Known course: id=11, name="ART 10 · Studio Practice"',
        ]
    )
    turn = MockProvider().chat(
        prompt,
        [{"role": "user", "content": "Which students are struggling most in phil 210?"}],
        tools,
    )
    assert turn.tool_calls[0].name == "get_course_summary"
    assert turn.tool_calls[0].arguments == {"course_id": 7}
    assert turn.tool_calls[0].arguments["course_id"] != 210

    context_prompt = "\n".join(
        ['Course in context: id=7, name="PHIL 210"', prompt]
    )
    explicit_other_course = MockProvider().chat(
        context_prompt,
        [{"role": "user", "content": "How is HIST 101 doing?"}],
        tools,
    )
    assert explicit_other_course.tool_calls[0].name == "get_course_summary"
    assert explicit_other_course.tool_calls[0].arguments == {"course_id": 8}

    context_course = MockProvider().chat(
        context_prompt,
        [{"role": "user", "content": "How is this course doing?"}],
        tools,
    )
    assert context_course.tool_calls[0].name == "get_course_summary"
    assert context_course.tool_calls[0].arguments == {"course_id": 7}

    unmatched = MockProvider().chat(
        prompt,
        [{"role": "user", "content": "How is BIO 404 doing?"}],
        tools,
    )
    assert unmatched.text == "Open a course before asking for a course summary."
    assert unmatched.tool_calls == []

    art_ten = MockProvider().chat(
        prompt,
        [{"role": "user", "content": "How is ART 10 doing?"}],
        tools,
    )
    assert art_ten.tool_calls[0].name == "get_course_summary"
    assert art_ten.tool_calls[0].arguments == {"course_id": 11}

    absent_art = MockProvider().chat(
        prompt,
        [{"role": "user", "content": "How is ART 100 doing?"}],
        tools,
    )
    assert absent_art.text == "Open a course before asking for a course summary."
    assert absent_art.tool_calls == []

    ambiguous = MockProvider().chat(
        prompt
        + '\nKnown course: id=9, name="PHIL 210 · Political Philosophy"',
        [{"role": "user", "content": "How is PHIL 210 doing?"}],
        tools,
    )
    assert ambiguous.text == "Open a course before asking for a course summary."
    assert ambiguous.tool_calls == []


def test_mock_chat_renders_successful_and_failed_tool_results():
    success = providers_mod.tool_result_message(
        [
            {
                "tool_use_id": "mock_tool_1",
                "content": json.dumps(
                    {
                        "label": "Student #5",
                        "average_percent": 63.5,
                        "best_percent": 70.0,
                        "worst_percent": 57.1,
                        "misconceptions": [
                            "treats correlation as causation",
                            "cites source without engaging it",
                        ],
                    }
                ),
                "is_error": False,
            }
        ]
    )
    success_turn = MockProvider().chat("", [success], [])
    for fact in (
        "Student #5",
        "63.5%",
        "70.0%",
        "57.1%",
        "treats correlation as causation",
        "cites source without engaging it",
    ):
        assert fact in success_turn.text

    course_success = providers_mod.tool_result_message(
        [
            {
                "tool_use_id": "mock_tool_1",
                "content": json.dumps(
                    {
                        "name": "PHIL 210",
                        "student_count": 2,
                        "assignments": [
                            {
                                "name": "Essay",
                                "submissions": 4,
                                "graded": 1,
                                "pending": 1,
                                "grading": 1,
                                "failed": 1,
                                "ungraded": 3,
                            }
                        ],
                        "students_needing_attention": [
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
                        ],
                    }
                ),
                "is_error": False,
            }
        ]
    )
    course_turn = MockProvider().chat("", [course_success], [])
    attention_start = course_turn.text.index("Lowest graded averages:")
    assert "Essay: 1 graded of 4 submissions" in course_turn.text
    assert "1 pending" in course_turn.text
    assert "1 grading" in course_turn.text
    assert "1 failed" in course_turn.text
    assert "3 ungraded" not in course_turn.text
    student_two = course_turn.text.index("Student #2 50.0%", attention_start)
    student_one = course_turn.text.index("Student #1 75.0%", attention_start)
    assert attention_start < student_two < student_one

    failure = providers_mod.tool_result_message(
        [
            {
                "tool_use_id": "mock_tool_1",
                "content": json.dumps({"error": "No course with id 210."}),
                "is_error": True,
            }
        ]
    )
    failure_turn = MockProvider().chat("", [failure], [])
    assert failure_turn.text.startswith("I couldn't retrieve that data:")
    assert "No course with id 210." in failure_turn.text
    assert "Here is what I found from the course data." not in failure_turn.text


# --------------------------------------------------------------------------
# validation / clamping
# --------------------------------------------------------------------------


def test_validate_clamps_scores_and_flags_anomalies():
    payload = {
        "criteria": [
            {"key": "thesis", "score": 99, "comment": "Way over."},
            {"key": "evidence", "score": -4, "comment": "Negative."},
            {"key": "gibberish", "score": 5, "comment": "Not in the rubric."},
        ],
        "summary_feedback": "  Overall good.  ",
        "misconceptions": ["Conflates Legality With Morality", "conflates legality with morality"],
        "strengths": ["Clear thesis", "Clear thesis"],
    }
    validated = engine.validate_grade_payload(payload, RUBRIC_CRITERIA)

    scores = {c["key"]: c["score"] for c in validated.criteria}
    assert scores["thesis"] == 10.0  # clamped to max_points
    assert scores["evidence"] == 0.0  # clamped to zero
    assert scores["mechanics"] == 0.0  # skipped by the model -> filled in
    assert [c["key"] for c in validated.criteria] == ["thesis", "evidence", "mechanics"]
    assert all(c["max_points"] for c in validated.criteria)

    assert validated.overall_score == 10.0
    assert validated.max_score == 25.0
    assert validated.percentage == 40.0
    assert validated.summary_feedback == "Overall good."
    assert validated.misconceptions == ["conflates legality with morality"]
    assert validated.strengths == ["Clear thesis"]

    joined = " | ".join(validated.anomalies)
    assert "clamped" in joined
    assert "gibberish" in joined
    assert "mechanics" in joined


def test_validate_handles_dupes_and_junk_scores():
    payload = {
        "criteria": [
            {"key": "Thesis", "score": "8 out of 10", "comment": "Fine."},
            {"key": "thesis", "score": 3, "comment": "Duplicate."},
            {"key": "evidence", "score": "not a number", "comment": ""},
            "garbage",
        ],
        "summary_feedback": "",
        "misconceptions": ["  MIXED   case  ", "", None, {"tag": "nested tag"}],
        "strengths": "single strength",
    }
    validated = engine.validate_grade_payload(payload, RUBRIC_CRITERIA, label="Student #2")

    scores = {c["key"]: c["score"] for c in validated.criteria}
    assert scores["thesis"] == 8.0  # case-insensitive key match, number extracted
    assert scores["evidence"] == 0.0
    assert validated.misconceptions == ["mixed case", "nested tag"]
    assert validated.strengths == ["single strength"]
    joined = " | ".join(validated.anomalies)
    assert "twice" in joined
    assert "non-numeric" in joined
    assert "summary" in joined


def test_validate_rejects_non_object_payloads():
    with pytest.raises(ProviderResponseError):
        engine.validate_grade_payload(["nope"], RUBRIC_CRITERIA)
    with pytest.raises(ProviderResponseError):
        engine.validate_grade_payload({"summary_feedback": "x"}, RUBRIC_CRITERIA)


def test_normalize_tags_dedupes_and_caps():
    tags = engine.normalize_tags(["A", "a", "  b  ", "B.", "c"] + [f"tag{i}" for i in range(20)])
    assert tags[:3] == ["a", "b", "c"]
    assert len(tags) == engine.MAX_MISCONCEPTIONS


# --------------------------------------------------------------------------
# request building / anonymization
# --------------------------------------------------------------------------


def test_build_grade_request_is_anonymized(client, db, course_setup):
    assignment_id = course_setup["assignment"]["id"]
    resp = upload(client, assignment_id, [("amara_osei.pdf", PDF_BYTES, "application/pdf")])
    assert resp.status_code == 201, resp.text

    submission = db.scalars(select(Submission).order_by(Submission.id)).first()
    request = engine.build_grade_request(db, submission)

    prompt_and_text = request.system_prompt + "\n".join(
        b.get("text", "") for b in request.content_blocks if b.get("type") == "text"
    )
    assert "Amara" not in prompt_and_text
    assert "Osei" not in prompt_and_text
    assert "Student #1" in prompt_and_text
    assert "Essay 1" in prompt_and_text
    assert "- [thesis] Thesis (max 10 points)" in request.system_prompt

    # Document block precedes the text block (AI_NOTES) and carries no private
    # keys once rendered for the API.
    assert request.content_blocks[0]["type"] == "document"
    assert request.content_blocks[-1]["type"] == "text"
    public = providers_mod.public_blocks(request.content_blocks)
    assert not any(k.startswith("_") for block in public for k in block)
    assert request.provider == config.MOCK_PROVIDER
    assert request.max_score == 25.0


def _text_pdf(text: str = "The student essay body.") -> bytes:
    """A minimal single-page PDF with a real text layer."""
    content = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
        b"<< /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += str(index).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    start = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += ("%010d 00000 n \n" % offset).encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n"
        + str(start).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def test_extract_pdf_text_reads_a_text_layer_and_survives_junk():
    assert "student essay body" in engine.extract_pdf_text(_text_pdf())
    # A scan (or a corrupt file) yields nothing rather than raising.
    assert engine.extract_pdf_text(b"%PDF-1.4 not really a pdf") == ""


def test_openai_pdf_submission_degrades_to_extracted_text(
    client, db, course_setup, tmp_path, monkeypatch
):
    """A PDF + an OpenAI skill must still grade — AI_NOTES: degrade gracefully.

    Privacy mode is pinned to ``off`` so this exercises the engine's own
    degrade path. Under the default ``swap`` mode the Privacy Guard replaces
    the document block with pseudonymized extracted text for every cloud call
    — that behaviour is covered in tests/test_privacy.py.
    """
    data_dir = tmp_path / "engine-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    config.save_privacy_settings({"mode": config.PRIVACY_MODE_OFF})

    assignment_id = course_setup["assignment"]["id"]
    upload(client, assignment_id, [("student_01.pdf", _text_pdf(), "application/pdf")])

    skill = db.get(Skill, course_setup["skill_id"])
    skill.provider = "openai"
    skill.model = "gpt-5.6-sol"
    skill.max_tokens = 0  # unset -> the registry's budget for that model
    db.commit()

    submission = db.scalars(select(Submission).order_by(Submission.id)).first()
    request = engine.build_grade_request(db, submission)
    assert request.provider == "openai"
    # Registry-sized budget, and a text fallback the provider can fall back to.
    assert request.max_tokens == config.max_tokens_for("openai", "gpt-5.6-sol")
    document = request.content_blocks[0]
    assert document["type"] == "document"
    assert "student essay body" in document["_text"]
    assert any("extracted" in note.lower() for note in request.notes)

    # The provider now produces text parts instead of raising Unsupported.
    provider = OpenAIProvider(model="gpt-5.6-sol", api_key="k", client=object())
    parts = provider._content_parts(request.content_blocks)
    assert any(p["type"] == "text" and "student essay body" in p["text"] for p in parts)


# --------------------------------------------------------------------------
# upload safety
# --------------------------------------------------------------------------


def test_sanitize_filename_blocks_traversal():
    assert grading_router.sanitize_filename("../../../etc/passwd") == "passwd"
    assert grading_router.sanitize_filename("..%2f..%2fboot.pdf").endswith("boot.pdf")
    assert grading_router.sanitize_filename("C:\\Users\\prof\\essay 1.pdf") == "essay_1.pdf"
    assert grading_router.sanitize_filename("") == "submission"
    assert grading_router.sanitize_filename("...") == "submission"
    assert "/" not in grading_router.sanitize_filename("a/b/c.pdf")


def test_detect_media_type_sniffs_magic_bytes():
    assert grading_router.detect_media_type(PDF_BYTES, "text/plain") == "application/pdf"
    assert grading_router.detect_media_type(PNG_BYTES, None) == "image/png"
    assert grading_router.detect_media_type(b"\xff\xd8\xff\xe0abc", None) == "image/jpeg"
    # A .pdf name with non-PDF content is not trusted.
    assert grading_router.detect_media_type(b"just text", "application/pdf") is None


def test_upload_validates_types_and_stores_safely(client, db, course_setup, tmp_path):
    assignment_id = course_setup["assignment"]["id"]
    resp = upload(
        client,
        assignment_id,
        [
            ("../../../etc/passwd.pdf", PDF_BYTES, "application/pdf"),
            ("notes.txt", b"plain text, not a submission", "text/plain"),
            ("scan.png", PNG_BYTES, "image/png"),
            ("empty.pdf", b"", "application/pdf"),
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()

    assert [r["filename"] for r in body["rejected"]] == ["notes.txt", "empty.pdf"]
    assert any("unsupported" in r["reason"] for r in body["rejected"])
    assert [u["filename"] for u in body["uploaded"]] == ["passwd.pdf", "scan.png"]

    upload_root = (tmp_path / "submissions" / str(assignment_id)).resolve()
    for item in body["uploaded"]:
        stored = item["stored_path"]
        assert str(upload_root) == str(__import__("pathlib").Path(stored).resolve().parent)
    assert (upload_root / "passwd.pdf").exists()


def test_upload_suggests_and_prefills_student_mapping(client, db, course_setup):
    assignment_id = course_setup["assignment"]["id"]
    students = {s["name"]: s for s in course_setup["students"]}

    resp = upload(
        client,
        assignment_id,
        [
            ("ps1_student_02.pdf", PDF_BYTES, "application/pdf"),
            ("amara-osei-essay.pdf", PDF_BYTES, "application/pdf"),
            ("scan9987.pdf", PDF_BYTES, "application/pdf"),
        ],
    )
    body = resp.json()
    by_file = {u["filename"]: u for u in body["uploaded"]}

    assert by_file["ps1_student_02.pdf"]["student_id"] == students["Ben Whitaker"]["id"]
    assert by_file["ps1_student_02.pdf"]["auto_assigned"] is True
    assert by_file["amara-osei-essay.pdf"]["student_id"] == students["Amara Osei"]["id"]
    unmatched = by_file["scan9987.pdf"]
    assert unmatched["student_id"] is None
    assert unmatched["needs_confirmation"] is True
    assert body["needs_confirmation"] == 1

    # The professor confirms the leftover file.
    confirm = client.post(
        f"/api/assignments/{assignment_id}/submissions/mapping",
        json={
            "mapping": [
                {
                    "submission_id": unmatched["submission_id"],
                    "student_id": students["Claudia Moreno"]["id"],
                }
            ]
        },
    )
    assert confirm.status_code == 200, confirm.text
    assert confirm.json()["unmapped"] == 0

    # Two files may not point at the same student.
    clash = client.post(
        f"/api/assignments/{assignment_id}/submissions/mapping",
        json={
            "mapping": [
                {
                    "submission_id": unmatched["submission_id"],
                    "student_id": students["Amara Osei"]["id"],
                }
            ]
        },
    )
    assert clash.status_code == 409


def test_upload_honours_the_confirmed_mapping_from_the_ui(client, course_setup):
    """static/js/grading.js posts files + parallel student_ids + a mapping JSON."""
    assignment_id = course_setup["assignment"]["id"]
    students = {s["name"]: s for s in course_setup["students"]}

    resp = client.post(
        f"/api/assignments/{assignment_id}/submissions",
        files=[
            ("files", ("ps1_student_02.pdf", io.BytesIO(PDF_BYTES), "application/pdf")),
            ("files", ("junk.txt", io.BytesIO(b"not a pdf"), "text/plain")),
            ("files", ("anonymous.pdf", io.BytesIO(PDF_BYTES), "application/pdf")),
        ],
        data={
            # Parallel to `files` — index 1 is the rejected text file.
            "student_ids": [str(students["Claudia Moreno"]["id"]), "", ""],
            "mapping": json.dumps(
                [{"filename": "anonymous.pdf", "student_id": students["Amara Osei"]["id"]}]
            ),
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    by_file = {u["filename"]: u for u in body["uploaded"]}

    # The professor's choice beats the "student_02" filename heuristic.
    assert by_file["ps1_student_02.pdf"]["student_id"] == students["Claudia Moreno"]["id"]
    assert by_file["ps1_student_02.pdf"]["confirmed"] is True
    # The JSON mapping covers files the parallel array left blank.
    assert by_file["anonymous.pdf"]["student_id"] == students["Amara Osei"]["id"]
    assert body["needs_confirmation"] == 0
    assert body["rejected"][0]["filename"] == "junk.txt"

    # A second upload may not steal a student who already has a submission.
    again = client.post(
        f"/api/assignments/{assignment_id}/submissions",
        files=[("files", ("late.pdf", io.BytesIO(PDF_BYTES), "application/pdf"))],
        data={"student_ids": [str(students["Amara Osei"]["id"])]},
    ).json()
    assert again["uploaded"][0]["student_id"] is None
    assert any("already has a submission" in w for w in again["warnings"])


def test_suggest_student_reasons(db, course_setup, client):
    from app.models import Student

    students = list(db.scalars(select(Student).order_by(Student.student_number)).all())

    match, confidence, reason = grading_router.suggest_student("hw3_student_03.pdf", students)
    assert match.student_number == 3 and confidence > 0.9 and "student #3" in reason

    match, confidence, _ = grading_router.suggest_student("whitaker_ben.pdf", students)
    assert match.name == "Ben Whitaker" and confidence >= 0.9

    match, confidence, reason = grading_router.suggest_student("untitled.pdf", students)
    assert match is None and reason == "no confident match"


# --------------------------------------------------------------------------
# end-to-end grading with MockProvider
# --------------------------------------------------------------------------


def test_grade_all_end_to_end_with_mock_provider(client, db, course_setup):
    assignment_id = course_setup["assignment"]["id"]
    resp = upload(
        client,
        assignment_id,
        [
            ("essay1_student_01.pdf", PDF_BYTES, "application/pdf"),
            ("essay1_student_02.pdf", PDF_BYTES, "application/pdf"),
            ("essay1_student_03.pdf", PDF_BYTES, "application/pdf"),
        ],
    )
    uploaded = resp.json()["uploaded"]
    assert all(u["auto_assigned"] for u in uploaded)

    status = client.get(f"/api/assignments/{assignment_id}/grading/status").json()
    assert status["counts"]["pending"] == 3
    assert status["in_progress"] is False

    run = client.post(f"/api/assignments/{assignment_id}/grade-all", json={})
    assert run.status_code == 200, run.text
    assert run.json()["queued"] == 3

    # TestClient runs BackgroundTasks before returning, so the run is complete.
    status = client.get(f"/api/assignments/{assignment_id}/grading/status").json()
    assert status["counts"]["graded"] == 3
    assert status["counts"]["failed"] == 0
    assert status["percent_complete"] == 100.0
    assert status["in_progress"] is False
    assert all(row["error"] is None for row in status["submissions"])

    results = client.get(f"/api/assignments/{assignment_id}/results").json()
    assert results["counts"]["graded"] == 3
    assert 0 < results["average_percent"] <= 100
    percentages = set()
    for row in results["results"]:
        result = row["result"]
        assert result is not None
        assert row["status"] == "graded"
        assert result["model"] == config.MOCK_MODEL
        assert result["max_score"] == 25.0
        assert 0 <= result["overall_score"] <= 25.0
        assert [c["key"] for c in result["criteria"]] == ["thesis", "evidence", "mechanics"]
        for crit in result["criteria"]:
            assert crit["score"] <= crit["max_points"]
        assert all(tag == tag.lower() for tag in result["misconceptions"])
        assert len(set(result["misconceptions"])) == len(result["misconceptions"])
        assert result["summary_feedback"]
        percentages.add(result["percentage"])
    assert len(percentages) > 1, "hash-derived scores must vary across submissions"

    # Rows really landed in the DB, one result per submission.
    assert db.query(GradeResult).count() == 3

    # Re-running without force is a no-op; forcing re-grades deterministically.
    again = client.post(f"/api/assignments/{assignment_id}/grade-all", json={}).json()
    assert again["queued"] == 0
    assert len(again["skipped"]) == 3

    before = client.get(f"/api/assignments/{assignment_id}/results").json()
    forced = client.post(
        f"/api/assignments/{assignment_id}/grade-all", json={"force": True}
    ).json()
    assert forced["queued"] == 3
    after = client.get(f"/api/assignments/{assignment_id}/results").json()
    assert [r["result"]["overall_score"] for r in after["results"]] == [
        r["result"]["overall_score"] for r in before["results"]
    ]
    assert db.query(GradeResult).count() == 3, "re-grading replaces, never duplicates"


def test_grade_one_and_failure_states(client, db, course_setup):
    assignment_id = course_setup["assignment"]["id"]
    body = upload(
        client,
        assignment_id,
        [
            ("essay1_student_01.pdf", PDF_BYTES, "application/pdf"),
            ("mystery-scan.pdf", PDF_BYTES, "application/pdf"),
        ],
    ).json()
    mapped = next(u for u in body["uploaded"] if u["auto_assigned"])
    unmapped = next(u for u in body["uploaded"] if not u["auto_assigned"])

    # Unmapped submissions cannot be graded (anonymization needs the number).
    blocked = client.post(f"/api/submissions/{unmapped['submission_id']}/grade")
    assert blocked.status_code == 409

    ok = client.post(f"/api/submissions/{mapped['submission_id']}/grade")
    assert ok.status_code == 200
    single = client.get(f"/api/submissions/{mapped['submission_id']}/result").json()
    assert single["status"] == "graded"
    assert single["result"]["overall_score"] > 0

    # A missing file on disk fails that submission, with a readable error.
    submission = db.get(Submission, unmapped["submission_id"])
    submission.student_id = course_setup["students"][2]["id"]
    db.commit()
    from pathlib import Path

    Path(submission.file_path).unlink()
    resp = client.post(f"/api/submissions/{submission.id}/grade")
    assert resp.status_code == 200
    status = client.get(f"/api/assignments/{assignment_id}/grading/status").json()
    failed = [row for row in status["submissions"] if row["status"] == "failed"]
    assert len(failed) == 1
    assert "missing" in failed[0]["error"].lower()
    assert status["counts"]["graded"] == 1

    assert client.get("/api/assignments/99999/grading/status").status_code == 404
    assert client.post("/api/submissions/99999/grade").status_code == 404


def test_stuck_grading_rows_can_be_recovered(client, db, course_setup):
    assignment_id = course_setup["assignment"]["id"]
    body = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    ).json()
    submission_id = body["uploaded"][0]["submission_id"]

    # Simulate a crash mid-run: the row is stranded in `grading`.
    submission = db.get(Submission, submission_id)
    submission.status = "grading"
    db.commit()

    assert client.post(f"/api/submissions/{submission_id}/grade").status_code == 409
    reset = client.post(f"/api/assignments/{assignment_id}/grading/reset").json()
    assert reset["reset"] == [submission_id]
    assert client.post(f"/api/submissions/{submission_id}/grade").status_code == 200
    assert client.get(f"/api/submissions/{submission_id}/result").json()["status"] == "graded"


def test_professor_can_edit_a_returned_grade(client, db, course_setup):
    """grading.html's "Save feedback" form PATCHes the result endpoint."""
    assignment_id = course_setup["assignment"]["id"]
    body = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    ).json()
    submission_id = body["uploaded"][0]["submission_id"]
    assert client.post(f"/api/submissions/{submission_id}/grade").status_code == 200

    before = client.get(f"/api/submissions/{submission_id}/result").json()["result"]
    edited = "Rewritten in my own voice: the thesis is sharper than the score suggests."

    resp = client.patch(
        f"/api/submissions/{submission_id}/result",
        json={
            "summary_feedback": edited,
            "criteria": [{"key": "thesis", "score": 10, "comment": "Full marks on reflection."}],
        },
    )
    assert resp.status_code == 200, resp.text

    after = client.get(f"/api/submissions/{submission_id}/result").json()["result"]
    assert after["summary_feedback"] == edited
    thesis = next(c for c in after["criteria"] if c["key"] == "thesis")
    assert thesis["score"] == 10.0
    assert thesis["comment"] == "Full marks on reflection."
    # The overall score follows the edit; untouched criteria are preserved.
    assert after["overall_score"] != before["overall_score"]
    assert after["overall_score"] == sum(c["score"] for c in after["criteria"])
    assert len(after["criteria"]) == len(before["criteria"])

    # Scores are clamped to the rubric maximum, and unknown ids 404.
    client.patch(
        f"/api/submissions/{submission_id}/result",
        json={"criteria": [{"key": "thesis", "score": 999}]},
    )
    capped = client.get(f"/api/submissions/{submission_id}/result").json()["result"]
    assert next(c for c in capped["criteria"] if c["key"] == "thesis")["score"] == 10.0
    assert client.patch("/api/submissions/99999/result", json={}).status_code == 404


def test_submission_file_download_and_delete(client, db, course_setup):
    assignment_id = course_setup["assignment"]["id"]
    body = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    ).json()
    submission_id = body["uploaded"][0]["submission_id"]

    download = client.get(f"/api/submissions/{submission_id}/file")
    assert download.status_code == 200
    assert download.content == PDF_BYTES

    assert client.delete(f"/api/submissions/{submission_id}").status_code == 200
    assert client.get(f"/api/submissions/{submission_id}/file").status_code == 404
    assert db.query(Submission).count() == 0


def test_delete_assignment_file_cleanup_removes_submission_directory(
    client, course_setup, tmp_path
):
    assignment_id = course_setup["assignment"]["id"]
    uploaded = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    )
    assert uploaded.status_code == 201, uploaded.text
    submission_directory = tmp_path / "submissions" / str(assignment_id)
    assert submission_directory.is_dir()

    deleted = client.delete(f"/api/assignments/{assignment_id}")

    assert deleted.status_code == 200, deleted.text
    assert not submission_directory.exists()
    assert client.get(f"/api/assignments/{assignment_id}").status_code == 404
    assert_no_delete_backups(tmp_path / "submissions")


def test_delete_assignment_cleanup_failure_preserves_row(
    client, course_setup, tmp_path, monkeypatch
):
    assignment_id = course_setup["assignment"]["id"]
    uploaded = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    )
    assert uploaded.status_code == 201, uploaded.text
    submission_directory = tmp_path / "submissions" / str(assignment_id)

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(grading_router, "reversible_delete", fail_cleanup)
    with pytest.raises(OSError, match="cleanup failed"):
        client.delete(f"/api/assignments/{assignment_id}")

    assert client.get(f"/api/assignments/{assignment_id}").status_code == 200
    assert submission_directory.is_dir()
    assert_no_delete_backups(tmp_path / "submissions")


def test_delete_submission_removes_file(client, course_setup, tmp_path):
    assignment_id = course_setup["assignment"]["id"]
    body = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    ).json()
    submission_id = body["uploaded"][0]["submission_id"]
    stored = tmp_path / "submissions" / str(assignment_id) / "essay1_student_01.pdf"
    assert stored.is_file()

    deleted = client.delete(f"/api/submissions/{submission_id}")

    assert deleted.status_code == 200, deleted.text
    assert not stored.exists()
    assert client.get(f"/api/submissions/{submission_id}/file").status_code == 404
    assert_no_delete_backups(tmp_path / "submissions")


def test_delete_submission_cleanup_failure_preserves_row(
    client, db, course_setup, tmp_path, monkeypatch
):
    assignment_id = course_setup["assignment"]["id"]
    body = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    ).json()
    submission_id = body["uploaded"][0]["submission_id"]
    stored = tmp_path / "submissions" / str(assignment_id) / "essay1_student_01.pdf"

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(grading_router, "reversible_delete", fail_cleanup)
    with pytest.raises(OSError, match="cleanup failed"):
        client.delete(f"/api/submissions/{submission_id}")

    db.expire_all()
    assert db.get(Submission, submission_id) is not None
    assert stored.is_file()
    assert_no_delete_backups(tmp_path / "submissions")


def test_deleting_course_removes_assignment_submission_directory(
    client, course_setup, tmp_path
):
    course_id = course_setup["course"]["id"]
    assignment_id = course_setup["assignment"]["id"]
    uploaded = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    )
    assert uploaded.status_code == 201, uploaded.text

    submission_directory = tmp_path / "submissions" / str(assignment_id)
    assert submission_directory.is_dir()
    assert any(submission_directory.iterdir())

    deleted = client.delete(f"/api/courses/{course_id}")

    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": course_id}
    assert not submission_directory.exists()
    assert_no_delete_backups(tmp_path / "submissions")


def test_course_delete_cleanup_failure_keeps_database_rows(
    client, course_setup, tmp_path, monkeypatch
):
    course_id = course_setup["course"]["id"]
    assignment_id = course_setup["assignment"]["id"]
    body = upload(
        client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")]
    ).json()
    stored = tmp_path / "submissions" / str(assignment_id) / "essay1_student_01.pdf"

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(courses_router, "reversible_delete", fail_cleanup)
    with pytest.raises(OSError, match="cleanup failed"):
        client.delete(f"/api/courses/{course_id}")

    course = client.get(f"/api/courses/{course_id}")
    assert course.status_code == 200
    assert [item["id"] for item in course.json()["assignments"]] == [assignment_id]
    assert stored.is_file()
    assert_no_delete_backups(tmp_path / "submissions")


def test_course_delete_second_cleanup_failure_restores_first_directory(
    client, db, course_setup, tmp_path, monkeypatch
):
    course_id = course_setup["course"]["id"]
    first_id = course_setup["assignment"]["id"]
    second = client.post(
        f"/api/courses/{course_id}/assignments",
        json={
            "name": "Essay 2",
            "rubric_id": course_setup["rubric"]["id"],
            "skill_id": course_setup["skill_id"],
        },
    ).json()
    second_id = second["id"]
    upload(client, first_id, [("first.pdf", PDF_BYTES, "application/pdf")])
    upload(client, second_id, [("second.pdf", PDF_BYTES + b"second", "application/pdf")])
    first_file = tmp_path / "submissions" / str(first_id) / "first.pdf"
    second_file = tmp_path / "submissions" / str(second_id) / "second.pdf"
    first_bytes = first_file.read_bytes()
    second_bytes = second_file.read_bytes()

    real_delete = courses_router.reversible_delete
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("cleanup failed")
        return real_delete(*args, **kwargs)

    monkeypatch.setattr(courses_router, "reversible_delete", fail_second)
    with pytest.raises(OSError, match="cleanup failed"):
        client.delete(f"/api/courses/{course_id}")

    db.expire_all()
    assert db.get(Course, course_id) is not None
    assert db.get(Assignment, first_id) is not None
    assert db.get(Assignment, second_id) is not None
    assert first_file.read_bytes() == first_bytes
    assert second_file.read_bytes() == second_bytes
    assert_no_delete_backups(tmp_path / "submissions")


def test_course_delete_commit_failure_restores_directories_and_rows(
    client, db, course_setup, tmp_path, monkeypatch
):
    course_id = course_setup["course"]["id"]
    first_id = course_setup["assignment"]["id"]
    second = client.post(
        f"/api/courses/{course_id}/assignments",
        json={"name": "Essay 2", "skill_id": course_setup["skill_id"]},
    ).json()
    second_id = second["id"]
    upload(client, first_id, [("first.pdf", PDF_BYTES, "application/pdf")])
    upload(client, second_id, [("second.pdf", PDF_BYTES + b"second", "application/pdf")])
    files = [
        tmp_path / "submissions" / str(first_id) / "first.pdf",
        tmp_path / "submissions" / str(second_id) / "second.pdf",
    ]
    contents = [path.read_bytes() for path in files]

    previous_dependency = app.dependency_overrides[get_db]
    original_commit = db.commit

    def fail_commit():
        raise RuntimeError("commit failed")

    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(db, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            client.delete(f"/api/courses/{course_id}")
    finally:
        monkeypatch.setattr(db, "commit", original_commit)
        app.dependency_overrides[get_db] = previous_dependency

    db.expire_all()
    assert db.get(Course, course_id) is not None
    assert db.get(Assignment, first_id) is not None
    assert db.get(Assignment, second_id) is not None
    assert [path.read_bytes() for path in files] == contents
    assert_no_delete_backups(tmp_path / "submissions")


def test_course_delete_restore_failure_does_not_skip_other_directories(
    client, db, course_setup, tmp_path, monkeypatch
):
    course_id = course_setup["course"]["id"]
    assignment_ids = [course_setup["assignment"]["id"]]
    for name in ("Essay 2", "Essay 3"):
        assignment = client.post(
            f"/api/courses/{course_id}/assignments",
            json={"name": name, "skill_id": course_setup["skill_id"]},
        ).json()
        assignment_ids.append(assignment["id"])

    directories = {}
    for index, assignment_id in enumerate(assignment_ids, start=1):
        filename = f"essay-{index}.pdf"
        content = PDF_BYTES + f"assignment-{index}".encode()
        upload(client, assignment_id, [(filename, content, "application/pdf")])
        directory = tmp_path / "submissions" / str(assignment_id)
        directories[directory] = (filename, content)

    previous_dependency = app.dependency_overrides[get_db]
    original_commit = db.commit
    original_restore = courses_router.ReversibleDelete.restore
    commit_error = RuntimeError("commit failed")
    restore_attempts = []

    def fail_commit():
        raise commit_error

    def fail_first_restore(token):
        restore_attempts.append(token)
        if len(restore_attempts) == 1:
            raise OSError("restore failed")
        original_restore(token)

    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(db, "commit", fail_commit)
    monkeypatch.setattr(courses_router.ReversibleDelete, "restore", fail_first_restore)
    try:
        with pytest.raises(RuntimeError, match="commit failed") as caught:
            client.delete(f"/api/courses/{course_id}")
    finally:
        monkeypatch.setattr(db, "commit", original_commit)
        app.dependency_overrides[get_db] = previous_dependency

    assert caught.value is commit_error
    assert isinstance(caught.value.__cause__, ExceptionGroup)
    assert any(isinstance(error, OSError) for error in caught.value.__cause__.exceptions)
    assert len(restore_attempts) == 3
    for token in restore_attempts[1:]:
        filename, content = directories[token.original]
        assert (token.original / filename).read_bytes() == content

    failed_token = restore_attempts[0]
    failed_filename, failed_content = directories[failed_token.original]
    assert not failed_token.original.exists()
    assert failed_token.backup.is_dir()
    assert (failed_token.backup / failed_filename).read_bytes() == failed_content

    db.expire_all()
    assert db.get(Course, course_id) is not None
    assert all(db.get(Assignment, assignment_id) is not None for assignment_id in assignment_ids)


def test_course_delete_discard_failure_does_not_skip_other_backups(
    client, db, course_setup, tmp_path, monkeypatch
):
    course_id = course_setup["course"]["id"]
    assignment_ids = [course_setup["assignment"]["id"]]
    for name in ("Essay 2", "Essay 3"):
        assignment = client.post(
            f"/api/courses/{course_id}/assignments",
            json={"name": name},
        ).json()
        assignment_ids.append(assignment["id"])

    directories = []
    for index, assignment_id in enumerate(assignment_ids, start=1):
        directory = tmp_path / "submissions" / str(assignment_id)
        directory.mkdir(parents=True)
        (directory / f"essay-{index}.txt").write_bytes(f"assignment-{index}".encode())
        directories.append(directory)

    original_discard = courses_router.ReversibleDelete.discard
    discard_attempts = []

    def fail_first_discard(token):
        discard_attempts.append(token)
        if len(discard_attempts) == 1:
            raise OSError("discard failed")
        original_discard(token)

    monkeypatch.setattr(courses_router.ReversibleDelete, "discard", fail_first_discard)
    with pytest.raises(ExceptionGroup) as caught:
        client.delete(f"/api/courses/{course_id}")

    assert len(discard_attempts) == 3
    assert any(
        isinstance(error, OSError) and str(error) == "discard failed"
        for error in caught.value.exceptions
    )
    db.expire_all()
    assert db.get(Course, course_id) is None
    assert all(db.get(Assignment, assignment_id) is None for assignment_id in assignment_ids)
    assert all(not directory.exists() for directory in directories)
    assert discard_attempts[0].backup.is_dir()
    assert all(not token.backup.exists() for token in discard_attempts[1:])


def test_assignment_delete_commit_failure_restores_directory_and_row(
    client, db, course_setup, tmp_path, monkeypatch
):
    assignment_id = course_setup["assignment"]["id"]
    upload(client, assignment_id, [("essay.pdf", PDF_BYTES, "application/pdf")])
    stored = tmp_path / "submissions" / str(assignment_id) / "essay.pdf"
    content = stored.read_bytes()
    previous_dependency = app.dependency_overrides[get_db]
    original_commit = db.commit

    def fail_commit():
        raise RuntimeError("commit failed")

    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(db, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            client.delete(f"/api/assignments/{assignment_id}")
    finally:
        monkeypatch.setattr(db, "commit", original_commit)
        app.dependency_overrides[get_db] = previous_dependency

    db.expire_all()
    assert db.get(Assignment, assignment_id) is not None
    assert stored.read_bytes() == content
    assert_no_delete_backups(tmp_path / "submissions")


def test_submission_delete_commit_failure_restores_file_and_row(
    client, db, course_setup, tmp_path, monkeypatch
):
    assignment_id = course_setup["assignment"]["id"]
    payload = upload(
        client, assignment_id, [("essay.pdf", PDF_BYTES, "application/pdf")]
    ).json()
    submission_id = payload["uploaded"][0]["submission_id"]
    stored = tmp_path / "submissions" / str(assignment_id) / "essay.pdf"
    content = stored.read_bytes()
    previous_dependency = app.dependency_overrides[get_db]
    original_commit = db.commit

    def fail_commit():
        raise RuntimeError("commit failed")

    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(db, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            client.delete(f"/api/submissions/{submission_id}")
    finally:
        monkeypatch.setattr(db, "commit", original_commit)
        app.dependency_overrides[get_db] = previous_dependency

    db.expire_all()
    assert db.get(Submission, submission_id) is not None
    assert stored.read_bytes() == content
    assert_no_delete_backups(tmp_path / "submissions")


def test_submission_upload_commit_failure_removes_new_files(
    client, db, course_setup, tmp_path, monkeypatch
):
    assignment_id = course_setup["assignment"]["id"]
    previous_dependency = app.dependency_overrides[get_db]
    original_commit = db.commit

    def fail_commit():
        raise RuntimeError("commit failed")

    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(db, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            upload(client, assignment_id, [("failed.pdf", PDF_BYTES, "application/pdf")])
    finally:
        monkeypatch.setattr(db, "commit", original_commit)
        app.dependency_overrides[get_db] = previous_dependency

    db.expire_all()
    assert db.query(Submission).filter_by(assignment_id=assignment_id).count() == 0
    root = tmp_path / "submissions"
    assert not [path for path in root.rglob("*") if path.is_file()]
    assert_no_delete_backups(root)


def test_submission_partial_write_failure_removes_partial_file(
    client, db, course_setup, tmp_path, monkeypatch
):
    assignment_id = course_setup["assignment"]["id"]

    def write_prefix_then_fail(path, data):
        with path.open("wb") as partial:
            partial.write(data[:5])
        raise OSError("write failed")

    monkeypatch.setattr(Path, "write_bytes", write_prefix_then_fail)

    with pytest.raises(OSError, match="write failed"):
        upload(client, assignment_id, [("partial.pdf", PDF_BYTES, "application/pdf")])

    db.expire_all()
    assert db.query(Submission).filter_by(assignment_id=assignment_id).count() == 0
    assignment_directory = tmp_path / "submissions" / str(assignment_id)
    assert not [path for path in assignment_directory.rglob("*") if path.is_file()]


def test_grading_page_route_matches_the_template_contract(client, course_setup):
    assignment_id = course_setup["assignment"]["id"]
    upload(client, assignment_id, [("essay1_student_01.pdf", PDF_BYTES, "application/pdf")])
    resp = client.get(f"/grading/{assignment_id}")
    assert resp.status_code == 200
    html = resp.text
    # The page must hand the JS the endpoints this router actually serves.
    assert f"/api/assignments/{assignment_id}/grade-all" in html
    assert f"/api/assignments/{assignment_id}/grading/status" in html
    assert "/api/submissions/{id}/grade" in html
    assert client.get(f"/assignments/{assignment_id}").status_code == 200
    assert client.get("/grading/99999").status_code == 404


# --------------------------------------------------------------------------
# Anthropic provider (fake client — no network, no key)
# --------------------------------------------------------------------------


class FakeBlock:
    def __init__(self, type_, **kwargs):
        self.type = type_
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeStopDetails:
    def __init__(self, category, explanation):
        self.category = category
        self.explanation = explanation


class FakeMessage:
    def __init__(self, content, stop_reason="end_turn", stop_details=None, model="fake"):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.model = model


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAnthropicClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def test_anthropic_grade_uses_structured_outputs_and_no_sampling_params():
    payload = {
        "criteria": [{"key": "thesis", "score": 9, "comment": "Strong."}],
        "summary_feedback": "Good.",
        "misconceptions": [],
        "strengths": ["clarity"],
    }
    fake = FakeAnthropicClient(
        FakeMessage([FakeBlock("text", text=json.dumps(payload))], model="claude-opus-5")
    )
    provider = AnthropicProvider(model="claude-opus-5", api_key="sk-test", max_tokens=4000, client=fake)

    result = provider.grade("system", [providers_mod.text_block("hi")], engine.GRADE_SCHEMA)
    assert result == payload

    sent = fake.messages.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["output_config"] == {
        "format": {"type": "json_schema", "schema": engine.GRADE_SCHEMA},
        "effort": config.GRADING_EFFORT,
    }
    # Current Claude models 400 on these — they must never be sent.
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in sent
    assert sent["messages"][0]["content"] == [{"type": "text", "text": "hi"}]
    # Server-side refusal fallback is opted in as a raw beta header + body key.
    assert sent["extra_headers"] == {"anthropic-beta": config.ANTHROPIC_FALLBACK_BETA}
    assert sent["extra_body"] == {"fallbacks": "default"}
    assert provider.served_model == "claude-opus-5"


def test_anthropic_fallback_opt_in_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_FALLBACKS", False)
    payload = {"criteria": [], "summary_feedback": "x", "misconceptions": [], "strengths": []}
    fake = FakeAnthropicClient(
        FakeMessage([FakeBlock("text", text=json.dumps(payload))], model="claude-opus-4-8")
    )
    provider = AnthropicProvider(model="claude-opus-5", api_key="sk-test", max_tokens=4000, client=fake)
    provider.grade("system", [providers_mod.text_block("hi")], engine.GRADE_SCHEMA)
    sent = fake.messages.calls[0]
    assert "extra_headers" not in sent and "extra_body" not in sent
    # The model that answered is recorded, not the one that was asked.
    assert provider.served_model == "claude-opus-4-8"


def test_anthropic_refusal_is_a_typed_error_not_a_crash():
    fake = FakeAnthropicClient(
        FakeMessage(
            [],
            stop_reason="refusal",
            stop_details=FakeStopDetails("cyber", "This looks like a jailbreak attempt."),
        )
    )
    provider = AnthropicProvider(model="claude-opus-5", api_key="sk-test", max_tokens=1000, client=fake)
    with pytest.raises(ProviderRefusalError) as excinfo:
        provider.grade("system", [providers_mod.text_block("x")], engine.GRADE_SCHEMA)
    assert excinfo.value.category == "cyber"
    assert "jailbreak" in str(excinfo.value)


def test_anthropic_truncation_and_bad_json_are_response_errors():
    truncated = FakeAnthropicClient(
        FakeMessage([FakeBlock("text", text='{"criteria": [')], stop_reason="max_tokens")
    )
    provider = AnthropicProvider(api_key="k", model="claude-opus-5", max_tokens=100, client=truncated)
    with pytest.raises(ProviderResponseError):
        provider.grade("s", [], engine.GRADE_SCHEMA)

    junk = FakeAnthropicClient(FakeMessage([FakeBlock("text", text="Sure! Here you go:")]))
    provider = AnthropicProvider(api_key="k", model="claude-opus-5", max_tokens=100, client=junk)
    with pytest.raises(ProviderResponseError):
        provider.grade("s", [], engine.GRADE_SCHEMA)


def test_anthropic_chat_parses_tool_inputs_as_dicts():
    fake = FakeAnthropicClient(
        FakeMessage(
            [
                FakeBlock("text", text="Let me look that up."),
                FakeBlock(
                    "tool_use",
                    id="toolu_1",
                    name="get_course_summary",
                    # Some transports hand back a JSON string; we parse, never match.
                    input='{"course_id": 4}',
                ),
            ],
            stop_reason="tool_use",
        )
    )
    provider = AnthropicProvider(api_key="k", model="claude-opus-5", max_tokens=1000, client=fake)
    turn = provider.chat(
        "You are Agora.",
        [
            {"role": "user", "content": "How is course 4 doing?"},
            providers_mod.tool_result_message(
                [{"tool_use_id": "toolu_0", "content": "nothing", "is_error": True}]
            ),
        ],
        [{"name": "get_course_summary", "description": "", "input_schema": {"type": "object"}}],
    )
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].arguments == {"course_id": 4}
    assert turn.text == "Let me look that up."

    sent = fake.messages.calls[0]
    assert sent["tools"][0]["input_schema"] == {"type": "object"}
    tool_msg = sent["messages"][1]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["is_error"] is True


# --------------------------------------------------------------------------
# OpenAI provider
# --------------------------------------------------------------------------


class FakeChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class FakeOpenAIMessage:
    def __init__(self, content="", refusal=None, tool_calls=None):
        self.content = content
        self.refusal = refusal
        self.tool_calls = tool_calls or []


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIClient:
    def __init__(self, response):
        self.chat = type("Chat", (), {"completions": FakeCompletions(response)})()


def test_openai_grade_uses_json_schema_response_format():
    payload = {
        "criteria": [{"key": "thesis", "score": 7, "comment": "OK."}],
        "summary_feedback": "Fine.",
        "misconceptions": ["hasty generalisation"],
        "strengths": [],
    }
    response = type("R", (), {"choices": [FakeChoice(FakeOpenAIMessage(json.dumps(payload)))], "model": "gpt"})()
    fake = FakeOpenAIClient(response)
    provider = OpenAIProvider(model="gpt-5.6-sol", api_key="sk-test", max_tokens=2000, client=fake)

    result = provider.grade("system", [providers_mod.text_block("body")], engine.GRADE_SCHEMA)
    assert result == payload

    sent = fake.chat.completions.calls[0]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"] is engine.GRADE_SCHEMA
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["max_completion_tokens"] == 2000
    assert sent["messages"][0] == {"role": "system", "content": "system"}
    assert "temperature" not in sent


def test_openai_degrades_gracefully_on_pdfs():
    fake = FakeOpenAIClient(type("R", (), {"choices": [], "model": "gpt"})())
    provider = OpenAIProvider(model="gpt-5.6-sol", api_key="k", max_tokens=100, client=fake)
    assert provider.supports_pdf() is False

    block = providers_mod.document_block(PDF_BYTES, filename="essay.pdf")
    with pytest.raises(ProviderUnsupportedError) as excinfo:
        provider.grade("s", [block], engine.GRADE_SCHEMA)
    assert "PDF" in str(excinfo.value)
    assert fake.chat.completions.calls == [], "no request should be attempted"

    # With extracted text available it degrades to text instead of failing.
    with_text = providers_mod.document_block(
        PDF_BYTES, filename="essay.pdf", text_fallback="The extracted essay text."
    )
    parts = provider._content_parts([with_text])
    assert parts[0]["type"] == "text"
    assert "The extracted essay text." in parts[0]["text"]


def test_openai_chat_parses_tool_call_arguments():
    call = type(
        "C",
        (),
        {
            "id": "call_1",
            "function": type("F", (), {"name": "navigate_to", "arguments": '{"page": "analytics"}'})(),
        },
    )()
    response = type(
        "R",
        (),
        {
            "choices": [FakeChoice(FakeOpenAIMessage("", tool_calls=[call]), "tool_calls")],
            "model": "gpt",
        },
    )()
    provider = OpenAIProvider(
        model="gpt-5.6-sol", api_key="k", max_tokens=100, client=FakeOpenAIClient(response)
    )
    turn = provider.chat("sys", [{"role": "user", "content": "open analytics"}], [
        {"name": "navigate_to", "description": "", "input_schema": {"type": "object"}}
    ])
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].arguments == {"page": "analytics"}


def test_openai_replays_a_tool_round_as_tool_calls():
    """Round 2 of a tool loop: the assistant turn must carry `tool_calls`.

    The assistant module replays the previous turn as Anthropic-shaped
    `tool_use` blocks. Dropping them leaves a `role="tool"` message with no
    call to answer, which the API rejects with a 400.
    """
    response = type(
        "R",
        (),
        {
            "choices": [FakeChoice(FakeOpenAIMessage("Course 4 is averaging 82%."), "stop")],
            "model": "gpt",
        },
    )()
    fake = FakeOpenAIClient(response)
    provider = OpenAIProvider(model="gpt-5.6-sol", api_key="k", max_tokens=500, client=fake)

    turn = provider.chat(
        "sys",
        [
            {"role": "user", "content": "How is course 4 doing?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking that up."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_course_summary",
                        "input": {"course_id": 4},
                    },
                ],
            },
            providers_mod.tool_result_message(
                [{"tool_use_id": "call_1", "content": '{"average": 82}', "is_error": False}]
            ),
        ],
        [{"name": "get_course_summary", "description": "", "input_schema": {"type": "object"}}],
    )
    assert turn.text == "Course 4 is averaging 82%."

    sent = fake.chat.completions.calls[0]["messages"]
    assistant = sent[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["type"] == "function"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_course_summary"
    # Arguments go over the wire as a JSON string, and must survive intact.
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"course_id": 4}
    # The tool answer follows the call that produced it.
    assert sent[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"average": 82}'}


def test_no_provider_payload_carries_the_original_filename(monkeypatch):
    """SPEC principle 2: the upload filename usually contains the student name."""
    leaky = "Smith_Jane_essay1.pdf"
    block = providers_mod.document_block(PDF_BYTES, filename=leaky)

    # Anthropic strips every private `_` key.
    assert leaky not in json.dumps(providers_mod.public_blocks([block]))

    # OpenAI, native-PDF path.
    provider = OpenAIProvider(model="gpt-5.6-sol", api_key="k", max_tokens=100, client=object())
    monkeypatch.setattr(provider, "supports_pdf", lambda: True)
    parts = provider._content_parts([block])
    assert parts[0]["file"]["filename"] == providers_mod.NEUTRAL_DOCUMENT_NAME
    assert leaky not in json.dumps(parts)

    # OpenAI, degraded text path.
    monkeypatch.setattr(provider, "supports_pdf", lambda: False)
    with_text = providers_mod.document_block(
        PDF_BYTES, filename=leaky, text_fallback="The essay body."
    )
    assert leaky not in json.dumps(provider._content_parts([with_text]))


def test_openai_content_filter_raises_a_refusal_like_anthropic():
    response = type(
        "R",
        (),
        {
            "choices": [FakeChoice(FakeOpenAIMessage(""), "content_filter")],
            "model": "gpt",
        },
    )()
    provider = OpenAIProvider(
        model="gpt-5.6-sol", api_key="k", max_tokens=100, client=FakeOpenAIClient(response)
    )
    with pytest.raises(ProviderRefusalError):
        provider.chat("sys", [{"role": "user", "content": "hi"}], [])


# --------------------------------------------------------------------------
# output budget
# --------------------------------------------------------------------------


def test_max_tokens_defaults_to_the_model_registry_entry():
    # A flat 8000 truncates a multi-criterion grade on a thinking-by-default
    # model, so the request is sized from the registry.
    assert AnthropicProvider(api_key="k", model="claude-opus-5").max_tokens == 16000
    assert AnthropicProvider(api_key="k", model="claude-haiku-4-5").max_tokens == 8000
    assert OpenAIProvider(api_key="k", model="gpt-5.6-sol").max_tokens == 16000
    # An explicit skill setting still wins.
    assert AnthropicProvider(api_key="k", model="claude-opus-5", max_tokens=2000).max_tokens == 2000
    # Unknown model -> the flat fallback.
    assert AnthropicProvider(api_key="k", model="claude-from-2029").max_tokens == (
        config.DEFAULT_MAX_TOKENS
    )


def test_long_grading_requests_use_the_streaming_api():
    payload = {
        "criteria": [{"key": "thesis", "score": 9, "comment": "Strong."}],
        "summary_feedback": "Good.",
        "misconceptions": [],
        "strengths": [],
    }
    message = FakeMessage([FakeBlock("text", text=json.dumps(payload))])

    class FakeStream:
        def __init__(self, response):
            self.response = response

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return self.response

    class StreamingMessages(FakeMessages):
        def __init__(self, response):
            super().__init__(response)
            self.streamed: list[dict] = []

        def stream(self, **kwargs):
            self.streamed.append(kwargs)
            return FakeStream(self.response)

    client = type("C", (), {})()
    client.messages = StreamingMessages(message)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5", client=client)
    assert provider.max_tokens == 16000
    provider.grade("s", [providers_mod.text_block("x")], engine.GRADE_SCHEMA)
    assert client.messages.streamed, "a 16k-token grade must stream"
    assert client.messages.calls == []


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------


def test_get_provider_resolves_skills_and_keys(db, monkeypatch):
    monkeypatch.delenv(providers_mod.FORCE_PROVIDER_ENV, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    skill = Skill(name="Mock", provider=config.MOCK_PROVIDER, model=config.MOCK_MODEL)
    assert isinstance(get_provider(skill, db=db), MockProvider)
    assert isinstance(get_provider("mock"), MockProvider)

    # No key configured -> a typed config error pointing at Settings.
    real = Skill(name="Real", provider="anthropic", model="claude-opus-5")
    with pytest.raises(ProviderConfigError) as excinfo:
        get_provider(real, db=db)
    assert "Settings" in str(excinfo.value)

    with pytest.raises(ProviderConfigError):
        get_provider({"provider": "wat"})

    # A stored, encrypted credential is enough to build the client wrapper.
    from app import security

    security.set_api_key(db, "anthropic", "sk-ant-testkey-123456")
    provider = get_provider(real, db=db)
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-opus-5"
    assert provider.api_key == "sk-ant-testkey-123456"

    # The env override forces the mock everywhere (offline demos / tests).
    monkeypatch.setenv(providers_mod.FORCE_PROVIDER_ENV, config.MOCK_PROVIDER)
    assert isinstance(get_provider(real, db=db), MockProvider)


def test_anthropic_haiku_omits_unsupported_effort():
    payload = {"criteria": [], "summary_feedback": "Synthetic", "misconceptions": [], "strengths": []}
    fake = FakeAnthropicClient(FakeMessage([FakeBlock("text", text=json.dumps(payload))]))
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="sk-test", client=fake)
    assert provider.grade("system", [], engine.GRADE_SCHEMA) == payload
    assert "effort" not in fake.messages.calls[0]["output_config"]
    assert fake.messages.calls[0]["output_config"]["format"]["type"] == "json_schema"
