"""Core module tests: schema round-trip, roster CSV import, analytics aggregation.

These build their own SQLite engine on a tmp path and override the `get_db`
dependency, so they are independent of import order with the other modules'
test files and never touch the real data/ directory.
"""

from __future__ import annotations

import io
import os
import re
import stat

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app import analytics, config, db as db_mod, security
from app.db import get_db
from app.main import app
from app.models import (
    ApiCredential,
    Assignment,
    Base,
    Course,
    GradeResult,
    Rubric,
    Skill,
    Student,
    Submission,
)
from app.routers.courses import import_roster_rows, parse_roster_csv
from app.seed import RUBRIC_CRITERIA, seed_demo


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
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
def seeded(db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "submissions")
    summary = seed_demo(db=db)
    assert summary["created"] is True
    return summary


def make_course(db, name="Test Course", term="Spring 2026") -> Course:
    course = Course(name=name, term=term)
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def test_models_round_trip(db):
    course = make_course(db)
    student = Student(course_id=course.id, name="Ada Lovelace", student_number=1,
                      email="ada@example.edu")
    rubric = Rubric(
        name="Essay Rubric",
        criteria=[
            {"key": "thesis", "title": "Thesis", "description": "", "max_points": 10},
            {"key": "evidence", "title": "Evidence", "description": "", "max_points": 5},
        ],
    )
    skill = Skill(name="Grader", system_prompt="Be strict.")
    db.add_all([student, rubric, skill])
    db.commit()

    assignment = Assignment(
        course_id=course.id, name="PS1", rubric_id=rubric.id, skill_id=skill.id
    )
    db.add(assignment)
    db.commit()

    submission = Submission(
        assignment_id=assignment.id,
        student_id=student.id,
        file_path="/tmp/ps1.pdf",
        mime_type="application/pdf",
        status="graded",
    )
    db.add(submission)
    db.commit()

    result = GradeResult(
        submission_id=submission.id,
        overall_score=12.5,
        max_score=15,
        summary_feedback="Good.",
        criteria=[
            {"key": "thesis", "score": 8.5, "max_points": 10, "comment": "Clear."},
            {"key": "evidence", "score": 4, "max_points": 5, "comment": "Thin."},
        ],
        misconceptions=["conflates legality with morality"],
        strengths=["clear thesis"],
        model=config.MOCK_MODEL,
    )
    db.add(result)
    db.commit()

    db.expire_all()
    fetched = db.get(Course, course.id)
    assert fetched.name == "Test Course"
    assert [s.name for s in fetched.students] == ["Ada Lovelace"]
    assert fetched.students[0].anon_label == "Student #1"
    assert fetched.assignments[0].submissions[0].grade_result.overall_score == 12.5
    # JSON columns survive the round-trip as real Python structures.
    stored = fetched.assignments[0].submissions[0].grade_result
    assert stored.criteria[0]["key"] == "thesis"
    assert stored.misconceptions == ["conflates legality with morality"]
    assert stored.percentage == pytest.approx(83.3)
    assert db.get(Rubric, rubric.id).total_points == 15

    # Deleting the course cascades to students, assignments, submissions, grades.
    db.delete(fetched)
    db.commit()
    assert db.query(Student).count() == 0
    assert db.query(Submission).count() == 0
    assert db.query(GradeResult).count() == 0


def test_init_db_adds_chat_message_course_id(tmp_path, monkeypatch):
    legacy_engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with legacy_engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE chat_messages (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT,
                tool_calls JSON,
                created_at DATETIME,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
            """
        )
    monkeypatch.setattr(db_mod, "engine", legacy_engine)

    db_mod.init_db()

    columns = {column["name"] for column in inspect(legacy_engine).get_columns("chat_messages")}
    assert "course_id" in columns
    legacy_engine.dispose()


def test_init_db_adds_chat_message_history_index(tmp_path, monkeypatch):
    legacy_engine = create_engine(f"sqlite:///{tmp_path / 'legacy-index.db'}")
    with legacy_engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE chat_messages (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT,
                tool_calls JSON,
                created_at DATETIME
            )
            """
        )
    monkeypatch.setattr(db_mod, "engine", legacy_engine)

    db_mod.init_db()
    indexes = [
        index
        for index in inspect(legacy_engine).get_indexes("chat_messages")
        if index["name"] == "ix_chat_messages_session_id_id"
    ]
    assert len(indexes) == 1
    assert indexes[0]["column_names"] == ["session_id", "id"]

    db_mod.init_db()
    indexes = [
        index
        for index in inspect(legacy_engine).get_indexes("chat_messages")
        if index["name"] == "ix_chat_messages_session_id_id"
    ]
    assert len(indexes) == 1

    with legacy_engine.connect() as connection:
        plan = connection.exec_driver_sql(
            "EXPLAIN QUERY PLAN SELECT * FROM chat_messages "
            "WHERE session_id = 1 ORDER BY id DESC LIMIT 24"
        ).all()
    assert any("ix_chat_messages_session_id_id" in row[-1] for row in plan)
    legacy_engine.dispose()


def test_student_number_unique_per_course(db):
    course_a = make_course(db, "A")
    course_b = make_course(db, "B")
    db.add(Student(course_id=course_a.id, name="One", student_number=1))
    db.add(Student(course_id=course_b.id, name="Other", student_number=1))
    db.commit()  # same number in a different course is fine

    db.add(Student(course_id=course_a.id, name="Clash", student_number=1))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# --------------------------------------------------------------------------
# security
# --------------------------------------------------------------------------


def test_api_key_encryption_round_trip(db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config, "SECRET_KEY_PATH", tmp_path / "cfg" / "secret.key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    security.reset_cache()

    token = security.encrypt("sk-ant-super-secret-value")
    assert "super-secret" not in token
    assert security.decrypt(token) == "sk-ant-super-secret-value"

    key_file = config.SECRET_KEY_PATH
    assert key_file.exists()
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600

    cred = security.set_api_key(db, "anthropic", "sk-ant-abc123456789")
    assert isinstance(cred, ApiCredential)
    assert "sk-ant" not in cred.key_encrypted
    assert security.get_api_key(db, "anthropic") == "sk-ant-abc123456789"
    assert security.get_api_key(db, "openai") is None
    assert security.mask_key("sk-ant-abc123456789").endswith("6789")

    security.reset_cache()


def test_model_registry_has_exact_anthropic_ids():
    ids = config.model_ids("anthropic")
    assert ids == ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
    assert config.default_model_for("anthropic") == "claude-opus-5"
    assert config.models_for("openai"), "OpenAI defaults must exist and be editable"


def test_launcher_rejects_non_loopback_host():
    import run

    assert config.HOST == "127.0.0.1"
    assert run.parse_args([]).host == "127.0.0.1"
    with pytest.raises(SystemExit) as exc_info:
        run.parse_args(["--host", "0.0.0.0"])
    assert exc_info.value.code == 2


# --------------------------------------------------------------------------
# roster CSV import
# --------------------------------------------------------------------------


def test_parse_roster_csv_with_header():
    rows, warnings = parse_roster_csv(
        "name,email\nAmara Osei,aosei@example.edu\nBen Whitaker,\n\n"
    )
    assert [r["name"] for r in rows] == ["Amara Osei", "Ben Whitaker"]
    assert rows[0]["email"] == "aosei@example.edu"
    assert rows[1]["email"] is None
    assert warnings == []


def test_parse_roster_csv_variants():
    # First/last name columns.
    rows, _ = parse_roster_csv("Last Name,First Name,Email\nOsei,Amara,a@x.edu\n")
    assert rows[0]["name"] == "Amara Osei"
    assert rows[0]["email"] == "a@x.edu"

    # No header at all.
    rows, warnings = parse_roster_csv("Amara Osei,aosei@example.edu\nBen Whitaker\n")
    assert [r["name"] for r in rows] == ["Amara Osei", "Ben Whitaker"]
    assert rows[0]["email"] == "aosei@example.edu"
    assert any("header" in w.lower() for w in warnings)

    # Explicit student numbers + a malformed email + a nameless row.
    rows, warnings = parse_roster_csv(
        "name,email,student number\nKeiko Tanaka,not-an-email,7\n,orphan@x.edu,8\n"
    )
    assert len(rows) == 1
    assert rows[0]["student_number"] == 7
    assert rows[0]["email"] is None
    assert len(warnings) == 2

    # Semicolon-delimited export.
    rows, _ = parse_roster_csv("name;email\nHugo Brandt;hb@x.edu\n")
    assert rows[0]["name"] == "Hugo Brandt"

    assert parse_roster_csv("")[0] == []


def test_import_roster_rows_assigns_numbers(db):
    course = make_course(db)
    rows, _ = parse_roster_csv(
        "name,email\nAmara Osei,a@x.edu\nBen Whitaker,b@x.edu\nClaudia Moreno,\n"
    )
    result = import_roster_rows(db, course.id, rows)
    assert result["created"] == 3
    assert [s["student_number"] for s in result["students"]] == [1, 2, 3]

    # Re-importing the same roster does not duplicate students.
    again = import_roster_rows(db, course.id, rows)
    assert again["created"] == 0
    assert db.query(Student).filter_by(course_id=course.id).count() == 3

    # A new name continues the numbering.
    more, _ = parse_roster_csv("name\nDaniel Park\n")
    result = import_roster_rows(db, course.id, more)
    assert result["students"][0]["student_number"] == 4


def test_roster_import_endpoint(client, db):
    course = client.post("/api/courses", json={"name": "PHIL 101", "term": "Fall"}).json()
    csv_bytes = b"name,email\nAmara Osei,aosei@example.edu\nBen Whitaker,bw@example.edu\n"

    resp = client.post(
        f"/api/courses/{course['id']}/roster/import",
        files={"roster.csv": ("roster.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 422  # wrong field name -> validation error

    resp = client.post(
        f"/api/courses/{course['id']}/roster/import",
        files={"file": ("roster.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["created"] == 2
    assert payload["parsed"] == 2

    listed = client.get(f"/api/courses/{course['id']}/students").json()
    assert [s["name"] for s in listed] == ["Amara Osei", "Ben Whitaker"]
    assert [s["student_number"] for s in listed] == [1, 2]
    # Names stay local; the anonymized handle is what goes to providers.
    assert listed[0]["anon_label"] == "Student #1"

    bad = client.post(
        f"/api/courses/{course['id']}/roster/import",
        files={"file": ("empty.csv", io.BytesIO(b"   \n"), "text/csv")},
    )
    assert bad.status_code == 400


def test_student_crud_endpoints(client, db, tmp_path):
    course = client.post("/api/courses", json={"name": "CS 200"}).json()
    created = client.post(
        f"/api/courses/{course['id']}/students", json={"name": "Grace Hopper"}
    ).json()
    assert created["student_number"] == 1

    clash = client.post(
        f"/api/courses/{course['id']}/students",
        json={"name": "Someone Else", "student_number": 1},
    )
    assert clash.status_code == 409

    patched = client.patch(
        f"/api/students/{created['id']}", json={"email": "gh@example.edu"}
    ).json()
    assert patched["email"] == "gh@example.edu"

    assignment = Assignment(course_id=course["id"], name="Final project")
    db.add(assignment)
    db.flush()
    submission_file = tmp_path / "final-project.pdf"
    submission_file.write_bytes(b"submitted work")
    submission = Submission(
        assignment_id=assignment.id,
        student_id=created["id"],
        file_path=str(submission_file),
        status="graded",
    )
    db.add(submission)
    db.flush()
    grade_result = GradeResult(
        submission_id=submission.id,
        overall_score=9,
        max_score=10,
    )
    db.add(grade_result)
    db.commit()
    submission_id = submission.id
    grade_result_id = grade_result.id

    db.expire_all()
    student = db.get(Student, created["id"])
    assert student is not None
    assert submission in student.submissions
    assert client.delete(f"/api/students/{created['id']}").status_code == 200
    assert client.get(f"/api/students/{created['id']}").status_code == 404
    db.expire_all()
    assert db.get(Student, created["id"]) is None
    preserved_submission = db.get(Submission, submission_id)
    assert preserved_submission is not None
    assert preserved_submission.student_id is None
    assert db.get(GradeResult, grade_result_id) is not None
    assert submission_file.exists()
    assert client.get("/api/courses/9999").status_code == 404


def test_course_detail_query_count_is_constant(client, db):
    one_course = Course(name="One-row course")
    twelve_course = Course(name="Twelve-row course")
    db.add_all([one_course, twelve_course])
    db.flush()

    for course, count in ((one_course, 1), (twelve_course, 12)):
        students = [
            Student(course_id=course.id, name=f"Student {index}", student_number=index)
            for index in range(1, count + 1)
        ]
        assignments = [
            Assignment(course_id=course.id, name=f"Assignment {index}")
            for index in range(1, count + 1)
        ]
        db.add_all([*students, *assignments])
        db.flush()
        submissions = [
            Submission(
                assignment_id=assignments[index].id,
                student_id=students[index].id,
                status="graded",
            )
            for index in range(count)
        ]
        db.add_all(submissions)
        db.flush()
        db.add_all(
            GradeResult(submission_id=submission.id, overall_score=8, max_score=10)
            for submission in submissions
        )
    db.commit()

    def measure(course_id):
        db.expire_all()
        statements = []

        def capture_selects(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        engine_ = db.get_bind()
        event.listen(engine_, "before_cursor_execute", capture_selects)
        try:
            response = client.get(f"/courses/{course_id}")
        finally:
            event.remove(engine_, "before_cursor_execute", capture_selects)
        return response, len(statements)

    one_response, one_count = measure(one_course.id)
    twelve_response, twelve_count = measure(twelve_course.id)

    assert one_response.status_code == 200
    assert twelve_response.status_code == 200
    assert one_count == twelve_count
    assert twelve_count <= 7


# --------------------------------------------------------------------------
# analytics
# --------------------------------------------------------------------------


def test_seed_demo_populates_course(db, seeded):
    course = db.get(Course, seeded["course_id"])
    assert len(course.students) == 12
    assert len(course.assignments) == 2
    assert db.query(Submission).count() == 24
    assert db.query(GradeResult).count() == 24
    assert all(s.status == "graded" for s in db.query(Submission).all())


def test_course_overview_aggregation(db, seeded):
    data = analytics.course_overview(db, seeded["course_id"])

    assert data["totals"]["students"] == 12
    assert data["totals"]["assignments"] == 2
    assert data["totals"]["submissions"] == 24
    assert data["totals"]["graded"] == 24
    assert 0 < data["totals"]["average_percent"] <= 100

    assert len(data["assignments"]) == 2
    for block in data["assignments"]:
        assert block["graded_count"] == 12
        assert sum(b["count"] for b in block["distribution"]) == 12
        assert block["low_percent"] <= block["average_percent"] <= block["high_percent"]
        assert block["max_score"] == sum(c["max_points"] for c in RUBRIC_CRITERIA)

    # Seeded data must actually vary, otherwise the charts are meaningless.
    percents = [b["average_percent"] for b in data["assignments"]]
    assert len(set(percents)) > 1
    buckets_used = {
        b["label"]
        for block in data["assignments"]
        for b in block["distribution"]
        if b["count"]
    }
    assert len(buckets_used) > 1
    assert [t["assignment_id"] for t in data["trend"]] == [
        b["assignment_id"] for b in data["assignments"]
    ]


def test_criteria_and_misconception_aggregation(db, seeded):
    crits = analytics.criteria_breakdown(db, seeded["course_id"])
    keys = {c["key"] for c in crits["criteria"]}
    assert keys == {c["key"] for c in RUBRIC_CRITERIA}
    for crit in crits["criteria"]:
        assert crit["count"] == 24
        assert 0 <= crit["average_percent"] <= 100
        assert crit["average_score"] <= crit["max_points"]
        assert len(crit["by_assignment"]) == 2
    # Sorted weakest-first for the dashboard.
    ordered = [c["average_percent"] for c in crits["criteria"]]
    assert ordered == sorted(ordered)
    assert crits["weakest"] == crits["criteria"][0]["key"]

    misc = analytics.misconception_counts(db, seeded["course_id"])
    assert misc["misconceptions"], "seeded grades must produce misconception tags"
    counts = [m["count"] for m in misc["misconceptions"]]
    assert counts == sorted(counts, reverse=True)
    assert all(m["tag"] == m["tag"].lower() for m in misc["misconceptions"])
    assert misc["total_mentions"] == sum(counts)
    top = misc["misconceptions"][0]
    assert 0 < top["student_count"] <= 12
    assert 0 < top["assignment_count"] <= 2

    limited = analytics.misconception_counts(db, seeded["course_id"], limit=2)
    assert len(limited["misconceptions"]) == 2


def test_student_timeline(db, seeded):
    student = (
        db.query(Student)
        .filter_by(course_id=seeded["course_id"], student_number=3)
        .one()
    )
    data = analytics.student_timeline(db, student.id)

    assert data["student"]["student_number"] == 3
    assert data["course"]["id"] == seeded["course_id"]
    assert data["graded_count"] == 2
    assert len(data["timeline"]) == 2
    assert data["worst_percent"] <= data["average_percent"] <= data["best_percent"]
    assert {c["key"] for c in data["criteria"]} == {c["key"] for c in RUBRIC_CRITERIA}
    assert all(0 <= c["average_percent"] <= 100 for c in data["criteria"])
    for point in data["timeline"]:
        assert point["max_score"] == sum(c["max_points"] for c in RUBRIC_CRITERIA)
        assert point["percent"] == pytest.approx(
            round(100 * point["score"] / point["max_score"], 1)
        )


def test_analytics_missing_course_raises(db):
    with pytest.raises(LookupError):
        analytics.course_overview(db, 424242)


def test_analytics_endpoints(client, db, seeded):
    course_id = seeded["course_id"]

    overview = client.get(f"/api/courses/{course_id}/analytics/overview")
    assert overview.status_code == 200
    assert overview.json()["totals"]["graded"] == 24

    criteria = client.get(f"/api/courses/{course_id}/analytics/criteria")
    assert criteria.status_code == 200
    assert len(criteria.json()["criteria"]) == len(RUBRIC_CRITERIA)

    misc = client.get(f"/api/courses/{course_id}/analytics/misconceptions?limit=3")
    assert misc.status_code == 200
    assert len(misc.json()["misconceptions"]) <= 3

    # `most_common(0)` returns nothing and a negative count raises, so neither
    # is a valid limit — omit the parameter to get every tag.
    assert (
        client.get(f"/api/courses/{course_id}/analytics/misconceptions?limit=0").status_code
        == 422
    )
    assert (
        client.get(f"/api/courses/{course_id}/analytics/misconceptions?limit=-1").status_code
        == 422
    )
    all_tags = client.get(f"/api/courses/{course_id}/analytics/misconceptions")
    assert all_tags.status_code == 200
    assert len(all_tags.json()["misconceptions"]) >= 3

    student = db.query(Student).filter_by(course_id=course_id, student_number=1).one()
    timeline = client.get(
        f"/api/courses/{course_id}/analytics/students/{student.id}/timeline"
    )
    assert timeline.status_code == 200
    assert timeline.json()["graded_count"] == 2
    alias = client.get(f"/api/students/{student.id}/analytics/timeline")
    assert alias.json() == timeline.json()

    assert client.get("/api/courses/999999/analytics/overview").status_code == 404
    assert (
        client.get(f"/api/courses/{course_id}/analytics/students/999999/timeline").status_code
        == 404
    )


def test_home_page_renders_the_average_score_tile(client, db, seeded):
    """The tile reads `stats.average_percent`; anything else renders a bare '%'."""
    from app.routers.courses import _quick_stats

    stats = _quick_stats(db)
    assert stats["average_percent"] is not None

    resp = client.get("/")
    assert resp.status_code == 200
    assert f'>{stats["average_percent"]}%<' in resp.text
    assert ">%<" not in resp.text


def test_home_page_aggregates_submission_status_without_loading_submission_rows(
    client, db
):
    empty_course = Course(name="Empty course")
    partial_course = Course(name="Partial course")
    graded_course = Course(name="Graded course")
    db.add_all([empty_course, partial_course, graded_course])
    db.flush()
    empty = Assignment(course_id=empty_course.id, name="Empty assignment")
    partial = Assignment(course_id=partial_course.id, name="Partial assignment")
    graded = Assignment(course_id=graded_course.id, name="Graded assignment")
    db.add_all([empty, partial, graded])
    db.flush()
    db.add_all(
        [
            Submission(assignment_id=partial.id, status="graded"),
            Submission(assignment_id=partial.id, status="pending"),
            Submission(assignment_id=graded.id, status="graded"),
            Submission(assignment_id=graded.id, status="graded"),
        ]
    )
    db.commit()

    statements = []

    def capture_selects(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    engine_ = db.get_bind()
    event.listen(engine_, "before_cursor_execute", capture_selects)
    try:
        response = client.get("/")
    finally:
        event.remove(engine_, "before_cursor_execute", capture_selects)

    assert response.status_code == 200
    compact_html = re.sub(r"\s+", " ", response.text)
    assert "<td>0 / 0</td>" in compact_html
    assert "<td>1 / 2</td>" in compact_html
    assert "<td>2 / 2</td>" in compact_html
    assert '<span class="pill pill--pending">no submissions</span>' in compact_html
    assert '<span class="pill pill--warning">1 ungraded</span>' in compact_html
    assert '<span class="pill pill--graded">all graded</span>' in compact_html
    assert '<span class="pill pill--graded">graded</span>' in compact_html

    grouped_status_queries = [
        statement
        for statement in statements
        if "GROUP BY assignments.id, submissions.status" in statement
    ]
    assert len(grouped_status_queries) == 1
    forbidden_columns = (
        "submissions.file_path",
        "submissions.original_filename",
        "submissions.mime_type",
        "submissions.error",
    )
    assert not any(
        column in statement for statement in statements for column in forbidden_columns
    )


def test_router_import_failure_aborts_startup(monkeypatch):
    import importlib

    main_module = importlib.import_module("app.main")
    monkeypatch.setattr(main_module, "ROUTER_MODULES", ["app.routers.does_not_exist"])
    with pytest.raises(ModuleNotFoundError):
        main_module._include_routers()


def test_health_endpoint_reports_core_routers(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert "app.routers.courses" in payload["routers"]
    assert "app.routers.analytics" in payload["routers"]
