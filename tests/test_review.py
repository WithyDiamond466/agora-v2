"""Human review record, release, terms gate, and export.

Runs against a tmp SQLite DB with MockProvider only; nothing leaves the box.
"""

from __future__ import annotations

import csv
import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import config, review, terms
from app.ai import grading as engine
from app.db import get_db
from app.main import app
from app.models import (
    Acceptance,
    Assignment,
    Base,
    Course,
    GradeResult,
    ReviewEvent,
    Rubric,
    Skill,
    Student,
    Submission,
)
from app.routers import grading as grading_router

CRITERIA = [
    {"key": "thesis", "title": "Thesis", "description": "", "max_points": 10},
    {"key": "evidence", "title": "Evidence", "description": "", "max_points": 10},
]


@pytest.fixture()
def Session(tmp_path):
    engine_ = create_engine(
        f"sqlite:///{tmp_path / 'review.db'}", connect_args={"check_same_thread": False}
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
def client(Session, db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "submissions")
    monkeypatch.setattr(grading_router, "session_factory", Session)

    def override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def _graded_assignment(db, *, provider="mock", n=3, released=False):
    """Course + n students + assignment with n graded submissions."""
    course = Course(name="PHIL 210", term="Fall 2026")
    db.add(course)
    db.flush()
    rubric = Rubric(name="R", criteria=CRITERIA)
    skill = Skill(name="S", provider=provider, model=config.MOCK_MODEL if provider == "mock" else "claude-opus-5")
    db.add_all([rubric, skill])
    db.flush()
    assignment = Assignment(course_id=course.id, name="Essay 1", skill_id=skill.id, rubric_id=rubric.id)
    db.add(assignment)
    db.flush()
    subs = []
    for i in range(1, n + 1):
        student = Student(course_id=course.id, name=f"Student {i}", student_number=i, email=f"s{i}@x.edu")
        db.add(student)
        db.flush()
        sub = Submission(
            assignment_id=assignment.id, student_id=student.id, original_filename=f"s{i}.pdf",
            mime_type="application/pdf", status="graded",
        )
        db.add(sub)
        db.flush()
        db.add(
            GradeResult(
                submission_id=sub.id,
                overall_score=12 + i,
                max_score=20,
                summary_feedback=f"Feedback for Student-0{i}.",
                criteria=[
                    {"key": "thesis", "score": 6 + i, "max_points": 10, "comment": "ok"},
                    {"key": "evidence", "score": 6, "max_points": 10, "comment": "fine"},
                ],
                misconceptions=[],
                strengths=[],
                model=config.MOCK_MODEL,
            )
        )
        subs.append(sub)
    db.commit()
    for s in subs:
        db.refresh(s)
    return assignment, subs


def _accept(db):
    terms.accept(db, "Dr Test")


def _events(db, submission_id):
    return [e.kind for e in db.scalars(select(ReviewEvent).where(ReviewEvent.submission_id == submission_id).order_by(ReviewEvent.id)).all()]


# --------------------------------------------------------------------------
# per-result state
# --------------------------------------------------------------------------


def test_new_result_is_unreviewed_and_seen_is_idempotent(db, client):
    assignment, subs = _graded_assignment(db)
    sub = subs[0]
    assert sub.grade_result.review_state == "unreviewed"

    first = client.post(f"/api/submissions/{sub.id}/review/seen").json()
    assert first["review"]["state"] == "seen"
    seen_at = first["review"]["seen_at"]
    second = client.post(f"/api/submissions/{sub.id}/review/seen").json()
    assert second["review"]["seen_at"] == seen_at
    assert _events(db, sub.id) == ["seen"]


def test_seen_needs_a_result(db, client):
    assignment, subs = _graded_assignment(db, n=1)
    sub = Submission(assignment_id=assignment.id, student_id=subs[0].student_id, status="pending")
    db.add(sub)
    db.commit()
    assert client.post(f"/api/submissions/{sub.id}/review/seen").status_code == 409


def test_edit_is_logged_and_counts_as_seen(db, client):
    assignment, subs = _graded_assignment(db)
    sub = subs[1]
    resp = client.patch(
        f"/api/submissions/{sub.id}/result",
        json={"summary_feedback": "Rewritten.", "criteria": [{"key": "thesis", "score": 3}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["review"]["state"] == "seen"
    assert body["result"]["review"]["edit_count"] == 1
    db.expire_all()
    result = db.get(GradeResult, body["result"]["id"])
    assert result.edit_log[0]["fields"] == ["criteria.thesis.score", "summary_feedback"]
    assert _events(db, sub.id) == ["seen", "edited"]

    # An identical save records nothing new.
    client.patch(f"/api/submissions/{sub.id}/result", json={"summary_feedback": "Rewritten."})
    assert _events(db, sub.id) == ["seen", "edited"]


def test_withhold_and_lift(db, client):
    assignment, subs = _graded_assignment(db)
    sub = subs[2]
    held = client.post(f"/api/submissions/{sub.id}/review/withhold", json={"withheld": True}).json()
    assert held["review"]["state"] == "withheld"
    lifted = client.post(f"/api/submissions/{sub.id}/review/withhold", json={"withheld": False}).json()
    assert lifted["review"]["state"] == "seen"  # withholding counts as having looked
    assert _events(db, sub.id) == ["withheld", "unwithheld"]
    events = client.get(f"/api/submissions/{sub.id}/review").json()["events"]
    assert [e["kind"] for e in events] == ["withheld", "unwithheld"]


def test_regrade_returns_result_to_unreviewed_and_keeps_history(db):
    assignment, subs = _graded_assignment(db, n=1)
    sub = subs[0]
    review.mark_seen(db, sub)
    review.record_edit(db, sub, ["summary_feedback"], commit=True)
    assignment = db.get(Assignment, assignment.id)
    review.approve(db, sub)
    review.release(db, assignment)
    db.refresh(sub)
    assert sub.grade_result.review_state == "released"

    validated = engine.validate_grade_payload(
        {"criteria": [{"key": "thesis", "score": 1, "comment": "x"}], "summary_feedback": "new",
         "misconceptions": [], "strengths": []},
        CRITERIA,
    )
    result = engine.persist_grade_result(db, sub, validated, "mock-grader-1")
    assert result.review_state == "unreviewed"
    assert result.edit_log is None
    kinds = _events(db, sub.id)
    assert kinds[-1] == "regraded"
    last = db.scalars(select(ReviewEvent).where(ReviewEvent.kind == "regraded")).first()
    assert last.detail["previous_state"] == "released"
    assert last.detail["previous_edits"] == 1


# --------------------------------------------------------------------------
# release
# --------------------------------------------------------------------------


def test_release_requires_terms(db, client):
    assignment, subs = _graded_assignment(db)
    client.post(f"/api/submissions/{subs[0].id}/review/approve")
    blocked = client.post(f"/api/assignments/{assignment.id}/release", json={})
    assert blocked.status_code == 409
    assert "/terms" in blocked.json()["detail"]

    summary = client.get(f"/api/assignments/{assignment.id}/release").json()
    assert summary["terms"]["accepted"] is False
    assert len(summary["releasable"]) == 1 and len(summary["unseen"]) == 2

    _accept(db)
    ok = client.post(f"/api/assignments/{assignment.id}/release", json={}).json()
    assert [r["submission_id"] for r in ok["released"]] == [subs[0].id]
    assert sorted(r["reason"] for r in ok["held"]) == ["not approved", "not approved"]
    assert ok["counts"] == {"graded": 3, "unreviewed": 2, "seen": 0, "approved": 0, "released": 1, "withheld": 0}


def test_release_never_bypasses_explicit_approval(db, client):
    assignment, subs = _graded_assignment(db)
    _accept(db)
    client.post(f"/api/submissions/{subs[0].id}/review/approve")
    client.post(f"/api/submissions/{subs[1].id}/review/withhold", json={"withheld": True})
    first = client.post(f"/api/assignments/{assignment.id}/release", json={"include_unseen": True}).json()
    assert [r["submission_id"] for r in first["released"]] == [subs[0].id]
    assert {r["reason"] for r in first["held"]} == {"not approved", "withheld"}
    client.post(f"/api/submissions/{subs[2].id}/review/approve")
    second = client.post(f"/api/assignments/{assignment.id}/release", json={}).json()
    assert [r["submission_id"] for r in second["released"]] == [subs[2].id]
    assert second["released"][0]["opened_before_release"] is True
    assert second["already_released"] == [subs[0].id]


def test_release_can_target_submissions(db, client):
    assignment, subs = _graded_assignment(db)
    _accept(db)
    for s in subs:
        client.post(f"/api/submissions/{s.id}/review/approve")
    only = client.post(
        f"/api/assignments/{assignment.id}/release", json={"submission_ids": [subs[1].id]}
    ).json()
    assert [r["submission_id"] for r in only["released"]] == [subs[1].id]
    assert only["counts"]["released"] == 1 and only["counts"]["approved"] == 2


def test_withholding_a_released_result_pulls_it_back(db, client):
    assignment, subs = _graded_assignment(db, n=1)
    _accept(db)
    client.post(f"/api/submissions/{subs[0].id}/review/approve")
    client.post(f"/api/assignments/{assignment.id}/release", json={})
    held = client.post(f"/api/submissions/{subs[0].id}/review/withhold", json={"withheld": True}).json()
    assert held["review"]["state"] == "withheld" and held["review"]["released_at"] is None
    assert client.get(f"/api/assignments/{assignment.id}/export?format=csv").status_code == 409


# --------------------------------------------------------------------------
# terms gate on grading
# --------------------------------------------------------------------------


def test_cloud_grading_is_gated_on_terms_but_mock_is_not(db, client):
    cloud, cloud_subs = _graded_assignment(db, provider="anthropic")
    blocked = client.post(f"/api/assignments/{cloud.id}/grade-all", json={"force": True})
    assert blocked.status_code == 409 and "terms" in blocked.json()["detail"].lower()
    single = client.post(f"/api/submissions/{cloud_subs[0].id}/grade?force=true")
    assert single.status_code == 409

    local, local_subs = _graded_assignment(db, provider="mock", n=1)
    ok = client.post(f"/api/assignments/{local.id}/grade-all", json={"force": True})
    assert ok.status_code == 200 and ok.json()["queued"] == 1

    _accept(db)
    allowed = client.post(f"/api/assignments/{cloud.id}/grade-all", json={"force": True})
    assert allowed.status_code == 200


def test_terms_api_and_page(db, client):
    before = client.get("/api/terms/status").json()
    assert before["accepted"] is False and before["version"] == terms.TERMS_VERSION
    page = client.get("/terms")
    assert page.status_code == 200 and "Accept terms" in page.text
    assert terms.SYLLABUS_STATEMENT[:40] in page.text

    accepted = client.post("/api/terms/accept", json={"signed_by": "Dr Test", "agree": True})
    assert accepted.status_code == 201 and accepted.json()["accepted"] is True
    again = client.post("/api/terms/accept", json={})
    assert again.json()["id"] == accepted.json()["id"]  # idempotent per version
    assert db.scalars(select(Acceptance)).all()[0].signed_by == "Dr Test"
    assert "accepted" in client.get("/terms").text
    assert client.get("/api/terms").json()["disclosure"] == terms.AI_DISCLOSURE


def test_old_acceptance_does_not_count(db, client):
    db.add(Acceptance(kind="terms", version="2000-01-01", signed_by="x"))
    db.commit()
    status = client.get("/api/terms/status").json()
    assert status["accepted"] is False and status["previous_version"] == "2000-01-01"


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def test_export_only_released_rows_with_disclosure(db, client):
    assignment, subs = _graded_assignment(db)
    _accept(db)
    assert client.get(f"/api/assignments/{assignment.id}/export").status_code == 409  # nothing released
    client.post(f"/api/submissions/{subs[0].id}/review/approve")
    client.post(f"/api/submissions/{subs[2].id}/review/approve")
    client.post(f"/api/submissions/{subs[2].id}/review/withhold", json={"withheld": True})
    client.post(f"/api/assignments/{assignment.id}/release", json={})

    resp = client.get(f"/api/assignments/{assignment.id}/export?format=csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'filename="essay-1-feedback-' in resp.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig"))))
    assert [r["student_name"] for r in rows] == ["Student 1"]
    row = rows[0]
    assert row["score"] == "13" and row["max_score"] == "20" and row["percent"] == "65"
    assert row["ai_disclosure"] == terms.AI_DISCLOSURE
    assert row["feedback"].endswith(terms.AI_DISCLOSURE)
    assert "Thesis 7/10: ok" in row["criteria"]
    assert row["email"] == "s1@x.edu"

    unknown = client.get(f"/api/assignments/{assignment.id}/export?format=docx")
    assert unknown.status_code == 409
    assert client.get("/api/exporters").json()["exporters"][0]["id"] == "csv"

    record = client.get(f"/api/assignments/{assignment.id}/release/record").json()
    assert len(record["rows"]) == 1 and record["rows"][0]["disclosure"] == terms.AI_DISCLOSURE


def test_export_requires_terms(db, client):
    assignment, subs = _graded_assignment(db, n=1)
    assert client.get(f"/api/assignments/{assignment.id}/export").status_code == 409
    assert client.get(f"/api/assignments/{assignment.id}/release/record").status_code == 409


# --------------------------------------------------------------------------
# page + status surfaces
# --------------------------------------------------------------------------


def test_grading_page_and_status_carry_review_state(db, client):
    assignment, subs = _graded_assignment(db)
    client.post(f"/api/submissions/{subs[0].id}/review/seen")
    html = client.get(f"/grading/{assignment.id}").text
    assert "Release feedback" in html
    assert 'data-review="seen"' in html and 'data-review="unreviewed"' in html
    assert "Approve and next" in html and "Withhold" in html
    assert 'id="release-dialog"' in html
    status = client.get(f"/api/assignments/{assignment.id}/grading/status").json()
    states = {row["id"]: row["review_state"] for row in status["submissions"]}
    assert states[subs[0].id] == "seen" and states[subs[1].id] == "unreviewed"
    result = client.get(f"/api/submissions/{subs[0].id}/result").json()
    assert result["review_state"] == "seen" and result["result"]["review"]["edit_count"] == 0
