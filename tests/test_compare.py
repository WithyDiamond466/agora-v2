"""Model comparison: same question or same submission, several models, nothing saved."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config, security, terms
from app import skills_service as svc
from app.db import get_db
from app.main import app
from app.models import Assignment, Base, Course, Rubric, Student, Submission
from app.routers import grading as grading_router

CRITERIA = [
    {"key": "thesis", "title": "Thesis", "description": "", "max_points": 10},
    {"key": "evidence", "title": "Evidence", "description": "", "max_points": 10},
]


@pytest.fixture()
def Session(tmp_path):
    engine_ = create_engine(
        f"sqlite:///{tmp_path / 'cmp.db'}", connect_args={"check_same_thread": False}
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


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "UPLOAD_DIR", data_dir / "submissions")
    monkeypatch.setattr(config, "MODEL_REGISTRY_PATH", data_dir / "model_registry.json")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config, "SECRET_KEY_PATH", tmp_path / "cfg" / "secret.key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGORA_AI_PROVIDER", raising=False)
    security.reset_cache()
    yield
    security.reset_cache()


@pytest.fixture()
def client(Session, db, monkeypatch):
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


def _submission(db, tmp_path):
    course = Course(name="C")
    db.add(course)
    db.flush()
    rubric = Rubric(name="R", criteria=CRITERIA)
    db.add(rubric)
    db.flush()
    skill = svc.create_skill(db, name="S", provider="mock")
    student = Student(course_id=course.id, name="Ada", student_number=1)
    assignment = Assignment(course_id=course.id, name="A1", rubric_id=rubric.id, skill_id=skill.id)
    db.add_all([student, assignment])
    db.flush()
    path = tmp_path / "essay.txt"
    path.write_text("A thesis with evidence.")
    sub = Submission(
        assignment_id=assignment.id, student_id=student.id, file_path=str(path),
        original_filename="essay.txt", mime_type="text/plain",
    )
    db.add(sub)
    db.commit()
    return skill, assignment, sub


def test_candidates_list_availability(client, db, monkeypatch):
    before = client.get("/api/compare/candidates").json()["candidates"]
    by = {(c["provider"], c["model"]): c for c in before}
    assert by[("auto", "auto")]["available"] is False
    assert by[("anthropic", "claude-opus-5")]["available"] is False
    assert by[("mock", config.MOCK_MODEL)]["available"] is True
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    after = {(c["provider"], c["model"]): c for c in client.get("/api/compare/candidates").json()["candidates"]}
    assert after[("anthropic", "claude-opus-5")]["available"] is True
    assert after[("auto", "auto")]["available"] is True and "anthropic" in after[("auto", "auto")]["label"]


def test_skill_compare_runs_each_candidate(client, db):
    skill = svc.create_skill(db, name="S", system_prompt="Be terse.")
    resp = client.post(
        f"/api/skills/{skill.id}/compare",
        json={"message": "How strict are you?", "candidates": [{"provider": "mock"}, {"provider": "openai"}]},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[0]["provider"] == "mock" and results[0]["reply"]
    assert results[1]["used_mock"] is True and "no api key configured for openai" in results[1]["note"].lower()
    assert all(isinstance(r["elapsed_ms"], int) for r in results)
    assert client.post(f"/api/skills/{skill.id}/compare", json={"message": "x", "candidates": []}).status_code == 422


def test_submission_compare_grades_without_persisting(client, db, tmp_path):
    skill, assignment, sub = _submission(db, tmp_path)
    resp = client.post(
        f"/api/submissions/{sub.id}/compare",
        json={"candidates": [{"provider": "mock"}, {"provider": "openai", "model": "gpt-5.6-sol"}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["persisted"] is False
    mock, openai = body["results"]
    assert mock["provider"] == "mock" and mock["error"] is None
    assert [c["key"] for c in mock["grade"]["criteria"]] == ["thesis", "evidence"]
    assert mock["grade"]["max_score"] == 20 and mock["grade"]["mode"] == "grade"
    assert openai["grade"] is None and "no api key configured for openai" in openai["error"].lower()
    db.expire_all()
    assert db.get(Submission, sub.id).grade_result is None
    assert db.get(Submission, sub.id).status == "pending"


def test_submission_compare_gates_cloud_on_terms(client, db, tmp_path, monkeypatch):
    skill, assignment, sub = _submission(db, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    blocked = client.post(
        f"/api/submissions/{sub.id}/compare", json={"candidates": [{"provider": "openai"}]}
    )
    assert blocked.status_code == 409 and "/terms" in blocked.json()["detail"]
    # Mock-only comparisons are not gated.
    assert client.post(f"/api/submissions/{sub.id}/compare", json={"candidates": [{"provider": "mock"}]}).status_code == 200
    terms.accept(db)
    allowed = client.post(
        f"/api/submissions/{sub.id}/compare", json={"candidates": [{"provider": "openai"}]}
    )
    # Accepted terms let it run; with a fake key the provider call fails cleanly per candidate.
    assert allowed.status_code == 200
    entry = allowed.json()["results"][0]
    assert entry["provider"] == "openai" and (entry["error"] or entry["grade"])


def test_compare_requires_a_mapped_student(client, db, tmp_path):
    skill, assignment, sub = _submission(db, tmp_path)
    orphan = Submission(assignment_id=assignment.id, original_filename="x.txt", mime_type="text/plain", file_path=sub.file_path)
    db.add(orphan)
    db.commit()
    assert client.post(f"/api/submissions/{orphan.id}/compare", json={"candidates": [{"provider": "mock"}]}).status_code == 409
    assert client.post("/api/submissions/99999/compare", json={"candidates": [{"provider": "mock"}]}).status_code == 404


def test_pages_render_compare_ui(client, db, tmp_path):
    skill, assignment, sub = _submission(db, tmp_path)
    skill_page = client.get(f"/skills/{skill.id}").text
    assert "Compare models" in skill_page and 'id="candidate-json"' in skill_page
    assert "/static/js/compare.js" in skill_page
    grading_page = client.get(f"/grading/{assignment.id}").text
    assert 'id="compare-dialog"' in grading_page and "Mock grader" in grading_page
