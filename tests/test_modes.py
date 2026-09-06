"""Skill modes: grade / feedback / selective built in, professor-added on disk."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config, security
from app import skills_service as svc
from app.ai import grading as engine
from app.ai import modes
from app.ai.providers import MockProvider
from app.db import get_db
from app.main import app
from app.models import Assignment, Base, Course, GradeResult, Rubric, Student, Submission
from app.routers import grading as grading_router

CRITERIA = [
    {"key": "thesis", "title": "Thesis", "description": "States a thesis.", "max_points": 10},
    {"key": "evidence", "title": "Evidence", "description": "Uses sources.", "max_points": 10},
    {"key": "mechanics", "title": "Mechanics", "description": "Prose.", "max_points": 5},
]


@pytest.fixture()
def Session(tmp_path):
    engine_ = create_engine(
        f"sqlite:///{tmp_path / 'modes.db'}", connect_args={"check_same_thread": False}
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
    monkeypatch.setattr(config, "EXPORT_DIR", data_dir / "exports")
    monkeypatch.setattr(config, "MODEL_REGISTRY_PATH", data_dir / "model_registry.json")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config, "SECRET_KEY_PATH", tmp_path / "cfg" / "secret.key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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


def _setup(db, tmp_path, *, mode="grade", ai_criteria=None):
    course = Course(name="C")
    db.add(course)
    db.flush()
    rubric = Rubric(name="R", criteria=CRITERIA)
    db.add(rubric)
    db.flush()
    skill = svc.create_skill(db, name=f"S-{mode}", provider="mock", mode=mode)
    student = Student(course_id=course.id, name="Ada", student_number=1)
    assignment = Assignment(
        course_id=course.id, name="A1", rubric_id=rubric.id, skill_id=skill.id, ai_criteria=ai_criteria
    )
    db.add_all([student, assignment])
    db.flush()
    path = tmp_path / f"essay-{mode}.txt"
    path.write_text("A thesis. Some evidence. Clean prose.")
    sub = Submission(
        assignment_id=assignment.id, student_id=student.id, file_path=str(path),
        original_filename="essay.txt", mime_type="text/plain",
    )
    db.add(sub)
    db.commit()
    return assignment, sub


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_builtin_modes_and_schemas():
    ids = [m.id for m in modes.list_modes()]
    assert ids[:3] == ["grade", "feedback", "selective"]
    grade = modes.get_mode("grade")
    assert grade.schema() == engine.GRADE_SCHEMA
    feedback = modes.get_mode("feedback")
    item = feedback.schema()["properties"]["criteria"]["items"]
    assert "score" not in item["properties"] and item["required"] == ["key", "comment"]
    assert modes.get_mode("no-such-mode").id == "grade"  # unknown → default, not a crash
    assert modes.get_mode(None).id == "grade"


def test_custom_mode_round_trips_through_disk():
    spec = {
        "id": "peer-review",
        "label": "Peer-review notes",
        "description": "Questions for the seminar.",
        "scored": False,
        "instructions": "Write three discussion questions the student's essay raises, addressed to the seminar.",
    }
    saved = modes.save_custom_mode(spec)
    assert (modes.custom_modes_dir() / "peer-review.json").is_file()
    assert saved.scored is False and saved.selective is False and not saved.builtin
    assert modes.get_mode("peer-review").label == "Peer-review notes"
    assert modes.is_known_mode("PEER-REVIEW")
    assert modes.delete_custom_mode("peer-review") is True
    assert not modes.is_known_mode("peer-review")


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"id": "grade", "label": "x", "instructions": "y" * 30}, "built-in"),
        ({"id": "Bad Id", "label": "x", "instructions": "y" * 30}, "'id'"),
        ({"id": "ok-id", "label": "", "instructions": "y" * 30}, "label"),
        ({"id": "ok-id", "label": "x", "instructions": "short"}, "instructions"),
        ({"id": "ok-id", "label": "x", "instructions": "y" * 30, "scored": "yes"}, "scored"),
    ],
)
def test_custom_mode_validation(bad, message):
    with pytest.raises(modes.ModeError) as exc:
        modes.validate_mode_spec(bad)
    assert message in str(exc.value)


def test_modes_api(client, db):
    listed = client.get("/api/modes").json()
    assert listed["default"] == "grade" and [m["id"] for m in listed["modes"]][:3] == ["grade", "feedback", "selective"]
    created = client.post(
        "/api/modes",
        json={"id": "socratic", "label": "Socratic", "scored": False,
              "instructions": "Respond only with questions that push the argument further."},
    )
    assert created.status_code == 201 and created.json()["builtin"] is False
    assert client.post("/api/modes", json={"id": "grade", "label": "x", "instructions": "y" * 30}).status_code == 400
    skill = svc.create_skill(db, name="Q", mode="socratic")
    assert skill.mode == "socratic"
    blocked = client.delete("/api/modes/socratic")
    assert blocked.status_code == 409 and "Q" in blocked.json()["detail"]
    svc.update_skill(db, skill.id, mode="feedback")
    assert client.delete("/api/modes/socratic").json()["deleted"] == "socratic"
    assert client.delete("/api/modes/socratic").status_code == 404
    assert client.delete("/api/modes/grade").status_code == 400


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


def test_feedback_mode_keeps_comments_and_drops_scores(db, tmp_path):
    assignment, sub = _setup(db, tmp_path, mode="feedback")
    request = engine.build_grade_request(db, sub)
    assert request.mode == "feedback" and request.scored is False
    assert "Do not assign scores" in request.system_prompt
    assert "Comment on every rubric criterion" in request.system_prompt
    assert "score" not in request.schema["properties"]["criteria"]["items"]["properties"]

    payload = {
        "criteria": [
            {"key": "thesis", "score": 9, "comment": "Clear."},   # a stray score is ignored
            {"key": "evidence", "comment": "Cite the second source."},
        ],
        "summary_feedback": "Good start.",
        "misconceptions": [],
        "strengths": ["clarity"],
    }
    validated = engine.validate_grade_payload(
        payload, CRITERIA, scored=False, mode="feedback"
    )
    assert [c["score"] for c in validated.criteria] == [None, None, None]
    assert validated.criteria[0]["comment"] == "Clear."
    assert (validated.overall_score, validated.max_score, validated.percentage) == (0.0, 0.0, 0.0)
    assert any("skipped criterion 'mechanics'" in a for a in validated.anomalies)
    assert not any("clamped" in a or "recorded as 0" in a for a in validated.anomalies)

    result, _ = engine.grade_submission(db, sub, provider=MockProvider())
    assert result.mode == "feedback" and result.max_score == 0
    assert all(c["score"] is None for c in result.criteria)


def test_selective_mode_splits_criteria_and_marks_manual(db, tmp_path):
    assignment, sub = _setup(db, tmp_path, mode="selective", ai_criteria=["thesis", "evidence"])
    request = engine.build_grade_request(db, sub)
    assert request.mode == "selective" and request.scored is True
    assert [c["key"] for c in request.criteria] == ["thesis", "evidence"]
    assert [c["key"] for c in request.manual_criteria] == ["mechanics"]
    assert "[mechanics]" not in request.system_prompt and "[thesis]" in request.system_prompt
    assert "must not mention or score them" in request.system_prompt

    result, validated = engine.grade_submission(db, sub, provider=MockProvider())
    keys = [c["key"] for c in result.criteria]
    assert keys == ["thesis", "evidence", "mechanics"]
    manual = result.criteria[-1]
    assert manual["manual"] is True and manual["score"] is None and manual["max_points"] == 5
    assert result.max_score == 25  # the whole rubric, so the professor's part shows as missing
    assert result.overall_score == sum(c["score"] for c in result.criteria[:2])
    assert result.mode == "selective"


def test_selective_mode_with_nothing_ticked_sends_everything(db, tmp_path):
    assignment, sub = _setup(db, tmp_path, mode="selective", ai_criteria=None)
    request = engine.build_grade_request(db, sub)
    assert [c["key"] for c in request.criteria] == ["thesis", "evidence", "mechanics"]
    assert request.manual_criteria == []
    assert any("no criteria are ticked" in n for n in request.notes)


def test_grade_mode_unchanged(db, tmp_path):
    assignment, sub = _setup(db, tmp_path, mode="grade", ai_criteria=["thesis"])  # ignored: not selective
    request = engine.build_grade_request(db, sub)
    assert request.schema == engine.GRADE_SCHEMA
    assert len(request.criteria) == 3 and request.manual_criteria == []
    result, _ = engine.grade_submission(db, sub, provider=MockProvider())
    assert result.mode == "grade" and result.max_score == 25


# --------------------------------------------------------------------------
# API + pages
# --------------------------------------------------------------------------


def test_assignment_ai_criteria_api(client, db, tmp_path):
    assignment, sub = _setup(db, tmp_path, mode="selective")
    bad = client.patch(f"/api/assignments/{assignment.id}", json={"ai_criteria": ["thesis", "nope"]})
    assert bad.status_code == 400 and "nope" in bad.json()["detail"]
    ok = client.patch(f"/api/assignments/{assignment.id}", json={"ai_criteria": ["evidence", "thesis", "evidence"]})
    assert ok.json()["ai_criteria"] == ["evidence", "thesis"]
    cleared = client.patch(f"/api/assignments/{assignment.id}", json={"ai_criteria": []})
    assert cleared.json()["ai_criteria"] is None
    page = client.get(f"/grading/{assignment.id}").text
    assert "Criteria the AI grades" in page and 'name="ai_criteria"' in page
    assert "AI selective grading" in page


def test_manual_scores_can_be_saved_and_feedback_totals_stay_zero(client, db, tmp_path):
    assignment, sub = _setup(db, tmp_path, mode="selective", ai_criteria=["thesis"])
    engine.grade_submission(db, sub, provider=MockProvider())
    db.refresh(sub)
    before = client.get(f"/api/submissions/{sub.id}/result").json()["result"]
    assert before["review"]["needs_manual_scores"] == 2
    saved = client.patch(
        f"/api/submissions/{sub.id}/result",
        json={"criteria": [{"key": "evidence", "score": 8}, {"key": "mechanics", "score": 99}]},
    ).json()["result"]
    by_key = {c["key"]: c for c in saved["criteria"]}
    assert by_key["evidence"]["score"] == 8 and by_key["mechanics"]["score"] == 5  # clamped
    assert saved["review"]["needs_manual_scores"] == 0
    assert saved["overall_score"] == by_key["thesis"]["score"] + 13

    fb_assignment, fb_sub = _setup(db, tmp_path, mode="feedback")
    engine.grade_submission(db, fb_sub, provider=MockProvider())
    edited = client.patch(
        f"/api/submissions/{fb_sub.id}/result", json={"criteria": [{"key": "thesis", "score": 10}]}
    ).json()["result"]
    assert (edited["overall_score"], edited["max_score"]) == (0, 0)
    page = client.get(f"/grading/{fb_assignment.id}").text
    assert "feedback only" in page and "unscored" in page


def test_skill_mode_api_and_bundle_round_trip(client, db):
    created = client.post("/api/skills", json={"name": "FB", "mode": "feedback"}).json()
    assert created["mode"] == "feedback" and created["mode_label"] == "AI feedback"
    assert client.post("/api/skills", json={"name": "X", "mode": "bogus"}).status_code == 400
    patched = client.patch(f"/api/skills/{created['id']}", json={"mode": "selective"}).json()
    assert patched["mode"] == "selective"
    page = client.get(f"/skills/{created['id']}").text
    assert 'name="mode"' in page and "AI selective grading" in page

    # A custom mode travels inside the bundle and is learned on import.
    modes.save_custom_mode({
        "id": "seminar", "label": "Seminar prep", "scored": False,
        "instructions": "Write two questions for the seminar based on the essay's weakest step.",
    })
    skill = svc.create_skill(db, name="Sem", mode="seminar")
    name, payload = svc.export_skill_bytes(db, skill.id)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["mode"] == "seminar" and manifest["mode_spec"]["scored"] is False

    modes.delete_custom_mode("seminar")
    assert not modes.is_known_mode("seminar")
    imported = svc.import_skill(db, payload)
    assert imported.mode == "seminar" and modes.is_known_mode("seminar")

    # A bundle naming a mode nobody defined falls back to grade.
    manifest2 = dict(manifest, mode="mystery", mode_spec=None)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest2))
        zf.writestr("system_prompt.md", "x")
    assert svc.import_skill(db, buf.getvalue()).mode == "grade"


def test_insight_ignores_unscored_criteria(db, tmp_path):
    from app.insight import derive_observations

    assignment, sub = _setup(db, tmp_path, mode="feedback")
    result, _ = engine.grade_submission(db, sub, provider=MockProvider())
    made = derive_observations(db, sub, result) if callable(derive_observations) else []
    assert not [o for o in made if o.kind in ("criterion_low", "criterion_high")]
