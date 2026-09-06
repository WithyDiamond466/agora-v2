"""No fixed default model: provider resolution follows the professor's keys."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config, security
from app import skills_service as svc
from app.ai import grading as engine
from app.ai.providers import ProviderConfigError, get_provider, resolve_provider
from app.db import get_db
from app.main import app
from app.models import Assignment, Base, Course, Rubric, Skill, Student, Submission


@pytest.fixture()
def db(tmp_path):
    engine_ = create_engine(
        f"sqlite:///{tmp_path / 'resolve.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine_)
    Session = sessionmaker(bind=engine_, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine_.dispose()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "MODEL_REGISTRY_PATH", data_dir / "model_registry.json")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config, "SECRET_KEY_PATH", tmp_path / "cfg" / "secret.key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGORA_AI_PROVIDER", raising=False)
    # The local model counts as "available" only when enabled; start disabled.
    config.save_privacy_settings({"local_model": {"enabled": False}})
    security.reset_cache()
    yield
    security.reset_cache()


@pytest.fixture()
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_nothing_available_is_reported_not_raised(db):
    r = resolve_provider(db, "auto")
    assert r.available is False and r.provider == config.DEFAULT_PROVIDER
    assert "add an anthropic or openai api key" in r.note.lower()
    pinned = resolve_provider(db, "openai", "gpt-5.6-sol")
    assert pinned.available is False and (pinned.provider, pinned.model) == ("openai", "gpt-5.6-sol")
    assert "no api key configured for openai" in pinned.note.lower()
    # Using it is what raises, with the same actionable message.
    with pytest.raises(ProviderConfigError) as exc:
        get_provider({"provider": "auto"}, db=db)
    assert "add an anthropic or openai api key" in str(exc.value).lower()
    with pytest.raises(ProviderConfigError):
        get_provider({"provider": "openai"}, db=db)


def test_auto_takes_the_first_provider_with_a_key(db, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    r = resolve_provider(db, "auto")
    assert (r.provider, r.model) == ("openai", config.default_model_for("openai"))
    assert r.substituted and "first provider with a key" in (r.note or "")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    r2 = resolve_provider(db, "auto")
    assert r2.provider == "anthropic" and r2.model == config.default_model_for("anthropic")
    assert r2.note is None  # first in the order, nothing to explain


def test_preferred_provider_wins_when_it_has_a_key(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    config.set_preferred_provider("openai")
    r = resolve_provider(db, "auto")
    assert r.provider == "openai" and r.note is None

    monkeypatch.delenv("OPENAI_API_KEY")
    r2 = resolve_provider(db, "auto")
    assert r2.provider == "anthropic"
    assert "preferred provider (openai)" in r2.note


def test_pinned_provider_is_never_swapped(db, monkeypatch):
    """A pinned provider means it: no silent substitution, just a clear error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    r = resolve_provider(db, "openai", "gpt-5.6-sol")
    assert (r.provider, r.model, r.available) == ("openai", "gpt-5.6-sol", False)
    assert r.requested_provider == "openai" and not r.substituted
    assert "set the skill to automatic" in r.note


def test_pinned_provider_with_key_keeps_its_model(db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    r = resolve_provider(db, "anthropic", "claude-sonnet-5")
    assert (r.provider, r.model, r.note) == ("anthropic", "claude-sonnet-5", None)
    auto_model = resolve_provider(db, "anthropic", "auto")
    assert auto_model.model == config.default_model_for("anthropic")


def test_mock_and_local_are_always_themselves(db):
    assert resolve_provider(db, "mock").model == config.MOCK_MODEL
    config.save_privacy_settings({"local_model": {"enabled": True}})
    local = resolve_provider(db, "local")
    assert local.provider == "local" and local.available
    # A 4B local model is never a silent stand-in for a frontier one...
    assert resolve_provider(db, "auto").available is False
    # ...unless the professor makes it the preferred provider.
    config.set_preferred_provider("local")
    chosen = resolve_provider(db, "auto")
    assert chosen.provider == "local" and chosen.available and chosen.note is None


def test_force_env_overrides_everything(db, monkeypatch):
    monkeypatch.setenv("AGORA_AI_PROVIDER", "mock")
    r = resolve_provider(db, "anthropic", "claude-opus-5")
    assert (r.provider, r.model) == ("mock", config.MOCK_MODEL)


def test_get_provider_understands_auto(db, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    built = get_provider({"provider": "auto", "model": "auto"}, db=db)
    assert built.name == "openai" and built.model == config.default_model_for("openai")


def test_grade_request_uses_the_resolved_provider(db, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "up")
    course = Course(name="C")
    db.add(course)
    db.flush()
    rubric = Rubric(name="R", criteria=[{"key": "k", "title": "K", "max_points": 5}])
    skill = svc.create_skill(db, name="Auto skill")  # provider auto by default
    db.add(rubric)
    db.flush()
    student = Student(course_id=course.id, name="A", student_number=1)
    assignment = Assignment(course_id=course.id, name="A1", rubric_id=rubric.id, skill_id=skill.id)
    db.add_all([student, assignment])
    db.flush()
    path = tmp_path / "essay.txt"
    path.write_text("An essay by Student-01.")
    sub = Submission(
        assignment_id=assignment.id, student_id=student.id, file_path=str(path),
        original_filename="essay.txt", mime_type="text/plain",
    )
    db.add(sub)
    db.commit()
    request = engine.build_grade_request(db, sub)
    assert request.provider == "openai"
    assert request.model == config.default_model_for("openai")
    assert any("first provider with a key" in n for n in request.notes)


def test_skill_defaults_and_updates_keep_auto_coherent(db):
    skill = svc.create_skill(db, name="S")
    assert (skill.provider, skill.model) == ("auto", "auto")
    pinned = svc.update_skill(db, skill.id, provider="anthropic", model="claude-sonnet-5")
    assert (pinned.provider, pinned.model) == ("anthropic", "claude-sonnet-5")
    back = svc.update_skill(db, skill.id, provider="auto")
    assert (back.provider, back.model) == ("auto", "auto")
    manifest = svc.build_manifest(back)
    assert manifest["provider"] == "auto" and manifest["model"] == "auto"


def test_settings_preferred_provider_api(db, client):
    assert client.get("/api/settings").json()["preferred_provider"] is None
    saved = client.post("/api/settings/keys", json={"provider": "openai", "api_key": "sk-openai-first"})
    assert saved.status_code == 201 and saved.json()["preferred_provider"] == "openai"
    # A second key does not steal the preference.
    client.post("/api/settings/keys", json={"provider": "anthropic", "api_key": "sk-ant-second"})
    status = client.get("/api/settings").json()
    assert status["preferred_provider"] == "openai"
    assert {p["provider"]: p["preferred"] for p in status["providers"]}["openai"] is True

    changed = client.post("/api/settings/preferred-provider", json={"provider": "anthropic"})
    assert changed.json()["preferred_provider"] == "anthropic"
    assert client.post("/api/settings/preferred-provider", json={"provider": "skynet"}).status_code == 400
    cleared = client.post("/api/settings/preferred-provider", json={"provider": ""})
    assert cleared.json()["preferred_provider"] is None
    page = client.get("/settings")
    assert page.status_code == 200 and "Preferred provider" in page.text


def test_test_drive_reports_resolution(db, client, monkeypatch):
    skill = svc.create_skill(db, name="Drive")
    none = client.post(f"/api/skills/{skill.id}/try", json={"message": "hi"}).json()
    assert none["used_mock"] is True and none["requested_provider"] == "auto"
    assert "add an anthropic or openai api key" in (none["note"] or "").lower()
    page = client.get(f"/skills/{skill.id}")
    assert page.status_code == 200 and "Automatic" in page.text
