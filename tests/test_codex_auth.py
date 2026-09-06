"""OpenAI login via the Codex CLI: import an API key when that is what the login holds."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import codex_auth, config, security
from app.db import get_db
from app.main import app
from app.models import Base


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'codex.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "MODEL_REGISTRY_PATH", data_dir / "model_registry.json")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config, "SECRET_KEY_PATH", tmp_path / "cfg" / "secret.key")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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


def _write(tmp_path, data):
    home = tmp_path / "codex"
    home.mkdir(exist_ok=True)
    (home / "auth.json").write_text(json.dumps(data))


def _jwt(claims):
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJub25lIn0.{body}.sig"


def test_missing_login(client):
    info = client.get("/api/settings/openai-login").json()
    assert info["state"] == "missing" and "codex login" in info["message"]
    assert client.post("/api/settings/openai-login/import").status_code == 409


def test_chatgpt_login_is_recognised_but_never_used(client, tmp_path, db):
    _write(tmp_path, {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {"id_token": _jwt({"email": "prof@example.edu"}), "access_token": "secret-access",
                   "refresh_token": "secret-refresh", "account_id": "acct"},
    })
    info = client.get("/api/settings/openai-login").json()
    assert info["state"] == "chatgpt" and info["email"] == "prof@example.edu"
    assert "Codex only" in info["message"] and info["api_keys_url"].startswith("https://platform.openai.com")
    assert "secret" not in json.dumps(info)
    blocked = client.post("/api/settings/openai-login/import")
    assert blocked.status_code == 409
    assert client.get("/api/settings/keys").json() == []
    page = client.get("/settings").text
    assert "Log in with your OpenAI account" in page and "prof@example.edu" in page
    assert "Create an API key" in page and "secret-access" not in page


def test_apikey_login_is_imported_encrypted(client, tmp_path, db):
    _write(tmp_path, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-proj-codex-imported-123456", "tokens": None})
    info = client.get("/api/settings/openai-login").json()
    assert info["state"] == "apikey"
    page = client.get("/settings").text
    assert "Use the key from my Codex login" in page
    imported = client.post("/api/settings/openai-login/import")
    assert imported.status_code == 201, imported.text
    body = imported.json()
    assert body["provider"] == "openai" and body["label"] == "Codex CLI login"
    assert body["masked_key"].endswith("3456") and "codex-imported" not in body["masked_key"]
    assert body["preferred_provider"] == "openai"
    assert security.get_api_key(db, "openai") == "sk-proj-codex-imported-123456"
    # Importing again replaces the active key rather than stacking a duplicate default.
    again = client.post("/api/settings/openai-login/import").json()
    creds = client.get("/api/settings/keys").json()
    assert [c["active"] for c in creds if c["provider"] == "openai"].count(True) == 1
    assert again["id"] != body["id"]


def test_unreadable_file(client, tmp_path):
    home = tmp_path / "codex"
    home.mkdir()
    (home / "auth.json").write_text("{not json")
    assert client.get("/api/settings/openai-login").json()["state"] == "unreadable"
    assert codex_auth.api_key() is None
