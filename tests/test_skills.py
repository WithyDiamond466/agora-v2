"""Skills module tests: CRUD, knowledge docs, bundle round-trip, key encryption.

Everything runs against a tmp SQLite DB and a tmp data dir — the real
``data/`` tree and ``~/.config/agora`` are never touched, and no network call
is ever made (the test drive falls back to the mock path).
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from app import config, security
from app import skills_service as svc
from app.db import get_db
from app.main import app
from app.models import ApiCredential, Base, KnowledgeDoc, Skill

BUNDLE_MIME = "application/zip"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'skills.db'}", connect_args={"check_same_thread": False}
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
def isolated_paths(tmp_path, monkeypatch):
    """Relocate DATA_DIR (skill docs, exports) and the secret key."""
    data_dir = tmp_path / "data"
    (data_dir / "exports").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
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
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def assert_no_delete_backups(root):
    if root.exists():
        assert not [path for path in root.rglob("*") if ".agora-delete-" in path.name]


def make_skill(db, name="Diff-Eq Grader", **kw) -> Skill:
    return svc.create_skill(
        db,
        name=name,
        description=kw.pop("description", "Grades MA232 problem sets."),
        system_prompt=kw.pop("system_prompt", "You are a strict but fair grader.\nBe specific."),
        **kw,
    )


# --------------------------------------------------------------------------
# CRUD (service + API)
# --------------------------------------------------------------------------


def test_create_update_delete_skill_service(db):
    skill = make_skill(db)
    assert skill.id
    # No fixed default model: a new skill resolves its provider at grade time.
    assert skill.provider == config.AUTO_PROVIDER
    assert skill.model == config.AUTO_MODEL
    assert skill.max_tokens == config.DEFAULT_MAX_TOKENS

    updated = svc.update_skill(
        db, skill.id, name="Diff-Eq Grader v4", model="claude-sonnet-5", max_tokens=99
    )
    assert updated.name == "Diff-Eq Grader v4"
    assert updated.model == "claude-sonnet-5"
    assert updated.max_tokens == 256  # clamped to the floor

    with pytest.raises(svc.SkillError):
        svc.create_skill(db, name="   ")
    with pytest.raises(svc.SkillNotFound):
        svc.get_skill(db, 99999)

    assert [s.id for s in svc.list_skills(db)] == [skill.id]
    svc.delete_skill(db, skill.id)
    assert svc.list_skills(db) == []


def test_skill_crud_endpoints(client, db):
    created = client.post(
        "/api/skills",
        json={
            "name": "Essay Grader",
            "description": "PHIL 101 essays",
            "system_prompt": "Grade the essay.",
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
        },
    )
    assert created.status_code == 201, created.text
    payload = created.json()
    assert payload["model"] == "claude-haiku-4-5"
    assert payload["docs"] == []

    listed = client.get("/api/skills").json()
    assert [s["name"] for s in listed] == ["Essay Grader"]

    patched = client.patch(
        f"/api/skills/{payload['id']}", json={"description": "Updated"}
    ).json()
    assert patched["description"] == "Updated"
    # A field that was not sent must be left alone.
    assert patched["system_prompt"] == "Grade the essay."

    assert client.get("/api/skills/424242").status_code == 404
    assert client.delete(f"/api/skills/{payload['id']}").status_code == 200
    assert client.get("/api/skills").json() == []


# --------------------------------------------------------------------------
# knowledge docs
# --------------------------------------------------------------------------


def test_add_knowledge_doc_sanitizes_and_validates(db):
    skill = make_skill(db)

    doc = svc.add_knowledge_doc(db, skill.id, "../../../etc/passwd.md", b"# syllabus\nbe kind")
    assert doc.filename == "passwd.md"
    stored = Path(doc.file_path)
    assert stored.parent == svc.skill_dir(skill.id)
    assert stored.read_bytes() == b"# syllabus\nbe kind"
    assert doc.mime_type == "text/markdown"
    assert doc.size_bytes == len(b"# syllabus\nbe kind")

    # Windows-style separators and nasty characters are stripped too.
    doc2 = svc.add_knowledge_doc(db, skill.id, r"C:\Users\prof\notes;rm -rf.txt", b"notes")
    assert "/" not in doc2.filename and "\\" not in doc2.filename
    assert doc2.filename.endswith(".txt")

    # Same name twice does not clobber.
    doc3 = svc.add_knowledge_doc(db, skill.id, "passwd.md", b"second")
    assert doc3.filename != doc.filename
    assert Path(doc.file_path).read_bytes() == b"# syllabus\nbe kind"

    with pytest.raises(svc.UnsupportedDocument):
        svc.add_knowledge_doc(db, skill.id, "malware.exe", b"MZ")
    with pytest.raises(svc.UnsupportedDocument):
        svc.add_knowledge_doc(db, skill.id, "huge.pdf", b"x" * (svc.MAX_DOC_BYTES + 1))
    with pytest.raises(svc.UnsupportedDocument):
        svc.add_knowledge_doc(db, skill.id, "empty.txt", b"")

    assert len(svc.list_docs(db, skill.id)) == 3
    svc.delete_knowledge_doc(db, doc.id, skill.id)
    assert not stored.exists()
    assert len(svc.list_docs(db, skill.id)) == 2
    assert_no_delete_backups(svc.skills_root())


def test_doc_upload_endpoints(client, db):
    skill = make_skill(db)

    resp = client.post(
        f"/api/skills/{skill.id}/docs",
        files={"file": ("rubric.md", io.BytesIO(b"# rubric"), "text/markdown")},
        data={"title": "Rubric conventions"},
    )
    assert resp.status_code == 201, resp.text
    doc = resp.json()
    assert doc["title"] == "Rubric conventions"

    bad = client.post(
        f"/api/skills/{skill.id}/docs",
        files={"file": ("virus.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert bad.status_code == 400

    download = client.get(f"/api/skills/{skill.id}/docs/{doc['id']}/download")
    assert download.status_code == 200
    assert download.content == b"# rubric"

    listed = client.get(f"/api/skills/{skill.id}/docs").json()
    assert len(listed) == 1
    assert client.delete(f"/api/skills/{skill.id}/docs/{doc['id']}").status_code == 200
    assert client.get(f"/api/skills/{skill.id}/docs").json() == []


def test_deleting_skill_removes_its_directory(db):
    skill = make_skill(db)
    svc.add_knowledge_doc(db, skill.id, "syllabus.txt", b"hello")
    directory = svc.skill_dir(skill.id)
    assert directory.is_dir()

    svc.delete_skill(db, skill.id)
    assert not directory.exists()
    assert db.query(KnowledgeDoc).count() == 0
    assert_no_delete_backups(svc.skills_root())


def test_skill_cleanup_failure_preserves_database_rows(db, monkeypatch):
    skill = make_skill(db)
    doc = svc.add_knowledge_doc(db, skill.id, "syllabus.txt", b"hello")
    stored = Path(doc.file_path)

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(svc, "reversible_delete", fail_cleanup)
    with pytest.raises(OSError, match="cleanup failed"):
        svc.delete_skill(db, skill.id)

    assert db.get(Skill, skill.id) is not None
    assert db.get(KnowledgeDoc, doc.id) is not None
    assert stored.is_file()
    assert_no_delete_backups(svc.skills_root())


def test_knowledge_doc_cleanup_failure_preserves_database_row(db, monkeypatch):
    skill = make_skill(db)
    doc = svc.add_knowledge_doc(db, skill.id, "syllabus.txt", b"hello")
    stored = Path(doc.file_path)

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(svc, "reversible_delete", fail_cleanup)
    with pytest.raises(OSError, match="cleanup failed"):
        svc.delete_knowledge_doc(db, doc.id, skill.id)

    assert db.get(KnowledgeDoc, doc.id) is not None
    assert stored.is_file()
    assert_no_delete_backups(svc.skills_root())


def test_skill_delete_commit_failure_restores_directory_and_rows(db, monkeypatch):
    skill = make_skill(db)
    doc = svc.add_knowledge_doc(db, skill.id, "syllabus.txt", b"hello")
    stored = Path(doc.file_path)
    content = stored.read_bytes()
    original_commit = db.commit

    def fail_commit():
        raise RuntimeError("commit failed")

    monkeypatch.setattr(db, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            svc.delete_skill(db, skill.id)
    finally:
        monkeypatch.setattr(db, "commit", original_commit)

    db.expire_all()
    assert db.get(Skill, skill.id) is not None
    assert db.get(KnowledgeDoc, doc.id) is not None
    assert stored.read_bytes() == content
    assert_no_delete_backups(svc.skills_root())


def test_knowledge_doc_delete_commit_failure_restores_file_and_row(db, monkeypatch):
    skill = make_skill(db)
    doc = svc.add_knowledge_doc(db, skill.id, "syllabus.txt", b"hello")
    stored = Path(doc.file_path)
    content = stored.read_bytes()
    original_commit = db.commit

    def fail_commit():
        raise RuntimeError("commit failed")

    monkeypatch.setattr(db, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            svc.delete_knowledge_doc(db, doc.id, skill.id)
    finally:
        monkeypatch.setattr(db, "commit", original_commit)

    db.expire_all()
    assert db.get(KnowledgeDoc, doc.id) is not None
    assert stored.read_bytes() == content
    assert_no_delete_backups(svc.skills_root())


def test_knowledge_doc_commit_failure_removes_new_file(db, monkeypatch):
    skill = make_skill(db)
    original_commit = db.commit

    def fail_commit():
        raise RuntimeError("commit failed")

    monkeypatch.setattr(db, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            svc.add_knowledge_doc(db, skill.id, "failed.txt", b"never committed")
    finally:
        monkeypatch.setattr(db, "commit", original_commit)

    db.expire_all()
    assert db.query(KnowledgeDoc).filter_by(skill_id=skill.id).count() == 0
    assert not [path for path in svc.skill_dir(skill.id).glob("*") if path.is_file()]
    assert_no_delete_backups(svc.skills_root())


def test_knowledge_doc_partial_write_failure_removes_partial_file(db, monkeypatch):
    skill = make_skill(db)
    original_write_bytes = Path.write_bytes

    def fail_after_partial_write(path, data):
        original_write_bytes(path, data[:5])
        raise OSError("write failed")

    monkeypatch.setattr(Path, "write_bytes", fail_after_partial_write)

    with pytest.raises(OSError, match="write failed"):
        svc.add_knowledge_doc(db, skill.id, "partial.txt", b"partially written")

    assert db.query(KnowledgeDoc).filter_by(skill_id=skill.id).count() == 0
    assert not (svc.skill_dir(skill.id) / "partial.txt").exists()


def test_knowledge_doc_runs_no_fallible_session_work_after_commit(db, monkeypatch):
    skill = make_skill(db)
    data = b"persisted knowledge"
    commit_finished = False
    original_commit = db.commit
    original_expire = db.expire

    def tracked_commit():
        nonlocal commit_finished
        original_commit()
        commit_finished = True

    def pre_commit_expire(*args, **kwargs):
        assert not commit_finished
        return original_expire(*args, **kwargs)

    def fail_refresh(*_args, **_kwargs):
        raise AssertionError("refresh called")

    monkeypatch.setattr(db, "commit", tracked_commit)
    monkeypatch.setattr(db, "expire", pre_commit_expire)
    monkeypatch.setattr(db, "refresh", fail_refresh)

    doc = svc.add_knowledge_doc(db, skill.id, "persisted.txt", data)

    assert commit_finished
    Session = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
    with Session() as fresh_db:
        stored = fresh_db.get(KnowledgeDoc, doc.id)
        assert stored is not None
        assert Path(stored.file_path).read_bytes() == data


def test_knowledge_doc_delete_runs_no_session_work_after_commit(db, monkeypatch):
    skill = make_skill(db)
    doc = svc.add_knowledge_doc(db, skill.id, "deleted.txt", b"delete me")
    doc_id = doc.id
    stored = Path(doc.file_path)
    commit_finished = False
    original_commit = db.commit
    original_expire = db.expire

    def tracked_commit():
        nonlocal commit_finished
        original_commit()
        commit_finished = True

    def pre_commit_expire(*args, **kwargs):
        assert not commit_finished
        return original_expire(*args, **kwargs)

    def fail_refresh(*_args, **_kwargs):
        raise AssertionError("refresh called")

    monkeypatch.setattr(db, "commit", tracked_commit)
    monkeypatch.setattr(db, "expire", pre_commit_expire)
    monkeypatch.setattr(db, "refresh", fail_refresh)

    assert svc.delete_knowledge_doc(db, doc_id, skill.id) == doc_id
    assert commit_finished

    Session = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
    with Session() as fresh_db:
        assert fresh_db.get(KnowledgeDoc, doc_id) is None
    assert not stored.exists()
    assert_no_delete_backups(svc.skills_root())


# --------------------------------------------------------------------------
# export / import round trip
# --------------------------------------------------------------------------


def test_export_bundle_layout(db):
    skill = make_skill(db, name="Diff-Eq Grader v4")
    svc.add_knowledge_doc(db, skill.id, "syllabus.md", b"# MA232 syllabus")
    svc.add_knowledge_doc(db, skill.id, "worked-example.pdf", b"%PDF-1.4 fake")

    path = svc.export_skill(db, skill.id)
    assert path.suffix == ".agoraskill"
    assert path.name == "diff-eq-grader-v4.agoraskill"

    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
        assert names == [
            "docs/syllabus.md",
            "docs/worked-example.pdf",
            "manifest.json",
            "system_prompt.md",
        ]
        manifest = json.loads(zf.read("manifest.json"))
        assert zf.read("system_prompt.md").decode() == skill.system_prompt

    assert manifest["format_version"] == svc.FORMAT_VERSION
    assert manifest["name"] == "Diff-Eq Grader v4"
    assert manifest["description"] == skill.description
    assert manifest["provider"] == skill.provider
    assert manifest["model"] == skill.model
    assert manifest["settings"] == {"max_tokens": skill.max_tokens}


def test_export_import_round_trip(db):
    skill = make_skill(db, name="Diff-Eq Grader v4", model="claude-sonnet-5")
    svc.add_knowledge_doc(db, skill.id, "syllabus.md", b"# MA232 syllabus", title="Syllabus")
    svc.add_knowledge_doc(db, skill.id, "example.pdf", b"%PDF-1.4 fake")
    path = svc.export_skill(db, skill.id)

    imported = svc.import_skill(db, path)
    assert imported.id != skill.id
    # Same name already exists locally, so the copy is disambiguated.
    assert imported.name == "Diff-Eq Grader v4 (imported)"
    assert imported.description == skill.description
    assert imported.system_prompt == skill.system_prompt
    assert imported.provider == skill.provider
    assert imported.model == "claude-sonnet-5"
    assert imported.max_tokens == skill.max_tokens

    names = sorted(d.filename for d in imported.knowledge_docs)
    assert names == ["example.pdf", "syllabus.md"]
    assert {d.title for d in imported.knowledge_docs} >= {"Syllabus"}
    for doc in imported.knowledge_docs:
        stored = Path(doc.file_path)
        assert stored.parent == svc.skill_dir(imported.id)
        assert stored.is_file()
        assert doc.size_bytes == stored.stat().st_size
    syllabus = next(d for d in imported.knowledge_docs if d.filename == "syllabus.md")
    assert Path(syllabus.file_path).read_bytes() == b"# MA232 syllabus"

    # Importing again makes a third distinct skill, never mutates the original.
    third = svc.import_skill(db, path.read_bytes())
    assert third.name == "Diff-Eq Grader v4 (imported 2)"
    assert db.query(Skill).count() == 3
    assert db.get(Skill, skill.id).name == "Diff-Eq Grader v4"


def _bundle(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def _manifest(**overrides) -> bytes:
    base = {
        "format_version": 1,
        "name": "Bundle Skill",
        "description": "d",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "settings": {"max_tokens": 4000},
    }
    base.update(overrides)
    return json.dumps(base).encode()


def test_import_skill_partial_write_failure_is_atomic(db, monkeypatch):
    payload = _bundle(
        {
            "manifest.json": _manifest(),
            "docs/first.txt": b"first document",
            "docs/second.txt": b"second document",
        }
    )
    original_write_bytes = Path.write_bytes
    write_error = OSError("write failed")
    writes = 0

    def fail_second_write(path, data):
        nonlocal writes
        writes += 1
        if writes == 2:
            original_write_bytes(path, data[:5])
            raise write_error
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_second_write)

    with pytest.raises(OSError, match="write failed") as caught:
        svc.import_skill(db, payload)

    assert caught.value is write_error

    assert db.query(Skill).count() == 0
    assert db.query(KnowledgeDoc).count() == 0
    assert not list(svc.skills_root().glob("*"))


def test_import_skill_commit_failure_is_atomic(db, monkeypatch):
    payload = _bundle(
        {
            "manifest.json": _manifest(),
            "docs/imported.txt": b"never committed",
        }
    )
    original_commit = db.commit
    commit_error = RuntimeError("commit failed")

    def fail_commit():
        raise commit_error

    monkeypatch.setattr(db, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failed") as caught:
            svc.import_skill(db, payload)
    finally:
        monkeypatch.setattr(db, "commit", original_commit)

    assert caught.value is commit_error

    Session = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
    with Session() as fresh_db:
        assert fresh_db.query(Skill).count() == 0
        assert fresh_db.query(KnowledgeDoc).count() == 0
    assert not list(svc.skills_root().glob("*"))


def test_import_skill_preserves_preexisting_directory_on_reused_id(db):
    original = make_skill(db, name="Original skill")
    assert original.id == 1
    directory = svc.skill_dir(original.id)
    directory.mkdir(parents=True)
    keeper = directory / "keeper.txt"
    keeper.write_bytes(b"keep this exact content")

    db.execute(delete(Skill).where(Skill.id == original.id))
    db.commit()

    probe = Skill(name="ID probe", provider=config.MOCK_PROVIDER, model=config.MOCK_MODEL)
    db.add(probe)
    db.flush()
    assert probe.id == 1
    db.rollback()

    payload = _bundle(
        {
            "manifest.json": _manifest(),
            "docs/imported.txt": b"must not be written",
        }
    )
    with pytest.raises(svc.SkillError, match="already exists"):
        svc.import_skill(db, payload)

    assert db.query(Skill).count() == 0
    assert db.query(KnowledgeDoc).count() == 0
    assert keeper.read_bytes() == b"keep this exact content"


def test_import_rejects_zip_slip(db, tmp_path):
    """A bundle must never be able to write outside the skill directory."""
    victim = tmp_path / "pwned.md"
    escapes = [
        "docs/../../../../pwned.md",
        "../pwned.md",
        "/etc/pwned.md",
        r"docs\..\..\pwned.md",
        "docs/nested/deeper.md",
    ]
    for entry in escapes:
        payload = _bundle(
            {"manifest.json": _manifest(), "system_prompt.md": b"p", entry: b"owned"}
        )
        with pytest.raises(svc.BundleError):
            svc.import_skill(db, payload)
        assert not victim.exists()

    # Nothing was created for any of the rejected bundles.
    assert db.query(Skill).count() == 0
    assert list(svc.skills_root().rglob("pwned.md")) == []


def test_import_rejects_symlink_members(db, tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("manifest.json", _manifest())
        info = zipfile.ZipInfo("docs/link.md")
        info.external_attr = (0o120777 << 16)  # symlink
        zf.writestr(info, str(tmp_path / "secret.txt"))
    with pytest.raises(svc.BundleError):
        svc.import_skill(db, buffer.getvalue())
    assert db.query(Skill).count() == 0


def test_import_rejects_bad_manifests(db):
    cases = [
        {},  # no manifest.json at all
        {"manifest.json": b"not json"},
        {"manifest.json": b"[1, 2, 3]"},
        {"manifest.json": _manifest(format_version=99)},
        {"manifest.json": _manifest(format_version="banana")},
        {"manifest.json": json.dumps({"name": "x"}).encode()},  # no format_version
        {"manifest.json": _manifest(name="")},
        {"manifest.json": _manifest(settings="lots")},
    ]
    for members in cases:
        with pytest.raises(svc.BundleError):
            svc.import_skill(db, _bundle(members))

    with pytest.raises(svc.BundleError):
        svc.import_skill(db, b"this is not a zip file at all")

    # A disallowed document type inside an otherwise valid bundle is refused.
    with pytest.raises(svc.SkillError):
        svc.import_skill(
            db, _bundle({"manifest.json": _manifest(), "docs/payload.exe": b"MZ"})
        )
    assert db.query(Skill).count() == 0


def test_import_reports_a_corrupt_member_as_a_bundle_error(db, client):
    """A CRC/length mismatch raises inside `zf.read`, not at open time.

    Left uncaught that is an HTTP 500 instead of the 400 the router maps
    BundleError to.
    """
    payload = bytearray(
        _bundle(
            {
                "manifest.json": _manifest(),
                "system_prompt.md": b"prompt",
                "docs/syllabus.md": b"# syllabus contents, long enough to corrupt",
            }
        )
    )
    # Corrupt the stored doc payload so its CRC no longer matches.
    start = payload.index(b"# syllabus contents")
    payload[start : start + 8] = b"XXXXXXXX"

    with pytest.raises(svc.BundleError):
        svc.import_skill(db, bytes(payload))
    assert db.query(Skill).count() == 0

    resp = client.post(
        "/api/skills/import",
        files={"file": ("broken.agoraskill", io.BytesIO(bytes(payload)), "application/zip")},
    )
    assert resp.status_code == 400, resp.text


def test_docx_knowledge_docs_are_rejected_at_upload(db):
    """They were accepted, then silently dropped from every grading request."""
    assert ".docx" not in svc.ALLOWED_DOC_TYPES
    skill = svc.create_skill(db, name="Docx Skill")
    with pytest.raises(svc.UnsupportedDocument):
        svc.add_knowledge_doc(db, skill.id, "conventions.docx", b"PK\x03\x04 not really")


def test_import_normalizes_unknown_provider_and_ignores_dotfiles(db):
    payload = _bundle(
        {
            "manifest.json": _manifest(provider="skynet", model="", settings={}),
            "system_prompt.md": "grade kindly".encode(),
            "docs/.DS_Store": b"junk",
            "docs/notes.txt": b"notes",
        }
    )
    skill = svc.import_skill(db, payload)
    assert skill.provider == config.AUTO_PROVIDER
    assert skill.model == config.AUTO_MODEL
    assert skill.max_tokens == config.DEFAULT_MAX_TOKENS
    assert [d.filename for d in skill.knowledge_docs] == ["notes.txt"]


def test_export_import_endpoints_round_trip(client, db):
    skill = make_skill(db, name="Exportable")
    svc.add_knowledge_doc(db, skill.id, "syllabus.md", b"# syllabus")

    resp = client.get(f"/api/skills/{skill.id}/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(BUNDLE_MIME)
    assert "exportable.agoraskill" in resp.headers["content-disposition"]
    bundle = resp.content
    assert zipfile.ZipFile(io.BytesIO(bundle)).namelist()

    preview = client.post(
        "/api/skills/import/preview",
        files={"file": ("x.agoraskill", io.BytesIO(bundle), BUNDLE_MIME)},
    )
    assert preview.status_code == 200
    assert preview.json()["name"] == "Exportable"
    assert preview.json()["documents"] == ["syllabus.md"]

    imported = client.post(
        "/api/skills/import",
        files={"file": ("x.agoraskill", io.BytesIO(bundle), BUNDLE_MIME)},
        data={"name": "Colleague's copy"},
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["name"] == "Colleague's copy"
    assert [d["filename"] for d in imported.json()["docs"]] == ["syllabus.md"]

    evil = _bundle({"manifest.json": _manifest(), "docs/../../evil.md": b"x"})
    rejected = client.post(
        "/api/skills/import",
        files={"file": ("evil.agoraskill", io.BytesIO(evil), BUNDLE_MIME)},
    )
    assert rejected.status_code == 400
    assert client.get("/api/skills/999/export").status_code == 404


# --------------------------------------------------------------------------
# settings: API key encryption
# --------------------------------------------------------------------------


def test_api_key_saved_encrypted_and_never_returned(client, db):
    secret = "sk-ant-api03-notarealkey-0123456789"

    resp = client.post(
        "/api/settings/keys",
        json={"provider": "anthropic", "api_key": secret, "label": "laptop"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert secret not in json.dumps(body)
    assert body["masked_key"].startswith("sk-a") and body["masked_key"].endswith("6789")
    assert body["active"] is True

    # Ciphertext on disk, plaintext only via the decrypting accessor.
    cred = db.get(ApiCredential, body["id"])
    assert secret not in cred.key_encrypted
    assert security.decrypt(cred.key_encrypted) == secret
    assert security.get_api_key(db, "anthropic") == secret
    key_file = Path(config.SECRET_KEY_PATH)
    assert key_file.exists()
    if os.name != "nt":
        assert oct(os.stat(key_file).st_mode)[-3:] == "600"

    listed = client.get("/api/settings/keys")
    assert secret not in listed.text
    summary = client.get("/api/settings")
    assert secret not in summary.text
    anthropic = next(p for p in summary.json()["providers"] if p["provider"] == "anthropic")
    assert anthropic["configured"] is True
    assert anthropic["source"] == "stored"
    openai = next(p for p in summary.json()["providers"] if p["provider"] == "openai")
    assert openai["configured"] is False
    assert openai["masked_key"] == ""

    client.post("/api/settings/keys", json={"provider": "anthropic", "api_key": "sk-ant-second-key"})
    creds = client.get("/api/settings/keys").json()
    assert [c["active"] for c in creds if c["provider"] == "anthropic"].count(True) == 1
    assert security.get_api_key(db, "anthropic") == "sk-ant-second-key"

    # Re-activating the first credential brings the original key back.
    assert client.post(f"/api/settings/keys/{body['id']}/activate").status_code == 200
    assert security.get_api_key(db, "anthropic") == secret

    assert client.delete(f"/api/settings/keys/{body['id']}").status_code == 200
    assert client.delete("/api/settings/keys/98765").status_code == 404
    assert (
        client.post("/api/settings/keys", json={"provider": "skynet", "api_key": "x" * 12}).status_code
        == 400
    )


def test_key_form_field_alias_test_and_delete_by_provider(client, db):
    """settings.html posts the field as ``key`` and deletes by provider name."""
    secret = "sk-ant-form-field-key-0001"
    saved = client.post("/api/settings/keys", json={"provider": "anthropic", "key": secret})
    assert saved.status_code == 201, saved.text
    assert security.get_api_key(db, "anthropic") == secret

    checked = client.post("/api/settings/keys/anthropic/test").json()
    assert checked["ok"] is True
    assert secret not in json.dumps(checked)
    assert checked["masked_key"].endswith("0001")

    missing = client.post("/api/settings/keys/openai/test").json()
    assert missing["ok"] is False
    assert "No API key" in missing["message"]
    assert client.post("/api/settings/keys/skynet/test").status_code == 400

    assert client.delete("/api/settings/keys/anthropic").status_code == 200
    assert security.get_api_key(db, "anthropic") is None
    assert client.delete("/api/settings/keys/anthropic").status_code == 404

    too_short = client.post("/api/settings/keys", json={"provider": "anthropic", "key": "abc"})
    assert too_short.status_code == 400


def test_settings_page_never_renders_a_key(client, db):
    secret = "sk-ant-page-render-key-99"
    client.post("/api/settings/keys", json={"provider": "anthropic", "api_key": secret})
    page = client.get("/settings")
    assert page.status_code == 200
    assert secret not in page.text
    assert "key configured" in page.text
    assert "not configured" in page.text  # openai has no key


def test_bulk_default_models(client):
    resp = client.post(
        "/api/settings/models",
        json={"anthropic": "claude-haiku-4-5", "openai": "gpt-5.6-terra"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["defaults"] == {
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5.6-terra",
    }
    assert config.default_model_for("anthropic") == "claude-haiku-4-5"
    assert config.default_model_for("openai") == "gpt-5.6-terra"
    assert client.post("/api/settings/models", json={"anthropic": "nope"}).status_code == 400


def test_default_model_selection(client):
    registry = client.get("/api/settings/models")
    assert registry.status_code == 200
    assert registry.json()["registry"]["anthropic"]["default"] == "claude-opus-5"

    ok = client.post(
        "/api/settings/models/default", json={"provider": "anthropic", "model": "claude-sonnet-5"}
    )
    assert ok.status_code == 200
    assert ok.json()["default"] == "claude-sonnet-5"
    assert config.default_model_for("anthropic") == "claude-sonnet-5"
    assert Path(config.MODEL_REGISTRY_PATH).exists()

    bad = client.post(
        "/api/settings/models/default", json={"provider": "anthropic", "model": "gpt-4"}
    )
    assert bad.status_code == 400


# --------------------------------------------------------------------------
# test drive
# --------------------------------------------------------------------------


def test_try_skill_falls_back_to_mock_without_a_key(client, db, monkeypatch):
    import app.routers.skills as skills_router

    def fail_factory(*_args, **_kwargs):
        raise AssertionError("no-key fallback must not instantiate a provider")

    monkeypatch.setattr(skills_router, "_load_provider_factory", lambda: fail_factory)
    skill = make_skill(
        db,
        name="Strict Essay Grader",
        system_prompt="Grade in a restrained philosophy voice.\nBe specific.",
    )
    svc.add_knowledge_doc(db, skill.id, "syllabus.md", b"# MA232", title="syllabus.md")

    resp = client.post(f"/api/skills/{skill.id}/try", json={"message": "Grade 2+2=5"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["used_mock"] is True
    assert body["provider"] == config.MOCK_PROVIDER
    assert body["model"] == config.MOCK_MODEL
    assert "Strict Essay Grader" in body["reply"]
    assert "Grade in a restrained philosophy voice." in body["reply"]
    assert "syllabus.md" in body["reply"]
    assert "Grade 2+2=5" in body["reply"]
    assert "Ask me about a course" not in body["reply"]
    assert body["error"] is None

    assert client.post("/api/skills/999/try", json={"message": "hi"}).status_code == 404
    assert client.post(f"/api/skills/{skill.id}/try", json={"message": ""}).status_code == 422


def test_try_skill_uses_provider_layer_when_available(client, db, monkeypatch):
    """With a key configured, the endpoint must go through get_provider()."""
    import app.routers.skills as skills_router

    seen: dict[str, object] = {}

    class FakeProvider:
        def chat(self, messages, tools=None, system=None, model=None, max_tokens=None):
            seen["messages"] = messages
            seen["system"] = system
            seen["model"] = model
            return {"text": "graded by the real provider"}

    def fake_get_provider(provider, model=None, api_key=None):
        seen["provider"] = provider
        seen["api_key"] = api_key
        return FakeProvider()

    monkeypatch.setattr(skills_router, "_load_provider_factory", lambda: fake_get_provider)
    client.post(
        "/api/settings/keys",
        json={"provider": "anthropic", "api_key": "sk-ant-configured-key"},
    )

    skill = make_skill(db, system_prompt="Be terse.")
    svc.add_knowledge_doc(db, skill.id, "syllabus.md", b"# MA232 syllabus")
    body = client.post(
        f"/api/skills/{skill.id}/try", json={"message": "How strict are you?"}
    ).json()

    assert body["used_mock"] is False
    assert body["provider"] == "anthropic"
    assert body["model"] == config.default_model_for("anthropic")  # auto skill resolved
    assert body["reply"] == "graded by the real provider"
    assert seen["provider"] == "anthropic"
    assert seen["api_key"] == "sk-ant-configured-key"
    assert seen["messages"] == [{"role": "user", "content": "How strict are you?"}]
    assert "Be terse." in seen["system"]
    assert "MA232 syllabus" in seen["system"]  # knowledge docs ride along


def test_try_skill_matches_the_engine_factory_contract(client, db):
    """Against the real ``app.ai.providers`` surface, with no key configured."""
    pytest.importorskip("app.ai.providers")
    skill = make_skill(db)

    body = client.post(f"/api/skills/{skill.id}/try", json={"message": "hello"}).json()
    assert body["engine_available"] is True
    assert body["used_mock"] is True
    assert body["provider"] == config.MOCK_PROVIDER
    assert body["error"] is None
    assert body["reply"].strip()
    assert "Ask me about a course" not in body["reply"]


def test_try_skill_handles_engine_style_signatures(client, db, monkeypatch):
    """get_provider(skill_or_settings, db) + chat(system_prompt, messages, tools)."""
    import app.routers.skills as skills_router

    seen: dict[str, object] = {}

    class EngineStyleProvider:
        def chat(self, system_prompt, messages, tools=None):
            seen["system_prompt"] = system_prompt
            seen["messages"] = messages
            return type("ChatTurn", (), {"text": "engine style reply", "tool_calls": []})()

    def get_provider(skill_or_settings=None, db=None):
        seen["settings"] = skill_or_settings
        seen["db_passed"] = db is not None
        return EngineStyleProvider()

    monkeypatch.setattr(skills_router, "_load_provider_factory", lambda: get_provider)
    client.post("/api/settings/keys", json={"provider": "anthropic", "api_key": "sk-ant-key-y"})
    skill = make_skill(db, system_prompt="Be terse.")

    body = client.post(f"/api/skills/{skill.id}/try", json={"message": "ping"}).json()
    assert body["reply"] == "engine style reply"
    assert body["used_mock"] is False
    assert seen["settings"] == {"provider": "anthropic", "model": config.default_model_for("anthropic")}
    assert seen["db_passed"] is True
    assert "Be terse." in seen["system_prompt"]
    assert seen["messages"] == [{"role": "user", "content": "ping"}]


def test_try_skill_survives_a_provider_error(client, db, monkeypatch):
    import app.routers.skills as skills_router

    class Boom:
        def chat(self, messages, **kw):
            raise RuntimeError("rate limited")

    monkeypatch.setattr(skills_router, "_load_provider_factory", lambda: (lambda *a, **k: Boom()))
    client.post("/api/settings/keys", json={"provider": "anthropic", "api_key": "sk-ant-key-x"})
    skill = make_skill(db)

    body = client.post(f"/api/skills/{skill.id}/try", json={"message": "hi"}).json()
    assert body["used_mock"] is True
    assert "rate limited" in body["error"]
    assert body["reply"].strip()


# --------------------------------------------------------------------------
# page routes (templates are owned by the frontend module)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,template", [("/skills", "skills.html"), ("/settings", "settings.html")])
def test_page_routes_render(client, db, path, template):
    make_skill(db)
    assert client.get(path).status_code == 200


def test_skill_detail_page(client, db):
    skill = make_skill(db)
    # The 404 path never touches a template, so it is always exercised.
    assert client.get("/skills/999999").status_code == 404
    assert client.get(f"/skills/{skill.id}").status_code == 200


def test_router_is_mounted(client):
    payload = client.get("/api/health").json()
    assert "app.routers.skills" in payload["routers"]
    assert "app.routers.settings" in payload["routers"]
