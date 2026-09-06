import sqlite3
import zipfile
from pathlib import Path
import pytest
from app.backups import create_backup, restore_backup


def sample(tmp_path):
    root = tmp_path / 'original'; root.mkdir()
    file = root / 'submissions' / 'one.pdf'; file.parent.mkdir(); file.write_bytes(b'synthetic work')
    with sqlite3.connect(root / 'agora.db') as db:
        db.executescript('CREATE TABLE submissions (id INTEGER PRIMARY KEY, file_path TEXT, status TEXT, error TEXT); CREATE TABLE knowledge_docs (id INTEGER PRIMARY KEY, file_path TEXT); CREATE TABLE api_credentials (id INTEGER, key_encrypted TEXT);')
        db.execute('INSERT INTO submissions VALUES (1, ?, ?, NULL)', (str(file), 'grading'))
        db.execute("INSERT INTO api_credentials VALUES (1, 'synthetic-credential-marker')")
    return root


def test_portable_backup_restores_files_and_resets_jobs_without_credentials(tmp_path):
    source = sample(tmp_path)
    backup = create_backup(source, tmp_path / 'course.zip')
    destination = restore_backup(backup, tmp_path / 'restored')
    with sqlite3.connect(destination / 'agora.db') as db:
        path, status = db.execute('SELECT file_path, status FROM submissions').fetchone()
        assert Path(path).read_bytes() == b'synthetic work'
        assert Path(path).is_relative_to(destination)
        assert status == 'pending'
        assert db.execute('SELECT COUNT(*) FROM api_credentials').fetchone()[0] == 0
    with sqlite3.connect(source / 'agora.db') as db:
        assert db.execute('SELECT COUNT(*) FROM api_credentials').fetchone()[0] == 1
    with zipfile.ZipFile(backup) as z:
        assert b'synthetic-credential-marker' not in z.read('agora.db')
    with pytest.raises(ValueError, match='existing coursework'):
        restore_backup(backup, destination)
    with pytest.raises(ValueError, match='never overwritten'):
        create_backup(source, backup)


def test_restore_rejects_traversal_and_leaves_no_destination(tmp_path):
    archive = create_backup(sample(tmp_path), tmp_path / 'course.zip')
    with zipfile.ZipFile(archive, 'a') as z:
        z.writestr('files/../outside.txt', 'bad')
    with pytest.raises(ValueError, match='Unsafe path'):
        restore_backup(archive, tmp_path / 'restored')
    assert not (tmp_path / 'outside.txt').exists()
    assert not (tmp_path / 'restored').exists()
