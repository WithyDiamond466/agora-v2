"""Portable backups of local coursework. Provider credentials are excluded."""
from __future__ import annotations
import json
from contextlib import closing
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
import zipfile

VERSION = 1
MAX_BYTES = 4 * 1024**3
DIRECTORIES = {'submissions', 'knowledge', 'skills', 'modes'}
SETTINGS = {'app_settings.json', 'privacy_settings.json', 'insight_settings.json', 'model_registry.json'}


def _relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if not value or '\\' in value or ':' in value or posix.is_absolute() or '..' in posix.parts:
        raise ValueError('Unsafe path in backup')
    return Path(*posix.parts)


def _rewrite_paths(db: sqlite3.Connection, transform) -> None:
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ('submissions', 'knowledge_docs'):
        if table not in tables:
            raise ValueError('This is not a complete Agora database')
        for row_id, value in db.execute(f'SELECT id, file_path FROM {table}').fetchall():
            if value:
                db.execute(f'UPDATE {table} SET file_path=? WHERE id=?', (transform(value), row_id))
    if 'api_credentials' in tables:
        db.execute('DELETE FROM api_credentials')
    # In-process jobs cannot survive a restore.
    db.execute("UPDATE submissions SET status='pending', error=NULL WHERE status='grading'")
    db.commit()


def create_backup(data_dir: Path, output: Path) -> Path:
    data_dir = data_dir.resolve()
    output = output.resolve()
    if output.is_relative_to(data_dir):
        raise ValueError('Save backups outside the active data directory')
    if output.exists():
        raise ValueError('Choose a new backup filename; existing files are never overwritten')
    source_db = data_dir / 'agora.db'
    if not source_db.is_file():
        raise ValueError('No Agora database found in that data directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='agora-backup-') as scratch:
        snapshot = Path(scratch) / 'agora.db'
        source = sqlite3.connect(source_db.as_uri() + '?mode=ro', uri=True)
        target = sqlite3.connect(snapshot)
        try:
            source.backup(target)
            def portable(value):
                path = Path(value).resolve()
                relative = path.relative_to(data_dir)
                if not path.is_file() or path.is_symlink():
                    raise ValueError('A referenced coursework file is missing or is a symlink')
                if relative.parts[0] not in DIRECTORIES:
                    raise ValueError('A referenced file is outside the coursework folders')
                return relative.as_posix()
            _rewrite_paths(target, portable)
            # Remove deleted credential payloads from free pages in the snapshot.
            target.execute('VACUUM')
        finally:
            source.close()
            target.close()
        try:
            with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json', json.dumps({'format': 'agora-backup', 'version': VERSION}))
                archive.write(snapshot, 'agora.db')
                for path in sorted(data_dir.rglob('*')):
                    relative = path.relative_to(data_dir)
                    if relative.parts[0] not in DIRECTORIES and relative.as_posix() not in SETTINGS:
                        continue
                    if path.is_symlink():
                        raise ValueError('Backups do not follow symlinks')
                    if path.is_file() and not any(part.startswith('.') for part in relative.parts):
                        archive.write(path, 'files/' + relative.as_posix())
            os.chmod(output, 0o600)
        except Exception:
            output.unlink(missing_ok=True)
            raise
    return output


def restore_backup(archive_path: Path, destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Restore requires a new directory; existing coursework is never overwritten')
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='.agora-restore-', dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > 50000 or sum(m.file_size for m in members) > MAX_BYTES:
                raise ValueError('Backup exceeds restore limits')
            if len({m.filename for m in members}) != len(members):
                raise ValueError('Duplicate archive paths')
            manifest = json.loads(archive.read('manifest.json'))
            if manifest != {'format': 'agora-backup', 'version': VERSION}:
                raise ValueError('Unsupported backup format or version')
            for member in members:
                name = member.filename
                if name == 'manifest.json':
                    continue
                if name == 'agora.db':
                    relative = Path(name)
                elif name.startswith('files/'):
                    relative = _relative(name[6:])
                    if relative.parts[0] not in DIRECTORIES and relative.as_posix() not in SETTINGS:
                        raise ValueError('Unexpected backup file')
                else:
                    raise ValueError('Unexpected backup entry')
                if member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unexpected directory or symlink entry')
                path = scratch / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target)
                os.chmod(path, 0o600)
        with closing(sqlite3.connect(scratch / 'agora.db')) as db:
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchone():
                raise ValueError('Backup database is inconsistent')
            def located(value):
                relative = _relative(value)
                if relative.parts[0] not in DIRECTORIES or not (scratch / relative).is_file():
                    raise ValueError('Backup is missing a referenced coursework file')
                return str(destination / relative)
            _rewrite_paths(db, located)
        scratch.rename(destination)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return destination
