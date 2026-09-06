"""Native-file cleanup helpers."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


@dataclass
class ReversibleDelete:
    original: Path
    backup: Path

    def restore(self) -> None:
        if self.original.exists():
            _remove_path(self.original)
        _copy_path(self.backup, self.original)
        _remove_path(self.backup)

    def discard(self) -> None:
        _remove_path(self.backup)


def restore_all(tokens: Iterable[ReversibleDelete]) -> None:
    errors: list[Exception] = []
    for token in reversed(list(tokens)):
        try:
            token.restore()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise ExceptionGroup("Could not restore deleted native files", errors)


def reraise_delete_failure(
    original: Exception,
    rollback: Callable[[], object],
    tokens: Iterable[ReversibleDelete],
) -> None:
    secondary_errors: list[Exception] = []
    try:
        rollback()
    except Exception as rollback_error:
        secondary_errors.append(rollback_error)
    try:
        restore_all(tokens)
    except ExceptionGroup as restore_errors:
        secondary_errors.extend(restore_errors.exceptions)
    if secondary_errors:
        raise original from ExceptionGroup("Delete recovery failed", secondary_errors)
    raise original


def discard_all(tokens: Iterable[ReversibleDelete]) -> None:
    errors: list[Exception] = []
    for token in tokens:
        try:
            token.discard()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise ExceptionGroup("Could not discard deleted native-file backups", errors)


def reversible_delete(
    path: Path | str,
    root: Path | str,
    *,
    direct_child: bool = False,
) -> ReversibleDelete | None:
    """Copy a validated path to a sibling backup, then remove the original."""
    resolved_root = Path(root).resolve()
    target = Path(path).resolve()
    inside_root = (
        target.parent == resolved_root if direct_child else resolved_root in target.parents
    )
    if not inside_root:
        raise ValueError(f"Native file is outside {resolved_root}")
    if not target.exists():
        return None

    backup = target.with_name(f".{target.name}.agora-delete-{uuid.uuid4().hex}")
    try:
        _copy_path(target, backup)
    except Exception:
        if backup.exists():
            _remove_path(backup)
        raise

    token = ReversibleDelete(original=target, backup=backup)
    try:
        _remove_path(target)
    except Exception:
        token.restore()
        raise
    return token
