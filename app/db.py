"""Engine / session plumbing.

Usage in routers::

    from fastapi import Depends
    from sqlalchemy.orm import Session
    from app.db import get_db

    @router.get("/api/things")
    def things(db: Session = Depends(get_db)): ...
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app import config
from app.models import Base

config.ensure_dirs()

_connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}

engine: Engine = create_engine(
    config.DATABASE_URL,
    connect_args=_connect_args,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, connection_record):  # pragma: no cover - driver hook
    """SQLite needs FKs turned on explicitly; WAL keeps the UI responsive."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
    except Exception:
        # Non-SQLite drivers (or in-memory edge cases) — nothing to do.
        pass


#: Columns added after a table first shipped. `create_all` only creates missing
#: tables, so a database made by an earlier build needs the column added by
#: hand. Keep entries here forever — they are cheap and idempotent.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("pseudonym_map", "retired_at", "DATETIME"),
    (
        "chat_messages",
        "course_id",
        "INTEGER REFERENCES courses(id) ON DELETE SET NULL",
    ),
    # Fall readiness: skill modes, selective grading, the human-review record.
    ("skills", "mode", "VARCHAR(40) NOT NULL DEFAULT 'grade'"),
    ("assignments", "ai_criteria", "JSON"),
    ("grade_results", "mode", "VARCHAR(40) NOT NULL DEFAULT 'grade'"),
    ("grade_results", "seen_at", "DATETIME"),
    ("grade_results", "approved_at", "DATETIME"),
    ("grade_results", "released_at", "DATETIME"),
    ("grade_results", "withheld_at", "DATETIME"),
    ("grade_results", "edit_log", "JSON"),
)

def _add_missing_columns() -> None:
    """Idempotent ALTER TABLE pass for columns added to existing tables."""
    inspector = inspect(engine)
    try:
        tables = set(inspector.get_table_names())
    except Exception:  # pragma: no cover - inspection failure is not fatal
        return
    for table, column, ddl_type in _ADDED_COLUMNS:
        if table not in tables:
            continue
        existing = {col["name"] for col in inspector.get_columns(table)}
        if column in existing:
            continue
        with engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))


def init_db() -> None:
    """Create every table. Called on app startup; idempotent."""
    config.ensure_dirs()
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id_id "
                "ON chat_messages (session_id, id)"
            )
        )


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Standalone transactional session for scripts/seeding."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


__all__ = ["engine", "SessionLocal", "init_db", "get_db", "session_scope", "Base"]
