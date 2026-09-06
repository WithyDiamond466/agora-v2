#!/usr/bin/env python
"""Render smoke test for the frontend module (templates/ + static/).

Three passes, no network and no real provider:

  1. Jinja compile — every file in templates/ is parsed with the app's own
     environment, so a syntax error (a stray {% endif %}, a bad filter name)
     fails here instead of as an HTTP 500 later.
  2. Live render — the app is booted against a throwaway SQLite file seeded
     with the demo course, then every page route is fetched with TestClient
     and checked for the markers the frontend JS needs to find.
  3. `node --check` on every static/js file when node is available.

    .venv/bin/python scripts/render_smoke.py

Exits non-zero on the first failure and prints what broke.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"  FAIL  {message}")


def ok(message: str) -> None:
    print(f"  ok    {message}")


# --------------------------------------------------------------------------
# 1. Jinja compile
# --------------------------------------------------------------------------


def compile_templates() -> None:
    from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

    from app import config

    print("templates — jinja compile")
    env = Environment(loader=FileSystemLoader(str(config.TEMPLATES_DIR)), autoescape=True)
    for path in sorted(Path(config.TEMPLATES_DIR).glob("*.html")):
        try:
            env.get_template(path.name)
        except TemplateSyntaxError as exc:
            fail(f"{path.name}:{exc.lineno} {exc.message}")
        else:
            ok(path.name)


# --------------------------------------------------------------------------
# 2. Live render of every page route
# --------------------------------------------------------------------------

# marker substrings each page must contain once rendered
PAGE_MARKERS: dict[str, list[str]] = {
    "/": ["stat-row", "Courses"],
    "/courses/{course_id}": ['id="nudges-panel"', 'id="course-context"', "insight.js"],
    "/students/{student_id}": [
        'id="student-card"',
        'data-action="refresh-card"',
        'id="observation-feed"',
        'id="student-context"',
        "insight.js",
    ],
    "/courses/{course_id}/analytics": ["chart-mount"],
    "/grading/{assignment_id}": [
        'id="privacy-banner"',
        "data-privacy-chip",
        "data-privacy-report",
        "privacy.js",
        "grading.js",
    ],
    "/skills": ["Skills"],
    "/settings": [
        'id="privacy-settings"',
        "data-privacy-mode",
        "data-pseudonym-body",
        "data-local-form",
        "data-local-health",
        "privacy.js",
    ],
}


def render_pages() -> None:
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from app.db import get_db
    from app.main import app
    from app.models import Assignment, Base, Course, Student
    from app.seed import seed_demo

    print("\npages — live render (demo data, MockProvider only)")
    tmp = Path(tempfile.mkdtemp(prefix="agora-smoke-"))
    engine = create_engine(f"sqlite:///{tmp / 'smoke.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        seed_demo(session)
        course = session.scalars(select(Course)).first()
        student = session.scalars(select(Student)).first()
        assignment = session.scalars(select(Assignment)).first()
        ids = {
            "course_id": course.id if course else 1,
            "student_id": student.id if student else 1,
            "assignment_id": assignment.id if assignment else 1,
        }

        app.dependency_overrides[get_db] = lambda: session
        client = TestClient(app)
        for pattern, markers in PAGE_MARKERS.items():
            path = pattern.format(**ids)
            response = client.get(path)
            if response.status_code != 200:
                fail(f"GET {path} → {response.status_code}")
                continue
            missing = [m for m in markers if m not in response.text]
            if missing:
                fail(f"GET {path} rendered without: {', '.join(missing)}")
            else:
                ok(f"GET {path}")
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()
        engine.dispose()
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# 3. node --check
# --------------------------------------------------------------------------


def check_js() -> None:
    node = shutil.which("node")
    print("\nstatic/js — syntax")
    if not node:
        print("  skip  node is not installed")
        return
    for path in sorted((ROOT / "static" / "js").glob("*.js")):
        result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            fail(f"{path.name}: {result.stderr.strip().splitlines()[0] if result.stderr else 'parse error'}")
        else:
            ok(path.name)


def main() -> int:
    os.environ.setdefault("AGORA_DISABLE_BROWSER", "1")
    compile_templates()
    render_pages()
    check_js()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s).")
        return 1
    print("All render smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
