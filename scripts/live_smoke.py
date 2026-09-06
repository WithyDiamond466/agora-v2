#!/usr/bin/env python
"""Real-provider end-to-end smoke: demo course → cloud grading → review → release → export.

The test suite never touches a network. This script is the one place a real
key is spent, on purpose, so the path a professor will use in the fall has
actually been exercised end to end.

    ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python scripts/live_smoke.py
    OPENAI_API_KEY=sk-...        .venv/bin/python scripts/live_smoke.py --provider openai
    .venv/bin/python scripts/live_smoke.py --provider auto --count 2 --compare

It uses a throw-away data directory (nothing in data/ is touched), seeds the
demo course, accepts the terms, grades N demo submissions with the chosen
provider, exercises the review + release + CSV export path, optionally runs
the model comparison, and prints what came back. Exit code is non-zero if any
step fails. Actual cost depends on the selected model, input, and provider pricing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Keep redirected console output readable on Windows as well as Unix.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="auto", help="anthropic | openai | auto | local | mock")
    parser.add_argument("--model", default=None, help="model id (default: the provider's default)")
    parser.add_argument("--count", type=int, default=2, help="how many demo submissions to grade")
    parser.add_argument("--mode", default="grade", help="skill mode: grade | feedback | selective")
    parser.add_argument("--compare", action="store_true", help="also run the model comparison on one submission")
    parser.add_argument("--keep", action="store_true", help="keep the temp data dir and print its path")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="agora-live-"))
    os.environ["AGORA_DATA_DIR"] = str(tmp / "data")
    os.environ["AGORA_DATABASE_URL"] = f"sqlite:///{tmp / 'data' / 'agora.db'}"
    os.environ["AGORA_CONFIG_DIR"] = str(tmp / "cfg")
    os.environ.pop("AGORA_AI_PROVIDER", None)

    from fastapi.testclient import TestClient

    from app import config
    from app.db import init_db, session_scope
    from app.main import app
    from app.seed import seed_demo

    init_db()
    with session_scope() as db:
        seed_demo(db)
    client = TestClient(app)

    def step(title: str) -> None:
        print(f"\n== {title}")

    def ok(resp, *codes):
        codes = codes or (200,)
        if resp.status_code not in codes:
            print(f"   FAILED {resp.request.method} {resp.url}: {resp.status_code} {resp.text[:400]}")
            raise SystemExit(2)
        return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp

    step("providers")
    settings = ok(client.get("/api/settings"))
    for p in settings["providers"]:
        print(f"   {p['provider']:<10} configured={p['configured']} source={p['source']} preferred={p.get('preferred')}")
    resolved = ok(client.get("/api/compare/candidates"))["candidates"][0]
    print(f"   auto → {resolved['label']} (available={resolved['available']})")

    step("terms")
    print("   before:", ok(client.get("/api/terms/status"))["accepted"])
    ok(client.post("/api/terms/accept", json={"signed_by": "live smoke"}), 201)
    print("   after: ", ok(client.get("/api/terms/status"))["accepted"])

    step("skill + assignment")
    skills = ok(client.get("/api/skills"))
    skill = skills[0]
    patch = {"provider": args.provider, "mode": args.mode}
    if args.model:
        patch["model"] = args.model
    skill = ok(client.patch(f"/api/skills/{skill['id']}", json=patch))
    print(f"   skill {skill['name']!r}: provider={skill['provider']} model={skill['model']} mode={skill['mode']}")
    courses = ok(client.get("/api/courses"))
    course_id = courses[0]["id"]
    assignments = ok(client.get(f"/api/courses/{course_id}/assignments"))
    assignment = assignments[0]
    if args.mode == "selective":
        rubric = ok(client.get(f"/api/assignments/{assignment['id']}"))["rubric"]
        keys = [c["key"] for c in rubric["criteria"]][:1]
        ok(client.patch(f"/api/assignments/{assignment['id']}", json={"ai_criteria": keys}))
        print(f"   selective: AI grades {keys}")
    subs = ok(client.get(f"/api/assignments/{assignment['id']}/submissions"))[: args.count]
    ids = [s["id"] for s in subs]
    print(f"   assignment {assignment['name']!r}: regrading {ids}")

    step(f"grading with {args.provider} (force regrade of {len(ids)})")
    started = time.time()
    run = ok(client.post(f"/api/assignments/{assignment['id']}/grade-all", json={"force": True, "submission_ids": ids}))
    print(f"   queued={run['queued']} skipped={len(run['skipped'])}")
    # TestClient runs background tasks before returning, so results are ready.
    status = ok(client.get(f"/api/assignments/{assignment['id']}/grading/status"))
    rows = [r for r in status["submissions"] if r["id"] in ids]
    failed = 0
    for r in rows:
        line = f"   #{r['student_number']:02d} {r['status']:<7}"
        if r["status"] == "graded":
            line += f" {r['overall_score']}/{r['max_score']} review={r['review_state']} mode={r['mode']}"
        else:
            failed += 1
            line += f" ERROR {r['error']}"
        print(line)
    print(f"   {time.time() - started:.1f}s")
    if failed:
        print("   grading failed — see errors above")
        return 3
    first = ok(client.get(f"/api/submissions/{ids[0]}/result"))
    print(f"   model that answered: {first['result']['model']}")
    print(f"   feedback: {(first['result']['summary_feedback'] or '')[:300]!r}")
    scan = ok(client.get(f"/api/submissions/{ids[0]}/privacy"))
    if scan.get("scan"):
        print(f"   privacy: mode={scan['scan']['mode']} swapped={scan['scan'].get('findings', {}).get('count', '?')}")

    step("review → release → export")
    ok(client.post(f"/api/submissions/{ids[0]}/review/seen"))
    ok(client.patch(f"/api/submissions/{ids[0]}/result", json={"summary_feedback": (first["result"]["summary_feedback"] or "") + "\n\n(reviewed in live smoke)"}))
    ok(client.post(f"/api/submissions/{ids[0]}/review/approve"))
    summary = ok(client.get(f"/api/assignments/{assignment['id']}/release"))
    print(f"   releasable={len(summary['releasable'])} unseen={len(summary['unseen'])} withheld={len(summary['withheld'])}")
    released = ok(client.post(f"/api/assignments/{assignment['id']}/release", json={}))
    print(f"   {released['message']}")
    export = client.get(f"/api/assignments/{assignment['id']}/export?format=csv")
    ok(export)
    reader = csv.DictReader(io.StringIO(export.content.decode("utf-8-sig")))
    rows = list(reader)
    assert len(rows) == 1 and all(r["ai_disclosure"] for r in rows), "Expected one disclosed, approved export row"
    print(f"   csv rows={len(rows)} disclosure present=True")
    events = ok(client.get(f"/api/submissions/{ids[0]}/review"))["events"]
    print("   events:", " → ".join(e["kind"] for e in events))

    if args.compare:
        step("compare on one submission")
        cands = [{"provider": args.provider, "model": args.model}, {"provider": "mock"}]
        cmp = ok(client.post(f"/api/submissions/{ids[-1]}/compare", json={"candidates": cands}))
        for r in cmp["results"]:
            if r["grade"]:
                print(f"   {r['provider']} · {r['model']}: {r['grade']['overall_score']}/{r['grade']['max_score']} in {r['elapsed_ms']} ms")
            else:
                raise AssertionError(f"Comparison failed: {r['error']}")

    step("chat assistant")
    chat = ok(client.post("/api/chat", json={"message": "Which students are struggling most in this course?", "context": {"course_id": course_id}}))
    print(f"   {chat.get('provider', '?')} · {chat.get('model', '?')}: {(chat.get('reply') or '')[:200]!r}")

    assert chat.get("reply") and "Open a course before" not in chat["reply"], "Chat did not use course context"
    print("\nOK" + (f" — data kept at {tmp}" if args.keep else ""))
    if not args.keep:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
