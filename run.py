#!/usr/bin/env python
"""Launch Agora: start uvicorn on 127.0.0.1:8811 and open the browser.

    python run.py            # normal start
    python run.py --demo     # seed the demo course first
    python run.py --no-browser --reload

The uvicorn Server object is parked on `app.state.server` so POST /api/shutdown
can stop the process cleanly instead of leaving a zombie holding the port.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from app import config

log = logging.getLogger("agora.run")


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def open_browser_when_ready(url: str, host: str, port: int, timeout: float = 15.0) -> None:
    """Poll the port, then open the browser — avoids a 'connection refused' tab."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(host, port):
            try:
                webbrowser.open(url)
            except Exception as exc:  # noqa: BLE001 - headless machines have no browser
                log.info("Could not open a browser (%s). Visit %s", exc, url)
            return
        time.sleep(0.2)
    log.warning("Server did not come up within %.0fs; visit %s manually", timeout, url)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run.py", description="Start the Agora server.")
    parser.add_argument("--host", choices=("127.0.0.1",), default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument(
        "--demo", action="store_true", help="seed a fully graded demo course before starting"
    )
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev)")
    parser.add_argument("--seed-only", action="store_true", help="seed the demo data and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.HOST = args.host
    config.PORT = args.port

    from app.db import init_db

    init_db()

    if args.demo or args.seed_only:
        from app.seed import seed_demo

        summary = seed_demo(reset=False)
        print(
            f"Seeded demo course (id={summary['course_id']}): "
            f"{summary.get('students', 0)} students, "
            f"{summary.get('graded', 0)} graded submissions."
        )
        if args.seed_only:
            return 0

    if port_in_use(args.host, args.port):
        print(
            f"Something is already listening on {args.host}:{args.port}.\n"
            f"If that is an old Agora instance: "
            f"curl -X POST http://{args.host}:{args.port}/api/shutdown",
            file=sys.stderr,
        )
        return 1

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Thread(
            target=open_browser_when_ready, args=(url, args.host, args.port), daemon=True
        ).start()

    print(f"Agora running at {url}  (Ctrl-C or POST /api/shutdown to stop)")

    if args.reload:
        # Reload mode needs the import string; /api/shutdown falls back to SIGINT.
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)
        return 0

    from app.main import app

    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    )
    app.state.server = server
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
