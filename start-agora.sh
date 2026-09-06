#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python3.12 -m venv .venv
fi
if [[ ! -f .venv/.agora-installed ]]; then
  .venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt
  touch .venv/.agora-installed
fi
exec .venv/bin/python run.py "$@"
