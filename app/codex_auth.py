"""Reuse the professor's OpenAI login from the Codex CLI — where OpenAI allows it.

``codex login`` (OpenAI's CLI) caches credentials at ``~/.codex/auth.json``
(or ``$CODEX_HOME/auth.json``). Two kinds exist:

* ``auth_mode: "apikey"`` — the file holds an OpenAI API key. That key is an
  ordinary platform credential, so Agora can import it as the OpenAI key,
  saving the professor a paste.
* ``auth_mode: "chatgpt"`` — the file holds OAuth tokens for a ChatGPT plan.
  OpenAI issues those for Codex itself; they are not platform API credentials
  and OpenAI's terms limit them to Codex. Agora reads only the signed-in
  email (from the id token's claims) to say who is logged in, and points the
  professor at an API key. No token is ever stored, sent, or logged here.

OpenAI's "Sign in with ChatGPT" for third-party apps is not generally
available as of 2026-09; when it is, this module is where it plugs in.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Optional

API_KEYS_URL = "https://platform.openai.com/api-keys"


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def auth_path() -> Path:
    return codex_home() / "auth.json"


def _jwt_claims(token: Any) -> dict[str, Any]:
    """Unverified payload of a JWT — enough to read an email; nothing is trusted."""
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _email_from(tokens: dict[str, Any]) -> Optional[str]:
    claims = _jwt_claims(tokens.get("id_token"))
    email = claims.get("email")
    if not email:
        profile = claims.get("https://api.openai.com/profile")
        if isinstance(profile, dict):
            email = profile.get("email")
    return str(email) if email else None


def status() -> dict[str, Any]:
    """What the Codex CLI login can offer Agora. Never includes a secret."""
    path = auth_path()
    base = {"path": str(path), "api_keys_url": API_KEYS_URL}
    if not path.is_file():
        return {
            **base,
            "state": "missing",
            "message": "No Codex CLI login found on this machine. Paste an OpenAI API key, or "
            "run `codex login --with-api-key` and come back.",
        }
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {**base, "state": "unreadable", "message": "The Codex login file could not be read."}
    if not isinstance(data, dict):
        return {**base, "state": "unreadable", "message": "The Codex login file is not what Agora expects."}

    api_key = str(data.get("OPENAI_API_KEY") or "").strip()
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    mode = str(data.get("auth_mode") or ("apikey" if api_key else "chatgpt")).strip().lower()
    email = _email_from(tokens)
    if api_key:
        return {
            **base,
            "state": "apikey",
            "email": email,
            "message": "The Codex CLI is logged in with an OpenAI API key. Agora can use that key.",
        }
    if tokens.get("access_token"):
        who = f" as {email}" if email else ""
        return {
            **base,
            "state": "chatgpt",
            "email": email,
            "message": (
                f"The Codex CLI is signed in with a ChatGPT account{who}. OpenAI issues those "
                "tokens for Codex only, and its terms keep them there, so Agora does not use "
                "them. Create an API key on the OpenAI platform and paste it here instead."
            ),
        }
    return {**base, "state": "missing", "mode": mode, "message": "The Codex login file holds no usable credential."}


def api_key() -> Optional[str]:
    """The API key from the Codex login, when it is that kind of login."""
    if status().get("state") != "apikey":
        return None
    try:
        with open(auth_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    key = str(data.get("OPENAI_API_KEY") or "").strip() if isinstance(data, dict) else ""
    return key or None


__all__ = ["API_KEYS_URL", "codex_home", "auth_path", "status", "api_key"]
