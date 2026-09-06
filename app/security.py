"""API-key encryption at rest.

The Fernet key lives at ``~/.config/agora/secret.key`` (0600), created on first
use. It is deliberately outside the repo and outside the data dir so backing up
``data/`` never leaks credentials.
"""

from __future__ import annotations

import os
import stat
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.models import ApiCredential

_fernet: Optional[Fernet] = None


class SecretError(RuntimeError):
    """Raised when a stored credential cannot be decrypted."""


def _key_path() -> str:
    return str(config.SECRET_KEY_PATH)


def load_or_create_key() -> bytes:
    """Return the Fernet key, creating a 0600 file on first run."""
    path = config.SECRET_KEY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, stat.S_IRWXU)  # 0700
    except OSError:
        pass

    if path.exists():
        key = path.read_bytes().strip()
        if key:
            return key

    key = Fernet.generate_key()
    # Create with 0600 from the start — no window where it is world-readable.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    os.chmod(str(path), 0o600)
    return key


def get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(load_or_create_key())
    return _fernet


def reset_cache() -> None:
    """Drop the cached Fernet (tests that relocate AGORA_CONFIG_DIR)."""
    global _fernet
    _fernet = None


def encrypt(plaintext: str) -> str:
    """Encrypt a secret into a storable ASCII token."""
    if plaintext is None:
        raise ValueError("nothing to encrypt")
    return get_fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str:
    """Decrypt a stored token back to plaintext."""
    try:
        return get_fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, TypeError) as exc:
        raise SecretError(
            "Could not decrypt stored API key — the secret.key file may have "
            "been replaced. Re-enter the key in Settings."
        ) from exc


# Aliases used by the settings/engine modules.
encrypt_api_key = encrypt
decrypt_api_key = decrypt


def mask_key(plaintext: str) -> str:
    """Display form for the UI: never render a full key."""
    if not plaintext:
        return ""
    if len(plaintext) <= 8:
        return "•" * len(plaintext)
    return f"{plaintext[:4]}{'•' * 8}{plaintext[-4:]}"


def get_api_key(db: Session, provider: str) -> Optional[str]:
    """Return the active plaintext key for a provider, or None.

    Environment variables win when present so a developer can run without
    storing anything (ANTHROPIC_API_KEY / OPENAI_API_KEY).
    """
    env_name = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    if env_name:
        env_val = os.environ.get(env_name)
        if env_val:
            return env_val

    cred = db.scalars(
        select(ApiCredential)
        .where(ApiCredential.provider == provider, ApiCredential.active.is_(True))
        .order_by(ApiCredential.id.desc())
    ).first()
    if cred is None:
        return None
    return decrypt(cred.key_encrypted)


def set_api_key(db: Session, provider: str, plaintext: str, label: str | None = None) -> ApiCredential:
    """Store/replace the active key for a provider (deactivates older ones)."""
    for old in db.scalars(
        select(ApiCredential).where(
            ApiCredential.provider == provider, ApiCredential.active.is_(True)
        )
    ).all():
        old.active = False
    cred = ApiCredential(
        provider=provider,
        key_encrypted=encrypt(plaintext.strip()),
        label=label,
        active=True,
    )
    db.add(cred)
    db.commit()
    db.refresh(cred)
    return cred


def has_api_key(db: Session, provider: str) -> bool:
    try:
        return get_api_key(db, provider) is not None
    except SecretError:
        return False


__all__ = [
    "load_or_create_key",
    "get_fernet",
    "encrypt",
    "decrypt",
    "encrypt_api_key",
    "decrypt_api_key",
    "mask_key",
    "get_api_key",
    "set_api_key",
    "has_api_key",
    "reset_cache",
    "SecretError",
]
