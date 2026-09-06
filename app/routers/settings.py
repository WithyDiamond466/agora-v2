"""Settings: API credentials (encrypted at rest) and default model selection.

Plaintext keys enter through POST only. They are encrypted by ``app.security``
before touching the DB and are **never** returned by any endpoint — the UI only
ever sees a mask like ``sk-a••••••••6789``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config, security
from app.db import get_db
from app.models import ApiCredential

router = APIRouter()
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

#: Same env vars app.security consults when resolving a key.
_ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class KeyIn(BaseModel):
    """The settings form posts ``key``; the JSON API also accepts ``api_key``."""

    provider: str = Field(min_length=1, max_length=50)
    api_key: Optional[str] = Field(default=None, max_length=500)
    key: Optional[str] = Field(default=None, max_length=500)
    label: Optional[str] = Field(default=None, max_length=200)

    def secret(self) -> str:
        return (self.api_key or self.key or "").strip()


class DefaultModelIn(BaseModel):
    provider: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=100)


class PreferredProviderIn(BaseModel):
    #: Empty/None clears the preference (``auto`` then follows the key order).
    provider: Optional[str] = Field(default=None, max_length=50)


def _check_provider(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider not in config.PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider {provider!r}. Known: {', '.join(config.PROVIDERS)}",
        )
    return provider


def _masked(cred: ApiCredential) -> str:
    """Mask a stored credential for display; never leak plaintext."""
    try:
        return security.mask_key(security.decrypt(cred.key_encrypted))
    except security.SecretError:
        return "unreadable — re-enter this key"


def credential_dict(cred: ApiCredential) -> dict[str, Any]:
    return {
        "id": cred.id,
        "provider": cred.provider,
        "label": cred.label,
        "active": bool(cred.active),
        "masked_key": _masked(cred),
        "created_at": cred.created_at.isoformat() if cred.created_at else None,
        "updated_at": cred.updated_at.isoformat() if cred.updated_at else None,
    }


def _credentials(db: Session, provider: str | None = None) -> list[ApiCredential]:
    stmt = select(ApiCredential).order_by(ApiCredential.provider, ApiCredential.id.desc())
    if provider:
        stmt = stmt.where(ApiCredential.provider == provider)
    return list(db.scalars(stmt).all())


def _get_credential(db: Session, cred_id: int) -> ApiCredential:
    cred = db.get(ApiCredential, cred_id)
    if cred is None:
        raise HTTPException(status_code=404, detail=f"Credential {cred_id} not found")
    return cred


def provider_status(db: Session) -> list[dict[str, Any]]:
    """Per-provider key/model state for the settings page (no plaintext)."""
    registry = config.load_model_registry()
    creds = _credentials(db)
    preferred = config.preferred_provider()
    out: list[dict[str, Any]] = []
    for provider in config.PROVIDERS:
        entry = registry.get(provider, {})
        mine = [c for c in creds if c.provider == provider]
        active = next((c for c in mine if c.active), None)
        # Env vars win at call time (see security.get_api_key), so say so here.
        env_key = os.environ.get(_ENV_VARS.get(provider, ""), "")
        if env_key:
            source, masked = "env", security.mask_key(env_key)
        elif active is not None:
            source, masked = "stored", _masked(active)
        elif provider == config.LOCAL_PROVIDER:
            enabled = bool(config.local_model_settings().get("enabled", True))
            source, masked = ("local" if enabled else "none"), ""
        else:
            source, masked = "none", ""
        out.append(
            {
                "provider": provider,
                "label": entry.get("label", provider.title()),
                "configured": source != "none",
                "preferred": provider == preferred,
                "source": source,
                "masked_key": masked,
                "credentials": [credential_dict(c) for c in mine],
                "default_model": entry.get("default"),
                "editable_models": bool(entry.get("editable")),
                "models": entry.get("models", []),
            }
        )
    return out


# --------------------------------------------------------------------------
# JSON API
# --------------------------------------------------------------------------


@router.get("/api/settings")
def get_settings(db: Session = Depends(get_db)) -> dict[str, Any]:
    return {
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "data_dir": str(config.DATA_DIR),
        "secret_key_path": str(config.SECRET_KEY_PATH),
        "providers": provider_status(db),
        "mock_provider": config.MOCK_PROVIDER,
        "default_provider": config.DEFAULT_PROVIDER,
        "preferred_provider": config.preferred_provider(),
        "auto_provider": config.AUTO_PROVIDER,
        "auto_order": list(config.AUTO_PROVIDER_ORDER),
    }


@router.get("/api/settings/models")
def get_models() -> dict[str, Any]:
    """The model registry (with any Settings overrides applied)."""
    return {
        "registry": config.load_model_registry(),
        "default_provider": config.DEFAULT_PROVIDER,
        "preferred_provider": config.preferred_provider(),
    }


@router.get("/api/settings/openai-login")
def openai_login_status() -> dict[str, Any]:
    """Can the professor's Codex CLI login stand in for pasting an OpenAI key?"""
    from app import codex_auth  # local import: reads a file outside the repo

    return codex_auth.status()


@router.post("/api/settings/openai-login/import", status_code=201)
def openai_login_import(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Store the OpenAI API key the Codex CLI is logged in with, encrypted."""
    from app import codex_auth

    key = codex_auth.api_key()
    if not key:
        info = codex_auth.status()
        raise HTTPException(status_code=409, detail=info["message"])
    cred = security.set_api_key(db, "openai", key, label="Codex CLI login")
    if config.preferred_provider() is None:
        config.set_preferred_provider("openai")
    data = credential_dict(cred)
    data["preferred_provider"] = config.preferred_provider()
    data["message"] = f"OpenAI key imported from the Codex CLI login ({data['masked_key']})."
    return data


@router.post("/api/settings/preferred-provider")
def set_preferred_provider(payload: PreferredProviderIn) -> dict[str, Any]:
    """The provider ``auto`` skills try first. Empty clears it."""
    value = (payload.provider or "").strip().lower()
    if value:
        value = _check_provider(value)
    saved = config.set_preferred_provider(value or None)
    return {
        "preferred_provider": saved,
        "message": f"Skills set to automatic will try {saved} first." if saved else (
            "Preference cleared — automatic skills follow the key order: "
            + ", ".join(config.AUTO_PROVIDER_ORDER) + "."
        ),
    }


@router.get("/api/settings/keys")
def list_keys(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [credential_dict(c) for c in _credentials(db)]


@router.post("/api/settings/keys", status_code=201)
def save_key(payload: KeyIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Store (and activate) an API key, encrypted at rest."""
    provider = _check_provider(payload.provider)
    plaintext = payload.secret()
    if len(plaintext) < 8:
        raise HTTPException(status_code=400, detail="That does not look like an API key")
    cred = security.set_api_key(db, provider, plaintext, label=payload.label)
    # The first key the professor adds becomes what "auto" skills try first.
    if config.preferred_provider() is None and config.is_cloud_provider(provider):
        config.set_preferred_provider(provider)
    data = credential_dict(cred)
    data["preferred_provider"] = config.preferred_provider()
    return data


def _resolve_credentials(db: Session, key_ref: str) -> list[ApiCredential]:
    """``key_ref`` is either a credential id or a provider name."""
    ref = (key_ref or "").strip()
    if ref.isdigit():
        return [_get_credential(db, int(ref))]
    provider = _check_provider(ref)
    creds = _credentials(db, provider)
    if not creds:
        raise HTTPException(status_code=404, detail=f"No stored key for {provider}")
    return creds


@router.post("/api/settings/keys/{key_ref}/activate")
def activate_key(key_ref: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Make one stored credential the active one for its provider."""
    cred = _resolve_credentials(db, key_ref)[0]
    for other in _credentials(db, cred.provider):
        other.active = other.id == cred.id
    db.commit()
    db.refresh(cred)
    return credential_dict(cred)


@router.post("/api/settings/keys/{provider}/test")
def test_key(provider: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Check that a key is present and that a client can be built for it.

    Deliberately does *not* spend a request against the provider; it reports
    configuration health only.
    """
    provider = _check_provider(provider)
    try:
        key = security.get_api_key(db, provider)
    except security.SecretError as exc:
        return {"ok": False, "provider": provider, "message": str(exc)}
    if not key:
        return {
            "ok": False,
            "provider": provider,
            "message": f"No API key configured for {provider}.",
        }

    try:
        from app.ai import providers as providers_mod  # noqa: PLC0415 - lazy

        built = providers_mod.get_provider({"provider": provider}, db)
        model = getattr(built, "model", config.default_model_for(provider))
    except ImportError:
        model = config.default_model_for(provider)
    except Exception as exc:  # noqa: BLE001 - report, never 500 the settings page
        return {"ok": False, "provider": provider, "message": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "masked_key": security.mask_key(key),
        "message": f"Key stored ({security.mask_key(key)}); ready to use {model}.",
    }


@router.delete("/api/settings/keys/{key_ref}")
def delete_key(key_ref: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Delete one credential by id, or every stored key for a provider."""
    creds = _resolve_credentials(db, key_ref)
    deleted = [c.id for c in creds]
    for cred in creds:
        db.delete(cred)
    db.commit()
    return {"deleted": deleted[0] if len(deleted) == 1 else deleted, "deleted_ids": deleted}


def _save_default(provider: str, model: str) -> str:
    registry = config.load_model_registry()
    known = [m.get("id") for m in registry.get(provider, {}).get("models", [])]
    if model not in known:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model!r} is not in the registry for {provider}. Known: {known}",
        )
    overrides = _load_overrides()
    overrides.setdefault(provider, {})["default"] = model
    config.save_model_registry(overrides)
    return model


@router.post("/api/settings/models/default")
def set_default_model(payload: DefaultModelIn) -> dict[str, Any]:
    """Pick the default model for a provider from ``config.MODEL_REGISTRY``."""
    provider = _check_provider(payload.provider)
    model = _save_default(provider, payload.model.strip())
    return {"provider": provider, "default": model, "registry": config.load_model_registry()}


@router.post("/api/settings/models")
def set_default_models(payload: dict[str, Any]) -> dict[str, Any]:
    """Bulk form of the above: ``{"anthropic": "claude-sonnet-5", ...}``.

    Also accepts the single-provider ``{"provider": ..., "model": ...}`` shape.
    """
    if "provider" in payload and "model" in payload:
        provider = _check_provider(str(payload["provider"]))
        return {
            "defaults": {provider: _save_default(provider, str(payload["model"]).strip())},
            "registry": config.load_model_registry(),
        }

    defaults: dict[str, str] = {}
    for provider, model in payload.items():
        if not isinstance(model, str) or not model.strip():
            continue
        defaults[_check_provider(provider)] = _save_default(
            _check_provider(provider), model.strip()
        )
    if not defaults:
        raise HTTPException(status_code=400, detail="No provider/model pairs were supplied")
    return {"defaults": defaults, "registry": config.load_model_registry()}


def _load_overrides() -> dict[str, Any]:
    """Current on-disk registry overrides (empty when nothing saved yet)."""
    try:
        with open(config.MODEL_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------
# page route
# --------------------------------------------------------------------------


def credentials_map(db: Session) -> dict[str, dict[str, Any]]:
    """``{provider: {configured, masked, label, source}}`` for settings.html."""
    return {
        entry["provider"]: {
            "configured": entry["configured"],
            "masked": entry["masked_key"],
            "masked_key": entry["masked_key"],
            "source": entry["source"],
            "label": (entry["credentials"][0]["label"] if entry["credentials"] else None),
        }
        for entry in provider_status(db)
    }


def _openai_login_status() -> dict[str, Any]:
    try:
        from app import codex_auth

        return codex_auth.status()
    except Exception:  # noqa: BLE001 - the page must render whatever is on disk
        return {"state": "missing", "message": ""}


@router.get("/settings", response_class=HTMLResponse)
def page_settings(request: Request, db: Session = Depends(get_db)) -> Any:
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            # settings.html reads `providers` as a list of provider ids.
            "providers": list(config.PROVIDERS),
            "credentials": credentials_map(db),
            "provider_status": provider_status(db),
            "model_registry": config.load_model_registry(),
            "preferred_provider": config.preferred_provider(),
            "auto_order": list(config.AUTO_PROVIDER_ORDER),
            "openai_login": _openai_login_status(),
            "secret_key_path": str(config.SECRET_KEY_PATH),
            "db_path": str(config.DB_PATH),
            "data_dir": str(config.DATA_DIR),
            "app_version": config.APP_VERSION,
        },
    )


__all__ = ["router", "provider_status", "credentials_map", "credential_dict"]


# ==========================================================================
# Increment 1 · PRIVACY MODULE endpoints (appended — nothing above changed)
#
# Settings → Privacy: the guard's mode, the local model's health, and the
# pseudonym map ("who is Person-A?"). The map is read-only data that never
# leaves this machine; it is what lets the UI swap codes back to real names.
#
# The settings page is owned by the frontend module, so these are JSON only —
# settings.html can pull them through the existing `data-api` glue.
# ==========================================================================

from app.models import Course, PseudonymMap  # noqa: E402 - appended module section


class PrivacyLocalModelIn(BaseModel):
    enabled: Optional[bool] = None
    base_url: Optional[str] = Field(default=None, max_length=500)
    model: Optional[str] = Field(default=None, max_length=200)


class PrivacyIn(BaseModel):
    mode: Optional[str] = Field(default=None, max_length=20)
    llm_sweep: Optional[bool] = None
    local_model: Optional[PrivacyLocalModelIn] = None


def privacy_status(db: Session) -> dict[str, Any]:
    """Everything Settings → Privacy renders (no PII beyond the local map)."""
    from app.ai import privacy as privacy_guard  # lazy: keeps settings import-light

    settings = config.load_privacy_settings()
    courses = db.scalars(select(Course).order_by(Course.name)).all()
    entries_by_course = privacy_guard.pseudonym_entries_by_course(
        db, [course.id for course in courses]
    )
    return {
        "mode": settings["mode"],
        "modes": [
            {
                "id": config.PRIVACY_MODE_SWAP,
                "label": "Swap (recommended)",
                "description": (
                    "Replace names, emails, phones and id numbers with stable codes "
                    "before anything is sent to a cloud model. Cloud calls receive "
                    "locally extracted text instead of the original file."
                ),
            },
            {
                "id": config.PRIVACY_MODE_WARN,
                "label": "Warn only",
                "description": (
                    "Detect identifiers and report them, but send the submission "
                    "unchanged."
                ),
            },
            {
                "id": config.PRIVACY_MODE_OFF,
                "label": "Off",
                "description": "Legacy behaviour: the native file goes straight to the provider.",
            },
        ],
        "llm_sweep": settings["llm_sweep"],
        "llm_sweep_active": privacy_guard.sweep_allowed(),
        "local_model": settings["local_model"],
        "local_model_is_local": config.local_base_url_is_local(
            settings["local_model"].get("base_url")
        ),
        "courses": [
            {
                "id": course.id,
                "name": course.name,
                "pseudonyms": len(entries_by_course[course.id]),
            }
            for course in courses
        ],
    }


@router.get("/api/settings/privacy")
def get_privacy_settings(db: Session = Depends(get_db)) -> dict[str, Any]:
    return privacy_status(db)


@router.post("/api/settings/privacy")
def set_privacy_settings(payload: PrivacyIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Set the Privacy Guard mode and the local model's address/name."""
    update: dict[str, Any] = {}
    if payload.mode is not None:
        update["mode"] = payload.mode
    if payload.llm_sweep is not None:
        update["llm_sweep"] = payload.llm_sweep
    if payload.local_model is not None:
        local = {k: v for k, v in payload.local_model.model_dump().items() if v is not None}
        if local:
            update["local_model"] = local
    if not update:
        raise HTTPException(status_code=400, detail="Nothing to update")
    try:
        config.save_privacy_settings(update)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return privacy_status(db)


@router.get("/api/settings/local-model/health")
def local_model_health(db: Session = Depends(get_db)) -> dict[str, Any]:
    """"local model: connected, gemma-3-4b-it" — never raises, always answers."""
    try:
        from app.ai import providers as providers_mod  # noqa: PLC0415 - lazy

        info = providers_mod.local_model_health()
    except Exception as exc:  # noqa: BLE001 - report, never 500 the settings page
        settings = config.local_model_settings()
        return {
            "ok": False,
            "provider": config.LOCAL_PROVIDER,
            "base_url": settings.get("base_url"),
            "model": settings.get("model"),
            "enabled": bool(settings.get("enabled", True)),
            "message": f"{type(exc).__name__}: {exc}",
        }
    info["label"] = (
        f"local model: {info['message']}" if info.get("ok") else "local model: not reachable"
    )
    return info


@router.get("/api/settings/privacy/pseudonyms")
def list_pseudonyms(
    course_id: Optional[int] = None, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """The pseudonym map — "who is Person-A?". Local only, never sent anywhere."""
    from app.ai import privacy as privacy_guard  # noqa: PLC0415 - lazy

    if course_id is not None:
        course = db.get(Course, course_id)
        if course is None:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
        courses = [course]
    else:
        courses = list(db.scalars(select(Course).order_by(Course.name)).all())

    entries_by_course = privacy_guard.pseudonym_entries_by_course(
        db, [course.id for course in courses]
    )
    out = []
    for course in courses:
        entries = entries_by_course[course.id]
        out.append(
            {
                "course_id": course.id,
                "course_name": course.name,
                "count": len(entries),
                "entries": entries,
            }
        )
    return {"courses": out, "mode": config.privacy_mode()}


@router.delete("/api/settings/privacy/pseudonyms/{entry_id}")
def delete_pseudonym(entry_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Forget one mapping. The next sighting gets a fresh code.

    The identity is wiped, but the code is retired rather than freed: reissuing
    "Person-A" to the next new face would silently change the meaning of every
    stored result, scan report and card that already mentions it.
    """
    from app.ai import privacy as privacy_guard  # noqa: PLC0415 - lazy

    row = db.get(PseudonymMap, entry_id)
    if row is None or row.retired_at is not None:
        raise HTTPException(status_code=404, detail=f"Pseudonym {entry_id} not found")
    code = privacy_guard.retire_pseudonym(db, row)
    return {"deleted": entry_id, "code": code, "retired": True}


__all__ += [
    "privacy_status",
    "PrivacyIn",
]
