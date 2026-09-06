"""Central configuration for Agora.

Everything path-, model-, and server-related lives here. Other modules import
from this file rather than hardcoding values (SPEC: "model list lives in
app/config.py MODEL_REGISTRY, not scattered in code").
"""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

#: Everything mutable/user-generated lives under data/ (gitignored).
DATA_DIR = Path(os.environ.get("AGORA_DATA_DIR", BASE_DIR / "data"))

DB_FILENAME = "agora.db"
DB_PATH = DATA_DIR / DB_FILENAME
DATABASE_URL = os.environ.get("AGORA_DATABASE_URL", f"sqlite:///{DB_PATH}")

#: Uploaded student submissions (PDF/PNG/JPEG).
UPLOAD_DIR = DATA_DIR / "submissions"
#: Knowledge documents attached to skills.
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
#: Scratch space for skill bundle export/import.
EXPORT_DIR = DATA_DIR / "exports"

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

#: Encryption key for API credentials. NEVER inside the repo or the data dir.
CONFIG_DIR = Path(os.environ.get("AGORA_CONFIG_DIR", Path.home() / ".config" / "agora"))
SECRET_KEY_PATH = CONFIG_DIR / "secret.key"

#: User-editable model registry overrides (written by the Settings page).
MODEL_REGISTRY_PATH = DATA_DIR / "model_registry.json"


def ensure_dirs() -> None:
    """Create every directory the app writes to. Safe to call repeatedly."""
    for path in (DATA_DIR, UPLOAD_DIR, KNOWLEDGE_DIR, EXPORT_DIR, TEMPLATES_DIR, STATIC_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = int(os.environ.get("AGORA_PORT", "8811"))
APP_NAME = "Agora"
APP_VERSION = "2.0.0"


def base_url() -> str:
    return f"http://{HOST}:{PORT}"


# --------------------------------------------------------------------------
# Providers / models
# --------------------------------------------------------------------------

PROVIDERS = ("anthropic", "openai")
DEFAULT_PROVIDER = "anthropic"
#: A skill may say "auto": grade with whatever provider the professor has a
#: key for (Settings → preferred provider first). See
#: ``app.ai.providers.resolve_provider``. New skills start here — there is no
#: fixed default model; the app shifts with whatever the professor has access to.
AUTO_PROVIDER = "auto"
AUTO_MODEL = "auto"
DEFAULT_SKILL_PROVIDER = AUTO_PROVIDER
#: The order ``auto`` tries cloud providers in when no preferred provider is
#: set. The local model joins only when the professor makes it the preferred
#: provider — a 4B model on a laptop is not a silent stand-in for a frontier one.
AUTO_PROVIDER_ORDER = ("anthropic", "openai")

APP_SETTINGS_FILENAME = "app_settings.json"


def app_settings_path() -> Path:
    """Resolved at call time so tests can repoint ``DATA_DIR``."""
    return DATA_DIR / APP_SETTINGS_FILENAME


def load_app_settings() -> dict[str, Any]:
    """Small app-level preferences (never secrets). Never raises."""
    try:
        with open(app_settings_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def save_app_settings(patch: dict[str, Any]) -> dict[str, Any]:
    current = load_app_settings()
    current.update(patch)
    ensure_dirs()
    with open(app_settings_path(), "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2)
    return current


def preferred_provider() -> Optional[str]:
    """The provider ``auto`` tries first (set in Settings, or on first key save)."""
    value = str(load_app_settings().get("preferred_provider") or "").strip().lower()
    return value or None


def set_preferred_provider(provider: Optional[str]) -> Optional[str]:
    value = str(provider or "").strip().lower() or None
    save_app_settings({"preferred_provider": value})
    return value

#: Per-provider model registry.
#:
#: Anthropic IDs are exact (docs/AI_NOTES.md) — no date suffixes, and these
#: models reject temperature/top_p/top_k and the `thinking` parameter.
#: OpenAI IDs are sensible defaults and are meant to be edited in Settings
#: (their lineup moves fast), which is why the registry is overridable on disk.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "anthropic": {
        "label": "Anthropic",
        "default": "claude-opus-5",
        "editable": False,
        "models": [
            {
                "id": "claude-opus-5",
                "label": "Claude Opus 5",
                "notes": "Highest capability. Default grading model.",
                "max_tokens": 16000,
                "supports_pdf": True,
                "supports_structured_output": True,
            },
            {
                "id": "claude-sonnet-5",
                "label": "Claude Sonnet 5",
                "notes": "Balanced speed/quality. Good for bulk grading runs.",
                "max_tokens": 16000,
                "supports_pdf": True,
                "supports_structured_output": True,
            },
            {
                "id": "claude-haiku-4-5",
                "label": "Claude Haiku 4.5",
                "notes": "Fastest/cheapest. Chat assistant and quick passes.",
                "max_tokens": 8000,
                "supports_pdf": True,
                "supports_structured_output": True,
            },
        ],
    },
    "openai": {
        "label": "OpenAI",
        "default": "gpt-5.6-sol",
        "editable": True,
        "models": [
            {
                "id": "gpt-5.6-sol",
                "label": "GPT-5.6 Sol",
                "notes": "Default OpenAI grading model. Editable in Settings.",
                "max_tokens": 16000,
                "supports_pdf": False,
                "supports_structured_output": True,
            },
            {
                "id": "gpt-5.6-terra",
                "label": "GPT-5.6 Terra",
                "notes": "Cheaper/faster tier. Editable in Settings.",
                "max_tokens": 16000,
                "supports_pdf": False,
                "supports_structured_output": True,
            },
        ],
    },
}

#: Mock provider is always available (no key, no network) — used by tests and
#: by `python run.py --demo`.
MOCK_PROVIDER = "mock"
MOCK_MODEL = "mock-grader-1"

DEFAULT_MODEL = MODEL_REGISTRY[DEFAULT_PROVIDER]["default"]

#: Fallback output budget for a model the registry does not know about. Sized
#: for grading: on thinking-by-default models (Opus 5) `max_tokens` caps
#: thinking *plus* the JSON grade, so a tight cap truncates the result. Prefer
#: ``max_tokens_for(provider, model)``, which reads the registry entry.
DEFAULT_MAX_TOKENS = 16000

#: Effort for grading requests on Anthropic models. Opus 5 thinks by default and
#: `max_tokens` caps thinking plus the JSON grade, so a modest effort keeps a
#: rubric grade well inside the budget; `medium` is the documented sweet spot
#: for short structured tasks. Override with AGORA_GRADING_EFFORT.
GRADING_EFFORT = os.environ.get("AGORA_GRADING_EFFORT", "medium").strip().lower() or "medium"

#: Server-side refusal fallback (beta). Opus 5's safety classifiers can decline
#: a request with a normal 200 + ``stop_reason: "refusal"``; with this on the
#: API re-runs the request on Anthropic's recommended fallback model instead of
#: handing the refusal back, so one flagged essay does not strand a batch.
#: Set AGORA_ANTHROPIC_FALLBACKS=0 to turn it off.
ANTHROPIC_FALLBACKS = os.environ.get("AGORA_ANTHROPIC_FALLBACKS", "1").strip() != "0"
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _load_overrides() -> dict[str, Any]:
    try:
        with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_model_registry() -> dict[str, dict[str, Any]]:
    """Return the model registry with any user overrides merged in.

    Overrides are stored as ``{provider: {"default": id, "models": [...]}}``.
    Only providers already known to the app are honoured.
    """
    registry = json.loads(json.dumps(MODEL_REGISTRY))  # deep copy
    for provider, override in _load_overrides().items():
        if provider not in registry or not isinstance(override, dict):
            continue
        if override.get("models"):
            registry[provider]["models"] = override["models"]
        if override.get("default"):
            registry[provider]["default"] = override["default"]
    return registry


def save_model_registry(overrides: dict[str, Any]) -> None:
    """Persist model-registry overrides (Settings page)."""
    ensure_dirs()
    with open(MODEL_REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(overrides, fh, indent=2)


def models_for(provider: str) -> list[dict[str, Any]]:
    return list(load_model_registry().get(provider, {}).get("models", []))


def model_ids(provider: str) -> list[str]:
    return [m["id"] for m in models_for(provider)]


def default_model_for(provider: str) -> str:
    entry = load_model_registry().get(provider)
    if not entry:
        return DEFAULT_MODEL
    return entry.get("default") or DEFAULT_MODEL


def is_known_model(provider: str, model: str) -> bool:
    return model in model_ids(provider)


def provider_for_model(model: str | None) -> Optional[str]:
    """Which registered provider owns a model id (None when unknown)."""
    if not model:
        return None
    for provider, entry in load_model_registry().items():
        if any(m.get("id") == model for m in entry.get("models", [])):
            return provider
    return None


def model_entry(provider: str, model: str | None) -> dict[str, Any]:
    """The registry row for one model (empty dict when unknown)."""
    if not model:
        return {}
    for entry in models_for(provider):
        if entry.get("id") == model:
            return entry
    return {}


def max_tokens_for(provider: str, model: str | None) -> int:
    """Output budget the registry advertises for a model.

    Used as the default request size so a grading response is never truncated
    by a flat cap that is smaller than what the model actually supports.
    """
    try:
        value = int(model_entry(provider, model).get("max_tokens") or 0)
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else DEFAULT_MAX_TOKENS


def model_supports_pdf(provider: str, model: str | None) -> bool:
    """Whether this model can ingest a PDF natively (registry-declared)."""
    entry = model_entry(provider, model)
    if not entry:
        return provider != "openai"
    return bool(entry.get("supports_pdf", True))


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------

ALLOWED_SUBMISSION_TYPES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


# ==========================================================================
# Increment 1 · PRIVACY MODULE (appended — do not reorder the sections above)
#
# Local (llama.cpp) provider registry + Privacy Guard settings. Everything
# here is additive: existing constants are extended, never rewritten.
# ==========================================================================

# -- local provider --------------------------------------------------------

LOCAL_PROVIDER = "local"
LOCAL_DEFAULT_MODEL = "gemma-3-4b-it"
LOCAL_DEFAULT_BASE_URL = "http://127.0.0.1:3782/v1"
#: llama.cpp needs no key; the OpenAI-compatible wire format wants *something*.
LOCAL_DUMMY_API_KEY = "no-key-needed"
#: The local server is a small model on a laptop — give it room, but bound it.
LOCAL_TIMEOUT_SECONDS = float(os.environ.get("AGORA_LOCAL_TIMEOUT", "180"))

MODEL_REGISTRY[LOCAL_PROVIDER] = {
    "label": "Local (llama.cpp)",
    "default": LOCAL_DEFAULT_MODEL,
    "editable": True,
    "local": True,
    "models": [
        {
            "id": LOCAL_DEFAULT_MODEL,
            "label": "Gemma 3 4B Instruct",
            "notes": "Runs on this machine via llama.cpp. Used by the Privacy Guard.",
            "max_tokens": 4096,
            "supports_pdf": False,
            "supports_structured_output": True,
        }
    ],
}

#: ``local`` joins the selectable providers so a Skill can be pointed at it and
#: Settings can render its status. Appended, so nothing above changes shape.
PROVIDERS = tuple(PROVIDERS) + (LOCAL_PROVIDER,)

#: Providers whose requests leave this machine. The Privacy Guard only rewrites
#: submissions bound for these; ``local``/``mock`` stay on the box and are exempt.
CLOUD_PROVIDERS = ("anthropic", "openai")
ON_DEVICE_PROVIDERS = (LOCAL_PROVIDER, MOCK_PROVIDER)


def is_cloud_provider(provider: str | None) -> bool:
    """True when a call to this provider would leave the machine."""
    return str(provider or "").strip().lower() in CLOUD_PROVIDERS


def requires_api_key(provider: str | None) -> bool:
    name = str(provider or "").strip().lower()
    return name not in ON_DEVICE_PROVIDERS


# -- privacy settings ------------------------------------------------------

PRIVACY_MODE_OFF = "off"
PRIVACY_MODE_WARN = "warn"
PRIVACY_MODE_SWAP = "swap"
PRIVACY_MODES = (PRIVACY_MODE_OFF, PRIVACY_MODE_WARN, PRIVACY_MODE_SWAP)
DEFAULT_PRIVACY_MODE = PRIVACY_MODE_SWAP

PRIVACY_SETTINGS_FILENAME = "privacy_settings.json"

DEFAULT_PRIVACY_SETTINGS: dict[str, Any] = {
    "mode": DEFAULT_PRIVACY_MODE,
    "llm_sweep": True,
    "local_model": {
        "enabled": True,
        "base_url": LOCAL_DEFAULT_BASE_URL,
        "model": LOCAL_DEFAULT_MODEL,
    },
}


def check_local_base_url(url: str | None) -> tuple[bool, str]:
    """Return whether an HTTP address is strictly loopback-only."""
    raw = str(url or "").strip()
    if not raw:
        return False, "no address is configured."
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        return False, f"{parsed.scheme or '(none)'!r} is not an http(s) address."
    host = (parsed.hostname or "").strip().strip("[]")
    if not host:
        return False, "it has no host."
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True, ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False, (
            f"{host!r} is a hostname, not a loopback address, so it may resolve "
            "outside this machine."
        )
    if address.is_loopback:
        return True, ""
    return False, f"{host} is not a loopback address."


def local_base_url_is_local(url: str | None) -> bool:
    return check_local_base_url(url)[0]


def privacy_settings_path() -> Path:
    """Resolved at call time so tests can repoint ``DATA_DIR``."""
    return DATA_DIR / PRIVACY_SETTINGS_FILENAME


def load_privacy_settings() -> dict[str, Any]:
    """Privacy Guard settings with defaults filled in (never raises)."""
    settings = json.loads(json.dumps(DEFAULT_PRIVACY_SETTINGS))
    try:
        with open(privacy_settings_path(), "r", encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        stored = {}
    if isinstance(stored, dict):
        mode = str(stored.get("mode") or "").strip().lower()
        if mode in PRIVACY_MODES:
            settings["mode"] = mode
        if "llm_sweep" in stored:
            settings["llm_sweep"] = bool(stored.get("llm_sweep"))
        local = stored.get("local_model")
        if isinstance(local, dict):
            if local.get("base_url"):
                base_url = str(local["base_url"]).strip()
                if check_local_base_url(base_url)[0]:
                    settings["local_model"]["base_url"] = base_url
            if local.get("model"):
                settings["local_model"]["model"] = str(local["model"]).strip()
            if "enabled" in local:
                settings["local_model"]["enabled"] = bool(local.get("enabled"))
    return settings


def save_privacy_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Merge + persist Privacy Guard settings; returns the stored result."""
    current = load_privacy_settings()
    mode = str(settings.get("mode") or "").strip().lower()
    if mode:
        if mode not in PRIVACY_MODES:
            raise ValueError(
                f"Unknown privacy mode {mode!r}. Known: {', '.join(PRIVACY_MODES)}"
            )
        current["mode"] = mode
    if "llm_sweep" in settings:
        current["llm_sweep"] = bool(settings.get("llm_sweep"))
    local = settings.get("local_model")
    if isinstance(local, dict):
        if local.get("base_url"):
            base_url = str(local["base_url"]).strip()
            ok, reason = check_local_base_url(base_url)
            if not ok:
                raise ValueError(
                    f"{base_url!r} is not a local address: {reason} The local model reads "
                    "the student's text before it is pseudonymized, so it must use a "
                    "loopback address."
                )
            current["local_model"]["base_url"] = base_url
        if local.get("model"):
            current["local_model"]["model"] = str(local["model"]).strip()
        if "enabled" in local:
            current["local_model"]["enabled"] = bool(local.get("enabled"))
    ensure_dirs()
    with open(privacy_settings_path(), "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2)
    return current


def privacy_mode() -> str:
    return load_privacy_settings()["mode"]


def local_model_settings() -> dict[str, Any]:
    return dict(load_privacy_settings()["local_model"])
