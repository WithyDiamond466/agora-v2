"""Provider abstraction: Anthropic, OpenAI, and a deterministic Mock.

Every provider implements the same two calls:

* ``grade(system_prompt, content_blocks, schema) -> dict`` — structured output,
  guaranteed to be a parsed JSON object matching ``schema`` (see
  ``app.ai.grading.GRADE_SCHEMA``).
* ``chat(system_prompt, messages, tools) -> ChatTurn`` — one assistant turn,
  with tool calls parsed into real dicts (never string-matched).

Content blocks
--------------
Blocks are written in **Anthropic shape** (``{"type": "text" | "document" |
"image", ...}``) because that is the richest form; every provider translates
them to its own wire format. Blocks may carry private ``_``-prefixed keys
(``_filename``, ``_text``) that never reach any API — they are used for
filename-derived mock scoring and for graceful degradation when a provider
cannot ingest a PDF natively.

API drift notes (docs/AI_NOTES.md is authoritative and overrides priors):
* Anthropic models reject ``temperature`` / ``top_p`` / ``top_k`` and
  ``thinking: {budget_tokens: N}``. We send none of them.
* Structured outputs use ``output_config={"format": {"type": "json_schema",
  "schema": ...}}``; the first text block is then guaranteed-valid JSON.
* ``stop_reason == "refusal"`` must be checked *before* reading content.
* Model IDs live in ``app.config.MODEL_REGISTRY``, never inline here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from app import config
from app.ai.mock_content import MOCK_CRITERION_COMMENTS

log = logging.getLogger("agora.ai.providers")

try:  # pragma: no cover - import guard; the SDK is in requirements.txt
    import anthropic
except Exception:  # noqa: BLE001
    anthropic = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    import openai
except Exception:  # noqa: BLE001
    openai = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# typed exception chain
# --------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Base class for every failure the engine surfaces to the UI."""

    def __init__(self, message: str, *, provider: str = "", model: str = ""):
        super().__init__(message)
        self.provider = provider
        self.model = model


class ProviderConfigError(ProviderError):
    """Missing API key, unknown provider, unusable model selection."""


class ProviderAuthError(ProviderError):
    """The key was rejected (401/403)."""


class ProviderRateLimitError(ProviderError):
    """429 after the SDK's own retries."""


class ProviderConnectionError(ProviderError):
    """Network/DNS/timeout — nothing reached the API."""


class ProviderResponseError(ProviderError):
    """A response came back but is unusable (truncated, not JSON, wrong shape)."""


class ProviderRefusalError(ProviderError):
    """``stop_reason == "refusal"`` — a graded-failed state, never a crash."""

    def __init__(self, message: str, *, category: str = "", **kwargs: Any):
        super().__init__(message, **kwargs)
        self.category = category


class ProviderUnsupportedError(ProviderError):
    """The selected provider/model cannot ingest this input (e.g. native PDF)."""


# --------------------------------------------------------------------------
# content blocks
# --------------------------------------------------------------------------

PRIVATE_PREFIX = "_"

#: Filenames are never sent to a provider — the professor's upload name usually
#: contains the student's name ("Smith_Jane_essay1.pdf"). Anything that needs a
#: name on the wire uses this one.
NEUTRAL_DOCUMENT_NAME = "attachment.pdf"


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def document_block(
    data: bytes,
    media_type: str = "application/pdf",
    filename: str | None = None,
    text_fallback: str | None = None,
) -> dict[str, Any]:
    """A PDF as an Anthropic ``document`` block (+ private degradation hints)."""
    block: dict[str, Any] = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }
    if filename:
        block["_filename"] = filename
    if text_fallback:
        block["_text"] = text_fallback
    return block


def image_block(
    data: bytes, media_type: str = "image/png", filename: str | None = None
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }
    if filename:
        block["_filename"] = filename
    return block


def block_for_file(
    data: bytes, media_type: str, filename: str | None = None
) -> dict[str, Any]:
    """Pick the right block type for an uploaded submission/knowledge file."""
    if media_type == "application/pdf":
        return document_block(data, media_type, filename)
    if media_type in ("image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"):
        return image_block(data, "image/jpeg" if media_type == "image/jpg" else media_type, filename)
    # Anything else is treated as text so it still reaches the model somehow.
    return text_block(data.decode("utf-8", "replace"))


def public_block(block: dict[str, Any]) -> dict[str, Any]:
    """Strip private ``_`` keys so the block is API-legal."""
    return {k: v for k, v in block.items() if not k.startswith(PRIVATE_PREFIX)}


def public_blocks(blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [public_block(b) for b in blocks]


def blocks_filename(blocks: Sequence[dict[str, Any]]) -> str:
    """First private filename hint in a block list (used by MockProvider)."""
    for block in blocks:
        name = block.get("_filename")
        if name:
            return str(name)
    return ""


def blocks_text(blocks: Sequence[dict[str, Any]]) -> str:
    parts = [str(b.get("text", "")) for b in blocks if b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# chat turn
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One tool invocation requested by the model. ``arguments`` is a real dict."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ChatTurn:
    """One assistant turn: prose, tool calls, and why generation stopped.

    ``stop_reason`` is normalised across providers to the Anthropic vocabulary:
    ``end_turn`` | ``tool_use`` | ``max_tokens`` | ``refusal`` | ``pause_turn``.
    ``content`` holds the provider-native assistant content so a caller can
    replay it verbatim in the next request (required for tool-use loops).
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    content: Any = None
    model: str = ""
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "stop_reason": self.stop_reason,
            "model": self.model,
        }


def tool_result_message(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the single message that carries every tool result of a round.

    ``results`` items: ``{"tool_use_id": str, "content": str, "is_error": bool}``.
    Providers translate this to their own shape (Anthropic: one user message of
    ``tool_result`` blocks; OpenAI: one ``role="tool"`` message per result).
    """
    return {"role": "tool_results", "results": list(results)}


def _as_dict(value: Any) -> dict[str, Any]:
    """Tool inputs are parsed, never string-matched (SPEC non-negotiable)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if value is None:
        return {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


# --------------------------------------------------------------------------
# base provider
# --------------------------------------------------------------------------


class Provider(ABC):
    """Common interface. Instances are cheap; build one per grading run."""

    name: str = "provider"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        max_tokens: int | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or config.default_model_for(self.name)
        self.api_key = api_key
        # Default the output budget from the model's registry entry, not a flat
        # constant: on thinking-by-default models max_tokens caps thinking plus
        # the response, so an undersized cap truncates the JSON grade.
        self.max_tokens = int(max_tokens or config.max_tokens_for(self.name, self.model))
        self._client = client

    # -- required surface -------------------------------------------------

    @abstractmethod
    def grade(
        self, system_prompt: str, content_blocks: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a parsed JSON object conforming to ``schema``."""

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatTurn:
        """Return one assistant turn (text and/or tool calls)."""

    # -- helpers ----------------------------------------------------------

    def model_entry(self) -> dict[str, Any]:
        for entry in config.models_for(self.name):
            if entry.get("id") == self.model:
                return entry
        return {}

    def supports_pdf(self) -> bool:
        return bool(self.model_entry().get("supports_pdf", True))

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "max_tokens": self.max_tokens}

    def _error(self, cls: type[ProviderError], message: str, **kwargs: Any) -> ProviderError:
        return cls(message, provider=self.name, model=self.model, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} model={self.model!r}>"


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

#: At or above this, grading switches to the streaming API (AI_NOTES: long
#: outputs). The registry's default budget (16000) is well past it, so the
#: default grading path streams.
STREAM_THRESHOLD_TOKENS = 8000


class AnthropicProvider(Provider):
    """Claude. Structured outputs, native PDF, no sampling parameters."""

    name = "anthropic"

    @property
    def client(self) -> Any:
        if self._client is None:
            if anthropic is None:  # pragma: no cover - dependency guard
                raise self._error(
                    ProviderConfigError, "The `anthropic` package is not installed."
                )
            if not self.api_key:
                raise self._error(
                    ProviderConfigError,
                    "No Anthropic API key configured. Add one in Settings.",
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    # -- error translation -------------------------------------------------

    def _translate(self, exc: Exception) -> ProviderError:
        """Most-specific-first, per docs/AI_NOTES.md."""
        if anthropic is not None:
            if isinstance(exc, getattr(anthropic, "AuthenticationError", ())):
                return self._error(ProviderAuthError, "Anthropic rejected the API key.")
            if isinstance(exc, getattr(anthropic, "PermissionDeniedError", ())):
                return self._error(
                    ProviderAuthError, "This Anthropic key lacks access to that model."
                )
            if isinstance(exc, getattr(anthropic, "NotFoundError", ())):
                return self._error(
                    ProviderConfigError, f"Unknown Anthropic model {self.model!r}."
                )
            if isinstance(exc, getattr(anthropic, "RateLimitError", ())):
                return self._error(
                    ProviderRateLimitError,
                    "Anthropic rate limit hit — wait a moment and retry the failed items.",
                )
            if isinstance(exc, getattr(anthropic, "BadRequestError", ())):
                return self._error(
                    ProviderResponseError, f"Anthropic rejected the request: {exc}"
                )
            if isinstance(exc, getattr(anthropic, "APIStatusError", ())):
                status = getattr(exc, "status_code", 0)
                return self._error(
                    ProviderResponseError, f"Anthropic API error ({status}): {exc}"
                )
            if isinstance(exc, getattr(anthropic, "APIConnectionError", ())):
                return self._error(
                    ProviderConnectionError,
                    "Could not reach the Anthropic API — check the network connection.",
                )
        if isinstance(exc, ProviderError):
            return exc
        return self._error(ProviderError, f"Anthropic call failed: {exc}")

    # -- request plumbing --------------------------------------------------

    def _create(self, *, stream: bool, **kwargs: Any) -> Any:
        """One request. NOTE: no temperature/top_p/top_k, no `thinking`."""
        client = self.client
        try:
            if stream and hasattr(client.messages, "stream"):
                with client.messages.stream(**kwargs) as streamed:
                    return streamed.get_final_message()
            return client.messages.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated into the typed chain
            raise self._translate(exc) from exc

    def _check_stop(self, response: Any, *, expect_json: bool) -> None:
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            explanation = getattr(details, "explanation", "") or (
                "Claude declined to process this submission."
            )
            raise self._error(
                ProviderRefusalError,
                f"Claude refused this request: {explanation}",
                category=getattr(details, "category", "") or "",
            )
        if expect_json and stop_reason == "max_tokens":
            raise self._error(
                ProviderResponseError,
                "The response was cut off by max_tokens before the JSON was complete — "
                "raise the skill's max_tokens and retry.",
            )

    @staticmethod
    def _first_text(response: Any) -> str:
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "text":
                return getattr(block, "text", "") or ""
        return ""

    # -- Provider surface --------------------------------------------------

    def _fallback_kwargs(self) -> dict[str, Any]:
        """Opt into the server-side refusal fallback (see config.ANTHROPIC_FALLBACKS).

        Sent as a raw header + body key so it works whatever the installed SDK
        knows about the beta; the API ignores neither.
        """
        if not config.ANTHROPIC_FALLBACKS:
            return {}
        return {
            "extra_headers": {"anthropic-beta": config.ANTHROPIC_FALLBACK_BETA},
            "extra_body": {"fallbacks": "default"},
        }

    def grade(
        self, system_prompt: str, content_blocks: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        response = self._create(
            stream=self.max_tokens >= STREAM_THRESHOLD_TOKENS,
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": public_blocks(content_blocks)}],
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                # Haiku 4.5 supports structured output but not effort.
                **({} if self.model.startswith("claude-haiku-") else {"effort": config.GRADING_EFFORT}),
            },
            **self._fallback_kwargs(),
        )
        # Refusals are checked BEFORE touching .content (AI_NOTES).
        self._check_stop(response, expect_json=True)
        # When the fallback ran, the grade was written by a different model;
        # record the one that actually answered.
        self.served_model = getattr(response, "model", None) or self.model

        raw = self._first_text(response)
        if not raw.strip():
            raise self._error(ProviderResponseError, "Claude returned an empty response.")
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise self._error(
                ProviderResponseError, "Structured output was not valid JSON."
            ) from exc
        if not isinstance(parsed, dict):
            raise self._error(
                ProviderResponseError, "Structured output was not a JSON object."
            )
        return parsed

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._normalize_messages(messages),
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = [self._normalize_tool(t) for t in tools]

        response = self._create(stream=False, **kwargs)
        self._check_stop(response, expect_json=False)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(response, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", "") or "",
                        name=getattr(block, "name", "") or "",
                        arguments=_as_dict(getattr(block, "input", None)),
                    )
                )
        return ChatTurn(
            text="\n".join(p for p in text_parts if p).strip(),
            tool_calls=tool_calls,
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            content=getattr(response, "content", None),
            model=getattr(response, "model", self.model),
            raw=response,
        )

    # -- message/tool normalisation ---------------------------------------

    @staticmethod
    def _normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
        out = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema") or tool.get("parameters") or {},
        }
        if tool.get("strict"):
            out["strict"] = True
        return out

    @classmethod
    def _normalize_messages(cls, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Accept plain text, native blocks, or ``tool_result_message`` payloads."""
        out: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            if role in ("tool", "tool_results"):
                results = message.get("results")
                if results is None and isinstance(message.get("content"), list):
                    results = message["content"]
                blocks = []
                for res in results or []:
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": res.get("tool_use_id", ""),
                        "content": res.get("content", ""),
                    }
                    if res.get("is_error"):
                        block["is_error"] = True
                    blocks.append(block)
                out.append({"role": "user", "content": blocks})
                continue

            content = message.get("content", "")
            if isinstance(content, list):
                content = [
                    public_block(b) if isinstance(b, dict) else b for b in content
                ]
            out.append({"role": role, "content": content})
        return out


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


class OpenAIProvider(Provider):
    """OpenAI chat completions with a JSON-schema response format.

    Model IDs come from ``config.MODEL_REGISTRY`` (editable in Settings) — their
    lineup moves fast, so nothing is hardcoded here. Where a capability is
    uncertain (native PDF ingestion) we degrade with a clear, typed error rather
    than guessing an API shape (docs/AI_NOTES.md).
    """

    name = "openai"

    @property
    def client(self) -> Any:
        if self._client is None:
            if openai is None:  # pragma: no cover - dependency guard
                raise self._error(ProviderConfigError, "The `openai` package is not installed.")
            if not self.api_key:
                raise self._error(
                    ProviderConfigError, "No OpenAI API key configured. Add one in Settings."
                )
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def _translate(self, exc: Exception) -> ProviderError:
        if openai is not None:
            if isinstance(exc, getattr(openai, "AuthenticationError", ())):
                return self._error(ProviderAuthError, "OpenAI rejected the API key.")
            if isinstance(exc, getattr(openai, "PermissionDeniedError", ())):
                return self._error(
                    ProviderAuthError, "This OpenAI key lacks access to that model."
                )
            if isinstance(exc, getattr(openai, "NotFoundError", ())):
                return self._error(
                    ProviderConfigError,
                    f"Unknown OpenAI model {self.model!r} — edit the model list in Settings.",
                )
            if isinstance(exc, getattr(openai, "RateLimitError", ())):
                return self._error(
                    ProviderRateLimitError, "OpenAI rate limit hit — retry the failed items."
                )
            if isinstance(exc, getattr(openai, "BadRequestError", ())):
                return self._error(ProviderResponseError, f"OpenAI rejected the request: {exc}")
            if isinstance(exc, getattr(openai, "APIStatusError", ())):
                status = getattr(exc, "status_code", 0)
                return self._error(ProviderResponseError, f"OpenAI API error ({status}): {exc}")
            if isinstance(exc, getattr(openai, "APIConnectionError", ())):
                return self._error(
                    ProviderConnectionError,
                    "Could not reach the OpenAI API — check the network connection.",
                )
        if isinstance(exc, ProviderError):
            return exc
        return self._error(ProviderError, f"OpenAI call failed: {exc}")

    # -- translation of neutral blocks ------------------------------------

    def _content_parts(self, blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for block in blocks:
            btype = block.get("type")
            if btype == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image":
                source = block.get("source", {})
                media = source.get("media_type", "image/png")
                data = source.get("data", "")
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
                )
            elif btype == "document":
                parts.extend(self._document_parts(block))
        return parts

    def _document_parts(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        """PDFs: native only when the registry entry opts in; else degrade.

        The professor's original upload filename (``_filename``) is a private
        hint used for mock scoring only: it usually contains the student's
        name, so it must never reach a provider (SPEC principle 2). Every
        attachment goes out under a neutral name.
        """
        source = block.get("source", {})
        if self.supports_pdf():
            return [
                {
                    "type": "file",
                    "file": {
                        "filename": NEUTRAL_DOCUMENT_NAME,
                        "file_data": (
                            f"data:{source.get('media_type', 'application/pdf')};base64,"
                            f"{source.get('data', '')}"
                        ),
                    },
                }
            ]
        fallback = block.get("_text")
        if fallback:
            return [
                {
                    "type": "text",
                    "text": (
                        "[Extracted text of the attached PDF — this model cannot read the "
                        f"PDF natively]\n\n{fallback}"
                    ),
                }
            ]
        raise self._error(
            ProviderUnsupportedError,
            f"{self.model} cannot read PDF submissions directly. Grade this assignment "
            "with an Anthropic skill, or enable native PDF for this model in Settings "
            "once your account supports it.",
        )

    def _completion(self, **kwargs: Any) -> Any:
        """Call chat.completions, degrading over the max-tokens parameter rename."""
        try:
            return self.client.chat.completions.create(
                max_completion_tokens=self.max_tokens, **kwargs
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            if "max_completion_tokens" in str(exc):
                try:
                    return self.client.chat.completions.create(
                        max_tokens=self.max_tokens, **kwargs
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    raise self._translate(retry_exc) from retry_exc
            raise self._translate(exc) from exc

    @staticmethod
    def _message(response: Any) -> Any:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        return getattr(choices[0], "message", None)

    _FINISH_REASONS = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
    }

    # -- Provider surface --------------------------------------------------

    def grade(
        self, system_prompt: str, content_blocks: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        parts = self._content_parts(content_blocks)
        response = self._completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": parts},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "agora_grade_result", "strict": True, "schema": schema},
            },
        )
        message = self._message(response)
        if message is None:
            raise self._error(ProviderResponseError, "OpenAI returned no choices.")
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise self._error(ProviderRefusalError, f"The model refused this request: {refusal}")

        finish = getattr((getattr(response, "choices", None) or [None])[0], "finish_reason", "")
        if finish == "length":
            raise self._error(
                ProviderResponseError,
                "The response was cut off before the JSON was complete — raise max_tokens.",
            )

        raw = getattr(message, "content", "") or ""
        if isinstance(raw, list):  # some SDK versions return part objects
            raw = "".join(getattr(p, "text", "") or "" for p in raw)
        if not raw.strip():
            raise self._error(ProviderResponseError, "OpenAI returned an empty response.")
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise self._error(
                ProviderResponseError, "Structured output was not valid JSON."
            ) from exc
        if not isinstance(parsed, dict):
            raise self._error(ProviderResponseError, "Structured output was not a JSON object.")
        return parsed

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._normalize_messages(system_prompt, messages),
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or t.get("parameters") or {},
                    },
                }
                for t in tools
            ]
        response = self._completion(**kwargs)
        message = self._message(response)
        if message is None:
            raise self._error(ProviderResponseError, "OpenAI returned no choices.")
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise self._error(ProviderRefusalError, f"The model refused this request: {refusal}")

        tool_calls = []
        for call in getattr(message, "tool_calls", None) or []:
            fn = getattr(call, "function", None)
            tool_calls.append(
                ToolCall(
                    id=getattr(call, "id", "") or "",
                    name=getattr(fn, "name", "") or "",
                    # Arguments arrive as a JSON string — parse, never string-match.
                    arguments=_as_dict(getattr(fn, "arguments", None)),
                )
            )
        finish = getattr((getattr(response, "choices", None) or [None])[0], "finish_reason", "stop")
        if finish == "content_filter":
            # Same contract as Anthropic's stop_reason == "refusal": a refusal
            # is a typed error both providers raise, never a silent turn.
            raise self._error(
                ProviderRefusalError,
                "The model declined to answer: the request tripped OpenAI's content filter.",
                category="content_filter",
            )
        text = getattr(message, "content", "") or ""
        if isinstance(text, list):
            text = "".join(getattr(p, "text", "") or "" for p in text)
        return ChatTurn(
            text=text.strip(),
            tool_calls=tool_calls,
            stop_reason=self._FINISH_REASONS.get(finish, "end_turn"),
            content=message,
            model=getattr(response, "model", self.model),
            raw=response,
        )

    @staticmethod
    def _assistant_message(content: Any) -> dict[str, Any]:
        """Replay one assistant turn in OpenAI shape.

        Blocks arrive in Anthropic shape (``text`` + ``tool_use``). ``tool_use``
        must become ``tool_calls`` on the assistant message — otherwise the
        following ``role="tool"`` messages have no call to answer and the API
        rejects the request with a 400.
        """
        if not isinstance(content, (list, tuple)):
            return {"role": "assistant", "content": content}

        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                texts.append(str(block.get("text", "")))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": json.dumps(_as_dict(block.get("input"))),
                        },
                    }
                )
        text = "\n".join(t for t in texts if t).strip()
        # An assistant turn that only called tools carries no content.
        out: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out

    def _normalize_messages(
        self, system_prompt: str, messages: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        for message in messages:
            role = message.get("role", "user")
            if role in ("tool", "tool_results"):
                results = message.get("results")
                if results is None and isinstance(message.get("content"), list):
                    results = message["content"]
                for res in results or []:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": res.get("tool_use_id", ""),
                            "content": str(res.get("content", "")),
                        }
                    )
                continue
            content = message.get("content", "")
            if role == "assistant":
                out.append(self._assistant_message(content))
                continue
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                # Anthropic-shaped tool results carried on a user message.
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(block.get("content", "")),
                            }
                        )
                continue
            if isinstance(content, list):
                content = self._content_parts([b for b in content if isinstance(b, dict)])
            out.append({"role": role, "content": content})
        return out


# --------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------

MOCK_MISCONCEPTIONS = [
    "conflates legality with morality",
    "misapplies utilitarian calculus",
    "treats correlation as causation",
    "strawmans the opposing view",
    "cites source without engaging it",
    "ignores stakeholder scope",
    "assumes technological inevitability",
]

MOCK_STRENGTHS = [
    "clear thesis statement",
    "well-chosen concrete case",
    "honest engagement with objections",
    "tight, readable prose",
    "careful use of course readings",
]


#: Matches the rubric lines rendered by ``app.ai.grading.render_rubric_text``.
_RUBRIC_LINE_RE = re.compile(
    r"^\s*-\s*\[(?P<key>[^\]]+)\]\s*(?P<title>[^(]*)\(max\s*(?P<max>[0-9]+(?:\.[0-9]+)?)\s*points\)",
    re.MULTILINE,
)
_NUMBER_RE = re.compile(r"\d+")
#: The mock has no model, so it reads the payload the tool already returned
#: rather than announcing that it read something.
def _pct(value: Any) -> str:
    return "%.1f%%" % float(value or 0)


def _mock_payload_answer(payload: dict[str, Any]) -> str:
    """Render one decoded local tool payload without exposing raw JSON."""
    parts: list[str] = []
    if "label" in payload and "average_percent" in payload:
        parts.append(f"{payload.get('label')} averages {_pct(payload.get('average_percent'))}")
        if payload.get("best_percent") is not None:
            parts.append(f"best {_pct(payload.get('best_percent'))}")
        if payload.get("worst_percent") is not None:
            parts.append(f"lowest {_pct(payload.get('worst_percent'))}")
        tags = []
        for misconception in payload.get("misconceptions") or []:
            tag = misconception.get("tag") if isinstance(misconception, dict) else misconception
            if tag:
                tags.append(str(tag))
        if tags:
            parts.append("recurring misconceptions: " + ", ".join(tags))
        return "; ".join(parts) + "."

    assignments = payload.get("assignments") or []
    if "name" in payload and "student_count" in payload:
        head = f"{payload.get('name')} has {payload.get('student_count')} students"
        totals = payload.get("totals") or {}
        if totals.get("average_percent") is not None:
            head += f"; course average {_pct(totals.get('average_percent'))}"
        parts.append(head + ".")
        for assignment in assignments:
            line = "%s: %s graded of %s submissions" % (
                assignment.get("name"),
                assignment.get("graded"),
                assignment.get("submissions"),
            )
            outstanding = [
                "%s %s" % (assignment.get(field), field)
                for field in ("pending", "grading", "failed")
                if assignment.get(field)
            ]
            if outstanding:
                line += " (" + ", ".join(outstanding) + ")"
            parts.append(line + ".")
        attention = payload.get("students_needing_attention") or []
        if attention:
            parts.append(
                "Lowest graded averages: "
                + ", ".join(
                    f"{row.get('label')} {_pct(row.get('average_percent'))}"
                    for row in attention
                )
                + "."
            )
        return "\n".join(parts)

    progress = payload.get("progress") or {}
    if "name" in payload and progress:
        parts.append(str(payload.get("name")))
        for field in ("graded", "pending", "failed", "ungraded", "submissions"):
            if field in progress:
                parts.append(f"{field}: {progress.get(field)}")
        if payload.get("average_percent") is not None:
            parts.append(f"average: {_pct(payload.get('average_percent'))}")
        return "; ".join(parts) + "."

    for field in ("note", "status", "navigating_to", "warning"):
        if payload.get(field) is not None:
            parts.append(str(payload.get(field)))
    if parts:
        return " ".join(parts)
    return "The tool returned data I do not have a summary for."


def _mock_tool_answer(messages: list[dict[str, Any]]) -> Optional[str]:
    results: list[dict[str, Any]] | None = None
    for message in reversed(messages):
        if message.get("role") == "tool_results":
            results = message.get("results") or []
            break
    if results is None:
        return None

    payloads: list[dict[str, Any]] = []
    for result in results:
        try:
            loaded = json.loads(result.get("content") or "")
        except (TypeError, ValueError):
            loaded = {}
        payload = loaded if isinstance(loaded, dict) else {}
        if result.get("is_error"):
            error = payload.get("error") or "The tool returned an unreadable error."
            return f"I couldn't retrieve that data: {error}"
        if payload:
            payloads.append(payload)
    if not payloads:
        return "The tool returned nothing I can summarise."
    return " ".join(_mock_payload_answer(payload) for payload in payloads)


#: A student is only ever referred to by "#NN" or "student NN" — never by a
#: bare number, which is almost always the course code (e.g. "PHIL 210").
_STUDENT_REF_RE = re.compile(r"(?:#|student\s+)(\d{1,3})\b", re.IGNORECASE)
#: The course id is already in the system prompt; scraping it out of the
#: question turns "PHIL 210" into course 210, which does not exist.
_CONTEXT_COURSE_RE = re.compile(r"Course in context: id=(\d+)")
_KNOWN_COURSE_RE = re.compile(r"^Known course: id=(\d+), name=(.+)$", re.MULTILINE)

MOCK_CHAT_TOOLS = (
    "get_course_summary",
    "get_student_summary",
    "get_assignment_results",
    "navigate_to",
    "start_grading",
)

_PAGE_KEYWORDS = {
    "analytics": "analytics",
    "dashboard": "analytics",
    "skill": "skills",
    "setting": "settings",
    "grading": "grading",
    "home": "home",
    "course": "courses",
}


def _fraction(*parts: Any) -> float:
    """Stable 0..1 value — the whole point of the mock is reproducibility."""
    digest = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _band(pct: float) -> str:
    if pct >= 0.9:
        return "strong"
    if pct >= 0.78:
        return "solid"
    if pct >= 0.62:
        return "developing"
    return "weak"


class MockProvider(Provider):
    """Deterministic, offline, schema-valid. Every test and ``--demo`` uses it.

    Scores derive from a hash of the submission filename, so the same file
    always grades the same way while a class of files shows real spread.
    """

    name = config.MOCK_PROVIDER

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        kwargs.pop("api_key", None)
        super().__init__(model or config.MOCK_MODEL, api_key=None, **kwargs)

    def supports_pdf(self) -> bool:
        return True

    # -- rubric discovery --------------------------------------------------

    @staticmethod
    def _criteria_from_prompt(system_prompt: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for match in _RUBRIC_LINE_RE.finditer(system_prompt or ""):
            out.append(
                {
                    "key": match.group("key").strip(),
                    "title": match.group("title").strip(),
                    "max_points": float(match.group("max")),
                }
            )
        return out

    @staticmethod
    def _criteria_from_rubric(rubric: Any) -> list[dict[str, Any]]:
        if isinstance(rubric, dict):
            criteria = rubric.get("criteria") or []
        elif isinstance(rubric, (list, tuple)):
            criteria = list(rubric)
        else:
            criteria = getattr(rubric, "criteria", None) or []
        out = []
        for crit in criteria:
            if not isinstance(crit, dict) or not crit.get("key"):
                continue
            try:
                max_points = float(crit.get("max_points") or 0)
            except (TypeError, ValueError):
                max_points = 0.0
            out.append(
                {
                    "key": str(crit["key"]),
                    "title": str(crit.get("title") or crit["key"]),
                    "max_points": max_points,
                }
            )
        return out

    # -- Provider surface --------------------------------------------------

    def grade(  # type: ignore[override]
        self,
        system_prompt: str = "",
        content_blocks: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        *,
        rubric: Any = None,
        submission: Any = None,
        filename: str | None = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        """Schema-valid grade result.

        The extra keyword arguments (``rubric``, ``submission``, ``filename``)
        are optional conveniences for callers that have the rubric in hand
        (``app.seed``); the three-positional-argument contract is unchanged.
        """
        blocks = list(content_blocks or [])
        criteria = self._criteria_from_rubric(rubric) or self._criteria_from_prompt(system_prompt)
        if not criteria:
            criteria = [{"key": "overall", "title": "Overall", "max_points": 100.0}]

        name = filename or ""
        if not name and isinstance(submission, dict):
            name = str(submission.get("filename") or submission.get("file_path") or "")
        if not name:
            name = blocks_filename(blocks)
        if not name:
            # Fall back to the anonymized header text so distinct students still
            # get distinct grades.
            name = blocks_text(blocks)[:120] or "submission"

        ability = 0.52 + 0.44 * _fraction(name, "ability")
        result_criteria: list[dict[str, Any]] = []
        for crit in criteria:
            max_points = float(crit.get("max_points") or 0)
            noise = (_fraction(name, crit["key"]) - 0.5) * 0.34
            fraction = min(1.0, max(0.25, ability + noise))
            score = round(fraction * max_points * 2) / 2 if max_points else 0.0
            band = _band(score / max_points if max_points else 1.0)
            result_criteria.append(
                {
                    "key": crit["key"],
                    "score": score,
                    "comment": MOCK_CRITERION_COMMENTS.get(
                        crit["key"], MOCK_CRITERION_COMMENTS["thesis"]
                    )[band],
                }
            )

        total = sum(float(c.get("max_points") or 0) for c in criteria)
        earned = sum(c["score"] for c in result_criteria)
        pct = earned / total if total else 1.0

        tag_count = 0 if pct >= 0.9 else 1 if pct >= 0.78 else 2 if pct >= 0.62 else 3
        start = int(_fraction(name, "tags") * len(MOCK_MISCONCEPTIONS))
        misconceptions = [
            MOCK_MISCONCEPTIONS[(start + i) % len(MOCK_MISCONCEPTIONS)] for i in range(tag_count)
        ]

        strength_count = 3 if pct >= 0.9 else 2 if pct >= 0.7 else 1
        s_start = int(_fraction(name, "strengths") * len(MOCK_STRENGTHS))
        strengths = [
            MOCK_STRENGTHS[(s_start + i) % len(MOCK_STRENGTHS)] for i in range(strength_count)
        ]

        if pct >= 0.9:
            summary = "Excellent work. The argument is disciplined and the objection is met head-on."
        elif pct >= 0.78:
            summary = "A good submission with a clear line of argument; evidence could do more work."
        elif pct >= 0.62:
            summary = "The beginnings of a real argument are here, but it stays at the level of summary."
        else:
            summary = "This submission does not yet meet the rubric; see the per-criterion notes."

        return {
            "criteria": result_criteria,
            "summary_feedback": summary,
            "misconceptions": misconceptions,
            "strengths": strengths,
        }

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatTurn:
        """Canned but structurally honest tool-calling behaviour.

        Selection is keyword-driven *inside the mock only* — real providers pick
        tools themselves, and callers still parse ``ToolCall.arguments`` as a
        dict, exactly as they must in production.
        """
        tool_names = [t.get("name", "") for t in (tools or [])]
        last_user = ""
        for message in messages:
            role = message.get("role")
            if role == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    content = blocks_text([b for b in content if isinstance(b, dict)])
                last_user = str(content)

        tool_answer = _mock_tool_answer(messages)
        if tool_answer is not None:
            return ChatTurn(
                text=tool_answer,
                tool_calls=[],
                stop_reason="end_turn",
                model=self.model,
            )

        lowered = last_user.lower()
        numbers = [int(n) for n in _NUMBER_RE.findall(last_user)]
        first_id = numbers[0] if numbers else 1

        chosen: Optional[ToolCall] = None
        if "grade" in lowered and "start_grading" in tool_names:
            chosen = ToolCall("mock_tool_1", "start_grading", {"assignment_id": first_id})
        elif (
            any(word in lowered for word in ("go to", "open ", "navigate", "take me"))
            and "navigate_to" in tool_names
        ):
            page = "home"
            for keyword, target in _PAGE_KEYWORDS.items():
                if keyword in lowered:
                    page = target
                    break
            chosen = ToolCall("mock_tool_1", "navigate_to", {"page": page})
        elif _STUDENT_REF_RE.search(last_user) and "get_student_summary" in tool_names:
            student_number = int(_STUDENT_REF_RE.search(last_user).group(1))
            context_course = _CONTEXT_COURSE_RE.search(system_prompt)
            if context_course is None:
                return ChatTurn(
                    text=f"Open a course before asking about Student #{student_number}.",
                    tool_calls=[],
                    stop_reason="end_turn",
                    model=self.model,
                )
            chosen = ToolCall(
                "mock_tool_1",
                "get_student_summary",
                {
                    "course_id": int(context_course.group(1)),
                    "student_number": student_number,
                },
            )
        elif (
            any(word in lowered for word in ("assignment", "result", "score"))
            and "get_assignment_results" in tool_names
        ):
            chosen = ToolCall(
                "mock_tool_1", "get_assignment_results", {"assignment_id": first_id}
            )
        elif (
            any(
                word in lowered
                for word in (
                    "course",
                    "class",
                    "how is",
                    "summary",
                    "struggl",
                    "students",
                    "weakest",
                    "misconception",
                )
            )
            and "get_course_summary" in tool_names
        ):
            matched_ids = set()
            for known_id, encoded_name in _KNOWN_COURSE_RE.findall(system_prompt):
                try:
                    course_name = json.loads(encoded_name)
                except (TypeError, ValueError):
                    continue
                if not isinstance(course_name, str):
                    continue
                course_code = course_name.split(" · ", 1)[0]
                if (
                    re.search(
                        rf"(?<![A-Za-z0-9]){re.escape(course_name)}(?![A-Za-z0-9])",
                        last_user,
                        re.IGNORECASE,
                    )
                    or re.search(
                        rf"(?<![A-Za-z0-9]){re.escape(course_code)}(?![A-Za-z0-9])",
                        last_user,
                        re.IGNORECASE,
                    )
                ):
                    matched_ids.add(int(known_id))
            if len(matched_ids) > 1:
                return ChatTurn(
                    text="Open a course before asking for a course summary.",
                    tool_calls=[],
                    stop_reason="end_turn",
                    model=self.model,
                )
            context_course = _CONTEXT_COURSE_RE.search(system_prompt)
            if len(matched_ids) == 1:
                course_id = matched_ids.pop()
            elif context_course is not None:
                course_id = int(context_course.group(1))
            else:
                return ChatTurn(
                    text="Open a course before asking for a course summary.",
                    tool_calls=[],
                    stop_reason="end_turn",
                    model=self.model,
                )
            chosen = ToolCall(
                "mock_tool_1",
                "get_course_summary",
                {"course_id": course_id},
            )

        if chosen is not None:
            return ChatTurn(
                text="", tool_calls=[chosen], stop_reason="tool_use", model=self.model
            )
        return ChatTurn(
            text=(
                "MockProvider here — no API key is configured, so I am answering from a canned "
                "script. Ask me about a course, a student, or say \"grade assignment 1\" to see "
                "tool calling work."
            ),
            tool_calls=[],
            stop_reason="end_turn",
            model=self.model,
        )


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    config.MOCK_PROVIDER: MockProvider,
}

#: Set to a provider name to force it everywhere (tests / offline demos).
FORCE_PROVIDER_ENV = "AGORA_AI_PROVIDER"


def _settings_from(skill_or_settings: Any) -> dict[str, Any]:
    """Accept a Skill row, a settings dict, a provider name, or None."""
    if skill_or_settings is None:
        return {}
    if isinstance(skill_or_settings, str):
        return {"provider": skill_or_settings}
    if isinstance(skill_or_settings, dict):
        return dict(skill_or_settings)
    return {
        "provider": getattr(skill_or_settings, "provider", None),
        "model": getattr(skill_or_settings, "model", None),
        "max_tokens": getattr(skill_or_settings, "max_tokens", None),
    }


@dataclass
class Resolution:
    """Which provider/model a request will actually use, and why."""

    provider: str
    model: str
    requested_provider: str
    requested_model: Optional[str] = None
    #: Human-readable reason when the answer differs from what was asked.
    note: Optional[str] = None
    #: False when nothing could run: the request is still shaped for
    #: ``provider`` so callers can inspect it, and ``get_provider`` raises.
    available: bool = True

    @property
    def substituted(self) -> bool:
        return self.provider != self.requested_provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "requested_provider": self.requested_provider,
            "requested_model": self.requested_model,
            "substituted": self.substituted,
            "available": self.available,
            "note": self.note,
        }


def _has_key(db: Any, provider_name: str) -> bool:
    from app import security  # local import keeps app.ai import-light

    try:
        if db is not None:
            return bool(security.get_api_key(db, provider_name))
    except Exception:  # noqa: BLE001 - unreadable key == no key
        return False
    env_name = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider_name, "")
    return bool(os.environ.get(env_name)) if env_name else False


def _provider_available(db: Any, provider_name: str) -> bool:
    """Could a request to this provider actually be made right now?"""
    if provider_name == config.MOCK_PROVIDER:
        return True
    if provider_name == config.LOCAL_PROVIDER:
        return bool(config.local_model_settings().get("enabled", True))
    if provider_name in config.CLOUD_PROVIDERS:
        return _has_key(db, provider_name)
    return False


def _model_for(provider_name: str, model: Optional[str], *, keep: bool) -> str:
    """``keep`` = the caller's model belongs to this provider; otherwise the default."""
    if provider_name == config.MOCK_PROVIDER:
        return config.MOCK_MODEL
    if provider_name == config.LOCAL_PROVIDER and not (keep and model):
        return str(config.local_model_settings().get("model") or config.LOCAL_DEFAULT_MODEL)
    if keep and model and model != config.AUTO_MODEL:
        return model
    return config.default_model_for(provider_name)


def resolve_provider(
    db: Any, provider: Optional[str], model: Optional[str] = None
) -> Resolution:
    """Turn what a skill asks for into what can run on this machine right now.

    * a pinned provider is used as asked — never swapped for another one. If
      it has no key, ``available`` is False and ``get_provider`` raises the
      usual "No API key configured" error when the request is actually made;
    * ``auto`` goes to the preferred provider (Settings), then through
      ``config.AUTO_PROVIDER_ORDER``, taking the first with a key and saying
      so in ``note``. The local model is used only when it is the preferred
      provider. Nothing available → ``available=False`` with the reason.

    ``AGORA_AI_PROVIDER`` still forces a provider for every request (tests).
    """
    forced = os.environ.get(FORCE_PROVIDER_ENV, "").strip().lower()
    requested = str(provider or config.DEFAULT_SKILL_PROVIDER).strip().lower()
    requested_model = (model or "").strip() or None
    if forced:
        keep = forced == requested
        return Resolution(forced, _model_for(forced, requested_model, keep=keep), requested, requested_model)

    if requested != config.AUTO_PROVIDER:
        if requested not in PROVIDER_CLASSES:
            raise ProviderConfigError(
                f"Unknown provider {requested!r}. Known providers: "
                f"{', '.join(sorted(PROVIDER_CLASSES))}."
            )
        available = _provider_available(db, requested)
        return Resolution(
            requested,
            _model_for(requested, requested_model, keep=True),
            requested,
            requested_model,
            note=None if available else (
                f"No API key configured for {requested}. Add one in Settings, set the skill "
                f"to automatic, or set its provider to '{config.MOCK_PROVIDER}' to grade offline."
            ),
            available=available,
        )

    preferred = config.preferred_provider()
    candidates: list[str] = [preferred] if preferred else []
    candidates.extend(p for p in config.AUTO_PROVIDER_ORDER if p not in candidates)
    for name in candidates:
        if name not in PROVIDER_CLASSES or not _provider_available(db, name):
            continue
        if preferred is None:
            note = None if name == config.AUTO_PROVIDER_ORDER[0] else (
                f"Using {name} — the first provider with a key configured."
            )
        else:
            note = None if name == preferred else (
                f"No key for the preferred provider ({preferred}); using {name}."
            )
        return Resolution(name, _model_for(name, requested_model, keep=False), requested, requested_model, note)

    fallback = preferred or config.DEFAULT_PROVIDER
    return Resolution(
        fallback,
        _model_for(fallback, None, keep=False),
        requested,
        requested_model,
        note=NO_PROVIDER_MESSAGE,
        available=False,
    )


NO_PROVIDER_MESSAGE = (
    "No AI provider is available: add an Anthropic or OpenAI API key in Settings, "
    "or enable the local model."
)


def get_provider(skill_or_settings: Any = None, db: Any = None) -> Provider:
    """Build the provider for a Skill (or explicit settings).

    Keys are read through ``app.security`` (env var first, then the encrypted
    ``ApiCredential`` row) and are never logged or persisted here. ``auto``
    (and a model of ``auto``) go through ``resolve_provider`` first.
    """
    settings = _settings_from(skill_or_settings)
    provider_name = (
        os.environ.get(FORCE_PROVIDER_ENV) or settings.get("provider") or config.DEFAULT_PROVIDER
    )
    provider_name = str(provider_name).strip().lower()
    if provider_name == config.AUTO_PROVIDER or str(settings.get("model") or "") == config.AUTO_MODEL:
        resolution = resolve_provider(db, provider_name, settings.get("model"))
        if not resolution.available and provider_name == config.AUTO_PROVIDER:
            raise ProviderConfigError(NO_PROVIDER_MESSAGE, provider=provider_name)
        provider_name = resolution.provider
        settings["model"] = resolution.model

    provider_cls = PROVIDER_CLASSES.get(provider_name)
    if provider_cls is None:
        raise ProviderConfigError(
            f"Unknown provider {provider_name!r}. Known providers: "
            f"{', '.join(sorted(PROVIDER_CLASSES))}."
        )

    model = settings.get("model") or None
    max_tokens = settings.get("max_tokens") or None

    if provider_cls is MockProvider:
        return MockProvider(model=model or config.MOCK_MODEL, max_tokens=max_tokens)

    # --- Increment 1 · privacy module: local provider glue ---------------
    # The llama.cpp server runs on this machine and needs no API key, so it
    # short-circuits the credential lookup below (same shape as MockProvider).
    if provider_name == config.LOCAL_PROVIDER:
        return LocalProvider(
            model=model,
            max_tokens=max_tokens,
            base_url=settings.get("base_url"),
        )
    # --- end privacy module glue -----------------------------------------

    if model and not config.is_known_model(provider_name, model):
        log.warning(
            "Model %r is not in the registry for %s — using it anyway (Settings may be ahead).",
            model,
            provider_name,
        )
    model = model or config.default_model_for(provider_name)

    api_key: Optional[str] = settings.get("api_key")
    if not api_key:
        from app import security  # local import keeps app.ai import-light

        if db is not None:
            try:
                api_key = security.get_api_key(db, provider_name)
            except security.SecretError as exc:
                raise ProviderConfigError(str(exc), provider=provider_name, model=model) from exc
        else:
            env_name = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(
                provider_name, ""
            )
            api_key = os.environ.get(env_name) if env_name else None

    if not api_key:
        raise ProviderConfigError(
            f"No API key configured for {provider_name}. Add one in Settings, or set the "
            f"skill's provider to '{config.MOCK_PROVIDER}' to grade offline.",
            provider=provider_name,
            model=model,
        )

    return provider_cls(model=model, api_key=api_key, max_tokens=max_tokens)


__all__ = [
    "ChatTurn",
    "ToolCall",
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "MockProvider",
    "Resolution",
    "resolve_provider",
    "get_provider",
    "tool_result_message",
    "text_block",
    "document_block",
    "image_block",
    "block_for_file",
    "public_block",
    "public_blocks",
    "NEUTRAL_DOCUMENT_NAME",
    "ProviderError",
    "ProviderConfigError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderConnectionError",
    "ProviderResponseError",
    "ProviderRefusalError",
    "ProviderUnsupportedError",
    "MOCK_CHAT_TOOLS",
]


# ==========================================================================
# Increment 1 · PRIVACY MODULE — LocalProvider (appended; nothing above moved)
#
# An OpenAI-compatible client for a llama.cpp server running on this machine
# (default: Gemma 3 4B at http://127.0.0.1:3782/v1). It implements the full
# Provider surface, so it can grade, chat, back a Skill, or — its first job —
# run the Privacy Guard's local PII sweep. No API key: llama.cpp ignores it,
# but the wire format wants a bearer token, so a dummy is sent.
# ==========================================================================

try:  # pragma: no cover - import guard; httpx is in requirements.txt
    import httpx
except Exception:  # noqa: BLE001
    httpx = None  # type: ignore[assignment]


#: Fenced ```json blocks are stripped before parsing (small models love them).
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

LOCAL_JSON_INSTRUCTION = (
    "Reply with a single JSON object and nothing else. No prose, no markdown "
    "fences, no explanation before or after the JSON."
)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object out of a small model's reply.

    Tolerates code fences and leading/trailing chatter by falling back to the
    first balanced ``{...}`` span. Raises ``ValueError`` when nothing parses.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    stripped = _FENCE_RE.sub("", text).strip()
    for candidate in (stripped, text):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(stripped[start : index + 1])
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = stripped.find("{", start + 1)
    raise ValueError("no JSON object found in the response")


class LocalProvider(Provider):
    """llama.cpp / any OpenAI-compatible server running on localhost."""

    name = config.LOCAL_PROVIDER

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        client: Any = None,
        timeout: float | None = None,
    ) -> None:
        settings = config.local_model_settings()
        raw_base = base_url or settings.get("base_url") or config.LOCAL_DEFAULT_BASE_URL
        ok, reason = config.check_local_base_url(str(raw_base))
        if not ok:
            raise ProviderConfigError(
                f"The local provider address {raw_base!r} is not loopback-only: {reason}",
                provider=self.name,
                model=model or settings.get("model") or config.LOCAL_DEFAULT_MODEL,
            )
        super().__init__(
            model or settings.get("model") or config.LOCAL_DEFAULT_MODEL,
            api_key=api_key or config.LOCAL_DUMMY_API_KEY,
            max_tokens=max_tokens,
            client=client,
        )
        self.base_url = str(raw_base).rstrip("/")
        self.timeout = float(timeout or config.LOCAL_TIMEOUT_SECONDS)

    # -- transport ---------------------------------------------------------

    @property
    def client(self) -> Any:
        """An httpx-like client. Tests inject their own (``client=`` kwarg)."""
        if self._client is None:
            if httpx is None:  # pragma: no cover - dependency guard
                raise self._error(ProviderConfigError, "The `httpx` package is not installed.")
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or config.LOCAL_DUMMY_API_KEY}",
        }

    def _unreachable(self, exc: Exception) -> ProviderError:
        return self._error(
            ProviderConnectionError,
            f"Could not reach the local model at {self.base_url} ({exc}). Start the "
            "llama.cpp server, or change the address in Settings → Local model.",
        )

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            if method == "GET":
                return self.client.get(self._url(path), headers=self._headers())
            return self.client.post(self._url(path), json=payload or {}, headers=self._headers())
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - network shapes vary by client
            raise self._unreachable(exc) from exc

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """One OpenAI-compatible ``/chat/completions`` call. Returns the body."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens or self.max_tokens),
            # llama.cpp *does* accept sampling params; determinism matters more
            # to us than variety for both grading and PII detection.
            "temperature": 0,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools

        response = self._request("POST", "/chat/completions", payload)
        status = int(getattr(response, "status_code", 0) or 0)
        if status and status >= 400:
            body = str(getattr(response, "text", ""))[:400]
            raise self._error(
                ProviderResponseError,
                f"The local model server returned HTTP {status}: {body}",
            )
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise self._error(
                ProviderResponseError, "The local model server returned a non-JSON body."
            ) from exc
        if not isinstance(body, dict):
            raise self._error(
                ProviderResponseError, "The local model server returned an unexpected body."
            )
        return body

    @staticmethod
    def _choice(body: dict[str, Any]) -> dict[str, Any]:
        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return {}
        return choices[0]

    @classmethod
    def _message(cls, body: dict[str, Any]) -> dict[str, Any]:
        message = cls._choice(body).get("message")
        return message if isinstance(message, dict) else {}

    @classmethod
    def _text_of(cls, body: dict[str, Any]) -> str:
        content = cls._message(body).get("content")
        if isinstance(content, list):  # some servers return part objects
            return "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        return str(content or "")

    # -- block flattening --------------------------------------------------

    def _blocks_to_text(self, content_blocks: Sequence[dict[str, Any]]) -> str:
        """Flatten Anthropic-shaped blocks into plain text for a local model.

        The server here is text-only, so a PDF has to arrive as its extracted
        text (``_text``, set by the engine's degradation path). Images cannot
        be graded locally at all — say so instead of silently grading nothing.
        """
        parts: list[str] = []
        for block in content_blocks or []:
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text", "")))
            elif btype == "document":
                fallback = block.get("_text")
                if fallback:
                    parts.append(f"[Extracted text of the attached document]\n\n{fallback}")
                else:
                    raise self._error(
                        ProviderUnsupportedError,
                        "The local model cannot read PDFs. Attach the extracted text, or "
                        "grade this submission with a cloud skill.",
                    )
            elif btype == "image":
                raise self._error(
                    ProviderUnsupportedError,
                    "The local model is text-only and cannot read image submissions.",
                )
        return "\n\n".join(p for p in parts if p).strip()

    # -- Provider surface --------------------------------------------------

    def supports_pdf(self) -> bool:
        return False

    def grade(
        self, system_prompt: str, content_blocks: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Structured grading with llama.cpp's ``json_schema`` when available.

        Falls back to strict-JSON prompting plus one retry, because a server
        built without grammar support answers 400 to ``response_format``.
        """
        user_text = self._blocks_to_text(content_blocks) or "(the submission was empty)"
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{LOCAL_JSON_INSTRUCTION}"},
            {"role": "user", "content": user_text},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "agora_grade_result", "strict": True, "schema": schema},
        }

        attempts: list[dict[str, Any] | None] = [response_format, None]
        last_error: Exception | None = None
        for index, fmt in enumerate(attempts):
            try:
                body = self.complete(messages, response_format=fmt)
                raw = self._text_of(body)
                if self._choice(body).get("finish_reason") == "length" and not raw.strip():
                    raise self._error(
                        ProviderResponseError,
                        "The local model hit its token budget before answering.",
                    )
                return parse_json_object(raw)
            except (ProviderResponseError, ValueError) as exc:
                last_error = exc
                if index + 1 < len(attempts):
                    log.warning(
                        "Local model grade attempt %d failed (%s) — retrying with "
                        "strict-JSON prompting",
                        index + 1,
                        exc,
                    )
                    messages = [
                        messages[0],
                        {
                            "role": "user",
                            "content": (
                                f"{user_text}\n\n{LOCAL_JSON_INSTRUCTION}\n"
                                f"The JSON must match this schema: {json.dumps(schema)}"
                            ),
                        },
                    ]
        raise self._error(
            ProviderResponseError,
            f"The local model did not return a usable JSON grade: {last_error}",
        )

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatTurn:
        wire_tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or t.get("parameters") or {},
                    },
                }
                for t in tools
            ]
            if tools
            else None
        )
        body = self.complete(
            self._normalize_messages(system_prompt, messages), tools=wire_tools
        )
        message = self._message(body)
        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=str(call.get("id") or ""),
                    name=str(fn.get("name") or ""),
                    # Arguments arrive as a JSON string — parse, never string-match.
                    arguments=_as_dict(fn.get("arguments")),
                )
            )
        finish = str(self._choice(body).get("finish_reason") or "stop")
        return ChatTurn(
            text=self._text_of(body).strip(),
            tool_calls=tool_calls,
            stop_reason=OpenAIProvider._FINISH_REASONS.get(finish, "end_turn"),
            content=message,
            model=str(body.get("model") or self.model),
            raw=body,
        )

    def _normalize_messages(
        self, system_prompt: str, messages: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Same shapes the other providers accept, flattened to text."""
        out: list[dict[str, Any]] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        for message in messages:
            role = message.get("role", "user")
            if role in ("tool", "tool_results"):
                results = message.get("results")
                if results is None and isinstance(message.get("content"), list):
                    results = message["content"]
                for res in results or []:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": res.get("tool_use_id", ""),
                            "content": str(res.get("content", "")),
                        }
                    )
                continue
            content = message.get("content", "")
            if role == "assistant":
                out.append(OpenAIProvider._assistant_message(content))
                continue
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(block.get("content", "")),
                            }
                        )
                continue
            if isinstance(content, list):
                content = self._blocks_to_text(
                    [b for b in content if isinstance(b, dict)]
                )
            out.append({"role": role, "content": content})
        return out

    # -- health ------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Settings → "local model: connected, gemma-3-4b-it". Never raises."""
        info: dict[str, Any] = {
            "ok": False,
            "provider": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "server_models": [],
            "message": "",
        }
        try:
            response = self._request("GET", "/models")
        except ProviderError as exc:
            info["message"] = str(exc)
            return info
        status = int(getattr(response, "status_code", 0) or 0)
        if status and status >= 400:
            info["message"] = f"The local model server returned HTTP {status}."
            return info
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            body = {}
        served: list[str] = []
        if isinstance(body, dict):
            for entry in body.get("data") or body.get("models") or []:
                if isinstance(entry, dict):
                    name = entry.get("id") or entry.get("name")
                    if name:
                        served.append(str(name))
        info["ok"] = True
        info["server_models"] = served
        # llama.cpp reports the gguf path as the model id — show the file name.
        served_label = served[0].replace("\\", "/").rsplit("/", 1)[-1] if served else self.model
        info["served_model"] = served_label
        info["message"] = f"connected, {served_label}"
        return info


def local_provider(**kwargs: Any) -> LocalProvider:
    """Build a LocalProvider from Settings (one line for every caller)."""
    return LocalProvider(**kwargs)


def local_model_health(**kwargs: Any) -> dict[str, Any]:
    settings = config.local_model_settings()
    info = LocalProvider(**kwargs).health()
    info["enabled"] = bool(settings.get("enabled", True))
    return info


PROVIDER_CLASSES[config.LOCAL_PROVIDER] = LocalProvider

__all__ += [
    "LocalProvider",
    "local_provider",
    "local_model_health",
    "parse_json_object",
    "LOCAL_JSON_INSTRUCTION",
]


# ==========================================================================
# Increment 1 · INSIGHT MODULE (appended — nothing above was touched)
#
# The Student Insight layer asks a provider for JSON in an arbitrary schema
# (the student card). Real providers already do exactly that through
# ``grade(system_prompt, blocks, schema)`` — the schema is an argument, so
# nothing about it is grading-specific. MockProvider is the exception: it
# ignores the schema and always answers with a grade payload.
#
# So the mock gets ONE extra method. It reads the JSON input block the
# consolidator embeds in its prompt (the same "the mock parses the prompt back
# out" contract the rubric line format already uses) and answers with a
# deterministic, schema-valid card, so tests and `--demo` exercise the real
# consolidation path — prompt → provider → parse → validate — offline.
# ==========================================================================


def _mock_card_from_input(data: dict[str, Any]) -> dict[str, Any]:
    """Deterministic student card built from the consolidator's DATA block."""
    student = data.get("student") or {}
    label = str(student.get("label") or "This student")
    graded = int(data.get("graded_assignments") or 0)
    average = data.get("average_percent")
    timeline = [p for p in (data.get("timeline") or []) if isinstance(p, dict)]
    criteria = [c for c in (data.get("criteria") or []) if isinstance(c, dict)]

    strengths: list[dict[str, Any]] = []
    for entry in criteria:
        if not entry.get("high_count") or not entry.get("evidence_high"):
            continue
        strengths.append(
            {
                "text": (
                    f"Holds full marks on {entry.get('title')} in "
                    f"{entry['high_count']} of {graded or entry['high_count']} assignments."
                ),
                "evidence": list(entry["evidence_high"]),
            }
        )
    for entry in data.get("repeated_strengths") or []:
        if not isinstance(entry, dict) or int(entry.get("count") or 0) < 2:
            continue
        strengths.append(
            {
                "text": f"Graders named {entry.get('phrase')} on {entry['count']} assignments.",
                "evidence": list(entry.get("evidence") or []),
            }
        )

    weaknesses: list[dict[str, Any]] = []
    for entry in criteria:
        if not entry.get("low_count") or not entry.get("evidence_low"):
            continue
        weaknesses.append(
            {
                "text": (
                    f"{entry.get('title')} lands at or below half marks on "
                    f"{entry['low_count']} of {graded or entry['low_count']} assignments."
                ),
                "evidence": list(entry["evidence_low"]),
            }
        )

    if average is not None and graded:
        summary = (
            f"{label} averages {average}% over {graded} graded "
            f"{'assignment' if graded == 1 else 'assignments'}"
        )
        if weaknesses:
            summary += f", weakest on {criteria[0].get('title')}"
        summary += "."
    else:
        summary = f"{label} has no graded work on file yet."

    percents = [p.get("percent") for p in timeline if p.get("percent") is not None]
    if len(percents) >= 2:
        delta = round(float(percents[-1]) - float(percents[0]), 1)
        movement = "up" if delta > 0 else "down" if delta < 0 else "level"
        trajectory = (
            f"Scores are {movement} {abs(delta)} points, {percents[0]}% to "
            f"{percents[-1]}%, across {len(percents)} assignments."
        )
    elif percents:
        trajectory = f"One assignment graded so far, at {percents[0]}%."
    else:
        trajectory = "Nothing graded yet, so there is no trend."

    return {
        "summary": summary,
        "trajectory": trajectory,
        "strengths": strengths[:4],
        "weaknesses": weaknesses[:4],
        "misconception_state": [
            {
                "tag": state.get("tag"),
                "status": state.get("status"),
                "evidence": list(state.get("evidence") or []),
            }
            for state in (data.get("misconception_state") or [])
            if isinstance(state, dict)
        ],
    }


def _mock_structured_json(
    self: MockProvider,
    system_prompt: str = "",
    content_blocks: list[dict[str, Any]] | None = None,
    schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Schema-shaped JSON for callers that are not grading.

    Recognised today: the Student Insight card. Anything else falls through to
    the ordinary grade payload, so the method is safe for any caller.
    """
    properties = (schema or {}).get("properties") or {}
    if "misconception_state" in properties:
        try:
            data = parse_json_object(blocks_text(list(content_blocks or [])))
        except ValueError:
            data = {}
        return _mock_card_from_input(data)
    return self.grade(system_prompt, list(content_blocks or []), schema or {}, **kwargs)


#: Bound after the class body rather than edited into it — this increment's
#: module ownership rule for a file another module owns is append, never
#: reorder or rewrite what is already there.
MockProvider.structured_json = _mock_structured_json  # type: ignore[attr-defined]
