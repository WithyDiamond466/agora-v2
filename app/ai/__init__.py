"""Agora AI engine: provider abstraction + the grading pipeline.

``app.ai.assistant`` (chat tool-calling) is owned by the chat module and is
deliberately **not** imported here, so importing the engine never drags in a
module that may still be mid-build.
"""

from __future__ import annotations

from app.ai.grading import (
    GRADE_SCHEMA,
    GradeRequest,
    ValidatedGrade,
    build_grade_request,
    grade_submission,
    persist_grade_result,
    render_rubric_text,
    validate_grade_payload,
)
from app.ai.providers import (
    AnthropicProvider,
    ChatTurn,
    MockProvider,
    OpenAIProvider,
    Provider,
    ProviderAuthError,
    ProviderConfigError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderResponseError,
    ProviderUnsupportedError,
    ToolCall,
    get_provider,
    tool_result_message,
)

__all__ = [
    "GRADE_SCHEMA",
    "GradeRequest",
    "ValidatedGrade",
    "build_grade_request",
    "grade_submission",
    "persist_grade_result",
    "render_rubric_text",
    "validate_grade_payload",
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "MockProvider",
    "ChatTurn",
    "ToolCall",
    "get_provider",
    "tool_result_message",
    "ProviderError",
    "ProviderConfigError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderConnectionError",
    "ProviderResponseError",
    "ProviderRefusalError",
    "ProviderUnsupportedError",
]
