"""Env-driven LLM connection credentials for the agent framework (pure, no I/O)."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from trading_agent_framework.utils.errors import ConfigurationError

# The OpenAI SDK needs a non-blank key to build a client; local servers (vLLM, llama.cpp) ignore it.
PLACEHOLDER_API_KEY = "not-needed"

_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


class ReasoningEffort(StrEnum):
    """Reasoning effort sent to the model with every agent call (`xhigh` is the most thinking)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


def normalize_base_url(url: str) -> str:
    """Reduce a pasted OpenAI-compatible endpoint to the base URL `ChatOpenAI` expects (e.g. `http://host:8005/v1`)."""
    cleaned = url.strip()
    scheme, separator, rest = cleaned.partition("://")
    cleaned = f"{scheme}{separator}{re.sub(r'/{2,}', '/', rest)}".rstrip("/")
    if cleaned.endswith(_CHAT_COMPLETIONS_SUFFIX):
        cleaned = cleaned[: -len(_CHAT_COMPLETIONS_SUFFIX)]
    return cleaned.rstrip("/")


@dataclass(frozen=True, slots=True)
class LLMCredentials:
    """Connection settings for the OpenAI-compatible LLM server.

    Attributes:
        base_url: Server URL, normalised to end in `/v1`.
        api_key: API key; a placeholder when the server needs none. Never shown in `repr`.
        default_model: Model to use when the strategy does not name one.
        max_input_tokens: The server's context window (vLLM `max_model_len`), for history compaction; None when unknown.
        reasoning_effort: Sent as `reasoning_effort` with every model call; None sends nothing (the server default applies).
    """

    base_url: str
    api_key: str = field(repr=False)
    default_model: str | None = None
    # The server's context window (vLLM `max_model_len`), for history compaction; `None` when unknown.
    max_input_tokens: int | None = None
    # Sent as `reasoning_effort` with every agent model call; `None` sends nothing (the server default applies).
    reasoning_effort: ReasoningEffort | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMCredentials:
        """Build the credentials from environment variables.

        Reads `LLM_BASE_URL` (required), `LLM_API_KEY`, `LLM_MODEL`, and the optional context-window and reasoning-effort variables.

        Args:
            env: Mapping to read instead of `os.environ` (tests).

        Returns:
            The credentials, with a blank `LLM_MODEL` read as unset and a missing key replaced by a placeholder.

        Raises:
            ConfigurationError: If `LLM_BASE_URL` is missing or blank.
        """
        source = env if env is not None else os.environ

        base_url = source.get("LLM_BASE_URL")
        if not base_url or not base_url.strip():
            raise ConfigurationError("Missing or blank LLM_BASE_URL environment variable")

        api_key = source.get("LLM_API_KEY")
        if not api_key or not api_key.strip():
            api_key = PLACEHOLDER_API_KEY

        default_model = source.get("LLM_MODEL")
        if default_model is not None and not default_model.strip():
            default_model = None

        return cls(base_url=normalize_base_url(base_url), api_key=api_key, default_model=default_model, max_input_tokens=_max_input_tokens(source), reasoning_effort=_reasoning_effort(source))


def _max_input_tokens(source: Mapping[str, str]) -> int | None:
    """`LLM_MAX_INPUT_TOKENS` as a positive integer; blank or missing is `None`."""
    raw = (source.get("LLM_MAX_INPUT_TOKENS") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ConfigurationError(f"LLM_MAX_INPUT_TOKENS must be a positive integer (the server's context window), got {raw!r}")
    return value


def _reasoning_effort(source: Mapping[str, str]) -> ReasoningEffort | None:
    """`LLM_REASONING_EFFORT` as a `ReasoningEffort` (case-insensitive); blank or missing is `None`."""
    raw = (source.get("LLM_REASONING_EFFORT") or "").strip()
    if not raw:
        return None
    try:
        return ReasoningEffort(raw.lower())
    except ValueError:
        allowed = ", ".join(effort.value for effort in ReasoningEffort)
        raise ConfigurationError(f"LLM_REASONING_EFFORT must be one of {allowed}, got {raw!r}") from None
