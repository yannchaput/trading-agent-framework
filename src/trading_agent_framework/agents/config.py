"""Env-driven LLM connection credentials for the agent framework (pure, no I/O)."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from trading_agent_framework.utils.errors import ConfigurationError

# The OpenAI SDK needs a non-blank key to build a client; local servers (vLLM, llama.cpp) ignore it.
PLACEHOLDER_API_KEY = "not-needed"

_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


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
    base_url: str
    api_key: str = field(repr=False)
    default_model: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMCredentials:
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

        return cls(base_url=normalize_base_url(base_url), api_key=api_key, default_model=default_model)
