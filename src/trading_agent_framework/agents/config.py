"""Env-driven LLM connection credentials for the agent framework (pure, no I/O)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from trading_agent_framework.utils.errors import ConfigurationError


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
            raise ConfigurationError("Missing or blank LLM_API_KEY environment variable")

        default_model = source.get("LLM_MODEL")
        if default_model is not None and not default_model.strip():
            default_model = None

        return cls(base_url=base_url, api_key=api_key, default_model=default_model)
