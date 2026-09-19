from __future__ import annotations

import pytest

from trading_agent_framework.agents.config import PLACEHOLDER_API_KEY, LLMCredentials, normalize_base_url
from trading_agent_framework.utils.errors import ConfigurationError


def test_from_env_reads_base_url_api_key_and_default_model() -> None:
    env = {
        "LLM_BASE_URL": "http://localhost:8000/v1",
        "LLM_API_KEY": "key",
        "LLM_MODEL": "qwen3-8b",
    }

    credentials = LLMCredentials.from_env(env)

    assert credentials.base_url == "http://localhost:8000/v1"
    assert credentials.api_key == "key"
    assert credentials.default_model == "qwen3-8b"


def test_from_env_default_model_is_optional() -> None:
    env = {"LLM_BASE_URL": "http://localhost:8000/v1", "LLM_API_KEY": "key"}

    credentials = LLMCredentials.from_env(env)

    assert credentials.default_model is None


@pytest.mark.parametrize("blank_model", ["", "   "])
def test_from_env_blank_default_model_normalizes_to_none(blank_model: str) -> None:
    env = {"LLM_BASE_URL": "http://localhost:8000/v1", "LLM_API_KEY": "key", "LLM_MODEL": blank_model}

    credentials = LLMCredentials.from_env(env)

    assert credentials.default_model is None


@pytest.mark.parametrize("base_url_value", [None, "", "   "])
def test_from_env_missing_or_blank_base_url_raises(base_url_value: str | None) -> None:
    env = {"LLM_API_KEY": "key"}
    if base_url_value is not None:
        env["LLM_BASE_URL"] = base_url_value

    with pytest.raises(ConfigurationError):
        LLMCredentials.from_env(env)


@pytest.mark.parametrize("api_key_value", [None, "", "   "])
def test_from_env_missing_or_blank_api_key_falls_back_to_the_placeholder(api_key_value: str | None) -> None:
    env = {"LLM_BASE_URL": "http://localhost:8000/v1"}
    if api_key_value is not None:
        env["LLM_API_KEY"] = api_key_value

    credentials = LLMCredentials.from_env(env)

    assert credentials.api_key == PLACEHOLDER_API_KEY


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:8005//v1/chat/completions", "http://localhost:8005/v1"),
        ("http://localhost:8005/v1/chat/completions", "http://localhost:8005/v1"),
        ("http://localhost:8005/v1/chat/completions/", "http://localhost:8005/v1"),
        ("http://localhost:8005/v1/", "http://localhost:8005/v1"),
        ("http://localhost:8005//v1", "http://localhost:8005/v1"),
        ("  http://localhost:8005/v1  ", "http://localhost:8005/v1"),
        ("https://api.example.com/v1", "https://api.example.com/v1"),
    ],
)
def test_normalize_base_url(raw: str, expected: str) -> None:
    assert normalize_base_url(raw) == expected


def test_from_env_normalizes_the_base_url() -> None:
    env = {"LLM_BASE_URL": "http://localhost:8005//v1/chat/completions"}

    assert LLMCredentials.from_env(env).base_url == "http://localhost:8005/v1"


def test_from_env_defaults_to_os_environ_when_no_mapping_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    credentials = LLMCredentials.from_env()

    assert credentials.base_url == "http://localhost:9000/v1"
    assert credentials.api_key == "env-key"


def test_repr_does_not_leak_the_api_key() -> None:
    credentials = LLMCredentials(base_url="http://localhost:8000/v1", api_key="supersecretkey")

    text = repr(credentials)

    assert "supersecretkey" not in text
    assert "LLMCredentials" in text
