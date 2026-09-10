from __future__ import annotations

import os

import pytest

from trading_agent_framework.config.env import (
    TRADING_MODES,
    AlpacaCredentials,
    TradingMode,
    load_strategy_env,
)
from trading_agent_framework.errors import ConfigurationError


def _write_env(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_strategy_file_present_is_used_and_loaded_into_environ(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    strategy_path = tmp_path / "env" / ".env.momentum.paper"
    _write_env(strategy_path, "ALPACA_API_KEY=strategy-key\n")

    # Register a pre-call value with monkeypatch so its teardown restores/removes
    # ALPACA_API_KEY afterward, even though load_dotenv (called inside
    # load_strategy_env) writes directly to the real os.environ and bypasses
    # monkeypatch's own tracking.
    monkeypatch.setenv("ALPACA_API_KEY", "unset")
    result = load_strategy_env("momentum", "paper", project_root=tmp_path)

    assert result == strategy_path
    assert os.environ.get("ALPACA_API_KEY") == "strategy-key"


def test_strategy_file_absent_falls_back_to_env_dot_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    fallback_path = tmp_path / "env" / ".env"
    _write_env(fallback_path, "ALPACA_API_KEY=fallback-key\n")

    # See comment in test_strategy_file_present_is_used_and_loaded_into_environ:
    # this registers a pre-call value so monkeypatch's teardown restores/removes
    # ALPACA_API_KEY after load_dotenv writes to the real os.environ.
    monkeypatch.setenv("ALPACA_API_KEY", "unset")
    result = load_strategy_env("momentum", "paper", project_root=tmp_path)

    assert result == fallback_path
    assert os.environ.get("ALPACA_API_KEY") == "fallback-key"


def test_both_absent_falls_back_to_root_dot_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    root_path = tmp_path / ".env"
    _write_env(root_path, "ALPACA_API_KEY=root-key\n")

    # See comment in test_strategy_file_present_is_used_and_loaded_into_environ:
    # this registers a pre-call value so monkeypatch's teardown restores/removes
    # ALPACA_API_KEY after load_dotenv writes to the real os.environ.
    monkeypatch.setenv("ALPACA_API_KEY", "unset")
    result = load_strategy_env("momentum", "paper", project_root=tmp_path)

    assert result == root_path
    assert os.environ.get("ALPACA_API_KEY") == "root-key"


def test_nothing_present_raises_configuration_error(tmp_path) -> None:
    with pytest.raises(ConfigurationError):
        load_strategy_env("momentum", "paper", project_root=tmp_path)


def test_override_true_replaces_pre_existing_environ_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "pre-existing")
    root_path = tmp_path / ".env"
    _write_env(root_path, "ALPACA_API_KEY=from-file\n")

    load_strategy_env("momentum", "paper", project_root=tmp_path)

    assert os.environ.get("ALPACA_API_KEY") == "from-file"


def test_unknown_trading_mode_raises_configuration_error(tmp_path) -> None:
    root_path = tmp_path / ".env"
    _write_env(root_path, "ALPACA_API_KEY=root-key\n")

    with pytest.raises(ConfigurationError):
        load_strategy_env("momentum", "staging", project_root=tmp_path)


@pytest.mark.parametrize(
    ("raw_value", "expected_is_paper"),
    [
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("true", True),
    ],
)
def test_from_env_is_paper_fail_safe_toward_paper(raw_value, expected_is_paper) -> None:
    env = {
        "ALPACA_API_KEY": "key",
        "ALPACA_API_SECRET": "secret",
        "ALPACA_IS_PAPER": raw_value,
    }

    credentials = AlpacaCredentials.from_env(env)

    assert credentials.is_paper is expected_is_paper


def test_from_env_is_paper_defaults_true_when_absent() -> None:
    env = {"ALPACA_API_KEY": "key", "ALPACA_API_SECRET": "secret"}

    credentials = AlpacaCredentials.from_env(env)

    assert credentials.is_paper is True


@pytest.mark.parametrize("secret_value", [None, "", "   "])
def test_from_env_missing_or_blank_secret_raises_configuration_error(secret_value) -> None:
    env = {"ALPACA_API_KEY": "key"}
    if secret_value is not None:
        env["ALPACA_API_SECRET"] = secret_value

    with pytest.raises(ConfigurationError):
        AlpacaCredentials.from_env(env)


def test_repr_does_not_leak_api_key_or_secret() -> None:
    creds = AlpacaCredentials(api_key="supersecretkey", api_secret="supersecret")

    text = repr(creds)

    assert "supersecretkey" not in text
    assert "supersecret" not in text
    assert "AlpacaCredentials" in text


def test_trading_modes_are_derived_from_the_enum() -> None:
    assert TRADING_MODES == frozenset({"live", "paper", "backtesting"})
    assert {mode.value for mode in TradingMode} == TRADING_MODES


def test_trading_mode_is_a_plain_string() -> None:
    assert TradingMode("paper") is TradingMode.PAPER
    assert f"{TradingMode.LIVE}" == "live"
