from __future__ import annotations

import os

import pytest

from trading_agent_framework.config.env import (
    IBKR_LIVE_PORT,
    IBKR_PAPER_PORT,
    TRADING_MODES,
    AlpacaCredentials,
    BrokerKind,
    BrokerSettings,
    FredCredentials,
    IbkrSettings,
    TradingMode,
    load_strategy_env,
)
from trading_agent_framework.utils.errors import ConfigurationError


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


def test_fred_credentials_from_env() -> None:
    creds = FredCredentials.from_env({"FRED_API_KEY": "abc123"})
    assert creds.api_key == "abc123"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_fred_credentials_missing_or_blank_key_raises(value) -> None:
    env = {} if value is None else {"FRED_API_KEY": value}
    with pytest.raises(ConfigurationError, match="FRED_API_KEY"):
        FredCredentials.from_env(env)


def test_fred_credentials_repr_does_not_leak_the_key() -> None:
    assert "supersecret" not in repr(FredCredentials(api_key="supersecret"))


# --- broker selection ---------------------------------------------------------------


def test_broker_settings_default_to_alpaca_paper() -> None:
    assert BrokerSettings.from_env({}) == BrokerSettings(kind=BrokerKind.ALPACA, is_paper=True)


@pytest.mark.parametrize("raw", ["ibkr", "IBKR", " ibkr "])
def test_broker_settings_read_broker_case_insensitively(raw: str) -> None:
    assert BrokerSettings.from_env({"BROKER": raw}).kind is BrokerKind.IBKR


def test_broker_settings_reject_an_unknown_broker() -> None:
    with pytest.raises(ConfigurationError, match="Unknown BROKER 'schwab'"):
        BrokerSettings.from_env({"BROKER": "schwab"})


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("0", False), ("no", False), ("true", True), ("anything", True)])
def test_broker_api_is_paper_is_fail_safe_toward_paper(raw: str, expected: bool) -> None:
    assert BrokerSettings.from_env({"BROKER_API_IS_PAPER": raw}).is_paper is expected


@pytest.mark.parametrize(
    "read",
    [
        BrokerSettings.from_env,
        AlpacaCredentials.for_trading,
        AlpacaCredentials.for_news,
        AlpacaCredentials.for_data,
        lambda env: IbkrSettings.from_env(True, env),
    ],
)
def test_the_renamed_alpaca_is_paper_variable_is_rejected_everywhere(read) -> None:
    env = {"ALPACA_IS_PAPER": "true", "ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"}

    with pytest.raises(ConfigurationError, match="ALPACA_IS_PAPER was renamed to BROKER_API_IS_PAPER"):
        read(env)


# --- Alpaca credential groups --------------------------------------------------------


def test_trading_credentials_read_alpaca_api_and_the_broker_paper_flag() -> None:
    env = {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s", "BROKER_API_IS_PAPER": "false"}

    creds = AlpacaCredentials.for_trading(env)

    assert (creds.api_key, creds.api_secret, creds.is_paper) == ("k", "s", False)


def test_news_credentials_read_only_the_news_pair() -> None:
    env = {"ALPACA_NEWS_API_KEY": "nk", "ALPACA_NEWS_API_SECRET": "ns", "ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"}

    creds = AlpacaCredentials.for_news(env)

    assert (creds.api_key, creds.api_secret, creds.is_paper) == ("nk", "ns", True)


def test_data_credentials_read_the_data_pair_and_their_own_paper_flag() -> None:
    env = {"ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds", "ALPACA_DATA_IS_PAPER": "no", "BROKER_API_IS_PAPER": "true"}

    creds = AlpacaCredentials.for_data(env)

    assert (creds.api_key, creds.api_secret, creds.is_paper) == ("dk", "ds", False)


def test_data_credentials_default_to_paper() -> None:
    assert AlpacaCredentials.for_data({"ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds"}).is_paper is True


@pytest.mark.parametrize(
    ("read", "missing"),
    [
        (AlpacaCredentials.for_trading, "ALPACA_API_KEY / ALPACA_API_SECRET"),
        (AlpacaCredentials.for_news, "ALPACA_NEWS_API_KEY / ALPACA_NEWS_API_SECRET"),
        (AlpacaCredentials.for_data, "ALPACA_DATA_API_KEY / ALPACA_DATA_API_SECRET"),
    ],
)
def test_each_group_names_its_own_missing_variables(read, missing: str) -> None:
    with pytest.raises(ConfigurationError, match=missing):
        read({})


def test_groups_never_fall_back_to_the_trading_pair() -> None:
    with pytest.raises(ConfigurationError, match="ALPACA_NEWS_API_KEY"):
        AlpacaCredentials.for_news({"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"})


def test_a_blank_secret_is_missing() -> None:
    with pytest.raises(ConfigurationError, match="ALPACA_API_SECRET"):
        AlpacaCredentials.for_trading({"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "   "})


# --- IBKR settings -------------------------------------------------------------------


def test_ibkr_settings_defaults_depend_on_the_paper_flag() -> None:
    assert IbkrSettings.from_env(True, {}) == IbkrSettings(host="127.0.0.1", port=IBKR_PAPER_PORT, client_id=1, is_paper=True)
    assert IbkrSettings.from_env(False, {}).port == IBKR_LIVE_PORT


def test_ibkr_settings_read_host_port_and_client_id() -> None:
    env = {"IBKR_HOST": "10.0.0.5", "IBKR_PORT": "7497", "IBKR_CLIENT_ID": "7"}

    assert IbkrSettings.from_env(True, env) == IbkrSettings(host="10.0.0.5", port=7497, client_id=7, is_paper=True)


@pytest.mark.parametrize("variable", ["IBKR_PORT", "IBKR_CLIENT_ID"])
def test_ibkr_settings_reject_non_integer_numbers(variable: str) -> None:
    with pytest.raises(ConfigurationError, match=f"{variable} must be an integer"):
        IbkrSettings.from_env(True, {variable: "abc"})
