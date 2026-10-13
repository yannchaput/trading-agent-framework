from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from trading_agent_framework.dashboard import account_reader
from trading_agent_framework.utils.errors import ConfigurationError

KEYS = "ALPACA_API_KEY=trade-key\nALPACA_API_SECRET=trade-secret\nALPACA_DATA_API_KEY=data-key\nALPACA_DATA_API_SECRET=data-secret\n"


def _env(env_dir: Path, name: str, body: str) -> None:
    env_dir.mkdir(exist_ok=True)
    (env_dir / name).write_text(body, encoding="utf-8")


def test_strategies_are_the_alpaca_env_files_of_the_mode(tmp_path: Path) -> None:
    _env(tmp_path, ".env.alpha.paper", "BROKER=alpaca\n" + KEYS)
    _env(tmp_path, ".env.beta.paper", KEYS)  # no BROKER: alpaca
    _env(tmp_path, ".env.gamma.paper", "BROKER=ibkr\n")
    _env(tmp_path, ".env.alpha.live", "BROKER=alpaca\n" + KEYS)
    _env(tmp_path, ".env.alpha.backtesting", "BROKER=alpaca\n" + KEYS)
    _env(tmp_path, ".env.example", KEYS)

    assert account_reader.strategies_for_mode("paper", tmp_path) == ["alpha", "beta"]
    assert account_reader.strategies_for_mode("live", tmp_path) == ["alpha"]


def test_no_env_dir_lists_no_strategy(tmp_path: Path) -> None:
    assert account_reader.strategies_for_mode("paper", tmp_path / "missing") == []


def test_credentials_come_from_the_file_without_touching_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_DATA_API_KEY", "ALPACA_DATA_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    _env(tmp_path, ".env.alpha.live", "BROKER=alpaca\nBROKER_API_IS_PAPER=false\n" + KEYS)

    trading, data = account_reader.load_account_credentials("alpha", "live", tmp_path)

    assert (trading.api_key, trading.api_secret, trading.is_paper) == ("trade-key", "trade-secret", False)
    assert (data.api_key, data.api_secret) == ("data-key", "data-secret")
    assert "ALPACA_API_KEY" not in os.environ


def test_blank_keys_raise_a_configuration_error_without_the_values(tmp_path: Path) -> None:
    _env(tmp_path, ".env.alpha.paper", "ALPACA_API_KEY=\nALPACA_API_SECRET=s3cret\n")

    with pytest.raises(ConfigurationError, match="ALPACA_API_KEY") as excinfo:
        account_reader.load_account_credentials("alpha", "paper", tmp_path)

    assert "s3cret" not in str(excinfo.value)


def test_a_missing_env_file_raises_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"\.env\.alpha\.paper"):
        account_reader.load_account_credentials("alpha", "paper", tmp_path)


def test_load_strategy_pnl_builds_the_clients_from_the_strategys_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, ".env.alpha.paper", "BROKER_API_IS_PAPER=true\n" + KEYS)
    built: dict[str, object] = {}
    monkeypatch.setattr(account_reader, "build_trading_client", lambda creds: built.setdefault("trading", creds))
    monkeypatch.setattr(account_reader, "build_stock_data_client", lambda creds: built.setdefault("data", creds))
    monkeypatch.setattr(account_reader, "load_account_pnl", lambda trading, data, today: (trading, data, today))

    result = account_reader.load_strategy_pnl("alpha", "paper", tmp_path, date(2026, 10, 13))

    trading, data, today = result  # ty: ignore[not-iterable]
    assert trading is built["trading"] and data is built["data"]
    assert trading.api_key == "trade-key" and trading.is_paper is True
    assert data.api_key == "data-key"
    assert today == date(2026, 10, 13)
