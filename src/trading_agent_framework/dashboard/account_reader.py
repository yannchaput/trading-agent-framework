"""A strategy's Alpaca account for the Live and Paper tabs: env-file discovery, credentials and read-only fetches.

Streamlit-free. Keys are read with `dotenv_values` -- never `load_dotenv` -- so one strategy's keys never reach
`os.environ` and switching strategies cannot mix accounts.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from trading_agent_framework.brokers.alpaca.client import build_stock_data_client, build_trading_client
from trading_agent_framework.config.env import AlpacaCredentials, find_project_root
from trading_agent_framework.dashboard.pnl import AccountPnl
from trading_agent_framework.utils.errors import ConfigurationError


def default_env_dir() -> Path:
    return find_project_root() / "env"


def _env_values(path: Path) -> dict[str, str]:
    return {key: value or "" for key, value in dotenv_values(path).items()}


def strategies_for_mode(mode: str, env_dir: Path) -> list[str]:
    """The strategies with an `env_dir/.env.<strategy>.<mode>` file trading on Alpaca (no `BROKER` means alpaca)."""
    if not env_dir.is_dir():
        return []
    prefix, suffix = ".env.", f".{mode}"
    names = []
    for path in env_dir.glob(f"{prefix}*{suffix}"):
        broker = (_env_values(path).get("BROKER") or "alpaca").strip().lower()
        if broker == "alpaca":
            names.append(path.name[len(prefix) : -len(suffix)])
    return sorted(names)


def load_account_credentials(strategy: str, mode: str, env_dir: Path) -> tuple[AlpacaCredentials, AlpacaCredentials]:
    """The trading and the market-data keys of `env_dir/.env.<strategy>.<mode>`."""
    path = env_dir / f".env.{strategy}.{mode}"
    if not path.is_file():
        raise ConfigurationError(f"{path} not found")
    values = _env_values(path)
    return AlpacaCredentials.for_trading(env=values), AlpacaCredentials.for_data(env=values)


def load_account_pnl(trading: Any, data: Any, today: date) -> AccountPnl:
    raise NotImplementedError  # Task 4


def load_strategy_pnl(strategy: str, mode: str, env_dir: Path, today: date) -> AccountPnl:
    """The daily P&L of the strategy's Alpaca account, fetched now."""
    trading_creds, data_creds = load_account_credentials(strategy, mode, env_dir)
    return load_account_pnl(build_trading_client(trading_creds), build_stock_data_client(data_creds), today)
