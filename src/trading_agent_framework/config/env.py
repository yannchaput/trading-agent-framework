from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.utils.errors import ConfigurationError

logger = logging.getLogger(__name__)


class TradingMode(StrEnum):
    """The three ways a strategy can run; values match the env-file suffixes."""

    LIVE = "live"
    PAPER = "paper"
    BACKTESTING = "backtesting"


TRADING_MODES: frozenset[str] = frozenset(mode.value for mode in TradingMode)

_FALSE_PAPER_VALUES = frozenset({"false", "0", "no"})

# Old name -> new name. Still present = a stale env file: fail loudly, no alias.
_RENAMED_VARIABLES = {"ALPACA_IS_PAPER": "BROKER_API_IS_PAPER"}

IBKR_PAPER_PORT = 4002  # IB Gateway's default paper-trading API port
IBKR_LIVE_PORT = 4001  # IB Gateway's default live-trading API port


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default cwd) to the first directory containing pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    searched = [current]
    while True:
        if (current / "pyproject.toml").is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
        searched.append(current)
    searched_display = ", ".join(str(path) for path in searched)
    raise ConfigurationError(
        f"Could not find project root (no pyproject.toml found in: {searched_display})"
    )


def resolve_env_file(
    strategy_name: str,
    trading_mode: str,
    project_root: Path | None = None,
) -> Path:
    """Resolve the env file to load, in priority order.

    Priority: env/.env.{strategy}.{mode} -> env/.env -> <root>/.env, first that exists.
    """
    if trading_mode not in TRADING_MODES:
        raise ConfigurationError(
            f"Unknown trading_mode {trading_mode!r}; expected one of {sorted(TRADING_MODES)}"
        )

    root = project_root if project_root is not None else find_project_root()

    candidates = [
        root / "env" / f".env.{strategy_name}.{trading_mode}",
        root / "env" / ".env",
        root / ".env",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    candidates_display = ", ".join(str(path) for path in candidates)
    raise ConfigurationError(f"No env file found; searched: {candidates_display}")


def load_strategy_env(
    strategy_name: str,
    trading_mode: str = "paper",
    project_root: Path | None = None,
) -> Path:
    """Resolve and load the env file for a strategy, returning the resolved path."""
    root = project_root if project_root is not None else find_project_root()
    env_path = resolve_env_file(strategy_name, trading_mode, project_root=root)
    load_dotenv(env_path, override=True)
    logger.info("Loaded environment from %s", env_path)
    return env_path


def _reject_renamed(source: Mapping[str, str]) -> None:
    for old, new in _RENAMED_VARIABLES.items():
        if old in source:
            raise ConfigurationError(f"{old} was renamed to {new}; rename it in your env file")


def _is_paper(raw: str | None) -> bool:
    """Fail-safe toward paper: only an explicit false/0/no means live."""
    return (raw or "true").strip().lower() not in _FALSE_PAPER_VALUES


def _int_setting(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from None


class BrokerKind(StrEnum):
    """Which broker trades in paper/live mode (`BROKER`); meaningless in backtesting."""

    ALPACA = "alpaca"
    IBKR = "ibkr"


@dataclass(frozen=True, slots=True)
class BrokerSettings:
    kind: BrokerKind
    is_paper: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> BrokerSettings:
        source = env if env is not None else os.environ
        _reject_renamed(source)
        raw_kind = (source.get("BROKER") or BrokerKind.ALPACA.value).strip().lower()
        try:
            kind = BrokerKind(raw_kind)
        except ValueError:
            valid = ", ".join(k.value for k in BrokerKind)
            raise ConfigurationError(f"Unknown BROKER {raw_kind!r}; expected one of: {valid}") from None
        return cls(kind=kind, is_paper=_is_paper(source.get("BROKER_API_IS_PAPER")))


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    """One Alpaca key pair. Each component reads its OWN group -- trading, news or data --
    and groups never fall back to each other."""

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    is_paper: bool = True

    @classmethod
    def for_trading(cls, env: Mapping[str, str] | None = None) -> AlpacaCredentials:
        """`ALPACA_API_KEY`/`ALPACA_API_SECRET`: the Alpaca trading broker (`BROKER=alpaca`)."""
        return cls._read(env, "ALPACA_API", paper_variable="BROKER_API_IS_PAPER")

    @classmethod
    def for_news(cls, env: Mapping[str, str] | None = None) -> AlpacaCredentials:
        """`ALPACA_NEWS_API_*`: the news tool, in every mode and for every broker."""
        return cls._read(env, "ALPACA_NEWS_API", paper_variable=None)

    @classmethod
    def for_data(cls, env: Mapping[str, str] | None = None) -> AlpacaCredentials:
        """`ALPACA_DATA_API_*`: prices, bars and the calendar (paper/live, both brokers) and
        `AlpacaBacktestData`. `ALPACA_DATA_IS_PAPER` picks the endpoint the calendar call uses."""
        return cls._read(env, "ALPACA_DATA_API", paper_variable="ALPACA_DATA_IS_PAPER")

    @classmethod
    def _read(cls, env: Mapping[str, str] | None, prefix: str, *, paper_variable: str | None) -> AlpacaCredentials:
        source = env if env is not None else os.environ
        _reject_renamed(source)
        key_name, secret_name = f"{prefix}_KEY", f"{prefix}_SECRET"
        api_key = source.get(key_name) or ""
        api_secret = source.get(secret_name) or ""
        if not api_key.strip() or not api_secret.strip():
            raise ConfigurationError(f"Missing or blank {key_name} / {secret_name} environment variables")
        is_paper = True if paper_variable is None else _is_paper(source.get(paper_variable))
        return cls(api_key=api_key, api_secret=api_secret, is_paper=is_paper)


@dataclass(frozen=True, slots=True)
class IbkrSettings:
    """Where IB Gateway listens. No key/secret: the Gateway holds the IBKR login."""

    host: str
    port: int
    client_id: int
    is_paper: bool

    @classmethod
    def from_env(cls, is_paper: bool, env: Mapping[str, str] | None = None) -> IbkrSettings:
        source = env if env is not None else os.environ
        _reject_renamed(source)
        host = (source.get("IBKR_HOST") or "127.0.0.1").strip()
        port = _int_setting(source, "IBKR_PORT", IBKR_PAPER_PORT if is_paper else IBKR_LIVE_PORT)
        client_id = _int_setting(source, "IBKR_CLIENT_ID", 1)
        return cls(host=host, port=port, client_id=client_id, is_paper=is_paper)


@dataclass(frozen=True, slots=True)
class FredCredentials:
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FredCredentials:
        source = env if env is not None else os.environ
        api_key = source.get("FRED_API_KEY")
        if not api_key or not api_key.strip():
            raise ConfigurationError("Missing or blank FRED_API_KEY environment variable")
        return cls(api_key=api_key)
