from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.errors import ConfigurationError

logger = logging.getLogger(__name__)

TRADING_MODES: frozenset[str] = frozenset({"live", "paper", "backtesting"})

_FALSE_PAPER_VALUES = frozenset({"false", "0", "no"})


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


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    is_paper: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AlpacaCredentials:
        source = env if env is not None else os.environ

        api_key = source.get("ALPACA_API_KEY")
        api_secret = source.get("ALPACA_API_SECRET")

        if not api_key or not api_key.strip():
            raise ConfigurationError(
                "Missing or blank ALPACA_API_KEY / ALPACA_API_SECRET environment variables"
            )
        if not api_secret or not api_secret.strip():
            raise ConfigurationError(
                "Missing or blank ALPACA_API_KEY / ALPACA_API_SECRET environment variables"
            )

        raw_is_paper = source.get("ALPACA_IS_PAPER", "true")
        is_paper = raw_is_paper.strip().lower() not in _FALSE_PAPER_VALUES

        return cls(api_key=api_key, api_secret=api_secret, is_paper=is_paper)
