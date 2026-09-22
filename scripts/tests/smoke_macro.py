#!/usr/bin/env python3
"""Manual check of the FRED macro tool against the live FRED API.

NOT part of the automated test suite. Run it by hand:

    uv run python scripts/tests/smoke_macro.py

Needs FRED_API_KEY in env/.env.alpaca.integration-tests (or another loaded env file) and a
paper-trading Alpaca broker (only used to build a Strategy/clock; no orders are placed).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.core import Strategy
from trading_agent_framework.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-macro"


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(f"Credentials file not found: {ENV_FILE}")
    load_dotenv(ENV_FILE, override=True)
    try:
        return AlpacaCredentials.for_trading()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    creds = _load_credentials()
    broker = AlpacaBroker.from_credentials(
        STRATEGY_NAME, trading=creds, data=AlpacaCredentials.for_data(), news=AlpacaCredentials.for_news, with_stream=False
    )
    strategy = Strategy(broker)
    [get_fred_series] = macro_tools(strategy)

    result = get_fred_series("M2SL", limit=5)
    print(f"get_fred_series(M2SL) -> {result}")
    _check("error" not in result, f"get_fred_series returned an error: {result.get('error')}")
    _check(len(result["observations"]) > 0, "no observations returned")

    print("\nPASS: get_fred_series returned observations.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
