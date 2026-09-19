#!/usr/bin/env python3
"""Manual check of the Alpaca news tool against Alpaca's live news feed.

NOT part of the automated test suite. Run it by hand:

    uv run python scripts/tests/smoke_news.py

Credentials come from env/.env.alpaca.integration-tests, as in smoke_alpaca_data.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.core import Strategy
from trading_agent_framework.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-news"


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(f"Credentials file not found: {ENV_FILE}")
    load_dotenv(ENV_FILE, override=True)
    try:
        return AlpacaCredentials.from_env()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    creds = _load_credentials()
    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=False)
    strategy = Strategy(broker)
    [search_news] = news_tools(strategy)

    result = search_news(symbols="SPY,QQQ", limit=5)
    print(f"search_news(SPY,QQQ) -> count={result.get('count')}")
    _check("error" not in result, f"search_news returned an error: {result.get('error')}")
    _check(result["count"] >= 0, "search_news returned a negative count")
    if result["articles"]:
        print(f"first headline: {result['articles'][0]['headline']}")

    print("\nPASS: search_news returned without error.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
