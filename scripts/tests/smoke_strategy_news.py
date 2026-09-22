#!/usr/bin/env python3
"""Manual end-to-end check of the news_builtin strategy against a real LLM server and Alpaca news.

NOT part of the automated test suite. Run it by hand:

    uv run python scripts/tests/smoke_strategy_news.py iteration   # one agent iteration (paper account only)
    uv run python scripts/tests/smoke_strategy_news.py backtest    # a 2-week real backtest

LLM_BASE_URL / LLM_MODEL come from env/.env.news_builtin.backtesting. `iteration` takes its Alpaca
credentials from env/.env.alpaca.integration-tests (paper) and refuses to run against a live account,
since the agent may place paper orders. `backtest` only reads news and market data.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config import load_strategy_env
from trading_agent_framework.config.env import AlpacaCredentials, TradingMode
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy
from trading_agent_framework.utils.clock import MARKET_TZ

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"
STRATEGY_NAME = "news_builtin"


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _iteration() -> None:
    if not PAPER_ENV_FILE.is_file():
        raise SmokeTestFailure(f"Paper credentials file not found: {PAPER_ENV_FILE}")
    load_dotenv(PAPER_ENV_FILE, override=True)
    # LLM settings only: override=False keeps the paper Alpaca credentials loaded above.
    load_dotenv(PROJECT_ROOT / "env" / ".env.news_builtin.backtesting", override=False)
    creds = AlpacaCredentials.for_trading()
    if not creds.is_paper:
        raise SmokeTestFailure("Refusing to run one agent iteration against a live account.")
    broker = AlpacaBroker.from_credentials(
        STRATEGY_NAME, trading=creds, data=AlpacaCredentials.for_data(), news=AlpacaCredentials.for_news, with_stream=False
    )
    strategy = NewsBinaryStrategy(broker, mode=TradingMode.PAPER)
    strategy.initialize()
    strategy.on_trading_iteration()
    print("PASS: one agent iteration completed (see the log line above for the agent's answer).")


def _backtest() -> None:
    load_strategy_env(STRATEGY_NAME, TradingMode.BACKTESTING.value, PROJECT_ROOT)
    creds = AlpacaCredentials.for_trading()
    broker = AlpacaBroker.from_credentials(
        STRATEGY_NAME, trading=creds, data=AlpacaCredentials.for_data(), news=AlpacaCredentials.for_news, with_stream=False
    )
    strategy = NewsBinaryStrategy(broker, mode=TradingMode.BACKTESTING)
    strategy.backtesting_start = datetime(2025, 3, 3, tzinfo=MARKET_TZ)
    strategy.backtesting_end = datetime(2025, 3, 14, tzinfo=MARKET_TZ)
    result = strategy.run_backtesting()
    for name in ("metrics.json", "settings.json", "equity.parquet"):
        if not (result.run_dir / name).is_file():
            raise SmokeTestFailure(f"missing {name} in {result.run_dir}")
    print(f"PASS: backtest report written to {result.run_dir}")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"iteration", "backtest"}:
        print(__doc__)
        return 2
    {"iteration": _iteration, "backtest": _backtest}[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
