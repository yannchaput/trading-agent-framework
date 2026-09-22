#!/usr/bin/env python3
"""Manual paper-account check of the market data layer against Alpaca's live IEX feed.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_alpaca_data.py

Credentials come from env/.env.alpaca.integration-tests, as in smoke_alpaca_orders.py. The
script is read-only (it places no orders) and works at any time of day. It fails only on things
that must always hold: SPY has a last trade, daily bars come back full-length, regular-hours
minute bars stay inside 09:30-16:00, and the indicators return values. It also prints two things
no fake can show: what Alpaca does with an unknown symbol, and how many minute bars the
include_after_hours=False filter keeps.
"""

from __future__ import annotations

import sys
from datetime import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.core import Strategy
from trading_agent_framework.utils.errors import ConfigurationError, TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-data"
SYMBOLS = ["SPY", "QQQ", "AAPL"]
UNKNOWN_SYMBOL = "ZZZZZZ"
DAY_BARS = 30
MINUTE_BARS = 60


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(
            f"Credentials file not found: {ENV_FILE}\n"
            "Create it with ALPACA_API_KEY / ALPACA_API_SECRET, ALPACA_DATA_API_KEY / "
            "ALPACA_DATA_API_SECRET, ALPACA_NEWS_API_KEY / ALPACA_NEWS_API_SECRET and "
            "BROKER_API_IS_PAPER=true before running this script."
        )
    load_dotenv(ENV_FILE, override=True)
    try:
        creds = AlpacaCredentials.for_trading()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc
    if not creds.is_paper:
        raise SmokeTestFailure("BROKER_API_IS_PAPER is not true in the credentials file; refusing to run.")
    return creds


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    creds = _load_credentials()
    broker = AlpacaBroker.from_credentials(
        STRATEGY_NAME, trading=creds, data=AlpacaCredentials.for_data(), news=AlpacaCredentials.for_news, with_stream=False
    )
    strategy = Strategy(broker)

    price = strategy.get_last_price("SPY")
    print(f"get_last_price(SPY) = {price}")
    _check(isinstance(price, Decimal) and price > 0, "no last trade price for SPY")

    prices = strategy.get_last_prices(SYMBOLS)
    print(f"get_last_prices({SYMBOLS}) = {prices}")
    _check(all(value is not None for value in prices.values()), f"missing last prices: {prices}")

    try:
        mixed = strategy.get_last_prices([UNKNOWN_SYMBOL, "SPY"])
        print(f"unknown symbol in a batch -> {mixed}")
    except TradingFrameworkError as exc:
        print(f"unknown symbol in a batch -> raised {type(exc).__name__}: {exc}")

    quote = strategy.get_quote("SPY")
    mid = quote.mid if quote is not None else None
    print(f"get_quote(SPY) = {quote} (mid={mid}); an empty side is normal outside market hours")

    day = strategy.get_historical_prices("SPY", DAY_BARS, "day")
    _check(day is not None and len(day.df) == DAY_BARS, f"expected {DAY_BARS} daily bars for SPY")
    assert day is not None
    print(f"daily bars: {day.df.index[0]} .. {day.df.index[-1]}\n{day.df.tail(3)}")

    day_regular = strategy.get_historical_prices("SPY", DAY_BARS, "day", include_after_hours=False)
    _check(day_regular is not None, "no daily bars for SPY with include_after_hours=False")
    assert day_regular is not None
    print(f"daily bars, include_after_hours=False: {len(day_regular.df)} bars")
    _check(
        len(day_regular.df) == len(day.df),
        "include_after_hours=False should be a no-op for day bars "
        f"(got {len(day_regular.df)} vs {len(day.df)})",
    )

    extended = strategy.get_historical_prices("SPY", MINUTE_BARS, "minute")
    regular = strategy.get_historical_prices(
        "SPY", MINUTE_BARS, "minute", include_after_hours=False
    )
    _check(extended is not None and regular is not None, "no minute bars for SPY")
    assert extended is not None and regular is not None
    print(f"minute bars, extended hours: {len(extended.df)} ({extended.df.index[0]} .. {extended.df.index[-1]})")
    print(f"minute bars, regular hours:  {len(regular.df)} ({regular.df.index[0]} .. {regular.df.index[-1]})")
    _check(
        all(time(9, 30) <= ts.time() < time(16, 0) for ts in regular.df.index),
        "include_after_hours=False returned bars outside 09:30-16:00",
    )

    many = strategy.get_historical_prices_for_assets(SYMBOLS, DAY_BARS, "day")
    print(f"get_historical_prices_for_assets: {({a.symbol: len(b.df) for a, b in many.items()})}")
    _check({asset.symbol for asset in many} == set(SYMBOLS), "missing daily bars in the batch")

    sma = strategy.indicators.sma("SPY", length=20)
    bbands = strategy.indicators.bbands("SPY", length=20, std=2)
    rsi = strategy.indicators.rsi("SPY", timestep="minute", length=14)
    print(f"sma(20) = {sma}\nbbands(20, 2) = {bbands}\nrsi(14, minute) = {rsi}")
    _check(sma is not None and bbands is not None and rsi is not None, "an indicator returned None")

    print("\nPASS: last prices, quote, day/minute bars, batch bars and indicators all returned data.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
