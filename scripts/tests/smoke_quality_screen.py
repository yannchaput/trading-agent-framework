#!/usr/bin/env python3
"""Manual run of the fundamentals quality screen against real SEC EDGAR and Yahoo data.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_quality_screen.py

`SEC_EDGAR_USER_AGENT` comes from env/.env.alpaca.integration-tests, as in the other smoke scripts.
The script is read-only. The first run downloads one SEC company-facts payload per symbol (about
4 MB each) and writes reduced copies under cache/sec/annual/; a second run the same day makes no
SEC request and finishes in a few seconds. It fails only on things that must always hold: every
symbol is either a candidate or rejected with a reason, an unknown ticker is `no_data`, a bank is
rejected, and the two Alphabet listings never both become candidates. (The bank's reason is printed, not
asserted: the numeric gates run before the sector gate, and a bank usually fails those first.)
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.fundamentals import ScreenParams, build_quality_screen
from trading_agent_framework.utils.errors import TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

UNKNOWN_SYMBOL = "ZZZZZZ"
BANK_SYMBOL = "JPM"
SYMBOLS = [
    "GOOGL", "GOOG", "CMG", "HLT", "QSR", "UBER", "CP", "LOW", "MDLZ", "BKNG", "MSFT",
    "AAPL", "NVDA", "COST", "KO", "V", "NEE", "TSM", BANK_SYMBOL, UNKNOWN_SYMBOL,
]  # fmt: skip


class SmokeTestFailure(Exception):
    pass


def _last_close(symbol: str) -> Decimal | None:
    import yfinance as yf

    try:
        closes = yf.Ticker(symbol).history(period="5d")["Close"].dropna()
    except Exception as exc:  # a manual script: any Yahoo failure is just "no price" for that symbol
        print(f"  no price for {symbol}: {exc}")
        return None
    return None if closes.empty else Decimal(str(round(float(closes.iloc[-1]), 4)))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(ENV_FILE)
    try:
        screen = build_quality_screen(PROJECT_ROOT, params=ScreenParams(top_n=len(SYMBOLS)))
        started = time.perf_counter()
        result = screen.run(SYMBOLS, as_of=datetime.now(UTC), price_of=_last_close)
        elapsed = time.perf_counter() - started
    except TradingFrameworkError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"\nCandidates ({len(result.candidates)}), screened in {elapsed:.1f}s:")
    print(f"{'#':>2} {'symbol':<6} {'score':>5} {'fcf yield':>9} {'fcf margin':>10} {'margin sd':>9} {'growth':>7} {'debt x':>6} {'mkt cap $B':>10}  fiscal year / filed")
    for c in result.candidates:
        debt = f"{c.net_debt_to_operating_income:6.2f}" + ("" if c.debt_reported else "*")
        print(
            f"{c.rank:>2} {c.symbol:<6} {c.score:5.2f} {c.fcf_yield:9.2%} {c.fcf_margin:10.2%} {c.operating_margin_stdev:9.2%} "
            f"{c.revenue_growth:7.2%} {debt} {float(c.market_cap) / 1e9:10.1f}  {c.fiscal_year_end} / {c.filed}"
        )
    print("  (* = no debt figure reported: counted as zero)")
    print(f"\nRejections ({len(result.rejections)}):")
    for symbol, reason in sorted(result.rejections.items()):
        print(f"  {symbol:<6} {reason}")

    candidates = {c.symbol for c in result.candidates}
    try:
        _check(candidates | set(result.rejections) == set(SYMBOLS), "a symbol is neither a candidate nor rejected")
        _check(not candidates & set(result.rejections), "a symbol is both a candidate and rejected")
        _check(result.rejections.get(UNKNOWN_SYMBOL) == "no_data", f"{UNKNOWN_SYMBOL} should be no_data")
        _check(BANK_SYMBOL in result.rejections, f"{BANK_SYMBOL} (a bank) should be rejected")
        _check(not {"GOOGL", "GOOG"} <= candidates, "both Alphabet listings became candidates")
    except SmokeTestFailure as exc:
        print(f"\nFAILED: {exc}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
