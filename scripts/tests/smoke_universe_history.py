#!/usr/bin/env python3
"""Manual run of the point-in-time universe history against the real Yahoo.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_universe_history.py

Read-only apart from the bar cache; no credentials. It builds the history from 2017-06 to the current month with
`ensure_universe_history`, writing the history file to a temporary directory (the real
`data/universe/us_stock_universe_history.json` is not touched) and using the REAL bar cache,
`cache/universe_history/`, so a rerun (or a later backtest) reuses the download.

**The first run is slow**: it checks the whole candidate pool for common stocks and downloads its daily bars in
batches of 100, which takes tens of minutes to over an hour and hits Yahoo hard. A rerun reuses the cached bars and
the pool saved in the cache. Run it when you can wait, not in a loop.

Four checks, each printed as PASS or FAIL; the exit code is non-zero on any FAIL:

1. NVDA is in the 2017-06 list (it traded about $150 then; its split-adjusted price, under $1 from today's
   viewpoint, would have failed the $10 screen).
2. Every 2017-06 symbol's last traded close before 2017-06-01 (the close of 2017-05-31, or of the last session it
   traded in the 10 before) is at least $10, read from the cached raw bars, not from the history.
3. The list has at most 1,200 symbols.
4. At least one 2017-06 symbol is absent from the latest month's list (the universe does move; a list that never
   changes would be today's file again).
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import parse_yahoo_frame
from trading_agent_framework.strategies.common.universe_history import UniverseHistory
from trading_agent_framework.strategies.cross_momentum.batch_us_stock_universe import (
    _DEFAULT_HISTORY_CACHE,
    ensure_universe_history,
)

START = date(2017, 6, 1)
CHECK_MONTH = "2017-06"
BEFORE_MONTH = date(2017, 5, 31)  # the last session before 2017-06
MIN_PRICE = 10.0
TOP_N = 1200


def check_nvda_in_2017_06(history: UniverseHistory, bars_dir: Path) -> tuple[bool, str]:
    symbols = history.months[CHECK_MONTH]
    if "NVDA" not in symbols:
        return False, f"NVDA is not among the {len(symbols)} symbols of {CHECK_MONTH}"
    return True, f"NVDA is number {symbols.index('NVDA') + 1} of {len(symbols)} in {CHECK_MONTH}"


def check_prices_in_2017_06(history: UniverseHistory, bars_dir: Path) -> tuple[bool, str]:
    below: list[str] = []
    for symbol in history.months[CHECK_MONTH]:
        frame = parse_yahoo_frame(pd.read_parquet(bars_dir / f"{symbol}.parquet"))
        closes = frame.loc[frame.index.date <= BEFORE_MONTH, "traded_close"]
        if closes.empty or float(closes.iloc[-1]) < MIN_PRICE:
            below.append(symbol)
    if below:
        return False, f"{len(below)} symbol(s) of {CHECK_MONTH} under ${MIN_PRICE:g} on {BEFORE_MONTH}: {', '.join(below[:20])}"
    return True, f"all {len(history.months[CHECK_MONTH])} symbols of {CHECK_MONTH} traded at ${MIN_PRICE:g} or more on {BEFORE_MONTH}"


def check_at_most_top_n(history: UniverseHistory, bars_dir: Path) -> tuple[bool, str]:
    largest_month, largest = max(((month, len(symbols)) for month, symbols in history.months.items()), key=lambda item: item[1])
    return largest <= TOP_N, f"the longest list has {largest} symbols ({largest_month}), the limit is {TOP_N}"


def check_universe_moves(history: UniverseHistory, bars_dir: Path) -> tuple[bool, str]:
    latest = max(history.months)
    gone = sorted(set(history.months[CHECK_MONTH]) - set(history.months[latest]))
    return bool(gone), f"{len(gone)} symbol(s) of {CHECK_MONTH} are not in {latest}'s list (e.g. {', '.join(gone[:5]) or 'none'})"


CHECKS = (
    ("1. NVDA is in the 2017-06 list", check_nvda_in_2017_06),
    ("2. every 2017-06 symbol traded at $10 or more on 2017-05-31", check_prices_in_2017_06),
    ("3. no list has more than 1,200 symbols", check_at_most_top_n),
    ("4. a 2017-06 symbol is absent from the latest month's list", check_universe_moves),
)


def run_checks(history: UniverseHistory, bars_dir: Path) -> bool:
    """Print PASS or FAIL for each check; True when all passed."""
    ok_all = True
    for title, check in CHECKS:
        try:
            ok, detail = check(history, bars_dir)
        except Exception as exc:  # a manual script: report any parse or file failure as a FAIL
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        ok_all = ok_all and ok
        print(f"{'PASS' if ok else 'FAIL'}: {title}\n      {detail}")
    return ok_all


def main() -> int:
    end = date.today().replace(day=1)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            history = ensure_universe_history(START, end, path=Path(tmp) / "us_stock_universe_history.json")
        except Exception as exc:
            print(f"FAIL: building the history {START:%Y-%m} to {end:%Y-%m}\n      {type(exc).__name__}: {exc}")
            return 1
    print(f"built {len(history.months)} monthly lists, {min(history.months)} to {max(history.months)} (pool of {history.pool_date})")
    return 0 if run_checks(history, _DEFAULT_HISTORY_CACHE / "bars") else 1


if __name__ == "__main__":
    sys.exit(main())
