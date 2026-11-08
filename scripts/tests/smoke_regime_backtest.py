#!/usr/bin/env python3
"""Manual smoke test: a backtest's `Regime` line equals the regime CLI's value for every session.

Runs a do-nothing strategy through `Strategy.run_backtesting()` over 2025-03-03..2025-04-30 against the real Yahoo
(`YahooBacktestData` and `YahooRegimeData`), then recomputes the regime from a separate `YahooRegimeData().series()`
download with the helpers of `regime_cli` and checks that every `Regime` row of the run's `indicators.parquet` matches
(acceptance 2 of the VIX market regime design).

The run's output goes to a temporary project root, so nothing is written under the repository's logs/, memory/ or data/.
Excluded from `tests/` on purpose: it hits the real network and is meant to be run by hand.

Run: uv run python scripts/tests/smoke_regime_backtest.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.placeholder import PlaceholderBroker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.regime import REGIME_LINE, RegimeParameters, regime_by_session
from trading_agent_framework.core.regime_cli import lookback_days, trim
from trading_agent_framework.core.regime_data import YahooRegimeData
from trading_agent_framework.core.strategy import Strategy

ET = ZoneInfo("America/New_York")
START = datetime(2025, 3, 3, tzinfo=ET)
END = datetime(2025, 4, 30, tzinfo=ET)
MAX_SHOWN = 5


class IdleStrategy(Strategy):
    """Does nothing; the run only exists to log the regime each session."""

    sleeptime = "1D"
    benchmark_symbol = "SPY"


def backtest_regimes(project_root: Path) -> dict[date, int]:
    """Run the idle backtest and read back its `Regime` rows.

    Args:
        project_root: Where the run's logs are written (a temporary directory).

    Returns:
        The regime the run logged, by New York session date.
    """
    strategy = IdleStrategy(PlaceholderBroker("idle"), mode=TradingMode.BACKTESTING, project_root=project_root)
    result = strategy.run_backtesting(start=START, end=END, timestep="day", data_source=YahooBacktestData, budget=10000)
    indicators = pd.read_parquet(result.run_dir / "indicators.parquet")
    rows = indicators[indicators["name"] == REGIME_LINE]
    stamps = pd.DatetimeIndex(rows["datetime"]).tz_convert(ET)
    return {ts.date(): int(value) for ts, value in zip(stamps, rows["value"], strict=True)}


def expected_regimes(sessions: list[date]) -> dict[date, int]:
    """The regime of each session, from a download separate from the backtest's.

    Args:
        sessions: The sessions to read.

    Returns:
        The regime by session date (sessions without a reading are absent).
    """
    series = trim(YahooRegimeData().series(min(sessions) - timedelta(days=lookback_days())), max(sessions))
    readings = regime_by_session(series.sessions, series.spy, series.vix, series.vix3m, RegimeParameters())
    return {d: readings[d].regime for d in sessions if d in readings}


def main() -> int:
    """Run the backtest and compare its regime with the recomputed one.

    Returns:
        The process exit code: 0 when every session matches, 1 otherwise.
    """
    with tempfile.TemporaryDirectory(prefix="smoke_regime_backtest_") as tmp:
        actual = backtest_regimes(Path(tmp))
    if not actual:
        print("FAIL: the backtest logged no Regime row")
        return 1
    expected = expected_regimes(sorted(actual))
    problems = [f"{d}: backtest {actual[d]}, recompute {expected.get(d)}" for d in sorted(actual) if expected.get(d) != actual[d]]
    if problems:
        print(f"FAIL: regime differs on {len(problems)} of {len(actual)} sessions, first:")
        for line in problems[:MAX_SHOWN]:
            print(f"  {line}")
        return 1
    print(f"regime matches on {len(actual)} sessions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
