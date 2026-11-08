#!/usr/bin/env python3
"""Manual run of the regime series against the real Yahoo.

NOT part of the automated test suite (it needs the network). Read-only, no credentials:

    uv run python scripts/tests/smoke_regime_data.py

Prints the last 5 aligned rows of `YahooRegimeData().closes_before(today, 503)`, its length (expect 503, the last
row dated before today) and today's regime reading.
"""

from __future__ import annotations

from datetime import date

from trading_agent_framework.core.regime import RegimeParameters, classify_regime
from trading_agent_framework.core.regime_data import YahooRegimeData


def main() -> None:
    """Fetch, print the tail and classify."""
    length = RegimeParameters().min_bars
    series = YahooRegimeData().closes_before(date.today(), length)
    print("session      SPY       VIX     VIX3M")
    for i in range(-5, 0):
        print(f"{series.sessions[i]}  {series.spy[i]:8.2f}  {series.vix[i]:6.2f}  {series.vix3m[i]:6.2f}")
    print(f"rows: {len(series)} (expected {length}); last session {series.sessions[-1]} < {date.today()}")
    print(classify_regime(series.spy, series.vix, series.vix3m))


if __name__ == "__main__":
    main()
