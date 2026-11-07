#!/usr/bin/env python3
"""Manual run of the Yahoo traded close/volume against the real Yahoo.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_traded_prices.py

Read-only, no credentials. Three checks, each printed as PASS or FAIL; the exit code is non-zero on
any FAIL:

1. NVDA over January 2015 through `YahooBacktestData`: its first session's `traded_close` is what
   the stock really printed (between $15 and $25), while its adjusted `close` is under $1 (NVDA has
   split since: 4:1 in 2021, 10:1 in 2024).
2. AAPL, MSFT, NVDA and KO over 2023: the `open/high/low/close/volume` rebuilt from an
   `auto_adjust=False, actions=True` download equal `yfinance.utils.auto_adjust` applied to the
   same raw download, to 1e-9 relative. If this fails, the rebuild does not match yfinance's
   arithmetic and the filters' adjusted series have moved. The difference against a separate
   `auto_adjust=True` download is printed as INFO only: Yahoo's adjusted values vary between requests.
3. KO over 2012: its mean traded dollar volume (traded close x traded volume) is above its adjusted
   one (close x volume), because the adjusted close of a dividend payer is lower the further back
   it goes.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import (
    TRADED_COLUMNS,
    YahooBacktestData,
    extract_ticker_frame,
    parse_yahoo_frame,
    yfinance_download,
)
from trading_agent_framework.entities.asset import Asset

MARKET_TZ = ZoneInfo("America/New_York")
NVDA_2015_TRADED_RANGE = (15.0, 25.0)
NVDA_2015_MAX_ADJUSTED_CLOSE = 1.0
REBUILD_SYMBOLS = ("AAPL", "MSFT", "NVDA", "KO")
REBUILD_TOLERANCE = 1e-9
OHLCV = ["open", "high", "low", "close", "volume"]


def _window(start: date, end: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(start, datetime.min.time(), tzinfo=MARKET_TZ),
        datetime.combine(end, datetime.min.time(), tzinfo=MARKET_TZ),
    )


def check_nvda_2015() -> tuple[bool, str]:
    """Check that NVDA's January 2015 traded close is the real price, while the adjusted close is far lower.

    Returns:
        A (passed, detail) pair; the detail is printed next to the verdict.
    """
    start, end = _window(date(2015, 1, 1), date(2015, 1, 31))
    source = YahooBacktestData(start, end)
    asset = Asset("NVDA")
    source.load([asset], start, end, "day")
    bars = source.bars(asset, end, 100, "day")
    if bars is None or bars.df.empty:
        return False, "no NVDA bars for January 2015"
    first = bars.df.iloc[0]
    traded, adjusted = float(first["traded_close"]), float(first["close"])
    low, high = NVDA_2015_TRADED_RANGE
    ok = low <= traded <= high and adjusted < NVDA_2015_MAX_ADJUSTED_CLOSE
    detail = f"NVDA {bars.df.index[0].date()}: traded_close ${traded:.2f} (want ${low:.0f}-${high:.0f}), adjusted close ${adjusted:.4f} (want under ${NVDA_2015_MAX_ADJUSTED_CLOSE:.0f})"
    return ok, detail


def _download_raw(symbol: str, *, auto_adjust: bool, start: str, end: str) -> pd.DataFrame:
    kwargs = {} if auto_adjust else {"actions": True}
    raw = yfinance_download()([symbol], start=start, end=end, auto_adjust=auto_adjust, progress=False, group_by="ticker", **kwargs)
    return extract_ticker_frame(raw, symbol)


def _worst_relative_difference(left: pd.DataFrame, right: pd.DataFrame) -> float:
    return float(((left[OHLCV] - right[OHLCV]).abs() / right[OHLCV].abs()).to_numpy().max())


# The design's guarantee: the rebuilt adjusted bars ARE yfinance's own `auto_adjust` arithmetic. So
# they are compared with `yfinance.utils.auto_adjust` applied to the SAME raw download, at 1e-9.
# They cannot be compared at 1e-9 with a separate `auto_adjust=True` request: Yahoo's adjusted
# values vary from one request to the next (up to 3.3e-04 between two back-to-back AAPL downloads),
# so that difference is printed as information only and never fails the script.
def check_rebuilt_adjusted_equal_yfinance() -> tuple[bool, str]:
    """Check that the rebuilt adjusted bars equal yfinance's own `auto_adjust` on the same download.

    Returns:
        A (passed, detail) pair; the detail is printed next to the verdict.
    """
    from yfinance.utils import auto_adjust

    # Through today, so every later split is covered (Yahoo adjusts to the download date).
    end = (date.today() + timedelta(days=1)).isoformat()
    lines: list[str] = []
    ok = True
    for symbol in REBUILD_SYMBOLS:
        raw = _download_raw(symbol, auto_adjust=False, start="2023-01-01", end=end)
        rebuilt = parse_yahoo_frame(raw)
        reference = auto_adjust(raw.copy())
        reference.columns = [str(c).lower() for c in reference.columns]
        reference = reference[OHLCV].astype("float64").dropna()
        rebuilt = rebuilt[rebuilt.index.year == 2023]
        reference = reference[reference.index.year == 2023]
        if rebuilt.empty or list(rebuilt.index.date) != list(reference.index.date):
            ok = False
            lines.append(f"{symbol}: dates differ ({len(rebuilt)} vs {len(reference)} rows)")
            continue
        reference.index = rebuilt.index  # safe: the dates were just checked equal, and only the tz-aware index differs
        worst = _worst_relative_difference(rebuilt, reference)
        if not (worst <= REBUILD_TOLERANCE):  # also fails on NaN, where `worst > tolerance` is False
            ok = False
        lines.append(f"{symbol}: {len(rebuilt)} rows, worst relative difference vs yfinance auto_adjust {worst:.2e}")
        plain = parse_yahoo_frame(_download_raw(symbol, auto_adjust=True, start="2023-01-01", end=end))
        plain = plain[plain.index.year == 2023]
        if list(plain.index) == list(rebuilt.index):
            lines.append(f"  INFO {symbol}: vs a separate auto_adjust=True download {_worst_relative_difference(rebuilt, plain):.2e} (Yahoo's adjusted values vary between requests; not a failure)")
        else:
            lines.append(f"  INFO {symbol}: separate auto_adjust=True download has different dates; not compared")
    return ok, "\n      ".join(lines)


def check_ko_2012_dollar_volume() -> tuple[bool, str]:
    """Check that KO's 2012 traded dollar volume is above the adjusted one.

    Returns:
        A (passed, detail) pair; the detail is printed next to the verdict.
    """
    start, end = _window(date(2012, 1, 1), date(2012, 12, 31))
    source = YahooBacktestData(start, end)
    asset = Asset("KO")
    source.load([asset], start, end, "day")
    bars = source.bars(asset, end, 400, "day")
    if bars is None or bars.df.empty:
        return False, "no KO bars for 2012"
    df = bars.df
    traded = float((df[TRADED_COLUMNS[0]] * df[TRADED_COLUMNS[1]]).mean())
    adjusted = float((df["close"] * df["volume"]).mean())
    return traded > adjusted, (f"KO 2012 ({len(df)} sessions): mean traded dollar volume ${traded / 1e6:.1f}M, adjusted ${adjusted / 1e6:.1f}M (ratio {traded / adjusted:.3f})")


def main() -> int:
    """Run the three checks and print PASS or FAIL for each.

    Returns:
        The process exit code: 0 on success or when skipped, 1 on failure.
    """
    checks = (
        ("1. NVDA 2015 traded close vs adjusted close", check_nvda_2015),
        ("2. rebuilt adjusted bars equal yfinance auto_adjust on the same download (1e-9)", check_rebuilt_adjusted_equal_yfinance),
        ("3. KO 2012 traded dollar volume above adjusted", check_ko_2012_dollar_volume),
    )
    failed = False
    for title, check in checks:
        try:
            ok, detail = check()
        except Exception as exc:  # a manual script: report any Yahoo/parse failure as a FAIL
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        failed = failed or not ok
        print(f"{'PASS' if ok else 'FAIL'}: {title}\n      {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
