"""Spike: does an inverse ETF held open-to-close on `in_market_fall` days make money?

Design (rules fixed before running): docs/superpowers/specs/2026-11-23-market-fall-intraday-spike-design.md

Run:
  uv run python scripts/regime/fall_intraday_spike.py check   # self-check on synthetic data
  uv run python scripts/regime/fall_intraday_spike.py dev     # 2016 -> 2024: run, read
  uv run python scripts/regime/fall_intraday_spike.py test    # 2025 -> last session, ONCE, chosen cell only
"""

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_agent_framework.utils.helpers import in_market_fall

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "cache" / "fall_intraday"
COST = 0.0003  # 3 bps per side
NY = "America/New_York"
ENV_STRATEGY, ENV_MODE = "bull_bear", "paper"  # env/.env.bull_bear.paper borrowed for its read-only ALPACA_DATA_API_* keys; no market_fall env exists


def fall_days(closes: pd.DataFrame) -> pd.Series:
    """Whether `in_market_fall` is true before each session's open: row d is decided from rows before d (row 0 is False)."""
    vix, vix3m, vixy = closes.vix.tolist(), closes.vix3m.tolist(), closes.vixy.tolist()
    flags = [False] + [in_market_fall(vix[:i], vix3m[:i], vixy[:i]) for i in range(1, len(closes))]
    return pd.Series(flags, index=closes.index)


def daily_closes() -> pd.DataFrame:
    """Yahoo ^VIX, ^VIX3M and VIXY closes from 2008-01-02 (columns vix, vix3m, vixy; date index). VIXY is NaN before 2011."""
    import yfinance as yf

    raw = yf.download(["^VIX", "^VIX3M", "VIXY"], start="2008-01-02", progress=False, auto_adjust=True)["Close"]
    raw = raw.rename(columns={"^VIX": "vix", "^VIX3M": "vix3m", "VIXY": "vixy"}).dropna(subset=["vix", "vix3m"])
    raw.index = raw.index.date
    return raw[["vix", "vix3m", "vixy"]]


def minute_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Regular-session 1-minute bars (open, high, low, close; New York index), Alpaca SIP, split/dividend adjusted.

    Cached per symbol-year under cache/fall_intraday/. shortcut: the current year is never cached, so a partial year is
    never frozen; fine for a spike, drop the exception if the script is re-run daily.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(start.year, end.year + 1):
        path = CACHE / f"{symbol}_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        df = _download_year(symbol, year)
        if year < date.today().year:
            df.to_parquet(path)
        frames.append(df)
    out = pd.concat(frames)
    return out[(out.index.date >= start) & (out.index.date <= end)]


def _download_year(symbol: str, year: int) -> pd.DataFrame:
    """One calendar year of regular-session minute bars from Alpaca (empty frame when the symbol has none)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from trading_agent_framework.config.env import AlpacaCredentials, load_strategy_env

    load_strategy_env(ENV_STRATEGY, ENV_MODE)
    creds = AlpacaCredentials.for_data()
    client = StockHistoricalDataClient(creds.api_key, creds.api_secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime(year, 1, 1),
        end=datetime(year, 12, 31, 23, 59),
        feed=DataFeed.SIP,
        adjustment=Adjustment.ALL,
    )
    df = client.get_stock_bars(request).df
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz=NY))
    return regular_session(df.droplevel("symbol")[["open", "high", "low", "close"]])


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the 09:30-15:59 New York bars of a UTC-indexed frame (early-close days simply end sooner)."""
    df = df.copy()
    df.index = df.index.tz_convert(NY)
    return df.between_time("09:30", "15:59")


def inverse_of(bars: pd.DataFrame) -> pd.DataFrame:
    """The synthetic -1 x SPY: per session each price p becomes 2 - p / session_open (high and low swap)."""
    out = []
    for _, day in bars.groupby(bars.index.date):
        first = day.open.iloc[0]
        out.append(pd.DataFrame({"open": 2 - day.open / first, "high": 2 - day.low / first, "low": 2 - day.high / first, "close": 2 - day.close / first}))
    return pd.concat(out)


def day_return(bars: pd.DataFrame, stop: float | None, cost: float = COST) -> tuple[float, str]:
    """One session held open to close: the multiplier on the money traded (both costs in) and how it ended.

    The trailing stop sits `stop` below the highest high of the EARLIER bars (seeded with the entry open); a bar that
    opens at or under it fills at its open, else one that trades down to it fills at the level. No re-entry.
    """
    o, h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    entry, exit_price, how, peak = o[0], c[-1], "close", o[0]
    if stop is not None:
        for i in range(len(o)):
            level = peak * (1 - stop)
            if o[i] <= level:
                exit_price, how = o[i], "stop"
                break
            if lo[i] <= level:
                exit_price, how = level, "stop"
                break
            peak = max(peak, h[i])
    return exit_price * (1 - cost) / (entry * (1 + cost)), how


def close_to(value: float, expected: float, costs: bool = False) -> bool:
    """Compare to `expected`, optionally applying the 3 bps entry and exit cost to it first."""
    if costs:
        expected = expected * (1 - COST) / (1 + COST)
    return abs(value - expected) < 1e-9


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """One New York session of 1-minute bars from (open, high, low, close) rows, starting 09:30."""
    idx = pd.date_range(datetime(2024, 3, 4, 9, 30), periods=len(rows), freq="min", tz=NY)
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def synthetic_closes(n: int) -> pd.DataFrame:
    """Random-walk VIX-like closes with a stressed stretch at rows 150-160 (VIX above VIX3M)."""
    rng = np.random.default_rng(1)
    idx = [date(2020, 1, 1) + pd.Timedelta(days=i) for i in range(n)]
    vix3m = 20 + rng.random(n)
    vix = vix3m * (0.8 + 0.05 * rng.random(n))
    vix[150:160] = vix3m[150:160] * 1.05
    vixy = 20 + rng.random(n)
    return pd.DataFrame({"vix": vix, "vix3m": vix3m, "vixy": vixy}, index=idx)


def check() -> None:
    """Assertions on synthetic data: no look-ahead, stop rules, gap fills, the inverse series."""
    # 1. no look-ahead: changing day k's closes never changes day k's verdict
    closes = synthetic_closes(300)
    base = fall_days(closes)
    for k in range(200, 300, 9):
        moved = closes.copy()
        moved.iloc[k] = moved.iloc[k] * [3.0, 1.0, 2.0]
        assert fall_days(moved).iloc[k] == base.iloc[k]
    assert base.iloc[151:160].any()  # the stressed stretch is seen the day after
    # 2. no stop: open 100 -> close 90, cost 3 bps each side
    bars = frame([(100, 101, 99, 100), (100, 100, 90, 90)])
    mult, exit_ = day_return(bars, None)
    assert exit_ == "close" and close_to(mult, 0.9, costs=True)
    # 3. stop 2%: high 105 on bar 1, bar 2 low 102 -> stop at 105 * 0.98 = 102.9, filled at 102.9
    bars = frame([(100, 105, 100, 104), (104, 104, 102, 103), (103, 120, 103, 120)])
    mult, exit_ = day_return(bars, 0.02)
    assert exit_ == "stop" and close_to(mult, 1.029, costs=True)
    # 4. a bar's own high does not raise its stop: bar 2 high 110 then low 104 -> not stopped (stop is 98 from bar 1)
    bars = frame([(100, 100, 99, 100), (100, 110, 104, 105)])
    assert day_return(bars, 0.02)[1] == "close"
    # 5. gap through: stop 98, next bar opens 95 -> filled at 95
    bars = frame([(100, 100, 99, 99), (95, 96, 94, 95)])
    assert close_to(day_return(bars, 0.02)[0], 0.95, costs=True)
    # 6. inverse: SPY 100 -> 99 gives 1 -> 1.01
    assert close_to(inverse_of(frame([(100, 100, 99, 99)])).iloc[-1]["close"], 1.01)
    # 7. regular session in New York time, whatever the UTC offset: summer 13:30 UTC and winter 14:30 UTC are both 09:30
    for day, open_utc in (("2024-07-01", "13:30"), ("2024-12-02", "14:30")):
        idx = pd.date_range(f"{day} {open_utc}", periods=3, freq="min", tz="UTC").append(
            pd.DatetimeIndex([pd.Timestamp(f"{day} {open_utc}", tz="UTC") - pd.Timedelta(minutes=1), pd.Timestamp(f"{day} {open_utc}", tz="UTC") + pd.Timedelta(hours=7)])
        )
        raw = pd.DataFrame(1.0, index=idx, columns=["open", "high", "low", "close"])
        kept = regular_session(raw)
        assert len(kept) == 3 and str(kept.index[0].time()) == "09:30:00", day  # pre-market and 16:30 bars dropped
    print("self-check OK")


def main() -> None:
    """Entry point: `check`, `dev` or `test` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
