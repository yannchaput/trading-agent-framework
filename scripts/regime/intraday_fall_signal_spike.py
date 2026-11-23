"""Spike: a 10-minute VIXY / SPY signal that trades an inverse S&P ETF intraday, with SHV overnight.

Design (rules fixed before running): docs/superpowers/specs/2026-11-23-intraday-fall-signal-spike-design.md

Run:
  uv run python scripts/regime/intraday_fall_signal_spike.py check   # self-check on synthetic data
  uv run python scripts/regime/intraday_fall_signal_spike.py dev     # 2016-01-04 -> 2024-06-28: run, read
  uv run python scripts/regime/intraday_fall_signal_spike.py test    # 2024-07-01 -> last session, ONCE, chosen cell only

A decision at time t (the close of a 10-minute block) reads only 1-minute bars stamped before t and trades at the open
of the bar stamped t.
"""

import importlib.util
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fall = _load("fall_intraday_spike", REPO / "scripts" / "regime" / "fall_intraday_spike.py")
NY, is_early_close, trim_early_closes, regular_session = _fall.NY, _fall.is_early_close, _fall.trim_early_closes, _fall.regular_session
ENV_STRATEGY, ENV_MODE = _fall.ENV_STRATEGY, _fall.ENV_MODE
CACHE = REPO / "cache" / "intraday_fall_signal"
TEST_GUARD = CACHE / "test.done"
SESSION_MINUTES = 390  # 09:30 -> 15:59


def minute_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Regular-session 1-minute bars (open, high, low, close, volume; New York index), Alpaca SIP, adjusted.

    Cached per symbol-year under cache/intraday_fall_signal/. shortcut: the current year is never cached, so a partial
    year is never frozen; fine for a spike.
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
    out = trim_early_closes(pd.concat(frames))
    return out[(out.index.date >= start) & (out.index.date <= end)]


def _download_year(symbol: str, year: int) -> pd.DataFrame:
    """One calendar year of regular-session minute bars with volume from Alpaca (empty frame when there are none)."""
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
    cols = ["open", "high", "low", "close", "volume"]
    if df.empty:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz=NY))
    return regular_session(df.droplevel("symbol")[cols])


def daily_yahoo() -> pd.DataFrame:
    """Yahoo adjusted VIXY close with its SMA50 (the 50 closes BEFORE each day) and SHV open and close, date index."""
    import yfinance as yf

    raw = yf.download(["VIXY", "SHV"], start="2015-06-01", progress=False, auto_adjust=True)
    out = pd.DataFrame(
        {
            "vixy_close": raw["Close"]["VIXY"],
            "vixy_sma50": raw["Close"]["VIXY"].rolling(50).mean().shift(1),
            "shv_open": raw["Open"]["SHV"],
            "shv_close": raw["Close"]["SHV"],
        }
    )
    out.index = out.index.date
    return out


def decision_times(session: date, early: bool) -> list[pd.Timestamp]:
    """Decision times 09:40, 09:50, ... 15:40 (12:40 on early closes), New York time."""
    last = "12:40" if early else "15:40"
    return list(pd.date_range(f"{session} 09:40", f"{session} {last}", freq="10min", tz=NY))


def day_context(prev: list[dict], sma50: float) -> dict | None:
    """What D-1..D-5 give a session: D-1 closes and VIXY VWAP, VIXY's SMA50, each day's cumulative VIXY volume.

    `prev[i]` = {"spy": df, "vixy": df} minute frames of the i+1-th previous session (most recent first); None when
    fewer than 5. `vixy_cumvol` is a list of (cum, reach): cum[m] = VIXY volume of the day's minutes 0..m (since
    09:30), reach = the minute count the day's bars cover (an early close or a thin end reaches less than 390).
    """
    if len(prev) < 5:
        return None
    d1 = prev[0]["vixy"]
    cumvol = []
    for p in prev[:5]:
        v = p["vixy"]
        minute = ((v.index.hour - 9) * 60 + v.index.minute - 30).to_numpy()
        per_minute = np.zeros(SESSION_MINUTES)
        per_minute[minute] = v.volume.to_numpy()
        cumvol.append((np.cumsum(per_minute), int(minute.max()) + 1))
    return {
        "vixy_d1_close": float(d1.close.iloc[-1]),
        "vixy_d1_vwap": float((d1.close * d1.volume).sum() / d1.volume.sum()),
        "spy_d1_close": float(prev[0]["spy"].close.iloc[-1]),
        "vixy_sma50": sma50,
        "vixy_cumvol": cumvol,
    }


def _now_and_before(df: pd.DataFrame, t: pd.Timestamp, minutes: int) -> tuple[float, float]:
    """Last close before t, and its reference `minutes` earlier.

    The reference is the close of the last bar stamped before t - minutes, or the session's first open when that moment
    is the 09:30 open or earlier (a window never crosses a night).
    """
    done = df[df.index < t]
    now = float(done.close.iloc[-1])
    ref_time = t - timedelta(minutes=minutes)
    if ref_time <= t.normalize() + timedelta(hours=9, minutes=30):
        return now, float(df.open.iloc[0])
    return now, float(done[done.index < ref_time].close.iloc[-1])


def observe(spy: pd.DataFrame, vixy: pd.DataFrame, ctx: dict, t: pd.Timestamp) -> dict:
    """The spec's signal values at decision time t, from bars stamped before t only."""
    vixy_now, vixy_hour_ref = _now_and_before(vixy, t, 60)
    spy_now, spy_hour_ref = _now_and_before(spy, t, 60)
    _, vixy_30_ref = _now_and_before(vixy, t, 30)
    _, spy_30_ref = _now_and_before(spy, t, 30)
    elapsed = int((t - t.normalize()).total_seconds() // 60) - (9 * 60 + 30)
    past = [cum[elapsed - 1] for cum, reach in ctx["vixy_cumvol"] if reach >= elapsed]
    today_volume = float(vixy[vixy.index < t].volume.sum())
    return {
        "vixy_day": vixy_now / ctx["vixy_d1_close"] - 1,
        "vixy_hour": vixy_now / vixy_hour_ref - 1,
        "vixy_vs_d1avg": vixy_now / ctx["vixy_d1_vwap"] - 1,
        "vixy_trend": vixy_now > ctx["vixy_sma50"],
        "rvol": today_volume / float(np.mean(past)) if past and np.mean(past) > 0 else 0.0,
        "spy_day": spy_now / ctx["spy_d1_close"] - 1,
        "spy_hour": spy_now / spy_hour_ref - 1,
        "vixy_30": vixy_now / vixy_30_ref - 1,
        "spy_30": spy_now / spy_30_ref - 1,
    }


def close_to(value: float, expected: float) -> bool:
    """Float compare at 1e-9."""
    return abs(value - expected) < 1e-9


def synth_day(d: date, drift: float = 0.0) -> pd.DataFrame:
    """A random-walk session of 1-minute bars with volume, 09:30-15:59 New York, seeded by the date."""
    rng = np.random.default_rng(d.toordinal())
    idx = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="min", tz=NY)
    close = 20 * np.exp(np.cumsum(rng.normal(drift / 390, 0.001, len(idx))))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) * 1.0005, "low": np.minimum(open_, close) * 0.9995, "close": close, "volume": rng.integers(100, 1000, len(idx)).astype(float)},
        index=idx,
    )


def five_days_before(d: date) -> list[date]:
    """The five weekdays before d, most recent first."""
    out, cur = [], d
    while len(out) < 5:
        cur -= timedelta(days=1)
        if cur.weekday() < 5:
            out.append(cur)
    return out


def check() -> None:
    """Assertions on synthetic data: no look-ahead, window clips, same-clock volume, thin minutes, D-1 values."""
    day = date(2024, 3, 4)
    spy, vixy = synth_day(day), synth_day(day, drift=0.002)
    prev = [{"spy": synth_day(d), "vixy": synth_day(d)} for d in five_days_before(day)]
    ctx = day_context(prev, sma50=10.0)
    assert ctx is not None
    t = pd.Timestamp("2024-03-04 11:00", tz=NY)
    base = observe(spy, vixy, ctx, t)
    # 1. no look-ahead: changing any bar that closes after t never changes the values at t
    moved = vixy.copy()
    moved.loc[moved.index >= t, ["open", "high", "low", "close"]] *= 5
    assert observe(spy, moved, ctx, t) == base
    # 2. vixy_hour at 09:50 is clipped at today's open (window 09:30 -> 09:50), never yesterday
    t950 = pd.Timestamp("2024-03-04 09:50", tz=NY)
    assert close_to(observe(spy, vixy, ctx, t950)["vixy_hour"], vixy.close[vixy.index < t950].iloc[-1] / vixy.open.iloc[0] - 1)
    # 3. rvol compares the same clock time: today's volume to 11:00 over the mean D-1..D-5 volume to 11:00
    assert close_to(base["rvol"], vixy.volume[vixy.index < t].sum() / np.mean([p["vixy"].volume[p["vixy"].index.time < t.time()].sum() for p in prev]))
    # 4. a thin minute: drop the 10:59 VIXY bar -> "now" is the 10:58 close
    thin = vixy.drop(pd.Timestamp("2024-03-04 10:59", tz=NY))
    assert close_to(observe(spy, thin, ctx, t)["vixy_day"], thin.close[thin.index < t].iloc[-1] / ctx["vixy_d1_close"] - 1)
    # 5. D-1 VWAP and a missing history
    d1 = prev[0]["vixy"]
    assert close_to(ctx["vixy_d1_vwap"], (d1.close * d1.volume).sum() / d1.volume.sum())
    assert day_context(prev[:4], 10.0) is None
    # 6. decision times: 37 normal, 19 on an early close, first 09:40
    assert len(decision_times(date(2024, 3, 4), False)) == 37 and len(decision_times(date(2024, 11, 29), True)) == 19
    assert decision_times(date(2024, 3, 4), False)[0] == pd.Timestamp("2024-03-04 09:40", tz=NY)
    # 7. a D-k session that ended early (12:59) is left out of the volume mean once t is past its end
    short = [{"spy": prev[0]["spy"][prev[0]["spy"].index.hour < 13], "vixy": prev[0]["vixy"][prev[0]["vixy"].index.hour < 13]}, *prev[1:]]
    ctx_short = day_context(short, 10.0)
    assert ctx_short is not None
    t_late = pd.Timestamp("2024-03-04 13:10", tz=NY)
    expected = vixy.volume[vixy.index < t_late].sum() / np.mean([p["vixy"].volume[p["vixy"].index.time < t_late.time()].sum() for p in prev[1:]])
    assert close_to(observe(spy, vixy, ctx_short, t_late)["rvol"], expected)
    print("self-check OK")


def main() -> None:
    """Entry point: `check`, `dev` or `test` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
