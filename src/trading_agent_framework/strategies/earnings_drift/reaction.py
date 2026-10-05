"""Reaction-day features from daily bars (spec §3.4). Pure; float64 on purpose (indicator maths on `Bars.df`).

Bars are matched by their market-time DATE, because live Alpaca stamps a daily bar at its open (midnight ET) and
every backtest source at its close (16:00 ET): both fall on the session's date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from trading_agent_framework.utils.clock import MARKET_TZ


@dataclass(frozen=True, slots=True)
class ReactionFeatures:
    gap_pct: float
    return_pct: float
    abnormal_pct: float
    hold_ratio: float | None  # None when the day's high is not above the previous close
    close_location: float
    rel_volume: float
    dollar_volume_20d: float
    runup_20d_pct: float | None
    runup_60d_pct: float | None
    atr14_pct: float | None
    close: float
    reaction_low: float


def bar_dates(frame: pd.DataFrame) -> list[date]:
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("daily bars must be indexed by tz-aware timestamps")
    return [stamp.date() for stamp in index.tz_convert(MARKET_TZ)]


def rows_through(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    """The rows on or before `day`, indexed by date (a duplicated date keeps its last row)."""
    dated = frame.copy()
    dated.index = pd.Index(bar_dates(frame))
    dated = dated[~dated.index.duplicated(keep="last")]
    return dated[[stamp <= day for stamp in dated.index]]


def _runup(closes: pd.Series, sessions: int) -> float | None:
    """The close before the reaction day against the close `sessions` sessions earlier; None without the history."""
    if len(closes) < sessions + 2:
        return None
    base = float(closes.iloc[-(sessions + 2)])
    return float(closes.iloc[-2]) / base - 1 if base > 0 else None


def _atr_pct(rows: pd.DataFrame, length: int = 14) -> float | None:
    if len(rows) < length + 1:
        return None
    window = rows.iloc[-(length + 1) :]
    previous = window["close"].shift(1)
    true_range = pd.concat([window["high"] - window["low"], (window["high"] - previous).abs(), (window["low"] - previous).abs()], axis=1).max(axis=1)
    close = float(window["close"].iloc[-1])
    return float(true_range.iloc[1:].mean()) / close if close > 0 else None


def reaction_features(stock: pd.DataFrame, benchmark: pd.DataFrame, day: date, *, baseline_sessions: int = 20) -> ReactionFeatures | None:
    """Features of the reaction session `day`; None without a bar for `day` (stock and benchmark) and the baseline before it."""
    rows = rows_through(stock, day)
    bench = rows_through(benchmark, day)
    if rows.empty or rows.index[-1] != day or len(rows) < baseline_sessions + 2:
        return None
    if len(bench) < 2 or bench.index[-1] != day:
        return None
    reaction, previous = rows.iloc[-1], rows.iloc[-2]
    prev_close = float(previous["close"])
    open_, high, low, close = (float(reaction[key]) for key in ("open", "high", "low", "close"))
    baseline = rows.iloc[-(baseline_sessions + 1) : -1]
    mean_volume = float(baseline["volume"].mean())
    bench_prev = float(bench["close"].iloc[-2])
    if prev_close <= 0 or mean_volume <= 0 or bench_prev <= 0:
        return None
    return_pct = close / prev_close - 1
    return ReactionFeatures(
        gap_pct=open_ / prev_close - 1,
        return_pct=return_pct,
        abnormal_pct=return_pct - (float(bench["close"].iloc[-1]) / bench_prev - 1),
        hold_ratio=(close - prev_close) / (high - prev_close) if high > prev_close else None,
        close_location=(close - low) / (high - low) if high > low else 0.5,
        rel_volume=float(reaction["volume"]) / mean_volume,
        dollar_volume_20d=float((baseline["close"] * baseline["volume"]).mean()),
        runup_20d_pct=_runup(rows["close"], 20),
        runup_60d_pct=_runup(rows["close"], 60),
        atr14_pct=_atr_pct(rows),
        close=close,
        reaction_low=low,
    )
