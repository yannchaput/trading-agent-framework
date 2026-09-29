"""Pure intraday features over minute bars (spec §2): session slicing, 5-minute contexts, VWAP, RVOL, RS, ATR, EMA.

float64 throughout: this is indicator maths on `Bars.df`, the codebase's float boundary. `bar_stamp` says how
the source stamps a minute bar: "open" (live Alpaca `Bars`) or "close" (every `BacktestDataSource`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd

BarStamp = Literal["open", "close"]
_MINUTE = pd.Timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class BarContext:
    """One completed intraday bar (5-minute by default) and the session state as of its close."""

    time: datetime  # the bar's close
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    rs: float  # session return minus beta x benchmark session return, at this close
    rvol: float | None  # cumulative volume over the baseline at this minute; None without a baseline
    session_open: float  # the session's first regular-hours open
    session_high: float  # the session's high up to this bar


@dataclass(frozen=True, slots=True)
class Levels:
    """The latest completed bar's close and the levels the exit review watches."""

    close: float
    vwap: float
    ema: float | None
    atr: float | None


def minute_starts(index: pd.DatetimeIndex, bar_stamp: BarStamp) -> pd.DatetimeIndex:
    """When each minute bar started: live Alpaca bars are stamped at their open, backtest bars at their close."""
    return index if bar_stamp == "open" else index - _MINUTE


def session_slice(df: pd.DataFrame, session_open: datetime, session_close: datetime, bar_stamp: BarStamp) -> pd.DataFrame:
    """The regular-session rows of a minute frame: the bars that start in `[open, close)`."""
    starts = minute_starts(df.index, bar_stamp)
    return df[(starts >= session_open) & (starts < session_close)]


def minute_of_session(df: pd.DataFrame, session_open: datetime, bar_stamp: BarStamp) -> pd.Series:
    """Minutes since the open at which each row's bar started (0 = the opening minute)."""
    starts = minute_starts(df.index, bar_stamp)
    return pd.Series(((starts - session_open) // _MINUTE).astype(int), index=df.index)


def vwap_series(df: pd.DataFrame) -> pd.Series:
    """Session VWAP after each row: cumulative typical price x volume over cumulative volume (NaN before any volume)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].astype(float)
    cumulative_volume = volume.cumsum()
    return (typical * volume).cumsum() / cumulative_volume.where(cumulative_volume > 0)


def cumulative_volume_by_minute(df: pd.DataFrame, session_open: datetime, bar_stamp: BarStamp) -> pd.Series:
    """Cumulative volume at every minute-of-session up to the last bar; a minute with no bar carries the previous total."""
    if df.empty:
        return pd.Series(dtype=float)
    minutes = minute_of_session(df, session_open, bar_stamp)
    cumulative = pd.Series(df["volume"].astype(float).cumsum().to_numpy(), index=minutes.to_numpy())
    cumulative = cumulative[~cumulative.index.duplicated(keep="last")]
    return cumulative.reindex(range(int(cumulative.index.max()) + 1)).ffill().fillna(0.0)


def rvol_baseline(prior_sessions: Sequence[pd.Series]) -> pd.Series:
    """Mean cumulative volume at each minute-of-session over prior sessions; a minute only averages the sessions that reached it (early closes)."""
    if not prior_sessions:
        return pd.Series(dtype=float)
    return pd.concat(list(prior_sessions), axis=1).mean(axis=1, skipna=True).sort_index()


def rvol_at(cumulative_volume: float, minute: int, baseline: pd.Series) -> float | None:
    """Today's cumulative volume over the baseline's at the same minute; None when the baseline has nothing there."""
    expected = baseline.get(minute)
    if expected is None or not expected > 0:  # NaN > 0 is False
        return None
    return float(cumulative_volume) / float(expected)


def intraday_contexts(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    session_open: datetime,
    now: datetime,
    bar_stamp: BarStamp,
    beta: float,
    baseline: pd.Series,
    minutes: int = 5,
) -> list[BarContext]:
    """Completed `minutes`-minute bars of one session, oldest first, each with VWAP, RS and RVOL at its close.

    `df`/`benchmark_df` are already `session_slice`d. A bucket counts as complete once its close time has
    passed AND either its last minute is in the data or a full minute has gone by since it closed: live
    minute bars land a few seconds late, and a bucket read without its last minute would pass for a
    finished bar. A bucket with no trade at all is simply absent.
    """
    if df.empty:
        return []
    mos = minute_of_session(df, session_open, bar_stamp)
    buckets = mos // minutes
    vwaps = vwap_series(df)
    cumulative = df["volume"].astype(float).cumsum()
    open_price = float(df["open"].iloc[0])
    bench_closes, bench_open = _benchmark_closes(benchmark_df, session_open, bar_stamp, minutes)
    contexts: list[BarContext] = []
    session_high = -math.inf
    for bucket_value, rows in df.groupby(buckets.to_numpy(), sort=True):
        bucket = int(bucket_value)
        close_time = session_open + timedelta(minutes=(bucket + 1) * minutes)
        last_minute = (bucket + 1) * minutes - 1
        label = rows.index[-1]
        has_last_minute = int(mos.loc[label]) == last_minute
        if close_time > now or (not has_last_minute and now - close_time < timedelta(minutes=1)):
            break
        high = float(rows["high"].max())
        session_high = max(session_high, high)
        close = float(rows["close"].iloc[-1])
        bench_return = _benchmark_return(bench_closes, bench_open, bucket)
        contexts.append(
            BarContext(
                time=close_time,
                open=float(rows["open"].iloc[0]),
                high=high,
                low=float(rows["low"].min()),
                close=close,
                volume=float(rows["volume"].sum()),
                vwap=float(vwaps.loc[label]),
                rs=(close / open_price - 1) - beta * bench_return,
                rvol=rvol_at(float(cumulative.loc[label]), last_minute, baseline),
                session_open=open_price,
                session_high=session_high,
            )
        )
    return contexts


def _benchmark_closes(benchmark_df: pd.DataFrame, session_open: datetime, bar_stamp: BarStamp, minutes: int) -> tuple[pd.Series, float | None]:
    if benchmark_df.empty:
        return pd.Series(dtype=float), None
    buckets = minute_of_session(benchmark_df, session_open, bar_stamp) // minutes
    closes = benchmark_df["close"].astype(float).groupby(buckets.to_numpy()).last()
    return closes, float(benchmark_df["open"].iloc[0])


def _benchmark_return(closes: pd.Series, bench_open: float | None, bucket: int) -> float:
    if not bench_open:
        return 0.0
    upto = closes[closes.index <= bucket]
    return 0.0 if upto.empty else float(upto.iloc[-1]) / bench_open - 1


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous = close.shift(1)
    return pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)


def daily_atr(df: pd.DataFrame, length: int) -> float | None:
    """Mean true range of the last `length` daily bars; None with fewer than `length + 1` bars."""
    if len(df) < length + 1:
        return None
    tr = _true_range(df["high"].astype(float), df["low"].astype(float), df["close"].astype(float))
    return float(tr.iloc[-length:].mean())


def bar_atr(contexts: Sequence[BarContext], length: int) -> float | None:
    """Mean true range of the last `length` intraday bars (fewer early in the session); None without bars."""
    if not contexts:
        return None
    frame = pd.DataFrame({"high": [c.high for c in contexts], "low": [c.low for c in contexts], "close": [c.close for c in contexts]})
    return float(_true_range(frame["high"], frame["low"], frame["close"]).iloc[-length:].mean())


def ema_last(values: Sequence[float], length: int) -> float | None:
    """The last value of an exponential moving average with span `length`; None without values."""
    if not values:
        return None
    return float(pd.Series(list(values), dtype=float).ewm(span=length, adjust=False).mean().iloc[-1])


def beta(stock_closes: pd.Series, bench_closes: pd.Series, lookback: int) -> float:
    """Beta of daily returns over the last `lookback` returns; 1.0 when there is too little data to say."""
    joined = pd.concat([stock_closes.astype(float), bench_closes.astype(float)], axis=1, join="inner").dropna()
    returns = joined.pct_change().dropna().iloc[-lookback:]
    if len(returns) < 10:
        return 1.0
    variance = returns.iloc[:, 1].var()
    if not variance > 0:
        return 1.0
    return float(returns.iloc[:, 0].cov(returns.iloc[:, 1]) / variance)


def zscores(values: Mapping[str, float]) -> dict[str, float]:
    """Cross-sectional z-scores (population std); all 0.0 when there is no spread to measure."""
    if len(values) < 2:
        return dict.fromkeys(values, 0.0)
    series = pd.Series(dict(values), dtype=float)
    std = series.std(ddof=0)
    if not std > 0:
        return dict.fromkeys(values, 0.0)
    return {str(key): float(value) for key, value in ((series - series.mean()) / std).items()}


def latest_levels(contexts: Sequence[BarContext], *, ema_length: int, atr_length: int) -> Levels | None:
    """The latest bar's close and VWAP, the EMA of the closes and the bar ATR; None without bars."""
    if not contexts:
        return None
    last = contexts[-1]
    return Levels(close=last.close, vwap=last.vwap, ema=ema_last([c.close for c in contexts], ema_length), atr=bar_atr(contexts, atr_length))
