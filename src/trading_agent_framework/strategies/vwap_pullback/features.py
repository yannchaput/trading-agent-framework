"""Pure intraday features over minute bars (spec §2): session slicing, 5-minute contexts, VWAP, RVOL, RS, ATR, EMA.

float64 throughout: this is indicator maths on `Bars.df`, the codebase's float boundary. `bar_stamp` says how
the source stamps a minute bar: "open" (live Alpaca `Bars`) or "close" (every `BacktestDataSource`).

Every function here is pure (no I/O, no clock): callers pass `now` and the frames explicitly, so the same
code serves live trading and backtests and is unit-tested on synthetic bars.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd

# How a minute bar is timestamped by its source. Live Alpaca bars carry their OPEN time (the 09:30 bar
# covers 09:30-09:31); every backtest data source indexes a bar by its CLOSE (the same bar is stamped
# 09:31) so that "index <= cutoff" means "already knowable". Every function that maps a row to a minute
# of the session must know which convention it is reading.
BarStamp = Literal["open", "close"]
_MINUTE = pd.Timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class BarContext:
    """One completed intraday bar (5-minute by default) and the session state as of its close.

    This is the only view of price the state machine and the agents get: session-level measures (VWAP,
    RS, RVOL, running high) are frozen at the bar's close, so replaying bars later gives the same answer.
    """

    time: datetime  # the bar's close
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float  # session VWAP up to and including this bar
    rs: float  # session return minus beta x benchmark session return, at this close
    rvol: float | None  # cumulative volume over the baseline at this minute; None without a baseline
    session_open: float  # the session's first regular-hours open
    session_high: float  # the session's high up to this bar


@dataclass(frozen=True, slots=True)
class Levels:
    """The latest completed bar's close and the levels the exit review watches."""

    close: float
    vwap: float
    ema: float | None  # EMA of the 5-minute closes (None only without bars)
    atr: float | None  # ATR of the 5-minute bars, the unit of the exit agent's trailing distance


def minute_starts(index: pd.DatetimeIndex, bar_stamp: BarStamp) -> pd.DatetimeIndex:
    """When each minute bar started: live Alpaca bars are stamped at their open, backtest bars at their close."""
    return index if bar_stamp == "open" else index - _MINUTE


def session_slice(df: pd.DataFrame, session_open: datetime, session_close: datetime, bar_stamp: BarStamp) -> pd.DataFrame:
    """The regular-session rows of a minute frame: the bars that start in `[open, close)`.

    Comparing bar START times makes both stamp conventions agree: the pre-market bar that closes at 09:30
    starts at 09:29 and is dropped, the 15:59 bar is kept whatever it is stamped with.
    """
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
    # `.where(> 0)` turns a zero denominator into NaN instead of a division error; a NaN VWAP makes every
    # "close above/below VWAP" comparison False, so a bar with no volume yet cannot pass a VWAP gate.
    return (typical * volume).cumsum() / cumulative_volume.where(cumulative_volume > 0)


def cumulative_volume_by_minute(df: pd.DataFrame, session_open: datetime, bar_stamp: BarStamp) -> pd.Series:
    """Cumulative volume at every minute-of-session up to the last bar; a minute with no bar carries the previous total."""
    if df.empty:
        return pd.Series(dtype=float)
    minutes = minute_of_session(df, session_open, bar_stamp)
    cumulative = pd.Series(df["volume"].astype(float).cumsum().to_numpy(), index=minutes.to_numpy())
    cumulative = cumulative[~cumulative.index.duplicated(keep="last")]
    # Illiquid names skip minutes (no trade, no bar). Reindex to every minute and forward-fill so the
    # baseline has a value at each minute RVOL may be asked about.
    return cumulative.reindex(range(int(cumulative.index.max()) + 1)).ffill().fillna(0.0)


def rvol_baseline(prior_sessions: Sequence[pd.Series]) -> pd.Series:
    """Mean cumulative volume at each minute-of-session over prior sessions; a minute only averages the sessions that reached it (early closes)."""
    if not prior_sessions:
        return pd.Series(dtype=float)
    # Sessions are aligned on minute-of-session; a shorter (early-close) session is NaN past its close and
    # `skipna` leaves it out of those minutes' mean instead of dragging the average towards zero.
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
    buckets = mos // minutes  # bucket 0 = minutes 0-4 of the session, bucket 1 = 5-9, ...
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
        # Completeness gate (plan deviation 7). Stop at the first bucket that is still forming, or that
        # closed less than a minute ago without its last minute bar: live data may simply not have
        # arrived yet. Later buckets are necessarily not complete either, hence `break`.
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
                # Session-level measures are read at the bucket's LAST row, i.e. as of the bar's close.
                vwap=float(vwaps.loc[label]),
                # Beta-adjusted relative strength: the stock's own move once the market's share of it is removed.
                rs=(close / open_price - 1) - beta * bench_return,
                # RVOL is measured at the bucket's last minute even if that minute had no trade, so a gap
                # in today's data does not shift which baseline minute it is compared with.
                rvol=rvol_at(float(cumulative.loc[label]), last_minute, baseline),
                session_open=open_price,
                session_high=session_high,
            )
        )
    return contexts


def _benchmark_closes(benchmark_df: pd.DataFrame, session_open: datetime, bar_stamp: BarStamp, minutes: int) -> tuple[pd.Series, float | None]:
    """The benchmark's last close per bucket and its session open; `(empty, None)` without benchmark data."""
    if benchmark_df.empty:
        return pd.Series(dtype=float), None
    buckets = minute_of_session(benchmark_df, session_open, bar_stamp) // minutes
    closes = benchmark_df["close"].astype(float).groupby(buckets.to_numpy()).last()
    return closes, float(benchmark_df["open"].iloc[0])


def _benchmark_return(closes: pd.Series, bench_open: float | None, bucket: int) -> float:
    """The benchmark's session return as of `bucket`'s close.

    Uses the latest benchmark bucket at or before `bucket` (never a later one: that would be look-ahead).
    Without benchmark data the return is 0, so RS degrades to the stock's plain session return.
    """
    if not bench_open:
        return 0.0
    upto = closes[closes.index <= bucket]
    return 0.0 if upto.empty else float(upto.iloc[-1]) / bench_open - 1


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Per-bar true range: the widest of high-low and the gaps from the previous close."""
    previous = close.shift(1)
    return pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)


def daily_atr(df: pd.DataFrame, length: int) -> float | None:
    """Mean true range of the last `length` daily bars; None with fewer than `length + 1` bars."""
    if len(df) < length + 1:  # the first bar has no previous close, so `length` true ranges need `length + 1` bars
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
    # Inner join on dates: a day missing from either series (halt, holiday mismatch) is dropped from both.
    joined = pd.concat([stock_closes.astype(float), bench_closes.astype(float)], axis=1, join="inner").dropna()
    returns = joined.pct_change().dropna().iloc[-lookback:]
    # Fewer than 10 paired returns (or a flat benchmark) says nothing reliable: assume market-like beta 1.0.
    if len(returns) < 10:
        return 1.0
    variance = returns.iloc[:, 1].var()
    if not variance > 0:
        return 1.0
    return float(returns.iloc[:, 0].cov(returns.iloc[:, 1]) / variance)


def zscores(values: Mapping[str, float]) -> dict[str, float]:
    """Cross-sectional z-scores (population std); all 0.0 when there is no spread to measure."""
    # A single symbol, or identical values, cannot be ranked: every z-score is 0 so the feature neither
    # helps nor hurts the composite, instead of dividing by a zero standard deviation.
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
