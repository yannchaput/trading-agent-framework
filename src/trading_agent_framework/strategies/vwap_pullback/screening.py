"""Pure candidate selection (spec §2): stage 1 on daily bars, stage 2 on intraday contexts.

Stage 1 runs once per session before the open and narrows the ~1,200-symbol universe to the names that
are liquid, volatile enough and trending (`select_stage1`). Stage 2 runs every tick on today's 5-minute
bars and keeps the few names showing abnormal, market-independent strength right now (`rank_stage2`).
Both are pure: `Scanner` fetches the data and applies the results.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import pandas as pd

from trading_agent_framework.strategies.vwap_pullback.features import BarContext, beta, daily_atr, zscores
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters

_MOMENTUM_SESSIONS = 20  # window for both the momentum return and the average dollar volume


@dataclass(frozen=True, slots=True)
class DailyProfile:
    """One symbol's stage-1 measures, from its daily bars up to yesterday's close."""

    symbol: str
    last_close: float
    daily_atr: float
    atr_pct: float  # daily ATR / last close: how much the stock typically moves in a day
    dollar_volume: float  # 20-session average close x volume
    momentum: float  # 20-session return
    beta: float  # to the benchmark (SPY); 1.0 when it cannot be estimated


@dataclass(frozen=True, slots=True)
class IntradaySnapshot:
    """One symbol's stage-2 measures, as of its latest completed 5-minute bar."""

    symbol: str
    ret: float  # session return: last close / session open - 1
    rs: float  # beta-adjusted relative strength vs the benchmark
    rvol: float | None  # relative volume; None without a baseline (fails the floor)
    last_close: float
    vwap: float


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """A symbol tracked this tick, with the scores the entry agent later reads."""

    symbol: str
    composite: float  # mean of z(ret), z(rs), z(rvol) across the symbols passing the floor
    z_rs: float
    z_rvol: float
    on_floor: bool = True  # False for a sticky symbol below the floor: its scores are placeholders, not measurements


def daily_profile(symbol: str, daily: pd.DataFrame, bench_daily: pd.DataFrame | None, params: VwapPullbackParameters) -> DailyProfile | None:
    """One symbol's stage-1 measures from its daily bars; None with too little history or no usable price."""
    # Need enough bars for both the 20-session momentum (21 closes) and the ATR (length + 1 bars).
    if len(daily) < max(_MOMENTUM_SESSIONS + 1, params.atr_length + 1):
        return None
    close = daily["close"].astype(float)
    last_close = float(close.iloc[-1])
    atr = daily_atr(daily, params.atr_length)
    if atr is None or not last_close > 0:
        return None
    dollar_volume = float((close * daily["volume"].astype(float)).tail(_MOMENTUM_SESSIONS).mean())
    momentum = last_close / float(close.iloc[-(_MOMENTUM_SESSIONS + 1)]) - 1
    # Without benchmark data every symbol gets beta 1.0, i.e. RS becomes "return minus the market's return".
    stock_beta = beta(close, bench_daily["close"], params.beta_lookback_sessions) if bench_daily is not None and not bench_daily.empty else 1.0
    return DailyProfile(symbol=symbol, last_close=last_close, daily_atr=atr, atr_pct=atr / last_close, dollar_volume=dollar_volume, momentum=momentum, beta=stock_beta)


def select_stage1(profiles: Sequence[DailyProfile], params: VwapPullbackParameters) -> list[DailyProfile]:
    """Filter on price, ATR% band and dollar-volume percentile, then keep the top `stage1_size` by mean z(ATR%), z(momentum).

    The volume cut is a percentile of the whole profiled universe, not an absolute number: Alpaca's IEX
    volume is a small slice of the consolidated tape.
    """
    if not profiles:
        return []
    # "Top 60% by dollar volume" == at or above the 40th percentile of the whole universe.
    threshold = float(pd.Series([p.dollar_volume for p in profiles]).quantile(1 - params.dollar_volume_percentile))
    low, high = params.atr_pct_band
    eligible = [p for p in profiles if p.last_close >= params.min_price and low <= p.atr_pct <= high and p.dollar_volume >= threshold]
    # Rank on volatility and trend together; z-scores are taken among the eligible names only.
    z_atr = zscores({p.symbol: p.atr_pct for p in eligible})
    z_momentum = zscores({p.symbol: p.momentum for p in eligible})
    # Symbol as the tie-breaker keeps the selection deterministic from run to run.
    ranked = sorted(eligible, key=lambda p: (-(z_atr[p.symbol] + z_momentum[p.symbol]) / 2, p.symbol))
    return ranked[: params.stage1_size]


def snapshot_from(symbol: str, contexts: Sequence[BarContext]) -> IntradaySnapshot | None:
    """Stage-2 measures from the latest completed bar; None before the first one."""
    if not contexts:
        return None
    last = contexts[-1]
    return IntradaySnapshot(symbol=symbol, ret=last.close / last.session_open - 1, rs=last.rs, rvol=last.rvol, last_close=last.close, vwap=last.vwap)


def stage2_funnel(snapshots: Sequence[IntradaySnapshot], ranked: Sequence[RankedCandidate], params: VwapPullbackParameters) -> str:
    """One log line saying where the stage-2 candidates are lost: each floor condition, the floor itself, then the tracked set.

    The three floor counts overlap (a symbol can fail several), so they do not add up to `floor_fail`.
    """
    no_rvol = sum(1 for s in snapshots if s.rvol is None or s.rvol < params.rvol_min)
    no_rs = sum(1 for s in snapshots if s.rs <= 0)
    below_vwap = sum(1 for s in snapshots if s.last_close <= s.vwap)
    passing = sum(1 for s in snapshots if s.rvol is not None and s.rvol >= params.rvol_min and s.rs > 0 and s.last_close > s.vwap)
    return (
        f"stage 2: {len(snapshots)} with bars | rvol<{params.rvol_min:g}: {no_rvol}, rs<=0: {no_rs}, below vwap: {below_vwap} "
        f"| pass floor: {passing} | tracked: {len(ranked)}"
    )


def rank_stage2(snapshots: Sequence[IntradaySnapshot], params: VwapPullbackParameters, sticky: Collection[str]) -> list[RankedCandidate]:
    """The top `tracked_size` symbols passing the floor by composite z-score, then every `sticky` symbol not already in (by name).

    Floor: RVOL at least `rvol_min`, RS above 0, last close above VWAP. A symbol with no RVOL baseline
    cannot pass it. A sticky symbol (a setup already past WATCH) stays tracked whatever its rank; it is
    returned with `on_floor=False` (placeholder zeros) when it is not among the symbols passing the floor.
    """
    # Hard floor: unusual volume, outperforming the market, and holding above VWAP.
    passing = [s for s in snapshots if s.rvol is not None and s.rvol >= params.rvol_min and s.rs > 0 and s.last_close > s.vwap]
    z_ret = zscores({s.symbol: s.ret for s in passing})
    z_rs = zscores({s.symbol: s.rs for s in passing})
    z_rvol = zscores({s.symbol: s.rvol for s in passing if s.rvol is not None})
    candidates = [
        RankedCandidate(symbol=s.symbol, composite=(z_ret[s.symbol] + z_rs[s.symbol] + z_rvol[s.symbol]) / 3, z_rs=z_rs[s.symbol], z_rvol=z_rvol[s.symbol])
        for s in passing
    ]
    ranked = sorted(candidates, key=lambda c: (-c.composite, c.symbol))[: params.tracked_size]
    # Sticky symbols: a setup that is already impulsing, pulling back or in a trade must keep being
    # tracked even if its ranking fades, otherwise its pullback (the whole point) would never be seen.
    kept = {c.symbol for c in ranked}
    by_symbol = {c.symbol: c for c in candidates}
    extras = [by_symbol.get(symbol, RankedCandidate(symbol=symbol, composite=0.0, z_rs=0.0, z_rvol=0.0, on_floor=False)) for symbol in sorted(set(sticky) - kept)]
    return ranked + extras
