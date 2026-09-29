"""Pure candidate selection (spec §2): stage 1 on daily bars, stage 2 on intraday contexts."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import pandas as pd

from trading_agent_framework.strategies.vwap_pullback.features import BarContext, beta, daily_atr, zscores
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters

_MOMENTUM_SESSIONS = 20


@dataclass(frozen=True, slots=True)
class DailyProfile:
    symbol: str
    last_close: float
    daily_atr: float
    atr_pct: float
    dollar_volume: float  # 20-session average close x volume
    momentum: float  # 20-session return
    beta: float


@dataclass(frozen=True, slots=True)
class IntradaySnapshot:
    symbol: str
    ret: float
    rs: float
    rvol: float | None
    last_close: float
    vwap: float


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    symbol: str
    composite: float
    z_rs: float
    z_rvol: float


def daily_profile(symbol: str, daily: pd.DataFrame, bench_daily: pd.DataFrame | None, params: VwapPullbackParameters) -> DailyProfile | None:
    """One symbol's stage-1 measures from its daily bars; None with too little history or no usable price."""
    if len(daily) < max(_MOMENTUM_SESSIONS + 1, params.atr_length + 1):
        return None
    close = daily["close"].astype(float)
    last_close = float(close.iloc[-1])
    atr = daily_atr(daily, params.atr_length)
    if atr is None or not last_close > 0:
        return None
    dollar_volume = float((close * daily["volume"].astype(float)).tail(_MOMENTUM_SESSIONS).mean())
    momentum = last_close / float(close.iloc[-(_MOMENTUM_SESSIONS + 1)]) - 1
    stock_beta = beta(close, bench_daily["close"], params.beta_lookback_sessions) if bench_daily is not None and not bench_daily.empty else 1.0
    return DailyProfile(symbol=symbol, last_close=last_close, daily_atr=atr, atr_pct=atr / last_close, dollar_volume=dollar_volume, momentum=momentum, beta=stock_beta)


def select_stage1(profiles: Sequence[DailyProfile], params: VwapPullbackParameters) -> list[DailyProfile]:
    """Filter on price, ATR% band and dollar-volume percentile, then keep the top `stage1_size` by mean z(ATR%), z(momentum).

    The volume cut is a percentile of the whole profiled universe, not an absolute number: Alpaca's IEX
    volume is a small slice of the consolidated tape.
    """
    if not profiles:
        return []
    threshold = float(pd.Series([p.dollar_volume for p in profiles]).quantile(1 - params.dollar_volume_percentile))
    low, high = params.atr_pct_band
    eligible = [p for p in profiles if p.last_close >= params.min_price and low <= p.atr_pct <= high and p.dollar_volume >= threshold]
    z_atr = zscores({p.symbol: p.atr_pct for p in eligible})
    z_momentum = zscores({p.symbol: p.momentum for p in eligible})
    ranked = sorted(eligible, key=lambda p: (-(z_atr[p.symbol] + z_momentum[p.symbol]) / 2, p.symbol))
    return ranked[: params.stage1_size]


def snapshot_from(symbol: str, contexts: Sequence[BarContext]) -> IntradaySnapshot | None:
    """Stage-2 measures from the latest completed bar; None before the first one."""
    if not contexts:
        return None
    last = contexts[-1]
    return IntradaySnapshot(symbol=symbol, ret=last.close / last.session_open - 1, rs=last.rs, rvol=last.rvol, last_close=last.close, vwap=last.vwap)


def rank_stage2(snapshots: Sequence[IntradaySnapshot], params: VwapPullbackParameters, sticky: Collection[str]) -> list[RankedCandidate]:
    """The top `tracked_size` symbols passing the floor by composite z-score, then every `sticky` symbol not already in (by name).

    Floor: RVOL at least `rvol_min`, RS above 0, last close above VWAP. A symbol with no RVOL baseline
    cannot pass it. A sticky symbol (a setup already past WATCH) stays tracked whatever its rank; it gets
    zeros when it is not among the symbols passing the floor.
    """
    passing = [s for s in snapshots if s.rvol is not None and s.rvol >= params.rvol_min and s.rs > 0 and s.last_close > s.vwap]
    z_ret = zscores({s.symbol: s.ret for s in passing})
    z_rs = zscores({s.symbol: s.rs for s in passing})
    z_rvol = zscores({s.symbol: s.rvol for s in passing if s.rvol is not None})
    candidates = [
        RankedCandidate(symbol=s.symbol, composite=(z_ret[s.symbol] + z_rs[s.symbol] + z_rvol[s.symbol]) / 3, z_rs=z_rs[s.symbol], z_rvol=z_rvol[s.symbol])
        for s in passing
    ]
    ranked = sorted(candidates, key=lambda c: (-c.composite, c.symbol))[: params.tracked_size]
    kept = {c.symbol for c in ranked}
    by_symbol = {c.symbol: c for c in candidates}
    extras = [by_symbol.get(symbol, RankedCandidate(symbol=symbol, composite=0.0, z_rs=0.0, z_rvol=0.0)) for symbol in sorted(set(sticky) - kept)]
    return ranked + extras
