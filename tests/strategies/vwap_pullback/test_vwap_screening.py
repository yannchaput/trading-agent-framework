from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from tests.fakes import et, make_bars_frame

from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.screening import (
    DailyProfile,
    IntradaySnapshot,
    daily_profile,
    rank_stage2,
    select_stage1,
    snapshot_from,
    stage2_funnel,
)

PARAMS = VwapPullbackParameters()


def _daily(closes: list[float]) -> pd.DataFrame:
    return make_bars_frame(closes, start=et(2026, 6, 1))


def test_daily_profile_measures_atr_momentum_and_dollar_volume() -> None:
    closes = [100.0] * 50 + [100.0 + i for i in range(21)]  # +20% over the last 20 sessions
    profile = daily_profile("AAA", _daily(closes), _daily([400.0] * 71), PARAMS)
    assert profile is not None
    assert profile.last_close == 120.0
    assert profile.momentum == pytest.approx(120 / 100 - 1)
    assert profile.daily_atr == pytest.approx(2.0, abs=0.1)
    assert profile.dollar_volume == pytest.approx(sum(closes[-20:]) / 20 * 1000)


def test_daily_profile_needs_enough_history() -> None:
    assert daily_profile("AAA", _daily([100.0] * 10), None, PARAMS) is None


def _profile(symbol: str, *, close: float = 100.0, atr_pct: float = 0.02, dollar_volume: float = 1e6, momentum: float = 0.0) -> DailyProfile:
    return DailyProfile(symbol=symbol, last_close=close, daily_atr=close * atr_pct, atr_pct=atr_pct, dollar_volume=dollar_volume, momentum=momentum, beta=1.0)


def test_select_stage1_filters_price_atr_band_and_volume_percentile_then_ranks() -> None:
    profiles = [
        _profile("CHEAP", close=3.0),
        _profile("WILD", atr_pct=0.10),
        _profile("CALM", atr_pct=0.01),
        _profile("THIN", dollar_volume=1.0),
        _profile("FAST", atr_pct=0.05, momentum=0.2),
        _profile("SLOW", atr_pct=0.02, momentum=0.0),
    ]
    chosen = select_stage1(profiles, PARAMS)
    assert [p.symbol for p in chosen] == ["FAST", "SLOW"]
    assert [p.symbol for p in select_stage1(profiles, dataclasses.replace(PARAMS, stage1_size=1))] == ["FAST"]


def _snapshot(symbol: str, *, ret: float = 0.02, rs: float = 0.01, rvol: float | None = 2.0, above_vwap: bool = True) -> IntradaySnapshot:
    return IntradaySnapshot(symbol=symbol, ret=ret, rs=rs, rvol=rvol, last_close=101.0, vwap=100.0 if above_vwap else 102.0)


def test_rank_stage2_applies_the_floor_and_keeps_the_top_n() -> None:
    snapshots = [
        _snapshot("LOWVOL", rvol=1.2),
        _snapshot("WEAK", rs=-0.01),
        _snapshot("UNDER", above_vwap=False),
        _snapshot("A", ret=0.05, rs=0.04, rvol=4.0),
        _snapshot("B", ret=0.02, rs=0.01, rvol=2.0),
        _snapshot("C", ret=0.01, rs=0.005, rvol=1.6),
    ]
    ranked = rank_stage2(snapshots, dataclasses.replace(PARAMS, tracked_size=2), sticky=set())
    assert [c.symbol for c in ranked] == ["A", "B"]
    assert ranked[0].z_rs > 0 and ranked[0].z_rvol > 0


def test_rank_stage2_keeps_sticky_symbols_even_when_they_fail_the_floor() -> None:
    ranked = rank_stage2([_snapshot("A"), _snapshot("HELD", rs=-0.01)], PARAMS, sticky={"HELD", "GONE"})
    assert [c.symbol for c in ranked] == ["A", "GONE", "HELD"]
    assert ranked[2].composite == 0.0
    assert [c.on_floor for c in ranked] == [True, False, False]  # placeholders must not overwrite measured scores


def test_rank_stage2_excludes_missing_rvol() -> None:
    assert rank_stage2([_snapshot("NEW", rvol=None)], PARAMS, sticky=set()) == []


def test_snapshot_from_uses_the_latest_context() -> None:
    context = BarContext(time=et(2026, 9, 1, 9, 35), open=100, high=101, low=99.5, close=101, volume=1000, vwap=100.5, rs=0.008, rvol=2.5, session_open=100, session_high=101)
    snapshot = snapshot_from("AAA", [context])
    assert snapshot == IntradaySnapshot(symbol="AAA", ret=pytest.approx(0.01), rs=0.008, rvol=2.5, last_close=101, vwap=100.5)
    assert snapshot_from("AAA", []) is None


def test_stage2_funnel_counts_each_floor_condition_and_the_survivors() -> None:
    def snap(symbol: str, *, rvol: float | None, rs: float, close: float, vwap: float = 10.0) -> IntradaySnapshot:
        return IntradaySnapshot(symbol=symbol, ret=0.01, rs=rs, rvol=rvol, last_close=close, vwap=vwap)

    snapshots = [
        snap("OK", rvol=2.0, rs=0.01, close=11.0),
        snap("LOWVOL", rvol=1.0, rs=0.01, close=11.0),
        snap("NOBASE", rvol=None, rs=0.01, close=11.0),
        snap("WEAK", rvol=2.0, rs=-0.01, close=9.0),  # fails RS and VWAP, not RVOL
    ]
    line = stage2_funnel(snapshots, rank_stage2(snapshots, PARAMS, sticky=()), PARAMS)
    assert line == "stage 2: 4 with bars | rvol<1.5: 2, rs<=0: 1, below vwap: 1 | pass floor: 1 | tracked: 1"
