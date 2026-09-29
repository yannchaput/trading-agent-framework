from __future__ import annotations

import functools
from datetime import timedelta

import pytest
from tests.fakes import et

from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState, advance, back_to_pullback, health, mark_in_trade, step

PARAMS = VwapPullbackParameters()
ATR = 2.0  # daily ATR: impulse = 1.6 move, large red body > 0.5


def bar(minute: int, o: float, h: float, low: float, c: float, v: float, *, vwap: float, rs: float = 0.01, rvol: float = 2.0, high_so_far: float | None = None) -> BarContext:
    return BarContext(
        time=et(2026, 9, 1, 9, 30) + timedelta(minutes=minute), open=o, high=h, low=low, close=c, volume=v,
        vwap=vwap, rs=rs, rvol=rvol, session_open=100.0, session_high=high_so_far if high_so_far is not None else h,
    )


B1 = bar(5, 100, 101, 99.9, 100.9, 1000, vwap=100.3)  # move 0.5 ATR: still WATCH
B2 = bar(10, 100.9, 101.8, 100.8, 101.7, 1000, vwap=100.8)  # move 0.9 ATR: IMPULSE
B3 = bar(15, 101.7, 101.75, 101.2, 101.3, 500, vwap=101.0, high_so_far=101.8)  # 28% retrace: PULLBACK
B4 = bar(20, 101.3, 101.4, 101.1, 101.25, 400, vwap=101.05, high_so_far=101.8)  # still pulling back
B5 = bar(25, 101.25, 101.9, 101.2, 101.85, 800, vwap=101.1, high_so_far=101.9)  # resumption: TRIGGERED


def _run(*bars: BarContext) -> Setup:
    return advance(Setup(symbol="AAA"), list(bars), ATR, PARAMS)


def test_a_healthy_path_goes_watch_impulse_pullback_triggered() -> None:
    assert _run(B1).state is SetupState.WATCH
    impulse = _run(B1, B2)
    assert impulse.state is SetupState.IMPULSE
    assert impulse.impulse_high == 101.8
    assert impulse.impulse_avg_volume == 1000
    pullback = _run(B1, B2, B3, B4)
    assert pullback.state is SetupState.PULLBACK
    assert pullback.pullback_low == 101.1
    assert pullback.pullback_avg_volume == 450
    triggered = _run(B1, B2, B3, B4, B5)
    assert triggered.state is SetupState.TRIGGERED
    assert triggered.trigger_close == 101.85
    assert triggered.triggered_at == B5.time


def test_health_flags() -> None:
    flags = health(_run(B1, B2, B3, B4))
    assert flags["state"] == "pullback"
    assert flags["vol_ratio"] == 0.45
    assert flags["duration_ratio"] == 1.0
    assert flags["retracement_pct"] == pytest.approx(30.6, abs=0.1)
    assert flags["above_vwap"] is True


def test_health_reports_the_pullback_depth_not_the_shallower_retracement_after_the_bounce() -> None:
    triggered = _run(B1, B2, B3, B4, B5)
    assert triggered.state is SetupState.TRIGGERED
    assert triggered.retracement < triggered.max_retracement  # the trigger bar closed higher than the pullback's low close
    assert health(triggered)["retracement_pct"] == pytest.approx(30.6, abs=0.1)


@pytest.mark.parametrize(
    ("last_bar", "reason"),
    [
        (bar(20, 101.3, 101.35, 100.9, 100.95, 400, vwap=101.0), "close below VWAP"),
        (bar(20, 101.3, 101.4, 101.1, 101.25, 400, vwap=101.05, rs=-0.001), "relative strength lost"),
        (bar(20, 100.8, 100.85, 100.6, 100.65, 400, vwap=100.5), "retraced more than 61.8% of the impulse"),
        (bar(20, 101.4, 101.45, 100.8, 100.85, 400, vwap=100.5), "large bearish candle"),
        (bar(20, 101.3, 101.35, 101.1, 101.2, 1600, vwap=101.0), "heavy selling volume"),
    ],
)
def test_each_broken_reason(last_bar: BarContext, reason: str) -> None:
    broken = _run(B1, B2, B3, last_bar)
    assert broken.state is SetupState.BROKEN
    assert broken.broken_reason == reason


def test_a_pullback_longer_than_the_impulse_breaks() -> None:
    broken = _run(B1, B2, B3, B4, bar(25, 101.25, 101.3, 101.15, 101.2, 300, vwap=101.05))
    assert broken.state is SetupState.BROKEN
    assert broken.broken_reason == "pullback lasted longer than the impulse"


def test_a_new_high_without_trigger_volume_extends_the_impulse() -> None:
    extended = _run(B1, B2, B3, B4, bar(25, 101.25, 102.0, 101.2, 101.95, 300, vwap=101.1, high_so_far=102.0))
    assert extended.state is SetupState.IMPULSE
    assert extended.impulse_high == 102.0
    assert extended.pullback_low is None


def test_a_trigger_is_stale_after_one_more_bar() -> None:
    later = _run(B1, B2, B3, B4, B5, bar(30, 101.85, 101.9, 101.5, 101.6, 300, vwap=101.15, high_so_far=101.9))
    assert later.state is not SetupState.TRIGGERED  # this bar makes a new high without trigger volume: back to IMPULSE
    assert later.trigger_close is None and later.triggered_at is None


def test_replaying_all_bars_at_once_equals_one_bar_per_tick() -> None:
    bars = [B1, B2, B3, B4, B5]
    one_by_one = functools.reduce(lambda setup, b: advance(setup, [b], ATR, PARAMS), bars, Setup(symbol="AAA"))
    at_once = _run(*bars)
    assert at_once == one_by_one
    assert advance(at_once, bars, ATR, PARAMS) == at_once  # bars already seen are skipped


def test_in_trade_ignores_price_action_and_back_to_pullback_clears_the_trigger() -> None:
    in_trade = mark_in_trade(_run(B1, B2, B3, B4, B5))
    after = step(in_trade, bar(30, 101.8, 101.9, 99.0, 99.5, 5000, vwap=101.0), ATR, PARAMS)
    assert after.state is SetupState.IN_TRADE
    reverted = back_to_pullback(in_trade)
    assert reverted.state is SetupState.PULLBACK and reverted.trigger_close is None
