"""Pure per-candidate state machine (spec §3), advanced on completed 5-minute bars.

WATCH -> IMPULSE -> PULLBACK -> TRIGGERED -> IN_TRADE -> DONE, or BROKEN (terminal for the session).
`advance` replays every bar newer than the last one seen, so a tick that finds several new bars gives
exactly the result of one bar per tick. A trigger lasts one bar: the next bar re-evaluates the setup as
a pullback (it may trigger again).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters


class SetupState(StrEnum):
    WATCH = "watch"
    IMPULSE = "impulse"
    PULLBACK = "pullback"
    TRIGGERED = "triggered"
    IN_TRADE = "in_trade"
    DONE = "done"
    BROKEN = "broken"


_FROZEN = frozenset({SetupState.IN_TRADE, SetupState.DONE, SetupState.BROKEN})


@dataclass(frozen=True, slots=True)
class Setup:
    symbol: str
    state: SetupState = SetupState.WATCH
    last_bar_time: datetime | None = None
    bars_seen: int = 0
    volume_total: float = 0.0
    impulse_high: float | None = None
    impulse_bars: int = 0  # bars from the open up to the impulse high
    impulse_volume_total: float = 0.0  # their volume
    pullback_low: float | None = None
    pullback_bars: int = 0
    pullback_volume_total: float = 0.0
    prev_high: float | None = None
    trigger_close: float | None = None
    triggered_at: datetime | None = None
    retracement: float = 0.0  # of the impulse leg (session open -> impulse high), at the last close
    largest_red_body_atr: float = 0.0
    last_close: float | None = None
    last_vwap: float | None = None
    last_rs: float | None = None
    last_rvol: float | None = None
    broken_reason: str | None = None

    @property
    def impulse_avg_volume(self) -> float:
        return self.impulse_volume_total / self.impulse_bars if self.impulse_bars else 0.0

    @property
    def pullback_avg_volume(self) -> float:
        return self.pullback_volume_total / self.pullback_bars if self.pullback_bars else 0.0


def advance(setup: Setup, bars: Sequence[BarContext], daily_atr: float, params: VwapPullbackParameters) -> Setup:
    """Replay every bar newer than `setup.last_bar_time`, oldest first."""
    for bar in bars:
        if setup.last_bar_time is None or bar.time > setup.last_bar_time:
            setup = step(setup, bar, daily_atr, params)
    return setup


def step(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    """The setup after one more completed bar."""
    seen = replace(
        setup, last_bar_time=bar.time, bars_seen=setup.bars_seen + 1, volume_total=setup.volume_total + bar.volume,
        last_close=bar.close, last_vwap=bar.vwap, last_rs=bar.rs, last_rvol=bar.rvol,
    )
    if seen.state in _FROZEN:
        return seen
    if seen.state is SetupState.TRIGGERED:  # the trigger was the previous bar: re-evaluate as a pullback
        seen = back_to_pullback(seen)
    if seen.state is SetupState.WATCH:
        return replace(_watch(seen, bar, daily_atr, params), prev_high=bar.high)
    return replace(_impulse_or_pullback(seen, bar, daily_atr, params), prev_high=bar.high)


def _watch(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    move_atr = (bar.session_high - bar.session_open) / daily_atr if daily_atr > 0 else 0.0
    if move_atr >= params.impulse_move_atr and bar.rvol is not None and bar.rvol >= params.rvol_min and bar.rs > 0 and bar.close > bar.vwap:
        return replace(setup, state=SetupState.IMPULSE, impulse_high=bar.session_high, impulse_bars=setup.bars_seen, impulse_volume_total=setup.volume_total)
    return setup


def _impulse_or_pullback(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    assert setup.impulse_high is not None  # set on entering IMPULSE
    leg = setup.impulse_high - bar.session_open
    retracement = (setup.impulse_high - bar.close) / leg if leg > 0 else 0.0
    setup = replace(setup, retracement=max(retracement, 0.0))
    red_body = bar.open - bar.close
    reason = _broken_reason(setup, bar, red_body, retracement, daily_atr, params)
    if reason is not None:
        return replace(setup, state=SetupState.BROKEN, broken_reason=reason)
    if setup.state is SetupState.PULLBACK:
        if setup.prev_high is not None and bar.close > setup.prev_high and bar.volume > setup.pullback_avg_volume:
            return replace(setup, state=SetupState.TRIGGERED, trigger_close=bar.close, triggered_at=bar.time)
        if bar.high > setup.impulse_high:
            return _new_high(setup, bar)
        pulled = replace(
            setup,
            pullback_low=min(setup.pullback_low if setup.pullback_low is not None else bar.low, bar.low),
            pullback_bars=setup.pullback_bars + 1,
            pullback_volume_total=setup.pullback_volume_total + bar.volume,
            largest_red_body_atr=max(setup.largest_red_body_atr, red_body / daily_atr if red_body > 0 and daily_atr > 0 else 0.0),
        )
        if pulled.pullback_bars > pulled.impulse_bars:
            return replace(pulled, state=SetupState.BROKEN, broken_reason="pullback lasted longer than the impulse")
        return pulled
    if bar.high > setup.impulse_high:
        setup = _new_high(setup, bar)
    if retracement >= params.pullback_min_retrace:
        return replace(
            setup, state=SetupState.PULLBACK, pullback_low=bar.low, pullback_bars=1, pullback_volume_total=bar.volume,
            largest_red_body_atr=red_body / daily_atr if red_body > 0 and daily_atr > 0 else 0.0,
        )
    return setup


def _broken_reason(setup: Setup, bar: BarContext, red_body: float, retracement: float, daily_atr: float, params: VwapPullbackParameters) -> str | None:
    if bar.close < bar.vwap:
        return "close below VWAP"
    if bar.rs <= 0:
        return "relative strength lost"
    if retracement > params.pullback_max_retrace:
        return f"retraced more than {params.pullback_max_retrace:.1%} of the impulse"
    if red_body > params.bearish_body_atr * daily_atr:
        return "large bearish candle"
    if red_body > 0 and setup.impulse_avg_volume > 0 and bar.volume > params.selling_volume_ratio * setup.impulse_avg_volume:
        return "heavy selling volume"
    return None


def _new_high(setup: Setup, bar: BarContext) -> Setup:
    return replace(
        setup, state=SetupState.IMPULSE, impulse_high=bar.high, impulse_bars=setup.bars_seen, impulse_volume_total=setup.volume_total,
        pullback_low=None, pullback_bars=0, pullback_volume_total=0.0, largest_red_body_atr=0.0, retracement=0.0,
    )


def back_to_pullback(setup: Setup) -> Setup:
    """A trigger that was not (or could not be) acted on: keep the pullback, drop the trigger."""
    return replace(setup, state=SetupState.PULLBACK, trigger_close=None, triggered_at=None)


def mark_in_trade(setup: Setup) -> Setup:
    return replace(setup, state=SetupState.IN_TRADE)


def mark_done(setup: Setup) -> Setup:
    return replace(setup, state=SetupState.DONE)


def health(setup: Setup) -> dict[str, object]:
    """The health flags the entry agent reads (spec §3), rounded for the prompt."""
    return {
        "state": setup.state.value,
        "above_vwap": setup.last_close is not None and setup.last_vwap is not None and setup.last_close > setup.last_vwap,
        "vol_ratio": round(setup.pullback_avg_volume / setup.impulse_avg_volume, 2) if setup.impulse_avg_volume else None,
        "duration_ratio": round(setup.pullback_bars / setup.impulse_bars, 2) if setup.impulse_bars else None,
        "retracement_pct": round(100 * setup.retracement, 1),
        "rs_now_pct": round(100 * setup.last_rs, 2) if setup.last_rs is not None else None,
        "rvol_now": round(setup.last_rvol, 2) if setup.last_rvol is not None else None,
        "largest_red_body_atr": round(setup.largest_red_body_atr, 2),
    }
