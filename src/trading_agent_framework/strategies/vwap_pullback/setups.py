"""Pure per-candidate state machine (spec §3), advanced on completed 5-minute bars.

WATCH -> IMPULSE -> PULLBACK -> TRIGGERED -> IN_TRADE -> DONE, or BROKEN (terminal for the session).
`advance` replays every bar newer than the last one seen, so a tick that finds several new bars gives
exactly the result of one bar per tick. A trigger lasts one bar: the next bar re-evaluates the setup as
a pullback (it may trigger again).

The pattern being detected: the stock shows abnormal strength (IMPULSE), gives some of it back in an
orderly way while staying above VWAP (PULLBACK), then buyers return (TRIGGERED = the entry signal).
Anything that says the move has failed -- losing VWAP, losing relative strength, a deep or heavy or
long pullback -- ends the setup for the day (BROKEN).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters


class SetupState(StrEnum):
    WATCH = "watch"  # tracked, no impulse yet
    IMPULSE = "impulse"  # strong move off the open; watching for a pullback
    PULLBACK = "pullback"  # orderly retracement above VWAP; watching for resumption
    TRIGGERED = "triggered"  # this bar resumed upward: the desk enters it now
    IN_TRADE = "in_trade"  # an entry was accepted (the entry order may still be pending)
    DONE = "done"  # the trade closed: one trade per symbol per session
    BROKEN = "broken"  # the pattern failed; ignored for the rest of the session. See '_broken_reason' method.


# States the price action no longer moves: the setup is either traded or dead.
_FROZEN = frozenset({SetupState.IN_TRADE, SetupState.DONE, SetupState.BROKEN})


@dataclass(frozen=True, slots=True)
class Setup:
    """A candidate's pattern state. Immutable: every transition returns a new `Setup` (`dataclasses.replace`)."""

    symbol: str
    state: SetupState = SetupState.WATCH
    last_bar_time: datetime | None = None  # close of the last bar processed; `advance` skips bars up to it
    bars_seen: int = 0  # bars processed since the open
    volume_total: float = 0.0  # their total volume
    impulse_high: float | None = None
    impulse_bars: int = 0  # bars from the open up to the impulse high
    impulse_volume_total: float = 0.0  # their volume
    pullback_low: float | None = None  # lowest low of the pullback: the protective stop is placed under it
    pullback_bars: int = 0
    pullback_volume_total: float = 0.0
    prev_high: float | None = None  # previous bar's high: a close above it is the resumption signal
    trigger_close: float | None = None  # close of the triggering bar, the entry reference for R
    triggered_at: datetime | None = None
    retracement: float = 0.0  # of the impulse leg (session open -> impulse high), at the last close
    max_retracement: float = 0.0  # the deepest `retracement` since the impulse high: the pullback's depth (it shrinks again on the bounce)
    largest_red_body_atr: float = 0.0  # biggest down-bar body of the pullback, in daily ATRs (a health flag)
    last_close: float | None = None
    last_vwap: float | None = None
    last_rs: float | None = None
    last_rvol: float | None = None
    broken_reason: str | None = None  # why the setup went BROKEN, for the logs

    @property
    def impulse_avg_volume(self) -> float:
        """Average bar volume of the impulse (0 before one exists)."""
        return self.impulse_volume_total / self.impulse_bars if self.impulse_bars else 0.0

    @property
    def pullback_avg_volume(self) -> float:
        """Average bar volume of the pullback (0 before one exists)."""
        return self.pullback_volume_total / self.pullback_bars if self.pullback_bars else 0.0


def advance(setup: Setup, bars: Sequence[BarContext], daily_atr: float, params: VwapPullbackParameters) -> Setup:
    """Replay every bar newer than `setup.last_bar_time`, oldest first.

    `bars` is the whole session so far; bars already processed are skipped, so calling this every tick
    with the full list is both cheap and idempotent.
    """
    for bar in bars:
        if setup.last_bar_time is None or bar.time > setup.last_bar_time:
            setup = step(setup, bar, daily_atr, params)
    return setup


def step(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    """The setup after one more completed bar."""
    # Book-keeping first, whatever the state: counters and the latest bar's measures.
    seen = replace(
        setup,
        last_bar_time=bar.time,
        bars_seen=setup.bars_seen + 1,
        volume_total=setup.volume_total + bar.volume,
        last_close=bar.close,
        last_vwap=bar.vwap,
        last_rs=bar.rs,
        last_rvol=bar.rvol,
    )
    if seen.state in _FROZEN:
        return seen
    if seen.state is SetupState.TRIGGERED:  # the trigger was the previous bar: re-evaluate as a pullback
        seen = back_to_pullback(seen)
    if seen.state is SetupState.WATCH:
        return replace(_watch(seen, bar, daily_atr, params), prev_high=bar.high)
    return replace(_impulse_or_pullback(seen, bar, daily_atr, params), prev_high=bar.high)


def _watch(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    """WATCH -> IMPULSE when the session has moved enough, on unusual volume, beating the market, above VWAP."""
    move_atr = (bar.session_high - bar.session_open) / daily_atr if daily_atr > 0 else 0.0
    if move_atr >= params.impulse_move_atr and bar.rvol is not None and bar.rvol >= params.rvol_min and bar.rs > 0 and bar.close > bar.vwap:
        # The impulse spans every bar since the open (its average volume is the pullback's yardstick).
        return replace(setup, state=SetupState.IMPULSE, impulse_high=bar.session_high, impulse_bars=setup.bars_seen, impulse_volume_total=setup.volume_total)
    return setup


def _impulse_or_pullback(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    """One bar for a setup in IMPULSE or PULLBACK.

    Order matters: (1) broken checks, (2) in PULLBACK, the trigger, then a new high, then the pullback
    update and its duration check; (3) in IMPULSE, a new high, then the retracement that starts a pullback.
    """
    assert setup.impulse_high is not None  # set on entering IMPULSE
    leg = setup.impulse_high - bar.session_open
    retracement = (setup.impulse_high - bar.close) / leg if leg > 0 else 0.0
    setup = replace(setup, retracement=max(retracement, 0.0), max_retracement=max(setup.max_retracement, retracement))
    red_body = bar.open - bar.close  # > 0 only for a down bar
    reason = _broken_reason(setup, bar, red_body, retracement, daily_atr, params)
    if reason is not None:
        return replace(setup, state=SetupState.BROKEN, broken_reason=reason)
    if setup.state is SetupState.PULLBACK:
        # Resumption: close above the previous bar's high on more volume than the pullback's average.
        # (Above VWAP is already guaranteed: a close below it broke the setup just above.)
        if setup.prev_high is not None and bar.close > setup.prev_high and bar.volume > setup.pullback_avg_volume:
            return replace(setup, state=SetupState.TRIGGERED, trigger_close=bar.close, triggered_at=bar.time)
        # A new high without trigger volume: the impulse simply continued; start over from IMPULSE.
        if setup.impulse_high is not None and bar.high > setup.impulse_high:
            return _new_high(setup, bar)
        pulled = replace(
            setup,
            pullback_low=min(setup.pullback_low if setup.pullback_low is not None else bar.low, bar.low),
            pullback_bars=setup.pullback_bars + 1,
            pullback_volume_total=setup.pullback_volume_total + bar.volume,
            largest_red_body_atr=max(setup.largest_red_body_atr, red_body / daily_atr if red_body > 0 and daily_atr > 0 else 0.0),
        )
        # A pullback that lasts longer than the impulse that built the move is distribution, not a pause.
        if pulled.pullback_bars > pulled.impulse_bars:
            return replace(pulled, state=SetupState.BROKEN, broken_reason="pullback lasted longer than the impulse")
        return pulled
    # IMPULSE: extend the impulse on a new high, then check whether this bar retraced enough to start a pullback.
    if setup.impulse_high is not None and bar.high > setup.impulse_high:
        setup = _new_high(setup, bar)
    if retracement >= params.pullback_min_retrace:
        # The retracing bar is the pullback's first bar.
        return replace(
            setup,
            state=SetupState.PULLBACK,
            pullback_low=bar.low,
            pullback_bars=1,
            pullback_volume_total=bar.volume,
            largest_red_body_atr=red_body / daily_atr if red_body > 0 and daily_atr > 0 else 0.0,
        )
    return setup


def _broken_reason(setup: Setup, bar: BarContext, red_body: float, retracement: float, daily_atr: float, params: VwapPullbackParameters) -> str | None:
    """Return a string explaining why this bar invalidates the pattern, or None if the pattern still holds.

    Checks five failure conditions: loss of VWAP support, loss of relative strength, excessive retracement,
    oversized bearish candle, or abnormally heavy selling volume. Evaluated on every IMPULSE/PULLBACK bar
    before any state transitions.

    Args:
        setup: Current setup state tracking impulse/pullback metrics.
        bar: The bar being evaluated.
        red_body: Size of the down-bar body (open - close), positive only for down bars.
        retracement: Depth of pullback as a fraction of the impulse leg (0.0 to 1.0+).
        daily_atr: Daily Average True Range, used to normalize candle sizes.
        params: Strategy parameters controlling break thresholds.

    Returns:
        A string describing why the pattern broke, or None if all conditions pass.
    """
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
    """Back to IMPULSE at a new high: the impulse now runs to this bar and any pullback so far is discarded."""
    return replace(
        setup,
        state=SetupState.IMPULSE,
        impulse_high=bar.high,
        impulse_bars=setup.bars_seen,
        impulse_volume_total=setup.volume_total,
        pullback_low=None,
        pullback_bars=0,
        pullback_volume_total=0.0,
        largest_red_body_atr=0.0,
        retracement=0.0,
        max_retracement=0.0,
    )


def back_to_pullback(setup: Setup) -> Setup:
    """A trigger that was not (or could not be) acted on: keep the pullback, drop the trigger."""
    return replace(setup, state=SetupState.PULLBACK, trigger_close=None, triggered_at=None)


def mark_in_trade(setup: Setup) -> Setup:
    """An entry was accepted: freeze the setup (price action no longer moves it)."""
    return replace(setup, state=SetupState.IN_TRADE)


def mark_done(setup: Setup) -> Setup:
    """The trade closed: the symbol is not traded again this session."""
    return replace(setup, state=SetupState.DONE)
