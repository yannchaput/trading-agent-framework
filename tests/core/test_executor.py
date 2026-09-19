from __future__ import annotations

import logging
import signal
from datetime import date, datetime, timedelta
from typing import cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, FatalStrategyError

MONDAY = date(2026, 9, 14)


class Recorder(Strategy):
    """Records every lifecycle hook with the clock time it ran at."""

    sleeptime = "2H"

    def __init__(self, broker: FakeBroker, **kwargs: object) -> None:
        super().__init__(broker, **kwargs)  # ty: ignore[invalid-argument-type]
        self.calls: list[tuple[str, datetime]] = []
        self.errors: list[BaseException] = []

    def _record(self, hook: str) -> None:
        self.calls.append((hook, self.get_datetime()))

    def initialize(self) -> None:
        self._record("initialize")

    def before_market_opens(self) -> None:
        self._record("before_market_opens")

    def before_starting_trading(self) -> None:
        self._record("before_starting_trading")

    def on_trading_iteration(self) -> None:
        self._record("on_trading_iteration")

    def before_market_closes(self) -> None:
        self._record("before_market_closes")

    def after_market_closes(self) -> None:
        self._record("after_market_closes")

    def on_strategy_end(self) -> None:
        self._record("on_strategy_end")

    def on_abrupt_closing(self) -> None:
        self._record("on_abrupt_closing")

    def on_bot_crash(self, error: BaseException) -> None:
        self.errors.append(error)
        super().on_bot_crash(error)

    @property
    def fake_clock(self) -> FakeClock:
        return cast(FakeClock, self.clock)

    def hooks(self) -> list[str]:
        return [hook for hook, _ in self.calls]

    def times(self, hook: str) -> list[datetime]:
        return [when for name, when in self.calls if name == hook]


def _strategy(
    cls: type[Recorder] = Recorder,
    *,
    start: datetime | None = None,
    sessions: int = 1,
    **kwargs: object,
) -> Recorder:
    clock = FakeClock(start or et(2026, 9, 14, 7), weekday_sessions(MONDAY, sessions))
    return cls(FakeBroker(clock), **kwargs)


def _run(cls: type[Recorder] = Recorder, **kwargs: object) -> Recorder:
    strategy = _strategy(cls, **kwargs)  # ty: ignore[invalid-argument-type]
    strategy.executor.run()
    return strategy


def _day(day: int) -> list[tuple[str, datetime]]:
    return [
        ("before_market_opens", et(2026, 9, day, 8, 30)),
        ("before_starting_trading", et(2026, 9, day, 9, 30)),
        ("on_trading_iteration", et(2026, 9, day, 9, 30)),
        ("on_trading_iteration", et(2026, 9, day, 11, 30)),
        ("on_trading_iteration", et(2026, 9, day, 13, 30)),
        ("on_trading_iteration", et(2026, 9, day, 15, 30)),
        ("before_market_closes", et(2026, 9, day, 15, 59)),
        ("after_market_closes", et(2026, 9, day, 16, 0)),
    ]


# --- lifecycle order and timing ------------------------------------------------------


def test_hook_order_and_times_over_two_sessions() -> None:
    strategy = _run(sessions=2)
    assert strategy.calls == [
        ("initialize", et(2026, 9, 14, 7)),
        *_day(14),
        *_day(15),
        ("on_strategy_end", et(2026, 9, 15, 16, 0)),
    ]


def test_mid_session_start_skips_before_market_opens_and_anchors_on_start() -> None:
    strategy = _run(start=et(2026, 9, 14, 10, 15))
    assert "before_market_opens" not in strategy.hooks()
    assert strategy.times("before_starting_trading") == [et(2026, 9, 14, 10, 15)]
    assert strategy.times("on_trading_iteration") == [
        et(2026, 9, 14, 10, 15),
        et(2026, 9, 14, 12, 15),
        et(2026, 9, 14, 14, 15),
    ]


def test_before_market_opens_runs_with_zero_minutes_before_opening() -> None:
    class NoLead(Recorder):
        minutes_before_opening = 0

    strategy = _run(NoLead)
    assert strategy.times("before_market_opens") == [et(2026, 9, 14, 9, 30)]


def test_five_minute_sleeptime_stops_before_the_closing_window() -> None:
    class FiveMinutes(Recorder):
        sleeptime = "5M"

    iterations = _run(FiveMinutes).times("on_trading_iteration")
    assert len(iterations) == 78
    assert iterations[0] == et(2026, 9, 14, 9, 30)
    assert iterations[-1] == et(2026, 9, 14, 15, 55)


def test_daily_sleeptime_runs_once_per_session() -> None:
    class Daily(Recorder):
        sleeptime = "1D"

    assert _run(Daily, sessions=2).times("on_trading_iteration") == [
        et(2026, 9, 14, 9, 30),
        et(2026, 9, 15, 9, 30),
    ]


def test_daily_sleeptime_before_market_closes_fires_near_close() -> None:
    class Daily(Recorder):
        sleeptime = "1D"

    strategy = _run(Daily)
    assert strategy.times("before_market_closes") == [et(2026, 9, 14, 15, 59)]


def test_two_day_sleeptime_skips_every_other_session() -> None:
    class EveryOtherDay(Recorder):
        sleeptime = "2D"

    strategy = _run(EveryOtherDay, sessions=3)
    assert strategy.times("on_trading_iteration") == [
        et(2026, 9, 14, 9, 30),
        et(2026, 9, 16, 9, 30),
    ]
    assert len(strategy.times("before_starting_trading")) == 3
    assert strategy.times("before_market_closes") == [
        et(2026, 9, 14, 15, 59),
        et(2026, 9, 15, 15, 59),
        et(2026, 9, 16, 15, 59),
    ]


def test_overrun_skips_missed_ticks_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    class Slow(Recorder):
        sleeptime = "10M"

        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            if self.first_iteration:
                self.fake_clock.advance(25 * 60)

    with caplog.at_level(logging.WARNING, logger="trading_agent_framework"):
        strategy = _run(Slow)
    assert strategy.times("on_trading_iteration")[:3] == [
        et(2026, 9, 14, 9, 30),
        et(2026, 9, 14, 10, 0),
        et(2026, 9, 14, 10, 10),
    ]
    assert "skipped 2 tick(s)" in caplog.text


def test_sleeptime_is_reread_after_each_iteration() -> None:
    class Changer(Recorder):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            self.sleeptime = "1H"

    assert _run(Changer).times("on_trading_iteration")[:3] == [
        et(2026, 9, 14, 9, 30),
        et(2026, 9, 14, 10, 30),
        et(2026, 9, 14, 11, 30),
    ]


def test_first_iteration_is_true_only_once() -> None:
    class FirstFlag(Recorder):
        def __init__(self, broker: FakeBroker) -> None:
            super().__init__(broker)
            self.flags: list[bool] = []

        def on_trading_iteration(self) -> None:
            self.flags.append(self.first_iteration)

    strategy = _run(FirstFlag)
    assert cast(FirstFlag, strategy).flags == [True, False, False, False]


def test_sleep_advances_the_clock_inside_a_hook() -> None:
    class Sleeper(Recorder):
        def on_trading_iteration(self) -> None:
            if self.first_iteration:
                self.sleep(90)
                self._record("after_sleep")

    assert _run(Sleeper).times("after_sleep") == [et(2026, 9, 14, 9, 31, 30)]


def test_no_sessions_runs_initialize_and_end_only() -> None:
    assert _run(sessions=0).hooks() == ["initialize", "on_strategy_end"]


# --- errors -----------------------------------------------------------------------


def test_iteration_crash_calls_on_bot_crash_and_trading_continues() -> None:
    class Crashy(Recorder):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            if self.first_iteration:
                raise RuntimeError("boom")

    strategy = _run(Crashy)
    assert [str(e) for e in strategy.errors] == ["boom"]
    assert "on_abrupt_closing" in strategy.hooks()  # lumibot's default on_bot_crash
    assert len(strategy.times("on_trading_iteration")) == 4
    assert strategy.hooks()[-1] == "on_strategy_end"


def test_a_fatal_strategy_error_aborts_the_run_after_the_usual_crash_bookkeeping() -> None:
    class GivesUp(Recorder):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            raise FatalStrategyError("give up")

    strategy = _strategy(GivesUp)

    with pytest.raises(FatalStrategyError, match="give up"):
        strategy.executor.run()

    assert len(strategy.times("on_trading_iteration")) == 1  # an ordinary crash would have gone on to 4
    assert [str(e) for e in strategy.errors] == ["give up"]  # on_bot_crash ran, as for a failed initialize
    assert strategy.hooks()[-1] == "on_strategy_end"  # the run still winds down cleanly


def test_lifecycle_hook_crash_is_reported_and_the_session_goes_on() -> None:
    class BadOpen(Recorder):
        def before_market_opens(self) -> None:
            raise RuntimeError("open failed")

    strategy = _run(BadOpen)
    assert [str(e) for e in strategy.errors] == ["open failed"]
    assert "before_starting_trading" in strategy.hooks()


def test_initialize_crash_propagates_without_on_strategy_end() -> None:
    class BadInit(Recorder):
        def initialize(self) -> None:
            raise RuntimeError("bad init")

    strategy = _strategy(BadInit)
    with pytest.raises(RuntimeError, match="bad init"):
        strategy.executor.run()
    assert [str(e) for e in strategy.errors] == ["bad init"]
    assert "on_strategy_end" not in strategy.hooks()
    assert cast(FakeBroker, strategy.broker).calls[-1] == "stop_stream"


def test_invalid_sleeptime_fails_fast_after_initialize() -> None:
    class Bad(Recorder):
        sleeptime = "soon"

    strategy = _strategy(Bad)
    with pytest.raises(ConfigurationError):
        strategy.executor.run()
    assert strategy.hooks() == ["initialize", "on_strategy_end"]


def test_calendar_errors_are_retried(caplog: pytest.LogCaptureFixture) -> None:
    strategy = _strategy()
    strategy.fake_clock.next_session_errors = [BrokerError("calendar down")]
    with caplog.at_level(logging.ERROR, logger="trading_agent_framework"):
        strategy.executor.run()
    assert strategy.times("before_market_opens") == [et(2026, 9, 14, 8, 30)]
    assert "retrying" in caplog.text


# --- stop, parameters, setup/teardown ------------------------------------------------


def test_stop_ends_the_run_after_the_current_hook() -> None:
    class Stopper(Recorder):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            self.stop()

    strategy = _run(Stopper, sessions=2)
    assert strategy.executor.stopped is True
    assert strategy.hooks() == [
        "initialize",
        "before_market_opens",
        "before_starting_trading",
        "on_trading_iteration",
        "on_strategy_end",
    ]


def test_initialize_receives_matching_parameters() -> None:
    class WithParams(Recorder):
        def initialize(self, symbol: str = "", quantity: int = 1) -> None:
            self.received = (symbol, quantity)

    strategy = _run(WithParams, parameters={"symbol": "SPY", "unused": 3})
    assert cast(WithParams, strategy).received == ("SPY", 1)


def test_orders_are_synced_and_the_stream_started_before_initialize() -> None:
    class SeesSetup(Recorder):
        def initialize(self) -> None:
            self.calls_at_init = list(cast(FakeBroker, self.broker).calls)

    strategy = _run(SeesSetup)
    broker = cast(FakeBroker, strategy.broker)
    assert cast(SeesSetup, strategy).calls_at_init == ["sync_open_orders", "start_stream"]
    assert broker.calls == ["sync_open_orders", "start_stream", "stop_stream"]
    assert broker.tracker.listeners == []


def test_stop_stream_failure_still_removes_listener_and_restores_sigterm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ExplodingStopStreamBroker(FakeBroker):
        def stop_stream(self, timeout: float = 5.0) -> None:
            self.calls.append("stop_stream")
            raise RuntimeError("stream teardown boom")

    clock = FakeClock(et(2026, 9, 14, 7), weekday_sessions(MONDAY, 1))
    broker = ExplodingStopStreamBroker(clock)
    strategy = Recorder(broker)
    previous_handler = signal.getsignal(signal.SIGTERM)

    with caplog.at_level(logging.ERROR, logger="trading_agent_framework"):
        strategy.executor.run()  # must not raise despite stop_stream blowing up

    assert broker.tracker.listeners == []
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert strategy.hooks()[-1] == "on_strategy_end"
    assert "stop_stream failed during teardown" in caplog.text


def test_wait_until_uses_the_clocks_max_wait_slice() -> None:
    class Hello(Strategy):
        sleeptime = "1D"

        def on_trading_iteration(self) -> None:
            pass

    clock = FakeClock(et(2026, 1, 5, 9, 0))
    clock.max_wait_slice = 10.0  # ty: ignore[invalid-assignment]
    strategy = Hello(FakeBroker(clock))
    executor = strategy.executor

    executor.wait_until(clock.now() + timedelta(seconds=25))

    # 10s, 10s, 5s -- never a bare 60s slice from the old module constant.
    assert clock.waits == [10.0, 10.0, 5.0]
