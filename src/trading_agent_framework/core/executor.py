"""Single-threaded driver of a `Strategy`'s lifecycle.

`StrategyExecutor.run()` walks the market calendar one session at a time:
market regime refresh -> before_market_opens -> before_starting_trading ->
on_trading_iteration every `sleeptime` (from `iteration_start_time` when set) -> before_market_closes ->
after_market_closes. Every wait goes
through the strategy's `MarketClock`, so a simulated clock can later drive the
very same loop for backtesting. Order events are dispatched on the executor
thread while it waits.
"""

from __future__ import annotations

import inspect
import logging
import signal
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from types import FrameType
from typing import TYPE_CHECKING

from trading_agent_framework.core.events import OrderEventQueue, QueuedOrderEvent
from trading_agent_framework.core.timing import next_tick, parse_sleeptime
from trading_agent_framework.entities.enums import OrderEvent
from trading_agent_framework.utils.clock import MARKET_TZ, MarketSession
from trading_agent_framework.utils.errors import BrokerError, FatalStrategyError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

logger = logging.getLogger(__name__)

CALENDAR_RETRY_SECONDS = 60.0

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class StrategyExecutor:
    """Runs one strategy's hooks on the calling thread, session after session."""

    def __init__(self, strategy: Strategy) -> None:
        self.strategy = strategy
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._events = OrderEventQueue(self._wake)
        self._session_index = 0
        self._last_iteration_session: int | None = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        """Ask the run to end at the next check; safe from any thread or hook."""
        self._stop.set()
        self._wake.set()

    def run(self) -> None:
        """Run the strategy's whole lifecycle on the calling thread, until the calendar ends or `stop()` is called.

        Setup: adopt open orders from before a restart, start the order stream, then `initialize`. Sessions follow.
        Ctrl-C and SIGTERM (turned into KeyboardInterrupt for this run) call `on_abrupt_closing` instead of
        propagating. Any other exception (a failed `initialize`, a `FatalStrategyError`) propagates after teardown.

        Teardown always runs: `on_strategy_end` runs only if `initialize` succeeded, after pending order events are
        dispatched. The stream is stopped, this run's event queue is detached and SIGTERM is restored, even if a
        hook fails.
        """
        strategy = self.strategy
        broker = strategy.broker
        self._stop.clear()
        # Order events from the broker's stream thread are queued and dispatched here, on this thread.
        broker.tracker.listeners.append(self._events)
        restore_sigterm = self._install_sigterm_handler()
        # Only a strategy whose `initialize` succeeded gets `on_strategy_end`.
        initialized = False
        try:
            # Adopt orders left open by a previous run before the stream can report on them.
            broker.sync_open_orders()
            broker.start_stream()
            self._initialize()
            initialized = True
            parse_sleeptime(strategy.sleeptime)  # fail fast on a bad sleeptime
            self._run_sessions()
        except KeyboardInterrupt:
            logger.warning("Strategy %s interrupted; running on_abrupt_closing", strategy.name)
            self._call_hook(strategy.on_abrupt_closing)
        finally:
            try:
                if initialized:
                    # Flush events still queued so the last fills reach the strategy before it ends.
                    self._dispatch_events()
                    self._call_hook(strategy.on_strategy_end)
            finally:
                # Teardown must finish even if stopping the stream fails, so the error is logged, not raised.
                try:
                    broker.stop_stream()
                except Exception:
                    logger.exception("stop_stream failed during teardown")
                broker.tracker.listeners.remove(self._events)
                restore_sigterm()

    def wait_until(self, deadline: datetime, until: Callable[[], bool] | None = None) -> bool:
        """Wait on the clock until `deadline`, dispatching order events meanwhile.

        Returns True as soon as `until()` holds; False at the deadline or on `stop()`.
        """
        clock = self.strategy.clock
        while True:
            self._dispatch_events()
            if until is not None and until():
                return True
            remaining = (deadline - clock.now()).total_seconds()
            if remaining <= 0 or self._stop.is_set():
                return False
            clock.wait(min(remaining, clock.max_wait_slice), self._wake)
            self._wake.clear()

    def wait_for(self, until: Callable[[], bool], timeout: float | None = None) -> bool:
        """`wait_until` with a relative timeout; no timeout means a one-year horizon."""
        horizon = timedelta(days=365) if timeout is None else timedelta(seconds=timeout)
        return self.wait_until(self._now() + horizon, until)

    # --- run phases ----------------------------------------------------------------

    def _initialize(self) -> None:
        strategy = self.strategy
        signature = inspect.signature(strategy.initialize)
        # Populate the initialize() kwargs with the strategy's parameters that match its signature subclassing.
        kwargs = {name: value for name, value in strategy.parameters.items() if name in signature.parameters and signature.parameters[name].kind in _KEYWORD_KINDS}
        logger.info("Initializing strategy %s", strategy.name)
        try:
            strategy.initialize(**kwargs)
        except Exception as exc:
            # calls logging exception method: exc_info being true by default, the exception will be logged
            logger.exception("initialize failed for strategy %s", strategy.name)
            self._on_bot_crash(exc)
            raise

    def _run_sessions(self) -> None:
        while not self._stop.is_set():
            session = self._next_session()
            if session is None:
                return
            self._run_session(session)
            self._session_index += 1

    def _next_session(self) -> MarketSession | None:
        clock = self.strategy.clock
        while not self._stop.is_set():
            try:
                return clock.next_session()
            except BrokerError:
                logger.exception(
                    "Could not get the next market session; retrying in %.0fs",
                    CALENDAR_RETRY_SECONDS,
                )
                self.wait_until(clock.now() + timedelta(seconds=CALENDAR_RETRY_SECONDS))
        return None

    def _run_session(self, session: MarketSession) -> None:
        strategy = self.strategy
        logger.info(
            "Next session for %s: %s -> %s",
            strategy.name,
            session.open.isoformat(),
            session.close.isoformat(),
        )
        started_before_open = self._now() < session.open
        self.wait_until(session.open - timedelta(minutes=strategy.minutes_before_opening))
        if self._stop.is_set():
            return
        self._refresh_regime()
        if started_before_open:
            self._call_hook(strategy.before_market_opens)
            self.wait_until(session.open)
            if self._stop.is_set():
                return
        self._call_hook(strategy.before_starting_trading)
        self._trade(session)
        if self._stop.is_set():
            return
        self._call_hook(strategy.before_market_closes)
        self.wait_until(session.close + timedelta(minutes=strategy.minutes_after_closing))
        if self._stop.is_set():
            return
        self._call_hook(strategy.after_market_closes)

    def _trade(self, session: MarketSession) -> None:
        """Run the session's trading iterations, from its first iteration time until the trading cutoff.

        The cutoff is `minutes_before_closing` before the close: no iteration starts at or after it. Iterations
        start at the first iteration time (`_iteration_start`), then follow `strategy.sleeptime`: an interval
        (e.g. "5m") ticks on a fixed grid, a session count (e.g. "1D") iterates once per due session.
        """
        strategy = self.strategy
        stop_at = session.close - timedelta(minutes=strategy.minutes_before_closing)
        start = self._iteration_start(session)
        # The first iteration time is already in the closing window (or after it): nothing to run this session.
        if start >= stop_at:
            logger.warning(
                "No iteration for %s this session: iteration_start_time %s is not before %s",
                strategy.name,
                start.isoformat(),
                stop_at.isoformat(),
            )
            self.wait_until(stop_at)
            return
        # A late start (e.g. a restart after the first iteration time) runs at once, not at the missed tick.
        tick = max(start, self._now())
        while not self._stop.is_set():
            self.wait_until(min(tick, stop_at))
            if self._stop.is_set() or self._now() >= stop_at:
                return
            # Re-read per tick: the strategy may change its sleeptime during an iteration.
            sleeptime = parse_sleeptime(strategy.sleeptime)
            # Session-based sleeptime: skip this session unless enough sessions have passed since the last iteration.
            if sleeptime.sessions is not None and not self._iteration_due(sleeptime.sessions):
                self.wait_until(stop_at)
                return
            self._iterate()
            # Re-parse after the iteration, since it may have changed the sleeptime.
            sleeptime = parse_sleeptime(strategy.sleeptime)
            if sleeptime.interval is None:
                self.wait_until(stop_at)
                return  # session-based sleeptime: one iteration per due session
            # Next tick on the interval grid after now; ticks that already passed (an overrun) are skipped and logged.
            tick, skipped = next_tick(tick, sleeptime.interval, self._now())
            if skipped:
                logger.warning(
                    "Iteration of %s overran its %s sleeptime; skipped %d tick(s)",
                    strategy.name,
                    strategy.sleeptime,
                    skipped,
                )

    def _iteration_start(self, session: MarketSession) -> datetime:
        """The session's first iteration time: the open, or `iteration_start_time` (market time) when later."""
        start_time = self.strategy.iteration_start_time
        if start_time is None:
            return session.open
        session_date = session.open.astimezone(MARKET_TZ).date()
        return max(session.open, datetime.combine(session_date, start_time, tzinfo=MARKET_TZ))

    def _iteration_due(self, sessions: int) -> bool:
        last = self._last_iteration_session
        return last is None or self._session_index - last >= sessions

    def _iterate(self) -> None:
        """Run one `on_trading_iteration` and mark this session as iterated.

        Any exception is reported through `on_bot_crash` and the run carries on, except `FatalStrategyError`, which
        is reported the same way and then re-raised to end the run. `first_iteration` is cleared either way.
        """
        strategy = self.strategy
        self._last_iteration_session = self._session_index
        logger.info("Trading iteration of %s at %s | portfolio value: %s", strategy.name, self._now().isoformat(), self._portfolio_value())
        try:
            strategy.on_trading_iteration()
        except FatalStrategyError as exc:
            # The one deliberate way for an iteration to end the run (any other failure is survivable, e.g. live
            # trading must ride out a bad tick). Same handling as a failed `initialize`: report, then propagate.
            logger.exception("on_trading_iteration aborted strategy %s", strategy.name)
            self._on_bot_crash(exc)
            raise
        except Exception as exc:
            logger.exception("on_trading_iteration failed for strategy %s", strategy.name)
            self._on_bot_crash(exc)
        finally:
            strategy.first_iteration = False

    # --- helpers -------------------------------------------------------------------

    def _portfolio_value(self) -> str:
        """The portfolio value for the iteration log; a failed read is reported, never allowed to skip the tick."""
        try:
            return f"{self.strategy.get_portfolio_value():.2f}"
        except Exception as exc:  # BrokerError live, BacktestError in a backtest: either way the tick must still run
            return f"unavailable ({exc})"

    def _refresh_regime(self) -> None:
        """Once per session, also on a mid-session start. A framework step, not a strategy hook: a failure is
        logged and the session carries on, without `on_bot_crash`."""
        try:
            self.strategy._refresh_regime()
        except Exception:
            logger.exception("Market regime refresh failed for strategy %s", self.strategy.name)

    def _call_hook(self, hook: Callable[[], None]) -> None:
        try:
            hook()
        except Exception as exc:
            name = getattr(hook, "__name__", repr(hook))
            logger.exception("%s failed for strategy %s", name, self.strategy.name)
            self._on_bot_crash(exc)

    def _on_bot_crash(self, error: BaseException) -> None:
        try:
            self.strategy.on_bot_crash(error)
        except Exception:
            logger.exception("on_bot_crash raised for strategy %s", self.strategy.name)

    def _now(self) -> datetime:
        return self.strategy.clock.now()

    def _dispatch_events(self) -> None:
        for item in self._events.drain():
            self._dispatch(item)

    def _dispatch(self, item: QueuedOrderEvent) -> None:
        strategy = self.strategy
        order = item.order
        try:
            if item.event is OrderEvent.NEW:
                strategy.on_new_order(order)
            elif item.event is OrderEvent.CANCELED:
                strategy.on_canceled_order(order)
            elif item.event in (OrderEvent.FILLED, OrderEvent.PARTIALLY_FILLED):
                if item.price is None or item.quantity is None:
                    logger.warning("Fill event for order %s has no fill data", order.identifier)
                    return
                hook = strategy.on_filled_order if item.event is OrderEvent.FILLED else strategy.on_partially_filled_order
                try:
                    position = strategy.get_position(order.asset)
                except Exception:
                    logger.exception(
                        "Could not fetch position for %s while dispatching a fill; calling the fill hook with position=None",
                        order.asset,
                    )
                    position = None
                hook(position, order, item.price, item.quantity, 1)
            elif item.event is OrderEvent.ERROR:
                logger.warning(
                    "Order %s for %s was rejected: %s",
                    order.identifier,
                    strategy.name,
                    order.error_message,
                )
        except Exception as exc:
            logger.exception("Order hook failed for %s (%s)", strategy.name, item.event)
            self._on_bot_crash(exc)

    def _install_sigterm_handler(self) -> Callable[[], None]:
        """Turn SIGTERM into KeyboardInterrupt for this run; returns the undo function."""
        # Python only allows signal handlers on the main thread. Off it, do nothing
        # and return a no-op undo so callers can always call the result.
        if threading.current_thread() is not threading.main_thread():
            return lambda: None  # signal handlers can only be installed on the main thread
        # Remember whatever handler was active (it may be None if it was set outside Python).
        previous = signal.getsignal(signal.SIGTERM)

        # Raising KeyboardInterrupt lets a SIGTERM (sent by a process supervisor, for example)
        # take the same shutdown path as Ctrl-C, so the run ends cleanly instead of being killed.
        def interrupt(signum: int, frame: FrameType | None) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)

        # Put the previous handler back once the run ends. If there was none (None),
        # fall back to SIG_DFL, the OS default, since signal.signal() cannot accept None.
        def restore() -> None:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)

        return restore
