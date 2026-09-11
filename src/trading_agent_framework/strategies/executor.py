"""Single-threaded driver of a `Strategy`'s lifecycle.

`StrategyExecutor.run()` walks the market calendar one session at a time:
before_market_opens -> before_starting_trading -> on_trading_iteration every
`sleeptime` -> before_market_closes -> after_market_closes. Every wait goes
through the strategy's `MarketClock`, so a simulated clock can later drive the
very same loop for backtesting.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from trading_agent_framework.clock import MarketSession
from trading_agent_framework.errors import BrokerError
from trading_agent_framework.strategies.events import OrderEventQueue
from trading_agent_framework.strategies.timing import next_tick, parse_sleeptime

if TYPE_CHECKING:
    from trading_agent_framework.strategies.strategy import Strategy

logger = logging.getLogger(__name__)

# Waits are sliced so clock drift (e.g. a suspended laptop) is corrected at least this often;
# order events and stop() interrupt a slice through the wake event anyway.
MAX_WAIT_SLICE_SECONDS = 60.0
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
        strategy = self.strategy
        broker = strategy.broker
        self._stop.clear()
        broker.tracker.listeners.append(self._events)
        initialized = False
        try:
            broker.sync_open_orders()
            broker.start_stream()
            self._initialize()
            initialized = True
            parse_sleeptime(strategy.sleeptime)  # fail fast on a bad sleeptime
            self._run_sessions()
        finally:
            if initialized:
                self._call_hook(strategy.on_strategy_end)
            broker.stop_stream()
            broker.tracker.listeners.remove(self._events)

    def wait_until(self, deadline: datetime) -> None:
        """Wait on the clock until `deadline` or `stop()`."""
        clock = self.strategy.clock
        while not self._stop.is_set():
            remaining = (deadline - clock.now()).total_seconds()
            if remaining <= 0:
                return
            clock.wait(min(remaining, MAX_WAIT_SLICE_SECONDS), self._wake)
            self._wake.clear()

    # --- run phases ----------------------------------------------------------------

    def _initialize(self) -> None:
        strategy = self.strategy
        signature = inspect.signature(strategy.initialize)
        kwargs = {
            name: value
            for name, value in strategy.parameters.items()
            if name in signature.parameters and signature.parameters[name].kind in _KEYWORD_KINDS
        }
        logger.info("Initializing strategy %s", strategy.name)
        try:
            strategy.initialize(**kwargs)
        except Exception as exc:
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
        strategy = self.strategy
        stop_at = session.close - timedelta(minutes=strategy.minutes_before_closing)
        tick = max(session.open, self._now())
        while not self._stop.is_set():
            self.wait_until(min(tick, stop_at))
            if self._stop.is_set() or self._now() >= stop_at:
                return
            sleeptime = parse_sleeptime(strategy.sleeptime)
            if sleeptime.sessions is not None and not self._iteration_due(sleeptime.sessions):
                self.wait_until(stop_at)
                return
            self._iterate()
            sleeptime = parse_sleeptime(strategy.sleeptime)  # the iteration may change it
            if sleeptime.interval is None:
                self.wait_until(stop_at)
                return  # session-based sleeptime: one iteration per due session
            tick, skipped = next_tick(tick, sleeptime.interval, self._now())
            if skipped:
                logger.warning(
                    "Iteration of %s overran its %s sleeptime; skipped %d tick(s)",
                    strategy.name,
                    strategy.sleeptime,
                    skipped,
                )

    def _iteration_due(self, sessions: int) -> bool:
        last = self._last_iteration_session
        return last is None or self._session_index - last >= sessions

    def _iterate(self) -> None:
        strategy = self.strategy
        self._last_iteration_session = self._session_index
        logger.debug("Trading iteration of %s at %s", strategy.name, self._now().isoformat())
        try:
            strategy.on_trading_iteration()
        except Exception as exc:
            logger.exception("on_trading_iteration failed for strategy %s", strategy.name)
            self._on_bot_crash(exc)
        finally:
            strategy.first_iteration = False

    # --- helpers -------------------------------------------------------------------

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
