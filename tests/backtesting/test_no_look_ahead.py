"""Integration test: the no-look-ahead guarantee across a full simulated multi-session
run. This is the test that protects the whole premise of the backtesting subsystem
(design spec, section 8): every price a strategy observes during on_trading_iteration()
must come from a bar that had already closed by the time of that observation, and the
bar for the CURRENT, still-forming session must never be visible.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from tests.backtesting.fakes import FakeBacktestDataSource

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MarketSession

ET = ZoneInfo("America/New_York")
AAPL = Asset("AAPL")


def _sessions(first_day: date, count: int) -> list[MarketSession]:
    sessions: list[MarketSession] = []
    day = first_day
    while len(sessions) < count:
        if day.weekday() < 5:
            sessions.append(MarketSession(
                open=datetime.combine(day, time(9, 30), tzinfo=ET),
                close=datetime.combine(day, time(16, 0), tzinfo=ET),
            ))
        day += timedelta(days=1)
    return sessions


def _close_indexed_bars(sessions: list[MarketSession], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
            "close": closes, "volume": [1000.0] * len(closes),
        },
        index=pd.DatetimeIndex([s.close for s in sessions], name="timestamp"),
    )


class RecordingStrategy(Strategy):
    sleeptime = "1D"

    def initialize(self) -> None:
        self.vars.observations = []

    def on_trading_iteration(self) -> None:
        now = self.clock.now()
        bars = self.get_historical_prices(AAPL, 10, "day")
        last_price = self.get_last_price(AAPL)
        latest_bar_close = (
            bars.df.index[-1].to_pydatetime() if bars is not None and not bars.df.empty else None
        )
        self.vars.observations.append(
            {"now": now, "latest_bar_close": latest_bar_close, "last_price": last_price}
        )


def test_strategy_never_observes_a_bar_that_has_not_closed_yet(tmp_path: Path) -> None:
    sessions = _sessions(date(2026, 1, 5), 5)
    closes = [150.0, 151.0, 149.0, 152.0, 153.0]
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _close_indexed_bars(sessions, closes))

    clock = BacktestClock(start=sessions[0].open - timedelta(hours=1), sessions=sessions)
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance

    strategy = RecordingStrategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
    strategy.executor.run()

    observations = strategy.vars.observations
    assert len(observations) == 5  # one iteration per session (sleeptime="1D")

    for i, obs in enumerate(observations):
        if obs["latest_bar_close"] is None:
            assert i == 0  # only the very first session has no prior closed bar at all
            continue
        # The chokepoint itself: bar_end <= cutoff, always, for every observation.
        assert obs["latest_bar_close"] <= obs["now"]
        # The sharper claim: session i's own iteration never sees session i's own bar --
        # only a strictly earlier session's.
        assert obs["latest_bar_close"] < sessions[i].close

    # Concretely: session 3's (index 2) iteration sees session 2's (index 1) close.
    assert observations[2]["last_price"] == Decimal(str(closes[1]))
    assert observations[2]["latest_bar_close"] == sessions[1].close


class OrderPlacingStrategy(Strategy):
    sleeptime = "1D"

    def initialize(self) -> None:
        self.vars.hooks_called = []
        self.vars.fill_args = None

    def before_market_opens(self) -> None:
        self.vars.hooks_called.append("before_market_opens")

    def before_starting_trading(self) -> None:
        self.vars.hooks_called.append("before_starting_trading")

    def on_trading_iteration(self) -> None:
        self.vars.hooks_called.append("on_trading_iteration")
        if self.first_iteration:
            self.submit_order(self.create_order(AAPL, 5, "buy"))

    def on_filled_order(self, position, order, price, quantity, multiplier) -> None:
        self.vars.fill_args = (order.identifier, price, quantity)

    def before_market_closes(self) -> None:
        self.vars.hooks_called.append("before_market_closes")

    def after_market_closes(self) -> None:
        self.vars.hooks_called.append("after_market_closes")


def test_full_simulated_session_dispatches_hooks_in_order_and_fills_next_bar(tmp_path: Path) -> None:
    sessions = _sessions(date(2026, 1, 5), 3)
    closes = [150.0, 151.0, 152.0]
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _close_indexed_bars(sessions, closes))

    clock = BacktestClock(start=sessions[0].open - timedelta(hours=1), sessions=sessions)
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance

    strategy = OrderPlacingStrategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
    strategy.executor.run()

    assert strategy.vars.hooks_called[:3] == [
        "before_market_opens", "before_starting_trading", "on_trading_iteration",
    ]
    assert strategy.vars.hooks_called.count("before_market_opens") == 3
    assert strategy.vars.hooks_called.count("on_trading_iteration") == 3
    assert strategy.vars.hooks_called.count("before_market_closes") == 3
    assert strategy.vars.hooks_called.count("after_market_closes") == 3

    # Submitted during session 1's iteration; fills against session 2's bar, and
    # on_filled_order (dispatched on the executor thread, per executor.py's design)
    # has fired with that fill's data by the time the run ends.
    assert strategy.vars.fill_args is not None
    identifier, price, quantity = strategy.vars.fill_args
    assert quantity == Decimal(5)
    assert price == Decimal("151.0")
