"""Builders shared by the earnings_drift tests (not a test module)."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from tests.fakes import FakeClock, FrameDataSource, et, weekday_sessions

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.strategies.earnings_drift.book import DriftState, JsonlLog
from trading_agent_framework.strategies.earnings_drift.desk import Desk
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.reaction import ReactionFeatures
from trading_agent_framework.strategies.earnings_drift.screening import Candidate
from trading_agent_framework.strategies.earnings_drift.surprise import PickedSurprise, Surprise

DEFAULT_DAY = date(2026, 9, 1)


def make_features(**overrides: float | None) -> ReactionFeatures:
    values: dict[str, float | None] = dict(
        gap_pct=0.06,
        return_pct=0.08,
        abnormal_pct=0.075,
        hold_ratio=0.8,
        close_location=0.9,
        rel_volume=3.0,
        dollar_volume_20d=50_000_000.0,
        runup_20d_pct=0.02,
        runup_60d_pct=0.05,
        atr14_pct=0.02,
        close=100.0,
        reaction_low=95.0,
    )
    values.update(overrides)
    return ReactionFeatures(**values)  # type: ignore[arg-type]


def make_candidate(symbol: str = "AAA", *, day: date = DEFAULT_DAY, accepted_at: datetime | None = None, surprise: Surprise | None = None, **feature_overrides: float | None) -> Candidate:
    accepted = accepted_at or et(day.year, day.month, day.day, 7, 0)
    headline = f"{symbol} Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate"
    return Candidate(
        event=EarningsEvent(symbol, accepted, f"0000000000-26-{symbol}", f"{symbol.lower()}-8k.htm"),
        reaction_day=day,
        surprise=PickedSurprise(surprise or Surprise(Decimal("1.52"), Decimal("1.20"), Decimal("1100000000"), Decimal("1000000000")), headline, accepted),
        reaction=make_features(**feature_overrides),
        headlines=((f"{day.isoformat()} 07:01", headline),),
    )


RIG_SESSIONS = weekday_sessions(date(2026, 9, 1), 8)
RIG_DATES = [s.open.date() for s in RIG_SESSIONS]
# AAA: entry fills at day 1's open (101); a stop placed at day 1's close (102) trails to 106 on day 2 and fills on day 3 at 97.52.
AAA_ROWS = [
    (100.0, 101.0, 99.0, 100.0, 1e6),
    (101.0, 103.0, 100.0, 102.0, 1e6),
    (102.0, 106.0, 101.0, 105.0, 1e6),
    (104.0, 104.0, 95.0, 96.0, 1e6),
] + [(96.0, 97.0, 95.5, 96.0, 1e6)] * 4
BBB_ROWS = [(50.0, 50.5, 49.5, 50.0, 1e6)] * 8


def rig_frame(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([s.close for s in RIG_SESSIONS[: len(rows)]], name="timestamp"))


class DeskRig:
    """A `Desk` over a real `BacktestBroker` on daily bars, starting at day 0's close (the cycle's time)."""

    def __init__(self, tmp_path: Path, *, params: DriftParams | None = None, candidates: list[Candidate] | None = None, budget: Decimal = Decimal("100000")) -> None:
        self.clock = FakeClock(RIG_SESSIONS[0].close, RIG_SESSIONS)
        frames = {("AAA", "day"): rig_frame(AAA_ROWS), ("BBB", "day"): rig_frame(BBB_ROWS)}
        self.broker = BacktestBroker("earnings_drift", data_source=FrameDataSource(frames, RIG_SESSIONS), clock=self.clock, budget=budget, timestep="day")
        self.strategy = Strategy(self.broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
        self.state = DriftState()
        self.trades_path, self.decisions_path = tmp_path / "trades.jsonl", tmp_path / "decisions.jsonl"
        self.desk = Desk(self.strategy, params or DriftParams(), self.state, trade_log=JsonlLog(lambda: self.trades_path), decision_log=JsonlLog(lambda: self.decisions_path))
        self.day = 0
        self.delivered: set[str] = set()
        default = [make_candidate("AAA", day=RIG_DATES[0], close=100.0, reaction_low=99.0), make_candidate("BBB", day=RIG_DATES[0], close=50.0, abnormal_pct=0.04)]
        self.desk.begin_session(RIG_DATES[0], RIG_DATES[:1], default if candidates is None else candidates, {})

    def advance(self) -> None:
        """Move to the next session's close and let the broker fill what it can (no hook delivered)."""
        before = self.clock.now()
        self.day += 1
        self.clock.advance((RIG_SESSIONS[self.day].close - before).total_seconds())
        self.broker.on_advance(before, self.clock.now())

    def deliver(self) -> None:
        """Deliver every fill the desk has not seen yet, as the executor would."""
        for order in self.strategy.get_orders():
            if order.is_filled() and order.identifier not in self.delivered:
                self.delivered.add(order.identifier)
                self.desk.on_order_filled(order)

    def next_close(self, candidates: list[Candidate] | None = None) -> None:
        self.advance()
        self.deliver()
        self.desk.begin_session(RIG_DATES[self.day], RIG_DATES[: self.day + 1], candidates or [], {})

    def open_aaa(self, quantity: int = 10, trail: float = 8.0) -> None:
        assert "error" not in self.desk.buy("AAA", quantity, trail, "beat and raise")
        self.next_close()

    def lines(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
