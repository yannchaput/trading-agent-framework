"""Builders shared by the earnings_drift tests (not a test module)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from tests.fakes import et

from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
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
