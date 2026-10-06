"""Hard gates on an earnings event, and the candidate that passes them (spec §3.5). Pure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.reaction import ReactionFeatures
from trading_agent_framework.strategies.earnings_drift.surprise import PickedSurprise, Surprise

REJECT_REASONS = ("already_held", "no_surprise_data", "eps_miss", "no_bars", "weak_reaction", "faded", "low_volume", "illiquid")


@dataclass(frozen=True, slots=True)
class Candidate:
    event: EarningsEvent
    reaction_day: date
    surprise: PickedSurprise
    reaction: ReactionFeatures
    headlines: tuple[tuple[str, str], ...] = ()  # (ET minute, headline): the picked one first, then the others oldest first, at most 5

    @property
    def symbol(self) -> str:
        return self.event.symbol


def gate(surprise: Surprise | None, reaction: ReactionFeatures | None, params: DriftParams, *, held: bool) -> str | None:
    """The first gate an event fails, in `REJECT_REASONS` order; None when it passes them all."""
    if held:
        return "already_held"
    if surprise is None:
        return "no_surprise_data"
    if not surprise.eps_beat:
        return "eps_miss"
    if reaction is None:
        return "no_bars"
    if reaction.abnormal_pct < params.min_abnormal_pct:
        return "weak_reaction"
    if reaction.hold_ratio is None or reaction.hold_ratio < params.min_hold_ratio or reaction.close_location < params.min_close_location:
        return "faded"
    if reaction.rel_volume < params.min_rel_volume:
        return "low_volume"
    if reaction.close < params.min_price or reaction.dollar_volume_20d < params.min_dollar_volume:
        return "illiquid"
    return None
