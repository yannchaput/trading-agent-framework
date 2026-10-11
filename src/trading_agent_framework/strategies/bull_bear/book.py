"""The book: the stocks bull_bear holds after a review, the ones the judge put in (pure).

The judge decides membership: code neither ranks the chosen stocks against each other nor gives a holding a
preferred slot. The book lists them in rank order, the order their buys are sent in.
"""

from __future__ import annotations

from collections.abc import Collection

from trading_agent_framework.strategies.bull_bear.debate_set import DebateSet


def build_book(debate: DebateSet, chosen: Collection[str]) -> tuple[str, ...]:
    """The chosen debated stocks, in rank order (`debate.stocks` is in rank order)."""
    wanted = set(chosen)
    return tuple(stock.symbol for stock in debate.stocks if stock.symbol in wanted)
