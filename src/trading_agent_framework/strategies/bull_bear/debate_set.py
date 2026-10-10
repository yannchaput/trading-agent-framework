"""The stocks the agents debate this week, and the holdings code sells without asking them (pure).

The top `shortlist_size` by momentum rank, plus every holding still ranked within `retention_rank` (cross_momentum's
own sell threshold). A holding ranked worse, or unranked (no data, failed a filter, left the universe), is a forced
exit: it never reaches the agents.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow


@dataclass(frozen=True, slots=True)
class DebateStock:
    rank: int
    row: MomentumRow
    held: bool

    @property
    def symbol(self) -> str:
        return self.row.symbol


@dataclass(frozen=True, slots=True)
class ForcedExit:
    symbol: str
    reason: str  # "unranked" or "rank N"


@dataclass(frozen=True, slots=True)
class DebateSet:
    stocks: tuple[DebateStock, ...]  # in rank order
    forced_exits: tuple[ForcedExit, ...]  # by symbol

    @property
    def symbols(self) -> list[str]:
        return [stock.symbol for stock in self.stocks]

    @property
    def held(self) -> list[str]:
        """The held stocks in the debate, sorted (the judge must keep or drop each)."""
        return sorted(stock.symbol for stock in self.stocks if stock.held)

    def stock(self, symbol: str) -> DebateStock:
        return next(stock for stock in self.stocks if stock.symbol == symbol)


def build_debate_set(ranked: Sequence[RankedRow], holdings: Collection[str], *, shortlist_size: int, retention_rank: int) -> DebateSet:
    """`holdings` are the stocks held, the parking instrument excluded (`Rebalancer.holdings()`)."""
    held = set(holdings)
    rank_of = {entry.row.symbol: entry.rank for entry in ranked}
    stocks = tuple(
        DebateStock(entry.rank, entry.row, entry.row.symbol in held)
        for entry in ranked
        if entry.rank <= shortlist_size or (entry.row.symbol in held and entry.rank <= retention_rank)
    )
    forced = tuple(
        ForcedExit(symbol, "unranked" if symbol not in rank_of else f"rank {rank_of[symbol]}")
        for symbol in sorted(held)
        if symbol not in rank_of or rank_of[symbol] > retention_rank
    )
    return DebateSet(stocks=stocks, forced_exits=forced)
