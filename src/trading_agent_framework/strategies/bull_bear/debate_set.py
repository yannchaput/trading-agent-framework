"""The stocks the agents debate this week, and the holdings code sells without asking them (pure).

The top `candidate_count` by momentum rank, plus every holding still ranked within `retention_rank` (cross_momentum's
own sell threshold). A holding ranked worse, or unranked (no data, failed a filter other than max_volatility, which
the pipeline applies to new entries only, left the universe), is a forced exit: it never reaches the agents. A stock in its re-entry cooldown (`excluded`) is never debated; a
holding still in cooldown (its sell did not fill) is a forced exit.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow


@dataclass(frozen=True, slots=True)
class DebateStock:
    """A stock in the debate.

    Attributes:
        rank: Momentum rank (1 is the strongest).
        row: The scored stock.
        held: Whether the account holds it.
    """

    rank: int
    row: MomentumRow
    held: bool

    @property
    def symbol(self) -> str:
        """Return the ticker.

        Returns:
            The stock's symbol.
        """
        return self.row.symbol


@dataclass(frozen=True, slots=True)
class ForcedExit:
    """A holding that must be sold whatever the judge says.

    Attributes:
        symbol: Ticker.
        reason: Why: `unranked`, `rank N` or `cooldown`.
    """

    symbol: str
    reason: str  # "unranked", "rank N" or "cooldown"


@dataclass(frozen=True, slots=True)
class DebateSet:
    """The stocks the agents debate in one review.

    Attributes:
        stocks: The candidates and the holdings kept in the debate, in rank order.
        forced_exits: The holdings sold without a debate, by symbol.
    """

    stocks: tuple[DebateStock, ...]  # in rank order
    forced_exits: tuple[ForcedExit, ...]  # by symbol

    @property
    def symbols(self) -> list[str]:
        """Return the tickers in the debate.

        Returns:
            The symbols, in rank order.
        """
        return [stock.symbol for stock in self.stocks]

    @property
    def held(self) -> list[str]:
        """The held stocks in the debate, sorted (each gets a verdict from the judge)."""
        return sorted(stock.symbol for stock in self.stocks if stock.held)

    def stock(self, symbol: str) -> DebateStock:
        """Look up a stock in the debate.

        Args:
            symbol: Ticker.

        Returns:
            The debate entry for the symbol.

        Raises:
            StopIteration: If the symbol is not in the debate.
        """
        return next(stock for stock in self.stocks if stock.symbol == symbol)


def build_debate_set(
    ranked: Sequence[RankedRow],
    holdings: Collection[str],
    *,
    candidate_count: int,
    retention_rank: int,
    excluded: Collection[str] = (),
) -> DebateSet:
    """`holdings` are the stocks held, the parking instrument excluded (`Rebalancer.holdings()`); `excluded` the stocks in cooldown."""
    held, blocked = set(holdings), set(excluded)
    rank_of = {entry.row.symbol: entry.rank for entry in ranked}
    # Debate the top `candidate_count` ranks and the holdings still within the retention rank, never a stock in cooldown.
    stocks = tuple(
        DebateStock(entry.rank, entry.row, entry.row.symbol in held)
        for entry in ranked
        if entry.row.symbol not in blocked
        and (entry.rank <= candidate_count or (entry.row.symbol in held and entry.rank <= retention_rank))  # hysteresis: held stocks still ranked within the retention threshold are debated
    )
    forced = tuple(
        ForcedExit(symbol, "cooldown" if symbol in blocked else "unranked" if symbol not in rank_of else f"rank {rank_of[symbol]}")
        for symbol in sorted(held)
        if symbol in blocked or symbol not in rank_of or rank_of[symbol] > retention_rank
    )
    return DebateSet(stocks=stocks, forced_exits=forced)
