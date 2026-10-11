"""The book: the stocks bull_bear holds after a review, built by code from the judge's verdicts (pure).

The survivors are the debated stocks the judge did not veto. Holdings come first, so a kept holding ranked within
`retention_rank` keeps its slot (hysteresis in cross_momentum's spirit: `build_debate_set` already drops holdings
ranked worse than `retention_rank`), then new names; each group in rank order. The first `book_size` survivors form
the book. The pipeline caps the vetoes so that at least `book_size` stocks survive.
"""

from __future__ import annotations

from collections.abc import Collection

from trading_agent_framework.strategies.bull_bear.debate_set import DebateSet


def build_book(debate: DebateSet, vetoed: Collection[str], book_size: int) -> tuple[str, ...]:
    """The book's symbols in buy order: kept holdings, then kept new names, each by rank, cut at `book_size`.

    More kept holdings than `book_size` (after `book_size` was lowered) keeps the best-ranked holdings.
    """
    vetoes = set(vetoed)
    survivors = [stock for stock in debate.stocks if stock.symbol not in vetoes]  # `debate.stocks` is in rank order
    holdings = [stock.symbol for stock in survivors if stock.held]
    newcomers = [stock.symbol for stock in survivors if not stock.held]
    return tuple([*holdings, *newcomers][:book_size])
