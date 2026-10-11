from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.book import build_book
from trading_agent_framework.strategies.bull_bear.debate_set import DebateSet, DebateStock
from trading_agent_framework.strategies.common.scoring import MomentumRow, score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG


def _row(symbol: str) -> MomentumRow:
    row = score_stock(symbol, [100.0 + i for i in range(300)], [1e6] * 300, CONFIG)
    assert row is not None
    return row


def _debate(ranks: list[int], held: frozenset[int] = frozenset()) -> DebateSet:
    """The debated stocks S01, S02, ... in rank order, as `build_debate_set` returns them; `held` are the ranks owned."""
    return DebateSet(stocks=tuple(DebateStock(rank, _row(f"S{rank:02d}"), rank in held) for rank in sorted(ranks)), forced_exits=())


def _symbols(ranks: list[int]) -> tuple[str, ...]:
    return tuple(f"S{rank:02d}" for rank in ranks)


def test_without_a_veto_the_book_is_the_top_book_size_ranks() -> None:
    assert build_book(_debate(list(range(1, 16))), [], 10) == _symbols(list(range(1, 11)))


def test_a_vetoed_stock_is_replaced_by_the_next_ranked_survivor() -> None:
    book = build_book(_debate(list(range(1, 16))), ["S01", "S05", "S07"], 10)

    assert book == _symbols([2, 3, 4, 6, 8, 9, 10, 11, 12, 13])


def test_kept_holdings_come_first_then_new_names_in_rank_order() -> None:
    """The spec's example: holdings at ranks 2, 9 and 22; the judge vetoes rank 9 and ranks 1, 5 and 7."""
    debate = _debate([*range(1, 16), 22], held=frozenset({2, 9, 22}))

    book = build_book(debate, ["S09", "S01", "S05", "S07"], 10)

    assert book == _symbols([2, 22, 3, 4, 6, 8, 10, 11, 12, 13])


def test_more_surviving_holdings_than_book_size_keeps_the_best_ranked_holdings() -> None:
    debate = _debate(list(range(1, 16)), held=frozenset({2, 4, 6, 8}))

    assert build_book(debate, [], 3) == _symbols([2, 4, 6])
