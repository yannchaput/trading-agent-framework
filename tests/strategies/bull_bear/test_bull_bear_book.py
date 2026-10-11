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


def test_the_book_is_the_chosen_stocks_in_rank_order() -> None:
    assert build_book(_debate(list(range(1, 16))), ["S07", "S02", "S11"]) == _symbols([2, 7, 11])


def test_a_holding_gets_no_preferred_slot() -> None:
    debate = _debate([*range(1, 16), 22], held=frozenset({22}))

    assert build_book(debate, ["S22", "S01"]) == _symbols([1, 22])


def test_nothing_chosen_is_an_empty_book() -> None:
    assert build_book(_debate(list(range(1, 16))), []) == ()


def test_a_chosen_symbol_outside_the_debate_is_ignored() -> None:
    assert build_book(_debate([1, 2, 3]), ["S02", "ZZZ"]) == _symbols([2])
