from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.debate_set import ForcedExit, build_debate_set
from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow, score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG


def _row(symbol: str) -> MomentumRow:
    row = score_stock(symbol, [100.0 + i for i in range(300)], [1e6] * 300, CONFIG)
    assert row is not None
    return row


def _ranked(count: int) -> list[RankedRow]:
    return [RankedRow(rank, _row(f"S{rank:02d}")) for rank in range(1, count + 1)]


def test_the_top_15_are_debated_in_rank_order() -> None:
    debate = build_debate_set(_ranked(40), [], shortlist_size=15, retention_rank=35)

    assert debate.symbols == [f"S{rank:02d}" for rank in range(1, 16)]
    assert debate.held == [] and debate.forced_exits == ()


def test_a_holding_ranked_16_to_35_joins_the_debate_tagged_held() -> None:
    debate = build_debate_set(_ranked(40), ["S20", "S35", "S03"], shortlist_size=15, retention_rank=35)

    assert debate.symbols == [*(f"S{rank:02d}" for rank in range(1, 16)), "S20", "S35"]
    assert debate.held == ["S03", "S20", "S35"]
    assert debate.stock("S20").rank == 20 and debate.stock("S20").held
    assert not debate.stock("S01").held


def test_a_holding_ranked_below_the_retention_rank_or_unranked_is_forced_out() -> None:
    debate = build_debate_set(_ranked(40), ["S36", "GONE", "S02"], shortlist_size=15, retention_rank=35)

    assert debate.forced_exits == (ForcedExit("GONE", "unranked"), ForcedExit("S36", "rank 36"))
    assert "S36" not in debate.symbols and "GONE" not in debate.symbols


def test_a_holding_inside_the_shortlist_is_debated_once_and_never_forced_out() -> None:
    """SHV never reaches here: the pipeline passes `Rebalancer.holdings()`, which excludes the parking instrument."""
    debate = build_debate_set(_ranked(20), ["S05"], shortlist_size=15, retention_rank=35)

    assert debate.symbols.count("S05") == 1 and debate.stock("S05").held
    assert debate.forced_exits == ()
