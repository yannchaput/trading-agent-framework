from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.cooldown import advance_cooldowns


def test_a_stock_sold_by_the_judge_starts_a_cooldown_of_the_full_length() -> None:
    assert advance_cooldowns({}, started=["AAA"], length=4) == {"AAA": 4}


def test_each_completed_review_counts_a_cooldown_down_and_drops_it_after_the_last() -> None:
    cooldowns: dict[str, int] = {"AAA": 4}
    seen = []
    for _ in range(4):
        cooldowns = advance_cooldowns(cooldowns, started=[], length=4)
        seen.append(dict(cooldowns))

    assert seen == [{"AAA": 3}, {"AAA": 2}, {"AAA": 1}, {}]  # excluded at 4 reviews, back at the 5th


def test_a_length_of_zero_starts_no_cooldown() -> None:
    assert advance_cooldowns({"AAA": 2}, started=["BBB"], length=0) == {"AAA": 1}
