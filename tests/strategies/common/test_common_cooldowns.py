from __future__ import annotations

from trading_agent_framework.strategies.common.cooldowns import advance_cooldowns


def test_a_stock_sold_starts_a_cooldown_of_the_full_length() -> None:
    assert advance_cooldowns({}, started=["AAA"], length=4) == {"AAA": 4}


def test_each_completed_review_counts_a_cooldown_down_and_drops_it_after_the_last() -> None:
    cooldowns: dict[str, int] = {"AAA": 4}
    seen = []
    for _ in range(4):
        cooldowns = advance_cooldowns(cooldowns, started=[], length=4)
        seen.append(dict(cooldowns))

    assert seen == [{"AAA": 3}, {"AAA": 2}, {"AAA": 1}, {}]  # excluded at 4 reviews, back at the 5th


def test_each_completed_review_counts_every_cooldown_down_and_drops_it_at_zero() -> None:
    assert advance_cooldowns({"A": 3, "B": 1}, started=[], length=4) == {"A": 2}


def test_a_symbol_sold_again_restarts_its_cooldown() -> None:
    assert advance_cooldowns({"HLT": 2}, started=["HLT"], length=4) == {"HLT": 4}


def test_a_length_of_zero_starts_no_cooldown() -> None:
    assert advance_cooldowns({"AAA": 2}, started=["BBB"], length=0) == {"AAA": 1}


def test_a_length_of_zero_disables_the_cooldown() -> None:
    assert advance_cooldowns({}, started=["HLT"], length=0) == {}


def test_a_sale_keeps_the_symbol_out_of_the_next_reviews_exactly() -> None:
    cooldowns = advance_cooldowns({}, started=["HLT"], length=4)  # sold at review R
    for _ in range(4):  # reviews R+1 .. R+4 start with HLT still cooling down
        assert "HLT" in cooldowns
        cooldowns = advance_cooldowns(cooldowns, started=[], length=4)
    assert "HLT" not in cooldowns  # back at review R+5


def test_the_input_cooldowns_are_not_modified() -> None:
    before = {"A": 2}

    advance_cooldowns(before, started=["B"], length=4)

    assert before == {"A": 2}


def test_every_stock_started_in_the_same_review_begins_its_cooldown() -> None:
    assert advance_cooldowns({"A": 3}, started=["B", "C"], length=4) == {"A": 2, "B": 4, "C": 4}
