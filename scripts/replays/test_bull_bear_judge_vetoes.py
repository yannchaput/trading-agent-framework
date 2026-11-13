"""Tests of the judge-vetoes replay's pure helpers (run by hand: `uv run pytest scripts/replays/test_bull_bear_judge_vetoes.py`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from trading_agent_framework.strategies.bull_bear import prompts

_SPEC = importlib.util.spec_from_file_location("bull_bear_judge_vetoes", Path(__file__).with_name("bull_bear_judge_vetoes.py"))
assert _SPEC is not None and _SPEC.loader is not None
jv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(jv)


def test_the_new_prompt_adds_the_grounds_after_the_out_rule_and_keeps_the_rest() -> None:
    """The candidate prompt is today's plus the grounds paragraph, and obeys the strategy's prompt rules."""
    new = jv.new_judge_system(prompts.JUDGE_SYSTEM)

    assert new.replace(jv._ADDITION, "") == prompts.JUDGE_SYSTEM
    assert "weak grounds" in new and "strong grounds" in new
    assert not any(word in new for word in ("held", "weight_pct", "forced"))  # the prompt rules of the strategy's tests
    with pytest.raises(ValueError, match="out rule"):
        jv.new_judge_system("another prompt")


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("Debt surged 34x, bear rates high", "weak"),
        ("Revenue fell 12% and leverage rose", "weak"),
        ("Revenue -42% to $1.8M", "weak"),
        ("7-month runway and debt up", "strong"),  # a strong ground wins
        ("Shelf offering filed, dilution ahead", "strong"),
        ("Phase 3 trial readout pending", "strong"),
        ("Bull's case is thin", None),
    ],
)
def test_ground_reads_what_an_out_rests_on(reason: str, expected: str | None) -> None:
    """A strong ground wins over a weak one; a reason with neither has no ground."""
    assert jv.ground(reason) == expected


def test_book_return_sums_the_ins_at_the_stock_weight_and_skips_a_stock_without_a_return() -> None:
    """Only the stocks put in count, each at the stock weight."""
    verdicts = {"A": "in", "B": "out", "C": "in", "D": "in"}

    assert jv.book_return(verdicts, {"A": 0.10, "B": 0.50, "C": -0.20}, 0.1) == pytest.approx(-0.01)


def _group(day: str, new_out: int, rerun_out: int, n: int = 30) -> tuple[list[tuple[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """`n` stocks of one day: the first `new_out` are out for the new arm, the first `rerun_out` for the rerun."""
    symbols = [f"S{i}" for i in range(n)]
    new = {symbol: "out" if i < new_out else "in" for i, symbol in enumerate(symbols)}
    rerun = {symbol: "out" if i < rerun_out else "in" for i, symbol in enumerate(symbols)}
    return [(day, symbol) for symbol in symbols], {day: new}, {day: rerun}


def test_decide_keeps_only_when_the_three_gates_pass() -> None:
    """Each gate alone can turn the verdict to DROP."""
    weak, weak_new, weak_rerun = _group("weak", new_out=18, rerun_out=30)  # 60% against 100%: under two thirds
    strong, strong_new, strong_rerun = _group("strong", new_out=28, rerun_out=30)  # 93% against 100%: within 10 points
    arms = {"new": weak_new | strong_new, "old-2": weak_rerun | strong_rerun}

    assert jv.decide(weak, strong, arms, -0.001, 0.002)["verdict"] == "KEEP"
    assert jv.decide(weak, strong, arms, -0.003, 0.002)["verdict"] == "DROP"  # worse than the rerun gap
    assert jv.decide(weak, strong, arms, None, 0.002)["verdict"] == "DROP"  # no return: not measurable
    assert jv.decide(weak[:29], strong, arms, 0.01, 0.002)["verdict"] == "DROP"  # a group under 30 stocks
    assert jv.decide(strong, strong, arms, 0.01, 0.002)["verdict"] == "DROP"  # weak grounds still put out: 93% of the rerun's
    assert jv.decide(weak, weak, arms, 0.01, 0.002)["verdict"] == "DROP"  # strong grounds released: 40 points under
