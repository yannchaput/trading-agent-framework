from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from trading_agent_framework.agents.submit import HandoffError
from trading_agent_framework.strategies.common.handoff import covering, optional_text, required_text, text, weight


def _cover(raw: Any, expected: Sequence[str] = ("AAA", "BBB", "CCC")) -> list[tuple[str, Any]]:
    return covering(raw, list(expected), what="cases", unknown=lambda k: f"{k} unknown", missing=lambda keys: f"missing {', '.join(keys)}")


def test_covering_returns_one_item_per_key_in_submission_order() -> None:
    raw = [{"symbol": " ccc "}, {"symbol": "aaa"}, {"symbol": "BBB"}]
    assert [k for k, _ in _cover(raw)] == ["CCC", "AAA", "BBB"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("x", "cases must be a list of objects"),
        ([1], "cases item 1 must be an object, got int"),
        ([{"symbol": "AAA"}, {}], "item 2 has no symbol"),
        ([{"symbol": "ZZZ"}], "ZZZ unknown"),
        ([{"symbol": "AAA"}, {"symbol": "aaa"}], "AAA appears twice"),
        ([{"symbol": "CCC"}], "missing AAA, BBB"),
    ],
)
def test_covering_errors(raw: Any, message: str) -> None:
    with pytest.raises(HandoffError) as exc:
        _cover(raw)
    assert str(exc.value) == message


def test_covering_reads_the_named_field() -> None:
    result = covering([{"ticker": "aa"}], ["AA"], what="w", field="ticker", unknown=str, missing=str)
    assert result[0][0] == "AA"


def test_required_text() -> None:
    assert required_text({"argument": "  hi "}, "argument", "AAA", 10) == "hi"
    for item in ({}, {"argument": "  "}, {"argument": 3}):
        with pytest.raises(HandoffError) as exc:
            required_text(item, "argument", "AAA", 10)
        assert str(exc.value) == "AAA has no argument: give one short sentence"
    with pytest.raises(HandoffError) as exc:
        required_text({"argument": "x" * 11}, "argument", "AAA", 10)
    assert str(exc.value) == "the argument for AAA is 11 characters: keep it under 10"


def test_optional_text() -> None:
    assert optional_text({}, "what_changed", "AAA", 10) is None
    assert optional_text({"what_changed": " "}, "what_changed", "AAA", 10) is None
    assert optional_text({"what_changed": " ok "}, "what_changed", "AAA", 10) == "ok"
    with pytest.raises(HandoffError) as exc:
        optional_text({"what_changed": 3}, "what_changed", "AAA", 10)
    assert str(exc.value) == "the what_changed for AAA must be text"
    with pytest.raises(HandoffError) as exc:
        optional_text({"what_changed": "x" * 11}, "what_changed", "AAA", 10)
    assert str(exc.value) == "the what_changed for AAA is 11 characters: keep it under 10"


def test_reason_text_keeps_its_messages() -> None:
    with pytest.raises(HandoffError, match="^AAA needs a reason: give one short sentence$"):
        text({}, "AAA", 10, required=True)
    assert text({}, "AAA", 10, required=False) is None
    with pytest.raises(HandoffError, match="^the reason for AAA must be text$"):
        text({"reason": 3}, "AAA", 10, required=False)
    with pytest.raises(HandoffError, match="^the reason for AAA is 11 characters: keep it under 10$"):
        text({"reason": "x" * 11}, "AAA", 10, required=True)


def test_weight_example_is_the_callers() -> None:
    with pytest.raises(HandoffError, match="such as 0.08$"):
        weight({"weight": "x"}, "AAA")
    with pytest.raises(HandoffError, match="such as 0.25$"):
        weight({"weight": "x"}, "AAA", example="0.25")
    assert weight({"weight": "0.1"}, "AAA") == 0.1
