from __future__ import annotations

import logging

import pytest

from trading_agent_framework.agents.tools.text import LengthGate, TextLimit, count_step, review_texts, text_tools, trim_text

LIMIT = TextLimit(stated=10, hard=20)


def _gate(attempts: int = 3) -> LengthGate:
    return LengthGate(attempts, name="submit_x", log_tag="strat")


# --- trim_text (moved from strategies/common/handoff.py) ------------------------------------------


def test_a_text_within_the_limit_is_only_stripped() -> None:
    assert trim_text("  A fact. Another.  ", max_chars=600, min_chars=250) == "A fact. Another."


def test_a_long_text_keeps_the_most_whole_sentences_that_fit() -> None:
    text = "a" * 300 + ". " + "b" * 250 + "! " + "c" * 100 + "."
    assert trim_text(text, max_chars=600, min_chars=250) == "a" * 300 + ". " + "b" * 250 + "!"


def test_a_decimal_point_is_not_a_sentence_end() -> None:
    text = "Revenue " + "r" * 280 + ". Q3 EPS $1.58 vs $1.47 est " + "e" * 300 + "."
    assert trim_text(text, max_chars=600, min_chars=250) == "Revenue " + "r" * 280 + "."


def test_with_no_long_enough_sentence_the_text_is_cut_at_a_word_with_an_ellipsis() -> None:
    text = "Short. " + "word " * 200
    trimmed = trim_text(text, max_chars=600, min_chars=250)
    assert len(trimmed) <= 600 and trimmed.endswith("word…")


def test_a_text_with_no_space_is_cut_hard_with_an_ellipsis() -> None:
    assert trim_text("x" * 700, max_chars=600, min_chars=250) == "x" * 599 + "…"


# --- count_characters ---------------------------------------------------------------------------


def test_count_characters_returns_stripped_lengths_in_order() -> None:
    (count,) = text_tools()
    assert count.__name__ == "count_characters"
    assert count(texts=["  abc ", "", "x" * 401]) == {"counts": [3, 0, 401]}


def test_past_two_counts_in_a_run_the_count_tool_says_submit_now_until_the_gate_is_reset() -> None:
    # the 2026-11-16 backtest: researchers re-counted a 707-character note against 500 until the step limit
    gate = _gate()
    (count,) = text_tools(lambda: gate)
    assert count(texts=["abc"]) == {"counts": [3]}
    assert count(texts=["ab"]) == {"counts": [2]}
    stop = {"error": "you have counted twice: call submit_x now with your texts; it tells you if one is still too long"}
    assert count(texts=["a"]) == stop
    assert count(texts=["a"]) == stop
    gate.reset()
    assert count(texts=["a"]) == {"counts": [1]}


def test_with_no_armed_gate_the_count_tool_never_stops_counting() -> None:
    (count,) = text_tools(lambda: None)
    assert [count(texts=["a"]) for _ in range(4)] == [{"counts": [1]}] * 4


# --- LengthGate ---------------------------------------------------------------------------------


def test_an_over_long_text_is_refused_naming_each_count_and_stated_limit() -> None:
    result = _gate().review([("a", "x" * 11, LIMIT), ("b", "ok", LIMIT), ("c", "y" * 15, LIMIT)])
    assert result == {
        "error": "too long: a has 11 characters (limit 10); c has 15 characters (limit 10). Count them with count_characters, shorten those texts and call submit_x again with the whole submission."
    }


def test_the_last_attempt_is_accepted_trimmed_at_the_hard_cap(caplog: pytest.LogCaptureFixture) -> None:
    gate = _gate(3)
    texts = [("a", "One two. " * 4, LIMIT), ("b", "x" * 15, LIMIT)]  # 35 and 15 characters
    assert "error" in gate.review(texts) and "error" in gate.review(texts)
    with caplog.at_level(logging.WARNING):
        kept = gate.review(texts)
    assert kept == [trim_text("One two. " * 4, max_chars=20, min_chars=5), "x" * 15]  # b: over stated, under hard: kept whole
    assert "submit_x: a trimmed from 35 to 17 characters (limit 20)" in caplog.text
    assert "[strat]" in caplog.text


def test_only_length_refusals_use_an_attempt() -> None:
    gate = _gate(2)
    assert gate.review([("a", "short", LIMIT)]) == ["short"]  # within limits: no attempt used
    assert "error" in gate.review([("a", "x" * 11, LIMIT)])
    assert gate.review([("a", "x" * 11, LIMIT)]) == ["x" * 11]


def test_reset_gives_the_attempts_back() -> None:
    gate = _gate(3)
    texts = [("a", "x" * 11, LIMIT)]
    assert "error" in gate.review(texts) and "error" in gate.review(texts)
    gate.reset()
    assert "error" in gate.review(texts) and "error" in gate.review(texts)
    assert gate.review(texts) == ["x" * 11]


def test_one_attempt_never_refuses() -> None:
    assert _gate(1).review([("a", "x" * 25, LIMIT)]) == ["x" * 19 + "…"]


def test_boundaries_are_inclusive_and_whitespace_is_not_counted() -> None:
    assert _gate().review([("a", "  " + "x" * 10 + "  ", LIMIT)]) == ["x" * 10]
    gate = _gate(1)
    assert gate.review([("a", "x" * 20, LIMIT)]) == ["x" * 20]  # exactly the hard cap is not trimmed


def test_non_strings_pass_through_untouched() -> None:
    assert _gate().review([("a", None, LIMIT), ("b", 3, LIMIT)]) == [None, 3]


def test_two_texts_with_the_same_label_are_kept_apart() -> None:
    gate = _gate(2)
    texts = [("reason for XLK, IGV", "short", LIMIT), ("reason for XLK, IGV", "x" * 15, LIMIT)]
    result = gate.review(texts)
    assert result == {"error": "too long: reason for XLK, IGV has 15 characters (limit 10). Count them with count_characters, shorten those texts and call submit_x again with the whole submission."}
    assert gate.review(texts) == ["short", "x" * 15]


def test_a_refusal_is_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        _gate().review([("a", "x" * 11, LIMIT)])
    assert any(record.levelno == logging.INFO and "submit_x: refused for length" in record.getMessage() for record in caplog.records)


def test_invalid_limits_and_attempts_raise() -> None:
    for stated, hard in ((0, 5), (10, 9)):
        with pytest.raises(ValueError):
            TextLimit(stated, hard)
    with pytest.raises(ValueError):
        LengthGate(0, name="submit_x", log_tag="strat")


def test_count_step_names_the_tool_and_the_texts() -> None:
    step = count_step("submit_x", "every `note` (10 characters at most)")
    assert "submit only texts whose last count" not in step.lower() and "tells you if a text is still too long" in step
    assert "`count_characters`" in step and "`submit_x`" in step and "every `note` (10 characters at most)" in step


# --- review_texts ---------------------------------------------------------------------------------


def test_review_texts_labels_every_field_of_every_entry_then_the_top_level_texts() -> None:
    entries = [{"s": "A", "f": "x" * 11, "g": "ok"}]
    result = review_texts(_gate(2), entries, {"f": LIMIT, "g": LIMIT}, lambda e: e["s"], {"view": ("y" * 12, LIMIT)})
    assert isinstance(result, dict)
    assert result["error"].startswith("too long: f for A has 11 characters (limit 10); view has 12 characters (limit 10). ")


def test_review_texts_skips_an_absent_or_none_field() -> None:
    entries = [{"s": "A", "f": "ok"}, {"s": "B", "f": "ok", "g": None}]
    result = review_texts(_gate(2), entries, {"f": LIMIT, "g": LIMIT}, lambda e: e["s"])
    assert result == (entries, {})


def test_review_texts_writes_trimmed_values_back_to_the_right_entry() -> None:
    long_text = "word " * 5
    entries = [{"s": "A, B", "f": long_text}, {"s": "A, B", "f": "short"}]
    result = review_texts(_gate(1), entries, {"f": LIMIT}, lambda e: e["s"])
    assert not isinstance(result, dict)
    reviewed, top = result
    assert len(reviewed[0]["f"]) <= 20
    assert reviewed[1]["f"] == "short"
    assert top == {}
    assert entries[0]["f"] == long_text


def test_review_texts_passes_malformed_entries_through() -> None:
    assert review_texts(_gate(2), "not a list", {"f": LIMIT}, str) == ("not a list", {})
    mixed = [1, {"s": "A", "f": 42}]
    assert review_texts(_gate(2), mixed, {"f": LIMIT}, lambda e: e["s"]) == (mixed, {})
