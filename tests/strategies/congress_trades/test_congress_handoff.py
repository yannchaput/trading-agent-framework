from __future__ import annotations

import inspect
from decimal import Decimal
from typing import Any

import pytest

from trading_agent_framework.strategies.congress_trades.handoff import (
    HandoffError,
    HandoffRecorder,
    HoldingSubmission,
    OrderReport,
    OrderView,
    TargetPosition,
    submit_tools,
    validate_holdings,
    validate_report,
    validate_target,
)
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams

PARAMS = CongressParams()
BASELINE = {"AAPL": (Decimal(5_000_001), Decimal(25_000_000)), "BE": (Decimal(1_500_002), Decimal(6_000_000)), "DBX": (Decimal(250_001), Decimal(500_000))}
KNOWN = {"AAPL", "BE", "DBX", "DIS", "NVDA"}


def holdings_of(raw: Any, **overrides: Any) -> list[HoldingSubmission]:
    kwargs: dict[str, Any] = dict(known_tickers=KNOWN, baseline=BASELINE, max_holdings=PARAMS.max_holdings, reason_max_chars=PARAMS.reason_max_chars)
    return validate_holdings(raw, **{**kwargs, **overrides})


def keep(ticker: str) -> dict[str, Any]:
    low, high = BASELINE[ticker]
    return {"ticker": ticker, "value_low": int(low), "value_high": int(high)}


def all_kept() -> list[dict[str, Any]]:
    return [keep(t) for t in BASELINE]


# --- submit_holdings --------------------------------------------------------------------------------


def test_holdings_equal_to_the_baseline_need_no_reason() -> None:
    result = holdings_of(all_kept())

    assert [h.ticker for h in result] == ["AAPL", "BE", "DBX"]
    assert all(h.reason is None and not h.dropped for h in result)
    assert result[0].value_low == Decimal(5_000_001)


def test_numbers_may_arrive_as_floats_or_strings_with_commas() -> None:
    raw = [{"ticker": "aapl", "value_low": 5000001.0, "value_high": "25,000,000"}, keep("BE"), keep("DBX")]

    assert holdings_of(raw)[0].ticker == "AAPL"


def test_a_changed_value_needs_a_reason() -> None:
    raw = [{**keep("AAPL"), "value_high": 20_000_000}, keep("BE"), keep("DBX")]

    with pytest.raises(HandoffError, match="AAPL.*needs a reason"):
        holdings_of(raw)
    raw[0]["reason"] = "the October sale is larger than the estimate"
    assert holdings_of(raw)[0].reason == "the October sale is larger than the estimate"


def test_a_new_ticker_known_from_a_filing_needs_a_reason() -> None:
    raw = [*all_kept(), {"ticker": "NVDA", "value_low": 250_001, "value_high": 500_000}]

    with pytest.raises(HandoffError, match="NVDA.*needs a reason"):
        holdings_of(raw)
    raw[-1]["reason"] = "bought in a trade report the baseline could not read"
    assert [h.ticker for h in holdings_of(raw)][-1] == "NVDA"


def test_a_ticker_in_no_known_filing_is_rejected() -> None:
    raw = [*all_kept(), {"ticker": "ZZZZ", "value_low": 1, "value_high": 2, "reason": "i think so"}]

    with pytest.raises(HandoffError, match="ZZZZ does not appear in any known filing"):
        holdings_of(raw)


def test_a_baseline_holding_cannot_be_silently_omitted() -> None:
    with pytest.raises(HandoffError, match="no entry for BE"):
        holdings_of([keep("AAPL"), keep("DBX")])


def test_a_baseline_holding_can_be_dropped_with_a_reason() -> None:
    raw = [keep("AAPL"), {"ticker": "BE", "drop": True, "reason": "sold in full according to the October report"}, keep("DBX")]

    result = holdings_of(raw)

    assert [(h.ticker, h.dropped) for h in result] == [("AAPL", False), ("BE", True), ("DBX", False)]


def test_dropping_needs_a_reason_and_only_a_baseline_holding_can_be_dropped() -> None:
    with pytest.raises(HandoffError, match="dropping BE needs a reason"):
        holdings_of([keep("AAPL"), {"ticker": "BE", "drop": True}, keep("DBX")])
    with pytest.raises(HandoffError, match="nothing to drop"):
        holdings_of([*all_kept(), {"ticker": "NVDA", "drop": True, "reason": "x"}])
    with pytest.raises(HandoffError, match="drop for AAPL must be true or false"):
        holdings_of([{**keep("AAPL"), "drop": "yes"}, keep("BE"), keep("DBX")])


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"value_low": 10, "value_high": 5}, "value_low"),
        ({"value_low": -1}, "non-negative"),
        ({"value_high": 0, "value_low": 0}, "value_high"),
        ({"value_low": "lots"}, "number of dollars"),
        ({"value_high": None}, "needs value_high"),
        ({"value_high": True}, "needs value_high"),
        ({"value_high": "NaN"}, "finite"),
    ],
)
def test_bad_values_are_rejected(mutation: dict[str, Any], message: str) -> None:
    raw = [{**keep("AAPL"), "reason": "x", **mutation}, keep("BE"), keep("DBX")]

    with pytest.raises(HandoffError, match=message):
        holdings_of(raw)


def test_holdings_duplicates_shape_and_size_are_rejected() -> None:
    with pytest.raises(HandoffError, match="appears twice"):
        holdings_of([*all_kept(), keep("AAPL")])
    with pytest.raises(HandoffError, match="must be a list of objects"):
        holdings_of("AAPL, BE")
    with pytest.raises(HandoffError, match="item 1 must be an object"):
        holdings_of(["AAPL"])
    with pytest.raises(HandoffError, match="item 1 has no ticker"):
        holdings_of([{"value_low": 1, "value_high": 2}])
    with pytest.raises(HandoffError, match="at most 2 holdings"):
        holdings_of(all_kept(), max_holdings=2)


def test_a_too_long_reason_is_rejected() -> None:
    raw = [{**keep("AAPL"), "value_high": 1_000_000_000, "reason": "x" * 301}, keep("BE"), keep("DBX")]

    with pytest.raises(HandoffError, match="keep it under 300"):
        holdings_of(raw)


def test_no_baseline_and_nothing_submitted_is_valid() -> None:
    assert holdings_of([], baseline={}) == []


# --- submit_target ----------------------------------------------------------------------------------

TIERS = {"AAPL": 8, "BE": 7, "DBX": 5}


def target_of(raw: Any, **overrides: Any) -> list[TargetPosition]:
    kwargs: dict[str, Any] = dict(
        holdings=TIERS,
        max_positions=PARAMS.max_positions,
        min_weight=PARAMS.min_weight,
        max_weight=PARAMS.max_position_weight,
        max_total_weight=PARAMS.max_total_weight,
        reason_max_chars=PARAMS.reason_max_chars,
    )
    return validate_target(raw, **{**kwargs, **overrides})


def row(ticker: str, weight: Any, reason: str = "by tier") -> dict[str, Any]:
    return {"ticker": ticker, "weight": weight, "reason": reason}


def test_a_target_in_tier_order_is_accepted() -> None:
    result = target_of([row("AAPL", 0.15), row("BE", 0.10), row("DBX", 0.05)])

    assert [(p.ticker, p.weight) for p in result] == [("AAPL", 0.15), ("BE", 0.1), ("DBX", 0.05)]


def test_a_higher_tier_holding_with_a_smaller_weight_is_rejected() -> None:
    with pytest.raises(HandoffError, match=r"BE \(value tier 7\).*smaller than DBX \(lower value tier 5\)"):
        target_of([row("AAPL", 0.15), row("BE", 0.03), row("DBX", 0.05)])


def test_equal_weights_across_tiers_are_accepted_for_example_when_capped() -> None:
    target_of([row("AAPL", 0.15), row("BE", 0.15), row("DBX", 0.15)], max_total_weight=0.95)


def test_the_tier_rule_does_not_apply_to_a_dropped_holding_but_it_needs_a_reason() -> None:
    result = target_of([row("AAPL", 0.15), row("BE", 0, "not tradable on this broker"), row("DBX", 0.05)])

    assert [p.weight for p in result] == [0.15, 0.0, 0.05]


def test_a_weight_of_zero_still_needs_a_reason_and_so_does_every_other_entry() -> None:
    with pytest.raises(HandoffError, match="BE needs a reason"):
        target_of([row("AAPL", 0.15), {"ticker": "BE", "weight": 0}, row("DBX", 0.05)])
    with pytest.raises(HandoffError, match="AAPL needs a reason"):
        target_of([{"ticker": "AAPL", "weight": 0.15}, row("BE", 0.1), row("DBX", 0.05)])


@pytest.mark.parametrize(
    "weight, message",
    [
        (0.005, "between 0.01 and 0.15"),
        (0.16, "between 0.01 and 0.15"),
        (-0.05, "cannot be negative"),
        ("heavy", "must be a number"),
        (True, "must be a number"),
        (float("nan"), "finite"),
        (None, "must be a number"),
    ],
)
def test_a_weight_out_of_range_or_not_a_number_is_rejected(weight: Any, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        target_of([row("AAPL", weight), row("BE", 0.1), row("DBX", 0.05)])


def test_weights_summing_over_the_total_cap_are_rejected() -> None:
    with pytest.raises(HandoffError, match="sum to at most 0.30"):
        target_of([row("AAPL", 0.15), row("BE", 0.15), row("DBX", 0.15)], max_total_weight=0.30)


def test_a_holding_cannot_be_left_out() -> None:
    with pytest.raises(HandoffError, match="no entry for DBX"):
        target_of([row("AAPL", 0.15), row("BE", 0.10)])


def test_a_ticker_that_is_not_a_holding_or_is_repeated_is_rejected() -> None:
    with pytest.raises(HandoffError, match="NVDA is not one of the holdings"):
        target_of([*[row(t, 0.05) for t in TIERS], row("NVDA", 0.05)])
    with pytest.raises(HandoffError, match="appears twice"):
        target_of([row("AAPL", 0.15), row("AAPL", 0.15), row("BE", 0.1), row("DBX", 0.05)])


def test_more_positions_than_allowed_are_rejected_but_dropped_ones_do_not_count() -> None:
    with pytest.raises(HandoffError, match="at most 2 positions, got 3"):
        target_of([row("AAPL", 0.15), row("BE", 0.1), row("DBX", 0.05)], max_positions=2)
    target_of([row("AAPL", 0.15), row("BE", 0.1), row("DBX", 0, "over the position limit")], max_positions=2)


def test_no_holdings_and_an_empty_list_is_valid_and_so_is_dropping_everything() -> None:
    assert target_of([], holdings={}) == []
    assert all(p.weight == 0 for p in target_of([row(t, 0, "hold cash today") for t in TIERS]))


def test_within_a_tier_weights_may_differ() -> None:
    target_of([row("AAPL", 0.15), row("BE", 0.10), row("DBX", 0.05)], holdings={"AAPL": 8, "BE": 8, "DBX": 5})
    with pytest.raises(HandoffError, match="higher tier"):
        target_of([row("AAPL", 0.05), row("BE", 0.10), row("DBX", 0.07)], holdings={"AAPL": 8, "BE": 8, "DBX": 5})


# --- submit_trade_report ----------------------------------------------------------------------------


def order(order_id: str, *, status: str = "filled", checked: bool = True, symbol: str = "AAPL", side: str = "buy") -> OrderView:
    return OrderView(order_id, symbol, side, 10.0, status, 10.0 if status == "filled" else 0.0, checked)


def report_of(raw: Any, orders: dict[str, OrderView]) -> list[OrderReport]:
    return validate_report(raw, orders=orders, reason_max_chars=PARAMS.reason_max_chars)


def test_a_report_naming_every_checked_filled_order_is_accepted() -> None:
    orders = {"o1": order("o1"), "o2": order("o2", symbol="BE")}

    result = report_of([{"order_id": "o1"}, {"order_id": "o2"}], orders)

    assert [r.order_id for r in result] == ["o1", "o2"]
    assert all(r.reason is None for r in result)


def test_an_order_that_was_not_checked_is_rejected() -> None:
    with pytest.raises(HandoffError, match="has not been checked: call check_orders first"):
        report_of([{"order_id": "o1"}], {"o1": order("o1", checked=False)})


@pytest.mark.parametrize("status", ["working", "partially_filled", "canceled", "rejected"])
def test_an_order_not_fully_filled_is_accepted_only_with_a_reason(status: str) -> None:
    orders = {"o1": order("o1", status=status)}

    with pytest.raises(HandoffError, match=f"which is {status}.*needs a reason"):
        report_of([{"order_id": "o1"}], orders)
    assert report_of([{"order_id": "o1", "reason": "the broker refused it"}], orders)[0].reason == "the broker refused it"


def test_a_report_must_name_every_order_once_and_only_orders_of_this_run() -> None:
    orders = {"o1": order("o1"), "o2": order("o2")}

    with pytest.raises(HandoffError, match="no entry for o2"):
        report_of([{"order_id": "o1"}], orders)
    with pytest.raises(HandoffError, match="appears twice"):
        report_of([{"order_id": "o1"}, {"order_id": "o1"}, {"order_id": "o2"}], orders)
    with pytest.raises(HandoffError, match="o9 is not an order you placed"):
        report_of([{"order_id": "o9"}], orders)
    with pytest.raises(HandoffError, match="item 1 has no order_id"):
        report_of([{"reason": "x"}], orders)


def test_an_empty_report_is_valid_only_when_no_order_was_placed() -> None:
    assert report_of([], {}) == []
    with pytest.raises(HandoffError, match="no entry for o1"):
        report_of([], {"o1": order("o1")})


# --- the recorder and the tools ---------------------------------------------------------------------


def test_the_recorder_records_the_first_valid_submission_and_refuses_a_second() -> None:
    recorder = HandoffRecorder(PARAMS)
    recorder.expect_holdings(KNOWN, BASELINE)

    assert recorder.submit("holdings", all_kept()) == {"status": "recorded"}
    assert recorder.submitted
    assert [h.ticker for h in recorder.submission] == ["AAPL", "BE", "DBX"]
    assert recorder.submit("holdings", all_kept()) == {"error": "already recorded for this run"}
    assert [h.ticker for h in recorder.submission] == ["AAPL", "BE", "DBX"]


def test_an_invalid_submission_returns_an_error_and_can_be_corrected() -> None:
    recorder = HandoffRecorder(PARAMS)
    recorder.expect_target(TIERS)

    bad = recorder.submit("target", [row("AAPL", 0.15), row("BE", 0.03), row("DBX", 0.05)])
    assert "error" in bad and recorder.last_error == bad["error"] and not recorder.submitted
    assert recorder.submission is None

    assert recorder.submit("target", [row("AAPL", 0.15), row("BE", 0.1), row("DBX", 0.05)]) == {"status": "recorded"}
    assert recorder.last_error is None


def test_a_submit_tool_for_another_stage_is_refused() -> None:
    recorder = HandoffRecorder(PARAMS)

    assert "not expected" in recorder.submit("target", [])["error"]
    recorder.expect_holdings(KNOWN, BASELINE)
    assert "submit_trade_report is not expected" in recorder.submit("report", [])["error"]


def test_arming_a_stage_clears_the_previous_submission() -> None:
    recorder = HandoffRecorder(PARAMS)
    recorder.expect_holdings(KNOWN, BASELINE)
    recorder.submit("holdings", all_kept())

    recorder.expect_target(TIERS)

    assert not recorder.submitted and recorder.submission is None and recorder.last_error is None


def test_the_report_is_validated_against_the_orders_at_submission_time() -> None:
    orders = {"o1": order("o1", status="working")}
    recorder = HandoffRecorder(PARAMS)
    recorder.expect_report(lambda: orders)

    assert "error" in recorder.submit("report", [{"order_id": "o1"}])
    orders["o1"] = order("o1")  # the order filled in the meantime
    assert recorder.submit("report", [{"order_id": "o1"}]) == {"status": "recorded"}


def test_the_tools_call_the_recorder_and_have_one_line_docstrings_and_real_annotations() -> None:
    recorder = HandoffRecorder(PARAMS)
    tools = submit_tools(recorder)

    assert set(tools) == {"submit_holdings", "submit_target", "submit_trade_report"}
    for name, tool in tools.items():
        assert tool.__name__ == name
        assert tool.__doc__ is not None and "\n" not in tool.__doc__.strip()
    assert inspect.signature(tools["submit_target"]).parameters["positions"].annotation == list[dict[str, Any]]
    recorder.expect_target({})
    assert tools["submit_target"]([]) == {"status": "recorded"}
