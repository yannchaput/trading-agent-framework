from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal
from functools import reduce
from operator import add
from typing import Any

import pytest

from trading_agent_framework.strategies.common.handoff import HandoffError, OrderView
from trading_agent_framework.strategies.insider_trades.form4.classify import Label, Labelled
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Trade
from trading_agent_framework.strategies.insider_trades.form4.signal import SignalClass, StockFacts, summarise
from trading_agent_framework.strategies.insider_trades.handoff import InsiderRecorder, submit_tools, validate_note, validate_target
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams

PARAMS = InsiderParams()
WATCHLIST = PARAMS.watchlist
TODAY = date(2025, 10, 14)
BUYING, SELLING, NONE = SignalClass.BUYING, SignalClass.SELLING, SignalClass.NONE


def labelled(accession: str, cik: str, label: Label, *, plan: bool = False) -> Labelled:
    trade = Form4Trade(accession, date(2025, 10, 1), "AAPL", cik, f"OWNER {cik}", "CEO", "S", date(2025, 10, 1), Decimal(100), Decimal(10), Decimal(900), plan, ())
    return Labelled(trade, label, "rule", {})


def facts_of(*trades: Labelled) -> StockFacts:
    return summarise("AAPL", list(trades), today=TODAY, window_start=date(2025, 9, 14))


FACTS = facts_of(
    labelled("A1", "0001", Label.ROUTINE),
    labelled("A2", "0001", Label.OPPORTUNISTIC),
    labelled("A3", "0002", Label.ROUTINE, plan=True),
    labelled("A4", "0003", Label.ROUTINE, plan=True),
    labelled("A5", "0003", Label.ROUTINE),
)


def override(**fields: Any) -> dict[str, Any]:
    return {"insider_id": "0001", "from_label": "routine", "to_label": "opportunistic", "reason": "the footnote says it was discretionary", **fields}


def note_of(summary: Any = "Cook sold outside his habit.", overrides: Any = (), *, facts: StockFacts = FACTS) -> Any:
    return validate_note(summary, list(overrides), facts=facts, reason_max_chars=PARAMS.reason_max_chars)


def classes_of(**by_ticker: SignalClass) -> dict[str, SignalClass]:
    return {ticker: by_ticker.get(ticker, NONE) for ticker in WATCHLIST}


def rows(weights: dict[str, float], reasons: dict[str, str] | None = None) -> list[dict[str, Any]]:
    return [{"ticker": t, "weight": w, **({"reason": reasons[t]} if reasons and t in reasons else {})} for t, w in weights.items()]


def target_of(raw: Any, classes: dict[str, SignalClass]) -> Any:
    return validate_target(raw, classes=classes, params=PARAMS)


def even() -> dict[str, float]:
    return {t: PARAMS.even_weight for t in WATCHLIST}


# --- submit_note ------------------------------------------------------------------------------------


def test_a_note_before_the_facts_were_fetched_is_refused() -> None:
    recorder = InsiderRecorder(PARAMS)
    recorder.expect_note(FACTS)

    assert recorder.submit("note", {"summary": "all quiet", "overrides": []}) == {"error": "call get_insider_trades for AAPL first"}
    assert not recorder.submitted and recorder.note is None

    recorder.facts_fetched("aapl")
    assert recorder.submit("note", {"summary": "all quiet", "overrides": []}) == {"status": "recorded"}
    assert recorder.submitted and recorder.note is not None and recorder.note.summary == "all quiet"


def test_arming_the_note_again_forgets_that_the_facts_were_fetched() -> None:
    recorder = InsiderRecorder(PARAMS)
    recorder.expect_note(FACTS)
    recorder.facts_fetched("AAPL")
    recorder.expect_note(FACTS)

    assert "call get_insider_trades for AAPL first" in recorder.submit("note", {"summary": "x", "overrides": []})["error"]


def test_a_valid_note_keeps_the_summary_and_the_overrides() -> None:
    note = note_of("  Cook sold outside his habit.  ", [override(from_label="ROUTINE", to_label=" Opportunistic ")])

    assert note.summary == "Cook sold outside his habit."
    [o] = note.overrides
    assert (o.insider_id, o.from_label, o.to_label) == ("0001", Label.ROUTINE, Label.OPPORTUNISTIC)
    assert note_of(overrides=[]).overrides == ()


@pytest.mark.parametrize("summary", [None, "", "   ", 7])
def test_a_note_needs_a_summary(summary: Any) -> None:
    with pytest.raises(HandoffError, match="summary"):
        note_of(summary)


def test_a_summary_over_the_limit_is_refused() -> None:
    with pytest.raises(HandoffError, match=f"under {PARAMS.reason_max_chars}"):
        note_of("x" * (PARAMS.reason_max_chars + 1))


def test_an_override_of_plan_only_trades_is_refused() -> None:
    with pytest.raises(HandoffError) as exc:
        note_of(overrides=[override(insider_id="0002")])

    assert str(exc.value) == "0002's routine trades are 10b5-1 plan trades: they stay routine"


def test_an_override_of_a_label_with_one_plan_and_one_other_trade_is_accepted() -> None:
    assert note_of(overrides=[override(insider_id="0003")]).overrides[0].insider_id == "0003"


def test_an_override_needs_a_reason_and_a_known_insider() -> None:
    with pytest.raises(HandoffError, match="needs a reason"):
        note_of(overrides=[override(reason=" ")])
    with pytest.raises(HandoffError, match="9999.*0001, 0002, 0003"):
        note_of(overrides=[override(insider_id="9999")])
    with pytest.raises(HandoffError, match="no insider_id"):
        note_of(overrides=[{"from_label": "routine", "to_label": "opportunistic", "reason": "r"}])


def test_an_override_needs_allowed_labels_that_differ_and_that_the_insider_has() -> None:
    with pytest.raises(HandoffError, match="routine, opportunistic, unclassified"):
        note_of(overrides=[override(to_label="boring")])
    with pytest.raises(HandoffError, match="from_label"):
        note_of(overrides=[override(from_label=None)])
    with pytest.raises(HandoffError, match="differ"):
        note_of(overrides=[override(to_label="routine")])
    with pytest.raises(HandoffError, match="0001 has no unclassified trades"):
        note_of(overrides=[override(from_label="unclassified", to_label="routine")])


def test_overrides_must_be_a_list_of_objects() -> None:
    with pytest.raises(HandoffError, match="overrides must be a list"):
        validate_note("s", "none", facts=FACTS, reason_max_chars=300)


# --- submit_target ----------------------------------------------------------------------------------


def test_ten_even_weights_with_float_noise_are_accepted() -> None:
    assert reduce(add, [0.095] * 10) != 0.95  # a running sum carries float noise the tolerance absorbs

    result = target_of(rows({t: 0.095 for t in WATCHLIST}), classes_of())

    assert [w.ticker for w in result] == list(WATCHLIST) and all(w.reason is None for w in result)


def test_a_weight_within_tolerance_of_even_needs_no_reason() -> None:
    weights = {**even(), "AAPL": 0.0954}

    assert target_of(rows(weights), classes_of())[0].weight == 0.0954


def test_tickers_are_stripped_and_upper_cased() -> None:
    raw = [{**r, "ticker": f" {r['ticker'].lower()} "} for r in rows(even())]

    assert [w.ticker for w in target_of(raw, classes_of())] == list(WATCHLIST)


def test_a_buying_stock_below_a_none_stock_is_refused() -> None:
    classes = classes_of(AAPL=BUYING)
    weights = {**{t: 0.08 for t in WATCHLIST}, "AAPL": 0.05}
    reasons = {t: "r" for t in WATCHLIST}

    with pytest.raises(HandoffError, match="AAPL.*buying.*0.0500.*0.0800"):
        target_of(rows(weights, reasons), classes)
    assert target_of(rows({**weights, "AAPL": 0.12}, reasons), classes)


def test_a_selling_stock_above_a_none_stock_is_refused() -> None:
    classes = classes_of(MSFT=SELLING)
    weights = {**{t: 0.08 for t in WATCHLIST}, "MSFT": 0.12}
    reasons = {t: "r" for t in WATCHLIST}

    with pytest.raises(HandoffError, match="MSFT.*selling.*0.1200.*0.0800"):
        target_of(rows(weights, reasons), classes)
    assert target_of(rows({**weights, "MSFT": 0.04}, reasons), classes)


def test_the_ordering_rule_is_skipped_when_a_class_is_empty() -> None:
    classes = {**classes_of(AAPL=BUYING), **{t: SELLING for t in WATCHLIST[1:]}}
    weights = {**{t: 0.05 for t in WATCHLIST}, "AAPL": 0.2}
    reasons = {t: "r" for t in WATCHLIST}

    assert len(target_of(rows(weights, reasons), classes)) == 10


def test_with_no_signal_anywhere_the_weights_must_be_even() -> None:
    weights = {**even(), "AAPL": 0.12, "MSFT": 0.07}

    with pytest.raises(HandoffError, match="0.0950"):
        target_of(rows(weights, {"AAPL": "r", "MSFT": "r"}), classes_of())


def test_a_weight_away_from_even_needs_a_reason() -> None:
    classes = classes_of(AAPL=BUYING)
    weights = {**{t: 0.08 for t in WATCHLIST}, "AAPL": 0.2}

    with pytest.raises(HandoffError, match="needs a reason"):
        target_of(rows(weights), classes)


def test_every_watchlist_stock_must_get_a_weight_between_floor_and_cap() -> None:
    classes = classes_of()
    with pytest.raises(HandoffError, match="no entry for KO"):
        target_of(rows({t: w for t, w in even().items() if t != "KO"}), classes)
    with pytest.raises(HandoffError, match="ZZZZ is not on the watchlist"):
        target_of(rows({**even(), "ZZZZ": 0.05}), classes)
    with pytest.raises(HandoffError, match="AAPL appears twice"):
        target_of([*rows(even()), {"ticker": "AAPL", "weight": 0.095}], classes)
    with pytest.raises(HandoffError, match=r"AAPL.*between 0.03 and 0.2"):
        target_of(rows({**even(), "AAPL": 0.0}), classes)
    with pytest.raises(HandoffError, match=r"AAPL.*between 0.03 and 0.2"):
        target_of(rows({**even(), "AAPL": 0.25}), classes)
    with pytest.raises(HandoffError, match="must be a number"):
        target_of([{"ticker": "AAPL", "weight": "lots"}], classes)


def test_the_weights_may_not_sum_over_the_limit() -> None:
    classes = classes_of(AAPL=BUYING)
    weights = {**{t: 0.1 for t in WATCHLIST}, "AAPL": 0.2}
    reasons = {t: "r" for t in WATCHLIST}

    with pytest.raises(HandoffError, match="sum to 1.1000.*at most 0.95"):
        target_of(rows(weights, reasons), classes)


# --- the recorder and the tools ---------------------------------------------------------------------


def order(order_id: str, *, status: str = "filled", checked: bool = True) -> OrderView:
    return OrderView(order_id, "AAPL", "buy", 10.0, status, 10.0 if status == "filled" else 0.0, checked)


def trade_recorder(classes: dict[str, SignalClass], seen: list[dict[str, float]], orders: dict[str, OrderView] | None = None) -> InsiderRecorder:
    recorder = InsiderRecorder(PARAMS)

    def on_target(weights: dict[str, float]) -> dict[str, Any]:
        seen.append(weights)
        return {"shortfalls": ["AAPL: 3 shares short"]}

    recorder.expect_trade(classes, on_target, lambda: orders or {})
    return recorder


def test_a_valid_target_arms_the_report_and_returns_what_on_target_returns() -> None:
    seen: list[dict[str, float]] = []
    recorder = trade_recorder(classes_of(), seen, {"o1": order("o1")})
    tools = submit_tools(recorder)
    assert recorder.missing_tool == "submit_target"

    weights = rows({t: 0.095 for t in WATCHLIST})
    assert tools["submit_target"](weights) == {"status": "recorded", "shortfalls": ["AAPL: 3 shares short"]}

    assert seen == [{t: 0.095 for t in WATCHLIST}]
    assert recorder.missing_tool == "submit_trade_report" and not recorder.submitted
    assert recorder.target is not None and len(recorder.target) == 10

    assert tools["submit_trade_report"]([{"order_id": "o1"}]) == {"status": "recorded"}
    assert recorder.submitted and recorder.report is not None and recorder.report[0].order_id == "o1"
    assert recorder.target is not None


def test_the_target_is_accepted_once() -> None:
    recorder = trade_recorder(classes_of(), [])
    tools = submit_tools(recorder)
    tools["submit_target"](rows(even()))

    assert tools["submit_target"](rows(even())) == {"error": "target already recorded for this run"}
    assert recorder.last_error == "target already recorded for this run"


def test_an_invalid_target_leaves_the_target_stage_armed_and_does_not_call_on_target() -> None:
    seen: list[dict[str, float]] = []
    recorder = trade_recorder(classes_of(), seen)

    bad = recorder.submit("target", rows({**even(), "AAPL": 0.25}))

    assert "error" in bad and recorder.last_error == bad["error"]
    assert seen == [] and recorder.target is None and recorder.missing_tool == "submit_target"


def test_an_on_target_that_raises_propagates() -> None:
    recorder = InsiderRecorder(PARAMS)

    def boom(weights: dict[str, float]) -> dict[str, Any]:
        raise RuntimeError("programming error")

    recorder.expect_trade(classes_of(), boom, dict)

    with pytest.raises(RuntimeError, match="programming error"):
        recorder.submit("target", rows(even()))
    assert recorder.target is None and recorder.missing_tool == "submit_target"


def test_a_report_is_validated_against_the_orders_at_submission_time() -> None:
    orders = {"o1": order("o1", checked=False)}
    recorder = trade_recorder(classes_of(), [], orders)
    recorder.submit("target", rows(even()))

    assert "has not been checked" in recorder.submit("report", [{"order_id": "o1"}])["error"]
    orders["o1"] = order("o1")
    assert recorder.submit("report", [{"order_id": "o1"}]) == {"status": "recorded"}
    assert recorder.submit("report", [{"order_id": "o1"}]) == {"error": "already recorded for this run"}


def test_a_stage_that_is_not_armed_is_refused() -> None:
    recorder = InsiderRecorder(PARAMS)
    assert "submit_note is not expected" in recorder.submit("note", {"summary": "s", "overrides": []})["error"]

    trade = trade_recorder(classes_of(), [])
    assert "submit_trade_report is not expected" in trade.submit("report", [])["error"]
    assert "submit_note is not expected" in trade.submit("note", {})["error"]


def test_expect_report_arms_the_report_only_and_keeps_the_target() -> None:
    recorder = trade_recorder(classes_of(), [])
    recorder.submit("target", rows(even()))
    recorder.expect_report(lambda: {"o2": order("o2")})

    assert recorder.missing_tool == "submit_trade_report" and not recorder.submitted and recorder.report is None
    assert recorder.target is not None
    assert recorder.submit("report", [{"order_id": "o2"}]) == {"status": "recorded"}


def test_the_tools_call_the_recorder_and_have_one_line_docstrings_and_real_annotations() -> None:
    recorder = InsiderRecorder(PARAMS)
    tools = submit_tools(recorder)

    assert set(tools) == {"submit_note", "submit_target", "submit_trade_report"}
    for name, tool in tools.items():
        assert tool.__name__ == name
        assert tool.__doc__ is not None and "\n" not in tool.__doc__.strip()
    assert list(inspect.signature(tools["submit_note"]).parameters) == ["summary", "overrides"]
    assert inspect.signature(tools["submit_note"]).parameters["overrides"].annotation == list[dict[str, Any]]
    assert inspect.signature(tools["submit_target"]).parameters["weights"].annotation == list[dict[str, Any]]
    assert inspect.signature(tools["submit_trade_report"]).parameters["orders"].annotation == list[dict[str, Any]]

    recorder.expect_note(FACTS)
    recorder.facts_fetched("AAPL")
    assert tools["submit_note"]("a summary", []) == {"status": "recorded"}
    assert recorder.note is not None and recorder.note.summary == "a summary"
