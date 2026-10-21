"""The hand-off between the insider_trades agents: submit tools, their validation and the recorder.

The researcher ends its run (one per changed stock that has trades in its window) with `submit_note`; the trader submits the
target (`submit_target`) and then its report (`submit_trade_report`). A tool validates its argument in code against what the pipeline armed the recorder with, records the
first valid submission, and otherwise returns `{"error": ...}` so the model can correct itself in the same run. Free text from an
agent is never trusted. Same pattern as `congress_trades/handoff.py`.

The rules that matter for the money are enforced here: the target names every watchlist stock exactly once, weights stay within the
floor and the cap, a stock the code classed as buying never gets less than one with no signal and a selling one never more, and
with no signal anywhere the book is split evenly. The researcher can only relabel an insider's trades through an override that
names a trade it may change (a 10b5-1 plan trade stays routine), after reading that stock's facts.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema from the
function's real annotations (as `memory/tools.py` does).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trading_agent_framework.strategies.common import handoff as shared
from trading_agent_framework.strategies.common.handoff import HandoffError, OrderReport, OrderView, validate_report
from trading_agent_framework.strategies.insider_trades.form4.classify import Label
from trading_agent_framework.strategies.insider_trades.form4.signal import Override, SignalClass, StockFacts
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams

NOTE, TARGET, REPORT = "note", "target", "report"
_TOOL_NAMES = {NOTE: "submit_note", TARGET: "submit_target", REPORT: "submit_trade_report"}


@dataclass(frozen=True, slots=True)
class Note:
    summary: str
    overrides: tuple[Override, ...]


@dataclass(frozen=True, slots=True)
class TargetWeight:
    ticker: str
    weight: float  # a fraction of portfolio value
    reason: str | None


# --- validation (pure) ------------------------------------------------------------------------------


def _label(item: Mapping[str, Any], field: str, who: str) -> Label:
    value = item.get(field)
    allowed = ", ".join(label.value for label in Label)
    if not isinstance(value, str):
        raise HandoffError(f"{field} for {who} must be one of: {allowed}")
    try:
        return Label(value.strip().lower())
    except ValueError:
        raise HandoffError(f"{field} for {who} is {value.strip()!r}: it must be one of: {allowed}") from None


def validate_note(summary: Any, overrides: Any, *, facts: StockFacts, reason_max_chars: int) -> Note:
    """A one-paragraph summary of the stock's insider activity, and the overrides the researcher wants.

    An override relabels one insider's trades from `from_label` to `to_label`: the insider must be in the facts, must have trades
    of `from_label` that are not 10b5-1 plan trades (a plan fixed the trade in advance, so it stays routine), and the labels differ.
    """
    if not isinstance(summary, str) or not summary.strip():
        raise HandoffError("the summary must be text: one short paragraph on the insider activity")
    summary = summary.strip()
    if len(summary) > reason_max_chars:
        raise HandoffError(f"the summary is {len(summary)} characters: keep it under {reason_max_chars}")
    by_id = {insider.insider_id: insider for insider in facts.insiders}
    result = []
    for index, item in enumerate(shared.items(overrides, "overrides"), start=1):
        insider_id = shared.key(item, "insider_id", index)
        insider = by_id.get(insider_id)
        if insider is None:
            raise HandoffError(f"{insider_id} is not an insider of {facts.ticker} in the window ({', '.join(sorted(by_id)) or 'there are none: submit an empty list'})")
        from_label, to_label = _label(item, "from_label", insider_id), _label(item, "to_label", insider_id)
        if to_label is from_label:
            raise HandoffError(f"the override for {insider_id} must change the label: to_label must differ from from_label ({from_label.value})")
        trades = [lt.trade for lt in insider.labelled if lt.label is from_label]
        if not trades:
            have = ", ".join(sorted({lt.label.value for lt in insider.labelled}))
            raise HandoffError(f"{insider_id} has no {from_label.value} trades in the window (labels it has: {have})")
        if all(trade.plan for trade in trades):
            raise HandoffError(f"{insider_id}'s {from_label.value} trades are 10b5-1 plan trades: they stay routine")
        reason = shared.text(item, f"the override for {insider_id}", reason_max_chars, required=True)
        assert reason is not None
        result.append(Override(insider_id, from_label, to_label, reason))
    return Note(summary, tuple(result))


def validate_target(raw: Any, *, classes: Mapping[str, SignalClass], params: InsiderParams) -> list[TargetWeight]:
    """One weight per watchlist stock (`classes` keys), each in [min_weight, max_position_weight], summing to at most max_total_weight.

    A buying stock never gets less than a stock with no signal and a selling one never more; with no signal anywhere every weight is
    the even weight. A weight that is not the even weight needs a reason. Limits give `weight_tolerance` of slack (float noise).
    """
    tol = params.weight_tolerance
    floor, cap, even = params.min_weight, params.max_position_weight, params.even_weight
    entries = shared.items(raw, "weights")
    watchlist = ", ".join(classes)
    rows: dict[str, tuple[float, Mapping[str, Any]]] = {}
    for index, item in enumerate(entries, start=1):
        ticker = shared.key(item, "ticker", index)
        if ticker not in classes:
            raise HandoffError(f"{ticker} is not on the watchlist ({watchlist})")
        if ticker in rows:
            raise HandoffError(f"{ticker} appears twice")
        weight = shared.weight(item, ticker)
        if not floor - tol <= weight <= cap + tol:
            raise HandoffError(f"the weight for {ticker} is {weight:.4f}: it must be between {floor:g} and {cap:g}")
        rows[ticker] = (weight, item)
    missing = [ticker for ticker in classes if ticker not in rows]
    if missing:
        raise HandoffError(f"no entry for {', '.join(missing)}: give every watchlist stock a weight ({watchlist})")
    total = sum(weight for weight, _ in rows.values())
    if total > params.max_total_weight + tol:
        raise HandoffError(f"the weights sum to {total:.4f}: they must sum to at most {params.max_total_weight:g}")
    _check_signal_order(rows, classes, tol)
    if all(signal is SignalClass.NONE for signal in classes.values()):
        off = [ticker for ticker, (weight, _) in rows.items() if abs(weight - even) > tol]
        if off:
            raise HandoffError(f"no stock has a signal, so every weight must be the even weight {even:.4f}: {', '.join(off)} differ")
    targets = []
    for ticker, (weight, item) in rows.items():
        reason = shared.text(item, f"{ticker} (its weight is not the even weight {even:.4f})" if abs(weight - even) > tol else ticker, params.reason_max_chars, required=abs(weight - even) > tol)
        targets.append(TargetWeight(ticker, weight, reason))
    return targets


def _check_signal_order(rows: Mapping[str, tuple[float, Mapping[str, Any]]], classes: Mapping[str, SignalClass], tol: float) -> None:
    """A buying stock must not get less than a stock with no signal, a selling one not more, and a buying one not less than a selling one."""
    weights = {signal: {ticker: rows[ticker][0] for ticker in rows if classes[ticker] is signal} for signal in SignalClass}
    none = weights[SignalClass.NONE]
    if weights[SignalClass.BUYING] and none:
        low = min(weights[SignalClass.BUYING], key=lambda t: weights[SignalClass.BUYING][t])
        high = max(none, key=lambda t: none[t])
        if weights[SignalClass.BUYING][low] < none[high] - tol:
            raise HandoffError(
                f"{low} (buying) has weight {weights[SignalClass.BUYING][low]:.4f}, smaller than {high} (no signal) at {none[high]:.4f}: a buying stock must not get less than a stock with no signal"
            )
    if weights[SignalClass.SELLING] and none:
        high = max(weights[SignalClass.SELLING], key=lambda t: weights[SignalClass.SELLING][t])
        low = min(none, key=lambda t: none[t])
        if weights[SignalClass.SELLING][high] > none[low] + tol:
            raise HandoffError(
                f"{high} (selling) has weight {weights[SignalClass.SELLING][high]:.4f}, larger than {low} (no signal) at {none[low]:.4f}: a selling stock must not get more than a stock with no signal"
            )
    buying, selling = weights[SignalClass.BUYING], weights[SignalClass.SELLING]
    if buying and selling:  # implied by the two checks above when a stock has no signal, but not when none does
        low = min(buying, key=lambda t: buying[t])
        high = max(selling, key=lambda t: selling[t])
        if buying[low] < selling[high] - tol:
            raise HandoffError(f"{low} (buying) has weight {buying[low]:.4f}, smaller than {high} (selling) at {selling[high]:.4f}: a buying stock must not get less than a selling stock")


# --- the recorder -----------------------------------------------------------------------------------


class InsiderRecorder:
    """Holds what the current stage must validate against, and the first valid submission for it.

    `expect_note` arms the researcher's note for one stock; `expect_trade` arms the target, and a valid target arms the report
    itself; `expect_report` arms the report alone (a later tick retrying unfilled orders). `submitted` is true once the armed stage's
    FINAL submission is in: the note, or the report. `target` stays readable after the report.
    """

    def __init__(self, params: InsiderParams) -> None:
        self._params = params
        self._stage: str | None = None
        self._context: dict[str, Any] = {}
        self._submitted = False
        self._fetched: set[str] = set()
        self.note: Note | None = None
        self.target: list[TargetWeight] | None = None
        self.report: list[OrderReport] | None = None
        self.last_error: str | None = None

    def expect_note(self, facts: StockFacts) -> None:
        self._fetched = set()
        self.note = None
        self._arm(NOTE, {"facts": facts})

    def facts_fetched(self, ticker: str) -> None:
        """The research tools' `on_fetch`: the researcher has read this stock's trades."""
        self._fetched.add(ticker.strip().upper())

    def expect_trade(
        self,
        classes: Mapping[str, SignalClass],
        on_target: Callable[[dict[str, float]], dict[str, Any]],
        orders: Callable[[], Mapping[str, OrderView]],
    ) -> None:
        """`on_target(weights)` follows a valid target and returns what the tool adds to its answer; `orders` is the desk's orders NOW."""
        self.target, self.report = None, None
        self._arm(TARGET, {"classes": dict(classes), "on_target": on_target, "orders": orders})

    def expect_report(self, orders: Callable[[], Mapping[str, OrderView]]) -> None:
        """`orders` returns the desk's orders as they are NOW (called at submission time)."""
        self.report = None
        self._arm(REPORT, {"orders": orders})

    def _arm(self, stage: str, context: dict[str, Any]) -> None:
        self._stage, self._context = stage, context
        self._submitted, self.last_error = False, None

    @property
    def submitted(self) -> bool:
        """Whether the armed stage's final submission (the note, or the report) is in."""
        return self._submitted

    @property
    def missing_tool(self) -> str:
        """The tool the armed stage still waits for: what a forced retry forces."""
        return _TOOL_NAMES.get(self._stage or "", "")

    def submit(self, stage: str, raw: Any) -> dict[str, Any]:
        """What a submit tool returns: `{"status": "recorded"}` or `{"error": ...}` (recorded in `last_error`).

        `raw` is `{"summary", "overrides"}` for the note, the weights list for the target, the orders list for the report.
        """
        if stage == TARGET and self.target is not None:
            return self._fail("target already recorded for this run")
        if self._stage != stage:
            return self._fail(f"{_TOOL_NAMES[stage]} is not expected at this point")
        if self._submitted:
            return self._fail("already recorded for this run")
        try:
            if stage == NOTE:
                return self._submit_note(raw)
            if stage == TARGET:
                return self._submit_target(raw)
            self.report = validate_report(raw, orders=self._context["orders"](), reason_max_chars=self._params.reason_max_chars)
        except HandoffError as exc:
            return self._fail(str(exc))
        self._submitted, self.last_error = True, None
        return {"status": "recorded"}

    def _submit_note(self, raw: Any) -> dict[str, Any]:
        facts: StockFacts = self._context["facts"]
        if facts.ticker.upper() not in self._fetched:
            raise HandoffError(f"call get_insider_trades for {facts.ticker} first")
        if not isinstance(raw, Mapping):
            raise HandoffError("the note needs a summary and overrides")
        self.note = validate_note(raw.get("summary"), raw.get("overrides"), facts=facts, reason_max_chars=self._params.reason_max_chars)
        self._submitted, self.last_error = True, None
        return {"status": "recorded"}

    def _submit_target(self, raw: Any) -> dict[str, Any]:
        target = validate_target(raw, classes=self._context["classes"], params=self._params)
        extra = self._context["on_target"]({t.ticker: t.weight for t in target})
        self.target = target
        self._arm(REPORT, {"orders": self._context["orders"]})
        return {"status": "recorded", **extra}

    def _fail(self, message: str) -> dict[str, Any]:
        self.last_error = message
        return {"error": message}


# --- the tools --------------------------------------------------------------------------------------


def submit_tools(recorder: InsiderRecorder) -> dict[str, Callable[..., dict[str, Any]]]:
    """The three submit tools, closures over `recorder`, keyed by name. Each agent is given only its own."""

    def submit_note(summary: str, overrides: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit your note on this stock: a short summary of its insider activity, and overrides (objects with insider_id, from_label, to_label and a reason) or an empty list."""
        return recorder.submit(NOTE, {"summary": summary, "overrides": overrides})

    def submit_target(weights: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit the target mix: one object per watchlist stock with ticker, weight (fraction of portfolio value) and a reason when the weight is not the even weight."""
        return recorder.submit(TARGET, weights)

    def submit_trade_report(orders: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit the trade report: one object per order you placed with order_id and, for an order not fully filled, a reason."""
        return recorder.submit(REPORT, orders)

    return {"submit_note": submit_note, "submit_target": submit_target, "submit_trade_report": submit_trade_report}
