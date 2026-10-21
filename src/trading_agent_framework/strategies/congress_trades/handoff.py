"""The hand-off between the three agents: submit tools, their validation and the recorder.

Each agent ends its run by calling one submit tool (`submit_holdings`, `submit_target`, `submit_trade_report`). A tool
validates its argument against what the pipeline armed the recorder with for this stage, records the first valid
submission, and otherwise returns `{"error": ...}` so the model can correct itself in the same run. Free text from an agent
is never trusted. Same pattern as `bill_ackman/handoff.py`.

The rules that matter for the money are enforced here, in code: a holding in a higher value tier never gets a smaller weight than one in a lower tier,
weights stay within the caps, and a trade report is accepted only when it names every order the desk placed and every
order has been checked.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema from
the function's real annotations (as `memory/tools.py` does).
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from trading_agent_framework.strategies.common import handoff as shared
from trading_agent_framework.strategies.common.handoff import HandoffError, OrderView, validate_report
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams

_EPS = 1e-9

HOLDINGS, TARGET, REPORT = "holdings", "target", "report"
_TOOL_NAMES = {HOLDINGS: "submit_holdings", TARGET: "submit_target", REPORT: "submit_trade_report"}


@dataclass(frozen=True, slots=True)
class HoldingSubmission:
    ticker: str
    value_low: Decimal
    value_high: Decimal
    reason: str | None
    dropped: bool = False


@dataclass(frozen=True, slots=True)
class TargetPosition:
    ticker: str
    weight: float  # a fraction of portfolio value; 0 means the holding is dropped
    reason: str


# --- field readers (pure; the shared ones are in `common/handoff.py`) ---------------------------------


def _money(item: Mapping[str, Any], field: str, ticker: str) -> Decimal:
    value = item.get(field)
    if isinstance(value, bool) or value is None:
        raise HandoffError(f"{ticker} needs {field}: a number of dollars")
    try:
        amount = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        raise HandoffError(f"{field} for {ticker} must be a number of dollars") from None
    if not amount.is_finite() or amount < 0:
        raise HandoffError(f"{field} for {ticker} must be a finite, non-negative number of dollars")
    return amount


# --- validation (pure) ------------------------------------------------------------------------------


def validate_holdings(
    raw: Any,
    *,
    known_tickers: Collection[str],
    baseline: Mapping[str, tuple[Decimal, Decimal]],
    max_holdings: int,
    reason_max_chars: int,
) -> list[HoldingSubmission]:
    """What she owns today, as the research agent concludes.

    Every baseline holding (ticker -> (value_low, value_high), computed by code) must appear, either kept or dropped with
    `drop: true` and a reason. A kept holding that differs from the baseline, or is not in it, needs a reason. Every kept
    ticker must be one that appears in a known filing. At most `max_holdings` are kept.
    """
    entries = shared.items(raw, "holdings")
    known = set(known_tickers)
    seen: set[str] = set()
    submissions = []
    for index, item in enumerate(entries, start=1):
        ticker = shared.key(item, "ticker", index)
        if ticker in seen:
            raise HandoffError(f"{ticker} appears twice")
        seen.add(ticker)
        drop = item.get("drop", False)
        if not isinstance(drop, bool):
            raise HandoffError(f"drop for {ticker} must be true or false")
        if drop:
            if ticker not in baseline:
                raise HandoffError(f"{ticker} is not in the baseline holdings, so there is nothing to drop: leave it out")
            reason = shared.text(item, f"dropping {ticker}", reason_max_chars, required=True)
            low, high = baseline[ticker]
            submissions.append(HoldingSubmission(ticker, low, high, reason, dropped=True))
            continue
        if ticker not in known:
            raise HandoffError(f"{ticker} does not appear in any known filing")
        low, high = _money(item, "value_low", ticker), _money(item, "value_high", ticker)
        if high <= 0 or low > high:
            raise HandoffError(f"{ticker}: value_low {low} and value_high {high} must satisfy 0 <= value_low <= value_high and value_high > 0")
        changed = baseline.get(ticker) != (low, high)
        reason = shared.text(item, f"{ticker} (it differs from the baseline)" if changed else ticker, reason_max_chars, required=changed)
        submissions.append(HoldingSubmission(ticker, low, high, reason))
    missing = sorted(set(baseline) - seen)
    if missing:
        raise HandoffError(f"no entry for {', '.join(missing)}: keep each baseline holding (with its value) or drop it with drop true and a reason")
    kept = sum(1 for s in submissions if not s.dropped)
    if kept > max_holdings:
        raise HandoffError(f"submit at most {max_holdings} holdings, got {kept}: drop the smallest")
    return submissions


def validate_target(
    raw: Any,
    *,
    holdings: Mapping[str, int],
    max_positions: int,
    min_weight: float,
    max_weight: float,
    max_total_weight: float,
    reason_max_chars: int,
) -> list[TargetPosition]:
    """One entry per holding (ticker -> value tier): a weight in [min_weight, max_weight], or 0 to drop it with a reason.

    The weights sum to at most `max_total_weight`, at most `max_positions` are held, and a holding in a higher tier never
    has a smaller weight than one in a lower tier.
    """
    entries = shared.items(raw, "positions")
    seen: set[str] = set()
    positions = []
    for index, item in enumerate(entries, start=1):
        ticker = shared.key(item, "ticker", index)
        if ticker not in holdings:
            raise HandoffError(f"{ticker} is not one of the holdings ({', '.join(sorted(holdings)) or 'there are none: submit an empty list'})")
        if ticker in seen:
            raise HandoffError(f"{ticker} appears twice")
        seen.add(ticker)
        weight = shared.weight(item, ticker)
        if weight < -_EPS:
            raise HandoffError(f"the weight for {ticker} is {weight:.4f}: it cannot be negative")
        reason = shared.text(item, ticker, reason_max_chars, required=True)
        assert reason is not None
        if weight <= _EPS:
            weight = 0.0
        elif not min_weight - _EPS <= weight <= max_weight + _EPS:
            raise HandoffError(f"the weight for {ticker} is {weight:.4f}: it must be between {min_weight} and {max_weight}, or 0 to drop it")
        positions.append(TargetPosition(ticker, weight, reason))
    missing = sorted(set(holdings) - seen)
    if missing:
        raise HandoffError(f"no entry for {', '.join(missing)}: give each holding a weight, or 0 with a reason to drop it")
    kept = [position for position in positions if position.weight > 0]
    if len(kept) > max_positions:
        raise HandoffError(f"hold at most {max_positions} positions, got {len(kept)}: drop the smallest with weight 0 and a reason")
    total = sum(position.weight for position in kept)
    if total > max_total_weight + _EPS:
        raise HandoffError(f"the weights sum to {total:.4f}: they must sum to at most {max_total_weight:.2f}")
    _check_tier_order(kept, holdings)
    return positions


def _check_tier_order(kept: list[TargetPosition], tiers: Mapping[str, int]) -> None:
    """A holding in a higher tier must not be given a smaller weight than one in a lower tier."""
    by_tier: dict[int, list[TargetPosition]] = {}
    for position in kept:
        by_tier.setdefault(tiers[position.ticker], []).append(position)
    ordered = sorted(by_tier)
    for index, high_tier in enumerate(ordered):
        smallest_high = min(by_tier[high_tier], key=lambda p: p.weight)
        for low_tier in ordered[:index]:
            biggest_low = max(by_tier[low_tier], key=lambda p: p.weight)
            if smallest_high.weight < biggest_low.weight - _EPS:
                raise HandoffError(
                    f"{smallest_high.ticker} (value tier {high_tier}) has weight {smallest_high.weight:.4f}, smaller than {biggest_low.ticker} (lower value tier {low_tier}) "
                    f"at {biggest_low.weight:.4f}: a holding in a higher tier must not get a smaller weight"
                )


# --- the recorder -----------------------------------------------------------------------------------


class HandoffRecorder:
    """Holds what the current stage must validate against, and the first valid submission for it."""

    def __init__(self, params: CongressParams) -> None:
        self._params = params
        self._stage: str | None = None
        self._context: dict[str, Any] = {}
        self._submission: Any = None
        self._submitted = False
        self.last_error: str | None = None

    def expect_holdings(self, known_tickers: Collection[str], baseline: Mapping[str, tuple[Decimal, Decimal]]) -> None:
        self._arm(HOLDINGS, {"known_tickers": set(known_tickers), "baseline": dict(baseline)})

    def expect_target(self, holdings: Mapping[str, int]) -> None:
        """`holdings` maps each ticker the portfolio agent must place to its value tier."""
        self._arm(TARGET, {"holdings": dict(holdings)})

    def expect_report(self, orders: Callable[[], Mapping[str, OrderView]]) -> None:
        """`orders` returns the desk's orders as they are NOW (called at submission time)."""
        self._arm(REPORT, {"orders": orders})

    def _arm(self, stage: str, context: dict[str, Any]) -> None:
        self._stage, self._context = stage, context
        self._submission, self._submitted, self.last_error = None, False, None

    @property
    def submitted(self) -> bool:
        """Whether the armed stage has a valid submission."""
        return self._submitted

    @property
    def submission(self) -> Any:
        """The armed stage's valid submission (a list of `HoldingSubmission`, `TargetPosition` or `OrderReport`), else None."""
        return self._submission

    def submit(self, stage: str, raw: Any) -> dict[str, Any]:
        """What a submit tool returns: `{"status": "recorded"}` or `{"error": ...}` (recorded in `last_error`)."""
        if self._stage != stage:
            return self._fail(f"{_TOOL_NAMES[stage]} is not expected at this point")
        if self._submitted:
            return self._fail("already recorded for this run")
        params = self._params
        try:
            if stage == HOLDINGS:
                value = validate_holdings(
                    raw,
                    known_tickers=self._context["known_tickers"],
                    baseline=self._context["baseline"],
                    max_holdings=params.max_holdings,
                    reason_max_chars=params.reason_max_chars,
                )
            elif stage == TARGET:
                value = validate_target(
                    raw,
                    holdings=self._context["holdings"],
                    max_positions=params.max_positions,
                    min_weight=params.min_weight,
                    max_weight=params.max_position_weight,
                    max_total_weight=params.max_total_weight,
                    reason_max_chars=params.reason_max_chars,
                )
            else:
                value = validate_report(raw, orders=self._context["orders"](), reason_max_chars=params.reason_max_chars)
        except HandoffError as exc:
            return self._fail(str(exc))
        self._submission, self._submitted, self.last_error = value, True, None
        return {"status": "recorded"}

    def _fail(self, message: str) -> dict[str, Any]:
        self.last_error = message
        return {"error": message}


# --- the tools --------------------------------------------------------------------------------------


def submit_tools(recorder: HandoffRecorder) -> dict[str, Callable[..., dict[str, Any]]]:
    """The three submit tools, closures over `recorder`, keyed by name. Each agent is given only its own."""

    def submit_holdings(holdings: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit what she owns today: objects with ticker, value_low, value_high (dollars) and a reason when you changed the baseline, or drop true with a reason to remove one."""
        return recorder.submit(HOLDINGS, holdings)

    def submit_target(positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit the target mix: one object per holding with ticker, weight (fraction of portfolio value, 0 to drop it) and reason."""
        return recorder.submit(TARGET, positions)

    def submit_trade_report(orders: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit the trade report: one object per order you placed with order_id and, for an order not fully filled, a reason."""
        return recorder.submit(REPORT, orders)

    return {"submit_holdings": submit_holdings, "submit_target": submit_target, "submit_trade_report": submit_trade_report}
