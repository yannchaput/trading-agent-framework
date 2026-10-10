"""The hand-off between the four agents: submit tools, their validation and the recorder.

Each agent ends its run by calling one submit tool. A tool validates its argument against what the pipeline armed
the recorder with for this stage, records the first valid submission, and otherwise returns `{"error": ...}` so the
model can correct itself in the same run. Free text from an agent is never trusted.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema
from the function's real annotations (as `bill_ackman/handoff.py` and `memory/tools.py` do).
"""

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams

LEVELS = ("low", "medium", "high")
CONCERNS = ("valuation", "momentum_exhaustion", "earnings", "fundamentals", "news_event", "sector", "none")

NOTE, BULL, BEAR, PICKS = "note", "bull", "bear", "picks"
_TOOL_NAMES = {NOTE: "submit_note", BULL: "submit_bull_case", BEAR: "submit_bear_case", PICKS: "submit_picks"}


class HandoffError(ValueError):
    """A submission that breaks a rule; the message tells the model what is wrong and what is allowed."""


@dataclass(frozen=True, slots=True)
class Note:
    symbol: str
    note: str


@dataclass(frozen=True, slots=True)
class BullCase:
    symbol: str
    conviction: str  # one of LEVELS
    argument: str


@dataclass(frozen=True, slots=True)
class BearCase:
    symbol: str
    risk: str  # one of LEVELS
    concern: str  # one of CONCERNS
    argument: str


@dataclass(frozen=True, slots=True)
class Choice:
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class Picks:
    picks: tuple[Choice, ...]  # in the judge's order: the buy order
    drops: tuple[Choice, ...]  # the held debate stocks not picked


# --- validation (pure) --------------------------------------------------------------------------


def _items(raw: Any, what: str) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list | tuple):
        raise HandoffError(f"{what} must be a list of objects")
    items = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise HandoffError(f"{what} item {index} must be an object, got {type(item).__name__}")
        items.append(item)
    return items


def _symbol(item: Mapping[str, Any], index: int) -> str:
    value = item.get("symbol")
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"item {index} has no symbol")
    return value.strip().upper()


def _text(item: Mapping[str, Any], key: str, symbol: str, max_chars: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"{symbol} has no {key}: give one short sentence")
    text = value.strip()
    if len(text) > max_chars:
        raise HandoffError(f"the {key} for {symbol} is {len(text)} characters: keep it under {max_chars}")
    return text


def _one_of(item: Mapping[str, Any], key: str, symbol: str, allowed: Sequence[str]) -> str:
    value = item.get(key)
    choice = value.strip().lower() if isinstance(value, str) else None
    if choice is None or choice not in allowed:
        raise HandoffError(f"the {key} for {symbol} must be one of {', '.join(allowed)}")
    return choice


def _covering(raw: Any, expected: Sequence[str]) -> list[tuple[str, Mapping[str, Any]]]:
    """(symbol, item) for exactly one item per symbol in `expected`, in submission order."""
    items = _items(raw, "cases")
    wanted = set(expected)
    seen: set[str] = set()
    covered = []
    for index, item in enumerate(items, start=1):
        symbol = _symbol(item, index)
        if symbol not in wanted:
            raise HandoffError(f"{symbol} is not in the debate: give cases only for {', '.join(expected)}")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        covered.append((symbol, item))
    missing = [symbol for symbol in expected if symbol not in seen]
    if missing:
        raise HandoffError(f"no case for {', '.join(missing)}: give exactly one case per stock")
    return covered


def validate_note(symbol: Any, note: Any, *, expected: str, max_chars: int) -> Note:
    if not isinstance(symbol, str) or symbol.strip().upper() != expected:
        raise HandoffError(f"this note is for {expected}: submit it with symbol {expected}")
    if not isinstance(note, str) or not note.strip():
        raise HandoffError(f"the note for {expected} is empty: write the dated facts you found, or say that you found none")
    text = note.strip()
    if len(text) > max_chars:
        raise HandoffError(f"the note for {expected} is {len(text)} characters: keep it under {max_chars}")
    return Note(expected, text)


def validate_bull_cases(raw: Any, *, expected: Sequence[str], max_chars: int) -> list[BullCase]:
    return [BullCase(symbol, _one_of(item, "conviction", symbol, LEVELS), _text(item, "argument", symbol, max_chars)) for symbol, item in _covering(raw, expected)]


def validate_bear_cases(raw: Any, *, expected: Sequence[str], max_chars: int) -> list[BearCase]:
    return [
        BearCase(symbol, _one_of(item, "risk", symbol, LEVELS), _one_of(item, "concern", symbol, CONCERNS), _text(item, "argument", symbol, max_chars))
        for symbol, item in _covering(raw, expected)
    ]


def validate_picks(picks_raw: Any, drops_raw: Any, *, allowed: Sequence[str], held: Collection[str], min_picks: int, max_picks: int, max_chars: int) -> Picks:
    """`min_picks` to `max_picks` unique stocks from `allowed`; `drops` names exactly the `held` stocks not picked."""
    items = _items(picks_raw, "picks")
    if not min_picks <= len(items) <= max_picks:
        raise HandoffError(f"pick between {min_picks} and {max_picks} stocks, got {len(items)}")
    allowed_set = set(allowed)
    picked: set[str] = set()
    picks = []
    for index, item in enumerate(items, start=1):
        symbol = _symbol(item, index)
        if symbol not in allowed_set:
            raise HandoffError(f"{symbol} is not in the debate (choose from {', '.join(allowed)})")
        if symbol in picked:
            raise HandoffError(f"{symbol} appears twice")
        picked.add(symbol)
        picks.append(Choice(symbol, _text(item, "reason", symbol, max_chars)))
    must_drop = [symbol for symbol in held if symbol not in picked]
    dropped: set[str] = set()
    drops = []
    for index, item in enumerate(_items([] if drops_raw is None else drops_raw, "drops"), start=1):
        symbol = _symbol(item, index)
        if symbol not in must_drop:
            raise HandoffError(f"{symbol} cannot be dropped: drops lists only the held stocks you did not pick ({', '.join(must_drop) or 'none'})")
        if symbol in dropped:
            raise HandoffError(f"{symbol} appears twice")
        dropped.add(symbol)
        drops.append(Choice(symbol, _text(item, "reason", symbol, max_chars)))
    missing = [symbol for symbol in must_drop if symbol not in dropped]
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        raise HandoffError(f"{', '.join(missing)} {verb} held and not picked: list each in drops with a reason")
    return Picks(picks=tuple(picks), drops=tuple(drops))


# --- the recorder ---------------------------------------------------------------------------------


class HandoffRecorder:
    """Holds what the current stage must validate against, and the first valid submission for it."""

    def __init__(self, params: BullBearParams) -> None:
        self._params = params
        self._stage: str | None = None
        self._context: dict[str, Any] = {}
        self._submission: Any = None
        self._submitted = False
        self.last_error: str | None = None

    def expect_note(self, symbol: str) -> None:
        self._arm(NOTE, {"expected": symbol})

    def expect_bull(self, symbols: Sequence[str]) -> None:
        self._arm(BULL, {"expected": list(symbols)})

    def expect_bear(self, symbols: Sequence[str]) -> None:
        self._arm(BEAR, {"expected": list(symbols)})

    def expect_picks(self, allowed: Sequence[str], held: Sequence[str]) -> None:
        self._arm(PICKS, {"allowed": list(allowed), "held": list(held)})

    def _arm(self, stage: str, context: dict[str, Any]) -> None:
        self._stage, self._context = stage, context
        self._submission, self._submitted, self.last_error = None, False, None

    @property
    def submitted(self) -> bool:
        """Whether the armed stage has a valid submission."""
        return self._submitted

    @property
    def submission(self) -> Any:
        """The armed stage's valid submission (`Note`, `list[BullCase]`, `list[BearCase]` or `Picks`), else None."""
        return self._submission

    def submit(self, stage: str, *args: Any) -> dict[str, Any]:
        """What a submit tool returns: `{"status": "recorded"}` or `{"error": ...}` (recorded in `last_error`)."""
        if self._stage != stage:
            return self._fail(f"{_TOOL_NAMES[stage]} is not expected at this point of the review")
        if self._submitted:
            return self._fail("already recorded for this stage")
        params, context = self._params, self._context
        try:
            if stage == NOTE:
                value: Any = validate_note(args[0], args[1], expected=context["expected"], max_chars=params.note_max_chars)
            elif stage == BULL:
                value = validate_bull_cases(args[0], expected=context["expected"], max_chars=params.argument_max_chars)
            elif stage == BEAR:
                value = validate_bear_cases(args[0], expected=context["expected"], max_chars=params.argument_max_chars)
            else:
                value = validate_picks(
                    args[0],
                    args[1],
                    allowed=context["allowed"],
                    held=context["held"],
                    min_picks=params.min_picks,
                    max_picks=params.max_picks,
                    max_chars=params.reason_max_chars,
                )
        except HandoffError as exc:
            return self._fail(str(exc))
        self._submission, self._submitted, self.last_error = value, True, None
        return {"status": "recorded"}

    def _fail(self, message: str) -> dict[str, Any]:
        self.last_error = message
        return {"error": message}


# --- the tools ------------------------------------------------------------------------------------


def submit_tools(recorder: HandoffRecorder) -> dict[str, Callable[..., dict[str, Any]]]:
    """The four submit tools, closures over `recorder`, keyed by name. Each agent is given only its own."""

    def submit_note(symbol: str, note: str) -> dict[str, Any]:
        """Submit your note on the stock: its symbol and the dated facts you found (no opinion)."""
        return recorder.submit(NOTE, symbol, note)

    def submit_bull_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one case per stock: objects with symbol, conviction (low, medium or high) and argument."""
        return recorder.submit(BULL, cases)

    def submit_bear_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one case per stock: objects with symbol, risk (low, medium or high), concern and argument."""
        return recorder.submit(BEAR, cases)

    def submit_picks(picks: list[dict[str, Any]], drops: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Submit the stocks that win the debate (objects with symbol and reason) and, for each held stock you do not pick, a drop (symbol and reason)."""
        return recorder.submit(PICKS, picks, drops)

    return {"submit_note": submit_note, "submit_bull_case": submit_bull_case, "submit_bear_case": submit_bear_case, "submit_picks": submit_picks}
