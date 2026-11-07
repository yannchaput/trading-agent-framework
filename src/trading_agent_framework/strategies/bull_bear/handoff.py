"""The hand-off between the four agents: submit tools, their validation and the recorder.

Each agent ends its run by calling one submit tool. A tool validates its argument against what the pipeline armed
the recorder with for this stage, records the first valid submission, and otherwise returns `{"error": ...}` so the
model can correct itself in the same run. Free text from an agent is never trusted.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema
from the function's real annotations (as `bill_ackman/handoff.py` and `memory/tools.py` do).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.common.handoff import HandoffError, SubmitRecorder, covering, required_text

LEVELS = ("low", "medium", "high")
CONCERNS = ("valuation", "momentum_exhaustion", "earnings", "fundamentals", "news_event", "sector", "none")
VERDICTS = ("in", "out")
STRENGTHS = ("strong", "standard", "weak")

NOTE, BULL, BEAR, VERDICT = "note", "bull", "bear", "verdict"


class NoteTooLong(HandoffError):
    """A note over the hard limit; `note` is the refused (stripped) text, so the recorder can spot a resubmission."""

    def __init__(self, message: str, note: str) -> None:
        """Keep the over-long note so it can be shortened and resubmitted."""
        super().__init__(message)
        self.note = note


@dataclass(frozen=True, slots=True)
class Note:
    """A research note on one stock."""

    symbol: str
    note: str


@dataclass(frozen=True, slots=True)
class BullCase:
    """The bull side's case for one stock: a conviction level and an argument."""

    symbol: str
    conviction: str  # one of LEVELS
    argument: str


@dataclass(frozen=True, slots=True)
class BearCase:
    """The bear side's case against one stock: a conviction level and an argument."""

    symbol: str
    risk: str  # one of LEVELS
    concern: str  # one of CONCERNS
    argument: str


@dataclass(frozen=True, slots=True)
class Verdict:
    """The judge's call on one stock."""

    symbol: str
    verdict: str  # one of VERDICTS
    reason: str
    what_changed: str | None = None  # the new fact behind a verdict that differs from the judge's previous one
    strength: str | None = None  # one of STRENGTHS, set on every verdict by `validate_verdicts`


@dataclass(frozen=True, slots=True)
class Verdicts:
    """The judge's verdicts for a batch of candidates."""

    verdicts: tuple[Verdict, ...]  # one per candidate, in submission order

    @property
    def chosen(self) -> list[str]:
        """The symbols the judge put in, in submission order."""
        return [verdict.symbol for verdict in self.verdicts if verdict.verdict == "in"]


# --- validation (pure) --------------------------------------------------------------------------


def _one_of(item: Mapping[str, Any], key: str, symbol: str, allowed: Sequence[str]) -> str:
    """The string at `key`, checked to be one of `allowed`."""
    value = item.get(key)
    choice = value.strip().lower() if isinstance(value, str) else None
    if choice is None or choice not in allowed:
        raise HandoffError(f"the {key} for {symbol} must be one of {', '.join(allowed)}")
    return choice


def _covering(raw: Any, expected: Sequence[str], what: str = "cases") -> list[tuple[str, Mapping[str, Any]]]:
    """(symbol, item) for exactly one item per symbol in `expected`, in submission order."""
    return covering(
        raw,
        expected,
        what=what,
        unknown=lambda symbol: f"{symbol} is not in the debate: give cases only for {', '.join(expected)}",
        missing=lambda symbols: f"no case for {', '.join(symbols)}: give exactly one case per stock",
    )


def validate_note(symbol: Any, note: Any, *, expected: str, max_chars: int, ask_chars: int) -> Note:
    """`max_chars` is the hard limit; `ask_chars` the length the model is told (the hard limit is never quoted)."""
    if not isinstance(symbol, str) or symbol.strip().upper() != expected:
        raise HandoffError(f"this note is for {expected}: submit it with symbol {expected}")
    if not isinstance(note, str) or not note.strip():
        raise HandoffError(f"the note for {expected} is empty: write the dated facts you found, or say that you found none")
    text = note.strip()
    if len(text) > max_chars:
        raise NoteTooLong(
            f"the note for {expected} is {len(text)} characters; a note is at most {ask_chars}: "
            f"rewrite it to about {int(ask_chars * 0.9)} characters by dropping your least important facts, not by trimming words",
            text,
        )
    return Note(expected, text)


def trim_note(text: str, *, max_chars: int, min_chars: int) -> str:
    """`text` cut to at most `max_chars`, never reworded: the longest run of whole sentences that fits, if it has
    at least `min_chars`; otherwise a cut at a word with a trailing ellipsis (or hard, when there is no space).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    end = 0  # length of the longest prefix that ends a sentence (. ! or ? before whitespace or at the end) and fits
    for index, char in enumerate(text[:max_chars]):
        if char in ".!?" and (index + 1 == len(text) or text[index + 1].isspace()):
            end = index + 1
    whole = text[:end].rstrip()
    if len(whole) >= min_chars:
        return whole
    head = text[: max_chars - 1]
    cut = max((index for index, char in enumerate(head) if char.isspace()), default=-1)
    if cut <= 0:
        return head + "…"
    return head[:cut].rstrip() + "…"


def validate_bull_cases(raw: Any, *, expected: Sequence[str], max_chars: int) -> list[BullCase]:
    """Validate the bull cases against the expected symbols and the length limit."""
    return [BullCase(symbol, _one_of(item, "conviction", symbol, LEVELS), required_text(item, "argument", symbol, max_chars)) for symbol, item in _covering(raw, expected)]


def validate_bear_cases(raw: Any, *, expected: Sequence[str], max_chars: int) -> list[BearCase]:
    """Validate the bear cases against the expected symbols and the length limit."""
    return [
        BearCase(symbol, _one_of(item, "risk", symbol, LEVELS), _one_of(item, "concern", symbol, CONCERNS), required_text(item, "argument", symbol, max_chars))
        for symbol, item in _covering(raw, expected)
    ]


def _what_changed(item: Mapping[str, Any], symbol: str, verdict: str, before: str | None, max_chars: int) -> str | None:
    """Required, as one short sentence, when `verdict` differs from the judge's previous verdict `before`."""
    value = item.get("what_changed")
    given = isinstance(value, str) and bool(value.strip())
    if before is not None and before != verdict and not given:
        raise HandoffError(f"your previous verdict on {symbol} was {before}: to put it {verdict}, give what_changed, the new and material fact")
    return required_text(item, "what_changed", symbol, max_chars) if given else None


def _strength(item: Mapping[str, Any], symbol: str, verdict: str) -> str:
    """The strength of a stock (required on every verdict, one of STRENGTHS); a weak stock cannot be put in."""
    strength = _one_of(item, "strength", symbol, STRENGTHS)
    if verdict == "in" and strength == "weak":
        raise HandoffError(f"{symbol} is weak: a weak stock goes out, put it out or rate it standard or strong, with what_changed if your previous verdict was in")
    return strength


def validate_verdicts(raw: Any, *, expected: Sequence[str], max_in: int, max_chars: int, previous: Mapping[str, str] | None = None) -> Verdicts:
    """One verdict (in or out) with a reason per `expected` symbol; a strength on every verdict (a weak stock is always out); at most `max_in` in; `what_changed` on every flip from `previous`."""
    before = previous or {}
    verdicts = []
    for symbol, item in _covering(raw, expected, "verdicts"):
        verdict = _one_of(item, "verdict", symbol, VERDICTS)
        reason = required_text(item, "reason", symbol, max_chars)
        changed = _what_changed(item, symbol, verdict, before.get(symbol), max_chars)
        verdicts.append(Verdict(symbol, verdict, reason, changed, _strength(item, symbol, verdict)))
    chosen = sum(1 for verdict in verdicts if verdict.verdict == "in")
    if chosen > max_in:
        raise HandoffError(f"{chosen} stocks in: put at most {max_in} in and the others out")
    return Verdicts(tuple(verdicts))


# --- the recorder ---------------------------------------------------------------------------------


class HandoffRecorder(SubmitRecorder):
    """Holds what the current stage must validate against, and the first valid submission for it."""

    TOOL_NAMES = {NOTE: "submit_note", BULL: "submit_bull_case", BEAR: "submit_bear_case", VERDICT: "submit_verdicts"}
    NOT_EXPECTED_SUFFIX = " of the review"
    ALREADY_RECORDED = "already recorded for this stage"

    def __init__(self, params: BullBearParams) -> None:
        """Create a recorder bound to the strategy parameters."""
        super().__init__()
        self._params = params
        self._refused_note: str | None = None

    def expect_note(self, symbol: str) -> None:
        """Arm the recorder to accept a note for `symbol`."""
        self._arm(NOTE, {"expected": symbol})

    @property
    def note_symbol(self) -> str | None:
        """The stock whose note is expected while the note stage is armed (the researcher's current stock), else None."""
        return self._context["expected"] if self._stage == NOTE else None

    def expect_bull(self, symbols: Sequence[str]) -> None:
        """Arm the recorder to accept bull cases for `symbols`."""
        self._arm(BULL, {"expected": list(symbols)})

    def expect_bear(self, symbols: Sequence[str]) -> None:
        """Arm the recorder to accept bear cases for `symbols`."""
        self._arm(BEAR, {"expected": list(symbols)})

    def expect_verdicts(self, symbols: Sequence[str], *, max_in: int, previous: Mapping[str, str] | None = None) -> None:
        """Arm the recorder to accept at most `max_in` verdicts for `symbols`, given the previous run's strengths."""
        self._arm(VERDICT, {"expected": list(symbols), "max_in": max_in, "previous": dict(previous or {})})

    def _arm(self, stage: str, context: dict[str, Any]) -> None:
        """Arm the recorder for `stage` with its validation context."""
        super()._arm(stage, context)
        self._refused_note = None

    @property
    def refused_note(self) -> str | None:
        """The most recent note refused for length since the stage was armed (kept across other refusals), else None."""
        return self._refused_note

    def _validate(self, stage: str, *args: Any) -> Any:
        """Validate the submitted payload for `stage`."""
        params, context = self._params, self._context
        if stage == NOTE:
            try:
                return validate_note(args[0], args[1], expected=context["expected"], max_chars=params.note_hard_max_chars, ask_chars=params.note_max_chars)
            except NoteTooLong as exc:
                message = f"this is the same text you sent before: {exc}" if exc.note == self._refused_note else str(exc)
                self._refused_note = exc.note
                raise HandoffError(message) from None
        if stage == BULL:
            return validate_bull_cases(args[0], expected=context["expected"], max_chars=params.argument_max_chars)
        if stage == BEAR:
            return validate_bear_cases(args[0], expected=context["expected"], max_chars=params.argument_max_chars)
        return validate_verdicts(args[0], expected=context["expected"], max_in=context["max_in"], max_chars=params.reason_max_chars, previous=context["previous"])


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

    def submit_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one verdict per stock: objects with symbol, strength (strong, standard or weak), verdict (in or out; a weak stock is always out), one short reason, and what_changed when the verdict differs from your previous_verdict."""  # noqa: E501
        return recorder.submit(VERDICT, verdicts)

    return {
        "submit_note": submit_note,
        "submit_bull_case": submit_bull_case,
        "submit_bear_case": submit_bear_case,
        "submit_verdicts": submit_verdicts,
    }
