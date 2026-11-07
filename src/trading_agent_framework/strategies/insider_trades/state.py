"""What the strategy remembers between daily checks.

`StateStore` keeps, in one small JSON file per mode written atomically: the accessions of the Form 4 filings the last completed
run was built from (`processed`, the "nothing new" test) and the ones inside each ticker's window then (`in_window`), the
researcher's note per ticker, that run's target, the tickers this strategy has ever ordered (`traded`: the only positions it
may sell), a trade left unfilled (`pending_trade`) and the abandoned-run streak. A missing or corrupt file is an empty state:
the next check then treats every known filing as new and builds the portfolio again.
`StateStore` never raises on an I/O problem: bookkeeping must not stop a run (the per-run log, `RunLog`, is in `strategies/common/run_log.py`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_agent_framework.strategies.common.json_state import InvalidState, JsonStateStore, is_int, valid_strings, valid_weights
from trading_agent_framework.strategies.common.pipeline import PendingTrade, decode_pending, valid_pending
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "InsiderTradesState")

STATE_VERSION = 1

_NOTE_CLASSES = ("buying", "selling", "none")
_NOTE_KEYS = {"summary", "class", "overrides", "as_of"}
_OVERRIDE_KEYS = {"insider_id", "from_label", "to_label", "reason"}


@dataclass(frozen=True, slots=True)
class InsiderState:
    """Persisted state of the insider-trades strategy."""
    processed: list[str] = field(default_factory=list)  # accessions of every known P/S filing at the last completed run
    in_window: dict[str, list[str]] = field(default_factory=dict)  # ticker -> accessions inside its window at that run
    notes: dict[str, dict[str, Any]] = field(default_factory=dict)  # ticker -> {summary, class, overrides, as_of} from that run
    target: dict[str, float] = field(default_factory=dict)  # ticker -> weight of that run's portfolio
    traded: list[str] = field(default_factory=list)  # tickers ever ordered by this strategy
    pending_trade: PendingTrade | None = None
    abandoned_streak: int = 0  # abandoned runs in a row
    last_run: str | None = None  # ISO date of the last completed run


def _valid_in_window(value: Any) -> bool:
    """Whether `value` maps strings to valid string lists."""
    return isinstance(value, dict) and all(isinstance(k, str) and valid_strings(v) for k, v in value.items())


def _valid_override(value: Any) -> bool:
    """Whether `value` is a well-formed override record."""
    return isinstance(value, dict) and set(value) == _OVERRIDE_KEYS and all(isinstance(v, str) for v in value.values())


def _valid_note(value: Any) -> bool:
    """Whether `value` is a well-formed note."""
    return (
        isinstance(value, dict)
        and set(value) == _NOTE_KEYS
        and isinstance(value["summary"], str)
        and value["class"] in _NOTE_CLASSES
        and isinstance(value["overrides"], list)
        and all(_valid_override(item) for item in value["overrides"])
        and isinstance(value["as_of"], str)
    )


def _valid_notes(value: Any) -> bool:
    """Whether `value` maps strings to well-formed notes."""
    return isinstance(value, dict) and all(isinstance(k, str) and _valid_note(v) for k, v in value.items())


def _valid(raw: dict[str, Any]) -> bool:
    """Whether the raw state file has the expected shape."""
    last_run = raw.get("last_run")
    streak = raw.get("abandoned_streak", 0)
    return (
        valid_strings(raw.get("processed", []))
        and _valid_in_window(raw.get("in_window", {}))
        and _valid_notes(raw.get("notes", {}))
        and valid_weights(raw.get("target", {}))
        and valid_strings(raw.get("traded", []))
        and valid_pending(raw.get("pending_trade"))
        and is_int(streak)
        and streak >= 0
        and (last_run is None or isinstance(last_run, str))
    )


def _decode(raw: dict[str, Any]) -> InsiderState:
    """Build an `InsiderState` from the state file's JSON."""
    if not _valid(raw):
        raise InvalidState
    return InsiderState(
        processed=list(raw.get("processed", [])),
        in_window={ticker: list(accessions) for ticker, accessions in raw.get("in_window", {}).items()},
        notes={ticker: {**note, "overrides": [dict(item) for item in note["overrides"]]} for ticker, note in raw.get("notes", {}).items()},
        target=dict(raw.get("target", {})),
        traded=list(raw.get("traded", [])),
        pending_trade=decode_pending(raw.get("pending_trade")),
        abandoned_streak=raw.get("abandoned_streak", 0),
        last_run=raw.get("last_run"),
    )


class StateStore(JsonStateStore[InsiderState]):
    """JSON file store for the insider-trades state."""
    def __init__(self, path: Path, *, fresh: bool = False) -> None:
        """`fresh` wipes the file at construction; used in backtests so one run never inherits the last."""
        super().__init__(path, version=STATE_VERSION, empty=InsiderState, decode=_decode, encode=asdict, log=logger, fresh=fresh)
