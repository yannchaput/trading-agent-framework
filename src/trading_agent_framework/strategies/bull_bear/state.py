"""What bull_bear remembers between reviews.

`StateStore` keeps the date of the last completed review (a completed Tuesday is not reviewed again after a
restart), the abandoned-review streak, the judge's latest verdict on every stock it judged and the re-entry
cooldowns (the stocks the judge sold, and the completed reviews each stays out of the candidates), in one small JSON
file per mode, written atomically. The verdict also carries the judge's strength (strong, standard or weak; never weak
on an `in`), so the next review shows the previous strength of every stock. A missing or invalid file is an empty
state; a version-2 file (verdicts without strengths) still loads whole, its verdicts showing no strength until a
review judges the stock again, and a save writes them back as they are (a version-3 entry may lack a strength; one
that has a strength must hold a valid one); a version-1 file (the picker, the veto judge) keeps its review date and
streak and nothing else. It never raises on an I/O problem: bookkeeping must not stop a review (the per-review log is a
`RunLog`, in `strategies/common/run_log.py`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_agent_framework.strategies.bull_bear.handoff import STRENGTHS
from trading_agent_framework.strategies.common.json_state import InvalidState, JsonStateStore, is_int
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "BullBearState")

STATE_VERSION = 3
_UNRATED_VERSION = 2  # the file before strengths: a verdict is {verdict, reason, date}
_PICKER_VERSION = 1  # the picker's and the veto judge's file: a review date and a streak
_VERDICT_KEYS = {"verdict", "reason", "date"}
_VERDICTS = {"in", "out"}


@dataclass(frozen=True, slots=True)
class BullBearState:
    """What bull_bear carries from one review to the next.

    Attributes:
        last_completed_review: ISO market date of the last completed review.
        abandoned_streak: Abandoned reviews in a row.
        verdicts: Symbol -> the judge's latest `{verdict, reason, date, strength}`; no strength on a version-2 verdict.
        cooldowns: Symbol -> completed reviews it stays out of the candidates.
    """

    last_completed_review: str | None = None  # ISO market date of the last completed review
    abandoned_streak: int = 0  # abandoned reviews in a row
    verdicts: dict[str, dict[str, str]] = field(default_factory=dict)  # symbol -> the judge's latest {verdict, reason, date, strength}; no strength on a version-2 verdict
    cooldowns: dict[str, int] = field(default_factory=dict)  # symbol -> completed reviews it stays out of the candidates


def _valid_dates(raw: dict[str, Any]) -> bool:
    """The review date (a string or None) and the abandoned streak (a non-negative integer) every version carries."""
    last = raw.get("last_completed_review")
    streak = raw.get("abandoned_streak", 0)
    return (last is None or isinstance(last, str)) and is_int(streak) and streak >= 0


def _valid_verdict(entry: Any, version: int) -> bool:
    """One `verdicts` entry: {verdict, reason, date} in version 2; in version 3 also a `strength`.

    The strength may be absent (a version-2 verdict not judged again) but, when there, is a known one (never weak on an
    `in`).
    """
    allowed = _VERDICT_KEYS if version == _UNRATED_VERSION else _VERDICT_KEYS | {"strength"}
    if not isinstance(entry, dict) or not _VERDICT_KEYS <= set(entry) <= allowed or not all(isinstance(value, str) for value in entry.values()) or entry["verdict"] not in _VERDICTS:
        return False
    return "strength" not in entry or (entry["strength"] in STRENGTHS and not (entry["verdict"] == "in" and entry["strength"] == "weak"))


def _valid(raw: dict[str, Any]) -> bool:
    if raw["version"] not in (_UNRATED_VERSION, STATE_VERSION):
        return False
    verdicts = raw.get("verdicts", {})
    cooldowns = raw.get("cooldowns", {})
    return (
        _valid_dates(raw)
        and isinstance(verdicts, dict)
        and all(_valid_verdict(entry, raw["version"]) for entry in verdicts.values())
        and isinstance(cooldowns, dict)
        and all(is_int(left) and left > 0 for left in cooldowns.values())
    )


def _decode(raw: dict[str, Any]) -> BullBearState:
    if raw["version"] == _PICKER_VERSION and _valid_dates(raw):
        # The picker's file: nothing else of it applies, but a completed Tuesday is not reviewed again after the upgrade.
        return BullBearState(last_completed_review=raw.get("last_completed_review"), abandoned_streak=raw.get("abandoned_streak", 0))
    if not _valid(raw):
        raise InvalidState
    return BullBearState(
        last_completed_review=raw.get("last_completed_review"),
        abandoned_streak=raw.get("abandoned_streak", 0),
        verdicts={symbol: dict(entry) for symbol, entry in raw.get("verdicts", {}).items()},
        cooldowns=dict(raw.get("cooldowns", {})),
    )


class StateStore(JsonStateStore[BullBearState]):
    """JSON store of the bull_bear state."""

    def __init__(self, path: Path, *, fresh: bool = False) -> None:
        """Open the store.

        Args:
            path: State file.
            fresh: Delete any existing file first, so a backtest never inherits the last run's state.
        """
        super().__init__(path, version=STATE_VERSION, empty=BullBearState, decode=_decode, encode=asdict, log=logger, accepts={_PICKER_VERSION, _UNRATED_VERSION, STATE_VERSION}, fresh=fresh)
