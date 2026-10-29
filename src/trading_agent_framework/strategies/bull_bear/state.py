"""What bull_bear remembers between reviews, and the per-review log.

`StateStore` keeps the date of the last completed review (a completed Tuesday is not reviewed again after a
restart), the abandoned-review streak, the judge's latest verdict on every stock it judged and the re-entry
cooldowns (the stocks the judge sold, and the completed reviews each stays out of the candidates), in one small JSON
file per mode, written atomically. The verdict also carries the judge's strength (strong, standard or weak; never weak
on an `in`), so the next review shows the previous strength of every stock. A missing or invalid file is an empty
state; a version-2 file (verdicts without strengths) still loads whole, its verdicts showing no strength until a
review judges the stock again, and a save writes them back as they are (a version-3 entry may lack a strength; one
that has a strength must hold a valid one); a version-1 file (the picker, the veto judge) keeps its review date and
streak and nothing else. `ReviewLog` appends one JSON line per review to the
run directory. Neither ever raises on an I/O problem: bookkeeping must not stop a review.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.bull_bear.handoff import STRENGTHS
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "BullBearState")

STATE_VERSION = 3
_UNRATED_VERSION = 2  # the file before strengths: a verdict is {verdict, reason, date}
_PICKER_VERSION = 1  # the picker's and the veto judge's file: a review date and a streak
_VERDICT_KEYS = {"verdict", "reason", "date"}
_VERDICTS = {"in", "out"}


def state_path(project_root: Path, mode: TradingMode) -> Path:
    """`<project_root>/data/bull_bear_state_<mode>.json`."""
    return project_root / "data" / f"bull_bear_state_{mode.value}.json"


@dataclass(frozen=True, slots=True)
class BullBearState:
    last_completed_review: str | None = None  # ISO market date of the last completed review
    abandoned_streak: int = 0  # abandoned reviews in a row
    verdicts: dict[str, dict[str, str]] = field(default_factory=dict)  # symbol -> the judge's latest {verdict, reason, date, strength}; no strength on a version-2 verdict
    cooldowns: dict[str, int] = field(default_factory=dict)  # symbol -> completed reviews it stays out of the candidates
    history: list[dict[str, Any]] = field(default_factory=list)  # the last feedback_reviews reviews: {date, verdicts: {symbol: {verdict, reason, close}}}


def _valid_review(entry: Any) -> bool:
    """One `history` entry: {date, verdicts: {symbol: {verdict, reason, close}}}."""
    if not isinstance(entry, dict) or set(entry) != {"date", "verdicts"} or not isinstance(entry["date"], str) or not isinstance(entry["verdicts"], dict):
        return False
    return all(
        isinstance(item, dict)
        and set(item) == {"verdict", "reason", "close"}
        and isinstance(item["verdict"], str)
        and item["verdict"] in _VERDICTS
        and isinstance(item["reason"], str)
        and isinstance(item["close"], int | float)
        and not isinstance(item["close"], bool)
        for item in entry["verdicts"].values()
    )


def _valid_dates(raw: dict[str, Any]) -> bool:
    """The review date (a string or None) and the abandoned streak (a non-negative integer) every version carries."""
    last = raw.get("last_completed_review")
    streak = raw.get("abandoned_streak", 0)
    return (last is None or isinstance(last, str)) and isinstance(streak, int) and not isinstance(streak, bool) and streak >= 0


def _valid_verdict(entry: Any, version: int) -> bool:
    """One `verdicts` entry: {verdict, reason, date} in version 2; in version 3 also a `strength`, which may be absent (a version-2 verdict
    not judged again) but, when there, is a known one (never weak on an `in`)."""
    allowed = _VERDICT_KEYS if version == _UNRATED_VERSION else _VERDICT_KEYS | {"strength"}
    if not isinstance(entry, dict) or not _VERDICT_KEYS <= set(entry) <= allowed or not all(isinstance(value, str) for value in entry.values()) or entry["verdict"] not in _VERDICTS:
        return False
    return "strength" not in entry or (entry["strength"] in STRENGTHS and not (entry["verdict"] == "in" and entry["strength"] == "weak"))


def _valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("version") not in (_UNRATED_VERSION, STATE_VERSION):
        return False
    verdicts = raw.get("verdicts", {})
    cooldowns = raw.get("cooldowns", {})
    history = raw.get("history", [])
    return (
        _valid_dates(raw)
        and isinstance(verdicts, dict)
        and all(_valid_verdict(entry, raw["version"]) for entry in verdicts.values())
        and isinstance(cooldowns, dict)
        and all(isinstance(left, int) and not isinstance(left, bool) and left > 0 for left in cooldowns.values())
        and isinstance(history, list)
        and all(_valid_review(entry) for entry in history)
    )


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> BullBearState:
        """The saved state; an empty one when the file is missing, unreadable, or not a valid state (warned about)."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return BullBearState()
        except (OSError, ValueError) as exc:
            logger.log_warning(f"state file {self._path} is unreadable, starting from an empty state: {exc}")
            return BullBearState()
        if isinstance(raw, dict) and raw.get("version") == _PICKER_VERSION and _valid_dates(raw):
            # The picker's file: nothing else of it applies, but a completed Tuesday is not reviewed again after the upgrade.
            return BullBearState(last_completed_review=raw.get("last_completed_review"), abandoned_streak=raw.get("abandoned_streak", 0))
        if not _valid(raw):
            logger.log_warning(f"state file {self._path} is not a valid state, starting from an empty state")
            return BullBearState()
        return BullBearState(
            last_completed_review=raw.get("last_completed_review"),
            abandoned_streak=raw.get("abandoned_streak", 0),
            verdicts={symbol: dict(entry) for symbol, entry in raw.get("verdicts", {}).items()},
            cooldowns=dict(raw.get("cooldowns", {})),
            history=[{"date": entry["date"], "verdicts": {symbol: dict(item) for symbol, item in entry["verdicts"].items()}} for entry in raw.get("history", [])],
        )

    def save(self, state: BullBearState) -> None:
        """Write the state atomically (temporary file in the same directory, then replace); an I/O error is logged."""
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self._path.parent, prefix=f"{self._path.name}.", suffix=".tmp")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"version": STATE_VERSION, **asdict(state)}, handle)
            os.replace(temporary, self._path)
        except OSError as exc:
            logger.log_warning(f"state could not be saved to {self._path}: {exc}")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def wipe(self) -> None:
        """Delete the state file (a backtest starts clean); safe when there is none."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.log_warning(f"state file {self._path} could not be deleted: {exc}")


class ReviewLog:
    """`reviews.jsonl`: one JSON line per review. With no path (a strategy run outside a runner) it does nothing."""

    def __init__(self, path: Path | None) -> None:
        self._path = path

    def append(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.log_warning(f"review log {self._path} could not be written: {exc}")
