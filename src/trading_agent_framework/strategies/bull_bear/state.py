"""What bull_bear remembers between reviews, and the per-review log.

`StateStore` keeps the date of the last completed review (a completed Tuesday is not reviewed again after a
restart), the abandoned-review streak and the last picks, in one small JSON file per mode, written atomically. A
missing or invalid file is an empty state. `ReviewLog` appends one JSON line per review to the run directory.
Neither ever raises on an I/O problem: bookkeeping must not stop a review.
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
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "BullBearState")

STATE_VERSION = 1
_PICK_KEYS = {"symbol", "reason", "date"}


def state_path(project_root: Path, mode: TradingMode) -> Path:
    """`<project_root>/data/bull_bear_state_<mode>.json`."""
    return project_root / "data" / f"bull_bear_state_{mode.value}.json"


@dataclass(frozen=True, slots=True)
class BullBearState:
    last_completed_review: str | None = None  # ISO market date of the last completed review
    abandoned_streak: int = 0  # abandoned reviews in a row
    last_picks: list[dict[str, str]] = field(default_factory=list)  # [{symbol, reason, date}] of the last completed review


def _valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return False
    last = raw.get("last_completed_review")
    streak = raw.get("abandoned_streak", 0)
    picks = raw.get("last_picks", [])
    return (
        (last is None or isinstance(last, str))
        and isinstance(streak, int)
        and not isinstance(streak, bool)
        and streak >= 0
        and isinstance(picks, list)
        and all(isinstance(pick, dict) and set(pick) == _PICK_KEYS and all(isinstance(value, str) for value in pick.values()) for pick in picks)
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
        if not _valid(raw):
            logger.log_warning(f"state file {self._path} is not a valid state, starting from an empty state")
            return BullBearState()
        return BullBearState(
            last_completed_review=raw.get("last_completed_review"),
            abandoned_streak=raw.get("abandoned_streak", 0),
            last_picks=[dict(pick) for pick in raw.get("last_picks", [])],
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
