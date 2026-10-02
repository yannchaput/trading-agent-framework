"""What the strategy remembers between reviews, and the per-review log.

`StateStore` keeps the fail counters, the previous review's ranking and verdicts, and the abandoned-review streak in one
small JSON file per mode, written atomically. A missing or corrupt file is an empty state: the strategy then
simply starts its counters again. `ReviewLog` appends one JSON line per review to the run directory.
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

logger = ColorLogger(logging.getLogger(__name__), "BillAckmanState")

STATE_VERSION = 1


def state_path(project_root: Path, mode: TradingMode) -> Path:
    """`<project_root>/data/bill_ackman_state_<mode>.json`."""
    return project_root / "data" / f"bill_ackman_state_{mode.value}.json"


@dataclass(frozen=True, slots=True)
class ReviewState:
    last_review: str | None = None  # ISO date of the last completed review
    fail_counts: dict[str, int] = field(default_factory=dict)  # holding -> consecutive fails (always >= 1)
    last_ranking: list[str] = field(default_factory=list)
    last_verdicts: dict[str, str] = field(default_factory=dict)  # symbol -> "survive" | "fail"
    abandoned_streak: int = 0  # abandoned reviews in a row


def _valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return False
    last_review = raw.get("last_review")
    fail_counts, ranking, verdicts = raw.get("fail_counts", {}), raw.get("last_ranking", []), raw.get("last_verdicts", {})
    streak = raw.get("abandoned_streak", 0)
    return (
        (last_review is None or isinstance(last_review, str))
        and isinstance(fail_counts, dict)
        and all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 1 for k, v in fail_counts.items())
        and isinstance(ranking, list)
        and all(isinstance(symbol, str) for symbol in ranking)
        and isinstance(verdicts, dict)
        and all(isinstance(k, str) and v in ("survive", "fail") for k, v in verdicts.items())
        and isinstance(streak, int)
        and not isinstance(streak, bool)
        and streak >= 0
    )


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> ReviewState:
        """The saved state; an empty one when the file is missing, unreadable, or not a valid state (warned about)."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ReviewState()
        except (OSError, ValueError) as exc:
            logger.log_warning(f"state file {self._path} is unreadable, starting from an empty state: {exc}")
            return ReviewState()
        if not _valid(raw):
            logger.log_warning(f"state file {self._path} is not a valid state, starting from an empty state")
            return ReviewState()
        return ReviewState(
            last_review=raw.get("last_review"),
            fail_counts=dict(raw.get("fail_counts", {})),
            last_ranking=list(raw.get("last_ranking", [])),
            last_verdicts=dict(raw.get("last_verdicts", {})),
            abandoned_streak=raw.get("abandoned_streak", 0),
        )

    def save(self, state: ReviewState) -> None:
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
