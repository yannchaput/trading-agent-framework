"""What the strategy remembers between reviews.

`StateStore` keeps the fail counters, the re-entry cooldowns, the previous review's ranking and verdicts (with their
reasons), and the abandoned-review streak in one
small JSON file per mode, written atomically. A missing or corrupt file is an empty state: the strategy then
simply starts its counters again. It never raises on an I/O problem: bookkeeping must not stop a review (the per-review
log is a `RunLog`, in `strategies/common/run_log.py`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_agent_framework.strategies.common.json_state import InvalidState, JsonStateStore, is_int
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "BillAckmanState")

STATE_VERSION = 2


@dataclass(frozen=True, slots=True)
class ReviewState:
    """What bill_ackman carries from one review to the next.

    Attributes:
        last_review: ISO date of the last completed review.
        fail_counts: Holding -> consecutive fails (always >= 1).
        last_ranking: The last ranking, best first.
        last_verdicts: Symbol -> `{verdict, reason, concern, date}`.
        abandoned_streak: Abandoned reviews in a row.
        cooldowns: Symbol -> completed reviews it stays out (always >= 1).
    """

    last_review: str | None = None  # ISO date of the last completed review
    fail_counts: dict[str, int] = field(default_factory=dict)  # holding -> consecutive fails (always >= 1)
    last_ranking: list[str] = field(default_factory=list)
    last_verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)  # symbol -> {verdict, reason, concern, date}
    abandoned_streak: int = 0  # abandoned reviews in a row
    cooldowns: dict[str, int] = field(default_factory=dict)  # symbol -> completed reviews it stays out (always >= 1)


_VERDICT_KEYS = {"verdict", "reason", "concern", "date"}


def _valid_counts(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(k, str) and is_int(v) and v >= 1 for k, v in value.items())


def _valid_verdict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _VERDICT_KEYS
        and value["verdict"] in ("survive", "fail")
        and isinstance(value["reason"], str)
        and (value["concern"] is None or isinstance(value["concern"], str))
        and isinstance(value["date"], str)
    )


def _valid(raw: dict[str, Any]) -> bool:
    last_review = raw.get("last_review")
    ranking, verdicts = raw.get("last_ranking", []), raw.get("last_verdicts", {})
    streak = raw.get("abandoned_streak", 0)
    return (
        (last_review is None or isinstance(last_review, str))
        and _valid_counts(raw.get("fail_counts", {}))
        and _valid_counts(raw.get("cooldowns", {}))
        and isinstance(ranking, list)
        and all(isinstance(symbol, str) for symbol in ranking)
        and isinstance(verdicts, dict)
        and all(isinstance(k, str) and _valid_verdict(v) for k, v in verdicts.items())
        and is_int(streak)
        and streak >= 0
    )


def _decode(raw: dict[str, Any]) -> ReviewState:
    if not _valid(raw):
        raise InvalidState
    return ReviewState(
        last_review=raw.get("last_review"),
        fail_counts=dict(raw.get("fail_counts", {})),
        last_ranking=list(raw.get("last_ranking", [])),
        last_verdicts={symbol: dict(record) for symbol, record in raw.get("last_verdicts", {}).items()},
        abandoned_streak=raw.get("abandoned_streak", 0),
        cooldowns=dict(raw.get("cooldowns", {})),
    )


class StateStore(JsonStateStore[ReviewState]):
    """JSON store of the bill_ackman state."""

    def __init__(self, path: Path, *, fresh: bool = False) -> None:
        """`fresh` wipes the file at construction; used in backtests so one run never inherits the last."""
        super().__init__(path, version=STATE_VERSION, empty=ReviewState, decode=_decode, encode=asdict, log=logger, fresh=fresh)
