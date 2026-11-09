"""What sector_pods remembers between weekly reviews: the last completed week's views and the abandoned streak."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from trading_agent_framework.strategies.common.json_state import InvalidState, JsonStateStore, is_int
from trading_agent_framework.utils.log import ColorLogger

STATE_VERSION = 1


@dataclass(frozen=True)
class WeekState:
    """Carried from one review to the next.

    Attributes:
        last_completed: ISO market date of the last completed review.
        abandoned_streak: Abandoned reviews in a row.
        previous: `{"date", "pods", "risk", "pm"}` of the last completed week; empty before the first one.
    """

    last_completed: str | None = None
    abandoned_streak: int = 0
    previous: dict[str, Any] = field(default_factory=dict)


def _decode(raw: dict[str, Any]) -> WeekState:
    last, streak, previous = raw.get("last_completed"), raw.get("abandoned_streak", 0), raw.get("previous", {})
    if last is not None:
        if not isinstance(last, str):
            raise InvalidState
        date.fromisoformat(last)  # a ValueError is an invalid state too
    if not is_int(streak) or streak < 0 or not isinstance(previous, dict):
        raise InvalidState
    return WeekState(last_completed=last, abandoned_streak=streak, previous=previous)


class StateStore(JsonStateStore[WeekState]):
    """JSON store of the sector_pods `WeekState`."""

    def __init__(self, path: Path, *, log: ColorLogger, fresh: bool = False) -> None:
        """Open the store.

        Args:
            path: State file.
            log: Logger for warnings about an unreadable or invalid file.
            fresh: Delete any existing file first, so a backtest never inherits the last run's state.
        """
        super().__init__(path, version=STATE_VERSION, empty=WeekState, decode=_decode, encode=asdict, log=log, fresh=fresh)
