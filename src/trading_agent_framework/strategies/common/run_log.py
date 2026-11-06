"""The per-run JSONL log shared by the strategies: one JSON line per record in the run directory."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "RunLog")


class RunLog:
    """One JSON line per record. With no path (a strategy run outside a runner) it does nothing; it never raises on an I/O or serialization problem.

    `path` may be a callable, read at every append (the run directory is known only once the run starts).
    """

    def __init__(self, path: Path | Callable[[], Path | None] | None, *, what: str = "run log") -> None:
        self._path = path
        self._what = what

    def append(self, record: dict[str, Any]) -> None:
        path = self._path() if callable(self._path) else self._path
        if path is None:
            return
        try:
            line = json.dumps(record, default=str) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except (OSError, TypeError, ValueError) as exc:
            logger.log_warning(f"{self._what} {path} could not be written: {exc}")
