"""The per-run log shared by the strategies that run once per filing: `runs.jsonl`, one JSON line per run."""

import json
import logging
from pathlib import Path
from typing import Any

from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "RunLog")


class RunLog:
    """`runs.jsonl`: one JSON line per run. With no path (a strategy run outside a runner) it does nothing; it never raises on an I/O problem."""

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
            logger.log_warning(f"run log {self._path} could not be written: {exc}")
