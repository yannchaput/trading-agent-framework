"""What the strategy remembers between daily checks, and the per-run log.

`StateStore` keeps, in one small JSON file per mode written atomically: the DocIDs of the filings the last completed run was
built from (`processed`, the "nothing new" test), that run's holdings and target, the tickers this strategy has ever ordered
(`traded`: the only positions it may sell), a trade left unfilled (`pending_trade`) and the abandoned-run streak. A missing or
corrupt file is an empty state: the next check then treats every known filing as new and builds the portfolio again.
`RunLog` appends one line per run to the run directory. Neither ever raises on an I/O problem: bookkeeping must not stop a run.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "CongressTradesState")

STATE_VERSION = 1


def state_path(project_root: Path, mode: TradingMode) -> Path:
    """`<project_root>/data/congress_trades_state_<mode>.json`."""
    return project_root / "data" / f"congress_trades_state_{mode.value}.json"


@dataclass(frozen=True, slots=True)
class PendingTrade:
    """A target the trading stage could not reach: later ticks re-run only that stage against it, `days` ticks used so far."""

    target: dict[str, float]  # ticker -> weight, as submitted by the portfolio agent
    days: int = 0


@dataclass(frozen=True, slots=True)
class CongressState:
    processed: list[str] = field(default_factory=list)  # DocIDs of the yearly report and PTRs the last completed run was built from
    holdings: dict[str, dict[str, int]] = field(default_factory=dict)  # ticker -> {tier, value_low, value_high} after that run
    target: dict[str, float] = field(default_factory=dict)  # ticker -> weight of that run's portfolio
    traded: list[str] = field(default_factory=list)  # tickers ever ordered by this strategy
    pending_trade: PendingTrade | None = None
    abandoned_streak: int = 0  # abandoned runs in a row
    last_run: str | None = None  # ISO date of the last completed run


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_weight(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _valid_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _valid_weights(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(k, str) and _is_weight(v) for k, v in value.items())


def _valid_holdings(value: Any) -> bool:
    keys = {"tier", "value_low", "value_high"}
    return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, dict) and set(v) == keys and all(_is_int(n) for n in v.values()) for k, v in value.items())


def _valid_pending(value: Any) -> bool:
    return value is None or (isinstance(value, dict) and set(value) == {"target", "days"} and _valid_weights(value["target"]) and _is_int(value["days"]) and value["days"] >= 0)


def _valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return False
    last_run = raw.get("last_run")
    streak = raw.get("abandoned_streak", 0)
    return (
        _valid_strings(raw.get("processed", []))
        and _valid_holdings(raw.get("holdings", {}))
        and _valid_weights(raw.get("target", {}))
        and _valid_strings(raw.get("traded", []))
        and _valid_pending(raw.get("pending_trade"))
        and _is_int(streak)
        and streak >= 0
        and (last_run is None or isinstance(last_run, str))
    )


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> CongressState:
        """The saved state; an empty one when the file is missing, unreadable, or not a valid state (warned about)."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return CongressState()
        except (OSError, ValueError) as exc:
            logger.log_warning(f"state file {self._path} is unreadable, starting from an empty state: {exc}")
            return CongressState()
        if not _valid(raw):
            logger.log_warning(f"state file {self._path} is not a valid state, starting from an empty state")
            return CongressState()
        pending = raw.get("pending_trade")
        return CongressState(
            processed=list(raw.get("processed", [])),
            holdings={ticker: dict(record) for ticker, record in raw.get("holdings", {}).items()},
            target=dict(raw.get("target", {})),
            traded=list(raw.get("traded", [])),
            pending_trade=None if pending is None else PendingTrade(target=dict(pending["target"]), days=pending["days"]),
            abandoned_streak=raw.get("abandoned_streak", 0),
            last_run=raw.get("last_run"),
        )

    def save(self, state: CongressState) -> None:
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


class RunLog:
    """`runs.jsonl`: one JSON line per run. With no path (a strategy run outside a runner) it does nothing."""

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
