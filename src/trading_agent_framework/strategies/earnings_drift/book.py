"""What earnings_drift remembers between cycles, and its two run logs (spec §8).

`StateStore` keeps the open trades, the failure streaks and the symbols this strategy has ever traded (the orphan
rule) in one small JSON file per strategy name and mode, written atomically. A missing, unreadable or invalid file is an empty state
(with a warning). `JsonlLog` appends JSON lines to the run directory. Neither raises on an I/O problem: bookkeeping
must not stop a cycle.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "EarningsDriftState")

STATE_VERSION = 1


class TradeState(StrEnum):
    PENDING = "pending"  # entry submitted, not filled yet
    OPEN = "open"  # entry filled: shares held, protected by a stop (or being sold)


def _dec(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


@dataclass
class Trade:
    symbol: str
    entry_order_id: str
    trail_percent: Decimal
    thesis: str
    accession_number: str | None = None
    state: TradeState = TradeState.PENDING
    quantity: Decimal = Decimal(0)  # filled shares
    entry_price: Decimal | None = None
    opened_on: date | None = None  # the entry fill's session date
    stop_order_id: str | None = None
    exit_order_id: str | None = None
    exit_reason: str | None = None  # set when an exit sell is submitted
    backstop: bool = False  # the stop was (re)placed by a guardrail; shown once to the agent
    reaction_low: Decimal | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_order_id": self.entry_order_id,
            "trail_percent": str(self.trail_percent),
            "thesis": self.thesis,
            "accession_number": self.accession_number,
            "state": self.state.value,
            "quantity": str(self.quantity),
            "entry_price": None if self.entry_price is None else str(self.entry_price),
            "opened_on": None if self.opened_on is None else self.opened_on.isoformat(),
            "stop_order_id": self.stop_order_id,
            "exit_order_id": self.exit_order_id,
            "exit_reason": self.exit_reason,
            "backstop": self.backstop,
            "reaction_low": None if self.reaction_low is None else str(self.reaction_low),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Trade:
        try:
            trail = Decimal(str(data["trail_percent"]))
            quantity = Decimal(str(data["quantity"]))
            entry_price, reaction_low = _dec(data.get("entry_price")), _dec(data.get("reaction_low"))
        except InvalidOperation as exc:
            raise ValueError(f"invalid number in trade {data.get('symbol')!r}") from exc
        for number in (quantity, trail, entry_price, reaction_low):
            if number is not None and not number.is_finite():
                raise ValueError(f"non-finite number in trade {data.get('symbol')!r}")
        if quantity < 0 or trail <= 0:
            raise ValueError(f"quantity must be >= 0 and trail_percent > 0 in trade {data.get('symbol')!r}")
        opened = data.get("opened_on")
        if not isinstance(data["symbol"], str) or not isinstance(data["entry_order_id"], str) or not isinstance(data["thesis"], str):
            raise TypeError("symbol, entry_order_id and thesis must be strings")
        return cls(
            symbol=data["symbol"],
            entry_order_id=data["entry_order_id"],
            trail_percent=trail,
            thesis=data["thesis"],
            accession_number=data.get("accession_number"),
            state=TradeState(data["state"]),
            quantity=quantity,
            entry_price=entry_price,
            opened_on=None if opened is None else date.fromisoformat(opened),
            stop_order_id=data.get("stop_order_id"),
            exit_order_id=data.get("exit_order_id"),
            exit_reason=data.get("exit_reason"),
            backstop=bool(data.get("backstop", False)),
            reaction_low=reaction_low,
        )


@dataclass
class DriftState:
    trades: dict[str, Trade] = field(default_factory=dict)  # symbol -> its pending or open trade
    agent_failure_streak: int = 0
    hollow_scan_streak: int = 0
    traded_symbols: set[str] = field(default_factory=set)  # every symbol this strategy has bought (orphan adoption)


def state_path(project_root: Path, mode: TradingMode, name: str = "earnings_drift") -> Path:
    """`<project_root>/data/<name>_state_<mode>.json`: one file per strategy name, so the agent and the baseline never share one."""
    return project_root / "data" / f"{name}_state_{mode.value}.json"


def _streak(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid streak {value!r}")
    return value


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> DriftState:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return DriftState()
        except (OSError, ValueError) as exc:
            logger.log_warning(f"state file {self._path} is unreadable, starting from an empty state: {exc}")
            return DriftState()
        try:
            if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
                raise ValueError("missing or other version")
            trades = {symbol: Trade.from_json(data) for symbol, data in raw.get("trades", {}).items()}
            if any(symbol != trade.symbol for symbol, trade in trades.items()):
                raise ValueError("a trade is filed under another symbol")
            traded = raw.get("traded_symbols", [])
            if not isinstance(traded, list) or not all(isinstance(s, str) for s in traded):
                raise ValueError("invalid traded_symbols")
            return DriftState(
                trades=trades,
                agent_failure_streak=_streak(raw.get("agent_failure_streak", 0)),
                hollow_scan_streak=_streak(raw.get("hollow_scan_streak", 0)),
                traded_symbols=set(traded),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.log_warning(f"state file {self._path} is not a valid state, starting from an empty state: {exc}")
            return DriftState()

    def save(self, state: DriftState) -> None:
        """Write the state atomically (temporary file in the same directory, then replace); an I/O error is logged."""
        payload = {
            "version": STATE_VERSION,
            "trades": {symbol: trade.to_json() for symbol, trade in state.trades.items()},
            "agent_failure_streak": state.agent_failure_streak,
            "hollow_scan_streak": state.hollow_scan_streak,
            "traded_symbols": sorted(state.traded_symbols),
        }
        temporary: str | None = None
        try:
            text = json.dumps(payload, default=str)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self._path.parent, prefix=f"{self._path.name}.", suffix=".tmp")
            try:
                handle = os.fdopen(descriptor, "w", encoding="utf-8")
            except BaseException:
                os.close(descriptor)
                raise
            with handle:
                handle.write(text)
            os.replace(temporary, self._path)
        except (OSError, TypeError, ValueError) as exc:
            logger.log_warning(f"state could not be saved to {self._path}: {exc}")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def wipe(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.log_warning(f"state file {self._path} could not be deleted: {exc}")


class JsonlLog:
    """One JSON line per record in the run directory; with no path (outside a runner) it does nothing."""

    def __init__(self, path: Callable[[], Path | None]) -> None:
        self._path = path

    def append(self, record: dict[str, Any]) -> None:
        path = self._path()
        if path is None:
            return
        try:
            line = json.dumps(record, default=str) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except (OSError, TypeError, ValueError) as exc:
            logger.log_warning(f"could not append to {path}: {exc}")


def sessions_held(opened_on: date, today: date, trading_dates: Sequence[date]) -> int:
    """Sessions from `opened_on` through `today`, both included, counted on `trading_dates`."""
    return sum(1 for day in trading_dates if opened_on <= day <= today)
