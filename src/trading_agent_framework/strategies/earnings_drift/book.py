"""What earnings_drift remembers between cycles (spec §8).

`StateStore` keeps the open trades, the failure streaks and the symbols this strategy has ever traded (the orphan
rule) in one small JSON file per strategy name and mode, written atomically. A missing, unreadable or invalid file is an empty state
(with a warning). It never raises on an I/O problem: bookkeeping must not stop a cycle (the two run logs are `RunLog`s, in
`strategies/common/run_log.py`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

from trading_agent_framework.strategies.common.json_state import InvalidState, JsonStateStore
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
        backstop = data.get("backstop", False)
        if not isinstance(backstop, bool):  # bool("false") is True: a hand-edited flag must not be guessed
            raise ValueError(f"backstop must be true or false in trade {data.get('symbol')!r}")
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
            backstop=backstop,
            reaction_low=reaction_low,
        )


@dataclass
class DriftState:
    trades: dict[str, Trade] = field(default_factory=dict)  # symbol -> its pending or open trade
    agent_failure_streak: int = 0
    hollow_scan_streak: int = 0
    traded_symbols: set[str] = field(default_factory=set)  # every symbol this strategy has bought (orphan adoption)


def _streak(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid streak {value!r}")
    return value


def _decode(raw: dict[str, Any]) -> DriftState:
    trades = {symbol: Trade.from_json(data) for symbol, data in raw.get("trades", {}).items()}
    if any(symbol != trade.symbol for symbol, trade in trades.items()):
        raise InvalidState("a trade is filed under another symbol")
    traded = raw.get("traded_symbols", [])
    if not isinstance(traded, list) or not all(isinstance(s, str) for s in traded):
        raise InvalidState("invalid traded_symbols")
    return DriftState(
        trades=trades,
        agent_failure_streak=_streak(raw.get("agent_failure_streak", 0)),
        hollow_scan_streak=_streak(raw.get("hollow_scan_streak", 0)),
        traded_symbols=set(traded),
    )


def _encode(state: DriftState) -> dict[str, Any]:
    return {
        "trades": {symbol: trade.to_json() for symbol, trade in state.trades.items()},
        "agent_failure_streak": state.agent_failure_streak,
        "hollow_scan_streak": state.hollow_scan_streak,
        "traded_symbols": sorted(state.traded_symbols),
    }


class StateStore(JsonStateStore[DriftState]):
    def __init__(self, path: Path, *, fresh: bool = False) -> None:
        super().__init__(path, version=STATE_VERSION, empty=DriftState, decode=_decode, encode=_encode, log=logger, fresh=fresh)


def sessions_held(opened_on: date, today: date, trading_dates: Sequence[date]) -> int:
    """Sessions from `opened_on` through `today`, both included, counted on `trading_dates`."""
    return sum(1 for day in trading_dates if opened_on <= day <= today)
