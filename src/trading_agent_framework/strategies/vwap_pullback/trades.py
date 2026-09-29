"""Pure trade records (spec §5): one `Trade` per entry, the session's `TradeBook`, and the exit-review triggers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters

_CENT = Decimal("0.01")


class TradeStatus(StrEnum):
    PENDING = "pending"  # entry order working, nothing filled
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Trade:
    symbol: str
    entry_order_id: str
    planned_quantity: Decimal
    stop_price: Decimal  # the initial stop, which defines R
    r_per_share: Decimal
    catalyst: str
    reason: str
    entered_at: datetime
    status: TradeStatus = TradeStatus.PENDING
    quantity: Decimal = Decimal(0)  # shares held now
    filled_quantity: Decimal = Decimal(0)  # shares bought in total
    entry_price: Decimal | None = None
    stop_level: Decimal = Decimal(0)  # the working stop's level (initial stop, raised, or the trail's start)
    stop_kind: str | None = None  # "stop" | "trail"
    trail_price: Decimal | None = None
    stop_order_id: str | None = None
    exit_order_ids: list[str] = field(default_factory=list)
    tp1_done: bool = False
    last_review_at: datetime | None = None
    review_flags: frozenset[str] = frozenset()
    exit_reason: str | None = None
    realised_pnl: Decimal = Decimal(0)
    closed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.stop_level:
            self.stop_level = self.stop_price

    def order_ids(self) -> set[str]:
        return {i for i in (self.entry_order_id, self.stop_order_id, *self.exit_order_ids) if i is not None}

    def record_entry_fill(self, total_filled: Decimal, avg_price: Decimal) -> None:
        """The entry's cumulative fill (not an increment): holdings and entry price follow it."""
        self.filled_quantity = total_filled
        self.quantity = total_filled
        self.entry_price = avg_price
        self.status = TradeStatus.OPEN

    def record_exit_fill(self, quantity: Decimal, price: Decimal, at: datetime) -> None:
        if self.entry_price is not None:
            self.realised_pnl = (self.realised_pnl + (price - self.entry_price) * quantity).quantize(_CENT)
        self.quantity -= quantity
        if self.quantity <= 0:
            self.quantity = Decimal(0)
            self.status = TradeStatus.CLOSED
            self.closed_at = at

    def unrealised_pnl(self, last_price: Decimal) -> Decimal:
        if self.entry_price is None:
            return Decimal(0)
        return ((last_price - self.entry_price) * self.quantity).quantize(_CENT)

    def unrealised_r(self, last_price: Decimal) -> float:
        if self.entry_price is None or self.r_per_share <= 0:
            return 0.0
        return round(float((last_price - self.entry_price) / self.r_per_share), 2)

    def to_json(self) -> dict[str, object]:
        risked = self.r_per_share * self.filled_quantity
        return {
            "symbol": self.symbol,
            "catalyst": self.catalyst,
            "reason": self.reason,
            "entered_at": self.entered_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "entry_price": str(self.entry_price) if self.entry_price is not None else None,
            "filled_quantity": str(self.filled_quantity),
            "stop_price": str(self.stop_price),
            "r_per_share": str(self.r_per_share),
            "realised_pnl": str(self.realised_pnl),
            "realised_r": round(float(self.realised_pnl / risked), 2) if risked > 0 else None,
            "tp1_done": self.tp1_done,
            "stop_kind": self.stop_kind,
            "exit_reason": self.exit_reason,
        }


class TradeBook:
    """The session's trades: active ones by symbol (one per symbol), closed ones in order."""

    def __init__(self) -> None:
        self._active: dict[str, Trade] = {}
        self.closed: list[Trade] = []

    def add(self, trade: Trade) -> None:
        self._active[trade.symbol] = trade

    def get(self, symbol: str) -> Trade | None:
        return self._active.get(symbol)

    def active(self) -> list[Trade]:
        return list(self._active.values())

    def open_trades(self) -> list[Trade]:
        return [t for t in self._active.values() if t.status is TradeStatus.OPEN]

    def pending(self) -> list[Trade]:
        return [t for t in self._active.values() if t.status is TradeStatus.PENDING]

    def by_order_id(self, order_id: str) -> Trade | None:
        return next((t for t in self._active.values() if order_id in t.order_ids()), None)

    def discard(self, symbol: str) -> None:
        self._active.pop(symbol, None)

    def archive(self, trade: Trade) -> None:
        self._active.pop(trade.symbol, None)
        self.closed.append(trade)

    def session_pnl(self, last_prices: Mapping[str, Decimal]) -> Decimal:
        """Realised P&L of every trade plus the open trades' unrealised P&L at `last_prices` (a missing price counts 0)."""
        total = sum((t.realised_pnl for t in [*self.closed, *self._active.values()]), Decimal(0))
        for trade in self.open_trades():
            price = last_prices.get(trade.symbol)
            if price is not None:
                total += trade.unrealised_pnl(price)
        return total


def trade_flags(trade: Trade, *, last_close: Decimal, vwap: float | None, ema: float | None) -> frozenset[str]:
    """The exit-review events that currently hold for `trade`: +1R reached (before TP1), close below VWAP, close below the EMA."""
    flags: set[str] = set()
    if trade.entry_price is not None and not trade.tp1_done and last_close >= trade.entry_price + trade.r_per_share:
        flags.add("reached_1r")
    if vwap is not None and last_close < Decimal(str(vwap)):
        flags.add("below_vwap")
    if ema is not None and last_close < Decimal(str(ema)):
        flags.add("below_ema")
    return frozenset(flags)


def exit_review_due(trade: Trade, flags: frozenset[str], *, now: datetime, has_new_headline: bool, params: VwapPullbackParameters) -> bool:
    """Whether the exit agent should look at `trade` now: a flag that was not there at the last review, a new headline, or enough time."""
    if trade.status is not TradeStatus.OPEN:
        return False
    if flags - trade.review_flags or has_new_headline:
        return True
    since = trade.last_review_at or trade.entered_at
    return now - since >= timedelta(minutes=params.exit_review_minutes)
