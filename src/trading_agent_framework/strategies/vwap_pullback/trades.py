"""Pure trade records: one `Trade` per entry and the session's `TradeBook`.

Accounting model (the desk relies on it everywhere):
- `Trade.quantity` is the number of shares held *as the desk knows it*. It only changes when a fill is
  BOOKED: an entry fill (`record_entry_fill`) or an exit fill (`record_exit_fill`), which `Desk` calls from
  the order hooks. A fill the broker has reported on an order but that is not booked yet is still inside
  `quantity`.
- `Trade.exit_order_ids` lists the exit sells not booked yet; the desk removes an id once it books that
  order. So "shares still free to sell or protect" = `quantity` minus the whole quantity of every listed
  exit sell (see `Desk._free_quantity`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

_CENT = Decimal("0.01")


class TradeStatus(StrEnum):
    PENDING = "pending"  # entry order working, nothing filled
    OPEN = "open"  # entry (partly) filled: shares held, protected by a stop
    CLOSED = "closed"  # every share sold; moved to `TradeBook.closed`


@dataclass
class Trade:
    """One entry and everything the desk knows about the position it opened (mutable, updated in place)."""

    symbol: str
    entry_order_id: str
    planned_quantity: Decimal  # the size the entry order asked for
    stop_price: Decimal  # the initial stop, which defines R
    r_per_share: Decimal
    entered_at: datetime
    status: TradeStatus = TradeStatus.PENDING
    quantity: Decimal = Decimal(0)  # shares held now (as booked; see the module docstring)
    filled_quantity: Decimal = Decimal(0)  # shares bought in total
    entry_price: Decimal | None = None  # average entry fill price
    stop_level: Decimal = Decimal(0)  # the working stop's level (the initial stop: nothing moves it)
    stop_order_id: str | None = None  # the working protective stop (None while it is being replaced)
    exit_order_ids: list[str] = field(default_factory=list)  # exit sells not booked yet
    exit_reason: str | None = None
    realised_pnl: Decimal = Decimal(0)
    closed_at: datetime | None = None

    def __post_init__(self) -> None:
        # The working stop starts at the initial stop.
        if not self.stop_level:
            self.stop_level = self.stop_price

    def order_ids(self) -> set[str]:
        """Every order id that can still move this trade: the entry, the working stop, the unbooked exits."""
        return {i for i in (self.entry_order_id, self.stop_order_id, *self.exit_order_ids) if i is not None}

    def record_entry_fill(self, total_filled: Decimal, avg_price: Decimal) -> None:
        """The entry's cumulative fill (not an increment): holdings and entry price follow it."""
        self.filled_quantity = total_filled
        self.quantity = total_filled
        self.entry_price = avg_price
        self.status = TradeStatus.OPEN

    def record_exit_fill(self, quantity: Decimal, price: Decimal, at: datetime) -> None:
        """Book `quantity` shares sold at `price`: realise their P&L and close the trade when nothing is left."""
        if self.entry_price is not None:
            self.realised_pnl = (self.realised_pnl + (price - self.entry_price) * quantity).quantize(_CENT)
        self.quantity -= quantity
        if self.quantity <= 0:
            self.quantity = Decimal(0)
            self.status = TradeStatus.CLOSED
            self.closed_at = at

    def unrealised_pnl(self, last_price: Decimal) -> Decimal:
        """Open P&L of the shares still held at `last_price` (0 before the entry fills)."""
        if self.entry_price is None:
            return Decimal(0)
        return ((last_price - self.entry_price) * self.quantity).quantize(_CENT)

    def to_json(self) -> dict[str, object]:
        """One line of `trades.jsonl` (written when the trade closes); Decimals as strings to stay exact."""
        risked = self.r_per_share * self.filled_quantity
        return {
            "symbol": self.symbol,
            "entered_at": self.entered_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "entry_price": str(self.entry_price) if self.entry_price is not None else None,
            "filled_quantity": str(self.filled_quantity),
            "stop_price": str(self.stop_price),
            "r_per_share": str(self.r_per_share),
            "realised_pnl": str(self.realised_pnl),
            "realised_r": round(float(self.realised_pnl / risked), 2) if risked > 0 else None,
            "exit_reason": self.exit_reason,
        }


class TradeBook:
    """The session's trades: active ones by symbol (one per symbol), closed ones in order."""

    def __init__(self) -> None:
        self._active: dict[str, Trade] = {}  # PENDING and OPEN trades, one per symbol
        self.closed: list[Trade] = []

    def add(self, trade: Trade) -> None:
        self._active[trade.symbol] = trade

    def get(self, symbol: str) -> Trade | None:
        """The active (pending or open) trade in `symbol`, if any."""
        return self._active.get(symbol)

    def active(self) -> list[Trade]:
        return list(self._active.values())

    def open_trades(self) -> list[Trade]:
        return [t for t in self._active.values() if t.status is TradeStatus.OPEN]

    def pending(self) -> list[Trade]:
        return [t for t in self._active.values() if t.status is TradeStatus.PENDING]

    def by_order_id(self, order_id: str) -> Trade | None:
        """The active trade an order belongs to (how order hooks find their trade); None for an unknown id."""
        return next((t for t in self._active.values() if order_id in t.order_ids()), None)

    def discard(self, symbol: str) -> None:
        """Forget a trade whose entry ended unfilled (nothing to archive)."""
        self._active.pop(symbol, None)

    def archive(self, trade: Trade) -> None:
        """Move a closed trade out of the active set."""
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
