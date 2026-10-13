"""Average-cost P&L, pure: the backtest's per-session realized P&L and the Live/Paper tabs' daily P&L."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution. `side` "buy" is a buy, anything else a sell; `fee` is paid on top, in dollars."""

    time: datetime
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0


@dataclass(frozen=True, slots=True)
class Sale:
    """The part of a sell fill that closed held shares, against their average cost."""

    symbol: str
    qty: float
    gross: float  # qty * sell price
    fees: float  # the sell fee's share for these shares
    basis: float  # the sold shares' average cost, buy fees included

    @property
    def pnl(self) -> float:
        return self.gross - self.fees - self.basis


class AverageCostBook:
    """Replays time-ordered fills: each symbol's shares and remaining cost (every buy fee in it). A sale leaves at
    the average cost and pays its own fee; the shares sold beyond what is held are counted in `unmatched`."""

    def __init__(self) -> None:
        self._holdings: dict[str, tuple[float, float]] = {}
        self.unmatched: dict[str, float] = {}

    def apply(self, fill: Fill) -> Sale | None:
        shares, cost = self._holdings.get(fill.symbol, (0.0, 0.0))
        sale = None
        if fill.side.lower() == "buy":
            shares, cost = shares + fill.qty, cost + fill.qty * fill.price + fill.fee
        else:
            closed = min(fill.qty, shares) if shares > _EPS else 0.0
            if fill.qty - closed > _EPS:
                self.unmatched[fill.symbol] = self.unmatched.get(fill.symbol, 0.0) + fill.qty - closed
            if closed > 0:
                basis = cost * closed / shares
                sale = Sale(fill.symbol, closed, closed * fill.price, fill.fee * closed / fill.qty, basis)
                shares, cost = shares - closed, cost - basis
        if shares > _EPS:
            self._holdings[fill.symbol] = (shares, cost)
        else:
            self._holdings.pop(fill.symbol, None)
        return sale

    def holdings(self) -> dict[str, tuple[float, float]]:
        return dict(self._holdings)
