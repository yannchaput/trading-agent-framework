"""Average-cost P&L, pure: the backtest's per-session realized P&L and the Live/Paper tabs' daily P&L."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from trading_agent_framework.utils.clock import MARKET_TZ

# Earning Per Share threshold for floating-point comparisons: a share's cost or proceeds below this is treated as zero.
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


@dataclass(frozen=True, slots=True)
class Fee:
    """A fee activity, signed like Alpaca's `net_amount` (a fee is negative)."""

    day: date
    amount: float


@dataclass(frozen=True, slots=True)
class Dividend:
    """A dividend credit or its tax withholding, signed like Alpaca's `net_amount` (a withholding is negative)."""

    day: date
    amount: float


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """An open position as the broker reports it now."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float | None
    market_value: float | None
    unrealized_pl: float | None
    unrealized_plpc: float | None


@dataclass(frozen=True, slots=True)
class AccountPnl:
    """`days`: `{date, realized, unrealized, change}` per trading day from the first fill to today, oldest first.
    `unrealized` is a level (the open positions' P&L at the close) and `realized` that day's flow, so neither is the
    day's P&L: `change` is `realized` plus the move of `unrealized` since the previous day (from 0 before the first fill).
    `unrealized_now` is on the replay's average cost, the basis of the realized P&L; `broker_unrealized` is Alpaca's
    own figure (it re-bases overnight positions to compressed FIFO, so the two differ after a partial sale).
    `mismatches`: `(symbol, replayed shares, broker shares)` where the fills do not explain the positions;
    `unmatched_sales`: shares sold with no held shares to cost them against; `unpriced`: held symbols with no close."""

    days: list[dict[str, Any]]
    realized_total: float
    unrealized_now: float
    broker_unrealized: float
    positions: list[BrokerPosition]
    mismatches: list[tuple[str, float, float]]
    unmatched_sales: dict[str, float]
    unpriced: list[str]


def market_day(moment: datetime) -> date:
    return moment.astimezone(MARKET_TZ).date()


def days_between(days: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    """The `AccountPnl.days` rows dated from `start` to `end`, both included (empty when `start` is after `end`)."""
    return [day for day in days if start <= date.fromisoformat(day["date"]) <= end]


def _close_on_or_before(closes: Mapping[date, float], day: date) -> float | None:
    earlier = [d for d in closes if d <= day]
    return closes[max(earlier)] if earlier else None


def _unrealized_today(positions: Sequence[BrokerPosition], held: Mapping[str, tuple[float, float]]) -> float:
    """Today's unrealized P&L on the replay's cost basis: the replayed shares at the position's current price, against
    their replayed cost. A position the fills do not explain (other quantity, not replayed, no price) keeps the
    broker's own `unrealized_pl`."""
    total = 0.0
    for position in positions:
        replayed = held.get(position.symbol)
        if replayed is not None and position.current_price is not None and abs(replayed[0] - position.qty) <= 1e-6:
            total += replayed[0] * position.current_price - replayed[1]
        else:
            total += position.unrealized_pl or 0.0
    return total


def daily_account_pnl(
    fills: Sequence[Fill],
    fees: Sequence[Fee],
    closes: Mapping[str, Mapping[date, float]],
    positions: Sequence[BrokerPosition],
    today: date,
    dividends: Sequence[Dividend] = (),
) -> AccountPnl:
    """Realized P&L of each day's sales (net of fees, at average cost, plus that day's fee and dividend activities) and the
    unrealized P&L of the shares held at each day's close; today's unrealized is the broker's, not a close's.
    A day is any day with a close, a fill, a fee or a dividend between the first fill and today, and today."""
    broker_unrealized = sum(p.unrealized_pl or 0.0 for p in positions)
    unrealized_now = broker_unrealized  # replaced below by the replay's figure once today's fills are in
    ordered = sorted(fills, key=lambda f: f.time)
    book = AverageCostBook()
    days: list[dict[str, Any]] = []
    unpriced: set[str] = set()
    previous_unrealized = 0.0
    if ordered:
        fills_by_day: dict[date, list[Fill]] = {}
        for fill in ordered:
            fills_by_day.setdefault(market_day(fill.time), []).append(fill)
        cash_by_day: dict[date, float] = {}  # fee activities and dividends, both signed like Alpaca's net_amount
        for activity in (*fees, *dividends):
            cash_by_day[activity.day] = cash_by_day.get(activity.day, 0.0) + activity.amount
        first = market_day(ordered[0].time)
        calendar = {d for series in closes.values() for d in series} | set(fills_by_day) | set(cash_by_day) | {today}
        for day in sorted(d for d in calendar if first <= d <= today):
            realized = cash_by_day.get(day, 0.0)
            for fill in fills_by_day.get(day, []):
                sale = book.apply(fill)
                if sale is not None:
                    realized += sale.pnl
            if day == today:
                unrealized_now = _unrealized_today(positions, book.holdings())
                unrealized = unrealized_now
            else:
                unrealized = 0.0
                for symbol, (shares, cost) in book.holdings().items():
                    close = _close_on_or_before(closes.get(symbol, {}), day)
                    if close is None:
                        unpriced.add(symbol)
                    else:
                        unrealized += shares * close - cost
            days.append(
                {
                    "date": day.isoformat(),
                    "realized": round(realized, 2),
                    "unrealized": round(unrealized, 2),
                    "change": round(realized + unrealized - previous_unrealized, 2),
                }
            )
            previous_unrealized = unrealized

    held = {symbol: shares for symbol, (shares, _) in book.holdings().items()}
    broker = {p.symbol: p.qty for p in positions}
    mismatches = [(symbol, held.get(symbol, 0.0), broker.get(symbol, 0.0)) for symbol in sorted(held.keys() | broker.keys()) if abs(held.get(symbol, 0.0) - broker.get(symbol, 0.0)) > 1e-6]
    return AccountPnl(
        days=days,
        realized_total=round(sum(d["realized"] for d in days), 2),
        unrealized_now=round(unrealized_now, 2),
        broker_unrealized=round(broker_unrealized, 2),
        positions=list(positions),
        mismatches=mismatches,
        unmatched_sales=dict(book.unmatched),
        unpriced=sorted(unpriced),
    )
