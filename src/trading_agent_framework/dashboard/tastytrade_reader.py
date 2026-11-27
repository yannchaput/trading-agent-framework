"""A Tastytrade account for the Live and Paper tabs: its transactions, positions and end-of-day equity, as the types `pnl.py` replays.

Streamlit-free and GET only.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Protocol

from trading_agent_framework.brokers.tastytrade.orders import ACTION_SIDES, from_tastytrade_symbol, parse_position
from trading_agent_framework.brokers.tastytrade.values import decimal_of
from trading_agent_framework.dashboard.pnl import BrokerPosition, Dividend, Fee, Fill
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.utils.errors import BrokerError


class TastytradeReadSession(Protocol):
    """The only `TastytradeSession` calls the dashboard makes: GETs. No write method, so the reader cannot trade."""

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """GET a path and return its data."""
        ...

    def get_all(self, path: str, *, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """GET every item of a paginated path."""
        ...


# Money movements that are the owner's capital, not P&L: left out without a word.
_CAPITAL_MOVEMENTS = frozenset({"Deposit", "Withdrawal", "Transfer"})


def _number(item: Mapping[str, Any], field: str) -> float:
    """A numeric field as a float; `BrokerError` when it is missing or not a finite number."""
    return float(decimal_of(item.get(field), field))


def _signed(item: Mapping[str, Any], field: str) -> float:
    """A money field as a signed number: Tastytrade sends the size, and the direction in `<field>-effect`."""
    value = _number(item, field)
    return -value if item.get(f"{field}-effect") == "Debit" else value


def fetch_transactions(session: TastytradeReadSession, account_number: str) -> tuple[tuple[list[Fill], list[Fee], list[Dividend]], list[str]]:
    """Every equity fill, account fee and dividend (credits and their withholding), oldest first, and the money movements left out.

    A fill's fee is what `net-value` takes from `value`, so it covers every fee column of the trade. Of the money
    movements only the dividends and fees count: they also hold the deposits, withdrawals and transfers, which are
    capital and not P&L. Any other sub-type is not counted either, but described in the second result so the page can say so.
    """
    # shortcut: only equity trades are read, add the multiplier and the other types before showing an options or futures strategy
    fills: list[Fill] = []
    fees: list[Fee] = []
    dividends: list[Dividend] = []
    ignored: list[str] = []
    for item in session.get_all(f"/accounts/{account_number}/transactions", params={"sort": "Asc"}):
        kind = item.get("transaction-type")
        if kind == "Trade" and item.get("instrument-type") == "Equity":
            side = ACTION_SIDES.get(item.get("action", ""))
            if side is None:
                raise BrokerError(f"Tastytrade trade has an unknown action: {item.get('action')!r}")
            fills.append(
                Fill(
                    time=datetime.fromisoformat(item["executed-at"]),
                    symbol=from_tastytrade_symbol(item["symbol"]),
                    side=side.value,
                    qty=_number(item, "quantity"),
                    price=_number(item, "price"),
                    fee=round(_signed(item, "value") - _signed(item, "net-value"), 6),
                )
            )
        elif kind == "Money Movement":
            sub_type = item.get("transaction-sub-type")
            if sub_type == "Dividend":
                dividends.append(Dividend(date.fromisoformat(item["transaction-date"]), _signed(item, "net-value")))
            elif sub_type == "Fee":
                fees.append(Fee(date.fromisoformat(item["transaction-date"]), _signed(item, "net-value")))
            elif sub_type not in _CAPITAL_MOVEMENTS:
                ignored.append(f"{item['transaction-date']} {sub_type} {_signed(item, 'net-value'):g}")
    return (fills, fees, dividends), ignored


def _to_position(item: Mapping[str, Any]) -> BrokerPosition | None:
    """One open equity position priced at its mark, `None` for a closed or non-equity one.

    Tastytrade sends no unrealized figure: it is the move from its average open price, which leaves out the fees.
    """
    position = parse_position(item, "")  # the broker's own parser; the strategy name is not used here
    if position is None:
        return None
    quantity = float(position.quantity) * (-1 if position.side is PositionSide.SHORT else 1)
    average = _number(item, "average-open-price")
    mark = None if item.get("mark-price") is None else _number(item, "mark-price")
    unrealized = None if mark is None else round((mark - average) * quantity, 6)
    return BrokerPosition(
        symbol=position.asset.symbol,
        qty=quantity,
        avg_entry_price=average,
        current_price=mark,
        market_value=None if mark is None else round(mark * quantity, 6),
        unrealized_pl=unrealized,
        unrealized_plpc=None if unrealized is None or not average else round(unrealized / (average * abs(quantity)), 6),
    )


def fetch_positions(session: TastytradeReadSession, account_number: str) -> list[BrokerPosition]:
    """Every open equity position with its current mark."""
    items = session.get(f"/accounts/{account_number}/positions", params={"include-marks": "true"}).get("items") or []
    return [position for position in map(_to_position, items) if position is not None]


def fetch_equity_snapshots(session: TastytradeReadSession, account_number: str, start: date, today: date) -> dict[date, float]:
    """The account's net liquidating value at each close from `start` to `today` (pending deposits are not in it)."""
    params = {"start-date": start.isoformat(), "end-date": today.isoformat(), "time-of-day": "EOD"}
    snapshots = session.get_all(f"/accounts/{account_number}/balance-snapshots", params=params)
    # The filter is repeated here: a row without a time of day is the balance now, and must never replace a close.
    return {date.fromisoformat(s["snapshot-date"]): _number(s, "net-liquidating-value") for s in snapshots if s.get("time-of-day") == "EOD"}
