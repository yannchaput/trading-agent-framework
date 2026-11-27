from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest
from tests.fakes import FakeTastytradeSession

from trading_agent_framework.dashboard import tastytrade_reader
from trading_agent_framework.dashboard.pnl import BrokerPosition, Dividend, Fee, Fill
from trading_agent_framework.utils.errors import BrokerError

ACCOUNT = "5WX00000"


def _session(resource: str, *items: dict[str, Any]) -> FakeTastytradeSession:
    """The shared fake with one route: any other path is a 404."""
    session = FakeTastytradeSession()
    session.responses[("GET", f"/accounts/{ACCOUNT}/{resource}")] = {"items": list(items)}
    return session


def _calls(session: FakeTastytradeSession) -> list[tuple[str, Any]]:
    return [(path, params) for _, path, _, params in session.calls]


def _trade(action: str, symbol: str = "XLP", quantity: str = "0.5", price: str = "92.0", value: str = "46.0", net: str = "46.1", effect: str = "Debit") -> dict[str, Any]:
    return {
        "transaction-type": "Trade",
        "instrument-type": "Equity",
        "action": action,
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "value": value,
        "value-effect": effect,
        "net-value": net,
        "net-value-effect": effect,
        "executed-at": "2026-11-27T16:53:29.670Z",
        "transaction-date": "2026-11-27",
    }


def _movement(sub_type: str, amount: str, effect: str = "Credit", day: str = "2026-11-30") -> dict[str, Any]:
    return {"transaction-type": "Money Movement", "transaction-sub-type": sub_type, "net-value": amount, "net-value-effect": effect, "transaction-date": day}


def _transactions(*items: dict[str, Any]) -> tuple[list[Fill], list[Fee], list[Dividend]]:
    activities, _ = tastytrade_reader.fetch_transactions(_session("transactions", *items), ACCOUNT)
    return activities


def test_a_buys_fee_is_what_the_net_value_adds_to_the_trade_value() -> None:
    fills, _, _ = _transactions(_trade("Buy to Open"))

    (fill,) = fills
    assert (fill.symbol, fill.side, fill.qty, fill.price) == ("XLP", "buy", 0.5, 92.0)
    assert fill.fee == pytest.approx(0.1)
    assert fill.time == datetime.fromisoformat("2026-11-27T16:53:29.670+00:00")


def test_a_sells_fee_is_what_the_net_value_takes_off_the_proceeds() -> None:
    fills, _, _ = _transactions(_trade("Sell to Close", value="47.0", net="46.85", effect="Credit"))

    assert fills[0].side == "sell"
    assert fills[0].fee == pytest.approx(0.15)


def test_a_class_share_gets_the_projects_symbol() -> None:
    fills, _, _ = _transactions(_trade("Buy to Open", symbol="BRK/B"))

    assert fills[0].symbol == "BRK.B"


def test_dividends_and_fees_are_signed_cash_and_capital_moved_is_not_pnl() -> None:
    _, fees, dividends = _transactions(
        _movement("Deposit", "558.0"),
        _movement("Withdrawal", "20.0", "Debit"),
        _movement("Transfer", "5.0"),
        _movement("Dividend", "1.2"),
        _movement("Dividend", "0.18", "Debit"),  # the non-resident tax withheld
        _movement("Fee", "2.0", "Debit"),
    )

    assert dividends == [Dividend(date(2026, 11, 30), 1.2), Dividend(date(2026, 11, 30), -0.18)]
    assert fees == [Fee(date(2026, 11, 30), -2.0)]


def test_transactions_are_asked_oldest_first() -> None:
    session = _session("transactions")

    tastytrade_reader.fetch_transactions(session, ACCOUNT)

    assert _calls(session) == [(f"/accounts/{ACCOUNT}/transactions", {"sort": "Asc"})]


def _position(**overrides: Any) -> dict[str, Any]:
    return {"instrument-type": "Equity", "symbol": "XLU", "quantity": "2.0", "quantity-direction": "Long", "average-open-price": "40.0", "mark-price": "41.0", **overrides}


def test_positions_are_priced_at_the_mark_which_is_asked_for() -> None:
    session = _session("positions", _position())

    positions = tastytrade_reader.fetch_positions(session, ACCOUNT)

    assert positions == [BrokerPosition("XLU", 2.0, 40.0, 41.0, 82.0, 2.0, 0.025)]
    assert _calls(session) == [(f"/accounts/{ACCOUNT}/positions", {"include-marks": "true"})]


def test_a_short_position_is_negative_and_gains_when_the_mark_falls() -> None:
    session = _session("positions", _position(**{"quantity-direction": "Short", "mark-price": "39.0"}))

    (position,) = tastytrade_reader.fetch_positions(session, ACCOUNT)

    assert (position.qty, position.market_value, position.unrealized_pl, position.unrealized_plpc) == (-2.0, -78.0, 2.0, 0.025)


def test_a_position_without_a_mark_has_no_price_and_a_closed_one_is_left_out() -> None:
    unmarked = _position()
    del unmarked["mark-price"]
    session = _session("positions", unmarked, _position(symbol="XLP", quantity="0.0"))

    assert tastytrade_reader.fetch_positions(session, ACCOUNT) == [BrokerPosition("XLU", 2.0, 40.0, None, None, None, None)]


def test_equity_closes_are_the_end_of_day_net_liquidating_values() -> None:
    session = _session(
        "balance-snapshots",
        {"snapshot-date": "2026-11-25", "time-of-day": "EOD", "net-liquidating-value": "100.0"},
        {"snapshot-date": "2026-11-27", "time-of-day": "EOD", "net-liquidating-value": "99.815"},
    )

    closes = tastytrade_reader.fetch_equity_snapshots(session, ACCOUNT, date(2026, 11, 20), date(2026, 11, 27))

    assert closes == {date(2026, 11, 25): 100.0, date(2026, 11, 27): 99.815}
    assert _calls(session) == [(f"/accounts/{ACCOUNT}/balance-snapshots", {"start-date": "2026-11-20", "end-date": "2026-11-27", "time-of-day": "EOD"})]


@pytest.mark.parametrize("broken", [{"action": "Allocate"}, {"price": None}, {"quantity": "NaN"}])
def test_an_unreadable_trade_is_a_broker_error_never_a_guessed_fill(broken: dict[str, Any]) -> None:
    with pytest.raises(BrokerError):
        _transactions({**_trade("Buy to Open"), **broken})


@pytest.mark.parametrize("broken", [{"quantity-direction": "Sideways"}, {"average-open-price": None}])
def test_an_unreadable_position_is_a_broker_error(broken: dict[str, Any]) -> None:
    with pytest.raises(BrokerError):
        tastytrade_reader.fetch_positions(_session("positions", _position(**broken)), ACCOUNT)


def test_an_uncounted_money_movement_is_reported_but_the_owners_own_money_is_not() -> None:
    session = _session(
        "transactions",
        _movement("Deposit", "558.0"),
        _movement("Withdrawal", "20.0", "Debit"),
        _movement("Transfer", "5.0"),
        _movement("Dividend", "1.2"),
        _movement("Withholding", "0.18", "Debit", day="2026-12-01"),
    )

    _, ignored = tastytrade_reader.fetch_transactions(session, ACCOUNT)

    assert ignored == ["2026-12-01 Withholding -0.18"]


def test_a_current_balance_row_never_replaces_a_close() -> None:
    # Only the EOD rows are closes: a row without a time of day is the balance now, whatever day it is dated.
    session = _session(
        "balance-snapshots",
        {"snapshot-date": "2026-11-25", "time-of-day": "EOD", "net-liquidating-value": "100.0"},
        {"snapshot-date": "2026-11-25", "net-liquidating-value": "657.8"},
    )

    assert tastytrade_reader.fetch_equity_snapshots(session, ACCOUNT, date(2026, 11, 20), date(2026, 11, 27)) == {date(2026, 11, 25): 100.0}
