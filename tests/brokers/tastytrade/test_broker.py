from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any

import pytest
from tests.fakes import (
    FakeClock,
    FakeStockHistoricalDataClient,
    FakeTastytradeSession,
    FakeTradingClient,
    et,
    make_alpaca_trade,
    make_tt_fill,
    make_tt_order,
    make_tt_position,
)

from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker
from trading_agent_framework.brokers.tastytrade.client import TastytradeHTTPError
from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError, ConfigurationError, OrderValidationError

D = Decimal
ACCOUNT = "5WT00001"
ORDERS = f"/accounts/{ACCOUNT}/orders"
AAPL, MSFT = Asset("AAPL"), Asset("MSFT")
SETTINGS = TastytradeSettings(client_secret="secret", refresh_token="refresh", account_number=ACCOUNT, is_paper=True)
ACCOUNTS = {"items": [{"account": {"account-number": ACCOUNT, "margin-or-cash": "Cash", "is-closed": False}, "authority-level": "owner"}]}


@pytest.fixture
def session() -> FakeTastytradeSession:
    session = FakeTastytradeSession()
    session.responses[("GET", "/instruments/equities/AAPL")] = {"symbol": "AAPL", "is-fractional-quantity-eligible": True}
    session.responses[("GET", "/instruments/equities/MSFT")] = {"symbol": "MSFT", "is-fractional-quantity-eligible": False}
    session.responses[("GET", "/customers/me/accounts")] = ACCOUNTS
    session.responses[("POST", ORDERS)] = accept()
    return session


def _market_data() -> AlpacaMarketData:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 100.5), "MSFT": make_alpaca_trade("MSFT", 100.5)}
    return AlpacaMarketData(data, FakeTradingClient())


@pytest.fixture
def broker(session: FakeTastytradeSession) -> TastytradeBroker:
    return TastytradeBroker("s", session, account_number=ACCOUNT, streamer_url="wss://streamer.cert.tastyworks.com", market_data=_market_data(), clock=FakeClock(et(2026, 10, 15, 10)))


@pytest.fixture
def seen(broker: TastytradeBroker) -> list[OrderEvent]:
    received: list[OrderEvent] = []
    broker.tracker.listeners.append(lambda order, event: received.append(event))
    return received


def accept(*, order_id: int = 1001, status: str = "Routed", fills: tuple[dict[str, Any], ...] = ()) -> Any:
    """A POST answer echoing the submitted order back, as Tastytrade does."""

    def answer(json: dict[str, Any], params: Any) -> dict[str, Any]:
        leg = json["legs"][0]
        placed = make_tt_order(
            order_id=order_id,
            status=status,
            external=json["external-identifier"],
            symbol=leg["symbol"],
            action=leg["action"],
            quantity=leg.get("quantity"),
            fills=fills,
            **{"order-type": json["order-type"]},
        )
        return {"order": placed, "buying-power-effect": {}}

    return answer


def _buy(quantity: str = "10", **overrides: object) -> Order:
    fields: dict[str, object] = {"strategy_name": "s", "asset": AAPL, "side": OrderSide.BUY, "quantity": D(quantity)}
    fields.update(overrides)
    return Order(**fields)  # ty: ignore[invalid-argument-type]


def _limit(price: str = "150") -> Order:
    return _buy(order_type=OrderType.LIMIT, limit_price=D(price))


# --- submit ----------------------------------------------------------------------------------


def test_submit_posts_the_order_tagged_with_its_external_identifier(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())

    [body] = session.posted(ORDERS)
    assert body["external-identifier"] == f"s:{order.identifier}" == order.client_order_id
    assert (body["order-type"], body["price"], body["legs"][0]["quantity"]) == ("Limit", "150.00", "10")
    assert order.status is OrderStatus.SUBMITTED
    assert broker.tracker.get_tracked_order(order.identifier) is order


def test_eligibility_is_looked_up_once_per_symbol(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    broker.submit_order(_buy())
    broker.submit_order(_buy())

    assert [c for c in session.calls if c[1] == "/instruments/equities/AAPL"] == [("GET", "/instruments/equities/AAPL", None, None)]


def test_an_unknown_symbol_is_an_order_validation_error(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    with pytest.raises(OrderValidationError, match="does not know the US stock ZZZZ"):
        broker.submit_order(_buy(asset=Asset("ZZZZ")))

    assert session.posted(ORDERS) == []


def test_a_rejection_sets_the_error_before_raising_and_untracks(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = TastytradeHTTPError(422, f"Tastytrade POST {ORDERS}: 422 insufficient_buying_power: Not enough buying power")
    order = _buy()

    with pytest.raises(BrokerError, match="insufficient_buying_power"):
        broker.submit_order(order)

    assert order.status is OrderStatus.ERROR
    assert "insufficient_buying_power" in (order.error_message or "")
    assert broker.tracker.get_tracked_order(order.identifier) is None


def test_an_order_filled_in_the_post_response_fills_once(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    session.responses[("POST", ORDERS)] = accept(status="Filled", fills=(make_tt_fill("f1", "1", "2.00"),))
    order = broker.submit_order(_buy("1", order_type=OrderType.LIMIT, limit_price=D(2)))

    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D(1))
    # the account streamer then delivers the same snapshot
    broker._events.apply(make_tt_order(status="Filled", external=order.client_order_id, quantity="1", fills=[make_tt_fill("f1", "1", "2.00")]))

    assert order.filled_quantity == D(1)
    assert seen.count(OrderEvent.FILLED) == 1


def test_an_uncertain_submission_found_at_tastytrade_is_adopted_not_resent(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1001, status="Live", external="s:fixed")]}
    order = _buy(client_order_id="s:fixed")

    result = broker.submit_order(order)

    assert result.status is OrderStatus.NEW
    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert len(session.posted(ORDERS)) == 1


def test_a_non_eligible_notional_order_becomes_whole_shares_at_alpacas_price(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    broker.submit_order(Order("s", MSFT, OrderSide.BUY, notional=D(1000)))

    [body] = session.posted(ORDERS)
    assert (body["order-type"], body["legs"][0]["quantity"]) == ("Market", "9")  # 1000 / 100.5


# --- cancel and modify ------------------------------------------------------------------------


def test_cancel_deletes_the_current_tastytrade_id(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("DELETE", f"{ORDERS}/1001")] = {}
    order = broker.submit_order(_limit())

    broker.cancel_order(order)

    assert session.calls[-1][:2] == ("DELETE", f"{ORDERS}/1001")


def test_modify_replaces_the_order_and_returns_the_same_order(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = make_tt_order(order_id=1002, status="Contingent", external=order.client_order_id, **{"replaces-order-id": 1001})

    result = broker.modify_order(order, limit_price=D(151))

    assert result is order
    assert order.limit_price == D(151)
    [(_, _, body, _)] = [c for c in session.calls if c[0] == "PUT"]
    assert "legs" not in body
    assert body["price"] == "151.00"
    assert broker._events.tt_id_of(order) == "1002"
    # OrderEvent.MODIFIED goes through the tracker, which notifies no listener for it (tracker.py), as for IBKR.


def test_modify_ignores_the_original_ids_cancel(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    broker._events.apply(make_tt_order(status="Live", external=order.client_order_id))
    session.responses[("PUT", f"{ORDERS}/1001")] = make_tt_order(order_id=1002, status="Contingent", external=order.client_order_id)
    broker.modify_order(order, limit_price=D(151))

    broker._events.apply(make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id))

    assert order.status is OrderStatus.NEW


def test_a_failed_modify_restores_the_prices_and_the_original_id(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = TastytradeHTTPError(422, "422 order_not_editable: no")

    with pytest.raises(BrokerError, match="order_not_editable"):
        broker.modify_order(order, limit_price=D(151))

    assert order.limit_price == D(150)
    broker._events.apply(make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id))
    assert order.status is OrderStatus.CANCELED


def test_an_uncertain_modify_found_at_tastytrade_succeeds(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = BrokerUncertainError("read timeout")
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1002, status="Contingent", external=order.client_order_id, **{"replaces-order-id": 1001})]}

    broker.modify_order(order, limit_price=D(151))

    assert broker._events.tt_id_of(order) == "1002"


def test_a_market_order_cannot_be_modified(broker: TastytradeBroker) -> None:
    order = broker.submit_order(_buy())

    with pytest.raises(OrderValidationError, match="can be modified"):
        broker.modify_order(order, limit_price=D(1))


# --- positions, account, orders -----------------------------------------------------------------


def test_close_position_sells_a_fractional_holding_of_an_eligible_stock_in_full(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(quantity="4.57")]}

    order = broker.close_position(AAPL)

    assert order is not None
    [body] = session.posted(ORDERS)
    assert (body["order-type"], body["legs"][0]["action"], body["legs"][0]["quantity"]) == ("Market", "Sell to Close", "4.57")


def test_close_position_floors_a_non_eligible_holding(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(symbol="MSFT", quantity="4.57")]}

    broker.close_position(MSFT)

    assert session.posted(ORDERS)[0]["legs"][0]["quantity"] == "4"


def test_close_position_below_one_non_eligible_share_sends_nothing(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(symbol="MSFT", quantity="0.5")]}

    assert broker.close_position(MSFT) is None
    assert session.posted(ORDERS) == []


def test_close_position_without_a_position_returns_none(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": []}

    assert broker.close_position(AAPL) is None


def test_close_all_positions_cancels_live_orders_first(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=7), make_tt_order(order_id=8, cancellable=False)]}
    session.responses[("DELETE", f"{ORDERS}/7")] = {}
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(quantity="3")]}

    closed = broker.close_all_positions()

    assert [c[1] for c in session.calls if c[0] == "DELETE"] == [f"{ORDERS}/7"]
    assert len(closed) == 1


def test_pull_positions_keeps_equities(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(), make_tt_position(symbol="AAPL  261218C00200000", instrument_type="Equity Option")]}

    [position] = broker.pull_positions()

    assert (position.asset, position.quantity, position.side) == (AAPL, D(10), PositionSide.LONG)


def test_get_account_parses_the_balances(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/balances")] = {"cash-balance": "1000", "net-liquidating-value": "2500", "equity-buying-power": "1000"}

    assert broker.get_account() == AccountBalances(cash=D(1000), portfolio_value=D(2500), buying_power=D(1000))


def test_pull_orders_parses_the_newest_and_skips_unsupported_ones(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", ORDERS)] = {"items": [make_tt_order(order_id=2), make_tt_order(order_id=1, **{"order-type": "Marketable Limit"})]}

    [order] = broker.pull_orders(limit=10)

    assert order.identifier == "abc"
    [(_, _, _, params)] = [c for c in session.calls if c[:2] == ("GET", ORDERS)]
    assert params == {"sort": "Desc"}


def test_pull_order_finds_ours_by_external_identifier(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", ORDERS)] = {"items": [make_tt_order(order_id=2, external="s:other"), make_tt_order(order_id=1)]}

    order = broker.pull_order("abc")

    assert order is not None
    assert order.raw["id"] == 1
    assert broker.pull_order("missing") is None


def test_sync_open_orders_adopts_only_this_strategys_working_orders(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = {
        "items": [
            make_tt_order(order_id=1, external="s:mine", fills=[make_tt_fill("f1", "4", "150")]),
            make_tt_order(order_id=2, external="other:theirs"),
            make_tt_order(order_id=3, external="s:done", status="Filled"),
            make_tt_order(order_id=4, external="s:old", **{"replacing-order-id": 5}),
        ]
    }

    [adopted] = broker.sync_open_orders()

    assert (adopted.identifier, adopted.filled_quantity) == ("mine", D(4))
    assert broker._events.tt_id_of(adopted) == "1"
    broker._events.apply(make_tt_order(order_id=1, external="s:mine", fills=[make_tt_fill("f1", "4", "150")]))
    assert adopted.filled_quantity == D(4)  # its existing fill is not counted twice
    assert broker.sync_open_orders() == []  # already tracked


def test_reconcile_re_reads_active_orders(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")])

    broker.reconcile()

    assert order.status is OrderStatus.FILL


# --- lifecycle --------------------------------------------------------------------------------


def test_from_settings_checks_the_account(session: FakeTastytradeSession) -> None:
    broker = TastytradeBroker.from_settings("s", SETTINGS, data=AlpacaCredentials(api_key="k", api_secret="s"), session=session, market_data=_market_data())

    assert broker.account_number == ACCOUNT
    assert broker.is_paper
    assert not session.closed


def test_from_settings_closes_the_session_when_the_check_fails(session: FakeTastytradeSession) -> None:
    session.responses[("GET", "/customers/me/accounts")] = {"items": []}

    with pytest.raises(ConfigurationError, match="TASTYTRADE_ACCOUNT_NUMBER"):
        TastytradeBroker.from_settings("s", SETTINGS, data=AlpacaCredentials(api_key="k", api_secret="s"), session=session, market_data=_market_data())

    assert session.closed


def test_stop_stream_closes_the_session(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    broker.stop_stream()

    assert session.closed


def test_market_data_comes_from_alpaca(broker: TastytradeBroker) -> None:
    assert broker.get_last_price(AAPL) == D("100.5")


# --- review rulings: uncertain outcomes, refreshes and malformed snapshots --------------------------------


def test_a_bare_client_order_id_is_prefixed_with_the_strategy(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_buy(client_order_id="bare"))

    [body] = session.posted(ORDERS)
    assert body["external-identifier"] == order.client_order_id == "s:bare"


def test_a_2xx_without_an_order_id_found_at_tastytrade_is_adopted_not_failed(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = {"order": {"status": "Received"}}
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1001, status="Live", external="s:fixed")]}
    order = _buy(client_order_id="s:fixed")

    result = broker.submit_order(order)

    assert result.status is OrderStatus.NEW
    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert broker._events.tt_id_of(order) == "1001"
    assert len(session.posted(ORDERS)) == 1


def test_a_2xx_with_no_order_at_all_found_at_tastytrade_is_adopted_not_failed(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = {}
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1001, status="Live", external="s:fixed")]}
    order = _buy(client_order_id="s:fixed")

    broker.submit_order(order)

    assert broker._events.tt_id_of(order) == "1001"
    assert order.status is OrderStatus.NEW


def test_a_malformed_snapshot_in_the_post_response_does_not_fail_the_submission(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    bad_fill = {"quantity": "1", "fill-price": "2.00"}  # no fill-id: the events layer refuses it

    def answer(json: dict[str, Any], params: Any) -> dict[str, Any]:
        return {"order": make_tt_order(order_id=1001, status="Filled", external=json["external-identifier"], fills=[bad_fill])}

    session.responses[("POST", ORDERS)] = answer
    order = broker.submit_order(_buy("1"))

    assert order.status is OrderStatus.SUBMITTED
    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert broker._events.tt_id_of(order) == "1001"


def test_a_failed_modify_re_reads_the_original_so_a_fill_during_the_replace_is_not_lost(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = TastytradeHTTPError(422, "422 order_not_editable: no")
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")])

    with pytest.raises(BrokerError, match="order_not_editable"):
        broker.modify_order(order, limit_price=D(151))

    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D(10))
    assert order.limit_price == D(150)


def test_a_modify_whose_2xx_has_no_order_id_is_looked_up_like_an_uncertain_one(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = {}
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1002, status="Contingent", external=order.client_order_id, **{"replaces-order-id": 1001})]}

    broker.modify_order(order, limit_price=D(151))

    assert broker._events.tt_id_of(order) == "1002"
    assert order.limit_price == D(151)


def test_reconcile_treats_an_expired_order_as_ended(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    order = broker.submit_order(_limit())
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Expired", external=order.client_order_id)

    broker.reconcile()

    assert not order.is_active()
    assert seen == [OrderEvent.CANCELED]


def test_reconcile_survives_a_malformed_snapshot_and_still_reads_the_other_orders(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    first = broker.submit_order(_limit())
    session.responses[("POST", ORDERS)] = accept(order_id=1002)
    second = broker.submit_order(_limit())
    bad_fill = {"quantity": "1", "fill-price": "2.00"}
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Filled", external=first.client_order_id, fills=[bad_fill])
    session.responses[("GET", f"{ORDERS}/1002")] = make_tt_order(order_id=1002, status="Cancelled", external=second.client_order_id)

    broker.reconcile()

    assert first.is_active()
    assert second.status is OrderStatus.CANCELED


def test_reconcile_still_re_reads_active_orders_when_adoption_fails(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("GET", f"{ORDERS}/live")] = BrokerError("boom")
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id)

    broker.reconcile()

    assert order.status is OrderStatus.CANCELED


def test_sync_open_orders_skips_a_malformed_order_and_adopts_the_rest(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    bad_fill = {"quantity": "1", "fill-price": "2.00"}
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1, external="s:bad", fills=[bad_fill]), make_tt_order(order_id=2, external="s:good")]}

    [adopted] = broker.sync_open_orders()

    assert adopted.identifier == "good"
    assert broker.tracker.get_tracked_order("bad") is None


# --- review 2: an unknown outcome keeps the order; preset client ids are not re-adopted ------------------------


def _lookups_fail(session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = TastytradeHTTPError(503, "down")
    session.responses[("GET", ORDERS)] = TastytradeHTTPError(503, "down")


def _lookups_empty(session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", ORDERS)] = {"items": []}


def test_an_uncertain_submission_the_stream_already_confirmed_is_kept(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    def post(json: dict[str, Any], params: Any) -> Any:
        broker._events.apply(make_tt_order(order_id=1001, status="Live", external=json["external-identifier"]))
        return BrokerUncertainError("read timeout")

    session.responses[("POST", ORDERS)] = post
    _lookups_fail(session)
    order = _limit()

    result = broker.submit_order(order)

    assert result is order
    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert broker._events.tt_id_of(order) == "1001"
    assert order.error_message is None
    broker._events.apply(make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")]))
    assert order.status is OrderStatus.FILL
    assert seen.count(OrderEvent.FILLED) == 1
    assert len(session.posted(ORDERS)) == 1


def test_an_uncertain_submission_whose_lookups_all_fail_keeps_the_order_tracked(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    _lookups_fail(session)
    order = _limit()

    with pytest.raises(BrokerUncertainError, match="check the account"):
        broker.submit_order(order)

    assert order.status is OrderStatus.SUBMITTED
    assert order.error_message is None
    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert len(session.posted(ORDERS)) == 1
    # the stream's external-identifier fallback can still adopt it
    broker._events.apply(make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")]))
    assert order.status is OrderStatus.FILL
    assert seen.count(OrderEvent.FILLED) == 1


def test_reconcile_resolves_an_unknown_submission_that_exists(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    _lookups_fail(session)
    order = _limit()
    with pytest.raises(BrokerUncertainError):
        broker.submit_order(order)
    snapshot = make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")])
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", ORDERS)] = {"items": [snapshot]}

    broker.reconcile()

    assert broker._events.tt_id_of(order) == "1001"
    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D(10))
    assert len(session.posted(ORDERS)) == 1


def test_reconcile_leaves_an_unknown_submission_while_lookups_still_fail(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    _lookups_fail(session)
    order = _limit()
    with pytest.raises(BrokerUncertainError):
        broker.submit_order(order)

    broker.reconcile()

    assert order.status is OrderStatus.SUBMITTED
    assert order.error_message is None
    assert broker.tracker.get_tracked_order(order.identifier) is order


def test_a_2xx_without_an_id_whose_lookups_fail_is_also_kept(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = {}
    _lookups_fail(session)
    order = _limit()

    with pytest.raises(BrokerUncertainError):
        broker.submit_order(order)

    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert order.error_message is None


def test_a_live_lookup_failure_alone_is_not_proof_of_absence(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    session.responses[("GET", f"{ORDERS}/live")] = TastytradeHTTPError(503, "down")
    session.responses[("GET", ORDERS)] = {"items": []}
    order = _limit()

    with pytest.raises(BrokerUncertainError):
        broker.submit_order(order)

    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert order.error_message is None


def test_an_uncertain_modify_with_failing_lookups_leaves_the_original_superseded(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    broker._events.apply(make_tt_order(status="Live", external=order.client_order_id))
    session.responses[("PUT", f"{ORDERS}/1001")] = BrokerUncertainError("read timeout")
    _lookups_fail(session)

    with pytest.raises(BrokerUncertainError):
        broker.modify_order(order, limit_price=D(151))

    # the replace may have happened: the original's Cancelled must not cancel the logical order
    broker._events.apply(make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id))
    assert order.is_active()
    # and the replacement, if it exists, is adopted through the external identifier
    broker._events.apply(make_tt_order(order_id=1002, status="Live", external=order.client_order_id, **{"replaces-order-id": 1001}))
    assert broker._events.tt_id_of(order) == "1002"


def test_reconcile_resolves_an_unknown_modify_whose_replacement_exists(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = BrokerUncertainError("read timeout")
    _lookups_fail(session)
    with pytest.raises(BrokerUncertainError):
        broker.modify_order(order, limit_price=D(151))
    replacement = make_tt_order(order_id=1002, status="Live", external=order.client_order_id, **{"replaces-order-id": 1001})
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [replacement]}
    session.responses[("GET", f"{ORDERS}/1002")] = replacement

    broker.reconcile()

    assert broker._events.tt_id_of(order) == "1002"
    assert order.limit_price == D(151)


def test_an_uncertain_modify_the_stream_already_confirmed_succeeds(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())

    def put(json: dict[str, Any], params: Any) -> Any:
        broker._events.apply(make_tt_order(order_id=1002, status="Live", external=order.client_order_id, **{"replaces-order-id": 1001}))
        return BrokerUncertainError("read timeout")

    session.responses[("PUT", f"{ORDERS}/1001")] = put
    _lookups_fail(session)

    broker.modify_order(order, limit_price=D(151))

    assert broker._events.tt_id_of(order) == "1002"
    assert order.limit_price == D(151)


def test_reconcile_does_not_re_adopt_an_order_with_a_preset_client_order_id(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    order = broker.submit_order(_buy(order_type=OrderType.LIMIT, limit_price=D(150), client_order_id="s:fixed"))
    live = make_tt_order(order_id=1001, status="Live", external="s:fixed")
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [live]}
    session.responses[("GET", f"{ORDERS}/1001")] = live

    broker.reconcile()

    assert broker.tracker.get_all_tracked_orders() == [order]
    broker._events.apply(make_tt_order(order_id=1001, status="Filled", external="s:fixed", fills=[make_tt_fill("f1", "10", "150")]))
    assert order.status is OrderStatus.FILL
    assert seen.count(OrderEvent.FILLED) == 1


def test_sync_open_orders_skips_an_id_the_events_layer_already_knows(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1, external="s:mine")]}
    [adopted] = broker.sync_open_orders()
    broker.tracker.untrack(adopted)  # even if the tracker forgot it, the id is registered

    assert broker.sync_open_orders() == []


def test_close_all_positions_continues_past_a_failing_position_and_raises_at_the_end(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(symbol="MSFT", quantity="3"), make_tt_position(symbol="AAPL", quantity="3")]}
    session.responses[("GET", "/instruments/equities/MSFT")] = TastytradeHTTPError(500, "boom")

    with pytest.raises(BrokerError, match="MSFT"):
        broker.close_all_positions()

    [body] = session.posted(ORDERS)  # AAPL was still closed
    assert body["legs"][0]["symbol"] == "AAPL"
    assert len([c for c in session.calls if c[1] == f"/accounts/{ACCOUNT}/positions"]) == 1  # pulled once


def test_cancel_without_a_tastytrade_id_is_a_broker_error(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    with pytest.raises(BrokerError, match="no Tastytrade order id"):
        broker.cancel_order(_limit())

    assert not [c for c in session.calls if c[0] == "DELETE"]


def test_reconcile_does_not_fail_an_unknown_submission_the_stream_adopted_meanwhile(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    _lookups_fail(session)
    order = _limit()
    with pytest.raises(BrokerUncertainError):
        broker.submit_order(order)
    live = make_tt_order(order_id=1001, status="Live", external=order.client_order_id)
    broker._events.apply(live)
    _lookups_empty(session)  # e.g. a lagging listing
    session.responses[("GET", f"{ORDERS}/1001")] = live

    broker.reconcile()

    assert order.status is OrderStatus.NEW
    assert order.error_message is None
    assert broker.tracker.get_tracked_order(order.identifier) is order


# --- review 3: unknown writes are decided later, never inside the failing call -------------------------------


def _later(broker: TastytradeBroker, seconds: float = 31) -> None:
    """Move the broker's FakeClock on, past the grace period before a miss counts."""
    broker.clock.advance(seconds)  # ty: ignore[unresolved-attribute]


def _unknown_submit(broker: TastytradeBroker, session: FakeTastytradeSession, *, lookups: str = "fail", post: object = None) -> Order:
    session.responses[("POST", ORDERS)] = post if post is not None else BrokerUncertainError("read timeout")
    (_lookups_fail if lookups == "fail" else _lookups_empty)(session)
    order = _limit()
    with pytest.raises(BrokerUncertainError):
        broker.submit_order(order)
    return order


def _unknown_modify(broker: TastytradeBroker, session: FakeTastytradeSession, *, lookups: str = "fail") -> Order:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = BrokerUncertainError("read timeout")
    (_lookups_fail if lookups == "fail" else _lookups_empty)(session)
    with pytest.raises(BrokerUncertainError):
        broker.modify_order(order, limit_price=D(151))
    return order


def test_a_clean_miss_right_after_an_uncertain_submit_keeps_the_order(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session, lookups="empty")

    assert order.status is OrderStatus.SUBMITTED
    assert order.error_message is None
    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert len(session.posted(ORDERS)) == 1


def test_a_2xx_without_an_id_and_a_clean_miss_keeps_the_order(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session, lookups="empty", post={})

    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert order.error_message is None


def test_the_first_conclusive_miss_keeps_the_order_and_the_second_fails_it_with_an_error_event(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    order = _unknown_submit(broker, session, lookups="empty")

    _later(broker)
    broker.reconcile()
    assert order.is_active()
    assert seen == []
    _later(broker)
    broker.reconcile()

    assert order.status is OrderStatus.ERROR
    assert order.error_message
    assert seen == [OrderEvent.ERROR]
    assert broker.tracker.get_tracked_order(order.identifier) is None
    broker.reconcile()
    assert seen == [OrderEvent.ERROR]  # resolved once
    assert len(session.posted(ORDERS)) == 1


def test_inconclusive_attempts_never_advance_the_miss_counter(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session, lookups="empty")
    _later(broker)
    broker.reconcile()  # one conclusive miss
    _lookups_fail(session)
    broker.reconcile()
    broker.reconcile()
    broker.reconcile()
    assert order.is_active()

    _lookups_empty(session)
    _later(broker)
    broker.reconcile()  # the second conclusive miss, however many failures came between

    assert order.status is OrderStatus.ERROR


def test_a_late_appearing_order_after_a_timeout_ends_adopted_not_failed(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session, lookups="empty")
    broker.reconcile()  # empty: a lagging listing
    live = make_tt_order(order_id=1001, status="Live", external=order.client_order_id)
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [live]}
    session.responses[("GET", f"{ORDERS}/1001")] = live

    broker.reconcile()
    broker.reconcile()
    broker.reconcile()

    assert broker._events.tt_id_of(order) == "1001"
    assert order.status is OrderStatus.NEW
    assert order.error_message is None
    assert broker.tracker.get_tracked_order(order.identifier) is order


def test_the_stream_adopting_during_the_submit_lookup_wins(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    order = _limit()

    def live(json: Any, params: Any) -> dict[str, Any]:
        broker._events.apply(make_tt_order(order_id=1001, status="Live", external=order.client_order_id))
        return {"items": []}  # the listing lags

    session.responses[("GET", f"{ORDERS}/live")] = live
    session.responses[("GET", ORDERS)] = {"items": []}

    broker.submit_order(order)  # confirmed by the stream: a normal success

    assert broker._events.tt_id_of(order) == "1001"
    assert order.error_message is None
    assert broker.tracker.get_tracked_order(order.identifier) is order
    broker._events.apply(make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")]))
    assert seen.count(OrderEvent.FILLED) == 1
    assert OrderEvent.ERROR not in seen


def test_the_stream_adopting_between_a_resolve_lookup_and_its_decision_wins(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    order = _unknown_submit(broker, session, lookups="empty")
    _later(broker)
    broker.reconcile()  # first conclusive miss

    lookups: list[int] = []

    def live(json: Any, params: Any) -> dict[str, Any]:
        lookups.append(1)
        if len(lookups) == 2:  # reconcile's own adoption sweep is the first lookup, the resolver's is the second
            broker._events.apply(make_tt_order(order_id=1001, status="Live", external=order.client_order_id))
        return {"items": []}

    session.responses[("GET", f"{ORDERS}/live")] = live
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Live", external=order.client_order_id)

    _later(broker)
    broker.reconcile()  # would be the second miss, but the stream got there first

    assert order.is_active()
    assert OrderEvent.ERROR not in seen
    assert broker._events.tt_id_of(order) == "1001"


def test_the_stream_adopting_the_replacement_during_the_modify_lookup_wins(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = BrokerUncertainError("read timeout")

    def live(json: Any, params: Any) -> dict[str, Any]:
        broker._events.apply(make_tt_order(order_id=1002, status="Live", external=order.client_order_id, **{"replaces-order-id": 1001}))
        return {"items": []}

    session.responses[("GET", f"{ORDERS}/live")] = live
    session.responses[("GET", ORDERS)] = {"items": []}
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id)

    broker.modify_order(order, limit_price=D(151))

    assert broker._events.tt_id_of(order) == "1002"
    assert order.limit_price == D(151)
    assert order.is_active()


def test_a_clean_miss_right_after_an_uncertain_modify_keeps_the_replace_open(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session, lookups="empty")

    broker._events.apply(make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id))  # the replace happened
    assert order.is_active()
    assert order.limit_price == D(151)  # not restored yet
    broker._events.apply(make_tt_order(order_id=1002, status="Live", external=order.client_order_id, **{"replaces-order-id": 1001}))
    assert broker._events.tt_id_of(order) == "1002"


def test_the_second_conclusive_miss_restores_an_unknown_modify(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session, lookups="empty")
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id)

    _later(broker)
    broker.reconcile()
    assert order.limit_price == D(151)
    assert order.is_active()
    _later(broker)
    broker.reconcile()

    assert order.limit_price == D(150)
    assert order.status is OrderStatus.CANCELED  # the original's real state is no longer hidden


def test_a_restored_modify_leaves_the_original_cancellable(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session, lookups="empty")
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Live", external=order.client_order_id)
    session.responses[("DELETE", f"{ORDERS}/1001")] = {}
    _later(broker)
    broker.reconcile()
    _later(broker)
    broker.reconcile()  # restored: the original is the live order again

    broker.cancel_order(order)

    assert session.calls[-1][:2] == ("DELETE", f"{ORDERS}/1001")


def test_modify_on_an_unresolved_modify_raises_without_sending_a_put(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session)
    puts = len([c for c in session.calls if c[0] == "PUT"])

    with pytest.raises(BrokerUncertainError, match="unresolved"):
        broker.modify_order(order, limit_price=D(152))

    assert len([c for c in session.calls if c[0] == "PUT"]) == puts
    assert order.limit_price == D(151)


def test_modify_resolves_a_pending_modify_first_and_then_goes_to_the_new_id(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session)
    replacement = make_tt_order(order_id=1002, status="Live", external=order.client_order_id, **{"replaces-order-id": 1001})
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [replacement]}
    session.responses[("PUT", f"{ORDERS}/1002")] = make_tt_order(order_id=1003, status="Contingent", external=order.client_order_id, **{"replaces-order-id": 1002})

    broker.modify_order(order, limit_price=D(152))

    assert broker._events.tt_id_of(order) == "1003"
    assert order.limit_price == D(152)


def test_cancel_on_an_unresolved_modify_raises_without_sending_a_delete_then_goes_to_the_new_id(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session)

    with pytest.raises(BrokerUncertainError, match="unresolved"):
        broker.cancel_order(order)
    assert not [c for c in session.calls if c[0] == "DELETE"]

    replacement = make_tt_order(order_id=1002, status="Live", external=order.client_order_id, **{"replaces-order-id": 1001})
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [replacement]}
    session.responses[("DELETE", f"{ORDERS}/1002")] = {}
    broker.cancel_order(order)

    assert session.calls[-1][:2] == ("DELETE", f"{ORDERS}/1002")


def test_cancel_on_an_unresolved_submit_is_uncertain_not_a_missing_id(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session)

    with pytest.raises(BrokerUncertainError, match="unresolved"):
        broker.cancel_order(order)

    assert not [c for c in session.calls if c[0] == "DELETE"]


def test_cancel_resolves_an_unresolved_submit_first(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session)
    live = make_tt_order(order_id=1001, status="Live", external=order.client_order_id)
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [live]}
    session.responses[("DELETE", f"{ORDERS}/1001")] = {}

    broker.cancel_order(order)

    assert session.calls[-1][:2] == ("DELETE", f"{ORDERS}/1001")


@pytest.mark.parametrize("pull", [lambda b: b.pull_orders(), lambda b: b.pull_order("x")])
def test_pull_calls_resolve_unknown_writes(broker: TastytradeBroker, session: FakeTastytradeSession, pull: Any) -> None:
    order = _unknown_submit(broker, session)
    live = make_tt_order(order_id=1001, status="Live", external=order.client_order_id)
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [live]}
    session.responses[("GET", ORDERS)] = {"items": []}

    pull(broker)

    assert broker._events.tt_id_of(order) == "1001"


# --- review 4: misses are spaced in time; resolution is atomic against the stream ------------------------------


def test_back_to_back_cancels_after_a_timeout_do_not_rule_the_order_out(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    order = _unknown_submit(broker, session, lookups="empty")

    for _ in range(3):
        with pytest.raises(BrokerUncertainError):
            broker.cancel_order(order)

    assert order.is_active()
    assert order.error_message is None
    assert seen == []
    # the order existed after all (listing lag): the stream reports it and it fills
    broker._events.apply(make_tt_order(order_id=1001, status="Live", external=order.client_order_id))
    broker._events.apply(make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")]))
    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D(10))


def test_two_spaced_misses_rule_a_submit_out_through_cancel_calls(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    order = _unknown_submit(broker, session, lookups="empty")

    _later(broker)
    with pytest.raises(BrokerUncertainError):
        broker.cancel_order(order)  # the first counted miss
    _later(broker)
    with pytest.raises(BrokerError):
        broker.cancel_order(order)  # the second one fails the order; there is nothing left to cancel

    assert order.status is OrderStatus.ERROR
    assert seen == [OrderEvent.ERROR]
    assert not [c for c in session.calls if c[0] == "DELETE"]


def test_a_miss_counts_only_after_the_grace_period_since_the_record_and_since_the_previous_miss(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session, lookups="empty")

    _later(broker, 10)
    broker.reconcile()  # 10 s after the record: ignored
    _later(broker, 10)
    broker.reconcile()  # 20 s: still ignored
    _later(broker, 11)
    broker.reconcile()  # 31 s: the first counted miss
    _later(broker, 9)
    broker.reconcile()  # 9 s after it: ignored
    assert order.is_active()
    _later(broker, 21)
    broker.reconcile()  # 30 s after the first counted miss: the second

    assert order.status is OrderStatus.ERROR


def test_back_to_back_cancels_after_an_uncertain_modify_do_not_restore_it(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session, lookups="empty")
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id)

    for _ in range(3):
        with pytest.raises(BrokerUncertainError):
            broker.cancel_order(order)

    assert order.is_active()
    assert order.limit_price == D(151)
    # the replace happened after all: the replacement fills
    broker._events.apply(make_tt_order(order_id=1002, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "151")], **{"replaces-order-id": 1001}))
    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D(10))


def test_two_spaced_misses_restore_a_modify_through_modify_calls(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session, lookups="empty")
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Live", external=order.client_order_id)
    session.responses[("PUT", f"{ORDERS}/1001")] = make_tt_order(order_id=1002, status="Contingent", external=order.client_order_id, **{"replaces-order-id": 1001})

    _later(broker)
    with pytest.raises(BrokerUncertainError):
        broker.modify_order(order, limit_price=D(152))
    _later(broker)
    broker.modify_order(order, limit_price=D(152))  # the second miss restored the original; this modify goes through

    assert broker._events.tt_id_of(order) == "1002"
    assert order.limit_price == D(152)


def test_the_resolver_fails_a_submit_atomically_against_the_stream(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session, lookups="empty")
    _later(broker)
    broker.reconcile()
    _later(broker)
    blocked: list[bool] = []
    threads: list[threading.Thread] = []

    def stream_adopts() -> None:
        broker._events.apply(make_tt_order(order_id=1001, status="Live", external=order.client_order_id))

    def during_the_decision(o: Order, event: OrderEvent) -> None:
        if event is OrderEvent.ERROR:
            thread = threading.Thread(target=stream_adopts)
            thread.start()
            thread.join(0.3)
            blocked.append(thread.is_alive())  # the stream thread must wait for the decision to finish
            threads.append(thread)

    broker.tracker.listeners.append(during_the_decision)

    broker.reconcile()
    threads[0].join(5)

    assert blocked == [True]
    assert order.status is OrderStatus.ERROR  # the decision was whole: the late adoption changes nothing
    assert broker.tracker.get_tracked_order(order.identifier) is None


def test_the_resolver_restores_a_modify_atomically_against_the_stream(broker: TastytradeBroker, session: FakeTastytradeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    order = _unknown_modify(broker, session, lookups="empty")
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Live", external=order.client_order_id)
    _later(broker)
    broker.reconcile()
    _later(broker)
    blocked: list[bool] = []
    threads: list[threading.Thread] = []
    real_abort = broker._events.abort_replace

    def stream_adopts() -> None:
        broker._events.apply(make_tt_order(order_id=1002, status="Live", external=order.client_order_id, **{"replaces-order-id": 1001}))

    def abort_while_the_stream_tries(tt_id: str) -> None:
        thread = threading.Thread(target=stream_adopts)
        thread.start()
        thread.join(0.3)
        blocked.append(thread.is_alive())
        threads.append(thread)
        real_abort(tt_id)

    monkeypatch.setattr(broker._events, "abort_replace", abort_while_the_stream_tries)

    broker.reconcile()
    threads[0].join(5)

    assert blocked == [True]
    assert order.limit_price == D(150)


def test_an_unknown_modify_whose_original_ended_is_cleaned_up(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_modify(broker, session, lookups="empty")
    # the original filled completely while superseded: its fills count, so the replace cannot have happened
    broker._events.apply(make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")]))
    assert order.status is OrderStatus.FILL
    assert order.limit_price == D(151)

    broker.reconcile()

    assert order.limit_price == D(150)
    assert "1001" not in broker._events._superseded
    assert not broker._unknown


# --- final review ------------------------------------------------------------------------------------


def test_after_a_restart_during_a_replace_the_originals_cancel_does_not_cancel_the_adopted_replacement(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    original = make_tt_order(order_id=1001, status="Replace Requested", external="s:abc", **{"replacing-order-id": 2002})
    replacement = make_tt_order(order_id=2002, status="Contingent", external="s:abc", **{"replaces-order-id": 1001, "price": "151.0"})
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [original, replacement]}

    [adopted] = broker.sync_open_orders()
    broker._events.apply(make_tt_order(order_id=1001, status="Cancelled", external="s:abc", **{"replacing-order-id": 2002}))

    assert adopted.is_active()
    assert OrderEvent.CANCELED not in seen
    assert broker._events.tt_id_of(adopted) == "2002"
    broker._events.apply(make_tt_order(order_id=2002, status="Filled", external="s:abc", fills=[make_tt_fill("g1", "10", "151")], **{"replaces-order-id": 1001}))
    assert (adopted.status, adopted.filled_quantity) == (OrderStatus.FILL, D(10))


def test_the_original_listed_after_its_replacement_is_superseded_too(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    original = make_tt_order(order_id=1001, status="Replace Requested", external="s:abc", **{"replacing-order-id": 2002})
    replacement = make_tt_order(order_id=2002, status="Contingent", external="s:abc", **{"replaces-order-id": 1001})
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [replacement, original]}

    [adopted] = broker.sync_open_orders()
    broker._events.apply(make_tt_order(order_id=1001, status="Cancelled", external="s:abc"))

    assert adopted.is_active()


def test_a_get_failure_while_conforming_a_submit_is_a_plain_broker_error() -> None:
    import httpx

    from trading_agent_framework.brokers.tastytrade.client import TastytradeSession

    posts: list[httpx.Request] = []

    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 900})
        if request.method == "POST":
            posts.append(request)
        raise httpx.ConnectTimeout("down")

    real = TastytradeSession(SETTINGS, transport=httpx.MockTransport(api), sleep=lambda _: None)
    broker = TastytradeBroker("s", real, account_number=ACCOUNT, streamer_url="wss://x", market_data=_market_data(), clock=FakeClock(et(2026, 10, 15, 10)))

    with pytest.raises(BrokerError) as excinfo:
        broker.submit_order(_buy())

    assert not isinstance(excinfo.value, BrokerUncertainError)
    assert posts == []


def test_unknown_writes_are_looked_up_in_the_window_of_their_own_day(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = _unknown_submit(broker, session, lookups="empty")  # recorded on 2026-10-15
    _later(broker, 15 * 3600)  # now 2026-10-16
    session.calls.clear()

    broker.reconcile()

    [(_, _, _, params)] = [c for c in session.calls if c[:2] == ("GET", ORDERS)]
    assert params["start-date"] == "2026-10-15"
    assert order.is_active()


def test_a_short_position_is_not_closed_with_a_buy(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(direction="Short", quantity="5")]}

    with pytest.raises(BrokerError, match="short"):
        broker.close_position(AAPL)

    assert session.posted(ORDERS) == []


def test_close_all_positions_reports_a_short_and_still_closes_the_rest(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(symbol="MSFT", direction="Short", quantity="5"), make_tt_position(quantity="3")]}

    with pytest.raises(BrokerError, match="MSFT"):
        broker.close_all_positions()

    [body] = session.posted(ORDERS)
    assert body["legs"][0]["symbol"] == "AAPL"


def test_an_unknown_modify_reported_by_the_stream_without_replaces_order_id_is_adopted_not_rolled_back(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    order = _unknown_modify(broker, session, lookups="empty")  # lookups match on replaces-order-id and find nothing
    broker._events.apply(make_tt_order(order_id=2002, status="Live", external=order.client_order_id))  # no replaces-order-id
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id)
    session.responses[("GET", f"{ORDERS}/2002")] = make_tt_order(order_id=2002, status="Live", external=order.client_order_id)

    for _ in range(3):
        _later(broker)
        broker.reconcile()

    assert broker._events.tt_id_of(order) == "2002"
    assert order.is_active()
    assert order.limit_price == D(151)
    assert OrderEvent.CANCELED not in seen


# --- real payload shapes recorded by the sandbox probe (2026-10-16) -------------------------------------------

# GET /instruments/equities/AAPL: no `is-fractional-quantity-eligible` key at all
PROBE_AAPL_INSTRUMENT = {
    "id": 815,
    "active": True,
    "cusip": "037833100",
    "description": "APPLE INC",
    "instrument-type": "Equity",
    "is-closing-only": False,
    "is-etf": False,
    "is-fraud-risk": False,
    "is-illiquid": False,
    "is-index": False,
    "is-options-closing-only": False,
    "listed-market": "XNAS",
    "overnight-trading-permitted": True,
    "short-description": "AAPL",
    "streamer-symbol": "AAPL",
    "symbol": "AAPL",
    "tick-sizes": [{"threshold": "1.0", "value": "0.0001"}, {"value": "0.01"}],
}
PROBE_PREFLIGHT_422 = {
    "data": {"id": 1940879},
    "error": {
        "code": "preflight_check_failure",
        "message": "One or more preflight checks failed",
        "errors": [{"code": "fractional_trading_invalid_symbol", "message": "Fractional trading of AAPL is currently not allowed"}],
    },
}


def _probe_order(order_id: int, external: str, **overrides: Any) -> dict[str, Any]:
    """A resting $5 limit buy as Tastytrade's sandbox returned it: ints for whole-share size and quantity."""
    order: dict[str, Any] = {
        "id": order_id,
        "account-number": ACCOUNT,
        "time-in-force": "Day",
        "order-type": "Limit",
        "size": 1,
        "price": "5.0",
        "price-effect": "Debit",
        "status": "Received",
        "underlying-symbol": "AAPL",
        "underlying-instrument-type": "Equity",
        "external-identifier": external,
        "cancellable": True,
        "editable": True,
        "edited": False,
        "legs": [{"instrument-type": "Equity", "symbol": "AAPL", "action": "Buy to Open", "quantity": 1, "remaining-quantity": 1, "fills": []}],
    }
    order.update(overrides)
    return order


def _real_session_broker(handler: Any) -> tuple[TastytradeBroker, list[Any]]:
    import httpx

    from trading_agent_framework.brokers.tastytrade.client import TastytradeSession

    requests: list[httpx.Request] = []

    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 900})
        requests.append(request)
        return handler(request)

    real = TastytradeSession(SETTINGS, transport=httpx.MockTransport(api), sleep=lambda _: None)
    broker = TastytradeBroker("s", real, account_number=ACCOUNT, streamer_url="wss://x", market_data=_market_data(), clock=FakeClock(et(2026, 10, 15, 10)))
    return broker, requests


def test_a_resting_limit_buy_in_the_real_post_shape_is_submitted(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", "/instruments/equities/AAPL")] = PROBE_AAPL_INSTRUMENT
    session.responses[("POST", ORDERS)] = lambda json, params: {"order": _probe_order(1940881, json["external-identifier"]), "buying-power-effect": {}}

    order = broker.submit_order(_buy("1", order_type=OrderType.LIMIT, limit_price=D(5)))

    assert order.status is OrderStatus.SUBMITTED  # "Received" outside market hours
    assert broker._events.tt_id_of(order) == "1940881"
    assert (order.quantity, order.limit_price) == (D(1), D(5))


def test_the_real_put_shape_is_the_new_order_directly_and_the_replace_chain_works(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    session.responses[("GET", "/instruments/equities/AAPL")] = PROBE_AAPL_INSTRUMENT
    session.responses[("POST", ORDERS)] = lambda json, params: {"order": _probe_order(1940881, json["external-identifier"]), "buying-power-effect": {}}
    order = broker.submit_order(_buy("1", order_type=OrderType.LIMIT, limit_price=D(5)))
    external = order.client_order_id
    # PUT answers with data = the NEW order, not wrapped in `order`
    session.responses[("PUT", f"{ORDERS}/1940881")] = _probe_order(1940882, external, price="6.0", **{"replaces-order-id": 1940881})

    result = broker.modify_order(order, limit_price=D(6))

    assert result is order
    assert broker._events.tt_id_of(order) == "1940882"
    assert order.limit_price == D(6)
    # the original then shows Cancelled with replacing-order-id: it must not cancel the logical order
    broker._events.apply(_probe_order(1940881, external, status="Cancelled", cancellable=False, edited=True, **{"replacing-order-id": 1940882}))
    assert order.is_active()
    assert OrderEvent.CANCELED not in seen
    # the replacement's own statuses count
    broker._events.apply(_probe_order(1940882, external, status="Live", price="6.0", **{"replaces-order-id": 1940881}))
    assert order.status is OrderStatus.NEW
    session.responses[("DELETE", f"{ORDERS}/1940882")] = {}
    broker.cancel_order(order)
    assert session.calls[-1][:2] == ("DELETE", f"{ORDERS}/1940882")
    broker._events.apply(_probe_order(1940882, external, status="Cancelled", **{"replaces-order-id": 1940881}))
    assert order.status is OrderStatus.CANCELED


def test_a_restart_during_the_real_replace_chain_adopts_only_the_replacement(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    original = _probe_order(1940881, "s:abc", status="Replace Requested", **{"replacing-order-id": 1940882})
    replacement = _probe_order(1940882, "s:abc", price="6.0", **{"replaces-order-id": 1940881})
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [original, replacement]}

    [adopted] = broker.sync_open_orders()
    broker._events.apply(_probe_order(1940881, "s:abc", status="Cancelled", cancellable=False, **{"replacing-order-id": 1940882}))

    assert broker._events.tt_id_of(adopted) == "1940882"
    assert adopted.limit_price == D(6)
    assert adopted.is_active()
    assert OrderEvent.CANCELED not in seen


def test_the_real_aapl_instrument_has_no_fractional_flag_so_a_fractional_quantity_is_floored(broker: TastytradeBroker, session: FakeTastytradeSession, caplog: pytest.LogCaptureFixture) -> None:
    session.responses[("GET", "/instruments/equities/AAPL")] = PROBE_AAPL_INSTRUMENT
    session.responses[("POST", ORDERS)] = lambda json, params: {"order": _probe_order(1940881, json["external-identifier"]), "buying-power-effect": {}}

    with caplog.at_level("WARNING"):
        broker.submit_order(_buy("2.7"))

    [body] = session.posted(ORDERS)
    assert body["legs"][0]["quantity"] == "2"
    assert "whole shares" in caplog.text


def test_the_sandbox_refusing_a_fractional_order_fails_it_with_the_reason_and_is_not_a_success() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": PROBE_AAPL_INSTRUMENT})
        return httpx.Response(422, json=PROBE_PREFLIGHT_422)  # carries data.id of a Rejected order

    broker, requests = _real_session_broker(handler)
    # an eligible-looking symbol is needed to even send a fractional quantity: force the flag as a real eligible one would
    broker._eligible["AAPL"] = True
    order = _buy("0.5")

    with pytest.raises(BrokerError, match="Fractional trading of AAPL is currently not allowed"):
        broker.submit_order(order)

    assert order.status is OrderStatus.ERROR
    assert "Fractional trading of AAPL is currently not allowed" in (order.error_message or "")
    assert "preflight_check_failure" in (order.error_message or "")
    assert broker.tracker.get_tracked_order(order.identifier) is None
    assert broker._events.tt_id_of(order) is None  # the 422's data.id is not mistaken for a placed order
    assert len([r for r in requests if r.method == "POST"]) == 1
