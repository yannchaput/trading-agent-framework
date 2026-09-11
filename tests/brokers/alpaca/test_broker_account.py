from __future__ import annotations

from decimal import Decimal
from typing import cast

import pytest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from tests.fakes import (
    FakeTradingClient,
    make_alpaca_account,
    make_alpaca_order,
    make_api_error,
    make_close_position_response,
    make_failed_close_details,
)

from trading_agent_framework.brokers.alpaca import broker as broker_module
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError, OrderValidationError

_OLD_ID = "11111111-1111-1111-1111-111111111111"
_NEW_ID = "22222222-2222-2222-2222-222222222222"


def _broker(client: FakeTradingClient) -> AlpacaBroker:
    return AlpacaBroker("momentum", client)


def _tracked_limit_order(broker: AlpacaBroker) -> Order:
    order = Order(
        strategy_name="momentum",
        asset=Asset("AAPL"),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal(1),
        limit_price=Decimal("100"),
        identifier=_OLD_ID,
    )
    broker.tracker.track_unprocessed(order)
    return order


def test_default_clock_is_an_alpaca_market_clock() -> None:
    assert isinstance(_broker(FakeTradingClient()).clock, AlpacaMarketClock)


def test_from_credentials_records_the_account_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_module, "build_trading_client", lambda creds: FakeTradingClient())
    creds = AlpacaCredentials(api_key="k", api_secret="s", is_paper=False)
    broker = AlpacaBroker.from_credentials("momentum", creds, with_stream=False)
    assert broker.is_paper is False


def test_get_account_parses_balances() -> None:
    client = FakeTradingClient()
    client.account_response = make_alpaca_account()
    assert _broker(client).get_account() == AccountBalances(
        cash=Decimal("10000.50"),
        portfolio_value=Decimal("25000.25"),
        buying_power=Decimal("20000"),
    )


def test_get_account_wraps_client_errors() -> None:
    client = FakeTradingClient()
    client.raises["get_account"] = make_api_error(500)
    with pytest.raises(BrokerError, match="account"):
        _broker(client).get_account()


def test_modify_order_replaces_and_tracks_the_new_order() -> None:
    client = FakeTradingClient()
    client.replace_response = make_alpaca_order(
        id=_NEW_ID, type="limit", limit_price="101.00", client_order_id="momentum:x"
    )
    broker = _broker(client)
    old = _tracked_limit_order(broker)

    new = broker.modify_order(old, limit_price=Decimal("101"))

    [(order_id, request)] = client.replace_calls
    assert order_id == _OLD_ID
    assert request.limit_price == 101.0
    assert new.identifier == _NEW_ID
    assert old.status == OrderStatus.CANCELED
    assert broker.tracker.get_tracked_order(_NEW_ID) is new


def test_modify_order_without_prices_is_a_validation_error() -> None:
    broker = _broker(FakeTradingClient())
    with pytest.raises(OrderValidationError):
        broker.modify_order(_tracked_limit_order(broker))


def test_modify_order_wraps_client_errors() -> None:
    client = FakeTradingClient()
    client.raises["replace_order_by_id"] = make_api_error(422)
    broker = _broker(client)
    with pytest.raises(BrokerError, match=_OLD_ID):
        broker.modify_order(_tracked_limit_order(broker), limit_price=Decimal("101"))


def test_close_position_sends_a_percentage_and_tracks_the_order() -> None:
    client = FakeTradingClient()
    client.close_position_response = make_alpaca_order(id=_NEW_ID, side="sell")
    broker = _broker(client)

    order = broker.close_position(Asset("AAPL"), Decimal("0.5"))

    [(symbol, request)] = client.close_position_calls
    assert symbol == "AAPL"
    assert request.percentage == "50.000000000"
    assert order is not None
    assert broker.tracker.get_tracked_order(_NEW_ID) is order


def test_close_position_returns_none_without_a_position() -> None:
    client = FakeTradingClient()
    client.raises["close_position"] = make_api_error(404)
    assert _broker(client).close_position(Asset("AAPL")) is None


def test_close_position_wraps_other_errors() -> None:
    client = FakeTradingClient()
    client.raises["close_position"] = make_api_error(403)
    with pytest.raises(BrokerError, match="AAPL"):
        _broker(client).close_position(Asset("AAPL"))


def test_close_all_positions_tracks_placed_orders_and_skips_failures() -> None:
    client = FakeTradingClient()
    client.close_all_response = [
        make_close_position_response(make_alpaca_order(id=_NEW_ID, side="sell"), "AAPL"),
        make_close_position_response(make_failed_close_details("TSLA"), "TSLA"),
    ]
    broker = _broker(client)

    closed = broker.close_all_positions(cancel_orders=False)

    assert client.close_all_calls == [False]
    assert [o.identifier for o in closed] == [_NEW_ID]
    assert broker.tracker.get_tracked_order(_NEW_ID) is closed[0]


def test_close_all_positions_wraps_client_errors() -> None:
    client = FakeTradingClient()
    client.raises["close_all_positions"] = make_api_error(500)
    with pytest.raises(BrokerError):
        _broker(client).close_all_positions()


def test_sync_open_orders_adopts_only_this_strategys_untracked_orders() -> None:
    client = FakeTradingClient()
    mine = make_alpaca_order(id=_NEW_ID, client_order_id="momentum:abc")
    already_tracked = make_alpaca_order(id=_OLD_ID, client_order_id="momentum:def")
    other = make_alpaca_order(client_order_id="other:ghi")
    manual = make_alpaca_order(client_order_id="manual-order")
    client.orders_response = [mine, already_tracked, other, manual]
    broker = _broker(client)
    _tracked_limit_order(broker)  # identifier _OLD_ID

    adopted = broker.sync_open_orders()

    request = cast(GetOrdersRequest, client.last_orders_request)
    assert request.status == QueryOrderStatus.OPEN
    assert [o.identifier for o in adopted] == [_NEW_ID]
    assert broker.tracker.get_tracked_order(_NEW_ID) is adopted[0]
    assert len(broker.tracker.get_all_tracked_orders()) == 2


def test_sync_open_orders_wraps_client_errors() -> None:
    client = FakeTradingClient()
    client.raises["get_orders"] = make_api_error(500)
    with pytest.raises(BrokerError, match="open orders"):
        _broker(client).sync_open_orders()
