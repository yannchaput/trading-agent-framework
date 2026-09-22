from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pytest
from tests.fakes import (
    FakeClock,
    FakeIB,
    FakeStockHistoricalDataClient,
    FakeTradingClient,
    et,
    make_alpaca_trade,
    make_ib_fill,
    make_ib_portfolio_item,
    make_ib_summary,
    make_ib_trade,
)

from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.ibkr.broker import IbkrBroker
from trading_agent_framework.brokers.ibkr.client import IbkrConnection
from trading_agent_framework.config.env import AlpacaCredentials, IbkrSettings
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, OrderValidationError

AAPL = Asset("AAPL")
SETTINGS = IbkrSettings(host="127.0.0.1", port=4002, client_id=1, is_paper=True)


@pytest.fixture
def ib() -> FakeIB:
    return FakeIB()


@pytest.fixture
def broker(ib: FakeIB) -> Iterator[IbkrBroker]:
    connection = IbkrConnection(SETTINGS, ib_factory=lambda: ib, call_timeout=2.0, sleep=lambda s: None)
    connection.start()
    connection.connect()
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 100.5)}
    built = IbkrBroker(
        "s", connection, market_data=AlpacaMarketData(data, FakeTradingClient()), client_id=1,
        clock=FakeClock(et(2026, 9, 22, 10)), ack_timeout=0.2,
    )
    yield built
    connection.disconnect()


def _buy(quantity: str = "10", **overrides: object) -> Order:
    fields: dict[str, object] = {"strategy_name": "s", "asset": AAPL, "side": OrderSide.BUY, "quantity": Decimal(quantity)}
    fields.update(overrides)
    return Order(**fields)  # ty: ignore[invalid-argument-type]


# --- submit -----------------------------------------------------------------------------


def test_submit_places_a_smart_order_tagged_with_the_client_order_id(broker: IbkrBroker, ib: FakeIB) -> None:
    order = broker.submit_order(_buy())

    [(contract, ib_order)] = ib.placed
    assert (contract.symbol, contract.exchange) == ("AAPL", "SMART")
    assert contract.conId  # qualified
    assert ib_order.orderRef == f"s:{order.identifier}" == order.client_order_id
    assert order.status is OrderStatus.SUBMITTED
    assert broker.tracker.get_tracked_order(order.identifier) is order


def test_contracts_are_qualified_once_per_symbol(broker: IbkrBroker, ib: FakeIB) -> None:
    calls: list[str] = []
    original = ib.qualifyContractsAsync

    async def counting(*contracts):
        calls.append(contracts[0].symbol)
        return await original(*contracts)

    ib.qualifyContractsAsync = counting  # ty: ignore[invalid-assignment]
    broker.submit_order(_buy())
    broker.submit_order(_buy())

    assert calls == ["AAPL"]


def test_an_unknown_symbol_is_a_validation_error(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.unknown_symbols = {"AAPL"}

    with pytest.raises(OrderValidationError, match="does not know the US stock AAPL"):
        broker.submit_order(_buy())


def test_a_rejection_sets_the_error_before_raising_and_untracks(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.place_status = "Inactive"
    ib.place_log_message = "No trading permissions"
    ib.place_log_error_code = 201
    order = _buy()

    with pytest.raises(BrokerError, match="IBKR rejected order .*No trading permissions"):
        broker.submit_order(order)

    assert order.status is OrderStatus.ERROR
    assert order.error_message == "No trading permissions (IBKR error 201)"
    assert broker.tracker.get_tracked_order(order.identifier) is None


def test_notional_orders_are_rejected(broker: IbkrBroker) -> None:
    with pytest.raises(OrderValidationError, match="notional"):
        broker.submit_order(Order(strategy_name="s", asset=AAPL, side=OrderSide.BUY, notional=Decimal(100)))


def test_fractional_quantities_are_floored(broker: IbkrBroker, ib: FakeIB) -> None:
    broker.submit_order(_buy("2.6"))

    assert ib.placed[-1][1].totalQuantity == 2.0


# --- cancel / modify ----------------------------------------------------------------------


def test_cancel_cancels_the_matching_ib_order(broker: IbkrBroker, ib: FakeIB) -> None:
    order = broker.submit_order(_buy())

    broker.cancel_order(order)

    assert [o.orderRef for o in ib.canceled] == [order.client_order_id]


def test_cancel_of_an_unknown_order_is_a_broker_error(broker: IbkrBroker) -> None:
    with pytest.raises(BrokerError, match="no open IBKR order"):
        broker.cancel_order(_buy(client_order_id="s:missing"))


def test_orders_of_another_client_id_cannot_be_cancelled_or_modified(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.all_trades.append(make_ib_trade(order_ref="s:old", client_id=9, status="Submitted"))
    order = _buy(identifier="old", client_order_id="s:old")

    with pytest.raises(BrokerError, match="placed by client id 9"):
        broker.cancel_order(order)
    with pytest.raises(BrokerError, match="placed by client id 9"):
        broker.modify_order(order, limit_price=Decimal(99))


def test_modify_edits_the_order_in_place(broker: IbkrBroker, ib: FakeIB) -> None:
    order = broker.submit_order(_buy(order_type=OrderType.LIMIT, limit_price=Decimal(100)))

    modified = broker.modify_order(order, limit_price=Decimal("99.5"))

    assert modified is order
    assert order.limit_price == Decimal("99.5")
    assert len(ib.all_trades) == 1
    assert ib.placed[-1][1].lmtPrice == 99.5
    assert ib.placed[-1][1].orderId == ib.placed[0][1].orderId


# --- pulls and positions ------------------------------------------------------------------


def test_pull_positions_reads_the_portfolio(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.portfolio_items = [make_ib_portfolio_item(symbol="AAPL", position=10.0), make_ib_portfolio_item(symbol="MSFT", position=0.0)]

    [position] = broker.pull_positions()

    assert (position.asset, position.quantity, position.side) == (AAPL, Decimal(10), PositionSide.LONG)


def test_pull_orders_and_pull_order(broker: IbkrBroker) -> None:
    order = broker.submit_order(_buy())

    assert [o.identifier for o in broker.pull_orders()] == [order.identifier]
    pulled = broker.pull_order(order.identifier)
    assert pulled is not None and pulled.client_order_id == order.client_order_id
    assert broker.pull_order("nope") is None


def test_close_position_sells_the_floored_fraction(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.portfolio_items = [make_ib_portfolio_item(symbol="AAPL", position=10.0)]

    order = broker.close_position(AAPL, Decimal("0.35"))

    assert order is not None
    assert (order.side, order.quantity, order.order_type) == (OrderSide.SELL, Decimal(3), OrderType.MARKET)


def test_close_position_without_a_position_is_none(broker: IbkrBroker) -> None:
    assert broker.close_position(AAPL) is None


def test_close_all_positions_cancels_then_sells_everything(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.portfolio_items = [make_ib_portfolio_item(symbol="AAPL", position=10.0), make_ib_portfolio_item(symbol="MSFT", position=5.0)]

    closed = broker.close_all_positions()

    assert ib.global_cancels == 1
    assert sorted((o.asset.symbol, o.quantity) for o in closed) == [("AAPL", Decimal(10)), ("MSFT", Decimal(5))]


def test_sync_open_orders_adopts_this_strategys_untracked_orders(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.foreign_open_trades = [make_ib_trade(order_ref="s:old", client_id=9, perm_id=77), make_ib_trade(order_ref="other:x", perm_id=78)]

    adopted = broker.sync_open_orders()

    assert [o.identifier for o in adopted] == ["old"]
    assert broker.tracker.get_tracked_order("old") is adopted[0]
    assert broker.sync_open_orders() == []


# --- account ------------------------------------------------------------------------------


def test_get_account(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.summary = make_ib_summary(cash="1000", net_liquidation="2500", buying_power="1000")

    assert broker.get_account() == AccountBalances(cash=Decimal(1000), portfolio_value=Decimal(2500), buying_power=Decimal(1000))


def test_configure_account_refuses_a_paper_mismatch(broker: IbkrBroker) -> None:
    broker.is_paper = False

    with pytest.raises(ConfigurationError, match="BROKER_API_IS_PAPER"):
        broker.configure_account()


def test_more_than_one_managed_account_is_refused(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.accounts = ["DU1", "DU2"]

    with pytest.raises(ConfigurationError, match="exactly one account"):
        broker.get_account()


# --- data, events, reconcile ----------------------------------------------------------------


def test_market_data_comes_from_alpaca(broker: IbkrBroker) -> None:
    assert broker.get_last_price(AAPL) == Decimal("100.5")


def test_the_stream_feeds_the_tracker(broker: IbkrBroker, ib: FakeIB) -> None:
    broker.start_stream()
    order = broker.submit_order(_buy())
    trade = ib.all_trades[0]

    # Emitted on the loop thread, where ib_async emits them in production.
    broker._connection.call(lambda _: ib.orderStatusEvent.emit(trade))
    broker._connection.call(lambda _: ib.execDetailsEvent.emit(trade, make_ib_fill(order_ref=order.client_order_id, cum_qty=10.0)))

    assert order.status is OrderStatus.FILL


def test_reconcile_applies_missed_fills_once(broker: IbkrBroker, ib: FakeIB) -> None:
    order = broker.submit_order(_buy())
    ib.executions = [make_ib_fill(order_ref=order.client_order_id, exec_id="e9", shares=10.0, cum_qty=10.0)]

    broker.reconcile()
    broker.reconcile()

    assert order.status is OrderStatus.FILL
    assert order.filled_quantity == Decimal(10)


def test_from_settings_connects_and_checks_the_account(ib: FakeIB) -> None:
    connection = IbkrConnection(SETTINGS, ib_factory=lambda: ib)

    built = IbkrBroker.from_settings(
        "s", SETTINGS, data=AlpacaCredentials("k", "s"), connection=connection,
        market_data=AlpacaMarketData(FakeStockHistoricalDataClient(), FakeTradingClient()),
    )
    try:
        assert ib.connect_calls == [("127.0.0.1", 4002, 1)]
        assert built.is_paper is True
        assert built.account_id == "DU123"
    finally:
        connection.disconnect()


def test_from_settings_disconnects_when_the_account_check_fails(ib: FakeIB) -> None:
    live = IbkrSettings(host="127.0.0.1", port=4001, client_id=1, is_paper=False)  # but the account is DU123
    connection = IbkrConnection(live, ib_factory=lambda: ib)

    with pytest.raises(ConfigurationError, match="BROKER_API_IS_PAPER"):
        IbkrBroker.from_settings(
            "s", live, data=AlpacaCredentials("k", "s"), connection=connection,
            market_data=AlpacaMarketData(FakeStockHistoricalDataClient(), FakeTradingClient()),
        )

    assert ib.connected is False
