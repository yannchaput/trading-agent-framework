"""`Strategy.spendable_cash` and the buy budget check in `Strategy.submit_order`."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import InsufficientFundsError, OrderValidationError


class _SellFillsAfterAccountRead(FakeBroker):
    """A broker whose open sell fills (cash grows, order leaves the book) right after the account is read."""

    def get_account(self) -> AccountBalances:
        account = self.account
        for order in self.tracker.get_active_orders():
            self.tracker.untrack(order)
        return account


def _broker(cls: type[FakeBroker] = FakeBroker) -> FakeBroker:
    broker = cls(FakeClock(et(2026, 9, 14, 9, 0)))
    broker.last_prices.update({"AAA": Decimal(100), "BBB": Decimal(200)})
    return broker


def _open(strategy: Strategy, broker: FakeBroker, symbol: str, quantity: int, side: str, limit_price: int | None = None, stop_price: int | None = None) -> None:
    """Track an open order without going through the check."""
    broker.tracker.track_unprocessed(strategy.create_order(symbol, quantity, side, limit_price=limit_price, stop_price=stop_price))


def test_spendable_is_the_smaller_term_less_two_percent_of_the_book() -> None:
    assert Strategy(_broker()).spendable_cash() == Decimal("9500")  # min(20000, 10000) - 2% x 25000


def test_buying_power_binds_when_it_is_smaller() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    broker.account = AccountBalances(cash=Decimal(10000), portfolio_value=Decimal(25000), buying_power=Decimal(4000))
    assert strategy.spendable_cash() == Decimal("3500")


def test_open_sells_credit_only_their_unfilled_part() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    _open(strategy, broker, "AAA", 10, "sell", limit_price=100)
    broker.tracker.get_active_orders()[0].add_transaction(Decimal(100), Decimal(4))
    assert strategy.spendable_cash() == Decimal("10100")  # 9500 + 6 x 100


def test_open_buys_are_deducted() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    _open(strategy, broker, "BBB", 5, "buy")  # no limit: last price 200
    assert strategy.spendable_cash() == Decimal("8500")


def test_a_stop_sell_brings_no_credit() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    _open(strategy, broker, "AAA", 10, "sell", stop_price=90)
    assert strategy.spendable_cash() == Decimal("9500")


def test_a_sell_filling_between_the_reads_is_not_credited_twice() -> None:
    broker = _broker(_SellFillsAfterAccountRead)
    strategy = Strategy(broker)
    _open(strategy, broker, "AAA", 10, "sell", limit_price=100)
    assert strategy.spendable_cash() == Decimal("9500")


def test_a_buy_over_spendable_raises_with_the_max_quantity() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    with pytest.raises(InsufficientFundsError) as err:
        strategy.submit_order(strategy.create_order("AAA", 101, "buy"))  # $10,100 > the $10,000 of cash
    assert err.value.max_quantity == Decimal(95)  # the retry lands back inside the buffer
    assert err.value.spendable == Decimal(9500)
    assert isinstance(err.value, OrderValidationError)
    assert "buy 101 AAA needs ~$10,100.00, spendable is $9,500.00" in str(err.value)
    assert "at most 95 shares" in str(err.value)
    assert broker.submitted == []


def test_a_buy_at_spendable_passes() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    strategy.submit_order(strategy.create_order("AAA", 95, "buy"))
    assert len(broker.submitted) == 1


def test_a_buy_into_the_buffer_passes() -> None:
    # The 2% buffer absorbs price drift between sizing and sending: only the cash itself is a hard limit.
    broker = _broker()
    strategy = Strategy(broker)
    strategy.submit_order(strategy.create_order("AAA", 100, "buy"))  # $10,000: over spendable $9,500, within cash $10,000
    assert len(broker.submitted) == 1


def test_a_limit_buy_is_priced_at_its_limit() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    strategy.submit_order(strategy.create_order("AAA", 100, "buy", limit_price=90))  # $9,000 <= $9,500
    assert len(broker.submitted) == 1


def test_a_sell_and_a_buy_without_a_price_pass_unchecked() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    strategy.submit_order(strategy.create_order("AAA", 500, "sell"))
    strategy.submit_order(strategy.create_order("ZZZ", 500, "buy"))  # no last price for ZZZ
    assert len(broker.submitted) == 2


def test_a_fully_invested_book_gives_max_quantity_zero() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    broker.account = AccountBalances(cash=Decimal(100), portfolio_value=Decimal(25000), buying_power=Decimal(100))
    with pytest.raises(InsufficientFundsError) as err:
        strategy.submit_order(strategy.create_order("AAA", 2, "buy"))  # $200 > the $100 of cash
    assert err.value.max_quantity == Decimal(0)
    assert err.value.spendable == Decimal(-400)


def test_submit_orders_checks_each_buy_against_the_ones_before() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    first, second = strategy.create_order("AAA", 60, "buy"), strategy.create_order("AAA", 60, "buy")
    with pytest.raises(InsufficientFundsError):
        strategy.submit_orders([first, second])
    assert broker.submitted == [first]


class _CountingBroker(FakeBroker):
    """Counts price lookups, to show open orders are priced with one batched call."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.single_lookups = 0
        self.batch_lookups = 0

    def get_last_price(self, asset: Asset) -> Decimal | None:
        self.single_lookups += 1
        return super().get_last_price(asset)

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        self.batch_lookups += 1
        return super().get_last_prices(assets)


def test_open_market_orders_are_priced_with_one_batched_lookup() -> None:
    broker = _broker(_CountingBroker)
    assert isinstance(broker, _CountingBroker)
    strategy = Strategy(broker)
    for _ in range(20):
        _open(strategy, broker, "AAA", 1, "buy")
    assert strategy.spendable_cash() == Decimal("7500")  # 9500 - 20 x 100
    assert (broker.batch_lookups, broker.single_lookups) == (1, 0)


def test_open_stop_sells_need_no_price_lookup() -> None:
    """A stop or trailing-stop sell credits nothing, so it is not quoted (earnings_drift and vwap hold one per position)."""
    broker = _broker(_CountingBroker)
    assert isinstance(broker, _CountingBroker)
    strategy = Strategy(broker)
    _open(strategy, broker, "AAA", 10, "sell", stop_price=90)
    broker.tracker.track_unprocessed(strategy.create_order("BBB", 5, "sell", trail_percent=8))
    assert strategy.spendable_cash() == Decimal("9500")
    assert (broker.batch_lookups, broker.single_lookups) == (0, 0)


def test_an_open_stop_buy_is_still_priced_and_debited() -> None:
    broker = _broker(_CountingBroker)
    assert isinstance(broker, _CountingBroker)
    strategy = Strategy(broker)
    _open(strategy, broker, "AAA", 10, "buy", stop_price=110)
    assert strategy.spendable_cash() == Decimal("8500")  # 9500 - 10 x the last price 100
    assert broker.batch_lookups == 1


def test_open_stop_limit_and_trailing_sells_bring_no_credit() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    _open(strategy, broker, "AAA", 10, "sell", limit_price=95, stop_price=90)  # a stop-limit: its limit is no promise to sell
    broker.tracker.track_unprocessed(strategy.create_order("BBB", 5, "sell", trail_price=10))
    assert strategy.spendable_cash() == Decimal("9500")


def test_an_open_limit_buy_is_deducted_at_its_limit_price() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    _open(strategy, broker, "BBB", 5, "buy", limit_price=150)  # last price 200
    assert strategy.spendable_cash() == Decimal("8750")  # 9500 - 5 x 150


def test_submit_orders_lets_an_earlier_sell_fund_a_later_buy() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    sell, buy = strategy.create_order("BBB", 10, "sell", limit_price=200), strategy.create_order("AAA", 110, "buy")
    strategy.submit_orders([sell, buy])  # the buy alone ($11,000) is over the $9,500, the open sell adds $2,000
    assert broker.submitted == [sell, buy]


def test_spendable_cash_uses_a_given_account_snapshot_instead_of_reading_again() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    snapshot = AccountBalances(cash=Decimal(5000), portfolio_value=Decimal(10000), buying_power=Decimal(8000))
    broker.account = AccountBalances(cash=Decimal(0), portfolio_value=Decimal(0), buying_power=Decimal(0))
    _open(strategy, broker, "AAA", 10, "sell", limit_price=100)  # the open orders are still read
    assert strategy.spendable_cash(account=snapshot) == Decimal("5800")  # min(8000, 5000 + 1000) - 2% x 10000
