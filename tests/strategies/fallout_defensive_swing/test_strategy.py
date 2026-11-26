from __future__ import annotations

from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.brokers.fees import TradingFeeFactory
from trading_agent_framework.config.env import BrokerKind, TradingMode
from trading_agent_framework.entities import AccountBalances, Asset, Order, OrderEvent, OrderSide, Position, PositionSide
from trading_agent_framework.strategies.fallout_defensive_swing import FalloutDefensiveSwingStrategy
from trading_agent_framework.strategies.fallout_defensive_swing.rule import Phase, SwingState
from trading_agent_framework.utils.strategy_factory import Strategies, build_strategy

NAME = "fallout_defensive_swing"
TODAY = date(2026, 11, 30)
YESTERDAY = date(2026, 11, 27)


class FillingBroker(FakeBroker):
    """A cash account whose market orders fill while the strategy waits, 0.02% worse than the last price.

    Symbols in `stuck` never fill (until canceled); with `ignores_cancel` the cancel is never confirmed either.
    """

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock, strategy_name=NAME)
        self.last_prices = {"XLP": D(100), "XLU": D(50)}
        self.account = AccountBalances(cash=D(768), portfolio_value=D(768), buying_power=D(768))
        self.stuck: set[str] = set()
        self.ignores_cancel = False
        clock.on_wait = self.process_orders

    def hold(self, xlp: str, xlu: str, *, cash: str = "0") -> None:
        for symbol, quantity in (("XLP", D(xlp)), ("XLU", D(xlu))):
            self._move(symbol, quantity)
        self._set_cash(D(cash))

    def quantity(self, symbol: str) -> D:
        return next((p.quantity for p in self.positions if p.asset.symbol == symbol), D(0))

    def process_orders(self) -> None:
        for order in self.tracker.get_active_orders():
            symbol = order.asset.symbol
            if symbol not in self.stuck:
                self.fill(order)
            elif order in self.canceled and not self.ignores_cancel:
                self.tracker.process_trade_event(order, OrderEvent.CANCELED)

    def fill(self, order: Order) -> None:
        assert order.quantity is not None
        sign = 1 if order.side is OrderSide.BUY else -1
        price = self.last_prices[order.asset.symbol] * (1 + sign * D("0.0002"))
        self.tracker.process_trade_event(order, OrderEvent.FILLED, price=price, filled_quantity=order.quantity)
        self._move(order.asset.symbol, sign * order.quantity)
        self._set_cash(self.account.cash - sign * price * order.quantity)

    def _move(self, symbol: str, change: D) -> None:
        quantity = self.quantity(symbol) + change
        self.positions = [p for p in self.positions if p.asset.symbol != symbol]
        if quantity > 0:
            self.positions.append(Position(strategy_name=NAME, asset=Asset(symbol), quantity=quantity, side=PositionSide.LONG))

    def _set_cash(self, cash: D) -> None:
        book = cash + sum((p.quantity * self.last_prices[p.asset.symbol] for p in self.positions), D(0))
        self.account = AccountBalances(cash=cash, portfolio_value=book, buying_power=cash)


def _broker() -> FillingBroker:
    return FillingBroker(FakeClock(et(2026, 11, 30, 10, 30)))


def _strategy(broker: FillingBroker, tmp_path: Path, state: SwingState | None = None) -> FalloutDefensiveSwingStrategy:
    strategy = FalloutDefensiveSwingStrategy(broker, mode=TradingMode.PAPER, project_root=tmp_path, fees=TradingFeeFactory(BrokerKind.TASTYTRADE))
    strategy.initialize()
    if state is not None:
        strategy.store.save(state)
    return strategy


def _holding(*, last_sale: date | None = None) -> SwingState:
    return SwingState(phase=Phase.HOLDING, basis=D(750), last_sale_session=last_sale)


def test_initialize_ticks_hourly_from_1030(tmp_path: Path) -> None:
    strategy = _strategy(_broker(), tmp_path)

    assert strategy.sleeptime == "60M"
    assert f"{strategy.iteration_start_time:%H:%M}" == "10:30"
    assert strategy.state_file().name == "fallout_defensive_swing_state_paper.json"


def test_first_tick_buys_the_mix_in_equal_parts_and_saves_the_basis(tmp_path: Path) -> None:
    broker = _broker()
    strategy = _strategy(broker, tmp_path)

    strategy.on_trading_iteration()

    assert [(o.asset.symbol, o.side) for o in broker.submitted] == [("XLP", OrderSide.BUY), ("XLU", OrderSide.BUY)]
    notionals = [o.quantity * broker.last_prices[o.asset.symbol] for o in broker.submitted if o.quantity is not None]
    spendable = D(768) - D("0.02") * D(768)
    assert abs(notionals[0] - spendable / 2) < D("0.01")
    assert abs(notionals[0] - notionals[1]) < D("0.30")  # the 2nd leg is capped by the cash left after the 1st fill
    assert all(o.quantity is not None and -o.quantity.as_tuple().exponent <= 6 for o in broker.submitted)  # type: ignore[operator]
    state = strategy.store.load()
    spent = D(768) - broker.account.cash
    assert state.phase is Phase.HOLDING
    assert state.basis == spent + D("0.22")  # two $0.10 fractional orders plus CAT, each rounded up to $0.11


def test_at_the_target_on_a_later_session_sells_everything(tmp_path: Path) -> None:
    broker = _broker()
    broker.hold("3.8", "7.6")  # worth 760 at 100 / 50: above 750 * 1.01 + fees
    strategy = _strategy(broker, tmp_path, _holding(last_sale=YESTERDAY))

    strategy.on_trading_iteration()

    assert [(o.asset.symbol, o.side, o.quantity) for o in broker.submitted] == [("XLP", OrderSide.SELL, D("3.8")), ("XLU", OrderSide.SELL, D("7.6"))]
    state = strategy.store.load()
    assert state.phase is Phase.CASH and not state.selling
    assert state.sale_prices == {"XLP": D("99.98"), "XLU": D("49.99")}
    assert state.last_sale_session == TODAY


def test_no_sale_on_the_session_of_the_last_sale(tmp_path: Path) -> None:
    broker = _broker()
    broker.hold("3.8", "7.6")
    strategy = _strategy(broker, tmp_path, _holding(last_sale=TODAY))

    strategy.on_trading_iteration()

    assert broker.submitted == []


def test_in_cash_one_percent_under_the_sale_prices_buys_back(tmp_path: Path) -> None:
    broker = _broker()
    broker.last_prices = {"XLP": D(99), "XLU": D("49.5")}
    cash = SwingState(phase=Phase.CASH, sale_prices={"XLP": D(100), "XLU": D(50)}, last_sale_session=TODAY)
    strategy = _strategy(broker, tmp_path, cash)

    strategy.on_trading_iteration()

    assert [o.side for o in broker.submitted] == [OrderSide.BUY, OrderSide.BUY]
    assert strategy.store.load().phase is Phase.HOLDING
    assert strategy.store.load().last_sale_session == TODAY  # still unsettled: no sale before tomorrow


def test_a_position_with_a_fresh_state_is_adopted_not_bought_again(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    broker = _broker()
    broker.hold("3.8", "7.6")
    strategy = _strategy(broker, tmp_path)

    strategy.on_trading_iteration()

    assert broker.submitted == []
    state = strategy.store.load()
    assert (state.phase, state.basis) == (Phase.HOLDING, D(760))
    assert "adopting" in caplog.text


def test_a_mix_sold_outside_the_strategy_waits_for_a_dip_and_for_settlement(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    broker = _broker()
    strategy = _strategy(broker, tmp_path, _holding())

    strategy.on_trading_iteration()

    assert broker.submitted == []
    state = strategy.store.load()
    assert (state.phase, state.sale_prices, state.last_sale_session) == (Phase.CASH, {"XLP": D(100), "XLU": D(50)}, TODAY)
    assert "adopting" in caplog.text


def test_one_sell_not_filled_stays_selling_and_the_next_session_sells_the_rest(tmp_path: Path) -> None:
    broker = _broker()
    broker.hold("3.8", "7.6")
    broker.stuck = {"XLU"}
    strategy = _strategy(broker, tmp_path, _holding())

    strategy.on_trading_iteration()

    state = strategy.store.load()
    assert (state.phase, state.selling, state.last_sale_session) == (Phase.HOLDING, True, TODAY)
    assert [o.asset.symbol for o in broker.canceled] == ["XLU"]

    broker.stuck = set()
    broker.last_prices = {"XLP": D(100), "XLU": D(40)}  # far below target: the rest goes anyway
    broker.clock.advance(24 * 3600)  # type: ignore[attr-defined]
    strategy.on_trading_iteration()

    state = strategy.store.load()
    assert state.phase is Phase.CASH and not state.selling
    assert state.sale_prices == {"XLP": D("99.98"), "XLU": D("39.992")}
    assert broker.positions == []


def test_a_sale_that_fills_nothing_is_not_forced_later(tmp_path: Path) -> None:
    broker = _broker()
    broker.hold("3.8", "7.6")
    broker.stuck = {"XLP", "XLU"}
    strategy = _strategy(broker, tmp_path, _holding())

    strategy.on_trading_iteration()

    assert strategy.store.load() == _holding()


def test_a_buy_that_does_not_fill_is_canceled_and_only_the_filled_leg_is_the_basis(tmp_path: Path) -> None:
    broker = _broker()
    broker.stuck = {"XLU"}
    strategy = _strategy(broker, tmp_path)

    strategy.on_trading_iteration()

    assert [o.asset.symbol for o in broker.canceled] == ["XLU"]
    state = strategy.store.load()
    assert state.phase is Phase.HOLDING and state.basis is not None
    assert state.basis < D(400)


def test_while_an_order_is_still_open_nothing_is_saved_and_ticks_do_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    broker = _broker()
    broker.stuck, broker.ignores_cancel = {"XLU"}, True
    strategy = _strategy(broker, tmp_path)

    strategy.on_trading_iteration()
    strategy.on_trading_iteration()

    assert len(broker.submitted) == 2
    assert strategy.store.load() == SwingState()
    assert "still open" in caplog.text


def test_a_missing_price_does_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    broker = _broker()
    del broker.last_prices["XLU"]
    strategy = _strategy(broker, tmp_path)

    strategy.on_trading_iteration()

    assert broker.submitted == []
    assert "no price for XLU" in caplog.text


def test_backtest_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr("trading_agent_framework.core.strategy.Strategy.run_backtesting", lambda self, **kwargs: seen.update(kwargs))
    strategy = _strategy(_broker(), tmp_path)

    strategy.run_backtesting()

    assert seen["timestep"] == "minute" and seen["agent_telemetry"] is False
    assert seen["data_source"].__name__ == "AlpacaBacktestData"
    assert strategy.resolved_window() == (et(2026, 11, 3, 0), et(2026, 11, 25, 0))
    assert strategy.parameters["budget"] == 768


def test_the_factory_builds_it() -> None:
    broker = FakeBroker(FakeClock(et(2026, 11, 30, 10, 30)), strategy_name=NAME)

    assert isinstance(build_strategy(Strategies(NAME), broker, TradingMode.BACKTESTING), FalloutDefensiveSwingStrategy)
