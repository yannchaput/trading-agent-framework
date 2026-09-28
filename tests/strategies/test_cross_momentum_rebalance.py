"""`CrossMomentumStrategy.rebalance` cash budgeting.

Market orders fill at a later bar's open (plus fees), so the cash the strategy computes at decision
time is only an estimate. `cash_buffer_pct` holds a slice of it back so the buys still land.
"""

from types import SimpleNamespace

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.utils.errors import BrokerError


class FakeStrategy:
    """Just enough of `Strategy` for `rebalance`; records the orders it places."""

    def __init__(self, *, cash, positions=(), last_prices=None, cash_buffer_pct=0.05, min_trade_pct=0.01, reject=(), price_errors=()):
        self.parameters = {
            "sell_rank_threshold": 35,
            "cash_buffer_pct": cash_buffer_pct,
            "parking": {"symbol": "SHV", "min_trade_pct": min_trade_pct},
        }
        self._cash = cash
        self._positions = list(positions)
        self._last_prices = last_prices or {}
        self._reject = set(reject)
        self._price_errors = set(price_errors)
        self.portfolio_value = cash + sum(p.quantity * self._last_prices[p.asset.symbol] for p in self._positions)
        self.orders = []
        self.warnings: list[str] = []

    def get_positions(self):
        return self._positions

    def get_cash(self):
        return self._cash

    def get_last_price(self, symbol):
        if symbol in self._price_errors:
            raise BrokerError(f"no quote for {symbol}")
        return self._last_prices.get(symbol)

    def create_order(self, symbol, quantity, side, **kwargs):
        return SimpleNamespace(symbol=symbol, quantity=quantity, side=side)

    def submit_order(self, order):
        if order.symbol in self._reject:
            raise RuntimeError(f"rejected {order.symbol}")
        self.orders.append(order)

    _price_or_zero = CrossMomentumStrategy._price_or_zero

    def log_info(self, *args, **kwargs): ...

    def log_warning(self, message, *args, **kwargs):
        self.warnings.append(message)

    def log_error(self, message, *args, **kwargs):
        self.warnings.append(message)


def _target(symbol, weight, price, rank):
    return {"symbol": symbol, "target_weight": weight, "price": price, "rank": rank}


def _planned_buy_cost(fake, prices):
    return sum(o.quantity * prices[o.symbol] for o in fake.orders if o.side == "buy")


def test_buys_leave_the_cash_buffer_unspent():
    prices = {"AAA": 100.0, "BBB": 50.0}
    fake = FakeStrategy(cash=1000.0, last_prices=prices, cash_buffer_pct=0.05)
    target = [_target("AAA", 0.5, 100.0, 1), _target("BBB", 0.5, 50.0, 2)]

    CrossMomentumStrategy.rebalance(fake, target, {"AAA": 1, "BBB": 2})

    assert _planned_buy_cost(fake, prices) <= 1000.0 * 0.95 + 1e-6


def test_buffer_applies_to_estimated_sell_proceeds_too():
    prices = {"OLD": 10.0, "AAA": 100.0, "BBB": 50.0}
    held = SimpleNamespace(asset=SimpleNamespace(symbol="OLD"), quantity=100.0)  # worth 1000
    fake = FakeStrategy(cash=0.0, positions=[held], last_prices=prices, cash_buffer_pct=0.05)

    target = [_target("AAA", 0.5, 100.0, 1), _target("BBB", 0.5, 50.0, 2)]

    CrossMomentumStrategy.rebalance(fake, target, {"AAA": 1, "BBB": 2})  # OLD is unranked -> sold

    assert _planned_buy_cost(fake, prices) <= 1000.0 * 0.95 + 1e-6


def test_zero_buffer_spends_everything_available():
    prices = {"AAA": 100.0, "BBB": 50.0}
    fake = FakeStrategy(cash=1000.0, last_prices=prices, cash_buffer_pct=0.0)
    target = [_target("AAA", 0.5, 100.0, 1), _target("BBB", 0.5, 50.0, 2)]

    CrossMomentumStrategy.rebalance(fake, target, {"AAA": 1, "BBB": 2})

    assert _planned_buy_cost(fake, prices) > 1000.0 * 0.99


def _held(symbol, quantity):
    return SimpleNamespace(asset=SimpleNamespace(symbol=symbol), quantity=quantity)


def _orders(fake, symbol, side):
    return [o.quantity for o in fake.orders if o.symbol == symbol and o.side == side]


def test_idle_cash_is_swept_into_shv():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "buy") == [3.0]
    assert _orders(fake, "SHV", "buy") == [13.0]  # 950 - 300 = 650 parked, the 5% reserve stays cash


def test_an_exposure_drop_trims_the_position_and_parks_the_proceeds():
    fake = FakeStrategy(cash=0.0, positions=[_held("AAA", 10.0)], last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.4, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "sell") == [6.0]
    assert _orders(fake, "SHV", "buy") == [11.0]  # target 950 - 400 = 550


def test_an_exposure_rise_sells_shv_before_funding_the_stock_buys():
    fake = FakeStrategy(cash=0.0, positions=[_held("SHV", 20.0)], last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.9, 100.0, 1)], {"AAA": 1})

    assert [(o.symbol, o.side, o.quantity) for o in fake.orders] == [("SHV", "sell", 19.0), ("AAA", "buy", 9.0)]


def test_shv_within_its_band_is_left_alone_and_never_exited_as_unranked():
    fake = FakeStrategy(cash=500.0, positions=[_held("SHV", 10.0)], last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.45, 100.0, 1)], {"AAA": 1})  # SHV target 500 = held

    assert _orders(fake, "SHV", "sell") == []
    assert _orders(fake, "SHV", "buy") == []


def test_zero_parking_target_sells_the_whole_shv_holding():
    fake = FakeStrategy(cash=0.0, positions=[_held("SHV", 3.333333)], last_prices={"AAA": 100.0, "SHV": 30.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 1.0, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "SHV", "sell") == [3.333333]


def test_a_trim_below_the_minimum_trade_is_skipped():
    fake = FakeStrategy(cash=9950.0, positions=[_held("AAA", 0.5)], last_prices={"AAA": 100.0, "SHV": 50.0})

    # 50 held vs target 20: above the band, but the 30 trim is under 1% of 10 000
    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.002, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "sell") == []


def test_an_shv_buy_below_the_minimum_trade_is_skipped():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.945, 100.0, 1)], {"AAA": 1})  # SHV target 5 < 10

    assert _orders(fake, "SHV", "buy") == []


def test_a_missing_shv_price_skips_parking_but_not_the_stock_orders():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "buy") == [3.0]
    assert [o for o in fake.orders if o.symbol == "SHV"] == []
    assert any("SHV" in message for message in fake.warnings)


def test_hysteresis_holdings_reduce_the_parking_target():
    prices = {"AAA": 100.0, "HYS": 50.0, "SHV": 50.0}
    fake = FakeStrategy(cash=800.0, positions=[_held("HYS", 4.0)], last_prices=prices)  # HYS worth 200

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1, "HYS": 30})

    assert _orders(fake, "HYS", "sell") == []
    assert _orders(fake, "SHV", "buy") == [9.0]  # 950 - 300 - 200 = 450


def test_a_rejected_shv_buy_is_logged_and_does_not_raise():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0, "SHV": 50.0}, reject={"SHV"})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "buy") == [3.0]
    assert any("SHV" in message for message in fake.warnings)


def test_backtests_preload_the_parking_symbol_once():
    fake = SimpleNamespace(parameters={"parking": {"symbol": "SHV"}}, vars=SimpleNamespace(universe=["AAA", "SHV", "BBB"]))

    assets = CrossMomentumStrategy._backtest_preload_assets(fake)

    assert [a.symbol for a in assets] == ["AAA", "SHV", "BBB"]
    fake.vars.universe = ["AAA"]
    assert CrossMomentumStrategy._backtest_preload_assets(fake) == [Asset(symbol="AAA"), Asset(symbol="SHV")]


def test_a_failing_shv_quote_skips_parking_but_not_the_stock_orders():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0, "SHV": 50.0}, price_errors={"SHV"})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "buy") == [3.0]
    assert [o for o in fake.orders if o.symbol == "SHV"] == []
    assert any("SHV" in message for message in fake.warnings)


def test_a_failing_quote_for_a_hysteresis_holding_does_not_abort_the_rebalance():
    prices = {"AAA": 100.0, "HYS": 50.0, "SHV": 50.0}
    fake = FakeStrategy(cash=800.0, positions=[_held("HYS", 4.0)], last_prices=prices, price_errors={"HYS"})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1, "HYS": 30})

    assert _orders(fake, "AAA", "buy") == [3.0]
    assert any("HYS" in message for message in fake.warnings)
