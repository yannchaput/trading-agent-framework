"""`CrossMomentumStrategy.rebalance` cash budgeting.

Market orders fill at a later bar's open (plus commission), so the cash the strategy computes at decision
time is only an estimate. `cash_buffer_pct` holds a slice of it back so the buys still land.
"""

from types import SimpleNamespace

from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy


class FakeStrategy:
    """Just enough of `Strategy` for `rebalance`; records the orders it places."""

    def __init__(self, *, cash, positions=(), last_prices=None, cash_buffer_pct=0.05):
        self.parameters = {"sell_rank_threshold": 35, "cash_buffer_pct": cash_buffer_pct}
        self._cash = cash
        self._positions = list(positions)
        self._last_prices = last_prices or {}
        self.portfolio_value = cash + sum(p.quantity * self._last_prices[p.asset.symbol] for p in self._positions)
        self.orders = []

    def get_positions(self):
        return self._positions

    def get_cash(self):
        return self._cash

    def get_last_price(self, symbol):
        return self._last_prices[symbol]

    def create_order(self, symbol, quantity, side, **kwargs):
        return SimpleNamespace(symbol=symbol, quantity=quantity, side=side)

    def submit_order(self, order):
        self.orders.append(order)

    def log_info(self, *args, **kwargs): ...
    def log_warning(self, *args, **kwargs): ...
    def log_error(self, *args, **kwargs): ...


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
