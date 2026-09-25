"""`CrossMomentumStrategy._compute_indicators_for_ticker` error isolation.

A universe-wide `compute_target_portfolio()` pass must not fail entirely
because one ticker's broker request errors out (e.g. a symbol Alpaca
rejects) -- that ticker should be skipped like any other filtered-out
ticker, not blow up the whole rebalance.
"""

from types import SimpleNamespace

from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.utils.errors import BrokerError


class FakeStrategy:
    """Just enough of `Strategy` for `_compute_indicators_for_ticker`."""

    def __init__(self, *, bars=None, raises=None):
        self.parameters = {"min_trading_days": 250, "skip_days": 21, "volatility_window": 20}
        self.vars = SimpleNamespace(alpaca_rate_limiter=SimpleNamespace(wait=lambda: None))
        self._bars = bars
        self._raises = raises
        self.warnings: list[str] = []

    def get_historical_prices(self, ticker, length, timestep):
        if self._raises is not None:
            raise self._raises
        return self._bars

    def log_error(self, *args, **kwargs): ...

    def log_warning(self, message, *args, **kwargs):
        self.warnings.append(message)


def test_a_broker_error_for_one_ticker_is_skipped_not_raised():
    fake = FakeStrategy(raises=BrokerError("invalid symbol: BRK-A"))

    result = CrossMomentumStrategy._compute_indicators_for_ticker(fake, "BRK-A")

    assert result is None


def test_a_broker_error_for_one_ticker_logs_a_warning_naming_it():
    fake = FakeStrategy(raises=BrokerError("invalid symbol: BRK-A"))

    CrossMomentumStrategy._compute_indicators_for_ticker(fake, "BRK-A")

    assert any("BRK-A" in message for message in fake.warnings)
