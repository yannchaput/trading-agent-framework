"""`CrossMomentumStrategy._compute_indicators_for_ticker` error isolation.

A universe-wide `compute_target_portfolio()` pass must not fail entirely
because one ticker's broker request errors out (e.g. a symbol Alpaca
rejects) -- that ticker should be skipped like any other filtered-out
ticker, not blow up the whole rebalance.
"""

from datetime import date, datetime, time
from types import SimpleNamespace

import pandas as pd

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BrokerError


class FakeStrategy:
    """Just enough of `Strategy` for `_compute_indicators_for_ticker`."""

    def __init__(self, *, bars=None, raises=None, now=None):
        self.parameters = {"min_trading_days": 250, "skip_days": 21, "volatility_window": 20}
        self.vars = SimpleNamespace(alpaca_rate_limiter=SimpleNamespace(wait=lambda: None), filter_source=None, filter_inputs={})
        self._bars = bars
        self._raises = raises
        self._now = now or datetime(2026, 10, 6, 12, 0, tzinfo=MARKET_TZ)
        self.requested_lengths: list[int] = []
        self.warnings: list[str] = []

    def get_historical_prices(self, ticker, length, timestep):
        self.requested_lengths.append(length)
        if self._raises is not None:
            raise self._raises
        return self._bars

    def get_datetime(self):
        return self._now

    _market_date = CrossMomentumStrategy._market_date

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


def _daily_bars(sessions: int, last: date) -> Bars:
    """`sessions` midnight-stamped daily bars ending on `last`; the last one is a -50% crash."""
    dates = [d.date() for d in pd.bdate_range(end=last, periods=sessions)]
    closes = [100.0 + i for i in range(sessions)]
    closes[-1] = closes[-2] * 0.5
    index = pd.DatetimeIndex([datetime.combine(d, time(0), tzinfo=MARKET_TZ) for d in dates])
    frame = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1e6] * sessions}, index=index)
    return Bars(Asset("AAA"), "day", frame)


def test_the_scan_ignores_todays_partial_bar():
    bars = _daily_bars(301, date(2026, 10, 6))  # 300 completed sessions + Tuesday's crash, still forming
    fake = FakeStrategy(bars=bars)

    result = CrossMomentumStrategy._compute_indicators_for_ticker(fake, "AAA")

    assert fake.requested_lengths == [301]
    assert result is not None
    assert result["trading_days"] == 300
    assert result["price"] == bars.df["close"].iloc[-2]  # Monday's close, not the crash
    assert result["close_series"].index[-1] == date(2026, 10, 5)


def test_the_scan_keeps_300_bars_when_today_has_no_bar_yet():
    fake = FakeStrategy(bars=_daily_bars(301, date(2026, 10, 5)))  # a bar per session through Monday

    result = CrossMomentumStrategy._compute_indicators_for_ticker(fake, "AAA")

    assert result is not None
    assert result["trading_days"] == 300
    assert result["close_series"].index[-1] == date(2026, 10, 5)
