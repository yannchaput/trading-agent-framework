from __future__ import annotations

import logging

import pytest
from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.regime import RegimeParameters
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import BacktestDataError, BrokerError

UPTREND = [100 * 1.001**i for i in range(300)]
DOWNTREND = [100 * 0.999**i for i in range(300)]


def _strategy(closes: list[float] | None = None, cls: type[Strategy] = Strategy) -> tuple[Strategy, FakeBroker, list[tuple]]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 8, 30)))
    if closes is not None:
        broker.bar_frames["SPY"] = make_bars_frame(closes, start=et(2025, 1, 6))
    strategy = cls(broker, mode=TradingMode.BACKTESTING)
    lines: list[tuple] = []
    strategy.add_line = lambda name, value, **kw: lines.append((name, value, kw["plot_name"]))  # ty: ignore[invalid-assignment]
    return strategy, broker, lines


def test_the_regime_is_unknown_before_the_first_refresh() -> None:
    strategy, _, _ = _strategy(UPTREND)
    assert strategy.regime is None


def test_a_refresh_sets_logs_and_charts_the_regime(caplog: pytest.LogCaptureFixture) -> None:
    strategy, broker, lines = _strategy(UPTREND)
    with caplog.at_level(logging.INFO):
        strategy._refresh_regime()
    assert strategy.regime == 1
    assert lines == [("Regime", 1, "Regime")]
    assert broker.bars_calls == [(("SPY",), 273, "day", True)]
    assert "Market regime +1 (bullish)" in caplog.text


def test_a_downtrend_is_logged_as_minus_one() -> None:
    strategy, _, lines = _strategy(DOWNTREND)
    strategy._refresh_regime()
    assert strategy.regime == -1
    assert lines == [("Regime", -1, "Regime")]


def test_the_benchmark_and_the_parameters_come_from_the_class() -> None:
    class Small(Strategy):
        benchmark_symbol = "QQQ"
        regime_params = RegimeParameters(sma_fast=3, sma_slow=5, vol_window=2, vol_lookback=4)

    strategy, broker, _ = _strategy(cls=Small)
    broker.bar_frames["QQQ"] = make_bars_frame([7, 6, 5, 4, 3, 2, 1], start=et(2026, 9, 1))
    strategy._refresh_regime()
    assert strategy.regime == -1
    assert broker.bars_calls == [(("QQQ",), 7, "day", True)]


@pytest.mark.parametrize("error", [BrokerError("data down"), BacktestDataError("no session")])
def test_a_data_failure_keeps_the_previous_regime(error: Exception, caplog: pytest.LogCaptureFixture) -> None:
    strategy, broker, lines = _strategy(UPTREND)
    strategy._refresh_regime()
    broker.market_data_error = error  # ty: ignore[invalid-assignment]
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
    assert strategy.regime == 1
    assert len(lines) == 1  # nothing charted for the failed refresh
    assert "Market regime not refreshed" in caplog.text


@pytest.mark.parametrize("closes", [None, UPTREND[:100]])
def test_too_little_history_leaves_the_regime_unknown_and_warns_once(closes: list[float] | None, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, lines = _strategy(closes)
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
        strategy._refresh_regime()
    assert strategy.regime is None and lines == []
    assert caplog.text.count("Market regime unavailable") == 1


def test_a_corrupt_close_keeps_the_previous_regime(caplog: pytest.LogCaptureFixture) -> None:
    strategy, broker, lines = _strategy(UPTREND)
    strategy._refresh_regime()
    broker.bar_frames["SPY"] = make_bars_frame([*UPTREND[:-1], 0.0], start=et(2025, 1, 6))
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
    assert strategy.regime == 1 and len(lines) == 1
    assert "Market regime not refreshed" in caplog.text


def test_outside_backtesting_the_regime_is_set_and_nothing_is_charted() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 8, 30)))
    broker.bar_frames["SPY"] = make_bars_frame(UPTREND, start=et(2025, 1, 6))
    strategy = Strategy(broker, mode=TradingMode.PAPER)
    strategy._refresh_regime()  # the real add_line: a no-op outside backtesting
    assert strategy.regime == 1
