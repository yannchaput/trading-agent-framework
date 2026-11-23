from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest
from tests.fakes import FakeBroker, FakeClock, FakeRegimeSource, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.regime_data import RegimeSeries
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import RegimeDataError

ROWS = 503


def _series(spy: list[float], vix: float = 15.0, vix3m: float = 18.0, rows: int = ROWS, vixy: list[float] | None = None) -> RegimeSeries:
    """`rows` sessions ending 2026-09-11 (the Friday before the clock's date), weekdays only."""
    sessions: list[date] = []
    day = date(2026, 9, 11)
    while len(sessions) < rows:
        if day.weekday() < 5:
            sessions.append(day)
        day -= timedelta(days=1)
    sessions.reverse()
    closes = spy[-rows:]
    return RegimeSeries(tuple(sessions), tuple(closes), tuple([vix] * rows), tuple([vix3m] * rows), tuple(vixy or ()))


BULL = _series([100 * 1.001**i for i in range(ROWS)])
# A fall from the 52w high with an inverted term structure: SPY below its SMA50 and off its high, VIX above VIX3M.
BEAR = _series([100 * 1.001**i for i in range(300)] + [100 * 1.001**300 * 0.995 ** (i + 1) for i in range(ROWS - 300)], vix=30.0, vix3m=25.0)


CALM_SPY = [100 * 1.001**i for i in range(ROWS)]


def _strategy(series: RegimeSeries | None = None, cls: type[Strategy] = Strategy, clock_at=None) -> tuple[Strategy, FakeRegimeSource, FakeBroker, list[tuple]]:
    broker = FakeBroker(FakeClock(clock_at or et(2026, 9, 14, 8, 30)))
    strategy = cls(broker, mode=TradingMode.BACKTESTING)
    source = FakeRegimeSource(series)
    strategy.regime_source = source
    lines: list[tuple] = []
    strategy.add_line = lambda name, value, **kw: lines.append((name, value, kw["plot_name"]))  # ty: ignore[invalid-assignment]
    return strategy, source, broker, lines


def test_the_regime_is_unknown_before_the_first_refresh() -> None:
    strategy, source, _, _ = _strategy(BULL)
    assert strategy.regime is None
    assert source.calls == []


@pytest.mark.parametrize("clock", [et(2026, 9, 14, 8, 30), et(2026, 9, 14, 10, 15)])
def test_a_refresh_reads_503_closes_strictly_before_the_session_date(clock) -> None:
    strategy, source, _, _ = _strategy(BULL, clock_at=clock)
    strategy._refresh_regime()
    assert source.calls == [(date(2026, 9, 14), 503)]


def test_a_refresh_sets_logs_and_charts_the_regime(caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, _, lines = _strategy(BEAR)
    with caplog.at_level(logging.INFO):
        strategy._refresh_regime()
    assert strategy.regime == -1
    assert lines == [("Regime", -1, "Regime")]
    for text in ("Market regime -1 (bearish)", "score ", "VIX3M", "SMA50", "-session high", "episode low"):
        assert text in caplog.text


def test_the_regime_never_reads_the_strategys_own_bars() -> None:
    strategy, _, broker, _ = _strategy(BULL)
    strategy._refresh_regime()
    assert broker.bars_calls == []


def test_the_regime_ignores_benchmark_symbol(caplog: pytest.LogCaptureFixture) -> None:
    class Qqq(Strategy):
        benchmark_symbol = "QQQ"

    strategy, source, _, _ = _strategy(BULL, cls=Qqq)
    with caplog.at_level(logging.INFO):
        strategy._refresh_regime()
    assert len(source.calls) == 1
    assert "SPY" in caplog.text and "QQQ" not in caplog.text


@pytest.mark.parametrize("error", [RegimeDataError("yahoo down"), ValueError("bad close")])
def test_a_data_failure_keeps_the_previous_regime(error: Exception, caplog: pytest.LogCaptureFixture) -> None:
    strategy, source, _, lines = _strategy(BEAR)
    strategy._refresh_regime()
    assert strategy.regime == -1
    source.error = error
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
    assert strategy.regime == -1
    assert len(lines) == 1  # nothing charted for the failed refresh
    assert "Market regime not refreshed (still -1)" in caplog.text


@pytest.mark.parametrize("series", [None, _series([100.0] * 100, rows=100)])
def test_too_little_history_leaves_the_regime_unknown_and_warns_once(series: RegimeSeries | None, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, _, lines = _strategy(series)
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
        strategy._refresh_regime()
    assert strategy.regime is None and lines == []
    assert caplog.text.count("Market regime unavailable") == 1
    assert "SPY/VIX/VIX3M" in caplog.text


def test_outside_backtesting_the_regime_is_set_and_nothing_is_charted() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 8, 30)))
    strategy = Strategy(broker, mode=TradingMode.PAPER)
    strategy.regime_source = FakeRegimeSource(BULL)
    strategy._refresh_regime()  # the real add_line: a no-op outside backtesting
    assert strategy.regime == 1


def test_a_refresh_sets_the_reading_and_the_market_fall_flag() -> None:
    strategy, _, _, _ = _strategy(BEAR)
    strategy._refresh_regime()
    assert strategy.market_fall is True
    assert strategy.regime_reading is not None and strategy.regime_reading.regime == strategy.regime


def test_a_calm_market_has_no_fall() -> None:  # BULL has no VIXY column: the ratio leg only
    strategy, _, _, _ = _strategy(BULL)
    strategy._refresh_regime()
    assert strategy.market_fall is False


def test_a_vixy_jump_alone_turns_the_flag_on() -> None:
    vixy = [20.0] * (ROWS - 1) + [22.5]  # +12.5% over 5 sessions, ratio calm
    strategy, _, _, _ = _strategy(_series(CALM_SPY, vixy=vixy))
    strategy._refresh_regime()
    assert strategy.market_fall is True


def test_a_missing_vixy_column_is_all_nan_for_the_rule() -> None:
    strategy, _, _, _ = _strategy(_series(CALM_SPY, vixy=[float("nan")] * ROWS))
    strategy._refresh_regime()
    assert strategy.market_fall is False


def test_the_market_fall_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, _, _ = _strategy(BEAR)
    with caplog.at_level(logging.INFO):
        strategy._refresh_regime()
    assert "Market fall warning ON" in caplog.text
    assert "VIXY" not in caplog.text.split("Market fall warning")[1]  # no VIXY data: that part is omitted


def test_the_vixy_change_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    vixy = [20.0] * (ROWS - 1) + [22.5]
    strategy, _, _, _ = _strategy(_series(CALM_SPY, vixy=vixy))
    with caplog.at_level(logging.INFO):
        strategy._refresh_regime()
    assert "VIXY +12.5% over 5 sessions" in caplog.text


def test_a_data_failure_keeps_the_previous_flag_and_reading() -> None:
    strategy, source, _, _ = _strategy(BEAR)
    strategy._refresh_regime()
    first_reading = strategy.regime_reading
    source.error = RegimeDataError("x")
    strategy._refresh_regime()
    assert strategy.market_fall is True and strategy.regime_reading is first_reading


def test_too_little_history_leaves_the_flag_unknown() -> None:
    strategy, _, _, _ = _strategy(_series([100.0] * 100, rows=100))
    strategy._refresh_regime()
    assert strategy.market_fall is None and strategy.regime_reading is None


def test_a_zero_vixy_close_counts_as_missing_and_the_regime_still_refreshes(caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, _, _ = _strategy(_series(CALM_SPY, vixy=[20.0] * (ROWS - 1) + [0.0]))
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
    assert strategy.regime is not None and strategy.regime_reading is not None
    assert strategy.market_fall is False
    assert "Market regime not refreshed" not in caplog.text


def test_a_zero_vixy_close_next_to_a_real_jump_does_not_raise() -> None:
    vixy = [20.0] * (ROWS - 1) + [22.5]
    vixy[10] = 0.0
    strategy, _, _, _ = _strategy(_series(CALM_SPY, vixy=vixy))
    strategy._refresh_regime()
    assert strategy.market_fall is True


@pytest.mark.parametrize("vix, vix3m", [(0.0, 18.0), (15.0, 0.0)])
def test_a_zero_vix_or_vix3m_close_is_a_warning_and_keeps_the_previous_values(vix: float, vix3m: float, caplog: pytest.LogCaptureFixture) -> None:
    strategy, source, _, _ = _strategy(BEAR)
    strategy._refresh_regime()
    before = (strategy.regime, strategy.regime_reading, strategy.market_fall)
    source.series = _series(CALM_SPY, vix=vix, vix3m=vix3m)
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
    assert (strategy.regime, strategy.regime_reading, strategy.market_fall) == before
    assert "Market regime not refreshed" in caplog.text
