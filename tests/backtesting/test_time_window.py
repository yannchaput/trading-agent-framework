"""Tests for the predefined backtest windows."""

from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window


def test_every_latest_window_keeps_its_base_start_and_ends_later() -> None:
    """A `_latest` twin differs from its base window only by a later end."""
    latest = [w for w in PredefinedWindow if w.endswith("_latest")]
    assert len(latest) == len(PredefinedWindow) // 2
    for window in latest:
        base_start, base_end = backtest_window(PredefinedWindow(window.removesuffix("_latest")))
        start, end = backtest_window(window)
        assert start == base_start
        assert end > base_end
        assert start.tzinfo is not None and end.tzinfo is not None
