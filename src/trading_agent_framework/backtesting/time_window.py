"""Factory method returning a tuple for backend start and end time.

The factory provides prebuilt configurations.
"""

from datetime import datetime
from enum import StrEnum

from trading_agent_framework.utils.clock import MARKET_TZ


class PredefinedWindow(StrEnum):
    """Predefined time windows for backtesting periods.

    Each window has a `_latest` twin with the same start and the later `_LATEST_END`.
    """

    DECADE = "decade"
    DECADE_LATEST = "decade_latest"
    SEMI_DECADE = "5y"
    SEMI_DECADE_LATEST = "5y_latest"
    BI_YEAR = "2y"
    BI_YEAR_LATEST = "2y_latest"
    YEAR = "year"
    YEAR_LATEST = "year_latest"
    HALF_YEAR = "half_year"
    HALF_YEAR_LATEST = "half_year_latest"
    BI_MONTH = "bi-month"
    BI_MONTH_LATEST = "bi-month_latest"
    MONTH = "month"
    MONTH_LATEST = "month_latest"
    WEEK = "week"
    WEEK_LATEST = "week_latest"


_LATEST_SUFFIX = "_latest"
_END = datetime(2026, 10, 5, tzinfo=MARKET_TZ)
_LATEST_END = datetime(2026, 11, 14, tzinfo=MARKET_TZ)
_STARTS = {
    PredefinedWindow.DECADE: datetime(2016, 1, 1, tzinfo=MARKET_TZ),
    PredefinedWindow.SEMI_DECADE: datetime(2021, 9, 20, tzinfo=MARKET_TZ),
    PredefinedWindow.BI_YEAR: datetime(2024, 9, 2, tzinfo=MARKET_TZ),
    PredefinedWindow.YEAR: datetime(2025, 9, 29, tzinfo=MARKET_TZ),
    PredefinedWindow.HALF_YEAR: datetime(2026, 1, 1, tzinfo=MARKET_TZ),
    PredefinedWindow.BI_MONTH: datetime(2026, 7, 28, tzinfo=MARKET_TZ),
    PredefinedWindow.MONTH: datetime(2026, 8, 25, tzinfo=MARKET_TZ),
    PredefinedWindow.WEEK: datetime(2026, 9, 22, tzinfo=MARKET_TZ),
}


def backtest_window(window: PredefinedWindow) -> tuple[datetime, datetime]:
    """Resolve a predefined window to dates.

    Args:
        window: The predefined window.

    Returns:
        The (start, end) pair, both timezone-aware at midnight in the market timezone.

    Raises:
        ValueError: If the window has no start date.
    """
    base = PredefinedWindow(window.removesuffix(_LATEST_SUFFIX))
    start = _STARTS.get(base)
    if start is None:
        raise ValueError(f"Unknown window: {window}")
    return start, (_END if base == window else _LATEST_END)
