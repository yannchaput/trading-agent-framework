"""
Factory method returning a tuple for backend start and end time.
The factory provides prebuilt configurations.
"""

from datetime import datetime
from enum import StrEnum

from trading_agent_framework.utils.clock import MARKET_TZ


class PredefinedWindow(StrEnum):
    """
    Backtesting periods predefined time Windows
    """

    DECADE = "decade"
    SEMI_DECADE = "5y"
    YEAR = "year"
    HALF_YEAR = "half_year"
    BI_MONTH = "bi-month"
    MONTH = "month"
    WEEK = "week"


def backtest_window(window: PredefinedWindow) -> tuple[datetime, datetime]:
    match window:
        case PredefinedWindow.DECADE:
            return (datetime(2016, 1, 1, tzinfo=MARKET_TZ), datetime(2026, 9, 23, tzinfo=MARKET_TZ))
        case PredefinedWindow.SEMI_DECADE:
            return (datetime(2021, 9, 20, tzinfo=MARKET_TZ), datetime(2026, 9, 23, tzinfo=MARKET_TZ))
        case PredefinedWindow.YEAR:
            return (datetime(2025, 9, 29, tzinfo=MARKET_TZ), datetime(2026, 9, 23, tzinfo=MARKET_TZ))
        case PredefinedWindow.HALF_YEAR:
            return (datetime(2026, 1, 1, tzinfo=MARKET_TZ), datetime(2026, 9, 23, tzinfo=MARKET_TZ))
        case PredefinedWindow.BI_MONTH:
            return (datetime(2026, 7, 28, tzinfo=MARKET_TZ), datetime(2026, 9, 23, tzinfo=MARKET_TZ))
        case PredefinedWindow.MONTH:
            return (datetime(2026, 8, 25, tzinfo=MARKET_TZ), datetime(2026, 9, 23, tzinfo=MARKET_TZ))
        case PredefinedWindow.WEEK:
            return (datetime(2026, 9, 22, tzinfo=MARKET_TZ), datetime(2026, 9, 26, tzinfo=MARKET_TZ))
        case _:
            raise ValueError(f"Unknown window: {window}")
