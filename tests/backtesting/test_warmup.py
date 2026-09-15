"""Tests for warmup_calendar_days() pure helper."""

import pytest

from trading_agent_framework.backtesting.warmup import (
    DEFAULT_HOLIDAY_BUFFER_DAYS,
    warmup_calendar_days,
)


class TestWarmupCalendarDays:
    """Test warmup_calendar_days() pure helper function."""

    def test_zero_trading_days_returns_zero_regardless_of_buffer(self):
        """warmup_calendar_days(0) == 0 regardless of holiday_buffer_days."""
        assert warmup_calendar_days(0) == 0
        assert warmup_calendar_days(0, holiday_buffer_days=99) == 0
        assert warmup_calendar_days(0, holiday_buffer_days=0) == 0

    def test_worked_example_300_trading_days(self):
        """Worked example: warmup_calendar_days(300) == 435.

        ceil(300*7/5) + 15 = 420 + 15 = 435
        """
        assert warmup_calendar_days(300) == 435

    def test_holiday_buffer_days_is_configurable_and_additive(self):
        """holiday_buffer_days is configurable and additive.

        warmup_calendar_days(300, holiday_buffer_days=0) == 420
        """
        assert warmup_calendar_days(300, holiday_buffer_days=0) == 420
        assert warmup_calendar_days(300, holiday_buffer_days=15) == 435
        assert warmup_calendar_days(300, holiday_buffer_days=30) == 450

    def test_monotonically_non_decreasing(self):
        """Results are monotonically non-decreasing across the sequence."""
        trading_days_list = [0, 1, 10, 50, 300, 1000]
        results = [warmup_calendar_days(td) for td in trading_days_list]
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1]

    def test_negative_trading_days_raises_value_error(self):
        """warmup_calendar_days(-1) raises ValueError matching 'trading_days must be >= 0'."""
        with pytest.raises(ValueError, match="trading_days must be >= 0"):
            warmup_calendar_days(-1)
