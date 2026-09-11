from __future__ import annotations

from datetime import timedelta

import pytest
from tests.fakes import et

from trading_agent_framework.core.timing import SleepTime, next_tick, parse_sleeptime
from trading_agent_framework.errors import ConfigurationError


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5, SleepTime(interval=timedelta(minutes=5))),
        ("30S", SleepTime(interval=timedelta(seconds=30))),
        ("30s", SleepTime(interval=timedelta(seconds=30))),
        ("5M", SleepTime(interval=timedelta(minutes=5))),
        ("5m", SleepTime(interval=timedelta(minutes=5))),
        ("2T", SleepTime(interval=timedelta(minutes=2))),
        ("2H", SleepTime(interval=timedelta(hours=2))),
        (" 3h ", SleepTime(interval=timedelta(hours=3))),
        ("1D", SleepTime(sessions=1)),
        ("2d", SleepTime(sessions=2)),
    ],
)
def test_parse_sleeptime_valid(value: int | str, expected: SleepTime) -> None:
    assert parse_sleeptime(value) == expected


@pytest.mark.parametrize(
    "value", ["", "M", "5", "5X", "1.5H", "-5M", "0M", "5 M M", 0, -1, True, 5.0, None]
)
def test_parse_sleeptime_invalid(value: object) -> None:
    with pytest.raises(ConfigurationError):
        parse_sleeptime(value)  # ty: ignore[invalid-argument-type]


def test_sleeptime_requires_exactly_one_field() -> None:
    with pytest.raises(ValueError):
        SleepTime()
    with pytest.raises(ValueError):
        SleepTime(interval=timedelta(minutes=1), sessions=1)


def test_next_tick_without_overrun() -> None:
    tick, skipped = next_tick(et(2026, 9, 14, 9, 30), timedelta(minutes=5), et(2026, 9, 14, 9, 31))
    assert tick == et(2026, 9, 14, 9, 35)
    assert skipped == 0


def test_next_tick_counts_skipped_ticks_after_overrun() -> None:
    tick, skipped = next_tick(et(2026, 9, 14, 9, 30), timedelta(minutes=5), et(2026, 9, 14, 9, 43))
    assert tick == et(2026, 9, 14, 9, 45)
    assert skipped == 2  # 09:35 and 09:40 were missed


def test_next_tick_is_strictly_after_now() -> None:
    tick, skipped = next_tick(et(2026, 9, 14, 9, 30), timedelta(minutes=5), et(2026, 9, 14, 9, 35))
    assert tick == et(2026, 9, 14, 9, 40)
    assert skipped == 1


def test_next_tick_when_now_is_before_last_tick() -> None:
    tick, skipped = next_tick(et(2026, 9, 14, 9, 30), timedelta(minutes=5), et(2026, 9, 14, 9, 0))
    assert tick == et(2026, 9, 14, 9, 35)
    assert skipped == 0
