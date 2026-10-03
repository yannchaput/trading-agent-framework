from __future__ import annotations

from datetime import datetime, timedelta

from trading_agent_framework.backtesting.data.chunked import CHUNK, OVERLAP, chunk_count, chunk_index, chunk_window
from trading_agent_framework.utils.clock import MARKET_TZ

START = datetime(2021, 1, 4, tzinfo=MARKET_TZ)
LONG_END = START + 3 * CHUNK + timedelta(days=100)  # four chunks, the last one 100 days long


def test_a_window_of_one_year_or_less_is_one_chunk() -> None:
    end = START + CHUNK
    assert chunk_count(START, end) == 1
    assert chunk_window(0, START, end) == (START, end)
    assert chunk_index(START - timedelta(days=5), START, end) == 0
    assert chunk_index(end + timedelta(days=5), START, end) == 0


def test_a_longer_window_is_split_year_by_year() -> None:
    assert chunk_count(START, LONG_END) == 4
    assert chunk_window(0, START, LONG_END) == (START, START + CHUNK)  # never fetched before `start`
    assert chunk_window(1, START, LONG_END) == (START + CHUNK - OVERLAP, START + 2 * CHUNK)
    assert chunk_window(3, START, LONG_END) == (START + 3 * CHUNK - OVERLAP, LONG_END)  # the last ends exactly at `end`


def test_chunk_index_is_clamped_and_a_boundary_belongs_to_the_next_chunk() -> None:
    assert chunk_index(START - timedelta(days=1), START, LONG_END) == 0
    assert chunk_index(START + CHUNK - timedelta(seconds=1), START, LONG_END) == 0
    assert chunk_index(START + CHUNK, START, LONG_END) == 1
    assert chunk_index(LONG_END + timedelta(days=10), START, LONG_END) == 3
