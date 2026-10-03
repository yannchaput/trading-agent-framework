from __future__ import annotations

import gc
import weakref
from datetime import datetime, timedelta

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.data.chunked import CHUNK, OVERLAP, YearChunkedData, chunk_count, chunk_index, chunk_window
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestDataError

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


A = Asset("AAA")
B = Asset("BBB")


class _Factory:
    """Builds a FakeBacktestDataSource per call, recording its window; keeps only weak references, so a test
    can check that a dropped chunk source is really freed. Every source serves `frame` for A at any timestep."""

    def __init__(self, frame=None) -> None:
        self.frame = frame
        self.windows: list[tuple[datetime, datetime]] = []
        self.refs: list[weakref.ref[FakeBacktestDataSource]] = []

    def __call__(self, start: datetime, end: datetime) -> FakeBacktestDataSource:
        source = FakeBacktestDataSource()
        if self.frame is not None:
            source.set_bars(A, self.frame)
        self.windows.append((start, end))
        self.refs.append(weakref.ref(source))
        return source

    def built(self, i: int) -> FakeBacktestDataSource:
        source = self.refs[i]()
        assert source is not None, f"source {i} was freed"
        return source


def test_a_short_window_serves_minutes_from_one_source_over_the_whole_window() -> None:
    end = START + timedelta(days=200)
    frame = make_close_indexed_frame([10.0, 11.0, 12.0], start=START + timedelta(days=1), freq="1min")
    factory = _Factory(frame)
    data = YearChunkedData(START, end, inner=factory)
    data.load([A], START, end, "minute")
    bars = data.bars(A, START + timedelta(days=2), 2, "minute")
    assert factory.windows == [(START, end), (START, end)]  # the whole-window source, then the minute one
    assert factory.built(1).load_windows == [(START, end)]
    assert bars is not None and list(bars.df["close"]) == [11.0, 12.0]


def test_a_minute_load_before_any_bars_call_targets_the_first_chunk() -> None:
    factory = _Factory()
    YearChunkedData(START, LONG_END, inner=factory).load([A], START, LONG_END, "minute")
    assert factory.windows[1] == chunk_window(0, START, LONG_END)


def test_a_minute_load_with_the_whole_window_loads_only_the_current_chunk() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.bars(A, START + CHUNK + timedelta(days=5), 70, "day")  # the scanner's daily read moves the position
    data.load([A, B], START, LONG_END, "minute")
    window = chunk_window(1, START, LONG_END)
    assert factory.windows[1:] == [window]
    assert factory.built(1).load_calls == [(A, B)]
    assert factory.built(1).load_windows == [window]
    assert factory.built(0).load_calls == []  # nothing reached the whole-window source


def test_a_cutoff_in_the_next_year_replaces_the_minute_source_and_frees_the_previous_one() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.bars(A, START + timedelta(days=10), 5, "minute")
    data.bars(A, START + CHUNK + timedelta(days=1), 5, "minute")
    assert factory.windows[1:] == [chunk_window(0, START, LONG_END), chunk_window(1, START, LONG_END)]
    gc.collect()
    assert factory.refs[1]() is None  # chunk 0's frames are gone
    assert factory.built(2).bars_calls == [(A, START + CHUNK + timedelta(days=1), 5, "minute")]


def test_a_minute_bars_call_for_an_asset_never_loaded_falls_through_to_the_chunk_source() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.load([A], START, LONG_END, "minute")
    data.bars(B, START + timedelta(days=3), 5, "minute")
    assert factory.built(1).bars_calls == [(B, START + timedelta(days=3), 5, "minute")]
    assert factory.built(0).bars_calls == []


def test_day_bars_load_and_sessions_go_to_the_whole_window_source() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.load([A], START, LONG_END, "day")
    data.bars(A, START + 2 * CHUNK, 5, "day")
    assert data.sessions(START, LONG_END) == []
    assert factory.windows == [(START, LONG_END)]  # no minute source was ever built
    assert factory.built(0).load_windows == [(START, LONG_END)]
    assert factory.built(0).bars_calls == [(A, START + 2 * CHUNK, 5, "day")]


def test_a_day_read_past_the_end_leaves_minutes_on_the_last_chunk() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.bars(A, LONG_END + timedelta(days=1), 1, "day")  # the runner's final benchmark read
    data.load([A], START, LONG_END, "minute")
    data.bars(A, LONG_END, 5, "minute")
    assert factory.windows[1:] == [chunk_window(3, START, LONG_END)]


def test_a_minute_cutoff_in_the_overlap_is_served_and_one_before_it_raises() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.bars(A, START + CHUNK + timedelta(days=1), 5, "minute")
    data.bars(A, START + CHUNK - timedelta(days=10), 5, "minute")  # inside chunk 1's 30-day overlap
    assert len(factory.windows) == 2  # served by chunk 1, nothing rebuilt
    with pytest.raises(BacktestDataError, match="before"):
        data.bars(A, START + CHUNK - OVERLAP - timedelta(days=1), 5, "minute")


def test_the_name_mirrors_the_inner_source() -> None:
    assert YearChunkedData(START, LONG_END, inner=_Factory()).name == "fake"
