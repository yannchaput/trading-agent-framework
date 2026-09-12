from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.data.cache import CachedDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BacktestDataError

AAPL = Asset("AAPL")
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 10, tzinfo=UTC)


def test_first_load_fetches_from_the_inner_source_and_writes_a_parquet_file(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0, 101.0, 102.0], start=START))
    cache = CachedDataSource(inner, tmp_path)

    cache.load([AAPL], START, END, "day")

    files = list((tmp_path / "fake").glob("AAPL_day_*.parquet"))
    assert len(files) == 1
    assert len(inner.load_calls) == 1


def test_second_load_of_the_same_window_does_not_touch_the_inner_source_again(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0, 101.0], start=START))
    cache = CachedDataSource(inner, tmp_path)

    cache.load([AAPL], START, END, "day")
    cache.load([AAPL], START, END, "day")

    assert len(inner.load_calls) == 1  # second call was a cache hit


def test_bars_reads_from_the_cached_parquet_file(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0, 101.0, 102.0], start=START))
    cache = CachedDataSource(inner, tmp_path)
    cache.load([AAPL], START, END, "day")

    result = cache.bars(AAPL, END, 2, "day")

    assert result is not None
    assert list(result.df["close"]) == [101.0, 102.0]


def test_bars_without_a_prior_load_falls_through_to_the_inner_source(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0], start=START))
    cache = CachedDataSource(inner, tmp_path)

    result = cache.bars(AAPL, START, 1, "day")

    assert result is not None
    assert len(inner.bars_calls) == 1


def test_cache_writes_a_meta_json_sidecar_with_provenance(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0], start=START))
    cache = CachedDataSource(inner, tmp_path)

    cache.load([AAPL], START, END, "day")

    [meta_path] = list((tmp_path / "fake").glob("*.meta.json"))
    meta = json.loads(meta_path.read_text())
    assert meta["provider"] == "fake"
    assert meta["symbol"] == "AAPL"
    assert meta["rows"] == 1
    assert "fetched_at" in meta


def test_sessions_delegates_to_the_inner_source(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    cache = CachedDataSource(inner, tmp_path)
    assert cache.sessions(START, END) == inner.sessions(START, END)


def test_cache_name_matches_the_inner_sources_name(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    cache = CachedDataSource(inner, tmp_path)
    assert cache.name == "fake"


def test_bars_wraps_a_corrupt_cache_file_in_backtest_data_error(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0], start=START))
    cache = CachedDataSource(inner, tmp_path)
    cache.load([AAPL], START, END, "day")

    [parquet_path] = list((tmp_path / "fake").glob("*.parquet"))
    parquet_path.write_bytes(b"not a parquet file")

    with pytest.raises(BacktestDataError):
        cache.bars(AAPL, END, 1, "day")
