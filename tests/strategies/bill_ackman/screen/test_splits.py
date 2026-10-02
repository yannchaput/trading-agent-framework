from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.strategies.bill_ackman.screen.splits import Split, SplitHistory, restate_shares, split_rows
from trading_agent_framework.utils.errors import FundamentalsError

FETCHED = datetime(2026, 10, 2, tzinfo=UTC)
FOUR_FOR_ONE = [(date(2020, 8, 31), 4.0)]


class FakeFetch:
    """Stands in for the Yahoo lookup: counts calls, and fails while `error` is set."""

    def __init__(self, splits: list[Split]) -> None:
        self.splits = splits
        self.calls: list[str] = []
        self.error: FundamentalsError | None = None

    def __call__(self, symbol: str) -> list[Split]:
        self.calls.append(symbol)
        if self.error is not None:
            raise self.error
        return list(self.splits)


class Clock:
    """A settable wall clock: split freshness is judged by it, never by a screen date."""

    def __init__(self, now: datetime = FETCHED) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _history(cache_file: Path, fetch: FakeFetch, clock: Clock | None = None) -> SplitHistory:
    return SplitHistory(cache_file, fetch=fetch, wall_clock=clock or Clock())


def _splits(history: SplitHistory, max_age_days: int = 30) -> list[Split]:
    return history.splits("AAPL", max_age_days=max_age_days)


def test_is_stale_only_when_fetched_more_than_max_age_before_as_of() -> None:
    assert is_stale(FETCHED, datetime(2026, 11, 2, tzinfo=UTC), 30) is True
    assert is_stale(FETCHED, datetime(2026, 11, 1, tzinfo=UTC), 30) is False


def test_a_file_fetched_today_is_fresh_for_every_past_date() -> None:
    assert is_stale(FETCHED, datetime(2020, 1, 1, tzinfo=UTC), 30) is False


def test_restate_shares_without_splits_is_unchanged() -> None:
    assert restate_shares(1000, date(2026, 1, 31), []) == Decimal(1000)


def test_restate_shares_applies_a_split_dated_after_the_count() -> None:
    assert restate_shares(1000, date(2020, 7, 17), FOUR_FOR_ONE) == Decimal(4000)


def test_restate_shares_ignores_a_split_on_or_before_the_count_date() -> None:
    assert restate_shares(1000, date(2020, 8, 31), FOUR_FOR_ONE) == Decimal(1000)
    assert restate_shares(1000, date(2021, 1, 1), FOUR_FOR_ONE) == Decimal(1000)


def test_restate_shares_compounds_several_splits_including_a_reverse_split() -> None:
    splits = [(date(2014, 6, 9), 7.0), (date(2020, 8, 31), 4.0), (date(2023, 5, 1), 0.1)]

    assert restate_shares(1000, date(2015, 1, 1), splits) == Decimal("400.0")


def test_split_rows_keeps_the_non_zero_rows_of_a_yahoo_history() -> None:
    index = pd.to_datetime(["2020-08-28", "2020-08-31", "2020-09-01"]).tz_localize("America/New_York")
    frame = pd.DataFrame({"Close": [499.0, 129.0, 134.0], "Stock Splits": [0.0, 4.0, 0.0]}, index=index)

    assert split_rows(frame) == [(date(2020, 8, 31), 4.0)]


def test_split_rows_of_a_history_with_no_split_is_empty() -> None:
    index = pd.to_datetime(["2026-01-02"]).tz_localize("America/New_York")
    frame = pd.DataFrame({"Close": [10.0], "Stock Splits": [0.0]}, index=index)

    assert split_rows(frame) == []


def test_split_rows_rejects_an_empty_history_as_a_failed_lookup() -> None:
    # yfinance reports a failed download as an empty frame; that must not read as "never split".
    with pytest.raises(FundamentalsError, match="no history"):
        split_rows(pd.DataFrame({"Close": []}))


def test_split_history_fetches_once_and_caches_on_disk(tmp_path: Path) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    cache_file = tmp_path / "splits.json"

    assert _splits(_history(cache_file, fetch)) == FOUR_FOR_ONE
    assert _splits(_history(cache_file, fetch)) == FOUR_FOR_ONE  # a new instance reads the file
    assert fetch.calls == ["AAPL"]


def test_split_history_caches_an_empty_history(tmp_path: Path) -> None:
    fetch = FakeFetch([])
    history = _history(tmp_path / "splits.json", fetch)

    assert _splits(history) == []
    assert _splits(history) == []
    assert fetch.calls == ["AAPL"]


def test_split_history_ignores_the_screen_date_and_judges_freshness_by_the_wall_clock(tmp_path: Path) -> None:
    # Split history must match today's split-adjusted prices, a wall-clock fact: an entry fetched
    # 2026-10-01 is stale on 2027-03-01 whatever date a backtest is simulating.
    fetch = FakeFetch(FOUR_FOR_ONE)
    clock = Clock(datetime(2026, 10, 1, tzinfo=UTC))
    history = _history(tmp_path / "splits.json", fetch, clock)
    _splits(history, max_age_days=1)

    clock.now = datetime(2027, 3, 1, tzinfo=UTC)
    _splits(history, max_age_days=1)

    assert fetch.calls == ["AAPL", "AAPL"]


def test_split_history_serves_an_entry_fetched_within_max_age_by_the_wall_clock(tmp_path: Path) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    clock = Clock(datetime(2026, 10, 1, 12, tzinfo=UTC))
    history = _history(tmp_path / "splits.json", fetch, clock)
    _splits(history, max_age_days=1)

    clock.now = datetime(2026, 10, 2, 11, tzinfo=UTC)  # 23 hours later
    _splits(history, max_age_days=1)

    assert fetch.calls == ["AAPL"]


def test_split_history_refetches_a_stale_entry(tmp_path: Path) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    clock = Clock()
    history = _history(tmp_path / "splits.json", fetch, clock)

    _splits(history)
    clock.now = datetime(2026, 11, 15, tzinfo=UTC)
    _splits(history)

    assert fetch.calls == ["AAPL", "AAPL"]


def test_a_failed_lookup_with_no_cached_copy_raises_and_caches_nothing(tmp_path: Path) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    fetch.error = FundamentalsError("yahoo is down")
    history = _history(tmp_path / "splits.json", fetch)

    with pytest.raises(FundamentalsError, match="yahoo is down"):
        _splits(history)

    fetch.error = None
    assert _splits(history) == FOUR_FOR_ONE


def test_a_failed_refresh_keeps_the_stale_copy_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    clock = Clock()
    history = _history(tmp_path / "splits.json", fetch, clock)
    _splits(history)
    fetch.error = FundamentalsError("yahoo is down")
    clock.now = datetime(2026, 11, 15, tzinfo=UTC)

    with caplog.at_level(logging.WARNING):
        splits = _splits(history)

    assert splits == FOUR_FOR_ONE
    assert "yahoo is down" in caplog.text


def test_a_corrupt_cache_file_is_treated_as_empty(tmp_path: Path) -> None:
    cache_file = tmp_path / "splits.json"
    cache_file.write_text("{not json", encoding="utf-8")
    fetch = FakeFetch(FOUR_FOR_ONE)

    assert _splits(_history(cache_file, fetch)) == FOUR_FOR_ONE
    assert fetch.calls == ["AAPL"]


def _write_cache(cache_file: Path, entry: object) -> None:
    cache_file.write_text(json.dumps({"AAPL": entry}), encoding="utf-8")


GOOD_ENTRY = {"fetched_at": FETCHED.isoformat(), "splits": [["2020-08-31", 4.0]]}


def test_a_well_formed_cache_entry_is_served_without_a_fetch(tmp_path: Path) -> None:
    cache_file = tmp_path / "splits.json"
    _write_cache(cache_file, GOOD_ENTRY)
    fetch = FakeFetch([])

    assert _splits(_history(cache_file, fetch)) == FOUR_FOR_ONE
    assert fetch.calls == []


@pytest.mark.parametrize(
    "entry",
    [
        "not a dict",
        None,
        [],
        {"splits": [["2020-08-31", 4.0]]},
        {**GOOD_ENTRY, "fetched_at": "yesterday"},
        {**GOOD_ENTRY, "fetched_at": "2026-10-02T00:00:00"},
        {**GOOD_ENTRY, "fetched_at": 20261002},
        {"fetched_at": FETCHED.isoformat()},
        {**GOOD_ENTRY, "splits": "none"},
        {**GOOD_ENTRY, "splits": [["not-a-date", 4.0]]},
        {**GOOD_ENTRY, "splits": [[20200831, 4.0]]},
        {**GOOD_ENTRY, "splits": [["2020-08-31", "four"]]},
        {**GOOD_ENTRY, "splits": [["2020-08-31", None]]},
        {**GOOD_ENTRY, "splits": [["2020-08-31", 0]]},
        {**GOOD_ENTRY, "splits": [["2020-08-31", -2.0]]},
        {**GOOD_ENTRY, "splits": [["2020-08-31", float("nan")]]},
        {**GOOD_ENTRY, "splits": [["2020-08-31", float("inf")]]},
        {**GOOD_ENTRY, "splits": [["2020-08-31"]]},
        {**GOOD_ENTRY, "splits": ["2020-08-31"]},
    ],
    ids=[
        "string", "null", "list", "no-fetched-at", "bad-fetched-at", "naive-fetched-at", "int-fetched-at", "no-splits", "splits-not-a-list",
        "bad-date", "int-date", "text-ratio", "null-ratio", "zero-ratio", "negative-ratio", "nan-ratio", "inf-ratio", "short-row", "row-not-a-pair",
    ],
)
def test_a_malformed_cache_entry_is_a_cache_miss_never_a_raw_exception(tmp_path: Path, entry: object) -> None:
    cache_file = tmp_path / "splits.json"
    _write_cache(cache_file, entry)
    fetch = FakeFetch(FOUR_FOR_ONE)

    assert _splits(_history(cache_file, fetch)) == FOUR_FOR_ONE
    assert fetch.calls == ["AAPL"]


def test_a_malformed_entry_does_not_poison_the_others(tmp_path: Path) -> None:
    cache_file = tmp_path / "splits.json"
    cache_file.write_text(json.dumps({"AAPL": "junk", "MSFT": {**GOOD_ENTRY, "splits": []}}), encoding="utf-8")
    fetch = FakeFetch(FOUR_FOR_ONE)
    history = _history(cache_file, fetch)

    assert history.splits("MSFT", max_age_days=30) == []
    assert fetch.calls == []


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), float("-inf"), 0.0, -2.0])
def test_split_rows_ignores_a_row_whose_ratio_is_not_finite_and_positive(ratio: float) -> None:
    index = pd.to_datetime(["2020-08-28", "2020-08-31", "2020-09-01"]).tz_localize("America/New_York")
    frame = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Stock Splits": [ratio, 4.0, 0.0]}, index=index)

    assert split_rows(frame) == [(date(2020, 8, 31), 4.0)]


def test_a_save_uses_a_unique_temp_file_and_leaves_no_temp_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_file = tmp_path / "cache" / "splits.json"
    replaced: list[str] = []
    real_replace = os.replace

    def spy(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        replaced.append(os.fspath(source))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", spy)
    history = _history(cache_file, FakeFetch(FOUR_FOR_ONE))

    history.splits("AAPL", max_age_days=30)
    history.splits("MSFT", max_age_days=30)

    assert len(set(replaced)) == 2  # a fresh name per write, not a fixed splits.tmp
    assert all(Path(name).name.startswith("splits.") and name.endswith(".tmp") for name in replaced)
    assert all(not Path(name).exists() for name in replaced)
    assert list(cache_file.parent.glob("*.tmp")) == []


def test_a_failed_save_warns_and_still_serves_the_fetched_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    cache_file = tmp_path / "cache" / "splits.json"

    def broken(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    fetch = FakeFetch(FOUR_FOR_ONE)
    history = _history(cache_file, fetch)

    with caplog.at_level(logging.WARNING):
        assert _splits(history) == FOUR_FOR_ONE
        assert _splits(history) == FOUR_FOR_ONE  # kept in memory: no second fetch

    assert fetch.calls == ["AAPL"]
    assert "disk full" in caplog.text
    assert list(cache_file.parent.glob("*.tmp")) == []
    assert not cache_file.exists()


def test_an_unwritable_cache_directory_never_escapes(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the cache directory should be", encoding="utf-8")
    history = _history(blocker / "splits.json", FakeFetch(FOUR_FOR_ONE))

    with caplog.at_level(logging.WARNING):
        assert _splits(history) == FOUR_FOR_ONE

    assert "could not be saved" in caplog.text
