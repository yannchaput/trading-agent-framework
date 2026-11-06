"""`UniverseBacktestData`: the preload of a point-in-time backtest, in batches, with a coverage check.

yfinance turns a rate limit into an empty column, and `YahooBacktestData.load` only warns about one, so a throttled
preload of thousands of tickers would silently shrink the universe for the whole run. This source retries and fails
instead; every symbol it is asked for was ranked on real bars, so an empty frame after the retries is a failure.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.common import batch_us_stock_universe as batch
from trading_agent_framework.strategies.common.batch_us_stock_universe import RateLimitGate
from trading_agent_framework.strategies.common.universe_backtest_data import UniverseBacktestData, universe_parameters
from trading_agent_framework.strategies.common.universe_history import UniverseHistory
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import UniverseHistoryError

_START = datetime(2024, 1, 2, tzinfo=MARKET_TZ)
_END = datetime(2024, 1, 5, tzinfo=MARKET_TZ)  # the fake bars run to 2024-01-08: the window trims them


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.t += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """No real waiting: time.sleep does nothing and the shared rate-limit gate runs on a fake clock."""
    fake = _Clock()
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(batch, "rate_limit_gate", RateLimitGate(30.0, 300.0, now=fake.now, sleep=fake.sleep))
    return fake


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0, "Adj Close": 10.0, "Volume": 1_000_000.0, "Dividends": 0.0, "Stock Splits": 0.0},
        index=pd.bdate_range("2024-01-02", periods=5),
    )


class _Download:
    """Stands in for `yfinance.download`: a ticker-grouped frame per requested symbol, `missing` ones absent."""

    def __init__(self, missing: tuple[str, ...] = ()) -> None:
        self.calls: list[dict] = []
        self.missing = set(missing)

    def __call__(self, symbols: list[str], **kwargs: object) -> pd.DataFrame:
        self.calls.append({"symbols": list(symbols), **kwargs})
        frames = {symbol: _frame() for symbol in symbols if symbol not in self.missing}
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _source(download: object, record: dict | None = None) -> UniverseBacktestData:
    return UniverseBacktestData(_START, _END, download=download, today=lambda: date(2024, 2, 1), record=record)  # type: ignore[arg-type]


def _assets(*symbols: str) -> list[Asset]:
    return [Asset(symbol=symbol) for symbol in symbols]


def _names(count: int) -> list[str]:
    return [f"S{index:03d}" for index in range(count)]


@pytest.mark.usefixtures("clock")
def test_the_assets_are_downloaded_in_batches_and_served(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "HISTORY_BATCH_SIZE", 2)
    download = _Download()
    source = _source(download)

    source.load(_assets("A", "B", "C", "D", "E"), _START, _END, "day")

    assert [call["symbols"] for call in download.calls] == [["A", "B"], ["C", "D"], ["E"]]
    # Raw bars through today, ticker-grouped, like YahooBacktestData: Yahoo's end is exclusive.
    assert all(call["start"] == "2024-01-02" and call["end"] == "2024-02-02" for call in download.calls)
    assert all(call["auto_adjust"] is False and call["actions"] is True and call["group_by"] == "ticker" for call in download.calls)
    bars = source.bars(Asset("E"), datetime(2024, 1, 31, tzinfo=MARKET_TZ), 10, "day")
    assert bars is not None and len(bars.df) == 4  # trimmed to the window end (2024-01-05)
    assert source.missing == []


@pytest.mark.usefixtures("clock")
def test_assets_already_loaded_are_not_downloaded_again() -> None:
    download = _Download()
    source = _source(download)
    source.load(_assets("A", "B"), _START, _END, "day")

    source.load(_assets("A", "B"), _START, _END, "day")

    assert len(download.calls) == 1


@pytest.mark.usefixtures("clock")
def test_an_all_empty_batch_is_retried() -> None:
    real = _Download()
    attempts: list[list[str]] = []

    def download(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        attempts.append(list(symbols))
        return pd.DataFrame() if len(attempts) == 1 else real(symbols, **kwargs)  # Yahoo swallowed a rate limit

    source = _source(download)
    source.load(_assets("A", "B"), _START, _END, "day")

    assert attempts == [["A", "B"], ["A", "B"]]
    assert source.missing == []
    assert source.bars(Asset("A"), datetime(2024, 1, 31, tzinfo=MARKET_TZ), 10, "day") is not None


def test_the_symbols_left_empty_are_asked_once_more_after_a_pause(clock: _Clock) -> None:
    real = _Download()
    asked: list[list[str]] = []

    def download(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        asked.append(list(symbols))
        return real([s for s in symbols if s != "B"], **kwargs) if len(asked) == 1 else real(symbols, **kwargs)  # the first pass drops B

    source = _source(download)
    source.load(_assets("A", "B", "C"), _START, _END, "day")

    assert asked == [["A", "B", "C"], ["B"]]
    assert clock.sleep_calls == [30.0]  # after a pause through the shared rate-limit gate
    assert source.missing == []
    assert source.bars(Asset("B"), datetime(2024, 1, 31, tzinfo=MARKET_TZ), 10, "day") is not None


@pytest.mark.usefixtures("clock")
def test_up_to_one_percent_still_empty_proceeds_and_is_recorded() -> None:
    names = _names(200)
    download = _Download(missing=("S007", "S150"))  # 2 of 200 = exactly 1%
    record: dict = {}
    source = _source(download, record)

    source.load(_assets(*names), _START, _END, "day")

    assert source.missing == ["S007", "S150"]
    assert record["preload_missing"] == 2
    calls = len(download.calls)
    # The empty frames are cached: asking for the bars never falls back to a lazy single-ticker download.
    assert source.bars(Asset("S007"), datetime(2024, 1, 31, tzinfo=MARKET_TZ), 10, "day") is None
    assert len(download.calls) == calls


@pytest.mark.usefixtures("clock")
def test_a_second_load_adds_to_the_missing_count() -> None:
    record: dict = {}
    source = _source(_Download(missing=("S007",)), record)

    source.load(_assets(*_names(150)), _START, _END, "day")
    source.load(_assets(*_names(250)), _START, _END, "day")  # S150..S249 are new; S007 is already settled

    assert record["preload_missing"] == 1
    assert source.missing == ["S007"]


@pytest.mark.usefixtures("clock")
def test_more_than_one_percent_still_empty_raises_naming_the_symbols() -> None:
    download = _Download(missing=("S007", "S050", "S150"))  # 3 of 200 = 1.5%

    with pytest.raises(UniverseHistoryError, match=r"3 of 200.*S007.*S050.*S150"):
        _source(download).load(_assets(*_names(200)), _START, _END, "day")


@pytest.mark.usefixtures("clock")
def test_a_download_that_fails_every_attempt_raises() -> None:
    def download(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("yahoo down")

    with pytest.raises(UniverseHistoryError, match="yahoo down"):
        _source(download).load(_assets("A"), _START, _END, "day")


def test_universe_parameters_record_the_history_and_the_pool_counts() -> None:
    history = UniverseHistory(
        built=date(2026, 11, 3), pool_date=date(2026, 10, 30), top_n=1200, min_price=10.0, min_dollar_volume=20e6, lookback=63,
        months={"2024-02": ("A",)}, pool_size=5400, unanswered=("U1", "U2"), bars_absent=("B1",),
    )

    parameters = universe_parameters(history)

    assert parameters == {
        "mode": "point_in_time", "history_built": "2026-11-03", "pool_date": "2026-10-30", "top_n": 1200,
        "pool_size": 5400, "unanswered_count": 2, "bars_absent_count": 1, "preload_missing": 0,
    }
    json.dumps(parameters)
