"""bull_bear backtests trade the point-in-time universe history, never today's universe file.

`run_backtesting` first makes sure the history covers the window (a failure stops the run before it starts), records
it in `parameters` (so `settings.json` shows it) and preloads the union of the window's lists; the pipeline then asks
for the list of each review's date.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et
from tests.strategies.bull_bear.test_bull_bear_strategy import _FakeAgents

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.bull_bear import agent_bull_bear
from trading_agent_framework.strategies.bull_bear.agent_bull_bear import BullBearStrategy
from trading_agent_framework.strategies.common import batch_us_stock_universe as batch
from trading_agent_framework.strategies.common.batch_us_stock_universe import RateLimitGate
from trading_agent_framework.strategies.common.universe_backtest_data import UniverseBacktestData
from trading_agent_framework.strategies.common.universe_history import UniverseHistory
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError, FatalStrategyError, UniverseHistoryError


def _history(**months: tuple[str, ...]) -> UniverseHistory:
    """`_history(m2024_02=("A", "B"))`: the key's underscore form stands for "2024-02"."""
    return UniverseHistory(
        built=date(2026, 11, 3),
        pool_date=date(2026, 10, 30),
        top_n=1200,
        min_price=5.0,
        min_dollar_volume=20_000_000.0,
        lookback=63,
        months={key[1:].replace("_", "-"): symbols for key, symbols in months.items()},
    )


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


HISTORY = _history(m2024_02=("A", "B", "SHV"), m2024_03=("B", "C"))


def _strategy(mode: TradingMode, now: datetime | None = None, **kwargs: Any) -> BullBearStrategy:
    if mode is not TradingMode.BACKTESTING:
        kwargs.setdefault("bars_source", object())
    return BullBearStrategy(FakeBroker(FakeClock(now or et(2024, 2, 1, 7), []), "bull_bear"), mode=mode, **kwargs)


# --- the pipeline reads the list of the review's month --------------------------------------------------------


@pytest.mark.parametrize(("review_date", "expected"), [(date(2024, 2, 13), ["A", "B"]), (date(2024, 3, 12), ["B", "C"])])  # SHV is parking, never a stock
def test_a_backtest_ranks_the_month_list_of_each_review_date(review_date: date, expected: list[str]) -> None:
    strategy = _strategy(TradingMode.BACKTESTING)
    strategy._history = HISTORY

    assert strategy._universe_on(review_date) == expected


def test_a_month_missing_from_the_history_aborts_the_run_instead_of_abandoning_a_review() -> None:
    strategy = _strategy(TradingMode.BACKTESTING)
    strategy._history = HISTORY

    with pytest.raises(FatalStrategyError, match="2024-04"):
        strategy._universe_on(date(2024, 4, 9))


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_paper_and_live_rank_the_universe_they_were_built_with(mode: TradingMode) -> None:
    strategy = _strategy(mode, universe=["AAA", "SHV", "BBB"])

    assert strategy._universe_on(date(2026, 10, 6)) == ["AAA", "BBB"] == strategy._universe_on(date(2027, 1, 5))


def test_the_pipeline_is_wired_to_the_per_date_list(tmp_path: Path) -> None:
    strategy = _strategy(TradingMode.BACKTESTING, project_root=tmp_path, sector_of=lambda symbol: "Technology", fundamentals_of=lambda symbol, as_of: {})
    strategy.broker.news = FakeNewsProvider()
    strategy._agents = cast(AgentManager, _FakeAgents())
    strategy._history = HISTORY

    strategy.initialize()

    assert strategy.pipeline is not None
    assert strategy.pipeline._universe_on(date(2024, 3, 12)) == ["B", "C"]


# --- construction -----------------------------------------------------------------------------------------------


def test_a_backtest_can_be_built_without_a_universe() -> None:
    assert _strategy(TradingMode.BACKTESTING).universe == []


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_paper_and_live_still_refuse_to_run_without_a_universe(mode: TradingMode) -> None:
    with pytest.raises(ConfigurationError, match="universe"):
        _strategy(mode)
    with pytest.raises(ConfigurationError, match="universe"):
        _strategy(mode, universe=[])


# --- run_backtesting --------------------------------------------------------------------------------------------


class Recorder:
    """Stands in for `ensure_universe_history` and for `Strategy.run_backtesting`."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.ensure_calls: list[tuple[date, date]] = []
        self.runs: list[dict[str, Any]] = []
        self.counts: dict[str, Any] = {}

    def ensure(self, start: date, end: date) -> UniverseHistory:
        self.ensure_calls.append((start, end))
        if self.error is not None:
            raise self.error
        return replace(HISTORY, **self.counts) if self.counts else HISTORY

    def run(self, strategy: Any, **kwargs: Any) -> str:
        self.runs.append(kwargs)
        return "result"


def _recorder(monkeypatch: pytest.MonkeyPatch, error: Exception | None = None) -> Recorder:
    recorder = Recorder(error)
    monkeypatch.setattr(agent_bull_bear, "ensure_universe_history", recorder.ensure)
    monkeypatch.setattr(agent_bull_bear.Strategy, "run_backtesting", lambda self, **kwargs: recorder.run(self, **kwargs))
    return recorder


def _runnable() -> BullBearStrategy:
    return _strategy(TradingMode.BACKTESTING, parameters={"backtesting_start": datetime(2024, 2, 5, tzinfo=MARKET_TZ), "backtesting_end": datetime(2024, 3, 29, tzinfo=MARKET_TZ)})


def test_run_backtesting_ensures_the_history_over_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _recorder(monkeypatch)
    strategy = _runnable()

    assert strategy.run_backtesting() == "result"

    assert recorder.ensure_calls == [(date(2024, 2, 5), date(2024, 3, 29))]
    assert strategy._history is HISTORY


def test_run_backtesting_takes_the_window_from_the_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _recorder(monkeypatch)

    _runnable().run_backtesting(start=datetime(2024, 3, 4, tzinfo=MARKET_TZ), end=datetime(2024, 3, 22, tzinfo=MARKET_TZ))

    assert recorder.ensure_calls == [(date(2024, 3, 4), date(2024, 3, 22))]
    assert recorder.runs[0]["start"] == datetime(2024, 3, 4, tzinfo=MARKET_TZ)


def test_bull_bear_backtest_preloads_the_union_and_records_the_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _recorder(monkeypatch)
    strategy = _runnable()

    strategy.run_backtesting()

    assert [asset.symbol for asset in recorder.runs[0]["preload_assets"]] == ["A", "B", "C", "SHV", "SPY"]  # the union, the parking symbol once, the benchmark
    assert strategy.parameters["universe"] == {
        "mode": "point_in_time",
        "history_built": "2026-11-03",
        "pool_date": "2026-10-30",
        "top_n": 1200,
        "pool_size": 0,
        "unanswered_count": 0,
        "bars_absent_count": 0,
        "preload_missing": 0,
    }
    json.dumps(strategy.parameters["universe"])  # strictly serialisable: settings.json writes parameters out


def test_bull_bear_backtest_preloads_through_the_checked_batch_source_and_records_what_it_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _recorder(monkeypatch)
    strategy = _runnable()
    strategy.run_backtesting()
    window = (datetime(2023, 1, 1, tzinfo=MARKET_TZ), datetime(2024, 3, 29, tzinfo=MARKET_TZ))
    source = recorder.runs[0]["data_source"](*window)
    assert isinstance(source, UniverseBacktestData)

    # Yahoo drops B from every download; with the tolerance lifted for the test, the run goes on and says so.
    monkeypatch.setattr(batch, "MAX_ABSENT_FRACTION", 0.5)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    now = [0.0]
    monkeypatch.setattr(batch, "rate_limit_gate", RateLimitGate(30.0, 300.0, now=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))

    def download(symbols: list[str], **kwargs: Any) -> pd.DataFrame:
        row = {"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0, "Adj Close": 10.0, "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0}
        frames = {symbol: pd.DataFrame([row], index=pd.to_datetime(["2024-02-05"])) for symbol in symbols if symbol != "B"}
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    monkeypatch.setattr(UniverseBacktestData, "_real_download", lambda self: download)
    source.load(recorder.runs[0]["preload_assets"], *window, "day")

    assert source.missing == ["B"]
    assert strategy.parameters["universe"]["preload_missing"] == 1


def test_bull_bear_backtest_records_the_pool_counts_of_the_history(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _recorder(monkeypatch)
    recorder.counts = {"pool_size": 5400, "unanswered": ("U1",), "bars_absent": ("B1", "B2")}
    strategy = _runnable()

    strategy.run_backtesting()

    universe = strategy.parameters["universe"]
    assert (universe["pool_size"], universe["unanswered_count"], universe["bars_absent_count"]) == (5400, 1, 2)


def test_a_failing_history_stops_the_bull_bear_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _recorder(monkeypatch, UniverseHistoryError("no pool"))

    with pytest.raises(UniverseHistoryError, match="no pool"):
        _runnable().run_backtesting()

    assert recorder.runs == []
