"""cross_momentum backtests trade the point-in-time universe history, never today's universe file.

On each review date the scan reads that month's list; `run_backtesting` first makes sure the history covers the
window (a failure stops the run before it starts), records it in `parameters` (so `settings.json` shows it) and
preloads the union of the window's lists.
"""

import json
from dataclasses import replace
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.common import batch_us_stock_universe as batch
from trading_agent_framework.strategies.common.batch_us_stock_universe import RateLimitGate
from trading_agent_framework.strategies.common.universe_backtest_data import UniverseBacktestData
from trading_agent_framework.strategies.common.universe_history import UniverseHistory
from trading_agent_framework.strategies.cross_momentum import agent_cross_momentum
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import FatalStrategyError, UniverseHistoryError

PARKING = {"symbol": "SHV", "trend_assets": ("GLD", "IEF")}


def _download_without(missing: str):
    """A `yfinance.download` stand-in: one bar per requested symbol, `missing` never comes back."""

    def download(symbols, **kwargs):
        row = {"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0, "Adj Close": 10.0, "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0}
        frames = {symbol: pd.DataFrame([row], index=pd.to_datetime(["2024-02-05"])) for symbol in symbols if symbol != missing}
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    return download


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


HISTORY = _history(m2024_02=("A", "B"), m2024_03=("B", "C", "GLD"))


def _backtest(now: datetime, history: UniverseHistory | None = HISTORY) -> CrossMomentumStrategy:
    strategy = CrossMomentumStrategy(FakeBroker(FakeClock(now, []), "cross_momentum"), mode=TradingMode.BACKTESTING, universe=["TODAY"])
    strategy.vars.universe_history = history
    return strategy


# --- the scan reads the list of its own month ---------------------------------------------------------------------


@pytest.mark.parametrize(("review_date", "expected"), [(date(2024, 2, 13), ["A", "B"]), (date(2024, 3, 12), ["B", "C"])])  # the sleeve's GLD is never scanned as a stock
def test_backtest_scans_the_month_list_of_each_review_date(review_date: date, expected: list[str]):
    fake = SimpleNamespace(vars=SimpleNamespace(universe_history=HISTORY, universe=["TODAY"]), parameters={"parking": PARKING}, _market_date=lambda: review_date)

    fake._sleeve_symbols = lambda: CrossMomentumStrategy._sleeve_symbols(fake)
    CrossMomentumStrategy._use_month_universe(fake)

    assert fake.vars.universe == expected


def test_the_review_day_scan_gets_its_months_universe(monkeypatch: pytest.MonkeyPatch):
    """Through `on_trading_iteration`: the list is in place before `compute_target_portfolio` runs."""
    strategy = _backtest(et(2024, 2, 13, 12))  # a Tuesday
    seen: list[list[str]] = []
    monkeypatch.setattr(strategy, "compute_target_portfolio", lambda: seen.append(list(strategy.vars.universe)) or ([], {}))
    monkeypatch.setattr(strategy, "_risk_exposure", lambda target: 1.0)
    monkeypatch.setattr(strategy, "_breadth_exposure", lambda: 1.0)
    monkeypatch.setattr(strategy, "rebalance", lambda target, ranks: None)
    strategy.vars.diagnostics_logger = SimpleNamespace(compute_and_persist=lambda: None, set_last_rebalance_weights=lambda weights: None)
    strategy.parameters = {**strategy.parameters, "volatility_targeting": {"enabled": False}}
    monkeypatch.setattr(agent_cross_momentum, "save_equity_history", lambda *args: None)

    strategy.on_trading_iteration()

    assert seen == [["A", "B"]]


def test_a_month_missing_from_the_history_aborts_the_run_instead_of_skipping_the_scan():
    """The executor rides out an exception in a tick, which would drop the week's scan silently."""
    strategy = _backtest(et(2024, 4, 9, 12))

    with pytest.raises(FatalStrategyError, match="2024-04"):
        strategy._use_month_universe()


def test_paper_and_live_keep_the_universe_they_were_built_with():
    strategy = CrossMomentumStrategy(FakeBroker(FakeClock(et(2026, 10, 6, 12), []), "cross_momentum"), mode=TradingMode.PAPER, universe=["AAA", "BBB"], bars_source=object())

    strategy._use_month_universe()

    assert strategy.vars.universe == ["AAA", "BBB"]


# --- construction -----------------------------------------------------------------------------------------------


def test_a_backtest_can_be_built_without_a_universe():
    strategy = CrossMomentumStrategy(FakeBroker(FakeClock(et(2024, 2, 13, 12), []), "cross_momentum"), mode=TradingMode.BACKTESTING)

    assert strategy.vars.universe == []


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_paper_and_live_still_refuse_to_run_without_a_universe(mode: TradingMode):
    broker = FakeBroker(FakeClock(et(2026, 10, 6, 12), []), "cross_momentum")

    with pytest.raises(SystemExit):
        CrossMomentumStrategy(broker, mode=mode, bars_source=object())


# --- run_backtesting --------------------------------------------------------------------------------------------


class Recorder:
    """Stands in for `ensure_universe_history` and for `Strategy.run_backtesting`."""

    def __init__(self, history: UniverseHistory | None = HISTORY, error: Exception | None = None):
        self.history, self.error = history, error
        self.ensure_calls: list[tuple[date, date]] = []
        self.runs: list[dict] = []

    def ensure(self, start: date, end: date) -> UniverseHistory:
        self.ensure_calls.append((start, end))
        if self.error is not None:
            raise self.error
        return self.history

    def run(self, strategy, **kwargs):
        self.runs.append(kwargs)
        return "result"


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(agent_cross_momentum, "ensure_universe_history", recorder.ensure)
    monkeypatch.setattr(agent_cross_momentum.Strategy, "run_backtesting", lambda self, **kwargs: recorder.run(self, **kwargs))
    return recorder


def _runnable(tmp_path, **parameters) -> CrossMomentumStrategy:
    strategy = CrossMomentumStrategy(
        FakeBroker(FakeClock(et(2024, 2, 1, 7), []), "cross_momentum"),
        mode=TradingMode.BACKTESTING,
        parameters={"backtesting_start": datetime(2024, 2, 5, tzinfo=MARKET_TZ), "backtesting_end": datetime(2024, 3, 29, tzinfo=MARKET_TZ), **parameters},
    )
    strategy.vars.history_file_path = tmp_path / "history.json"
    strategy.vars.breadth_file_path = tmp_path / "breadth.json"
    return strategy


def test_run_backtesting_ensures_the_history_over_the_window(recorder: Recorder, tmp_path):
    strategy = _runnable(tmp_path)

    assert strategy.run_backtesting() == "result"

    assert recorder.ensure_calls == [(date(2024, 2, 5), date(2024, 3, 29))]
    assert strategy.vars.universe_history is HISTORY


def test_run_backtesting_takes_the_window_from_the_overrides(recorder: Recorder, tmp_path):
    strategy = _runnable(tmp_path)

    strategy.run_backtesting(start=datetime(2024, 3, 4, tzinfo=MARKET_TZ), end=datetime(2024, 3, 22, tzinfo=MARKET_TZ))

    assert recorder.ensure_calls == [(date(2024, 3, 4), date(2024, 3, 22))]
    assert recorder.runs[0]["start"] == datetime(2024, 3, 4, tzinfo=MARKET_TZ)


def test_backtest_preloads_the_union_plus_sleeve_and_benchmark(recorder: Recorder, tmp_path):
    strategy = _runnable(tmp_path)

    strategy.run_backtesting()

    assert [asset.symbol for asset in recorder.runs[0]["preload_assets"]] == ["A", "B", "C", "SHV", "GLD", "IEF", "SPY"]
    assert all(isinstance(asset, Asset) for asset in recorder.runs[0]["preload_assets"])


def test_backtest_records_the_universe_in_parameters(recorder: Recorder, tmp_path):
    strategy = _runnable(tmp_path)

    strategy.run_backtesting()

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


def test_backtest_records_the_pool_counts_of_the_history(monkeypatch: pytest.MonkeyPatch, tmp_path):
    counted = replace(HISTORY, pool_size=5400, unanswered=("U1", "U2"), bars_absent=("B1", "B2", "B3"))
    recorder = Recorder(history=counted)
    monkeypatch.setattr(agent_cross_momentum, "ensure_universe_history", recorder.ensure)
    monkeypatch.setattr(agent_cross_momentum.Strategy, "run_backtesting", lambda self, **kwargs: recorder.run(self, **kwargs))
    strategy = _runnable(tmp_path)

    strategy.run_backtesting()

    universe = strategy.parameters["universe"]
    assert (universe["pool_size"], universe["unanswered_count"], universe["bars_absent_count"]) == (5400, 2, 3)
    json.dumps(strategy.parameters["universe"])


def test_backtest_preloads_through_the_checked_batch_source_and_records_what_it_misses(recorder: Recorder, tmp_path, monkeypatch: pytest.MonkeyPatch):
    strategy = _runnable(tmp_path)
    strategy.run_backtesting()
    source = recorder.runs[0]["data_source"](datetime(2023, 1, 1, tzinfo=MARKET_TZ), datetime(2024, 3, 29, tzinfo=MARKET_TZ))
    assert isinstance(source, UniverseBacktestData)

    # Yahoo drops B from every download; with the tolerance lifted for the test, the run goes on and says so.
    monkeypatch.setattr(batch, "MAX_ABSENT_FRACTION", 0.5)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    now = [0.0]
    monkeypatch.setattr(batch, "rate_limit_gate", RateLimitGate(30.0, 300.0, now=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    monkeypatch.setattr(UniverseBacktestData, "_real_download", lambda self: _download_without("B"))
    source.load(recorder.runs[0]["preload_assets"], datetime(2023, 1, 1, tzinfo=MARKET_TZ), datetime(2024, 3, 29, tzinfo=MARKET_TZ), "day")

    assert source.missing == ["B"]
    assert strategy.parameters["universe"]["preload_missing"] == 1


def test_a_failing_history_stops_the_backtest_before_it_starts(tmp_path, monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(error=UniverseHistoryError("no pool"))
    monkeypatch.setattr(agent_cross_momentum, "ensure_universe_history", recorder.ensure)
    monkeypatch.setattr(agent_cross_momentum.Strategy, "run_backtesting", lambda self, **kwargs: recorder.run(self, **kwargs))
    strategy = _runnable(tmp_path)

    with pytest.raises(UniverseHistoryError, match="no pool"):
        strategy.run_backtesting()

    assert recorder.runs == []
