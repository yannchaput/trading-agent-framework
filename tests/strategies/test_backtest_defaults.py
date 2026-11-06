"""Every strategy's `run_backtesting` resolves its window, budget, benchmark, slippage and warmup the same way.

The defaults live in the strategy's `parameters` (slippage: the `Strategy.backtesting_slippage` default, 5 bps per side);
the override of a strategy adds only its data source, preloads, timestep and telemetry. Each case stubs the base
`Strategy.run_backtesting` and reads what the override passed; the base resolution is tested in tests/core/test_strategy.py.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, et, make_session
from tests.strategies.earnings_drift.test_drift_scanner import StaticEvents

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import REGIME_WARMUP_SLACK, Strategy
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.bull_bear import BullBearStrategy, agent_bull_bear
from trading_agent_framework.strategies.common.universe_history import UniverseHistory
from trading_agent_framework.strategies.congress_trades import CongressTradesStrategy
from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy, agent_cross_momentum
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.insider_trades import InsiderTradesStrategy
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy

NOW = et(2026, 9, 14, 10)
HISTORY = UniverseHistory(
    built=NOW.date(),
    pool_date=NOW.date(),
    top_n=1200,
    min_price=5.0,
    min_dollar_volume=20_000_000.0,
    lookback=63,
    months={f"{year}-{month:02d}": ("AAA", "BBB") for year in range(2010, 2028) for month in range(1, 13)},
)


def _broker(name: str) -> FakeBroker:
    return FakeBroker(FakeClock(NOW, [make_session(NOW.date())]), name)


def _build(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Strategy:
    mode = TradingMode.BACKTESTING
    if name in ("bull_bear", "cross_momentum"):
        monkeypatch.setattr(agent_bull_bear, "ensure_universe_history", lambda start, end: HISTORY)
        monkeypatch.setattr(agent_cross_momentum, "ensure_universe_history", lambda start, end: HISTORY)
    builders: dict[str, Callable[[], Strategy]] = {
        "bill_ackman": lambda: BillAckmanStrategy(_broker(name), mode=mode, universe=["AAA"], project_root=tmp_path, screen=object()),
        "bull_bear": lambda: BullBearStrategy(_broker(name), mode=mode, universe=["AAA"], project_root=tmp_path, bars_source=object(), sector_of=lambda s: "Tech", fundamentals_of=lambda s, d: {}),
        "congress_trades": lambda: CongressTradesStrategy(_broker(name), mode=mode, project_root=tmp_path, source=object()),
        "cross_momentum": lambda: CrossMomentumStrategy(_broker(name), mode=mode, project_root=tmp_path, universe=["AAA"]),
        "earnings_drift": lambda: EarningsDriftStrategy(_broker(name), mode=mode, universe=["AAA"], project_root=tmp_path, settings=DriftParams(), event_source=StaticEvents([])),
        "insider_trades": lambda: InsiderTradesStrategy(_broker(name), mode=mode, project_root=tmp_path, source=object()),
        "news_builtin": lambda: NewsBinaryStrategy(_broker(name), mode=mode, project_root=tmp_path),
        "vwap_pullback": lambda: VwapPullbackStrategy(_broker(name), mode=mode, universe=["AAA"], project_root=tmp_path),
    }
    strategy = builders[name]()
    # the per-instance copy: a pre-step (universe record) must not touch the class dict
    strategy.parameters = dict(strategy.parameters)
    return strategy


YEAR = (et(2025, 9, 29), et(2026, 10, 5))
# name -> (window, parameters["warmup_trading_days"], timestep); the budget is 10000 and the benchmark SPY everywhere
EXPECTED: dict[str, tuple[tuple[datetime, datetime], int, str]] = {
    "bill_ackman": (YEAR, 260, "day"),
    "bull_bear": ((et(2024, 9, 2), et(2026, 10, 5)), 300, "day"),
    "congress_trades": (YEAR, 10, "day"),
    "cross_momentum": ((et(2016, 1, 1), et(2026, 10, 5)), 300, "day"),
    "earnings_drift": (YEAR, 283, "day"),
    "insider_trades": (YEAR, 10, "day"),
    "news_builtin": (YEAR, 10, "minute"),
    "vwap_pullback": ((et(2021, 9, 20), et(2026, 10, 5)), 75, "minute"),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_strategy_backtest_defaults_unchanged(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """What the runner receives: the real `Strategy.run_backtesting` resolution, with a fake data source and a stubbed runner."""
    seen: dict[str, Any] = {}
    real = Strategy.run_backtesting
    monkeypatch.setattr("trading_agent_framework.backtesting.runner.run_backtest", lambda strategy, **kw: seen.update(kw))
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: real(self, **{**kwargs, "data_source": FakeBacktestDataSource()}))
    strategy = _build(name, tmp_path, monkeypatch)
    (start, end), warmup, timestep = EXPECTED[name]

    strategy.run_backtesting()

    assert seen["start"] == start and seen["end"] == end
    assert seen["budget"] == Decimal(10000)
    assert seen["benchmark"] == "SPY"
    assert seen["slippage"] == Decimal("0.0005")
    assert seen["timestep"] == timestep
    if timestep == "day":
        regime_floor = strategy.regime_params.min_bars + REGIME_WARMUP_SLACK
        assert seen["warmup_trading_days"] == max(warmup, regime_floor)
    else:
        assert seen["warmup_trading_days"] == warmup
