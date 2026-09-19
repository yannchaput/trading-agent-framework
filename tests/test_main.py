from __future__ import annotations

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework import main as main_module
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.news_builtin import NewsBuiltinStrategy


def test_registry_lists_both_strategies() -> None:
    assert set(main_module.AGENT_STRATEGIES) == {"cross_momentum", "news_builtin"}


def test_news_builtin_builder_returns_the_strategy() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="news_builtin")

    strategy = main_module._build_news_builtin(broker, TradingMode.BACKTESTING)  # ty: ignore[invalid-argument-type]

    assert isinstance(strategy, NewsBuiltinStrategy)
    assert strategy.is_backtesting


def test_cross_momentum_builder_returns_none_without_a_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: [])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="cross_momentum")

    assert main_module._build_cross_momentum(broker, TradingMode.BACKTESTING) is None  # ty: ignore[invalid-argument-type]


def test_an_unknown_strategy_name_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["agent", "nope", "backtesting"])

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 1
