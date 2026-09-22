from __future__ import annotations

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework import main as main_module
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy


def test_registry_lists_both_strategies() -> None:
    assert set(main_module.AGENT_STRATEGIES) == {"cross_momentum", "news_binary"}


def test_news_binary_builder_returns_the_strategy() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="news_binary")

    strategy = main_module._build_news_binary(broker, TradingMode.BACKTESTING)

    assert isinstance(strategy, NewsBinaryStrategy)
    assert strategy.is_backtesting


def test_cross_momentum_builder_returns_none_without_a_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: [])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="cross_momentum")

    assert main_module._build_cross_momentum(broker, TradingMode.BACKTESTING) is None


def test_an_unknown_strategy_name_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["agent", "nope", "backtesting"])

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 1


def test_backtesting_mode_builds_a_placeholder_and_never_a_live_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    from trading_agent_framework.backtesting.placeholder import PlaceholderBroker

    seen: list[object] = []
    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", lambda *a, **k: pytest.fail("a live broker was built for a backtest"))
    monkeypatch.setitem(main_module.AGENT_STRATEGIES, "probe", lambda broker, mode: seen.append(broker))

    main_module._run_strategy(Console(), TradingMode.BACKTESTING, "probe")

    [broker] = seen
    assert isinstance(broker, PlaceholderBroker)
    assert broker.strategy_name == "probe"


def test_a_configuration_error_exits_cleanly_in_paper_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    from trading_agent_framework.utils.errors import ConfigurationError

    def refuse(strategy_name: str):
        raise ConfigurationError("Missing or blank ALPACA_API_KEY / ALPACA_API_SECRET environment variables")

    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", refuse)

    with pytest.raises(SystemExit) as excinfo:
        main_module._run_strategy(Console(), TradingMode.PAPER, "news_binary")

    assert excinfo.value.code == 1
