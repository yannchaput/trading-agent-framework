from __future__ import annotations

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework import main as main_module
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.congress_trades import CongressTradesStrategy
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy


def test_registry_lists_the_strategies() -> None:
    assert set(main_module.AGENT_STRATEGIES) == {
        "bill_ackman",
        "congress_trades",
        "cross_momentum",
        "earnings_drift",
        "earnings_drift_baseline",
        "news_binary",
        "vwap_pullback_continuation",
    }


def test_news_binary_builder_returns_the_strategy() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="news_binary")

    strategy = main_module._build_news_binary(broker, TradingMode.BACKTESTING)

    assert isinstance(strategy, NewsBinaryStrategy)
    assert strategy.is_backtesting


def test_congress_trades_builder_returns_the_strategy_without_needing_a_universe_file() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="congress_trades")

    strategy = main_module._build_congress_trades(broker, TradingMode.BACKTESTING)

    assert isinstance(strategy, CongressTradesStrategy)
    assert strategy.is_backtesting


def test_bill_ackman_builder_returns_the_strategy_with_the_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: ["AAA", "BBB"])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")

    strategy = main_module._build_bill_ackman(broker, TradingMode.BACKTESTING)

    assert isinstance(strategy, BillAckmanStrategy)
    assert strategy.universe == ["AAA", "BBB"]
    assert strategy.is_backtesting


def test_bill_ackman_builder_returns_none_without_a_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: [])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")

    assert main_module._build_bill_ackman(broker, TradingMode.BACKTESTING) is None


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


def test_earnings_drift_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: ["AAA"])
    agent = main_module._build_earnings_drift(FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="earnings_drift"), TradingMode.BACKTESTING)
    baseline = main_module._build_earnings_drift_baseline(FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="earnings_drift_baseline"), TradingMode.BACKTESTING)
    assert isinstance(agent, EarningsDriftStrategy) and agent.settings.agent_enabled
    assert isinstance(baseline, EarningsDriftStrategy) and not baseline.settings.agent_enabled
    for strategy in (agent, baseline):  # each builder picks its own window; the length is a parameter, so only its shape is pinned
        start, end = strategy.parameters["backtesting_start"], strategy.parameters["backtesting_end"]
        assert start.tzinfo is not None and end.tzinfo is not None  # run_backtest refuses naive bounds
        assert start < end


def test_earnings_drift_builder_returns_none_without_a_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: [])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="earnings_drift")
    assert main_module._build_earnings_drift(broker, TradingMode.BACKTESTING) is None
