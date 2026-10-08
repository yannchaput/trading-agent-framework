from __future__ import annotations

import pytest

from trading_agent_framework import main as main_module
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.utils.strategy_factory import Strategies


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
    monkeypatch.setattr(main_module, "build_strategy", lambda strategy, broker, mode: seen.append(broker))

    main_module._run_strategy(Console(), TradingMode.BACKTESTING, Strategies.NEWS_BINARY)

    [broker] = seen
    assert isinstance(broker, PlaceholderBroker)
    assert broker.strategy_name == "news_binary"


def test_a_configuration_error_exits_cleanly_in_paper_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    from trading_agent_framework.utils.errors import ConfigurationError

    def refuse(strategy_name: str):
        raise ConfigurationError("Missing or blank ALPACA_API_KEY / ALPACA_API_SECRET environment variables")

    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", refuse)

    with pytest.raises(SystemExit) as excinfo:
        main_module._run_strategy(Console(), TradingMode.PAPER, Strategies.NEWS_BINARY)

    assert excinfo.value.code == 1
