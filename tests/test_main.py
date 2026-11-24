from __future__ import annotations

import pytest

from trading_agent_framework import main as main_module
from trading_agent_framework.backtesting.time_window import PredefinedWindow
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
    monkeypatch.setattr(main_module, "build_strategy", lambda strategy, broker, mode, window: seen.append(broker))

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


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_a_strategy_never_runs_against_the_tastytrade_sandbox(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: TradingMode) -> None:
    from rich.console import Console

    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", lambda *a, **k: pytest.fail("a sandbox broker was built for a strategy"))
    monkeypatch.setenv("BROKER", "tastytrade")
    monkeypatch.delenv("BROKER_API_IS_PAPER", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        main_module._run_strategy(Console(width=500), mode, Strategies.NEWS_BINARY)

    assert excinfo.value.code == 1
    assert "Tastytrade sandbox" in capsys.readouterr().out


def test_live_tastytrade_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console
    from tests.fakes import FakeBroker, FakeClock, et

    built: list[str] = []
    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", lambda name: built.append(name) or FakeBroker(FakeClock(et(2026, 10, 15, 10)), strategy_name=name))
    monkeypatch.setattr(main_module, "build_strategy", lambda strategy, broker, mode, window: None)
    monkeypatch.setenv("BROKER", "tastytrade")
    monkeypatch.setenv("BROKER_API_IS_PAPER", "false")

    main_module._run_strategy(Console(width=500), TradingMode.LIVE, Strategies.NEWS_BINARY)

    assert built == ["news_binary"]


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> list[tuple[Strategies, TradingMode, PredefinedWindow | None]]:
    calls: list[tuple[Strategies, TradingMode, PredefinedWindow | None]] = []
    monkeypatch.setattr("sys.argv", ["agent", *argv])
    monkeypatch.setattr(main_module, "_run_strategy", lambda console, mode, strategy, window=None: calls.append((strategy, mode, window)))
    main_module.main()
    return calls


def test_window_option_reaches_the_run_wherever_it_sits(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _run_main(monkeypatch, "--window", "2y_latest", "bull_bear", "backtesting")
    after = _run_main(monkeypatch, "bull_bear", "backtesting", "--window", "2y_latest")

    expected = [(Strategies.BULL_BEAR, TradingMode.BACKTESTING, PredefinedWindow.BI_YEAR_LATEST)]
    assert before == expected
    assert after == expected


def test_without_the_window_option_the_run_gets_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main(monkeypatch, "bull_bear", "backtesting") == [(Strategies.BULL_BEAR, TradingMode.BACKTESTING, None)]


def test_an_unknown_window_exits_and_lists_the_valid_ones(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["agent", "--window", "bi_year_latest", "bull_bear", "backtesting"])

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code != 0
    assert "2y_latest" in capsys.readouterr().err


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_a_window_is_refused_outside_backtesting(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: TradingMode) -> None:
    from rich.console import Console

    monkeypatch.setattr(main_module, "find_project_root", lambda: pytest.fail("the run started despite --window"))
    monkeypatch.setattr(main_module, "build_broker", lambda *a, **k: pytest.fail("a broker was built despite --window"))

    with pytest.raises(SystemExit) as excinfo:
        main_module._run_strategy(Console(width=500), mode, Strategies.BULL_BEAR, PredefinedWindow.MONTH)

    assert excinfo.value.code == 1
    assert "backtesting" in capsys.readouterr().out


def test_the_window_reaches_build_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    seen: list[PredefinedWindow | None] = []
    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_strategy", lambda strategy, broker, mode, window: seen.append(window))

    main_module._run_strategy(Console(), TradingMode.BACKTESTING, Strategies.BULL_BEAR, PredefinedWindow.MONTH)

    assert seen == [PredefinedWindow.MONTH]
