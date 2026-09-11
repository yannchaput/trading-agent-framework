from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework import core
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import executor as executor_module
from trading_agent_framework.core import strategy as strategy_module
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError
from trading_agent_framework.utils.log import ANSI_BLUE, ANSI_RESET


class Hello(Strategy):
    sleeptime = "1D"

    def on_trading_iteration(self) -> None:
        self.log_info("hello from iteration")


def _strategy(
    tmp_path: Path,
    *,
    is_paper: bool = True,
    mode: TradingMode = TradingMode.PAPER,
    cls: type[Strategy] = Hello,
) -> Strategy:
    clock = FakeClock(et(2026, 9, 14, 4, 30), weekday_sessions(date(2026, 9, 14), 1))
    return cls(FakeBroker(clock, is_paper=is_paper), mode=mode, project_root=tmp_path)


def _log_content(tmp_path: Path, mode: str) -> str:
    [log_file] = (tmp_path / "logs" / "momentum" / mode).glob(f"*_{mode}/{mode}.log")
    return log_file.read_text(encoding="utf-8")


def test_run_paper_trading_logs_the_banner_and_the_iterations(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    assert isinstance(strategy.broker, FakeBroker)
    strategy.broker.positions = [
        Position(
            strategy_name="momentum",
            asset=Asset("AAPL"),
            quantity=Decimal(10),
            side=PositionSide.LONG,
        )
    ]

    strategy.run_paper_trading()

    content = _log_content(tmp_path, "paper")
    assert "PAPER TRADING MODE" in content
    assert "Broker account: PAPER" in content
    assert "05:00:00 until market opens" in content
    assert "11:30:00 until market closes" in content
    assert "Initial cash: 10000" in content
    assert "Position: 10 AAPL" in content
    assert f"{ANSI_BLUE}hello from iteration{ANSI_RESET}" in content


def test_run_live_trading_writes_the_live_log(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, is_paper=False, mode=TradingMode.LIVE)
    strategy.run_live_trading()
    assert "LIVE TRADING MODE" in _log_content(tmp_path, "live")
    assert strategy.trading_mode is TradingMode.LIVE


def test_live_trading_refuses_a_paper_account(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="live mode against a paper"):
        _strategy(tmp_path).run_live_trading()
    assert not (tmp_path / "logs").exists()


def test_paper_trading_refuses_a_live_account(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="paper mode against a live"):
        _strategy(tmp_path, is_paper=False).run_paper_trading()
    assert not (tmp_path / "logs").exists()


def test_banner_survives_account_lookup_errors(tmp_path: Path) -> None:
    class BrokenAccountBroker(FakeBroker):
        def get_account(self) -> AccountBalances:
            raise BrokerError("account service down")

    clock = FakeClock(et(2026, 9, 14, 4, 30), weekday_sessions(date(2026, 9, 14), 1))
    strategy = Hello(BrokenAccountBroker(clock), project_root=tmp_path)

    strategy.run_paper_trading()  # must not raise despite the account lookup failing

    content = _log_content(tmp_path, "paper")
    assert "Could not fetch account/position info: account service down" in content
    assert "hello from iteration" in content


def test_banner_survives_calendar_errors(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    assert isinstance(strategy.clock, FakeClock)
    strategy.clock.next_session_errors = [BrokerError("calendar down")]

    strategy.run_paper_trading()

    content = _log_content(tmp_path, "paper")
    assert "Market calendar unavailable: calendar down" in content
    assert "hello from iteration" in content


class Dispatch(Strategy):
    def __init__(self, broker: FakeBroker, **kwargs: object) -> None:
        super().__init__(broker, **kwargs)  # ty: ignore[invalid-argument-type]
        self.ran: list[str] = []

    def run_paper_trading(self) -> None:
        self.ran.append("paper")

    def run_live_trading(self) -> None:
        self.ran.append("live")

    def run_backtesting(self) -> None:
        self.ran.append("backtesting")


@pytest.mark.parametrize("mode", list(TradingMode))
def test_run_strategy_dispatches_on_the_trading_mode(tmp_path: Path, mode: TradingMode) -> None:
    strategy = _strategy(tmp_path, mode=mode, cls=Dispatch)
    strategy.run_strategy()
    assert isinstance(strategy, Dispatch)
    assert strategy.ran == [mode.value]


def test_run_backtesting_is_not_implemented_yet(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="backtesting"):
        _strategy(tmp_path, mode=TradingMode.BACKTESTING).run_strategy()


def test_core_package_reexports() -> None:
    assert core.Strategy is strategy_module.Strategy
    assert core.StrategyExecutor is executor_module.StrategyExecutor
