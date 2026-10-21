from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.strategies.common.trade_desk import TradeDesk
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams

NO_TARGET = {"error": "no target yet: call submit_target first"}


def _desk(tmp_path: Path, *, log_tag: str = "insider_trades") -> TradeDesk:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="insider_trades")
    broker.last_prices = {"AAA": Decimal("50")}
    broker.account = AccountBalances(cash=Decimal("5000"), portfolio_value=Decimal("10000"), buying_power=Decimal("5000"))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    return TradeDesk(strategy, CongressParams(), log_tag=log_tag)


def test_with_no_target_every_order_is_refused_until_one_is_set(tmp_path: Path) -> None:
    desk = _desk(tmp_path)
    desk.begin_run(None, traded=["AAA"])
    place = {t.__name__: t for t in desk.tools()}["place_order"]

    assert place("AAA", "sell", 1) == NO_TARGET
    assert place("AAA", "buy", 1) == NO_TARGET
    assert desk.orders == [] and not desk.has_target

    desk.set_target({"AAA": 0.10})

    assert desk.has_target and "order_id" in place("AAA", "buy", 1)


def test_a_refusal_with_no_target_does_not_read_the_account(tmp_path: Path) -> None:
    desk = _desk(tmp_path)
    desk.begin_run(None, traded=[])
    broker = desk._strategy.broker

    def boom() -> AccountBalances:
        raise AssertionError("the account must not be read")

    broker.get_account = boom  # type: ignore[method-assign]
    place = {t.__name__: t for t in desk.tools()}["place_order"]

    assert place("AAA", "buy", 1) == NO_TARGET


def test_the_audit_with_no_target_reports_only_the_unfilled_orders(tmp_path: Path) -> None:
    desk = _desk(tmp_path)
    desk.begin_run(None, traded=["AAA"])

    audit = desk.audit()

    assert audit.unfilled == [] and audit.shortfalls == [] and audit.in_flight == [] and audit.complete


def test_set_target_keeps_the_orders_already_placed_this_run(tmp_path: Path) -> None:
    desk = _desk(tmp_path)
    desk.begin_run({"AAA": 0.1}, [])
    place = {t.__name__: t for t in desk.tools()}["place_order"]
    assert "order_id" in place("AAA", "buy", 1)

    desk.set_target({"aaa": 0.2, "BBB": 0.0})

    assert len(desk.orders) == 1 and desk.traded == ["AAA"] and desk.has_target


def test_the_log_tag_prefixes_refusals(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    desk = _desk(tmp_path)
    desk.begin_run({"AAA": 0.1}, [])
    place = {t.__name__: t for t in desk.tools()}["place_order"]

    with caplog.at_level(logging.WARNING):
        place("AAA", "hold", 1)
        desk.begin_run(None, [])
        place("AAA", "buy", 1)

    assert "[insider_trades] order refused: side must be 'buy' or 'sell'" in caplog.text
    assert "[insider_trades] order refused: no target yet: call submit_target first" in caplog.text
    assert "[congress_trades]" not in caplog.text


def test_the_log_tag_prefixes_the_order_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    desk = _desk(tmp_path, log_tag="congress_trades")
    desk.begin_run({"AAA": 0.1}, [])
    place = {t.__name__: t for t in desk.tools()}["place_order"]

    with caplog.at_level(logging.INFO):
        place("AAA", "buy", 1)

    assert "[congress_trades] buy 1 AAA @ ~$50.00" in caplog.text
