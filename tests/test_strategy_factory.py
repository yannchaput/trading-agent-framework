from __future__ import annotations

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.congress_trades import CongressTradesStrategy
from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy
from trading_agent_framework.utils import strategy_factory
from trading_agent_framework.utils.strategy_factory import Strategies, build_strategy


def _broker(strategy: Strategies) -> FakeBroker:
    return FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name=strategy.value)


def _universe_of(strategy: object) -> list[str]:
    # CrossMomentumStrategy keeps it in `vars` (minus the sleeve assets); the others expose `universe`
    return strategy.vars.universe if isinstance(strategy, CrossMomentumStrategy) else strategy.universe  # type: ignore[attr-defined]


def _set_universe(monkeypatch: pytest.MonkeyPatch, universe: list[str]) -> None:
    monkeypatch.setattr(strategy_factory, "load_cross_momentum_universe", lambda: universe)


def test_strategy_names_are_the_cli_names() -> None:
    assert {strategy.value for strategy in Strategies} == {
        "bill_ackman",
        "congress_trades",
        "cross_momentum",
        "earnings_drift",
        "earnings_drift_baseline",
        "news_binary",
        "vwap_pullback_continuation",
    }


def test_every_strategy_has_a_builder() -> None:
    assert set(strategy_factory._STRATEGY_BUILDERS) == set(Strategies)


def test_news_binary_returns_the_strategy() -> None:
    strategy = build_strategy(Strategies.NEWS_BINARY, _broker(Strategies.NEWS_BINARY), TradingMode.BACKTESTING)

    assert isinstance(strategy, NewsBinaryStrategy)
    assert strategy.is_backtesting


def test_congress_trades_returns_the_strategy_without_needing_a_universe_file(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_universe(monkeypatch, [])

    strategy = build_strategy(Strategies.CONGRESS_TRADES, _broker(Strategies.CONGRESS_TRADES), TradingMode.BACKTESTING)

    assert isinstance(strategy, CongressTradesStrategy)
    assert strategy.is_backtesting


@pytest.mark.parametrize(
    ("name", "strategy_class"),
    [
        (Strategies.BILL_ACKMAN, BillAckmanStrategy),
        (Strategies.CROSS_MOMENTUM, CrossMomentumStrategy),
        (Strategies.EARNINGS_DRIFT, EarningsDriftStrategy),
        (Strategies.EARNINGS_DRIFT_BASELINE, EarningsDriftStrategy),
        (Strategies.VWAP_PULLBACK_CONTINUATION, VwapPullbackStrategy),
    ],
)
def test_universe_strategies_receive_the_universe(monkeypatch: pytest.MonkeyPatch, name: Strategies, strategy_class: type) -> None:
    _set_universe(monkeypatch, ["AAA", "BBB"])

    strategy = build_strategy(name, _broker(name), TradingMode.BACKTESTING)

    assert isinstance(strategy, strategy_class)
    assert _universe_of(strategy) == ["AAA", "BBB"]
    assert strategy.is_backtesting


@pytest.mark.parametrize(
    "name",
    [
        Strategies.BILL_ACKMAN,
        Strategies.CROSS_MOMENTUM,
        Strategies.EARNINGS_DRIFT,
        Strategies.EARNINGS_DRIFT_BASELINE,
        Strategies.VWAP_PULLBACK_CONTINUATION,
    ],
)
def test_universe_strategies_return_none_and_warn_without_a_universe(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, name: Strategies) -> None:
    _set_universe(monkeypatch, [])

    with caplog.at_level("WARNING"):
        assert build_strategy(name, _broker(name), TradingMode.BACKTESTING) is None

    assert "batch-universe" in caplog.text


def test_earnings_drift_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_universe(monkeypatch, ["AAA"])
    agent = build_strategy(Strategies.EARNINGS_DRIFT, _broker(Strategies.EARNINGS_DRIFT), TradingMode.BACKTESTING)
    baseline = build_strategy(Strategies.EARNINGS_DRIFT_BASELINE, _broker(Strategies.EARNINGS_DRIFT_BASELINE), TradingMode.BACKTESTING)

    assert isinstance(agent, EarningsDriftStrategy) and agent.settings.agent_enabled
    assert isinstance(baseline, EarningsDriftStrategy) and not baseline.settings.agent_enabled
    for strategy in (agent, baseline):  # each builder picks its own window; the length is a parameter, so only its shape is pinned
        start, end = strategy.parameters["backtesting_start"], strategy.parameters["backtesting_end"]
        assert start.tzinfo is not None and end.tzinfo is not None  # run_backtest refuses naive bounds
        assert start < end
