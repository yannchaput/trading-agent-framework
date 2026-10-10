"""A module to create a strategy instance from a strategy name and parameters."""

import logging
from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any

from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.bull_bear import BullBearStrategy
from trading_agent_framework.strategies.congress_trades import CongressTradesStrategy
from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.utils import load_cross_momentum_universe
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy

StrategyBuilder = Callable[[Broker, TradingMode], Strategy | None]

logger = logging.getLogger(__name__)


class Strategies(StrEnum):
    BILL_ACKMAN = auto()
    BULL_BEAR = auto()
    CONGRESS_TRADES = auto()
    CROSS_MOMENTUM = auto()
    EARNINGS_DRIFT = auto()
    EARNINGS_DRIFT_BASELINE = auto()
    NEWS_BINARY = auto()
    VWAP_PULLBACK_CONTINUATION = auto()


def _build_with_universe(strategy_class: Callable[..., Strategy], broker: Broker, mode: TradingMode, **kwargs: Any) -> Strategy | None:
    """Build a strategy that trades the shared stock universe; None (with a warning) when its file is missing."""
    universe = load_cross_momentum_universe()
    if not universe:
        logger.warning("Universe file not found — run `uv run batch-universe` before executing this strategy.")
        return None
    return strategy_class(broker=broker, mode=mode, universe=universe, **kwargs)


def _build_cross_momentum(broker: Broker, mode: TradingMode) -> Strategy | None:
    return _build_with_universe(CrossMomentumStrategy, broker, mode)


def _build_news_binary(broker: Broker, mode: TradingMode) -> Strategy | None:
    return NewsBinaryStrategy(broker=broker, mode=mode)


def _build_vwap_pullback(broker: Broker, mode: TradingMode) -> Strategy | None:
    return _build_with_universe(VwapPullbackStrategy, broker, mode)


def _build_bill_ackman(broker: Broker, mode: TradingMode) -> Strategy | None:
    return _build_with_universe(BillAckmanStrategy, broker, mode)


def _build_bull_bear(broker: Broker, mode: TradingMode) -> Strategy | None:
    return _build_with_universe(BullBearStrategy, broker, mode)


def _build_congress_trades(broker: Broker, mode: TradingMode) -> Strategy | None:
    return CongressTradesStrategy(broker=broker, mode=mode)


def _build_earnings_drift(broker: Broker, mode: TradingMode) -> Strategy | None:
    return _build_with_universe(EarningsDriftStrategy, broker, mode)


def _build_earnings_drift_baseline(broker: Broker, mode: TradingMode) -> Strategy | None:
    """The code-only baseline over 5 years: every gated candidate, the default trail, no LLM."""
    start, end = backtest_window(PredefinedWindow.YEAR)
    return _build_with_universe(
        EarningsDriftStrategy,
        broker,
        mode,
        settings=DriftParams(agent_enabled=False),
        parameters={"backtesting_start": start, "backtesting_end": end},
    )


# Every Strategies member has a builder (pinned by a test), so build_strategy needs no "unknown" branch.
_STRATEGY_BUILDERS: dict[Strategies, StrategyBuilder] = {
    Strategies.BILL_ACKMAN: _build_bill_ackman,
    Strategies.BULL_BEAR: _build_bull_bear,
    Strategies.CONGRESS_TRADES: _build_congress_trades,
    Strategies.CROSS_MOMENTUM: _build_cross_momentum,
    Strategies.EARNINGS_DRIFT: _build_earnings_drift,
    Strategies.EARNINGS_DRIFT_BASELINE: _build_earnings_drift_baseline,
    Strategies.NEWS_BINARY: _build_news_binary,
    Strategies.VWAP_PULLBACK_CONTINUATION: _build_vwap_pullback,
}


def build_strategy(strategy: Strategies, broker: Broker, mode: TradingMode) -> Strategy | None:
    """
    Build a strategy instance from a strategy name, Broker and trading mode.

    Args:
        strategy (Strategies): The strategy name.
        broker (Broker): The broker instance.
        mode (TradingMode): The trading mode.

    Returns:
        Strategy | None: The strategy instance, or None if it needs a universe file that does not exist.
    """
    return _STRATEGY_BUILDERS[strategy](broker, mode)
