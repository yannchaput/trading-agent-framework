import logging
import sys
from collections.abc import Callable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import trading_agent_framework as tr
from trading_agent_framework.backtesting.placeholder import PlaceholderBroker
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.factory import build_broker
from trading_agent_framework.config import find_project_root, load_strategy_env
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.utils import load_cross_momentum_universe
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError

StrategyBuilder = Callable[[Broker, TradingMode], Strategy | None]


def _build_cross_momentum(broker: Broker, mode: TradingMode) -> Strategy | None:
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run batch_stock_universe.py before executing this strategy.", style="bold red")
        return None
    return CrossMomentumStrategy(broker=broker, mode=mode, universe=universe)


def _build_news_binary(broker: Broker, mode: TradingMode) -> Strategy | None:
    return NewsBinaryStrategy(broker=broker, mode=mode)


# MAPPING OF STRATEGY NAMES TO STRATEGY BUILDERS
def _build_vwap_pullback(broker: Broker, mode: TradingMode) -> Strategy | None:
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run `uv run batch-universe` before executing this strategy.", style="bold red")
        return None
    return VwapPullbackStrategy(broker=broker, mode=mode, universe=universe)


def _build_bill_ackman(broker: Broker, mode: TradingMode) -> Strategy | None:
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run `uv run batch-universe` before executing this strategy.", style="bold red")
        return None
    return BillAckmanStrategy(broker=broker, mode=mode, universe=universe)


def _build_earnings_drift(broker: Broker, mode: TradingMode) -> Strategy | None:
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run `uv run batch-universe` before executing this strategy.", style="bold red")
        return None
    return EarningsDriftStrategy(broker=broker, mode=mode, universe=universe)


def _build_earnings_drift_baseline(broker: Broker, mode: TradingMode) -> Strategy | None:
    """The code-only baseline over 5 years: every gated candidate, the default trail, no LLM."""
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run `uv run batch-universe` before executing this strategy.", style="bold red")
        return None
    start, end = backtest_window(PredefinedWindow.YEAR)
    return EarningsDriftStrategy(
        broker=broker,
        mode=mode,
        universe=universe,
        settings=DriftParams(agent_enabled=False),
        parameters={"backtesting_start": start, "backtesting_end": end},
    )


AGENT_STRATEGIES: dict[str, StrategyBuilder] = {
    "bill_ackman": _build_bill_ackman,
    "cross_momentum": _build_cross_momentum,
    "earnings_drift": _build_earnings_drift,
    "earnings_drift_baseline": _build_earnings_drift_baseline,
    "news_binary": _build_news_binary,
    "vwap_pullback_continuation": _build_vwap_pullback,
}


def _run_strategy(console: Console, trading_mode: TradingMode, strategy_name: str) -> None:
    """
    Load and run the selected strategy.

    Args:
        console (Console): the standard output
        trading_mode (TradingMode): the trading mode.
        strategy_name (str) : the strategy name

    Returns:
        None
    """
    console.print(f"Running strategy: [bold cyan2]{strategy_name}[/bold cyan2] in [bold dark_red]{trading_mode.value}[/bold dark_red] mode")

    # Load strategy builder
    builder = AGENT_STRATEGIES[strategy_name]
    project_root = find_project_root()
    # Get environment file
    load_strategy_env(strategy_name, trading_mode.value, project_root)
    if trading_mode is TradingMode.BACKTESTING:
        broker: Broker = PlaceholderBroker(strategy_name)
    else:
        try:
            broker = build_broker(strategy_name)
        except (ConfigurationError, BrokerError) as exc:
            console.print(str(exc), style="bold red")
            raise SystemExit(1) from exc
    strategy = builder(broker, trading_mode)
    if strategy is None:
        return

    # run strategy according to the trading mode
    strategy.run_strategy()


def main() -> None:
    """
    Main entry point of the program.
    It checks the command-line arguments for the strategy name and trading mode, validates them, and then runs the selected strategy.
    """
    # Temporary bootstrap logger: root has no handlers until setup_strategy_logging
    # runs (inside Strategy.run_strategy), so early config-loading logs (e.g.
    # load_strategy_env) would otherwise be dropped.
    logging.basicConfig(level=logging.INFO)

    console = Console()
    panel_text = Text()
    panel_text.append("🤖 Yann trading bot 🤖", style="bold green")
    panel_text.append(f"\nversion: {tr.__version__}")
    panel_text.justify = "center"
    banner = Panel(
        panel_text,
        title="",
        border_style="cyan",
        expand=False,
        box=box.SQUARE,
        padding=(1, 3),
    )
    console.print(banner, "\n")

    def print_available_strategies():
        """Print the available strategies from the AGENT_STRATEGIES mapping."""
        for name in AGENT_STRATEGIES.keys():
            console.print(f"  - {name}", style="white")

    def print_available_trading_modes():
        """Print the available trading modes from the TradingMode enum."""
        for mode in TradingMode:
            console.print(f"  - {mode.value}", style="magenta")

    # Check if the strategy is provided as a command-line argument
    if len(sys.argv) < 3:
        # The "[" need to be escaped in the console to avoid being interpreted as optional arguments by the shell
        # The closing bracket does not need to be escaped because it is not interpreted as an optional argument by the shell.
        console.print("Usage: python main.py \\[strategy_name] \\[trading_mode]", style="bold red")
        console.print("Available strategies:", style="bold white")
        print_available_strategies()
        console.print("Available trading modes:", style="bold magenta")
        print_available_trading_modes()
        raise SystemExit(1)

        # Check if the provided strategy name is valid
    strategy_name = sys.argv[1]
    if strategy_name not in AGENT_STRATEGIES:
        console.print(f"Invalid strategy name: {strategy_name}", style="bold red")
        console.print("Available strategies:", style="bold yellow")
        print_available_strategies()
        raise SystemExit(1)

    # check if the provided trading mode is valid
    trading_mode = sys.argv[2]
    if trading_mode not in [mode.value for mode in TradingMode]:
        console.print(f"Invalid trading mode: {trading_mode}", style="bold red")
        console.print("Available trading modes:", style="bold yellow")
        print_available_trading_modes()
        raise SystemExit(1)

    # Get the trading mode from the command-line argument and convert it to the TradingMode enum
    enum_mode = TradingMode(trading_mode)
    _run_strategy(console, enum_mode, strategy_name)


if __name__ == "__main__":
    main()
