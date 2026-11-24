"""Command-line entry point: `agent [--window WINDOW] <strategy> <live|paper|backtesting>`."""

import argparse
import logging

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import trading_agent_framework as tr
from trading_agent_framework.backtesting.placeholder import PlaceholderBroker
from trading_agent_framework.backtesting.time_window import PredefinedWindow
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.factory import build_broker
from trading_agent_framework.config import find_project_root, load_strategy_env
from trading_agent_framework.config.env import BrokerKind, BrokerSettings, TradingMode
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError
from trading_agent_framework.utils.strategy_factory import Strategies, build_strategy


def _refuse_tastytrade_sandbox() -> None:
    """Strategies never run against the Tastytrade sandbox: it clears trades and positions every 24 hours."""
    settings = BrokerSettings.from_env()
    if settings.kind is BrokerKind.TASTYTRADE and settings.is_paper:
        raise ConfigurationError(
            "BROKER=tastytrade with BROKER_API_IS_PAPER=true is the Tastytrade sandbox, which clears trades and positions "
            "every 24 hours: forward-test on Alpaca paper (BROKER=alpaca) and use BROKER=tastytrade with "
            "BROKER_API_IS_PAPER=false for live trading only"
        )


def _run_strategy(console: Console, trading_mode: TradingMode, strategy: Strategies, window: PredefinedWindow | None = None) -> None:
    """Load and run the selected strategy.

    Args:
        console (Console): the standard output
        trading_mode (TradingMode): the trading mode.
        strategy (Strategies): the strategy to run
        window (PredefinedWindow | None): backtest window replacing the strategy's default; only valid in backtesting mode.

    Returns:
        None
    """
    if window is not None and trading_mode is not TradingMode.BACKTESTING:
        console.print(f"--window only applies in backtesting mode, not {trading_mode.value}", style="bold red")
        raise SystemExit(1)
    strategy_name = strategy.value
    console.print(f"Running strategy: [bold cyan2]{strategy_name}[/bold cyan2] in [bold dark_red]{trading_mode.value}[/bold dark_red] mode")

    project_root = find_project_root()
    # Get environment file
    load_strategy_env(strategy_name, trading_mode.value, project_root)
    if trading_mode is TradingMode.BACKTESTING:
        broker: Broker = PlaceholderBroker(strategy_name)
    else:
        try:
            _refuse_tastytrade_sandbox()
            broker = build_broker(strategy_name)
        except (ConfigurationError, BrokerError) as exc:
            console.print(str(exc), style="bold red")
            raise SystemExit(1) from exc
    built = build_strategy(strategy, broker, trading_mode, window)
    if built is None:
        return

    # run strategy according to the trading mode
    built.run_strategy()


def main() -> None:
    """Main entry point of the program.

    It checks the command-line arguments for the strategy name and trading mode, validates them, and then runs the selected strategy.
    """
    # Temporary bootstrap logger: root has no handlers until setup_strategy_logging
    # runs (inside Strategy.run_strategy), so early config-loading logs (e.g.
    # load_strategy_env, build_strategy) would otherwise be dropped.
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
        """Print the available strategies from the Strategies enum."""
        for name in Strategies:
            console.print(f"  - {name.value}", style="white")

    def print_available_trading_modes():
        """Print the available trading modes from the TradingMode enum."""
        for mode in TradingMode:
            console.print(f"  - {mode.value}", style="magenta")

    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("strategy", nargs="?")
    parser.add_argument("trading_mode", nargs="?")
    parser.add_argument("--window", choices=[window.value for window in PredefinedWindow], help="backtest window replacing the strategy's default (backtesting only)")
    args = parser.parse_intermixed_args()

    # Check if the strategy and the trading mode are provided as command-line arguments
    if args.strategy is None or args.trading_mode is None:
        # The "[" need to be escaped in the console to avoid being interpreted as optional arguments by the shell
        # The closing bracket does not need to be escaped because it is not interpreted as an optional argument by the shell.
        console.print("Usage: agent \\[--window window] \\[strategy_name] \\[trading_mode]", style="bold red")
        console.print("Available strategies:", style="bold white")
        print_available_strategies()
        console.print("Available trading modes:", style="bold magenta")
        print_available_trading_modes()
        raise SystemExit(1)

    # Check if the provided strategy name is valid
    try:
        strategy = Strategies(args.strategy)
    except ValueError:
        console.print(f"Invalid strategy name: {args.strategy}", style="bold red")
        console.print("Available strategies:", style="bold yellow")
        print_available_strategies()
        raise SystemExit(1) from None

    # check if the provided trading mode is valid
    trading_mode = args.trading_mode
    if trading_mode not in [mode.value for mode in TradingMode]:
        console.print(f"Invalid trading mode: {trading_mode}", style="bold red")
        console.print("Available trading modes:", style="bold yellow")
        print_available_trading_modes()
        raise SystemExit(1)

    # Get the trading mode from the command-line argument and convert it to the TradingMode enum
    enum_mode = TradingMode(trading_mode)
    _run_strategy(console, enum_mode, strategy, PredefinedWindow(args.window) if args.window else None)


if __name__ == "__main__":
    main()
