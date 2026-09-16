import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import trading_agent_framework as tr
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config import find_project_root, load_strategy_env
from trading_agent_framework.config.env import AlpacaCredentials, TradingMode
from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.utils import load_cross_momentum_universe

# MAPPING OF STRATEGY NAMES TO CLASSES
AGENT_STRATEGIES = {
    "cross_momentum": CrossMomentumStrategy,
}


def main() -> None:
    """
    Main entry point of the program.
    It checks the command-line arguments for the strategy name and trading mode, validates them, and then runs the selected strategy.
    """
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

    console.print(f"Running strategy: [bold cyan2]{strategy_name}[/bold cyan2] in [bold dark_red]{enum_mode.value}[/bold dark_red] mode")

    # Load strategy class
    strategy_class = AGENT_STRATEGIES[strategy_name]
    project_root = find_project_root()
    # Get environment file
    load_strategy_env(strategy_name, enum_mode.value, project_root)
    creds = AlpacaCredentials.from_env()
    if not creds.api_key or not creds.api_secret:
        console.print("No credentials are sent as environment variables for the broker.", style="bold red")
        raise SystemExit(1)
    broker = AlpacaBroker.from_credentials(strategy_name, creds)
    # TODO: change strategy init argument order: no need of *args at the beginning any more
    universe = load_cross_momentum_universe()
    if universe:
        strategy = strategy_class(broker=broker, mode=enum_mode, universe=universe)
    else:
        console.print("Universe file not found — run batch_stock_universe.py before executing this strategy.", style="bold red")
        return

    # run strategy according to the trading mode
    strategy.run_strategy()


if __name__ == "__main__":
    main()
