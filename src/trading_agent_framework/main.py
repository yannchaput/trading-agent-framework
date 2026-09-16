import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import trading_agent_framework as tr
from trading_agent_framework.config import find_project_root, load_strategy_env
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy

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
    # Get environment file
    env_file_path: Path = load_strategy_env(strategy_name, enum_mode.value, find_project_root())
    print(env_file_path.as_uri)
    # if strategy_name == "warren_buffett":
    #     # For some strategies like Warren Buffet, we need to pass the backtesting universe to the strategy constructor
    #     strategy = strategy_class(mode=mode, broker=broker, name=strategy_name, universe=WARREN_BUFFETT_BACKTEST_UNIVERSE)
    # elif strategy_name.startswith("cross_momentum"):
    #     # Load pre-computed universe from batch script (data/universe/stock_universe.json).
    #     # Falls back to broker's full equity list if the file doesn't exist yet.
    #     universe = load_cross_momentum_universe()
    #     if universe:
    #         strategy = strategy_class(mode=mode, broker=broker, name=strategy_name, universe=universe)
    #     else:
    #         console.print("Universe file not found — run batch_stock_universe.py before executing this strategy.", style="bold red")
    #         return

    # else:
    #     strategy = strategy_class(mode=mode, broker=broker, name=strategy_name)
    # strategy.run_strategy()


if __name__ == "__main__":
    main()
