#!/usr/bin/env python3
"""Manual smoke test: run a trivial buy-and-hold strategy through
Strategy.run_backtesting() against real Yahoo Finance data, over a short recent
window, and print the resulting metrics and output files.

Excluded from `tests/` on purpose (see scripts/tests/smoke_alpaca_data.py and its
siblings for the same convention) -- this hits the real network (Yahoo Finance) and is
meant to be run by hand, not in CI.

Credentials come from env/.env.smoke_backtest.paper or env/.env (the fallback
strategy-env resolver) so the placeholder broker can be constructed; the script will
fail clearly if neither file exists.

Run: uv run python scripts/tests/smoke_backtest.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials, TradingMode, find_project_root
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import ConfigurationError, TradingFrameworkError

PROJECT_ROOT = find_project_root()
ET = ZoneInfo("America/New_York")
AAPL = Asset("AAPL")


class BuyAndHold(Strategy):
    sleeptime = "1D"
    benchmark_symbol = "SPY"

    def on_trading_iteration(self) -> None:
        if self.first_iteration:
            price = self.get_last_price(AAPL)
            if price is not None:
                quantity = (self.get_cash() * Decimal("0.9") / price).quantize(Decimal(1))
                if quantity > 0:
                    self.submit_order(self.create_order(AAPL, quantity, "buy"))


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    """Load Alpaca credentials from env, attempting strategy-specific then fallback file."""
    env_file = PROJECT_ROOT / "env" / ".env.smoke_backtest.paper"
    if not env_file.is_file():
        # Fall back to the global env file.
        env_file = PROJECT_ROOT / "env" / ".env"
    if not env_file.is_file():
        raise SmokeTestFailure(
            f"Credentials file not found: {env_file}\n"
            "Create env/.env.smoke_backtest.paper or env/.env with ALPACA_API_KEY, "
            "ALPACA_API_SECRET, and ALPACA_IS_PAPER=true before running this script."
        )
    load_dotenv(env_file, override=True)
    try:
        creds = AlpacaCredentials.from_env()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {env_file}: {exc}") from exc
    if not creds.is_paper:
        raise SmokeTestFailure("ALPACA_IS_PAPER is not true in the credentials file; refusing to run.")
    return creds


def main() -> int:
    try:
        creds = _load_credentials()
        print("Credentials loaded. Building placeholder broker...")

        # Strategy.run_backtesting() rebinds the broker before the first bar
        # is touched, so any Broker instance works as a placeholder.
        placeholder_broker = AlpacaBroker.from_credentials(
            "smoke_backtest", creds, with_stream=False
        )

        strategy = BuyAndHold(placeholder_broker, mode=TradingMode.BACKTESTING, project_root=PROJECT_ROOT)

        end = datetime.now(ET)
        start = end - timedelta(days=120)
        print(f"Running BuyAndHold backtest from {start.date()} to {end.date()}...")

        result = strategy.run_backtesting(start=start, end=end)

        print(f"\nRun directory: {result.run_dir}")
        print(f"Sharpe (strategy): {result.metrics.get('sharpe_strategy', 'N/A')}")
        print(f"Max drawdown (strategy): {result.metrics.get('max_drawdown_strategy', 'N/A')}")
        print(f"Total return (strategy): {result.metrics.get('total_return_strategy', 'N/A')}")
        print("\nOutput files:")
        for filename in ("settings.json", "metrics.json", "equity.parquet", "trades.parquet", "indicators.parquet"):
            path = result.run_dir / filename
            status = "OK" if path.is_file() else "MISSING"
            print(f"  {filename}: {status}")

        print("\nPASS: backtest completed and results directory populated.")
        return 0

    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    except TradingFrameworkError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
