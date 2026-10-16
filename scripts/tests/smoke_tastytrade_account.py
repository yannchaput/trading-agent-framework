"""Manual smoke test: authenticate against the Tastytrade SANDBOX, check the account, print balances and positions.

Read-only: places no orders. Credentials in env/.env.alpaca.integration-tests: TASTYTRADE_CLIENT_SECRET,
TASTYTRADE_REFRESH_TOKEN, TASTYTRADE_ACCOUNT_NUMBER (sandbox ones) and ALPACA_DATA_API_KEY /
ALPACA_DATA_API_SECRET. Always uses the sandbox, whatever BROKER and BROKER_API_IS_PAPER say.

    uv run python scripts/tests/smoke_tastytrade_account.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker
from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings, find_project_root
from trading_agent_framework.entities.asset import Asset

STRATEGY_NAME = "smoke_tastytrade"
ENV_FILE = find_project_root() / "env" / ".env.alpaca.integration-tests"


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"SKIP: credentials file not found: {ENV_FILE}")
        return 0
    load_dotenv(ENV_FILE, override=True)
    if not os.environ.get("TASTYTRADE_REFRESH_TOKEN", "").strip():
        print(f"SKIP: no TASTYTRADE_* credentials in {ENV_FILE}")
        return 0
    broker = TastytradeBroker.from_settings(STRATEGY_NAME, TastytradeSettings.from_env(is_paper=True), data=AlpacaCredentials.for_data())
    try:
        print(f"Sandbox account {broker.account_number}: {broker.get_account()}")
        for position in broker.pull_positions():
            print(f"  {position.asset.symbol}: {position.quantity} {position.side.value} @ {position.avg_fill_price}")
        print(f"SPY last price (Alpaca IEX): {broker.get_last_price(Asset('SPY'))}")
        print(f"Next session: {broker.clock.next_session()}")
    finally:
        broker.stop_stream()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
