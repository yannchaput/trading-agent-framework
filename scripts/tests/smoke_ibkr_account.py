"""Manual smoke test: connect to a PAPER IB Gateway, check the account, print balances and positions.

Read-only: places no orders. Needs IB Gateway running and logged into a paper account, and
`env/.env.ibkr.integration-tests` with BROKER=ibkr, BROKER_API_IS_PAPER=true, optional IBKR_*,
and ALPACA_DATA_API_KEY / ALPACA_DATA_API_SECRET.

    uv run python scripts/tests/smoke_ibkr_account.py
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from trading_agent_framework.brokers.factory import build_broker
from trading_agent_framework.brokers.ibkr.broker import IbkrBroker
from trading_agent_framework.config.env import BrokerKind, BrokerSettings, find_project_root
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BrokerError

STRATEGY_NAME = "smoke_ibkr"
ENV_FILE = find_project_root() / "env" / ".env.ibkr.integration-tests"


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"SKIP: credentials file not found: {ENV_FILE}")
        return 0
    load_dotenv(ENV_FILE, override=True)
    settings = BrokerSettings.from_env()
    if settings.kind is not BrokerKind.IBKR or not settings.is_paper:
        print("Refusing to run: needs BROKER=ibkr and BROKER_API_IS_PAPER=true.", file=sys.stderr)
        return 1
    try:
        broker = build_broker(STRATEGY_NAME)  # connects and runs configure_account()
    except BrokerError as exc:
        print(f"SKIP: could not connect to IB Gateway: {exc}")
        return 0
    assert isinstance(broker, IbkrBroker)
    try:
        print(f"Account {broker.account_id}: {broker.get_account()}")
        for position in broker.pull_positions():
            print(f"  {position.asset.symbol}: {position.quantity} @ {position.avg_fill_price}")
        print(f"SPY last price (Alpaca IEX): {broker.get_last_price(Asset('SPY'))}")
        print(f"Next session: {broker.clock.next_session()}")
    finally:
        broker.stop_stream()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
