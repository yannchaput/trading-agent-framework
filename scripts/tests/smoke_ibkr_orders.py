"""Manual smoke test: order round trip against a PAPER IB Gateway.

Refuses to run unless the logged-in account is a paper (DU...) account. Places a far-from-market
limit buy, modifies it and cancels it; then buys and sells 1 share at market (fills only during
market hours). Same env file as smoke_ibkr_account.py.

    uv run python scripts/tests/smoke_ibkr_orders.py
"""

from __future__ import annotations

import sys
import time
from decimal import Decimal

from dotenv import load_dotenv

from trading_agent_framework.brokers.factory import build_broker
from trading_agent_framework.brokers.ibkr.account import is_paper_account
from trading_agent_framework.brokers.ibkr.broker import IbkrBroker
from trading_agent_framework.config.env import BrokerKind, BrokerSettings, find_project_root
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order

STRATEGY_NAME = "smoke_ibkr"
ENV_FILE = find_project_root() / "env" / ".env.ibkr.integration-tests"
SPY = Asset("SPY")


def _wait_for(order: Order, statuses: set[OrderStatus], seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while order.status not in statuses and time.monotonic() < deadline:
        time.sleep(0.2)
    print(f"  {order.identifier}: {order.status}")


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"Credentials file not found: {ENV_FILE}", file=sys.stderr)
        return 1
    load_dotenv(ENV_FILE, override=True)
    settings = BrokerSettings.from_env()
    if settings.kind is not BrokerKind.IBKR or not settings.is_paper:
        print("Refusing to run: needs BROKER=ibkr and BROKER_API_IS_PAPER=true.", file=sys.stderr)
        return 1
    broker = build_broker(STRATEGY_NAME)
    assert isinstance(broker, IbkrBroker)
    try:
        if not is_paper_account(broker.account_id):
            print("Refusing to run: IB Gateway is not logged into a paper account.", file=sys.stderr)
            return 1
        broker.tracker.listeners.append(lambda order, event: print(f"  event {event} for {order.identifier}"))
        broker.sync_open_orders()
        broker.start_stream()
        last = broker.get_last_price(SPY)
        if last is None:
            print("No SPY price from Alpaca.", file=sys.stderr)
            return 1
        far = (last * Decimal("0.5")).quantize(Decimal("0.01"))
        print(f"1) limit buy 1 SPY @ {far} (last {last})")
        limit = broker.submit_order(Order(STRATEGY_NAME, SPY, OrderSide.BUY, OrderType.LIMIT, quantity=Decimal(1), limit_price=far))
        _wait_for(limit, {OrderStatus.NEW})
        print(f"2) modify to {far - 1}")
        broker.modify_order(limit, limit_price=far - 1)
        print("3) cancel")
        broker.cancel_order(limit)
        _wait_for(limit, {OrderStatus.CANCELED})
        print("4) market buy 1, then sell it (market hours only)")
        buy = broker.submit_order(Order(STRATEGY_NAME, SPY, OrderSide.BUY, quantity=Decimal(1)))
        _wait_for(buy, {OrderStatus.FILL}, 60)
        if buy.status is OrderStatus.FILL:
            sell = broker.close_position(SPY)
            if sell is not None:
                _wait_for(sell, {OrderStatus.FILL}, 60)
    finally:
        broker.stop_stream()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
