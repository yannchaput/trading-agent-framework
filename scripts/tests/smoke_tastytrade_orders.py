"""Manual smoke test: order round trip against the Tastytrade SANDBOX, with the account streamer running.

Built on the sandbox's simulated fills (they ignore the market): limit orders at $3 or more rest
forever, limit orders under $3 fill at once, market orders fill at $1. The sandbox clears trades and
positions every 24 hours. Same credentials as smoke_tastytrade_account.py; always the sandbox.

  1. $5 limit buy of 1 AAPL           -> rests (NEW)
  2. modify it to $6                   -> still working at $6; the original id's Cancelled must not cancel it
  3. cancel it                         -> CANCELED
  4. $2 limit buy of 1 AAPL           -> FILL; the position appears
  5. close the AAPL position (market) -> FILL; the position is gone

    uv run python scripts/tests/smoke_tastytrade_orders.py
"""

from __future__ import annotations

import os
import time
from decimal import Decimal

from dotenv import load_dotenv

from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker
from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings, find_project_root
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order

STRATEGY_NAME = "smoke_tastytrade"
ENV_FILE = find_project_root() / "env" / ".env.alpaca.integration-tests"
AAPL = Asset("AAPL")


def _wait_for(order: Order, statuses: set[OrderStatus], seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while order.status not in statuses and time.monotonic() < deadline:
        time.sleep(0.2)
    print(f"  {order.identifier}: {order.status}")
    return order.status in statuses


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"SKIP: credentials file not found: {ENV_FILE}")
        return 0
    load_dotenv(ENV_FILE, override=True)
    if not os.environ.get("TASTYTRADE_REFRESH_TOKEN", "").strip():
        print(f"SKIP: no TASTYTRADE_* credentials in {ENV_FILE}")
        return 0
    broker = TastytradeBroker.from_settings(STRATEGY_NAME, TastytradeSettings.from_env(is_paper=True), data=AlpacaCredentials.for_data())
    results: list[tuple[str, bool]] = []
    events: list[tuple[str, OrderEvent]] = []

    def check(name: str, ok: bool) -> None:
        results.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    broker.tracker.listeners.append(lambda order, event: events.append((order.identifier, event)))
    try:
        broker.sync_open_orders()
        broker.start_stream()
        time.sleep(2)  # let the streamer subscribe

        print("1) limit buy 1 AAPL @ $5 (rests in the sandbox)")
        resting = broker.submit_order(Order(STRATEGY_NAME, AAPL, OrderSide.BUY, OrderType.LIMIT, quantity=Decimal(1), limit_price=Decimal(5)))
        check("a $5 limit rests (NEW)", _wait_for(resting, {OrderStatus.NEW}))

        print("2) modify to $6")
        broker.modify_order(resting, limit_price=Decimal(6))
        time.sleep(3)  # the original's Cancelled arrives on the stream meanwhile
        check("the modified order is still working at $6", resting.status is OrderStatus.NEW and resting.limit_price == Decimal(6))
        check("no cancel event reached the modified order", (resting.identifier, OrderEvent.CANCELED) not in events)

        print("3) cancel")
        broker.cancel_order(resting)
        check("cancel -> CANCELED", _wait_for(resting, {OrderStatus.CANCELED}))

        print("4) limit buy 1 AAPL @ $2 (fills at once in the sandbox)")
        filled = broker.submit_order(Order(STRATEGY_NAME, AAPL, OrderSide.BUY, OrderType.LIMIT, quantity=Decimal(1), limit_price=Decimal(2)))
        check("a $2 limit fills (FILL)", _wait_for(filled, {OrderStatus.FILL}))
        position = next((p for p in broker.pull_positions() if p.asset == AAPL), None)
        check("the AAPL position appears", position is not None and position.quantity >= 1)

        print("5) close the AAPL position at market (fills at $1 in the sandbox)")
        close = broker.close_position(AAPL)
        check("the close fills (FILL)", close is not None and _wait_for(close, {OrderStatus.FILL}))
        time.sleep(2)
        check("the AAPL position is gone", all(p.asset != AAPL for p in broker.pull_positions()))
    finally:
        broker.stop_stream()
    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
