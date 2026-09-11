#!/usr/bin/env python3
"""Manual paper-trading sanity check for the Alpaca broker order layer.

NOT part of the automated test suite -- the suite never touches the network
(Global Constraint 7). Run this by hand, with real (paper) Alpaca credentials,
whenever you want to verify the whole order lifecycle actually works end to
end against Alpaca's live paper API rather than a fake client:

    uv run python scripts/tests/smoke_alpaca_orders.py

Credentials come from env/.env.alpaca.integration-tests -- a dedicated,
git-ignored credentials file for this script, loaded directly by path rather
than through the strategy-oriented env/.env.{strategy}.{mode} resolver, since
it isn't any particular strategy's runtime configuration.

The script:
  1. submits a 1-share LIMIT buy order for SPY, priced far below market so it
     rests instead of filling
  2. starts the trade-update stream and waits for the broker to observe it go
     `new`
  3. calls pull_orders() and confirms the submitted order is visible there
  4. cancels the order
  5. waits for the stream to report it `canceled`
  6. stops the stream and prints a PASS/FAIL summary

Refuses to run at all against a non-paper (live) account.
"""

from __future__ import annotations

import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import ConfigurationError, TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-test"
SYMBOL = "SPY"
RESTING_LIMIT_PRICE = Decimal("100.00")  # nowhere near SPY's real market price
STREAM_STARTUP_GRACE_SECONDS = 2.0
WAIT_TIMEOUT_SECONDS = 20.0
POLL_INTERVAL_SECONDS = 0.5

logger = logging.getLogger("smoke_alpaca_orders")


class SmokeTestFailure(Exception):
    """Raised for any step that didn't reach the expected state."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(
            f"Credentials file not found: {ENV_FILE}\n"
            "Create it with ALPACA_API_KEY / ALPACA_API_SECRET / ALPACA_IS_PAPER=true "
            "before running this script."
        )
    load_dotenv(ENV_FILE, override=True)
    try:
        creds = AlpacaCredentials.from_env()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc

    if not creds.is_paper:
        raise SmokeTestFailure(
            "ALPACA_IS_PAPER is not true in the credentials file. Refusing to run "
            "this script against a live account."
        )
    return creds


def _wait_for_status(
    broker: AlpacaBroker,
    identifier: str,
    target: OrderStatus,
    *,
    timeout: float = WAIT_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout
    last_status: OrderStatus | None = None
    while time.monotonic() < deadline:
        order = broker.tracker.get_tracked_order(identifier)
        if order is not None:
            last_status = order.status
            if order.status is target:
                return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise SmokeTestFailure(
        f"Timed out after {timeout}s waiting for order {identifier} to reach "
        f"{target} (last observed status: {last_status})"
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    # Surface every orders.map_event/map_status warning at INFO so an
    # unmapped live wire value (the thing no automated test can observe) is
    # impossible to miss in this script's output.
    logging.getLogger("trading_agent_framework.brokers.alpaca.orders").setLevel(logging.INFO)

    print(f"Loading credentials from {ENV_FILE} ...")
    creds = _load_credentials()
    print(f"Paper account confirmed (ALPACA_IS_PAPER=true). Building {STRATEGY_NAME} broker ...")

    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=True)
    broker.start_stream()
    print(f"Stream started; waiting {STREAM_STARTUP_GRACE_SECONDS}s for it to connect ...")
    time.sleep(STREAM_STARTUP_GRACE_SECONDS)

    submitted: Order | None = None
    try:
        order = Order(
            strategy_name=STRATEGY_NAME,
            asset=Asset(symbol=SYMBOL),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            time_in_force=TimeInForce.DAY,
            limit_price=RESTING_LIMIT_PRICE,
        )

        print(f"Submitting a resting limit buy: 1 {SYMBOL} @ {RESTING_LIMIT_PRICE} ...")
        submitted = broker.submit_order(order)
        print(f"Submitted. Broker order id: {submitted.identifier}")

        print("Waiting for the stream to report the order as NEW ...")
        _wait_for_status(broker, submitted.identifier, OrderStatus.NEW)
        print("Order confirmed NEW via the stream.")

        print("Calling pull_orders() and checking the order is visible ...")
        pulled = broker.pull_orders()
        if not any(o.identifier == submitted.identifier for o in pulled):
            raise SmokeTestFailure(
                f"Order {submitted.identifier} not found in pull_orders() "
                f"({len(pulled)} orders returned)"
            )
        print(f"Order found in pull_orders() ({len(pulled)} orders returned).")

        print("Cancelling the order ...")
        broker.cancel_order(submitted)

        print("Waiting for the stream to report the order as CANCELED ...")
        _wait_for_status(broker, submitted.identifier, OrderStatus.CANCELED)
        print("Order confirmed CANCELED via the stream.")

    finally:
        print("Stopping stream ...")
        broker.stop_stream()

        # Best-effort safety net: if anything above failed after submission
        # but before/without a successful cancel, make sure nothing is left
        # resting on the account.
        if submitted is not None:
            order = broker.tracker.get_tracked_order(submitted.identifier)
            if order is not None and order.is_active():
                print(f"Cleanup: cancelling still-active order {submitted.identifier} ...")
                try:
                    broker.cancel_order(submitted)
                except TradingFrameworkError:
                    logger.exception("cleanup cancel failed")

    print(
        "\nPASS: submit -> new -> pull_orders -> cancel -> canceled, all confirmed via the stream."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
