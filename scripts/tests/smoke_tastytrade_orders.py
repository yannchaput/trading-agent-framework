"""Manual smoke test: order round trip against the Tastytrade SANDBOX, with the account streamer running.

PLACES SANDBOX ORDERS (and run_smoke_tests.sh runs it, as it runs every smoke_*.py), so it needs sandbox
credentials; without them it prints SKIP. Any failure exits non-zero. Same credentials as
smoke_tastytrade_account.py; always the sandbox. The sandbox clears trades and positions every 24 hours.

Its fills are NOT the documented synthetic ones ($1 market fills, limits under $3 filling at once): during
US market hours (2026-10-19) it filled market orders at the real price and left a $1 limit buy resting.
So a $5 limit buy of AAPL rests, and the fill step is a market buy (about one AAPL share of sandbox cash).

  1. $5 limit buy of 1 AAPL           -> rests (NEW during market hours; SUBMITTED while the US market is
                                         closed: the sandbox keeps a resting order in "Received" then)
  2. modify it to $6                   -> still working at $6; the original id's Cancelled must not cancel it
  3. cancel it                         -> CANCELED
  4. market buy of 1 AAPL             -> FILL; the position appears         (US market hours only, else SKIP)
  5. close the AAPL position (market) -> FILL; the position is gone         (US market hours only, else SKIP)

Run it during US market hours to exercise the fill steps and the Live status; outside them steps 4-5 are
skipped. Fractional and notional orders are not tested: the sandbox refuses them for AAPL.

The account streamer is NOT checked: since about 2026-09 the cert streamer has refused every connect with
"queue-declare-failed", a Tastytrade-side outage (https://github.com/joaquinbejar/tastytrade/issues/160),
so a subscription check would always FAIL. The script prints a NOTE with the outcome and, when the stream
is down, reads order statuses over REST (`broker.reconcile()`) instead. Once the sandbox streamer works
again, order notifications are exercised by the same steps and a NOTE says the stream subscribed;
restore a `check("the account stream subscribes", ...)` then.

It refuses to start if the strategy already has open orders or the account already holds AAPL (leftovers
of an earlier run would make the checks spurious). Whatever happens, it then cancels the orders it left
active and sells the AAPL it bought, before closing the session. An order whose outcome is unknown
(BrokerUncertainError) is reported as a FAIL, left tracked and never resent.

    uv run python scripts/tests/smoke_tastytrade_orders.py
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from decimal import Decimal

from dotenv import load_dotenv

from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker
from trading_agent_framework.brokers.tastytrade.stream import TastytradeAccountStream
from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings, find_project_root
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError

STRATEGY_NAME = "smoke_tastytrade"
ENV_FILE = find_project_root() / "env" / ".env.alpaca.integration-tests"
AAPL = Asset("AAPL")


POLL_INTERVAL = 1.0  # seconds between REST re-reads when the account stream is down


def _wait_for(order: Order, statuses: set[OrderStatus], poll: Callable[[], None] | None, seconds: float = 20.0) -> bool:
    """Wait for `order` to reach one of `statuses`; `poll` (the REST fallback) re-reads the orders when the stream is down."""
    deadline = time.monotonic() + seconds
    next_poll = time.monotonic()
    while order.status not in statuses and time.monotonic() < deadline:
        if poll is not None and time.monotonic() >= next_poll:
            poll()
            next_poll = time.monotonic() + POLL_INTERVAL
            continue
        time.sleep(0.2)
    print(f"  {order.identifier}: {order.status}")
    return order.status in statuses


def _market_open(broker: TastytradeBroker) -> bool:
    """Whether the regular US session is open now (the sandbox only works and fills orders then)."""
    session = broker.clock.next_session()
    return session is not None and session.open <= broker.clock.now() < session.close


def _aapl_quantity(broker: TastytradeBroker) -> Decimal:
    return sum((p.quantity for p in broker.pull_positions() if p.asset == AAPL), Decimal(0))


def _cleanup(broker: TastytradeBroker, poll: Callable[[], None] | None, *, bought_aapl: bool, uncertain: bool) -> None:
    """Leave the sandbox as found: each step is isolated, so a failing one never masks the original error."""
    active = broker.tracker.get_active_orders()  # only this script's orders: it refuses to start with leftovers
    for order in active:
        try:
            broker.cancel_order(order)
        except Exception as exc:
            print(f"  CLEANUP: could not cancel order {order.identifier}: {exc!r}")
    for order in active:
        try:
            _wait_for(order, {OrderStatus.CANCELED, OrderStatus.FILL, OrderStatus.ERROR}, poll, seconds=10.0)
        except Exception as exc:
            print(f"  CLEANUP: waiting for order {order.identifier} failed: {exc!r}")
    if not bought_aapl:
        return
    if uncertain:  # an unresolved write could still fill: selling now might oversell
        print("  CLEANUP: a write had an unknown outcome, so the AAPL position is NOT closed; check the account (the sandbox resets daily)")
        return
    try:
        held = _aapl_quantity(broker)  # nothing was held at the start, so everything held was bought here
        if held > 0:
            print(f"  CLEANUP: selling the {held} AAPL this script bought")
            sell = broker.submit_order(Order(STRATEGY_NAME, AAPL, OrderSide.SELL, OrderType.MARKET, quantity=held))
            _wait_for(sell, {OrderStatus.FILL}, poll)
    except Exception as exc:
        print(f"  CLEANUP: could not close the AAPL position: {exc!r}")


def _report(results: list[tuple[str, bool]]) -> int:
    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    print("OK")
    return 0


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
    state = {"bought_aapl": False, "uncertain": False}
    market_open = _market_open(broker)
    poll: Callable[[], None] | None = None  # set to broker.reconcile when the account stream is down

    def check(name: str, ok: bool) -> None:
        results.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    def write(name: str, action: Callable[[], object]) -> bool:
        """Run a broker write; an unknown outcome or a rejection is a FAIL line, not a crash."""
        try:
            action()
        except BrokerUncertainError as exc:
            state["uncertain"] = True
            results.append((f"{name}: outcome unknown", False))
            print(f"  FAIL  UNCERTAIN: {name}: the outcome is unknown, the order is left tracked and was not resent ({exc})")
            return False
        except BrokerError as exc:
            check(f"{name}: accepted", False)
            print(f"        {exc}")
            return False
        return True

    broker.tracker.listeners.append(lambda order, event: events.append((order.identifier, event)))
    try:
        leftovers = broker.sync_open_orders()  # adopted into the tracker, so the cleanup below cancels them
        held = _aapl_quantity(broker)
        if leftovers or held > 0:
            print(
                f"FAIL: leftovers of an earlier run: {len(leftovers)} open order(s) of {STRATEGY_NAME!r} (cancelled below) "
                f"and {held} AAPL held (close it by hand, or wait for the sandbox's daily reset), then rerun"
            )
            return 1
        print("0) the account stream subscribes (not a check: the sandbox streamer is down, see the docstring)")
        broker.start_stream()
        if broker.wait_for_stream(10.0):
            print("  NOTE: the account stream subscribed: order notifications are being delivered")
        else:
            poll = broker.reconcile
            print(f"  NOTE: the account stream did not subscribe ({broker.stream_error}); reading order statuses over REST (reconcile) instead")
            logging.getLogger(TastytradeAccountStream.__module__).setLevel(logging.ERROR)  # its reason is reported: hush the retries

        print("1) limit buy 1 AAPL @ $5 (rests in the sandbox)")
        resting = Order(STRATEGY_NAME, AAPL, OrderSide.BUY, OrderType.LIMIT, quantity=Decimal(1), limit_price=Decimal(5))
        if write("submit the $5 limit", lambda: broker.submit_order(resting)):
            if market_open:
                check("a $5 limit rests (NEW: Live)", _wait_for(resting, {OrderStatus.NEW}, poll))
            else:
                check("a $5 limit rests (SUBMITTED: Received while the market is closed, or NEW)", _wait_for(resting, {OrderStatus.NEW, OrderStatus.SUBMITTED}, poll))

            print("2) modify to $6")
            if write("modify to $6", lambda: broker.modify_order(resting, limit_price=Decimal(6))):
                time.sleep(3)  # the original's Cancelled arrives on the stream meanwhile
                check("the modified order is still working at $6", resting.status in {OrderStatus.NEW, OrderStatus.SUBMITTED} and resting.limit_price == Decimal(6))
                check("no cancel event reached the modified order", (resting.identifier, OrderEvent.CANCELED) not in events)

            print("3) cancel")
            if write("cancel", lambda: broker.cancel_order(resting)):
                check("cancel -> CANCELED", _wait_for(resting, {OrderStatus.CANCELED}, poll))

        if state["uncertain"]:
            print("an earlier write had an unknown outcome: stopping before the fill steps")
            return 1

        if not market_open:
            print("4) SKIP: the US market is closed, the sandbox does not fill orders now (run during market hours)")
            print("5) SKIP: nothing to close")
            return _report(results)
        print("4) market buy 1 AAPL (the sandbox fills it at the market price)")
        filled = Order(STRATEGY_NAME, AAPL, OrderSide.BUY, OrderType.MARKET, quantity=Decimal(1))
        state["bought_aapl"] = True  # from the submit on: the buy may fill even if the call fails
        if write("submit the market buy", lambda: broker.submit_order(filled)):
            check("the market buy fills (FILL)", _wait_for(filled, {OrderStatus.FILL}, poll))
            print(f"  filled at {filled.avg_fill_price} (Alpaca IEX last trade {broker.get_last_price(AAPL)})")
            position = next((p for p in broker.pull_positions() if p.asset == AAPL), None)
            check("the AAPL position appears", position is not None and position.quantity >= 1)

            print("5) close the AAPL position at market")
            closed: list[Order | None] = []
            if write("close the AAPL position", lambda: closed.append(broker.close_position(AAPL))):
                close = closed[0]
                check("the close fills (FILL)", close is not None and _wait_for(close, {OrderStatus.FILL}, poll))
                time.sleep(2)
                check("the AAPL position is gone", all(p.asset != AAPL for p in broker.pull_positions()))
    finally:
        _cleanup(broker, poll, bought_aapl=state["bought_aapl"], uncertain=state["uncertain"])
        broker.stop_stream()
    return _report(results)


if __name__ == "__main__":
    raise SystemExit(main())
