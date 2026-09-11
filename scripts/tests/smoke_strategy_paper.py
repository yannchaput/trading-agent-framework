#!/usr/bin/env python3
"""Manual paper-trading sanity check for the strategy framework.

NOT part of the automated test suite (the suite never touches the network).
Run by hand with real *paper* Alpaca credentials:

    uv run python scripts/tests/smoke_strategy_paper.py

Credentials come from env/.env.alpaca.integration-tests, the same git-ignored
file smoke_alpaca_orders.py uses. A script-local always-open clock replaces
Alpaca's calendar so the check also runs outside market hours. Outside regular
hours Alpaca may refuse to replace a queued order; the script then reports FAIL
with the broker's message -- rerun during regular hours before suspecting code.

The strategy (sleeptime 30S):
  1. iteration 1 -- logs cash and portfolio value, submits a 1-share SPY limit
     buy far below market so it rests
  2. iteration 2 -- modifies that order's limit price
  3. iteration 3 -- cancels it, waits for the cancel, then stops the run

PASS requires three iterations, a successful modify, no hook errors,
on_new_order and on_canceled_order to have fired, and on_strategy_end last. The run log lands in
logs/smoke/paper/<timestamp>_paper/paper.log (open it with VS Code's ANSI
Colors plugin). Refuses to run against a live account.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from decimal import Decimal

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.clock import MarketClock, MarketSession
from trading_agent_framework.config.env import AlpacaCredentials, TradingMode, find_project_root
from trading_agent_framework.entities.enums import OrderSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import TradingFrameworkError
from trading_agent_framework.strategies import Strategy

PROJECT_ROOT = find_project_root()
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke"
SYMBOL = "SPY"
RESTING_LIMIT_PRICE = Decimal("100.00")  # nowhere near SPY's real price
MODIFIED_LIMIT_PRICE = Decimal("101.00")
CANCEL_TIMEOUT_SECONDS = 20


class AlwaysOpenClock(MarketClock):
    """One session opening now and closing in an hour, whatever the real calendar says."""

    def __init__(self) -> None:
        start = self.now()
        self._session = MarketSession(open=start, close=start + timedelta(hours=1))

    def next_session(self) -> MarketSession | None:
        return self._session if self._session.close > self.now() else None


class SmokeStrategy(Strategy):
    sleeptime = "30S"
    minutes_before_closing = 0

    def initialize(self) -> None:
        self.vars.hooks = []
        self.vars.errors = []
        self.vars.order = None
        self.vars.modified = False
        self.vars.cancel_confirmed = False

    def _record(self, hook: str) -> None:
        self.vars.hooks.append(hook)
        self.log_info(f"hook: {hook}")

    def on_trading_iteration(self) -> None:
        self._record("on_trading_iteration")
        iteration = self.vars.hooks.count("on_trading_iteration")
        if iteration == 1:
            self.log_info(f"cash={self.get_cash()} portfolio_value={self.get_portfolio_value()}")
            order = self.create_order(SYMBOL, 1, OrderSide.BUY, limit_price=RESTING_LIMIT_PRICE)
            self.vars.order = self.submit_order(order)
        elif iteration == 2:
            self.vars.order = self.modify_order(self.vars.order, limit_price=MODIFIED_LIMIT_PRICE)
            self.vars.modified = True
        elif iteration == 3:
            self.cancel_order(self.vars.order)
            self.vars.cancel_confirmed = self.wait_for_order_execution(
                self.vars.order, timeout=CANCEL_TIMEOUT_SECONDS
            )
            self.stop()

    def on_new_order(self, order: Order) -> None:
        self._record("on_new_order")

    def on_canceled_order(self, order: Order) -> None:
        self._record("on_canceled_order")

    def on_strategy_end(self) -> None:
        self._record("on_strategy_end")

    def on_bot_crash(self, error: BaseException) -> None:
        # The executor catches hook failures and keeps trading; record them so they fail the run.
        self.vars.errors.append(repr(error))
        self.log_error(f"hook failed: {error!r}")


def _failures(strategy: SmokeStrategy) -> list[str]:
    hooks: list[str] = strategy.vars.hooks
    failures = []
    if strategy.vars.errors:
        failures.append(f"hook errors: {strategy.vars.errors}")
    if hooks.count("on_trading_iteration") != 3:
        failures.append(f"expected 3 iterations, got {hooks.count('on_trading_iteration')}")
    if not strategy.vars.modified:
        failures.append("modify_order did not succeed")
    for hook in ("on_new_order", "on_canceled_order"):
        if hook not in hooks:
            failures.append(f"{hook} never fired")
    if not strategy.vars.cancel_confirmed:
        failures.append("cancel was not confirmed by the trade stream")
    if not hooks or hooks[-1] != "on_strategy_end":
        failures.append("on_strategy_end did not run last")
    return failures


def main() -> int:
    load_dotenv(ENV_FILE, override=True)
    creds = AlpacaCredentials.from_env()
    if not creds.is_paper:
        print("Refusing to run: ALPACA_IS_PAPER is false (live account).", file=sys.stderr)
        return 1

    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds)
    strategy = SmokeStrategy(
        broker, mode=TradingMode.PAPER, clock=AlwaysOpenClock(), project_root=PROJECT_ROOT
    )
    started = datetime.now()
    try:
        strategy.run_paper_trading()
    except TradingFrameworkError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        order = strategy.vars.__dict__.get("order")
        if order is not None and order.is_active():
            broker.cancel_order(order)  # never leave a resting order behind

    failures = _failures(strategy)
    elapsed = (datetime.now() - started).total_seconds()
    if failures:
        print(f"FAIL after {elapsed:.0f}s: " + "; ".join(failures), file=sys.stderr)
        print(f"hooks: {strategy.vars.hooks}", file=sys.stderr)
        return 1
    print(f"PASS after {elapsed:.0f}s: hooks {strategy.vars.hooks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
