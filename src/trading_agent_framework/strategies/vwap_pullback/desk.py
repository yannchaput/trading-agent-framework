"""Every order the vwap_pullback strategy places: entries, protective stops, the flatten.

The only module of the package that submits, cancels or modifies orders. The strategy reaches it through
`enter_triggered` every tick and through its order hooks (`on_order_filled`/`on_order_canceled`).
`enter_long` returns `{"error": ...}` instead of raising; hook paths log and never raise. A position is
never left without a stop: a stop that cannot be placed is replaced by an immediate market sell.

Invariants every method preserves:
- never leave shares without a protective stop (or, after the flatten, without a sell);
- never sell twice, or more than is held: brokers refuse a sell above held minus pending sells, so the
  working stop is always cancelled (and the cancel waited for) before any other exit sell;
- never carry this strategy's position past the close; never touch a position no order of ours points at.

Accounting (see `trades.py`): `trade.quantity` only drops when a fill is booked; `_free_quantity` gives the
shares no unbooked exit sell covers, which is what any new stop or exit sell may be sized on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.vwap_pullback import risk
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.setups import SetupState, back_to_pullback, mark_done, mark_in_trade
from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeStatus
from trading_agent_framework.utils.errors import BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy
    from trading_agent_framework.strategies.vwap_pullback.session import SessionState

# `_release_stop`'s answer when the stop has already filled: the trade is out, nothing else to do.
STOPPED_OUT = "already_stopped_out"
# Market-data / account failures a desk read may hit: live BrokerError, backtest BacktestError.
_DATA_ERRORS = (BrokerError, BacktestError)


class Desk:
    """The order desk of one strategy run. Reads the session from `strategy.vars.session` on every call.

    Sections: views (read-only), entries, order events (hooks), order helpers, session boundaries (flatten, restart).
    """

    def __init__(self, strategy: Strategy, params: VwapPullbackParameters, *, trade_log: Callable[[], Path | None] = lambda: None) -> None:
        self._strategy = strategy
        self._params = params
        self._trade_log = trade_log  # where closed trades are appended as JSON lines (None: not logged)
        # Stops this desk cancelled itself (hand-offs, flatten): their CANCELED hook must not be taken for an
        # outside cancel, which would place the stop again right after the desk removed it on purpose.
        self._expected_cancels: set[str] = set()

    @property
    def state(self) -> SessionState:
        """The current session (replaced every session by the strategy; never cached here)."""
        return self._strategy.vars.session

    # --- views ---------------------------------------------------------------------

    def last_close(self, symbol: str) -> Decimal | None:
        """Latest 5-minute close as Decimal (no I/O: from the last scan); None without bars."""
        contexts = self.state.contexts.get(symbol)
        return Decimal(str(contexts[-1].close)) if contexts else None

    def session_pnl(self) -> Decimal:
        """Realised + open P&L of today's trades, open trades marked at their last 5-minute close."""
        prices = {t.symbol: p for t in self.state.book.open_trades() if (p := self.last_close(t.symbol)) is not None}
        return self.state.book.session_pnl(prices)

    def breaker_tripped(self) -> bool:
        """Daily loss limit reached: no new entries for the rest of the session."""
        return risk.circuit_breaker_tripped(self.session_pnl(), self.state.session_open_equity, self._params)

    def free_slots(self) -> int:
        """Spec §4: pending entries count as taken, pending full exits (no free share left, an exit sell working) as freed."""
        book = self.state.book
        holding = [t for t in book.open_trades() if not (self._free_quantity(t) <= 0 and self._exit_pending(t))]
        return risk.free_slots(len(holding), len(book.pending()), self._params)

    # --- entries -------------------------------------------------------------------

    def enter_triggered(self) -> list[str]:
        """Try every triggered setup, best stage-2 score first; the symbols whose entry was submitted.

        The score is the filter when triggers outnumber free slots: the best-ranked take the slots. A
        triggered symbol with no score this tick (tracked only because its setup is under way) goes after
        every scored one. A refused entry is logged by `enter_long` and the next trigger is tried.
        """
        state = self.state
        triggered = [symbol for symbol, setup in state.setups.items() if setup.state is SetupState.TRIGGERED]
        if not triggered or state.flattened or not risk.in_entry_window(self._strategy.get_datetime(), self._params) or self.breaker_tripped():
            return []
        # Unscored last (False sorts before True), then the higher score, then the symbol: the same order from run to run.
        triggered.sort(key=lambda symbol: (symbol not in state.scores, -state.scores.get(symbol, 0.0), symbol))
        entered: list[str] = []
        for index, symbol in enumerate(triggered):
            if self.free_slots() <= 0:
                self._strategy.log_info(f"no free position slot for {', '.join(triggered[index:])}")
                break
            if "error" not in self.enter_long(symbol):
                entered.append(symbol)
        return entered

    def enter_long(self, symbol: str) -> dict[str, Any]:
        """Enter a triggered setup: validate, size (`risk.plan_entry`), submit a marketable limit buy.

        Every rule is checked here, and a refusal comes back as `{"error": ...}` (and is logged as a warning,
        so the run log says why a triggered setup was not entered).
        The protective stop is NOT placed here: it goes in when the entry fills (`on_order_filled` -> `_protect`).
        """
        result = self._enter_long(symbol)
        if "error" in result:
            self._strategy.log_warning(f"entry {symbol.strip().upper()} refused: {result['error']}")
        return result

    def _enter_long(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        state = self.state
        setup = state.setups.get(symbol)
        # Guards, cheapest first; each names the rule so the warning in the log explains itself.
        if setup is None or setup.state is not SetupState.TRIGGERED:
            current = setup.state.value if setup is not None else "untracked"
            return {"error": f"{symbol} has no triggered setup right now (state: {current}); only triggered setups can be entered"}
        if state.flattened:
            return {"error": "the session is already flattened; no more entries today"}
        now = self._strategy.get_datetime()
        if not risk.in_entry_window(now, self._params):
            return {"error": f"entries are only allowed between {self._params.no_entry_before:%H:%M} and {self._params.no_entry_after:%H:%M}"}
        if self.breaker_tripped():
            return {"error": "the daily loss limit is reached; no more entries today"}
        if self.free_slots() <= 0:
            return {"error": "no free position slot"}
        if self._has_exposure(symbol):
            return {"error": f"{symbol} already has a position or an open order"}
        info = state.candidates.get(symbol)
        if info is None:
            return {"error": f"{symbol} is not a candidate this session"}
        try:
            last = self._strategy.get_last_price(symbol)
            account = self._strategy.broker.get_account()
        except _DATA_ERRORS as exc:
            return {"error": f"price or account unavailable: {exc}"}
        if last is None or setup.trigger_close is None or setup.pullback_low is None:
            return {"error": f"no price for {symbol}"}
        try:
            plan = risk.plan_entry(
                trigger_close=setup.trigger_close,
                pullback_low=setup.pullback_low,
                last_price=last,
                daily_atr=info.daily_atr,
                equity=account.portfolio_value,
                buying_power=account.buying_power,
                cash=account.cash,
                pending_sell_proceeds=self._pending_sell_proceeds(),
                params=self._params,
            )
        except risk.EntryRefused as exc:
            return {"error": str(exc)}
        try:
            submitted = self._strategy.submit_order(self._strategy.create_order(symbol, plan.quantity, "buy", limit_price=plan.limit_price))
        except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
            return {"error": str(exc)}
        # The trade exists from submission (PENDING): it takes a slot, and the fill/cancel hooks find it by order id.
        state.book.add(
            Trade(
                symbol=symbol,
                entry_order_id=submitted.identifier,
                planned_quantity=plan.quantity,
                stop_price=plan.stop_price,
                r_per_share=plan.r_per_share,
                entered_at=now,
            )
        )
        state.setups[symbol] = mark_in_trade(setup)  # freeze the setup: one trade per symbol per session
        self._strategy.log_info(f"entry {symbol}: {plan.quantity} at limit {plan.limit_price}, stop {plan.stop_price}, R {plan.r_per_share}")
        return {
            "symbol": symbol,
            "quantity": int(plan.quantity),
            "limit_price": float(plan.limit_price),
            "stop_price": float(plan.stop_price),
            "r_per_share": float(plan.r_per_share),
            "status": "entry submitted",
        }

    def _has_exposure(self, symbol: str) -> bool:
        """Whether `symbol` already has a trade, an open order or a position (any of them: no second entry)."""
        if self.state.book.get(symbol) is not None:
            return True
        if any(order.asset.symbol == symbol for order in self._strategy.broker.tracker.get_active_orders()):
            return True
        try:
            return self._strategy.get_position(symbol) is not None
        except _DATA_ERRORS:
            return True  # unknown: refuse rather than double up

    def _pending_sell_proceeds(self) -> Decimal:
        """What working market/limit sells should bring in (a stop only sells if triggered)."""
        total = Decimal(0)
        for order in self._strategy.broker.tracker.get_active_orders():
            if order.side is OrderSide.SELL and order.order_type in (OrderType.MARKET, OrderType.LIMIT) and order.quantity is not None:
                price = order.limit_price or self.last_close(order.asset.symbol)
                if price is not None:
                    total += order.quantity * price
        return total

    # --- order events ----------------------------------------------------------------

    def on_order_filled(self, order: Order, price: Decimal, quantity: Decimal) -> None:
        """FILLED hook (executor thread): book the fill on its trade.

        Entry fill -> the trade opens and gets its protective stop. Exit fill (stop or flatten sell) -> shares
        booked out, the order's id dropped from the unbooked list, the trade archived when nothing is left. A buy
        of ours with no trade at all is sold at once (`_sell_orphan_fill`).
        """
        state = getattr(self._strategy.vars, "session", None)
        trade = state.book.by_order_id(order.identifier) if state is not None else None
        if trade is None:
            self._sell_orphan_fill(order, quantity)
            return
        # FILLED is terminal, so the order's cumulative filled_quantity is the whole fill (partial events included).
        filled = order.filled_quantity if order.filled_quantity > 0 else quantity
        fill_price = order.avg_fill_price if order.avg_fill_price is not None else price
        if order.identifier == trade.entry_order_id:
            trade.record_entry_fill(filled, fill_price)
            self._protect(trade)
            return
        trade.record_exit_fill(filled, fill_price, self._strategy.get_datetime())
        if order.identifier in trade.exit_order_ids:
            trade.exit_order_ids.remove(order.identifier)  # booked: its shares have left trade.quantity, so it no longer counts as pending
        if order.identifier == trade.stop_order_id:
            trade.stop_order_id = None
            trade.exit_reason = trade.exit_reason or "stop"
        if trade.status is TradeStatus.CLOSED:
            self._archive(trade)

    def _sell_orphan_fill(self, order: Order, quantity: Decimal) -> None:
        """A buy of this strategy's that filled with no trade to protect it (e.g. adopted after a restart): never carry it."""
        if order.side is not OrderSide.BUY or self._strategy.broker.tracker.get_tracked_order(order.identifier) is None:
            return
        filled = order.filled_quantity if order.filled_quantity > 0 else quantity
        if filled <= 0:
            return
        symbol = order.asset.symbol
        self._strategy.log_error(f"a buy of {filled} {symbol} by this strategy filled with no trade to protect it; selling it at market now")
        try:
            self._strategy.submit_order(self._strategy.create_order(symbol, filled, "sell"))
        except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
            self._strategy.log_error(f"market sell of {filled} {symbol} failed (unprotected fill): {exc}")

    def on_order_canceled(self, order: Order) -> None:
        """CANCELED/expired hook: settle whatever the ended order leaves behind.

        Order of the checks matters: an exit sell is settled first even when the desk cancelled it itself
        (its partial fill must be booked); then the desk's own stop cancels are ignored; then an ended entry
        is settled, and a stop cancelled from OUTSIDE the strategy is placed again at once.
        """
        if getattr(self._strategy.vars, "session", None) is None:
            return
        trade = self.state.book.by_order_id(order.identifier)
        if trade is not None and order.identifier in trade.exit_order_ids:
            self._settle_exit(trade, order)  # also when the desk cancelled it itself: the partial fill is booked exactly once
            return
        if order.identifier in self._expected_cancels:
            self._expected_cancels.discard(order.identifier)
            return
        if trade is None:
            return
        if order.identifier == trade.entry_order_id:
            self._settle_entry(trade, order)
        elif order.identifier == trade.stop_order_id:
            self._strategy.log_warning(f"the stop for {trade.symbol} was cancelled outside the strategy; placing it again")
            self._record_stop_partial(trade, order)
            trade.stop_order_id = None
            if trade.status is not TradeStatus.CLOSED:
                self._submit_stop(trade, self._free_quantity(trade))

    def _settle_exit(self, trade: Trade, order: Order) -> None:
        """An exit sell that ended cancelled/expired: book its filled part once, then keep the rest protected (or sold, once flattened)."""
        self._expected_cancels.discard(order.identifier)
        trade.exit_order_ids.remove(order.identifier)
        if order.filled_quantity > 0:
            trade.record_exit_fill(order.filled_quantity, order.avg_fill_price or trade.stop_level, self._strategy.get_datetime())
        if trade.status is TradeStatus.CLOSED:
            trade.exit_reason = trade.exit_reason or "exit"
            self._archive(trade)
            return
        self._reprotect(trade)

    def _reprotect(self, trade: Trade) -> None:
        """Cover every free share of an open trade: after the flatten sell what no stop covers, otherwise make the stop as big as the free shares."""
        stop = self._strategy.get_order(trade.stop_order_id) if trade.stop_order_id is not None else None
        if stop is not None and stop.is_filled():
            return  # its fill hook books the trade out
        if stop is not None and not stop.is_active():
            self._record_stop_partial(trade, stop)  # cancelled/expired/errored without a hook reaching us
            trade.stop_order_id = None
            stop = None
            if trade.status is TradeStatus.CLOSED:
                return
        free = self._free_quantity(trade)
        # The whole quantity: the stop's filled-but-unbooked shares are still inside trade.quantity (and so inside `free`).
        stop_qty = stop.quantity if stop is not None and stop.quantity is not None else Decimal(0)
        if self.state.flattened:
            if free - stop_qty > 0:
                trade.exit_reason = trade.exit_reason or "flatten"
                self._market_sell(trade, free - stop_qty, "exit sell ended unfilled after the flatten")
            return
        if stop_qty >= free:
            return
        if stop is not None:
            released = self._release_stop(trade)
            if released is not None:
                if released != STOPPED_OUT:
                    self._strategy.log_error(f"{trade.symbol}: could not re-size the stop: {released}")
                return
        self._strategy.log_warning(f"{trade.symbol}: {free} free shares, stop covers {stop_qty}; placing the stop again")
        self._submit_stop(trade, self._free_quantity(trade))

    def _record_stop_partial(self, trade: Trade, order: Order) -> None:
        """A stop that partly filled before ending cancelled/errored: book the filled part, archive the trade if that closed it."""
        if order.filled_quantity <= 0:
            return
        trade.record_exit_fill(order.filled_quantity, order.avg_fill_price or trade.stop_level, self._strategy.get_datetime())
        if trade.status is TradeStatus.CLOSED:
            trade.exit_reason = trade.exit_reason or "stop"
            self._archive(trade)

    def _settle_entry(self, trade: Trade, order: Order) -> None:
        """An entry that ended without a full fill: keep and protect what filled, or drop the trade."""
        if order.filled_quantity > 0:
            trade.record_entry_fill(order.filled_quantity, order.avg_fill_price or trade.stop_price)
            self._protect(trade)
            return
        self.state.book.discard(trade.symbol)
        setup = self.state.setups.get(trade.symbol)
        if setup is not None and setup.state is SetupState.IN_TRADE:
            self.state.setups[trade.symbol] = back_to_pullback(setup)
        self._strategy.log_info(f"entry {trade.symbol} ended unfilled ({order.status.value}); setup back to pullback")

    def _protect(self, trade: Trade) -> None:
        """Right after an entry fill: place the protective stop (or sell at once if the session is already flattened)."""
        if self.state.flattened:  # a late entry fill after the flatten: never carry it
            trade.exit_reason = trade.exit_reason or "filled after the end-of-day flatten"
            self._market_sell(trade, trade.quantity, "filled after the end-of-day flatten")
            return
        if trade.stop_order_id is None:
            self._submit_stop(trade, self._free_quantity(trade))

    def reconcile(self, now: datetime) -> None:
        """Start-of-tick housekeeping: expire entries from earlier ticks, drop entries the broker rejected, re-place a stop that is gone."""
        # Hooks can be missed: a broker rejection arrives as ERROR, which reaches no strategy hook. Every tick
        # therefore re-derives the book's consistency from the orders themselves before anything else runs.
        for trade in list(self.state.book.active()):
            if trade.status is TradeStatus.PENDING:
                order = self._strategy.get_order(trade.entry_order_id)
                # Entry ended (rejected/errored/cancelled) without a hook reaching us: keep what filled, or drop it.
                if order is None or (not order.is_active() and not order.is_filled()):
                    if order is None:
                        self.state.book.discard(trade.symbol)
                    else:
                        self._settle_entry(trade, order)
                # Entry still working from an earlier tick: the trigger is stale, cancel it (the hook settles it).
                elif order.is_active() and trade.entered_at < now:
                    self._cancel(order)
            elif trade.status is TradeStatus.OPEN:
                # Exit sells that ended without a hook: book their partial fills and re-protect what they left.
                for order_id in list(trade.exit_order_ids):
                    order = self._strategy.get_order(order_id)
                    if order is not None and not order.is_active() and not order.is_filled() and trade.status is TradeStatus.OPEN:
                        self._settle_exit(trade, order)  # ended without a hook reaching us
                # Then make sure the working stop covers every free share (a missing or undersized stop is fixed).
                if trade.status is TradeStatus.OPEN:
                    self._reprotect(trade)

    def _exit_pending(self, trade: Trade) -> bool:
        """An exit sell is working, or has filled with its fill not yet booked (booked ones leave `exit_order_ids`)."""
        orders = [self._strategy.get_order(i) for i in trade.exit_order_ids]
        return any(o is not None and (o.is_active() or o.is_filled()) for o in orders)

    # --- order helpers ---------------------------------------------------------------

    def _free_quantity(self, trade: Trade) -> Decimal:
        """Shares held that no working exit sell is already selling: what a new stop or exit sell may cover.

        Every listed exit sell is unbooked (`on_order_filled` and `_settle_exit` drop an id once they book it), so each
        one still working or already FILLED counts by its whole quantity: `trade.quantity` only drops when the desk books
        a fill, so its filled-but-unbooked shares are still counted in it and are already sold.
        """
        pending = Decimal(0)
        for order_id in trade.exit_order_ids:
            order = self._strategy.get_order(order_id)
            if order is not None and (order.is_active() or order.is_filled()) and order.quantity is not None:
                pending += order.quantity
        return max(Decimal(0), trade.quantity - pending)

    def _submit_stop(self, trade: Trade, quantity: Decimal) -> bool:
        """Place the trade's protective stop for `quantity`: True when placed; False when it fell back to a market sell (or that failed too)."""
        if quantity <= 0:
            return False
        order = self._strategy.create_order(trade.symbol, quantity, "sell", stop_price=trade.stop_level)
        try:
            submitted = self._strategy.submit_order(order)
        except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
            self._strategy.log_error(f"stop for {trade.symbol} could not be placed ({exc}); selling {quantity} now")
            self._market_sell(trade, quantity, "protective stop failed")
            return False
        trade.stop_order_id = submitted.identifier
        return True

    def _market_sell(self, trade: Trade, quantity: Decimal, reason: str) -> Order | None:
        """Submit a market sell for the trade; the order is listed as an unbooked exit. None (logged) if refused."""
        try:
            submitted = self._strategy.submit_order(self._strategy.create_order(trade.symbol, quantity, "sell"))
        except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
            self._strategy.log_error(f"market sell of {quantity} {trade.symbol} failed ({reason}): {exc}")
            return None
        trade.exit_order_ids.append(submitted.identifier)  # counted as pending by `_free_quantity` until booked
        return submitted

    def _cancel(self, order: Order) -> None:
        """Request a cancel; a broker refusal is logged, not raised (the outcome arrives through the hooks)."""
        try:
            self._strategy.cancel_order(order)
        except BrokerError as exc:
            self._strategy.log_warning(f"cancel of {order.identifier} ({order.asset.symbol}) failed: {exc}")

    def _release_stop(self, trade: Trade) -> str | None:
        """Cancel the trade's working stop and wait for it: None once released, `STOPPED_OUT`, or an error message.

        Alpaca and `BacktestBroker` refuse a sell above held minus pending sells, so the stop must go before
        any other exit sell. If the stop filled while the cancel was on its way, the trade is already out.
        """
        if trade.stop_order_id is None:
            return None
        order = self._strategy.get_order(trade.stop_order_id)
        if order is None:
            trade.stop_order_id = None
            return None
        if order.is_filled():
            return STOPPED_OUT
        if order.is_active():
            self._expected_cancels.add(order.identifier)  # so its CANCELED hook is not read as an outside cancel
            try:
                self._strategy.cancel_order(order)
            except BrokerError as exc:
                self._expected_cancels.discard(order.identifier)
                if order.is_filled():  # Alpaca raises when cancelling an already-filled order
                    return STOPPED_OUT
                return f"could not cancel the stop: {exc}"
            # Waits on the strategy clock and dispatches order hooks meanwhile (instant in a backtest: cancels are synchronous).
            self._strategy.wait_for_order_execution(order, timeout=self._params.cancel_wait_seconds)
            if order.is_filled():
                return STOPPED_OUT
            if order.is_active():
                return "the stop cancel was not confirmed in time; nothing else was changed"
        # Cancelled just now, or already inactive and unfilled (cancelled/expired, its hook not yet seen): book any partial fill.
        self._record_stop_partial(trade, order)
        trade.stop_order_id = None
        return STOPPED_OUT if trade.status is TradeStatus.CLOSED else None

    def _archive(self, trade: Trade) -> None:
        """A trade closed: move it out of the book, mark its setup DONE, append it to `trades.jsonl`."""
        self.state.book.archive(trade)
        setup = self.state.setups.get(trade.symbol)
        if setup is not None:
            self.state.setups[trade.symbol] = mark_done(setup)
        self._strategy.log_info(f"trade {trade.symbol} closed ({trade.exit_reason}): P&L {trade.realised_pnl}")
        path = self._trade_log()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trade.to_json()) + "\n")

    # --- session boundaries --------------------------------------------------------------

    def flatten_all(self, reason: str) -> None:
        """Cancel every entry and stop of this session's trades and market-sell what they hold (only this strategy's trades).

        Without a session (its preparation kept failing) there is no trade book: this strategy's open orders are
        cancelled and the positions they point at closed instead. A stop whose cancel is confirmed only after its
        wait is caught by a re-check once the flatten sells have had their wait: its free shares are sold then.
        """
        state = getattr(self._strategy.vars, "session", None)
        if state is None:
            self._strategy.log_warning(f"{reason} with no session state: closing what this strategy's open orders point at")
            self._close_orphans(known=set())
            return
        # From now on: no entries, and a late entry fill is sold instead of protected (`_protect`).
        state.flattened = True
        sells: list[Order] = []
        unconfirmed: list[Order] = []  # stops whose cancel was not confirmed in time (waited on below)
        for trade in list(state.book.active()):
            if trade.status is TradeStatus.PENDING:
                order = self._strategy.get_order(trade.entry_order_id)
                if order is not None and order.is_active():
                    self._cancel(order)
                continue
            released = self._release_stop(trade)
            if released == STOPPED_OUT:
                continue
            if released is not None:
                self._strategy.log_error(f"flatten {trade.symbol}: {released}")
                stop = self._strategy.get_order(trade.stop_order_id) if trade.stop_order_id is not None else None
                if stop is not None:
                    unconfirmed.append(stop)
                continue
            trade.exit_reason = reason
            free = self._free_quantity(trade)
            sell = self._market_sell(trade, free, reason) if free > 0 else None
            if sell is not None:
                sells.append(sell)
        # Live: give the sells (and the late stop cancels) time to complete, dispatching their hooks meanwhile.
        # No tick runs after the flatten, so this wait and the re-check below are the last chance before the close.
        if (sells or unconfirmed) and not self._strategy.is_backtesting:
            self._strategy.wait_for_orders_execution([*sells, *unconfirmed], timeout=self._params.flatten_wait_seconds)
            still_open = [order.asset.symbol for order in sells if order.is_active()]
            if still_open:
                self._strategy.log_error(f"flatten: sells still open after {self._params.flatten_wait_seconds:.0f}s for {', '.join(still_open)}")
        self._sell_unprotected(reason)

    def _sell_unprotected(self, reason: str) -> None:
        """After the flatten: market-sell the free shares of every trade whose stop is gone (cancelled, expired) unfilled."""
        for trade in list(self.state.book.open_trades()):
            stop = self._strategy.get_order(trade.stop_order_id) if trade.stop_order_id is not None else None
            if stop is not None and (stop.is_active() or stop.is_filled()):
                continue
            if stop is not None:
                self._record_stop_partial(trade, stop)
                if trade.status is TradeStatus.CLOSED:
                    continue
            trade.stop_order_id = None
            free = self._free_quantity(trade)
            if free <= 0:
                continue
            self._strategy.log_warning(f"flatten {trade.symbol}: its stop is gone and {free} shares are unsold; selling them now")
            trade.exit_reason = reason
            self._market_sell(trade, free, reason)

    def close_unknown_positions(self) -> None:
        """After a restart: cancel this strategy's open orders in symbols no trade of the session knows, then close the
        positions those orders pointed at.

        An order left from before a restart (an entry, a stop) has no trade to manage it, whether or not its symbol is
        held yet. A position without an order of ours is not ours to touch: the account may be shared.
        """
        state = self.state
        state.unknown_positions_checked = True
        self._close_orphans(known={trade.symbol for trade in state.book.active()})

    def _close_orphans(self, *, known: set[str]) -> None:
        """Cancel this strategy's active orders in symbols outside `known`, then close the positions they pointed at."""
        ours: dict[str, list[Order]] = {}
        for order in self._strategy.broker.tracker.get_active_orders():
            if order.asset.symbol not in known:
                ours.setdefault(order.asset.symbol, []).append(order)
        if not ours:
            return
        try:
            positions = self._strategy.get_positions()
        except _DATA_ERRORS as exc:
            self._strategy.log_warning(f"could not check for positions to close: {exc}")
            return
        orphans = [order for orders in ours.values() for order in orders]
        self._strategy.log_warning(f"cancelling {len(orphans)} open order(s) of this strategy with no trade to manage them: {', '.join(sorted(ours))}")
        for order in orphans:
            self._cancel(order)
        if not self._strategy.is_backtesting:
            self._strategy.wait_for_orders_execution(orphans, timeout=self._params.cancel_wait_seconds)
        for position in positions:
            symbol = position.asset.symbol
            if symbol not in ours:
                continue
            self._strategy.log_warning(f"closing {position.quantity} {symbol}: held with this strategy's open orders but no trade to manage it")
            try:
                self._strategy.close_position(symbol)
            except BrokerError as exc:
                self._strategy.log_error(f"could not close {symbol}: {exc}")
