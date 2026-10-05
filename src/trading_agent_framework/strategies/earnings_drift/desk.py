"""Every order earnings_drift places (spec §6): the agent's buy / trail / sell, stops on fills, the guardrails.

The only module of the strategy that submits or cancels orders. The agent reaches it through `tools.py`, the
strategy through `reconcile` / `ensure_stops` / `baseline_entries` and the order hooks. Tool paths return
`{"error": ...}` (logging a warning that names the guardrail) instead of raising; hook paths log and never raise.
A filled position is never left without a stop: a stop refused twice is replaced by an immediate market sell.

Orders are sent right after the close and fill at the next open. The agent's tools may run on LangGraph worker
threads, several at once: every public method holds one re-entrant lock (re-entrant because a cancel wait
dispatches order hooks on the same thread).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.earnings_drift.book import DriftState, JsonlLog, Trade, TradeState, sessions_held
from trading_agent_framework.strategies.earnings_drift.fact_sheet import fact_sheet
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.screening import Candidate
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

ALREADY_CLOSED = "already_closed"
_DATA_ERRORS = (BrokerError, BacktestError)


def _lean(order: Order) -> dict[str, Any]:
    return {"identifier": order.identifier, "symbol": order.asset.symbol, "side": order.side.value, "quantity": float(order.quantity or 0), "status": order.status.value}


def _number(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except InvalidOperation, ValueError:
        return None
    return number if number.is_finite() else None


class Desk:
    def __init__(
        self,
        strategy: Strategy,
        params: DriftParams,
        state: DriftState,
        *,
        save: Callable[[], None] = lambda: None,
        trade_log: JsonlLog,
        decision_log: JsonlLog,
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._state = state
        self._save_state = save
        self._trade_log = trade_log
        self._decision_log = decision_log
        self._lock = threading.RLock()
        # Stops this desk cancelled itself: their CANCELED hook must not be read as an outside cancel.
        self._expected_cancels: set[str] = set()
        self._today: date | None = None
        self._trading_dates: list[date] = []
        self._candidates: dict[str, Candidate] = {}
        self._decided: set[str] = set()
        self._cycle_buys = Decimal(0)  # estimated cost of the buys submitted this cycle
        self._cycle_sell_proceeds = Decimal(0)  # estimated proceeds of the sells submitted this cycle

    # --- session ---------------------------------------------------------------------------------

    def begin_session(self, today: date, trading_dates: Sequence[date], candidates: Sequence[Candidate], rejections: Mapping[str, str]) -> None:
        with self._lock:
            self._today = today
            self._trading_dates = sorted(set(trading_dates))
            self._candidates = {candidate.symbol: candidate for candidate in candidates}
            self._decided = set()
            self._cycle_buys = Decimal(0)
            self._cycle_sell_proceeds = Decimal(0)
            for symbol, reason in sorted(rejections.items()):
                self._log_decision(symbol, "rejected", reason=reason)

    def save(self) -> None:
        self._save_state()

    # --- views -----------------------------------------------------------------------------------

    @property
    def trades(self) -> list[Trade]:
        return list(self._state.trades.values())

    def exposed_symbols(self) -> set[str]:
        """Symbols with a trade of ours or any long position: an earnings event in one of them is `already_held`."""
        with self._lock:
            symbols = set(self._state.trades)
            try:
                symbols |= {position.asset.symbol for position in self._strategy.get_positions() if position.quantity > 0}
            except _DATA_ERRORS as exc:
                self._strategy.log_warning(f"[earnings_drift] positions unavailable, only this strategy's trades count as held: {exc}")
            return symbols

    def free_slots(self) -> int:
        """`max_positions` minus the trades not already being sold (a pending exit frees its slot)."""
        with self._lock:
            taken = sum(1 for trade in self._state.trades.values() if trade.exit_order_id is None)
            return max(0, self._params.max_positions - taken)

    def max_quantity(self, symbol: str) -> int:
        """Whole shares of a candidate the budget allows now: the slot cap, within what is still available (CLAUDE.md sizing rule)."""
        with self._lock:
            candidate = self._candidates.get(symbol.strip().upper())
            if candidate is None:
                return 0
            price = Decimal(str(candidate.reaction.close))
            try:
                account = self._strategy.broker.get_account()
            except _DATA_ERRORS as exc:
                self._strategy.log_warning(f"[earnings_drift] account unavailable, no buy can be sized: {exc}")
                return 0
            cap = account.portfolio_value / self._params.max_positions
            available = min(account.buying_power, account.cash + self._cycle_sell_proceeds) - self._cycle_buys
            budget = min(cap, available)
            if price <= 0 or budget <= 0:
                return 0
            return int((budget / price).to_integral_value(rounding=ROUND_FLOOR))

    def candidate_sheets(self) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._candidates.values(), key=lambda c: c.reaction.abnormal_pct, reverse=True)
            return [fact_sheet(candidate, self.max_quantity(candidate.symbol)) for candidate in ordered]

    def holdings_context(self) -> list[dict[str, Any]]:
        """Open trades not being sold, as the agent sees them; a backstop flag is shown once, then cleared."""
        with self._lock:
            rows: list[dict[str, Any]] = []
            for trade in sorted(self._open_trades(), key=lambda t: t.symbol):
                last = self._price(trade.symbol)
                pnl = None
                if last is not None and trade.entry_price:
                    pnl = round(float((last / trade.entry_price - 1) * 100), 1)
                status = "backstop" if trade.backstop else ("working" if self._stop_is_working(trade) else "missing")
                trade.backstop = False
                rows.append(
                    {
                        "symbol": trade.symbol,
                        "entry_date": trade.opened_on.isoformat() if trade.opened_on else None,
                        "entry_price": float(trade.entry_price) if trade.entry_price is not None else None,
                        "last_close": float(last) if last is not None else None,
                        "pnl_pct": pnl,
                        "sessions_held": self._sessions_held(trade),
                        "trail_percent": float(trade.trail_percent),
                        "stop_status": status,
                        "thesis": trade.thesis,
                        "reaction_low": float(trade.reaction_low) if trade.reaction_low is not None else None,
                    }
                )
            return rows

    def balances(self) -> dict[str, Any]:
        try:
            account = self._strategy.broker.get_account()
        except _DATA_ERRORS as exc:
            return {"error": str(exc), "free_slots": self.free_slots()}
        return {
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "free_slots": self.free_slots(),
        }

    # --- agent tools -----------------------------------------------------------------------------

    def buy(self, symbol: str, quantity: object, trail_percent: object, reason: str) -> dict[str, Any]:
        return self._buy(symbol, quantity, trail_percent, reason, decision="buy")

    def skip(self, symbol: str, reason: str) -> dict[str, Any]:
        with self._lock:
            symbol = str(symbol).strip().upper()
            if symbol not in self._candidates:
                return {"error": f"{symbol} is not one of today's candidates"}
            if symbol in self._decided:
                return {"error": f"{symbol} was already decided this session"}
            self._decided.add(symbol)
            self._log_decision(symbol, "skip", reason=self._clip(reason))
            return {"symbol": symbol, "status": "skipped"}

    def set_trailing_stop(self, symbol: str, trail_percent: object, reason: str) -> dict[str, Any]:
        with self._lock:
            symbol = str(symbol).strip().upper()
            trade = self._state.trades.get(symbol)
            if trade is None or trade.state is not TradeState.OPEN or trade.exit_order_id is not None:
                return self._refuse("order_limits", f"set_trailing_stop {symbol}: no open position (or it is being sold)")
            trail = _number(trail_percent)
            low, high = self._params.min_trail_percent, self._params.max_trail_percent
            if trail is None or not Decimal(str(low)) <= trail <= Decimal(str(high)):
                return self._refuse("order_limits", f"set_trailing_stop {symbol}: trail_percent must be between {low} and {high} (got {trail_percent!r})")
            if trail > trade.trail_percent:
                return self._refuse("order_limits", f"set_trailing_stop {symbol}: tighten-only, {trail}% is wider than the current {trade.trail_percent}%")
            if trail == trade.trail_percent:
                return {"symbol": symbol, "trail_percent": float(trail), "status": "unchanged"}
            released = self._release_stop(trade)
            if released == ALREADY_CLOSED:
                return {"status": ALREADY_CLOSED}
            if released is not None:
                return {"error": released}
            old = trade.trail_percent
            if self._submit_stop(trade, trail) is None:
                self._protect(trade, old)
                self._save_state()
                return {"error": f"the new stop was refused; the {old}% trail was placed again"}
            trade.trail_percent, trade.backstop = trail, False
            self._log_decision(symbol, "trail", trail_percent=float(trail), reason=self._clip(reason))
            self._save_state()
            return {"symbol": symbol, "trail_percent": float(trail), "status": "stop replaced"}

    def sell(self, symbol: str, reason: str) -> dict[str, Any]:
        with self._lock:
            symbol = str(symbol).strip().upper()
            trade = self._state.trades.get(symbol)
            if trade is None or trade.state is not TradeState.OPEN or trade.exit_order_id is not None:
                return self._refuse("order_limits", f"sell {symbol}: no open position (or it is being sold)")
            result = self._exit(trade, "agent_sell")
            if "error" not in result:
                self._log_decision(symbol, "sell", reason=self._clip(reason))
            return result

    # --- baseline and end of cycle ----------------------------------------------------------------

    def baseline_entries(self) -> list[str]:
        """Baseline mode: every candidate, best abnormal return first, at `max_quantity` and the default trail."""
        with self._lock:
            bought: list[str] = []
            for candidate in sorted(self._candidates.values(), key=lambda c: c.reaction.abnormal_pct, reverse=True):
                if self.free_slots() <= 0:
                    break
                quantity = self.max_quantity(candidate.symbol)
                if quantity < 1:
                    self._strategy.log_info(f"[earnings_drift] baseline: no budget left for {candidate.symbol}")
                    continue
                result = self._buy(candidate.symbol, quantity, self._params.default_trail_percent, "baseline", decision="baseline")
                if "error" not in result:
                    bought.append(candidate.symbol)
            return bought

    def record_undecided(self) -> None:
        with self._lock:
            for symbol in sorted(set(self._candidates) - self._decided):
                self._decided.add(symbol)
                self._log_decision(symbol, "undecided")

    # --- order hooks -----------------------------------------------------------------------------

    def on_order_filled(self, order: Order) -> None:
        with self._lock:
            trade = self._trade_for(order)
            if trade is None:
                return
            if order.identifier == trade.entry_order_id:
                if trade.state is TradeState.PENDING:
                    self._entry_filled(trade, order)
            else:
                self._close(trade, order)
            self._save_state()

    def on_order_canceled(self, order: Order) -> None:
        """An order ended unfilled (cancelled or expired): settle an entry, re-protect after a lost stop or exit sell."""
        with self._lock:
            if order.identifier in self._expected_cancels:
                self._expected_cancels.discard(order.identifier)
                return
            trade = self._trade_for(order)
            if trade is None:
                return
            if order.identifier == trade.entry_order_id:
                if trade.state is TradeState.PENDING:
                    if order.filled_quantity > 0:
                        self._entry_filled(trade, order)  # keep what filled, protected
                    else:
                        self._state.trades.pop(trade.symbol, None)
                        self._strategy.log_info(f"[earnings_drift] entry for {trade.symbol} ended {order.status.value} unfilled")
            elif order.identifier == trade.stop_order_id:
                trade.stop_order_id = None
                self._book_partial(trade, order)
                if trade.symbol in self._state.trades:
                    self._strategy.log_warning(f"guardrail stop_backstop: the stop for {trade.symbol} was cancelled outside the strategy; placing it again")
                    trade.backstop = True
                    self._protect(trade, trade.trail_percent)
            elif order.identifier == trade.exit_order_id:
                reason = trade.exit_reason or "agent_sell"
                trade.exit_order_id, trade.exit_reason = None, None
                self._book_partial(trade, order, reason)
                if trade.symbol in self._state.trades:
                    self._strategy.log_warning(f"guardrail stop_backstop: the {reason} sell of {trade.symbol} ended {order.status.value}; placing the stop again")
                    trade.backstop = True
                    self._protect(trade, trade.trail_percent)
            self._save_state()

    # --- internals -------------------------------------------------------------------------------

    def _buy(self, symbol: str, quantity: object, trail_percent: object, reason: str, *, decision: str) -> dict[str, Any]:
        with self._lock:
            symbol = str(symbol).strip().upper()
            shares, trail = _number(quantity), _number(trail_percent)
            low, high = self._params.min_trail_percent, self._params.max_trail_percent
            candidate = self._candidates.get(symbol)
            refusal = None
            if candidate is None:
                refusal = f"{symbol} is not one of today's candidates"
            elif symbol in self._state.trades:
                refusal = f"{symbol} is already held or being bought"
            elif shares is None or shares < 1 or shares != shares.to_integral_value():
                refusal = f"quantity must be a whole number of shares, at least 1 (got {quantity!r})"
            elif trail is None or not Decimal(str(low)) <= trail <= Decimal(str(high)):
                refusal = f"trail_percent must be between {low} and {high} (got {trail_percent!r})"
            elif self.free_slots() <= 0:
                refusal = f"max_positions ({self._params.max_positions}) reached"
            else:
                cap = self.max_quantity(symbol)
                if shares > cap:
                    refusal = f"quantity {shares} is above max_quantity {cap}"
            if refusal is not None:
                return self._refuse("order_limits", f"buy {symbol}: {refusal}")
            assert candidate is not None and shares is not None and trail is not None
            # Recorded before the broker is asked: an order that reached it although the client raised (a timeout) can
            # still fill, and `traded_symbols` is what lets the reconcile adopt such a position and give it a stop.
            self._state.traded_symbols.add(symbol)
            self._save_state()
            try:
                order = self._strategy.submit_order(self._strategy.create_order(symbol, shares, "buy"))
            except Exception as exc:  # a broker's _submit_order may re-raise after order.set_error (lumibot contract)
                self._strategy.log_warning(f"[earnings_drift] buy {symbol} refused by the broker: {exc}")
                return {"error": str(exc)}
            thesis = self._clip(reason)
            self._state.trades[symbol] = Trade(
                symbol=symbol,
                entry_order_id=order.identifier,
                trail_percent=trail,
                thesis=thesis,
                accession_number=candidate.event.accession_number,
                reaction_low=Decimal(str(candidate.reaction.reaction_low)),
            )
            self._cycle_buys += shares * Decimal(str(candidate.reaction.close))
            self._decided.add(symbol)
            self._log_decision(symbol, decision, quantity=int(shares), trail_percent=float(trail), reason=thesis)
            self._save_state()
            return {**_lean(order), "trail_percent": float(trail), "note": "fills at the next open; the trailing stop is placed when it fills"}

    def _entry_filled(self, trade: Trade, order: Order) -> None:
        trade.state = TradeState.OPEN
        trade.quantity = order.filled_quantity
        trade.entry_price = order.avg_fill_price
        trade.opened_on = self._now_date()
        self._strategy.log_info(f"[earnings_drift] entry {trade.symbol}: {trade.quantity} @ {trade.entry_price}; placing a {trade.trail_percent}% trailing stop")
        self._protect(trade, trade.trail_percent)

    def _close(self, trade: Trade, order: Order, reason: str | None = None) -> None:
        """A stop or exit sell filled: the trade leaves the book and is appended to `trades.jsonl`."""
        if reason is None:
            reason = (trade.exit_reason or "agent_sell") if order.identifier == trade.exit_order_id else "trail"
        exit_price = order.avg_fill_price
        quantity = order.filled_quantity or trade.quantity
        record: dict[str, Any] = {
            "symbol": trade.symbol,
            "accession_number": trade.accession_number,
            "entry_date": trade.opened_on.isoformat() if trade.opened_on else None,
            "entry_price": trade.entry_price,
            "exit_date": self._now_date().isoformat(),
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": None,
            "return_pct": None,
            "r_multiple": None,
            "sessions_held": self._sessions_held(trade),
            "exit_reason": reason,
            "trail_percent": trade.trail_percent,
            "thesis": trade.thesis,
            "agent_enabled": self._params.agent_enabled,
        }
        if exit_price is not None and trade.entry_price:
            ret = exit_price / trade.entry_price - 1
            record["pnl"] = (exit_price - trade.entry_price) * quantity
            record["return_pct"] = round(ret * 100, 2)
            record["r_multiple"] = round(ret / (trade.trail_percent / 100), 2)
        self._state.trades.pop(trade.symbol, None)
        self._trade_log.append(record)
        self._strategy.log_info(f"[earnings_drift] trade {trade.symbol} closed ({reason}): P&L {record['pnl']}")

    def _book_partial(self, trade: Trade, order: Order, reason: str = "trail") -> None:
        """Shares an ended stop or sell had already sold: deducted; the trade closes if none are left."""
        sold = order.filled_quantity
        if sold <= 0:
            return
        trade.quantity -= sold
        self._strategy.log_info(f"[earnings_drift] {trade.symbol}: {sold} shares sold by {order.identifier} before it ended; {trade.quantity} left")
        if trade.quantity <= 0:
            self._close(trade, order, reason)

    def _release_stop(self, trade: Trade) -> str | None:
        """Cancel the trade's working stop and wait for it: None once released, `ALREADY_CLOSED`, or an error message.

        Brokers refuse a sell above held minus pending sells, so the stop must go before any other exit sell. If the
        stop filled while the cancel was on its way (or before; its hook not seen yet), the trade is already out. The
        wait dispatches order hooks; in a backtest the cancel is synchronous, so it returns without moving the clock.
        """
        order = self._lookup(trade.stop_order_id)
        if order is None:
            trade.stop_order_id = None
            return None
        if order.is_filled():
            return ALREADY_CLOSED
        if order.is_active():
            self._expected_cancels.add(order.identifier)
            try:
                self._strategy.cancel_order(order)
            except BrokerError as exc:
                self._expected_cancels.discard(order.identifier)
                if order.is_filled():  # Alpaca raises when cancelling an already-filled order
                    return ALREADY_CLOSED
                return f"could not cancel the stop: {exc}"
            self._strategy.wait_for_order_execution(order, timeout=self._params.cancel_wait_seconds)
            if order.is_filled():
                return ALREADY_CLOSED
            if order.is_active():
                return "the stop cancel was not confirmed in time; nothing else was changed"
        trade.stop_order_id = None
        self._book_partial(trade, order)
        return None if trade.symbol in self._state.trades else ALREADY_CLOSED

    def _exit(self, trade: Trade, exit_reason: str) -> dict[str, Any]:
        """Release the stop, then a market sell of the whole trade at the next open; the stop goes back if the sell is refused."""
        released = self._release_stop(trade)
        if released == ALREADY_CLOSED:
            return {"status": ALREADY_CLOSED}
        if released is not None:
            return {"error": released}
        order = self._market_sell(trade, exit_reason)
        if order is None:
            self._protect(trade, trade.trail_percent)
            self._save_state()
            return {"error": "the sell was refused; the stop was placed again"}
        self._save_state()
        return _lean(order)

    def _protect(self, trade: Trade, trail: Decimal) -> bool:
        """Place the trade's trailing stop (two attempts); a market sell when both are refused. True when a stop is working."""
        if self._submit_stop(trade, trail) is not None or self._submit_stop(trade, trail) is not None:
            return True
        self._strategy.log_error(f"[earnings_drift] no stop could be placed for {trade.symbol}; selling {trade.quantity} at the next open")
        self._market_sell(trade, "backstop_sell")
        return False

    def _submit_stop(self, trade: Trade, trail: Decimal) -> Order | None:
        if trade.quantity <= 0:
            return None
        order = self._strategy.create_order(trade.symbol, trade.quantity, "sell", trail_percent=trail, time_in_force="gtc")
        try:
            submitted = self._strategy.submit_order(order)
        except Exception as exc:  # a broker's _submit_order may re-raise after order.set_error (lumibot contract)
            self._strategy.log_warning(f"[earnings_drift] stop for {trade.symbol} refused: {exc}")
            return None
        trade.stop_order_id = submitted.identifier
        return submitted

    def _market_sell(self, trade: Trade, exit_reason: str) -> Order | None:
        try:
            submitted = self._strategy.submit_order(self._strategy.create_order(trade.symbol, trade.quantity, "sell"))
        except Exception as exc:  # a broker's _submit_order may re-raise after order.set_error (lumibot contract)
            self._strategy.log_error(f"[earnings_drift] market sell of {trade.quantity} {trade.symbol} failed ({exit_reason}): {exc}")
            return None
        trade.exit_order_id, trade.exit_reason = submitted.identifier, exit_reason
        price = self._price(trade.symbol)
        if price is not None:
            self._cycle_sell_proceeds += trade.quantity * price
        return submitted

    def _open_trades(self) -> list[Trade]:
        return [t for t in self._state.trades.values() if t.state is TradeState.OPEN and t.exit_order_id is None]

    def _trade_for(self, order: Order) -> Trade | None:
        trade = self._state.trades.get(order.asset.symbol)
        if trade is None or order.identifier not in (trade.entry_order_id, trade.stop_order_id, trade.exit_order_id):
            return None
        return trade

    def _lookup(self, order_id: str | None) -> Order | None:
        if not order_id:
            return None
        try:
            return self._strategy.get_order(order_id)
        except Exception as exc:  # strategy.get_order can fall through to a raw SDK lookup
            self._strategy.log_warning(f"[earnings_drift] order {order_id} lookup failed: {exc}")
            return None

    def _stop_is_working(self, trade: Trade) -> bool:
        stop = self._lookup(trade.stop_order_id)
        return stop is not None and stop.is_active()

    def _price(self, symbol: str) -> Decimal | None:
        try:
            return self._strategy.get_last_price(symbol)
        except _DATA_ERRORS:
            return None

    def _now_date(self) -> date:
        return self._strategy.clock.now().astimezone(MARKET_TZ).date()

    def _sessions_held(self, trade: Trade) -> int:
        if trade.opened_on is None:
            return 0
        today = self._now_date()
        dates = [day for day in self._trading_dates if day <= today]
        if not dates or dates[-1] < today:
            dates.append(today)  # a hook between cycles: today's session counts
        return sessions_held(trade.opened_on, today, dates)

    def _refuse(self, guardrail: str, message: str) -> dict[str, Any]:
        self._strategy.log_warning(f"guardrail {guardrail}: {message}")
        return {"error": message}

    def _clip(self, reason: str) -> str:
        return str(reason or "").strip()[: self._params.reason_max_chars]

    def _log_decision(self, symbol: str, decision: str, **fields: Any) -> None:
        record: dict[str, Any] = {"date": self._today.isoformat() if self._today else None, "symbol": symbol, "decision": decision, **fields}
        candidate = self._candidates.get(symbol)
        if candidate is not None:
            record["features"] = {
                "abnormal_pct": candidate.reaction.abnormal_pct,
                "rel_volume": candidate.reaction.rel_volume,
                "hold_ratio": candidate.reaction.hold_ratio,
                "eps_surprise_pct": candidate.surprise.surprise.eps_surprise_pct,
                "sales_result": candidate.surprise.surprise.sales_result,
            }
        self._decision_log.append(record)
