"""`Broker` implementation simulating fills against a `BacktestDataSource`, with no
network calls. Cash and positions are Decimal-exact. Order submission here tracks the
order and queues it for the next bar; `on_advance`/`process_pending` (this module,
added alongside the fill engine) are what actually fill it -- see the module's second
half below the "--- fills" marker.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import ClassVar
from uuid import uuid4

from trading_agent_framework.backtesting import fills
from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, Ledger
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BacktestDataError, BacktestError, BrokerError, ConfigurationError, OrderValidationError

logger = logging.getLogger(__name__)


@dataclass
class _PendingOrder:
    order: Order
    asset: Asset
    last_evaluated: datetime
    # True when `last_evaluated` was the raw submission instant rather than an exact bar-close
    # boundary -- i.e. a bar was still forming at submission. The first bar `_process_pending`
    # finds closing after `last_evaluated` is then that in-progress bar (the submitting bar) and
    # must be skipped once rather than used to fill: "nothing fills within the submitting bar".
    needs_skip: bool = False


@dataclass(frozen=True)
class _Projection:
    """What `BacktestBroker._pending` will have done to the account once it all fills."""

    cash: Decimal
    pending_sells: dict[Asset, Decimal]


class BacktestBroker(Broker):
    """`Broker` implementation backed by simulated fills against a `BacktestDataSource`."""

    name: ClassVar[str] = "backtest"

    def __init__(
        self,
        strategy_name: str,
        *,
        data_source: BacktestDataSource,
        clock: MarketClock,
        budget: Decimal,
        timestep: str = "day",
        commission: Decimal = Decimal(0),
        slippage: Decimal = Decimal(0),
        tracker: OrderTracker | None = None,
        news_source: NewsProvider | None = None,
    ) -> None:
        super().__init__(strategy_name, tracker, clock=clock, is_paper=True)
        self._data_source = data_source
        self._timestep = timestep
        self._commission = commission
        self._slippage = slippage
        self._cash = budget
        self._positions: dict[Asset, Position] = {}
        self._news_source = news_source
        self._pending: dict[str, _PendingOrder] = {}
        self.ledger = Ledger()

    # --- Broker ABC: orders ------------------------------------------------------------

    def _conform_order(self, order: Order) -> Order:
        return order  # no broker-specific tick rounding to simulate

    def _submit_order(self, order: Order) -> Order:
        if order.notional is not None:
            raise OrderValidationError(
                "backtesting only supports quantity-based orders, not notional orders"
            )
        # Reject against the projection (cash/holdings once every pending order fills) *before*
        # tracking, so a rejected order never enters the tracker and the caller -- an LLM agent
        # through `submit_order`, which turns the exception into `{"error": ...}` -- finds out
        # in the same iteration. Without this the only signal is the fill-time rejection below,
        # a session later, which the agent never sees and so repeats.
        projected_rejection = self._projected_rejection_reason(order)
        if projected_rejection is not None:
            raise OrderValidationError(projected_rejection)
        if not order.client_order_id:
            order.client_order_id = f"{self.strategy_name}:{order.identifier}"
        self.tracker.track_unprocessed(order)
        self.tracker.process_trade_event(order, OrderEvent.NEW)
        now = self.clock.now()
        found = self._latest_bar_with_time(order.asset, now)
        # Case 1: a bar closed exactly at `now` -- nothing is mid-formation, so the next bar to
        # close is already a safe fill bar. Case 2: no bar closed exactly at `now` (either none
        # has closed yet, or the latest one closed strictly earlier) -- a bar is currently
        # forming and must be skipped once before any fill (see `_PendingOrder.needs_skip`).
        needs_skip = found is None or found[1] < now
        self._pending[order.identifier] = _PendingOrder(
            order=order, asset=order.asset, last_evaluated=now, needs_skip=needs_skip
        )
        return order

    def cancel_order(self, order: Order) -> None:
        self._pending.pop(order.identifier, None)
        self.tracker.process_trade_event(order, OrderEvent.CANCELED)

    def pull_order(self, identifier: str) -> Order | None:
        return self.tracker.get_tracked_order(identifier)

    def pull_orders(self, limit: int = 100) -> list[Order]:
        return self.tracker.get_all_tracked_orders()[:limit]

    def pull_positions(self) -> list[Position]:
        return list(self._positions.values())

    def news_provider(self) -> NewsProvider | None:
        """The injected news source, else an Alpaca provider built from the env credentials on first use."""
        if self._news_source is None:
            # Deferred: importing this module must not pull in `alpaca`, and a backtest that never
            # calls the news tool needs no Alpaca credentials for news.
            from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
            from trading_agent_framework.config.env import AlpacaCredentials

            try:
                self._news_source = AlpacaNewsProvider.from_credentials(AlpacaCredentials.from_env())
            except ConfigurationError as exc:
                raise BrokerError(f"no news source available for backtesting: {exc}") from exc
        return self._news_source

    def get_account(self) -> AccountBalances:
        """Balances, with `buying_power` net of orders still pending.

        `cash` and `portfolio_value` stay the settled, raw values: `Strategy.get_cash()`
        reads `cash`, and `cross_momentum.rebalance` adds its *own* estimate of pending sell
        proceeds to it, so netting `cash` here would make that strategy double-count. Only
        `buying_power` moves -- which is what a real broker does with open orders anyway. An
        agent sizes a buy against the smaller of it and its own `cash` + same-run sell proceeds,
        never against raw `cash` alone (that ignores the sell credit).
        """
        now = self.clock.now()
        portfolio_value = self._portfolio_value(now)
        return AccountBalances(
            cash=self._cash, portfolio_value=portfolio_value, buying_power=self._projection(now).cash
        )

    def modify_order(
        self, order: Order, *, limit_price: Decimal | None = None, stop_price: Decimal | None = None
    ) -> Order:
        if order.identifier not in self._pending:
            raise BacktestError(f"order {order.identifier} is not pending; cannot modify")
        replacement = dataclasses.replace(
            order,
            identifier=uuid4().hex,
            limit_price=limit_price if limit_price is not None else order.limit_price,
            stop_price=stop_price if stop_price is not None else order.stop_price,
            status=OrderStatus.UNPROCESSED,
            transactions=[],
            filled_quantity=Decimal(0),
        )
        pending = self._pending.pop(order.identifier)
        pending.order = replacement
        self._pending[replacement.identifier] = pending
        self.tracker.mark_replaced(order, replacement)
        return replacement

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        position = self._positions.get(asset)
        if position is None:
            return None
        quantity = (position.quantity * fraction).copy_abs()
        side = OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY
        order = Order(strategy_name=self.strategy_name, asset=asset, side=side, quantity=quantity)
        return self.submit_order(order)

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        if cancel_orders:
            for pending in list(self._pending.values()):
                self.cancel_order(pending.order)
        closed: list[Order] = []
        for asset in list(self._positions):
            order = self.close_position(asset)
            if order is not None:
                closed.append(order)
        return closed

    def sync_open_orders(self) -> list[Order]:
        return []  # a fresh backtest has no prior state to adopt

    # --- Broker ABC: market data --------------------------------------------------------

    def get_last_price(self, asset: Asset) -> Decimal | None:
        bar = self._latest_bar(asset, self.clock.now())
        return bar.close if bar is not None else None

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        return {asset: self.get_last_price(asset) for asset in assets}

    def get_quote(self, asset: Asset) -> Quote | None:
        bar = self._latest_bar(asset, self.clock.now())
        if bar is None:
            return None
        return Quote(
            asset=asset, bid=bar.close, ask=bar.close, bid_size=None, ask_size=None,
            timestamp=self.clock.now(),
        )

    def get_bars(
        self, assets: Sequence[Asset], length: int, timestep: str = "day", *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        result: dict[Asset, Bars] = {}
        for asset in assets:
            bars = self._source_bars(asset, self.clock.now(), length, timestep)
            if bars is not None:
                result[asset] = bars
        return result

    def start_stream(self) -> None:
        pass

    def stop_stream(self, timeout: float = 5.0) -> None:
        pass

    # --- shared bar lookup -----------------------------------------------------------------

    def _source_bars(
        self, asset: Asset, cutoff: datetime, length: int, timestep: str
    ) -> Bars | None:
        """THE gate (design spec, section 5.2): the single place this broker -- and so
        the strategy, whose every price request funnels through `get_bars`,
        `get_last_price`, `get_quote` or the fill engine -- reads a
        `BacktestDataSource`, plus the spec's defensive assertion on what comes back.

        `bars()` is contractually "the last `length` bars closed at or before
        `cutoff`", and every source enforces that itself. This re-checks it anyway,
        once, because the cost of a source getting it wrong is not an exception but a
        backtest that looks great and is worthless -- "cheap insurance against a future
        provider with a sloppy index", as the spec puts it. It is exactly the net that
        would have caught `AlpacaBacktestData`'s bar-open fallback (fixed at its own
        root in `data/alpaca.py`) had it produced a *future*-dated row rather than a
        past-dated one.

        Only the "past the cutoff" direction is checked, per the spec. A deliberately
        absent "too far in the past" check has no principled threshold: a source
        legitimately returns a stale latest bar over a weekend, a holiday, a halt, or
        for a thinly-traded or delisted symbol, so any bound tight enough to catch a
        mis-indexed bar would also fire on ordinary data. That failure mode is
        addressed where it can be decided correctly -- at the source, which alone knows
        a bar's date has no session -- not by a heuristic here.
        """
        bars = self._data_source.bars(asset, cutoff, length, timestep)
        if bars is None or bars.df.empty:
            return bars
        try:
            latest = bars.df.index.max()
            past_cutoff = bool(latest > cutoff)
        except TypeError as exc:  # e.g. a tz-naive index compared against a tz-aware cutoff
            raise BacktestDataError(
                f"{self._data_source.name} returned bars for {asset.symbol} whose index "
                f"cannot be compared against the {cutoff.isoformat()} cutoff: {exc}"
            ) from exc
        if past_cutoff:
            raise BacktestDataError(
                f"{self._data_source.name} returned a bar for {asset.symbol} closing at "
                f"{latest}, after the {cutoff.isoformat()} no-look-ahead cutoff; "
                "a data source must only return bars closed at or before the cutoff"
            )
        return bars

    def _latest_bar_with_time(
        self, asset: Asset, cutoff: datetime
    ) -> tuple[fills.Bar, datetime] | None:
        bars = self._source_bars(asset, cutoff, 1, self._timestep)
        if bars is None or bars.df.empty:
            return None
        row = bars.df.iloc[-1]
        bar = fills.Bar(
            open=Decimal(str(row["open"])), high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])), close=Decimal(str(row["close"])),
        )
        return bar, bars.df.index[-1].to_pydatetime()

    def _latest_bar(self, asset: Asset, cutoff: datetime) -> fills.Bar | None:
        found = self._latest_bar_with_time(asset, cutoff)
        return None if found is None else found[0]

    def _portfolio_value(self, cutoff: datetime) -> Decimal:
        return self._cash + self._positions_value(cutoff)

    def _positions_value(self, cutoff: datetime) -> Decimal:
        total = Decimal(0)
        for asset, position in self._positions.items():
            bar = self._latest_bar(asset, cutoff)
            price = bar.close if bar is not None else (position.avg_fill_price or Decimal(0))
            signed_qty = position.quantity if position.side is PositionSide.LONG else -position.quantity
            total += signed_qty * price
        return total

    # --- pending-order projection -------------------------------------------------------

    def _estimated_price(self, order: Order, cutoff: datetime) -> Decimal | None:
        """Best guess at what `order` will fill at, or `None` when no price is available.

        A limit price is the bound the fill engine itself honours; everything else is
        estimated at the latest close, read through `_source_bars` so the no-look-ahead
        gate governs this path exactly like every other price read.
        """
        if order.limit_price is not None:
            return order.limit_price
        bar = self._latest_bar(order.asset, cutoff)
        return None if bar is None else bar.close

    def _projection(self, cutoff: datetime) -> _Projection:
        """Cash and per-asset sell quantity once every currently pending order has filled.

        Pending sells are *credited* here, deliberately: orders fill in submission order, so a
        rebalance that submits its sells before its buys really is funded by those sells (the
        reason the hard check lives at fill time in the first place). An order whose price can't
        be estimated contributes its quantity but no cash; `_rejection_reason` remains the net.
        """
        cash = self._cash
        pending_sells: dict[Asset, Decimal] = {}
        for pending in self._pending.values():
            order = pending.order
            if order.quantity is None:  # notional orders never reach `_pending`
                continue
            if order.side is OrderSide.SELL:
                pending_sells[order.asset] = pending_sells.get(order.asset, Decimal(0)) + order.quantity
            price = self._estimated_price(order, cutoff)
            if price is None:
                continue
            _, commission_cost, notional = self._execution_terms(order, price)
            if order.side is OrderSide.BUY:
                cash -= notional + commission_cost
            else:
                cash += notional - commission_cost
        return _Projection(cash=cash, pending_sells=pending_sells)

    def _projected_rejection_reason(self, order: Order) -> str | None:
        """Why submitting `order` is already impossible given what is pending, else `None`.

        The submission-time mirror of `_rejection_reason`, which stays the authority at fill
        time: this one works off estimated prices, so it only rejects what is *certainly*
        impossible and lets anything it cannot price through.
        """
        if order.quantity is None:
            return None
        symbol = order.asset.symbol
        now = self.clock.now()
        projection = self._projection(now)
        if order.side is OrderSide.SELL:
            existing = self._positions.get(order.asset)
            held = existing.quantity if existing is not None and existing.side is PositionSide.LONG else Decimal(0)
            available = held - projection.pending_sells.get(order.asset, Decimal(0))
            if order.quantity > available:
                pending_note = "" if available == held else f" ({held} held, the rest already pending sale)"
                return (
                    f"insufficient position: selling {order.quantity} {symbol}, "
                    f"available {available}{pending_note}"
                )
            return None
        price = self._estimated_price(order, now)
        if price is None:
            return None  # unpriceable: let it through and let the fill-time check decide
        _, commission_cost, notional = self._execution_terms(order, price)
        needed = notional + commission_cost
        if needed > projection.cash:
            return (
                f"insufficient buying power: buying {order.quantity} {symbol} needs about "
                f"{needed} (commission included), buying power is {projection.cash}"
            )
        return None

    # --- fills: called by BacktestClock.on_advance --------------------------------------

    def on_advance(self, previous_now: datetime, new_now: datetime) -> None:
        """Registered as `BacktestClock.on_advance`: process fills, then sample equity."""
        self._process_pending(new_now)
        self._sample_equity(new_now)

    def _process_pending(self, cutoff: datetime) -> None:
        for identifier in list(self._pending):
            pending = self._pending[identifier]
            found = self._latest_bar_with_time(pending.asset, cutoff)
            if found is None:
                continue
            bar, bar_time = found
            if bar_time <= pending.last_evaluated:
                continue  # no new bar has closed for this asset since we last checked
            pending.last_evaluated = bar_time
            if pending.needs_skip:
                # This is the bar that was still forming at submission time -- "nothing fills
                # within the submitting bar" (design spec section 2/6.1). Skip it once; the
                # order becomes eligible starting with the next bar found after this one.
                pending.needs_skip = False
                continue
            order = pending.order
            try:
                result = fills.evaluate_fill(
                    order_type=order.order_type, side=order.side, bar=bar,
                    limit_price=order.limit_price, stop_price=order.stop_price,
                    stop_limit_price=order.stop_limit_price,
                )
            except ValueError as exc:
                order.set_error(exc)
                self.tracker.process_trade_event(order, OrderEvent.ERROR)
                del self._pending[identifier]
                continue
            if result is None:
                continue  # still doesn't touch the trigger; retried on the next bar
            rejection = self._rejection_reason(order, result.price)
            if rejection is not None:
                # Long-only cash account: the fill would overdraw cash or sell what isn't held. Like a
                # real broker's rejection, the order fails whole and nothing (cash, positions, ledger) moves.
                logger.warning("Rejected order %s at %s: %s", order.identifier, bar_time.isoformat(), rejection)
                order.set_error(BrokerError(rejection))
                self.tracker.process_trade_event(order, OrderEvent.ERROR)
                del self._pending[identifier]
                continue
            self._fill(order, result.price, bar_time)
            del self._pending[identifier]

    def _execution_terms(self, order: Order, raw_price: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        """`(execution_price, commission_cost, notional)` for filling `order` at `raw_price`."""
        assert order.quantity is not None  # notional orders are rejected at submission
        execution_price, commission_per_share = fills.apply_commission_and_slippage(
            raw_price, order.side, commission=self._commission, slippage=self._slippage
        )
        return execution_price, commission_per_share * order.quantity, execution_price * order.quantity

    def _rejection_reason(self, order: Order, raw_price: Decimal) -> str | None:
        """Why filling `order` at `raw_price` is impossible for a long-only cash account, else `None`."""
        assert order.quantity is not None
        symbol = order.asset.symbol
        _, commission_cost, notional = self._execution_terms(order, raw_price)
        if order.side is OrderSide.BUY:
            needed = notional + commission_cost
            if needed > self._cash:
                return f"insufficient cash: buying {order.quantity} {symbol} needs {needed} (commission included), cash is {self._cash}"
            return None
        existing = self._positions.get(order.asset)
        held = existing.quantity if existing is not None and existing.side is PositionSide.LONG else Decimal(0)
        if order.quantity > held:
            return f"insufficient position: selling {order.quantity} {symbol}, holding {held}"
        return None

    def _fill(self, order: Order, raw_price: Decimal, bar_time: datetime) -> None:
        assert order.quantity is not None  # notional orders are rejected at submission
        execution_price, commission_cost, notional = self._execution_terms(order, raw_price)
        quantity = order.quantity
        if order.side is OrderSide.BUY:
            self._cash -= notional + commission_cost
        else:
            self._cash += notional - commission_cost
        self._apply_to_position(order.asset, order.side, quantity, execution_price)
        self.ledger.record_fill(FillRecord(
            time=bar_time, identifier=order.identifier, symbol=order.asset.symbol,
            side=order.side, order_type=order.order_type, quantity=order.quantity,
            filled_quantity=quantity, price=execution_price, trade_cost=commission_cost,
            trade_slippage=(execution_price - raw_price).copy_abs(),
        ))
        order.avg_fill_price = execution_price
        self.tracker.process_trade_event(
            order, OrderEvent.FILLED, price=execution_price, filled_quantity=quantity
        )

    def _apply_to_position(self, asset: Asset, side: OrderSide, quantity: Decimal, price: Decimal) -> None:
        existing = self._positions.get(asset)
        held = existing.quantity if existing is not None else Decimal(0)
        new_quantity = held + quantity if side is OrderSide.BUY else held - quantity
        assert new_quantity >= 0  # long-only: `_rejection_reason` never lets a sell exceed the holding
        if new_quantity == 0:
            self._positions.pop(asset, None)
            return
        self._positions[asset] = Position(
            strategy_name=self.strategy_name, asset=asset, quantity=new_quantity,
            side=PositionSide.LONG, avg_fill_price=price,
        )

    def _sample_equity(self, cutoff: datetime) -> None:
        positions_value = self._positions_value(cutoff)
        self.ledger.record_equity(EquitySample(
            time=cutoff, portfolio_value=self._cash + positions_value,
            cash=self._cash, positions_value=positions_value,
        ))
