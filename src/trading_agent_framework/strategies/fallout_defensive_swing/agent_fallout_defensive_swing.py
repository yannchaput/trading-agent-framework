"""fallout_defensive_swing: hold a defensive ETF mix, sell it at a small profit, buy it back after a dip or a few days. No LLM.

Every 30 minutes `rule.decide` picks buy, sell or hold. Built for a cash account: it never sells on the session of its last sale,
because what a sale's unsettled proceeds bought must not be sold before they settle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime, time
from decimal import ROUND_DOWN, Decimal
from typing import Any

from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.fees import TradingFeeFactory
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities import Order, OrderSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.fallout_defensive_swing.rule import Action, Phase, SwingState, decide, mean_price_ratio, sell_target, trading_days_since
from trading_agent_framework.strategies.fallout_defensive_swing.store import StateStore
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import TradingFrameworkError

ORDER_WAIT_SECONDS = 300  # a market order fills in seconds; after this it is canceled and retried at a later tick
_SHARE_STEP = Decimal("0.000001")


class FalloutDefensiveSwingStrategy(Strategy):
    """Equal parts of a few defensive ETFs: sold at +`take_profit` net of fees, bought back `buyback_dip` lower."""

    parameters = {
        "symbols": ["XLP", "XLU"],
        "take_profit": Decimal("0.01"),
        "buyback_dip": Decimal("0.01"),
        "cash_reset_days": 10,  # trading days in cash before buying back without the dip; None never does
        "budget": 768,
        "benchmark_symbol": "SPY",
        "backtesting_start": datetime(2026, 11, 3, tzinfo=MARKET_TZ),
        "backtesting_end": datetime(2026, 11, 25, tzinfo=MARKET_TZ),
    }

    def __init__(self, broker: Broker, *, mode: TradingMode = TradingMode.PAPER, fees: TradingFeeFactory | None = None, **kwargs: Any) -> None:
        """Wire the strategy.

        Args:
            broker: Broker facade.
            mode: Trading mode.
            fees: Fee model used to net the sell fees out of the target and to add the buy fees to the basis;
                the one of `BROKER` when omitted.
            kwargs: Passed to `Strategy`.
        """
        super().__init__(broker, mode=mode, **kwargs)
        self.fees = fees if fees is not None else TradingFeeFactory.from_env()

    def initialize(self) -> None:
        """Tick every 30 minutes from 10:30 market time and open the state file (a backtest starts from an empty one)."""
        self.sleeptime = "30M"
        self.iteration_start_time = time(10, 30)
        self.store = StateStore(self.state_file(), log=self._log, fresh=self.is_backtesting)

    def on_trading_iteration(self) -> None:
        """One check: read the account, correct the saved state, then buy, sell or hold."""
        open_orders = self.broker.tracker.get_active_orders()
        if open_orders:
            self.log_warning(f"{len(open_orders)} order(s) still open: nothing is done until they are final")
            return
        symbols = self.parameters["symbols"]
        session = self.clock.now().astimezone(MARKET_TZ).date()
        held = {p.asset.symbol: p.quantity for p in self.get_positions() if p.asset.symbol in symbols and p.quantity > 0}
        latest = {asset.symbol: price for asset, price in self.get_last_prices(symbols).items()}
        prices: dict[str, Decimal] = {}
        for symbol in symbols:
            price = latest[symbol]
            if price is None:
                self.log_warning(f"no price for {symbol}: nothing is done this tick")
                return
            prices[symbol] = price
        value = sum((quantity * prices[symbol] for symbol, quantity in held.items()), Decimal(0))
        state = self._reconcile(held, prices, value, session)
        sell_fees = sum((self._sell_fee(quantity, quantity * prices[symbol]) for symbol, quantity in held.items()), Decimal(0))
        action = decide(
            state,
            value=value,
            sell_fees=sell_fees,
            prices=prices,
            session=session,
            take_profit=self.parameters["take_profit"],
            buyback_dip=self.parameters["buyback_dip"],
            cash_reset_days=self.parameters["cash_reset_days"],
        )
        self.log_info(self._describe(state, action, value - sell_fees, prices))
        if action is Action.RESET:
            assert state.last_sale_session is not None  # `decide` counts the reset from it
            self.log_warning(f"Strategy resets after {trading_days_since(state.last_sale_session, session)} days without trading activity")
        if action in (Action.BUY, Action.RESET):
            self._buy(state, prices)
        elif action is Action.SELL:
            self._sell(state, held, prices, session)

    def on_new_order(self, order: Order) -> None:
        """Called when the broker accepts an order."""
        self.log_info(f"submitted {order.side} {order.asset.symbol} {order.quantity}")

    def on_partially_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        """Called on each partial fill; `quantity` is the size of this fill."""
        self.log_info(f"partially filled {order.side} {order.asset.symbol} {quantity} at {price}")

    def on_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        """Called when an order is completely filled."""
        self.log_info(f"filled {order.side} {order.asset.symbol} {quantity} at {price}")

    def _reconcile(self, held: Mapping[str, Decimal], prices: dict[str, Decimal], value: Decimal, session: date) -> SwingState:
        """The saved state, corrected to what the account holds (restart, manual trade, late fill)."""
        state = self.store.load()
        if held and state.phase is not Phase.HOLDING:
            self.log_warning(f"the account holds the mix but the state says {state.phase}: adopting it with a basis of {value:.2f}")
            state = replace(state, phase=Phase.HOLDING, basis=value, selling=False)
        elif not held and state.phase is Phase.HOLDING:
            # Sold outside this tick, perhaps today: dated today so that nothing bought now is sold before tomorrow.
            self.log_warning("the state says holding but the account has none of the mix: adopting cash, buy-back below today's prices")
            state = replace(state, phase=Phase.CASH, sale_prices=prices, last_sale_session=session, selling=False)
        else:
            return state
        self.store.save(state)
        return state

    def _buy(self, state: SwingState, prices: Mapping[str, Decimal]) -> None:
        """Split the spendable cash equally over the mix; the filled cost, fees included, is the new basis."""
        per_symbol = self.spendable_cash() / len(prices)
        orders: list[Order] = []
        for symbol, price in prices.items():
            # read again: the earlier buy filled above its sizing price, so the cash left may be under `per_symbol`
            quantity = (min(per_symbol, self.spendable_cash()) / price).quantize(_SHARE_STEP, ROUND_DOWN)
            if quantity <= 0:
                self.log_warning(f"no cash to buy {symbol}")
                continue
            order = self._send(symbol, quantity, OrderSide.BUY)
            if order is not None:
                self._wait_or_cancel(order)
                orders.append(order)
        if self.broker.tracker.get_active_orders():
            return  # still open: saving now could miss a later fill; a later tick adopts what the account holds
        basis = Decimal(0)
        for order in orders:
            if order.filled_quantity > 0:
                cost = order.filled_quantity * (order.avg_fill_price or prices[order.asset.symbol])
                basis += cost + self._buy_fee(order.filled_quantity, cost)
        if basis > 0:
            self.store.save(replace(state, phase=Phase.HOLDING, basis=basis, selling=False))

    def _sell(self, state: SwingState, held: Mapping[str, Decimal], prices: Mapping[str, Decimal], session: date) -> None:
        """Sell everything held; if only part fills, `selling` stays set and the rest goes at the next allowed tick."""
        orders = [order for symbol, quantity in held.items() if (order := self._send(symbol, quantity, OrderSide.SELL)) is not None]
        for order in orders:
            self._wait_or_cancel(order)
        sold = {o.asset.symbol: o.avg_fill_price or prices[o.asset.symbol] for o in orders if o.filled_quantity > 0}
        if not sold:
            return
        # saved even with an order still open: ticks wait for it, and losing the record of what was sold would leave the rest stuck
        complete = len(orders) == len(held) and all(order.is_filled() for order in orders)
        earlier = state.sale_prices if state.selling else {}  # the legs an unfinished sale already sold
        self.store.save(
            replace(
                state,
                phase=Phase.CASH if complete else Phase.HOLDING,
                sale_prices={**earlier, **sold},
                last_sale_session=session,
                selling=not complete,
            )
        )

    def _buy_fee(self, shares: Decimal, value: Decimal) -> Decimal:
        """Fees of one buy order."""
        return self.fees.fees(buy_shares=shares, sell_shares=Decimal(0), buy_value=value, sell_value=Decimal(0)).buy

    def _sell_fee(self, shares: Decimal, value: Decimal) -> Decimal:
        """Fees of one sell order."""
        return self.fees.fees(buy_shares=Decimal(0), sell_shares=shares, buy_value=Decimal(0), sell_value=value).sell

    def _send(self, symbol: str, quantity: Decimal, side: OrderSide) -> Order | None:
        """Submit a market order; None, with a warning, when it raises, so the tick still records the other leg."""
        try:
            return self.submit_order(self.create_order(symbol, quantity, side))
        except TradingFrameworkError as exc:
            self.log_warning(f"submitting {side} {symbol} failed (check the account): {exc}")
            return None

    def _wait_or_cancel(self, order: Order) -> None:
        """Wait for a final status, canceling an order that takes too long."""
        if self.wait_for_order_execution(order, ORDER_WAIT_SECONDS):
            return
        self.log_warning(f"{order.side} {order.asset.symbol} not final after {ORDER_WAIT_SECONDS}s: canceling it")
        try:
            self.cancel_order(order)
        except TradingFrameworkError as exc:
            self.log_warning(f"could not cancel {order.side} {order.asset.symbol} (it may have filled): {exc}. If later ticks keep reporting an open order, check the account and restart")
            return
        self.wait_for_order_execution(order, ORDER_WAIT_SECONDS)

    def _describe(self, state: SwingState, action: Action, net_value: Decimal, prices: Mapping[str, Decimal]) -> str:
        """One line saying where the strategy stands and what it does."""
        if state.phase is Phase.HOLDING and state.basis is not None:
            detail = f", value after sell fees {net_value:.2f} ON a target {sell_target(state.basis, self.parameters['take_profit']):.2f}"
        elif state.phase is Phase.CASH:
            ratio = mean_price_ratio(prices, state.sale_prices)
            detail = f", price vs last sale {ratio - 1:+.2%} / buy-back at {-self.parameters['buyback_dip']:+.2%}"
        else:
            detail = ""
        return f"Phase [{state.phase}]{detail} -> {action}"

    def run_backtesting(self, **overrides: Any):
        """Backtest over the class `parameters` window on Alpaca minute bars."""
        defaults: dict[str, Any] = dict(data_source=AlpacaBacktestData, timestep="minute", agent_telemetry=False)
        return super().run_backtesting(**{**defaults, **overrides})
