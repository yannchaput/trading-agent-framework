"""Lean port of lumibot's `Strategy` template.

Subclasses override the lifecycle hooks (`initialize`, `on_trading_iteration`,
`before_market_opens`, ...) and the order-event hooks. Everything else is a thin
facade over the broker, so strategy code reads like lumibot strategy code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.stats_store import LLMStatsStore, llm_stats_db_path
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode, find_project_root
from trading_agent_framework.core.executor import StrategyExecutor
from trading_agent_framework.core.indicators import Indicators
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import (
    ACTIVE_ORDER_STATUSES,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.memory.store import MemoryStore, memory_db_path
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, LLMStatsError
from trading_agent_framework.utils.log import ColorLogger, setup_strategy_logging

if TYPE_CHECKING:
    from trading_agent_framework.backtesting.data.base import BacktestDataSource
    from trading_agent_framework.backtesting.runner import BacktestResult
    from trading_agent_framework.brokers.news import NewsProvider

logger = logging.getLogger(__name__)

Number = Decimal | int | float | str

_FINAL_STATUSES = frozenset(OrderStatus) - ACTIVE_ORDER_STATUSES


def _to_asset(asset: Asset | str) -> Asset:
    return asset if isinstance(asset, Asset) else Asset(asset)


def _to_decimal(value: Number) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _to_optional_decimal(value: Number | None) -> Decimal | None:
    return None if value is None else _to_decimal(value)


def _format_duration(delta: timedelta) -> str:
    """HH:MM:SS, with hours allowed past 24 (a weekend wait reads 65:30:00)."""
    total = max(0, int(delta.total_seconds()))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Strategy:
    """Base class for strategies, with lumibot's hook names and signatures."""

    sleeptime: int | str = "1M"
    minutes_before_opening: int = 60
    minutes_before_closing: int = 1
    minutes_after_closing: int = 0
    parameters: Mapping[str, Any] = MappingProxyType({})

    # backtesting defaults (Strategy.run_backtesting()), overridable per call
    backtesting_start: datetime | None = None
    backtesting_end: datetime | None = None
    budget: Decimal = Decimal("10000")
    benchmark_symbol: str = "SPY"

    # Agent call telemetry (tokens, latency, tool calls per model call) -> `memory/<strategy>/<mode>/llm_stats.sqlite`.
    # Off here for paper/live (opt in with `agent_telemetry = True`); `run_backtesting(agent_telemetry=True)` turns it on per run.
    agent_telemetry: bool = False

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        parameters: Mapping[str, Any] | None = None,
        clock: MarketClock | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.broker = broker
        self.trading_mode = mode
        # Copy all parameters passed to __init__ into self.parameters, overriding class-level defaults
        # This attribute only accepts framework dependant parameters.
        self.parameters = {**type(self).parameters, **(parameters or {})}
        self.clock = clock if clock is not None else broker.clock
        self.project_root = project_root or find_project_root()
        # A container for strategy-specific variables that can be set and read by the user; not persisted.
        # It is not used by the framework.
        self.vars = SimpleNamespace()  # Store all strategy specific parameters
        self.first_iteration = True
        self._log = ColorLogger(logger, self.name)
        self._indicators: Indicators | None = None
        self._memory: MemoryStore | None = None
        self._memory_mode: TradingMode | None = None
        self._agents: AgentManager | None = None
        # Name of the run directory (`<ts>_<mode>`), set by the runners; tags every llm_stats row.
        self.run_id: str | None = None
        self.executor = StrategyExecutor(self)

    @property
    def name(self) -> str:
        return self.broker.strategy_name

    @property
    def is_backtesting(self) -> bool:
        return self.trading_mode is TradingMode.BACKTESTING

    @property
    def indicators(self) -> Indicators:
        """pandas-ta-classic indicators over this strategy's bars, e.g. `indicators.sma(a, length=20)`."""
        if self._indicators is None:
            self._indicators = Indicators(self)
        return self._indicators

    @property
    def memory(self) -> MemoryStore:
        """Agent memory for this strategy and trading mode (lumibot's `strategy.memory`).

        Opened lazily at `memory/<strategy>/<mode>/memory.sqlite`; a backtest run starts empty.
        """
        if self._memory is None or self._memory_mode is not self.trading_mode:
            root = self.project_root if self.project_root is not None else find_project_root()
            self._memory = MemoryStore(
                memory_db_path(root, self.name, self.trading_mode),
                strategy_name=self.name,
                now=self.clock.now,
                fresh=self.is_backtesting,
            )
            self._memory_mode = self.trading_mode
        return self._memory

    @property
    def agents(self) -> AgentManager:
        """This strategy's LLM agents (lumibot's `strategy.agents`); built on first use."""
        if self._agents is None:
            self._agents = AgentManager(LLMCredentials.from_env)
            if self.agent_telemetry:
                self._agents.enable_telemetry(now=lambda: self.clock.now(), store=self._open_llm_stats_store())
        return self._agents

    def agent_telemetry_summary(self) -> dict[str, dict[str, Any]]:
        """Per-agent totals of the model calls recorded so far; empty when telemetry is off or no agent ran."""
        return self._agents.telemetry_summary() if self._agents is not None else {}

    def _open_llm_stats_store(self) -> LLMStatsStore | None:
        """Opened like `memory`: `llm_stats.sqlite` per strategy and mode, wiped when a backtest starts.

        Telemetry is observational, so an unusable database only costs the per-call rows (warned about),
        and a strategy run outside a runner (no `run_id`) keeps its in-memory totals only.
        """
        if self.run_id is None:
            return None
        try:
            return LLMStatsStore(llm_stats_db_path(self.project_root, self.name, self.trading_mode), run_id=self.run_id, fresh=self.is_backtesting)
        except LLMStatsError as exc:
            self.log_warning(f"LLM stats database unavailable, per-call agent telemetry will not be saved: {exc}")
            return None

    # --- lifecycle hooks -------------------------------------------------------

    def initialize(self) -> None:
        """Called once before trading starts; receives matching `parameters` as kwargs."""

    def on_trading_iteration(self) -> None:
        """Called every `sleeptime` while the market is open."""

    def before_market_opens(self) -> None:
        """Called `minutes_before_opening` before each session opens."""

    def before_starting_trading(self) -> None:
        """Called when each session opens, before its first iteration."""

    def before_market_closes(self) -> None:
        """Called `minutes_before_closing` before each session closes."""

    def after_market_closes(self) -> None:
        """Called `minutes_after_closing` after each session closes."""

    def on_strategy_end(self) -> None:
        """Called once when the run ends (stop, interrupt, or no more sessions)."""

    def on_bot_crash(self, error: BaseException) -> None:
        """Called when a hook raises; like lumibot, defaults to `on_abrupt_closing()`."""
        self.on_abrupt_closing()

    def on_abrupt_closing(self) -> None:
        """Called on Ctrl+C / SIGTERM, and by the default `on_bot_crash`."""

    # --- order-event hooks -----------------------------------------------------

    def on_new_order(self, order: Order) -> None:
        """Called when the broker accepts an order."""

    def on_canceled_order(self, order: Order) -> None:
        """Called when an order is canceled or expires."""

    def on_partially_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        """Called on each partial fill; `quantity` is the size of this fill."""

    def on_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        """Called when an order is completely filled."""

    # --- logging ---------------------------------------------------------------

    def log_debug(self, message: object) -> str:
        return self._log.log_debug(message, stacklevel=2)

    def log_info(self, message: object) -> str:
        return self._log.log_info(message, stacklevel=2)

    def log_warning(self, message: object) -> str:
        return self._log.log_warning(message, stacklevel=2)

    def log_error(self, message: object) -> str:
        return self._log.log_error(message, stacklevel=2)

    def log_critical(self, message: object) -> str:
        return self._log.log_critical(message, stacklevel=2)

    # --- control -------------------------------------------------------------------

    def sleep(self, seconds: float) -> None:
        """Pause for `seconds` of clock time; order hooks still fire meanwhile."""
        self.executor.wait_until(self.get_datetime() + timedelta(seconds=seconds))

    def stop(self) -> None:
        """End the run once the current hook returns; `on_strategy_end` still runs."""
        self.executor.stop()

    def add_line(
        self,
        name: str,
        value: Number,
        *,
        color: str | None = None,
        style: str = "solid",
        plot_name: str = "default_plot",
    ) -> None:
        """Record a charted value at the current simulated time (lumibot-compatible
        signature). No-op outside backtesting."""
        if not self.is_backtesting:
            return
        from trading_agent_framework.backtesting.broker import BacktestBroker
        from trading_agent_framework.backtesting.ledger import IndicatorLine

        if not isinstance(self.broker, BacktestBroker):
            return
        self.broker.ledger.record_line(
            IndicatorLine(
                time=self.clock.now(),
                name=name,
                value=_to_decimal(value),
                color=color,
                style=style,
                plot_name=plot_name,
            )
        )

    def wait_for_order_execution(self, order: Order, timeout: float | None = None) -> bool:
        """Wait until `order` is filled, canceled, expired or rejected; False on timeout/stop.

        Needs the broker's trade stream (the runners start it). Order hooks keep firing
        while waiting. A modified order is replaced: wait on the order `modify_order` returned.
        """
        return self.wait_for_orders_execution([order], timeout)

    def wait_for_orders_execution(self, orders: Sequence[Order], timeout: float | None = None) -> bool:
        return self.executor.wait_for(lambda: all(order.status in _FINAL_STATUSES for order in orders), timeout)

    # --- accounting --------------------------------------------------------------

    def get_datetime(self) -> datetime:
        return self.clock.now()

    def get_cash(self) -> Decimal:
        return self.broker.get_account().cash

    def get_portfolio_value(self) -> Decimal:
        return self.broker.get_account().portfolio_value

    @property
    def cash(self) -> Decimal:
        return self.get_cash()

    @property
    def portfolio_value(self) -> Decimal:
        return self.get_portfolio_value()

    def get_positions(self) -> list[Position]:
        return self.broker.pull_positions()

    def get_position(self, asset: Asset | str) -> Position | None:
        symbol = _to_asset(asset).symbol
        return next((p for p in self.get_positions() if p.asset.symbol == symbol), None)

    def get_orders(self) -> list[Order]:
        return self.broker.tracker.get_all_tracked_orders()

    def get_order(self, identifier: str) -> Order | None:
        tracked = self.broker.get_tracked_order(identifier)
        return tracked if tracked is not None else self.broker.pull_order(identifier)

    # --- market data -----------------------------------------------------------------

    def get_last_price(self, asset: Asset | str) -> Decimal | None:
        """Last traded price; for the quote midpoint use `get_quote(asset).mid`."""
        return self.broker.get_last_price(_to_asset(asset))

    def get_last_prices(self, assets: Iterable[Asset | str]) -> dict[Asset, Decimal | None]:
        return self.broker.get_last_prices([_to_asset(asset) for asset in assets])

    def get_quote(self, asset: Asset | str) -> Quote | None:
        return self.broker.get_quote(_to_asset(asset))

    def get_historical_prices(
        self,
        asset: Asset | str,
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> Bars | None:
        """The last `length` "minute" or "day" bars, oldest first; None without data."""
        target = _to_asset(asset)
        bars = self.broker.get_bars([target], length, timestep, include_after_hours=include_after_hours)
        return bars.get(target)

    def get_historical_prices_for_assets(
        self,
        assets: Iterable[Asset | str],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        return self.broker.get_bars(
            [_to_asset(asset) for asset in assets],
            length,
            timestep,
            include_after_hours=include_after_hours,
        )

    # --- trading -----------------------------------------------------------------

    def create_order(
        self,
        asset: Asset | str,
        quantity: Number,
        side: OrderSide | str,
        *,
        limit_price: Number | None = None,
        stop_price: Number | None = None,
        time_in_force: TimeInForce | str = TimeInForce.DAY,
    ) -> Order:
        """Build (not submit) an order; the type follows from the prices given."""
        limit = _to_optional_decimal(limit_price)
        stop = _to_optional_decimal(stop_price)
        if limit is not None and stop is not None:
            order_type = OrderType.STOP_LIMIT
        elif limit is not None:
            order_type = OrderType.LIMIT
        elif stop is not None:
            order_type = OrderType.STOP
        else:
            order_type = OrderType.MARKET
        is_stop_limit = order_type is OrderType.STOP_LIMIT
        return Order(
            strategy_name=self.name,
            asset=_to_asset(asset),
            side=OrderSide(side),
            order_type=order_type,
            quantity=_to_decimal(quantity),
            time_in_force=TimeInForce(time_in_force),
            limit_price=None if is_stop_limit else limit,
            stop_price=stop,
            stop_limit_price=limit if is_stop_limit else None,
        )

    def submit_order(self, order: Order) -> Order:
        return self.broker.submit_order(order)

    def submit_orders(self, orders: Sequence[Order]) -> list[Order]:
        return self.broker.submit_orders(orders)

    def cancel_order(self, order: Order) -> None:
        self.broker.cancel_order(order)

    def cancel_orders(self, orders: Iterable[Order]) -> None:
        for order in orders:
            self.cancel_order(order)

    def cancel_open_orders(self) -> None:
        self.cancel_orders(self.broker.tracker.get_active_orders())

    def modify_order(self, order: Order, limit_price: Number | None = None, stop_price: Number | None = None) -> Order:
        """Change an open order's prices; returns the replacement order (new identifier)."""
        return self.broker.modify_order(
            order,
            limit_price=_to_optional_decimal(limit_price),
            stop_price=_to_optional_decimal(stop_price),
        )

    def sell_all(self, cancel_open_orders: bool = True) -> list[Order]:
        return self.broker.close_all_positions(cancel_orders=cancel_open_orders)

    def close_position(self, asset: Asset | str, fraction: Number = 1) -> Order | None:
        return self.broker.close_position(_to_asset(asset), _to_decimal(fraction))

    def close_positions(self, assets: Iterable[Asset | str] | None = None) -> list[Order]:
        """Close each given asset's position; `None` means every position currently held."""
        targets = [p.asset for p in self.get_positions()] if assets is None else assets
        orders = [self.close_position(asset) for asset in targets]
        return [order for order in orders if order is not None]

    # --- runners (the lumibot-agent `WrappingStrategy` entry points) -------------------

    def run_strategy(self) -> None:
        if self.trading_mode is TradingMode.LIVE:
            self.run_live_trading()
        elif self.trading_mode is TradingMode.PAPER:
            self.run_paper_trading()
        else:
            self.run_backtesting()

    def run_paper_trading(self) -> None:
        self._run_trading(TradingMode.PAPER)

    def run_live_trading(self) -> None:
        self._run_trading(TradingMode.LIVE)

    def run_backtesting(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        budget: Number | None = None,
        data_source: BacktestDataSource | Callable[[datetime, datetime], BacktestDataSource] | None = None,
        preload_assets: Sequence[Asset] = (),
        benchmark: str | None = None,
        timestep: str = "day",
        commission: Number = Decimal(0),
        slippage: Number = Decimal(0),
        risk_free_rate: float = 0.0,
        warmup_trading_days: int = 0,
        news_source: NewsProvider | None = None,
        agent_telemetry: bool = True,
    ) -> BacktestResult:
        """Run this strategy against simulated time and simulated fills.

        `start`/`end`/`budget`/`benchmark` fall back to the `backtesting_start`/
        `backtesting_end`/`budget`/`benchmark_symbol` class attributes when omitted.

        Args:
            start: first simulated datetime (inclusive)
            end: last simulated datetime (inclusive)
            budget: starting cash for the backtest
            data_source: source of historical market data. One of:
                - omitted/`None` -- defaults to `YahooBacktestData`, constructed with
                  the warmup-widened `[start, end]` window described below.
                - a class/callable taking `(start, end)` -- called with that same
                  widened window, e.g. `data_source=YahooBacktestData` or
                  `data_source=AlpacaBacktestData` (both default their remaining
                  constructor args -- Alpaca's client/trading_client are built from
                  `AlpacaCredentials.from_env()` when omitted).
                - an already-built `BacktestDataSource` instance -- used as given,
                  NOT widened for warmup (construct it with your own window first).
                No on-disk cache by default -- wrap the result in
                `backtesting.CachedDataSource` for repeat runs.
            preload_assets: extra assets (e.g. a strategy's universe) to batch-fetch
                up front alongside the benchmark, in the same warmup-widened
                `load(...)` call `run_backtest` already makes -- avoids a strategy
                having to construct and preload its own data source just to avoid
                the per-ticker lazy-fetch path.
            benchmark: symbol to use for the backtest's benchmark performance (e.g SPY)
            timestep: "minute" or "day" bars for the backtest
            commission: per-trade commission (default 0), a single symmetric commission rate — a Decimal fraction of trade notional, applied identically to buys and sells.
            slippage: per-trade slippage (default 0)
            risk_free_rate: annualized risk-free rate (default 0.0) for Sharpe ratio calculation. The annual rate we get by placing the money.
            warmup_trading_days: extra trading days of history to make available before
                `start` (default 0, i.e. no widening) so a strategy's indicators aren't
                starved near `backtesting_start`. Computed once, here, via
                `backtesting.warmup.warmup_calendar_days`, and used both to construct
                a `data_source` class/callable and to widen `run_backtest`'s own eager
                preload call -- an explicit `data_source` instance is used as given
                and is not widened by this method.
            news_source: where the news tool gets historical news (a `NewsProvider`). Defaults to an
                Alpaca provider built lazily from `AlpacaCredentials.from_env()`; the tool's own
                `strategy.clock.now()` cutoff still applies, so no future article leaks.
            agent_telemetry: record every LLM call (tokens, latency, tool calls) -- per-agent totals go to
                `settings.json["agents"]` and each call to `memory/<strategy>/backtesting/llm_stats.sqlite`,
                which is wiped when the run starts (default `True`).
        """
        from trading_agent_framework.backtesting.data.base import BacktestDataSource as _BacktestDataSource
        from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
        from trading_agent_framework.backtesting.runner import run_backtest
        from trading_agent_framework.backtesting.warmup import warmup_calendar_days

        resolved_start = start if start is not None else self.backtesting_start
        resolved_end = end if end is not None else self.backtesting_end
        if resolved_start is None or resolved_end is None:
            raise ConfigurationError("run_backtesting needs start/end, either as arguments or as backtesting_start/backtesting_end class attributes")
        resolved_budget = _to_decimal(budget) if budget is not None else self.budget
        warmup_start = resolved_start - timedelta(days=warmup_calendar_days(warmup_trading_days))
        if data_source is None:
            resolved_source = YahooBacktestData(warmup_start, resolved_end)
        elif isinstance(data_source, _BacktestDataSource):
            resolved_source = data_source
        else:
            resolved_source = data_source(warmup_start, resolved_end)
        return run_backtest(
            self,
            start=resolved_start,
            end=resolved_end,
            budget=resolved_budget,
            data_source=resolved_source,
            preload_assets=preload_assets,
            benchmark=benchmark or self.benchmark_symbol,
            timestep=timestep,
            commission=_to_decimal(commission),
            slippage=_to_decimal(slippage),
            risk_free_rate=risk_free_rate,
            warmup_trading_days=warmup_trading_days,
            news_source=news_source,
            agent_telemetry=agent_telemetry,
        )

    def _run_trading(self, mode: TradingMode) -> None:
        if self.broker.is_paper != (mode is TradingMode.PAPER):
            account_kind = "paper" if self.broker.is_paper else "live"
            raise ConfigurationError(f"Refusing to run strategy {self.name!r} in {mode} mode against a {account_kind} broker account")
        self.trading_mode = mode
        log_file = setup_strategy_logging(self.name, mode, project_root=self.project_root)
        self.run_id = log_file.parent.name
        self._log_startup_banner(mode, log_file)
        self.executor.run()

    def _log_startup_banner(self, mode: TradingMode, log_file: Path) -> None:
        self.log_info(f"======== {mode.value.upper()} TRADING MODE ========")
        self.log_info(f"Logs will be saved to: {log_file.parent}")
        self.log_info(f"Broker account: {'PAPER' if self.broker.is_paper else 'LIVE'}")
        self.log_info(f"Parameters: {dict(self.parameters)}")
        self._log_market_conditions()
        try:
            self.log_info(f"Initial cash: {self.get_cash()}")
            for position in self.get_positions():
                self.log_info(f"Position: {position.quantity} {position.asset}")
        except BrokerError as exc:
            self.log_warning(f"Could not fetch account/position info: {exc}")

    def _log_market_conditions(self) -> None:
        try:
            session = self.clock.next_session()
        except BrokerError as exc:
            self.log_warning(f"Market calendar unavailable: {exc}")
            return
        if session is None:
            self.log_warning("No upcoming market session")
            return
        now = self.clock.now()
        if session.open > now:
            self.log_info(f"{_format_duration(session.open - now)} until market opens")
        else:
            self.log_info("Market is open")
        self.log_info(f"{_format_duration(session.close - now)} until market closes")
