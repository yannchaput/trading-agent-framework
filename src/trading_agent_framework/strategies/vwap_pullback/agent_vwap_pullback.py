"""VwapPullbackStrategy: an intraday VWAP pullback continuation strategy, in code only (no LLM).

Every tick (5 minutes): `Desk.reconcile` settles what the order hooks may have missed, `Scanner.scan`
advances the setups, `Desk.enter_triggered` enters the triggered ones (best stage-2 score first). A trade
ends on its protective stop or at the 15:50 flatten. Code owns everything (`Desk`): sizing, the protective
stop on every fill, the loss limit, the entry window and the flatten.
See docs/superpowers/specs/2026-10-01-vwap-pullback-pure-code-design.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any

from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.backtesting.data.chunked import YearChunkedData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.vwap_pullback.desk import Desk
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.scanner import Scanner
from trading_agent_framework.strategies.vwap_pullback.session import SessionState
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError


class VwapPullbackStrategy(Strategy):
    """Wires `Scanner` and `Desk` into the framework's lifecycle hooks.

    Hook map: `before_market_opens` prepares the session (stage 1); `on_trading_iteration` runs one tick
    (reconcile, scan, entries); `on_filled_order`/`on_canceled_order` forward to the desk (protective stops,
    trade bookkeeping); `before_market_closes` flattens at 15:50.
    """

    sleeptime = "5M"  # one tick per 5-minute bar
    minutes_before_closing = 10  # before_market_closes, and so the flatten, runs at 15:50

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.SEMI_DECADE)[0],
        "backtesting_end": backtest_window(PredefinedWindow.SEMI_DECADE)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 75,  # 70 daily bars for stage 1, plus the RVOL baseline sessions
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: VwapPullbackParameters | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.universe = list(universe)
        self.settings = settings or VwapPullbackParameters()
        self.desk: Desk | None = None
        self.scanner: Scanner | None = None

    # --- lifecycle -------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the desk and the scanner (once per run)."""
        self.vars.session = None
        if self.is_backtesting:
            # A backtest clock jumps a whole 5-minute tick at once: an entry filled on the tick's first bar would get
            # its stop (placed by the fill hook) only at the tick's end, with the bars in between never checked
            # against it. One-minute slices let the executor dispatch fills, and so place stops, bar by bar.
            # Set on this run's clock instance only: other strategies' clocks keep the class's infinite slice.
            self.clock.max_wait_slice = 60.0
        self._build_components()
        self.log_info(f"VwapPullbackStrategy initialized: {len(self.universe)} symbols, sleeptime {self.sleeptime}")

    def _build_components(self) -> None:
        """Create the desk and the scanner (separate from `initialize` so tests can build them on their own)."""
        self.desk = Desk(self, self.settings, trade_log=self._trade_log_path)
        self.scanner = Scanner(self, self.settings, self.universe, benchmark=self.parameters["benchmark_symbol"], preload=self._preload if self.is_backtesting else None)

    def before_market_opens(self) -> None:
        """Run stage 1 before the open, so the first tick can scan right away."""
        self._ensure_session()

    def on_trading_iteration(self) -> None:
        """One tick: make sure the session is prepared, run the restart check once, then reconcile, scan and enter."""
        if not self.is_backtesting and self.settings.live_bar_delay_seconds > 0:
            self.sleep(self.settings.live_bar_delay_seconds)  # let the last minute bar be published
        state = self._ensure_session()
        # No session (preparation failed; retried next tick) or already flattened: nothing to do this tick.
        if state is None or state.flattened:
            return
        assert self.desk is not None and self.scanner is not None
        # First tick of a session: cancel/close what a previous run of this strategy left behind (restart).
        if not state.unknown_positions_checked:
            self.desk.close_unknown_positions()
        # Reconcile first: expired/rejected entries and missing stops are settled before the scan and the entries.
        self.desk.reconcile(self.get_datetime())
        self.scanner.scan(state)
        self.desk.enter_triggered()

    def before_market_closes(self) -> None:
        """15:50 (`minutes_before_closing`): sell everything this strategy holds; the strategy is strictly intraday."""
        if self.desk is not None:
            self.desk.flatten_all("end-of-day flatten")

    def on_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        """Order hook (executor thread): an entry fill gets its protective stop, an exit fill is booked."""
        if self.desk is not None:
            self.desk.on_order_filled(order, price, quantity)

    def on_canceled_order(self, order: Order) -> None:
        """Order hook: settle an expired entry, re-place a stop cancelled from outside, re-protect after an exit dies."""
        if self.desk is not None:
            self.desk.on_order_canceled(order)

    def _ensure_session(self) -> SessionState | None:
        """This session's state, prepared on first need: `before_market_opens` does not run when a live run starts mid-session."""
        today = self.get_datetime().astimezone(MARKET_TZ).date()
        state = self.vars.session
        if state is None or state.day != today:
            assert self.scanner is not None
            try:
                self.vars.session = self.scanner.prepare_session()
            except (BrokerError, BacktestError) as exc:
                self.log_error(f"session preparation failed, skipping this tick: {exc}")
                self.vars.session = None
        return self.vars.session

    # --- backtesting ---------------------------------------------------------------------

    def _trade_log_path(self) -> Path | None:
        """`trades.jsonl` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "trades.jsonl"

    def _preload(self, assets: Sequence[Asset], timestep: str) -> None:
        """Backtests only: batch-load `assets` over the data source's own window before the scanner reads them.

        The window must equal the one `run_backtesting` builds the data source with (backtest start minus the
        warm-up, to the end): a source that caches one frame per asset would otherwise keep a shorter frame.
        """
        from trading_agent_framework.backtesting.broker import BacktestBroker
        from trading_agent_framework.backtesting.warmup import warmup_calendar_days

        if not isinstance(self.broker, BacktestBroker):
            return
        start = self.parameters["backtesting_start"] - timedelta(days=warmup_calendar_days(self.parameters["warmup_trading_days"]))
        self.broker.preload_bars(assets, start, self.parameters["backtesting_end"], timestep)

    def run_backtesting(self, **overrides: Any):
        """Backtest over the class `parameters` window on Alpaca minute bars (only the benchmark preloaded)."""
        # class parameters: the same window the data source is built with, so preload_bars matches it
        defaults: dict[str, Any] = dict(
            # Alpaca: minute bars with enough history (Yahoo keeps ~30 days of minutes), one year at a time
            # (a 5Y window of minutes for 150 symbols in one fetch was OOM-killed at 58 GB)
            data_source=partial(YearChunkedData, inner=AlpacaBacktestData),
            timestep="minute",
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            benchmark=self.parameters["benchmark_symbol"],
            budget=Decimal(str(self.parameters["budget"])),
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=False,  # no agent runs: nothing to record
        )
        return super().run_backtesting(**{**defaults, **overrides})
