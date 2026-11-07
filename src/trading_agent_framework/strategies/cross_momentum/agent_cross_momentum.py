"""Cross-Sectional Momentum Strategy (V5) -------------------------------------- Fractional share support — floor-based quantities in rebalance().

V5 baseline — copy of V2 with fractional share support (floor-based
quantities in rebalance()).

Implements:
  1. Load pre-computed universe (monthly batch)
  2. Daily: compute portfolio risk diagnostics (if enabled)
  3. Daily: record portfolio equity for realized-vol tracking
  4. Weekly (Tuesday, at rebalance_time): filter, score, rank, select top N on completed sessions, weight by inverse vol
  5. Portfolio risk overlay (beta/vol/corr) — first exposure leg
  6. Breadth overlay (share of scored stocks above their 100d SMA, stepped, with hysteresis) — second leg
  7. Fast/slow volatility targeting (equity-curve-based) — third exposure leg
  8. Combined via min(risk_exposure, breadth_exposure, vol_exposure) — most conservative wins
  9. Hysteresis: sell below rank 35, buy top 20; trim target positions above the ±20% band
  10. Park the de-risked capital (everything but the cash reserve) in a sleeve: GLD and IEF each take half while
      above their 200-day SMA, SHV holds the rest
"""

import calendar
import logging
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import traded_series
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.alpaca import AlpacaApiRateLimiter
from trading_agent_framework.config import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.common.batch_us_stock_universe import ensure_universe_history
from trading_agent_framework.strategies.common.rebalancer import in_flight_quantities
from trading_agent_framework.strategies.common.scoring import apply_filters, momentum_inputs, momentum_score
from trading_agent_framework.strategies.common.sessions import completed_bars, parse_rebalance_time
from trading_agent_framework.strategies.common.universe_backtest_data import UniverseBacktestData, universe_parameters
from trading_agent_framework.strategies.common.universe_history import UniverseHistory, market_day, month_universe, window_symbols
from trading_agent_framework.strategies.common.yahoo_daily_bars import UnusableBatch, YahooDailyBars, fetch_usable_batch
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError, BrokerUncertainError, YahooDataError
from trading_agent_framework.utils.helpers import fractional_qty, get_thread_capacity, parse_insufficient_buying_power

from .log_diagnostics import DiagnosticLogger
from .parameters import CONFIG
from .portfolio_risk_overlay import compute_risk_overlay
from .utils import (
    breadth_exposure,
    breadth_share,
    close_series,
    compute_volatility_exposure,
    get_cross_momentum_universe_last_date,
    inverse_volatility_weights,
    load_breadth_step,
    load_equity_history,
    next_breadth_step,
    save_breadth_step,
    save_equity_history,
    sleeve_symbols,
    sleeve_weights,
    trend_reading,
    unfilled_sell_proceeds,
)

if TYPE_CHECKING:
    from trading_agent_framework.brokers.base import Broker

logger = logging.getLogger(__name__)

# A position within ±20% of its target value is left alone (no top-up, no trim) to avoid overtrading.
_REBALANCE_BAND = 0.20

# Daily bars behind every computation (scores, filters, breadth, risk overlay), all from completed sessions.
_HISTORY_BARS = 300

# Paper/live need Yahoo's bars for at least this share of the universe to scan at all: below it the lookup is down
# or throttled, and scanning on the remainder would sell held names as "not ranked".
_MIN_YAHOO_COVERAGE = 0.5

# Seconds before each new attempt at a failed or incomplete Yahoo batch: the rebalance runs once a week, so a transient
# throttle is waited out (~4 minutes at most) rather than costing the week.
_YAHOO_RETRY_DELAYS = (60.0, 180.0)


@dataclass(frozen=True)
class _Book:
    """What one rebalance sizes against: open orders, the portfolio value, the minimum trade and the sizing prices."""

    incoming: dict[str, float]  # buys still open, by symbol
    outgoing: dict[str, float]  # sells still open, by symbol
    portfolio_value: float
    min_trade_value: float
    prices: dict[str, float]  # last trade of each target, 0.0 when unavailable

    def held_quantity(self, position) -> float:
        """The position counting open orders: held, plus buys in flight, minus sells in flight."""
        symbol = position.asset.symbol
        return max(0.0, float(position.quantity) + self.incoming.get(symbol, 0.0) - self.outgoing.get(symbol, 0.0))

    def sellable(self, position):
        """What is held and not already being sold; the exact position when nothing is."""
        pending_sell = self.outgoing.get(position.asset.symbol, 0.0)
        return position.quantity if pending_sell == 0 else max(0.0, fractional_qty(float(position.quantity) - pending_sell))


@dataclass
class _Sold:
    """The outcome of the sell phase."""

    sells: list[tuple[Order, float]] = field(default_factory=list)  # each submitted sell with the price it is expected to fill at
    hysteresis_value: float = 0.0
    kept_count: int = 0
    sold_count: int = 0
    unconfirmed_count: int = 0  # exits whose outcome the broker could not tell
    sleeve_positions: dict[str, Position] = field(default_factory=dict)


class CrossMomentumStrategy(Strategy):
    """Cross-sectional momentum strategy V5 — copy of V2 with fractional share support.

    Tracks actual daily portfolio equity to compute realized volatility
    using both a 20-day (fast) and 63-day (slow) window, then takes the
    more conservative estimate via max(vol_20d, 0.75 * vol_63d). Combined
    with the portfolio risk overlay and the market-breadth overlay via min() — the most conservative leg
    determines the final exposure multiplier. Held positions are trimmed to
    the scaled target, and the capital taken out of stocks is parked in a trend-filtered SHV/GLD/IEF sleeve.
    """

    # ── Backtesting params ───────────────────────────────────────────────────────
    parameters = {
        # "backtesting_start": datetime(2026, 4, 6, tzinfo=MARKET_TZ),
        # "backtesting_end": datetime(2026, 4, 24, tzinfo=MARKET_TZ),
        "backtesting_start": backtest_window(PredefinedWindow.DECADE)[0],
        "backtesting_end": backtest_window(PredefinedWindow.DECADE)[1],
        "benchmark_symbol": "SPY",
        # Warm-up extends the data window before backtesting_start so the 12-1m
        # momentum lookback (252 + 21 skip + 1 = 274 bars) has full history from
        # day one. Without it, every ticker trips min_trading_days=250 and the
        # first ~year of the backtest can never pass filters.
        "warmup_trading_days": 300,
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.BACKTESTING,
        universe: list[str] | None = None,
        diagnostics_file_path: str | None = None,
        bars_source: YahooDailyBars | None = None,
        **kwargs,
    ):
        """Create the strategy.

        Args:
            broker: Broker the strategy trades through.
            mode: Trading mode.
            universe: Tickers ranked; required in paper/live, while a backtest reads the point-in-time history instead.
            diagnostics_file_path: File the rebalance diagnostics are written to; none when omitted.
            bars_source: Yahoo daily-bars source for paper/live (injected in tests); a backtest reads the data gate.
            **kwargs: Passed on to `Strategy`.
        """
        super().__init__(broker, mode=mode, **kwargs)
        self.parameters = {**CONFIG, **self.parameters}

        # Paper/live trade the universe file's list, so they need one. A backtest has none at construction: its lists
        # come from the point-in-time history, loaded by `run_backtesting` and read month by month (`_use_month_universe`).
        if not universe and self.trading_mode is not TradingMode.BACKTESTING:
            self.log_warning("No universe provided to the strategy, this is a mandatory parameter")
            sys.exit(1)
        universe = universe or []
        # A sleeve asset (SHV, GLD, IEF) is never scored or bought as a stock, even if the universe file lists it
        sleeve = self._sleeve_symbols()
        self.vars.universe = [symbol for symbol in universe if symbol not in sleeve]
        self.vars.universe_history = None  # set by `run_backtesting`; None (paper/live) keeps `vars.universe` as built

        # Paper/live read every daily bar the decision uses from Yahoo (see `get_historical_prices`): Alpaca's IEX bars
        # carry ~5% of consolidated volume, skip a thin stock's quiet days and close at the last IEX trade, so the
        # filter and the ranks ran on other data than the backtest's. A backtest's bars are already Yahoo's.
        if bars_source is None and self.trading_mode is not TradingMode.BACKTESTING:
            bars_source = YahooDailyBars()
        self.vars.bars_source = bars_source
        self.vars.yahoo_bars = {}

        # Diagnostics (delegated to DiagnosticLogger)
        self.vars.diagnostics_logger = DiagnosticLogger(self, trading_mode=self.trading_mode, diagnostics_file_path=diagnostics_file_path)
        self.vars.target_closes = {}

        # Breadth overlay state: this rebalance's reading and the step it put the book on. The step is persisted
        # (like the equity history) so a crash or restart keeps the hysteresis; None means no history yet.
        self.vars.breadth = None
        self.vars.breadth_step = None
        self.vars.breadth_file_path = Path(f"data/cross_momentum_breadth_{self.trading_mode.value}.json")

        # Track actual portfolio equity for realized-vol computation
        self.vars.equity_history = []
        self.vars.history_file_path = Path(f"data/cross_momentum_ptf_history_{self.trading_mode.value}.json")

        # Rate limiter for Alpaca market-data calls (universe filtering bursts
        # past Alpaca's ~200 req/min limit; see support/alpaca_support.py).
        # Market data is Alpaca's under every broker (BROKER=ibkr included), so this limiter always applies.
        self.vars.alpaca_rate_limiter = AlpacaApiRateLimiter(trading_mode=self.trading_mode)

        self.log_info(
            f"CrossMomentumStrategy initialized with {len(self.vars.universe)} symbols",
        )

    def initialize(self):
        """Set up the strategy before trading starts."""
        self.sleeptime = "1D"
        # The executor runs each session's iteration, so the Tuesday rebalance, at this market time
        self.iteration_start_time = parse_rebalance_time(self.parameters["rebalance_time"])
        self.log_info(f"Sleeptime set to {self.sleeptime}, iteration start time {self.iteration_start_time}")
        # Track actual portfolio equity for realized-vol computation
        self.vars.equity_history = load_equity_history(self.vars.history_file_path)
        self.log_info(f"Initialize equity history for volatility exposure compute {self.vars.equity_history}")
        # Resume the breadth step where the bot left off (None if there is nothing usable on disk)
        self.vars.breadth_step = load_breadth_step(self.vars.breadth_file_path, len(self.parameters["breadth_overlay"]["exposures"]))
        self.log_info(f"Initialize breadth step: {self.vars.breadth_step}")

    # ── Fast/Slow realized volatility from equity curve ────────────────────

    def _compute_realized_volatility(self) -> float | None:
        """Compute annualized realized volatility using fast/slow windows.

        Uses max(vol_20d, 0.75 * vol_63d) as the risk volatility estimate,
        rather than just vol_20d.

        The 63d (quarterly) component catches structural risk earlier
        (e.g. August 2024 where 20d vol was still declining but 63d vol
        already showed elevated risk). The 0.75 haircut prevents the slow
        estimator from dominating exposure decisions long after a shock.

        Returns:
            Annualized volatility as a float, or None if insufficient history.
        """
        # Need at least 63 + 1 observations for the slow window
        if len(self.vars.equity_history) < 64:
            return None

        equity_series = pd.Series(self.vars.equity_history)
        daily_returns = equity_series.pct_change().dropna()

        if len(daily_returns) < 63:
            return None

        # Fast window: 20-day realized vol
        vol_20 = float(np.std(daily_returns.iloc[-20:], ddof=1) * np.sqrt(252))

        # Slow window: 63-day realized vol (quarterly)
        vol_63 = float(np.std(daily_returns.iloc[-63:], ddof=1) * np.sqrt(252))

        # Fast/Slow targeting: use the more conservative estimate.
        # max(vol_20d, 0.75 * vol_63d) catches structural risk early while
        # the 0.75 haircut prevents the slow estimator from dominating
        # forever after a shock.
        risk_vol = max(vol_20, 0.75 * vol_63)
        return risk_vol

    # ── Momentum / portfolio construction (from V2.1) ─────────────────────────

    def is_rebalance_day(self) -> bool:
        """True if today is a rebalance day."""
        dt = self.get_datetime()
        return dt.weekday() == self.parameters["day_of_week"]

    def _market_date(self) -> date:
        """Today's date in market time: the date of the session whose daily bar is still forming."""
        return self.get_datetime().astimezone(MARKET_TZ).date()

    def get_historical_prices(self, asset: Asset | str, length: int, timestep: str = "day", *, include_after_hours: bool = True) -> Bars | None:
        """The last `length` bars of `asset`.

        Paper/live serve daily bars from the Yahoo batch loaded for this scan (`_load_yahoo_bars`), so the stocks, SPY
        and the sleeve's trend assets are all read from the backtest's own source; a symbol Yahoo has no bars for gets
        None, never an Alpaca fallback on other data. Everything else goes to the framework.
        """
        if self.vars.bars_source is not None and timestep == "day":
            bars = self.vars.yahoo_bars.get(asset.symbol if isinstance(asset, Asset) else asset)
            return None if bars is None else Bars(asset=bars.asset, timestep="day", df=bars.pandas_df.tail(length))
        self.vars.alpaca_rate_limiter.wait()  # Alpaca's data API; a no-op in backtests
        return super().get_historical_prices(asset, length, timestep, include_after_hours=include_after_hours)

    def _refresh_regime(self) -> None:
        """The framework's regime refresh, fed with the benchmark's Yahoo bars in paper/live.

        The executor refreshes the regime once per session, before the open and so before any scan: `vars.yahoo_bars`
        is empty then (it is filled by `rebalance` and released when it ends), and the `get_historical_prices` override
        above answers an empty batch with None. The framework then read "no bars" as "fewer than `min_bars` daily
        bars", logged that once and never produced a regime in paper/live (2026-10-09). Falling back to Alpaca instead
        would put the regime on IEX bars while every decision runs on Yahoo's, which is the split the Yahoo switch
        removed. So the benchmark is fetched here, on its own, and handed to the framework's unchanged logic through
        `vars.yahoo_bars` for the duration of the call only. A backtest has no bars source and uses the framework as is.
        """
        if self.vars.bars_source is None:
            # call parent implementation based on Alpaca IEX if bars_source is not based on Yahoo
            super()._refresh_regime()
            return
        benchmark = self.benchmark_symbol
        try:
            # Fetch bars from Yahoo for the benchmark only, and hand them to the framework's logic through `vars.yahoo_bars` for the duration of the call
            fetched = self.vars.bars_source.bars([benchmark], self._market_date())
        except YahooDataError as exc:
            # Same policy as the framework: a failed refresh keeps the previous value and never costs a session
            self.log_warning(f"Market regime not refreshed (still {self.regime}): {exc}")
            return
        if benchmark not in fetched:
            self.log_warning(f"Market regime not refreshed (still {self.regime}): Yahoo has no bars for {benchmark}")
            return
        previous = self.vars.yahoo_bars
        self.vars.yahoo_bars = fetched
        try:
            # Hand over to parent refresh_regime which will call this class get_historical_prices fetching bars from vars.yahoo_bars
            super()._refresh_regime()
        finally:
            self.vars.yahoo_bars = previous  # not left behind: no stale bars outlive the call

    def _compute_indicators_for_ticker(self, ticker: str) -> dict | None:
        """Fetch OHLCV data and compute momentum indicators for a single ticker."""
        try:
            bars = self.get_historical_prices(ticker, length=_HISTORY_BARS + 1, timestep="day")
        except BrokerError as exc:
            self.log_warning(f"Skipping {ticker}: failed to fetch bars ({exc})")
            return None
        if bars is None or bars.empty:
            return None

        df = bars.pandas_df

        # Completed sessions only: today's bar is partial, and it made the same Tuesday's decision depend on the
        # hour it ran (2026-10-06). One extra bar is fetched so the window stays _HISTORY_BARS long.
        df = completed_bars(df, self._market_date()).tail(_HISTORY_BARS)
        if df.empty:
            return None

        closes = df["close"].tolist()
        # Cut from the same rows as `closes`: the filters read the price and volume the stock traded at, not the adjusted ones.
        traded = traded_series(df)
        inputs = momentum_inputs(
            closes,
            df["volume"].tolist(),
            self.parameters,
            traded_closes=traded[0] if traded else None,
            traded_volumes=traded[1] if traded else None,
        )
        if inputs is None:
            return None

        return {
            "symbol": ticker,
            "price": inputs.price,
            "ret_12_1m": inputs.ret_12_1m,
            "ret_6_1m": inputs.ret_6_1m,
            "ret_3m": inputs.ret_3m,
            "volatility": inputs.volatility,
            "avg_dollar_volume": inputs.avg_dollar_volume,
            "trading_days": inputs.trading_days,
            "closes": closes,
            "frame": df,
        }

    def _load_yahoo_bars(self) -> bool:
        """Fetch the scan's daily bars from Yahoo into `vars.yahoo_bars`: the universe, SPY and the sleeve's trend assets.

        A failed or incomplete batch is fetched again after each of `_YAHOO_RETRY_DELAYS` (a transient 429 must not cost
        the week's only rebalance). False (logged) when the last attempt still fails: the caller then skips the scan
        and holds the book.
        """
        universe = self.vars.universe
        try:
            bars = fetch_usable_batch(
                self.vars.bars_source,
                universe,
                extras=["SPY", *self.parameters["parking"]["trend_assets"]],
                today=self._market_date,
                held=lambda: {pos.asset.symbol for pos in self.get_positions()},
                min_coverage=_MIN_YAHOO_COVERAGE,
                retry_delays=_YAHOO_RETRY_DELAYS,
                sleep=self.sleep,  # clock time, so order hooks still fire meanwhile
                on_retry=lambda problem, delay: self.log_warning(f"{problem} — retry in {delay:.0f}s"),
            )
        except UnusableBatch as exc:
            self.log_error(f"{exc} — holding current positions.")
            return False
        self.vars.yahoo_bars = bars
        covered = sum(1 for symbol in universe if symbol in bars)
        self.log_info(f"Yahoo bars: {covered}/{len(universe)} symbols")
        return True

    def compute_target_portfolio(self) -> tuple[list[dict], dict[str, int]]:
        """Run the full pipeline: indicators → score → filter → rank → select → weight.

        Uses ThreadPoolExecutor for parallel processing of the universe.
        """
        self.log_info(f"Computing target portfolio for {len(self.vars.universe)} tickers...")
        self.vars.breadth = None  # no stale reading if this run returns early
        self.vars.yahoo_bars = {}  # nor last week's bars
        # An unusable batch (logged by _load_yahoo_bars) ends the scan with an empty target. rebalance() treats that as
        # "hold the book", so no sell is sent on data Yahoo did not deliver.
        if self.vars.bars_source is not None and not self._load_yahoo_bars():
            return [], {}

        scored: list[dict] = []
        skip_count = 0
        total = len(self.vars.universe)

        max_workers = get_thread_capacity()
        self.log_info(f"Using {max_workers} threads for parallel processing")

        def _process_ticker(ticker: str) -> dict | None:
            result = self._compute_indicators_for_ticker(ticker)
            if result is None:
                return None

            if not apply_filters(
                result["price"],
                result["avg_dollar_volume"],
                result["volatility"],
                result["trading_days"],
                self.parameters,
            ):
                self.log_debug(f"Skipping {ticker}: failed filters")
                return None

            result["score"] = momentum_score(
                result["ret_12_1m"],
                result["ret_6_1m"],
                result["ret_3m"],
                self.parameters["w_12m"],
                self.parameters["w_6m"],
                self.parameters["w_3m"],
            )
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(_process_ticker, ticker): ticker for ticker in self.vars.universe}

            for i, future in enumerate(as_completed(future_to_ticker), 1):
                result = future.result()
                if result is None:
                    skip_count += 1
                    continue
                scored.append(result)

                if i % 100 == 0:
                    self.log_info(f"  Processed {i}/{total} — {len(scored)} passed filters so far")

        self.log_info(f"Indicators: {len(scored)} passed filters, {skip_count} skipped")

        if not scored:
            self.log_error("No stocks passed filters — holding current positions.")
            return [], {}

        scored.sort(key=lambda x: x["score"], reverse=True)
        for idx, entry in enumerate(scored):
            entry["rank"] = idx + 1

        all_ranks = {entry["symbol"]: entry["rank"] for entry in scored}

        breadth_config = self.parameters["breadth_overlay"]
        self.vars.breadth = breadth_share(
            {entry["symbol"]: entry["closes"] for entry in scored},
            breadth_config["sma_window"],
            breadth_config["min_stocks"],
        )

        top_n = self.parameters["top_n"]
        selected = scored[:top_n]

        # Store closes for the selected stocks (consumed by portfolio risk overlay)
        self.vars.target_closes = {entry["symbol"]: close_series(entry["frame"]) for entry in selected}

        selected = inverse_volatility_weights(
            selected,
            self.parameters["max_position_pct"],
            self.parameters["min_position_pct"],
        )

        if selected:
            self.log_info(
                f"Target portfolio: {len(selected)} stocks selected. Top: {selected[0]['symbol']} (rank 1, score {selected[0]['score']:.4f})",
            )
        return selected, all_ranks

    def _breadth_exposure(self) -> float:
        """Exposure multiplier from market breadth (share of scored stocks above their SMA), with hysteresis.

        1.0 when the overlay is disabled or there is no reading; otherwise the current step's multiplier. The
        step is kept on `self.vars` and persisted, so a breadth hovering at a threshold doesn't flip the book
        week to week, even across a crash or restart.
        """
        config = self.parameters["breadth_overlay"]
        breadth = self.vars.breadth
        if not config["enabled"] or breadth is None:
            return 1.0
        step = next_breadth_step(breadth, self.vars.breadth_step, config["thresholds"], config["hysteresis"])
        self.vars.breadth_step = step
        save_breadth_step(self.vars.breadth_file_path, step, breadth, self.get_datetime().strftime("%Y-%m-%d"))
        exposure = breadth_exposure(step, config["exposures"])
        self.log_info(f"Breadth: {breadth:.0%} of stocks above their {config['sma_window']}d SMA -> step {step} (exposure {exposure:.0%})")
        return exposure

    def _risk_exposure(self, target: list[dict]) -> float:
        """Exposure multiplier from the portfolio risk overlay: beta/vol/corr of the target stocks against SPY.

        1.0 without targets or SPY data. The overlay aligns the series by session date; a target whose last bar
        is dated differently from SPY's is still logged, because on 2026-10-06 half of them were a session short
        for a reason never established.
        """
        if not target or not self.vars.target_closes:
            return 1.0
        spy_bars = self.get_historical_prices("SPY", length=_HISTORY_BARS + 1, timestep="day")
        if spy_bars is None or spy_bars.empty:
            self.log_warning("Risk overlay: SPY data unavailable — defaulting to NORMAL (100% exposure)")
            return 1.0
        spy = close_series(completed_bars(spy_bars.pandas_df, self._market_date()).tail(_HISTORY_BARS))
        if spy.empty:
            self.log_warning("Risk overlay: no completed SPY session — defaulting to NORMAL (100% exposure)")
            return 1.0
        spy_last = spy.index[-1]
        off_date = {symbol: series.index[-1] for symbol, series in self.vars.target_closes.items() if len(series) and series.index[-1] != spy_last}
        if off_date:
            listed = ", ".join(f"{symbol} {day}" for symbol, day in sorted(off_date.items()))
            self.log_warning(f"Risk overlay: last bar date differs from SPY's {spy_last}: {listed}")
        target_weights = {entry["symbol"]: entry["target_weight"] for entry in target}
        risk_state, risk_exposure, metrics = compute_risk_overlay(self.vars.target_closes, target_weights, spy)
        self.log_info(
            f"Risk overlay: {risk_state.upper()} (beta={metrics.get('beta_full_window')}, vol={metrics.get('vol_20d')}, corr={metrics.get('corr_20d')}, "
            f"obs={metrics.get('observations')}, exposure={risk_exposure:.0%})"
        )
        return risk_exposure

    def _price_or_zero(self, symbol: str) -> float:
        """Last price for valuing a holding, or 0.0 if the lookup fails or returns nothing.

        A failed quote must not abort a rebalance halfway through, after its sells are already submitted.
        """
        try:
            return float(self.get_last_price(symbol) or 0.0)
        except Exception as e:
            self.log_warning(f"No price for {symbol}: {e}")
            return 0.0

    def _trend_closes(self, symbol: str) -> list[float] | None:
        """A trend asset's completed daily closes, oldest first, or None (logged) when its bars are unavailable."""
        fallback = self.parameters["parking"]["symbol"]
        try:
            bars = self.get_historical_prices(symbol, length=_HISTORY_BARS + 1, timestep="day")
        except (BrokerError, BacktestError) as exc:
            self.log_warning(f"Sleeve: no bars for {symbol} ({exc}): its share goes to {fallback}")
            return None
        if bars is None or bars.empty:
            self.log_warning(f"Sleeve: no bars for {symbol}: its share goes to {fallback}")
            return None
        # Completed sessions only, like the stocks: a bar dated today is partial while the session is open
        return completed_bars(bars.pandas_df, self._market_date()).tail(_HISTORY_BARS)["close"].tolist()

    def _sleeve_weights(self) -> dict[str, float]:
        """This week's share of the parking sleeve per symbol (SHV first, then the trend assets), logged."""
        parking = self.parameters["parking"]
        window = parking["trend_sma_window"]
        trend_assets = tuple(parking["trend_assets"])
        closes_by_asset = {symbol: closes for symbol in trend_assets if (closes := self._trend_closes(symbol)) is not None}
        weights = sleeve_weights(closes_by_asset, trend_assets, window, parking["symbol"])

        readings = []
        for symbol in trend_assets:
            reading = trend_reading(closes_by_asset.get(symbol, []), window)
            if reading is None:
                readings.append(f"{symbol} off (no data)")
            elif weights[symbol] > 0:
                readings.append(f"{symbol} on (close {reading[0]:.2f} > SMA{window} {reading[1]:.2f})")
            else:
                readings.append(f"{symbol} off (close {reading[0]:.2f} <= SMA{window} {reading[1]:.2f})")
        allocation = " / ".join(f"{symbol} {weight:.0%}" for symbol, weight in weights.items())
        self.log_info(f"Sleeve: {', '.join(readings) + ' -> ' if readings else ''}{allocation}")
        return weights

    def rebalance(self, target: list[dict], all_ranks: dict[str, int]) -> None:
        """Rebalance the book to `target` (`_rebalance_book`), then release the scan's Yahoo bars whatever happens.

        The rebalance is their last reader (the sleeve's trend test); kept, they would pin ~15 MB for a week and serve
        last week's bars to any later daily read.
        """
        try:
            self._rebalance_book(target, all_ranks)
        finally:
            self.vars.yahoo_bars = {}

    def _send(self, side: str, quantity: float | Decimal, symbol: str, why: str) -> tuple[Order | None, Exception | None, bool]:
        """Create and submit one day order: (the submitted order or None, the exception raised or None, unknown).

        `unknown` is True on `BrokerUncertainError`: the order was sent but the broker could not confirm it. It stays
        tracked and is never resent here (logged with `why`); the caller adds no proceeds and keeps any cost deducted.
        """
        try:
            return self.submit_order(self.create_order(symbol, quantity, side, time_in_force="day")), None, False
        except BrokerUncertainError as exc:
            self.log_warning(f"{side} {quantity:g} {symbol} ({why}): outcome unknown at the broker, not resent -- check the account: {exc}")
            return None, exc, True
        except Exception as exc:
            return None, exc, False

    def _rebalance_book(self, target: list[dict], all_ranks: dict[str, int]) -> None:
        """Compare holdings to target, apply hysteresis, trim, submit orders, and park the rest in the sleeve.

        Args:
            target: Selected stocks with target_weight (top N, already weighted).
            all_ranks: Symbol→rank mapping for ALL scored stocks.
        """
        if not target:
            return

        parking = self.parameters["parking"]
        # Trend assets first, SHV last: the order of the sleeve buys once the stock buys are funded
        sleeve_order = (*parking["trend_assets"], parking["symbol"])
        portfolio_value = float(self.portfolio_value or 1.0)

        # Long positions only: IBKR and Tastytrade report a short as a positive quantity with side=SHORT, and a sell
        # against it would open more short (this strategy is long-only and never trades one)
        current_positions = []
        for pos in self.get_positions():
            if pos.side is PositionSide.LONG:
                current_positions.append(pos)
            else:
                self.log_warning(f"{pos.asset.symbol}: short position of {pos.quantity} ignored (this strategy is long-only and will not trade it)")
        # Orders still open from an earlier rebalance (a restart on a rebalance day sends this one again), read before
        # this run sends anything: a pending buy counts toward its position, a pending sell is not sent again, and the
        # buy budget is net of both. Cash is not debited by an open buy, nor credited by an open sell, until it fills.
        incoming, outgoing = in_flight_quantities(self.broker.tracker.get_active_orders())
        if incoming or outgoing:
            pending = [f"buy {q:g} {s}" for s, q in incoming.items()] + [f"sell {q:g} {s}" for s, q in outgoing.items()]
            self.log_warning(f"Counting orders still open from an earlier rebalance: {', '.join(pending)}")

        # Size at the last trade: an entry's "price" is the traded close of the last completed session, a session old by now.
        prices = {entry["symbol"]: self._price_or_zero(entry["symbol"]) for entry in target}
        for symbol, price in prices.items():
            if price <= 0:
                self.log_warning(f"No price for {symbol}: no buy or trim this week")
        book = _Book(incoming, outgoing, portfolio_value, portfolio_value * parking["min_trade_pct"], prices)

        # The trend data is fetched before any order is sent: an unexpected fetch error then aborts the rebalance cleanly
        weights = self._sleeve_weights()

        sold = self._sell_phase(target, all_ranks, current_positions, book, sleeve_order)
        sleeve = self._park_targets(target, book, sold, weights, sleeve_order)
        self._buy_phase(target, book, current_positions, sold.sells, sleeve)

        # The target positions never get a line unless they are bought, so say how the book is made up
        target_symbols = {entry["symbol"] for entry in target}
        held_targets = sum(1 for pos in current_positions if pos.asset.symbol in target_symbols)
        unconfirmed_note = f" ({sold.unconfirmed_count} unconfirmed)" if sold.unconfirmed_count else ""
        self.log_info(
            f"Rebalance summary: {len(target)} targets ({held_targets} already held, {len(target) - held_targets} not held), "
            f"{sold.kept_count} kept under hysteresis, {sold.sold_count} sold{unconfirmed_note}"
        )

    def _sell_phase(self, target: list[dict], all_ranks: dict[str, int], current_positions: list, book: _Book, sleeve_order: tuple[str, ...]) -> _Sold:
        """Phase 1: exits (not ranked or outside the hysteresis band) and trims of target positions above their band."""
        target_by_symbol = {entry["symbol"]: entry for entry in target}
        sell_threshold = self.parameters["sell_rank_threshold"]
        sold = _Sold()
        for pos in current_positions:
            symbol = pos.asset.symbol
            if symbol in sleeve_order:
                # The parking sleeve is never ranked: it must not be exited as "not ranked" below
                sold.sleeve_positions[symbol] = pos
                continue
            if symbol in target_by_symbol:
                # Trim a target position that sits above its band, so an exposure cut lowers the held book
                # too, not just new buys; the proceeds are parked or fund this week's buys.
                entry = target_by_symbol[symbol]
                price = book.prices[symbol]
                if price <= 0:
                    continue
                target_value = book.portfolio_value * entry["target_weight"]
                current_value = book.held_quantity(pos) * price
                trim_value = current_value - target_value
                if current_value > target_value * (1 + _REBALANCE_BAND) and trim_value >= book.min_trade_value:
                    trim_qty = min(fractional_qty(trim_value / price), float(book.sellable(pos)))
                    if trim_qty > 0:
                        self.log_info(f"Trimming {trim_qty} {symbol} @ ${price:.2f} (value ${current_value:,.0f} > target ${target_value:,.0f})")
                        order, exc, _ = self._send("sell", trim_qty, symbol, "trim above band")
                        if order is not None:
                            sold.sells.append((order, price))
                        elif exc is not None and not isinstance(exc, BrokerUncertainError):
                            self.log_error(f"Failed to submit trim order for {symbol}: {exc}")
                continue

            rank = all_ranks.get(symbol)
            if rank is None or rank > sell_threshold:
                exit_quantity = book.sellable(pos)
                if exit_quantity <= 0:
                    self.log_info(f"{symbol} is already being sold by an open order: not selling it again")
                    continue
                if rank is None:
                    # The position did not pass the scoring pipeline and was kicked out by the filters (volatility, price or not able to compute indicators)
                    self.log_warning(f"Selling {symbol} (not ranked — failed filters or price data unavailable)")
                else:
                    # The position is ranked but outside the sell threshold (hysteresis band)
                    self.log_warning(f"Selling {symbol} (rank {rank} > {sell_threshold}) - outside hysteresis band")
                order, exc, unknown = self._send("sell", exit_quantity, symbol, "exit")
                if unknown:  # sent, outcome unknown: counted as sold, no proceeds
                    sold.sold_count += 1
                    sold.unconfirmed_count += 1
                elif order is not None:
                    sold.sold_count += 1
                    try:
                        sold.sells.append((order, float(self.get_last_price(symbol) or 0.0)))
                    except Exception as e:
                        self.log_error(f"Failed to submit sell order for {symbol}: {e}")
                else:
                    self.log_error(f"Failed to submit sell order for {symbol}: {exc}")
            # Rank <= sell_threshold means keep the position inside the histeresis band (do not sell)
            else:
                self.log_info(f"Keeping {symbol} (rank {rank} ≤ {sell_threshold}, within hysteresis band)")
                sold.hysteresis_value += book.held_quantity(pos) * self._price_or_zero(symbol)
                sold.kept_count += 1
        return sold

    def _park_targets(self, target: list[dict], book: _Book, sold: _Sold, weights: dict[str, float], sleeve_order: tuple[str, ...]) -> list[tuple[str, float, float, float]]:
        """Size the parking sleeve and sell its excess (added to `sold.sells`).

        Returns (symbol, price, current value, target value) per priced sleeve symbol, for the parking buys.
        """
        # Parking target: everything not meant for stocks, except the cash reserve, goes to the sleeve
        stock_target_value = book.portfolio_value * sum(entry["target_weight"] for entry in target)
        parking_target = max(0.0, book.portfolio_value * (1 - self.parameters["cash_buffer_pct"]) - stock_target_value - sold.hysteresis_value)
        sleeve: list[tuple[str, float, float, float]] = []
        for symbol in sleeve_order:
            position = sold.sleeve_positions.get(symbol)
            price = self._price_or_zero(symbol)
            if price <= 0:
                self.log_warning(f"Parking: no price for {symbol} — no {symbol} order this week")
                continue
            value = (book.held_quantity(position) if position else book.incoming.get(symbol, 0.0)) * price
            symbol_target = parking_target * weights.get(symbol, 0.0)
            sleeve.append((symbol, price, value, symbol_target))
            self.log_info(f"Parking: {symbol} target ${symbol_target:,.0f} (current ${value:,.0f})")
            excess = value - symbol_target
            if position is not None and value > symbol_target * (1 + _REBALANCE_BAND) and excess >= book.min_trade_value:
                # A zero target sells the exact holding (less what is already being sold), so float flooring leaves no dust behind.
                sell_qty = float(book.sellable(position)) if symbol_target == 0 else min(fractional_qty(excess / price), float(book.sellable(position)))
                if sell_qty > 0:
                    self.log_info(f"Selling {sell_qty} {symbol} @ ${price:.2f} (parking above target)")
                    order, exc, _ = self._send("sell", sell_qty, symbol, "parking above target")
                    if order is not None:
                        sold.sells.append((order, price))
                    elif exc is not None and not isinstance(exc, BrokerUncertainError):
                        self.log_error(f"Failed to submit parking sell order for {symbol}: {exc}")
        return sleeve

    def _buy_phase(self, target: list[dict], book: _Book, current_positions: list, sells: list[tuple[Order, float]], sleeve: list[tuple[str, float, float, float]]) -> None:
        """Phases 2 and 3: buy the targets within the budget, then park what is left up to each sleeve target."""
        # The estimate is priced at the last trade, but orders fill a little later at the market (at the next
        # bar's open in a backtest) plus fees, so hold a fixed reserve back: it is what lets the last buys land
        # when prices move or sells fill lower.
        # Live market sells fill within seconds, so the broker's cash already holds the proceeds of the filled ones:
        # only the part still to fill is added. The cash is read BEFORE the sell states, so a sell filling in between
        # is counted nowhere (a slightly smaller budget), never twice. In a backtest no sell fills before the next
        # session, so all of them are added, as before.
        incoming, outgoing = book.incoming, book.outgoing
        cash = float(self.get_cash())
        pending_proceeds = unfilled_sell_proceeds(sells)
        filled_sells = sum(1 for order, _ in sells if not order.is_active())
        # Earlier open orders: a pending sell will bring its proceeds, a pending buy has already claimed its cost
        earlier_credit = sum(quantity * self._price_or_zero(symbol) for symbol, quantity in outgoing.items()) - sum(quantity * self._price_or_zero(symbol) for symbol, quantity in incoming.items())
        earlier_note = f" {'+' if earlier_credit >= 0 else '-'} ${abs(earlier_credit):,.0f} net from earlier open orders" if incoming or outgoing else ""
        self.log_info(f"Buy budget: cash ${cash:,.0f} + ${pending_proceeds:,.0f} from unfilled sells ({filled_sells} of {len(sells)} sells already filled){earlier_note}")
        spendable = cash + pending_proceeds + earlier_credit
        cash_reserve = spendable * self.parameters["cash_buffer_pct"]
        available_cash = spendable - cash_reserve
        positions_by_symbol = {p.asset.symbol: p for p in current_positions}
        for entry in target:
            # Available cash safe guard: if available cash is zero or negative, skip remaining target stocks
            if available_cash <= 0:
                self.log_warning("No available cash left for buying — skipping remaining target stocks.")
                break

            symbol = entry["symbol"]
            price = book.prices[symbol]
            if price <= 0:
                continue
            target_weight = entry["target_weight"]
            target_value = book.portfolio_value * target_weight

            current_pos = positions_by_symbol.get(symbol)
            current_value = (book.held_quantity(current_pos) if current_pos else incoming.get(symbol, 0.0)) * price

            # Compute the difference between target and current value position
            diff_value = target_value - current_value

            # Skip buying if the position is already above target, or within ±20% of the target value (avoids overtrading)
            if diff_value <= 0 or (current_value > 0 and abs(diff_value) / target_value < _REBALANCE_BAND):
                self.log_info(f"Holding {symbol} (rank {entry['rank']}, ${current_value:,.0f} vs target ${target_value:,.0f}): no buy needed")
                continue

            # Compute the fractional quantity to buy taking into account the existing position
            raw_qty = diff_value / price
            quantity = fractional_qty(raw_qty)
            # Current position is already above target, skip buying
            if quantity <= 0:
                continue

            cost = quantity * price
            # If the cost exceeds the remaining spendable cash (the reserve is already excluded), reduce the quantity to fit
            if cost > available_cash:
                quantity = fractional_qty(available_cash / price)
                cost = quantity * price
            # quantity can be equal to 0 for very tiny amount passed to fractional_qty
            if quantity <= 0:
                continue

            # cost <= available_cash here (fractional_qty floors), so available_cash stays non-negative
            available_cash -= cost

            self.log_info(f"Buying {quantity} {symbol} @ ${price:.2f} (target weight: {target_weight:.1%}, rank: {entry['rank']})")
            _, exc, unknown = self._send("buy", quantity, symbol, "toward target")
            if exc is not None and not unknown:
                self.log_warning(f"Failed to submit buy order for {symbol}: {exc}")
                # The local available_cash estimate (get_cash() + estimated unfilled
                # sell proceeds) can drift from the broker's real buying power —
                # e.g. this iteration's sells hadn't settled yet. Resync to the
                # broker's own reported value so the remaining targets in this loop
                # aren't sized against a number already proven wrong, instead of
                # repeating the same rejection down the list.
                real_buying_power = parse_insufficient_buying_power(exc)
                if real_buying_power is not None:
                    self.log_warning(f"Resyncing available cash to broker-reported buying power: ${real_buying_power:.2f}")
                    available_cash = real_buying_power * (1 - self.parameters["cash_buffer_pct"])
            # An unknown outcome keeps the cost deducted, with no buying-power resync

        # Phase 3: Park what the stock buys left, up to each sleeve target (the trend assets first, SHV last)
        for symbol, price, value, symbol_target in sleeve:
            if value >= symbol_target * (1 - _REBALANCE_BAND):
                continue
            buy_value = min(symbol_target - value, available_cash)
            if buy_value < book.min_trade_value:
                continue
            quantity = fractional_qty(buy_value / price)
            if quantity <= 0:
                continue
            available_cash -= quantity * price
            self.log_info(f"Buying {quantity} {symbol} @ ${price:.2f} (parking)")
            _, exc, unknown = self._send("buy", quantity, symbol, "parking")
            if exc is not None and not unknown:
                self.log_warning(f"Failed to submit parking buy order for {symbol}: {exc}")

    # ── Main iteration ────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        """Run one trading iteration: record the day's equity, then rebalance when the schedule says it is due.

        In paper/live it first checks that the universe file is recent and does nothing when it is missing or stale.
        """
        super().on_trading_iteration()

        # Record daily equity for realized-vol tracking
        equity = self.portfolio_value or 0.0
        if equity > 0:
            # Compute history file name based on trading mode
            self.vars.equity_history.append(float(equity))
            today_str = self.get_datetime().strftime("%Y-%m-%d")
            save_equity_history(self.vars.equity_history[-1], today_str, self.vars.history_file_path)

        today: datetime = self.get_datetime()

        if self.trading_mode is not TradingMode.BACKTESTING:
            last_universe_date = get_cross_momentum_universe_last_date()
            if not last_universe_date:
                self.log_warning("No universe date found — universe file missing or empty.")
                return
            if (today - last_universe_date).days > 30:
                self.log_warning(f"Warning: universe is {(today - last_universe_date).days} days old (last update: {last_universe_date:%Y-%m-%d}). Consider refreshing.")

        # Step 1: Daily diagnostics (observation only — backtesting only)
        if self.parameters.get("enable_diagnostics", True) and self.trading_mode is TradingMode.BACKTESTING:
            self.vars.diagnostics_logger.compute_and_persist()

        # Step 2: Weekly rebalance
        rebalance_day = calendar.day_name[self.parameters["day_of_week"]]
        self.log_info(f"Today is a {today.strftime('%A')} and the rebalance day is {rebalance_day}")
        if not self.is_rebalance_day():
            self.log_warning("Not a rebalance day — skipping.")
            return

        self.log_info("Today is a rebalance day — computing target portfolio...")

        # Backtests: the scan reads this month's list (no-op in paper/live)
        self._use_month_universe()

        # Paper/live: this loads the scan's Yahoo bars into vars.yahoo_bars. Steps 3-9 read them through
        # get_historical_prices, and rebalance() releases them at the end, so nothing below may fetch a fresh batch.
        target, all_ranks = self.compute_target_portfolio()

        # Step 3: Portfolio Risk Overlay — beta/vol/corr of today's target stocks against SPY, aligned by date.
        # Still reads vars.yahoo_bars (rebalance() has not run yet). After an unusable batch the target is empty, so
        # this returns 1.0 without reading SPY.
        risk_exposure = self._risk_exposure(target)

        # Step 4: Breadth overlay — market regime from the share of scored stocks above their SMA
        breadth_leg = self._breadth_exposure()

        # Step 5: Fast/Slow Volatility Targeting — compute realized vol
        # from actual equity curve using max(vol_20d, 0.75 * vol_63d).
        vt_config = self.parameters["volatility_targeting"]
        vol_exposure = 1.0
        realized_vol = None
        if vt_config["enabled"]:
            realized_vol = self._compute_realized_volatility()
            if realized_vol is not None:
                vol_exposure = compute_volatility_exposure(
                    portfolio_volatility=realized_vol,
                    target_volatility=vt_config["target_volatility"],
                    min_exposure=vt_config["min_exposure"],
                    max_exposure=vt_config["max_exposure"],
                )
                self.log_info(f"Vol targeting (fast/slow): risk_vol={realized_vol:.1%}, target={vt_config['target_volatility']:.0%}, vol_exposure={vol_exposure:.0%}")
            else:
                self.log_warning(f"Vol targeting: insufficient equity history ({len(self.vars.equity_history)} days, need 64) — defaulting to 100% exposure")

        # Step 6: Combine exposures via min() — most conservative leg wins.
        # Using min() rather than multiplication avoids two overlays
        # accidentally creating extremely low exposure.
        final_exposure = min(risk_exposure, vol_exposure, breadth_leg)
        if final_exposure < 1.0:
            self.log_info(f"Combined exposure: {final_exposure:.0%} (risk={risk_exposure:.0%}, breadth={breadth_leg:.0%}, vol={vol_exposure:.0%})")

        # Step 7: Scale target weights by final exposure multiplier
        if target and final_exposure < 1.0:
            for entry in target:
                entry["target_weight"] *= final_exposure
            self.log_info(f"Target weights scaled to {final_exposure:.0%} exposure (total weight: {sum(e['target_weight'] for e in target):.1%})")

        # Step 8: Store target weights for next week's diagnostics
        self.vars.diagnostics_logger.set_last_rebalance_weights({entry["symbol"]: entry["target_weight"] for entry in target})

        # Step 9: Rebalance
        self.rebalance(target, all_ranks)

    def _use_month_universe(self) -> None:
        """Backtests: make `vars.universe` the history's list for today's month, minus the sleeve assets.

        Everything downstream (the scan, the Yahoo batch, the breadth) reads `vars.universe`. A holding that dropped out
        of the list is not ranked and is sold by the existing rule. No history (paper/live) leaves the universe as built.
        A month the history lacks ends the run: the executor would otherwise log the tick as a failure and carry on,
        dropping the week's scan without a trace in the results.
        """
        history: UniverseHistory | None = self.vars.universe_history
        if history is None:
            return
        self.vars.universe = month_universe(history, self._market_date(), exclude=self._sleeve_symbols())

    def _sleeve_symbols(self) -> set[str]:
        """The parking sleeve's symbols (parking, trend assets): never ranked, never part of the stock universe."""
        return set(sleeve_symbols(self.parameters["parking"]))

    def _backtest_preload_assets(self, stocks: Sequence[str] | None = None, benchmark: str | None = None) -> list[Asset]:
        """The stocks (default `vars.universe`), the sleeve symbols and the benchmark, each once: what a backtest loads up front."""
        symbols = list(dict.fromkeys([*(self.vars.universe if stocks is None else stocks), *sleeve_symbols(self.parameters["parking"]), *([benchmark] if benchmark else [])]))
        return [Asset(symbol=symbol) for symbol in symbols]

    def run_backtesting(self, **overrides: Any):
        """Run the strategy in backtesting mode, on the point-in-time universe history.

        The history is the only universe of a backtest (never today's universe file, which holds today's winners). It is
        ensured over the window first, so a failure (`UniverseHistoryError`) stops the run before it starts; the first
        review date is inside the window, so its month is the first one needed.
        """
        start, end = self.resolved_window(overrides.get("start"), overrides.get("end"))
        history = ensure_universe_history(market_day(start), market_day(end))
        self.vars.universe_history = history
        # Recorded in settings.json (the runner writes `dict(strategy.parameters)`)
        universe = universe_parameters(history)
        self.parameters["universe"] = universe
        sleeve = self._sleeve_symbols()
        stocks = window_symbols(history, start, end, exclude=sleeve)
        benchmark = overrides.get("benchmark", self.parameters["benchmark_symbol"])

        if self.vars.history_file_path.exists():
            self.log_info(f"Deleting previous equity history file for a clean backtest: {self.vars.history_file_path}")
            self.vars.history_file_path.unlink()

        # A backtest must start with no breadth history, and never leave a step behind for paper/live
        # (their files are separate by mode).
        if self.vars.breadth_file_path.exists():
            self.log_info(f"Deleting previous breadth step file for a clean backtest: {self.vars.breadth_file_path}")
            self.vars.breadth_file_path.unlink()

        # Preloading every symbol any month's list holds (via `preload_assets`) avoids
        # `compute_target_portfolio`'s thread pool falling back to fetching it one
        # ticker at a time: hundreds of concurrent `yf.download` calls race
        # yfinance's shared session/crumb handling and trip Yahoo's rate limiting,
        # which surfaces as the backtest stalling partway through the universe
        # instead of a clean error.
        defaults: dict[str, Any] = dict(
            # Yahoo (AlpacaBacktestData has not enough history, approximatively 6 years), loaded in batches, failing when Yahoo returns too little
            data_source=partial(UniverseBacktestData, record=universe),
            preload_assets=self._backtest_preload_assets(stocks, benchmark),  # the window's stocks, the sleeve ETFs and the benchmark, in memory
        )
        return super().run_backtesting(**{**defaults, **overrides})
