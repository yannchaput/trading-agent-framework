"""
Cross-Sectional Momentum Strategy (V5)
--------------------------------------
Fractional share support — floor-based quantities in rebalance().

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.alpaca import AlpacaApiRateLimiter
from trading_agent_framework.config import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError, YahooDataError
from trading_agent_framework.utils.helpers import (
    get_thread_capacity,
)

from .log_diagnostics import DiagnosticLogger
from .parameters import CONFIG
from .portfolio_risk_overlay import compute_risk_overlay
from .utils import (
    annualized_volatility,
    apply_filters,
    breadth_exposure,
    breadth_share,
    close_series,
    completed_bars,
    compute_atr_from_df,
    compute_return_from_prices,
    compute_volatility_exposure,
    fractional_qty,
    get_cross_momentum_universe_last_date,
    inverse_volatility_weights,
    load_breadth_step,
    load_equity_history,
    momentum_score,
    next_breadth_step,
    parse_insufficient_buying_power,
    parse_rebalance_time,
    save_breadth_step,
    save_equity_history,
    sleeve_symbols,
    sleeve_weights,
    trend_reading,
)
from .yahoo_daily_bars import YahooDailyBars

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
        super().__init__(broker, mode=mode, **kwargs)
        self.parameters = {**CONFIG, **self.parameters}
        # The executor runs each session's iteration, so the Tuesday rebalance, at this market time
        self.iteration_start_time = parse_rebalance_time(self.parameters["rebalance_time"])

        if not universe:
            self.log_warning("No universe provided to the strategy, this is a mandatory parameter")
            sys.exit(1)
        # A sleeve asset (SHV, GLD, IEF) is never scored or bought as a stock, even if the universe file lists it
        sleeve = set(sleeve_symbols(self.parameters["parking"]))
        self.vars.universe = [symbol for symbol in universe if symbol not in sleeve]

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
        self.log_info(f"Sleeptime set to {self.sleeptime}")
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

    def _compute_indicators_for_ticker(self, ticker: str) -> dict | None:
        """Fetch OHLCV data and compute momentum indicators for a single ticker."""
        try:
            bars = self.get_historical_prices(ticker, length=_HISTORY_BARS + 1, timestep="day")
        except BrokerError as exc:
            self.log_warning(f"Skipping {ticker}: failed to fetch bars ({exc})")
            return None
        if bars is None or bars.empty:
            return None

        df = bars.pandas_df if hasattr(bars, "pandas_df") else bars
        if not isinstance(df, pd.DataFrame) or df.empty:
            self.log_error("The Bars instance has not the right type: consider using a pandas Dataframe.")
            return None

        # Completed sessions only: today's bar is partial, and it made the same Tuesday's decision depend on the
        # hour it ran (2026-10-06). One extra bar is fetched so the window stays _HISTORY_BARS long.
        df = completed_bars(df, self._market_date()).tail(_HISTORY_BARS)
        if df.empty:
            return None

        closes = df["close"].tolist()
        volumes = df["volume"].tolist()
        trading_days = len(closes)

        if trading_days < self.parameters["min_trading_days"]:
            return None

        current_price = closes[-1] if closes else 0.0

        skip = self.parameters["skip_days"]
        ret_12_1m = compute_return_from_prices(closes, 252, skip)
        ret_6_1m = compute_return_from_prices(closes, 126, skip)
        ret_3m = compute_return_from_prices(closes, 63, 0)

        if ret_12_1m is None or ret_6_1m is None or ret_3m is None:
            return None

        vol = annualized_volatility(closes, self.parameters["volatility_window"])

        if len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
        else:
            avg_volume = sum(volumes) / max(len(volumes), 1)
        avg_dollar_volume = avg_volume * current_price

        atr = compute_atr_from_df(df) if trading_days >= 15 else None

        return {
            "symbol": ticker,
            "price": current_price,
            "ret_12_1m": ret_12_1m,
            "ret_6_1m": ret_6_1m,
            "ret_3m": ret_3m,
            "volatility": vol,
            "avg_dollar_volume": avg_dollar_volume,
            "trading_days": trading_days,
            "atr": atr,
            "closes": closes,
            "close_series": close_series(df),
        }

    def _load_yahoo_bars(self) -> bool:
        """Fetch the scan's daily bars from Yahoo into `vars.yahoo_bars`: the universe, SPY and the sleeve's trend assets.

        A failed or incomplete batch is fetched again after each of `_YAHOO_RETRY_DELAYS` (a transient 429 must not cost
        the week's only rebalance). False (logged) when the last attempt still fails: the caller then skips the scan
        and holds the book.
        """
        for delay in (*_YAHOO_RETRY_DELAYS, None):
            bars, problem = self._fetch_yahoo_bars()
            if problem is None:
                self.vars.yahoo_bars = bars
                covered = sum(1 for symbol in self.vars.universe if symbol in bars)
                self.log_info(f"Yahoo bars: {covered}/{len(self.vars.universe)} symbols")
                return True
            if delay is None:
                self.log_error(f"{problem} — holding current positions.")
                return False
            self.log_warning(f"{problem} — retry in {delay:.0f}s")
            self.sleep(delay)  # clock time, so order hooks still fire meanwhile
        return False  # unreachable: the last delay is None

    def _fetch_yahoo_bars(self) -> tuple[dict[str, Bars], str | None]:
        """One attempt: the bars, and None when they are usable, else a description of the problem.

        Unusable when Yahoo fails, has bars for too few stocks, or misses a held stock (an unranked holding would be
        sold). Any other stock Yahoo has no bars for is simply filtered out, like any other ticker without data.
        """
        universe = self.vars.universe
        extras = ["SPY", *self.parameters["parking"]["trend_assets"]]
        try:
            bars = self.vars.bars_source.bars([*universe, *extras], self._market_date())
        except YahooDataError as exc:
            return {}, f"Yahoo lookup failed ({exc})"
        covered = sum(1 for symbol in universe if symbol in bars)
        if covered < len(universe) * _MIN_YAHOO_COVERAGE:
            return {}, f"Yahoo covers only {covered}/{len(universe)} symbols"
        # A holding outside the universe was never requested: the rebalance sells it as "not ranked" by design
        in_universe = set(universe)
        missing_held = sorted(({pos.asset.symbol for pos in self.get_positions()} & in_universe) - bars.keys())
        if missing_held:
            return {}, f"Yahoo has no bars for held {', '.join(missing_held)}"
        return bars, None

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
                dict(self.parameters),
            ):
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
        self.vars.target_closes = {entry["symbol"]: entry["close_series"] for entry in selected}

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
            f"Risk overlay: {risk_state.upper()} (beta={metrics.get('beta_63d')}, vol={metrics.get('vol_20d')}, corr={metrics.get('corr_20d')}, "
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

    def _rebalance_book(self, target: list[dict], all_ranks: dict[str, int]) -> None:
        """Compare holdings to target, apply hysteresis, trim, submit orders, and park the rest in the sleeve.

        Args:
            target: Selected stocks with target_weight (top N, already weighted).
            all_ranks: Symbol→rank mapping for ALL scored stocks.
        """
        if not target:
            return

        target_symbols = {entry["symbol"] for entry in target}
        target_by_symbol = {entry["symbol"]: entry for entry in target}
        parking = self.parameters["parking"]
        # Trend assets first, SHV last: the order of the sleeve buys once the stock buys are funded
        sleeve_order = (*parking["trend_assets"], parking["symbol"])

        sell_threshold = self.parameters["sell_rank_threshold"]
        portfolio_value = float(self.portfolio_value or 1.0)
        min_trade_value = portfolio_value * parking["min_trade_pct"]

        current_positions = self.get_positions()

        # Size at the last trade: an entry's "price" is the last COMPLETED close, a session old by now.
        prices = {entry["symbol"]: self._price_or_zero(entry["symbol"]) for entry in target}
        for symbol, price in prices.items():
            if price <= 0:
                self.log_warning(f"No price for {symbol}: no buy or trim this week")

        # The trend data is fetched before any order is sent: an unexpected fetch error then aborts the rebalance cleanly
        weights = self._sleeve_weights()

        # Phase 1: Sell (exits, trims, excess parking)
        estimated_sell_proceeds = 0.0
        hysteresis_value = 0.0
        kept_count = 0
        sold_count = 0
        sleeve_positions = {}
        for pos in current_positions:
            symbol = pos.asset.symbol
            if symbol in sleeve_order:
                # The parking sleeve is never ranked: it must not be exited as "not ranked" below
                sleeve_positions[symbol] = pos
                continue
            if symbol in target_symbols:
                # Trim a target position that sits above its band, so an exposure cut lowers the held book
                # too, not just new buys; the proceeds are parked or fund this week's buys.
                entry = target_by_symbol[symbol]
                price = prices[symbol]
                if price <= 0:
                    continue
                target_value = portfolio_value * entry["target_weight"]
                current_value = float(pos.quantity) * price
                trim_value = current_value - target_value
                if current_value > target_value * (1 + _REBALANCE_BAND) and trim_value >= min_trade_value:
                    trim_qty = fractional_qty(trim_value / price)
                    if trim_qty > 0:
                        self.log_info(f"Trimming {trim_qty} {symbol} @ ${price:.2f} (value ${current_value:,.0f} > target ${target_value:,.0f})")
                        try:
                            self.submit_order(self.create_order(symbol, trim_qty, "sell", time_in_force="day"))
                            estimated_sell_proceeds += trim_qty * price
                        except Exception as e:
                            self.log_error(f"Failed to submit trim order for {symbol}: {e}")
                continue

            rank = all_ranks.get(symbol)
            if rank is None or rank > sell_threshold:
                if rank is None:
                    # The position did not pass the scoring pipeline and was kicked out by the filters (volatility, price or not able to compute indicators)
                    self.log_warning(f"Selling {symbol} (not ranked — failed filters or price data unavailable)")
                else:
                    # The position is ranked but outside the sell threshold (hysteresis band)
                    self.log_warning(f"Selling {symbol} (rank {rank} > {sell_threshold}) - outside hysteresis band")
                try:
                    sell_order = self.create_order(symbol, pos.quantity, "sell", time_in_force="day")
                    self.submit_order(sell_order)
                    sold_count += 1
                    last_price = self.get_last_price(symbol) or 0.0
                    estimated_sell_proceeds += float(pos.quantity) * float(last_price)
                except Exception as e:
                    self.log_error(f"Failed to submit sell order for {symbol}: {e}")
            # Rank <= sell_threshold means keep the position inside the histeresis band (do not sell)
            else:
                self.log_info(f"Keeping {symbol} (rank {rank} ≤ {sell_threshold}, within hysteresis band)")
                hysteresis_value += float(pos.quantity) * self._price_or_zero(symbol)
                kept_count += 1

        # Parking target: everything not meant for stocks, except the cash reserve, goes to the sleeve
        stock_target_value = portfolio_value * sum(entry["target_weight"] for entry in target)
        parking_target = max(0.0, portfolio_value * (1 - self.parameters["cash_buffer_pct"]) - stock_target_value - hysteresis_value)
        sleeve: list[tuple[str, float, float, float]] = []  # (symbol, price, current value, target value), priced only
        for symbol in sleeve_order:
            position = sleeve_positions.get(symbol)
            price = self._price_or_zero(symbol)
            if price <= 0:
                self.log_warning(f"Parking: no price for {symbol} — no {symbol} order this week")
                continue
            value = float(position.quantity) * price if position else 0.0
            symbol_target = parking_target * weights.get(symbol, 0.0)
            sleeve.append((symbol, price, value, symbol_target))
            self.log_info(f"Parking: {symbol} target ${symbol_target:,.0f} (current ${value:,.0f})")
            excess = value - symbol_target
            if value > symbol_target * (1 + _REBALANCE_BAND) and excess >= min_trade_value:
                # A zero target sells the exact holding, so float flooring leaves no dust behind.
                # Guard against a missing position so static analysis does not treat it as a known quantity.
                sell_qty = float(position.quantity) if position is not None and symbol_target == 0 else fractional_qty(excess / price)
                if sell_qty > 0:
                    self.log_info(f"Selling {sell_qty} {symbol} @ ${price:.2f} (parking above target)")
                    try:
                        self.submit_order(self.create_order(symbol, sell_qty, "sell", time_in_force="day"))
                        estimated_sell_proceeds += sell_qty * price
                    except Exception as e:
                        self.log_error(f"Failed to submit parking sell order for {symbol}: {e}")

        # Phase 2: Buy
        # The estimate is priced at the last trade, but orders fill a little later at the market (at the next
        # bar's open in a backtest) plus fees, so hold a fixed reserve back: it is what lets the last buys land
        # when prices move or sells fill lower.
        cash_reserve = (float(self.get_cash()) + estimated_sell_proceeds) * self.parameters["cash_buffer_pct"]
        available_cash = float(self.get_cash()) + estimated_sell_proceeds - cash_reserve
        for entry in target:
            # Available cash safe guard: if available cash is zero or negative, skip remaining target stocks
            if available_cash <= 0:
                self.log_warning("No available cash left for buying — skipping remaining target stocks.")
                break

            symbol = entry["symbol"]
            price = prices[symbol]
            if price <= 0:
                continue
            target_weight = entry["target_weight"]
            target_value = portfolio_value * target_weight

            current_pos = next((p for p in current_positions if p.asset.symbol == symbol), None)
            current_value = float(current_pos.quantity) * price if current_pos else 0.0

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
            try:
                buy_order = self.create_order(symbol, quantity, "buy", time_in_force="day")
                self.submit_order(buy_order)
            except Exception as e:
                self.log_warning(f"Failed to submit buy order for {symbol}: {e}")
                # The local available_cash estimate (get_cash() + estimated unfilled
                # sell proceeds) can drift from the broker's real buying power —
                # e.g. this iteration's sells hadn't settled yet. Resync to the
                # broker's own reported value so the remaining targets in this loop
                # aren't sized against a number already proven wrong, instead of
                # repeating the same rejection down the list.
                real_buying_power = parse_insufficient_buying_power(e)
                if real_buying_power is not None:
                    self.log_warning(f"Resyncing available cash to broker-reported buying power: ${real_buying_power:.2f}")
                    available_cash = real_buying_power * (1 - self.parameters["cash_buffer_pct"])

        # Phase 3: Park what the stock buys left, up to each sleeve target (the trend assets first, SHV last)
        for symbol, price, value, symbol_target in sleeve:
            if value >= symbol_target * (1 - _REBALANCE_BAND):
                continue
            buy_value = min(symbol_target - value, available_cash)
            if buy_value < min_trade_value:
                continue
            quantity = fractional_qty(buy_value / price)
            if quantity <= 0:
                continue
            available_cash -= quantity * price
            self.log_info(f"Buying {quantity} {symbol} @ ${price:.2f} (parking)")
            try:
                self.submit_order(self.create_order(symbol, quantity, "buy", time_in_force="day"))
            except Exception as e:
                self.log_warning(f"Failed to submit parking buy order for {symbol}: {e}")

        # The target positions never get a line unless they are bought, so say how the book is made up
        held_targets = sum(1 for pos in current_positions if pos.asset.symbol in target_symbols)
        self.log_info(f"Rebalance summary: {len(target)} targets ({held_targets} already held, {len(target) - held_targets} not held), {kept_count} kept under hysteresis, {sold_count} sold")

    # ── Main iteration ────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        super().on_trading_iteration()

        # Record daily equity for realized-vol tracking
        equity = self.portfolio_value or 0.0
        if equity > 0:
            # Compute history file name based on trading mode
            self.vars.equity_history.append(float(equity))
            today_str = self.get_datetime().strftime("%Y-%m-%d")
            save_equity_history(self.vars.equity_history, today_str, self.vars.history_file_path)

        today: datetime = self.get_datetime()
        last_universe_date = get_cross_momentum_universe_last_date()

        if self.trading_mode is not TradingMode.BACKTESTING:
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
        vt_config = self.parameters.get("volatility_targeting", {})
        vol_exposure = 1.0
        realized_vol = None
        if vt_config.get("enabled", False):
            realized_vol = self._compute_realized_volatility()
            if realized_vol is not None:
                vol_exposure = compute_volatility_exposure(
                    portfolio_volatility=realized_vol,
                    target_volatility=vt_config.get("target_volatility", 0.20),
                    min_exposure=vt_config.get("min_exposure", 0.40),
                    max_exposure=vt_config.get("max_exposure", 1.00),
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

    def _backtest_preload_assets(self) -> list[Asset]:
        """The universe plus the sleeve symbols, each once: what a backtest loads up front."""
        symbols = list(dict.fromkeys([*self.vars.universe, *sleeve_symbols(self.parameters["parking"])]))
        return [Asset(symbol=symbol) for symbol in symbols]

    def run_backtesting(self, **overrides: Any):
        """Run the strategy in backtesting mode."""
        if self.vars.history_file_path.exists():
            self.log_info(f"Deleting previous equity history file for a clean backtest: {self.vars.history_file_path}")
            self.vars.history_file_path.unlink()

        # A backtest must start with no breadth history, and never leave a step behind for paper/live
        # (their files are separate by mode).
        if self.vars.breadth_file_path.exists():
            self.log_info(f"Deleting previous breadth step file for a clean backtest: {self.vars.breadth_file_path}")
            self.vars.breadth_file_path.unlink()

        # Preloading the whole universe (via `preload_assets`) avoids
        # `compute_target_portfolio`'s thread pool falling back to fetching it one
        # ticker at a time: hundreds of concurrent `yf.download` calls race
        # yfinance's shared session/crumb handling and trip Yahoo's rate limiting,
        # which surfaces as the backtest stalling partway through the universe
        # instead of a clean error.
        defaults: dict[str, Any] = dict(
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            budget=self.parameters["budget"],
            data_source=YahooBacktestData,  # AlpacaBacktestData has no enough history, approximatively 6 year history
            preload_assets=self._backtest_preload_assets(),  # preload the ticker universe and the sleeve ETFs in memory
            benchmark=self.parameters["benchmark_symbol"],
            warmup_trading_days=self.parameters["warmup_trading_days"],
        )
        return super().run_backtesting(**{**defaults, **overrides})
