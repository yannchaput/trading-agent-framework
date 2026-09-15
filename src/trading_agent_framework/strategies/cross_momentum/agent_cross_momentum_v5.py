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
  4. Weekly (Friday): filter, score, rank, select top N, weight by inverse vol
  5. Portfolio risk overlay (beta/vol/corr) — first exposure leg
  6. Fast/slow volatility targeting (equity-curve-based) — second exposure leg
  7. Combined via min(risk_exposure, vol_exposure) — most conservative wins
  8. Hysteresis: sell below rank 35, buy top 20
"""

import calendar
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lumibot.entities import Asset, TradingFee

from lumibot_trading_agent.support.alpaca_support import AlpacaApiRateLimiter, SafeAlpacaBacktesting
from lumibot_trading_agent.support.broker_factory import BrokerFactory
from lumibot_trading_agent.support.helpers import (
    TradingMode,
    build_logs,
    get_thread_capacity,
)
from lumibot_trading_agent.support.wrapping_strategy import WrappingStrategy

from .log_diagnostics import DiagnosticLogger
from .parameters import CONFIG
from .portfolio_risk_overlay import compute_risk_overlay
from .utils import (
    annualized_volatility,
    apply_filters,
    compute_atr_from_df,
    compute_return_from_prices,
    compute_volatility_exposure,
    fractional_qty,
    get_cross_momentum_universe_last_date,
    inverse_volatility_weights,
    load_equity_history,
    momentum_score,
    parse_insufficient_buying_power,
    save_equity_history,
)

logger = logging.getLogger(__name__)


# ── Backtesting params ───────────────────────────────────────────────────────
BACKTESTING_PARAMS = {
    # "backtesting_start": datetime(2022, 1, 1),
    # "backtesting_end": datetime(2026, 5, 31),
    # "backtesting_start": datetime(2016, 1, 1),
    # "backtesting_end": datetime(2026, 5, 31),
    "backtesting_start": datetime(2016, 1, 1),
    "backtesting_end": datetime(2026, 8, 15),
    # Warm-up extends the data window before backtesting_start so the 12-1m
    # momentum lookback (252 + 21 skip + 1 = 274 bars) has full history from
    # day one. Without it, every ticker trips min_trading_days=250 and the
    # first ~year of the backtest can never pass filters.
    "warm_up_trading_days": 300,
    "benchmark_asset": Asset("SPY", Asset.AssetType.STOCK),
    "budget": 10000,
    "buy_trading_fees": [TradingFee(percent_fee=0.001)],
    "sell_trading_fees": [TradingFee(percent_fee=0.001)],
}


class CrossMomentumStrategyV5(WrappingStrategy):
    """Cross-sectional momentum strategy V5 — copy of V2 with fractional share support.

    Tracks actual daily portfolio equity to compute realized volatility
    using both a 20-day (fast) and 63-day (slow) window, then takes the
    more conservative estimate via max(vol_20d, 0.75 * vol_63d). Combined
    with the portfolio risk overlay via min() — the most conservative leg
    determines the final exposure multiplier.
    """

    def __init__(
        self,
        *args,
        mode: TradingMode = TradingMode.BACKTESTING,
        universe: list[str] = [],
        diagnostics_file_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, mode=mode, **kwargs)
        self.params = {**CONFIG, **BACKTESTING_PARAMS}

        if not len(universe):
            self.log_message(
                "No universe provided to the strategy, this is a mandatory parameter",
                color="red",
            )
            sys.exit(1)
        self.__universe = universe or []

        # Diagnostics (delegated to DiagnosticLogger)
        self._diagnostics_logger = DiagnosticLogger(self, trading_mode=self._trading_mode, diagnostics_file_path=diagnostics_file_path)
        self._target_closes: dict[str, list[float]] = {}

        # Track actual portfolio equity for realized-vol computation
        self._equity_history: list[float] = []
        self._history_file_path: Path = Path(f"data/cross_momentum_ptf_history_{self._trading_mode.value}.json")

        # Rate limiter for Alpaca market-data calls (universe filtering bursts
        # past Alpaca's ~200 req/min limit; see support/alpaca_support.py).
        self._alpaca_rate_limiter = AlpacaApiRateLimiter(trading_mode=self._trading_mode)

        self.log_info(
            f"CrossMomentumStrategyV5 initialized with {len(self.__universe)} symbols",
        )

    def initialize(self):
        """Set up the strategy before trading starts."""
        self.sleeptime = "1D"
        self.log_info(f"Sleeptime set to {self.sleeptime}")
        # Track actual portfolio equity for realized-vol computation
        self._equity_history = load_equity_history(self._history_file_path)
        self.log_info(f"Initialize equity history for volatility exposure compute {self._equity_history}")

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
        if len(self._equity_history) < 64:
            return None

        equity_series = pd.Series(self._equity_history)
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
        return dt.weekday() == self.params["day_of_week"]

    def _compute_indicators_for_ticker(self, ticker: str) -> dict | None:
        """Fetch OHLCV data and compute momentum indicators for a single ticker."""
        self._alpaca_rate_limiter.wait()
        bars = self.get_historical_prices(ticker, length=300, timestep="day")
        if bars is None or bars.empty:
            return None

        df = bars.pandas_df if hasattr(bars, "pandas_df") else bars
        if df is None or df.empty:
            return None

        closes = df["close"].tolist()
        volumes = df["volume"].tolist()
        trading_days = len(closes)

        if trading_days < self.params["min_trading_days"]:
            return None

        current_price = closes[-1] if closes else 0.0

        skip = self.params["skip_days"]
        ret_12_1m = compute_return_from_prices(closes, 252, skip)
        ret_6_1m = compute_return_from_prices(closes, 126, skip)
        ret_3m = compute_return_from_prices(closes, 63, 0)

        if ret_12_1m is None or ret_6_1m is None or ret_3m is None:
            return None

        vol = annualized_volatility(closes, self.params["volatility_window"])

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
        }

    def compute_target_portfolio(self) -> tuple[list[dict], dict[str, int]]:
        """Run the full pipeline: indicators → score → filter → rank → select → weight.

        Uses ThreadPoolExecutor for parallel processing of the universe.
        """
        self.log_info(f"Computing target portfolio for {len(self.__universe)} tickers...")

        scored: list[dict] = []
        skip_count = 0
        total = len(self.__universe)

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
                self.params,
            ):
                return None

            result["score"] = momentum_score(
                result["ret_12_1m"],
                result["ret_6_1m"],
                result["ret_3m"],
                self.params["w_12m"],
                self.params["w_6m"],
                self.params["w_3m"],
            )
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(_process_ticker, ticker): ticker for ticker in self.__universe}

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

        top_n = self.params["top_n"]
        selected = scored[:top_n]

        # Store closes for the selected stocks (consumed by portfolio risk overlay)
        self._target_closes = {entry["symbol"]: entry["closes"] for entry in selected}

        selected = inverse_volatility_weights(
            selected,
            self.params["max_position_pct"],
            self.params["min_position_pct"],
        )

        if selected:
            self.log_info(
                f"Target portfolio: {len(selected)} stocks selected. Top: {selected[0]['symbol']} (rank 1, score {selected[0]['score']:.4f})",
            )
        return selected, all_ranks

    def rebalance(self, target: list[dict], all_ranks: dict[str, int]) -> None:
        """Compare current holdings to target, apply hysteresis, submit buy/sell orders.

        Args:
            target: Selected stocks with target_weight (top N, already weighted).
            all_ranks: Symbol→rank mapping for ALL scored stocks.
        """
        if not target:
            return

        target_symbols = {entry["symbol"] for entry in target}

        sell_threshold = self.params["sell_rank_threshold"]
        portfolio_value = self.portfolio_value or 1.0

        current_positions = self.get_positions()

        # Phase 1: Sell
        estimated_sell_proceeds = 0.0
        for pos in current_positions:
            symbol = pos.asset.symbol
            if symbol in target_symbols:
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
                    last_price = self.get_last_price(symbol) or 0.0
                    estimated_sell_proceeds += pos.quantity * last_price
                except Exception as e:
                    self.log_error(f"Failed to submit sell order for {symbol}: {e}")
            # Rank <= sell_threshold means keep the position inside the histeresis band (do not sell)
            else:
                self.log_info(f"Keeping {symbol} (rank {rank} ≤ {sell_threshold}, within hysteresis band)")

        # Phase 2: Buy
        available_cash = self.get_cash() + estimated_sell_proceeds
        for entry in target:
            # Available cash safe guard: if available cash is zero or negative, skip remaining target stocks
            if available_cash <= 0:
                self.log_warning("No available cash left for buying — skipping remaining target stocks.")
                break

            symbol = entry["symbol"]
            target_weight = entry["target_weight"]
            target_value = portfolio_value * target_weight

            current_pos = next((p for p in current_positions if p.asset.symbol == symbol), None)
            current_value = current_pos.quantity * entry["price"] if current_pos else 0.0

            # Compute the difference between target and current value position
            diff_value = target_value - current_value

            # If the position is already above target, skip buying
            if diff_value <= 0:
                continue

            # If the current position is within ±20% of the target value, skip buying to avoid overtrading
            if current_value > 0 and abs(diff_value) / target_value < 0.20:
                continue

            # Compute the fractional quantity to buy taking into account the existing position
            raw_qty = diff_value / entry["price"]
            quantity = fractional_qty(raw_qty)
            # Current position is already above target, skip buying
            if quantity <= 0:
                continue

            cost = quantity * entry["price"]
            # Estimate the cost of the remaining quantity to buy, if it exceeds 95% of available cash, reduce the quantity to fit within 95% of available cash
            if cost > available_cash * 0.95:
                quantity = fractional_qty(available_cash * 0.95 / entry["price"])
                cost = quantity * entry["price"]
            # quantity can be equal to 0 for very tiny amount passed to fractional_qty or available cash can become negative due to substraction below in the loop
            # provoking a negative division
            if quantity <= 0:
                continue

            # If we reach this line: either the cost < available cash * 0.95 and so available_cash will be positive
            # or the cost > available cash * 0.95 and so we reduced the quantity to fit within 95% of available cash and so available_cash will be positive
            available_cash -= cost

            self.log_info(f"Buying {quantity} {symbol} @ ${entry['price']:.2f} (target weight: {target_weight:.1%}, rank: {entry['rank']})")
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
                    available_cash = real_buying_power

    # ── Main iteration ────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        super().on_trading_iteration()

        # Record daily equity for realized-vol tracking
        equity = self.portfolio_value or 0.0
        if equity > 0:
            # Compute history file name based on trading mode
            self._equity_history.append(equity)
            today_str = self.get_datetime().strftime("%Y-%m-%d")
            save_equity_history(self._equity_history, today_str, self._history_file_path)

        today: datetime = self.get_datetime()
        last_universe_date = get_cross_momentum_universe_last_date()

        if self._trading_mode is not TradingMode.BACKTESTING:
            if not last_universe_date:
                self.log_warning("No universe date found — universe file missing or empty.")
                return
            if (today - last_universe_date).days > 30:
                self.log_warning(f"Warning: universe is {(today - last_universe_date).days} days old (last update: {last_universe_date:%Y-%m-%d}). Consider refreshing.")

        # Step 1: Daily diagnostics (observation only — backtesting only)
        if self.params.get("enable_diagnostics", True) and self._trading_mode is TradingMode.BACKTESTING:
            self._diagnostics_logger.compute_and_persist()

        # Step 2: Weekly rebalance
        rebalance_day = calendar.day_name[self.params["day_of_week"]]
        self.log_info(f"Today is a {today.strftime('%A')} and the rebalance day is {rebalance_day}")
        if not self.is_rebalance_day():
            self.log_warning("Not a rebalance day — skipping.")
            return

        self.log_info("Today is a rebalance day — computing target portfolio...")

        target, all_ranks = self.compute_target_portfolio()

        # Step 3: Portfolio Risk Overlay — compute beta/vol/corr from
        # today's target stocks' historical closes. This is the "hypothetical
        # risk of today's portfolio" leg.
        risk_exposure = 1.0
        risk_metrics = {}
        if target and self._target_closes:
            self._alpaca_rate_limiter.wait()
            spy_bars = self.get_historical_prices("SPY", length=300, timestep="day")
            spy_closes = None
            if spy_bars is not None and not spy_bars.empty:
                spy_df = spy_bars.pandas_df if hasattr(spy_bars, "pandas_df") else spy_bars
                if spy_df is not None and not spy_df.empty:
                    spy_closes = spy_df["close"].tolist()

            if spy_closes is None:
                self.log_warning("Risk overlay: SPY data unavailable — defaulting to NORMAL (100% exposure)")
            else:
                target_weights = {entry["symbol"]: entry["target_weight"] for entry in target}
                risk_state, risk_exposure, risk_metrics = compute_risk_overlay(
                    self._target_closes,
                    target_weights,
                    spy_closes,
                )
                self.log_message(
                    f"Risk overlay: {risk_state.upper()} (beta={risk_metrics.get('beta_63d')}, vol={risk_metrics.get('vol_20d')}, corr={risk_metrics.get('corr_20d')}, exposure={risk_exposure:.0%})",
                    color="cyan" if risk_state == "normal" else "red",
                )

        # Step 4: Fast/Slow Volatility Targeting — compute realized vol
        # from actual equity curve using max(vol_20d, 0.75 * vol_63d).
        vt_config = self.params.get("volatility_targeting", {})
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
                self.log_warning(f"Vol targeting: insufficient equity history ({len(self._equity_history)} days, need 64) — defaulting to 100% exposure")

        # Step 5: Combine exposures via min() — most conservative leg wins.
        # Using min() rather than multiplication avoids two overlays
        # accidentally creating extremely low exposure.
        final_exposure = min(risk_exposure, vol_exposure)
        if final_exposure < 1.0:
            self.log_info(f"Combined exposure: {final_exposure:.0%} (risk={risk_exposure:.0%}, vol={vol_exposure:.0%})")

        # Step 6: Scale target weights by final exposure multiplier
        if target and final_exposure < 1.0:
            for entry in target:
                entry["target_weight"] *= final_exposure
            self.log_info(f"Target weights scaled to {final_exposure:.0%} exposure (total weight: {sum(e['target_weight'] for e in target):.1%})")

        # Step 7: Store target weights for next week's diagnostics
        self._diagnostics_logger.set_last_rebalance_weights({entry["symbol"]: entry["target_weight"] for entry in target})

        # Step 8: Rebalance
        self.rebalance(target, all_ranks)

    def run_backtesting(self):
        """Run the strategy in backtesting mode using AlpacaBacktesting."""
        super().run_backtesting()
        log_dir: Path = build_logs(f"agent_{self.name}", TradingMode.BACKTESTING)
        diagnostics_file_path = str(log_dir / "diagnostics.parquet")
        self.log_info(f"Logs will be saved to: {log_dir}")

        # Delete equity curve history file from the previous run if exists, to ensure a clean backtest
        self.log_debug(f"Checking for existing equity history file: {self._history_file_path}")
        if self._history_file_path.exists():
            self.log_debug(f"Found existing equity history file: {self._history_file_path}")
            self._history_file_path.unlink()
            self.log_info(f"Deleted previous equity history file: {self._history_file_path}")

        self.backtest(
            SafeAlpacaBacktesting,
            name=self.name,
            backtesting_start=self.params["backtesting_start"],
            backtesting_end=self.params["backtesting_end"],
            warm_up_trading_days=self.params["warm_up_trading_days"],
            benchmark_asset=self.params["benchmark_asset"],
            buy_trading_fees=self.params["buy_trading_fees"],
            sell_trading_fees=self.params["sell_trading_fees"],
            budget=self.params["budget"],
            quiet_logs=False,
            logfile=str(log_dir / "backtest.log"),
            stats_file=str(log_dir / "stats.csv"),
            save_logfile=True,
            config=BrokerFactory.get_alpaca_config(),
            universe=self.__universe or [],
            diagnostics_file_path=diagnostics_file_path,
        )
