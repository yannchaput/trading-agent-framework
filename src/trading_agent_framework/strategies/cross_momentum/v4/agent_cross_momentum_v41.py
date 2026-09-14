"""
Cross-Sectional Momentum Strategy (V41)
---------------------------------------
Beta-aware volatility targeting — experimental variant.

Frozen V2 baseline with a single change: the portfolio exposure multiplier
is replaced by a beta-aware formula:

  vol_scalar  = target_vol / realized_vol_20d          (capped at 1.0)
  beta_scalar = stepped function on portfolio beta_63d  (only reduces)
  exposure    = min(vol_scalar, beta_scalar)

The beta scalar is a coarse stepped function:
    beta_63d ≤ 1.25  →  1.00  (no reduction)
    beta_63d ≤ 1.50  →  0.85
    beta_63d ≤ 1.75  →  0.70
    beta_63d > 1.75  →  0.55

Beta must only *reduce* exposure — never increase it. The vol scalar is
similarly capped at 1.00. The portfolio risk overlay (beta/vol/corr) from
V2 is preserved unchanged; the final exposure is min(risk_exposure,
beta_aware_exposure).

Why this experiment: 18 stocks with beta 1.7–2.5 are still one large
high-beta momentum trade. This specifically targets the 2021–2023 and 2024
drawdowns without interfering much during normal periods.
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lumibot.entities import Asset, TradingFee

from lumibot_trading_agent.support.alpaca_support import SafeAlpacaBacktesting
from lumibot_trading_agent.support.broker_factory import BrokerFactory
from lumibot_trading_agent.support.helpers import (
    TradingMode,
    build_logs,
    get_thread_capacity,
)
from lumibot_trading_agent.support.wrapping_strategy import WrappingStrategy

from ..log_diagnostics import DiagnosticLogger
from ..parameters import CONFIG
from ..portfolio_risk_overlay import compute_risk_overlay
from ..utils import (
    annualized_volatility,
    apply_filters,
    compute_atr_from_df,
    compute_return_from_prices,
    get_cross_momentum_universe_last_date,
    inverse_volatility_weights,
    load_equity_history,
    momentum_score,
    save_equity_history,
)

logger = logging.getLogger(__name__)

# ── Backtesting params ───────────────────────────────────────────────────────
BACKTESTING_PARAMS = {
    # "backtesting_start": datetime(2022, 1, 1),
    # "backtesting_end": datetime(2026, 5, 31),
    "backtesting_start": datetime(2016, 1, 1),
    "backtesting_end": datetime(2026, 5, 31),
    "benchmark_asset": Asset("SPY", Asset.AssetType.STOCK),
    "budget": 10000,
    "buy_trading_fees": [TradingFee(percent_fee=0.001)],
    "sell_trading_fees": [TradingFee(percent_fee=0.001)],
}

# ── Beta-aware scalar thresholds (V41 only) ──────────────────────────────────
# Coarse stepped function — beta must only *reduce* exposure, never increase.
_BETA_SCALAR_BANDS: list[tuple[float, float]] = [
    (1.25, 1.00),  # beta ≤ 1.25 → no reduction
    (1.50, 0.85),  # beta ≤ 1.50 → 85% exposure
    (1.75, 0.70),  # beta ≤ 1.75 → 70% exposure
    (float("inf"), 0.55),  # beta > 1.75 → 55% exposure
]


def _compute_beta_scalar(beta: float) -> float:
    """Compute the beta exposure scalar from portfolio beta.

    Beta only reduces exposure — the returned scalar is always ≤ 1.0.

    Args:
        beta: Portfolio beta relative to benchmark (63d).

    Returns:
        Exposure scalar in [0, 1].
    """
    for threshold, scalar in _BETA_SCALAR_BANDS:
        if beta <= threshold:
            return scalar
    return 1.0  # fallback (should never be reached)


class CrossMomentumStrategyV41(WrappingStrategy):
    """Cross-sectional momentum strategy V41 — beta-aware volatility targeting.

    Identical to V2 except the portfolio exposure multiplier uses a
    beta-aware formula: vol_scalar = target_vol / realized_vol_20d,
    beta_scalar = stepped function on portfolio_beta_63d, and
    exposure = min(vol_scalar, beta_scalar).

    The risk overlay from V2 is preserved unchanged. Final exposure is
    min(risk_exposure, beta_aware_exposure).
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
        self._equity_history: list[float] = load_equity_history()

        self.log_message(
            f"CrossMomentumStrategyV41 initialized with {len(self.__universe)} symbols",
            color="green",
        )

    def initialize(self):
        """Set up the strategy before trading starts."""
        self.sleeptime = "1D"
        self.log_message(f"Sleeptime set to {self.sleeptime}", color="yellow")

    # ── Realized volatility from equity curve (20d only) ───────────────────

    def _compute_realized_volatility_20d(self) -> float | None:
        """Compute annualized realized volatility from 20-day equity curve.

        Uses only the fast 20-day window (cf. V2's fast/slow max approach).
        The beta scalar provides the structural risk signal that V2's 63d
        slow window used to cover.

        Returns:
            Annualized volatility as a float, or None if insufficient history.
        """
        if len(self._equity_history) < 22:
            return None

        equity_series = pd.Series(self._equity_history)
        daily_returns = equity_series.pct_change().dropna()

        if len(daily_returns) < 20:
            return None

        vol_20 = float(np.std(daily_returns.iloc[-20:], ddof=1) * np.sqrt(252))
        return vol_20

    # ── Momentum / portfolio construction (identical to V2) ─────────────────

    def is_rebalance_day(self) -> bool:
        """True if today is a Friday (rebalance day)."""
        dt = self.get_datetime()
        return dt.weekday() == self.params["day_of_week"]

    def _compute_indicators_for_ticker(self, ticker: str) -> dict | None:
        """Fetch OHLCV data and compute momentum indicators for a single ticker."""
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
        self.log_message(
            f"Computing target portfolio for {len(self.__universe)} tickers...",
            color="yellow",
        )

        scored: list[dict] = []
        skip_count = 0
        total = len(self.__universe)

        max_workers = get_thread_capacity()
        self.log_message(f"Using {max_workers} threads for parallel processing", color="yellow")

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
                    self.log_message(
                        f"  Processed {i}/{total} — {len(scored)} passed filters so far",
                        color="yellow",
                    )

        self.log_message(
            f"Indicators: {len(scored)} passed filters, {skip_count} skipped",
            color="yellow",
        )

        if not scored:
            self.log_message("No stocks passed filters — holding current positions.", color="red")
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
            self.log_message(
                f"Target portfolio: {len(selected)} stocks selected. Top: {selected[0]['symbol']} (rank 1, score {selected[0]['score']:.4f})",
                color="green",
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
                self.log_message(
                    f"Selling {symbol} (rank {rank or 'N/A'} > {sell_threshold})",
                    color="red",
                )
                try:
                    sell_order = self.create_order(symbol, pos.quantity, "sell")
                    self.submit_order(sell_order)
                    last_price = self.get_last_price(symbol) or 0.0
                    estimated_sell_proceeds += pos.quantity * last_price
                except Exception as e:
                    self.log_message(f"Failed to submit sell order for {symbol}: {e}", color="red")
            else:
                self.log_message(
                    f"Keeping {symbol} (rank {rank} ≤ {sell_threshold}, within hysteresis band)",
                    color="yellow",
                )

        # Phase 2: Buy
        available_cash = self.get_cash() + estimated_sell_proceeds
        for entry in target:
            symbol = entry["symbol"]
            target_weight = entry["target_weight"]
            target_value = portfolio_value * target_weight

            current_pos = next((p for p in current_positions if p.asset.symbol == symbol), None)
            current_value = current_pos.quantity * entry["price"] if current_pos else 0.0

            diff_value = target_value - current_value

            if diff_value <= 0:
                continue

            if current_value > 0 and abs(diff_value) / target_value < 0.20:
                continue

            quantity = int(diff_value / entry["price"])
            if quantity <= 0:
                continue

            cost = quantity * entry["price"]
            if cost > available_cash * 0.95:
                quantity = int(available_cash * 0.95 / entry["price"])
                cost = quantity * entry["price"]
            if quantity <= 0:
                continue

            available_cash -= cost

            self.log_message(
                f"Buying {quantity} {symbol} @ ${entry['price']:.2f} (target weight: {target_weight:.1%}, rank: {entry['rank']})",
                color="green",
            )
            try:
                buy_order = self.create_order(symbol, quantity, "buy")
                self.submit_order(buy_order)
            except Exception as e:
                self.log_message(f"Failed to submit buy order for {symbol}: {e}", color="red")

    # ── Main iteration ────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        super().on_trading_iteration()

        # Record daily equity for realized-vol tracking
        equity = self.portfolio_value or 0.0
        if equity > 0:
            self._equity_history.append(equity)
            today_str = self.get_datetime().strftime("%Y-%m-%d")
            save_equity_history(self._equity_history, today_str)

        today = self.get_datetime()
        last_universe_date = get_cross_momentum_universe_last_date()

        if self._trading_mode is not TradingMode.BACKTESTING:
            if not last_universe_date:
                self.log_message("No universe date found — universe file missing or empty.", color="red")
                return
            if (today - last_universe_date).days > 30:
                self.log_message(
                    f"Warning: universe is {(today - last_universe_date).days} days old (last update: {last_universe_date:%Y-%m-%d}). Consider refreshing.",
                    color="yellow",
                )

        # Step 1: Daily diagnostics (observation only — backtesting only)
        if self.params.get("enable_diagnostics", True) and self._trading_mode is TradingMode.BACKTESTING:
            self._diagnostics_logger.compute_and_persist()

        # Step 2: Weekly rebalance
        if not self.is_rebalance_day():
            self.log_message("Not a rebalance day — skipping.", color="yellow")
            return

        self.log_message("Friday rebalance — computing target portfolio...", color="cyan")

        target, all_ranks = self.compute_target_portfolio()

        # Step 3: Portfolio Risk Overlay — compute beta/vol/corr from
        # today's target stocks' historical closes. This is the "hypothetical
        # risk of today's portfolio" leg (identical to V2).
        risk_exposure = 1.0
        risk_metrics: dict = {}
        if target and self._target_closes:
            spy_bars = self.get_historical_prices("SPY", length=300, timestep="day")
            spy_closes = None
            if spy_bars is not None and not spy_bars.empty:
                spy_df = spy_bars.pandas_df if hasattr(spy_bars, "pandas_df") else spy_bars
                if spy_df is not None and not spy_df.empty:
                    spy_closes = spy_df["close"].tolist()

            if spy_closes is None:
                self.log_message(
                    "Risk overlay: SPY data unavailable — defaulting to NORMAL (100% exposure)",
                    color="yellow",
                )
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

        # Step 4: Beta-Aware Volatility Targeting (V41 — replaces V2's fast/slow vol targeting)
        #
        # vol_scalar  = target_vol / realized_vol_20d           (capped at 1.0)
        # beta_scalar = stepped function on portfolio_beta_63d   (only reduces)
        # exposure    = min(vol_scalar, beta_scalar)
        vt_config = self.params.get("volatility_targeting", {})
        beta_aware_exposure = 1.0
        if vt_config.get("enabled", False):
            # ── vol_scalar: target_vol / realized_vol_20d ──────────────
            realized_vol_20d = self._compute_realized_volatility_20d()
            if realized_vol_20d is not None and realized_vol_20d > 0:
                target_vol = vt_config.get("target_volatility", 0.20)
                vol_scalar = target_vol / realized_vol_20d
                # Cap at 1.00 — vol targeting never increases exposure
                vol_scalar = min(1.0, vol_scalar)
                vol_scalar = max(vt_config.get("min_exposure", 0.40), vol_scalar)
                self.log_message(
                    f"Beta-aware vol targeting: realized_vol_20d={realized_vol_20d:.1%}, vol_scalar={vol_scalar:.0%}",
                    color="cyan",
                )
            else:
                vol_scalar = 1.0
                self.log_message(
                    f"Beta-aware vol targeting: insufficient equity history ({len(self._equity_history)} days, need 22) — vol_scalar=100%",
                    color="yellow",
                )

            # ── beta_scalar: stepped function on portfolio_beta_63d ────
            portfolio_beta = risk_metrics.get("beta_63d")
            if portfolio_beta is not None:
                beta_scalar = _compute_beta_scalar(portfolio_beta)
                self.log_message(
                    f"Beta-aware beta scalar: portfolio_beta_63d={portfolio_beta:.2f}, beta_scalar={beta_scalar:.0%}",
                    color="cyan",
                )
            else:
                beta_scalar = 1.0
                self.log_message(
                    "Beta-aware beta scalar: no portfolio beta available — beta_scalar=100%",
                    color="yellow",
                )

            # ── Combine: most conservative wins ────────────────────────
            beta_aware_exposure = min(vol_scalar, beta_scalar)
            if beta_aware_exposure < 1.0:
                self.log_message(
                    f"Beta-aware exposure: {beta_aware_exposure:.0%} (vol_scalar={vol_scalar:.0%}, beta_scalar={beta_scalar:.0%})",
                    color="yellow",
                )

        # Step 5: Combine exposures via min() — most conservative leg wins.
        # Risk overlay (constituent risk) × Beta-aware vol targeting.
        final_exposure = min(risk_exposure, beta_aware_exposure)
        if final_exposure < 1.0:
            self.log_message(
                f"Combined exposure: {final_exposure:.0%} (risk={risk_exposure:.0%}, beta_aware={beta_aware_exposure:.0%})",
                color="yellow",
            )

        # Step 6: Scale target weights by final exposure multiplier
        if target and final_exposure < 1.0:
            for entry in target:
                entry["target_weight"] *= final_exposure
            self.log_message(
                f"Target weights scaled to {final_exposure:.0%} exposure (total weight: {sum(e['target_weight'] for e in target):.1%})",
                color="yellow",
            )

        # Step 7: Store target weights for next week's diagnostics
        self._diagnostics_logger.set_last_rebalance_weights({entry["symbol"]: entry["target_weight"] for entry in target})

        # Step 8: Rebalance
        self.rebalance(target, all_ranks)

    def run_backtesting(self):
        """Run the strategy in backtesting mode using YahooDataBacktesting."""
        log_dir: Path = build_logs(f"agent_{self.name}", TradingMode.BACKTESTING)
        diagnostics_file_path = str(log_dir / "diagnostics.parquet")

        self.log_message(f"Logs will be saved to: {log_dir}", color="yellow")
        self.backtest(
            SafeAlpacaBacktesting,
            name=self.name,
            backtesting_start=self.params["backtesting_start"],
            backtesting_end=self.params["backtesting_end"],
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
