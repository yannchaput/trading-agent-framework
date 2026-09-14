"""
Cross-Sectional Momentum Strategy (V2.4)
----------------------------------
V2.3c + Market Breadth Risk Gate.

Key change from V2.3c:
  - V2.3c combined risk overlay (beta/vol/corr) and fast/slow volatility
    targeting via min(risk_exposure, vol_exposure).
  - V2.4 adds a market breadth regime gate as a third leg in the min() chain:
    min(risk_exposure, vol_exposure, regime_exposure). Breadth is computed
    from the eligible universe (stocks that pass filters): what fraction is
    above its SMA-200 and SMA-50? The regime thresholds are:
      - breadth_200d < 0.35 → bear (50% exposure)
      - breadth_50d < 0.45 → weak (75% exposure)
      - otherwise → normal (100% exposure)

Implements:
  1. Load pre-computed universe (monthly batch)
  2. Daily: compute portfolio risk diagnostics (if enabled)
  3. Daily: record portfolio equity for realized-vol tracking
  4. Weekly (Friday): filter, score, rank, select top N, weight by inverse vol
  5. V2.3a risk overlay (beta/vol/corr) — first leg
  6. V2.3c fast/slow volatility targeting (equity-curve-based) — second leg
  7. V2.4 market breadth regime gate — third leg
  8. Combined via min(risk_exposure, vol_exposure, regime_exposure)
  9. Hysteresis: sell below rank 35, buy top 20
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
    "backtesting_start": datetime(2022, 1, 1),
    "backtesting_end": datetime(2026, 5, 31),
    "benchmark_asset": Asset("SPY", Asset.AssetType.STOCK),
    "budget": 10000,
    "buy_trading_fees": [TradingFee(percent_fee=0.001)],
    "sell_trading_fees": [TradingFee(percent_fee=0.001)],
}

# ── Breadth regime thresholds (V2.4) ─────────────────────────────────────────
BREADTH_REGIME_CONFIG = {
    "bear": {
        "breadth_200d_threshold": 0.35,
        "exposure": 0.50,
    },
    "weak": {
        "breadth_50d_threshold": 0.45,
        "exposure": 0.75,
    },
    "normal": {
        "exposure": 1.00,
    },
}


def compute_volatility_exposure(
    portfolio_volatility: float,
    target_volatility: float = 0.20,
    min_exposure: float = 0.40,
    max_exposure: float = 1.00,
) -> float:
    """Compute exposure multiplier from realized portfolio volatility.

    Formula: exposure = target_vol / realized_vol, clamped to [min, max].

    Args:
        portfolio_volatility: Annualized realized portfolio volatility.
        target_volatility: Desired annualized volatility (default 20%).
        min_exposure: Floor on exposure (default 40%).
        max_exposure: Ceiling on exposure (default 100%).

    Returns:
        Exposure multiplier in [min_exposure, max_exposure].
    """
    if portfolio_volatility <= 0:
        return max_exposure

    exposure = target_volatility / portfolio_volatility
    return max(min_exposure, min(max_exposure, exposure))


def _compute_sma(closes: list[float], window: int) -> float | None:
    """Compute simple moving average from a list of closing prices.

    Args:
        closes: List of closing prices, most recent last.
        window: Lookback window for the SMA.

    Returns:
        SMA value or None if insufficient data.
    """
    if len(closes) < window:
        return None
    return float(np.mean(closes[-window:]))


def compute_breadth_regime(
    breadth_200d: float | None,
    breadth_50d: float | None,
) -> tuple[str, float]:
    """Classify market regime from breadth indicators.

    Decision tree (ordered — first match wins):
      1. breadth_200d < 0.35 → bear (50% exposure)
      2. breadth_50d < 0.45  → weak (75% exposure)
      3. otherwise            → normal (100% exposure)

    Args:
        breadth_200d: Fraction of eligible stocks above their SMA-200, or None.
        breadth_50d: Fraction of eligible stocks above their SMA-50, or None.

    Returns:
        (regime_label, regime_exposure) tuple.
    """
    config = BREADTH_REGIME_CONFIG

    if breadth_200d is not None and breadth_200d < config["bear"]["breadth_200d_threshold"]:
        return "bear", config["bear"]["exposure"]

    if breadth_50d is not None and breadth_50d < config["weak"]["breadth_50d_threshold"]:
        return "weak", config["weak"]["exposure"]

    return "normal", config["normal"]["exposure"]


class CrossMomentumStrategyV24(WrappingStrategy):
    """Cross-sectional momentum strategy V2.4 — market breadth regime gate.

    Adds a market breadth risk gate on top of the V2.3c risk + volatility
    overlay stack. Breadth ratios (stocks above SMA-200 / SMA-50) are
    computed from the eligible universe on each rebalance day and fed into
    a regime classifier. The final exposure is:
        min(risk_exposure, vol_exposure, regime_exposure)
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

        # V2.3c: enable and configure fast/slow volatility targeting
        self.params["volatility_targeting"] = {
            "enabled": True,
            "target_volatility": 0.20,
            "min_exposure": 0.40,
            "max_exposure": 1.00,
        }

        # V2.4: enable market breadth regime gate
        self.params["breadth_regime"] = {
            "enabled": True,
            "bear_threshold_200d": 0.35,
            "bear_exposure": 0.50,
            "weak_threshold_50d": 0.45,
            "weak_exposure": 0.75,
            "normal_exposure": 1.00,
        }

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

        # V2.3c: track actual portfolio equity for realized-vol computation
        self._equity_history: list[float] = load_equity_history()

        # V2.4: store latest breadth ratios (computed on rebalance days)
        self._breadth_200d: float | None = None
        self._breadth_50d: float | None = None

        self.log_message(
            f"CrossMomentumStrategyV24 initialized with {len(self.__universe)} symbols",
            color="green",
        )

    def initialize(self):
        """Set up the strategy before trading starts."""
        self.sleeptime = "1D"
        self.log_message(f"Sleeptime set to {self.sleeptime}", color="yellow")

    # ── Fast/Slow realized volatility from equity curve (V2.3c) ────────────

    def _compute_realized_volatility(self) -> float | None:
        """Compute annualized realized volatility using fast/slow windows.

        V2.3c enhancement: uses max(vol_20d, 0.75 * vol_63d) as the risk
        volatility estimate, rather than just vol_20d.

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

    # ── Market breadth regime (V2.4) ───────────────────────────────────────

    def _compute_breadth_from_scored(self, scored: list[dict]) -> tuple[float | None, float | None]:
        """Compute market breadth ratios from the scored stock list.

        For each stock that has the requisite SMA data, check whether its
        current price is above its SMA-200 and SMA-50. The breadth ratio
        is the fraction of eligible stocks that are above each threshold.

        Args:
            scored: All stocks that passed filters (not just the top N).

        Returns:
            (breadth_200d, breadth_50d) tuple. Each is None if insufficient data.
        """
        above_200 = 0
        above_50 = 0
        eligible_200 = 0
        eligible_50 = 0

        for entry in scored:
            if "above_sma_200" in entry:
                eligible_200 += 1
                if entry["above_sma_200"]:
                    above_200 += 1
            if "above_sma_50" in entry:
                eligible_50 += 1
                if entry["above_sma_50"]:
                    above_50 += 1

        breadth_200 = (above_200 / eligible_200) if eligible_200 >= 20 else None
        breadth_50 = (above_50 / eligible_50) if eligible_50 >= 20 else None

        return breadth_200, breadth_50

    # ── Momentum / portfolio construction (from V2.1) ─────────────────────────

    def is_rebalance_day(self) -> bool:
        """True if today is a Friday (rebalance day)."""
        dt = self.get_datetime()
        return dt.weekday() == self.params["day_of_week"]

    def _compute_indicators_for_ticker(self, ticker: str) -> dict | None:
        """Fetch OHLCV data and compute momentum indicators for a single ticker.

        V2.4: additionally computes SMA-200 and SMA-50 positions for breadth
        calculation. The closes list is used both for momentum scoring and
        for the SMA checks.
        """
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

        # V2.4: compute SMA positions for market breadth
        sma_200 = _compute_sma(closes, 200)
        sma_50 = _compute_sma(closes, 50)

        result = {
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

        # Only include SMA flags when we have enough data
        if sma_200 is not None:
            result["above_sma_200"] = current_price > sma_200
        if sma_50 is not None:
            result["above_sma_50"] = current_price > sma_50

        return result

    def compute_target_portfolio(self) -> tuple[list[dict], dict[str, int]]:
        """Run the full pipeline: indicators → score → filter → rank → select → weight.

        V2.4: after collecting all scored stocks, computes market breadth
        ratios (above SMA-200 and SMA-50) and stores them for the regime gate.

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

        # V2.4: compute market breadth from filtered universe BEFORE sorting
        if self.params.get("breadth_regime", {}).get("enabled", False):
            breadth_200, breadth_50 = self._compute_breadth_from_scored(scored)
            self._breadth_200d = breadth_200
            self._breadth_50d = breadth_50
            above_200_count = sum(1 for s in scored if s.get("above_sma_200"))
            eligible_200_count = sum(1 for s in scored if "above_sma_200" in s)
            above_50_count = sum(1 for s in scored if s.get("above_sma_50"))
            eligible_50_count = sum(1 for s in scored if "above_sma_50" in s)
            breadth_200_str = f"{breadth_200:.1%}" if breadth_200 is not None else "N/A"
            breadth_50_str = f"{breadth_50:.1%}" if breadth_50 is not None else "N/A"
            self.log_message(
                f"Breadth: 200d={breadth_200_str} (above={above_200_count}/{eligible_200_count}), 50d={breadth_50_str} (above={above_50_count}/{eligible_50_count})",
                color="cyan",
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

        # V2.3c: record daily equity for realized-vol tracking
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

        # Step 3: Portfolio Risk Overlay (V2.3a) — compute beta/vol/corr from
        # today's target stocks' historical closes. This is the "hypothetical
        # risk of today's portfolio" leg.
        risk_exposure = 1.0
        risk_metrics = {}
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

        # Step 4: V2.3c Fast/Slow Volatility Targeting — compute realized vol
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
                self.log_message(
                    f"Vol targeting (fast/slow): risk_vol={realized_vol:.1%}, target={vt_config['target_volatility']:.0%}, vol_exposure={vol_exposure:.0%}",
                    color="cyan",
                )
            else:
                self.log_message(
                    f"Vol targeting: insufficient equity history ({len(self._equity_history)} days, need 64) — defaulting to 100% exposure",
                    color="yellow",
                )

        # Step 5: V2.4 Market Breadth Regime — compute regime exposure from
        # breadth ratios computed during compute_target_portfolio().
        breadth_config = self.params.get("breadth_regime", {})
        regime_exposure = 1.0
        regime_label = "normal"
        if breadth_config.get("enabled", False):
            regime_label, regime_exposure = compute_breadth_regime(
                self._breadth_200d,
                self._breadth_50d,
            )
            if regime_label != "normal":
                self.log_message(
                    f"Breadth regime: {regime_label.upper()} (breadth_200d={self._breadth_200d}, breadth_50d={self._breadth_50d}, exposure={regime_exposure:.0%})",
                    color="red" if regime_label == "bear" else "yellow",
                )
            else:
                self.log_message(
                    f"Breadth regime: NORMAL (breadth_200d={self._breadth_200d}, breadth_50d={self._breadth_50d}, exposure={regime_exposure:.0%})",
                    color="cyan",
                )

        # Step 6: Combine exposures via min() — most conservative leg wins.
        # V2.4 adds regime_exposure as the third leg in the min() chain.
        # Using min() rather than multiplication avoids two overlays
        # accidentally creating extremely low exposure.
        final_exposure = min(risk_exposure, vol_exposure, regime_exposure)
        if final_exposure < 1.0:
            self.log_message(
                f"Combined exposure: {final_exposure:.0%} (risk={risk_exposure:.0%}, vol={vol_exposure:.0%}, regime={regime_exposure:.0%})",
                color="yellow",
            )

        # Step 7: Scale target weights by final exposure multiplier
        if target and final_exposure < 1.0:
            for entry in target:
                entry["target_weight"] *= final_exposure
            self.log_message(
                f"Target weights scaled to {final_exposure:.0%} exposure (total weight: {sum(e['target_weight'] for e in target):.1%})",
                color="yellow",
            )

        # Step 8: Store target weights for next week's diagnostics
        self._diagnostics_logger.set_last_rebalance_weights({entry["symbol"]: entry["target_weight"] for entry in target})

        # Step 9: Rebalance
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
