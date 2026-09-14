"""
Cross-Sectional Momentum Strategy (V2.2-C)
----------------------------------
V2.1 base (weekly momentum + SPY 200-day SMA regime overlay) augmented with a
daily short-term relative strength break deterioration layer (Signal C only).

Every trading day, current holdings are checked for stock-specific collapse
relative to the broad market: if a position's 5-day return underperforms SPY
by ≥8 percentage points, the full position is exited. Freed cash stays
uninvested until the next Friday rebalance.

V2.1 features kept:
  - Weekly Friday momentum ranking (12-1m, 6-1m, 3m scores)
  - Hysteresis: buy top 20, sell below rank 35
  - Inverse-volatility position sizing
  - SPY 200-day SMA market regime exposure overlay
  - ThreadPoolExecutor parallel indicator computation
  - Daily: Detect deterioration in the portfolio stocks by computing five-trading-day total return of the position vs SPY.
  If performance of the position is 8% under SPY, drop the position.

No LLM, no AI agents — pure rules-based.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from lumibot.entities import Asset, TradingFee

from lumibot_trading_agent.support.alpaca_support import SafeAlpacaBacktesting
from lumibot_trading_agent.support.broker_factory import BrokerFactory
from lumibot_trading_agent.support.helpers import (
    TradingMode,
    build_logs,
    get_thread_capacity,
)
from lumibot_trading_agent.support.wrapping_strategy import WrappingStrategy

from ..deterioration_state import DeteriorationStateManager
from ..parameters import CONFIG
from ..utils import (
    annualized_volatility,
    apply_filters,
    compute_atr_from_df,
    compute_return_from_prices,
    get_cross_momentum_universe_last_date,
    inverse_volatility_weights,
    momentum_score,
)

# ── Backtesting params ───────────────────────────────────────────────────────
BACKTESTING_PARAMS = {
    "backtesting_start": datetime(2022, 1, 1),
    "backtesting_end": datetime(2026, 5, 31),
    "benchmark_asset": Asset("SPY", Asset.AssetType.STOCK),
    "budget": 10000,
    "buy_trading_fees": [TradingFee(percent_fee=0.001)],
    "sell_trading_fees": [TradingFee(percent_fee=0.001)],
}

# ── Strategy class ───────────────────────────────────────────────────────────


class CrossMomentumStrategyV22c(WrappingStrategy):
    """Cross-sectional momentum strategy — weekly rebalance with daily relative strength deterioration.

    The deterioration layer (Signal C) exits positions whose 5-day return
    underperforms SPY by ≥8 percentage points. Friday rebalance is authoritative
    and can re-enter a previously exited position.

    Universe is loaded from a pre-computed JSON file (built monthly by
    batch_stock_universe.py), or passed directly as a constructor parameter.
    """

    def __init__(
        self,
        *args,
        mode: TradingMode = TradingMode.BACKTESTING,
        universe: list[str] | None = None,
        bk_log_dir: Path | None = None,
        **kwargs,
    ):
        """Initialize the strategy.

        Args:
            mode: Trading mode (backtesting, paper, live).
            universe: List of ticker symbols (required — must be non-empty).
            bk_log_dir: Path to backtesting log directory (used for deterioration event logging).
        """
        super().__init__(*args, mode=mode, **kwargs)
        self.params = {**CONFIG, **BACKTESTING_PARAMS}

        universe = universe or []
        if not len(universe):
            self.log_message("No universe provided to the strategy, this is a mandatory parameter", color="red")
            sys.exit(1)
        self.__universe = universe
        self._deterioration_state: DeteriorationStateManager | None = None
        self._backtest_log_dir: Path | None = bk_log_dir

        self.log_message(
            f"CrossMomentumStrategyV22c initialized with {len(self.__universe)} symbols",
            color="green",
        )

    def initialize(self):
        """Set up the strategy before trading starts."""
        self.sleeptime = "1D"
        # memory_dir = Path(".lumibot") / f"memory_{self._trading_mode.value}" / self.name
        memory_dir = self.memory.strategy_dir
        self._deterioration_state = DeteriorationStateManager(memory_dir)
        self.log_message(f"Sleeptime set to {self.sleeptime}", color="yellow")

    def is_rebalance_day(self) -> bool:
        """True if today is a Friday (rebalance day)."""
        dt = self.get_datetime()
        return dt.weekday() == self.params["day_of_week"]

    def compute_market_regime(self) -> dict:
        """Fetch SPY prices, compute 200-day SMA, return regime + exposure_multiplier.

        Returns:
            dict with keys: regime (str), exposure_multiplier (float),
                            spy_close (float|None), sma200 (float|None),
                            ratio (float|None)
        """
        sma_window = self.params["regime_sma_window"]
        buffer_pct = self.params["regime_buffer_pct"]
        bull_mult = self.params["regime_bull_multiplier"]
        neutral_mult = self.params["regime_neutral_multiplier"]
        bear_mult = self.params["regime_bear_multiplier"]
        data_days = self.params["regime_data_days"]

        bars = self.get_historical_prices("SPY", length=data_days, timestep="day")
        if bars is None or bars.empty:
            self.log_message("SPY data unavailable — defaulting to Neutral regime", color="yellow")
            return {
                "regime": "neutral",
                "exposure_multiplier": neutral_mult,
                "spy_close": None,
                "sma200": None,
                "ratio": None,
            }

        df = bars.pandas_df if hasattr(bars, "pandas_df") else bars
        if df is None or df.empty:
            self.log_message("SPY DataFrame empty — defaulting to Neutral regime", color="yellow")
            return {
                "regime": "neutral",
                "exposure_multiplier": neutral_mult,
                "spy_close": None,
                "sma200": None,
                "ratio": None,
            }

        closes = df["close"].tolist()

        if len(closes) < sma_window:
            self.log_message(
                f"SPY data insufficient ({len(closes)} days < {sma_window}) — defaulting to Neutral",
                color="yellow",
            )
            return {
                "regime": "neutral",
                "exposure_multiplier": neutral_mult,
                "spy_close": closes[-1] if closes else None,
                "sma200": None,
                "ratio": None,
            }

        recent_closes = closes[-sma_window:]
        sma200 = sum(recent_closes) / len(recent_closes)
        spy_close = closes[-1]

        if sma200 <= 0:
            self.log_message("SPY SMA200 is zero — defaulting to Neutral regime", color="yellow")
            return {
                "regime": "neutral",
                "exposure_multiplier": neutral_mult,
                "spy_close": spy_close,
                "sma200": sma200,
                "ratio": None,
            }

        ratio = spy_close / sma200

        if ratio > 1.0 + buffer_pct:
            regime = "bull"
            multiplier = bull_mult
        elif ratio < 1.0 - buffer_pct:
            regime = "bear"
            multiplier = bear_mult
        else:
            regime = "neutral"
            multiplier = neutral_mult

        self.log_message(
            f"Market regime: {regime} (SPY={spy_close:.2f}, SMA200={sma200:.2f}, ratio={ratio:.3f}, exposure={multiplier:.0%})",
            color="cyan",
        )

        return {
            "regime": regime,
            "exposure_multiplier": multiplier,
            "spy_close": spy_close,
            "sma200": sma200,
            "ratio": ratio,
        }

    # ── Daily relative strength check (Signal C) ──────────────────────────

    def _run_relative_strength_check(self) -> None:
        """Check all current holdings for short-term relative strength break.

        For each holding, compute the 5-trading-day return vs SPY using
        **yesterday's close** as the reference point.  The sell order fills
        at today's open, so the timeline is chronologically valid:

          See close[N-1] (yesterday's confirmed close)
          → signal triggers
          → sell at open[N] (today's open)  ✓

        If the stock underperformed SPY by ≥8 percentage points, exit the
        full position. Already-exited positions (from a prior day this week)
        are skipped.

        Relative strength calculation:
          stock_return_5d = close[-2] / close[-lookback - 2] - 1
          benchmark_return_5d = spy_close[-2] / spy_close[-lookback - 2] - 1
          relative_return_5d = stock_return_5d - benchmark_return_5d
          EXIT if relative_return_5d <= -0.08
        """
        positions = self.get_positions()
        if not positions:
            return

        lookback: int = self.params["deterioration_relative_strength_lookback_days"]
        threshold: float = self.params["deterioration_relative_return_threshold"]
        benchmark_symbol: str = self.params["deterioration_benchmark"]
        today_str = str(self.get_datetime().date())

        # Fetch benchmark (SPY) data once for all holdings.
        # +11 bars: +1 for the [-2] lag shift, +10 for safety margin.
        spy_bars = self.get_historical_prices(benchmark_symbol, length=lookback + 11, timestep="day")
        if spy_bars is None or spy_bars.empty:
            self.log_message(
                f"{benchmark_symbol} data unavailable — skipping relative strength check",
                color="yellow",
            )
            return

        spy_df = spy_bars.pandas_df if hasattr(spy_bars, "pandas_df") else spy_bars
        if spy_df is None or spy_df.empty:
            self.log_message(
                f"{benchmark_symbol} DataFrame empty — skipping relative strength check",
                color="yellow",
            )
            return

        spy_closes = spy_df["close"].tolist()
        # Need lookback+2 bars: close[-2] … close[-lookback-2]
        if len(spy_closes) < lookback + 2:
            self.log_message(
                f"{benchmark_symbol} data insufficient ({len(spy_closes)} days < {lookback + 2}) — skipping relative strength check",
                color="yellow",
            )
            return

        # close[-2] = yesterday's confirmed close (chronologically before today's open).
        spy_current = spy_closes[-2]
        spy_previous = spy_closes[-lookback - 2]
        if spy_previous is None or spy_previous <= 0:
            self.log_message(
                f"{benchmark_symbol} close {lookback + 1}d ago is invalid — skipping relative strength check",
                color="yellow",
            )
            return

        benchmark_return_5d = (spy_current / spy_previous) - 1.0

        self.log_message(
            f"Relative strength check: {benchmark_symbol} 5d return={benchmark_return_5d:.4f} (yesterday close ${spy_current:.2f} vs {lookback + 1}d ago close ${spy_previous:.2f})",
            color="yellow",
        )

        for pos in positions:
            symbol = pos.asset.symbol

            # Skip already-exited positions
            state = self._deterioration_state.get_position_state(symbol)
            if state == "exited":
                continue

            # Fetch stock OHLCV data (same +1 buffer for the lag shift)
            stock_bars = self.get_historical_prices(symbol, length=lookback + 11, timestep="day")
            if stock_bars is None or stock_bars.empty:
                continue

            stock_df = stock_bars.pandas_df if hasattr(stock_bars, "pandas_df") else stock_bars
            if stock_df is None or stock_df.empty or len(stock_df) < lookback + 2:
                continue

            stock_closes = stock_df["close"].tolist()
            stock_current = stock_closes[-2]
            stock_previous = stock_closes[-lookback - 2]

            if stock_previous is None or stock_previous <= 0:
                continue

            stock_return_5d = (stock_current / stock_previous) - 1.0
            relative_return_5d = stock_return_5d - benchmark_return_5d

            if relative_return_5d <= threshold:
                self.log_message(
                    f"RELATIVE STRENGTH EXIT: {symbol} | "
                    f"stock 5d={stock_return_5d:.4f} | {benchmark_symbol} 5d={benchmark_return_5d:.4f} | "
                    f"relative={relative_return_5d:.4f} (threshold={threshold}) | "
                    f"${stock_previous:.2f} → ${stock_current:.2f}",
                    color="red",
                )

                # Submit full exit (fills at today's open).
                try:
                    sell_order = self.create_order(symbol, pos.quantity, "sell")
                    self.submit_order(sell_order)
                    # Approximate cash released using yesterday's close.
                    cash_released = pos.quantity * stock_current
                except Exception as e:
                    self.log_message(
                        f"Failed to submit relative strength exit for {symbol}: {e}",
                        color="red",
                    )
                    continue

                # Record event and persist state
                event = {
                    "date": today_str,
                    "symbol": symbol,
                    "action": "exit",
                    "signal": "relative_strength",
                    "signal_count": 1,
                    "relative_strength_triggered": True,
                    "stock_return_5d": round(stock_return_5d, 6),
                    "benchmark_return_5d": round(benchmark_return_5d, 6),
                    "relative_return_5d": round(relative_return_5d, 6),
                    "relative_return_threshold": threshold,
                    "benchmark_symbol": benchmark_symbol,
                    "close": stock_current,
                    "close_5d_ago": stock_previous,
                    "position_qty_before": pos.quantity,
                    "position_qty_after": 0,
                    "cash_released": round(cash_released, 2),
                }
                self._deterioration_state.set_exited(symbol, event)
                self._append_deterioration_event(event)

    def _append_deterioration_event(self, event: dict) -> None:
        """Append a deterioration event to the JSONL log file.

        Only active during backtesting — in paper/live modes the deterioration
        state is persisted via DeteriorationStateManager in .lumibot/ instead.
        """
        if self._backtest_log_dir is None:
            return
        events_file = self._backtest_log_dir / "deterioration_events.jsonl"
        with open(events_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    # ── Weekly momentum pipeline (V2.1 — unchanged) ───────────────────────

    def _compute_indicators_for_ticker(self, ticker: str) -> dict | None:
        """Fetch OHLCV data and compute momentum indicators for a single ticker.

        Returns a dict with symbol, price, returns, volatility, dollar_volume,
        trading_days, atr, or None if data is insufficient.
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

        # Momentum returns (12-1m, 6-1m, 3m)
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
        }

    def compute_target_portfolio(self) -> tuple[list[dict], dict[str, int]]:
        """Run the full pipeline: indicators → score → filter → rank → select → weight.

        Uses ThreadPoolExecutor for parallel processing of the universe.

        Returns:
            (target, all_ranks) where target is a list of selected+weighted dicts
            and all_ranks is a mapping from symbol to rank for ALL scored stocks
            (needed for hysteresis in rebalance).
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
            f"Indicators: {len(scored)} passed filters, {skip_count} skipped (insufficient data or filters)",
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
            all_ranks: Symbol→rank mapping for ALL scored stocks (needed for hysteresis).
        """
        if not target:
            return

        target_symbols = {entry["symbol"] for entry in target}

        sell_threshold = self.params["sell_rank_threshold"]
        portfolio_value = self.portfolio_value or 1.0

        current_positions = self.get_positions()

        # Phase 1: Sell positions that should be removed (hysteresis)
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

        # Phase 2: Buy new positions / adjust existing to target weight
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

    # ── Main iteration ──────────────────────────────────────────────────────

    def on_trading_iteration(self):
        super().on_trading_iteration()

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

        # ── DAILY: Relative strength break check on current holdings ──
        self._run_relative_strength_check()

        # ── FRIDAY ONLY: Full momentum pipeline ──
        if not self.is_rebalance_day():
            self.log_message("Not a rebalance day — skipping momentum pipeline.", color="yellow")
            return

        regime_info = self.compute_market_regime()
        exposure_multiplier = regime_info["exposure_multiplier"]

        self.log_message(
            f"Friday rebalance — regime={regime_info['regime']}, exposure={exposure_multiplier:.0%} — computing target portfolio...",
            color="cyan",
        )

        target, all_ranks = self.compute_target_portfolio()

        if target and exposure_multiplier < 1.0:
            for entry in target:
                entry["target_weight"] *= exposure_multiplier
            self.log_message(
                f"Target weights scaled to {exposure_multiplier:.0%} exposure (total weight: {sum(e['target_weight'] for e in target):.1%})",
                color="yellow",
            )

        self.rebalance(target, all_ranks)

        # Reset deterioration state after Friday rebalance
        held_symbols = {p.asset.symbol for p in self.get_positions()}
        self._deterioration_state.reset_for_friday(held_symbols)

    def run_backtesting(self):
        """Run the strategy in backtesting mode using YahooDataBacktesting."""
        log_dir = build_logs(f"agent_{self.name}", TradingMode.BACKTESTING)
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
            universe=self.__universe,
            bk_log_dir=log_dir,
        )
