"""Diagnostic logging for cross-sectional momentum strategies.

Encapsulates the portfolio-risk-diagnostics pipeline that was previously
duplicated as a set of ``_compute_and_persist_diagnostics`` / ``_fetch_aligned_returns``
/ ``_build_weight_array`` / ``_compute_portfolio_daily_returns`` /
``_compute_sector_exposure`` / ``_flush_diagnostics`` methods on every strategy
variant. Strategies hold a single :class:`DiagnosticLogger` reference instead.

The logger duck-types the strategy — it only reads the public surface
(``get_historical_prices``, ``get_datetime``, ``portfolio_value``, ``params``,
``log_warning``) rather than importing the strategy classes, which keeps this
module free of circular imports.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from trading_agent_framework.brokers.alpaca.alpaca_support import AlpacaApiRateLimiter
from trading_agent_framework.config import TradingMode

from .risk_diagnostics import (
    PortfolioRiskDiagnostics,
    compute_average_pairwise_correlation,
    compute_effective_n,
    compute_herfindahl_index,
    compute_portfolio_beta,
    compute_portfolio_volatility,
    compute_weighted_pairwise_correlation,
)
from .sector_provider import SectorProvider
from .utils import diagnostics_to_dict

logger = logging.getLogger(__name__)


class DiagnosticLogger:
    """Compute and persist daily portfolio risk diagnostics.

    Owns all diagnostic state (last rebalance weights, rolling peak, sector
    provider, output path, header flag) that was previously spread across the
    strategy instances. ``SectorProvider`` is encapsulated here and created on
    construction unless an instance is injected (tests pass a fake to avoid
    network access).
    """

    def __init__(
        self,
        strategy: Any,
        trading_mode: TradingMode,
        diagnostics_file_path: str | None = None,
        sector_provider: SectorProvider | None = None,
    ) -> None:
        self._strategy = strategy

        # Rate limiter for Alpaca market-data calls (universe filtering bursts
        # past Alpaca's ~200 req/min limit; see support/alpaca_support.py).
        self._alpaca_rate_limiter = AlpacaApiRateLimiter(trading_mode=trading_mode)

        # Diagnostic state
        self._last_rebalance_weights: dict[str, float] = {}
        self._rolling_peak: float = 0.0
        self._sector_provider: SectorProvider | None = sector_provider or SectorProvider()
        self._diagnostics_file_path: str | None = diagnostics_file_path
        self._diagnostics_header_written: bool = False

    def set_last_rebalance_weights(self, weights: dict[str, float]) -> None:
        """Store the latest rebalance target weights for next-day diagnostics."""
        self._last_rebalance_weights = weights

    def compute_and_persist(self) -> None:
        """Compute portfolio risk diagnostics for today and persist to parquet."""
        diag_config = self._strategy.parameters.get("risk_diagnostics", {})
        if not diag_config.get("enabled", False):
            return

        # No previous rebalance yet → skip
        if not self._last_rebalance_weights:
            return

        holdings = list(self._last_rebalance_weights.keys())
        if not holdings:
            return

        benchmark = diag_config.get("benchmark", "SPY")
        beta_lookback = diag_config.get("beta_lookback_days", 63)
        volatility_lookbacks = diag_config.get("volatility_lookbacks", [20, 63])
        correlation_lookbacks = diag_config.get("correlation_lookbacks", [20, 63])
        min_obs = diag_config.get("min_overlap_observations", 10)
        max_lookback = max(volatility_lookbacks + correlation_lookbacks + [beta_lookback])

        # Fetch aligned returns for holdings + benchmark
        returns_df, benchmark_returns_series = self._fetch_aligned_returns(holdings, benchmark, max_lookback)

        if returns_df is None or returns_df.empty:
            return

        # Guard: if we lost stocks during fetch (silent get_historical_prices
        # failures), the renormalized weights would concentrate exposure and
        # produce wildly inflated vol / beta. Skip diagnostics for today.
        n_held = len(holdings)
        n_fetched = returns_df.shape[1]
        if n_fetched < n_held * 0.75:
            self._strategy.log_warning(
                f"Diagnostics: only {n_fetched}/{n_held} holdings have price data — skipping today to avoid concentration bias.",
            )
            return

        weights = self._build_weight_array(returns_df.columns.tolist())
        if weights is None:
            return

        # Volatility
        vol_20 = compute_portfolio_volatility(returns_df, weights, lookback=20, min_obs=min_obs) if 20 in volatility_lookbacks else None
        vol_63 = compute_portfolio_volatility(returns_df, weights, lookback=63, min_obs=min_obs) if 63 in volatility_lookbacks else None

        # Average pairwise correlation
        avg_corr_20 = compute_average_pairwise_correlation(returns_df, lookback=20, min_obs=min_obs) if 20 in correlation_lookbacks else None
        avg_corr_63 = compute_average_pairwise_correlation(returns_df, lookback=63, min_obs=min_obs) if 63 in correlation_lookbacks else None

        # Weighted correlation (63d only per spec)
        w_corr_63 = compute_weighted_pairwise_correlation(returns_df, weights, lookback=63, min_obs=min_obs) if 63 in correlation_lookbacks else None

        # Beta
        beta = None
        portfolio_daily_returns = self._compute_portfolio_daily_returns(returns_df, weights)
        if portfolio_daily_returns is not None and benchmark_returns_series is not None:
            beta = compute_portfolio_beta(portfolio_daily_returns, benchmark_returns_series, min_obs=min_obs)

        # Sector exposure
        largest_sector, largest_sector_exp, unknown_exp = self._compute_sector_exposure(holdings, weights)

        # Concentration
        hhi = compute_herfindahl_index(weights)
        eff_n = compute_effective_n(weights)

        # Drawdown — track rolling peak across days (not single-point Series)
        equity = float(self._strategy.portfolio_value or 0.0)
        if equity > self._rolling_peak:
            self._rolling_peak = equity
        dd = equity / self._rolling_peak - 1.0 if self._rolling_peak > 0 else 0.0
        rp = self._rolling_peak

        today = self._strategy.get_datetime()
        diag = PortfolioRiskDiagnostics(
            date=today,
            portfolio_volatility_20d=vol_20,
            portfolio_volatility_63d=vol_63,
            average_correlation_20d=avg_corr_20,
            average_correlation_63d=avg_corr_63,
            weighted_correlation_63d=w_corr_63,
            portfolio_beta_63d=beta,
            largest_sector=largest_sector,
            largest_sector_exposure=largest_sector_exp,
            unknown_sector_exposure=unknown_exp,
            herfindahl_index=hhi,
            effective_number_positions=eff_n,
            portfolio_equity=equity,
            rolling_peak=rp,
            drawdown=dd,
        )

        self._flush_diagnostics(diag)

    def _fetch_aligned_returns(
        self,
        holdings: list[str],
        benchmark: str,
        length: int,
    ) -> tuple[pd.DataFrame | None, pd.Series | None]:
        """Fetch aligned daily returns for holdings and benchmark.

        Uses get_historical_prices which returns data only through the current
        simulation date in backtesting — preventing look-ahead bias.

        Returns:
            (returns_df, benchmark_returns) where returns_df has symbols as columns
            and dates as index. Both are None if data is unavailable.
        """
        # Fetch benchmark data
        self._alpaca_rate_limiter.wait()
        bench_bars = self._strategy.get_historical_prices(benchmark, length=length, timestep="day")
        benchmark_returns = None
        if bench_bars is not None and not bench_bars.empty:
            bench_df = bench_bars.pandas_df if hasattr(bench_bars, "pandas_df") else bench_bars
            if bench_df is not None and not bench_df.empty:
                closes = bench_df["close"]
                benchmark_returns = closes.pct_change().dropna()
                benchmark_returns.name = benchmark

        # Fetch each holding's daily closes
        closes_map: dict[str, pd.Series] = {}
        for symbol in holdings:
            self._alpaca_rate_limiter.wait()
            bars = self._strategy.get_historical_prices(symbol, length=length, timestep="day")
            if bars is None or bars.empty:
                continue
            df = bars.pandas_df if hasattr(bars, "pandas_df") else bars
            if df is None or df.empty:
                continue
            closes_map[symbol] = df["close"]

        if len(closes_map) < 2:
            self._strategy.log_warning(
                f"Diagnostics: only {len(closes_map)}/{len(holdings)} holdings fetched price data",
            )
            return None, benchmark_returns

        # Convert to daily returns
        returns_map = {}
        for sym, closes in closes_map.items():
            r = closes.pct_change().dropna()
            if len(r) > 0:
                returns_map[sym] = r

        if len(returns_map) < 2:
            return None, benchmark_returns

        # Align all return series on date index (inner join)
        returns_df = pd.DataFrame(returns_map).dropna()
        if returns_df.empty or returns_df.shape[1] < 2:
            return None, benchmark_returns

        return returns_df, benchmark_returns

    def _build_weight_array(self, columns: list[str]) -> np.ndarray | None:
        """Build a weight array aligned with returns_df columns.

        Uses last-Friday rebalance target weights, normalized to sum to 1.
        """
        raw = []
        for sym in columns:
            raw.append(self._last_rebalance_weights.get(sym, 0.0))
        w = np.array(raw, dtype=float)
        total = np.sum(w)
        if total <= 0:
            return None
        return w / total

    def _compute_portfolio_daily_returns(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
    ) -> pd.Series | None:
        """Compute daily portfolio returns from constituent returns and weights.

        Uses the caller-provided weights to ensure consistency with the
        volatility and correlation diagnostics. Weights must be normalized
        and aligned with returns_df.columns order.
        """
        if returns_df.empty or weights is None or len(weights) == 0:
            return None
        if len(weights) != returns_df.shape[1]:
            return None
        w = np.asarray(weights, dtype=float)
        total = np.sum(w)
        if total <= 0:
            return None
        w = w / total
        portfolio_returns = returns_df.values @ w
        return pd.Series(portfolio_returns, index=returns_df.index)

    def _compute_sector_exposure(
        self,
        holdings: list[str],
        weights: np.ndarray,
    ) -> tuple[str | None, float | None, float]:
        """Compute sector exposure breakdown.

        Returns:
            (largest_sector_name, largest_sector_exposure, unknown_sector_exposure)
        """
        if self._sector_provider is None or len(holdings) == 0:
            return None, None, 0.0

        sector_map = self._sector_provider.get_sector_map(holdings)

        sector_weights: dict[str, float] = {}
        for i, sym in enumerate(holdings):
            sector = sector_map.get(sym, "UNKNOWN")
            w = float(weights[i]) if i < len(weights) else 0.0
            sector_weights[sector] = sector_weights.get(sector, 0.0) + w

        unknown_exp = sector_weights.pop("UNKNOWN", 0.0)

        if not sector_weights:
            return None, None, unknown_exp

        largest_sector = max(sector_weights, key=lambda k: sector_weights[k])
        largest_exp = sector_weights[largest_sector]

        return largest_sector, largest_exp, unknown_exp

    def _flush_diagnostics(self, diag: PortfolioRiskDiagnostics) -> None:
        """Append one diagnostic row to diagnostics.parquet."""
        if not self._diagnostics_file_path:
            return

        row = diagnostics_to_dict(diag)
        df = pd.DataFrame([row])

        try:
            if not self._diagnostics_header_written:
                df.to_parquet(self._diagnostics_file_path, index=False)
                self._diagnostics_header_written = True
            else:
                existing = pd.read_parquet(self._diagnostics_file_path)
                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_parquet(self._diagnostics_file_path, index=False)
        except Exception:
            logger.warning("Failed to persist diagnostics to parquet", exc_info=True)
