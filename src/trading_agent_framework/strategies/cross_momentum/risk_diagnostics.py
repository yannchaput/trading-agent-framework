"""Portfolio risk diagnostics — pure computation functions.

No Lumibot dependency. All functions take data in (arrays/DataFrames) and
return scalars or None. Designed for daily observation without triggering trades.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskDiagnostics:
    """Daily portfolio risk snapshot — observation only, never triggers trades."""

    date: datetime

    # Realized volatility (annualized)
    portfolio_volatility_20d: float | None = None
    portfolio_volatility_63d: float | None = None

    # Average pairwise correlation
    average_correlation_20d: float | None = None
    average_correlation_63d: float | None = None
    weighted_correlation_63d: float | None = None

    # Portfolio beta to benchmark
    portfolio_beta_63d: float | None = None

    # Sector exposure
    largest_sector: str | None = None
    largest_sector_exposure: float | None = None
    unknown_sector_exposure: float = 0.0

    # Concentration
    herfindahl_index: float | None = None
    effective_number_positions: float | None = None

    # Drawdown
    portfolio_equity: float = 0.0
    rolling_peak: float = 0.0
    drawdown: float = 0.0


# ── Pure computation functions ────────────────────────────────────────────────


def compute_portfolio_volatility(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    lookback: int,
    min_obs: int = 10,
) -> float | None:
    """Annualized portfolio volatility from daily returns and weights.

    Formula: sqrt(w.T @ cov @ w) * sqrt(252)

    Args:
        returns_df: DataFrame of daily close-to-close returns (columns = symbols).
        weights: 1-D array of portfolio weights matching column order.
        lookback: Number of trading days for the volatility estimate.
        min_obs: Minimum aligned observations required (window fallback).

    Returns:
        Annualized volatility (float) or None if data is insufficient.
    """
    n_assets = returns_df.shape[1]
    if n_assets < 1:
        return None

    recent = returns_df.iloc[-lookback:].dropna()
    if len(recent) < min_obs or len(recent) < 2:
        return None

    cols = recent.columns.tolist()
    w = _normalize_weights(weights, cols)
    if w is None:
        return None

    cov = recent.cov().values
    variance = w @ cov @ w

    if variance < 0:
        if variance > -1e-10:
            variance = 0.0
        else:
            logger.warning("Materially negative portfolio variance (%.2e). Returning None.", variance)
            return None

    return float(np.sqrt(variance) * np.sqrt(252))


def compute_average_pairwise_correlation(
    returns_df: pd.DataFrame,
    lookback: int,
    min_obs: int = 10,
) -> float | None:
    """Average pairwise Pearson correlation, excluding the diagonal.

    Formula: mean(corr[i][j]) for all i < j

    Args:
        returns_df: DataFrame of daily returns (columns = symbols).
        lookback: Lookback window in trading days.
        min_obs: Minimum aligned observations required.

    Returns:
        Mean upper-triangle correlation (float) or None if < 2 assets.
    """
    recent = returns_df.iloc[-lookback:].dropna()
    if len(recent) < min_obs:
        return None

    n = recent.shape[1]
    if n < 2:
        return None

    corr = recent.corr().values
    upper = corr[np.triu_indices(n, k=1)]
    return float(np.mean(upper))


def compute_weighted_pairwise_correlation(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    lookback: int,
    min_obs: int = 10,
) -> float | None:
    """Weight-aware average pairwise correlation.

    Formula: sum(w_i * w_j * corr[i][j]) / sum(w_i * w_j)  for i < j.
    Uses normalized absolute long weights.

    Args:
        returns_df: DataFrame of daily returns (columns = symbols).
        weights: Raw weight array matching column order.
        lookback: Lookback window in trading days.
        min_obs: Minimum aligned observations.

    Returns:
        Weighted correlation (float) or None if < 2 assets or insufficient data.
    """
    recent = returns_df.iloc[-lookback:].dropna()
    if len(recent) < min_obs:
        return None

    cols = recent.columns.tolist()
    n = len(cols)
    if n < 2:
        return None

    w = _normalize_weights(weights, cols)
    if w is None:
        return None

    corr = recent.corr().values

    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w_prod = w[i] * w[j]
            numerator += w_prod * corr[i, j]
            denominator += w_prod

    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def compute_portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    min_obs: int = 10,
) -> float | None:
    """Compute portfolio beta relative to a benchmark.

    beta = cov(portfolio, benchmark) / var(benchmark)

    No annualization is applied. Uses sample covariance and variance
    (ddof=1), whose (n-1) denominators cancel in the ratio.

    Args:
        portfolio_returns: Series of daily portfolio returns with date index.
        benchmark_returns: Series of daily benchmark returns with date index.
        min_obs: Minimum aligned trading observations required.

    Returns:
        Beta (float) or None if insufficient data or zero benchmark variance.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < min_obs:
        return None

    pr = aligned.iloc[:, 0]
    br = aligned.iloc[:, 1]

    bench_var = br.var()
    if bench_var == 0.0 or pd.isna(bench_var):
        return None

    cov = pr.cov(br)
    return float(cov / bench_var)


def compute_herfindahl_index(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index: sum(w_i^2).

    Uses normalized weights (sum to 1).
    """
    w = _normalize_weights(weights, None)
    if w is None:
        return 0.0
    return float(np.sum(w**2))


def compute_effective_n(weights: np.ndarray) -> float:
    """Effective number of positions: 1 / sum(w_i^2).

    Uses normalized weights.
    """
    hhi = compute_herfindahl_index(weights)
    if hhi == 0.0:
        return 0.0
    return float(1.0 / hhi)


def compute_drawdown(equity_series: pd.Series) -> tuple[float, float]:
    """Compute drawdown and rolling peak from an equity curve.

    drawdown = equity[-1] / rolling_peak - 1  (negative value when below peak)

    Args:
        equity_series: Series of portfolio equity values ordered chronologically.

    Returns:
        (drawdown, rolling_peak) tuple.
    """
    if equity_series.empty:
        return (0.0, 0.0)

    rolling_peak = equity_series.cummax().iloc[-1]
    current = equity_series.iloc[-1]

    if rolling_peak <= 0:
        return (0.0, float(rolling_peak))

    drawdown = current / rolling_peak - 1.0
    return (float(drawdown), float(rolling_peak))


# ── Drawdown precursor analytics (post-hoc) ────────────────────────────────────


def identify_drawdown_episodes(
    diagnostics_df: pd.DataFrame,
    equity_curve: pd.Series | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Identify the worst N drawdown episodes and extract risk snapshots.

    Groups consecutive drawdown < 0 rows into independent episodes, finds
    the true trough (minimum drawdown) within each episode, and returns
    the deepest N episodes with risk snapshots at T-60, T-40, T-20, T-10,
    T-5, T (trading observations, not calendar days).

    Args:
        diagnostics_df: DataFrame with one row per day (date-sorted).
        equity_curve: Series of portfolio equity (used only if no drawdown
                      column exists — for post-hoc reconstruction).
        top_n: Number of worst episodes to return.

    Returns:
        List of dicts, each containing episode info and diagnostic snapshots.
    """
    df = diagnostics_df.sort_values("date").reset_index(drop=True)

    # Build or use existing drawdown column
    if "drawdown" in df.columns:
        drawdowns = df["drawdown"].values
    elif equity_curve is not None:
        rolling_peak = equity_curve.cummax()
        drawdowns = equity_curve.values / rolling_peak.values - 1.0
    else:
        return []

    n = len(drawdowns)
    if n == 0:
        return []

    # ── Group contiguous drawdown < 0 rows into episodes ─────────────────
    episodes_raw: list[dict] = []
    i = 0
    while i < n:
        if drawdowns[i] >= 0:
            i += 1
            continue

        # Start of an episode — find prior peak (last time dd >= 0)
        peak_idx = i - 1
        while peak_idx >= 0 and drawdowns[peak_idx] < 0:
            peak_idx -= 1
        if peak_idx < 0:
            peak_idx = 0  # never at peak — start of data

        # Scan forward to end of episode (dd >= 0 or end of data)
        j = i
        while j < n and drawdowns[j] < 0:
            j += 1

        # Trough = minimum drawdown within this episode
        episode_slice = drawdowns[i:j]
        trough_offset = int(np.argmin(episode_slice))
        trough_idx = i + trough_offset

        # Recovery (first row after episode where dd >= 0)
        recovery_idx = j if j < n else None

        episodes_raw.append(
            {
                "peak_idx": peak_idx,
                "start_idx": i,
                "trough_idx": trough_idx,
                "end_idx": j - 1,
                "recovery_idx": recovery_idx,
                "drawdown_at_trough": float(drawdowns[trough_idx]),
                "drawdown_duration": j - peak_idx,
                "recovery_duration": (recovery_idx - trough_idx) if recovery_idx is not None else None,
            }
        )

        i = j  # continue scanning after this episode

    if not episodes_raw:
        return []

    # Sort deepest first, take top N
    episodes_raw.sort(key=lambda ep: ep["drawdown_at_trough"])
    top_episodes = episodes_raw[:top_n]

    # ── Build output with diagnostic snapshots ───────────────────────────
    snapshot_offsets = [-60, -40, -20, -10, -5, 0]
    metric_cols = [
        "portfolio_volatility_20d",
        "portfolio_volatility_63d",
        "average_correlation_20d",
        "average_correlation_63d",
        "weighted_correlation_63d",
        "portfolio_beta_63d",
        "largest_sector_exposure",
        "herfindahl_index",
        "effective_number_positions",
    ]

    episodes = []
    for ep in top_episodes:
        ti = ep["trough_idx"]
        result = {
            "peak_date": str(df.loc[ep["peak_idx"], "date"]),
            "peak_index": int(ep["peak_idx"]),
            "start_date": str(df.loc[ep["start_idx"], "date"]),
            "start_index": int(ep["start_idx"]),
            "trough_date": str(df.loc[ti, "date"]),
            "trough_index": int(ti),
            "drawdown_at_trough": ep["drawdown_at_trough"],
            "recovery_date": str(df.loc[ep["recovery_idx"], "date"]) if ep["recovery_idx"] is not None else None,
            "drawdown_duration": ep["drawdown_duration"],
            "recovery_duration": ep["recovery_duration"],
            "diagnostics_snapshots": {},
        }

        # Extract diagnostic snapshots at offsets (trading observations)
        for offset in snapshot_offsets:
            idx = ti + offset
            if 0 <= idx < n:
                snap: dict = {"index": int(idx), "date": str(df.loc[idx, "date"])}
                for col in metric_cols:
                    if col in df.columns:
                        val = df.loc[idx, col]
                        snap[col] = float(val) if not pd.isna(val) else None
                # Also include largest_sector if available
                if "largest_sector" in df.columns:
                    snap["largest_sector"] = str(df.loc[idx, "largest_sector"])
                result["diagnostics_snapshots"][f"T{offset:+d}"] = snap

        episodes.append(result)

    return episodes


def compute_baseline_stats(
    diagnostics_df: pd.DataFrame,
) -> dict:
    """Compute median and 90th percentile for key risk metrics.

    Args:
        diagnostics_df: DataFrame with diagnostic columns.

    Returns:
        Dict with keys like "portfolio_volatility_63d_median",
        "portfolio_volatility_63d_p90", etc.
    """
    metrics = [
        "portfolio_volatility_63d",
        "weighted_correlation_63d",
        "largest_sector_exposure",
        "herfindahl_index",
        "effective_number_positions",
    ]

    stats: dict = {}
    for metric in metrics:
        if metric in diagnostics_df.columns:
            values = diagnostics_df[metric].dropna()
            if len(values) > 0:
                stats[f"{metric}_median"] = float(values.median())
                stats[f"{metric}_p90"] = float(values.quantile(0.90))
            else:
                stats[f"{metric}_median"] = None
                stats[f"{metric}_p90"] = None

    return stats


# ── Internal helpers ──────────────────────────────────────────────────────────


def _normalize_weights(
    weights: np.ndarray,
    column_names: list[str] | None,
) -> np.ndarray | None:
    """Normalize weights to sum to 1 (absolute, long-only).

    If column_names is provided, ensure weights match the column count.
    """
    w = np.asarray(weights, dtype=float)
    if column_names is not None and len(w) != len(column_names):
        if len(w) > len(column_names):
            w = w[: len(column_names)]
        else:
            return None

    w = np.abs(w)
    total = np.sum(w)
    if total == 0.0:
        return None
    return w / total
