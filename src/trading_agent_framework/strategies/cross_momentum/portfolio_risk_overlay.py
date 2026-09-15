"""Portfolio Risk Overlay — pure computation functions.

No Lumibot dependency. Computes portfolio-level beta (63d), realized
volatility (20d), and average pairwise correlation (20d) from raw close
prices, then classifies the portfolio risk state as NORMAL, ELEVATED, or
CRITICAL to determine an exposure multiplier.

Thresholds are intentionally coarse (round numbers). Do not optimize them
without collecting data from more than the 5 known drawdown episodes.
"""

from enum import Enum

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

_MIN_OBS = 40  # minimum overlapping trading days for any metric

# Coarse thresholds — deliberately not optimised from the 5 known episodes
_BETA_ELEVATED = 1.7
_BETA_CRITICAL = 2.0
_VOL_ELEVATED = 0.30
_VOL_CRITICAL = 0.35
_CORR_ELEVATED = 0.30
_CORR_CRITICAL = 0.35


# ── Public types ───────────────────────────────────────────────────────────────


class RiskState(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


EXPOSURE_MAP: dict[RiskState, float] = {
    RiskState.NORMAL: 1.00,
    RiskState.ELEVATED: 0.70,
    RiskState.CRITICAL: 0.40,
}


# ── Classification ─────────────────────────────────────────────────────────────


def classify_risk_state(
    beta: float | None,
    vol: float | None,
    corr: float | None,
) -> RiskState:
    """Classify portfolio risk state from beta, vol, and correlation.

    Evaluation order: CRITICAL conditions first, then ELEVATED, then NORMAL.
    None metrics skip their condition — the evaluation is per-condition,
    not all-or-nothing.
    """
    # CRITICAL
    if beta is not None and beta >= _BETA_CRITICAL:
        return RiskState.CRITICAL
    if vol is not None and corr is not None and vol >= _VOL_CRITICAL and corr >= _CORR_CRITICAL:
        return RiskState.CRITICAL

    # ELEVATED
    if beta is not None and beta >= _BETA_ELEVATED:
        return RiskState.ELEVATED
    if vol is not None and corr is not None and vol >= _VOL_ELEVATED and corr >= _CORR_ELEVATED:
        return RiskState.ELEVATED

    return RiskState.NORMAL


# ── Returns construction ───────────────────────────────────────────────────────


def _build_aligned_returns(
    closes_map: dict[str, list[float]],
    weights_dict: dict[str, float],
    benchmark_closes: list[float] | None,
) -> tuple[pd.DataFrame | None, pd.Series | None, np.ndarray | None]:
    """Build aligned daily returns from raw close-price series.

    All series are truncated to the shortest common length, then converted
    to simple daily percentage returns (not log returns, for consistency
    with existing diagnostics). Weights are renormalized to include only
    symbols that made it into the returns DataFrame.

    Args:
        closes_map: {symbol: [closes]} for each target stock.
        weights_dict: {symbol: target_weight} — raw weights (may not sum to 1).
        benchmark_closes: List of benchmark (SPY) closes, or None.

    Returns:
        (returns_df, benchmark_returns, aligned_weights) tuple where:
        - returns_df: DataFrame (days × symbols) of daily returns
        - benchmark_returns: Series of daily benchmark returns (or None)
        - aligned_weights: np.ndarray normalized to sum=1, matching column order
        All three are None if closes_map is empty.
    """
    if not closes_map:
        return None, None, None

    # Truncate to shortest common length
    lengths = [len(closes) for closes in closes_map.values()]
    if benchmark_closes is not None:
        lengths.append(len(benchmark_closes))
    min_len = min(lengths)

    if min_len < 2:
        return None, None, None

    # Compute daily returns for each stock (simple pct change)
    returns_map: dict[str, pd.Series] = {}
    for sym, closes in closes_map.items():
        truncated = closes[-min_len:]
        r = pd.Series(truncated).pct_change().dropna().values
        if len(r) > 0:
            returns_map[sym] = r  # pyright: ignore[reportArgumentType]

    if not returns_map:
        return None, None, None

    # Build DataFrame aligned on observation index
    returns_df = pd.DataFrame(returns_map).dropna()
    if returns_df.empty or returns_df.shape[1] < 1:
        return None, None, None

    # Benchmark returns
    benchmark_returns = None
    if benchmark_closes is not None:
        truncated_bench = benchmark_closes[-min_len:]
        br = pd.Series(truncated_bench).pct_change().dropna().reset_index(drop=True)
        if len(br) > 0:
            benchmark_returns = br

    # Aligned weights
    cols = returns_df.columns.tolist()
    raw = np.array([weights_dict.get(sym, 0.0) for sym in cols], dtype=float)
    total = np.sum(raw)
    if total <= 0:
        return None, None, None
    aligned_weights = raw / total

    return returns_df, benchmark_returns, aligned_weights


# ── Metric computations ────────────────────────────────────────────────────────


def _compute_portfolio_vol(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    lookback: int,
) -> float | None:
    """Annualized portfolio volatility from daily returns and weights.

    Formula: sqrt(w.T @ cov @ w) * sqrt(252)
    """
    n_assets = returns_df.shape[1]
    if n_assets < 1:
        return None

    recent = returns_df.iloc[-lookback:].dropna()
    if len(recent) < 2:
        return None

    cols = recent.columns.tolist()
    if len(cols) != len(weights):
        return None

    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w) if np.sum(w) > 0 else w

    cov = recent.cov().values
    variance = w @ cov @ w

    if variance < 0:
        if variance > -1e-10:
            variance = 0.0
        else:
            return None

    return float(np.sqrt(variance) * np.sqrt(252))


def _compute_avg_corr(
    returns_df: pd.DataFrame,
    lookback: int,
) -> float | None:
    """Average pairwise Pearson correlation, excluding the diagonal.

    Formula: mean(corr[i][j]) for all i < j
    """
    recent = returns_df.iloc[-lookback:].dropna()
    n = recent.shape[1]
    if len(recent) < 2 or n < 2:
        return None

    corr = recent.corr().values
    upper = corr[np.triu_indices(n, k=1)]
    return float(np.mean(upper))


def _compute_portfolio_beta(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    benchmark_returns: pd.Series,
    min_obs: int = _MIN_OBS,
) -> float | None:
    """Portfolio beta relative to benchmark over full available history.

    beta = cov(portfolio, benchmark) / var(benchmark)
    """
    w = np.asarray(weights, dtype=float)
    total = np.sum(w)
    if total <= 0:
        return None
    w = w / total

    portfolio_returns = returns_df.values @ w
    pr = pd.Series(portfolio_returns, index=returns_df.index)

    aligned = pd.concat([pr, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < min_obs:
        return None

    pr_aligned = aligned.iloc[:, 0]
    br_aligned = aligned.iloc[:, 1]

    bench_var = br_aligned.var()
    if bench_var == 0.0 or pd.isna(bench_var):
        return None

    cov = pr_aligned.cov(br_aligned)
    return float(cov / bench_var)


# ── Orchestrator ───────────────────────────────────────────────────────────────


def compute_risk_overlay(
    closes_map: dict[str, list[float]],
    target_weights: dict[str, float],
    benchmark_closes: list[float] | None,
    min_obs: int = _MIN_OBS,
) -> tuple[str, float, dict]:
    """Compute portfolio risk metrics and determine exposure multiplier.

    This is the single entry point called by the strategy on rebalance day.
    It builds aligned returns from already-fetched closes, computes the three
    risk metrics (beta, vol, corr), classifies the risk state, and returns
    the state name, exposure multiplier, and a metrics dict for logging.

    Args:
        closes_map: {symbol: [closes]} — price series for each target stock.
        target_weights: {symbol: weight} — raw target weights (from inverse-vol).
        benchmark_closes: List of SPY close prices, or None if unavailable.
        min_obs: Minimum aligned observations for metric computation (default 40).

    Returns:
        (risk_state: str, exposure_multiplier: float, metrics: dict)
        risk_state is one of "normal", "elevated", "critical".
        exposure_multiplier is 1.00 / 0.70 / 0.40.
        metrics dict contains individual values for logging.
    """
    returns_df, bench_returns, aligned_weights = _build_aligned_returns(
        closes_map,
        target_weights,
        benchmark_closes,
    )

    beta = None
    vol = None
    corr = None

    if returns_df is not None and aligned_weights is not None:
        # Gate on minimum observations — fail-open if insufficient data
        if returns_df.shape[0] < min_obs:
            risk_state = RiskState.NORMAL
            return (
                risk_state.value,
                EXPOSURE_MAP[risk_state],
                {
                    "risk_state": risk_state.value,
                    "exposure_multiplier": EXPOSURE_MAP[risk_state],
                    "beta_63d": None,
                    "vol_20d": None,
                    "corr_20d": None,
                },
            )

        # Volatility (20d)
        if returns_df.shape[1] >= 1:
            vol = _compute_portfolio_vol(returns_df, aligned_weights, lookback=20)

        # Correlation (20d) — requires at least 2 assets
        if returns_df.shape[1] >= 2:
            corr = _compute_avg_corr(returns_df, lookback=20)

        # Beta (63d) — requires benchmark
        if bench_returns is not None:
            beta = _compute_portfolio_beta(returns_df, aligned_weights, bench_returns, min_obs=min_obs)

    risk_state = classify_risk_state(beta, vol, corr)
    multiplier = EXPOSURE_MAP[risk_state]

    metrics = {
        "risk_state": risk_state.value,
        "exposure_multiplier": multiplier,
        "beta_63d": beta,
        "vol_20d": vol,
        "corr_20d": corr,
    }

    return risk_state.value, multiplier, metrics
