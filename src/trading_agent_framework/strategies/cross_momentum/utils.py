"""
Utility functions for the cross-sectional momentum strategy.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.helpers import fractional_qty, parse_insufficient_buying_power  # noqa: F401  (moved to utils/helpers.py; re-exported)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from trading_agent_framework.entities.order import Order

    from .risk_diagnostics import PortfolioRiskDiagnostics

logger = logging.getLogger(__name__)

# Path and pruning window for persisted equity history. Both load and save prune to it; 70 leaves slack over the
# 64 values the fast/slow volatility targeting needs (`_compute_realized_volatility` in the strategy).
_EQUITY_HISTORY_FILE = Path("data/cross_momentum_ptf_history.json")
_MAX_HISTORY_ENTRIES = 70

# Path to the pre-computed US stock universe (produced by batch_us_stock_universe.py)
_UNIVERSE_FILE = Path("data/universe/us_stock_universe.json")


def inverse_volatility_weights(
    selected: list[dict],
    max_pct: float,
    min_pct: float,
) -> list[dict]:
    """Assign inverse-volatility weights, cap/floor, and normalize to 100%.

    Each dict in `selected` must have a "volatility" key (float > 0).
    Mutates each dict in-place to add a "target_weight" key (float 0..1).
    Returns the same list.

    Known limitation: the sequence is cap, redistribute the excess, floor, normalize. The floor only raises the total
    above 1.0, and the final normalization then scales every weight down. So no weight ends above `max_pct`, but a
    floored weight can end a little under `min_pct`. A floor that binds is not guaranteed to hold after normalization.
    """
    if not selected:
        return selected

    # Raw weights = 1 / volatility
    raw_weights = []
    for entry in selected:
        vol = entry.get("volatility", 0)
        if vol is None or vol <= 0:
            vol = 0.01  # fallback for degenerate case
        raw_weights.append(1.0 / vol)

    total_raw = sum(raw_weights)
    if total_raw <= 0:
        # All equal-weight as ultimate fallback
        eq = 1.0 / len(selected)
        for entry in selected:
            entry["target_weight"] = eq
        return selected

    # Normalize to 1.0, then apply caps
    weights = [rw / total_raw for rw in raw_weights]

    # Cap at max_pct
    capped = [min(w, max_pct) for w in weights]
    # Redistribute excess from capped positions to uncapped
    excess = sum(weights) - sum(capped)
    if excess > 0:
        uncapped_indices = [i for i, w in enumerate(capped) if w < max_pct]
        if uncapped_indices:
            total_uncapped_weight = sum(capped[i] for i in uncapped_indices)
            if total_uncapped_weight > 0:
                for i in uncapped_indices:
                    capped[i] += excess * (capped[i] / total_uncapped_weight)

    # Floor at min_pct (see the known limitation in the docstring: normalization below can undo it)
    for i, w in enumerate(capped):
        if w < min_pct:
            capped[i] = min_pct

    # Final normalization to 1.0
    total_capped = sum(capped)
    if total_capped > 0:
        final = [w / total_capped for w in capped]
    else:
        eq = 1.0 / len(selected)
        final = [eq] * len(selected)

    for i, entry in enumerate(selected):
        entry["target_weight"] = final[i]

    return selected


def unfilled_sell_proceeds(sells: Iterable[tuple[Order, float]]) -> float:
    """Estimated proceeds of the sells still to fill: each active sell's open quantity at its sizing price.

    A filled part is already in the broker's cash and must not be counted again (live market sells fill within
    seconds); a sell no longer active (filled, canceled, rejected) brings nothing more. In a backtest nothing fills
    before the next session, so every sell counts in full.
    """
    return sum(float(order.quantity - order.filled_quantity) * price for order, price in sells if order.is_active())


def compute_atr_from_df(df: pd.DataFrame, period: int = 14) -> float | None:
    """Compute Average True Range over `period` days from a DataFrame with OHLC columns."""
    if len(df) < period + 1:
        return None
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    tr_values = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_values.append(tr)
    if len(tr_values) < period:
        return None
    return sum(tr_values[-period:]) / period


def close_series(df: pd.DataFrame) -> pd.Series:
    """Closes indexed by session date in market time, one per date (the last), oldest first.

    Live Alpaca daily bars are stamped at midnight market time and backtest bars at the close; both map to the
    same date, so series from either source can be joined on it.
    """
    dates = [ts.date() for ts in pd.DatetimeIndex(df.index).tz_convert(MARKET_TZ)]
    series = pd.Series(df["close"].to_numpy(dtype=float), index=pd.Index(dates))
    return series.loc[~series.index.duplicated(keep="last")].sort_index()


def diagnostics_to_dict(diag: PortfolioRiskDiagnostics) -> dict:
    """Convert a PortfolioRiskDiagnostics instance to a plain dict for serialization."""
    return {
        "date": diag.date,
        "portfolio_volatility_20d": diag.portfolio_volatility_20d,
        "portfolio_volatility_63d": diag.portfolio_volatility_63d,
        "average_correlation_20d": diag.average_correlation_20d,
        "average_correlation_63d": diag.average_correlation_63d,
        "weighted_correlation_63d": diag.weighted_correlation_63d,
        "portfolio_beta_63d": diag.portfolio_beta_63d,
        "largest_sector": diag.largest_sector,
        "largest_sector_exposure": diag.largest_sector_exposure,
        "unknown_sector_exposure": diag.unknown_sector_exposure,
        "herfindahl_index": diag.herfindahl_index,
        "effective_number_positions": diag.effective_number_positions,
        "portfolio_equity": diag.portfolio_equity,
        "rolling_peak": diag.rolling_peak,
        "drawdown": diag.drawdown,
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


def breadth_share(
    closes_by_symbol: dict[str, list[float]],
    sma_window: int,
    min_stocks: int,
) -> float | None:
    """Share of stocks whose latest close is above their own `sma_window`-day simple moving average.

    A stock with fewer than `sma_window` closes, or a non-finite last close or SMA, is left out of both the
    numerator and the denominator. A close equal to its SMA does not count as above. Returns None when fewer
    than `min_stocks` stocks are valid, so a thin day can't trigger an exposure cut.
    """
    above = 0
    valid = 0
    for closes in closes_by_symbol.values():
        if len(closes) < sma_window:
            continue
        sma = sum(closes[-sma_window:]) / sma_window
        if not (math.isfinite(sma) and math.isfinite(closes[-1])):
            continue
        valid += 1
        if closes[-1] > sma:
            above += 1
    if valid < min_stocks:
        return None
    return above / valid


def next_breadth_step(
    breadth: float,
    previous_step: int | None,
    thresholds: tuple[float, ...],
    hysteresis: float,
) -> int:
    """Breadth step (0 = full exposure, higher = more defensive) with hysteresis on the way back up.

    `thresholds` runs from the least to the most defensive boundary, e.g. (0.50, 0.30): breadth below 0.50
    is at least step 1, below 0.30 step 2. A cut to a more defensive step is immediate; moving to a less
    defensive one needs breadth above that step's threshold plus `hysteresis`, and when breadth jumps several
    steps the result is the least defensive step whose bound is met.
    """
    raw = sum(1 for threshold in thresholds if breadth < threshold)
    if previous_step is None or raw >= previous_step:
        return raw
    step = previous_step
    while step > raw and breadth >= thresholds[step - 1] + hysteresis:
        step -= 1
    return step


def breadth_exposure(step: int, exposures: tuple[float, ...]) -> float:
    """Exposure multiplier for a breadth step."""
    return exposures[step]


def sleeve_symbols(parking: dict) -> tuple[str, ...]:
    """Every symbol of the parking sleeve: the fallback (SHV) first, then the trend assets."""
    return (parking["symbol"], *parking["trend_assets"])


def trend_reading(closes: list[float], sma_window: int) -> tuple[float, float] | None:
    """(last close, simple average of the last `sma_window` closes), or None with too few or non-finite values."""
    if len(closes) < sma_window:
        return None
    last = closes[-1]
    sma = sum(closes[-sma_window:]) / sma_window
    if not (math.isfinite(last) and math.isfinite(sma)):
        return None
    return last, sma


def sleeve_weights(
    closes_by_asset: dict[str, list[float]],
    trend_assets: tuple[str, ...],
    sma_window: int,
    fallback: str,
) -> dict[str, float]:
    """Share of the parking sleeve for the fallback and each trend asset; the shares sum to 1.0.

    Each trend asset gets 1/len(trend_assets) while its last close is strictly above its SMA; otherwise (or when
    its closes are missing, too short or non-finite) that share goes to `fallback`. No hysteresis: the weekly
    cadence, the rebalance band and the minimum trade already damp a close hovering at its SMA.
    """
    if not trend_assets:
        return {fallback: 1.0}
    share = 1.0 / len(trend_assets)
    weights = {fallback: 0.0, **{asset: 0.0 for asset in trend_assets}}
    for asset in trend_assets:
        reading = trend_reading(closes_by_asset.get(asset, []), sma_window)
        if reading is not None and reading[0] > reading[1]:
            weights[asset] = share
        else:
            weights[fallback] += share
    return weights


def load_breadth_step(path: Path, n_steps: int) -> int | None:
    """Load the persisted breadth step, or None if there is nothing usable.

    None covers a missing file, unreadable or corrupt JSON, a non-dict payload, and a `step` that is not an
    int (a bool does not count) in `0 <= step < n_steps`; the first reading then applies directly.
    """
    try:
        data = json.loads(path.read_text())
    except OSError, ValueError:  # ValueError covers bad JSON and a file that isn't valid text
        return None
    if not isinstance(data, dict):
        return None
    step = data.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < n_steps:
        return None
    return step


def save_breadth_step(path: Path, step: int, breadth: float, today_str: str) -> None:
    """Persist the breadth step so a crash or restart keeps the hysteresis.

    Written atomically (temp file + rename). A disk error only logs a warning: the trading loop must never
    fail because state couldn't be saved.
    """
    tmp_path = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps({"date": today_str, "step": step, "breadth": breadth}, indent=2))
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.warning("Failed to save breadth step to %s: %s", path, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _percentile_rank(values: list[float]) -> list[float]:
    """Return cross-sectional percentile ranks (0 to 1) for each value.

    Higher input values get higher ranks (1 = best/highest momentum).
    Ties receive the average rank.

    Not called by the strategy (no caller in src): kept with `compute_residual_momentum` for the beta experiments.
    """
    n = len(values)
    if n <= 1:
        return [0.5] * n

    arr = np.array(values, dtype=float)

    # argsort twice gives ranks: 0 = smallest, n-1 = largest
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(n, dtype=float)

    # Handle ties: average rank within tied groups
    sorted_arr = arr[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        if j > i + 1:
            mean_rank = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = mean_rank
        i = j

    # Normalize to [0, 1]
    return (ranks / (n - 1)).tolist()


def compute_residual_momentum(
    stock_closes: list[float],
    spy_closes: list[float],
    lookback_days: int,
    skip_days: int = 0,
) -> float | None:
    """Compute cumulative residual return after regressing out market beta.

    Not called by the strategy: residual (beta-adjusted) momentum was tested twice in the beta experiments and dropped.
    Kept so those experiments can be re-run.

    For each trading day t in the lookback window:

        stock_return_t = alpha + beta * SPY_return_t + epsilon_t

    Then the cumulative residual return is prod(1 + epsilon_t) - 1,
    optionally skipping the most recent *skip_days* residuals
    (e.g. skip_days=21 for "12-1 month").

    Args:
        stock_closes: List of daily close prices for the stock.
        spy_closes: List of daily close prices for SPY (same length).
        lookback_days: Number of trading days in the lookback window.
        skip_days: Number of most recent days to exclude (e.g. 21 for 1-month gap).

    Returns:
        Cumulative residual return as a float, or None if data is insufficient.
    """
    needed = lookback_days + skip_days + 1

    # Truncate both series to the shortest common usable length
    usable = min(len(stock_closes), len(spy_closes))
    if usable < needed:
        return None

    stock_window = stock_closes[-usable:]
    spy_window = spy_closes[-usable:]

    # Build daily simple returns for the full usable window
    stock_returns: list[float] = []
    spy_returns: list[float] = []
    for i in range(1, len(stock_window)):
        if stock_window[i - 1] <= 0 or stock_window[i] <= 0:
            return None
        if spy_window[i - 1] <= 0 or spy_window[i] <= 0:
            return None
        stock_returns.append(stock_window[i] / stock_window[i - 1] - 1)
        spy_returns.append(spy_window[i] / spy_window[i - 1] - 1)

    # We need at least the regression window worth of returns
    min_obs = max(lookback_days // 2, 20)
    if len(stock_returns) < min_obs:
        return None

    # OLS regression: stock_return = alpha + beta * spy_return + epsilon
    X = np.column_stack([np.ones(len(spy_returns)), spy_returns])
    y = np.array(stock_returns, dtype=float)

    try:
        coeffs, _residuals, _rank, _singular = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    alpha = float(coeffs[0])
    beta = float(coeffs[1])

    # Residual returns: epsilon_t = stock_return_t - (alpha + beta * SPY_return_t)
    epsilon = y - (alpha + beta * np.array(spy_returns, dtype=float))

    # Extract the lookback window, applying skip
    end = len(epsilon) - skip_days
    start = end - lookback_days
    if start < 0:
        return None

    window_epsilon = epsilon[start:end]

    # Cumulative residual return
    residual_cumulative = float(np.prod(1 + window_epsilon) - 1)

    return residual_cumulative


def _read_raw_entries(path: Path) -> list[dict]:
    """Read the raw list of {date, ptf_value} dicts from a JSON file.

    Returns [] if the file doesn't exist or is corrupt.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError, OSError:
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def load_equity_history(filepath: Path | None = None) -> list[float]:
    """Load persisted portfolio equity history from JSON.

    Returns the last ``MAX_HISTORY_ENTRIES`` values as a flat list of floats.
    Returns an empty list if the file doesn't exist, is corrupt, or has an
    unexpected schema.

    Args:
        filepath: Path to the JSON file. Defaults to ``data/cross_momentum_ptf_history.json``.

    Returns:
        List of portfolio values (floats), most recent last.
    """
    path = filepath or _EQUITY_HISTORY_FILE
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load equity history from %s: %s", path, exc)
        return []

    if not isinstance(data, list):
        logger.warning("Equity history file %s has unexpected format (not a list)", path)
        return []

    values: list[float] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        ptf_value = entry.get("ptf_value")
        if isinstance(ptf_value, (int, float)):
            values.append(float(ptf_value))

    # Prune to max entries
    if len(values) > _MAX_HISTORY_ENTRIES:
        values = values[-_MAX_HISTORY_ENTRIES:]

    return values


def save_equity_history(
    history: list[float],
    today_str: str,
    filepath: Path | None = None,
) -> None:
    """Persist the latest portfolio value to a JSON equity history file.

    The file is the authoritative source of historical dates — only
    ``history[-1]`` (today's value) is persisted. The full list is accepted
    as a parameter for ergonomic call sites (pass ``self._equity_history``).

    The file is written atomically (temp file + rename) to avoid corruption
    on crash mid-write. If the last entry already has the same date as
    ``today_str``, it is overwritten rather than creating a duplicate.
    Entries are pruned to the last ``_MAX_HISTORY_ENTRIES``.

    Args:
        history: List of portfolio values (floats), most recent last. Only
            ``history[-1]`` (today's latest value) is persisted — the file
            is the authoritative source for all historical dates.
        today_str: ISO date string (``YYYY-MM-DD``) for today's entry.
        filepath: Path to the JSON file. Defaults to ``data/cross_momentum_ptf_history.json``.
    """
    path = filepath or _EQUITY_HISTORY_FILE

    # Load existing entries to preserve their dates across restarts
    existing = _read_raw_entries(path)
    # Remove any existing entry for today (dedup on restart)
    existing = [e for e in existing if e.get("date") != today_str]
    # Append today's latest value
    existing.append({"date": today_str, "ptf_value": float(history[-1]) if history else 0.0})

    # Prune
    if len(existing) > _MAX_HISTORY_ENTRIES:
        existing = existing[-_MAX_HISTORY_ENTRIES:]

    # Atomic write: tmp file then rename
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(existing, indent=2))
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.warning("Failed to save equity history to %s: %s", path, exc)
        # Clean up tmp file if it exists
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def load_cross_momentum_universe() -> list[str]:
    """Load the pre-computed universe for cross-sectional momentum.

    Reads data/universe/us_stock_universe.json (produced by batch_us_stock_universe.py).
    Falls back to an empty list if the file doesn't exist.

    Returns:
        List of ticker symbols.
    """
    if _UNIVERSE_FILE.exists():
        try:
            data = json.loads(_UNIVERSE_FILE.read_text())
            symbols = data.get("symbols", [])
            logger.info("Loaded %s symbols from %s (dated %s)", len(symbols), _UNIVERSE_FILE, data.get("date", "unknown"))
            return symbols
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse universe file %s: %s", _UNIVERSE_FILE, e)
    return []


def get_cross_momentum_universe_last_date() -> datetime | None:
    """Get the last date of the pre-computed universe for cross-sectional momentum.

    Reads data/universe/us_stock_universe.json (produced by batch_us_stock_universe.py).
    Returns None if the file doesn't exist or if the date is not found.

    Returns:
        datetime: The last date of the universe, or None if not found.
    """
    if _UNIVERSE_FILE.exists():
        try:
            data = json.loads(_UNIVERSE_FILE.read_text())
            date_str = data.get("date")
            if date_str:
                return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse universe file %s: %s", _UNIVERSE_FILE, e)
    return None
