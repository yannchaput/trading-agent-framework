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
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from .parameters import CONFIG

if TYPE_CHECKING:
    from .risk_diagnostics import PortfolioRiskDiagnostics

logger = logging.getLogger(__name__)

# Path and pruning window for persisted equity history
_EQUITY_HISTORY_FILE = Path("data/cross_momentum_ptf_history.json")
_MAX_HISTORY_ENTRIES = 70

# Path to the pre-computed US stock universe (produced by batch_us_stock_universe.py)
_UNIVERSE_FILE = Path("data/universe/us_stock_universe.json")


def compute_return_from_prices(
    closes: list[float],
    lookback_days: int,
    skip_days: int = 0,
) -> float | None:
    """Compute percentage return over a lookback window, optionally skipping recent days.

    For "12-1 month momentum": lookback_days=252, skip_days=21.
    Returns (price[-skip_days-1] - price[-lookback_days-skip_days-1]) / price[-lookback_days-skip_days-1].
    """
    needed = lookback_days + skip_days + 1
    if len(closes) < needed:
        return None
    start_price = closes[-needed]
    end_price = closes[-skip_days - 1]
    if start_price <= 0:
        return None
    return (end_price - start_price) / start_price


def annualized_volatility(closes: list[float], window: int) -> float | None:
    """Annualized volatility from daily log returns over the most recent `window` days."""
    if len(closes) < window + 1:
        return None
    recent = closes[-window - 1 :]
    log_returns = []
    for i in range(1, len(recent)):
        if recent[i - 1] <= 0 or recent[i] <= 0:
            return None
        log_returns.append(math.log(recent[i] / recent[i - 1]))
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(252)


def momentum_score(
    ret_12_1m: float,
    ret_6_1m: float,
    ret_3m: float,
    w_12m: float,
    w_6m: float,
    w_3m: float,
) -> float:
    """Weighted momentum score. Returns a single float (higher = stronger momentum)."""
    return w_12m * ret_12_1m + w_6m * ret_6_1m + w_3m * ret_3m


def apply_filters(
    price: float,
    avg_dollar_volume: float,
    volatility: float | None,
    trading_days: int,
    params: dict,
) -> bool:
    """Check tradability filters. Returns True if the stock passes all gates."""
    if price < params["min_price"]:
        return False
    if avg_dollar_volume < params["min_dollar_volume"]:
        return False
    if volatility is not None and volatility > params["max_volatility"]:
        return False
    if trading_days < params["min_trading_days"]:
        return False
    return True


def inverse_volatility_weights(
    selected: list[dict],
    max_pct: float,
    min_pct: float,
) -> list[dict]:
    """Assign inverse-volatility weights, cap/floor, and normalize to 100%.

    Each dict in `selected` must have a "volatility" key (float > 0).
    Mutates each dict in-place to add a "target_weight" key (float 0..1).
    Returns the same list.
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

    # Floor at min_pct
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


def _percentile_rank(values: list[float]) -> list[float]:
    """Return cross-sectional percentile ranks (0 to 1) for each value.

    Higher input values get higher ranks (1 = best/highest momentum).
    Ties receive the average rank.
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


def compute_momentum_breadth_exposure(
    scored: list[dict],
    bands: list[tuple[float, float]] | None = None,
) -> tuple[float, float, int, int]:
    """Compute momentum breadth and the resulting exposure multiplier.

    Momentum breadth = fraction of scored stocks with a positive momentum score.
    The first matching band determines the exposure multiplier.

    Args:
        scored: List of scored stock dicts, each with a "score" key (float).
        bands: Breadth bands as [(min_breadth, exposure), ...]. Defaults to
               MOMENTUM_BREADTH_BANDS.

    Returns:
        (breadth, exposure, positive_count, total_count) tuple.
        breadth is in [0, 1]. exposure is in [0, 1].
    """
    if bands is None:
        bands = CONFIG["MOMENTUM_BREADTH_BANDS"]

    total = len(scored)
    if total == 0:
        return 0.0, 1.0, 0, 0

    positive_count = sum(1 for s in scored if s.get("score", 0) > 0)
    breadth = positive_count / total

    # Find the first matching band (bands are sorted descending by threshold)
    exposure = 1.0  # default if no band matches (shouldn't happen)
    for threshold, mult in cast(list[tuple[float, float]], bands):
        if breadth >= threshold:
            exposure = mult
            break

    return breadth, exposure, positive_count, total


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


def fractional_qty(value: float, decimals: int = 6) -> float:
    """Floor a quantity to the specified number of decimal places.

    Using floor (not round) guarantees the computed quantity never exceeds
    the dollar budget — a quantity that rounds up could produce a cost
    greater than available cash.

    Args:
        value: Raw quantity computed as dollar_amount / share_price.
        decimals: Number of decimal places to keep. Default 6 (Alpaca
            supports 9; 6 provides a safe margin).

    Returns:
        Floored quantity as a float.
    """
    factor = 10**decimals
    return math.floor(value * factor) / factor


def parse_insufficient_buying_power(error: Exception) -> float | None:
    """Extract the broker's real buying_power from a rejected Alpaca order error.

    Alpaca's APIError.__str__ returns the raw JSON error body, e.g.
    '{"buying_power":"132.45","code":40310000,"message":"insufficient buying power"}'.
    Returns None for any error that isn't this specific rejection shape, so a
    caller can safely try this against any broker exception.

    Args:
        error: The exception raised by strategy.submit_order().

    Returns:
        The broker-reported real buying power as a float, or None.
    """
    try:
        payload = json.loads(str(error))
    except json.JSONDecodeError, TypeError:
        return None
    if not isinstance(payload, dict) or payload.get("message") != "insufficient buying power":
        return None
    try:
        return float(payload["buying_power"])
    except KeyError, TypeError, ValueError:
        return None


def load_cross_momentum_universe() -> list[str]:
    """Load the pre-computed universe for cross-sectional momentum.

    Reads data/universe/stock_universe.json (produced by batch_stock_universe.py).
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

    Reads data/universe/stock_universe.json (produced by batch_stock_universe.py).
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
