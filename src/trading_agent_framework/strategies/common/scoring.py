"""Momentum scoring shared by cross_momentum and bull_bear (pure: no I/O, no pandas).

Moved from cross_momentum (`utils.py` and `CrossMomentumStrategy._compute_indicators_for_ticker`) so that bull_bear
shortlists on cross_momentum's own ranking by construction. `params` is cross_momentum's `CONFIG`, or any mapping
with the keys read here (`min_trading_days`, `skip_days`, `volatility_window`, the filter thresholds, the weights).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_YEAR, _HALF_YEAR, _QUARTER = 252, 126, 63
_DOLLAR_VOLUME_DAYS = 20


def compute_return_from_prices(
    closes: Sequence[float],
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


def annualized_volatility(closes: Sequence[float], window: int) -> float | None:
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
    params: Mapping[str, Any],
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


@dataclass(frozen=True, slots=True)
class MomentumInputs:
    price: float  # the last completed close
    ret_12_1m: float
    ret_6_1m: float
    ret_3m: float
    volatility: float | None  # annualized, over `volatility_window` sessions; None when it cannot be computed
    avg_dollar_volume: float  # the last 20 sessions' average volume times the last close
    trading_days: int


def momentum_inputs(closes: Sequence[float], volumes: Sequence[float], params: Mapping[str, Any]) -> MomentumInputs | None:
    """The returns, volatility and liquidity cross_momentum scores on, from completed sessions, oldest first.

    None when the history is shorter than `min_trading_days` or too short for one of the returns.
    """
    trading_days = len(closes)
    if trading_days < params["min_trading_days"]:
        return None
    skip = params["skip_days"]
    ret_12_1m = compute_return_from_prices(closes, _YEAR, skip)
    ret_6_1m = compute_return_from_prices(closes, _HALF_YEAR, skip)
    ret_3m = compute_return_from_prices(closes, _QUARTER, 0)
    if ret_12_1m is None or ret_6_1m is None or ret_3m is None:
        return None
    price = closes[-1]
    recent = volumes[-_DOLLAR_VOLUME_DAYS:] if len(volumes) >= _DOLLAR_VOLUME_DAYS else volumes
    avg_volume = sum(recent) / max(len(recent), 1)
    return MomentumInputs(
        price=price,
        ret_12_1m=ret_12_1m,
        ret_6_1m=ret_6_1m,
        ret_3m=ret_3m,
        volatility=annualized_volatility(closes, params["volatility_window"]),
        avg_dollar_volume=avg_volume * price,
        trading_days=trading_days,
    )


@dataclass(frozen=True, slots=True)
class MomentumRow:
    symbol: str
    inputs: MomentumInputs
    volatility: float  # `inputs.volatility`, known to be finite and above zero
    score: float
    closes: tuple[float, ...]  # the completed closes scored, oldest first (the fact sheet reads them)


def score_stock(symbol: str, closes: Sequence[float], volumes: Sequence[float], params: Mapping[str, Any]) -> MomentumRow | None:
    """A stock's momentum row, or None when it has no inputs, fails cross_momentum's filters, or has no volatility.

    cross_momentum ranks a stock whose volatility cannot be computed (non-positive closes); bull_bear sizes by
    volatility, so `score_stock` leaves such a stock out. It cannot happen with positive prices.
    """
    inputs = momentum_inputs(closes, volumes, params)
    if inputs is None or inputs.volatility is None or not (math.isfinite(inputs.volatility) and inputs.volatility > 0):
        return None
    if not apply_filters(inputs.price, inputs.avg_dollar_volume, inputs.volatility, inputs.trading_days, params):
        return None
    score = momentum_score(inputs.ret_12_1m, inputs.ret_6_1m, inputs.ret_3m, params["w_12m"], params["w_6m"], params["w_3m"])
    return MomentumRow(symbol=symbol, inputs=inputs, volatility=inputs.volatility, score=score, closes=tuple(closes))


@dataclass(frozen=True, slots=True)
class RankedRow:
    rank: int  # 1 = the strongest momentum
    row: MomentumRow


def rank(rows: Iterable[MomentumRow]) -> list[RankedRow]:
    """Rows sorted by score, best first, ranked from 1 (a stable sort: equal scores keep their input order)."""
    ordered = sorted(rows, key=lambda row: row.score, reverse=True)
    return [RankedRow(index, row) for index, row in enumerate(ordered, start=1)]
