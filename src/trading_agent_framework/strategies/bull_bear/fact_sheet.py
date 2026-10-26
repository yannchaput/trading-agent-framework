"""The fact sheet: one row per debate-set stock, computed by code from its momentum row (pure).

The agents read these numbers and must not recompute them; percentages are rounded to one decimal so a 25-stock
prompt stays short.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trading_agent_framework.strategies.bull_bear.debate_set import DebateStock
from trading_agent_framework.strategies.common.scoring import compute_return_from_prices

_YEAR = 252
_SMA = 200
_MONTH = 21
_WEEK = 5
_MONTH_BEFORE_THE_LAST_WEEK = _MONTH - _WEEK  # 16 sessions: the month's sessions that precede the last week


def _pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100, 1)


def drawdown_from_high(closes: Sequence[float]) -> float | None:
    """The last close against the highest close of the last 252 sessions (0 at a high, negative below it)."""
    window = closes[-_YEAR:]
    high = max(window) if window else 0.0
    return None if high <= 0 else closes[-1] / high - 1


def versus_sma200(closes: Sequence[float]) -> float | None:
    """The last close against its 200-session simple average; None with fewer closes."""
    if len(closes) < _SMA:
        return None
    sma = sum(closes[-_SMA:]) / _SMA
    return None if sma <= 0 else closes[-1] / sma - 1


def fact_sheet_row(stock: DebateStock, *, sector: str, fundamentals: Mapping[str, Any]) -> dict[str, Any]:
    """What every agent reads about a stock. Whether it is held is left out on purpose: no agent's view may depend on it."""
    row = stock.row
    inputs = row.inputs
    return {
        "symbol": row.symbol,
        "momentum_rank": stock.rank,
        "momentum_score": round(row.score, 3),
        "return_12m_skip_1m_pct": _pct(inputs.ret_12_1m),
        "return_6m_skip_1m_pct": _pct(inputs.ret_6_1m),
        "return_3m_pct": _pct(inputs.ret_3m),
        "return_1m_pct": _pct(compute_return_from_prices(row.closes, _MONTH, 0)),
        "return_1w_pct": _pct(compute_return_from_prices(row.closes, _WEEK, 0)),
        "return_1m_skip_1w_pct": _pct(compute_return_from_prices(row.closes, _MONTH_BEFORE_THE_LAST_WEEK, _WEEK)),
        "volatility_pct": _pct(row.volatility),
        "drawdown_from_52w_high_pct": _pct(drawdown_from_high(row.closes)),
        "vs_sma200_pct": _pct(versus_sma200(row.closes)),
        "sector": sector,
        **fundamentals,
    }
