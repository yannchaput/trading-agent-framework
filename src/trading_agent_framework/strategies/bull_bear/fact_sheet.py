"""The fact sheet: one row per debate-set stock, computed by code from its momentum row (pure).

The agents read these numbers and must not recompute them; percentages are rounded to one decimal so a 25-stock
prompt stays short.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from trading_agent_framework.strategies.bull_bear.debate_set import DebateStock
from trading_agent_framework.strategies.common.scoring import compute_return_from_prices

_YEAR = 252
_SMA = 200
_MONTH = 21


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


def fact_sheet_row(stock: DebateStock, *, sector: str, weight: float) -> dict[str, Any]:
    """`weight` is the current share of portfolio value (open orders counted), 0 when not held."""
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
        "volatility_pct": _pct(row.volatility),
        "drawdown_from_52w_high_pct": _pct(drawdown_from_high(row.closes)),
        "vs_sma200_pct": _pct(versus_sma200(row.closes)),
        "sector": sector,
        "held": stock.held,
        "weight_pct": _pct(weight),
    }
