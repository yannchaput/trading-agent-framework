"""A candidate as the agent sees it (spec §4). Pure. Percentages are in percent, rounded to 0.1.

Floats here are for the JSON context only; nothing is sized from them.
"""

from __future__ import annotations

from typing import Any

from trading_agent_framework.strategies.earnings_drift.events import release_timing
from trading_agent_framework.strategies.earnings_drift.screening import Candidate
from trading_agent_framework.utils.clock import MARKET_TZ


def _pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100, 1)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def fact_sheet(candidate: Candidate, max_quantity: int) -> dict[str, Any]:
    surprise, reaction = candidate.surprise.surprise, candidate.reaction
    sales = None
    if surprise.sales_actual is not None and surprise.sales_estimate is not None:
        sales = {
            "actual": float(surprise.sales_actual),
            "estimate": float(surprise.sales_estimate),
            "surprise_pct": _pct(surprise.sales_surprise_pct),
            "result": surprise.sales_result,
        }
    return {
        "symbol": candidate.symbol,
        "reported_at": candidate.event.accepted_at.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M"),
        "timing": release_timing(candidate.event.accepted_at, candidate.reaction_day),
        "eps": {
            "actual": float(surprise.eps_actual),
            "estimate": float(surprise.eps_estimate),
            "surprise_pct": _pct(surprise.eps_surprise_pct),
            "result": surprise.eps_result,
        },
        "sales": sales,
        "reaction": {
            "gap_pct": _pct(reaction.gap_pct),
            "return_pct": _pct(reaction.return_pct),
            "abnormal_pct": _pct(reaction.abnormal_pct),
            "hold_ratio": _round(reaction.hold_ratio, 2),
            "close_location": _round(reaction.close_location, 2),
            "rel_volume": _round(reaction.rel_volume, 1),
        },
        "context": {
            "runup_20d_pct": _pct(reaction.runup_20d_pct),
            "runup_60d_pct": _pct(reaction.runup_60d_pct),
            "atr14_pct": _pct(reaction.atr14_pct),
            "close": round(reaction.close, 2),
            "reaction_low": round(reaction.reaction_low, 2),
        },
        "max_quantity": max_quantity,
        "filing": {"accession_number": candidate.event.accession_number},
        "headlines": [{"time": when, "headline": headline} for when, headline in candidate.headlines],
    }
