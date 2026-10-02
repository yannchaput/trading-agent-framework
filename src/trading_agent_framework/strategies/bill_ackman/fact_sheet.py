"""The compact per-company dict the agents read (pure).

The screen's `Candidate` already holds the numbers; this module only picks, rounds and names them so the
LLM never does arithmetic and every field costs few tokens. `price` and `price_return_12m` come from the
strategy (clock-gated in a backtest), so they are passed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from trading_agent_framework.strategies.bill_ackman.screen import Candidate

TRADING_DAYS_PER_YEAR = 252


def price_return(closes: Sequence[float], lookback: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """The last close over the close `lookback` bars earlier, minus 1; None without that much history."""
    if len(closes) <= lookback:
        return None
    start = closes[-1 - lookback]
    return None if start <= 0 else closes[-1] / start - 1


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def fact_sheet(candidate: Candidate, *, price: float | None, price_return_12m: float | None) -> dict[str, Any]:
    """One company's fact sheet. `debt_reported` appears only when it is False (the debt ratio is then an absence)."""
    sheet: dict[str, Any] = {
        "symbol": candidate.symbol,
        "sic": candidate.sic,
        "market_cap_usd_bn": round(float(candidate.market_cap) / 1e9, 2),
        "fcf_yield": round(candidate.fcf_yield, 4),
        "fcf_margin_5y": round(candidate.fcf_margin, 4),
        "operating_margin": round(candidate.operating_margin, 4),
        "operating_margin_stdev": round(candidate.operating_margin_stdev, 4),
        "revenue_cagr_5y": round(candidate.revenue_growth, 4),
        "net_debt_to_operating_income": round(candidate.net_debt_to_operating_income, 2),
        "fiscal_year_end": candidate.fiscal_year_end.isoformat(),
        "filed": candidate.filed.isoformat(),
        "price": price,
        "price_return_12m": _rounded(price_return_12m, 4),
    }
    if not candidate.debt_reported:
        sheet["debt_reported"] = False
    return sheet


def unavailable_fact_sheet(symbol: str, *, reason: str, price: float | None, price_return_12m: float | None) -> dict[str, Any]:
    """A holding the screen could not describe (a data problem, not a failed gate): price facts and the reason only."""
    return {"symbol": symbol.upper(), "screen_unavailable": reason, "price": price, "price_return_12m": _rounded(price_return_12m, 4)}
