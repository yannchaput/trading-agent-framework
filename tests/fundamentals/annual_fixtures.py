"""A reduced annual-figures record for a healthy company, with knobs to break one thing at a time."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

YEARS = (2021, 2022, 2023, 2024, 2025)
FLOW_FIELDS = ("revenue", "operating_income", "operating_cash_flow", "capex")


def healthy_figures(
    *,
    years: Sequence[int] = YEARS,
    revenue: Sequence[int] = (100, 110, 120, 130, 140),
    operating_income: Sequence[int] = (20, 22, 24, 26, 28),
    operating_cash_flow: Sequence[int] = (25, 27, 29, 31, 33),
    capex: Sequence[int] = (5, 5, 5, 5, 5),
    debt: int | None = 50,
    debt_by_year: dict[int, int] | None = None,
    cash: int | None = 10,
    shares: int | None = 1000,
    drop: Collection[tuple[str, int]] = (),
) -> dict[str, Any]:
    """Calendar fiscal years, each filed on 15 February of the next year; balances and shares for the last year.

    Defaults: free cash flow 20, 22, 24, 26, 28 (a 20% margin every year), a 20% operating margin every
    year, net debt 40. `drop` removes single (field, year) flow rows. `debt_by_year` replaces the single
    last-year `debt` with one balance row per listed year (`debt` is then ignored).
    """
    series = {"revenue": revenue, "operating_income": operating_income, "operating_cash_flow": operating_cash_flow, "capex": capex}
    flows = [
        {"field": field, "start": f"{year}-01-01", "end": f"{year}-12-31", "value": series[field][index], "filed": f"{year + 1}-02-15"}
        for index, year in enumerate(years)
        for field in FLOW_FIELDS
        if (field, year) not in drop
    ]
    last = years[-1]
    debts = {last: debt} if debt_by_year is None and debt is not None else (debt_by_year or {})
    balances = [{"field": "debt", "end": f"{year}-12-31", "value": value, "filed": f"{year + 1}-02-15"} for year, value in debts.items()]
    if cash is not None:
        balances.append({"field": "cash", "end": f"{last}-12-31", "value": cash, "filed": f"{last + 1}-02-15"})
    share_rows = [] if shares is None else [{"end": f"{last + 1}-01-31", "value": shares, "filed": f"{last + 1}-02-15", "kind": "cover"}]
    return {"cik": "0000000001", "fetched_at": "2026-10-02T00:00:00+00:00", "status": "ok", "flows": flows, "balances": balances, "shares": share_rows}
