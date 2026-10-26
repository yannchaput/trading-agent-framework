"""Fundamentals for the fact sheet: a pure reduction of SEC statements plus a thin SEC source.

The debate's agents used to learn a company's growth, margins, cash flow and debt only if the researcher found them
and wrote them into a 500-character note. Code now reads the latest SEC figures for every candidate and puts them on
the fact sheet beside the price figures, so every stock is judged on the same evidence and the news only supports it.

`fundamentals_fields` is pure (statements in, rounded fields out); `SecFundamentals` reads the statements as of the
strategy clock (filings dated before the review day only) through an injected `SecEdgarClient`-shaped object and never raises for one stock's SEC failure.
Money is shown in billions of the REPORTING currency (labelled `currency`, never converted); a figure that cannot be
computed is left out, never `None`, never guessed.
"""

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from math import isfinite
from typing import Any, Protocol

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.utils.errors import FundamentalsError, FundamentalsNotFoundError

_DAYS_PER_MONTH = 30.44
_BILLION = 1e9
_NOT_A_FILER = "not an SEC filer"
_REQUEST_FAILED = "SEC request failed"
_NO_FIGURES = "no SEC figures"


class _SecClient(Protocol):
    def ticker_to_cik(self, symbol: str) -> str: ...

    def get_company_facts_payload(self, cik: str, *, as_of: datetime) -> dict[str, Any]: ...


def unavailable_fundamentals(reason: str) -> dict[str, Any]:
    """The fact-sheet fields of a stock whose fundamentals could not be read."""
    return {"fundamentals_unavailable": reason}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if isfinite(value) else None


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _months(row: Mapping[str, Any]) -> int | None:
    start, end = _day(row.get("start")), _day(row.get("end"))
    return None if start is None or end is None else round((end - start).days / _DAYS_PER_MONTH)


def _pct(value: float) -> float:
    return round(value * 100, 1) + 0.0


def _billions(value: float) -> float:
    return round(value / _BILLION, 3) + 0.0


def _prior_value(row: Mapping[str, Any] | None) -> float | None:
    prior = (row or {}).get("prior_year") or {}
    return _number(prior.get("value"))


def _margin(operating: float | None, revenue: float | None) -> float | None:
    return None if operating is None or revenue is None or revenue <= 0 else _pct(operating / revenue)


def fundamentals_fields(income: Mapping[str, Mapping[str, Any]], balance: Mapping[str, Mapping[str, Any]], *, stale: bool) -> dict[str, Any]:
    """The fact-sheet fundamentals fields from the SEC income statement and balance sheet (with their `prior_year`)."""
    revenue_row = income.get("revenue")
    operating_row = income.get("operating_income")
    cash_row = income.get("free_cash_flow")
    debt_row = balance.get("debt")

    revenue, revenue_before = _number((revenue_row or {}).get("value")), _prior_value(revenue_row)
    operating, operating_before = _number((operating_row or {}).get("value")), _prior_value(operating_row)
    cash, cash_before = _number((cash_row or {}).get("value")), _prior_value(cash_row)
    debt, debt_before = _number((debt_row or {}).get("value")), _prior_value(debt_row)

    figures: dict[str, Any] = {}
    if revenue is not None and revenue_before is not None and revenue_before > 0:
        figures["revenue_growth_pct"] = _pct(revenue / revenue_before - 1)
    for name, value in (
        ("operating_margin_pct", _margin(operating, revenue)),
        ("operating_margin_prior_year_pct", _margin(operating_before, revenue_before)),
    ):
        if value is not None:
            figures[name] = value
    if cash is not None:
        figures["free_cash_flow_bn"] = _billions(cash)
        if (months := _months(cash_row or {})) is not None:
            figures["free_cash_flow_months"] = months
    if cash_before is not None:
        figures["free_cash_flow_prior_year_bn"] = _billions(cash_before)
    if debt is not None:
        figures["total_debt_bn"] = _billions(debt)
    if debt_before is not None:
        figures["total_debt_prior_year_bn"] = _billions(debt_before)
    if not figures:
        return unavailable_fundamentals(_NO_FIGURES)

    fields: dict[str, Any] = {}
    period_row = revenue_row or next((row for row in income.values() if row.get("end") and row.get("filed")), None)
    if period_row is not None:
        if (end := _day(period_row.get("end"))) is not None:
            fields["fundamentals_period_end"] = end.isoformat()
        if (filed := _day(period_row.get("filed"))) is not None:
            fields["fundamentals_filed"] = filed.isoformat()
        if (months := _months(period_row)) is not None:
            fields["period_months"] = months
    fields.update(figures)
    currency = (revenue_row or next((row for row in (operating_row, cash_row, debt_row) if row and row.get("unit")), None) or {}).get("unit")
    if currency:
        fields["currency"] = currency
    if stale:
        fields["fundamentals_stale"] = True
    return fields


class SecFundamentals:
    """Reads one stock's latest SEC figures as of a time; a failure becomes `fundamentals_unavailable`, never an exception."""

    def __init__(self, client: _SecClient) -> None:
        self._client = client

    def __call__(self, symbol: str, as_of: datetime) -> dict[str, Any]:
        try:
            cik = self._client.ticker_to_cik(symbol)
            payload = self._client.get_company_facts_payload(cik, as_of=as_of)
        except FundamentalsNotFoundError:
            return unavailable_fundamentals(_NOT_A_FILER)
        except FundamentalsError:
            return unavailable_fundamentals(_REQUEST_FAILED)
        # A company-facts `filed` is a DATE (read as midnight), so a filing dated the review day would pass `filed <= as_of`
        # although it may have come after the close: the statements are read as of the instant before the review day starts.
        cutoff = datetime.combine(as_of.date(), time.min, tzinfo=as_of.tzinfo) - timedelta(microseconds=1)
        income = sec.income_statement(payload, as_of=cutoff)
        balance = sec.balance_sheet(payload, as_of=cutoff)
        return fundamentals_fields(income, balance, stale=sec.is_stale_statement(income, as_of=cutoff))
