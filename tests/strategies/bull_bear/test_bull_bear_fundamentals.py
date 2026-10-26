from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from trading_agent_framework.strategies.bull_bear.fundamentals import SecFundamentals, fundamentals_fields, unavailable_fundamentals
from trading_agent_framework.utils.errors import FundamentalsError, FundamentalsNotFoundError


def _flow(value, start, end, prior=None, unit="USD", filed="2026-03-25", form="10-Q"):
    row = {"value": value, "unit": unit, "filed": filed, "form": form, "start": start, "end": end}
    if prior is not None:
        row["prior_year"] = {"value": prior[0], "start": prior[1], "end": prior[2]}
    return row


MU_INCOME = {
    "revenue": _flow(23.86e9, "2025-11-28", "2026-02-26", (8.05e9, "2024-11-29", "2025-02-27")),
    "operating_income": _flow(16.1e9, "2025-11-28", "2026-02-26", (2.0e9, "2024-11-29", "2025-02-27")),
    "free_cash_flow": _flow(8.54e9, "2025-08-29", "2026-02-26", (-0.075e9, "2024-08-30", "2025-02-27")),
}
MU_BALANCE = {"debt": {"value": 9.56e9, "unit": "USD", "filed": "2026-03-25", "end": "2026-02-26", "prior_year": {"value": 11.54e9, "end": "2025-02-27"}}}


def test_the_fields_compare_the_latest_period_with_a_year_earlier():
    f = fundamentals_fields(MU_INCOME, MU_BALANCE, stale=False)
    assert f["revenue_growth_pct"] == 196.4
    assert f["operating_margin_pct"] == 67.5 and f["operating_margin_prior_year_pct"] == 24.8
    assert f["free_cash_flow_bn"] == 8.54 and f["free_cash_flow_prior_year_bn"] == -0.075
    assert f["total_debt_bn"] == 9.56 and f["total_debt_prior_year_bn"] == 11.54
    assert f["period_months"] == 3 and f["free_cash_flow_months"] == 6
    assert f["fundamentals_period_end"] == "2026-02-26" and f["fundamentals_filed"] == "2026-03-25"
    assert f["currency"] == "USD" and "fundamentals_stale" not in f and "fundamentals_unavailable" not in f


def test_no_prior_year_omits_growth_and_the_prior_margin():
    income = {
        "revenue": _flow(10e9, "2025-10-01", "2025-12-31"),
        "operating_income": _flow(2e9, "2025-10-01", "2025-12-31"),
    }
    f = fundamentals_fields(income, {}, stale=False)
    assert f["operating_margin_pct"] == 20.0
    assert "revenue_growth_pct" not in f and "operating_margin_prior_year_pct" not in f
    assert "free_cash_flow_bn" not in f and "total_debt_bn" not in f
    assert None not in f.values()


def test_revenue_at_or_below_zero_omits_the_margins_and_growth():
    income = {
        "revenue": _flow(0, "2025-10-01", "2025-12-31", (0, "2024-10-01", "2024-12-31")),
        "operating_income": _flow(-1e9, "2025-10-01", "2025-12-31", (-2e9, "2024-10-01", "2024-12-31")),
    }
    f = fundamentals_fields(income, {}, stale=False)
    for key in ("revenue_growth_pct", "operating_margin_pct", "operating_margin_prior_year_pct"):
        assert key not in f
    assert None not in f.values()


def test_an_ifrs_filer_keeps_its_currency():
    income = {"revenue": _flow(7.051e9, "2025-01-01", "2025-12-31", unit="EUR", form="20-F"), "operating_income": _flow(3.277e9, "2025-01-01", "2025-12-31", unit="EUR", form="20-F")}
    f = fundamentals_fields(income, {"debt": {"value": 0.738e9, "unit": "EUR", "filed": "2026-03-26", "end": "2025-12-31"}}, stale=False)
    assert f["currency"] == "EUR"
    assert f["total_debt_bn"] == 0.738 and f["period_months"] == 12


def test_the_currency_falls_back_to_the_first_money_field_without_revenue():
    f = fundamentals_fields({"free_cash_flow": _flow(1e9, "2025-10-01", "2025-12-31", unit="TWD")}, {}, stale=False)
    assert f["currency"] == "TWD" and f["free_cash_flow_bn"] == 1.0
    assert f["fundamentals_period_end"] == "2025-12-31" and f["period_months"] == 3


def test_stale_figures_are_flagged():
    assert fundamentals_fields(MU_INCOME, MU_BALANCE, stale=True)["fundamentals_stale"] is True


def test_no_figures_at_all_is_unavailable():
    assert fundamentals_fields({}, {}, stale=False) == {"fundamentals_unavailable": "no SEC figures"}
    assert unavailable_fundamentals("SEC request failed") == {"fundamentals_unavailable": "SEC request failed"}


class _FakeClient:
    def __init__(self, payload: dict[str, Any] | None = None, *, cik_error: Exception | None = None, facts_error: Exception | None = None) -> None:
        self.payload = payload or {}
        self.cik_error = cik_error
        self.facts_error = facts_error
        self.calls: list[dict[str, Any]] = []

    def ticker_to_cik(self, symbol: str) -> str:
        if self.cik_error is not None:
            raise self.cik_error
        return "0000000001"

    def get_company_facts_payload(self, cik: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.facts_error is not None:
            raise self.facts_error
        return self.payload


_AS_OF = datetime(2025, 6, 1, tzinfo=UTC)


def test_an_unknown_ticker_is_unavailable_not_an_error():
    client = _FakeClient(cik_error=FundamentalsNotFoundError("no such ticker"))
    assert SecFundamentals(client)("ZZZZ", _AS_OF) == {"fundamentals_unavailable": "not an SEC filer"}


def test_a_failed_sec_request_is_unavailable():
    client = _FakeClient(facts_error=FundamentalsError("HTTP 503"))
    assert SecFundamentals(client)("AAA", _AS_OF) == {"fundamentals_unavailable": "SEC request failed"}


def _revenue(val: float, start: str, end: str, filed: str, form: str = "10-K") -> dict[str, Any]:
    return {"val": val, "start": start, "end": end, "filed": filed, "form": form, "accn": filed}


def test_the_figures_are_read_as_of_the_given_time():
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            _revenue(4e9, "2023-01-01", "2023-12-31", "2024-02-20"),
                            _revenue(5e9, "2024-01-01", "2024-12-31", "2025-02-20"),
                            _revenue(9e9, "2025-01-01", "2025-12-31", "2026-02-20"),
                        ]
                    }
                }
            }
        }
    }
    client = _FakeClient(payload)
    f = SecFundamentals(client)("AAA", _AS_OF)
    assert client.calls == [{"as_of": _AS_OF}]
    assert f["fundamentals_period_end"] == "2024-12-31" and f["fundamentals_filed"] == "2025-02-20"
    assert f["period_months"] == 12 and f["currency"] == "USD" and f["revenue_growth_pct"] == 25.0
    assert "fundamentals_stale" not in f


def test_a_payload_with_nothing_known_at_the_time_is_unavailable():
    payload = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [_revenue(9e9, "2025-01-01", "2025-12-31", "2026-02-20")]}}}}}
    assert SecFundamentals(_FakeClient(payload))("AAA", _AS_OF) == {"fundamentals_unavailable": "no SEC figures"}


_NY = ZoneInfo("America/New_York")


def _filed_payload(*filings: tuple[float, str, str, str]) -> dict[str, Any]:
    rows = [_revenue(val, start, end, filed) for val, start, end, filed in filings]
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": rows}}}}}


_OLDER = (5e9, "2024-01-01", "2024-12-31", "2025-02-20")
_PRIOR = (4e9, "2023-01-01", "2023-12-31", "2024-02-20")


def test_a_filing_dated_the_review_day_is_hidden_even_at_midday():
    """A 10-Q filed on the review day (often after the close) must not reach the 12:00 review: `filed` is a date, read as midnight."""
    payload = _filed_payload(_PRIOR, _OLDER, (9e9, "2025-01-01", "2025-12-31", "2026-03-03"))
    client = _FakeClient(payload)
    review = datetime(2026, 3, 3, 12, 0, tzinfo=_NY)

    f = SecFundamentals(client)("AAA", review)

    assert f["fundamentals_period_end"] == "2024-12-31" and f["fundamentals_filed"] == "2025-02-20"
    assert f["revenue_growth_pct"] == 25.0
    assert client.calls == [{"as_of": review}]  # the client keeps the REAL as_of (cache freshness only), and no max_age_days


def test_a_filing_dated_the_day_before_is_visible():
    payload = _filed_payload(_PRIOR, _OLDER, (9e9, "2025-01-01", "2025-12-31", "2026-03-02"))

    f = SecFundamentals(_FakeClient(payload))("AAA", datetime(2026, 3, 3, 12, 0, tzinfo=_NY))

    assert f["fundamentals_period_end"] == "2025-12-31" and f["fundamentals_filed"] == "2026-03-02"
    assert f["revenue_growth_pct"] == 80.0


def test_the_only_filing_dated_the_review_day_is_unavailable():
    payload = _filed_payload((9e9, "2025-01-01", "2025-12-31", "2026-03-03"))

    assert SecFundamentals(_FakeClient(payload))("AAA", datetime(2026, 3, 3, 12, 0, tzinfo=_NY)) == {
        "fundamentals_unavailable": "no SEC figures"
    }


def test_the_cutoff_is_the_start_of_the_day_in_the_as_of_timezone():
    payload = _filed_payload(_PRIOR, _OLDER, (9e9, "2025-01-01", "2025-12-31", "2026-03-03"))
    # 2026-03-04 02:00 UTC is still 2026-03-03 in New York: the review day there, so the 03-03 filing stays hidden
    f = SecFundamentals(_FakeClient(payload))("AAA", datetime(2026, 3, 4, 2, 0, tzinfo=UTC).astimezone(_NY))
    assert f["fundamentals_period_end"] == "2024-12-31"
