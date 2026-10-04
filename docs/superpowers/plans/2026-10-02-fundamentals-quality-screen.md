# Fundamentals Quality Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A code-only screen that cuts a list of symbols to the top N companies that were simple, predictable, cash-generative, lightly indebted and reasonably priced on a given date, using only SEC data public on that date.

**Architecture:** Pure functions do the work (`sec.annual_figures` reduces a 4 MB SEC payload to a few KB; `quality.assess`/`quality.rank` apply the gates and the score). Two small I/O classes feed them lazily (`AnnualFiguresStore` over `SecEdgarClient`, `SplitHistory` over yfinance), and `QualityScreen.run` wires them. Time and prices are passed in; nothing here knows `Strategy`, a broker or an LLM.

**Tech Stack:** Python 3.14, `httpx` (SEC), `yfinance` (splits only), `pytest` with `httpx.MockTransport` and hand-written fakes, `uv`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-10-02-fundamentals-quality-screen-design.md`

## Global Constraints

- Branch: `feature/fundamentals-quality-screen`. One commit per task, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run tests with `uv run pytest ...` and lint with `uv run ruff check` (line length 200). Package manager is `uv`, never pip.
- The automated suite never touches the network. Use `httpx.MockTransport`, temp directories and hand-written fakes, not `MagicMock`.
- `sec.py` and `quality.py` stay pure: no I/O, no clock, no state, no logging.
- `edgar_client.py` is the only module that imports `httpx` for SEC access. `splits.py` is the only module that imports `yfinance` for splits, and only inside a function body.
- No raw library exception escapes: wrap failures in `FundamentalsError` / `FundamentalsNotFoundError`.
- A row is **known** on `as_of` when its `filed` date is strictly before `as_of.date()`.
- A fiscal year is identified by its period **end date**, never by XBRL's `fy` field. A fiscal-year period lasts 350 to 380 days.
- Statement figures are `int`, price and market cap are `Decimal`, ratios and the score are `float`.
- Freshness rule, used by both caches: stale when `fetched_at < as_of - max_age_days`.
- Defaults (`ScreenParams`): `years=5`, `min_growth_years=3`, `max_filing_age_months=18`, `max_net_debt_to_operating_income=4.0`, `excluded_sic_ranges=((4900, 4999), (6000, 6799))`, `weights=(0.4, 0.3, 0.3)`, `top_n=15`, `max_age_days=30`, `max_fetch_failure_ratio=0.2`.
- Rejection reasons, in gate order: `no_data`, `stale_filing`, `insufficient_history`, `operating_loss`, `negative_fcf`, `shrinking_revenue`, `too_much_debt`, `excluded_sector`, `duplicate_listing`, `no_price`, `no_split_data`.

## Review Focus

Inputs the spec implies but does not spell out, each pinned by a test in the task that owns the code:

1. **Two listings of one company (GOOGL and GOOG are both in the universe file).** Expected: one candidate slot, the other rejected as `duplicate_listing`; if the first has no price, the second still gets its chance. (Task 7)
2. **A year with zero or negative revenue.** Expected: `insufficient_history`, never a `ZeroDivisionError`. (Task 3)
3. **`price_of` raises a framework error, or returns 0.** Expected: that symbol is `no_price` and the run continues. (Task 7)
4. **A naive `as_of`.** Expected: a `ValueError` naming the problem at the top of `run`, not a `TypeError` inside a cache. (Task 7)
5. **An empty symbol list, a list where everything is rejected, or the same symbol twice.** Expected: an empty result with no division by zero, and a repeated symbol processed once. (Task 7)

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/trading_agent_framework/utils/errors.py` | modify | Add `FundamentalsNotFoundError`. |
| `src/trading_agent_framework/fundamentals/edgar_client.py` | modify | Uncached `fetch_json`; 404 raises `FundamentalsNotFoundError`. |
| `src/trading_agent_framework/fundamentals/sec.py` | modify | `annual_figures`, `parse_sic`, tag lists. |
| `src/trading_agent_framework/fundamentals/quality.py` | create | Types, gates (`assess`), score (`rank`). Pure. |
| `src/trading_agent_framework/fundamentals/freshness.py` | create | `is_stale`, the one freshness rule. Pure. |
| `src/trading_agent_framework/fundamentals/splits.py` | create | `restate_shares`, `split_rows` (pure) and `SplitHistory` (I/O). |
| `src/trading_agent_framework/fundamentals/annual_store.py` | create | `AnnualFiguresStore` (I/O). |
| `src/trading_agent_framework/fundamentals/screen.py` | create | `QualityScreen`, `build_quality_screen` (wiring). |
| `src/trading_agent_framework/fundamentals/__init__.py` | modify | Exports. |
| `tests/fundamentals/annual_fixtures.py` | create | `healthy_figures(...)` shared by two test files. |
| `tests/fundamentals/test_*.py` | create/modify | One test file per module. |
| `scripts/tests/smoke_quality_screen.py` | create | Manual run against real SEC and Yahoo. |
| `pyproject.toml`, `CLAUDE.md` | modify | `yfinance` core dependency; architecture notes. |

---

### Task 1: SEC client: uncached fetch and a not-found error

**Files:**
- Modify: `src/trading_agent_framework/utils/errors.py` (end of file)
- Modify: `src/trading_agent_framework/fundamentals/edgar_client.py` (`get_json`, `ticker_to_cik`, the two payload getters)
- Test: `tests/fundamentals/test_edgar_client.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `FundamentalsNotFoundError(FundamentalsError)` in `utils/errors.py`.
  - `SecEdgarClient.fetch_json(url: str) -> dict[str, Any]`: one uncached request; HTTP 404 raises `FundamentalsNotFoundError`, any other failure `FundamentalsError`.
  - `SecEdgarClient.fetch_company_facts_payload(cik: str) -> dict[str, Any]` and `SecEdgarClient.fetch_submissions_payload(cik: str) -> dict[str, Any]`: uncached.
  - `SecEdgarClient.ticker_to_cik(symbol)` now raises `FundamentalsNotFoundError` for an unknown ticker.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fundamentals/test_edgar_client.py`, and change its errors import to
`from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError, FundamentalsNotFoundError`:

```python
def test_fetch_json_makes_a_request_every_time_and_writes_nothing(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    client = _client(tmp_path, handler)

    assert client.fetch_json("https://data.sec.gov/x.json") == {"ok": True}
    assert client.fetch_json("https://data.sec.gov/x.json") == {"ok": True}
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_fetch_json_raises_not_found_on_a_404(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(404))

    with pytest.raises(FundamentalsNotFoundError):
        client.fetch_json("https://data.sec.gov/missing.json")


def test_fetch_json_raises_a_plain_fundamentals_error_on_a_server_error(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(500))

    with pytest.raises(FundamentalsError) as raised:
        client.fetch_json("https://data.sec.gov/x.json")
    assert not isinstance(raised.value, FundamentalsNotFoundError)


def test_fetch_json_raises_a_plain_fundamentals_error_on_a_throttle_page(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, text="<html>Request Rate Threshold Exceeded</html>"))

    with pytest.raises(FundamentalsError) as raised:
        client.fetch_json("https://data.sec.gov/x.json")
    assert not isinstance(raised.value, FundamentalsNotFoundError)


def test_get_json_also_raises_not_found_on_a_404(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(404))

    with pytest.raises(FundamentalsNotFoundError):
        client.get_json("https://data.sec.gov/missing.json", ("missing.json",))


def test_ticker_to_cik_raises_not_found_for_an_unknown_symbol(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, json=_TICKERS))

    with pytest.raises(FundamentalsNotFoundError, match="ZZZZ"):
        client.ticker_to_cik("ZZZZ")


def test_fetch_payload_methods_hit_the_sec_endpoints_without_caching(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={})

    client = _client(tmp_path, handler)

    client.fetch_company_facts_payload("0000320193")
    client.fetch_submissions_payload("0000320193")

    assert seen_urls == [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://data.sec.gov/submissions/CIK0000320193.json",
    ]
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_edgar_client.py -v`
Expected: collection error, `ImportError: cannot import name 'FundamentalsNotFoundError'`.

- [ ] **Step 3: Implement**

Append to `src/trading_agent_framework/utils/errors.py`:

```python
class FundamentalsNotFoundError(FundamentalsError):
    """Raised when SEC has nothing for the request (unknown ticker, HTTP 404): a real absence, not a failed lookup."""
```

In `src/trading_agent_framework/fundamentals/edgar_client.py`, change the errors import to
`from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError, FundamentalsNotFoundError`
and replace `get_json` with these two methods:

```python
def fetch_json(self, url: str) -> dict[str, Any]:
    """One uncached request. HTTP 404 raises `FundamentalsNotFoundError`, any other failure `FundamentalsError`."""
    self._rate_limit()
    try:
        response = self._client.get(url, headers=self._headers())
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise FundamentalsNotFoundError(f"Not found: {url}") from exc
        raise FundamentalsError(f"Failed to fetch {url}: {exc}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        # ValueError also covers json.JSONDecodeError: SEC's fair-access throttling
        # sometimes serves an HTML block page with a 2xx status instead of JSON.
        raise FundamentalsError(f"Failed to fetch {url}: {exc}") from exc


def get_json(self, url: str, cache_key: tuple[str, ...]) -> dict[str, Any]:
    cache_path = self._cache_path(*cache_key)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            # A corrupt/truncated cache entry (e.g. from an interrupted write) is
            # treated as a cache miss so it self-heals on the next fetch, rather than
            # permanently wedging the tool until someone deletes the file by hand.
            pass
    payload = self.fetch_json(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload
```

Replace `ticker_to_cik`, `get_company_facts_payload` and `get_submissions_payload` with:

```python
def ticker_to_cik(self, symbol: str) -> str:
    payload = self.get_json(SEC_COMPANY_TICKERS_URL, ("company_tickers.json",))
    try:
        return sec.parse_company_tickers(payload, symbol)
    except ValueError as exc:
        raise FundamentalsNotFoundError(str(exc)) from exc


def get_company_facts_payload(self, cik: str) -> dict[str, Any]:
    return self.get_json(_company_facts_url(cik), ("companyfacts", f"CIK{cik}.json"))


def fetch_company_facts_payload(self, cik: str) -> dict[str, Any]:
    """Uncached: the payload is about 4 MB, and the quality screen keeps only a reduced copy."""
    return self.fetch_json(_company_facts_url(cik))


def get_submissions_payload(self, cik: str) -> dict[str, Any]:
    return self.get_json(_submissions_url(cik), ("submissions", f"CIK{cik}.json"))


def fetch_submissions_payload(self, cik: str) -> dict[str, Any]:
    """Uncached, for a caller that needs one field of it."""
    return self.fetch_json(_submissions_url(cik))
```

Add these module-level helpers above `class SecEdgarClient`:

```python
def _company_facts_url(cik: str) -> str:
    return f"{SEC_DATA_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"


def _submissions_url(cik: str) -> str:
    return f"{SEC_DATA_BASE_URL}/submissions/CIK{cik}.json"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals tests/agents -q && uv run ruff check`
Expected: all pass (the existing `test_ticker_to_cik_unknown_symbol_raises_fundamentals_error` still passes: the new error is a subclass), no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/utils/errors.py src/trading_agent_framework/fundamentals/edgar_client.py tests/fundamentals/test_edgar_client.py
git commit -m "feat: uncached SEC fetch and FundamentalsNotFoundError

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `sec.annual_figures` and `sec.parse_sic`

**Files:**
- Modify: `src/trading_agent_framework/fundamentals/sec.py` (new constants after `_FORM_PRIORITY`; new functions at the end of the file)
- Test: `tests/fundamentals/test_sec.py` (append)

**Interfaces:**
- Consumes: the existing `INCOME_STATEMENT_TAGS`, `BALANCE_SHEET_TAGS`, `_parse_dt` in `sec.py`.
- Produces:
  - `sec.MIN_FISCAL_YEAR_DAYS = 350`, `sec.MAX_FISCAL_YEAR_DAYS = 380`.
  - `sec.annual_figures(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]` returning
    `{"flows": [...], "balances": [...], "shares": [...]}` where
    - a flow row is `{"field": str, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "value": int, "filed": "YYYY-MM-DD"}` with `field` in `revenue`, `operating_income`, `operating_cash_flow`, `capex`;
    - a balance row is `{"field": "debt" | "cash", "end": ..., "value": int, "filed": ...}`;
    - a share row is `{"end": ..., "value": int, "filed": ...}`, cover-page (`dei`) rows first, then weighted-average rows.
  - `sec.parse_sic(submissions_payload: dict[str, Any]) -> int | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fundamentals/test_sec.py`:

```python
def _annual(val: int, year: int, filed: str, *, form: str = "10-K") -> dict[str, object]:
    return {"val": val, "start": f"{year}-01-01", "end": f"{year}-12-31", "filed": filed, "form": form}


def _instant(val: int, end: str, filed: str, *, form: str = "10-K") -> dict[str, object]:
    return {"val": val, "end": end, "filed": filed, "form": form}


def _usd(*rows: dict[str, object]) -> dict[str, object]:
    return {"units": {"USD": list(rows)}}


def _share_units(*rows: dict[str, object]) -> dict[str, object]:
    return {"units": {"shares": list(rows)}}


def _company(gaap: dict[str, object] | None = None, dei: dict[str, object] | None = None) -> dict[str, object]:
    return {"facts": {"us-gaap": gaap or {}, "dei": dei or {}}}


def _rows(figures: dict[str, list[dict[str, object]]], kind: str, field: str) -> list[dict[str, object]]:
    return [row for row in figures[kind] if row["field"] == field]


def test_annual_figures_keeps_every_filed_version_of_a_fiscal_year() -> None:
    # Each 10-K repeats earlier years; the 2025 filing restates 2023.
    payload = _company(
        {
            "Revenues": _usd(
                _annual(100, 2023, "2024-02-15"),
                _annual(110, 2024, "2025-02-15"),
                _annual(101, 2023, "2026-02-15"),
                _annual(120, 2025, "2026-02-15"),
            )
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["end"], row["filed"], row["value"]) for row in revenue] == [
        ("2023-12-31", "2024-02-15", 100),
        ("2023-12-31", "2026-02-15", 101),
        ("2024-12-31", "2025-02-15", 110),
        ("2025-12-31", "2026-02-15", 120),
    ]
    assert revenue[0]["start"] == "2023-01-01"


def test_annual_figures_drops_quarters_and_filings_that_are_not_annual_reports() -> None:
    payload = _company(
        {
            "Revenues": _usd(
                {"val": 30, "start": "2025-10-01", "end": "2025-12-31", "filed": "2026-02-15", "form": "10-K"},
                _annual(999, 2025, "2026-01-20", form="8-K"),
                _annual(998, 2025, "2026-02-01", form="10-Q"),
                _annual(120, 2025, "2026-03-01", form="10-K/A"),
            )
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["filed"], row["value"]) for row in revenue] == [("2026-03-01", 120)]


def test_annual_figures_uses_the_first_tag_that_covers_each_period() -> None:
    payload = _company(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(_annual(120, 2025, "2026-02-15")),
            "Revenues": _usd(_annual(90, 2024, "2025-02-15"), _annual(125, 2025, "2026-02-15")),
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["end"], row["value"]) for row in revenue] == [("2024-12-31", 90), ("2025-12-31", 120)]


def test_annual_figures_reads_the_four_flow_fields_with_their_fallback_tags() -> None:
    payload = _company(
        {
            "Revenues": _usd(_annual(100, 2025, "2026-02-15")),
            "OperatingIncomeLoss": _usd(_annual(20, 2025, "2026-02-15")),
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": _usd(_annual(25, 2025, "2026-02-15")),
            "PaymentsToAcquireProductiveAssets": _usd(_annual(5, 2025, "2026-02-15")),
        }
    )

    flows = sec.annual_figures(payload)["flows"]

    assert {row["field"]: row["value"] for row in flows} == {"revenue": 100, "operating_income": 20, "operating_cash_flow": 25, "capex": 5}


def test_annual_figures_prefers_total_long_term_debt() -> None:
    payload = _company(
        {
            "LongTermDebt": _usd(_instant(90, "2025-12-31", "2026-02-15")),
            "LongTermDebtNoncurrent": _usd(_instant(70, "2025-12-31", "2026-02-15")),
        }
    )

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [90]


def test_annual_figures_sums_noncurrent_and_current_debt_when_there_is_no_total() -> None:
    payload = _company(
        {
            "LongTermDebtNoncurrent": _usd(_instant(70, "2024-12-31", "2025-02-15"), _instant(80, "2025-12-31", "2026-02-15")),
            "LongTermDebtCurrent": _usd(_instant(12, "2025-12-31", "2026-02-15")),
        }
    )

    debt = _rows(sec.annual_figures(payload), "balances", "debt")

    assert [(row["end"], row["value"]) for row in debt] == [("2024-12-31", 70), ("2025-12-31", 92)]


def test_annual_figures_falls_back_to_the_combined_debt_tag() -> None:
    payload = _company({"DebtLongtermAndShorttermCombinedAmount": _usd(_instant(55, "2025-12-31", "2026-02-15"))})

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [55]


def test_annual_figures_reads_cash_from_annual_reports_only() -> None:
    payload = _company(
        {
            "CashAndCashEquivalentsAtCarryingValue": _usd(
                _instant(40, "2025-12-31", "2026-02-15"),
                _instant(45, "2026-03-31", "2026-05-01", form="10-Q"),
            )
        }
    )

    assert [(row["end"], row["value"]) for row in _rows(sec.annual_figures(payload), "balances", "cash")] == [("2025-12-31", 40)]


def test_annual_figures_keeps_both_share_counts_cover_page_first() -> None:
    payload = _company(
        gaap={"WeightedAverageNumberOfDilutedSharesOutstanding": _share_units({"val": 1010, "start": "2026-01-01", "end": "2026-03-31", "filed": "2026-05-01", "form": "10-Q"})},
        dei={
            "EntityCommonStockSharesOutstanding": _share_units(
                _instant(1000, "2026-01-31", "2026-02-15"),
                _instant(1005, "2026-04-20", "2026-05-01", form="10-Q"),
                _instant(9999, "2026-05-01", "2026-05-02", form="8-K"),
            )
        },
    )

    shares = sec.annual_figures(payload)["shares"]

    assert shares == [
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15"},
        {"end": "2026-04-20", "value": 1005, "filed": "2026-05-01"},
        {"end": "2026-03-31", "value": 1010, "filed": "2026-05-01"},
    ]


def test_annual_figures_skips_rows_without_a_value_or_a_filing_date() -> None:
    payload = _company(
        {
            "Revenues": _usd(
                {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-15", "form": "10-K"},
                {"val": 100, "start": "2025-01-01", "end": "2025-12-31", "form": "10-K"},
            )
        }
    )

    assert sec.annual_figures(payload) == {"flows": [], "balances": [], "shares": []}


def test_annual_figures_of_an_empty_payload_is_empty() -> None:
    assert sec.annual_figures({}) == {"flows": [], "balances": [], "shares": []}


def test_parse_sic_reads_the_code_as_an_integer() -> None:
    assert sec.parse_sic({"sic": "3571", "sicDescription": "Electronic Computers"}) == 3571


@pytest.mark.parametrize("payload", [{}, {"sic": ""}, {"sic": None}, {"sic": "n/a"}])
def test_parse_sic_of_a_missing_or_malformed_code_is_none(payload: dict[str, object]) -> None:
    assert sec.parse_sic(payload) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_sec.py -v`
Expected: the new tests FAIL with `AttributeError: module 'trading_agent_framework.fundamentals.sec' has no attribute 'annual_figures'` (and `parse_sic`).

- [ ] **Step 3: Implement**

In `src/trading_agent_framework/fundamentals/sec.py`, add after the `_FORM_PRIORITY` line:

```python
# The quality screen's annual figures (`annual_figures`). A fiscal year is identified by its period END
# date: each 10-K repeats three years of figures, all tagged with the filing's own `fy`.
MIN_FISCAL_YEAR_DAYS = 350
MAX_FISCAL_YEAR_DAYS = 380
_ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})
_SHARE_COUNT_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})

ANNUAL_FLOW_TAGS: dict[str, list[str]] = {
    "revenue": INCOME_STATEMENT_TAGS["revenue"],
    "operating_income": INCOME_STATEMENT_TAGS["operating_income"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
```

Append at the end of the file:

```python
def _tag_rows(facts: dict[str, Any], tag: str, unit: str, forms: frozenset[str]) -> list[dict[str, Any]]:
    """One tag's facts in one unit, from `forms` only, that carry a value, a period end and a filing date."""
    rows = (facts.get(tag) or {}).get("units", {}).get(unit, [])
    return [row for row in rows if isinstance(row, dict) and row.get("form") in forms and row.get("val") is not None and row.get("end") and row.get("filed")]


def _is_fiscal_year(row: dict[str, Any]) -> bool:
    start, end = _parse_dt(row.get("start")), _parse_dt(row.get("end"))
    return start is not None and end is not None and MIN_FISCAL_YEAR_DAYS <= (end - start).days <= MAX_FISCAL_YEAR_DAYS


def _slim(row: dict[str, Any], *, with_start: bool = False) -> dict[str, Any]:
    slim = {"start": row["start"]} if with_start else {}
    return {**slim, "end": row["end"], "value": row["val"], "filed": row["filed"]}


def _first_source_per_period(sources: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Every filed version from the first source covering each period end, sorted by (end, filed).

    A later source only fills period ends that no earlier source has, so two tags reporting the same
    year never mix. Within the winning source every version is kept: a later 10-K restates earlier years,
    and the as-of selection needs each version's own `filed` date.
    """
    owner: dict[str, int] = {}
    kept: dict[tuple[str, str], dict[str, Any]] = {}
    for index, rows in enumerate(sources):
        for row in rows:
            if owner.setdefault(row["end"], index) == index:
                kept[(row["end"], row["filed"])] = row
    return sorted(kept.values(), key=lambda row: (row["end"], row["filed"]))


def _debt_sources(gaap: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Total debt, in order of preference: the total tag, noncurrent + current, the combined tag."""
    current = {(row["end"], row["filed"]): row["val"] for row in _tag_rows(gaap, "LongTermDebtCurrent", "USD", _ANNUAL_FORMS)}
    summed = [{"end": row["end"], "value": row["val"] + current.get((row["end"], row["filed"]), 0), "filed": row["filed"]} for row in _tag_rows(gaap, "LongTermDebtNoncurrent", "USD", _ANNUAL_FORMS)]
    return [
        [_slim(row) for row in _tag_rows(gaap, "LongTermDebt", "USD", _ANNUAL_FORMS)],
        summed,
        [_slim(row) for row in _tag_rows(gaap, "DebtLongtermAndShorttermCombinedAmount", "USD", _ANNUAL_FORMS)],
    ]


def annual_figures(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Reduce a company-facts payload (about 4 MB) to the rows the quality screen needs (a few KB).

    `flows` and `balances` come from annual reports only. `shares` keeps both the cover-page count
    (`dei`, listed first) and the weighted-average diluted count, from annual and quarterly reports:
    a multi-class company has no usable cover-page count in this API.
    """
    root = payload.get("facts", {})
    gaap, dei = root.get("us-gaap", {}), root.get("dei", {})

    flows: list[dict[str, Any]] = []
    for field, tags in ANNUAL_FLOW_TAGS.items():
        sources = [[_slim(row, with_start=True) for row in _tag_rows(gaap, tag, "USD", _ANNUAL_FORMS) if _is_fiscal_year(row)] for tag in tags]
        flows.extend({"field": field, **row} for row in _first_source_per_period(sources))

    cash_sources = [[_slim(row) for row in _tag_rows(gaap, tag, "USD", _ANNUAL_FORMS)] for tag in BALANCE_SHEET_TAGS["cash"]]
    balances: list[dict[str, Any]] = []
    for field, sources in (("debt", _debt_sources(gaap)), ("cash", cash_sources)):
        balances.extend({"field": field, **row} for row in _first_source_per_period(sources))

    cover = [_slim(row) for row in _tag_rows(dei, "EntityCommonStockSharesOutstanding", "shares", _SHARE_COUNT_FORMS)]
    weighted = [_slim(row) for row in _tag_rows(gaap, "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", _SHARE_COUNT_FORMS)]
    shares = [*_first_source_per_period([cover]), *_first_source_per_period([weighted])]

    return {"flows": flows, "balances": balances, "shares": shares}


def parse_sic(submissions_payload: dict[str, Any]) -> int | None:
    """The company's SIC industry code from a submissions payload, or `None` when absent or malformed."""
    text = str(submissions_payload.get("sic") or "").strip()
    return int(text) if text.isdigit() else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals/test_sec.py -v && uv run ruff check`
Expected: all pass, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/fundamentals/sec.py tests/fundamentals/test_sec.py
git commit -m "feat: reduce SEC company facts to annual figures for the quality screen

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `quality.py`: types and the numeric gates

**Files:**
- Create: `src/trading_agent_framework/fundamentals/quality.py`
- Create: `tests/fundamentals/annual_fixtures.py`
- Test: `tests/fundamentals/test_quality.py`

**Interfaces:**
- Consumes: `sec.MIN_FISCAL_YEAR_DAYS`, `sec.MAX_FISCAL_YEAR_DAYS`; the row shapes produced by `sec.annual_figures` (Task 2), wrapped in a record that also has `"status": "ok" | "absent"`.
- Produces (all in `fundamentals/quality.py`):
  - `ScreenParams` (frozen dataclass, defaults in Global Constraints).
  - `Survivor` (frozen dataclass): `symbol: str`, `fiscal_year_end: date`, `filed: date`, `free_cash_flow: int`, `fcf_margin: float`, `operating_margin: float`, `operating_margin_stdev: float`, `revenue_growth: float`, `net_debt_to_operating_income: float`, `debt_reported: bool`, `shares: int | None`, `counted_on: date | None`.
  - `assess(symbol: str, figures: Mapping[str, Any] | None, *, as_of: datetime, params: ScreenParams) -> Survivor | str`: a `Survivor`, or the rejection reason of the first failed gate among `no_data`, `stale_filing`, `insufficient_history`, `operating_loss`, `negative_fcf`, `shrinking_revenue`, `too_much_debt`.
  - Test helper `tests.fundamentals.annual_fixtures.healthy_figures(...)` (used again in Task 7).

- [ ] **Step 1: Write the fixture helper**

Create `tests/fundamentals/annual_fixtures.py`:

```python
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
    cash: int | None = 10,
    shares: int | None = 1000,
    drop: Collection[tuple[str, int]] = (),
) -> dict[str, Any]:
    """Calendar fiscal years, each filed on 15 February of the next year; balances and shares for the last year.

    Defaults: free cash flow 20, 22, 24, 26, 28 (a 20% margin every year), a 20% operating margin every
    year, net debt 40. `drop` removes single (field, year) flow rows.
    """
    series = {"revenue": revenue, "operating_income": operating_income, "operating_cash_flow": operating_cash_flow, "capex": capex}
    flows = [
        {"field": field, "start": f"{year}-01-01", "end": f"{year}-12-31", "value": series[field][index], "filed": f"{year + 1}-02-15"}
        for index, year in enumerate(years)
        for field in FLOW_FIELDS
        if (field, year) not in drop
    ]
    last = years[-1]
    balances = [{"field": field, "end": f"{last}-12-31", "value": value, "filed": f"{last + 1}-02-15"} for field, value in (("debt", debt), ("cash", cash)) if value is not None]
    share_rows = [] if shares is None else [{"end": f"{last + 1}-01-31", "value": shares, "filed": f"{last + 1}-02-15"}]
    return {"cik": "0000000001", "fetched_at": "2026-10-02T00:00:00+00:00", "status": "ok", "flows": flows, "balances": balances, "shares": share_rows}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/fundamentals/test_quality.py`:

```python
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tests.fundamentals.annual_fixtures import FLOW_FIELDS, healthy_figures
from trading_agent_framework.fundamentals.quality import ScreenParams, Survivor, assess

AS_OF = datetime(2026, 6, 1, tzinfo=UTC)
PARAMS = ScreenParams()


def _assess(figures: dict[str, object] | None, as_of: datetime = AS_OF) -> Survivor | str:
    return assess("ACME", figures, as_of=as_of, params=PARAMS)


def _survivor(figures: dict[str, object] | None, as_of: datetime = AS_OF) -> Survivor:
    outcome = _assess(figures, as_of)
    assert isinstance(outcome, Survivor), outcome
    return outcome


def test_a_healthy_company_survives_with_its_metrics() -> None:
    survivor = _survivor(healthy_figures())

    assert survivor.symbol == "ACME"
    assert survivor.fiscal_year_end == date(2025, 12, 31)
    assert survivor.filed == date(2026, 2, 15)
    assert survivor.free_cash_flow == 28
    assert survivor.fcf_margin == pytest.approx(0.2)
    assert survivor.operating_margin == pytest.approx(0.2)
    assert survivor.operating_margin_stdev == pytest.approx(0.0, abs=1e-12)
    assert survivor.revenue_growth == pytest.approx((140 / 100) ** (1 / 4) - 1)
    assert survivor.net_debt_to_operating_income == pytest.approx(40 / 28)
    assert survivor.debt_reported is True
    assert survivor.shares == 1000
    assert survivor.counted_on == date(2026, 1, 31)


def test_no_record_is_no_data() -> None:
    assert _assess(None) == "no_data"


def test_an_absent_record_is_no_data() -> None:
    assert _assess({"status": "absent", "flows": [], "balances": [], "shares": []}) == "no_data"


def test_nothing_filed_before_the_date_is_no_data() -> None:
    assert _assess(healthy_figures(), datetime(2021, 6, 1, tzinfo=UTC)) == "no_data"


def test_a_last_fiscal_year_older_than_18_months_is_stale() -> None:
    assert _assess(healthy_figures(), datetime(2027, 9, 1, tzinfo=UTC)) == "stale_filing"


def test_four_years_of_history_is_insufficient() -> None:
    figures = healthy_figures(drop={(field, 2021) for field in FLOW_FIELDS})

    assert _assess(figures) == "insufficient_history"


def test_a_gap_between_fiscal_years_is_insufficient() -> None:
    assert _assess(healthy_figures(years=(2020, 2022, 2023, 2024, 2025))) == "insufficient_history"


@pytest.mark.parametrize("field", ["operating_income", "operating_cash_flow", "capex"])
def test_a_year_missing_one_field_is_insufficient(field: str) -> None:
    assert _assess(healthy_figures(drop={(field, 2023)})) == "insufficient_history"


@pytest.mark.parametrize("revenue", [0, -5])
def test_a_year_without_positive_revenue_is_insufficient_and_never_divides_by_zero(revenue: int) -> None:
    assert _assess(healthy_figures(revenue=(100, 110, revenue, 130, 140))) == "insufficient_history"


def test_an_operating_loss_in_any_year_is_rejected() -> None:
    assert _assess(healthy_figures(operating_income=(20, 22, -1, 26, 28))) == "operating_loss"


def test_zero_operating_income_counts_as_a_loss() -> None:
    assert _assess(healthy_figures(operating_income=(20, 22, 0, 26, 28))) == "operating_loss"


def test_negative_free_cash_flow_in_any_year_is_rejected() -> None:
    assert _assess(healthy_figures(capex=(5, 5, 40, 5, 5))) == "negative_fcf"


def test_the_first_failed_gate_is_the_reason() -> None:
    figures = healthy_figures(operating_income=(20, 22, -1, 26, 28), capex=(5, 5, 40, 5, 5))

    assert _assess(figures) == "operating_loss"


def test_revenue_up_in_only_two_of_four_years_is_shrinking() -> None:
    assert _assess(healthy_figures(revenue=(100, 90, 85, 95, 140))) == "shrinking_revenue"


def test_revenue_ending_below_where_it_started_is_shrinking() -> None:
    assert _assess(healthy_figures(revenue=(100, 110, 120, 130, 95))) == "shrinking_revenue"


def test_revenue_up_in_three_of_four_years_passes() -> None:
    assert isinstance(_assess(healthy_figures(revenue=(100, 110, 105, 130, 140))), Survivor)


def test_net_debt_above_four_times_operating_income_is_too_much() -> None:
    assert _assess(healthy_figures(debt=200, cash=10)) == "too_much_debt"


def test_net_debt_of_exactly_four_times_operating_income_passes() -> None:
    assert isinstance(_assess(healthy_figures(debt=122, cash=10)), Survivor)


def test_net_cash_passes_with_a_negative_multiple() -> None:
    survivor = _survivor(healthy_figures(debt=0, cash=56))

    assert survivor.net_debt_to_operating_income == pytest.approx(-2.0)


def test_a_missing_debt_figure_counts_as_zero_and_is_flagged() -> None:
    survivor = _survivor(healthy_figures(debt=None, cash=10))

    assert survivor.debt_reported is False
    assert survivor.net_debt_to_operating_income == pytest.approx(-10 / 28)


def test_missing_cash_counts_as_zero() -> None:
    survivor = _survivor(healthy_figures(debt=50, cash=None))

    assert survivor.net_debt_to_operating_income == pytest.approx(50 / 28)


def test_a_filing_dated_today_is_not_known_yet() -> None:
    # The 2025 annual report is filed on 2026-02-15; SEC gives no time of day.
    assert _assess(healthy_figures(), datetime(2026, 2, 15, 23, 0, tzinfo=UTC)) == "insufficient_history"
    assert isinstance(_assess(healthy_figures(), datetime(2026, 2, 16, 0, 1, tzinfo=UTC)), Survivor)


def test_a_restatement_filed_after_the_date_is_ignored() -> None:
    figures = healthy_figures()
    figures["flows"].append({"field": "revenue", "start": "2025-01-01", "end": "2025-12-31", "value": 150, "filed": "2026-08-01"})

    before = _survivor(figures, datetime(2026, 6, 1, tzinfo=UTC))
    after = _survivor(figures, datetime(2026, 9, 1, tzinfo=UTC))

    assert before.operating_margin == pytest.approx(28 / 140)
    assert after.operating_margin == pytest.approx(28 / 150)
    assert after.filed == date(2026, 8, 1)


def test_the_share_count_is_the_latest_one_known_on_the_date() -> None:
    figures = healthy_figures()
    figures["shares"] = [
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15"},
        {"end": "2026-04-30", "value": 1100, "filed": "2026-05-10"},
        {"end": "2026-07-31", "value": 1200, "filed": "2026-08-10"},
    ]

    survivor = _survivor(figures)

    assert (survivor.shares, survivor.counted_on) == (1100, date(2026, 4, 30))


def test_the_cover_page_count_wins_a_tie_on_the_period_end() -> None:
    figures = healthy_figures()
    figures["shares"] = [
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15"},  # cover page: listed first by annual_figures
        {"end": "2026-01-31", "value": 990, "filed": "2026-02-15"},
    ]

    assert _survivor(figures).shares == 1000


def test_a_company_with_no_share_count_still_survives_the_numeric_gates() -> None:
    survivor = _survivor(healthy_figures(shares=None))

    assert (survivor.shares, survivor.counted_on) == (None, None)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_quality.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.fundamentals.quality'`.

- [ ] **Step 4: Implement**

Create `src/trading_agent_framework/fundamentals/quality.py`:

```python
"""Pure quality screen: which companies were simple, predictable, cash-generative, lightly indebted and
reasonably priced on a date, from annual SEC figures known on that date.

No I/O, no clock, no state (same rules as `sec.py`). `assess` applies the numeric gates to one company's
reduced annual figures (`sec.annual_figures`); `rank` scores the companies that were also priced.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from typing import Any

from trading_agent_framework.fundamentals.sec import MAX_FISCAL_YEAR_DAYS, MIN_FISCAL_YEAR_DAYS

_DAYS_PER_MONTH = 30.4375
_REQUIRED_FLOWS = ("revenue", "operating_income", "operating_cash_flow", "capex")


@dataclass(frozen=True, slots=True)
class ScreenParams:
    years: int = 5
    min_growth_years: int = 3
    max_filing_age_months: int = 18
    max_net_debt_to_operating_income: float = 4.0
    excluded_sic_ranges: tuple[tuple[int, int], ...] = ((4900, 4999), (6000, 6799))  # utilities; finance, insurance, real estate
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)  # fcf_yield, fcf_margin, operating-margin stability
    top_n: int = 15
    max_age_days: int = 30
    max_fetch_failure_ratio: float = 0.2


@dataclass(frozen=True, slots=True)
class Survivor:
    """A company that passed the numeric gates; the sector, price and split gates come after."""

    symbol: str
    fiscal_year_end: date
    filed: date  # filing date of the latest fiscal year's revenue figure
    free_cash_flow: int  # latest fiscal year
    fcf_margin: float  # mean over the window
    operating_margin: float  # latest fiscal year
    operating_margin_stdev: float
    revenue_growth: float  # compound annual rate over the window
    net_debt_to_operating_income: float
    debt_reported: bool
    shares: int | None
    counted_on: date | None


def _known(rows: Sequence[Mapping[str, Any]], cutoff: date) -> list[Mapping[str, Any]]:
    """Rows filed strictly before `cutoff`: SEC gives no filing time, so a row filed today is known tomorrow."""
    return [row for row in rows if date.fromisoformat(row["filed"]) < cutoff]


def _latest_versions(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, date], Mapping[str, Any]]:
    """(field, period end) -> the most recently filed version."""
    latest: dict[tuple[str, date], Mapping[str, Any]] = {}
    for row in sorted(rows, key=lambda row: row["filed"]):
        latest[(row["field"], date.fromisoformat(row["end"]))] = row
    return latest


def _consecutive(year_ends: Sequence[date]) -> bool:
    return all(MIN_FISCAL_YEAR_DAYS <= (later - earlier).days <= MAX_FISCAL_YEAR_DAYS for earlier, later in pairwise(year_ends))


def assess(symbol: str, figures: Mapping[str, Any] | None, *, as_of: datetime, params: ScreenParams) -> Survivor | str:
    """Apply the numeric gates to one company; the rejection reason is the first gate that fails."""
    if figures is None or figures.get("status") == "absent":
        return "no_data"
    cutoff = as_of.date()
    flows = _latest_versions(_known(figures.get("flows", []), cutoff))
    if not flows:
        return "no_data"

    year_ends = sorted(end for field, end in flows if field == "revenue")
    if not year_ends:
        return "insufficient_history"
    window = year_ends[-params.years :]
    latest_end = window[-1]
    if (cutoff - latest_end).days > params.max_filing_age_months * _DAYS_PER_MONTH:
        return "stale_filing"
    if len(window) < params.years or not _consecutive(window):
        return "insufficient_history"

    revenues: list[int] = []
    operating_incomes: list[int] = []
    free_cash_flows: list[int] = []
    for end in window:
        rows = [flows.get((field, end)) for field in _REQUIRED_FLOWS]
        if any(row is None for row in rows):
            return "insufficient_history"
        revenue, operating_income, operating_cash_flow, capex = (row["value"] for row in rows)  # ty: ignore[not-subscriptable]
        if revenue <= 0:
            return "insufficient_history"
        revenues.append(revenue)
        operating_incomes.append(operating_income)
        free_cash_flows.append(operating_cash_flow - capex)

    if any(value <= 0 for value in operating_incomes):
        return "operating_loss"
    if any(value <= 0 for value in free_cash_flows):
        return "negative_fcf"
    increases = sum(later > earlier for earlier, later in pairwise(revenues))
    if increases < params.min_growth_years or revenues[-1] < revenues[0]:
        return "shrinking_revenue"

    balances = _latest_versions(_known(figures.get("balances", []), cutoff))
    debt_row, cash_row = balances.get(("debt", latest_end)), balances.get(("cash", latest_end))
    net_debt = (debt_row["value"] if debt_row else 0) - (cash_row["value"] if cash_row else 0)
    if net_debt > params.max_net_debt_to_operating_income * operating_incomes[-1]:
        return "too_much_debt"

    margins = [income / revenue for income, revenue in zip(operating_incomes, revenues, strict=True)]
    share_rows = _known(figures.get("shares", []), cutoff)
    # `max` keeps the first of equal keys, and `annual_figures` lists cover-page counts first.
    share_row = max(share_rows, key=lambda row: (row["end"], row["filed"])) if share_rows else None
    return Survivor(
        symbol=symbol,
        fiscal_year_end=latest_end,
        filed=date.fromisoformat(flows[("revenue", latest_end)]["filed"]),
        free_cash_flow=free_cash_flows[-1],
        fcf_margin=statistics.fmean(fcf / revenue for fcf, revenue in zip(free_cash_flows, revenues, strict=True)),
        operating_margin=margins[-1],
        operating_margin_stdev=statistics.pstdev(margins),
        revenue_growth=(revenues[-1] / revenues[0]) ** (1 / (len(revenues) - 1)) - 1,
        net_debt_to_operating_income=net_debt / operating_incomes[-1],
        debt_reported=debt_row is not None,
        shares=share_row["value"] if share_row else None,
        counted_on=date.fromisoformat(share_row["end"]) if share_row else None,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals/test_quality.py -v && uv run ruff check`
Expected: all pass, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/fundamentals/quality.py tests/fundamentals/annual_fixtures.py tests/fundamentals/test_quality.py
git commit -m "feat: quality screen numeric gates over as-of annual figures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `quality.py`: sector gate, score and ranking

**Files:**
- Modify: `src/trading_agent_framework/fundamentals/quality.py` (append; add `from decimal import Decimal` to its imports)
- Test: `tests/fundamentals/test_quality.py` (append)

**Interfaces:**
- Consumes: `ScreenParams`, `Survivor` from Task 3.
- Produces (in `fundamentals/quality.py`):
  - `Priced` (frozen dataclass): `survivor: Survivor`, `sic: int | None`, `market_cap: Decimal` (must be > 0).
  - `Candidate` (frozen dataclass): `symbol: str`, `rank: int`, `score: float`, `sic: int | None`, `market_cap: Decimal`, `fcf_yield: float`, `fcf_margin: float`, `operating_margin: float`, `operating_margin_stdev: float`, `revenue_growth: float`, `net_debt_to_operating_income: float`, `debt_reported: bool`, `fiscal_year_end: date`, `filed: date`.
  - `ScreenResult` (frozen dataclass): `candidates: list[Candidate]`, `rejections: dict[str, str]`.
  - `sector_excluded(sic: int | None, params: ScreenParams) -> bool`.
  - `rank(priced: Sequence[Priced], params: ScreenParams) -> list[Candidate]`: best first, at most `params.top_n`, `rank` starting at 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fundamentals/test_quality.py`, and extend its imports to
`from decimal import Decimal` and
`from trading_agent_framework.fundamentals.quality import Priced, ScreenParams, Survivor, assess, rank, sector_excluded`:

```python
def _priced(symbol: str, *, market_cap: str, fcf: int = 28, fcf_margin: float = 0.2, stdev: float = 0.01, sic: int | None = 3571) -> Priced:
    survivor = Survivor(
        symbol=symbol,
        fiscal_year_end=date(2025, 12, 31),
        filed=date(2026, 2, 15),
        free_cash_flow=fcf,
        fcf_margin=fcf_margin,
        operating_margin=0.25,
        operating_margin_stdev=stdev,
        revenue_growth=0.08,
        net_debt_to_operating_income=1.5,
        debt_reported=True,
        shares=1000,
        counted_on=date(2026, 1, 31),
    )
    return Priced(survivor=survivor, sic=sic, market_cap=Decimal(market_cap))


@pytest.mark.parametrize(("sic", "excluded"), [(6021, True), (6000, True), (6799, True), (4911, True), (6800, False), (4899, False), (3571, False), (None, False)])
def test_sector_excluded_covers_utilities_and_finance_and_lets_an_unknown_code_pass(sic: int | None, excluded: bool) -> None:
    assert sector_excluded(sic, PARAMS) is excluded


def test_rank_scores_by_weighted_percentiles() -> None:
    priced = [
        _priced("AAA", market_cap="280", fcf_margin=0.2, stdev=0.00),  # yield 0.100: pct 1.0 | margin pct 0.5 | stdev pct 0.0
        _priced("BBB", market_cap="560", fcf_margin=0.3, stdev=0.02),  # yield 0.050: pct 0.5 | margin pct 1.0 | stdev pct 1.0
        _priced("CCC", market_cap="1120", fcf_margin=0.1, stdev=0.01),  # yield 0.025: pct 0.0 | margin pct 0.0 | stdev pct 0.5
    ]

    candidates = rank(priced, PARAMS)

    assert [(c.symbol, c.rank) for c in candidates] == [("AAA", 1), ("BBB", 2), ("CCC", 3)]
    assert [c.score for c in candidates] == pytest.approx([0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 1.0, 0.4 * 0.5 + 0.3 * 1.0 + 0.3 * 0.0, 0.3 * 0.5])
    assert [c.fcf_yield for c in candidates] == pytest.approx([0.1, 0.05, 0.025])


def test_rank_carries_the_survivor_metrics_into_the_candidate() -> None:
    (candidate,) = rank([_priced("AAA", market_cap="280", sic=5812)], PARAMS)

    assert candidate.sic == 5812
    assert candidate.market_cap == Decimal("280")
    assert candidate.fcf_margin == 0.2
    assert candidate.operating_margin == 0.25
    assert candidate.operating_margin_stdev == 0.01
    assert candidate.revenue_growth == 0.08
    assert candidate.net_debt_to_operating_income == 1.5
    assert candidate.debt_reported is True
    assert candidate.fiscal_year_end == date(2025, 12, 31)
    assert candidate.filed == date(2026, 2, 15)


def test_a_single_survivor_gets_percentile_one_on_every_metric() -> None:
    (candidate,) = rank([_priced("AAA", market_cap="280")], PARAMS)

    assert candidate.score == pytest.approx(0.4 + 0.3 + 0.3 * (1 - 1.0))


def test_tied_metrics_share_the_average_percentile() -> None:
    # Same yield (28/280 and 56/560), same margin, same stdev: every percentile is 0.5 for both.
    candidates = rank([_priced("AAA", market_cap="280", fcf=28), _priced("BBB", market_cap="560", fcf=56)], PARAMS)

    assert [c.score for c in candidates] == pytest.approx([0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5] * 2)


def test_equal_scores_are_ordered_by_market_cap_then_symbol() -> None:
    priced = [
        _priced("ZZZ", market_cap="280", fcf=28),
        _priced("MMM", market_cap="560", fcf=56),
        _priced("AAA", market_cap="280", fcf=28),
    ]

    assert [c.symbol for c in rank(priced, PARAMS)] == ["MMM", "AAA", "ZZZ"]


def test_rank_returns_at_most_top_n() -> None:
    priced = [_priced("AAA", market_cap="280"), _priced("BBB", market_cap="560"), _priced("CCC", market_cap="1120")]

    candidates = rank(priced, ScreenParams(top_n=2))

    assert [(c.symbol, c.rank) for c in candidates] == [("AAA", 1), ("BBB", 2)]


def test_rank_of_nothing_is_empty() -> None:
    assert rank([], PARAMS) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_quality.py -v`
Expected: collection error, `ImportError: cannot import name 'Priced'`.

- [ ] **Step 3: Implement**

In `src/trading_agent_framework/fundamentals/quality.py`, add `from decimal import Decimal` to the imports, then append:

```python
@dataclass(frozen=True, slots=True)
class Priced:
    """A survivor of every gate, with its sector code and its split-restated market cap (above zero)."""

    survivor: Survivor
    sic: int | None
    market_cap: Decimal


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    rank: int
    score: float
    sic: int | None
    market_cap: Decimal
    fcf_yield: float
    fcf_margin: float
    operating_margin: float
    operating_margin_stdev: float
    revenue_growth: float
    net_debt_to_operating_income: float
    debt_reported: bool
    fiscal_year_end: date
    filed: date


@dataclass(frozen=True, slots=True)
class ScreenResult:
    candidates: list[Candidate]
    rejections: dict[str, str]  # symbol -> reason of the first failed gate


def sector_excluded(sic: int | None, params: ScreenParams) -> bool:
    """Whether the SIC code falls in an excluded range; a company with no code passes."""
    return sic is not None and any(low <= sic <= high for low, high in params.excluded_sic_ranges)


def _percentiles(values: Sequence[float]) -> list[float]:
    """Each value's percentile rank in `values`: (rank - 1) / (n - 1), ties sharing their average rank."""
    count = len(values)
    if count == 1:
        return [1.0]
    order = sorted(range(count), key=values.__getitem__)
    ranks = [0.0] * count
    start = 0
    while start < count:
        stop = start
        while stop + 1 < count and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average_rank = (start + stop) / 2 + 1
        for position in range(start, stop + 1):
            ranks[order[position]] = average_rank
        start = stop + 1
    return [(rank - 1) / (count - 1) for rank in ranks]


def rank(priced: Sequence[Priced], params: ScreenParams) -> list[Candidate]:
    """Score the priced survivors against each other and return the best `params.top_n`, best first."""
    if not priced:
        return []
    yields = [float(Decimal(item.survivor.free_cash_flow) / item.market_cap) for item in priced]
    yield_pct = _percentiles(yields)
    margin_pct = _percentiles([item.survivor.fcf_margin for item in priced])
    stdev_pct = _percentiles([item.survivor.operating_margin_stdev for item in priced])
    yield_weight, margin_weight, stability_weight = params.weights
    scored = [(yield_weight * yield_pct[index] + margin_weight * margin_pct[index] + stability_weight * (1 - stdev_pct[index]), yields[index], item) for index, item in enumerate(priced)]
    scored.sort(key=lambda entry: (-entry[0], -entry[2].market_cap, entry[2].survivor.symbol))
    return [
        Candidate(
            symbol=item.survivor.symbol,
            rank=position,
            score=score,
            sic=item.sic,
            market_cap=item.market_cap,
            fcf_yield=fcf_yield,
            fcf_margin=item.survivor.fcf_margin,
            operating_margin=item.survivor.operating_margin,
            operating_margin_stdev=item.survivor.operating_margin_stdev,
            revenue_growth=item.survivor.revenue_growth,
            net_debt_to_operating_income=item.survivor.net_debt_to_operating_income,
            debt_reported=item.survivor.debt_reported,
            fiscal_year_end=item.survivor.fiscal_year_end,
            filed=item.survivor.filed,
        )
        for position, (score, fcf_yield, item) in enumerate(scored[: params.top_n], start=1)
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals/test_quality.py -v && uv run ruff check`
Expected: all pass, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/fundamentals/quality.py tests/fundamentals/test_quality.py
git commit -m "feat: quality screen sector gate, percentile score and ranking

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Splits and the freshness rule

**Files:**
- Create: `src/trading_agent_framework/fundamentals/freshness.py`
- Create: `src/trading_agent_framework/fundamentals/splits.py`
- Modify: `pyproject.toml` (the `dependencies` list)
- Test: `tests/fundamentals/test_splits.py`

**Interfaces:**
- Consumes: `FundamentalsError`; `ColorLogger` from `trading_agent_framework.utils.log`.
- Produces:
  - `freshness.is_stale(fetched_at: datetime, as_of: datetime, max_age_days: int) -> bool`.
  - `splits.Split = tuple[date, float]`.
  - `splits.restate_shares(shares: int, counted_on: date, splits: Sequence[Split]) -> Decimal`.
  - `splits.split_rows(history: Any) -> list[Split]`: pure parse of a yfinance history frame; raises `FundamentalsError` on an empty frame.
  - `splits.SplitHistory(cache_file: Path, *, fetch: Callable[[str], list[Split]] | None = None, wall_clock: Callable[[], datetime] | None = None)` with `splits(symbol: str, *, as_of: datetime, max_age_days: int) -> list[Split]`, raising `FundamentalsError` when the lookup fails and no cached copy exists.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"yfinance>=0.2.61",` to `dependencies`, after the `"vectorbt>=1.1.0,<2.0",` line. Leave the `backtesting-yahoo` extra and the `batch-universe` group as they are.

Run: `uv sync`
Expected: resolves with no change to the installed `yfinance` (it is already installed through the `batch-universe` group).

- [ ] **Step 2: Write the failing tests**

Create `tests/fundamentals/test_splits.py`:

```python
from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.fundamentals.splits import Split, SplitHistory, restate_shares, split_rows
from trading_agent_framework.utils.errors import FundamentalsError

FETCHED = datetime(2026, 10, 2, tzinfo=UTC)
FOUR_FOR_ONE = [(date(2020, 8, 31), 4.0)]


class FakeFetch:
    """Stands in for the Yahoo lookup: counts calls, and fails while `error` is set."""

    def __init__(self, splits: list[Split]) -> None:
        self.splits = splits
        self.calls: list[str] = []
        self.error: FundamentalsError | None = None

    def __call__(self, symbol: str) -> list[Split]:
        self.calls.append(symbol)
        if self.error is not None:
            raise self.error
        return list(self.splits)


def _history(cache_file: Path, fetch: FakeFetch) -> SplitHistory:
    return SplitHistory(cache_file, fetch=fetch, wall_clock=lambda: FETCHED)


def _splits(history: SplitHistory, as_of: datetime = FETCHED) -> list[Split]:
    return history.splits("AAPL", as_of=as_of, max_age_days=30)


def test_is_stale_only_when_fetched_more_than_max_age_before_as_of() -> None:
    assert is_stale(FETCHED, datetime(2026, 11, 2, tzinfo=UTC), 30) is True
    assert is_stale(FETCHED, datetime(2026, 11, 1, tzinfo=UTC), 30) is False


def test_a_file_fetched_today_is_fresh_for_every_past_date() -> None:
    assert is_stale(FETCHED, datetime(2020, 1, 1, tzinfo=UTC), 30) is False


def test_restate_shares_without_splits_is_unchanged() -> None:
    assert restate_shares(1000, date(2026, 1, 31), []) == Decimal(1000)


def test_restate_shares_applies_a_split_dated_after_the_count() -> None:
    assert restate_shares(1000, date(2020, 7, 17), FOUR_FOR_ONE) == Decimal(4000)


def test_restate_shares_ignores_a_split_on_or_before_the_count_date() -> None:
    assert restate_shares(1000, date(2020, 8, 31), FOUR_FOR_ONE) == Decimal(1000)
    assert restate_shares(1000, date(2021, 1, 1), FOUR_FOR_ONE) == Decimal(1000)


def test_restate_shares_compounds_several_splits_including_a_reverse_split() -> None:
    splits = [(date(2014, 6, 9), 7.0), (date(2020, 8, 31), 4.0), (date(2023, 5, 1), 0.1)]

    assert restate_shares(1000, date(2015, 1, 1), splits) == Decimal("400.0")


def test_split_rows_keeps_the_non_zero_rows_of_a_yahoo_history() -> None:
    index = pd.to_datetime(["2020-08-28", "2020-08-31", "2020-09-01"]).tz_localize("America/New_York")
    frame = pd.DataFrame({"Close": [499.0, 129.0, 134.0], "Stock Splits": [0.0, 4.0, 0.0]}, index=index)

    assert split_rows(frame) == [(date(2020, 8, 31), 4.0)]


def test_split_rows_of_a_history_with_no_split_is_empty() -> None:
    index = pd.to_datetime(["2026-01-02"]).tz_localize("America/New_York")
    frame = pd.DataFrame({"Close": [10.0], "Stock Splits": [0.0]}, index=index)

    assert split_rows(frame) == []


def test_split_rows_rejects_an_empty_history_as_a_failed_lookup() -> None:
    # yfinance reports a failed download as an empty frame; that must not read as "never split".
    with pytest.raises(FundamentalsError, match="no history"):
        split_rows(pd.DataFrame({"Close": []}))


def test_split_history_fetches_once_and_caches_on_disk(tmp_path: Path) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    cache_file = tmp_path / "splits.json"

    assert _splits(_history(cache_file, fetch)) == FOUR_FOR_ONE
    assert _splits(_history(cache_file, fetch)) == FOUR_FOR_ONE  # a new instance reads the file
    assert fetch.calls == ["AAPL"]


def test_split_history_caches_an_empty_history(tmp_path: Path) -> None:
    fetch = FakeFetch([])
    history = _history(tmp_path / "splits.json", fetch)

    assert _splits(history) == []
    assert _splits(history) == []
    assert fetch.calls == ["AAPL"]


def test_split_history_is_not_refetched_for_a_past_date(tmp_path: Path) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    history = _history(tmp_path / "splits.json", fetch)

    _splits(history)
    _splits(history, as_of=datetime(2021, 1, 4, tzinfo=UTC))

    assert fetch.calls == ["AAPL"]


def test_split_history_refetches_a_stale_entry(tmp_path: Path) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    history = _history(tmp_path / "splits.json", fetch)

    _splits(history)
    _splits(history, as_of=datetime(2026, 11, 15, tzinfo=UTC))

    assert fetch.calls == ["AAPL", "AAPL"]


def test_a_failed_lookup_with_no_cached_copy_raises_and_caches_nothing(tmp_path: Path) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    fetch.error = FundamentalsError("yahoo is down")
    history = _history(tmp_path / "splits.json", fetch)

    with pytest.raises(FundamentalsError, match="yahoo is down"):
        _splits(history)

    fetch.error = None
    assert _splits(history) == FOUR_FOR_ONE


def test_a_failed_refresh_keeps_the_stale_copy_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fetch = FakeFetch(FOUR_FOR_ONE)
    history = _history(tmp_path / "splits.json", fetch)
    _splits(history)
    fetch.error = FundamentalsError("yahoo is down")

    with caplog.at_level(logging.WARNING):
        splits = _splits(history, as_of=datetime(2026, 11, 15, tzinfo=UTC))

    assert splits == FOUR_FOR_ONE
    assert "yahoo is down" in caplog.text


def test_a_corrupt_cache_file_is_treated_as_empty(tmp_path: Path) -> None:
    cache_file = tmp_path / "splits.json"
    cache_file.write_text("{not json", encoding="utf-8")
    fetch = FakeFetch(FOUR_FOR_ONE)

    assert _splits(_history(cache_file, fetch)) == FOUR_FOR_ONE
    assert fetch.calls == ["AAPL"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_splits.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.fundamentals.freshness'`.

- [ ] **Step 4: Implement**

Create `src/trading_agent_framework/fundamentals/freshness.py`:

```python
"""The one freshness rule of the quality screen's on-disk caches (annual figures, split history).

A cached entry is stale when it was fetched more than `max_age_days` before `as_of`. `as_of` is the
caller's clock, so the same rule covers every mode: in a backtest `as_of` is simulated, and an entry
fetched today is fresh for every past date; in paper/live `as_of` is now, and entries refresh every
`max_age_days`. `fetched_at` is real time, but it is only ever compared with `as_of`, never with data.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def is_stale(fetched_at: datetime, as_of: datetime, max_age_days: int) -> bool:
    return fetched_at < as_of - timedelta(days=max_age_days)
```

Create `src/trading_agent_framework/fundamentals/splits.py`:

```python
"""Stock splits: restating a reported share count onto the basis of today's split-adjusted prices.

Bar prices are split-adjusted to today, but a share count in a filing is as reported on its date. A
2-for-1 split after the count would halve the computed market cap. `restate_shares` (pure) fixes the
count; `SplitHistory` is the I/O side and the only module that imports `yfinance` for splits (lazily).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.utils.errors import FundamentalsError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "SplitHistory")

Split = tuple[date, float]


def restate_shares(shares: int, counted_on: date, splits: Sequence[Split]) -> Decimal:
    """`shares`, counted on `counted_on`, restated for every split dated after that day.

    Splits later than the caller's `as_of` are applied too, on purpose: the price this count gets
    multiplied by is already adjusted for them, so this undoes a price adjustment and leaks nothing.
    """
    restated = Decimal(shares)
    for split_date, ratio in splits:
        if split_date > counted_on:
            restated *= Decimal(str(ratio))
    return restated


def split_rows(history: Any) -> list[Split]:
    """The splits in a yfinance `Ticker.history(actions=True)` frame, oldest first.

    yfinance reports a failed download as an empty frame, which must not be read as "never split":
    an empty frame raises `FundamentalsError`.
    """
    if history.empty or "Stock Splits" not in history.columns:
        raise FundamentalsError("split lookup returned no history")
    column = history["Stock Splits"]
    return [(stamp.date(), float(ratio)) for stamp, ratio in column[column != 0].items()]


def _fetch_from_yahoo(symbol: str) -> list[Split]:
    try:
        import yfinance as yf

        history = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=True)
    except Exception as exc:  # yfinance raises many unrelated types; none may escape raw
        raise FundamentalsError(f"split lookup failed for {symbol}: {exc}") from exc
    try:
        return split_rows(history)
    except FundamentalsError as exc:
        raise FundamentalsError(f"{exc} for {symbol}") from exc


class SplitHistory:
    """Split history per symbol, fetched lazily and cached in one JSON file."""

    def __init__(
        self,
        cache_file: Path,
        *,
        fetch: Callable[[str], list[Split]] | None = None,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._cache_file = cache_file
        self._fetch = fetch or _fetch_from_yahoo
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._entries: dict[str, dict[str, Any]] | None = None

    def splits(self, symbol: str, *, as_of: datetime, max_age_days: int) -> list[Split]:
        """`symbol`'s splits, oldest first (empty when it never split).

        Raises `FundamentalsError` when the lookup fails and there is no cached copy. A stale copy
        that cannot be refreshed is used, with a warning.
        """
        entries = self._load()
        entry = entries.get(symbol)
        if entry is not None and not is_stale(datetime.fromisoformat(entry["fetched_at"]), as_of, max_age_days):
            return _decode(entry)
        try:
            fetched = sorted(self._fetch(symbol))
        except FundamentalsError as exc:
            if entry is None:
                raise
            logger.log_warning(f"split history for {symbol} could not be refreshed, using the copy fetched {entry['fetched_at']}: {exc}")
            return _decode(entry)
        entries[symbol] = {"fetched_at": self._wall_clock().isoformat(), "splits": [[split_date.isoformat(), ratio] for split_date, ratio in fetched]}
        self._save(entries)
        return fetched

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._entries is None:
            try:
                loaded = json.loads(self._cache_file.read_text(encoding="utf-8"))
            except OSError, ValueError:
                loaded = {}  # no file yet, or one truncated by an interrupted write
            self._entries = loaded if isinstance(loaded, dict) else {}
        return self._entries

    def _save(self, entries: dict[str, dict[str, Any]]) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._cache_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries), encoding="utf-8")
        os.replace(temporary, self._cache_file)


def _decode(entry: dict[str, Any]) -> list[Split]:
    return [(date.fromisoformat(split_date), float(ratio)) for split_date, ratio in entry["splits"]]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals/test_splits.py -v && uv run ruff check`
Expected: all pass, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/trading_agent_framework/fundamentals/freshness.py src/trading_agent_framework/fundamentals/splits.py tests/fundamentals/test_splits.py
git commit -m "feat: split-aware share counts and the cache freshness rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `AnnualFiguresStore`

**Files:**
- Create: `src/trading_agent_framework/fundamentals/annual_store.py`
- Test: `tests/fundamentals/test_annual_store.py`

**Interfaces:**
- Consumes: `SecEdgarClient.ticker_to_cik`, `.fetch_company_facts_payload`, `.fetch_submissions_payload` and `FundamentalsNotFoundError` (Task 1); `sec.annual_figures`, `sec.parse_sic` (Task 2); `freshness.is_stale` (Task 5).
- Produces `annual_store.AnnualFiguresStore(client: SecEdgarClient, cache_dir: Path, *, wall_clock: Callable[[], datetime] | None = None)` with:
  - `cik(symbol: str) -> str | None`: `None` for a ticker SEC does not know.
  - `figures(symbol: str, *, as_of: datetime, max_age_days: int) -> dict[str, Any] | None`: the reduced record (`cik`, `fetched_at`, `status: "ok"`, `flows`, `balances`, `shares`, and `sic` once looked up), or `None` when the company is absent. Raises `FundamentalsError` on a transport failure with no cached copy.
  - `sic(symbol: str, *, as_of: datetime, max_age_days: int) -> int | None`: fetched once, saved in the record. Raises `FundamentalsError` on a transport failure.

- [ ] **Step 1: Write the failing tests**

Create `tests/fundamentals/test_annual_store.py`:

```python
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from trading_agent_framework.fundamentals.annual_store import AnnualFiguresStore
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils.errors import FundamentalsError

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL"}}
FACTS = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"val": 100, "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-15", "form": "10-K"}]}}}}}
FETCHED = datetime(2026, 10, 2, tzinfo=UTC)
LATER = datetime(2026, 11, 15, tzinfo=UTC)  # more than 30 days after FETCHED


class FakeSec:
    """A fake SEC: records every request and answers each endpoint with a settable status."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.facts_status = 200
        self.submissions_status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "/companyfacts/" in url:
            return httpx.Response(200, json=FACTS) if self.facts_status == 200 else httpx.Response(self.facts_status)
        if "/submissions/" in url:
            return httpx.Response(200, json={"sic": "3571"}) if self.submissions_status == 200 else httpx.Response(self.submissions_status)
        return httpx.Response(404)

    def count(self, fragment: str) -> int:
        return sum(fragment in url for url in self.requests)


def _store(tmp_path: Path, sec: FakeSec) -> AnnualFiguresStore:
    client = SecEdgarClient("TestApp test@example.com", tmp_path / "sec", min_request_interval_seconds=0.0, transport=httpx.MockTransport(sec))
    return AnnualFiguresStore(client, tmp_path / "sec" / "annual", wall_clock=lambda: FETCHED)


def _figures(store: AnnualFiguresStore, symbol: str = "AAPL", as_of: datetime = FETCHED) -> dict[str, object] | None:
    return store.figures(symbol, as_of=as_of, max_age_days=30)


def _sic(store: AnnualFiguresStore, as_of: datetime = FETCHED) -> int | None:
    return store.sic("AAPL", as_of=as_of, max_age_days=30)


def _record_file(tmp_path: Path) -> Path:
    return tmp_path / "sec" / "annual" / "CIK0000320193.json"


def test_the_first_call_fetches_and_writes_the_reduced_record_only(tmp_path: Path) -> None:
    sec = FakeSec()

    record = _figures(_store(tmp_path, sec))

    assert record is not None
    assert record["status"] == "ok"
    assert record["cik"] == "0000320193"
    assert record["fetched_at"] == FETCHED.isoformat()
    assert [row["value"] for row in record["flows"]] == [100]  # ty: ignore[not-iterable]
    assert json.loads(_record_file(tmp_path).read_text(encoding="utf-8")) == record
    assert not (tmp_path / "sec" / "companyfacts").exists()  # the 4 MB raw payload is not kept


def test_a_second_call_makes_no_request_in_memory_or_from_disk(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    _figures(store)
    _figures(store)
    _figures(_store(tmp_path, sec))  # a new process reads the file

    assert sec.count("/companyfacts/") == 1


def test_a_record_fetched_today_serves_every_past_date(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    _figures(store, as_of=datetime(2024, 1, 2, tzinfo=UTC))
    _figures(store, as_of=datetime(2025, 6, 2, tzinfo=UTC))

    assert sec.count("/companyfacts/") == 1


def test_a_stale_record_is_refetched(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    _figures(store)
    _figures(store, as_of=LATER)

    assert sec.count("/companyfacts/") == 2


def test_a_404_is_cached_as_an_absent_company(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.facts_status = 404
    store = _store(tmp_path, sec)

    assert _figures(store) is None
    assert _figures(store) is None
    assert sec.count("/companyfacts/") == 1
    assert json.loads(_record_file(tmp_path).read_text(encoding="utf-8"))["status"] == "absent"


def test_a_transport_error_raises_and_is_not_cached(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.facts_status = 500
    store = _store(tmp_path, sec)

    with pytest.raises(FundamentalsError):
        _figures(store)
    assert not _record_file(tmp_path).exists()

    sec.facts_status = 200
    assert _figures(store) is not None


def test_a_stale_record_survives_a_failed_refetch_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)
    _figures(store)
    sec.facts_status = 500

    with caplog.at_level(logging.WARNING):
        record = _figures(store, as_of=LATER)

    assert record is not None
    assert record["status"] == "ok"
    assert "0000320193" in caplog.text


def test_a_corrupt_record_file_is_refetched(tmp_path: Path) -> None:
    sec = FakeSec()
    _record_file(tmp_path).parent.mkdir(parents=True)
    _record_file(tmp_path).write_text("{not json", encoding="utf-8")

    record = _figures(_store(tmp_path, sec))

    assert record is not None
    assert sec.count("/companyfacts/") == 1


def test_a_ticker_sec_does_not_know_is_absent_without_a_request_or_a_file(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    assert store.cik("ZZZZ") is None
    assert _figures(store, "ZZZZ") is None
    assert sec.count("/companyfacts/") == 0
    assert not (tmp_path / "sec" / "annual").exists()


def test_the_ticker_map_is_requested_once(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    assert store.cik("AAPL") == "0000320193"
    assert store.cik("AAPL") == "0000320193"
    assert sec.count("company_tickers.json") == 1


def test_the_sic_code_is_fetched_once_and_saved_in_the_record(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    assert _sic(store) == 3571
    assert _sic(store) == 3571
    assert _sic(_store(tmp_path, sec)) == 3571

    assert sec.count("/submissions/") == 1
    assert json.loads(_record_file(tmp_path).read_text(encoding="utf-8"))["sic"] == 3571
    assert not (tmp_path / "sec" / "submissions").exists()


def test_the_sic_code_survives_a_refresh_of_the_record(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)
    _sic(store)

    assert _sic(store, as_of=LATER) == 3571
    assert sec.count("/companyfacts/") == 2
    assert sec.count("/submissions/") == 1


def test_a_company_with_no_submissions_has_no_sic_and_is_not_asked_again(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.submissions_status = 404
    store = _store(tmp_path, sec)

    assert _sic(store) is None
    assert _sic(store) is None
    assert sec.count("/submissions/") == 1


def test_a_transport_error_on_the_sic_lookup_raises_and_is_retried(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.submissions_status = 500
    store = _store(tmp_path, sec)

    with pytest.raises(FundamentalsError):
        _sic(store)

    sec.submissions_status = 200
    assert _sic(store) == 3571


def test_the_sic_of_an_absent_company_is_none_without_a_request(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.facts_status = 404
    store = _store(tmp_path, sec)

    assert _sic(store) is None
    assert sec.count("/submissions/") == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_annual_store.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.fundamentals.annual_store'`.

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/fundamentals/annual_store.py`:

```python
"""`AnnualFiguresStore`: each company's reduced annual SEC figures, fetched lazily and cached on disk.

The I/O side of the quality screen. A company-facts payload is about 4 MB; the store fetches it
uncached, keeps only `sec.annual_figures(...)` of it (a few KB) in `<cache_dir>/CIK<cik>.json`, and
holds every record it has read in memory for the life of the process.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.utils.errors import FundamentalsError, FundamentalsNotFoundError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "AnnualFiguresStore")

_EMPTY_FIGURES: dict[str, list[dict[str, Any]]] = {"flows": [], "balances": [], "shares": []}


class AnnualFiguresStore:
    def __init__(self, client: SecEdgarClient, cache_dir: Path, *, wall_clock: Callable[[], datetime] | None = None) -> None:
        self._client = client
        self._cache_dir = cache_dir
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._ciks: dict[str, str | None] = {}
        self._records: dict[str, dict[str, Any]] = {}

    def cik(self, symbol: str) -> str | None:
        """`symbol`'s CIK, or `None` for a ticker SEC does not know. Two share classes share one CIK."""
        if symbol not in self._ciks:
            # Kept in memory: the client re-reads and re-parses its 900 KB ticker map on every call.
            try:
                self._ciks[symbol] = self._client.ticker_to_cik(symbol)
            except FundamentalsNotFoundError:
                self._ciks[symbol] = None
        return self._ciks[symbol]

    def figures(self, symbol: str, *, as_of: datetime, max_age_days: int) -> dict[str, Any] | None:
        """The reduced record, or `None` when SEC has no such company.

        Raises `FundamentalsError` when the fetch fails and there is no cached copy. A stale copy that
        cannot be refreshed is used, with a warning.
        """
        record = self._record(symbol, as_of, max_age_days)
        return None if record is None or record["status"] == "absent" else record

    def sic(self, symbol: str, *, as_of: datetime, max_age_days: int) -> int | None:
        """The company's SIC industry code, fetched once and saved in its record."""
        record = self._record(symbol, as_of, max_age_days)
        if record is None or record["status"] == "absent":
            return None
        if "sic" not in record:
            try:
                code = sec.parse_sic(self._client.fetch_submissions_payload(record["cik"]))
            except FundamentalsNotFoundError:
                code = None
            self._write({**record, "sic": code})
        return self._records[record["cik"]]["sic"]

    def _record(self, symbol: str, as_of: datetime, max_age_days: int) -> dict[str, Any] | None:
        cik = self.cik(symbol)
        if cik is None:
            return None
        record = self._records.get(cik) or self._read(cik)
        if record is not None and not is_stale(datetime.fromisoformat(record["fetched_at"]), as_of, max_age_days):
            self._records[cik] = record
            return record
        base = {"cik": cik, "fetched_at": self._wall_clock().isoformat()}
        try:
            fresh = {**base, "status": "ok", **sec.annual_figures(self._client.fetch_company_facts_payload(cik))}
        except FundamentalsNotFoundError:
            fresh = {**base, "status": "absent", **_EMPTY_FIGURES}
        except FundamentalsError as exc:
            if record is None:
                raise
            logger.log_warning(f"annual figures for {symbol} (CIK {cik}) could not be refreshed, using the copy fetched {record['fetched_at']}: {exc}")
            self._records[cik] = record
            return record
        if record is not None and "sic" in record:
            fresh["sic"] = record["sic"]  # an industry code does not move with a new filing
        self._write(fresh)
        return fresh

    def _path(self, cik: str) -> Path:
        return self._cache_dir / f"CIK{cik}.json"

    def _read(self, cik: str) -> dict[str, Any] | None:
        try:
            record = json.loads(self._path(cik).read_text(encoding="utf-8"))
        except OSError, ValueError:
            return None  # no file yet, or one truncated by an interrupted write: a cache miss
        if not isinstance(record, dict) or "fetched_at" not in record or "status" not in record:
            return None
        return record

    def _write(self, record: dict[str, Any]) -> None:
        path = self._path(record["cik"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temporary, path)
        self._records[record["cik"]] = record
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals/test_annual_store.py -v && uv run ruff check`
Expected: all pass, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/fundamentals/annual_store.py tests/fundamentals/test_annual_store.py
git commit -m "feat: AnnualFiguresStore, the lazy reduced cache of SEC annual figures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `QualityScreen` wiring

**Files:**
- Create: `src/trading_agent_framework/fundamentals/screen.py`
- Modify: `src/trading_agent_framework/fundamentals/__init__.py`
- Test: `tests/fundamentals/test_screen.py`

**Interfaces:**
- Consumes:
  - `quality.ScreenParams`, `Survivor`, `Priced`, `ScreenResult`, `assess(symbol, figures, *, as_of, params)`, `sector_excluded(sic, params)`, `rank(priced, params)` (Tasks 3 and 4).
  - `splits.Split`, `splits.restate_shares(shares, counted_on, splits)`, `SplitHistory.splits(symbol, *, as_of, max_age_days)` (Task 5).
  - `AnnualFiguresStore.cik(symbol)`, `.figures(symbol, *, as_of, max_age_days)`, `.sic(symbol, *, as_of, max_age_days)` (Task 6).
  - `tests.fundamentals.annual_fixtures.healthy_figures` (Task 3).
- Produces (in `fundamentals/screen.py`, re-exported from `trading_agent_framework.fundamentals`):
  - `QualityScreen(store: FiguresSource, splits: SplitSource, *, params: ScreenParams | None = None)`.
  - `QualityScreen.run(symbols: Sequence[str], *, as_of: datetime, price_of: Callable[[str], Decimal | None]) -> ScreenResult`. Raises `ValueError` for a naive `as_of`, `FundamentalsError` when more than `params.max_fetch_failure_ratio` of the symbols failed on transport errors.
  - `build_quality_screen(project_root: Path, *, params: ScreenParams | None = None) -> QualityScreen`: real store and split history under `<project_root>/cache/`; raises `ConfigurationError` without `SEC_EDGAR_USER_AGENT`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fundamentals/test_screen.py`:

```python
from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from tests.fundamentals.annual_fixtures import healthy_figures
from trading_agent_framework.fundamentals import Candidate, QualityScreen, ScreenParams, ScreenResult, build_quality_screen
from trading_agent_framework.fundamentals.splits import Split
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, FundamentalsError

AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


class FakeStore:
    """Annual figures and SIC codes from dicts; symbols in `failing` raise like a transport error."""

    def __init__(
        self,
        records: dict[str, dict[str, object] | None],
        *,
        ciks: dict[str, str] | None = None,
        sics: dict[str, int] | None = None,
        failing: Collection[str] = (),
        sic_failing: Collection[str] = (),
    ) -> None:
        self.records = records
        self.ciks = ciks or {}
        self.sics = sics or {}
        self.failing = failing
        self.sic_failing = sic_failing

    def cik(self, symbol: str) -> str | None:
        return self.ciks.get(symbol, f"cik-{symbol}")

    def figures(self, symbol: str, *, as_of: datetime, max_age_days: int) -> dict[str, object] | None:
        if symbol in self.failing:
            raise FundamentalsError(f"network down for {symbol}")
        return self.records.get(symbol)

    def sic(self, symbol: str, *, as_of: datetime, max_age_days: int) -> int | None:
        if symbol in self.sic_failing:
            raise FundamentalsError(f"network down for {symbol}")
        return self.sics.get(symbol)


class FakeSplits:
    def __init__(self, splits: dict[str, list[Split]] | None = None, *, failing: Collection[str] = ()) -> None:
        self._splits = splits or {}
        self.failing = failing
        self.calls: list[str] = []

    def splits(self, symbol: str, *, as_of: datetime, max_age_days: int) -> list[Split]:
        self.calls.append(symbol)
        if symbol in self.failing:
            raise FundamentalsError(f"yahoo is down for {symbol}")
        return self._splits.get(symbol, [])


class Prices:
    """A `price_of` that records which symbols it was asked about."""

    def __init__(self, prices: dict[str, Decimal | None] | None = None, *, default: Decimal | None = Decimal("10")) -> None:
        self._prices = prices or {}
        self._default = default
        self.calls: list[str] = []

    def __call__(self, symbol: str) -> Decimal | None:
        self.calls.append(symbol)
        return self._prices.get(symbol, self._default)


def _run(
    store: FakeStore,
    symbols: list[str],
    *,
    splits: FakeSplits | None = None,
    price_of: Callable[[str], Decimal | None] | None = None,
    params: ScreenParams | None = None,
) -> ScreenResult:
    screen = QualityScreen(store, splits or FakeSplits(), params=params)
    return screen.run(symbols, as_of=AS_OF, price_of=price_of or Prices())


def _symbols(result: ScreenResult) -> list[str]:
    return [candidate.symbol for candidate in result.candidates]


def test_a_healthy_company_becomes_a_candidate_with_its_market_cap() -> None:
    result = _run(FakeStore({"AAA": healthy_figures()}, sics={"AAA": 5812}), ["AAA"])

    (candidate,) = result.candidates
    assert isinstance(candidate, Candidate)
    assert (candidate.symbol, candidate.rank, candidate.sic) == ("AAA", 1, 5812)
    assert candidate.market_cap == Decimal("10000")  # 1000 shares x 10
    assert candidate.fcf_yield == pytest.approx(28 / 10000)
    assert result.rejections == {}


def test_candidates_are_ranked_against_each_other() -> None:
    store = FakeStore({"AAA": healthy_figures(), "BBB": healthy_figures()})

    result = _run(store, ["AAA", "BBB"], price_of=Prices({"AAA": Decimal("20"), "BBB": Decimal("10")}))

    assert _symbols(result) == ["BBB", "AAA"]  # same company figures, BBB is cheaper


def test_a_split_after_the_share_count_restates_the_market_cap() -> None:
    # healthy_figures counts 1000 shares on 2026-01-31; a 2-for-1 follows.
    splits = FakeSplits({"AAA": [(date(2026, 3, 2), 2.0)]})

    result = _run(FakeStore({"AAA": healthy_figures()}), ["AAA"], splits=splits)

    assert result.candidates[0].market_cap == Decimal("20000")


def test_a_numeric_gate_failure_is_reported_with_its_reason() -> None:
    store = FakeStore({"AAA": healthy_figures(), "LOSS": healthy_figures(operating_income=(20, 22, -1, 26, 28)), "GONE": None})

    result = _run(store, ["AAA", "LOSS", "GONE"])

    assert _symbols(result) == ["AAA"]
    assert result.rejections == {"LOSS": "operating_loss", "GONE": "no_data"}


def test_an_excluded_sector_is_rejected() -> None:
    result = _run(FakeStore({"BANK": healthy_figures()}, sics={"BANK": 6021}), ["BANK"])

    assert result.rejections == {"BANK": "excluded_sector"}


def test_two_listings_of_one_company_take_one_slot() -> None:
    store = FakeStore({"GOOGL": healthy_figures(), "GOOG": healthy_figures()}, ciks={"GOOGL": "0001652044", "GOOG": "0001652044"})

    result = _run(store, ["GOOGL", "GOOG"])

    assert _symbols(result) == ["GOOGL"]
    assert result.rejections == {"GOOG": "duplicate_listing"}


def test_the_second_listing_gets_its_chance_when_the_first_has_no_price() -> None:
    store = FakeStore({"GOOGL": healthy_figures(), "GOOG": healthy_figures()}, ciks={"GOOGL": "0001652044", "GOOG": "0001652044"})

    result = _run(store, ["GOOGL", "GOOG"], price_of=Prices({"GOOGL": None}))

    assert _symbols(result) == ["GOOG"]
    assert result.rejections == {"GOOGL": "no_price"}


def test_prices_and_splits_are_asked_only_for_survivors_of_the_earlier_gates() -> None:
    store = FakeStore(
        {"AAA": healthy_figures(), "LOSS": healthy_figures(operating_income=(20, 22, -1, 26, 28)), "BANK": healthy_figures(), "AAA2": healthy_figures()},
        ciks={"AAA": "1", "AAA2": "1"},
        sics={"BANK": 6021},
    )
    prices, splits = Prices(), FakeSplits()

    _run(store, ["AAA", "LOSS", "GONE", "BANK", "AAA2"], splits=splits, price_of=prices)

    assert prices.calls == ["AAA"]
    assert splits.calls == ["AAA"]


@pytest.mark.parametrize("price", [None, Decimal("0"), Decimal("-1")])
def test_a_missing_or_non_positive_price_is_no_price(price: Decimal | None) -> None:
    result = _run(FakeStore({"AAA": healthy_figures()}), ["AAA"], price_of=Prices({"AAA": price}))

    assert result.rejections == {"AAA": "no_price"}


def test_a_price_lookup_that_raises_a_framework_error_is_no_price_and_the_run_continues() -> None:
    def price_of(symbol: str) -> Decimal | None:
        if symbol == "AAA":
            raise BrokerError("no quote")
        return Decimal("10")

    result = _run(FakeStore({"AAA": healthy_figures(), "BBB": healthy_figures()}), ["AAA", "BBB"], price_of=price_of)

    assert _symbols(result) == ["BBB"]
    assert result.rejections == {"AAA": "no_price"}


@pytest.mark.parametrize("shares", [None, 0])
def test_a_missing_or_zero_share_count_is_no_price(shares: int | None) -> None:
    result = _run(FakeStore({"AAA": healthy_figures(shares=shares)}), ["AAA"])

    assert result.rejections == {"AAA": "no_price"}


def test_a_failed_split_lookup_is_no_split_data() -> None:
    result = _run(FakeStore({"AAA": healthy_figures()}), ["AAA"], splits=FakeSplits(failing={"AAA"}))

    assert result.rejections == {"AAA": "no_split_data"}


def test_a_transport_failure_rejects_the_symbol_as_no_data() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    store = FakeStore({name: healthy_figures() for name in names}, failing={"EEE"})

    result = _run(store, names)  # 1 of 5 = 20%, not above the 20% limit

    assert result.rejections == {"EEE": "no_data"}
    assert len(result.candidates) == 4


def test_a_failed_sic_lookup_counts_as_a_transport_failure() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    store = FakeStore({name: healthy_figures() for name in names}, sic_failing={"EEE"})

    assert _run(store, names).rejections == {"EEE": "no_data"}


def test_a_hollow_screen_raises_instead_of_ranking_what_happened_to_download() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    store = FakeStore({name: healthy_figures() for name in names}, failing={"DDD", "EEE"})

    with pytest.raises(FundamentalsError, match="2 of 5"):
        _run(store, names)


def test_a_naive_as_of_is_refused_up_front() -> None:
    screen = QualityScreen(FakeStore({"AAA": healthy_figures()}), FakeSplits())

    with pytest.raises(ValueError, match="timezone-aware"):
        screen.run(["AAA"], as_of=datetime(2026, 6, 1), price_of=Prices())


def test_an_empty_symbol_list_gives_an_empty_result() -> None:
    assert _run(FakeStore({}), []) == ScreenResult(candidates=[], rejections={})


def test_a_list_where_everything_is_rejected_gives_no_candidates() -> None:
    result = _run(FakeStore({}), ["GONE", "ALSO"])

    assert result.candidates == []
    assert result.rejections == {"GONE": "no_data", "ALSO": "no_data"}


def test_a_symbol_listed_twice_is_processed_once() -> None:
    prices = Prices()

    result = _run(FakeStore({"AAA": healthy_figures()}), ["AAA", "AAA"], price_of=prices)

    assert _symbols(result) == ["AAA"]
    assert prices.calls == ["AAA"]
    assert result.rejections == {}


def test_top_n_limits_the_candidates() -> None:
    names = ["AAA", "BBB", "CCC"]

    result = _run(FakeStore({name: healthy_figures() for name in names}), names, params=ScreenParams(top_n=2))

    assert len(result.candidates) == 2


def test_each_run_logs_one_summary_line(caplog: pytest.LogCaptureFixture) -> None:
    store = FakeStore({"AAA": healthy_figures(), "LOSS": healthy_figures(operating_income=(20, 22, -1, 26, 28))})

    with caplog.at_level(logging.INFO):
        _run(store, ["AAA", "LOSS", "GONE"])

    assert "screened 3 symbols" in caplog.text
    assert "1 candidates" in caplog.text
    assert "no_data=1" in caplog.text
    assert "operating_loss=1" in caplog.text


def test_build_quality_screen_needs_a_sec_user_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)

    with pytest.raises(ConfigurationError, match="SEC_EDGAR_USER_AGENT"):
        build_quality_screen(tmp_path)


def test_build_quality_screen_builds_a_screen_under_the_project_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")

    screen = build_quality_screen(tmp_path, params=ScreenParams(top_n=3))

    assert isinstance(screen, QualityScreen)
    assert screen.params.top_n == 3
    assert (tmp_path / "cache" / "sec").is_dir()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_screen.py -v`
Expected: collection error, `ImportError: cannot import name 'Candidate' from 'trading_agent_framework.fundamentals'`.

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/fundamentals/screen.py`:

```python
"""`QualityScreen`: wires the annual-figures store and the split history to the pure quality gates.

Knows no `Strategy`, broker or LLM: the caller passes the date (`as_of`, its own clock) and a price
lookup. Every per-symbol problem becomes a rejection with a named reason; only a screen hollowed out
by transport failures raises.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from trading_agent_framework.fundamentals.annual_store import AnnualFiguresStore
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.fundamentals.quality import Priced, ScreenParams, ScreenResult, assess, rank, sector_excluded
from trading_agent_framework.fundamentals.splits import Split, SplitHistory, restate_shares
from trading_agent_framework.utils.errors import FundamentalsError, TradingFrameworkError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "QualityScreen")


class FiguresSource(Protocol):
    """What the screen needs from `AnnualFiguresStore`."""

    def cik(self, symbol: str) -> str | None: ...

    def figures(self, symbol: str, *, as_of: datetime, max_age_days: int) -> dict[str, Any] | None: ...

    def sic(self, symbol: str, *, as_of: datetime, max_age_days: int) -> int | None: ...


class SplitSource(Protocol):
    """What the screen needs from `SplitHistory`."""

    def splits(self, symbol: str, *, as_of: datetime, max_age_days: int) -> list[Split]: ...


class QualityScreen:
    def __init__(self, store: FiguresSource, splits: SplitSource, *, params: ScreenParams | None = None) -> None:
        self._store = store
        self._splits = splits
        self.params = params or ScreenParams()

    def run(self, symbols: Sequence[str], *, as_of: datetime, price_of: Callable[[str], Decimal | None]) -> ScreenResult:
        """Rank the companies among `symbols` that pass every gate on `as_of`, best first.

        `price_of` is called only for companies that passed every gate before the price gate.
        Raises `FundamentalsError` when more than `params.max_fetch_failure_ratio` of the symbols
        could not be fetched: a ranking of what happened to download is worse than none.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        params = self.params
        max_age_days = params.max_age_days
        unique = list(dict.fromkeys(symbols))
        rejections: dict[str, str] = {}
        priced: list[Priced] = []
        accepted_ciks: set[str | None] = set()
        transport_failures = 0

        for symbol in unique:
            try:
                outcome = assess(symbol, self._store.figures(symbol, as_of=as_of, max_age_days=max_age_days), as_of=as_of, params=params)
                if isinstance(outcome, str):
                    rejections[symbol] = outcome
                    continue
                sic = self._store.sic(symbol, as_of=as_of, max_age_days=max_age_days)
            except FundamentalsError as exc:
                transport_failures += 1
                rejections[symbol] = "no_data"
                logger.log_debug(f"{symbol}: {exc}")
                continue
            if sector_excluded(sic, params):
                rejections[symbol] = "excluded_sector"
                continue
            cik = self._store.cik(symbol)
            if cik in accepted_ciks:
                rejections[symbol] = "duplicate_listing"
                continue
            price = _price(price_of, symbol)
            if price is None or not outcome.shares or outcome.counted_on is None:
                rejections[symbol] = "no_price"
                continue
            try:
                splits = self._splits.splits(symbol, as_of=as_of, max_age_days=max_age_days)
            except FundamentalsError as exc:
                rejections[symbol] = "no_split_data"
                logger.log_debug(f"{symbol}: {exc}")
                continue
            accepted_ciks.add(cik)  # only now: a first listing with no price must not block the second
            priced.append(Priced(survivor=outcome, sic=sic, market_cap=restate_shares(outcome.shares, outcome.counted_on, splits) * price))

        if unique and transport_failures / len(unique) > params.max_fetch_failure_ratio:
            raise FundamentalsError(f"quality screen aborted: {transport_failures} of {len(unique)} symbols could not be fetched from SEC")

        candidates = rank(priced, params)
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(Counter(rejections.values()).items())) or "none"
        logger.log_info(f"screened {len(unique)} symbols as of {as_of.date()}: {len(candidates)} candidates from {len(priced)} survivors; rejected: {reasons}")
        return ScreenResult(candidates=candidates, rejections=rejections)


def _price(price_of: Callable[[str], Decimal | None], symbol: str) -> Decimal | None:
    """`price_of(symbol)` when it is a usable price; a failed or non-positive quote is no price."""
    try:
        price = price_of(symbol)
    except TradingFrameworkError as exc:
        logger.log_debug(f"{symbol}: no price: {exc}")
        return None
    return price if price is not None and price > 0 else None


def build_quality_screen(project_root: Path, *, params: ScreenParams | None = None) -> QualityScreen:
    """The real screen: SEC annual figures and Yahoo split history, cached under `<project_root>/cache/`.

    Raises `ConfigurationError` when `SEC_EDGAR_USER_AGENT` is missing or blank.
    """
    sec_cache = project_root / "cache" / "sec"
    client = SecEdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT", ""), sec_cache)
    store = AnnualFiguresStore(client, sec_cache / "annual")
    return QualityScreen(store, SplitHistory(project_root / "cache" / "splits.json"), params=params)
```

`SecEdgarClient.__init__` raises `ConfigurationError` on a blank user agent before it creates its cache directory, which is what `test_build_quality_screen_needs_a_sec_user_agent` relies on.

Replace `src/trading_agent_framework/fundamentals/__init__.py` with:

```python
"""SEC EDGAR fundamentals: `sec.py` (pure translation), `edgar_client.py` (cached I/O), and the quality
screen (`quality.py` pure gates and score, `annual_store.py` and `splits.py` I/O, `screen.py` wiring)."""

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.fundamentals.quality import Candidate, ScreenParams, ScreenResult
from trading_agent_framework.fundamentals.screen import QualityScreen, build_quality_screen
from trading_agent_framework.utils import get_version

__version__ = get_version("trading_agent_framework")

__all__ = ["Candidate", "QualityScreen", "ScreenParams", "ScreenResult", "SecEdgarClient", "build_quality_screen"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals -v && uv run ruff check`
Expected: all pass, no lint errors.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass. This catches an import cycle or a test elsewhere that depended on the old `fundamentals/__init__.py`.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/fundamentals/screen.py src/trading_agent_framework/fundamentals/__init__.py tests/fundamentals/test_screen.py
git commit -m "feat: QualityScreen wiring the store, splits and the pure gates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Smoke script and documentation

**Files:**
- Create: `scripts/tests/smoke_quality_screen.py`
- Modify: `CLAUDE.md` (Commands block; the `fundamentals/` architecture bullet; one new gotcha)

**Interfaces:**
- Consumes: `build_quality_screen(project_root)`, `QualityScreen.run(symbols, *, as_of, price_of)`, `ScreenResult`, `Candidate` (Task 7).
- Produces: a manual script; no code other tasks depend on.

- [ ] **Step 1: Write the smoke script**

Create `scripts/tests/smoke_quality_screen.py`:

```python
#!/usr/bin/env python3
"""Manual run of the fundamentals quality screen against real SEC EDGAR and Yahoo data.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_quality_screen.py

`SEC_EDGAR_USER_AGENT` comes from env/.env.alpaca.integration-tests, as in the other smoke scripts.
The script is read-only. The first run downloads one SEC company-facts payload per symbol (about
4 MB each) and writes reduced copies under cache/sec/annual/; a second run the same day makes no
SEC request and finishes in a few seconds. It fails only on things that must always hold: every
symbol is either a candidate or rejected with a reason, an unknown ticker is `no_data`, a bank is
rejected, and the two Alphabet listings never both become candidates. (The bank's reason is printed, not
asserted: the numeric gates run before the sector gate, and a bank usually fails those first.)
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.fundamentals import ScreenParams, build_quality_screen
from trading_agent_framework.utils.errors import TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

UNKNOWN_SYMBOL = "ZZZZZZ"
BANK_SYMBOL = "JPM"
SYMBOLS = [
    "GOOGL", "GOOG", "CMG", "HLT", "QSR", "UBER", "CP", "LOW", "MDLZ", "BKNG", "MSFT",
    "AAPL", "NVDA", "COST", "KO", "V", "NEE", "TSM", BANK_SYMBOL, UNKNOWN_SYMBOL,
]  # fmt: skip


class SmokeTestFailure(Exception):
    pass


def _last_close(symbol: str) -> Decimal | None:
    import yfinance as yf

    try:
        closes = yf.Ticker(symbol).history(period="5d")["Close"].dropna()
    except Exception as exc:  # a manual script: any Yahoo failure is just "no price" for that symbol
        print(f"  no price for {symbol}: {exc}")
        return None
    return None if closes.empty else Decimal(str(round(float(closes.iloc[-1]), 4)))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(ENV_FILE)
    try:
        screen = build_quality_screen(PROJECT_ROOT, params=ScreenParams(top_n=len(SYMBOLS)))
        started = time.perf_counter()
        result = screen.run(SYMBOLS, as_of=datetime.now(UTC), price_of=_last_close)
        elapsed = time.perf_counter() - started
    except TradingFrameworkError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"\nCandidates ({len(result.candidates)}), screened in {elapsed:.1f}s:")
    print(f"{'#':>2} {'symbol':<6} {'score':>5} {'fcf yield':>9} {'fcf margin':>10} {'margin sd':>9} {'growth':>7} {'debt x':>6} {'mkt cap $B':>10}  fiscal year / filed")
    for c in result.candidates:
        debt = f"{c.net_debt_to_operating_income:6.2f}" + ("" if c.debt_reported else "*")
        print(
            f"{c.rank:>2} {c.symbol:<6} {c.score:5.2f} {c.fcf_yield:9.2%} {c.fcf_margin:10.2%} {c.operating_margin_stdev:9.2%} "
            f"{c.revenue_growth:7.2%} {debt} {float(c.market_cap) / 1e9:10.1f}  {c.fiscal_year_end} / {c.filed}"
        )
    print("  (* = no debt figure reported: counted as zero)")
    print(f"\nRejections ({len(result.rejections)}):")
    for symbol, reason in sorted(result.rejections.items()):
        print(f"  {symbol:<6} {reason}")

    candidates = {c.symbol for c in result.candidates}
    try:
        _check(candidates | set(result.rejections) == set(SYMBOLS), "a symbol is neither a candidate nor rejected")
        _check(not candidates & set(result.rejections), "a symbol is both a candidate and rejected")
        _check(result.rejections.get(UNKNOWN_SYMBOL) == "no_data", f"{UNKNOWN_SYMBOL} should be no_data")
        _check(BANK_SYMBOL in result.rejections, f"{BANK_SYMBOL} (a bank) should be rejected")
        _check(not {"GOOGL", "GOOG"} <= candidates, "both Alphabet listings became candidates")
    except SmokeTestFailure as exc:
        print(f"\nFAILED: {exc}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the smoke script twice**

Run: `uv run python scripts/tests/smoke_quality_screen.py`
Expected: a table of candidates, a list of rejections with reasons, and `OK`. The first run takes roughly a minute (about 20 SEC downloads).

Run it again.
Expected: the same output, in a few seconds (only the Yahoo price lookups remain), and `ls cache/sec/annual | wc -l` shows about 19 small files while `cache/sec/companyfacts/` has gained none.

If a check fails, do not weaken the check: read the rejection reasons printed above it and fix the code (the most likely causes are a tag missing from `ANNUAL_FLOW_TAGS` and a share-count selection problem). `TSM` is expected to be rejected (`no_data` or `insufficient_history`: it files IFRS figures on a 20-F), and `JPM` usually fails a numeric gate before the sector gate is reached.

- [ ] **Step 3: Update `CLAUDE.md`**

In the Commands block, add after the `smoke_alpaca_data.py` line:

```
uv run python scripts/tests/smoke_quality_screen.py    # manual smoke test: fundamentals quality screen against real SEC + Yahoo (read-only)
```

Replace the `fundamentals/` architecture bullet (it starts with ``- `fundamentals/` -- SEC EDGAR client``) with:

```markdown
- `fundamentals/` -- SEC EDGAR client, trimmed and ported from lumibot's `SECFundamentals`: `sec.py` (**pure**: tag maps, as-of candidate filtering, statement-period matching, filings parsing, URL building, HTML stripping, and `annual_figures`, which reduces a 4 MB company-facts payload to the few KB the quality screen needs) and `edgar_client.py` (`SecEdgarClient`, the only module allowed to import `httpx` for SEC access -- cached to `<project_root>/cache/sec/`, rate-limited, requires `SEC_EDGAR_USER_AGENT`; `fetch_json` is its uncached path, and HTTP 404 raises `FundamentalsNotFoundError`). The **quality screen** (`build_quality_screen(project_root).run(symbols, as_of=, price_of=)`) ranks the companies that were simple, predictable, cash-generative, lightly indebted and reasonably priced on a date: `quality.py` (**pure**: `ScreenParams`, the numeric gates in `assess`, the percentile score in `rank`), `annual_store.py` (`AnnualFiguresStore`: lazy per-company fetch, reduced cache in `cache/sec/annual/`), `splits.py` (pure `restate_shares`; `SplitHistory`, the only module importing `yfinance` for splits, cached in `cache/splits.json`), `freshness.py` (the one cache rule), `screen.py` (wiring). It knows no `Strategy`, broker or LLM.
```

Add this bullet to "Key patterns / gotchas", after the "Research agent tools (news/macro/fundamentals) gate on ..." bullet:

```markdown
- **The quality screen's as-of rule is stricter than the fundamentals tools'.** `quality.assess` treats a row as known only when its `filed` date is strictly before `as_of`'s date (SEC gives no filing time, and annual reports are often filed after the close); the agent tools in `sec.py` use `filed <= as_of`. A fiscal year is keyed by its period END date, never by XBRL's `fy` (each 10-K repeats three years, all tagged with the filing's own `fy`). Its two caches share one freshness rule (`freshness.is_stale`): stale when `fetched_at < as_of - max_age_days`, with `as_of` the caller's clock -- so a backtest never refetches (a file fetched today is fresh for every past date) and paper/live refresh monthly, with no wall-clock value ever compared against data. Share counts are restated for every split dated after the count (`splits.restate_shares`), including splits after `as_of`: bar prices are already adjusted for those, so it undoes a price adjustment and leaks nothing.
```

- [ ] **Step 4: Final verification**

Run: `uv run pytest -q && uv run ruff check`
Expected: the full suite passes, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/tests/smoke_quality_screen.py CLAUDE.md
git commit -m "docs: quality screen smoke script and CLAUDE.md notes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
