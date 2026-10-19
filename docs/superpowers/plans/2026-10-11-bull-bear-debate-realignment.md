# bull_bear Debate Realignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bull_bear's judge decide, held-blind, which candidates are in the portfolio, remember and learn from its verdicts, and argue from correct SEC data shared with bill_ackman.

**Architecture:** The SEC layer (`fundamentals/sec.py`) becomes the single definition of total debt and cash flow, used by the agent tools and by bill_ackman's screen (whose output must not change). In bull_bear, the judge gives `in`/`out` on every candidate (at most 10 `in`), code builds the book from the `in` names in rank order (parking the rest in SHV below 5 names), and the state carries the judge's verdicts, a re-entry cooldown and a short verdict history that feeds outcome feedback back to the judge, which also gets two memory tools.

**Tech Stack:** Python 3.14, `uv`, pytest with hand-written fakes, ruff. No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-11-bull-bear-debate-realignment-design.md` (read it before Task 1).

## Global Constraints

- Work in the worktree `/home/yann/projets/trading-agent-framework-realign` (branch `feat/bull-bear-debate-realignment`); every command below runs from there.
- `uv` only (never pip); run tests with `uv run pytest`, lint with `uv run ruff check`.
- The automated suite never touches the network; tests use hand-written fakes (`tests/fakes.py` style), never `MagicMock`.
- `strategies/bull_bear/handoff.py` and `memory/tools.py` keep NO `from __future__ import annotations` (the agent layer reads real annotations); `agents/tools/fundamentals.py` has none either.
- No wall-clock time in a data tool: `as_of` is always `strategy.clock.now()`.
- bill_ackman's screen records must not change; `annual_store.SCHEMA_VERSION` stays 2.
- No change to cross_momentum, `strategies/common/scoring.py` or `strategies/common/rebalancer.py`.
- No agent ever gets an order tool.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (after a blank line).
- Defaults from the spec: `candidate_count` 15, `max_book_size` 10, `retention_rank` 35, `reentry_cooldown_reviews` 4, `feedback_reviews` 4, `judge_memory_budget` 6, `research_news` True, `research_tool_budget` 3, `note_max_chars` 500, weights [4%, 20%], cash buffer 2%, stale statement after 548 days.

## Review Focus

1. A 10-Q tags a 3-month and a year-to-date figure ending the same day: `prior_year` must pair a quarter with a quarter, never with nine months (Task 2, `test_a_quarter_is_never_compared_with_a_year_to_date_figure`).
2. A judge's previous verdict from months ago still requires `what_changed` to flip (Task 6, `test_a_judge_flipping_a_verdict_without_what_changed_is_refused_then_corrects_itself`).
3. A holding still in cooldown because its sell did not fill must be sold as a forced exit, not debated (Task 7, `test_a_holding_still_in_cooldown_is_sold_as_a_forced_exit`).
4. A stock in the feedback history that has no price now is dropped, never estimated (Task 8, `test_a_stock_without_a_price_now_is_left_out_and_never_estimated`).
5. A judge that puts nothing in completes the review and parks everything; it is not an abandonment (Task 5, `test_a_judge_putting_nothing_in_parks_everything`).

---

### Task 1: One total-debt and cash-flow definition in `fundamentals/sec.py`

**Files:**
- Modify: `src/trading_agent_framework/fundamentals/sec.py` (constants after `IFRS_BALANCE_SHEET_TAGS`, functions after `latest_fact`)
- Modify: `src/trading_agent_framework/strategies/bill_ackman/screen/annual_figures.py`
- Modify: `src/trading_agent_framework/fundamentals/CLAUDE.md`, `src/trading_agent_framework/strategies/bill_ackman/CLAUDE.md`
- Test: `tests/fundamentals/test_sec.py`

**Interfaces:**
- Produces: `sec.unit_rows(facts, tag, unit, forms) -> list[dict]`; `sec.TOTAL_DEBT_TAGS`, `sec.SUMMED_DEBT_TAGS`, `sec.COMBINED_DEBT_TAG`; `sec.total_debt_sources(gaap: dict, forms: frozenset[str]) -> list[list[dict]]` (rows `{"end", "value", "filed"}`); `sec.CASH_FLOW_TAGS: dict[str, list[str]]` with keys `operating_cash_flow`, `capex`.

- [ ] **Step 0: Install and record the screen's baseline output**

```bash
cd /home/yann/projets/trading-agent-framework-realign && uv sync
uv run python - <<'EOF'
import glob, json, os
from trading_agent_framework.strategies.bill_ackman.screen.annual_figures import annual_figures
out = {}
for path in sorted(glob.glob("/home/yann/projets/trading-agent-framework/cache/sec/companyfacts/*.json")):
    with open(path) as handle:
        out[os.path.basename(path)] = annual_figures(json.load(handle))
with open("/tmp/claude-1000/-home-yann-projets-trading-agent-framework/457a5b20-56a5-4926-aadb-d9c998ddf1fd/scratchpad/annual_baseline.json", "w") as handle:
    json.dump(out, handle, sort_keys=True)
print(len(out), "payloads reduced")
EOF
```

Expected: `183 payloads reduced` (the number of cached company-facts files; any count above 100 is fine, note it).

- [ ] **Step 1: Write the failing tests** (append to `tests/fundamentals/test_sec.py`)

```python
# --- total debt and cash flow: one definition for the agent tools and bill_ackman's screen ---------------------------

_ANNUAL = frozenset({"10-K", "10-K/A"})


def _usd(*rows: dict[str, object]) -> dict[str, object]:
    return {"units": {"USD": list(rows)}}


def test_total_debt_sources_follow_the_order_of_preference() -> None:
    year_end = {"end": "2025-12-31", "filed": "2026-02-01", "form": "10-K"}
    gaap = {
        "LongTermDebt": _usd({"val": 900, **year_end}),
        "LongTermDebtNoncurrent": _usd({"val": 700, **year_end}),
        "LongTermDebtCurrent": _usd({"val": 200, **year_end}),
        "DebtLongtermAndShorttermCombinedAmount": _usd({"val": 950, **year_end}),
    }

    sources = sec.total_debt_sources(gaap, _ANNUAL)

    assert [rows[0]["value"] if rows else None for rows in sources] == [900, None, 900, None, 950]
    assert sources[0][0] == {"end": "2025-12-31", "value": 900, "filed": "2026-02-01"}


def test_a_noncurrent_debt_without_its_current_part_counts_the_current_part_as_zero() -> None:
    gaap = {"LongTermDebtNoncurrent": _usd({"val": 700, "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K"})}

    assert sec.total_debt_sources(gaap, _ANNUAL)[2] == [{"end": "2025-12-31", "value": 700, "filed": "2026-02-01"}]


def test_total_debt_sources_read_only_the_forms_asked_for() -> None:
    gaap = {"LongTermDebt": _usd({"val": 900, "end": "2026-03-31", "filed": "2026-05-01", "form": "10-Q"})}

    assert sec.total_debt_sources(gaap, _ANNUAL)[0] == []
    assert sec.total_debt_sources(gaap, frozenset({"10-Q"}))[0] == [{"end": "2026-03-31", "value": 900, "filed": "2026-05-01"}]


def test_the_cash_flow_tags_are_shared_with_bill_ackmans_screen() -> None:
    from trading_agent_framework.strategies.bill_ackman.screen.annual_figures import ANNUAL_FLOW_TAGS

    assert ANNUAL_FLOW_TAGS["operating_cash_flow"] is sec.CASH_FLOW_TAGS["operating_cash_flow"]
    assert ANNUAL_FLOW_TAGS["capex"] is sec.CASH_FLOW_TAGS["capex"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/fundamentals/test_sec.py -q`
Expected: 4 failures, `AttributeError: module ... has no attribute 'total_debt_sources'` / `'CASH_FLOW_TAGS'`.

- [ ] **Step 3: Implement in `sec.py`**

Replace the comment above `IFRS_INCOME_STATEMENT_TAGS` (lines 44-46) with:

```python
# The same fields for filers of 20-F / 40-F reports, who tag under the `ifrs-full` taxonomy. bill_ackman's screen imports
# INCOME_STATEMENT_TAGS["revenue"] and ["operating_income"], BALANCE_SHEET_TAGS["cash"], CASH_FLOW_TAGS and
# `total_debt_sources`: changing those changes its records (bump its `annual_store.SCHEMA_VERSION`). Tag names checked
# against real payloads (UMC, Kinross, Equinox Gold, Gold Fields, Abivax, ...). `RevenueFromSaleOfGoods` is left out:
# it can be a component of revenue.
```

Add after `IFRS_BALANCE_SHEET_TAGS`:

```python
# Total debt, in order of preference: the total tags, then noncurrent + current sums, then the combined tag. Real filers
# use `LongTermDebt` (LOW), the lease-inclusive total (KO, MDLZ), `LongTermDebtNoncurrent` + `LongTermDebtCurrent` (QSR,
# NEE), or only the lease-inclusive pair (HLT). Known limitation: commercial paper and other short-term borrowings
# (`CommercialPaper`, `ShortTermBorrowings`) are not added, since filers disagree on whether the long-term tags include them.
TOTAL_DEBT_TAGS: tuple[str, ...] = ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities")
SUMMED_DEBT_TAGS: tuple[tuple[str, str], ...] = (
    ("LongTermDebtNoncurrent", "LongTermDebtCurrent"),
    ("LongTermDebtAndCapitalLeaseObligations", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
)
COMBINED_DEBT_TAG = "DebtLongtermAndShorttermCombinedAmount"

# Cash flow: the screen reads annual reports, the agent tools quarterly ones too.
CASH_FLOW_TAGS: dict[str, list[str]] = {
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
```

Add after `latest_fact`:

```python
def unit_rows(facts: dict[str, Any], tag: str, unit: str, forms: frozenset[str]) -> list[dict[str, Any]]:
    """One tag's facts in one unit, from `forms` only, that carry a value, a period end and a filing date."""
    rows = (facts.get(tag) or {}).get("units", {}).get(unit, [])
    return [row for row in rows if isinstance(row, dict) and row.get("form") in forms and row.get("val") is not None and row.get("end") and row.get("filed")]


def _debt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"end": row["end"], "value": row["val"], "filed": row["filed"]}


def _summed_debt(gaap: dict[str, Any], noncurrent_tag: str, current_tag: str, forms: frozenset[str]) -> list[dict[str, Any]]:
    """Noncurrent + current debt per filing; the current part counts as 0 when the filing lacks it."""
    current = {(row["end"], row["filed"]): row["val"] for row in unit_rows(gaap, current_tag, "USD", forms)}
    return [{"end": row["end"], "value": row["val"] + current.get((row["end"], row["filed"]), 0), "filed": row["filed"]} for row in unit_rows(gaap, noncurrent_tag, "USD", forms)]


def total_debt_sources(gaap: dict[str, Any], forms: frozenset[str]) -> list[list[dict[str, Any]]]:
    """Total-debt rows (`end`, `value`, `filed`; USD) from `forms`, one list per source, in order of preference."""
    return [
        *([_debt_row(row) for row in unit_rows(gaap, tag, "USD", forms)] for tag in TOTAL_DEBT_TAGS),
        *(_summed_debt(gaap, noncurrent, current, forms) for noncurrent, current in SUMMED_DEBT_TAGS),
        [_debt_row(row) for row in unit_rows(gaap, COMBINED_DEBT_TAG, "USD", forms)],
    ]
```

- [ ] **Step 4: Make the screen use them** (`screen/annual_figures.py`)

1. Import line becomes:
   `from trading_agent_framework.fundamentals.sec import BALANCE_SHEET_TAGS, CASH_FLOW_TAGS, INCOME_STATEMENT_TAGS, parse_dt, total_debt_sources, unit_rows`
2. In `ANNUAL_FLOW_TAGS`, replace the two literal lists with `"operating_cash_flow": CASH_FLOW_TAGS["operating_cash_flow"],` and `"capex": CASH_FLOW_TAGS["capex"],`.
3. Delete the functions `_tag_rows`, `_summed` and `_debt_sources`.
4. Replace every remaining call `_tag_rows(` with `unit_rows(` (in `annual_figures`: the flows loop, `cash_sources`, `cover`, `weighted_facts`).
5. Replace `("debt", _debt_sources(gaap))` with `("debt", total_debt_sources(gaap, _ANNUAL_FORMS))`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/fundamentals tests/strategies/bill_ackman -q`
Expected: all pass.

- [ ] **Step 6: Verify the screen's records did not change**

```bash
uv run python - <<'EOF'
import glob, json, os
from trading_agent_framework.strategies.bill_ackman.screen.annual_figures import annual_figures
with open("/tmp/claude-1000/-home-yann-projets-trading-agent-framework/457a5b20-56a5-4926-aadb-d9c998ddf1fd/scratchpad/annual_baseline.json") as handle:
    baseline = json.load(handle)
differ = []
for path in sorted(glob.glob("/home/yann/projets/trading-agent-framework/cache/sec/companyfacts/*.json")):
    with open(path) as handle:
        now = json.loads(json.dumps(annual_figures(json.load(handle)), sort_keys=True))
    if now != baseline[os.path.basename(path)]:
        differ.append(os.path.basename(path))
print(len(baseline), "compared;", "identical" if not differ else f"DIFFERENT: {differ}")
EOF
```

Expected: `183 compared; identical`. If anything differs, STOP and report the files: the spec requires a decision.

- [ ] **Step 7: Docs**

In `src/trading_agent_framework/fundamentals/CLAUDE.md`, replace `the \`us-gaap\` maps \`INCOME_STATEMENT_TAGS\`/\`BALANCE_SHEET_TAGS\` are imported by the bill_ackman screen and must not change, so filers` with `bill_ackman's screen imports \`INCOME_STATEMENT_TAGS["revenue"]\`/\`["operating_income"]\`, \`BALANCE_SHEET_TAGS["cash"]\`, \`CASH_FLOW_TAGS\` and \`total_debt_sources\` (the one definition of total debt: total tags, then noncurrent + current, then the combined tag) and those must not change without bumping its \`SCHEMA_VERSION\`, so filers`.

In `src/trading_agent_framework/strategies/bill_ackman/CLAUDE.md`, replace `\`annual_figures.py\` pure SEC reduction,` with `\`annual_figures.py\` pure SEC reduction, its total debt and cash-flow tags imported from \`fundamentals/sec.py\` (\`total_debt_sources\`, \`CASH_FLOW_TAGS\`, shared with the agent tools),`.

- [ ] **Step 8: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/fundamentals src/trading_agent_framework/strategies/bill_ackman tests/fundamentals
git commit -m "refactor(fundamentals): one total-debt and cash-flow definition shared with bill_ackman's screen

The screen's records are unchanged (183 cached payloads compared).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Every statement value carries its period and the prior year

**Files:**
- Modify: `src/trading_agent_framework/fundamentals/sec.py` (`statement_values` and new helpers above it)
- Test: `tests/fundamentals/test_sec.py`, `tests/agents/tools/test_fundamentals_tools.py:117`

**Interfaces:**
- Consumes: `sec._usd` test helper from Task 1 (same file).
- Produces: `statement_values` values shaped `{"value", "unit", "filed", "form", "start"?, "end"?, "prior_year"?: {"value", "start"?, "end"}}`; helpers `sec._period(row) -> dict`, `sec._a_year_before(end: datetime, other_end: datetime) -> bool`, constants `sec._YEAR_DAYS = 365`, `sec._PERIOD_TOLERANCE_DAYS = 15` (Task 3 uses them).

- [ ] **Step 1: Write the failing tests** (append to `tests/fundamentals/test_sec.py`)

```python
# --- statement periods and the prior year -------------------------------------------------------------------------


def _flow(val: object, start: str, end: str, filed: str = "2026-05-01", form: str = "10-Q", accn: str = "q1") -> dict[str, object]:
    return {"val": val, "start": start, "end": end, "filed": filed, "form": form, "accn": accn}


def test_a_value_carries_its_period_and_the_same_quarter_a_year_earlier() -> None:
    payload = {"facts": {"us-gaap": {"Revenues": _usd(_flow(120, "2026-01-01", "2026-03-31"), _flow(100, "2025-01-01", "2025-03-31"))}}}

    values = sec.statement_values(payload, {"revenue": ["Revenues"]}, as_of=_AS_OF)

    assert values["revenue"] == {
        "value": 120,
        "unit": "USD",
        "filed": "2026-05-01",
        "form": "10-Q",
        "start": "2026-01-01",
        "end": "2026-03-31",
        "prior_year": {"value": 100, "start": "2025-01-01", "end": "2025-03-31"},
    }


def test_a_quarter_is_never_compared_with_a_year_to_date_figure() -> None:
    """A 10-Q tags a 3-month and a 9-month figure ending the same day: each has its own prior year."""
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": _usd(
                    _flow(120, "2025-07-01", "2025-09-30", filed="2025-11-01"),
                    _flow(330, "2025-01-01", "2025-09-30", filed="2025-11-01"),
                    _flow(270, "2024-01-01", "2024-09-30", filed="2025-11-01"),  # only the 9-month prior year is reported
                )
            }
        }
    }

    values = sec.statement_values(payload, {"revenue": ["Revenues"]}, as_of=_AS_OF)

    assert values["revenue"]["value"] == 120 and values["revenue"]["start"] == "2025-07-01"
    assert "prior_year" not in values["revenue"]  # 270 covers nine months, not three


def test_a_balance_compares_with_the_balance_a_year_earlier() -> None:
    k1 = {"filed": "2026-02-01", "form": "10-K", "accn": "k1"}
    payload = {"facts": {"us-gaap": {"Assets": _usd({"val": 50, "end": "2025-12-31", **k1}, {"val": 40, "end": "2024-12-31", **k1})}}}

    values = sec.statement_values(payload, {"assets": ["Assets"]}, as_of=_AS_OF)

    assert values["assets"] == {"value": 50, "unit": "USD", "filed": "2026-02-01", "form": "10-K", "end": "2025-12-31", "prior_year": {"value": 40, "end": "2024-12-31"}}


def test_the_prior_year_is_point_in_time() -> None:
    """No look-ahead: a prior-year figure first filed after `as_of` (a later restatement) is not read."""
    payload = {"facts": {"us-gaap": {"Revenues": _usd(_flow(120, "2026-01-01", "2026-03-31"), _flow(100, "2025-01-01", "2025-03-31", filed="2026-07-01", accn="later"))}}}

    values = sec.statement_values(payload, {"revenue": ["Revenues"]}, as_of=_AS_OF)

    assert "prior_year" not in values["revenue"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/fundamentals/test_sec.py -q -k "period or quarter or balance_compares or point_in_time"`
Expected: 4 failures (values lack `start`/`end`/`prior_year`).

- [ ] **Step 3: Implement** (in `sec.py`, above `statement_values`)

```python
_YEAR_DAYS = 365
_PERIOD_TOLERANCE_DAYS = 15  # fiscal periods can end a few days off the calendar (52/53-week years)


def _period(row: dict[str, Any]) -> dict[str, Any]:
    """The `start` (flows only) and `end` a value covers."""
    return {**({"start": row["start"]} if row.get("start") else {}), **({"end": row["end"]} if row.get("end") else {})}


def _a_year_before(end: datetime, other_end: datetime) -> bool:
    return abs((end - other_end).days - _YEAR_DAYS) <= _PERIOD_TOLERANCE_DAYS


def _prior_year(candidates: list[dict[str, Any]], chosen: dict[str, Any]) -> dict[str, Any] | None:
    """The same-length period ending about a year before `chosen`, in its unit, from `candidates` (best first).

    A flow (with a `start`) matches only a flow of the same length, so a quarter is never paired with a year-to-date
    figure; a balance (no `start`) matches only a balance.
    """
    end, start = parse_dt(chosen.get("end")), parse_dt(chosen.get("start"))
    if end is None:
        return None
    for row in candidates:
        row_end, row_start = parse_dt(row.get("end")), parse_dt(row.get("start"))
        if row_end is None or row.get("unit") != chosen.get("unit") or not _a_year_before(end, row_end):
            continue
        if (start is None) != (row_start is None):
            continue
        if start is not None and row_start is not None and abs((row_end - row_start).days - (end - start).days) > _PERIOD_TOLERANCE_DAYS:
            continue
        return {"value": row["value"], **_period(row)}
    return None
```

In `statement_values`, replace the body of the `if _same_statement_period(candidate, anchor):` block with:

```python
            if _same_statement_period(candidate, anchor):
                value = {"value": candidate["value"], "unit": candidate["unit"], "filed": candidate["filed"], "form": candidate["form"], **_period(candidate)}
                prior = _prior_year(candidates, candidate)
                if prior is not None:
                    value["prior_year"] = prior
                values[field] = value
                break
```

Add to its docstring, after the first paragraph: `Each value carries its period (\`start\` for a flow, \`end\`) and, when the filings known at \`as_of\` report it, \`prior_year\`: the same-length period ending about a year earlier (\`value\`, \`start\`, \`end\`).`

- [ ] **Step 4: Update the one exact-shape assertion** in `tests/agents/tools/test_fundamentals_tools.py` (in `test_an_ifrs_filers_income_statement_and_balance_sheet_are_not_empty`):

```python
    assert income["values"]["revenue"] == {"value": 7051, "unit": "EUR", "filed": "2026-03-26", "form": "20-F", "end": "2025-12-31"}
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/fundamentals tests/agents/tools -q`
Expected: all pass.

- [ ] **Step 6: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/fundamentals tests/fundamentals tests/agents/tools
git commit -m "feat(fundamentals): statement values carry their period and the prior year

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The agent statements: total debt, cash flow, free cash flow, `stale`

**Files:**
- Modify: `src/trading_agent_framework/fundamentals/sec.py` (new constants and functions after `statement_values`)
- Modify: `src/trading_agent_framework/agents/tools/fundamentals.py:35-62`
- Modify: `src/trading_agent_framework/fundamentals/CLAUDE.md`
- Test: `tests/fundamentals/test_sec.py`, `tests/agents/tools/test_fundamentals_tools.py`

**Interfaces:**
- Consumes: `total_debt_sources`, `CASH_FLOW_TAGS` (Task 1); `_period`, `_a_year_before` (Task 2).
- Produces: `sec.IFRS_CASH_FLOW_TAGS`, `sec.STATEMENT_FORMS`, `sec.STALE_STATEMENT_DAYS = 548`; `sec.total_debt(gaap, *, end: str, as_of: datetime) -> dict | None`; `sec.balance_sheet(payload, *, as_of) -> dict[str, dict]`; `sec.income_statement(payload, *, as_of) -> dict[str, dict]`; `sec.free_cash_flow(values) -> dict | None`; `sec.is_stale_statement(values, *, as_of) -> bool`. Tool results gain `"stale": True` when stale (key absent otherwise).

- [ ] **Step 1: Write the failing tests** (append to `tests/fundamentals/test_sec.py`)

```python
# --- the agent statements: total debt, cash flow, staleness -------------------------------------------------------


def _q(val: object, end: str = "2026-03-31", filed: str = "2026-05-01", accn: str = "q1") -> dict[str, object]:
    return {"val": val, "end": end, "filed": filed, "form": "10-Q", "accn": accn}


def test_the_balance_sheets_debt_is_total_debt_not_the_current_portion() -> None:
    """VSXY's shape: the agent tool used to show $4M (the current portion) against $3.93B of liabilities."""
    payload = {"facts": {"us-gaap": {"Liabilities": _usd(_q(3_930)), "LongTermDebtCurrent": _usd(_q(4)), "LongTermDebtNoncurrent": _usd(_q(1_200))}}}

    values = sec.balance_sheet(payload, as_of=_AS_OF)

    assert values["debt"] == {"value": 1_204, "unit": "USD", "filed": "2026-05-01", "end": "2026-03-31"}
    assert values["liabilities"]["value"] == 3_930


def test_total_debt_carries_the_balance_a_year_earlier() -> None:
    payload = {"facts": {"us-gaap": {"Assets": _usd(_q(9_000)), "LongTermDebt": _usd(_q(900), _q(800, end="2025-03-31", filed="2025-05-01", accn="q0"))}}}

    assert sec.balance_sheet(payload, as_of=_AS_OF)["debt"]["prior_year"] == {"value": 800, "end": "2025-03-31"}


def test_an_ifrs_filers_debt_is_still_its_borrowings() -> None:
    assert sec.balance_sheet(_ifrs_payload(), as_of=_AS_OF)["debt"]["value"] == 738


def test_a_company_without_a_debt_figure_has_no_debt_field() -> None:
    assert "debt" not in sec.balance_sheet({"facts": {"us-gaap": {"Assets": _usd(_q(9_000))}}}, as_of=_AS_OF)


def test_the_income_statement_adds_cash_flow_and_free_cash_flow() -> None:
    year = {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "accn": "k1"}
    before = {"start": "2024-01-01", "end": "2024-12-31", "filed": "2026-02-01", "form": "10-K", "accn": "k1"}
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": _usd({"val": 1_000, **year}),
                "NetCashProvidedByUsedInOperatingActivities": _usd({"val": 300, **year}, {"val": 250, **before}),
                "PaymentsToAcquirePropertyPlantAndEquipment": _usd({"val": 100, **year}, {"val": 90, **before}),
            }
        }
    }

    values = sec.income_statement(payload, as_of=_AS_OF)

    assert values["operating_cash_flow"]["value"] == 300 and values["capex"]["value"] == 100
    assert values["free_cash_flow"] == {
        "value": 200,
        "unit": "USD",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "prior_year": {"value": 160, "start": "2024-01-01", "end": "2024-12-31"},
    }


def test_free_cash_flow_needs_both_figures_over_the_same_period() -> None:
    year = {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "accn": "k1"}
    payload = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": _usd({"val": 300, **year})}}}

    assert "free_cash_flow" not in sec.income_statement(payload, as_of=_AS_OF)


def test_an_ifrs_filers_cash_flow_is_read_from_the_ifrs_full_taxonomy() -> None:
    year = {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-03-26", "form": "20-F", "accn": "a1"}
    payload = {
        "facts": {
            "ifrs-full": {
                "CashFlowsFromUsedInOperatingActivities": {"units": {"EUR": [{"val": 500, **year}]}},
                "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": {"units": {"EUR": [{"val": 120, **year}]}},
            }
        }
    }

    values = sec.income_statement(payload, as_of=_AS_OF)

    assert values["free_cash_flow"]["value"] == 380 and values["free_cash_flow"]["unit"] == "EUR"


def test_a_statement_whose_newest_period_ended_over_18_months_ago_is_stale() -> None:
    """Ticker B (2026-02-17) returned 2011 figures, and the bear called the company's data 'unknowable'."""
    old = {"facts": {"us-gaap": {"Assets": _usd({"val": 1, "end": "2011-03-31", "filed": "2011-04-05", "form": "10-K", "accn": "o"})}}}
    fresh = {"facts": {"us-gaap": {"Assets": _usd(_q(1))}}}

    assert sec.is_stale_statement(sec.balance_sheet(old, as_of=_AS_OF), as_of=_AS_OF)
    assert not sec.is_stale_statement(sec.balance_sheet(fresh, as_of=_AS_OF), as_of=_AS_OF)
    assert not sec.is_stale_statement({}, as_of=_AS_OF)
```

And append to `tests/agents/tools/test_fundamentals_tools.py`:

```python
def test_the_income_statement_tool_has_cash_flow_and_flags_a_stale_statement() -> None:
    client = _FakeEdgarClient()
    row = {"val": 7, "start": "2010-01-01", "end": "2010-12-31", "filed": "2011-02-01", "form": "10-K", "accn": "a1"}
    client.company_facts = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [row]}}}}}

    result = _tools(client)["get_income_statement"]("B")  # type: ignore

    assert result["values"]["operating_cash_flow"]["value"] == 7
    assert result["stale"] is True


def test_a_fresh_statement_has_no_stale_key() -> None:
    client = _FakeEdgarClient()
    client.company_facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [{"val": 999, "filed": "2026-08-01", "form": "10-Q", "end": "2026-06-30", "accn": "a1"}]}}}}}

    assert "stale" not in _tools(client)["get_balance_sheet"]("AAPL")  # type: ignore
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/fundamentals/test_sec.py tests/agents/tools/test_fundamentals_tools.py -q`
Expected: the 10 new tests fail (`AttributeError: ... 'balance_sheet'`, missing `operating_cash_flow`, missing `stale`).

- [ ] **Step 3: Implement in `sec.py`**

After `CASH_FLOW_TAGS`:

```python
# Checked against the 17 cached ifrs-full payloads (2026-10-11): all 17 tag the operating total, 16 the PP&E purchase,
# 4 only the broader PP&E-and-other-assets purchase.
IFRS_CASH_FLOW_TAGS: dict[str, list[str]] = {
    "operating_cash_flow": ["CashFlowsFromUsedInOperatingActivities"],
    "capex": [
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets",
    ],
}

STATEMENT_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})
STALE_STATEMENT_DAYS = 548  # about 18 months: a statement whose newest period ended earlier is flagged `stale`
```

After `statement_values`:

```python
def _known_debt(gaap: dict[str, Any], as_of: datetime, end_matches: Callable[[str], bool]) -> dict[str, Any] | None:
    """The newest-filed total-debt row known at `as_of` whose end satisfies `end_matches`, from the most preferred source with one."""
    for rows in total_debt_sources(gaap, STATEMENT_FORMS):
        known = [row for row in rows if end_matches(row["end"]) and (filed := parse_dt(row["filed"])) is not None and _same_tz(filed, as_of) <= as_of]
        if known:
            return max(known, key=lambda row: row["filed"])
    return None


def total_debt(gaap: dict[str, Any], *, end: str, as_of: datetime) -> dict[str, Any] | None:
    """Total debt (USD) at the balance-sheet date `end`, with the figure a year earlier when reported; None without one."""
    target = parse_dt(end)
    row = _known_debt(gaap, as_of, lambda row_end: row_end == end)
    if row is None or target is None:
        return None
    value: dict[str, Any] = {"value": row["value"], "unit": "USD", "filed": row["filed"], "end": row["end"]}
    prior = _known_debt(gaap, as_of, lambda row_end: (other := parse_dt(row_end)) is not None and _a_year_before(target, other))
    if prior is not None:
        value["prior_year"] = {"value": prior["value"], "end": prior["end"]}
    return value


def balance_sheet(payload: dict[str, Any], *, as_of: datetime) -> dict[str, dict[str, Any]]:
    """The balance sheet as of `as_of`; a US GAAP filer's `debt` is its total debt (`total_debt`), never a current portion."""
    values = statement_values(payload, {**BALANCE_SHEET_TAGS, "debt": []}, as_of=as_of, ifrs_tag_map=IFRS_BALANCE_SHEET_TAGS)
    gaap = payload.get("facts", {}).get("us-gaap", {})
    end = next((value["end"] for value in values.values() if value.get("end")), None)
    if gaap and end is not None and "debt" not in values:
        debt = total_debt(gaap, end=end, as_of=as_of)
        if debt is not None:
            values["debt"] = debt
    return values


def free_cash_flow(values: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Operating cash flow minus capex over the same period and unit, with its prior year when both have one."""
    operating, capex = values.get("operating_cash_flow"), values.get("capex")
    if operating is None or capex is None or _period(operating) != _period(capex) or operating["unit"] != capex["unit"]:
        return None
    value: dict[str, Any] = {"value": operating["value"] - capex["value"], "unit": operating["unit"], **_period(operating)}
    before, capex_before = operating.get("prior_year"), capex.get("prior_year")
    if before is not None and capex_before is not None and _period(before) == _period(capex_before):
        value["prior_year"] = {"value": before["value"] - capex_before["value"], **_period(before)}
    return value


def income_statement(payload: dict[str, Any], *, as_of: datetime) -> dict[str, dict[str, Any]]:
    """The income statement and cash flow as of `as_of`, with `free_cash_flow` when both cash-flow figures cover one period."""
    values = statement_values(payload, {**INCOME_STATEMENT_TAGS, **CASH_FLOW_TAGS}, as_of=as_of, ifrs_tag_map={**IFRS_INCOME_STATEMENT_TAGS, **IFRS_CASH_FLOW_TAGS})
    fcf = free_cash_flow(values)
    if fcf is not None:
        values["free_cash_flow"] = fcf
    return values


def is_stale_statement(values: dict[str, dict[str, Any]], *, as_of: datetime) -> bool:
    """Whether the newest period in `values` ended more than `STALE_STATEMENT_DAYS` before `as_of`."""
    ends = [end for value in values.values() if (end := parse_dt(value.get("end"))) is not None]
    return bool(ends) and (as_of - _same_tz(max(ends), as_of)).days > STALE_STATEMENT_DAYS
```

Add `from collections.abc import Callable` to the imports at the top of `sec.py`.

- [ ] **Step 4: Rewire the tools** (`agents/tools/fundamentals.py`)

Replace `_statement`, `get_income_statement` and `get_balance_sheet` with:

```python
    def _statement(symbol: str, build: Callable[..., dict[str, dict[str, Any]]]) -> dict[str, Any]:
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            payload = edgar.get_company_facts_payload(cik, as_of=as_of, max_age_days=MAX_PAYLOAD_AGE_DAYS)
        except FundamentalsError as exc:
            return {"error": str(exc)}
        values = build(payload, as_of=as_of)
        result: dict[str, Any] = {"symbol": symbol.upper(), "as_of": as_of.isoformat(), "values": values}
        if sec.is_stale_statement(values, as_of=as_of):
            result["stale"] = True
        return result

    def get_income_statement(symbol: str) -> dict[str, Any]:
        """Get a company's income statement and cash flow (revenue, net income, free cash flow, ...) with the prior year, as of now."""
        return _statement(symbol, sec.income_statement)

    def get_balance_sheet(symbol: str) -> dict[str, Any]:
        """Get a company's balance sheet (assets, liabilities, total debt, ...) with the prior year, as of now."""
        return _statement(symbol, sec.balance_sheet)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/fundamentals tests/agents/tools tests/strategies/bill_ackman -q`
Expected: all pass.

- [ ] **Step 6: Check ticker B by hand** (network allowed: this is a manual check, not a test)

```bash
set -a; source /home/yann/projets/trading-agent-framework/env/.env.bull_bear.backtesting; set +a
uv run python - <<'EOF'
import os
from datetime import UTC, datetime
from pathlib import Path
from trading_agent_framework.fundamentals import sec
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
client = SecEdgarClient(os.environ["SEC_EDGAR_USER_AGENT"], Path("/home/yann/projets/trading-agent-framework/cache/sec"))
as_of = datetime(2026, 2, 17, 17, tzinfo=UTC)
cik = client.ticker_to_cik("B")
payload = client.get_company_facts_payload(cik, as_of=as_of, max_age_days=30)
values = sec.income_statement(payload, as_of=as_of)
print(cik, payload.get("entityName"), {k: (v["value"], v.get("end")) for k, v in values.items()}, "stale:", sec.is_stale_statement(values, as_of=as_of))
EOF
```

Report the output in the task summary: which company `B` maps to, the newest `end`, and the `stale` flag. If the env file name differs, find the file under `/home/yann/projets/trading-agent-framework/env/` that sets `SEC_EDGAR_USER_AGENT`. No code change follows from this step in this plan.

- [ ] **Step 7: Docs** — append to the `fundamentals/` bullet in `src/trading_agent_framework/fundamentals/CLAUDE.md`:

` The agent statements are \`sec.balance_sheet\` (a US GAAP filer's \`debt\` is \`total_debt\`, never a current portion; an IFRS filer's is \`Borrowings\`) and \`sec.income_statement\` (adds \`operating_cash_flow\`, \`capex\`, \`free_cash_flow\`); every value carries its period (\`start\` for a flow, \`end\`) and \`prior_year\`, the same-length period about a year earlier, so a 10-Q quarter is never paired with a year-to-date figure; the tools add \`stale: true\` when the newest period ended more than 548 days before \`as_of\`.`

- [ ] **Step 8: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/fundamentals src/trading_agent_framework/agents/tools tests/fundamentals tests/agents/tools
git commit -m "feat(fundamentals): agent statements show total debt, cash flow and free cash flow, and flag stale data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: A held-blind fact sheet with a one-week return; researcher and bear prompts

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/fact_sheet.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (step 3: the fact sheet call)
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py` (`_FACT_SHEET`, `RESEARCHER_SYSTEM`, `BEAR_SYSTEM`, one sentence of `JUDGE_SYSTEM`)
- Test: `tests/strategies/bull_bear/test_bull_bear_fact_sheet.py`, `tests/strategies/bull_bear/test_bull_bear_prompts.py`

**Interfaces:**
- Produces: `fact_sheet_row(stock: DebateStock, *, sector: str) -> dict` (no `weight`), keys without `held`/`weight_pct`, plus `return_1w_pct`.

- [ ] **Step 1: Rewrite `tests/strategies/bull_bear/test_bull_bear_fact_sheet.py`**

```python
from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear.debate_set import DebateStock
from trading_agent_framework.strategies.bull_bear.fact_sheet import fact_sheet_row
from trading_agent_framework.strategies.common.scoring import score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG


def _stock(closes: list[float]) -> DebateStock:
    row = score_stock("AAA", closes, [1e6] * len(closes), CONFIG)
    assert row is not None
    return DebateStock(rank=3, row=row, held=True)


def test_a_row_reports_the_momentum_facts_in_rounded_percentages() -> None:
    sheet = fact_sheet_row(_stock([100.0 + i for i in range(300)]), sector="Technology")

    assert sheet["symbol"] == "AAA" and sheet["momentum_rank"] == 3
    assert sheet["return_12m_skip_1m_pct"] == 200.0
    assert sheet["return_6m_skip_1m_pct"] == 50.0
    assert sheet["return_3m_pct"] == 18.8  # 0.1875
    assert sheet["return_1m_pct"] == 5.6  # (399 - 378) / 378
    assert sheet["return_1w_pct"] == 1.3  # (399 - 394) / 394
    assert sheet["drawdown_from_52w_high_pct"] == 0.0  # the last close is the high
    assert sheet["vs_sma200_pct"] == 33.2  # 399 / mean(200..399) - 1
    assert sheet["momentum_score"] == pytest.approx(1.188, abs=1e-3)
    assert sheet["sector"] == "Technology"
    assert isinstance(sheet["volatility_pct"], float)


def test_no_agent_is_told_whether_a_stock_is_held() -> None:
    sheet = fact_sheet_row(_stock([100.0 + i for i in range(300)]), sector="Technology")  # the stock IS held

    assert "held" not in sheet and "weight_pct" not in sheet


def test_the_drawdown_is_measured_from_the_52_week_high() -> None:
    # The last close falls 10% from 398 (a 50% fall would push the 20-day volatility over max_volatility and filter the stock out)
    closes = [100.0 + i for i in range(299)] + [398.0 * 0.9]

    sheet = fact_sheet_row(_stock(closes), sector="UNKNOWN")

    assert sheet["drawdown_from_52w_high_pct"] == pytest.approx(-10.0, abs=0.05)
```

And append to `tests/strategies/bull_bear/test_bull_bear_prompts.py`:

```python
def test_no_prompt_explains_held_or_weight() -> None:
    for text in (prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM):
        assert "held" not in text and "weight_pct" not in text


def test_the_fact_sheet_explains_the_one_week_return() -> None:
    assert "return_1w_pct" in prompts.BULL_SYSTEM and "return_1w_pct" in prompts.BEAR_SYSTEM


def test_the_researcher_reads_the_prior_year_and_is_not_asked_for_a_trend_it_cannot_see() -> None:
    assert "prior_year" in prompts.RESEARCHER_SYSTEM and "trend of revenue" not in prompts.RESEARCHER_SYSTEM


def test_the_bear_may_not_rate_news_the_price_already_absorbed_high_by_itself() -> None:
    assert "already reacted to" in prompts.BEAR_SYSTEM and "lasting consequence" in prompts.BEAR_SYSTEM
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_fact_sheet.py tests/strategies/bull_bear/test_bull_bear_prompts.py -q`
Expected: failures (`fact_sheet_row` requires `weight`; `held` in prompts; no `return_1w_pct`).

- [ ] **Step 3: Implement**

`fact_sheet.py`: add `_WEEK = 5` next to `_MONTH`; replace `fact_sheet_row` with:

```python
def fact_sheet_row(stock: DebateStock, *, sector: str) -> dict[str, Any]:
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
        "volatility_pct": _pct(row.volatility),
        "drawdown_from_52w_high_pct": _pct(drawdown_from_high(row.closes)),
        "vs_sma200_pct": _pct(versus_sma200(row.closes)),
        "sector": sector,
    }
```

`pipeline.py`, step 3: delete `weights = self._rebalancer.current_weights()` and make the sheets line
`sheets = {stock.symbol: fact_sheet_row(stock, sector=self._sector_of(stock.symbol)) for stock in debate.stocks}`.

`prompts.py`:

```python
_FACT_SHEET = (
    "Each stock comes with a fact sheet computed by code; do not recompute its numbers. momentum_rank is the stock's rank "
    "by momentum across the whole universe (1 is the strongest) and momentum_score the weighted 12-, 6- and 3-month return "
    "behind it. return_12m_skip_1m_pct and return_6m_skip_1m_pct leave out the last month; return_3m_pct, return_1m_pct "
    "and return_1w_pct (the last 5 sessions) do not. volatility_pct is the annualised volatility of the last 20 sessions. "
    "drawdown_from_52w_high_pct is how far the last close is below its one-year high. vs_sma200_pct is how far it is "
    "above (or below) its 200-day average."
)
```

In `RESEARCHER_SYSTEM`, replace `analyst actions) and the trend of revenue, operating income and debt in the latest filings. Your tool calls ` with `analyst actions) and what the latest filings show against the same period a year earlier: each statement value carries prior_year (revenue, operating income, free cash flow, total debt). Your tool calls ` (keep the surrounding string-literal layout).

In `BEAR_SYSTEM`, after `"A stock that is still rising is never momentum_exhaustion. "` insert: `"A news event the price has already reacted to (see return_1w_pct) is not a high risk by itself either: rate it high only when you can name its lasting consequence for the business (lost revenue, a fine, dilution, a damaged balance sheet). "`.

In `JUDGE_SYSTEM`, delete `A held stock you veto is sold. ` (Task 5 rewrites the rest).

- [ ] **Step 4: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): no agent sees whether a stock is held; one-week return; priced-in news is not a high risk

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The judge puts each candidate in or out; the book is what it puts in

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/parameters.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/debate_set.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/handoff.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/book.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py` (`JUDGE_SYSTEM`, `judge_task`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py` (module docstring only)
- Test: `tests/strategies/bull_bear/test_bull_bear_params.py`, `..._debate_set.py`, `..._handoff.py`, `..._book.py`, `..._pipeline.py`, `..._prompts.py`, `..._backtest.py`

**Interfaces:**
- Produces: `BullBearParams.candidate_count: int = 15`, `BullBearParams.max_book_size: int = 10` (`book_size`, `max_vetoes`, `debate_size` removed); `build_debate_set(ranked, holdings, *, candidate_count, retention_rank)`; `handoff.VERDICTS = ("in", "out")`, `Verdicts.chosen -> list[str]`, `validate_verdicts(raw, *, expected, max_in, max_chars)`, `HandoffRecorder.expect_verdicts(symbols, *, max_in)`; `build_book(debate, chosen) -> tuple[str, ...]` (rank order); judge context `constraints = {"max_in": ..., "reason_max_chars": ...}`; review log keys `verdicts`, `verdict_counts`, `book`, `targets`.

- [ ] **Step 1: Rewrite the parameter tests** (`test_bull_bear_params.py`, whole file)

```python
from __future__ import annotations

import re
from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams


def test_the_defaults_are_the_specs_and_valid() -> None:
    params = BullBearParams()

    assert (params.candidate_count, params.max_book_size, params.retention_rank) == (15, 10, 35)
    assert (params.min_weight, params.max_weight, params.cash_buffer, params.rebalance_band) == (0.04, 0.20, 0.02, 0.03)
    assert params.investable == pytest.approx(0.98)
    assert (params.rebalance_time, params.rebalance_weekday, params.parking_symbol) == ("12:00", 1, "SHV")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_weight": 0.03}, "min_weight must be above rebalance_band"),
        ({"max_book_size": 30}, "max_book_size x min_weight must fit"),
        ({"candidate_count": 35}, "retention_rank must be above candidate_count"),
        ({"retention_rank": 15}, "retention_rank must be above candidate_count"),
        ({"candidate_count": 0}, "candidate_count and max_book_size must be at least 1"),
        ({"max_book_size": 0}, "candidate_count and max_book_size must be at least 1"),
        ({"rebalance_time": "9:00"}, "rebalance_time"),
        ({"rebalance_weekday": 5}, "rebalance_weekday must be a weekday"),
        ({"agent_temperature": 3.0}, "agent_temperature"),
        ({"parking_symbol": " "}, "parking_symbol must not be blank"),
        ({"min_yahoo_coverage": 0.0}, "min_yahoo_coverage must be in (0, 1]"),
        ({"yahoo_retry_delays": (60.0, -1.0)}, "yahoo_retry_delays"),
        ({"research_tool_budget": 0}, "research_tool_budget must be at least 1"),
    ],
)
def test_an_invalid_value_is_refused_with_a_reason(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):  # messages contain "(0, 1]"
        BullBearParams(**overrides)


@pytest.mark.parametrize("field", ["shortlist_size", "min_picks", "max_picks", "book_size", "max_vetoes"])
def test_the_picker_and_veto_parameters_are_gone(field: str) -> None:
    overrides: dict[str, Any] = {field: 5}

    with pytest.raises(TypeError):
        BullBearParams(**overrides)


def test_there_is_no_derived_debate_size() -> None:
    assert not hasattr(BullBearParams(), "debate_size")
```

- [ ] **Step 2: Rewrite `test_bull_bear_book.py`** (keep its `_row`, `_debate`, `_symbols` helpers; replace the four tests)

```python
def test_the_book_is_the_chosen_stocks_in_rank_order() -> None:
    assert build_book(_debate(list(range(1, 16))), ["S07", "S02", "S11"]) == _symbols([2, 7, 11])


def test_a_holding_gets_no_preferred_slot() -> None:
    debate = _debate([*range(1, 16), 22], held=frozenset({22}))

    assert build_book(debate, ["S22", "S01"]) == _symbols([1, 22])


def test_nothing_chosen_is_an_empty_book() -> None:
    assert build_book(_debate(list(range(1, 16))), []) == ()


def test_a_chosen_symbol_outside_the_debate_is_ignored() -> None:
    assert build_book(_debate([1, 2, 3]), ["S02", "ZZZ"]) == _symbols([2])
```

- [ ] **Step 3: Update `test_bull_bear_debate_set.py`**: replace every `debate_size=15` with `candidate_count=15` (4 calls).

- [ ] **Step 4: Update the verdict tests in `test_bull_bear_handoff.py`**

Replace `_verdicts` with:

```python
def _verdicts(chosen: list[str], out: list[str] | None = None) -> list[dict[str, Any]]:
    return [{"symbol": s, "verdict": "in", "reason": "bull wins"} for s in chosen] + [{"symbol": s, "verdict": "out", "reason": "bear wins"} for s in out or []]
```

Replace `test_a_verdict_for_every_stock_is_recorded_with_its_vetoes`, `test_keeping_every_stock_is_valid_even_with_no_veto_allowed`, `test_verdicts_and_symbols_are_normalised_like_the_debaters_cases` and the `test_bad_verdicts_are_refused` parametrisation with:

```python
def test_a_verdict_for_every_stock_is_recorded_with_the_stocks_put_in() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=4)

    result = tools["submit_verdicts"](_verdicts(["AAA", "BBB", "CCC", "DDD"], ["EEE", "FFF"]))

    assert result == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "in", "bull wins")
    assert recorder.submission.chosen == ["AAA", "BBB", "CCC", "DDD"]


def test_putting_every_stock_out_is_valid() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=4)

    assert tools["submit_verdicts"](_verdicts([], DEBATE)) == {"status": "recorded"}
    assert recorder.submission.chosen == []


def test_verdicts_and_symbols_are_normalised_like_the_debaters_cases() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6)

    raw = [{"symbol": " aaa ", "verdict": "Out ", "reason": "  lawsuit filed  "}, *_verdicts(DEBATE[1:])]

    assert tools["submit_verdicts"](raw) == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "out", "lawsuit filed")


@pytest.mark.parametrize(
    ("verdicts", "message"),
    [
        ({"AAA": "in"}, "verdicts must be a list of objects"),
        (_verdicts(DEBATE[:5]), "no case for FFF"),
        (_verdicts([*DEBATE, "ZZZ"]), "ZZZ is not in the debate"),
        (_verdicts([*DEBATE, "AAA"]), "AAA appears twice"),
        ([{"symbol": "AAA", "verdict": "keep", "reason": "x"}, *_verdicts([], DEBATE[1:])], "the verdict for AAA must be one of in, out"),
        ([{"symbol": "AAA", "verdict": "in", "reason": " "}, *_verdicts([], DEBATE[1:])], "AAA has no reason"),
        ([{"symbol": "AAA", "verdict": "in", "reason": "x" * 301}, *_verdicts([], DEBATE[1:])], "the reason for AAA is 301 characters"),
        (_verdicts(DEBATE[:3], DEBATE[3:]), "3 stocks in: put at most 2 in and the others out"),
    ],
)
def test_bad_verdicts_are_refused(verdicts: Any, message: str) -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=2)

    result = tools["submit_verdicts"](verdicts)

    assert message in result["error"] and not recorder.submitted
```

- [ ] **Step 5: Update the pipeline tests** (`test_bull_bear_pipeline.py`)

Replace the `verdicts` helper:

```python
def verdicts(chosen: Collection[str] | None = None) -> Script:
    """The judge puts `chosen` in (default: the first 10 stocks it is shown, the best ranked) and the others out."""

    def step(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        symbols = [s["fact_sheet"]["symbol"] for s in ctx["stocks"]]
        put_in = set(symbols[:10]) if chosen is None else set(chosen)
        tools["submit_verdicts"]([{"symbol": s, "verdict": "in" if s in put_in else "out", "reason": "bull wins" if s in put_in else "bear wins"} for s in symbols])

    return step
```

Then make these edits:

1. Rename `test_a_review_debates_the_top_15_and_buys_the_top_10_the_judge_keeps` to `test_a_review_debates_the_top_15_and_buys_what_the_judge_puts_in` (body unchanged).
2. In `test_the_debaters_are_told_the_character_caps_that_reject_a_submission`: `assert judge["context"]["constraints"] == {"max_in": 10, "reason_max_chars": 77}`.
3. In `test_the_review_log_records_the_whole_review_as_one_line`, replace the two verdict asserts with:
   ```python
   assert line["verdicts"][0] == {"symbol": "S00", "verdict": "in", "reason": "bull wins"} and len(line["verdicts"]) == 15
   assert line["verdict_counts"] == {"in": 10, "out": 5} and "veto_cap" not in line and "judge_verdicts" not in line
   ```
4. In `test_holdings_in_the_retention_band_are_debated_and_the_others_are_forced_out`: `h.agents["judge"].script = verdicts(UNIVERSE[:10])`, and the verdict assert becomes `assert {"symbol": "S16", "verdict": "out", "reason": "bear wins"} in line["verdicts"]`; change the order comment to `# a holding put out is sold`.
5. Replace `test_a_kept_holding_takes_its_slot_before_better_ranked_new_names`, `test_vetoed_stocks_are_replaced_by_the_next_ranked_survivors`, `test_the_veto_cap_shrinks_so_that_book_size_stocks_always_survive`, `test_a_debate_set_of_exactly_book_size_leaves_the_judge_no_veto`, `test_a_veto_at_cap_zero_is_refused_and_the_retry_keeps_everything` and `test_a_judge_over_the_veto_cap_is_refused_and_corrects_itself_on_the_retry` with:

```python
def test_a_holding_put_in_stays_whatever_its_rank_and_gets_no_preferred_slot(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S16": 20}, params=BullBearParams(retention_rank=18))
    h.agents["judge"].script = verdicts([*UNIVERSE[:9], "S16"])

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["book"] == [*UNIVERSE[:9], "S16"]  # rank order: the holding comes last
    assert not any(order[0] == "S16" for order in h.orders)  # held at 10%, target ~9.8%: inside the band


def test_the_judges_choice_not_the_rank_decides_the_book(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].script = verdicts({"S03", "S14", "S07", "S11", "S12"})

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["book"] == ["S03", "S07", "S11", "S12", "S14"]
    assert [order[:2] for order in h.orders] == [(symbol, "buy") for symbol in line["book"]]


def test_fewer_than_five_stocks_put_in_leave_the_rest_parked(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].script = verdicts({"S03", "S14"})

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["book"] == ["S03", "S14"]
    assert line["targets"] == pytest.approx({"S03": 0.20, "S14": 0.20, "SHV": 0.58})
    assert any(order[0] == "SHV" and order[1] == "buy" for order in h.orders)


def test_a_judge_putting_nothing_in_parks_everything(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S00": 20})
    h.agents["judge"].script = verdicts(set())

    outcome = h.run()

    assert outcome.completed  # not an abandonment
    (line,) = h.log_lines()
    assert line["book"] == [] and line["verdict_counts"] == {"out": 15}
    assert line["targets"] == pytest.approx({"SHV": 0.98})
    assert ("S00", "sell", 20.0) in h.orders and any(order[0] == "SHV" and order[1] == "buy" for order in h.orders)


def test_a_judge_putting_more_than_max_book_size_in_is_refused_and_corrects_itself_on_the_retry(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].overrides = [verdicts(UNIVERSE[:11])]

    outcome = h.run()

    assert outcome.completed
    retry = h.agents["judge"].calls[1]
    assert retry["force_tool"] == "submit_verdicts" and "11 stocks in: put at most 10 in" in retry["task"]
    assert h.log_lines()[0]["book"] == UNIVERSE[:10]


def test_a_debate_smaller_than_max_book_size_still_completes(tmp_path: Path) -> None:
    h = _harness(tmp_path, series={symbol: series for symbol, series in _default_series().items() if symbol in UNIVERSE[:9]})

    assert h.run().completed
    assert h.log_lines()[0]["book"] == UNIVERSE[:9]
```

6. In `test_max_volatility_filters_new_entries_but_never_forces_a_holding_out`: replace `assert line["book"][0] == "S03"  # ...` with `assert "S03" in line["book"]  # the high-volatility holding the judge put in...`.
7. Rename `test_more_kept_holdings_than_book_size_keep_the_best_ranked_and_sell_the_rest` to `test_holdings_the_judge_puts_out_are_sold` (body unchanged).
8. Replace `test_a_debate_set_smaller_than_book_size_abandons_the_review` with:

```python
def test_no_stock_to_debate_abandons_the_review(tmp_path: Path) -> None:
    h = _harness(tmp_path, series={})

    outcome = h.run()

    assert not outcome.completed and h.agents["researcher"].calls == []
    (line,) = h.log_lines()
    assert line["stage"] == "data" and "no stock to debate" in line["error"]
```

9. In `test_a_broker_error_during_the_rebalance_logs_the_orders_already_sent`: replace the two lines after the comment with
   ```python
   assert line["book"] == UNIVERSE[:10] and line["verdict_counts"] == {"in": 10, "out": 5}
   assert len(line["verdicts"]) == 15
   ```
10. Rename `test_every_judge_verdict_is_logged_keeps_and_vetoes_alike` to `test_every_judge_verdict_is_logged_ins_and_outs_alike`; script `verdicts({"S00", "S14"})`; asserts `"[judge] S00: in: bull wins"`, `"[judge] S01: out: bear wins"`, `"[judge] S14: in: bull wins"`.
11. In `test_a_rejected_submission_is_logged_as_a_warning_with_the_reason_and_not_as_recorded`: overrides `[verdicts(UNIVERSE[:11])]` (comment: `# 11 in against a cap of 10: refused, then the retry puts the first 10 in`); message `"11 stocks in: put at most 10 in"`; last two asserts `assert "[judge] S10: in: bull wins" not in caplog.text` and `assert "[judge] S10: out: bear wins" in caplog.text`.

- [ ] **Step 6: Update the prompt and backtest tests**

In `test_bull_bear_prompts.py`, replace `test_the_judge_gives_a_verdict_per_stock_and_a_trend_alone_is_no_reason_to_veto` with:

```python
def test_the_judge_puts_each_stock_in_or_out_under_a_cap_and_a_trend_alone_is_no_reason_for_out() -> None:
    text = prompts.JUDGE_SYSTEM

    assert "in or out" in text and "max_in" in text
    assert "not a reason for out by itself" in text and "already reacted to" in text
    assert "veto" not in text.lower() and "veto" not in prompts.judge_task(300).lower()
```

In `test_bull_bear_backtest.py`, replace `_judge` and the settings line:

```python
def _judge(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    symbols = [s["fact_sheet"]["symbol"] for s in context["stocks"]]
    tools["submit_verdicts"]([{"symbol": s, "verdict": "in" if i < 5 else "out", "reason": "bull wins" if i < 5 else "bear wins"} for i, s in enumerate(symbols)])
```

```python
        settings=BullBearParams(candidate_count=6, max_book_size=5),  # six stocks: the judge rules on all six and puts five in
```

- [ ] **Step 7: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: many failures (`TypeError: ... candidate_count`, `max_in`, `"in"` not a verdict, ...).

- [ ] **Step 8: Implement**

`parameters.py` — replace the `book_size`/`max_vetoes` fields and the `retention_rank` comment:

```python
    retention_rank: int = 35  # a holding ranked worse (or unranked) is a forced exit; above candidate_count
    candidate_count: int = 15  # the top ranks the judge rules on every review (plus the holdings ranked up to retention_rank)
    max_book_size: int = 10  # most stocks the judge may put in; below 5 the rest of the money is parked
```

In `problems`, replace the four book/veto rules with:

```python
            "candidate_count and max_book_size must be at least 1": self.candidate_count < 1 or self.max_book_size < 1,
            "retention_rank must be above candidate_count": self.retention_rank <= self.candidate_count,
            "max_book_size x min_weight must fit in 1 - cash_buffer": self.max_book_size * self.min_weight > investable + _EPS,
```

Delete the `debate_size` property.

`debate_set.py` — rename the keyword `debate_size` to `candidate_count` in `build_debate_set` (signature, the `entry.rank <= candidate_count` test) and in the module docstring (`The top \`candidate_count\` by momentum rank, ...`).

`handoff.py`:

```python
VERDICTS = ("in", "out")
```

```python
@dataclass(frozen=True, slots=True)
class Verdicts:
    verdicts: tuple[Verdict, ...]  # one per candidate, in submission order

    @property
    def chosen(self) -> list[str]:
        """The symbols the judge put in, in submission order."""
        return [verdict.symbol for verdict in self.verdicts if verdict.verdict == "in"]
```

```python
def validate_verdicts(raw: Any, *, expected: Sequence[str], max_in: int, max_chars: int) -> Verdicts:
    """One verdict (in or out) with a reason per `expected` symbol, and at most `max_in` in."""
    verdicts = tuple(Verdict(symbol, _one_of(item, "verdict", symbol, VERDICTS), _text(item, "reason", symbol, max_chars)) for symbol, item in _covering(raw, expected, "verdicts"))
    chosen = sum(1 for verdict in verdicts if verdict.verdict == "in")
    if chosen > max_in:
        raise HandoffError(f"{chosen} stocks in: put at most {max_in} in and the others out")
    return Verdicts(verdicts)
```

`expect_verdicts(self, symbols: Sequence[str], *, max_in: int)` arms `{"expected": list(symbols), "max_in": max_in}`; in `submit`, the verdict branch passes `max_in=context["max_in"]`; the `submit_verdicts` docstring becomes `"""Submit one verdict per stock: objects with symbol, verdict (in or out) and one short reason."""`; the `submission` property docstring keeps `Verdicts`.

`book.py` (whole file):

```python
"""The book: the stocks bull_bear holds after a review, the ones the judge put in (pure).

The judge decides membership: code neither ranks the chosen stocks against each other nor gives a holding a
preferred slot. The book lists them in rank order, the order their buys are sent in.
"""

from __future__ import annotations

from collections.abc import Collection

from trading_agent_framework.strategies.bull_bear.debate_set import DebateSet


def build_book(debate: DebateSet, chosen: Collection[str]) -> tuple[str, ...]:
    """The chosen debated stocks, in rank order (`debate.stocks` is in rank order)."""
    wanted = set(chosen)
    return tuple(stock.symbol for stock in debate.stocks if stock.symbol in wanted)
```

`pipeline.py`:

- Module docstring first paragraph:
  ```
  One review: data → candidates → researcher (one run per stock) → bull → bear → judge → book → sizing → rebalance.

  The judge puts every candidate in or out; code validates (through the `HandoffRecorder`), builds the book from the
  stocks put in (rank order), sizes it, remembers (`StateStore`) and calls the `Rebalancer`. A debater stage that
  never yields a valid submission abandons the review: no order at all, forced exits included. One stock's failed
  research only gives it the note "research unavailable". Free text from an agent is logged and otherwise ignored.
  See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md and
  docs/superpowers/specs/2026-10-11-bull-bear-debate-realignment-design.md.
  ```
- Step 2: `debate = build_debate_set(ranked, holdings, candidate_count=params.candidate_count, retention_rank=params.retention_rank)`; replace the `if len(symbols) < params.book_size: ...` check and the `veto_cap = ...` line with:
  ```python
          if not symbols:
              raise ReviewAbandoned("data", "no stock to debate")
  ```
- Steps 6 to 8 (from the `# 6. Judge` comment down to the `record.update(...)` before the forced-exit warnings):

```python
        # 6. Judge: in or out on every candidate, on merit (no agent is told which stocks are held).
        self._recorder.expect_verdicts(symbols, max_in=params.max_book_size)
        context = {
            "current_datetime": moment,
            "stocks": [
                {
                    "fact_sheet": sheets[symbol],
                    "note": notes[symbol],
                    "bull": {"conviction": bull[symbol].conviction, "argument": bull[symbol].argument},
                    "bear": {"risk": bear[symbol].risk, "concern": bear[symbol].concern, "argument": bear[symbol].argument},
                }
                for symbol in symbols
            ],
            "constraints": {"max_in": params.max_book_size, "reason_max_chars": params.reason_max_chars},
        }
        self._run_stage("judge", "submit_verdicts", judge_task(params.reason_max_chars), context)
        verdicts: Verdicts = self._recorder.submission
        outs = ", ".join(f"{verdict.symbol} ({verdict.reason})" for verdict in verdicts.verdicts if verdict.verdict == "out")
        strategy.log_info(f"[judge] in: {', '.join(verdicts.chosen) or 'none'}; out: {outs or 'none'}")

        # 7. The book: every stock the judge put in, in rank order.
        book = build_book(debate, verdicts.chosen)
        strategy.log_info(f"[bull_bear] book: {', '.join(book) or 'empty, everything parked'}")

        # 8. Sizing and execution: inverse volatility within the weight bounds; below 5 stocks the rest is parked.
        volatilities = {symbol: debate.stock(symbol).row.volatility for symbol in book}
        total = min(params.investable, len(book) * params.max_weight)
        target_weights = capped_inverse_volatility(volatilities, total=total, min_weight=params.min_weight, max_weight=params.max_weight) if book else {}
        target = target_portfolio(target_weights, cash_buffer=params.cash_buffer)
        # The decision goes into the record before the first order: an abandonment during execution still says why.
        record.update(
            verdicts=[asdict(verdict) for verdict in verdicts.verdicts],
            verdict_counts=dict(Counter(verdict.verdict for verdict in verdicts.verdicts)),
            book=list(book),
            targets={**target.weights, params.parking_symbol: target.parking_weight},
        )
```

(Steps 9 onward are unchanged: the state still saves `last_book` from the verdict reasons.)

`prompts.py`:

```python
JUDGE_SYSTEM = (
    "You are the judge. For every stock, weigh the bull's case against the bear's and decide whether it belongs in the "
    "portfolio: in or out. Do not trade: code builds the portfolio from the stocks you put in, sizes the positions and "
    "places the orders.\n\n"
    f"{_FACT_SHEET}\n"
    "Each stock has the researcher's note, the bull's conviction and argument, and the bear's risk, concern and argument. "
    "Put a stock out when the bear names a concrete risk specific to it (an event, a filing, guidance, a lawsuit, a "
    "balance-sheet problem) that outweighs the bull's case. A strong or extended price trend is not a reason for out by "
    "itself: it is the momentum this portfolio buys. Neither is a news event the price has already reacted to (see "
    "return_1w_pct), unless the bear names its lasting consequence for the business. Put at most max_in stocks in, given "
    "in the constraints: when more deserve it, keep the strongest cases in and give the others out with your reason. You "
    "may put none in.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_verdicts exactly once, with one object per stock: the symbol, the verdict (in or "
    f"out) and one short reason. Give a verdict for every stock in stocks and no other. {_SUBMIT_RULE}"
)
```

```python
def judge_task(max_chars: int) -> str:
    return (
        "Give your verdict, in or out, on every stock in the context and submit them. "
        f"Each reason is at most {max_chars} characters; a longer one rejects the whole submission. The current datetime is in the context."
    )
```

`agent_bull_bear.py` module docstring: replace `builds a book of \`book_size\` stocks from those the judge keeps` with `builds the book from the stocks the judge puts in`, and the spec reference `2026-10-11-bull-bear-veto-judge-design.md` with `2026-10-11-bull-bear-debate-realignment-design.md`.

- [ ] **Step 9: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass.

- [ ] **Step 10: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): the judge puts each candidate in or out; the book is what it puts in

At most max_book_size (10) in, no veto cap, no held-first rule; below 5 names the rest is parked in SHV.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The judge remembers its verdicts and must say what changed to flip one

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/state.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/handoff.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_state.py`, `..._handoff.py`, `..._pipeline.py`, `..._prompts.py`

**Interfaces:**
- Consumes: Task 5's `validate_verdicts`, `expect_verdicts`, `Verdicts`.
- Produces: `STATE_VERSION = 2`; `BullBearState(last_completed_review, abandoned_streak, verdicts: dict[str, dict[str, str]])` (`last_book` removed; entries `{"verdict", "reason", "date"}`); `Verdict.what_changed: str | None = None`; `validate_verdicts(..., previous: Mapping[str, str] | None = None)`; `expect_verdicts(symbols, *, max_in, previous: Mapping[str, str] | None = None)`; `ReviewPipeline._review(now, record, state)`; judge stock context key `previous_verdict`; log verdict entries gain `what_changed` and `previous_verdict`.

- [ ] **Step 1: State tests** (`test_bull_bear_state.py`)

Replace `test_a_saved_state_loads_back` and `test_a_state_saved_before_the_veto_judge_loads_without_its_last_picks` with:

```python
def test_a_saved_state_loads_back(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data" / "state.json")
    state = BullBearState(last_completed_review="2026-10-06", abandoned_streak=0, verdicts={"AAA": {"verdict": "in", "reason": "bull wins", "date": "2026-10-06"}})

    store.save(state)

    assert store.load() == state
    assert json.loads((tmp_path / "data" / "state.json").read_text())["version"] == STATE_VERSION


def test_a_state_from_the_picker_or_the_veto_judge_is_an_empty_state_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "last_completed_review": "2026-10-06", "abandoned_streak": 1, "last_book": [{"symbol": "AAA", "reason": "won", "date": "2026-10-06"}]}))

    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == BullBearState()
    assert "starting from an empty state" in caplog.text
```

In the invalid-content parametrisation, replace `json.dumps({"version": STATE_VERSION, "last_book": [{"symbol": "AAA"}]})` with `json.dumps({"version": STATE_VERSION, "verdicts": {"AAA": {"verdict": "in"}}})` and add `json.dumps({"version": STATE_VERSION, "verdicts": ["AAA"]})`.

- [ ] **Step 2: Handoff tests** (append to `test_bull_bear_handoff.py`)

```python
def test_a_changed_verdict_needs_what_changed() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6, previous={"AAA": "in"})

    result = tools["submit_verdicts"](_verdicts(DEBATE[1:], ["AAA"]))

    assert "your previous verdict on AAA was in: to put it out, give what_changed" in result["error"]


def test_a_changed_verdict_with_what_changed_is_recorded_with_it() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6, previous={"AAA": "in", "BBB": "in"})
    raw = [{"symbol": "AAA", "verdict": "out", "reason": "bear wins", "what_changed": "8-K: guidance cut"}, *_verdicts(DEBATE[1:])]

    assert tools["submit_verdicts"](raw) == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "out", "bear wins", "8-K: guidance cut")
    assert recorder.submission.verdicts[1].what_changed is None


def test_an_unchanged_verdict_or_a_first_verdict_needs_no_what_changed() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6, previous={"AAA": "in"})

    assert tools["submit_verdicts"](_verdicts(DEBATE)) == {"status": "recorded"}


def test_a_long_what_changed_is_refused() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6, previous={"AAA": "in"})
    raw = [{"symbol": "AAA", "verdict": "out", "reason": "x", "what_changed": "y" * 301}, *_verdicts(DEBATE[1:])]

    assert "the what_changed for AAA is 301 characters" in tools["submit_verdicts"](raw)["error"]
```

- [ ] **Step 3: Pipeline tests** (`test_bull_bear_pipeline.py`)

In `test_a_review_debates_the_top_15_and_buys_what_the_judge_puts_in`, replace the two `state.last_book` asserts with:

```python
    assert set(state.verdicts) == set(UNIVERSE[:15])
    assert state.verdicts["S00"] == {"verdict": "in", "reason": "bull wins", "date": "2026-10-06"}
    assert state.verdicts["S14"]["verdict"] == "out"
```

In `test_the_review_log_records_the_whole_review_as_one_line`: `assert line["verdicts"][0] == {"symbol": "S00", "verdict": "in", "reason": "bull wins", "what_changed": None, "previous_verdict": None} and len(line["verdicts"]) == 15`.

Append:

```python
def test_the_judge_sees_its_previous_verdict_on_a_stock_it_judged_before(tmp_path: Path) -> None:
    earlier = {"S00": {"verdict": "out", "reason": "lawsuit", "date": "2026-09-29"}, "ZZZ": {"verdict": "in", "reason": "x", "date": "2026-09-29"}}
    h = _harness(tmp_path, state=BullBearState(verdicts=earlier))
    h.agents["judge"].script = verdicts(UNIVERSE[1:11])  # S00 stays out: no what_changed needed

    assert h.run().completed

    stocks = {s["fact_sheet"]["symbol"]: s for s in h.agents["judge"].calls[0]["context"]["stocks"]}
    assert stocks["S00"]["previous_verdict"] == {"verdict": "out", "reason": "lawsuit", "date": "2026-09-29"}
    assert "previous_verdict" not in stocks["S01"]
    (line,) = h.log_lines()
    assert line["verdicts"][0]["previous_verdict"] == "out"
    assert h.store.load().verdicts["ZZZ"] == earlier["ZZZ"]  # a stock not judged this time keeps its last verdict


def test_a_judge_flipping_a_verdict_without_what_changed_is_refused_then_corrects_itself(tmp_path: Path) -> None:
    # a verdict from months ago still counts
    h = _harness(tmp_path, state=BullBearState(verdicts={"S00": {"verdict": "out", "reason": "lawsuit", "date": "2026-07-07"}}))
    h.agents["judge"].overrides = [verdicts()]  # puts S00 in with no what_changed

    def with_what_changed(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        symbols = [s["fact_sheet"]["symbol"] for s in ctx["stocks"]]
        tools["submit_verdicts"]([{"symbol": s, "verdict": "in" if s in symbols[:10] else "out", "reason": "r", **({"what_changed": "lawsuit dismissed"} if s == "S00" else {})} for s in symbols])

    h.agents["judge"].script = with_what_changed

    assert h.run().completed

    assert "your previous verdict on S00 was out" in h.agents["judge"].calls[1]["task"]
    (line,) = h.log_lines()
    assert line["verdicts"][0]["what_changed"] == "lawsuit dismissed"
```

- [ ] **Step 4: Prompt test** (append to `test_bull_bear_prompts.py`)

```python
def test_the_judge_starts_from_its_previous_verdict_and_says_what_changed() -> None:
    assert "previous_verdict" in prompts.JUDGE_SYSTEM and "what_changed" in prompts.JUDGE_SYSTEM
```

- [ ] **Step 5: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: failures (`verdicts` not a field, `previous` unexpected keyword, no `previous_verdict`).

- [ ] **Step 6: Implement**

`state.py`: module docstring second sentence becomes `\`StateStore\` keeps the date of the last completed review (a completed Tuesday is not reviewed again after a restart), the abandoned-review streak and the judge's latest verdict on every stock it judged, in one small JSON file per mode, written atomically. A missing or invalid file, or one from an older version (the picker, the veto judge), is an empty state.`; then:

```python
STATE_VERSION = 2
_VERDICT_KEYS = {"verdict", "reason", "date"}
```

```python
@dataclass(frozen=True, slots=True)
class BullBearState:
    last_completed_review: str | None = None  # ISO market date of the last completed review
    abandoned_streak: int = 0  # abandoned reviews in a row
    verdicts: dict[str, dict[str, str]] = field(default_factory=dict)  # symbol -> the judge's latest {verdict, reason, date}


def _valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return False
    last = raw.get("last_completed_review")
    streak = raw.get("abandoned_streak", 0)
    verdicts = raw.get("verdicts", {})
    return (
        (last is None or isinstance(last, str))
        and isinstance(streak, int)
        and not isinstance(streak, bool)
        and streak >= 0
        and isinstance(verdicts, dict)
        and all(isinstance(entry, dict) and set(entry) == _VERDICT_KEYS and all(isinstance(value, str) for value in entry.values()) for entry in verdicts.values())
    )
```

In `load`, the returned state is `BullBearState(last_completed_review=raw.get("last_completed_review"), abandoned_streak=raw.get("abandoned_streak", 0), verdicts={symbol: dict(entry) for symbol, entry in raw.get("verdicts", {}).items()})`. Delete `_BOOK_KEYS`.

`handoff.py`:

```python
@dataclass(frozen=True, slots=True)
class Verdict:
    symbol: str
    verdict: str  # one of VERDICTS
    reason: str
    what_changed: str | None = None  # the new fact behind a verdict that differs from the judge's previous one
```

```python
def _what_changed(item: Mapping[str, Any], symbol: str, verdict: str, before: str | None, max_chars: int) -> str | None:
    """Required, as one short sentence, when `verdict` differs from the judge's previous verdict `before`."""
    value = item.get("what_changed")
    given = isinstance(value, str) and bool(value.strip())
    if before is not None and before != verdict and not given:
        raise HandoffError(f"your previous verdict on {symbol} was {before}: to put it {verdict}, give what_changed, the new and material fact")
    return _text(item, "what_changed", symbol, max_chars) if given else None


def validate_verdicts(raw: Any, *, expected: Sequence[str], max_in: int, max_chars: int, previous: Mapping[str, str] | None = None) -> Verdicts:
    """One verdict (in or out) with a reason per `expected` symbol; at most `max_in` in; `what_changed` on every flip from `previous`."""
    before = previous or {}
    verdicts = []
    for symbol, item in _covering(raw, expected, "verdicts"):
        verdict = _one_of(item, "verdict", symbol, VERDICTS)
        verdicts.append(Verdict(symbol, verdict, _text(item, "reason", symbol, max_chars), _what_changed(item, symbol, verdict, before.get(symbol), max_chars)))
    chosen = sum(1 for verdict in verdicts if verdict.verdict == "in")
    if chosen > max_in:
        raise HandoffError(f"{chosen} stocks in: put at most {max_in} in and the others out")
    return Verdicts(tuple(verdicts))
```

`expect_verdicts(self, symbols: Sequence[str], *, max_in: int, previous: Mapping[str, str] | None = None)` arms `{"expected": list(symbols), "max_in": max_in, "previous": dict(previous or {})}`; `submit` passes `previous=context["previous"]`; the tool docstring becomes `"""Submit one verdict per stock: objects with symbol, verdict (in or out), one short reason, and what_changed when it differs from your previous_verdict."""`.

`pipeline.py`:

- `run`: call `self._review(now, record, state)`; `_review(self, now: datetime, record: dict[str, Any], state: BullBearState) -> None`.
- Step 6: before `self._recorder.expect_verdicts(...)` add
  ```python
          previous = {symbol: state.verdicts[symbol] for symbol in symbols if symbol in state.verdicts}
  ```
  change the arming to `self._recorder.expect_verdicts(symbols, max_in=params.max_book_size, previous={symbol: entry["verdict"] for symbol, entry in previous.items()})`, and add to each stock's dict in the context, after `"bear": {...},`: `**({"previous_verdict": previous[symbol]} if symbol in previous else {}),`. Change the comment to `# 6. Judge: in or out on every candidate, on merit (no agent is told which stocks are held), starting from its own previous verdict on a stock when it gave one.`
- In the `record.update(...)` of step 8: `verdicts=[{**asdict(verdict), "previous_verdict": previous.get(verdict.symbol, {}).get("verdict")} for verdict in verdicts.verdicts],`.
- Step 9: replace the `reasons = ...` and `self._state.save(...)` lines with
  ```python
          decided = {verdict.symbol: {"verdict": verdict.verdict, "reason": verdict.reason, "date": today} for verdict in verdicts.verdicts}
          self._state.save(BullBearState(last_completed_review=today, abandoned_streak=0, verdicts={**state.verdicts, **decided}))
  ```
- `_submission_lines`, last line:
  ```python
      return [f"[{agent_name}] {verdict.symbol}: {verdict.verdict}: {verdict.reason}" + (f" (changed: {verdict.what_changed})" if verdict.what_changed else "") for verdict in submission.verdicts]
  ```

`prompts.py`, `JUDGE_SYSTEM`: after `"may put none in. "` (keep a space) insert `"A stock you judged before shows your previous_verdict: start from it and change it only for a new, material fact, and then give what_changed, one short sentence on that fact."` and make the closing instruction `"... with one object per stock: the symbol, the verdict (in or out), one short reason, and what_changed when the verdict differs from your previous_verdict. Give a verdict for every stock in stocks and no other. "`.

- [ ] **Step 7: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass.

- [ ] **Step 8: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): the judge sees its previous verdict and must give what_changed to flip it

State version 2 keeps the latest verdict per stock; an older state file loads empty.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: A stock the judge sold stays out of the candidates for 4 reviews

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/cooldown.py`
- Modify: `parameters.py`, `state.py`, `debate_set.py`, `pipeline.py` (all under `src/trading_agent_framework/strategies/bull_bear/`)
- Test: `tests/strategies/bull_bear/test_bull_bear_cooldown.py` (create), `..._debate_set.py`, `..._params.py`, `..._state.py`, `..._pipeline.py`

**Interfaces:**
- Consumes: Task 6's `BullBearState`, `_review(now, record, state)`.
- Produces: `BullBearParams.reentry_cooldown_reviews: int = 4`; `advance_cooldowns(current: Mapping[str, int], *, started: Collection[str], length: int) -> dict[str, int]`; `BullBearState.cooldowns: dict[str, int]`; `build_debate_set(..., excluded: Collection[str] = ())`; `ForcedExit.reason` may be `"cooldown"`; log keys `cooldowns_active`, `cooldowns_started`.

- [ ] **Step 1: Write the failing tests**

`tests/strategies/bull_bear/test_bull_bear_cooldown.py`:

```python
from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.cooldown import advance_cooldowns


def test_a_stock_sold_by_the_judge_starts_a_cooldown_of_the_full_length() -> None:
    assert advance_cooldowns({}, started=["AAA"], length=4) == {"AAA": 4}


def test_each_completed_review_counts_a_cooldown_down_and_drops_it_after_the_last() -> None:
    cooldowns: dict[str, int] = {"AAA": 4}
    seen = []
    for _ in range(4):
        cooldowns = advance_cooldowns(cooldowns, started=[], length=4)
        seen.append(dict(cooldowns))

    assert seen == [{"AAA": 3}, {"AAA": 2}, {"AAA": 1}, {}]  # excluded at 4 reviews, back at the 5th


def test_a_length_of_zero_starts_no_cooldown() -> None:
    assert advance_cooldowns({"AAA": 2}, started=["BBB"], length=0) == {"AAA": 1}
```

Append to `test_bull_bear_debate_set.py`:

```python
def test_a_stock_in_cooldown_is_not_debated() -> None:
    debate = build_debate_set(_ranked(40), [], candidate_count=15, retention_rank=35, excluded={"S03"})

    assert "S03" not in debate.symbols and len(debate.symbols) == 14


def test_a_holding_still_in_cooldown_is_a_forced_exit() -> None:
    debate = build_debate_set(_ranked(40), ["S03"], candidate_count=15, retention_rank=35, excluded={"S03"})

    assert debate.forced_exits == (ForcedExit("S03", "cooldown"),) and "S03" not in debate.symbols
```

In `test_bull_bear_params.py`: add `assert params.reentry_cooldown_reviews == 4` to the defaults test and `({"reentry_cooldown_reviews": -1}, "reentry_cooldown_reviews must be at least 0"),` to the invalid list.

In `test_bull_bear_state.py`: in `test_a_saved_state_loads_back` add `cooldowns={"BBB": 3}` to the state; add `json.dumps({"version": STATE_VERSION, "cooldowns": {"AAA": 0}})` and `json.dumps({"version": STATE_VERSION, "cooldowns": {"AAA": True}})` to the invalid list.

Append to `test_bull_bear_pipeline.py`:

```python
# --- the re-entry cooldown ---------------------------------------------------------------------------------


def test_a_holding_the_judge_puts_out_starts_its_cooldown(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S03": 20})
    h.agents["judge"].script = verdicts(set(UNIVERSE[:10]) - {"S03"})

    assert h.run().completed

    assert h.store.load().cooldowns == {"S03": 4}
    (line,) = h.log_lines()
    assert line["cooldowns_started"] == ["S03"] and line["cooldowns_active"] == {}
    assert ("S03", "sell", 20.0) in h.orders


def test_a_stock_put_out_that_was_not_held_starts_no_cooldown(tmp_path: Path) -> None:
    h = _harness(tmp_path)  # the default judge puts S10-S14 out; none is held

    assert h.run().completed
    assert h.store.load().cooldowns == {}


def test_a_stock_in_cooldown_is_not_debated_and_its_count_goes_down(tmp_path: Path) -> None:
    h = _harness(tmp_path, state=BullBearState(cooldowns={"S03": 2, "S05": 1}))

    assert h.run().completed

    (line,) = h.log_lines()
    debated = [stock["symbol"] for stock in line["debate_set"]]
    assert "S03" not in debated and "S05" not in debated and len(debated) == 13
    assert line["cooldowns_active"] == {"S03": 2, "S05": 1}
    assert h.store.load().cooldowns == {"S03": 1}


def test_an_abandoned_review_leaves_the_cooldowns_as_they_were(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S03": 20}, state=BullBearState(cooldowns={"S05": 2}))
    h.agents["judge"].script = does_nothing

    assert not h.run().completed
    assert h.store.load().cooldowns == {"S05": 2}


def test_a_holding_still_in_cooldown_is_sold_as_a_forced_exit(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S03": 20}, state=BullBearState(cooldowns={"S03": 3}))

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["forced_exits"] == [{"symbol": "S03", "reason": "cooldown"}]
    assert ("S03", "sell", 20.0) in h.orders
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: failures (no `cooldown` module, no `excluded`, no `cooldowns`).

- [ ] **Step 3: Implement**

`cooldown.py`:

```python
"""The re-entry cooldown: a stock the judge sold stays out of the candidates for a number of completed reviews (pure).

The audit of 73 sells (spec 2026-10-11-bull-bear-debate-realignment-design.md) found half of them bought back, most
within two weeks, and those re-entries then lagged the other candidates by 4.2 points over 4 weeks.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping


def advance_cooldowns(current: Mapping[str, int], *, started: Collection[str], length: int) -> dict[str, int]:
    """The cooldowns after a completed review: each running one counts down and is dropped after its last review,
    and each stock in `started` begins one of `length` reviews (none when `length` is 0)."""
    running = {symbol: left - 1 for symbol, left in current.items() if left > 1}
    return {**running, **({symbol: length for symbol in started} if length > 0 else {})}
```

`parameters.py`: field `reentry_cooldown_reviews: int = 4  # completed reviews a stock the judge sold stays out of the candidates`; rule `"reentry_cooldown_reviews must be at least 0": self.reentry_cooldown_reviews < 0,`.

`state.py`: field `cooldowns: dict[str, int] = field(default_factory=dict)  # symbol -> completed reviews it stays out of the candidates`; in `_valid` read `cooldowns = raw.get("cooldowns", {})` and add `and isinstance(cooldowns, dict) and all(isinstance(left, int) and not isinstance(left, bool) and left > 0 for left in cooldowns.values())`; `load` passes `cooldowns=dict(raw.get("cooldowns", {}))`; the module docstring mentions the cooldowns.

`debate_set.py`: `ForcedExit.reason` comment becomes `# "unranked", "rank N" or "cooldown"`; module docstring gains `A stock in its re-entry cooldown (\`excluded\`) is never debated; a holding still in cooldown (its sell did not fill) is a forced exit.`; the function:

```python
def build_debate_set(ranked: Sequence[RankedRow], holdings: Collection[str], *, candidate_count: int, retention_rank: int, excluded: Collection[str] = ()) -> DebateSet:
    """`holdings` are the stocks held, the parking instrument excluded (`Rebalancer.holdings()`); `excluded` the stocks in cooldown."""
    held, blocked = set(holdings), set(excluded)
    rank_of = {entry.row.symbol: entry.rank for entry in ranked}
    # Debate the top `candidate_count` ranks and the holdings still within the retention rank, never a stock in cooldown.
    stocks = tuple(
        DebateStock(entry.rank, entry.row, entry.row.symbol in held)
        for entry in ranked
        if entry.row.symbol not in blocked and (entry.rank <= candidate_count or (entry.row.symbol in held and entry.rank <= retention_rank))
    )
    forced = tuple(
        ForcedExit(symbol, "cooldown" if symbol in blocked else "unranked" if symbol not in rank_of else f"rank {rank_of[symbol]}")
        for symbol in sorted(held)
        if symbol in blocked or symbol not in rank_of or rank_of[symbol] > retention_rank
    )
    return DebateSet(stocks=stocks, forced_exits=forced)
```

`pipeline.py` (import `advance_cooldowns` from `...bull_bear.cooldown`):
- Step 2: `debate = build_debate_set(ranked, holdings, candidate_count=params.candidate_count, retention_rank=params.retention_rank, excluded=state.cooldowns)`; add `cooldowns_active=dict(state.cooldowns),` to that `record.update(...)`.
- Step 8 `record.update(...)`: before it compute `sold = [stock.symbol for stock in debate.stocks if stock.held and stock.symbol not in set(verdicts.chosen)]` and add `cooldowns_started=sold,`.
- Step 9 save: `BullBearState(last_completed_review=today, abandoned_streak=0, verdicts={**state.verdicts, **decided}, cooldowns=advance_cooldowns(state.cooldowns, started=sold, length=params.reentry_cooldown_reviews))`.

- [ ] **Step 4: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): a stock the judge sold stays out of the candidates for 4 reviews

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The judge sees how its recent `out` verdicts turned out

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/feedback.py`
- Modify: `parameters.py`, `state.py`, `pipeline.py`, `prompts.py` (under `src/trading_agent_framework/strategies/bull_bear/`)
- Test: `tests/strategies/bull_bear/test_bull_bear_feedback.py` (create), `..._params.py`, `..._state.py`, `..._pipeline.py`, `..._prompts.py`

**Interfaces:**
- Consumes: Task 7's state and pipeline.
- Produces: `BullBearParams.feedback_reviews: int = 4`; `BullBearState.history: list[dict]` (entries `{"date": str, "verdicts": {symbol: {"verdict", "reason", "close"}}}`); `recent_outcomes(history, closes_now: Mapping[str, float]) -> list[dict]` (entries `{"symbol", "date", "reason", "return_since_pct", "in_names_return_pct"}`); judge context key `recent_verdicts`; log key `recent_verdicts`.

- [ ] **Step 1: Write the failing tests**

`tests/strategies/bull_bear/test_bull_bear_feedback.py`:

```python
from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.feedback import recent_outcomes

HISTORY = [
    {"date": "2026-09-22", "verdicts": {"AAA": {"verdict": "in", "reason": "r", "close": 100.0}, "BBB": {"verdict": "out", "reason": "lawsuit", "close": 50.0}}},
    {
        "date": "2026-09-29",
        "verdicts": {
            "AAA": {"verdict": "in", "reason": "r", "close": 110.0},
            "CCC": {"verdict": "in", "reason": "r", "close": 20.0},
            "DDD": {"verdict": "out", "reason": "dilution", "close": 10.0},
        },
    },
]


def test_each_out_verdict_is_shown_newest_first_against_that_days_in_stocks() -> None:
    now = {"AAA": 121.0, "BBB": 60.0, "CCC": 22.0, "DDD": 9.0}

    assert recent_outcomes(HISTORY, now) == [
        {"symbol": "DDD", "date": "2026-09-29", "reason": "dilution", "return_since_pct": -10.0, "in_names_return_pct": 10.0},
        {"symbol": "BBB", "date": "2026-09-22", "reason": "lawsuit", "return_since_pct": 20.0, "in_names_return_pct": 21.0},
    ]


def test_a_stock_without_a_price_now_is_left_out_and_never_estimated() -> None:
    assert [outcome["symbol"] for outcome in recent_outcomes(HISTORY, {"AAA": 121.0, "BBB": 60.0, "CCC": 22.0})] == ["BBB"]


def test_a_review_whose_in_stocks_have_no_price_shows_none_for_them() -> None:
    assert recent_outcomes(HISTORY[:1], {"BBB": 60.0})[0]["in_names_return_pct"] is None


def test_no_history_means_no_feedback() -> None:
    assert recent_outcomes([], {"AAA": 1.0}) == []
```

In `test_bull_bear_params.py`: defaults test `assert params.feedback_reviews == 4`; invalid list `({"feedback_reviews": 0}, "feedback_reviews must be at least 1"),`.

In `test_bull_bear_state.py`: in `test_a_saved_state_loads_back` add `history=[{"date": "2026-10-06", "verdicts": {"AAA": {"verdict": "in", "reason": "bull wins", "close": 51.5}}}]`; invalid list add `json.dumps({"version": STATE_VERSION, "history": [{"date": "2026-10-06", "verdicts": {"AAA": {"verdict": "in", "reason": "r"}}}]})` and `json.dumps({"version": STATE_VERSION, "history": [{"date": "2026-10-06"}]})`.

In `test_bull_bear_prompts.py`:

```python
def test_the_judge_is_told_how_to_read_its_recent_verdicts() -> None:
    assert "recent_verdicts" in prompts.JUDGE_SYSTEM and "in_names_return_pct" in prompts.JUDGE_SYSTEM
```

Append to `test_bull_bear_pipeline.py`:

```python
# --- outcome feedback ----------------------------------------------------------------------------------------


def test_the_judge_sees_how_its_recent_out_verdicts_did_since(tmp_path: Path) -> None:
    series = _default_series()
    now_close = {symbol: s.closes[-1] for symbol, s in series.items()}
    history = [
        {
            "date": "2026-09-29",
            "verdicts": {
                "S00": {"verdict": "in", "reason": "r", "close": now_close["S00"] / 1.10},
                "S18": {"verdict": "out", "reason": "dilution", "close": now_close["S18"] / 0.95},  # S18 is no candidate now, but priced
            },
        }
    ]
    h = _harness(tmp_path, series=series, state=BullBearState(history=history))

    assert h.run().completed

    shown = h.agents["judge"].calls[0]["context"]["recent_verdicts"]
    assert shown == [{"symbol": "S18", "date": "2026-09-29", "reason": "dilution", "return_since_pct": -5.0, "in_names_return_pct": 10.0}]
    assert h.log_lines()[0]["recent_verdicts"] == shown


def test_a_completed_review_stores_each_verdict_with_the_close_it_was_made_at(tmp_path: Path) -> None:
    series = _default_series()
    earlier = [{"date": "2026-09-22", "verdicts": {}}, {"date": "2026-09-29", "verdicts": {}}]
    h = _harness(tmp_path, series=series, params=BullBearParams(feedback_reviews=2), state=BullBearState(history=earlier))

    assert h.run().completed

    history = h.store.load().history
    assert [review["date"] for review in history] == ["2026-09-29", "2026-10-06"]  # trimmed to feedback_reviews
    assert history[-1]["verdicts"]["S00"] == {"verdict": "in", "reason": "bull wins", "close": series["S00"].closes[-1]}
    assert history[-1]["verdicts"]["S14"]["verdict"] == "out"
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: failures (no `feedback` module, no `history`, no `recent_verdicts`).

- [ ] **Step 3: Implement**

`feedback.py`:

```python
"""What the judge sees of its recent `out` verdicts: how each stock did since, against the stocks it put in (pure).

Prices are the closes stored when the judge decided and the last closes the review loaded now, so no price after
`clock.now()` is ever read: in a backtest both come from the gated bars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


def _since(close_then: float, close_now: float | None) -> float | None:
    if close_now is None or close_then <= 0:
        return None
    return close_now / close_then - 1


def recent_outcomes(history: Sequence[Mapping[str, Any]], closes_now: Mapping[str, float]) -> list[dict[str, Any]]:
    """One entry per `out` verdict in `history`, newest review first: symbol, date, reason, `return_since_pct`, and
    `in_names_return_pct` (the mean return of that review's `in` stocks over the same span; None when none is priced).
    An `out` stock without a price now is left out, never estimated."""
    outcomes: list[dict[str, Any]] = []
    for review in reversed(history):
        decided = review["verdicts"]
        in_returns = [since for symbol, item in decided.items() if item["verdict"] == "in" and (since := _since(item["close"], closes_now.get(symbol))) is not None]
        in_mean = round(fmean(in_returns) * 100, 1) if in_returns else None
        for symbol, item in decided.items():
            since = _since(item["close"], closes_now.get(symbol)) if item["verdict"] == "out" else None
            if since is not None:
                outcomes.append({"symbol": symbol, "date": review["date"], "reason": item["reason"], "return_since_pct": round(since * 100, 1), "in_names_return_pct": in_mean})
    return outcomes
```

`parameters.py`: field `feedback_reviews: int = 4  # completed reviews whose out verdicts the judge sees with their outcome`; rule `"feedback_reviews must be at least 1": self.feedback_reviews < 1,`.

`state.py`: field `history: list[dict[str, Any]] = field(default_factory=list)  # the last feedback_reviews reviews: {date, verdicts: {symbol: {verdict, reason, close}}}`; add

```python
def _valid_review(entry: Any) -> bool:
    """One `history` entry: {date, verdicts: {symbol: {verdict, reason, close}}}."""
    if not isinstance(entry, dict) or set(entry) != {"date", "verdicts"} or not isinstance(entry["date"], str) or not isinstance(entry["verdicts"], dict):
        return False
    return all(
        isinstance(item, dict)
        and set(item) == {"verdict", "reason", "close"}
        and isinstance(item["verdict"], str)
        and isinstance(item["reason"], str)
        and isinstance(item["close"], int | float)
        and not isinstance(item["close"], bool)
        for item in entry["verdicts"].values()
    )
```

and in `_valid` `history = raw.get("history", [])` with `and isinstance(history, list) and all(_valid_review(entry) for entry in history)`; `load` passes `history=[{"date": entry["date"], "verdicts": {symbol: dict(item) for symbol, item in entry["verdicts"].items()}} for entry in raw.get("history", [])]`.

`pipeline.py` (import `recent_outcomes`):
- After `ranked = rank(rows)` (step 1): `closes_now = {symbol: data.closes[-1] for symbol, data in series.items() if data.closes}`.
- Step 6: `recent = recent_outcomes(state.history, closes_now)`; add `"recent_verdicts": recent,` to the judge `context` (after `"stocks"`); add `recent_verdicts=recent,` to the step-8 `record.update(...)`.
- Step 9 save adds `history=[*state.history, {"date": today, "verdicts": {verdict.symbol: {"verdict": verdict.verdict, "reason": verdict.reason, "close": closes_now[verdict.symbol]} for verdict in verdicts.verdicts}}][-params.feedback_reviews:]`.

`prompts.py`, `JUDGE_SYSTEM`: after the `previous_verdict` sentence insert `" recent_verdicts lists the stocks you put out at your last reviews, with each one's return since (return_since_pct) against the mean return of the stocks you put in that day (in_names_return_pct): an out verdict whose stock then beat the stocks you put in cost the portfolio."`

- [ ] **Step 4: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): the judge sees how its recent out verdicts did against the stocks it put in

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The judge keeps lessons with two memory tools

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py`
- Modify: `parameters.py`, `pipeline.py`, `prompts.py` (under `src/trading_agent_framework/strategies/bull_bear/`)
- Test: `tests/strategies/bull_bear/test_bull_bear_strategy.py`, `..._pipeline.py`, `..._params.py`, `..._prompts.py`

**Interfaces:**
- Consumes: `memory.tools.memory_tools(store)`, `Strategy.memory`, `agents.tools.only`, `agents.results.ToolCallRecord(name, args, result)`.
- Produces: `BullBearParams.judge_memory_budget: int = 6`; the judge agent's tools `search_memory`, `remember_lesson`, `submit_verdicts` with `exempt_tools=["submit_verdicts"]`; `ReviewPipeline._run_stage(...) -> list[ToolCallRecord]`; log key `memory = {"calls": int, "lessons": list[str]}`.

- [ ] **Step 1: Write the failing tests**

`test_bull_bear_strategy.py`: in `test_initialize_creates_the_four_agents_with_only_their_own_tools` the judge entry becomes `"judge": {"submit_verdicts", "search_memory", "remember_lesson"},`; replace `test_only_the_researcher_has_a_budget_exempt_tool_and_every_agent_uses_the_temperature` with:

```python
def test_the_researcher_and_the_judge_end_with_a_budget_exempt_submit_and_every_agent_uses_the_temperature(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=BullBearParams(agent_temperature=0.1))

    strategy.initialize()

    assert {created["name"]: created.get("exempt_tools") for created in agents.created} == {"researcher": ["submit_note"], "bull": None, "bear": None, "judge": ["submit_verdicts"]}
    assert {created["temperature"] for created in agents.created} == {0.1}
```

`test_bull_bear_pipeline.py`: import `ToolCallRecord` (`from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord`); give `FakeAgent` a `self.tool_calls: list[ToolCallRecord] = []` (in `__init__`) and return `AgentRunResult(output="done", tool_calls=self.tool_calls)`; append:

```python
def test_the_judge_runs_with_its_memory_budget_and_its_lessons_are_logged(tmp_path: Path) -> None:
    h = _harness(tmp_path, params=BullBearParams(judge_memory_budget=4))
    h.agents["judge"].tool_calls = [
        ToolCallRecord("search_memory", {"query": "S03"}, "{}"),
        ToolCallRecord("remember_lesson", {"text": "out on a shelf filing alone was wrong", "symbol": "CRDO"}, "{}"),
        ToolCallRecord("submit_verdicts", {}, "{}"),
    ]

    assert h.run().completed

    assert h.agents["judge"].calls[0]["tool_budget"] == 4
    assert h.log_lines()[0]["memory"] == {"calls": 2, "lessons": ["out on a shelf filing alone was wrong"]}
```

`test_bull_bear_params.py`: defaults `assert params.judge_memory_budget == 6`; invalid `({"judge_memory_budget": -1}, "judge_memory_budget must be at least 0"),`.

`test_bull_bear_prompts.py`:

```python
def test_the_judge_is_told_about_its_memory_tools() -> None:
    assert "search_memory" in prompts.JUDGE_SYSTEM and "remember_lesson" in prompts.JUDGE_SYSTEM
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: failures (judge tools, exempt tools, `memory` log key, budget).

- [ ] **Step 3: Implement**

`parameters.py`: field `judge_memory_budget: int = 6  # memory tool calls per judge run; submit_verdicts is exempt`; rule `"judge_memory_budget must be at least 0": self.judge_memory_budget < 0,`.

`agent_bull_bear.py`: add `_JUDGE_MEMORY_TOOLS = {"search_memory", "remember_lesson"}` next to `_RESEARCH_TOOLS`; in `initialize`, add `from trading_agent_framework.memory.tools import memory_tools` next to the other local imports and replace the judge creation with:

```python
            # The judge also keeps lessons between reviews (`strategy.memory`, wiped at the start of every backtest).
            lessons = only(memory_tools(self.memory), _JUDGE_MEMORY_TOOLS)
            self.agents.create(name="judge", system_prompt=JUDGE_SYSTEM, tools=[*lessons, submit["submit_verdicts"]], temperature=temperature, exempt_tools=["submit_verdicts"])
```

Update the comment above the bull/bear creation to `# The bull and the bear argue from the same evidence pack: no research tool, only their submit tool.`

`pipeline.py`:
- Import `ToolCallRecord` alongside `AgentRunResult`.
- `_run_stage(...) -> list[ToolCallRecord]`; docstring adds `Returns the tool calls of the run that submitted.`; inside the loop set `calls: list[ToolCallRecord] = []` before `try:`, `calls = result.tool_calls` right after `agent.run(...)`, and `return calls` instead of the bare `return`.
- Step 6: `judge_calls = self._run_stage("judge", "submit_verdicts", judge_task(params.reason_max_chars), context, tool_budget=params.judge_memory_budget)`; add `memory=_memory_record(judge_calls),` to the step-8 `record.update(...)`.
- Module level, next to `_submission_lines`:

```python
_MEMORY_TOOLS = ("search_memory", "remember_lesson")


def _memory_record(calls: Sequence[ToolCallRecord]) -> dict[str, Any]:
    """The judge's memory use in one review: how many memory calls, and the text of each lesson it recorded."""
    used = [call for call in calls if call.name in _MEMORY_TOOLS]
    return {"calls": len(used), "lessons": [str(call.args.get("text", "")) for call in used if call.name == "remember_lesson"]}
```

`prompts.py`, `JUDGE_SYSTEM`: after the `recent_verdicts` sentence insert `" You have two memory tools: search_memory finds the lessons you recorded at earlier reviews, and remember_lesson records a short new one, for example when recent_verdicts shows that an out verdict was right or wrong, and why. Look up your lessons before deciding, and record a lesson only when an outcome teaches something."`

- [ ] **Step 4: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass (the backtest test's manager accepts the extra tools).

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): the judge keeps lessons with search_memory and remember_lesson

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `research_news` switches the researcher's news tool off

**Files:**
- Modify: `parameters.py`, `prompts.py`, `agent_bull_bear.py` (under `src/trading_agent_framework/strategies/bull_bear/`)
- Test: `tests/strategies/bull_bear/test_bull_bear_strategy.py`, `..._prompts.py`, `..._params.py`

**Interfaces:**
- Produces: `BullBearParams.research_news: bool = True`; `prompts.researcher_system(news: bool) -> str`, `prompts.RESEARCHER_SYSTEM == researcher_system(True)`; `BullBearStrategy.parameters["research_news"]` mirrors the setting (recorded in `settings.json`).

- [ ] **Step 1: Write the failing tests**

`test_bull_bear_strategy.py`:

```python
def test_with_research_news_off_the_researcher_has_no_news_tool_and_no_news_source_is_needed(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=BullBearParams(research_news=False))
    strategy.broker.news = None  # type: ignore[attr-defined]

    strategy.initialize()

    researcher = next(created for created in agents.created if created["name"] == "researcher")
    assert _tool_names(researcher) == {"get_income_statement", "get_balance_sheet", "submit_note"}
    assert "news" not in researcher["system_prompt"].lower()
    assert strategy.parameters["research_news"] is False


def test_research_news_is_on_by_default_and_recorded_in_the_parameters(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)

    assert strategy.settings.research_news is True and strategy.parameters["research_news"] is True
```

`test_bull_bear_prompts.py`:

```python
def test_the_researcher_without_news_is_not_told_about_news() -> None:
    assert "news" in prompts.researcher_system(True) and "news" not in prompts.researcher_system(False).lower()
    assert prompts.RESEARCHER_SYSTEM == prompts.researcher_system(True)
```

`test_bull_bear_params.py`: defaults `assert params.research_news is True`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: failures (`research_news` unknown, no `researcher_system`).

- [ ] **Step 3: Implement**

`parameters.py`: field `research_news: bool = True  # the researcher's news tool; False for the news-off comparison run`.

`prompts.py`: replace `RESEARCHER_SYSTEM = (...)` with:

```python
def researcher_system(news: bool) -> str:
    """The researcher's prompt, with or without its news tool."""
    look_for = (
        "Look for what the fact sheet cannot show: recent company news (results, guidance, deals, lawsuits, management "
        "changes, analyst actions) and what the latest filings show"
        if news
        else "Look for what the fact sheet cannot show: what the latest filings show"
    )
    return (
        "You are the researcher of a long-only stock portfolio that buys strong momentum stocks. You receive one stock at a "
        "time and gather the facts a bull and a bear will argue from. Do not trade and do not give an opinion.\n\n"
        f"{_FACT_SHEET}\n"
        f"{look_for} against the same period a year earlier: each statement value carries prior_year (revenue, operating "
        "income, free cash flow, total debt). Your tool calls are limited: spend them on what matters most for this stock. "
        "Write facts only, each with its date, and no recommendation. Do not repeat the fact sheet's numbers. If you find "
        "nothing useful, say so in one line.\n\n"
        f"{_ENGLISH}\n\n"
        f"End your run by calling submit_note exactly once, with the stock's symbol and your note. {_SUBMIT_RULE}"
    )


RESEARCHER_SYSTEM = researcher_system(news=True)
```

`agent_bull_bear.py`:
- Import `researcher_system` instead of `RESEARCHER_SYSTEM`.
- In `__init__`, right after `self.settings = ...`: `self.parameters = {**self.parameters, "research_news": self.settings.research_news}  # recorded in settings.json, so the news-on and news-off runs can be told apart`.
- In `initialize`:

```python
        try:
            news = self.settings.research_news
            if news:
                self._require_news_provider()
            recorder = HandoffRecorder(self.settings)
            submit = submit_tools(recorder)
            from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
            from trading_agent_framework.agents.tools.news import news_tools
            from trading_agent_framework.memory.tools import memory_tools

            temperature = self.settings.agent_temperature
            sources = [*(news_tools(self) if news else []), *fundamentals_tools(self)]
            research = only(sources, _RESEARCH_TOOLS if news else _RESEARCH_TOOLS - {"search_news"})
            self.agents.create(name="researcher", system_prompt=researcher_system(news), tools=[*research, submit["submit_note"]], temperature=temperature, exempt_tools=["submit_note"])
```

  (the bull, bear and judge creation lines stay as Task 9 left them).
- The `initialize` log line ends with `, research news {'on' if self.settings.research_news else 'off'}`.

- [ ] **Step 4: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): research_news switches the researcher's news tool off for the comparison run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Docs and the manual checks

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md`
- Modify: `README.md` (lines 73, 301, 303: the bull_bear section)

- [ ] **Step 1: Rewrite the Architecture bullet of `strategies/bull_bear/CLAUDE.md`** with this text (one bullet, replacing the current one):

```markdown
- `bull_bear/` (`BullBearStrategy` in `agent_bull_bear.py`, registered as `"bull_bear"`): lumibot's bull vs bear debate over cross_momentum's ranking, once a week (Tuesday 12:00 ET, `sleeptime = "1D"`, `BullBearParams.rebalance_weekday` = 1: every other session returns at once; a Tuesday whose review completed is not run again after a restart, unlike cross_momentum). Code ranks the universe with `strategies/common/scoring.py` and cross_momentum's own `CONFIG` (never duplicated) and builds the candidates (`debate_set.py`): the top `candidate_count` (15) plus the holdings ranked up to `retention_rank` (35), minus the stocks in their re-entry cooldown (`max_volatility` gates new entries only: the pipeline scores a holding with a copy of `CONFIG` that lifts it, so `scoring.py`/cross_momentum are untouched; a holding ranked worse, unranked or still in cooldown is a forced exit, sold by code, never shown to the agents, and logged as a warning (`forced exit: <symbol> (<reason>)`) just before its sell order is sent: not when the review is abandoned, since nothing is sold then). The fact sheet (`fact_sheet.py`) has NO `held` or `weight_pct`: no agent is told what is held. Four agents hand over through submit tools validated by `HandoffRecorder` (`handoff.py`: `submit_note`, `submit_bull_case`, `submit_bear_case`, `submit_verdicts`; no `from __future__ import annotations`): a researcher run PER STOCK (`search_news` when `research_news` is on, plus the SEC `get_income_statement` (with cash flow and free cash flow) and `get_balance_sheet` (total debt), each value with its `prior_year`; `research_tool_budget` 3 with `submit_note` exempt; a failed run gives that stock the note "research unavailable"), then the bull and the bear, independently and from the same evidence, with no research tool (every stock rated low/medium/high; the bear names a concern, rates high only for a concrete named risk, may use `momentum_exhaustion` only for a reversal already under way (a negative one-month return and `drawdown_from_52w_high_pct` of -15 or below), and may not rate high a news event the price already reacted to (`return_1w_pct`) without naming its lasting consequence), then the judge: `in` or `out`, with a reason, on EVERY candidate, at most `max_book_size` (10) `in`, no veto cap; it sees its `previous_verdict` on a stock it judged before and must give `what_changed` to flip it (the submission is rejected otherwise), `recent_verdicts` (every `out` verdict of the last `feedback_reviews` (4) reviews with the stock's return since against that day's `in` stocks, from stored closes: no price after `clock.now()`), and has `search_memory` and `remember_lesson` (`strategy.memory`, wiped at the start of every backtest; `judge_memory_budget` 6, `submit_verdicts` exempt). Code builds the book (`book.build_book`: every `in` stock, in rank order; no held-first rule), sizes it by inverse volatility within [4%, 20%] summing to `min(98%, n x 20%)` (`sizing.capped_inverse_volatility`; below 5 stocks the rest is parked in SHV, zero `in` parks everything) and rebalances through `strategies/common/rebalancer.py`'s `Rebalancer`. A holding the judge puts out is sold and starts a cooldown of `reentry_cooldown_reviews` (4) completed reviews (`cooldown.advance_cooldowns`). No overlays. A bull, bear or judge with no valid submission after the forced retry, unusable data (`market_data.YahooBars`: one Yahoo batch in paper/live, fetched again after 60 s and 180 s; `GateBars` in backtests), or no candidate at all abandons the review with NO order, forced exits included, and changes no verdict, cooldown or history; a `BrokerError`/`BacktestError` abandons it too (stage `broker` before the rebalancer starts: no order was sent; stage `execution` during it: orders may already be out, logged in `reviews.jsonl` with the judge's verdicts, the book and the targets); a backtest raises `FatalStrategyError` at 3 abandoned in a row (paper/live log and carry on). State in `data/bull_bear_state_<mode>.json` (`StateStore`, `STATE_VERSION` 2: last completed review date, abandoned streak, the judge's latest verdict per stock, the cooldowns, the last `feedback_reviews` reviews' verdicts with their closes; a version-1 file (picker or veto judge) loads as an empty state; wiped by `initialize` in backtests only). `ReviewLog` appends one line per review (forced exits included) to `reviews.jsonl` in the run directory: `verdicts` (each with `what_changed` and `previous_verdict`), `verdict_counts`, `book`, `targets`, `cooldowns_active`, `cooldowns_started`, `recent_verdicts`, `memory` (calls and lessons). `research_news` (default on) is mirrored into `strategy.parameters`, so `settings.json` tells a news-off run apart; set it to False in `BullBearParams` for that run. `AgentHandle.run` wraps every exception, `ConfigurationError` included, as `AgentError`, so a misconfiguration surfacing DURING a review costs only that stock's note ("research unavailable"); the real misconfigurations (news provider when `research_news` is on, SEC user agent, `LLM_MODEL`) are caught at `initialize`.
```

In the Gotchas bullet, replace `**bull_bear fails fast without a news source, and its sectors are not point-in-time.** \`BullBearStrategy.initialize\` calls \`_require_news_provider\`` with `**bull_bear fails fast without a news source (unless \`research_news\` is off), and its sectors are not point-in-time.** \`BullBearStrategy.initialize\` calls \`_require_news_provider\` when \`research_news\` is on`.

- [ ] **Step 2: README** — in the bull_bear section:
  - Line 73: `\`bull_bear\` checks for a news source when it starts and refuses to run without one, in every mode (backtests included), unless \`BullBearParams.research_news\` is off: set \`ALPACA_NEWS_*\` in its env file before the first run.`
  - Line 301 (Env file row): `(required in every mode while \`research_news\` is on: the strategy refuses to start without a news source)`.
  - Line 303: replace the sentence from `and the judge gives a keep or veto verdict` through `and places every order.` with: `and the judge puts every stock in or out of the portfolio, with a reason, without being told which stocks are held (at most 10 in; a strong trend alone is no reason for out). It sees its previous verdict on each stock and must say what changed to flip it, sees how the stocks it recently put out did against those it put in, and can keep short lessons with two memory tools. Code holds exactly the stocks put in, sizes them by inverse volatility (4% to 20% each, up to 98% invested; with fewer than 5 stocks the rest stays in SHV), sells holdings that fell below rank 35 (the agents never see those) and any holding put out, which then cannot come back for 4 reviews, and places every order.`

- [ ] **Step 3: Manual check** (network): `uv run python scripts/tests/smoke_quality_screen.py` (with `SEC_EDGAR_USER_AGENT` exported as in Task 3 Step 6). Expected: the same pass output as before the branch; report it. If it fails for a reason unrelated to this branch (network, SEC rate limit), say so.

- [ ] **Step 4: Lint, full suite, commit**

```bash
uv run ruff check && uv run pytest -q
git add -A src/trading_agent_framework/strategies/bull_bear/CLAUDE.md README.md
git commit -m "docs(bull_bear): the realigned debate: held-blind in/out judge, verdict memory, cooldown, feedback

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Final whole-branch review** before merge (the user's rule for order code): `superpowers:requesting-code-review` on `main..feat/bull-bear-debate-realignment`.
