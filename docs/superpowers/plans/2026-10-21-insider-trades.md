# insider_trades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new `insider_trades` strategy: a researcher agent verifies code-labelled (routine / opportunistic) SEC Form 4
open-market trades on a 10-stock watchlist, and a trader agent tilts an evenly weighted long-only book toward opportunistic
insider buying and away from heavy opportunistic selling, through a guarded order desk.

**Architecture:** Same shape as `congress_trades`: pure parsing / classification / signal modules, one SEC source (the only
I/O), clock-gated research tools, submit tools validated by a recorder, a `TradeDesk` as the only order code, a JSON state
store and a pipeline that runs agents only when something changed. The desk and the trade-report hand-off move to
`strategies/common/` so both strategies share them.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff, pyright, stdlib `xml.etree.ElementTree`, the existing `SecEdgarClient`,
LangChain agents through `strategy.agents`, `YahooBacktestData`.

**Spec:** `docs/superpowers/specs/2026-10-21-insider-trades-design.md` (read it first; this plan argues from it).

## Global Constraints

- Every root and nested `CLAUDE.md` rule applies. In particular: money is `Decimal` (the desk keeps its existing float maths,
  it is not new); no `datetime.now` / `time.sleep` in strategy, pipeline, source or tool code — every cutoff comes from
  `strategy.clock.now()`; SEC HTTP only through `fundamentals/edgar_client.py`; tests never touch the network and use
  hand-written fakes, not `MagicMock`.
- Modules that define LangChain tool closures (`handoff.py`, `tools.py`, `strategies/common/trade_desk.py`) have NO
  `from __future__ import annotations`.
- A Form 4 is known only when its **filing date is strictly before** `strategy.clock.now()`'s market date (`MARKET_TZ`).
- Only form type `4`; `4/A` and Form 5 ignored and counted. Only non-derivative rows with code `P` or `S`.
- Only officers and directors count; a joint filing's rows count once, attributed to the first listed owner who is an
  officer or director.
- Labels per trade: 10b5-1 box checked ⇒ routine ("plan"); else the 3 calendar years before the trade's year decide.
- Signal window: filed in `[today - 30 days, today)`.
- Weights: every watchlist stock in `[0.03, 0.20]`, sum ≤ `0.95`, even weight = `0.95 / len(watchlist)`.
- Strategy name `insider_trades`; state file `data/insider_trades_state_<mode>.json`; log tag `[insider_trades]`.
- `congress_trades` behaviour does not change: its test assertions stay as they are (only import lines may move to
  `strategies/common/`).

## Review Focus

1. **A joint filing whose first owner is a 10%-owner fund and whose second is a director** — the rows count once, attributed
   to the director; the fund is listed as a co-filer. (Test in Task 3.)
2. **A P/S row with no price or no shares element** (some filings footnote a price range and leave the value out) — the row
   is skipped and counted as `unpriced`, the filing's other rows still count, nothing raises. (Test in Task 3.)
3. **One watchlist ticker fails to load** (CIK lookup 404, SEC's HTML throttle page) — in paper/live the other nine still
   run and the failed one keeps its previous note and is never read as "its filings expired"; in a backtest the load is
   retried on the next tick and the run is abandoned (streak), never run on partial data. (Tests in Tasks 6 and 10.)
4. **The same insider under two spellings of their name across years** — history is keyed on the owner CIK, so the label
   is unaffected. (Test in Task 4.)
5. **Float rounding in submitted weights** (ten weights of 0.095 summing to 0.9500000000000001, or 0.0950004 vs even)
   — accepted within the tolerance, never refused. (Test in Task 8.)

---

## File structure

```
src/trading_agent_framework/
  strategies/common/trade_desk.py      MOVED+GENERALISED from congress_trades/desk.py: TradeDesk, Audit, Shortfall, order_status, DeskParams
  strategies/common/handoff.py         MOVED from congress_trades/handoff.py: HandoffError, OrderView, OrderReport, validate_report, field readers
  strategies/common/run_log.py         MOVED from congress_trades/state.py: RunLog
  fundamentals/sec.py                  + older_pages (moved from earnings_drift/events.py), + raw_form4_document
  fundamentals/edgar_client.py         + get_submissions_page
  utils/errors.py                      + InsiderDataError
  strategies/insider_trades/
    __init__.py                        exports InsiderTradesStrategy
    CLAUDE.md
    parameters.py                      InsiderParams
    form4/__init__.py
    form4/parse.py                     PURE: Form 4 XML -> Form4Filing (Owner, Form4Trade)
    form4/classify.py                  PURE: Label, Labelled, label_trades
    form4/signal.py                    PURE: window, overrides, InsiderSummary, StockFacts, SignalClass, signal_class
    form4/source.py                    Form4Source: SEC submissions + XML, known-before-today, load reports
    facts.py                           InsiderFacts: the per-day labelled facts shared by the tools and the pipeline
    tools.py                           insider_research_tools: get_insider_trades, read_form4
    handoff.py                         InsiderRecorder, validate_note, validate_target, submit_tools
    state.py                           InsiderState, PendingTrade, StateStore, state_path
    pipeline.py                        InsiderPipeline
    prompts.py                         researcher_system, trader_system, task prompts, retry_prompt
    agent_insider_trades.py            InsiderTradesStrategy
  utils/strategy_factory.py            + Strategies.INSIDER_TRADES
scripts/tests/smoke_insider_trades.py
tests/strategies/insider_trades/...    one test file per module, fixtures/ for Form 4 XML
```

---

### Task 1: Share the order desk, the trade-report hand-off and the run log

**Files:**
- Create: `src/trading_agent_framework/strategies/common/trade_desk.py`, `strategies/common/handoff.py`, `strategies/common/run_log.py`
- Delete: `src/trading_agent_framework/strategies/congress_trades/desk.py`
- Modify: `strategies/congress_trades/handoff.py`, `state.py`, `pipeline.py`, `agent_congress_trades.py`, `strategies/common/CLAUDE.md`, `strategies/congress_trades/CLAUDE.md`
- Modify (import lines only): `tests/strategies/congress_trades/test_congress_desk.py`, `test_congress_handoff.py`, `test_congress_pipeline.py`, `test_congress_prompts.py`, `test_congress_state.py`
- Test: `tests/strategies/common/test_common_trade_desk.py`

**Interfaces:**
- Produces:
  - `class DeskParams(Protocol)`: `rebalance_band: float`, `min_trade_pct: float`, `order_wait_seconds: float` (`CongressParams` and later `InsiderParams` satisfy it structurally).
  - `TradeDesk(strategy: Strategy, params: DeskParams, *, log_tag: str)`; `begin_run(target: Mapping[str, float] | None, traded: Iterable[str]) -> None`; NEW `set_target(target: Mapping[str, float]) -> None`; `has_target: bool` property; unchanged `traded`, `orders`, `orders_view()`, `tools()`, `audit()`. Log lines and refusals use `f"[{log_tag}] ..."`.
  - `Audit`, `Shortfall`, `order_status` unchanged, from `strategies.common.trade_desk`.
  - From `strategies.common.handoff`: `HandoffError`, `OrderView`, `OrderReport`, `validate_report(raw, *, orders, reason_max_chars) -> list[OrderReport]`, and the readers renamed public: `items(raw, what)`, `key(item, field, index)`, `text(item, who, max_chars, *, required)`, `weight(item, ticker)`.
  - `strategies.common.run_log.RunLog(path: Path | None)` with `append(record: dict[str, Any]) -> None`.

- [ ] **Step 1: Write the failing tests** in `tests/strategies/common/test_common_trade_desk.py` (reuse the `FakeBroker` / `Strategy` set-up of `test_congress_desk.py`):

```python
def test_with_no_target_every_order_is_refused_until_one_is_set(...):
    desk = TradeDesk(strategy, CongressParams(), log_tag="insider_trades")
    desk.begin_run(None, traded=["AAA"])
    place = {t.__name__: t for t in desk.tools()}["place_order"]
    assert place("AAA", "sell", 1) == {"error": "no target yet: call submit_target first"}
    assert place("AAA", "buy", 1) == {"error": "no target yet: call submit_target first"}
    assert desk.orders == [] and not desk.has_target
    desk.set_target({"AAA": 0.10})
    assert desk.has_target and "order_id" in place("AAA", "buy", 1)

def test_set_target_keeps_the_orders_already_placed_this_run(...):
    # begin_run({"AAA": 0.1}, []), place a buy, set_target({"AAA": 0.2}) -> len(desk.orders) == 1

def test_the_log_tag_prefixes_refusals(caplog, ...):
    # a refused order logs "[insider_trades] order refused:"
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/common/test_common_trade_desk.py -v` — Expected: FAIL (`ModuleNotFoundError: trading_agent_framework.strategies.common.trade_desk`).

- [ ] **Step 3: Move the code.** `git mv` `congress_trades/desk.py` to `common/trade_desk.py`; type `params` as `DeskParams`; replace the hard-coded `[congress_trades]` with `log_tag`; a `None` target makes `_place` return the refusal above before any account read, and `audit()` return `Audit(unfilled, [], [])` with no shortfalls. Move `HandoffError`, `OrderView`, `OrderReport`, `validate_report` and the four readers to `common/handoff.py` (no `from __future__ import annotations`); `congress_trades/handoff.py` imports them. Move `RunLog` to `common/run_log.py`. Update congress imports; `CongressTradesStrategy.initialize` builds `TradeDesk(self, params, log_tag="congress_trades")`. Update both `CLAUDE.md` files (common: the three new modules; congress: `desk.py` now lives in common).

- [ ] **Step 4: Run** `uv run pytest tests/strategies/common tests/strategies/congress_trades -q && uv run ruff check && uv run pyright` — Expected: all pass, congress assertions untouched (`git diff tests/strategies/congress_trades` shows import lines only).

- [ ] **Step 5: Commit** — `git commit -m "refactor: share congress_trades' trade desk, report hand-off and run log via strategies/common (Task 1)"`

---

### Task 2: `InsiderParams` and `InsiderDataError`

**Files:**
- Create: `strategies/insider_trades/__init__.py` (empty for now), `strategies/insider_trades/parameters.py`
- Modify: `utils/errors.py`
- Test: `tests/strategies/insider_trades/test_insider_params.py`

**Interfaces:**
- Produces: `InsiderDataError(TradingFrameworkError)`. `@dataclass(frozen=True, slots=True) class InsiderParams` with exactly these fields and defaults:
  `watchlist: tuple[str, ...] = ("AAPL", "MSFT", "JPM", "BAC", "XOM", "CVX", "PFE", "INTC", "F", "KO")`,
  `signal_window_days: int = 30`, `history_years: int = 3`, `max_total_weight: float = 0.95`, `min_weight: float = 0.03`,
  `max_position_weight: float = 0.20`, `min_buy_holding_fraction: float = 0.01`, `min_sell_holding_fraction: float = 0.10`,
  `weight_tolerance: float = 0.001`, `rebalance_band: float = 0.01`, `min_trade_pct: float = 0.005`, `max_trade_retries: int = 3`,
  `max_consecutive_abandoned: int = 3`, `reason_max_chars: int = 300`, `agent_temperature: float | None = 0.3`,
  `order_wait_seconds: float = 60.0`; property `even_weight -> float` (= `max_total_weight / len(watchlist)`).

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_are_the_spec_values():
    p = InsiderParams()
    assert p.watchlist == ("AAPL", "MSFT", "JPM", "BAC", "XOM", "CVX", "PFE", "INTC", "F", "KO")
    assert (p.min_weight, p.max_position_weight, p.max_total_weight) == (0.03, 0.20, 0.95)
    assert (p.min_buy_holding_fraction, p.min_sell_holding_fraction) == (0.01, 0.10)
    assert p.signal_window_days == 30 and p.history_years == 3
    assert p.even_weight == pytest.approx(0.095)

@pytest.mark.parametrize("kwargs, message", [
    ({"watchlist": ()}, "watchlist must not be empty"),
    ({"watchlist": ("AAPL", "aapl")}, "watchlist must not repeat a ticker"),
    ({"min_weight": 0.10}, "watchlist x min_weight must fit in max_total_weight"),
    ({"max_position_weight": 0.05}, "the even weight must lie between min_weight and max_position_weight"),
    ({"min_sell_holding_fraction": 0.0}, "holding fractions must be in (0, 1]"),
    ({"min_buy_holding_fraction": 1.5}, "holding fractions must be in (0, 1]"),
    ({"signal_window_days": 0}, "signal_window_days must be at least 1"),
    ({"history_years": 0}, "history_years must be at least 1"),
    ({"min_weight": 0.005}, "min_weight must be at least rebalance_band"),
])
def test_invalid_settings_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        InsiderParams(**kwargs)
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/test_insider_params.py -v` — Expected: FAIL (module not found).
- [ ] **Step 3: Implement** in `parameters.py`, `__post_init__` in the `CongressParams` style (one `problems` dict, every failure joined in one `ValueError("InsiderParams: ...")`); watchlist entries upper-cased and stripped are compared for repeats; also keep congress' finite / band / temperature / retries checks.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(insider_trades): InsiderParams and InsiderDataError (Task 2)`

---

### Task 3: Form 4 parser (pure)

**Files:**
- Create: `strategies/insider_trades/form4/__init__.py`, `form4/parse.py`
- Modify: `fundamentals/sec.py` (+ `raw_form4_document`)
- Test: `tests/strategies/insider_trades/form4/test_parse.py`, fixtures `tests/strategies/insider_trades/form4/fixtures/*.xml` (synthetic, built from the SEC `ownershipDocument` schema; real ones replace or join them in Task 6's checkpoint)

**Interfaces:**
- Produces:
  - `sec.raw_form4_document(primary_document: str) -> str`: `"xslF345X05/wk-form4_1.xml" -> "wk-form4_1.xml"`; a name with no `/` is returned as is.
  - `@dataclass(frozen=True, slots=True) class Owner`: `cik: str`, `name: str`, `is_director: bool`, `is_officer: bool`, `officer_title: str`, `is_ten_percent_owner: bool`, `is_other: bool`; properties `is_insider` (director or officer) and `role` (`"CEO, director"`, `"director"`, `"CFO"`, `"officer"` when the title is blank, `"10% owner"`, `"other"`).
  - `@dataclass(frozen=True, slots=True) class Form4Trade`: `accession: str`, `filed: date`, `ticker: str`, `owner_cik: str`, `owner_name: str`, `role: str`, `code: str` (`"P"`/`"S"`), `trade_date: date`, `shares: Decimal`, `price: Decimal`, `owned_after: Decimal`, `plan: bool`, `footnotes: tuple[str, ...]`; properties `dollars` (`shares * price`) and `holding_fraction` (P: `shares / owned_after`, 1 when `owned_after` is 0; S: `shares / (shares + owned_after)`).
  - `@dataclass(frozen=True, slots=True) class Form4Filing`: `accession`, `filed`, `ticker`, `owners: tuple[Owner, ...]`, `attributed_to: Owner | None`, `co_filers: tuple[str, ...]` (names of the other owners), `trades: tuple[Form4Trade, ...]` (P/S of the attributed owner only), `plan: bool`, `other_codes: int`, `unpriced: int`, `footnotes: dict[str, str]`.
  - `parse_form4(xml: str, *, accession: str, filed: date, ticker: str) -> Form4Filing`; raises `InsiderDataError` on XML that does not parse or has no `ownershipDocument` root.

- [ ] **Step 1: Write the failing tests** (one fixture per case: `sale_ceo.xml`, `purchase_director.xml`, `joint_fund_then_director.xml`, `mixed_codes.xml`, `unpriced_row.xml`, `plan_checked.xml`, `ten_percent_only.xml`):

```python
def test_an_open_market_sale_by_the_ceo():
    f = parse_form4(fixture("sale_ceo.xml"), accession="0001-25-000001", filed=date(2025, 10, 3), ticker="AAPL")
    (t,) = f.trades
    assert (t.code, t.trade_date, t.shares, t.price, t.owned_after) == ("S", date(2025, 10, 1), Decimal(100), Decimal("255.10"), Decimal(3_280_295))
    assert t.role == "CEO, director" and t.owner_cik == "0001214156" and not t.plan
    assert t.dollars == Decimal("25510.00")
    assert t.holding_fraction == Decimal(100) / Decimal(3_280_395)

def test_a_joint_filing_counts_its_rows_once_for_the_first_insider():
    f = parse_form4(fixture("joint_fund_then_director.xml"), ...)
    assert f.attributed_to.name == "DOE JANE" and f.co_filers == ("BIG FUND LP",)
    assert {t.owner_name for t in f.trades} == {"DOE JANE"} and len(f.trades) == 2

def test_awards_exercises_and_derivative_rows_are_counted_not_kept():
    f = parse_form4(fixture("mixed_codes.xml"), ...)   # rows A, M, F, S + one derivative row
    assert [t.code for t in f.trades] == ["S"] and f.other_codes == 4

def test_a_row_without_a_price_is_skipped_and_counted():
    f = parse_form4(fixture("unpriced_row.xml"), ...)  # one S without <transactionPricePerShare><value>, one priced S
    assert len(f.trades) == 1 and f.unpriced == 1

def test_the_10b5_1_box_marks_every_trade_as_plan():
    assert all(t.plan for t in parse_form4(fixture("plan_checked.xml"), ...).trades)   # <aff10b5One>1</aff10b5One>, also "true"

def test_a_filing_by_a_ten_percent_owner_alone_has_no_trades():
    f = parse_form4(fixture("ten_percent_only.xml"), ...)
    assert f.attributed_to is None and f.trades == ()

def test_footnotes_referenced_by_a_row_travel_with_the_trade():
    # sale_ceo.xml's row references F1 ("Sold under a Rule 10b5-1 plan adopted ...") -> t.footnotes == ("Sold under ...",)

def test_garbage_raises():
    with pytest.raises(InsiderDataError):
        parse_form4("<html>Request Rate Threshold Exceeded</html>", accession="x", filed=date(2025, 1, 2), ticker="AAPL")

def test_raw_document_strips_the_xsl_folder():
    assert raw_form4_document("xslF345X05/wk-form4_1.xml") == "wk-form4_1.xml"
    assert raw_form4_document("form4.xml") == "form4.xml"
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/form4/test_parse.py -v` — Expected: FAIL (module not found).
- [ ] **Step 3: Implement** with `xml.etree.ElementTree.fromstring`; booleans are `"1"`/`"true"` (case-insensitive) true, anything else false; values are read from `<x><value>…</value></x>`; amounts as `Decimal(str)`. A P/S row missing shares, price, date or `sharesOwnedFollowingTransaction` counts as `unpriced`.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(insider_trades): Form 4 XML parser (Task 3)`

---

### Task 4: Routine / opportunistic labels (pure)

**Files:**
- Create: `strategies/insider_trades/form4/classify.py`
- Test: `tests/strategies/insider_trades/form4/test_classify.py`

**Interfaces:**
- Consumes: `Form4Trade` (Task 3).
- Produces: `class Label(StrEnum)`: `ROUTINE = "routine"`, `OPPORTUNISTIC = "opportunistic"`, `UNCLASSIFIED = "unclassified"`.
  `@dataclass(frozen=True, slots=True) class Labelled`: `trade: Form4Trade`, `label: Label`, `rule: str`, `months_by_year: dict[int, tuple[int, ...]]`.
  `label_trades(trades: Sequence[Form4Trade], history: Sequence[Form4Trade], *, years: int) -> list[Labelled]` — `history` is every known attributed trade of the issuer; an owner's history is the `trade_date`s of their P/S trades (keyed on `owner_cik`) in the `years` calendar years before `trade.trade_date.year`.

Rule strings (exact copy, shown to the researcher):
- plan: `"plan: the 10b5-1 box is checked"`
- routine: `f"routine: traded in {month name} in each of {y1}, {y2}, {y3}"` (years ascending; the earliest qualifying month)
- opportunistic: `f"opportunistic: traded in each of {y1}, {y2}, {y3}, never in the same month every year"`
- unclassified: `f"unclassified: no trade in {missing years ascending, comma-separated}"`

- [ ] **Step 1: Write the failing tests**

```python
def test_same_month_three_years_running_is_routine():           # history: 2022-03, 2023-03, 2024-03; trade 2025-06
    (l,) = label_trades([trade(date(2025, 6, 2))], history, years=3)
    assert l.label is Label.ROUTINE and l.rule == "routine: traded in March in each of 2022, 2023, 2024"
    assert l.months_by_year == {2022: (3,), 2023: (3,), 2024: (3,)}

def test_every_year_but_never_the_same_month_is_opportunistic(): # 2022-01, 2023-05, 2024-11
    assert label.label is Label.OPPORTUNISTIC and label.rule == "opportunistic: traded in each of 2022, 2023, 2024, never in the same month every year"

def test_a_missing_year_is_unclassified():                       # 2022-03, 2024-03
    assert label.label is Label.UNCLASSIFIED and label.rule == "unclassified: no trade in 2023"

def test_a_plan_trade_is_routine_whatever_the_history():          # opportunistic history, trade.plan=True
    assert label.label is Label.ROUTINE and label.rule == "plan: the 10b5-1 box is checked"

def test_trades_either_side_of_new_year_use_their_own_years():   # trades 2025-12-20 and 2026-01-05; history makes 2022-24 routine, 2023-25 opportunistic
    assert [l.label for l in labels] == [Label.ROUTINE, Label.OPPORTUNISTIC]

def test_history_is_keyed_on_the_owner_cik_not_the_name():       # same CIK, names "COOK TIMOTHY D" and "Cook Tim"
    assert label.label is Label.ROUTINE

def test_another_insiders_trades_do_not_count():                 # same months, other CIK
    assert label.label is Label.UNCLASSIFIED

def test_buys_and_sales_both_count_as_history(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/form4/test_classify.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `label_trades` (group history by `owner_cik` once; per trade, `months_by_year` over the `years` years before its year; routine when one month is present in all of them).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(insider_trades): routine/opportunistic labels per trade (Task 4)`

---

### Task 5: Window, overrides and a stock's signal class (pure)

**Files:**
- Create: `strategies/insider_trades/form4/signal.py`
- Test: `tests/strategies/insider_trades/form4/test_signal.py`

**Interfaces:**
- Consumes: `Labelled`, `Label` (Task 4).
- Produces:
  - `class SignalClass(StrEnum)`: `BUYING = "buying"`, `SELLING = "selling"`, `NONE = "none"`.
  - `@dataclass(frozen=True, slots=True) class Override`: `insider_id: str` (owner CIK), `from_label: Label`, `to_label: Label`, `reason: str`.
  - `window_trades(labelled: Sequence[Labelled], *, today: date, window_days: int) -> list[Labelled]` — `today - timedelta(window_days) <= filed < today`.
  - `apply_overrides(labelled: Sequence[Labelled], overrides: Sequence[Override]) -> list[Labelled]` — each override relabels that insider's trades whose label is `from_label` and that are NOT plan trades; the new `rule` is `f"override: {reason}"`.
  - `@dataclass(frozen=True, slots=True) class InsiderSummary`: `insider_id`, `name`, `role`, `bought: dict[Label, Decimal]`, `sold: dict[Label, Decimal]`, `max_buy_fraction: Decimal` (largest single OPPORTUNISTIC purchase `holding_fraction`, 0 if none), `sell_fraction: Decimal` (sum of OPPORTUNISTIC sales' `holding_fraction`), `labelled: tuple[Labelled, ...]`.
  - `@dataclass(frozen=True, slots=True) class StockFacts`: `ticker`, `today: date`, `window_start: date`, `insiders: tuple[InsiderSummary, ...]` (by total dollars, largest first); property `accessions -> frozenset[str]`.
  - `summarise(ticker: str, window: Sequence[Labelled], *, today: date, window_start: date) -> StockFacts`.
  - `signal_class(facts: StockFacts, *, min_buy_fraction: Decimal, min_sell_fraction: Decimal) -> SignalClass`.

Class rule (spec "A stock's signal class"): BUYING when any OPPORTUNISTIC purchase has `holding_fraction >= min_buy_fraction`; else SELLING when opportunistic dollars sold > opportunistic dollars bought AND some insider's `sell_fraction >= min_sell_fraction`; else NONE. BUYING wins when both hold. (`sell_fraction` as a sum of per-trade fractions is a documented approximation: 10 of 100 then 9 of 90 reads 0.20, not 0.19.)

- [ ] **Step 1: Write the failing tests**

```python
def test_the_window_is_the_30_days_before_today_by_filing_date():   # filed 2025-09-14 kept, 2025-09-13 dropped, 2025-10-14 (today) dropped; today 2025-10-14

def test_one_opportunistic_buy_above_the_threshold_is_buying_even_against_bigger_sales():
    # opp P $50k at fraction 0.02 + opp S $2M at fraction 0.30 -> BUYING

def test_a_negligible_opportunistic_buy_does_not_count():           # fraction 0.005 < 0.01 -> NONE

def test_heavy_opportunistic_selling_is_selling():                   # opp S fraction 0.12 -> SELLING

def test_light_opportunistic_selling_is_none():                      # opp S fraction 0.05 -> NONE

def test_routine_and_unclassified_dollars_never_move_the_class():    # routine P fraction 0.5, unclassified S 0.9 -> NONE

def test_an_override_relabels_one_insiders_trades_of_one_label_but_never_a_plan_trade():
    # insider X: opp S (plan=False) and routine S (plan=True); Override(X, ROUTINE, OPPORTUNISTIC, "...") changes nothing;
    # Override(X, OPPORTUNISTIC, ROUTINE, "plan named in footnote F2") -> no opportunistic trade left -> NONE; rule == "override: plan named in footnote F2"

def test_the_summary_splits_dollars_by_label():
    assert summary.bought == {Label.OPPORTUNISTIC: Decimal("50000.00")} and summary.sold[Label.ROUTINE] == Decimal("120000.00")
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/form4/test_signal.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** the six names above.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(insider_trades): signal window, overrides and signal class (Task 5)`

---

### Task 6: `Form4Source` (the only SEC I/O) and the smoke script — CHECKPOINT

**Files:**
- Create: `strategies/insider_trades/form4/source.py`, `scripts/tests/smoke_insider_trades.py`
- Modify: `fundamentals/sec.py` (+ `older_pages`, moved from `strategies/earnings_drift/events.py`, which now imports it from there; its test is unchanged), `fundamentals/edgar_client.py` (+ `get_submissions_page`), `fundamentals/CLAUDE.md`, root `CLAUDE.md` (commands list: the smoke script)
- Test: `tests/strategies/insider_trades/form4/test_source.py`, `tests/fundamentals/test_edgar_client.py` (one test for the page getter)

**Interfaces:**
- Consumes: `parse_form4`, `raw_form4_document` (Task 3), `SecEdgarClient`.
- Produces:
  - `SecEdgarClient.get_submissions_page(name: str) -> dict[str, Any]` — `f"{SEC_DATA_BASE_URL}/submissions/{name}"`, cached forever at `("submissions", name)`.
  - `@dataclass(frozen=True, slots=True) class LoadReport`: `loaded: tuple[str, ...]`, `failed: dict[str, str]`, `ignored_forms: int` (4/A and 5 seen).
  - `@dataclass(frozen=True, slots=True) class KnownInsiderData`: `filings: dict[str, list[Form4Filing]]` (ticker → filings filed strictly before the market date of `as_of`, oldest first), `failed: dict[str, str]`.
  - `class Form4Source`: `__init__(client: SecEdgarClient, watchlist: Sequence[str])`; `load(*, as_of: datetime, since: date) -> LoadReport` (per ticker: `ticker_to_cik`; `get_submissions_payload(cik, as_of=as_of, max_age_days=0)`; `older_pages(recent, since)` via `get_submissions_page`; rows of form exactly `"4"` with `filingDate >= since`; XML via `get_filing_text(cik, accession, raw_form4_document(primary))` and `parse_form4`. A `FundamentalsError` or `InsiderDataError` fails that ticker only, and the ticker's earlier filings are kept out of the result: a failed ticker is ALL or nothing); `known(as_of: datetime) -> KnownInsiderData`; `loaded_on: date | None` (market date of the last load, for the pipeline's once-a-day rule).

- [ ] **Step 1: Write the failing tests** with a hand-written fake client (`ticker_to_cik`, `get_submissions_payload`, `get_submissions_page`, `get_filing_text` returning fixture XML):

```python
def test_only_form_4_filed_since_the_history_start_is_read_across_recent_and_older_pages(): ...
def test_amendments_and_form_5_are_counted_not_read():  assert report.ignored_forms == 2
def test_known_keeps_filings_filed_strictly_before_the_market_date():
    # filed 2025-10-13 kept, 2025-10-14 dropped for as_of = et(2025, 10, 14, 10)
def test_the_raw_xml_is_fetched_not_the_xsl_rendering():  assert client.text_requests == [("320193", "0001-25-000001", "wk-form4_1.xml")]
def test_a_ticker_that_fails_is_reported_and_absent_while_the_others_load():
    # get_submissions_payload raises FundamentalsError("403") for "JPM" -> report.failed == {"JPM": "403"}; known(...).filings has no "JPM" key; known(...).failed == {"JPM": "403"}
def test_a_throttle_page_instead_of_xml_fails_the_ticker_not_the_load(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/form4/test_source.py tests/fundamentals -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `get_submissions_page`, move `older_pages`, implement `Form4Source`.
- [ ] **Step 4: Run** `uv run pytest tests/fundamentals tests/strategies/earnings_drift tests/strategies/insider_trades -q` — Expected: PASS.
- [ ] **Step 5: Write `scripts/tests/smoke_insider_trades.py`** (read-only, needs `SEC_EDGAR_USER_AGENT`, loads `env/.env` like the other smoke scripts): for `AAPL` and `JPM`, `Form4Source.load(as_of=<now>, since=<Jan 1 three years back>)`, then print the load report, the count of P/S trades per year, five labelled trades of the last 90 days with their rule, and the path of the first cached raw XML file.
- [ ] **Step 6: CHECKPOINT (needs the network; stop and ask the user to run it if the sandbox cannot reach sec.gov).** Run `uv run python scripts/tests/smoke_insider_trades.py`. Expected: both tickers load, at least one `S` trade parses with a price, the cached file is raw `ownershipDocument` XML (not HTML). Copy two real XML files (one sale, one with several owners or a 10b5-1 box if found) into `tests/strategies/insider_trades/form4/fixtures/real_*.xml`, add a parse test for each, and fix the parser against anything real that differs from the synthetic fixtures. Record what was verified in the strategy `CLAUDE.md` in Task 11.
- [ ] **Step 7: Commit** — `feat(insider_trades): Form4Source over SEC submissions and Form 4 XML, smoke script (Task 6)`

---

### Task 7: `InsiderFacts` and the research tools

**Files:**
- Create: `strategies/insider_trades/facts.py`, `strategies/insider_trades/tools.py`
- Test: `tests/strategies/insider_trades/test_insider_tools.py`

**Interfaces:**
- Consumes: `Form4Source.known` (through a `KnownSource` Protocol: `known(as_of: datetime) -> KnownInsiderData`), `label_trades`, `window_trades`, `summarise` (Tasks 4–5), `InsiderParams`.
- Produces:
  - `class InsiderFacts`: `__init__(strategy: Strategy, source: KnownSource, params: InsiderParams)`; `today() -> date` (market date of `strategy.clock.now()`); `known() -> KnownInsiderData` (cached per market date); `labelled(ticker: str) -> list[Labelled]` (every known trade of the ticker, labelled against all known trades of the ticker); `window(ticker: str) -> list[Labelled]`; `facts(ticker: str) -> StockFacts` (code labels); `filing(accession: str) -> Form4Filing | None` (known filings only). `labelled`/`window`/`facts` raise `InsiderDataError` for a failed ticker.
  - `insider_research_tools(strategy: Strategy, facts: InsiderFacts, on_fetch: Callable[[str], None]) -> list[Callable[..., dict[str, Any]]]` returning `get_insider_trades(ticker: str)` and `read_form4(accession_number: str)`; `get_insider_trades` calls `on_fetch(ticker)` after a successful answer (the recorder uses it, Task 8). Both use `RunMemo` like `congress_research_tools`, and both re-check the clock gate on what they return.

Payload shapes (lean; whole dollars as `int`, fractions rounded to 4 decimals, dates ISO):
- `get_insider_trades` → `{"ticker", "as_of", "window_start", "insiders": [{"insider_id", "name", "role", "bought": {label: int}, "sold": {label: int}, "max_buy_fraction", "sell_fraction", "months_by_year": {"2022": [3, 9], ...}, "trades": [{"accession", "filed", "traded", "code", "shares", "price", "label", "rule"}]}], "count"}`; at most 10 trades per insider, then `"more_trades": n`; a ticker not on the watchlist → `{"error": "<T> is not on the watchlist: <list>"}`.
- `read_form4` → `{"accession", "filed", "owners": [{"name", "role"}], "attributed_to", "co_filers", "plan", "rows": [{"code", "traded", "shares", "price", "owned_after"}], "footnotes": {"F1": text[:500]}}`; an unknown or not-yet-filed accession → `{"error": "<acc> is not a known Form 4: call get_insider_trades for the filings filed before <today>"}`.

- [ ] **Step 1: Write the failing tests** (fake `KnownSource`, `FakeBroker`/`FakeClock`):

```python
def test_get_insider_trades_returns_code_labels_with_their_evidence(): ...
def test_a_filing_dated_today_is_invisible_even_if_the_source_leaks_it(): ...   # source returns it anyway
def test_read_form4_refuses_an_accession_filed_today(): ...
def test_footnotes_are_capped_at_500_characters(): ...
def test_more_than_ten_trades_per_insider_are_cut_with_a_count(): ...
def test_a_ticker_off_the_watchlist_is_refused(): ...
def test_a_failed_ticker_answers_with_an_error_not_an_exception(): ...
def test_on_fetch_is_called_with_the_ticker_only_on_success(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/test_insider_tools.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `facts.py` and `tools.py` (no `from __future__ import annotations` in `tools.py`).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(insider_trades): per-day insider facts and the researcher's tools (Task 7)`

---

### Task 8: Hand-off: `submit_note`, `submit_target`, `submit_trade_report`

**Files:**
- Create: `strategies/insider_trades/handoff.py` (no `from __future__ import annotations`)
- Test: `tests/strategies/insider_trades/test_insider_handoff.py`

**Interfaces:**
- Consumes: `strategies.common.handoff` (Task 1), `Override`, `Label`, `SignalClass`, `StockFacts` (Task 5), `InsiderParams`.
- Produces:
  - `@dataclass(frozen=True, slots=True) class Note`: `summary: str`, `overrides: tuple[Override, ...]`.
  - `@dataclass(frozen=True, slots=True) class TargetWeight`: `ticker: str`, `weight: float`, `reason: str | None`.
  - `validate_note(summary: Any, overrides: Any, *, facts: StockFacts, reason_max_chars: int) -> Note`.
  - `validate_target(raw: Any, *, classes: Mapping[str, SignalClass], params: InsiderParams) -> list[TargetWeight]`.
  - `class InsiderRecorder(params: InsiderParams)`: `expect_note(facts: StockFacts)`; `facts_fetched(ticker: str)` (the tools' `on_fetch`); `expect_trade(classes: Mapping[str, SignalClass], on_target: Callable[[dict[str, float]], dict[str, Any]], orders: Callable[[], Mapping[str, OrderView]])` (arms TARGET; a valid target calls `on_target(weights)` and merges its result into `{"status": "recorded", **result}`, then arms REPORT with `orders`); `expect_report(orders)` (pending retries: REPORT only); `submitted: bool` (the armed stage's FINAL submission is in: the note, or the report); `target: list[TargetWeight] | None`; `note: Note | None`; `report: list[OrderReport] | None`; `missing_tool: str` (`"submit_note"`, `"submit_target"` or `"submit_trade_report"`: what a forced retry forces); `last_error: str | None`; `submit(stage: str, raw: Any) -> dict[str, Any]`.
  - `submit_tools(recorder) -> dict[str, Callable]` keyed `submit_note(summary: str, overrides: list[dict[str, Any]])`, `submit_target(weights: list[dict[str, Any]])`, `submit_trade_report(orders: list[dict[str, Any]])`.

Rules (each failure is `{"error": ...}` naming what is allowed):
- Note: refused until `facts_fetched(facts.ticker)` ran (`"call get_insider_trades for <T> first"`); `summary` non-blank and ≤ `reason_max_chars`; each override has `insider_id` among `facts.insiders`, `from_label` that insider has on at least one NON-plan trade (plan-only → `"<id>'s <label> trades are 10b5-1 plan trades: they stay routine"`), `to_label != from_label`, a reason. An empty override list is valid.
- Target: exactly the watchlist (missing / extra / repeated tickers refused, listed); each weight in `[min_weight - tol, max_position_weight + tol]`; sum ≤ `max_total_weight + tol`; ordering: `min(buying weights) >= max(none weights) - tol` and `max(selling weights) <= min(none weights) + tol` (skipped when a class is empty); all NONE ⇒ every `|w - even_weight| <= tol`; a reason is required when `|w - even_weight| > tol`. `tol = params.weight_tolerance`.
- Report: `validate_report` from common.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_note_before_the_facts_were_fetched_is_refused(): ...
def test_an_override_of_plan_only_trades_is_refused(): ...
def test_an_override_needs_a_reason_and_a_known_insider(): ...
def test_ten_even_weights_with_float_noise_are_accepted():   # [0.095] * 10 sums to 0.9500000000000001
def test_a_weight_within_tolerance_of_even_needs_no_reason():  # 0.0954 vs 0.095
def test_a_buying_stock_below_a_none_stock_is_refused(): ...
def test_a_selling_stock_above_a_none_stock_is_refused(): ...
def test_with_no_signal_anywhere_the_weights_must_be_even(): ...
def test_every_watchlist_stock_must_get_a_weight_between_floor_and_cap(): ...
def test_a_valid_target_arms_the_report_and_returns_what_on_target_returns():
    assert tools["submit_target"](weights) == {"status": "recorded", "shortfalls": [...]}
    assert recorder.missing_tool == "submit_trade_report" and not recorder.submitted
def test_the_target_is_accepted_once(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/test_insider_handoff.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(insider_trades): note, target and trade-report hand-offs (Task 8)`

---

### Task 9: State store

**Files:**
- Create: `strategies/insider_trades/state.py`
- Test: `tests/strategies/insider_trades/test_insider_state.py`

**Interfaces:**
- Produces: `state_path(project_root: Path, mode: TradingMode) -> Path` (`data/insider_trades_state_<mode>.json`); `PendingTrade(target: dict[str, float], days: int = 0)`; `@dataclass(frozen=True, slots=True) class InsiderState`: `processed: list[str]` (accessions of every known P/S filing at the last completed run), `in_window: dict[str, list[str]]` (ticker → accessions in its window then), `notes: dict[str, dict[str, Any]]` (ticker → `{"summary": str, "class": str, "overrides": [{"insider_id", "from_label", "to_label", "reason"}], "as_of": iso}`), `target: dict[str, float]`, `traded: list[str]`, `pending_trade: PendingTrade | None`, `abandoned_streak: int`, `last_run: str | None`; `StateStore(path)` with `load() -> InsiderState`, `save(state)`, `wipe()` — same contract as congress' (`STATE_VERSION = 1`, atomic write, invalid or corrupt file ⇒ empty state with a warning, never raises).

- [ ] **Step 1: Write the failing tests:** round trip of a full state; a missing file, a corrupt file, a wrong version and a note with a non-string summary each load as `InsiderState()`; `wipe()` with no file is safe.
- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/test_insider_state.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** (mirror `congress_trades/state.py`'s validators).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(insider_trades): state store (Task 9)`

---

### Task 10: The pipeline

**Files:**
- Create: `strategies/insider_trades/pipeline.py`
- Test: `tests/strategies/insider_trades/test_insider_pipeline.py` (a `Rig` like `test_congress_pipeline.py`'s: `FakeBroker`, scripted agent `Handle`s, a fake source with `load`/`known`/`loaded_on`)

**Interfaces:**
- Consumes: everything above; `AgentLookup`/`AgentLike` Protocols copied from congress' pipeline.
- Produces: `AGENT_RESEARCHER = "researcher"`, `AGENT_TRADER = "trader"`, `AGENT_TRADER_RETRY = "trader_retry"`; `RunOutcome(completed: bool, abandoned_streak: int)`; `class InsiderPipeline(*, strategy, params, source, facts, agents, recorder, state, run_log, desk, load_window: Callable[[], tuple[datetime, date]])` with `run() -> RunOutcome`. `load_window()` returns `(as_of, since)` for `Form4Source.load` (Task 11 supplies it).

Behaviour of `run()` (the spec's "Daily check" and "Agents" sections; log lines prefixed `[insider_trades]`, `[researcher]`, `[trader]`; one `runs.jsonl` record per tick with `outcome` among `nothing_new`, `completed`, `completed_pending`, `pending_complete`, `pending_retry`, `pending_gave_up`, `abandoned`):
1. **Load once per market date** (`source.loaded_on != today`): `report = source.load(as_of, since)`. Backtest (`strategy.is_backtesting`): any failed ticker ⇒ the run is abandoned at stage `"filings"` and the next tick loads again. Paper/live: failed tickers are logged; every ticker failing ⇒ abandoned at `"filings"`.
2. **Changes** per loaded ticker: `current = {accessions of its known P/S filings}`; `win = facts.facts(t).accessions`; changed when `current - processed` is non-empty or `state.in_window.get(t, []) - win` is non-empty; every ticker is changed when `state.last_run is None`. A failed ticker is never changed and keeps its stored note, class and `in_window`.
3. No change: pending ⇒ retry (below); else `nothing_new`, no agent.
4. Unchanged tickers reuse their stored note and class (their window is the same). Per changed ticker, in watchlist order: empty window ⇒ code note `{"summary": "no open-market insider trade in the last <n> days", "class": "none", "overrides": []}`; else arm `expect_note(facts)`, run the researcher (`_run_stage`, the congress loop: once, then once more with `force_tool=recorder.missing_tool`, else `RunAbandoned`) with context `{"current_datetime", "ticker", "window_days", "previous_note"}`, then class = `signal_class(summarise(apply_overrides(window, note.overrides)), ...)`.
5. Trader: `desk.begin_run(None, state.traded)`; `recorder.expect_trade(classes, on_target, desk.orders_view)` where `on_target(w)` calls `desk.set_target(w)` and returns `{"shortfalls": [{"symbol", "kind", "dollars", "detail"}]}` from `desk.audit()`; context `{"current_datetime", "stocks": [{"ticker", "class", "summary", "insiders": [{"name", "role", "opportunistic_bought", "opportunistic_sold", "max_buy_fraction", "sell_fraction"}], "current_weight", "previous_weight"}], "constraints": {"min_weight", "max_weight", "max_total_weight", "even_weight"}}`. The stage ends when the report is valid. No valid target ⇒ abandoned at `"trader"`. Target but no report ⇒ logged, the run still completes (pending decided by the audit).
6. Audit ⇒ `pending_trade = None if audit.complete else PendingTrade(target)`; save `processed`, `in_window`, `notes`, `target`, `traded`, streak 0, `last_run`.
7. **Pending retry** = congress' `_retry_pending` with `AGENT_TRADER_RETRY`, `desk.begin_run(stored target, traded)`, `recorder.expect_report(desk.orders_view)`, `max_trade_retries`.
8. **Abandoned** = congress' bookkeeping (streak + 1, processed not moved, desk `traded` merged, pending days + 1 when a retry failed).

- [ ] **Step 1: Write the failing tests**

```python
def test_first_run_researches_every_ticker_with_trades_and_builds_the_book(): ...
def test_a_quiet_day_runs_no_agent_and_sends_no_order(): ...
def test_a_new_routine_filing_still_wakes_the_researcher_for_that_ticker_only(): ...
def test_a_signal_expiring_reruns_the_trader_with_a_code_note_and_no_researcher(): ...
def test_an_override_changes_the_class_the_trader_must_respect(): ...
def test_the_trader_cannot_order_before_its_target_is_recorded(): ...
def test_unfilled_orders_leave_a_pending_trade_retried_by_trader_retry_only(): ...
def test_a_researcher_that_never_submits_abandons_the_run_and_keeps_the_filings_new(): ...
def test_in_a_backtest_a_failed_ticker_abandons_the_run_and_the_next_tick_loads_again(): ...
def test_in_paper_a_failed_ticker_keeps_its_note_and_the_others_run(): ...
def test_the_source_is_loaded_once_per_market_date(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/test_insider_pipeline.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat(insider_trades): pipeline, act only on change (Task 10)`

---

### Task 11: Prompts, the strategy, registration and docs

**Files:**
- Create: `strategies/insider_trades/prompts.py`, `agent_insider_trades.py`, `strategies/insider_trades/CLAUDE.md`
- Modify: `strategies/insider_trades/__init__.py` (export `InsiderTradesStrategy`), `utils/strategy_factory.py` (`Strategies.INSIDER_TRADES`, `_build_insider_trades`), `strategies/CLAUDE.md`, root `CLAUDE.md` (strategy list in the nested-docs bullet), `README.md` (a short "insider_trades" note: `env/.env.insider_trades.live` with `BROKER=tastytrade`, `.paper` with `BROKER=alpaca`, `.backtesting` with `BROKER=tastytrade`; no new variable — `SEC_EDGAR_USER_AGENT` is already documented)
- Test: `tests/strategies/insider_trades/test_insider_strategy.py`, `test_insider_prompts.py`, `tests/test_strategy_factory.py`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: `prompts.researcher_system() -> str`, `prompts.trader_system(params: InsiderParams) -> str`, `RESEARCHER_TASK`, `TRADER_TASK`, `TRADER_RETRY_TASK`, `retry_prompt(tool: str, error: str) -> str`; `class InsiderTradesStrategy(Strategy)` with `sleeptime = "1D"`, `iteration_start_time = time(10, 0)`, `parameters` = YEAR window, `"benchmark_symbol": "SPY"`, `"warmup_trading_days": 10`, `"budget": 10000`; `__init__(broker, *, mode, settings: InsiderParams | None = None, source: Form4Source-like | None = None, **kwargs)`.

Prompt content (English, congress style, for a local model): the researcher's one job (verify code's labels for ONE ticker, call `get_insider_trades` first, read a Form 4 only to check a footnote or a doubtful row, override only with a reason, plan trades stay routine, never recompute dollars) and its submit contract; the trader's job (one target for every watchlist stock between the floor and cap, even by default, more for `buying`, less for `selling`, reason for any weight off even naming a note fact; then orders through `place_order` sells first, `check_orders`, `submit_trade_report`), the desk's rules stated as in congress' `trader_system`, and the English rule. Neither prompt states how often the strategy runs.

`initialize()`: wipe state in a backtest; build `SecEdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT", ""), project_root / "cache" / "sec")` (a `ConfigurationError` ⇒ `FatalStrategyError`), `Form4Source`, `InsiderFacts`, `InsiderRecorder`, `TradeDesk(self, params, log_tag="insider_trades")`; agents: `researcher` (`insider_research_tools(self, facts, recorder.facts_fetched)` + `submit_note`), `trader` (`get_account_balance`, `get_positions`, `get_last_price`, `submit_target`, desk tools, `submit_trade_report`), `trader_retry` (the same without `submit_target`), all at `agent_temperature`. `load_window()`: backtest ⇒ `(self._backtest_end, date(self._backtest_start.year - history_years, 1, 1))` (both set in `run_backtesting` from the merged arguments, as earnings_drift sets `_backtest_end`); paper/live ⇒ `(clock now, date(now.year - history_years, 1, 1))`. `on_trading_iteration` = congress' (`FatalStrategyError` at `max_consecutive_abandoned` in a backtest only). `run_backtesting`: `YahooBacktestData`, `timestep="day"`, `preload_assets` = the watchlist + SPY, `benchmark="SPY"`, `agent_telemetry=True`.

- [ ] **Step 1: Write the failing tests** — mirror `test_congress_strategy.py` for: ticks at 10:00 once a session; three agents with exactly their tools (only `trader`/`trader_retry` have `place_order`; only `trader` has `submit_target`); each prompt requires English and names its submit tool(s); temperature from settings; a missing `SEC_EDGAR_USER_AGENT` refuses to start; backtest state wiped, paper kept; three abandoned runs abort a backtest, never paper/live; `run_backtesting` passes the YEAR window, Yahoo daily data, the watchlist + SPY preloaded; `load_window()` in a backtest starting 2025-09-29 gives `since == date(2022, 1, 1)`. In `tests/test_strategy_factory.py` add `"insider_trades"` to the names list and a build test like congress'.
- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades tests/test_strategy_factory.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement**, then write `strategies/insider_trades/CLAUDE.md` (architecture line per module, and gotchas: filing-date rule, officers/directors only and joint filings once, per-trade labels and the plan rule, signal classes, act only on change, the trader's two submits and the gated desk, what Task 6's checkpoint verified against real filings).
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check && uv run pyright` — Expected: all pass.
- [ ] **Step 5: Commit** — `feat(insider_trades): strategy, prompts, registration and docs (Task 11)`

---

### Task 12: Backtest integration and the real one-year run — CHECKPOINT

**Files:**
- Test: `tests/strategies/insider_trades/test_insider_backtest.py` (like `test_congress_backtest.py`: real `BacktestBroker`, `FakeBacktestDataSource`, a fake Form 4 source with filings on known dates, scripted agents)

- [ ] **Step 1: Write the tests**

```python
def test_the_bot_checks_every_session_but_acts_only_on_changes(): ...      # filings on 3 dates + one expiry -> agents run on exactly those 4 sessions
def test_the_agents_never_see_a_filing_before_its_filing_date(): ...       # the researcher's get_insider_trades on each day returns only filings filed before it
def test_orders_go_out_once_and_fill_on_a_later_bar(): ...
def test_a_dead_llm_aborts_a_real_backtest_instead_of_writing_a_flat_report(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/insider_trades/test_insider_backtest.py -v` — Expected: PASS (they exercise finished code; a failure here is a bug in Tasks 10–11, fixed there).
- [ ] **Step 3: Run the full suite** `uv run pytest -q && uv run ruff check && uv run pyright` — Expected: all pass.
- [ ] **Step 4: Commit** — `test(insider_trades): backtest integration (Task 12)`
- [ ] **Step 5: CHECKPOINT (needs the network and the LLM server; ask the user to run it if unavailable).** `uv run agent insider_trades backtesting`. Expected: the run completes over 2025-09-29 → 2026-10-16 and writes a report; `runs.jsonl` shows `nothing_new` on most days and `completed` on change days; no `abandoned` streak reaches 3; the logs show researcher notes followed by a trader target and a trade report on each change day. Report to the user: return vs SPY over the window and over Oct–Nov 2025, the number of change days, and how many stocks were ever `buying` / `selling`.
