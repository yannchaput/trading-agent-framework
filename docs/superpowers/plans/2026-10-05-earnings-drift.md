# earnings_drift (PEAD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new strategy `earnings_drift` that buys stocks after a strong, held, positive earnings reaction, with an
LLM agent deciding buy/skip, the trailing stop and the exits, behind code guardrails, plus a code-only baseline mode.

**Architecture:** Pure modules (events, surprise, reaction, screening, fact sheet) feed a `Scanner` that runs once
per session right after the close and builds today's candidates from SEC 8-K item 2.02 filings and Benzinga
headlines. A `Desk` is the only order code: the agent's `buy` / `set_trailing_stop` / `sell` tools call it, it places
a trailing stop on every entry fill, cancels and waits before any exit sell, and enforces the guardrails (each logs a
warning). Orders sent at the close fill at the next open, in backtests and live alike.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff, pandas (daily bars), httpx (via `SecEdgarClient`), LangChain (via
`AgentManager`, lazily).

**Spec:** `docs/superpowers/specs/2026-10-05-earnings-drift-design.md`

## Global Constraints

- Package: `src/trading_agent_framework/strategies/earnings_drift/`; strategy names in `main.py`: `"earnings_drift"`
  (agent mode, `PredefinedWindow.YEAR`) and `"earnings_drift_baseline"` (`agent_enabled=False`, `PredefinedWindow.SEMI_DECADE`).
- The daily cycle runs in `after_market_closes` (`minutes_after_closing = 0`); `sleeptime = "1D"`; `on_trading_iteration`
  only raises a pending `FatalStrategyError`. Paper/live first `strategy.sleep(live_bar_delay_seconds)` (300).
- Trailing stops: `trail_percent` in percent (8 = 8%), bounds `[3, 15]`, default 8, `time_in_force="gtc"`, tighten-only.
- Entries and exits: market orders, `time_in_force="day"`, sent after the close, filling at the next open.
- Every guardrail logs `log_warning("guardrail <name>: ...")` with `<name>` in `max_hold`, `stop_backstop`, `order_limits`,
  `orphan`, `baseline`.
- Gate order, first failure wins: `already_held`, `no_surprise_data`, `eps_miss`, `no_bars`, `weak_reaction`, `faded`,
  `low_volume`, `illiquid`. Scanner-only reasons: `no_news`.
- State file `data/earnings_drift_state_<mode>.json` (`STATE_VERSION = 1`), wiped in backtesting only. Run logs
  `trades.jsonl` and `decisions.jsonl` in `logs/<strategy>/<mode>/<run_id>/`.
- Money stays `Decimal`; bar maths (`reaction.py`) is float64 on `Bars.df`; agent-context dicts use floats for JSON only.
  No new float boundary in order sizing.
- No `time.sleep` / `datetime.now` in strategy code: `strategy.clock`, `strategy.sleep`.
- Tests never touch the network; hand-written fakes (`tests/fakes.py`, `tests/backtesting/fakes.py`), no `MagicMock`.
- `tools.py` has no `from __future__ import annotations` (the agent layer reads real annotations). Tool docstrings are one line.
- Run tests with `uv run pytest`, lint with `uv run ruff check`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Two agent tool calls in one model response run on parallel LangGraph worker threads.** Two `buy` calls at once
   must not both pass the `max_positions` / `max_quantity` checks. The desk holds one re-entrant lock in every public
   method; Task 9 adds `test_two_buys_on_parallel_threads_respect_max_positions`.
2. **A stop that already filled when the agent asks to sell or tighten** (the fill hook not delivered yet): nothing
   else must be sold. Task 10 adds `test_sell_after_the_stop_filled_sells_nothing`.
3. **A live restart with a lost or stale state file:** a position this strategy once traded is adopted, and an active
   sell order already working for it is adopted as its stop rather than doubled (a second stop would be refused and
   fall back to a market sell). Task 11 adds `test_an_orphan_with_a_working_sell_adopts_it_as_its_stop`.
4. **The benchmark has no bar for today** (live data delay, a data gap): no candidates, but max-hold counting must
   still count today. Task 8 adds `test_no_benchmark_bar_today_gives_no_candidates_and_counts_today`.
5. **A model that sends a fractional, negative, NaN or string quantity, or a trail outside [3, 15]:** refused with
   `{"error": ...}`, never an exception into LangGraph. Task 9 parametrizes
   `test_buy_refuses_bad_quantities_and_trails` over `2.5`, `0`, `-3`, `float("nan")`, `"ten"`, trail `2.9`, `15.1`.

---

### Task 1: Package and parameters

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/__init__.py`
- Create: `src/trading_agent_framework/strategies/earnings_drift/parameters.py`
- Test: `tests/strategies/earnings_drift/test_drift_params.py`

**Interfaces:**
- Produces: `DriftParams` (frozen dataclass, fields below, `ValueError("DriftParams: ...")` on invalid values).

- [ ] **Step 1: Write the failing test**

```python
# tests/strategies/earnings_drift/test_drift_params.py
from __future__ import annotations

import pytest

from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams


def test_defaults_match_the_spec() -> None:
    params = DriftParams()
    assert params.agent_enabled is True
    assert (params.max_positions, params.max_holding_sessions) == (8, 10)
    assert (params.min_trail_percent, params.default_trail_percent, params.max_trail_percent) == (3.0, 8.0, 15.0)
    assert (params.min_abnormal_pct, params.min_hold_ratio, params.min_close_location, params.min_rel_volume) == (0.03, 0.5, 0.5, 2.0)
    assert (params.min_price, params.min_dollar_volume) == (10.0, 20_000_000.0)
    assert params.live_bar_delay_seconds == 300.0
    assert params.tool_budget_per_item == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_positions": 0}, "max_positions"),
        ({"max_holding_sessions": 0}, "max_holding_sessions"),
        ({"min_trail_percent": 9.0}, "trail"),
        ({"default_trail_percent": 16.0}, "trail"),
        ({"min_trail_percent": 0.0}, "trail"),
        ({"min_hold_ratio": 1.5}, "min_hold_ratio"),
        ({"min_close_location": -0.1}, "min_close_location"),
        ({"min_rel_volume": float("nan")}, "finite"),
        ({"volume_baseline_sessions": 1}, "volume_baseline_sessions"),
        ({"bars_lookback_sessions": 20}, "bars_lookback_sessions"),
        ({"news_symbols_per_call": 0}, "news_symbols_per_call"),
        ({"live_bar_delay_seconds": -1.0}, "live_bar_delay_seconds"),
        ({"sec_hollow_fraction": 1.5}, "sec_hollow_fraction"),
        ({"agent_temperature": 3.0}, "agent_temperature"),
    ],
)
def test_invalid_values_are_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DriftParams(**overrides)  # type: ignore[arg-type]


def test_zero_live_delay_and_no_temperature_are_allowed() -> None:
    DriftParams(live_bar_delay_seconds=0.0, agent_temperature=None)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_params.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.earnings_drift'`

- [ ] **Step 3: Write the package and the parameters**

```python
# src/trading_agent_framework/strategies/earnings_drift/__init__.py
"""earnings_drift: post-earnings announcement drift with an agent that trades (spec 2026-10-05)."""
```

```python
# src/trading_agent_framework/strategies/earnings_drift/parameters.py
"""`DriftParams`: every threshold of the earnings_drift strategy in one frozen dataclass (spec §9)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DriftParams:
    agent_enabled: bool = True  # False: baseline mode, every gated candidate is bought at default_trail_percent
    max_positions: int = 8
    max_holding_sessions: int = 10  # sessions from the entry fill through the review, both included
    min_trail_percent: float = 3.0  # trailing stop bounds, in percent (8 = 8%), as Alpaca and IBKR take them
    default_trail_percent: float = 8.0
    max_trail_percent: float = 15.0
    min_abnormal_pct: float = 0.03  # reaction-day return minus the benchmark's, as a fraction
    min_hold_ratio: float = 0.5
    min_close_location: float = 0.5
    min_rel_volume: float = 2.0
    min_price: float = 10.0
    min_dollar_volume: float = 20_000_000.0
    volume_baseline_sessions: int = 20
    bars_lookback_sessions: int = 75  # daily bars read per name: the 60-session run-up plus margin
    surprise_lookback_hours: float = 2.0  # news is read from this long before the 8-K's acceptance
    news_symbols_per_call: int = 5
    news_limit: int = 50
    event_lookback_days: int = 10  # older SEC submission pages are read back to the cycle's date minus this
    live_bar_delay_seconds: float = 300.0  # paper/live wait after the close, so the daily bar is final
    cancel_wait_seconds: float = 30.0
    tool_budget_per_item: int = 4  # tool calls per candidate or holding in one agent run (order tools and skip exempt)
    agent_temperature: float | None = 0.3
    max_consecutive_agent_failures: int = 3
    max_consecutive_hollow_scans: int = 3
    sec_hollow_fraction: float = 0.5
    sec_hollow_min_failures: int = 20
    reason_max_chars: int = 300  # a longer reason is cut before it is logged or kept as a thesis

    def __post_init__(self) -> None:
        floats = (
            self.min_trail_percent,
            self.default_trail_percent,
            self.max_trail_percent,
            self.min_abnormal_pct,
            self.min_hold_ratio,
            self.min_close_location,
            self.min_rel_volume,
            self.min_price,
            self.min_dollar_volume,
            self.surprise_lookback_hours,
            self.live_bar_delay_seconds,
            self.cancel_wait_seconds,
            self.sec_hollow_fraction,
        )
        problems = {
            "thresholds must be finite": not all(math.isfinite(value) for value in floats),
            "max_positions must be at least 1": self.max_positions < 1,
            "max_holding_sessions must be at least 1": self.max_holding_sessions < 1,
            "trail percents must satisfy 0 < min <= default <= max <= 100": not (
                0 < self.min_trail_percent <= self.default_trail_percent <= self.max_trail_percent <= 100
            ),
            "min_hold_ratio must be in [0, 1]": not 0 <= self.min_hold_ratio <= 1,
            "min_close_location must be in [0, 1]": not 0 <= self.min_close_location <= 1,
            "min_rel_volume must be above 0": not self.min_rel_volume > 0,
            "min_price and min_dollar_volume must be at least 0": self.min_price < 0 or self.min_dollar_volume < 0,
            "volume_baseline_sessions must be at least 2": self.volume_baseline_sessions < 2,
            "bars_lookback_sessions must exceed volume_baseline_sessions + 1": self.bars_lookback_sessions <= self.volume_baseline_sessions + 1,
            "surprise_lookback_hours must be at least 0": self.surprise_lookback_hours < 0,
            "news_symbols_per_call and news_limit must be at least 1": self.news_symbols_per_call < 1 or self.news_limit < 1,
            "event_lookback_days must be at least 0": self.event_lookback_days < 0,
            "live_bar_delay_seconds and cancel_wait_seconds must be at least 0": self.live_bar_delay_seconds < 0 or self.cancel_wait_seconds < 0,
            "tool_budget_per_item must be at least 1": self.tool_budget_per_item < 1,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None
            and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
            "max_consecutive_agent_failures and max_consecutive_hollow_scans must be at least 1": self.max_consecutive_agent_failures < 1
            or self.max_consecutive_hollow_scans < 1,
            "sec_hollow_fraction must be in (0, 1]": not 0 < self.sec_hollow_fraction <= 1,
            "sec_hollow_min_failures must be at least 1": self.sec_hollow_min_failures < 1,
            "reason_max_chars must be at least 1": self.reason_max_chars < 1,
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ValueError("DriftParams: " + "; ".join(failed))
```

Note the test matches `"trail"` for the trail cases and `"finite"` for the NaN: the messages above contain those words.
`min_rel_volume=nan` trips both "finite" and "min_rel_volume must be above 0"; the joined message contains "finite".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_params.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift tests/strategies/earnings_drift/test_drift_params.py
git commit -m "feat: earnings_drift package and DriftParams (Task 1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Surprise parsing (`surprise.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/surprise.py`
- Test: `tests/strategies/earnings_drift/test_drift_surprise.py`

**Interfaces:**
- Produces:
  - `Surprise(eps_actual: Decimal, eps_estimate: Decimal, sales_actual: Decimal | None = None, sales_estimate: Decimal | None = None)`
    with properties `eps_beat: bool`, `eps_result: str` (`"BEAT"|"MISS"|"IN-LINE"`), `eps_surprise_pct: float | None`,
    `sales_result: str | None`, `sales_surprise_pct: float | None`.
  - `parse_surprise(headline: str) -> Surprise | None`
  - `PickedSurprise(surprise: Surprise, headline: str, created_at: datetime)`
  - `article_time(article: Mapping[str, Any]) -> datetime | None`
  - `articles_for(articles: Sequence[Mapping[str, Any]], symbol: str, *, start: datetime, end: datetime) -> list[Mapping[str, Any]]` (oldest first)
  - `pick_surprise(articles: Sequence[Mapping[str, Any]]) -> PickedSurprise | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_surprise.py
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal as D

import pytest

from trading_agent_framework.strategies.earnings_drift.surprise import Surprise, articles_for, parse_surprise, pick_surprise


@pytest.mark.parametrize(
    ("headline", "eps", "estimate", "sales", "sales_estimate"),
    [
        ("Omnicom Group Q3 EPS $1.77 Beats $1.68 Estimate, Sales $3.44B Beat $3.36B Estimate", "1.77", "1.68", "3440000000", "3360000000"),
        ("XYZ Corp Q2 Adj. EPS $(0.12) Misses $(0.05) Estimate, Sales $120.5M Beat $118M Estimate", "-0.12", "-0.05", "120500000", "118000000"),
        ("ABC Inc Q4 EPS $-0.30 Beats $-0.41 Estimate", "-0.30", "-0.41", None, None),
        ("Small Co FY25 Adj EPS $2.10 In-Line With $2.10 Estimate, Revenue $850K Miss $900K Estimate", "2.10", "2.10", "850000", "900000"),
        ("CORRECTION: Omnicom Group Q3 EPS $1.77 Beats $1.68 Estimate, Sales $3.44B Beat $3.36B Estimate", "1.77", "1.68", "3440000000", "3360000000"),
    ],
)
def test_parse_surprise_reads_the_numbers(headline: str, eps: str, estimate: str, sales: str | None, sales_estimate: str | None) -> None:
    surprise = parse_surprise(headline)
    assert surprise is not None
    assert (surprise.eps_actual, surprise.eps_estimate) == (D(eps), D(estimate))
    assert surprise.sales_actual == (D(sales) if sales else None)
    assert surprise.sales_estimate == (D(sales_estimate) if sales_estimate else None)


@pytest.mark.parametrize("headline", ["XYZ Q3 EPS $1.20 Up From $1.00 YoY", "Omnicom Group: Q3 Earnings Insights", "", "Earnings Scheduled For October 18, 2022"])
def test_headlines_without_an_estimate_are_not_surprises(headline: str) -> None:
    assert parse_surprise(headline) is None


def test_the_result_comes_from_the_numbers_not_the_verb() -> None:
    surprise = parse_surprise("Mislabelled Inc Q1 EPS $1.00 Misses $0.90 Estimate, Sales $10M Beat $12M Estimate")
    assert surprise is not None
    assert surprise.eps_beat is True and surprise.eps_result == "BEAT"
    assert surprise.sales_result == "MISS"


def test_surprise_percentages_and_results() -> None:
    beat = Surprise(D("1.52"), D("1.20"), D("110"), D("100"))
    assert beat.eps_surprise_pct == pytest.approx(0.26667, rel=1e-4)
    assert beat.sales_surprise_pct == pytest.approx(0.10)
    negative = Surprise(D("-0.30"), D("-0.41"))
    assert negative.eps_beat and negative.eps_surprise_pct == pytest.approx(0.26829, rel=1e-4)
    assert Surprise(D("0.10"), D("0")).eps_surprise_pct is None
    assert Surprise(D("2.10"), D("2.10")).eps_result == "IN-LINE"
    assert Surprise(D("1"), D("2")).eps_result == "MISS"
    assert Surprise(D("1"), D("2")).sales_result is None and Surprise(D("1"), D("2")).sales_surprise_pct is None


def _article(headline: str, created_at: str, symbols: list[str]) -> dict[str, object]:
    return {"headline": headline, "created_at": created_at, "symbols": symbols}


def test_articles_for_keeps_the_symbol_and_the_window_oldest_first() -> None:
    articles = [
        _article("late", "2026-09-01T21:00:00+00:00", ["AAA"]),
        _article("early", "2026-09-01T20:00:00+00:00", ["AAA", "BBB"]),
        _article("other symbol", "2026-09-01T20:30:00+00:00", ["BBB"]),
        _article("too old", "2026-09-01T10:00:00+00:00", ["AAA"]),
        _article("no time", "", ["AAA"]),
    ]
    kept = articles_for(articles, "aaa", start=datetime(2026, 9, 1, 18, tzinfo=UTC), end=datetime(2026, 9, 2, tzinfo=UTC))
    assert [a["headline"] for a in kept] == ["early", "late"]


def test_pick_prefers_the_latest_correction_then_the_earliest_parsable() -> None:
    original = _article("AAA Q3 EPS $1.77 Beats $1.68 Estimate, Sales $3.44M Miss $3.36B Estimate", "2026-09-01T20:06:37Z", ["AAA"])
    correction = _article("CORRECTION: AAA Q3 EPS $1.77 Beats $1.68 Estimate, Sales $3.44B Beat $3.36B Estimate", "2026-09-01T20:14:28Z", ["AAA"])
    noise = _article("AAA: Q3 Earnings Insights", "2026-09-01T21:09:47Z", ["AAA"])
    picked = pick_surprise([original, correction, noise])
    assert picked is not None and picked.headline.startswith("CORRECTION:")
    assert picked.surprise.sales_actual == D("3440000000")
    assert picked.created_at == datetime(2026, 9, 1, 20, 14, 28, tzinfo=UTC)
    first = pick_surprise([original, noise])
    assert first is not None and first.headline == original["headline"]
    assert pick_surprise([noise]) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_surprise.py -v`
Expected: FAIL with `ModuleNotFoundError` (`surprise`)

- [ ] **Step 3: Implement `surprise.py`**

```python
# src/trading_agent_framework/strategies/earnings_drift/surprise.py
"""Benzinga's earnings headline, parsed into numbers (spec §3.3). Pure: no I/O, no clock.

BEAT/MISS is computed from the numbers, never read from the headline's verb: the agent gets facts it does not
have to derive (a local model has misread the sign of "actual vs estimate" prints before).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

_NUMBER = r"\$?\(?\$?-?[\d,]*\.?\d+\)?"
_VERB = r"(?:Beats?|Miss(?:es)?|In-?Line\s+With|Inline\s+With|Meets?)"
_EPS = re.compile(
    rf"\b(?:Q[1-4]|FY)(?:\s*'?\d{{2,4}})?\s+(?:Adj(?:usted|\.)?\s+)?EPS\s+(?P<actual>{_NUMBER})\s+{_VERB}\s+(?P<estimate>{_NUMBER})\s+Estimate",
    re.IGNORECASE,
)
_SALES = re.compile(
    rf"\b(?:Sales|Revenue)\s+(?P<actual>{_NUMBER})(?P<actual_unit>[KMB])?\s+{_VERB}\s+(?P<estimate>{_NUMBER})(?P<estimate_unit>[KMB])?\s+Estimate",
    re.IGNORECASE,
)
_UNITS = {"": Decimal(1), "K": Decimal(1_000), "M": Decimal(1_000_000), "B": Decimal(1_000_000_000)}
_CORRECTION = "CORRECTION"


def _result(actual: Decimal, estimate: Decimal) -> str:
    if actual > estimate:
        return "BEAT"
    return "MISS" if actual < estimate else "IN-LINE"


def _surprise_pct(actual: Decimal, estimate: Decimal) -> float | None:
    if estimate == 0:
        return None
    return float((actual - estimate) / abs(estimate))


@dataclass(frozen=True, slots=True)
class Surprise:
    eps_actual: Decimal
    eps_estimate: Decimal
    sales_actual: Decimal | None = None
    sales_estimate: Decimal | None = None

    @property
    def eps_beat(self) -> bool:
        return self.eps_actual > self.eps_estimate

    @property
    def eps_result(self) -> str:
        return _result(self.eps_actual, self.eps_estimate)

    @property
    def eps_surprise_pct(self) -> float | None:
        return _surprise_pct(self.eps_actual, self.eps_estimate)

    @property
    def sales_result(self) -> str | None:
        if self.sales_actual is None or self.sales_estimate is None:
            return None
        return _result(self.sales_actual, self.sales_estimate)

    @property
    def sales_surprise_pct(self) -> float | None:
        if self.sales_actual is None or self.sales_estimate is None:
            return None
        return _surprise_pct(self.sales_actual, self.sales_estimate)


@dataclass(frozen=True, slots=True)
class PickedSurprise:
    surprise: Surprise
    headline: str
    created_at: datetime


def _number(text: str) -> Decimal | None:
    negative = "(" in text or "-" in text
    digits = re.sub(r"[^\d.]", "", text)
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None
    return -value if negative else value


def parse_surprise(headline: str) -> Surprise | None:
    """The EPS (and sales, when present) actual and estimate in a Benzinga headline; None without an EPS estimate."""
    eps = _EPS.search(headline or "")
    if eps is None:
        return None
    actual, estimate = _number(eps["actual"]), _number(eps["estimate"])
    if actual is None or estimate is None:
        return None
    sales_actual = sales_estimate = None
    sales = _SALES.search(headline, eps.end())
    if sales is not None:
        raw_actual, raw_estimate = _number(sales["actual"]), _number(sales["estimate"])
        if raw_actual is not None and raw_estimate is not None:
            sales_actual = raw_actual * _UNITS[(sales["actual_unit"] or "").upper()]
            sales_estimate = raw_estimate * _UNITS[(sales["estimate_unit"] or "").upper()]
    return Surprise(actual, estimate, sales_actual, sales_estimate)


def article_time(article: Mapping[str, Any]) -> datetime | None:
    """An article's aware `created_at`; None when missing or unparsable."""
    try:
        created = datetime.fromisoformat(str(article.get("created_at") or ""))
    except ValueError:
        return None
    return created if created.tzinfo is not None else None


def articles_for(articles: Sequence[Mapping[str, Any]], symbol: str, *, start: datetime, end: datetime) -> list[Mapping[str, Any]]:
    """The articles tagged with `symbol` and created in `[start, end]`, oldest first."""
    wanted = symbol.upper()
    kept: list[tuple[datetime, Mapping[str, Any]]] = []
    for article in articles:
        created = article_time(article)
        symbols = [str(s).upper() for s in article.get("symbols") or []]
        if created is not None and wanted in symbols and start <= created <= end:
            kept.append((created, article))
    return [article for _, article in sorted(kept, key=lambda pair: pair[0])]


def pick_surprise(articles: Sequence[Mapping[str, Any]]) -> PickedSurprise | None:
    """The surprise among `articles` (one symbol, oldest first): the latest CORRECTION, else the earliest parsable headline."""
    parsed: list[PickedSurprise] = []
    for article in articles:
        headline = str(article.get("headline") or "")
        surprise = parse_surprise(headline)
        created = article_time(article)
        if surprise is not None and created is not None:
            parsed.append(PickedSurprise(surprise, headline, created))
    corrections = [p for p in parsed if p.headline.strip().upper().startswith(_CORRECTION)]
    if corrections:
        return corrections[-1]
    return parsed[0] if parsed else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_surprise.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/surprise.py tests/strategies/earnings_drift/test_drift_surprise.py
git commit -m "feat: earnings_drift parses Benzinga EPS and sales surprise headlines (Task 2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Earnings events and reaction dates (`events.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/events.py`
- Test: `tests/strategies/earnings_drift/test_drift_events.py`

**Interfaces:**
- Consumes: `trading_agent_framework.fundamentals.sec.parse_dt`, `utils.clock.MARKET_TZ`.
- Produces:
  - `EarningsEvent(symbol: str, accepted_at: datetime, accession_number: str, primary_document: str)` (frozen)
  - `earnings_events(symbol: str, payloads: Iterable[Mapping[str, Any]]) -> list[EarningsEvent]` (oldest first)
  - `older_pages(payload: Mapping[str, Any], since: date) -> list[str]`
  - `session_close(day: date) -> datetime` (16:00 ET)
  - `reaction_date(accepted_at: datetime, trading_dates: Sequence[date]) -> date | None`
  - `events_reacting_on(events: Iterable[EarningsEvent], day: date, trading_dates: Sequence[date], now: datetime) -> list[EarningsEvent]`
  - `release_timing(accepted_at: datetime, reaction_day: date) -> str` (`"before_open"|"during_session"|"after_close"`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_events.py
from __future__ import annotations

from datetime import UTC, date, datetime

from tests.fakes import et

from trading_agent_framework.strategies.earnings_drift.events import (
    EarningsEvent,
    earnings_events,
    events_reacting_on,
    older_pages,
    reaction_date,
    release_timing,
)

# Tue 1 Sep .. Mon 7 Sep 2026 (the fake calendar ignores Labor Day): Wed 2, Thu 3, Fri 4, Mon 7.
DATES = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7)]


def _columns(rows: list[tuple[str, str, str, str]]) -> dict[str, list[str]]:
    """rows: (form, items, acceptanceDateTime, accessionNumber)."""
    return {
        "form": [r[0] for r in rows],
        "items": [r[1] for r in rows],
        "acceptanceDateTime": [r[2] for r in rows],
        "accessionNumber": [r[3] for r in rows],
        "primaryDocument": [f"{r[3]}.htm" for r in rows],
        "filingDate": [r[2][:10] for r in rows],
    }


def test_only_8k_filings_with_item_2_02_are_events_across_recent_and_older_pages() -> None:
    recent = {
        "filings": {
            "recent": _columns(
                [
                    ("8-K", "2.02,9.01", "2026-07-28T20:07:49.000Z", "acc-3"),
                    ("8-K", "7.01", "2026-06-01T12:00:00.000Z", "acc-x1"),
                    ("8-K/A", "2.02", "2026-07-29T12:00:00.000Z", "acc-x2"),
                    ("10-Q", "", "2026-07-30T12:00:00.000Z", "acc-x3"),
                    ("8-K", "1.01,2.03", "2026-05-01T12:00:00.000Z", "acc-x4"),
                    ("8-K", "2.02", "2026-04-28T20:08:25.000Z", "acc-2"),
                ]
            ),
            "files": [],
        }
    }
    page = _columns([("8-K", "2.02,7.01,9.01", "2026-01-20T21:01:00.000Z", "acc-1"), ("8-K", "2.02", "2026-04-28T20:08:25.000Z", "acc-2")])
    events = earnings_events("omc", [recent, page])
    assert [e.accession_number for e in events] == ["acc-1", "acc-2", "acc-3"]  # oldest first, the duplicate accession once
    assert events[0] == EarningsEvent("OMC", datetime(2026, 1, 20, 21, 1, tzinfo=UTC), "acc-1", "acc-1.htm")


def test_a_row_without_a_timestamp_is_skipped() -> None:
    payload = {"filings": {"recent": _columns([("8-K", "2.02", "", "acc-1")])}}
    assert earnings_events("AAA", [payload]) == []


def test_older_pages_are_those_reaching_since() -> None:
    payload = {
        "filings": {
            "recent": {},
            "files": [
                {"name": "CIK1-submissions-001.json", "filingFrom": "2024-01-01", "filingTo": "2026-01-31"},
                {"name": "CIK1-submissions-002.json", "filingFrom": "2010-01-01", "filingTo": "2023-12-31"},
                {"name": "bad.json"},
            ],
        }
    }
    assert older_pages(payload, date(2025, 9, 1)) == ["CIK1-submissions-001.json"]
    assert older_pages(payload, date(2023, 6, 1)) == ["CIK1-submissions-001.json", "CIK1-submissions-002.json"]


def test_reaction_date_is_the_first_session_closing_after_the_release() -> None:
    assert reaction_date(et(2026, 9, 2, 7, 0), DATES) == date(2026, 9, 2)  # before the open
    assert reaction_date(et(2026, 9, 2, 12, 0), DATES) == date(2026, 9, 2)  # during the session
    assert reaction_date(et(2026, 9, 2, 16, 0), DATES) == date(2026, 9, 3)  # at the close: next session
    assert reaction_date(et(2026, 9, 2, 16, 5), DATES) == date(2026, 9, 3)  # after the close
    assert reaction_date(et(2026, 9, 4, 16, 30), DATES) == date(2026, 9, 7)  # Friday evening: Monday
    assert reaction_date(et(2026, 9, 7, 16, 30), DATES) is None  # past the last known session


def _event(symbol: str, accepted_at: datetime, accession: str = "") -> EarningsEvent:
    return EarningsEvent(symbol, accepted_at, accession or f"{symbol}-{accepted_at.isoformat()}", "doc.htm")


def test_events_reacting_on_a_day() -> None:
    events = [
        _event("AAA", et(2026, 9, 2, 16, 5)),  # reacts Thu 3
        _event("BBB", et(2026, 9, 3, 7, 0)),  # reacts Thu 3
        _event("BBB", et(2026, 9, 3, 8, 0)),  # same symbol, same reaction: only the earliest is kept
        _event("CCC", et(2026, 9, 3, 16, 1)),  # reacts Fri 4
        _event("DDD", et(2026, 9, 2, 15, 0)),  # reacted Wed 2
    ]
    found = events_reacting_on(events, date(2026, 9, 3), DATES, now=et(2026, 9, 3, 16, 0))
    assert [(e.symbol, e.accepted_at) for e in found] == [("AAA", et(2026, 9, 2, 16, 5)), ("BBB", et(2026, 9, 3, 7, 0))]


def test_events_reacting_on_ignores_the_future_unknown_days_and_the_first_date() -> None:
    late = _event("AAA", et(2026, 9, 3, 15, 0))
    assert events_reacting_on([late], date(2026, 9, 3), DATES, now=et(2026, 9, 3, 14, 0)) == []  # not filed yet at `now`
    assert events_reacting_on([late], date(2026, 9, 5), DATES, now=et(2026, 9, 5, 16, 0)) == []  # not a trading date
    assert events_reacting_on([_event("AAA", et(2026, 9, 1, 7, 0))], date(2026, 9, 1), DATES, now=et(2026, 9, 1, 16, 0)) == []


def test_release_timing() -> None:
    assert release_timing(et(2026, 9, 2, 16, 5), date(2026, 9, 3)) == "after_close"
    assert release_timing(et(2026, 9, 3, 7, 0), date(2026, 9, 3)) == "before_open"
    assert release_timing(et(2026, 9, 3, 11, 0), date(2026, 9, 3)) == "during_session"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_events.py -v`
Expected: FAIL with `ModuleNotFoundError` (`events`)

- [ ] **Step 3: Implement `events.py`**

```python
# src/trading_agent_framework/strategies/earnings_drift/events.py
"""Earnings events from SEC submissions, and the session each one moves the stock in (spec §3.1). Pure.

An event is an 8-K carrying item 2.02 ("Results of Operations"), timed by its `acceptanceDateTime`. Its reaction
session is the first session whose close is strictly after that time. Sessions are given as trading dates (the
benchmark's daily bars) and every close is taken as 16:00 ET: early closes are a known limit (spec §11).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from trading_agent_framework.fundamentals.sec import parse_dt
from trading_agent_framework.utils.clock import MARKET_TZ

EARNINGS_ITEM = "2.02"
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    symbol: str
    accepted_at: datetime  # aware
    accession_number: str
    primary_document: str


def _columns(payload: Mapping[str, Any]) -> Mapping[str, Sequence[Any]]:
    """The columnar filing table: `filings.recent` of a submissions payload, or the top level of an older page."""
    filings = payload.get("filings")
    if isinstance(filings, Mapping):
        recent = filings.get("recent")
        return recent if isinstance(recent, Mapping) else {}
    return payload


def _at(columns: Mapping[str, Sequence[Any]], key: str, index: int) -> Any:
    values = columns.get(key) or []
    return values[index] if index < len(values) else None


def earnings_events(symbol: str, payloads: Iterable[Mapping[str, Any]]) -> list[EarningsEvent]:
    """Every 8-K carrying item 2.02 across `payloads`, oldest first, one per accession number."""
    seen: set[str] = set()
    events: list[EarningsEvent] = []
    for payload in payloads:
        columns = _columns(payload)
        for index, form in enumerate(columns.get("form") or []):
            if str(form).upper() != "8-K":
                continue
            items = [part.strip() for part in str(_at(columns, "items", index) or "").split(",")]
            if EARNINGS_ITEM not in items:
                continue
            accepted = parse_dt(_at(columns, "acceptanceDateTime", index))
            accession = str(_at(columns, "accessionNumber", index) or "")
            if accepted is None or accepted.tzinfo is None or not accession or accession in seen:
                continue
            seen.add(accession)
            events.append(EarningsEvent(symbol.upper(), accepted, accession, str(_at(columns, "primaryDocument", index) or "")))
    return sorted(events, key=lambda event: event.accepted_at)


def _iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def older_pages(payload: Mapping[str, Any], since: date) -> list[str]:
    """Names of the older submission pages whose `filingTo` is on or after `since`."""
    filings = payload.get("filings")
    files = filings.get("files") if isinstance(filings, Mapping) else None
    names: list[str] = []
    for entry in files or []:
        if not isinstance(entry, Mapping) or not entry.get("name"):
            continue
        filing_to = _iso_date(entry.get("filingTo"))
        if filing_to is not None and filing_to >= since:
            names.append(str(entry["name"]))
    return names


def session_close(day: date) -> datetime:
    return datetime.combine(day, SESSION_CLOSE, tzinfo=MARKET_TZ)


def reaction_date(accepted_at: datetime, trading_dates: Sequence[date]) -> date | None:
    """The first trading date whose 16:00 ET close is strictly after `accepted_at`; None past the last date."""
    for day in sorted(trading_dates):
        if session_close(day) > accepted_at:
            return day
    return None


def events_reacting_on(events: Iterable[EarningsEvent], day: date, trading_dates: Sequence[date], now: datetime) -> list[EarningsEvent]:
    """The events whose reaction date is `day` and that were filed by `now`: one per symbol, its earliest, by symbol."""
    dates = sorted(trading_dates)
    if day not in dates or dates.index(day) == 0:
        return []
    lower, upper = session_close(dates[dates.index(day) - 1]), session_close(day)
    earliest: dict[str, EarningsEvent] = {}
    for event in events:
        if lower <= event.accepted_at < upper and event.accepted_at <= now:
            kept = earliest.get(event.symbol)
            if kept is None or event.accepted_at < kept.accepted_at:
                earliest[event.symbol] = event
    return [earliest[symbol] for symbol in sorted(earliest)]


def release_timing(accepted_at: datetime, reaction_day: date) -> str:
    local = accepted_at.astimezone(MARKET_TZ)
    if local.date() < reaction_day:
        return "after_close"
    return "before_open" if local.time() < SESSION_OPEN else "during_session"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_events.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/events.py tests/strategies/earnings_drift/test_drift_events.py
git commit -m "feat: earnings_drift finds 8-K item 2.02 events and their reaction session (Task 3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: SEC event source (`event_source.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/event_source.py`
- Test: `tests/strategies/earnings_drift/test_drift_event_source.py`

**Interfaces:**
- Consumes: `SecEdgarClient` (`ticker_to_cik`, `get_submissions_payload(cik, as_of=, max_age_days=)`, `get_json(url, cache_key)`),
  `SEC_DATA_BASE_URL`; `earnings_events`, `older_pages` (Task 3).
- Produces:
  - `LoadReport(loaded: int, failed: dict[str, str])` with `is_hollow(fraction: float, min_failures: int) -> bool`
  - `EventProvider` (Protocol): `needs_load() -> bool`, `load(symbols, *, as_of: datetime, since: date) -> LoadReport`,
    `discard() -> None`, `all_events() -> list[EarningsEvent]`
  - `EventSource(client: SecEdgarClient, *, reload_every_cycle: bool)` implementing it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_event_source.py
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.earnings_drift.event_source import EventSource, LoadReport

_TICKERS = {"0": {"cik_str": 29989, "ticker": "OMC"}, "1": {"cik_str": 320193, "ticker": "AAPL"}}
_RECENT = {
    "filings": {
        "recent": {
            "form": ["8-K"],
            "items": ["2.02,9.01"],
            "acceptanceDateTime": ["2026-07-28T20:07:49.000Z"],
            "accessionNumber": ["acc-new"],
            "primaryDocument": ["new.htm"],
        },
        "files": [{"name": "CIK0000029989-submissions-001.json", "filingFrom": "2016-01-01", "filingTo": "2026-01-31"}],
    }
}
_PAGE = {"form": ["8-K"], "items": ["2.02"], "acceptanceDateTime": ["2026-01-20T21:01:00.000Z"], "accessionNumber": ["acc-old"], "primaryDocument": ["old.htm"]}


def _source(tmp_path: Path, requests: list[str], *, reload_every_cycle: bool = False) -> EventSource:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json=_TICKERS)
        if request.url.path == "/submissions/CIK0000029989.json":
            return httpx.Response(200, json=_RECENT)
        if request.url.path == "/submissions/CIK0000029989-submissions-001.json":
            return httpx.Response(200, json=_PAGE)
        return httpx.Response(404)

    client = SecEdgarClient("TestApp test@example.com", tmp_path, min_request_interval_seconds=0.0, transport=httpx.MockTransport(handler))
    return EventSource(client, reload_every_cycle=reload_every_cycle)


NOW = datetime(2026, 9, 1, 20, 0, tzinfo=UTC)


def test_load_reads_recent_filings_and_the_older_pages_reaching_since(tmp_path: Path) -> None:
    requests: list[str] = []
    source = _source(tmp_path, requests)
    report = source.load(["OMC"], as_of=NOW, since=date(2025, 12, 1))
    assert report == LoadReport(loaded=1, failed={})
    assert [e.accession_number for e in source.all_events()] == ["acc-old", "acc-new"]
    assert "/submissions/CIK0000029989-submissions-001.json" in requests


def test_an_older_page_before_since_is_not_fetched(tmp_path: Path) -> None:
    requests: list[str] = []
    source = _source(tmp_path, requests)
    source.load(["OMC"], as_of=NOW, since=date(2026, 6, 1))
    assert [e.accession_number for e in source.all_events()] == ["acc-new"]
    assert "/submissions/CIK0000029989-submissions-001.json" not in requests


def test_a_failing_symbol_is_reported_not_raised(tmp_path: Path) -> None:
    source = _source(tmp_path, [])
    report = source.load(["OMC", "NOPE", "AAPL"], as_of=NOW, since=date(2026, 6, 1))
    assert report.loaded == 1
    assert set(report.failed) == {"NOPE", "AAPL"}  # unknown ticker; AAPL's submissions answer 404


def test_needs_load_once_in_a_backtest_every_cycle_live_and_after_discard(tmp_path: Path) -> None:
    backtest = _source(tmp_path / "bt", [])
    assert backtest.needs_load()
    backtest.load(["OMC"], as_of=NOW, since=date(2026, 6, 1))
    assert not backtest.needs_load()
    backtest.discard()
    assert backtest.needs_load() and backtest.all_events() == []
    live = _source(tmp_path / "live", [], reload_every_cycle=True)
    live.load(["OMC"], as_of=NOW, since=date(2026, 6, 1))
    assert live.needs_load()


@pytest.mark.parametrize(
    ("loaded", "failed", "hollow"),
    [(10, 25, True), (10, 19, False), (35, 25, False), (0, 0, False)],
)
def test_is_hollow(loaded: int, failed: int, hollow: bool) -> None:
    report = LoadReport(loaded=loaded, failed={f"S{i}": "error" for i in range(failed)})
    assert report.is_hollow(0.5, 20) is hollow
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_event_source.py -v`
Expected: FAIL with `ModuleNotFoundError` (`event_source`)

- [ ] **Step 3: Implement `event_source.py`**

```python
# src/trading_agent_framework/strategies/earnings_drift/event_source.py
"""SEC submissions for the universe, reduced to earnings events and kept in memory (spec §3.2).

The only module of the strategy that talks to SEC EDGAR. Every payload goes through `SecEdgarClient`'s cache with
`max_age_days=0` against the strategy clock: a backtest never refetches a file fetched after its simulated date,
paper/live refetch at every load. A backtest loads once (`reload_every_cycle=False`), paper/live at every cycle.
Older pages (`filings.files`) never change and are cached for good.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

from trading_agent_framework.fundamentals.edgar_client import SEC_DATA_BASE_URL, SecEdgarClient
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent, earnings_events, older_pages
from trading_agent_framework.utils.errors import FundamentalsError


@dataclass(frozen=True, slots=True)
class LoadReport:
    loaded: int
    failed: dict[str, str] = field(default_factory=dict)  # symbol -> error message

    def is_hollow(self, fraction: float, min_failures: int) -> bool:
        """Too many symbols failed to trust the events: at least `min_failures`, and more than `fraction` of all."""
        total = self.loaded + len(self.failed)
        return total > 0 and len(self.failed) >= min_failures and len(self.failed) > fraction * total


class EventProvider(Protocol):
    def needs_load(self) -> bool: ...

    def load(self, symbols: Sequence[str], *, as_of: datetime, since: date) -> LoadReport: ...

    def discard(self) -> None: ...

    def all_events(self) -> list[EarningsEvent]: ...


class EventSource:
    def __init__(self, client: SecEdgarClient, *, reload_every_cycle: bool) -> None:
        self._client = client
        self._reload_every_cycle = reload_every_cycle
        self._events: dict[str, list[EarningsEvent]] | None = None

    def needs_load(self) -> bool:
        return self._events is None or self._reload_every_cycle

    def load(self, symbols: Sequence[str], *, as_of: datetime, since: date) -> LoadReport:
        events: dict[str, list[EarningsEvent]] = {}
        failed: dict[str, str] = {}
        for symbol in symbols:
            try:
                events[symbol.upper()] = self._symbol_events(symbol, as_of=as_of, since=since)
            except FundamentalsError as exc:
                failed[symbol.upper()] = str(exc)
        self._events = events
        return LoadReport(loaded=len(events), failed=failed)

    def discard(self) -> None:
        self._events = None

    def all_events(self) -> list[EarningsEvent]:
        return [event for events in (self._events or {}).values() for event in events]

    def _symbol_events(self, symbol: str, *, as_of: datetime, since: date) -> list[EarningsEvent]:
        cik = self._client.ticker_to_cik(symbol)
        recent = self._client.get_submissions_payload(cik, as_of=as_of, max_age_days=0)
        pages = [self._client.get_json(f"{SEC_DATA_BASE_URL}/submissions/{name}", ("submissions", name)) for name in older_pages(recent, since)]
        return earnings_events(symbol, [recent, *pages])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_event_source.py -v`
Expected: PASS. If `test_a_failing_symbol_is_reported_not_raised` fails because the 404 is not a `FundamentalsError`,
check `SecEdgarClient.fetch_json`: HTTP 404 raises `FundamentalsNotFoundError`, a `FundamentalsError` subclass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/event_source.py tests/strategies/earnings_drift/test_drift_event_source.py
git commit -m "feat: earnings_drift loads SEC submissions, older pages included, into an event index (Task 4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Reaction features (`reaction.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/reaction.py`
- Test: `tests/strategies/earnings_drift/test_drift_reaction.py`

**Interfaces:**
- Produces:
  - `ReactionFeatures(gap_pct, return_pct, abnormal_pct, hold_ratio: float | None, close_location, rel_volume, dollar_volume_20d,
    runup_20d_pct: float | None, runup_60d_pct: float | None, atr14_pct: float | None, close, reaction_low)` (frozen, floats)
  - `bar_dates(frame: pd.DataFrame) -> list[date]`
  - `rows_through(frame: pd.DataFrame, day: date) -> pd.DataFrame` (indexed by date, rows on or before `day`)
  - `reaction_features(stock: pd.DataFrame, benchmark: pd.DataFrame, day: date, *, baseline_sessions: int = 20) -> ReactionFeatures | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_reaction.py
from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest
from tests.fakes import weekday_sessions

from trading_agent_framework.strategies.earnings_drift.reaction import bar_dates, reaction_features
from trading_agent_framework.utils.clock import MARKET_TZ

SESSIONS = weekday_sessions(date(2026, 7, 1), 31)
DAY = SESSIONS[-1].open.date()
FLAT = (100.0, 101.0, 99.0, 100.0, 1_000_000.0)


def _frame(rows: list[tuple[float, float, float, float, float]], *, stamp: str = "close") -> pd.DataFrame:
    sessions = SESSIONS[-len(rows) :]
    index = [s.close if stamp == "close" else datetime.combine(s.open.date(), time(0), tzinfo=MARKET_TZ) for s in sessions]
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(index, name="timestamp"))


STOCK = _frame([FLAT] * 30 + [(106.0, 110.0, 105.0, 109.0, 4_000_000.0)])
SPY = _frame([(400.0, 401.0, 399.0, 400.0, 1e8)] * 30 + [(400.0, 403.0, 399.0, 402.0, 1e8)])


def test_reaction_features() -> None:
    features = reaction_features(STOCK, SPY, DAY)
    assert features is not None
    assert features.gap_pct == pytest.approx(0.06)
    assert features.return_pct == pytest.approx(0.09)
    assert features.abnormal_pct == pytest.approx(0.085)
    assert features.hold_ratio == pytest.approx(0.9)
    assert features.close_location == pytest.approx(0.8)
    assert features.rel_volume == pytest.approx(4.0)
    assert features.dollar_volume_20d == pytest.approx(1e8)
    assert features.runup_20d_pct == pytest.approx(0.0)
    assert features.runup_60d_pct is None  # 31 sessions are not enough for 60
    assert features.atr14_pct == pytest.approx((13 * 2 + 10) / 14 / 109)
    assert (features.close, features.reaction_low) == (109.0, 105.0)


def test_live_open_stamped_bars_give_the_same_dates_and_features() -> None:
    assert bar_dates(_frame([FLAT] * 3, stamp="open")) == bar_dates(_frame([FLAT] * 3))
    live = reaction_features(_frame([FLAT] * 30 + [(106.0, 110.0, 105.0, 109.0, 4e6)], stamp="open"), SPY, DAY)
    assert live == reaction_features(STOCK, SPY, DAY)


def test_bars_after_the_day_are_ignored() -> None:
    assert reaction_features(STOCK, SPY, SESSIONS[-2].open.date()) is not None
    features = reaction_features(STOCK, SPY, SESSIONS[-2].open.date())
    assert features is not None and features.return_pct == pytest.approx(0.0)


def test_no_features_without_the_day_or_the_baseline() -> None:
    assert reaction_features(STOCK.iloc[:-1], SPY, DAY) is None  # no bar for the day
    assert reaction_features(STOCK.iloc[-20:], SPY, DAY) is None  # 20 rows: no full 20-session baseline before the day
    assert reaction_features(STOCK, SPY.iloc[:-1], DAY) is None  # the benchmark has no bar for the day


def test_a_down_day_has_no_hold_ratio_and_a_flat_bar_sits_mid_range() -> None:
    down = reaction_features(_frame([FLAT] * 30 + [(99.0, 99.5, 97.0, 98.0, 2e6)]), SPY, DAY)
    assert down is not None and down.hold_ratio is None
    flat = reaction_features(_frame([FLAT] * 30 + [(100.0, 100.0, 100.0, 100.0, 2e6)]), SPY, DAY)
    assert flat is not None and flat.close_location == 0.5


def test_sixty_session_runup_with_enough_history() -> None:
    sessions = weekday_sessions(date(2026, 5, 1), 70)
    closes = [50.0] * 8 + [100.0] * 61 + [104.0]
    rows = [(c, c + 1, c - 1, c, 1e6) for c in closes]
    stock = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([s.close for s in sessions]))
    spy = pd.DataFrame([(400.0, 401.0, 399.0, 400.0, 1e8)] * 70, columns=stock.columns, index=stock.index)
    features = reaction_features(stock, spy, sessions[-1].open.date())
    assert features is not None
    assert features.runup_60d_pct == pytest.approx(0.0)  # close before the day (100) vs 60 sessions before that (100)
    closes[-62] = 80.0
    rows = [(c, c + 1, c - 1, c, 1e6) for c in closes]
    stock = pd.DataFrame(rows, columns=stock.columns, index=stock.index)
    assert reaction_features(stock, spy, sessions[-1].open.date()).runup_60d_pct == pytest.approx(0.25)  # type: ignore[union-attr]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_reaction.py -v`
Expected: FAIL with `ModuleNotFoundError` (`reaction`)

- [ ] **Step 3: Implement `reaction.py`**

```python
# src/trading_agent_framework/strategies/earnings_drift/reaction.py
"""Reaction-day features from daily bars (spec §3.4). Pure; float64 on purpose (indicator maths on `Bars.df`).

Bars are matched by their market-time DATE, because live Alpaca stamps a daily bar at its open (midnight ET) and
every backtest source at its close (16:00 ET): both fall on the session's date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from trading_agent_framework.utils.clock import MARKET_TZ


@dataclass(frozen=True, slots=True)
class ReactionFeatures:
    gap_pct: float
    return_pct: float
    abnormal_pct: float
    hold_ratio: float | None  # None when the day's high is not above the previous close
    close_location: float
    rel_volume: float
    dollar_volume_20d: float
    runup_20d_pct: float | None
    runup_60d_pct: float | None
    atr14_pct: float | None
    close: float
    reaction_low: float


def bar_dates(frame: pd.DataFrame) -> list[date]:
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("daily bars must be indexed by tz-aware timestamps")
    return [stamp.date() for stamp in index.tz_convert(MARKET_TZ)]


def rows_through(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    """The rows on or before `day`, indexed by date (a duplicated date keeps its last row)."""
    dated = frame.copy()
    dated.index = pd.Index(bar_dates(frame))
    dated = dated[~dated.index.duplicated(keep="last")]
    return dated[[stamp <= day for stamp in dated.index]]


def _runup(closes: pd.Series, sessions: int) -> float | None:
    """The close before the reaction day against the close `sessions` sessions earlier; None without the history."""
    if len(closes) < sessions + 2:
        return None
    base = float(closes.iloc[-(sessions + 2)])
    return float(closes.iloc[-2]) / base - 1 if base > 0 else None


def _atr_pct(rows: pd.DataFrame, length: int = 14) -> float | None:
    if len(rows) < length + 1:
        return None
    window = rows.iloc[-(length + 1) :]
    previous = window["close"].shift(1)
    true_range = pd.concat([window["high"] - window["low"], (window["high"] - previous).abs(), (window["low"] - previous).abs()], axis=1).max(axis=1)
    close = float(window["close"].iloc[-1])
    return float(true_range.iloc[1:].mean()) / close if close > 0 else None


def reaction_features(stock: pd.DataFrame, benchmark: pd.DataFrame, day: date, *, baseline_sessions: int = 20) -> ReactionFeatures | None:
    """Features of the reaction session `day`; None without a bar for `day` (stock and benchmark) and the baseline before it."""
    rows = rows_through(stock, day)
    bench = rows_through(benchmark, day)
    if rows.empty or rows.index[-1] != day or len(rows) < baseline_sessions + 2:
        return None
    if len(bench) < 2 or bench.index[-1] != day:
        return None
    reaction, previous = rows.iloc[-1], rows.iloc[-2]
    prev_close = float(previous["close"])
    open_, high, low, close = (float(reaction[key]) for key in ("open", "high", "low", "close"))
    baseline = rows.iloc[-(baseline_sessions + 1) : -1]
    mean_volume = float(baseline["volume"].mean())
    bench_prev = float(bench["close"].iloc[-2])
    if prev_close <= 0 or mean_volume <= 0 or bench_prev <= 0:
        return None
    return_pct = close / prev_close - 1
    return ReactionFeatures(
        gap_pct=open_ / prev_close - 1,
        return_pct=return_pct,
        abnormal_pct=return_pct - (float(bench["close"].iloc[-1]) / bench_prev - 1),
        hold_ratio=(close - prev_close) / (high - prev_close) if high > prev_close else None,
        close_location=(close - low) / (high - low) if high > low else 0.5,
        rel_volume=float(reaction["volume"]) / mean_volume,
        dollar_volume_20d=float((baseline["close"] * baseline["volume"]).mean()),
        runup_20d_pct=_runup(rows["close"], 20),
        runup_60d_pct=_runup(rows["close"], 60),
        atr14_pct=_atr_pct(rows),
        close=close,
        reaction_low=low,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_reaction.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/reaction.py tests/strategies/earnings_drift/test_drift_reaction.py
git commit -m "feat: earnings_drift computes reaction-day features from daily bars (Task 5)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Gates, candidates and the fact sheet (`screening.py`, `fact_sheet.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/screening.py`
- Create: `src/trading_agent_framework/strategies/earnings_drift/fact_sheet.py`
- Create: `tests/strategies/earnings_drift/drift_helpers.py` (shared test builders, extended in Task 9)
- Test: `tests/strategies/earnings_drift/test_drift_screening.py`

**Interfaces:**
- Consumes: `Surprise`, `PickedSurprise` (Task 2); `EarningsEvent`, `release_timing` (Task 3); `ReactionFeatures` (Task 5); `DriftParams`.
- Produces:
  - `Candidate(event: EarningsEvent, reaction_day: date, surprise: PickedSurprise, reaction: ReactionFeatures, headlines: tuple[tuple[str, str], ...] = ())`
    with property `symbol`.
  - `REJECT_REASONS: tuple[str, ...]`
  - `gate(surprise: Surprise | None, reaction: ReactionFeatures | None, params: DriftParams, *, held: bool) -> str | None`
  - `fact_sheet(candidate: Candidate, max_quantity: int) -> dict[str, Any]`
  - test helpers `make_features(**overrides)`, `make_candidate(symbol="AAA", *, day=..., accepted_at=None, surprise=None, **feature_overrides)`.

- [ ] **Step 1: Write the shared helpers and the failing tests**

```python
# tests/strategies/earnings_drift/drift_helpers.py
"""Builders shared by the earnings_drift tests (not a test module)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from tests.fakes import et

from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
from trading_agent_framework.strategies.earnings_drift.reaction import ReactionFeatures
from trading_agent_framework.strategies.earnings_drift.screening import Candidate
from trading_agent_framework.strategies.earnings_drift.surprise import PickedSurprise, Surprise

DEFAULT_DAY = date(2026, 9, 1)


def make_features(**overrides: float | None) -> ReactionFeatures:
    values: dict[str, float | None] = dict(
        gap_pct=0.06,
        return_pct=0.08,
        abnormal_pct=0.075,
        hold_ratio=0.8,
        close_location=0.9,
        rel_volume=3.0,
        dollar_volume_20d=50_000_000.0,
        runup_20d_pct=0.02,
        runup_60d_pct=0.05,
        atr14_pct=0.02,
        close=100.0,
        reaction_low=95.0,
    )
    values.update(overrides)
    return ReactionFeatures(**values)  # type: ignore[arg-type]


def make_candidate(symbol: str = "AAA", *, day: date = DEFAULT_DAY, accepted_at: datetime | None = None, surprise: Surprise | None = None, **feature_overrides: float | None) -> Candidate:
    accepted = accepted_at or et(day.year, day.month, day.day, 7, 0)
    headline = f"{symbol} Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate"
    return Candidate(
        event=EarningsEvent(symbol, accepted, f"0000000000-26-{symbol}", f"{symbol.lower()}-8k.htm"),
        reaction_day=day,
        surprise=PickedSurprise(surprise or Surprise(Decimal("1.52"), Decimal("1.20"), Decimal("1100000000"), Decimal("1000000000")), headline, accepted),
        reaction=make_features(**feature_overrides),
        headlines=((f"{day.isoformat()} 07:01", headline),),
    )
```

```python
# tests/strategies/earnings_drift/test_drift_screening.py
from __future__ import annotations

from datetime import date
from decimal import Decimal as D

import pytest
from tests.fakes import et
from tests.strategies.earnings_drift.drift_helpers import make_candidate, make_features

from trading_agent_framework.strategies.earnings_drift.fact_sheet import fact_sheet
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.screening import REJECT_REASONS, gate
from trading_agent_framework.strategies.earnings_drift.surprise import Surprise

PARAMS = DriftParams()
BEAT = Surprise(D("1.52"), D("1.20"))


def test_a_good_reaction_passes() -> None:
    assert gate(BEAT, make_features(), PARAMS, held=False) is None


@pytest.mark.parametrize(
    ("surprise", "features", "held", "reason"),
    [
        (BEAT, make_features(), True, "already_held"),
        (None, make_features(), False, "no_surprise_data"),
        (Surprise(D("1.00"), D("1.00")), make_features(), False, "eps_miss"),
        (BEAT, None, False, "no_bars"),
        (BEAT, make_features(abnormal_pct=0.0299), False, "weak_reaction"),
        (BEAT, make_features(hold_ratio=0.49), False, "faded"),
        (BEAT, make_features(hold_ratio=None), False, "faded"),
        (BEAT, make_features(close_location=0.49), False, "faded"),
        (BEAT, make_features(rel_volume=1.99), False, "low_volume"),
        (BEAT, make_features(close=9.99), False, "illiquid"),
        (BEAT, make_features(dollar_volume_20d=19_999_999.0), False, "illiquid"),
    ],
)
def test_each_gate(surprise, features, held: bool, reason: str) -> None:  # noqa: ANN001
    assert gate(surprise, features, PARAMS, held=held) == reason


def test_boundaries_pass_and_the_first_failure_wins() -> None:
    edge = make_features(abnormal_pct=0.03, hold_ratio=0.5, close_location=0.5, rel_volume=2.0, close=10.0, dollar_volume_20d=20_000_000.0)
    assert gate(BEAT, edge, PARAMS, held=False) is None
    assert gate(None, make_features(abnormal_pct=0.0), PARAMS, held=True) == "already_held"
    assert gate(BEAT, make_features(abnormal_pct=0.0, rel_volume=0.5), PARAMS, held=False) == "weak_reaction"
    assert REJECT_REASONS == ("already_held", "no_surprise_data", "eps_miss", "no_bars", "weak_reaction", "faded", "low_volume", "illiquid")


def test_fact_sheet_is_lean_and_labelled_by_code() -> None:
    candidate = make_candidate("AAA", day=date(2026, 9, 2), accepted_at=et(2026, 9, 1, 16, 5), runup_60d_pct=None)
    sheet = fact_sheet(candidate, max_quantity=13)
    assert sheet == {
        "symbol": "AAA",
        "reported_at": "2026-09-01 16:05",
        "timing": "after_close",
        "eps": {"actual": 1.52, "estimate": 1.2, "surprise_pct": 26.7, "result": "BEAT"},
        "sales": {"actual": 1_100_000_000.0, "estimate": 1_000_000_000.0, "surprise_pct": 10.0, "result": "BEAT"},
        "reaction": {"gap_pct": 6.0, "return_pct": 8.0, "abnormal_pct": 7.5, "hold_ratio": 0.8, "close_location": 0.9, "rel_volume": 3.0},
        "context": {"runup_20d_pct": 2.0, "runup_60d_pct": None, "atr14_pct": 2.0, "close": 100.0, "reaction_low": 95.0},
        "max_quantity": 13,
        "filing": {"accession_number": "0000000000-26-AAA"},
        "headlines": [{"time": "2026-09-02 07:01", "headline": "AAA Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate"}],
    }


def test_fact_sheet_without_sales() -> None:
    sheet = fact_sheet(make_candidate("BBB", surprise=Surprise(D("0.50"), D("0.40"))), max_quantity=0)
    assert sheet["sales"] is None and sheet["eps"]["result"] == "BEAT"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_screening.py -v`
Expected: FAIL with `ModuleNotFoundError` (`screening`)

- [ ] **Step 3: Implement `screening.py` and `fact_sheet.py`**

```python
# src/trading_agent_framework/strategies/earnings_drift/screening.py
"""Hard gates on an earnings event, and the candidate that passes them (spec §3.5). Pure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.reaction import ReactionFeatures
from trading_agent_framework.strategies.earnings_drift.surprise import PickedSurprise, Surprise

REJECT_REASONS = ("already_held", "no_surprise_data", "eps_miss", "no_bars", "weak_reaction", "faded", "low_volume", "illiquid")


@dataclass(frozen=True, slots=True)
class Candidate:
    event: EarningsEvent
    reaction_day: date
    surprise: PickedSurprise
    reaction: ReactionFeatures
    headlines: tuple[tuple[str, str], ...] = ()  # (ET minute, headline), oldest first, at most 5

    @property
    def symbol(self) -> str:
        return self.event.symbol


def gate(surprise: Surprise | None, reaction: ReactionFeatures | None, params: DriftParams, *, held: bool) -> str | None:
    """The first gate an event fails, in `REJECT_REASONS` order; None when it passes them all."""
    if held:
        return "already_held"
    if surprise is None:
        return "no_surprise_data"
    if not surprise.eps_beat:
        return "eps_miss"
    if reaction is None:
        return "no_bars"
    if reaction.abnormal_pct < params.min_abnormal_pct:
        return "weak_reaction"
    if reaction.hold_ratio is None or reaction.hold_ratio < params.min_hold_ratio or reaction.close_location < params.min_close_location:
        return "faded"
    if reaction.rel_volume < params.min_rel_volume:
        return "low_volume"
    if reaction.close < params.min_price or reaction.dollar_volume_20d < params.min_dollar_volume:
        return "illiquid"
    return None
```

```python
# src/trading_agent_framework/strategies/earnings_drift/fact_sheet.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_screening.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/screening.py src/trading_agent_framework/strategies/earnings_drift/fact_sheet.py tests/strategies/earnings_drift/drift_helpers.py tests/strategies/earnings_drift/test_drift_screening.py
git commit -m "feat: earnings_drift gates and the candidate fact sheet (Task 6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Trades, state file and run logs (`book.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/book.py`
- Test: `tests/strategies/earnings_drift/test_drift_book.py`

**Interfaces:**
- Produces:
  - `TradeState` (`PENDING = "pending"`, `OPEN = "open"`)
  - `Trade` (mutable dataclass): `symbol: str, entry_order_id: str, trail_percent: Decimal, thesis: str, accession_number: str | None = None,
    state: TradeState = PENDING, quantity: Decimal = 0, entry_price: Decimal | None = None, opened_on: date | None = None,
    stop_order_id: str | None = None, exit_order_id: str | None = None, exit_reason: str | None = None, backstop: bool = False,
    reaction_low: Decimal | None = None`; `to_json() -> dict`, `Trade.from_json(dict) -> Trade` (raises `ValueError`/`KeyError`/`TypeError` on bad input)
  - `DriftState(trades: dict[str, Trade], agent_failure_streak: int = 0, hollow_scan_streak: int = 0, traded_symbols: set[str])`
  - `STATE_VERSION = 1`, `state_path(project_root: Path, mode: TradingMode) -> Path`
  - `StateStore(path)`: `load() -> DriftState`, `save(state) -> None`, `wipe() -> None`
  - `JsonlLog(path: Callable[[], Path | None])`: `append(record: dict[str, Any]) -> None`
  - `sessions_held(opened_on: date, today: date, trading_dates: Sequence[date]) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_book.py
from __future__ import annotations

import json
import logging
from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.earnings_drift.book import (
    STATE_VERSION,
    DriftState,
    JsonlLog,
    StateStore,
    Trade,
    TradeState,
    sessions_held,
    state_path,
)


def _trade() -> Trade:
    return Trade(
        symbol="AAA",
        entry_order_id="e1",
        trail_percent=D("8.0"),
        thesis="beat and raise",
        accession_number="acc",
        state=TradeState.OPEN,
        quantity=D("11"),
        entry_price=D("109.5"),
        opened_on=date(2026, 9, 2),
        stop_order_id="s1",
        reaction_low=D("105"),
    )


def test_a_trade_round_trips_through_json() -> None:
    trade = _trade()
    assert Trade.from_json(json.loads(json.dumps(trade.to_json()))) == trade


def test_state_round_trips_and_is_versioned(tmp_path: Path) -> None:
    path = state_path(tmp_path, TradingMode.PAPER)
    assert path == tmp_path / "data" / "earnings_drift_state_paper.json"
    store = StateStore(path)
    state = DriftState(trades={"AAA": _trade()}, agent_failure_streak=2, hollow_scan_streak=1, traded_symbols={"AAA", "BBB"})
    store.save(state)
    assert json.loads(path.read_text())["version"] == STATE_VERSION
    assert store.load() == state


@pytest.mark.parametrize("content", ["not json", json.dumps({"version": 99, "trades": {}}), json.dumps({"version": 1, "trades": {"AAA": {"symbol": "AAA"}}})])
def test_a_bad_state_file_loads_empty_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str) -> None:
    path = tmp_path / "state.json"
    path.write_text(content)
    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == DriftState()
    assert "empty state" in caplog.text


def test_a_missing_file_is_an_empty_state_and_wipe_is_safe(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "none.json")
    assert store.load() == DriftState()
    store.wipe()
    store.save(DriftState())
    store.wipe()
    assert not (tmp_path / "none.json").exists()


def test_jsonl_log_appends_and_does_nothing_without_a_path(tmp_path: Path) -> None:
    path = tmp_path / "run" / "trades.jsonl"
    log = JsonlLog(lambda: path)
    log.append({"a": D("1.5")})
    log.append({"b": 2})
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"a": "1.5"}, {"b": 2}]
    JsonlLog(lambda: None).append({"ignored": True})


def test_sessions_held_counts_both_ends() -> None:
    dates = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7)]
    assert sessions_held(date(2026, 9, 2), date(2026, 9, 2), dates) == 1
    assert sessions_held(date(2026, 9, 2), date(2026, 9, 7), dates) == 4
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_book.py -v`
Expected: FAIL with `ModuleNotFoundError` (`book`)

- [ ] **Step 3: Implement `book.py`**

```python
# src/trading_agent_framework/strategies/earnings_drift/book.py
"""What earnings_drift remembers between cycles, and its two run logs (spec §8).

`StateStore` keeps the open trades, the failure streaks and the symbols this strategy has ever traded (the orphan
rule) in one small JSON file per mode, written atomically. A missing, unreadable or invalid file is an empty state
(with a warning). `JsonlLog` appends JSON lines to the run directory. Neither raises on an I/O problem: bookkeeping
must not stop a cycle.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "EarningsDriftState")

STATE_VERSION = 1


class TradeState(StrEnum):
    PENDING = "pending"  # entry submitted, not filled yet
    OPEN = "open"  # entry filled: shares held, protected by a stop (or being sold)


def _dec(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


@dataclass
class Trade:
    symbol: str
    entry_order_id: str
    trail_percent: Decimal
    thesis: str
    accession_number: str | None = None
    state: TradeState = TradeState.PENDING
    quantity: Decimal = Decimal(0)  # filled shares
    entry_price: Decimal | None = None
    opened_on: date | None = None  # the entry fill's session date
    stop_order_id: str | None = None
    exit_order_id: str | None = None
    exit_reason: str | None = None  # set when an exit sell is submitted
    backstop: bool = False  # the stop was (re)placed by a guardrail; shown once to the agent
    reaction_low: Decimal | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_order_id": self.entry_order_id,
            "trail_percent": str(self.trail_percent),
            "thesis": self.thesis,
            "accession_number": self.accession_number,
            "state": self.state.value,
            "quantity": str(self.quantity),
            "entry_price": None if self.entry_price is None else str(self.entry_price),
            "opened_on": None if self.opened_on is None else self.opened_on.isoformat(),
            "stop_order_id": self.stop_order_id,
            "exit_order_id": self.exit_order_id,
            "exit_reason": self.exit_reason,
            "backstop": self.backstop,
            "reaction_low": None if self.reaction_low is None else str(self.reaction_low),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Trade:
        try:
            trail = Decimal(str(data["trail_percent"]))
            quantity = Decimal(str(data["quantity"]))
            entry_price, reaction_low = _dec(data.get("entry_price")), _dec(data.get("reaction_low"))
        except InvalidOperation as exc:
            raise ValueError(f"invalid number in trade {data.get('symbol')!r}") from exc
        opened = data.get("opened_on")
        if not isinstance(data["symbol"], str) or not isinstance(data["entry_order_id"], str) or not isinstance(data["thesis"], str):
            raise TypeError("symbol, entry_order_id and thesis must be strings")
        return cls(
            symbol=data["symbol"],
            entry_order_id=data["entry_order_id"],
            trail_percent=trail,
            thesis=data["thesis"],
            accession_number=data.get("accession_number"),
            state=TradeState(data["state"]),
            quantity=quantity,
            entry_price=entry_price,
            opened_on=None if opened is None else date.fromisoformat(opened),
            stop_order_id=data.get("stop_order_id"),
            exit_order_id=data.get("exit_order_id"),
            exit_reason=data.get("exit_reason"),
            backstop=bool(data.get("backstop", False)),
            reaction_low=reaction_low,
        )


@dataclass
class DriftState:
    trades: dict[str, Trade] = field(default_factory=dict)  # symbol -> its pending or open trade
    agent_failure_streak: int = 0
    hollow_scan_streak: int = 0
    traded_symbols: set[str] = field(default_factory=set)  # every symbol this strategy has bought (orphan adoption)


def state_path(project_root: Path, mode: TradingMode) -> Path:
    """`<project_root>/data/earnings_drift_state_<mode>.json`."""
    return project_root / "data" / f"earnings_drift_state_{mode.value}.json"


def _streak(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid streak {value!r}")
    return value


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> DriftState:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return DriftState()
        except (OSError, ValueError) as exc:
            logger.log_warning(f"state file {self._path} is unreadable, starting from an empty state: {exc}")
            return DriftState()
        try:
            if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
                raise ValueError("missing or other version")
            trades = {symbol: Trade.from_json(data) for symbol, data in raw.get("trades", {}).items()}
            traded = raw.get("traded_symbols", [])
            if not isinstance(traded, list) or not all(isinstance(s, str) for s in traded):
                raise ValueError("invalid traded_symbols")
            return DriftState(
                trades=trades,
                agent_failure_streak=_streak(raw.get("agent_failure_streak", 0)),
                hollow_scan_streak=_streak(raw.get("hollow_scan_streak", 0)),
                traded_symbols=set(traded),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.log_warning(f"state file {self._path} is not a valid state, starting from an empty state: {exc}")
            return DriftState()

    def save(self, state: DriftState) -> None:
        """Write the state atomically (temporary file in the same directory, then replace); an I/O error is logged."""
        temporary: str | None = None
        payload = {
            "version": STATE_VERSION,
            "trades": {symbol: trade.to_json() for symbol, trade in state.trades.items()},
            "agent_failure_streak": state.agent_failure_streak,
            "hollow_scan_streak": state.hollow_scan_streak,
            "traded_symbols": sorted(state.traded_symbols),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self._path.parent, prefix=f"{self._path.name}.", suffix=".tmp")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temporary, self._path)
        except OSError as exc:
            logger.log_warning(f"state could not be saved to {self._path}: {exc}")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def wipe(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.log_warning(f"state file {self._path} could not be deleted: {exc}")


class JsonlLog:
    """One JSON line per record in the run directory; with no path (outside a runner) it does nothing."""

    def __init__(self, path: Callable[[], Path | None]) -> None:
        self._path = path

    def append(self, record: dict[str, Any]) -> None:
        path = self._path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.log_warning(f"could not append to {path}: {exc}")


def sessions_held(opened_on: date, today: date, trading_dates: Sequence[date]) -> int:
    """Sessions from `opened_on` through `today`, both included, counted on `trading_dates`."""
    return sum(1 for day in trading_dates if opened_on <= day <= today)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_book.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/book.py tests/strategies/earnings_drift/test_drift_book.py
git commit -m "feat: earnings_drift trade records, state file and run logs (Task 7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Scanner (`scanner.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/scanner.py`
- Test: `tests/strategies/earnings_drift/test_drift_scanner.py`

**Interfaces:**
- Consumes: `EventProvider`, `LoadReport` (Task 4); `events_reacting_on` (Task 3); `articles_for`, `pick_surprise`, `article_time` (Task 2);
  `reaction_features`, `bar_dates` (Task 5); `gate`, `Candidate` (Task 6); `strategy.get_historical_prices_for_assets`,
  `strategy.broker.news_provider()`.
- Produces:
  - `ScanResult(today: date, trading_dates: list[date], candidates: list[Candidate], rejections: dict[str, str], hollow: bool = False)`
  - `Scanner(strategy, params, universe: Sequence[str], source: EventProvider, *, benchmark: str = "SPY")` with
    `prepare(held: Collection[str]) -> ScanResult`. Raises `BrokerError`/`BacktestError` only when the benchmark's bars cannot be read.

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_scanner.py
from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

import pandas as pd
import pytest
from tests.fakes import FakeClock, FakeNewsProvider, FrameDataSource, et, weekday_sessions

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.strategies.earnings_drift.event_source import LoadReport
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.scanner import Scanner
from trading_agent_framework.utils.errors import BrokerError

SESSIONS = weekday_sessions(date(2026, 8, 3), 31)
DATES = [s.open.date() for s in SESSIONS]
TODAY = DATES[-1]
FLAT = (100.0, 100.5, 99.5, 100.0, 1_000_000.0)


def _frame(rows: Sequence[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    sessions = SESSIONS[-len(rows) :]
    return pd.DataFrame(list(rows), columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([s.close for s in sessions], name="timestamp"))


FRAMES = {
    ("AAA", "day"): _frame([FLAT] * 30 + [(106.0, 110.0, 105.0, 109.0, 5_000_000.0)]),
    ("BBB", "day"): _frame([FLAT] * 30 + [(106.0, 110.0, 105.0, 109.0, 5_000_000.0)]),
    ("CCC", "day"): _frame([FLAT] * 30 + [(100.0, 101.0, 99.0, 100.5, 1_500_000.0)]),
    ("SPY", "day"): _frame([(400.0, 401.0, 399.0, 400.0, 5e7)] * 30 + [(400.0, 402.0, 399.0, 400.8, 5e7)]),
}


class StaticEvents:
    """An `EventProvider` over fixed events; counts loads; `report` is what `load` answers."""

    def __init__(self, events: list[EarningsEvent], report: LoadReport | None = None, *, reload_every_cycle: bool = False) -> None:
        self.events, self.report, self.loads, self.reload = events, report or LoadReport(loaded=3), 0, reload_every_cycle
        self.loaded = False

    def needs_load(self) -> bool:
        return not self.loaded or self.reload

    def load(self, symbols: Sequence[str], *, as_of: datetime, since: date) -> LoadReport:
        self.loads += 1
        self.loaded = True
        return self.report

    def discard(self) -> None:
        self.loaded = False

    def all_events(self) -> list[EarningsEvent]:
        return list(self.events) if self.loaded else []


def _event(symbol: str, accepted_at: datetime) -> EarningsEvent:
    return EarningsEvent(symbol, accepted_at, f"acc-{symbol}", f"{symbol}.htm")


def _headline(symbol: str, text: str, when: datetime) -> dict[str, object]:
    return {"headline": text, "created_at": when.isoformat(), "symbols": [symbol]}


RELEASE = et(TODAY.year, TODAY.month, TODAY.day, 7, 0)
NEWS = {
    "AAA": [_headline("AAA", "AAA Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate", RELEASE + timedelta(minutes=1))],
    "CCC": [_headline("CCC", "CCC Q3 EPS $0.52 Beats $0.50 Estimate", RELEASE + timedelta(minutes=1))],
}
EVENTS = [
    _event("AAA", RELEASE),
    _event("BBB", RELEASE),  # no headline: no_surprise_data
    _event("CCC", RELEASE),  # weak reaction
    _event("AAA", RELEASE + timedelta(days=1)),  # reacts tomorrow: ignored today
]


def _scanner(tmp_path: Path, source: StaticEvents, *, news: object | None = None, frames: dict | None = None, now: datetime | None = None) -> Scanner:
    clock = FakeClock(now or SESSIONS[-1].close, SESSIONS)
    broker = BacktestBroker(
        "earnings_drift",
        data_source=FrameDataSource(frames or FRAMES, SESSIONS),
        clock=clock,
        budget=D("100000"),
        timestep="day",
        news_source=news if news is not None else FakeNewsProvider(NEWS),  # type: ignore[arg-type]
    )
    strategy = Strategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
    return Scanner(strategy, DriftParams(), ["AAA", "BBB", "CCC"], source)


def test_prepare_finds_today_s_candidates_and_rejections(tmp_path: Path) -> None:
    result = _scanner(tmp_path, StaticEvents(EVENTS)).prepare(held=set())
    assert result.today == TODAY and result.trading_dates == DATES and not result.hollow
    assert [c.symbol for c in result.candidates] == ["AAA"]
    candidate = result.candidates[0]
    assert candidate.reaction_day == TODAY
    assert candidate.reaction.abnormal_pct == pytest.approx(0.088)
    assert candidate.surprise.surprise.eps_actual == D("1.52")
    assert candidate.headlines == ((f"{TODAY.isoformat()} 07:01", NEWS["AAA"][0]["headline"]),)
    assert result.rejections == {"BBB": "no_surprise_data", "CCC": "weak_reaction"}


def test_a_held_symbol_is_rejected(tmp_path: Path) -> None:
    result = _scanner(tmp_path, StaticEvents(EVENTS)).prepare(held={"AAA"})
    assert result.candidates == [] and result.rejections["AAA"] == "already_held"


def test_events_load_once_in_a_backtest(tmp_path: Path) -> None:
    source = StaticEvents(EVENTS)
    scanner = _scanner(tmp_path, source)
    scanner.prepare(held=set())
    scanner.prepare(held=set())
    assert source.loads == 1


def test_a_hollow_load_gives_no_candidates_and_is_retried(tmp_path: Path) -> None:
    source = StaticEvents(EVENTS, LoadReport(loaded=5, failed={f"S{i}": "x" for i in range(25)}))
    scanner = _scanner(tmp_path, source)
    result = scanner.prepare(held=set())
    assert result.hollow and result.candidates == []
    assert source.needs_load()  # discarded: the next cycle loads again
    scanner.prepare(held=set())
    assert source.loads == 2


class _BrokenNews:
    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):  # noqa: ANN001, ANN201
        raise BrokerError("news down")


def test_a_news_failure_rejects_the_events_as_no_news(tmp_path: Path) -> None:
    result = _scanner(tmp_path, StaticEvents(EVENTS), news=_BrokenNews()).prepare(held=set())
    assert result.candidates == []
    assert result.rejections == {"AAA": "no_news", "BBB": "no_news", "CCC": "no_news"}


def test_news_is_asked_in_chunks_from_before_the_first_release(tmp_path: Path) -> None:
    news = FakeNewsProvider(NEWS)
    _scanner(tmp_path, StaticEvents(EVENTS), news=news).prepare(held=set())
    assert [call[0] for call in news.calls] == [("AAA", "BBB", "CCC")]
    assert news.calls[0][1] == RELEASE - timedelta(hours=2) and news.calls[0][2] == SESSIONS[-1].close


def test_no_benchmark_bar_today_gives_no_candidates_and_counts_today(tmp_path: Path) -> None:
    frames = dict(FRAMES)
    frames[("SPY", "day")] = FRAMES[("SPY", "day")].iloc[:-1]
    result = _scanner(tmp_path, StaticEvents(EVENTS), frames=frames).prepare(held=set())
    assert result.candidates == [] and result.trading_dates[-1] == TODAY
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError` (`scanner`)

- [ ] **Step 3: Implement `scanner.py`**

```python
# src/trading_agent_framework/strategies/earnings_drift/scanner.py
"""Today's candidates: earnings events → surprise → reaction → gates (spec §3.6).

Runs once per cycle, right after the close. Reads SEC through an `EventProvider`, news through the broker's
`NewsProvider` (cut at `strategy.clock.now()`), daily bars through `strategy.get_historical_prices_for_assets`
(in a backtest, the no-look-ahead gate). Never orders anything.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.earnings_drift.event_source import EventProvider
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent, events_reacting_on
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.reaction import bar_dates, reaction_features
from trading_agent_framework.strategies.earnings_drift.screening import Candidate, gate
from trading_agent_framework.strategies.earnings_drift.surprise import article_time, articles_for, pick_surprise
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

_MAX_HEADLINES = 5


@dataclass
class ScanResult:
    today: date
    trading_dates: list[date]
    candidates: list[Candidate] = field(default_factory=list)  # best abnormal return first
    rejections: dict[str, str] = field(default_factory=dict)  # symbol -> reason
    hollow: bool = False


class Scanner:
    def __init__(self, strategy: Strategy, params: DriftParams, universe: Sequence[str], source: EventProvider, *, benchmark: str = "SPY") -> None:
        self._strategy = strategy
        self._params = params
        self._universe = [symbol.upper() for symbol in universe]
        self._source = source
        self._benchmark = benchmark

    def prepare(self, held: Collection[str]) -> ScanResult:
        strategy, params = self._strategy, self._params
        now = strategy.clock.now()
        today = now.astimezone(MARKET_TZ).date()
        benchmark = strategy.get_historical_prices(self._benchmark, params.bars_lookback_sessions, "day")
        frame = benchmark.df if benchmark is not None else pd.DataFrame()
        trading_dates = sorted({d for d in (bar_dates(frame) if not frame.empty else []) if d <= today})
        result = ScanResult(today=today, trading_dates=trading_dates)
        if today not in trading_dates:
            strategy.log_warning(f"[earnings_drift] {self._benchmark} has no bar for {today}: no candidates this cycle")
            result.trading_dates = [*trading_dates, today]  # today still counts as a session for the holdings
            return result
        if self._source.needs_load():
            report = self._source.load(self._universe, as_of=now, since=today - timedelta(days=params.event_lookback_days))
            if report.failed:
                sample = ", ".join(sorted(report.failed)[:5])
                strategy.log_warning(f"[earnings_drift] SEC submissions failed for {len(report.failed)} of {len(self._universe)} symbols (e.g. {sample})")
            if report.is_hollow(params.sec_hollow_fraction, params.sec_hollow_min_failures):
                self._source.discard()
                strategy.log_error(f"[earnings_drift] hollow scan: SEC failed for {len(report.failed)} symbols; no candidates this cycle")
                result.hollow = True
                return result
        events = events_reacting_on(self._source.all_events(), today, trading_dates, now)
        if not events:
            strategy.log_info(f"[earnings_drift] no earnings reactions on {today}")
            return result
        news = self._news(events, now)
        bars = self._bars([event.symbol for event in events])
        held_symbols = {symbol.upper() for symbol in held}
        for event in events:
            reason, candidate = self._judge(event, today, now, news, bars, frame, held_symbols)
            if candidate is not None:
                result.candidates.append(candidate)
            elif reason is not None:
                result.rejections[event.symbol] = reason
        result.candidates.sort(key=lambda c: c.reaction.abnormal_pct, reverse=True)
        counts: dict[str, int] = {}
        for reason in result.rejections.values():
            counts[reason] = counts.get(reason, 0) + 1
        strategy.log_info(f"[earnings_drift] {today}: {len(events)} earnings reactions, {len(result.candidates)} candidates, rejected {counts}")
        return result

    def _judge(
        self,
        event: EarningsEvent,
        today: date,
        now: datetime,
        news: Mapping[str, list[Mapping[str, Any]]] | None,
        bars: Mapping[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        held: set[str],
    ) -> tuple[str | None, Candidate | None]:
        if event.symbol in held:
            return "already_held", None
        if news is None or event.symbol not in news:
            return "no_news", None
        lookback = timedelta(hours=self._params.surprise_lookback_hours)
        window = articles_for(news[event.symbol], event.symbol, start=event.accepted_at - lookback, end=now)
        picked = pick_surprise(window)
        frame = bars.get(event.symbol)
        reaction = reaction_features(frame, benchmark, today, baseline_sessions=self._params.volume_baseline_sessions) if frame is not None else None
        reason = gate(picked.surprise if picked else None, reaction, self._params, held=False)
        if reason is not None or picked is None or reaction is None:
            return reason, None
        headlines = []
        for article in window[:_MAX_HEADLINES]:
            created = article_time(article)
            if created is not None:
                headlines.append((created.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M"), str(article.get("headline") or "")))
        return None, Candidate(event=event, reaction_day=today, surprise=picked, reaction=reaction, headlines=tuple(headlines))

    def _news(self, events: Sequence[EarningsEvent], now: datetime) -> dict[str, list[Mapping[str, Any]]] | None:
        """Articles per event symbol, read in chunks; a failed chunk's symbols are missing (`no_news`); None without a provider."""
        try:
            provider = self._strategy.broker.news_provider()
        except BrokerError as exc:
            self._strategy.log_warning(f"[earnings_drift] no news provider: {exc}")
            return None
        if provider is None:
            self._strategy.log_warning("[earnings_drift] this broker has no news provider: every event is rejected as no_news")
            return None
        start = min(event.accepted_at for event in events) - timedelta(hours=self._params.surprise_lookback_hours)
        symbols = sorted({event.symbol for event in events})
        found: dict[str, list[Mapping[str, Any]]] = {}
        size = self._params.news_symbols_per_call
        for first in range(0, len(symbols), size):
            chunk = symbols[first : first + size]
            try:
                articles = provider.get_news(chunk, start=start, end=now, limit=self._params.news_limit)
            except BrokerError as exc:
                self._strategy.log_warning(f"[earnings_drift] news failed for {', '.join(chunk)}: {exc}")
                continue
            for symbol in chunk:
                found[symbol] = list(articles)
        return found

    def _bars(self, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
        try:
            fetched = self._strategy.get_historical_prices_for_assets([Asset(symbol) for symbol in symbols], self._params.bars_lookback_sessions, "day")
        except (BrokerError, BacktestError) as exc:
            self._strategy.log_warning(f"[earnings_drift] daily bars failed for {len(symbols)} symbols: {exc}")
            return {}
        return {asset.symbol: bars.df for asset, bars in fetched.items() if bars is not None and not bars.df.empty}
```

Check `test_news_is_asked_in_chunks_from_before_the_first_release`: three event symbols fit one chunk of 5,
`start = RELEASE - 2h`, `end = now` (the session close). `FakeNewsProvider.calls` holds `(symbols, start, end, limit, include_content)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_scanner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/scanner.py tests/strategies/earnings_drift/test_drift_scanner.py
git commit -m "feat: earnings_drift scanner builds today's candidates after the close (Task 8)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Desk part 1 — entries, stops on fill, closes, decisions, baseline (`desk.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/desk.py`
- Modify: `tests/strategies/earnings_drift/drift_helpers.py` (add `DeskRig`)
- Test: `tests/strategies/earnings_drift/test_drift_desk_entries.py`

**Interfaces:**
- Consumes: `Trade`, `TradeState`, `DriftState`, `JsonlLog`, `sessions_held` (Task 7); `Candidate` (Task 6); `fact_sheet` (Task 6); `DriftParams`.
- Produces (`Desk(strategy, params, state, *, save: Callable[[], None] = lambda: None, trade_log: JsonlLog, decision_log: JsonlLog)`):
  - `begin_session(today: date, trading_dates: Sequence[date], candidates: Sequence[Candidate], rejections: Mapping[str, str]) -> None`
  - views: `trades -> list[Trade]`, `exposed_symbols() -> set[str]`, `free_slots() -> int`, `max_quantity(symbol: str) -> int`,
    `candidate_sheets() -> list[dict]`, `holdings_context() -> list[dict]`, `balances() -> dict`
  - tools: `buy(symbol, quantity, trail_percent, reason) -> dict`, `skip(symbol, reason) -> dict`
  - `baseline_entries() -> list[str]`, `record_undecided() -> None`, `save() -> None`
  - hooks: `on_order_filled(order: Order) -> None`
  - `ALREADY_CLOSED = "already_closed"`
  - private helpers used by Tasks 10-11: `_protect(trade, trail) -> bool`, `_submit_stop(trade, trail) -> Order | None`,
    `_market_sell(trade, exit_reason) -> Order | None`, `_close(trade, order, reason=None)`, `_book_partial(trade, order, reason="trail")`,
    `_lookup(order_id) -> Order | None`, `_refuse(guardrail, message) -> dict`, `_sessions_held(trade) -> int`, `_now_date() -> date`.

- [ ] **Step 1: Add the desk rig to the helpers**

Append to `tests/strategies/earnings_drift/drift_helpers.py`:

```python
import json
from decimal import Decimal as D
from pathlib import Path
from typing import Any

import pandas as pd
from tests.fakes import FakeClock, FrameDataSource, weekday_sessions

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.strategies.earnings_drift.book import DriftState, JsonlLog
from trading_agent_framework.strategies.earnings_drift.desk import Desk
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams

RIG_SESSIONS = weekday_sessions(date(2026, 9, 1), 8)
RIG_DATES = [s.open.date() for s in RIG_SESSIONS]
# AAA: entry fills at day 1's open (101); a stop placed at day 1's close (102) trails to 106 on day 2 and fills on day 3 at 97.52.
AAA_ROWS = [
    (100.0, 101.0, 99.0, 100.0, 1e6),
    (101.0, 103.0, 100.0, 102.0, 1e6),
    (102.0, 106.0, 101.0, 105.0, 1e6),
    (104.0, 104.0, 95.0, 96.0, 1e6),
] + [(96.0, 97.0, 95.5, 96.0, 1e6)] * 4
BBB_ROWS = [(50.0, 50.5, 49.5, 50.0, 1e6)] * 8


def rig_frame(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([s.close for s in RIG_SESSIONS[: len(rows)]], name="timestamp"))


class DeskRig:
    """A `Desk` over a real `BacktestBroker` on daily bars, starting at day 0's close (the cycle's time)."""

    def __init__(self, tmp_path: Path, *, params: DriftParams | None = None, candidates: list[Candidate] | None = None, budget: D = D("100000")) -> None:
        self.clock = FakeClock(RIG_SESSIONS[0].close, RIG_SESSIONS)
        frames = {("AAA", "day"): rig_frame(AAA_ROWS), ("BBB", "day"): rig_frame(BBB_ROWS)}
        self.broker = BacktestBroker("earnings_drift", data_source=FrameDataSource(frames, RIG_SESSIONS), clock=self.clock, budget=budget, timestep="day")
        self.strategy = Strategy(self.broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
        self.state = DriftState()
        self.trades_path, self.decisions_path = tmp_path / "trades.jsonl", tmp_path / "decisions.jsonl"
        self.desk = Desk(self.strategy, params or DriftParams(), self.state, trade_log=JsonlLog(lambda: self.trades_path), decision_log=JsonlLog(lambda: self.decisions_path))
        self.day = 0
        self.delivered: set[str] = set()
        default = [make_candidate("AAA", day=RIG_DATES[0], close=100.0, reaction_low=99.0), make_candidate("BBB", day=RIG_DATES[0], close=50.0, abnormal_pct=0.04)]
        self.desk.begin_session(RIG_DATES[0], RIG_DATES[:1], default if candidates is None else candidates, {})

    def advance(self) -> None:
        """Move to the next session's close and let the broker fill what it can (no hook delivered)."""
        before = self.clock.now()
        self.day += 1
        self.clock.advance((RIG_SESSIONS[self.day].close - before).total_seconds())
        self.broker.on_advance(before, self.clock.now())

    def deliver(self) -> None:
        """Deliver every fill the desk has not seen yet, as the executor would."""
        for order in self.strategy.get_orders():
            if order.is_filled() and order.identifier not in self.delivered:
                self.delivered.add(order.identifier)
                self.desk.on_order_filled(order)

    def next_close(self, candidates: list[Candidate] | None = None) -> None:
        self.advance()
        self.deliver()
        self.desk.begin_session(RIG_DATES[self.day], RIG_DATES[: self.day + 1], candidates or [], {})

    def open_aaa(self, quantity: int = 10, trail: float = 8.0) -> None:
        assert "error" not in self.desk.buy("AAA", quantity, trail, "beat and raise")
        self.next_close()

    def lines(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
```

Move the `from __future__ import annotations` line and the imports to the top of the file when appending (ruff's
import sorting will flag a split import block).

- [ ] **Step 2: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_desk_entries.py
from __future__ import annotations

import logging
import threading
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import RIG_DATES, DeskRig, make_candidate

from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.strategies.earnings_drift.book import TradeState
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.utils.errors import OrderValidationError


def test_max_quantity_is_the_slot_budget_over_the_reaction_close(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    assert rig.desk.max_quantity("AAA") == 125  # 100000 / 8 / 100
    assert rig.desk.max_quantity("BBB") == 250
    assert rig.desk.max_quantity("ZZZ") == 0


def test_buy_submits_a_day_market_order_and_records_a_pending_trade(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    result = rig.desk.buy("aaa", 10, 8.0, "beat and raise")
    assert result["symbol"] == "AAA" and result["trail_percent"] == 8.0
    trade = rig.state.trades["AAA"]
    assert trade.state is TradeState.PENDING and trade.trail_percent == D("8.0") and trade.thesis == "beat and raise"
    assert trade.reaction_low == D("99.0") and trade.accession_number == "0000000000-26-AAA"
    order = rig.strategy.get_order(trade.entry_order_id)
    assert order is not None and order.side is OrderSide.BUY and order.order_type is OrderType.MARKET and order.time_in_force is TimeInForce.DAY
    assert rig.state.traded_symbols == {"AAA"}
    assert rig.desk.max_quantity("BBB") == 250  # still the 12500 slot cap: 99000 is left after the 1000 committed
    [line] = rig.lines(rig.decisions_path)
    assert line["decision"] == "buy" and line["quantity"] == 10 and line["features"]["abnormal_pct"] == 0.075


@pytest.mark.parametrize(
    ("symbol", "quantity", "trail", "message"),
    [
        ("ZZZ", 10, 8.0, "not one of today's candidates"),
        ("AAA", 2.5, 8.0, "whole number"),
        ("AAA", 0, 8.0, "whole number"),
        ("AAA", -3, 8.0, "whole number"),
        ("AAA", float("nan"), 8.0, "whole number"),
        ("AAA", "ten", 8.0, "whole number"),
        ("AAA", 10, 2.9, "trail_percent"),
        ("AAA", 10, 15.1, "trail_percent"),
        ("AAA", 126, 8.0, "above max_quantity 125"),
    ],
)
def test_buy_refuses_bad_quantities_and_trails(tmp_path: Path, caplog: pytest.LogCaptureFixture, symbol: str, quantity: object, trail: float, message: str) -> None:
    rig = DeskRig(tmp_path)
    with caplog.at_level(logging.WARNING):
        result = rig.desk.buy(symbol, quantity, trail, "x")  # type: ignore[arg-type]
    assert message in result["error"]
    assert "guardrail order_limits" in caplog.text
    assert rig.state.trades == {}


def test_buy_refuses_a_second_entry_and_a_full_book(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_positions=1))
    assert "error" not in rig.desk.buy("AAA", 10, 8.0, "x")
    assert "already held or being bought" in rig.desk.buy("AAA", 1, 8.0, "x")["error"]
    assert "max_positions (1) reached" in rig.desk.buy("BBB", 1, 8.0, "x")["error"]


def test_two_buys_on_parallel_threads_respect_max_positions(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_positions=1))
    results: list[dict] = []
    barrier = threading.Barrier(2)

    def buy(symbol: str) -> None:
        barrier.wait()
        results.append(rig.desk.buy(symbol, 5, 8.0, "x"))

    threads = [threading.Thread(target=buy, args=(s,)) for s in ("AAA", "BBB")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum("error" in r for r in results) == 1 and len(rig.state.trades) == 1


def test_a_broker_refusal_is_returned_as_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)

    def refuse(order):  # noqa: ANN001, ANN202
        raise OrderValidationError("insufficient cash")

    monkeypatch.setattr(rig.strategy, "submit_order", refuse)
    assert rig.desk.buy("AAA", 10, 8.0, "x") == {"error": "insufficient cash"}
    assert rig.state.trades == {}


def test_the_entry_fills_at_the_next_open_and_gets_its_trailing_stop(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    trade = rig.state.trades["AAA"]
    assert trade.state is TradeState.OPEN and trade.quantity == D(10) and trade.entry_price == D("101.0")
    assert trade.opened_on == RIG_DATES[1]
    stop = rig.strategy.get_order(trade.stop_order_id)  # type: ignore[arg-type]
    assert stop is not None and stop.order_type is OrderType.TRAIL and stop.trail_percent == D("8.0") and stop.time_in_force is TimeInForce.GTC
    assert stop.quantity == D(10) and stop.is_active()


def test_the_stop_fill_closes_the_trade_into_trades_jsonl(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    rig.next_close()
    rig.next_close()  # day 3: the low 95 crosses 106 x 0.92 = 97.52
    assert rig.state.trades == {}
    [line] = rig.lines(rig.trades_path)
    assert line["symbol"] == "AAA" and line["exit_reason"] == "trail"
    assert D(line["entry_price"]) == D("101") and D(line["exit_price"]) == D("97.52")
    assert line["sessions_held"] == 3 and line["entry_date"] == RIG_DATES[1].isoformat() and line["exit_date"] == RIG_DATES[3].isoformat()
    assert D(line["pnl"]) == D("-34.80") and line["agent_enabled"] is True and line["thesis"] == "beat and raise"


def test_a_stop_refused_twice_is_replaced_by_a_market_sell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    assert "error" not in rig.desk.buy("AAA", 10, 8.0, "x")
    real_submit = rig.strategy.submit_order

    def no_trailing_stops(order):  # noqa: ANN001, ANN202
        if order.order_type is OrderType.TRAIL:
            raise OrderValidationError("trailing stops are down")
        return real_submit(order)

    monkeypatch.setattr(rig.strategy, "submit_order", no_trailing_stops)
    with caplog.at_level(logging.WARNING):
        rig.next_close()
    trade = rig.state.trades["AAA"]
    assert trade.stop_order_id is None and trade.exit_reason == "backstop_sell"
    sell = rig.strategy.get_order(trade.exit_order_id)  # type: ignore[arg-type]
    assert sell is not None and sell.side is OrderSide.SELL and sell.order_type is OrderType.MARKET
    rig.next_close()
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "backstop_sell"


def test_skip_and_undecided_are_recorded_once(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    assert rig.desk.skip("BBB", "guidance cut") == {"symbol": "BBB", "status": "skipped"}
    assert "already decided" in rig.desk.skip("BBB", "again")["error"]
    assert "not one of today's candidates" in rig.desk.skip("ZZZ", "x")["error"]
    rig.desk.record_undecided()
    rig.desk.record_undecided()
    decisions = [(line["symbol"], line["decision"]) for line in rig.lines(rig.decisions_path)]
    assert decisions == [("BBB", "skip"), ("AAA", "undecided")]


def test_rejections_are_logged_at_begin_session(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.desk.begin_session(RIG_DATES[0], RIG_DATES[:1], [], {"CCC": "faded"})
    assert rig.lines(rig.decisions_path) == [{"date": RIG_DATES[0].isoformat(), "symbol": "CCC", "decision": "rejected", "reason": "faded"}]


def test_baseline_buys_every_candidate_best_first_up_to_the_free_slots(tmp_path: Path) -> None:
    candidates = [make_candidate("BBB", day=RIG_DATES[0], close=50.0, abnormal_pct=0.04), make_candidate("AAA", day=RIG_DATES[0], close=100.0, abnormal_pct=0.09)]
    rig = DeskRig(tmp_path, params=DriftParams(agent_enabled=False, max_positions=1), candidates=candidates)
    assert rig.desk.baseline_entries() == ["AAA"]
    trade = rig.state.trades["AAA"]
    assert trade.trail_percent == D("8.0") and trade.thesis == "baseline"
    order = rig.strategy.get_order(trade.entry_order_id)
    assert order is not None and order.quantity == D(1000)  # one slot: the whole 100000, at the reaction close of 100
```

Then add the context-view tests:

```python
def test_candidate_sheets_and_balances(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    sheets = rig.desk.candidate_sheets()
    assert [s["symbol"] for s in sheets] == ["AAA", "BBB"]  # best abnormal return first
    assert sheets[0]["max_quantity"] == 125
    assert rig.desk.balances() == {"portfolio_value": 100000.0, "cash": 100000.0, "buying_power": 100000.0, "free_slots": 8}


def test_holdings_context_lists_open_trades(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    [row] = rig.desk.holdings_context()
    assert row == {
        "symbol": "AAA",
        "entry_date": RIG_DATES[1].isoformat(),
        "entry_price": 101.0,
        "last_close": 102.0,
        "pnl_pct": 1.0,
        "sessions_held": 1,
        "trail_percent": 8.0,
        "stop_status": "working",
        "thesis": "beat and raise",
        "reaction_low": 99.0,
    }
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_desk_entries.py -v`
Expected: FAIL with `ModuleNotFoundError` (`desk`)

- [ ] **Step 4: Implement `desk.py` (part 1)**

```python
# src/trading_agent_framework/strategies/earnings_drift/desk.py
"""Every order earnings_drift places (spec §6): the agent's buy / trail / sell, stops on fills, the guardrails.

The only module of the strategy that submits or cancels orders. The agent reaches it through `tools.py`, the
strategy through `reconcile` / `ensure_stops` / `baseline_entries` and the order hooks. Tool paths return
`{"error": ...}` (logging a warning that names the guardrail) instead of raising; hook paths log and never raise.
A filled position is never left without a stop: a stop refused twice is replaced by an immediate market sell.

Orders are sent right after the close and fill at the next open. The agent's tools may run on LangGraph worker
threads, several at once: every public method holds one re-entrant lock (re-entrant because a cancel wait
dispatches order hooks on the same thread).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.earnings_drift.book import DriftState, JsonlLog, Trade, TradeState, sessions_held
from trading_agent_framework.strategies.earnings_drift.fact_sheet import fact_sheet
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.screening import Candidate
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

ALREADY_CLOSED = "already_closed"
_DATA_ERRORS = (BrokerError, BacktestError)


def _lean(order: Order) -> dict[str, Any]:
    return {"identifier": order.identifier, "symbol": order.asset.symbol, "side": order.side.value, "quantity": float(order.quantity or 0), "status": order.status.value}


def _number(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


class Desk:
    def __init__(
        self,
        strategy: Strategy,
        params: DriftParams,
        state: DriftState,
        *,
        save: Callable[[], None] = lambda: None,
        trade_log: JsonlLog,
        decision_log: JsonlLog,
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._state = state
        self._save_state = save
        self._trade_log = trade_log
        self._decision_log = decision_log
        self._lock = threading.RLock()
        # Stops this desk cancelled itself: their CANCELED hook must not be read as an outside cancel.
        self._expected_cancels: set[str] = set()
        self._today: date | None = None
        self._trading_dates: list[date] = []
        self._candidates: dict[str, Candidate] = {}
        self._decided: set[str] = set()
        self._cycle_buys = Decimal(0)  # estimated cost of the buys submitted this cycle
        self._cycle_sell_proceeds = Decimal(0)  # estimated proceeds of the sells submitted this cycle

    # --- session ---------------------------------------------------------------------------------

    def begin_session(self, today: date, trading_dates: Sequence[date], candidates: Sequence[Candidate], rejections: Mapping[str, str]) -> None:
        with self._lock:
            self._today = today
            self._trading_dates = sorted(set(trading_dates))
            self._candidates = {candidate.symbol: candidate for candidate in candidates}
            self._decided = set()
            self._cycle_buys = Decimal(0)
            self._cycle_sell_proceeds = Decimal(0)
            for symbol, reason in sorted(rejections.items()):
                self._log_decision(symbol, "rejected", reason=reason)

    def save(self) -> None:
        self._save_state()

    # --- views -----------------------------------------------------------------------------------

    @property
    def trades(self) -> list[Trade]:
        return list(self._state.trades.values())

    def exposed_symbols(self) -> set[str]:
        """Symbols with a trade of ours or any long position: an earnings event in one of them is `already_held`."""
        with self._lock:
            symbols = set(self._state.trades)
            try:
                symbols |= {position.asset.symbol for position in self._strategy.get_positions() if position.quantity > 0}
            except _DATA_ERRORS as exc:
                self._strategy.log_warning(f"[earnings_drift] positions unavailable, only this strategy's trades count as held: {exc}")
            return symbols

    def free_slots(self) -> int:
        """`max_positions` minus the trades not already being sold (a pending exit frees its slot)."""
        with self._lock:
            taken = sum(1 for trade in self._state.trades.values() if trade.exit_order_id is None)
            return max(0, self._params.max_positions - taken)

    def max_quantity(self, symbol: str) -> int:
        """Whole shares of a candidate the budget allows now: the slot cap, within what is still available (CLAUDE.md sizing rule)."""
        with self._lock:
            candidate = self._candidates.get(symbol.strip().upper())
            if candidate is None:
                return 0
            price = Decimal(str(candidate.reaction.close))
            try:
                account = self._strategy.broker.get_account()
            except _DATA_ERRORS as exc:
                self._strategy.log_warning(f"[earnings_drift] account unavailable, no buy can be sized: {exc}")
                return 0
            cap = account.portfolio_value / self._params.max_positions
            available = min(account.buying_power, account.cash + self._cycle_sell_proceeds) - self._cycle_buys
            budget = min(cap, available)
            if price <= 0 or budget <= 0:
                return 0
            return int((budget / price).to_integral_value(rounding=ROUND_FLOOR))

    def candidate_sheets(self) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._candidates.values(), key=lambda c: c.reaction.abnormal_pct, reverse=True)
            return [fact_sheet(candidate, self.max_quantity(candidate.symbol)) for candidate in ordered]

    def holdings_context(self) -> list[dict[str, Any]]:
        """Open trades not being sold, as the agent sees them; a backstop flag is shown once, then cleared."""
        with self._lock:
            rows: list[dict[str, Any]] = []
            for trade in sorted(self._open_trades(), key=lambda t: t.symbol):
                last = self._price(trade.symbol)
                pnl = None
                if last is not None and trade.entry_price:
                    pnl = round(float((last / trade.entry_price - 1) * 100), 1)
                status = "backstop" if trade.backstop else ("working" if self._stop_is_working(trade) else "missing")
                trade.backstop = False
                rows.append(
                    {
                        "symbol": trade.symbol,
                        "entry_date": trade.opened_on.isoformat() if trade.opened_on else None,
                        "entry_price": float(trade.entry_price) if trade.entry_price is not None else None,
                        "last_close": float(last) if last is not None else None,
                        "pnl_pct": pnl,
                        "sessions_held": self._sessions_held(trade),
                        "trail_percent": float(trade.trail_percent),
                        "stop_status": status,
                        "thesis": trade.thesis,
                        "reaction_low": float(trade.reaction_low) if trade.reaction_low is not None else None,
                    }
                )
            return rows

    def balances(self) -> dict[str, Any]:
        try:
            account = self._strategy.broker.get_account()
        except _DATA_ERRORS as exc:
            return {"error": str(exc), "free_slots": self.free_slots()}
        return {
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "free_slots": self.free_slots(),
        }

    # --- agent tools -----------------------------------------------------------------------------

    def buy(self, symbol: str, quantity: object, trail_percent: object, reason: str) -> dict[str, Any]:
        return self._buy(symbol, quantity, trail_percent, reason, decision="buy")

    def skip(self, symbol: str, reason: str) -> dict[str, Any]:
        with self._lock:
            symbol = str(symbol).strip().upper()
            if symbol not in self._candidates:
                return {"error": f"{symbol} is not one of today's candidates"}
            if symbol in self._decided:
                return {"error": f"{symbol} was already decided this session"}
            self._decided.add(symbol)
            self._log_decision(symbol, "skip", reason=self._clip(reason))
            return {"symbol": symbol, "status": "skipped"}

    # --- baseline and end of cycle ----------------------------------------------------------------

    def baseline_entries(self) -> list[str]:
        """Baseline mode: every candidate, best abnormal return first, at `max_quantity` and the default trail."""
        with self._lock:
            bought: list[str] = []
            for candidate in sorted(self._candidates.values(), key=lambda c: c.reaction.abnormal_pct, reverse=True):
                if self.free_slots() <= 0:
                    break
                quantity = self.max_quantity(candidate.symbol)
                if quantity < 1:
                    self._strategy.log_info(f"[earnings_drift] baseline: no budget left for {candidate.symbol}")
                    continue
                result = self._buy(candidate.symbol, quantity, self._params.default_trail_percent, "baseline", decision="baseline")
                if "error" not in result:
                    bought.append(candidate.symbol)
            return bought

    def record_undecided(self) -> None:
        with self._lock:
            for symbol in sorted(set(self._candidates) - self._decided):
                self._decided.add(symbol)
                self._log_decision(symbol, "undecided")

    # --- order hooks -----------------------------------------------------------------------------

    def on_order_filled(self, order: Order) -> None:
        with self._lock:
            trade = self._trade_for(order)
            if trade is None:
                return
            if order.identifier == trade.entry_order_id:
                if trade.state is TradeState.PENDING:
                    self._entry_filled(trade, order)
            else:
                self._close(trade, order)
            self._save_state()

    # --- internals -------------------------------------------------------------------------------

    def _buy(self, symbol: str, quantity: object, trail_percent: object, reason: str, *, decision: str) -> dict[str, Any]:
        with self._lock:
            symbol = str(symbol).strip().upper()
            shares, trail = _number(quantity), _number(trail_percent)
            low, high = self._params.min_trail_percent, self._params.max_trail_percent
            candidate = self._candidates.get(symbol)
            refusal = None
            if candidate is None:
                refusal = f"{symbol} is not one of today's candidates"
            elif symbol in self._state.trades:
                refusal = f"{symbol} is already held or being bought"
            elif shares is None or shares < 1 or shares != shares.to_integral_value():
                refusal = f"quantity must be a whole number of shares, at least 1 (got {quantity!r})"
            elif trail is None or not Decimal(str(low)) <= trail <= Decimal(str(high)):
                refusal = f"trail_percent must be between {low} and {high} (got {trail_percent!r})"
            elif self.free_slots() <= 0:
                refusal = f"max_positions ({self._params.max_positions}) reached"
            else:
                cap = self.max_quantity(symbol)
                if shares > cap:
                    refusal = f"quantity {shares} is above max_quantity {cap}"
            if refusal is not None:
                return self._refuse("order_limits", f"buy {symbol}: {refusal}")
            assert candidate is not None and shares is not None and trail is not None
            try:
                order = self._strategy.submit_order(self._strategy.create_order(symbol, shares, "buy"))
            except Exception as exc:  # a broker's _submit_order may re-raise after order.set_error (lumibot contract)
                self._strategy.log_warning(f"[earnings_drift] buy {symbol} refused by the broker: {exc}")
                return {"error": str(exc)}
            thesis = self._clip(reason)
            self._state.trades[symbol] = Trade(
                symbol=symbol,
                entry_order_id=order.identifier,
                trail_percent=trail,
                thesis=thesis,
                accession_number=candidate.event.accession_number,
                reaction_low=Decimal(str(candidate.reaction.reaction_low)),
            )
            self._state.traded_symbols.add(symbol)
            self._cycle_buys += shares * Decimal(str(candidate.reaction.close))
            self._decided.add(symbol)
            self._log_decision(symbol, decision, quantity=int(shares), trail_percent=float(trail), reason=thesis)
            self._save_state()
            return {**_lean(order), "trail_percent": float(trail), "note": "fills at the next open; the trailing stop is placed when it fills"}

    def _entry_filled(self, trade: Trade, order: Order) -> None:
        trade.state = TradeState.OPEN
        trade.quantity = order.filled_quantity
        trade.entry_price = order.avg_fill_price
        trade.opened_on = self._now_date()
        self._strategy.log_info(f"[earnings_drift] entry {trade.symbol}: {trade.quantity} @ {trade.entry_price}; placing a {trade.trail_percent}% trailing stop")
        self._protect(trade, trade.trail_percent)

    def _close(self, trade: Trade, order: Order, reason: str | None = None) -> None:
        """A stop or exit sell filled: the trade leaves the book and is appended to `trades.jsonl`."""
        if reason is None:
            reason = (trade.exit_reason or "agent_sell") if order.identifier == trade.exit_order_id else "trail"
        exit_price = order.avg_fill_price
        quantity = order.filled_quantity or trade.quantity
        record: dict[str, Any] = {
            "symbol": trade.symbol,
            "accession_number": trade.accession_number,
            "entry_date": trade.opened_on.isoformat() if trade.opened_on else None,
            "entry_price": trade.entry_price,
            "exit_date": self._now_date().isoformat(),
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": None,
            "return_pct": None,
            "r_multiple": None,
            "sessions_held": self._sessions_held(trade),
            "exit_reason": reason,
            "trail_percent": trade.trail_percent,
            "thesis": trade.thesis,
            "agent_enabled": self._params.agent_enabled,
        }
        if exit_price is not None and trade.entry_price:
            ret = exit_price / trade.entry_price - 1
            record["pnl"] = (exit_price - trade.entry_price) * quantity
            record["return_pct"] = round(ret * 100, 2)
            record["r_multiple"] = round(ret / (trade.trail_percent / 100), 2)
        self._state.trades.pop(trade.symbol, None)
        self._trade_log.append(record)
        self._strategy.log_info(f"[earnings_drift] trade {trade.symbol} closed ({reason}): P&L {record['pnl']}")

    def _book_partial(self, trade: Trade, order: Order, reason: str = "trail") -> None:
        """Shares an ended stop or sell had already sold: deducted; the trade closes if none are left."""
        sold = order.filled_quantity
        if sold <= 0:
            return
        trade.quantity -= sold
        self._strategy.log_info(f"[earnings_drift] {trade.symbol}: {sold} shares sold by {order.identifier} before it ended; {trade.quantity} left")
        if trade.quantity <= 0:
            self._close(trade, order, reason)

    def _protect(self, trade: Trade, trail: Decimal) -> bool:
        """Place the trade's trailing stop (two attempts); a market sell when both are refused. True when a stop is working."""
        if self._submit_stop(trade, trail) is not None or self._submit_stop(trade, trail) is not None:
            return True
        self._strategy.log_error(f"[earnings_drift] no stop could be placed for {trade.symbol}; selling {trade.quantity} at the next open")
        self._market_sell(trade, "backstop_sell")
        return False

    def _submit_stop(self, trade: Trade, trail: Decimal) -> Order | None:
        if trade.quantity <= 0:
            return None
        order = self._strategy.create_order(trade.symbol, trade.quantity, "sell", trail_percent=trail, time_in_force="gtc")
        try:
            submitted = self._strategy.submit_order(order)
        except Exception as exc:  # a broker's _submit_order may re-raise after order.set_error (lumibot contract)
            self._strategy.log_warning(f"[earnings_drift] stop for {trade.symbol} refused: {exc}")
            return None
        trade.stop_order_id = submitted.identifier
        return submitted

    def _market_sell(self, trade: Trade, exit_reason: str) -> Order | None:
        try:
            submitted = self._strategy.submit_order(self._strategy.create_order(trade.symbol, trade.quantity, "sell"))
        except Exception as exc:  # a broker's _submit_order may re-raise after order.set_error (lumibot contract)
            self._strategy.log_error(f"[earnings_drift] market sell of {trade.quantity} {trade.symbol} failed ({exit_reason}): {exc}")
            return None
        trade.exit_order_id, trade.exit_reason = submitted.identifier, exit_reason
        price = self._price(trade.symbol)
        if price is not None:
            self._cycle_sell_proceeds += trade.quantity * price
        return submitted

    def _open_trades(self) -> list[Trade]:
        return [t for t in self._state.trades.values() if t.state is TradeState.OPEN and t.exit_order_id is None]

    def _trade_for(self, order: Order) -> Trade | None:
        trade = self._state.trades.get(order.asset.symbol)
        if trade is None or order.identifier not in (trade.entry_order_id, trade.stop_order_id, trade.exit_order_id):
            return None
        return trade

    def _lookup(self, order_id: str | None) -> Order | None:
        if not order_id:
            return None
        try:
            return self._strategy.get_order(order_id)
        except Exception as exc:  # strategy.get_order can fall through to a raw SDK lookup
            self._strategy.log_warning(f"[earnings_drift] order {order_id} lookup failed: {exc}")
            return None

    def _stop_is_working(self, trade: Trade) -> bool:
        stop = self._lookup(trade.stop_order_id)
        return stop is not None and stop.is_active()

    def _price(self, symbol: str) -> Decimal | None:
        try:
            return self._strategy.get_last_price(symbol)
        except _DATA_ERRORS:
            return None

    def _now_date(self) -> date:
        return self._strategy.clock.now().astimezone(MARKET_TZ).date()

    def _sessions_held(self, trade: Trade) -> int:
        if trade.opened_on is None:
            return 0
        today = self._now_date()
        dates = [day for day in self._trading_dates if day <= today]
        if not dates or dates[-1] < today:
            dates.append(today)  # a hook between cycles: today's session counts
        return sessions_held(trade.opened_on, today, dates)

    def _refuse(self, guardrail: str, message: str) -> dict[str, Any]:
        self._strategy.log_warning(f"guardrail {guardrail}: {message}")
        return {"error": message}

    def _clip(self, reason: str) -> str:
        return str(reason or "").strip()[: self._params.reason_max_chars]

    def _log_decision(self, symbol: str, decision: str, **fields: Any) -> None:
        record: dict[str, Any] = {"date": self._today.isoformat() if self._today else None, "symbol": symbol, "decision": decision, **fields}
        candidate = self._candidates.get(symbol)
        if candidate is not None:
            record["features"] = {
                "abnormal_pct": candidate.reaction.abnormal_pct,
                "rel_volume": candidate.reaction.rel_volume,
                "hold_ratio": candidate.reaction.hold_ratio,
                "eps_surprise_pct": candidate.surprise.surprise.eps_surprise_pct,
                "sales_result": candidate.surprise.surprise.sales_result,
            }
        self._decision_log.append(record)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_desk_entries.py -v`
Expected: PASS. If `test_the_stop_fill_closes_the_trade_into_trades_jsonl` reads an exit price other than 97.52,
check that the stop was submitted at day 1's close exactly (the rig delivers fills at the close, so `needs_skip` is
False and the trail reference is day 1's close, 102).

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/desk.py tests/strategies/earnings_drift/drift_helpers.py tests/strategies/earnings_drift/test_drift_desk_entries.py
git commit -m "feat: earnings_drift desk buys, protects every fill with a trailing stop, logs decisions and trades (Task 9)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Desk part 2 — trail changes, agent sells, cancel hooks

**Files:**
- Modify: `src/trading_agent_framework/strategies/earnings_drift/desk.py` (add the methods below to `Desk`)
- Test: `tests/strategies/earnings_drift/test_drift_desk_exits.py`

**Interfaces:**
- Consumes: Task 9's `Desk` internals.
- Produces: `set_trailing_stop(symbol, trail_percent, reason) -> dict`, `sell(symbol, reason) -> dict`,
  `on_order_canceled(order: Order) -> None`; internals `_release_stop(trade) -> str | None`, `_exit(trade, exit_reason) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_desk_exits.py
from __future__ import annotations

import logging
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import DeskRig

from trading_agent_framework.entities.enums import OrderType
from trading_agent_framework.utils.errors import OrderValidationError


def _open(tmp_path: Path) -> DeskRig:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    return rig


def test_tightening_replaces_the_stop(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    old = rig.state.trades["AAA"].stop_order_id
    result = rig.desk.set_trailing_stop("AAA", 5.0, "lock in gains")
    assert result == {"symbol": "AAA", "trail_percent": 5.0, "status": "stop replaced"}
    trade = rig.state.trades["AAA"]
    assert trade.trail_percent == D("5.0") and trade.stop_order_id != old
    assert rig.strategy.get_order(old).is_canceled()  # type: ignore[union-attr]
    new = rig.strategy.get_order(trade.stop_order_id)  # type: ignore[arg-type]
    assert new is not None and new.is_active() and new.trail_percent == D("5.0")


def test_widening_is_refused_and_the_same_trail_is_unchanged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = _open(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert "tighten-only" in rig.desk.set_trailing_stop("AAA", 9.0, "more room")["error"]
    assert "guardrail order_limits" in caplog.text
    assert rig.desk.set_trailing_stop("AAA", 8.0, "same")["status"] == "unchanged"
    assert "between" in rig.desk.set_trailing_stop("AAA", 2.0, "too tight")["error"]
    assert "no open position" in rig.desk.set_trailing_stop("BBB", 5.0, "x")["error"]


def test_a_refused_new_stop_puts_the_old_trail_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _open(tmp_path)
    real_submit = rig.strategy.submit_order

    def refuse_five(order):  # noqa: ANN001, ANN202
        if order.order_type is OrderType.TRAIL and order.trail_percent == D("5.0"):
            raise OrderValidationError("no")
        return real_submit(order)

    monkeypatch.setattr(rig.strategy, "submit_order", refuse_five)
    assert "the 8.0% trail was placed again" in rig.desk.set_trailing_stop("AAA", 5.0, "x")["error"]
    trade = rig.state.trades["AAA"]
    stop = rig.strategy.get_order(trade.stop_order_id)  # type: ignore[arg-type]
    assert trade.trail_percent == D("8.0") and stop is not None and stop.is_active() and stop.trail_percent == D("8.0")


def test_sell_cancels_the_stop_and_sells_at_the_next_open(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    stop_id = rig.state.trades["AAA"].stop_order_id
    result = rig.desk.sell("AAA", "thesis broken")
    assert result["side"] == "sell"
    assert rig.strategy.get_order(stop_id).is_canceled()  # type: ignore[union-attr]
    assert rig.desk.free_slots() == 8  # a pending exit frees its slot
    assert "no open position" in rig.desk.sell("AAA", "again")["error"]  # being sold
    rig.next_close()
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "agent_sell" and D(line["exit_price"]) == D("102.0")


def test_sell_after_the_stop_filled_sells_nothing(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    rig.next_close()
    rig.advance()  # day 3: the stop fills at the broker; its hook has not reached the desk yet
    sells_before = [o for o in rig.strategy.get_orders() if o.side.value == "sell"]
    assert rig.desk.sell("AAA", "x") == {"status": "already_closed"}
    assert rig.desk.set_trailing_stop("AAA", 5.0, "x") == {"status": "already_closed"}
    assert [o for o in rig.strategy.get_orders() if o.side.value == "sell"] == sells_before


def test_a_stop_cancelled_outside_is_placed_again(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = _open(tmp_path)
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    rig.broker.cancel_order(stop)  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        rig.desk.on_order_canceled(stop)  # type: ignore[arg-type]
    assert "guardrail stop_backstop" in caplog.text
    trade = rig.state.trades["AAA"]
    assert trade.stop_order_id != stop.identifier and trade.backstop  # type: ignore[union-attr]
    assert rig.strategy.get_order(trade.stop_order_id).is_active()  # type: ignore[arg-type, union-attr]


def test_the_desk_s_own_cancel_hook_is_ignored(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    rig.desk.set_trailing_stop("AAA", 5.0, "x")
    new_id = rig.state.trades["AAA"].stop_order_id
    rig.desk.on_order_canceled(stop)  # type: ignore[arg-type]  # delivered late, as the executor would
    assert rig.state.trades["AAA"].stop_order_id == new_id


def test_an_unfilled_entry_that_ends_is_dropped_and_a_partial_one_is_protected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)
    rig.desk.buy("AAA", 10, 8.0, "x")
    rig.desk.buy("BBB", 10, 8.0, "y")
    aaa = rig.strategy.get_order(rig.state.trades["AAA"].entry_order_id)
    bbb = rig.strategy.get_order(rig.state.trades["BBB"].entry_order_id)
    protected: list[str] = []
    monkeypatch.setattr(rig.desk, "_protect", lambda trade, trail: protected.append(trade.symbol) or True)
    rig.broker.cancel_order(aaa)  # type: ignore[arg-type]
    rig.desk.on_order_canceled(aaa)  # type: ignore[arg-type]
    bbb.filled_quantity, bbb.avg_fill_price = D(4), D("50.1")  # type: ignore[union-attr]
    rig.broker.cancel_order(bbb)  # type: ignore[arg-type]
    rig.desk.on_order_canceled(bbb)  # type: ignore[arg-type]
    assert "AAA" not in rig.state.trades
    assert rig.state.trades["BBB"].quantity == D(4) and protected == ["BBB"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_desk_exits.py -v`
Expected: FAIL with `AttributeError: 'Desk' object has no attribute 'set_trailing_stop'`

- [ ] **Step 3: Add the methods to `Desk`**

In the "agent tools" section, after `skip`:

```python
    def set_trailing_stop(self, symbol: str, trail_percent: object, reason: str) -> dict[str, Any]:
        with self._lock:
            symbol = str(symbol).strip().upper()
            trade = self._state.trades.get(symbol)
            if trade is None or trade.state is not TradeState.OPEN or trade.exit_order_id is not None:
                return self._refuse("order_limits", f"set_trailing_stop {symbol}: no open position (or it is being sold)")
            trail = _number(trail_percent)
            low, high = self._params.min_trail_percent, self._params.max_trail_percent
            if trail is None or not Decimal(str(low)) <= trail <= Decimal(str(high)):
                return self._refuse("order_limits", f"set_trailing_stop {symbol}: trail_percent must be between {low} and {high} (got {trail_percent!r})")
            if trail > trade.trail_percent:
                return self._refuse("order_limits", f"set_trailing_stop {symbol}: tighten-only, {trail}% is wider than the current {trade.trail_percent}%")
            if trail == trade.trail_percent:
                return {"symbol": symbol, "trail_percent": float(trail), "status": "unchanged"}
            released = self._release_stop(trade)
            if released == ALREADY_CLOSED:
                return {"status": ALREADY_CLOSED}
            if released is not None:
                return {"error": released}
            old = trade.trail_percent
            if self._submit_stop(trade, trail) is None:
                self._protect(trade, old)
                self._save_state()
                return {"error": f"the new stop was refused; the {old}% trail was placed again"}
            trade.trail_percent, trade.backstop = trail, False
            self._log_decision(symbol, "trail", trail_percent=float(trail), reason=self._clip(reason))
            self._save_state()
            return {"symbol": symbol, "trail_percent": float(trail), "status": "stop replaced"}

    def sell(self, symbol: str, reason: str) -> dict[str, Any]:
        with self._lock:
            symbol = str(symbol).strip().upper()
            trade = self._state.trades.get(symbol)
            if trade is None or trade.state is not TradeState.OPEN or trade.exit_order_id is not None:
                return self._refuse("order_limits", f"sell {symbol}: no open position (or it is being sold)")
            result = self._exit(trade, "agent_sell")
            if "error" not in result:
                self._log_decision(symbol, "sell", reason=self._clip(reason))
            return result
```

In the "order hooks" section, after `on_order_filled`:

```python
    def on_order_canceled(self, order: Order) -> None:
        """An order ended unfilled (cancelled or expired): settle an entry, re-protect after a lost stop or exit sell."""
        with self._lock:
            if order.identifier in self._expected_cancels:
                self._expected_cancels.discard(order.identifier)
                return
            trade = self._trade_for(order)
            if trade is None:
                return
            if order.identifier == trade.entry_order_id:
                if trade.state is TradeState.PENDING:
                    if order.filled_quantity > 0:
                        self._entry_filled(trade, order)  # keep what filled, protected
                    else:
                        self._state.trades.pop(trade.symbol, None)
                        self._strategy.log_info(f"[earnings_drift] entry for {trade.symbol} ended {order.status.value} unfilled")
            elif order.identifier == trade.stop_order_id:
                trade.stop_order_id = None
                self._book_partial(trade, order)
                if trade.symbol in self._state.trades:
                    self._strategy.log_warning(f"guardrail stop_backstop: the stop for {trade.symbol} was cancelled outside the strategy; placing it again")
                    trade.backstop = True
                    self._protect(trade, trade.trail_percent)
            elif order.identifier == trade.exit_order_id:
                reason = trade.exit_reason or "agent_sell"
                trade.exit_order_id, trade.exit_reason = None, None
                self._book_partial(trade, order, reason)
                if trade.symbol in self._state.trades:
                    self._strategy.log_warning(f"guardrail stop_backstop: the {reason} sell of {trade.symbol} ended {order.status.value}; placing the stop again")
                    trade.backstop = True
                    self._protect(trade, trade.trail_percent)
            self._save_state()
```

In the internals section:

```python
    def _release_stop(self, trade: Trade) -> str | None:
        """Cancel the trade's working stop and wait for it: None once released, `ALREADY_CLOSED`, or an error message.

        Brokers refuse a sell above held minus pending sells, so the stop must go before any other exit sell. If the
        stop filled while the cancel was on its way (or before; its hook not seen yet), the trade is already out. The
        wait dispatches order hooks; in a backtest the cancel is synchronous, so it returns without moving the clock.
        """
        order = self._lookup(trade.stop_order_id)
        if order is None:
            trade.stop_order_id = None
            return None
        if order.is_filled():
            return ALREADY_CLOSED
        if order.is_active():
            self._expected_cancels.add(order.identifier)
            try:
                self._strategy.cancel_order(order)
            except BrokerError as exc:
                self._expected_cancels.discard(order.identifier)
                if order.is_filled():  # Alpaca raises when cancelling an already-filled order
                    return ALREADY_CLOSED
                return f"could not cancel the stop: {exc}"
            self._strategy.wait_for_order_execution(order, timeout=self._params.cancel_wait_seconds)
            if order.is_filled():
                return ALREADY_CLOSED
            if order.is_active():
                return "the stop cancel was not confirmed in time; nothing else was changed"
        trade.stop_order_id = None
        self._book_partial(trade, order)
        return None if trade.symbol in self._state.trades else ALREADY_CLOSED

    def _exit(self, trade: Trade, exit_reason: str) -> dict[str, Any]:
        """Release the stop, then a market sell of the whole trade at the next open; the stop goes back if the sell is refused."""
        released = self._release_stop(trade)
        if released == ALREADY_CLOSED:
            return {"status": ALREADY_CLOSED}
        if released is not None:
            return {"error": released}
        order = self._market_sell(trade, exit_reason)
        if order is None:
            self._protect(trade, trade.trail_percent)
            self._save_state()
            return {"error": "the sell was refused; the stop was placed again"}
        self._save_state()
        return _lean(order)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/ -v`
Expected: PASS (Tasks 1-10)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/desk.py tests/strategies/earnings_drift/test_drift_desk_exits.py
git commit -m "feat: earnings_drift desk tightens stops, sells on the agent's call, re-protects after lost orders (Task 10)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Desk part 3 — reconcile, max hold, stop backstop, orphans

**Files:**
- Modify: `src/trading_agent_framework/strategies/earnings_drift/desk.py` (add the methods below)
- Test: `tests/strategies/earnings_drift/test_drift_desk_reconcile.py`

**Interfaces:**
- Produces: `reconcile() -> list[str]` (the symbols sold for max hold), `ensure_stops() -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_desk_reconcile.py
from __future__ import annotations

import logging
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import RIG_DATES, DeskRig

from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.strategies.earnings_drift.book import TradeState
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams


def test_max_hold_sells_at_the_next_open(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_holding_sessions=2))
    rig.open_aaa()  # filled day 1
    assert rig.desk.reconcile() == []  # 1 session held
    rig.next_close()  # day 2: 2 sessions held
    with caplog.at_level(logging.WARNING):
        assert rig.desk.reconcile() == ["AAA"]
    assert "guardrail max_hold: AAA held 2 sessions" in caplog.text
    trade = rig.state.trades["AAA"]
    assert trade.exit_reason == "max_hold" and trade.stop_order_id is None
    rig.next_close()  # day 3: the sell fills at the open (104)
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "max_hold" and D(line["exit_price"]) == D("104.0")


def test_ensure_stops_replaces_a_stop_that_vanished_without_a_hook(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    rig.broker.cancel_order(stop)  # type: ignore[arg-type]  # its CANCELED hook never reaches the desk
    with caplog.at_level(logging.WARNING):
        rig.desk.ensure_stops()
    assert "guardrail stop_backstop: AAA has no working stop" in caplog.text
    trade = rig.state.trades["AAA"]
    assert trade.backstop and rig.strategy.get_order(trade.stop_order_id).is_active()  # type: ignore[arg-type, union-attr]
    assert rig.desk.holdings_context()[0]["stop_status"] == "backstop"
    assert rig.desk.holdings_context()[0]["stop_status"] == "working"  # shown once


def test_reconcile_settles_a_fill_whose_hook_was_missed(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.desk.buy("AAA", 10, 8.0, "x")
    rig.advance()  # filled at the broker, hook not delivered
    rig.desk.begin_session(RIG_DATES[1], RIG_DATES[:2], [], {})
    rig.desk.reconcile()
    trade = rig.state.trades["AAA"]
    assert trade.state is TradeState.OPEN and rig.strategy.get_order(trade.stop_order_id).is_active()  # type: ignore[arg-type, union-attr]


def test_reconcile_drops_an_entry_rejected_at_fill_time(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, budget=D("2000"))
    assert "error" not in rig.desk.buy("AAA", 2, 8.0, "x")  # 2 x 100 passes the submission check...
    entry = rig.strategy.get_order(rig.state.trades["AAA"].entry_order_id)
    entry.quantity = D(25)  # type: ignore[union-attr]  # ...then grows past the cash, so the fill is rejected whole
    rig.advance()
    rig.desk.begin_session(RIG_DATES[1], RIG_DATES[:2], [], {})
    rig.desk.reconcile()
    assert rig.state.trades == {}


def test_a_position_never_traded_by_this_strategy_is_left_alone(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.strategy.submit_order(rig.strategy.create_order("BBB", 10, "buy"))  # bought outside the desk
    rig.next_close()
    with caplog.at_level(logging.INFO):
        rig.desk.reconcile()
        rig.desk.reconcile()
    assert "BBB" not in rig.state.trades
    assert caplog.text.count("never traded by this strategy") == 1


def test_an_orphan_of_a_traded_symbol_is_adopted_with_a_stop(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.state.traded_symbols.add("BBB")
    rig.strategy.submit_order(rig.strategy.create_order("BBB", 10, "buy"))
    rig.next_close()
    with caplog.at_level(logging.WARNING):
        rig.desk.reconcile()
    assert "guardrail orphan: adopted 10 BBB" in caplog.text
    trade = rig.state.trades["BBB"]
    assert trade.state is TradeState.OPEN and trade.trail_percent == D("8.0") and trade.opened_on == RIG_DATES[1]
    assert rig.strategy.get_order(trade.stop_order_id).is_active()  # type: ignore[arg-type, union-attr]


def test_an_orphan_with_a_working_sell_adopts_it_as_its_stop(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.state.traded_symbols.add("BBB")
    rig.strategy.submit_order(rig.strategy.create_order("BBB", 10, "buy"))
    rig.next_close()
    working = rig.strategy.submit_order(rig.strategy.create_order("BBB", 10, "sell", trail_percent=6, time_in_force="gtc"))
    rig.desk.reconcile()
    trade = rig.state.trades["BBB"]
    assert trade.stop_order_id == working.identifier and trade.trail_percent == D("6")
    sells = [o for o in rig.strategy.get_orders() if o.side is OrderSide.SELL and o.order_type is OrderType.TRAIL]
    assert sells == [working]


def test_a_position_gone_outside_the_strategy_closes_the_trade_as_unknown(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    rig.broker.cancel_order(stop)  # type: ignore[arg-type]
    rig.strategy.submit_order(rig.strategy.create_order("AAA", 10, "sell"))  # sold by hand
    rig.advance()
    rig.desk.begin_session(RIG_DATES[2], RIG_DATES[:3], [], {})
    rig.desk.reconcile()
    assert rig.state.trades == {}
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "unknown"
```

`test_reconcile_drops_an_entry_rejected_at_fill_time` mutates `entry.quantity` after submission to force the
fill-time cash rejection: `BacktestBroker._rejection_reason` reads `order.quantity` at fill time and rejects a buy above
cash whole (`order.set_error`, error status), a state no other public path reaches in one session.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_desk_reconcile.py -v`
Expected: FAIL with `AttributeError: 'Desk' object has no attribute 'reconcile'`

- [ ] **Step 3: Add the methods to `Desk`**

Add `from trading_agent_framework.entities.enums import OrderSide, OrderType` and
`from trading_agent_framework.entities.position import Position` to the imports, initialise
`self._ignored_orphans: set[str] = set()` in `__init__`, then add a "cycle guardrails" section:

```python
    # --- cycle guardrails ------------------------------------------------------------------------

    def reconcile(self) -> list[str]:
        """Settle what hooks missed, adopt orphans, sell what reached the max holding period, back up the stops."""
        with self._lock:
            self._settle_missed()
            self._close_vanished()
            self._adopt_orphans()
            sold: list[str] = []
            for trade in sorted(self._open_trades(), key=lambda t: t.symbol):
                held = self._sessions_held(trade)
                if held < self._params.max_holding_sessions:
                    continue
                self._strategy.log_warning(f"guardrail max_hold: {trade.symbol} held {held} sessions; selling at the next open")
                if "error" not in self._exit(trade, "max_hold"):
                    sold.append(trade.symbol)
            self.ensure_stops()
            self._save_state()
            return sold

    def ensure_stops(self) -> None:
        """Every open trade not being sold has a working stop: a missing one is placed again (`stop_backstop`)."""
        with self._lock:
            for trade in list(self._open_trades()):
                stop = self._lookup(trade.stop_order_id)
                if stop is not None and stop.is_active():
                    continue
                if stop is not None and stop.is_filled():
                    self._close(trade, stop)
                    continue
                if stop is not None:
                    self._book_partial(trade, stop)
                if trade.symbol not in self._state.trades:
                    continue
                trade.stop_order_id = None
                self._strategy.log_warning(f"guardrail stop_backstop: {trade.symbol} has no working stop; placing a {trade.trail_percent}% trail")
                trade.backstop = True
                self._protect(trade, trade.trail_percent)
            self._save_state()

    def _settle_missed(self) -> None:
        """Apply fills and ends whose hooks never reached the desk (a restart, a dropped event)."""
        for trade in list(self._state.trades.values()):
            if trade.state is TradeState.PENDING:
                order = self._lookup(trade.entry_order_id)
                if order is None:
                    self._state.trades.pop(trade.symbol, None)
                    self._strategy.log_warning(f"[earnings_drift] entry {trade.entry_order_id} for {trade.symbol} is unknown to the broker; trade dropped")
                elif order.is_filled() or (not order.is_active() and order.filled_quantity > 0):
                    self._entry_filled(trade, order)
                elif not order.is_active():
                    self._state.trades.pop(trade.symbol, None)
                    detail = f": {order.error_message}" if order.error_message else ""
                    self._strategy.log_warning(f"[earnings_drift] entry for {trade.symbol} ended {order.status.value} unfilled{detail}")
                continue
            for attribute in ("stop_order_id", "exit_order_id"):
                if trade.symbol not in self._state.trades:
                    break
                order_id = getattr(trade, attribute)
                if order_id is None:
                    continue
                order = self._lookup(order_id)
                if order is not None and order.is_filled():
                    self._close(trade, order)
                elif order is None or not order.is_active():
                    reason = (trade.exit_reason or "agent_sell") if attribute == "exit_order_id" else "trail"
                    setattr(trade, attribute, None)
                    if attribute == "exit_order_id":
                        trade.exit_reason = None
                    if order is not None:
                        self._book_partial(trade, order, reason)

    def _positions(self) -> dict[str, Position] | None:
        try:
            return {p.asset.symbol: p for p in self._strategy.get_positions() if p.quantity > 0}
        except _DATA_ERRORS as exc:
            self._strategy.log_warning(f"[earnings_drift] positions unavailable, orphans and vanished positions not checked: {exc}")
            return None

    def _close_vanished(self) -> None:
        """An open trade whose position is gone (sold outside the strategy) is closed with reason `unknown`."""
        positions = self._positions()
        if positions is None:
            return
        for trade in list(self._open_trades()):
            if trade.symbol in positions:
                continue
            stop = self._lookup(trade.stop_order_id)
            if stop is not None and stop.is_active():
                self._expected_cancels.add(stop.identifier)
                try:
                    self._strategy.cancel_order(stop)
                except BrokerError as exc:
                    self._expected_cancels.discard(stop.identifier)
                    self._strategy.log_warning(f"[earnings_drift] cancel of the stop of vanished {trade.symbol} failed: {exc}")
            self._strategy.log_warning(f"[earnings_drift] the {trade.symbol} position is gone outside the strategy; trade closed as unknown")
            record = {"symbol": trade.symbol, "entry_date": trade.opened_on.isoformat() if trade.opened_on else None, "entry_price": trade.entry_price}
            self._trade_log.append({**record, "exit_date": self._now_date().isoformat(), "exit_price": None, "quantity": trade.quantity, "exit_reason": "unknown", "thesis": trade.thesis, "agent_enabled": self._params.agent_enabled})
            self._state.trades.pop(trade.symbol, None)

    def _adopt_orphans(self) -> None:
        """A long position with no trade: adopted when this strategy once traded the symbol, left alone (logged once) otherwise."""
        positions = self._positions()
        if positions is None:
            return
        active = [o for o in self._strategy.broker.tracker.get_active_orders() if o.side is OrderSide.SELL]
        for symbol, position in sorted(positions.items()):
            quantity = position.quantity
            if symbol in self._state.trades:
                continue
            if symbol not in self._state.traded_symbols:
                if symbol not in self._ignored_orphans:
                    self._ignored_orphans.add(symbol)
                    self._strategy.log_info(f"[earnings_drift] position {symbol} was never traded by this strategy; left alone")
                continue
            working = next((o for o in active if o.asset.symbol == symbol and o.order_type is OrderType.TRAIL), None)
            trail = working.trail_percent if working is not None and working.trail_percent is not None else Decimal(str(self._params.default_trail_percent))
            self._state.trades[symbol] = Trade(
                symbol=symbol,
                entry_order_id="",
                trail_percent=trail,
                thesis="adopted: a position without a trade record",
                state=TradeState.OPEN,
                quantity=quantity,
                entry_price=position.avg_fill_price,
                opened_on=self._now_date(),
                stop_order_id=working.identifier if working is not None else None,
                backstop=working is None,
            )
            self._strategy.log_warning(f"guardrail orphan: adopted {quantity} {symbol} with a {trail}% trailing stop")
```

`ensure_stops()` (called at the end of `reconcile`) places the stop of an orphan adopted without a working sell.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/ -v`
Expected: PASS (Tasks 1-11)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/desk.py tests/strategies/earnings_drift/test_drift_desk_reconcile.py
git commit -m "feat: earnings_drift desk reconciles, enforces max hold and the stop backstop, adopts orphans (Task 11)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Agent tools and prompt (`tools.py`, `prompts.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/tools.py`
- Create: `src/trading_agent_framework/strategies/earnings_drift/prompts.py`
- Test: `tests/strategies/earnings_drift/test_drift_tools.py`

**Interfaces:**
- Consumes: `Desk` (Tasks 9-11); `news_tools`, `fundamentals_tools`, `market_data_tools` (existing).
- Produces: `ORDER_TOOLS = ("buy", "set_trailing_stop", "sell", "skip")`, `desk_tools(desk) -> list[Callable[..., dict]]`,
  `research_tools(strategy) -> list[Callable[..., dict]]`; `DRIFT_SYSTEM: str`, `TASK_PROMPT: str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/earnings_drift/test_drift_tools.py
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import DeskRig

from trading_agent_framework.strategies.earnings_drift.prompts import DRIFT_SYSTEM, TASK_PROMPT
from trading_agent_framework.strategies.earnings_drift.tools import ORDER_TOOLS, desk_tools, research_tools


def test_desk_tools_are_the_order_tools_with_one_line_docstrings(tmp_path: Path) -> None:
    tools = desk_tools(DeskRig(tmp_path).desk)
    assert tuple(tool.__name__ for tool in tools) == ORDER_TOOLS
    for tool in tools:
        assert tool.__doc__ and "\n" not in tool.__doc__.strip()
        assert all(p.annotation is not inspect.Parameter.empty for p in inspect.signature(tool).parameters.values())


def test_desk_tools_reach_the_desk(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    tools = {tool.__name__: tool for tool in desk_tools(rig.desk)}
    assert tools["buy"]("AAA", 10, 8.0, "beat")["symbol"] == "AAA"
    assert tools["skip"]("BBB", "guidance cut") == {"symbol": "BBB", "status": "skipped"}
    assert "no open position" in tools["sell"]("AAA", "x")["error"]  # pending, not open yet
    assert "no open position" in tools["set_trailing_stop"]("AAA", 5.0, "x")["error"]


def test_research_tools_are_news_two_filing_tools_and_market_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")
    names = [tool.__name__ for tool in research_tools(DeskRig(tmp_path).strategy)]
    assert names == ["search_news", "get_filings", "get_filing_document", "get_last_price", "get_quote", "get_bars"]


def test_the_prompt_never_states_the_holding_limit_and_names_every_order_tool() -> None:
    for forbidden in ("10 sessions", "ten sessions", "max_holding", "holding period", "10 days"):
        assert forbidden not in DRIFT_SYSTEM.lower()
    for tool in ORDER_TOOLS:
        assert f"{tool}(" in DRIFT_SYSTEM
    assert "do not recompute" in DRIFT_SYSTEM.lower()
    assert TASK_PROMPT.strip()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_tools.py -v`
Expected: FAIL with `ModuleNotFoundError` (`tools`)

- [ ] **Step 3: Implement `tools.py` and `prompts.py`**

```python
# src/trading_agent_framework/strategies/earnings_drift/tools.py
"""The agent's tools (spec §5.1): the desk's order tools, `skip`, and the clock-gated research tools.

Each docstring is a single line on purpose: it is the tool description sent to the model on every call.
No `from __future__ import annotations`: the agent layer reads the real annotations.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.strategies.earnings_drift.desk import Desk

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

ORDER_TOOLS = ("buy", "set_trailing_stop", "sell", "skip")  # exempt from the per-run tool budget
_FILING_TOOLS = ("get_filings", "get_filing_document")


def desk_tools(desk: Desk) -> list[Callable[..., dict[str, Any]]]:
    def buy(symbol: str, quantity: int, trail_percent: float, reason: str) -> dict[str, Any]:
        """Buy a candidate at the next open; a trailing stop of trail_percent (3-15) is placed when it fills."""
        return desk.buy(symbol, quantity, trail_percent, reason)

    def set_trailing_stop(symbol: str, trail_percent: float, reason: str) -> dict[str, Any]:
        """Tighten a holding's trailing stop to trail_percent (never wider than the current one)."""
        return desk.set_trailing_stop(symbol, trail_percent, reason)

    def sell(symbol: str, reason: str) -> dict[str, Any]:
        """Sell a whole holding at the next open."""
        return desk.sell(symbol, reason)

    def skip(symbol: str, reason: str) -> dict[str, Any]:
        """Pass on a candidate, with the reason."""
        return desk.skip(symbol, reason)

    return [buy, set_trailing_stop, sell, skip]


def research_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """News search, the two SEC filing tools (the 8-K press release) and market data, all gated on the strategy clock."""
    from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
    from trading_agent_framework.agents.tools.market_data import market_data_tools
    from trading_agent_framework.agents.tools.news import news_tools

    filings = [tool for tool in fundamentals_tools(strategy) if tool.__name__ in _FILING_TOOLS]
    return [*news_tools(strategy), *filings, *market_data_tools(strategy)]
```

```python
# src/trading_agent_framework/strategies/earnings_drift/prompts.py
"""The earnings_drift agent's prompts (spec §5.3). The holding limit is code's (`max_hold`) and is never stated here."""

DRIFT_SYSTEM = """You manage a small long-only book of US stocks that have just reported earnings. You run once a day,
right after the close. Every order you place fills at the next open.

The context gives you:
- candidates: stocks whose earnings reaction today passed the code's gates (an EPS beat, a strong abnormal return
  that held into the close, high volume, enough liquidity). Every number in a fact sheet is computed by code:
  eps.result and sales.result say BEAT, MISS or IN-LINE. Do not recompute them.
- holdings: the stocks you hold, with the thesis you gave when buying, the sessions held, the P&L and the trailing
  stop (stop_status "backstop" means code had to place it again).
- balances.

For each candidate, decide:
- buy(symbol, quantity, trail_percent, reason) when the surprise looks real and the move can continue: the EPS beat
  comes with a sales beat, and guidance was held or raised. When the headlines do not say, read the 8-K press release
  with get_filing_document (its accession_number is in the fact sheet). quantity is at most max_quantity; buy less
  when your conviction is lower. trail_percent is between 3 and 15: about 2 to 3 times context.atr14_pct, wider for a
  volatile stock. Your reason is the thesis you will see on the following days.
- skip(symbol, reason) otherwise: a beat made of one-off items (tax, buybacks, asset sales), a guidance cut, missing
  sales, or a stock that had already run up a lot before the report.
Every candidate gets buy or skip.

For each holding:
- sell(symbol, reason) when the thesis is broken, for example a close below reaction_low, or news that undoes the
  surprise.
- set_trailing_stop(symbol, trail_percent, reason) to tighten the stop as gains build. A stop can only be tightened.
- Otherwise do nothing: holding needs no tool call.

Use search_news, get_filings, get_filing_document and get_bars only when they help a decision. A tool that returns
{"error": ...} did nothing: read the error and correct the call."""

TASK_PROMPT = "Decide on every candidate (buy or skip) and review every holding, using the context."
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift/tools.py src/trading_agent_framework/strategies/earnings_drift/prompts.py tests/strategies/earnings_drift/test_drift_tools.py
git commit -m "feat: earnings_drift agent tools and prompt (Task 12)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: The strategy, its registration and end-to-end backtests

**Files:**
- Create: `src/trading_agent_framework/strategies/earnings_drift/agent_earnings_drift.py`
- Modify: `src/trading_agent_framework/strategies/earnings_drift/__init__.py` (export the class)
- Modify: `src/trading_agent_framework/main.py` (two builders, two registry entries)
- Modify: `tests/test_main.py` (registry set, builder tests)
- Test: `tests/strategies/earnings_drift/test_drift_strategy.py`, `tests/strategies/earnings_drift/test_drift_backtest.py`

**Interfaces:**
- Consumes: everything above; `Strategy` hooks; `AgentManager.create(name=, system_prompt=, tools=, temperature=, exempt_tools=)`,
  `agents[name].run(task, context=, tool_budget=)`.
- Produces: `EarningsDriftStrategy(broker, *, mode=PAPER, universe, settings: DriftParams | None = None, event_source: EventProvider | None = None, **kwargs)`
  with `AGENT_NAME = "drift"`, `run_cycle() -> None`, `run_backtesting(**overrides)`.

- [ ] **Step 1: Write the failing strategy tests**

```python
# tests/strategies/earnings_drift/test_drift_strategy.py
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et, make_session
from tests.strategies.earnings_drift.drift_helpers import make_candidate
from tests.strategies.earnings_drift.test_drift_scanner import StaticEvents

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.scanner import ScanResult
from trading_agent_framework.strategies.earnings_drift.tools import ORDER_TOOLS
from trading_agent_framework.utils.errors import AgentError, FatalStrategyError

DAY = date(2026, 9, 1)


class _Handle:
    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.runs = error, []

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs.append((context, tool_budget))
        if self.error is not None:
            raise self.error
        return AgentRunResult(output="ok", tool_calls=[])


class _Manager:
    def __init__(self, handle: _Handle) -> None:
        self.handle, self.created = handle, []

    def create(self, **kwargs: Any) -> _Handle:
        self.created.append(kwargs)
        return self.handle

    def __getitem__(self, name: str) -> _Handle:
        return self.handle

    def telemetry_summary(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _strategy(tmp_path: Path, *, mode: TradingMode = TradingMode.BACKTESTING, settings: DriftParams | None = None, handle: _Handle | None = None) -> tuple[EarningsDriftStrategy, _Manager]:
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 16, 0), [make_session(DAY)]), "earnings_drift")
    strategy = EarningsDriftStrategy(broker, mode=mode, universe=["AAA"], project_root=tmp_path, settings=settings or DriftParams(live_bar_delay_seconds=0), event_source=StaticEvents([]))
    manager = _Manager(handle or _Handle())
    strategy._agents = cast(AgentManager, manager)
    strategy.initialize()
    return strategy, manager


def _stub_cycle(strategy: EarningsDriftStrategy, *, candidates: int = 1, holdings: int = 0, hollow: bool = False) -> list[str]:
    steps: list[str] = []
    assert strategy.scanner is not None and strategy.desk is not None
    desk = strategy.desk
    strategy.scanner.prepare = lambda held: steps.append("prepare") or ScanResult(DAY, [DAY], [make_candidate()] * candidates, {}, hollow=hollow)  # type: ignore[method-assign]
    desk.exposed_symbols = lambda: set()  # type: ignore[method-assign]
    desk.begin_session = lambda *args: steps.append("begin")  # type: ignore[method-assign]
    desk.reconcile = lambda: steps.append("reconcile") or []  # type: ignore[method-assign]
    desk.candidate_sheets = lambda: [{"symbol": "AAA"}] * candidates  # type: ignore[method-assign]
    desk.holdings_context = lambda: [{"symbol": "HHH"}] * holdings  # type: ignore[method-assign]
    desk.balances = lambda: {}  # type: ignore[method-assign]
    desk.baseline_entries = lambda: steps.append("baseline") or []  # type: ignore[method-assign]
    desk.ensure_stops = lambda: steps.append("ensure")  # type: ignore[method-assign]
    desk.record_undecided = lambda: steps.append("undecided")  # type: ignore[method-assign]
    desk.save = lambda: steps.append("save")  # type: ignore[method-assign]
    return steps


def test_initialize_creates_the_agent_with_order_and_research_tools(tmp_path: Path) -> None:
    _, manager = _strategy(tmp_path)
    [created] = manager.created
    names = [tool.__name__ for tool in created["tools"]]
    assert created["name"] == "drift" and names[:4] == list(ORDER_TOOLS) and "search_news" in names and "get_filing_document" in names
    assert created["exempt_tools"] == list(ORDER_TOOLS) and created["temperature"] == 0.3


def test_baseline_mode_creates_no_agent_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _, manager = _strategy(tmp_path, settings=DriftParams(agent_enabled=False))
    assert manager.created == [] and "guardrail baseline" in caplog.text


def test_a_missing_sec_identity_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 16, 0), [make_session(DAY)]), "earnings_drift")
    strategy = EarningsDriftStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA"], project_root=tmp_path)
    strategy._agents = cast(AgentManager, _Manager(_Handle()))
    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_the_cycle_runs_in_order_after_the_close(tmp_path: Path) -> None:
    strategy, manager = _strategy(tmp_path)
    steps = _stub_cycle(strategy, candidates=2, holdings=1)
    strategy.after_market_closes()
    assert steps == ["prepare", "begin", "reconcile", "ensure", "undecided", "save"]
    [(context, budget)] = manager.handle.runs
    assert set(context) == {"date", "candidates", "holdings", "balances"} and budget == 4 * 3


def test_no_candidates_and_no_holdings_means_no_agent_call(tmp_path: Path) -> None:
    strategy, manager = _strategy(tmp_path)
    _stub_cycle(strategy, candidates=0)
    strategy.after_market_closes()
    assert manager.handle.runs == []


def test_baseline_mode_buys_instead_of_asking_the_agent(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, settings=DriftParams(agent_enabled=False))
    steps = _stub_cycle(strategy)
    strategy.after_market_closes()
    assert "baseline" in steps


def test_three_agent_failures_end_a_backtest_the_next_morning(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, handle=_Handle(AgentError("model down")))
    _stub_cycle(strategy)
    for _ in range(2):
        strategy.after_market_closes()
    strategy.on_trading_iteration()  # two failures: carry on
    strategy.after_market_closes()
    with pytest.raises(FatalStrategyError, match="3 agent runs failed in a row"):
        strategy.on_trading_iteration()


def test_agent_failures_never_end_a_paper_run(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, mode=TradingMode.PAPER, handle=_Handle(AgentError("model down")))
    _stub_cycle(strategy)
    for _ in range(4):
        strategy.after_market_closes()
    strategy.on_trading_iteration()


def test_three_hollow_scans_end_a_backtest_the_next_morning(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    _stub_cycle(strategy, hollow=True)
    for _ in range(3):
        strategy.after_market_closes()
    with pytest.raises(FatalStrategyError, match="3 hollow scans in a row"):
        strategy.on_trading_iteration()


def test_order_hooks_reach_the_desk(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    assert strategy.desk is not None
    seen: list[str] = []
    strategy.desk.on_order_filled = lambda order: seen.append("filled")  # type: ignore[method-assign]
    strategy.desk.on_order_canceled = lambda order: seen.append("canceled")  # type: ignore[method-assign]
    order = object()
    strategy.on_filled_order(None, order, None, None, 1)  # type: ignore[arg-type]
    strategy.on_canceled_order(order)  # type: ignore[arg-type]
    assert seen == ["filled", "canceled"]
```

- [ ] **Step 2: Write the failing end-to-end backtest tests**

```python
# tests/strategies/earnings_drift/test_drift_backtest.py
"""End to end: a short daily backtest with a scripted agent (it calls the real desk tools), then the baseline."""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et, weekday_sessions
from tests.strategies.earnings_drift.test_drift_scanner import StaticEvents

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams

ALL = weekday_sessions(date(2026, 7, 20), 36)
HISTORY, WINDOW = ALL[:30], ALL[30:]
FLAT = (100.0, 100.5, 99.5, 100.0, 1_000_000.0)
AAA = [FLAT] * 30 + [
    (106.0, 110.0, 105.0, 109.0, 5_000_000.0),  # W0: the reaction
    (109.5, 111.0, 108.0, 110.0, 2_000_000.0),  # W1: the entry fills at 109.5
    (110.0, 112.0, 109.0, 111.0, 1_500_000.0),  # W2: the trail rises to 112
    (105.0, 106.0, 100.0, 101.0, 3_000_000.0),  # W3: 112 x 0.92 = 103.04 is crossed
    (101.0, 102.0, 100.0, 101.0, 1_000_000.0),
    (101.0, 102.0, 100.0, 101.0, 1_000_000.0),
]
SPY = [(400.0, 401.0, 399.0, 400.0, 5e7)] * 30 + [(400.0, 402.0, 399.0, 400.8, 5e7)] + [(400.8, 401.5, 400.0, 400.8, 5e7)] * 5
BBB = [(50.0, 50.5, 49.5, 50.0, 1_000_000.0)] * 36
RELEASE = et(WINDOW[0].open.year, WINDOW[0].open.month, WINDOW[0].open.day, 7, 0)


def _frame(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([s.close for s in ALL], name="timestamp"))


class _Handle:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.runs = 0

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs += 1
        for sheet in context["candidates"]:
            self.tools["buy"](sheet["symbol"], sheet["max_quantity"], 8.0, "real beat, held the gap")
        return AgentRunResult(output="ok", tool_calls=[])


class _Manager:
    def __init__(self) -> None:
        self.handle = _Handle()

    def create(self, *, name: str, tools: list[Any], **_: Any) -> _Handle:
        self.handle.tools = {tool.__name__: tool for tool in tools}
        return self.handle

    def __getitem__(self, name: str) -> _Handle:
        return self.handle

    def telemetry_summary(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _run(tmp_path: Path, settings: DriftParams) -> tuple[EarningsDriftStrategy, _Manager]:
    source = FakeBacktestDataSource()
    source.set_sessions(WINDOW)
    for symbol, rows in {"AAA": AAA, "BBB": BBB, "SPY": SPY}.items():
        source.set_bars(Asset(symbol), _frame(rows))
    news = FakeNewsProvider({"AAA": [{"headline": "AAA Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate", "created_at": (RELEASE + timedelta(minutes=1)).isoformat(), "symbols": ["AAA"]}]})
    events = StaticEvents([EarningsEvent("AAA", RELEASE, "acc-AAA", "aaa.htm")])
    strategy = EarningsDriftStrategy(
        FakeBroker(FakeClock(WINDOW[0].open - timedelta(hours=1)), strategy_name="earnings_drift"),
        mode=TradingMode.BACKTESTING,
        universe=["AAA", "BBB"],
        project_root=tmp_path,
        settings=settings,
        event_source=events,
    )
    manager = _Manager()
    strategy._agents = cast(AgentManager, manager)
    Strategy.run_backtesting(strategy, start=WINDOW[0].open - timedelta(hours=1), end=WINDOW[-1].close, data_source=source, news_source=news, budget=Decimal(10000), benchmark="SPY")
    return strategy, manager


def _lines(tmp_path: Path, name: str) -> list[dict[str, Any]]:
    (path,) = tmp_path.rglob(name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_agent_buys_after_the_close_and_the_trail_takes_it_out(tmp_path: Path) -> None:
    _, manager = _run(tmp_path, DriftParams())
    [trade] = _lines(tmp_path, "trades.jsonl")
    assert trade["symbol"] == "AAA" and trade["exit_reason"] == "trail" and trade["quantity"] == "11"
    assert Decimal(trade["entry_price"]) == Decimal("109.5") and Decimal(trade["exit_price"]) == Decimal("103.04")
    assert trade["entry_date"] == WINDOW[1].open.date().isoformat() and trade["sessions_held"] == 3
    assert manager.handle.runs == 3  # W0 (the candidate), W1 and W2 (the holding); nothing to review after W3
    assert [(d["symbol"], d["decision"]) for d in _lines(tmp_path, "decisions.jsonl")] == [("AAA", "buy")]


def test_the_baseline_takes_the_same_trade_without_an_agent(tmp_path: Path) -> None:
    _, manager = _run(tmp_path, DriftParams(agent_enabled=False))
    [trade] = _lines(tmp_path, "trades.jsonl")
    assert trade["exit_reason"] == "trail" and trade["thesis"] == "baseline" and trade["agent_enabled"] is False
    assert manager.handle.runs == 0
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/strategies/earnings_drift/test_drift_strategy.py tests/strategies/earnings_drift/test_drift_backtest.py -v`
Expected: FAIL with `ImportError: cannot import name 'EarningsDriftStrategy'`

- [ ] **Step 4: Implement the strategy**

```python
# src/trading_agent_framework/strategies/earnings_drift/agent_earnings_drift.py
"""EarningsDriftStrategy: post-earnings announcement drift, decided by an agent that trades through a guarded desk.

Once per session, right after the close (`after_market_closes`): `Scanner.prepare` builds today's candidates
(SEC 8-K item 2.02, Benzinga surprise, reaction-day gates), `Desk.reconcile` applies the guardrails, then the agent
decides (or, in baseline mode, code buys every candidate). Orders fill at the next open. `on_trading_iteration`
only raises a fatal error recorded by the previous cycle: the executor swallows exceptions from every other hook.
See docs/superpowers/specs/2026-10-05-earnings-drift-design.md.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.earnings_drift.book import DriftState, JsonlLog, StateStore, state_path
from trading_agent_framework.strategies.earnings_drift.desk import Desk
from trading_agent_framework.strategies.earnings_drift.event_source import EventProvider, EventSource
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.prompts import DRIFT_SYSTEM, TASK_PROMPT
from trading_agent_framework.strategies.earnings_drift.scanner import Scanner, ScanResult
from trading_agent_framework.strategies.earnings_drift.tools import ORDER_TOOLS, desk_tools, research_tools
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError, ConfigurationError, FatalStrategyError

_DATA_ERRORS = (BrokerError, BacktestError)


class EarningsDriftStrategy(Strategy):
    sleeptime = "1D"  # on_trading_iteration only raises a pending fatal error; the work is in after_market_closes
    minutes_after_closing = 0  # at the exact close: a backtest order then fills at the next open, not a session later
    AGENT_NAME = "drift"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.YEAR)[0],
        "backtesting_end": backtest_window(PredefinedWindow.YEAR)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 283,  # the regime minimum; also covers the 60-session run-up and the 20-session baselines
        "budget": 10000,
        "slippage": Decimal("0.0005"),
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: DriftParams | None = None,
        event_source: EventProvider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.universe = list(universe)
        self.settings = settings or DriftParams()
        self._event_source = event_source  # injected in tests; the SEC source is built in `initialize`
        self.desk: Desk | None = None
        self.scanner: Scanner | None = None
        self._state = DriftState()  # replaced by the loaded state in `initialize`
        self._pending_fatal: str | None = None  # raised by the next on_trading_iteration (backtests only)

    # --- lifecycle --------------------------------------------------------------------------------

    def initialize(self) -> None:
        store = StateStore(state_path(self.project_root, self.trading_mode))
        if self.is_backtesting:
            store.wipe()  # one run's trades must not leak into the next; paper and live state is never wiped
        self._state = store.load()
        state = self._state
        try:
            source = self._event_source or EventSource(_sec_client(self.project_root), reload_every_cycle=not self.is_backtesting)
            self.desk = Desk(
                self,
                self.settings,
                state,
                save=lambda: store.save(state),
                trade_log=JsonlLog(lambda: self._run_file("trades.jsonl")),
                decision_log=JsonlLog(lambda: self._run_file("decisions.jsonl")),
            )
            if self.settings.agent_enabled:
                self.agents.create(
                    name=self.AGENT_NAME,
                    system_prompt=DRIFT_SYSTEM,
                    tools=[*desk_tools(self.desk), *research_tools(self)],
                    temperature=self.settings.agent_temperature,
                    exempt_tools=list(ORDER_TOOLS),
                )
            else:
                self.log_warning("guardrail baseline: agent disabled, baseline mode (every gated candidate is bought at the default trail)")
        except ConfigurationError as exc:
            raise FatalStrategyError(str(exc)) from exc  # refuse to start rather than fail every cycle
        self.scanner = Scanner(self, self.settings, self.universe, source, benchmark=self.parameters["benchmark_symbol"])
        self.log_info(f"EarningsDriftStrategy initialized: {len(self.universe)} symbols, agent {'on' if self.settings.agent_enabled else 'off (baseline)'}")

    def on_trading_iteration(self) -> None:
        if self._pending_fatal is not None:
            raise FatalStrategyError(self._pending_fatal)

    def after_market_closes(self) -> None:
        if not self.is_backtesting and self.settings.live_bar_delay_seconds > 0:
            self.sleep(self.settings.live_bar_delay_seconds)  # let the day's bar become final
        self.run_cycle()

    def on_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        if self.desk is not None:
            self.desk.on_order_filled(order)

    def on_canceled_order(self, order: Order) -> None:
        if self.desk is not None:
            self.desk.on_order_canceled(order)

    # --- the daily cycle --------------------------------------------------------------------------

    def run_cycle(self) -> None:
        assert self.desk is not None and self.scanner is not None, "initialize() has not run"
        desk, settings, state = self.desk, self.settings, self._state
        try:
            scan = self.scanner.prepare(held=desk.exposed_symbols())
        except _DATA_ERRORS as exc:
            self.log_error(f"[earnings_drift] scan failed, only the stops are checked this cycle: {exc}")
            desk.ensure_stops()
            desk.save()
            return
        if scan.hollow:
            state.hollow_scan_streak += 1
            if self.is_backtesting and state.hollow_scan_streak >= settings.max_consecutive_hollow_scans:
                self._pending_fatal = f"{state.hollow_scan_streak} hollow scans in a row (SEC unreachable?), aborting the backtest"
        else:
            state.hollow_scan_streak = 0
        desk.begin_session(scan.today, scan.trading_dates, scan.candidates, scan.rejections)
        desk.reconcile()
        if settings.agent_enabled:
            self._run_agent(scan)
        else:
            desk.baseline_entries()
        desk.ensure_stops()
        desk.record_undecided()
        desk.save()

    def _run_agent(self, scan: ScanResult) -> None:
        assert self.desk is not None
        desk, state = self.desk, self._state
        candidates, holdings = desk.candidate_sheets(), desk.holdings_context()
        if not candidates and not holdings:
            return
        context = {"date": scan.today.isoformat(), "candidates": candidates, "holdings": holdings, "balances": desk.balances()}
        budget = self.settings.tool_budget_per_item * (len(candidates) + len(holdings))
        try:
            result = self.agents[self.AGENT_NAME].run(TASK_PROMPT, context=context, tool_budget=budget)
        except AgentError as exc:
            state.agent_failure_streak += 1
            self.log_error(f"[earnings_drift] agent run failed ({state.agent_failure_streak} in a row); today's candidates are dropped: {exc}")
            if self.is_backtesting and state.agent_failure_streak >= self.settings.max_consecutive_agent_failures:
                self._pending_fatal = f"{state.agent_failure_streak} agent runs failed in a row, aborting the backtest; last error: {exc}"
            return
        state.agent_failure_streak = 0
        self.log_info(f"[earnings_drift] agent: {len(candidates)} candidates, {len(holdings)} holdings, {len(result.tool_calls)} tool calls")

    def _run_file(self, name: str) -> Path | None:
        """`<name>` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / name

    # --- backtesting --------------------------------------------------------------------------------

    def run_backtesting(self, **overrides: Any):
        """Backtest over the `parameters` window on Alpaca SIP daily bars, the universe preloaded."""
        symbols = list(dict.fromkeys([*self.universe, self.parameters["benchmark_symbol"]]))
        defaults: dict[str, Any] = dict(
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            budget=Decimal(str(self.parameters["budget"])),
            data_source=AlpacaBacktestData,
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],
            benchmark=self.parameters["benchmark_symbol"],
            timestep="day",
            slippage=self.parameters["slippage"],
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=self.settings.agent_enabled,
        )
        return super().run_backtesting(**{**defaults, **overrides})


def _sec_client(project_root: Path) -> SecEdgarClient:
    """The SEC client for the event source; `ConfigurationError` without `SEC_EDGAR_USER_AGENT`."""
    return SecEdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT", ""), project_root / "cache" / "sec")
```

```python
# src/trading_agent_framework/strategies/earnings_drift/__init__.py
"""earnings_drift: post-earnings announcement drift with an agent that trades (spec 2026-10-05)."""

from trading_agent_framework.strategies.earnings_drift.agent_earnings_drift import EarningsDriftStrategy

__all__ = ["EarningsDriftStrategy"]
```

- [ ] **Step 5: Register both names in `main.py` and update `tests/test_main.py`**

In `src/trading_agent_framework/main.py`, add the imports and the builders next to `_build_bill_ackman`:

```python
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams


def _build_earnings_drift(broker: Broker, mode: TradingMode) -> Strategy | None:
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run `uv run batch-universe` before executing this strategy.", style="bold red")
        return None
    return EarningsDriftStrategy(broker=broker, mode=mode, universe=universe)


def _build_earnings_drift_baseline(broker: Broker, mode: TradingMode) -> Strategy | None:
    """The code-only baseline over 5 years: every gated candidate, the default trail, no LLM."""
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run `uv run batch-universe` before executing this strategy.", style="bold red")
        return None
    start, end = backtest_window(PredefinedWindow.SEMI_DECADE)
    return EarningsDriftStrategy(
        broker=broker,
        mode=mode,
        universe=universe,
        settings=DriftParams(agent_enabled=False),
        parameters={"backtesting_start": start, "backtesting_end": end},
    )
```

and in `AGENT_STRATEGIES`:

```python
    "earnings_drift": _build_earnings_drift,
    "earnings_drift_baseline": _build_earnings_drift_baseline,
```

In `tests/test_main.py`, update the registry test and add:

```python
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy


def test_registry_lists_the_strategies() -> None:
    assert set(main_module.AGENT_STRATEGIES) == {
        "bill_ackman",
        "cross_momentum",
        "earnings_drift",
        "earnings_drift_baseline",
        "news_binary",
        "vwap_pullback_continuation",
    }


def test_earnings_drift_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: ["AAA"])
    agent = main_module._build_earnings_drift(FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="earnings_drift"), TradingMode.BACKTESTING)
    baseline = main_module._build_earnings_drift_baseline(FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="earnings_drift_baseline"), TradingMode.BACKTESTING)
    assert isinstance(agent, EarningsDriftStrategy) and agent.settings.agent_enabled
    assert isinstance(baseline, EarningsDriftStrategy) and not baseline.settings.agent_enabled
    assert baseline.parameters["backtesting_start"] < agent.parameters["backtesting_start"]  # 5 years against 1


def test_earnings_drift_builder_returns_none_without_a_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: [])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="earnings_drift")
    assert main_module._build_earnings_drift(broker, TradingMode.BACKTESTING) is None
```

(Replace the existing `test_registry_lists_the_strategies`; do not keep two.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/earnings_drift/ tests/test_main.py -v`
Expected: PASS. If `test_the_agent_buys_after_the_close_and_the_trail_takes_it_out` finds no trade, log the
executor's hook times: the cycle must run at each session's exact close (`minutes_after_closing = 0`), and the entry
must fill on the next session's bar. If `manager.handle.runs` is 4, the W3 cycle saw the holding before the stop's
fill hook: check that the stop fill is dispatched during the wait to W3's close, before `after_market_closes`.

- [ ] **Step 7: Run the whole suite and the linter**

Run: `uv run pytest -q && uv run ruff check`
Expected: all tests pass; ruff reports no error.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/strategies/earnings_drift src/trading_agent_framework/main.py tests/strategies/earnings_drift tests/test_main.py
git commit -m "feat: EarningsDriftStrategy runs its cycle after the close; earnings_drift and earnings_drift_baseline registered (Task 13)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Smoke script and docs

**Files:**
- Create: `scripts/tests/smoke_earnings_events.py`
- Modify: `CLAUDE.md` (architecture entry, a gotcha bullet, the smoke command)
- Modify: `README.md` (strategy entry after `vwap_pullback_continuation`)
- Modify: `docs/superpowers/specs/2026-10-05-earnings-drift-design.md` §10 (the `earnings_drift_baseline` name)

**Interfaces:**
- Consumes: `EventSource`, `events_reacting_on`, `reaction_date`, `articles_for`, `pick_surprise`, `AlpacaNewsProvider.from_credentials`,
  `AlpacaCredentials.for_news`.

- [ ] **Step 1: Write the smoke script**

```python
#!/usr/bin/env python3
"""Manual check of earnings_drift's event detection against real SEC EDGAR and Alpaca news.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_earnings_events.py

Credentials come from env/.env.alpaca.integration-tests (ALPACA_NEWS_*, SEC_EDGAR_USER_AGENT), as in the other smoke
scripts; without SEC_EDGAR_USER_AGENT the script prints `SKIP:` and exits 0. Read-only. For a few large caps over the
last full quarter it prints each 8-K item 2.02 event, its reaction date and the parsed surprise, then the share of
events with a parsable Benzinga headline. It fails only when no event is found at all or no surprise parses.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.earnings_drift.event_source import EventSource
from trading_agent_framework.strategies.earnings_drift.events import reaction_date
from trading_agent_framework.strategies.earnings_drift.surprise import articles_for, pick_surprise
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"
SYMBOLS = ["OMC", "AAPL", "MSFT", "JPM", "NFLX", "CAT", "KO", "UNH"]


def main() -> int:
    if ENV_FILE.is_file():
        load_dotenv(ENV_FILE, override=True)
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "")
    if not user_agent.strip():
        print("SKIP: SEC_EDGAR_USER_AGENT is not set")
        return 0
    now = datetime.now(UTC)
    since = (now - timedelta(days=120)).date()
    source = EventSource(SecEdgarClient(user_agent, PROJECT_ROOT / "cache" / "sec"), reload_every_cycle=True)
    report = source.load(SYMBOLS, as_of=now, since=since)
    print(f"SEC: {report.loaded} symbols loaded, failed: {report.failed or 'none'}")
    news = AlpacaNewsProvider.from_credentials(AlpacaCredentials.for_news())
    events = [e for e in source.all_events() if e.accepted_at.date() >= since]
    if not events:
        print("FAIL: no 8-K item 2.02 event in the last 120 days")
        return 1
    parsed = 0
    weekdays = [since + timedelta(days=i) for i in range((now.date() - since).days + 2) if (since + timedelta(days=i)).weekday() < 5]
    for event in sorted(events, key=lambda e: e.accepted_at):
        reacts = reaction_date(event.accepted_at, weekdays)
        articles = news.get_news([event.symbol], start=event.accepted_at - timedelta(hours=2), end=event.accepted_at + timedelta(days=1), limit=50)
        picked = pick_surprise(articles_for(articles, event.symbol, start=event.accepted_at - timedelta(hours=2), end=event.accepted_at + timedelta(days=1)))
        parsed += picked is not None
        when = event.accepted_at.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M")
        if picked is None:
            print(f"{event.symbol:5} {when} -> reacts {reacts}: no surprise headline")
        else:
            s = picked.surprise
            print(f"{event.symbol:5} {when} -> reacts {reacts}: EPS {s.eps_actual} vs {s.eps_estimate} {s.eps_result}, sales {s.sales_result} | {picked.headline}")
    share = parsed / len(events)
    print(f"\n{parsed}/{len(events)} events with a parsed surprise ({share:.0%})")
    if parsed == 0:
        print("FAIL: no surprise headline parsed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TradingFrameworkError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
```

The weekday list ignores holidays: the smoke check prints reaction dates for a reader, it does not assert them.

- [ ] **Step 2: Run the smoke script by hand (needs network and credentials)**

Run: `uv run python scripts/tests/smoke_earnings_events.py`
Expected: `PASS` with most events showing a parsed surprise, or `SKIP:` without credentials. Write the printed share
of parsed events in the commit message: it is the Benzinga coverage figure the spec's §11 asks for. If it is below
about 70%, look at the unparsed headlines and stop to report before tuning the regexes.

- [ ] **Step 3: Update `CLAUDE.md`**

In "Commands", after the `smoke_quality_screen.py` line:

```bash
uv run python scripts/tests/smoke_earnings_events.py   # manual smoke test: earnings_drift events + Benzinga surprises against real SEC + Alpaca news (read-only)
```

In "Architecture", in the `strategies/` bullet, after the `bill_ackman/` description, add:

```markdown
`earnings_drift/` (`EarningsDriftStrategy`, registered as `"earnings_drift"` (agent, 1-year window) and `"earnings_drift_baseline"` (`DriftParams(agent_enabled=False)`, 5-year window)): post-earnings announcement drift. The cycle runs in `after_market_closes`, not on a tick. Pure `events` (SEC 8-K item 2.02, reaction session), `surprise` (Benzinga "EPS ... Estimate" headline parsing), `reaction` (daily-bar features), `screening` (gates), `fact_sheet`; `event_source.py` (the only SEC code, an in-memory event index), `scanner.py` (today's candidates), `desk.py` (the only order code: the agent's `buy` / `set_trailing_stop` / `sell` tools, a trailing stop on every entry fill, the guardrails), `book.py` (state file, `trades.jsonl`, `decisions.jsonl`).
```

In "Key patterns / gotchas", add a bullet:

```markdown
- **earnings_drift decides after the close, because of the backtest fill rule.** `BacktestBroker` skips the bar that is forming when an order arrives, so on daily bars an order sent at the 09:30 tick fills at the NEXT day's open, a session later than live. The cycle therefore runs in `after_market_closes` with `minutes_after_closing = 0`: an order sent at the exact close skips nothing and fills at the next open in both modes (Alpaca queues a DAY market order sent after hours). Paper/live first wait `live_bar_delay_seconds` (300) for the final daily bar. `on_trading_iteration` only raises a `FatalStrategyError` recorded by the previous cycle (3 agent failures or 3 hollow SEC scans in a backtest), because the executor swallows exceptions from every other hook. The agent places its own orders, but only through the desk's tools: `buy` (candidates only, whole shares up to `max_quantity`, a free slot, `trail_percent` in [3, 15]), `set_trailing_stop` (tighten-only), `sell`; each violation returns `{"error": ...}` and logs `guardrail order_limits`. Code guardrails, each logging `guardrail <name>`: `max_hold` (sold when `sessions_held` (entry session through today, both included) reaches 10, never stated in the prompt), `stop_backstop` (a missing stop is placed again; a stop refused twice becomes a market sell), `orphan` (a position of a symbol this strategy once traded is adopted, and a working trailing sell for it becomes its stop), `baseline`. The desk holds one re-entrant lock: tools may run on parallel LangGraph threads. In backtests the entry session has no stop (the fill hook places it at that bar's close); live, the hook places it seconds after the open. Reaction sessions take every close as 16:00 ET (early closes are a known limit).
```

- [ ] **Step 4: Update `README.md`**

After the `vwap_pullback_continuation` entry, add:

```markdown
#### 📈 `earnings_drift` — Post-earnings announcement drift (PEAD)
| Field | Value |
| --- | --- |
| **File** | `strategies/earnings_drift/agent_earnings_drift.py` (`EarningsDriftStrategy`) |
| **Model** | `LLM_*` (one agent, `"drift"`); none for `earnings_drift_baseline` |
| **Tools** | `buy`, `set_trailing_stop`, `sell`, `skip` (the desk), `search_news`, `get_filings`, `get_filing_document`, market data |
| **Asset universe** | the `cross_momentum` universe file (`uv run batch-universe`) |
| **Agent frequency** | once per session, right after the close; orders fill at the next open |
| **Trading modes** | backtest, paper, live |
| **Benchmark** | SPY |
| **Env file** | `env/.env.earnings_drift.<mode>`: `LLM_*`, `SEC_EDGAR_USER_AGENT`, `ALPACA_NEWS_*`, `ALPACA_DATA_*` (daily bars, also for backtests), plus the broker keys in paper/live |

Long only, 1 to 10 sessions. After each close, code finds the stocks that reacted today to an earnings release (SEC 8-K item 2.02), reads the Benzinga "EPS ... Estimate" headline and keeps an event only when EPS beat, the abnormal return against SPY is at least 3%, the stock held at least half its intraday gain and closed in the upper half of its range, on at least twice its 20-day volume, with a $10 price and $20M of daily dollar volume. The agent then buys or skips each candidate with a trailing stop of its choice (3% to 15%), and reviews its holdings (sell, or tighten the stop). Code guards every order: candidates only, sizes up to `max_quantity` (portfolio value / 8 positions), stops tighten-only, a stop on every fill, a sale after 10 sessions. Each guardrail logs a warning. `uv run agent earnings_drift_baseline backtesting` runs the same pipeline without the agent (every candidate, an 8% trail) over 5 years, to measure what the agent adds. Each run writes `trades.jsonl` and `decisions.jsonl` next to its report.
```

- [ ] **Step 5: Update the spec's §10**

Replace the sentence "The 5-year baseline run passes `start/end` from `PredefinedWindow.SEMI_DECADE` and
`settings=DriftParams(agent_enabled=False)`." with: "The 5-year baseline runs as its own registered strategy,
`earnings_drift_baseline` (`DriftParams(agent_enabled=False)`, `PredefinedWindow.SEMI_DECADE`, its own logs and
state file; env file `env/.env.earnings_drift_baseline.<mode>`, else `env/.env`)."

- [ ] **Step 6: Run the whole suite and the linter**

Run: `uv run pytest -q && uv run ruff check`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add scripts/tests/smoke_earnings_events.py CLAUDE.md README.md docs/superpowers/specs/2026-10-05-earnings-drift-design.md
git commit -m "docs: earnings_drift smoke script, CLAUDE.md and README entries (Task 14)

Benzinga coverage on the smoke run: <N>/<M> events with a parsed surprise.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
