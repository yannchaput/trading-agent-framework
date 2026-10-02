# Bill Ackman Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A daily three-agent strategy (`bill_ackman`: researcher, short seller, trader) over the fundamentals quality screen, with code owning the screen, the hand-offs, the state and every order, runnable in `live`, `paper` and `backtesting`.

**Architecture:** `ReviewPipeline` runs one review per session: the screen (code) gives at most 15 candidates, the researcher ranks 5, the short seller attacks the ranked names plus the current holdings, code applies a 2-consecutive-fails hysteresis, the trader picks weights, and `Rebalancer` places the orders (sells first, buys, then SHV). Each agent ends with one structured submit tool that code validates (`HandoffRecorder`); no agent has an order tool. The quality screen moves into `strategies/bill_ackman/screen/`; the SEC client and the generic SEC translation stay in `fundamentals/`.

**Tech Stack:** Python 3.14, `uv`, `pytest` (hand-written fakes, `FakeBroker`, `FakeBacktestDataSource`), `ruff`, LangChain only inside `AgentManager` (never imported by the new modules), `httpx` only in `edgar_client.py`.

**Spec:** `docs/superpowers/specs/2026-10-02-bill-ackman-strategy-design.md` (binding). The first spec, `docs/superpowers/specs/2026-10-02-fundamentals-quality-screen-design.md`, built the screen this plan moves and uses.

## Global Constraints

- Package manager is `uv`: `uv run pytest ...`, `uv run ruff check` (line length 200, rules E, F, I, UP, B). Never pip.
- Branch: `feature/bill-ackman-strategy`. The working tree may hold unrelated uncommitted changes (for example `TODO.md`): stage named files only, never `git add -A` or `git add .`.
- Commit trailer: the one in the executing session's attribution instructions. At planning time it is `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; the commit snippets below use it.
- The automated suite never touches the network (no SEC, no Yahoo, no LLM). Use hand-written fakes (`tests/fakes.py`, `tests/backtesting/fakes.py`), not `MagicMock`.
- **No agent has an order tool.** The three agents get only: researcher = `fundamentals_tools` + `market_data_tools` + `submit_ranking`; short seller = `fundamentals_tools` + `news_tools` + `market_data_tools` + `submit_verdicts`; trader = `submit_portfolio`. Free text from an agent is logged and otherwise ignored.
- `strategies/bill_ackman/handoff.py` has **no `from __future__ import annotations`**: the agent layer builds each tool's schema from real annotations. Every submit tool has a one-line docstring.
- Pure modules stay pure (no I/O, no clock, no logging): `fact_sheet.py`, `hysteresis.py`, `portfolio.py`, the validation half of `handoff.py`, `screen/quality.py`, `screen/annual_figures.py`, `fundamentals/sec.py`.
- Never let a raw exception escape: wrap failures in the framework errors (`utils/errors.py`). `ReviewPipeline.run` lets only a `ConfigurationError` from an agent and a non-framework bug out; `AgentError`, `FundamentalsError` and a stage that never submits abandon the review instead.
- No `time.sleep` / `datetime.now` in strategy code: time comes from `strategy.clock.now()` (the screen gets it converted to `MARKET_TZ`).
- LangChain is imported nowhere in the new package. `fundamentals_tools` and `news_tools` are imported inside `BillAckmanStrategy.initialize`, so importing the screen does not pull in the news stack.
- Money: the rebalancer sizes orders in floats and floors quantities with `fractional_qty` (6 decimals), exactly as `cross_momentum.rebalance` does; the broker receives `Decimal`s through `Strategy.create_order`. Weights and ratios are floats (dimensionless). This follows the existing precedent and is not a new class of float boundary elsewhere; it is called out in the final report for the user.
- Defaults (`AckmanParams`), verbatim from the spec: `research_top_n` 5, `max_positions` 5, `max_weight` 0.35, `min_weight` 0.05, `cash_buffer` 0.02, `forced_exit_fails` 2, `rebalance_band` 0.05, `min_trade_pct` 0.005, `parking_symbol` "SHV", `max_consecutive_abandoned` 3, `reason_max_chars` 300; `screen` is `ScreenParams()` (`top_n` 15). `min_weight >= rebalance_band`.
- Hand-off rules (verbatim from the spec): `submit_ranking` takes 1 to `research_top_n` unique ideas from the candidates (all of them if fewer); `submit_verdicts` takes exactly one `survive`/`fail` verdict per symbol asked about; `submit_portfolio` takes at most `max_positions` unique allowed stocks, each weight in [`min_weight`, `max_weight`], summing to at most `1 - cash_buffer`, and an empty list is valid; the first valid submission is final; an invalid one returns `{"error": ...}` and records nothing.
- Backtest window: `backtest_window(PredefinedWindow.BI_MONTH)` (`backtesting/time_window.py`), benchmark SPY, budget 100000, `warmup_trading_days` 10, Yahoo daily bars, `sleeptime "1D"`.
- Backtest state starts clean: `initialize` wipes the state file in backtesting mode only.

## Review Focus

Inputs and failure modes the spec implies but a task's main tests would not exercise. Each has a named test in the owning task.

1. **A price lookup that fails mid-review.** The review must still complete (fact sheets say `price: null`, the rebalancer skips what it cannot price) and never raise. Pinned by `test_a_failed_price_lookup_never_stops_a_review` (Task 10) and the rebalancer's failed-lookup tests (Task 8).
2. **No LLM configured (an agent run raises `ConfigurationError`).** It must propagate, trade nothing and leave the saved state untouched: it is a setup problem, not an abandoned review. Pinned by `test_a_configuration_error_from_an_agent_propagates_and_leaves_the_state_alone` (Task 10).
3. **A held position SEC has no data for (an ETF, a delisted name).** It is described by a reduced fact sheet, reviewed like any holding, and sold if the trader drops it. Pinned by `test_a_position_the_screen_does_not_know_can_be_dropped_by_the_trader_and_is_then_sold` (Task 10).
4. **A restart between reviews in live.** The counters must survive: the next review continues them. Pinned by `test_the_counters_survive_a_restart` (Task 10) and the round-trip tests (Task 9).
5. **An order still open from the previous review.** A backtest fills an order on the next bar, so the daily review always meets yesterday's orders pending; they must not be sent again. Pinned by the rebalancer's open-order tests (Task 8) and `lines[1]["orders"] == []` in the end-to-end backtest (Task 12).

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/trading_agent_framework/strategies/bill_ackman/screen/{quality,screen,splits,annual_store}.py` | moved from `fundamentals/` | the quality screen, unchanged apart from imports (and `top_n`, Task 2) |
| `src/trading_agent_framework/strategies/bill_ackman/screen/annual_figures.py` | new (cut from `fundamentals/sec.py`) | `annual_figures`, `parse_sic`, `ANNUAL_FLOW_TAGS`, fiscal-year constants |
| `src/trading_agent_framework/strategies/bill_ackman/screen/__init__.py` | new | exports `Candidate`, `QualityScreen`, `ScreenParams`, `ScreenResult`, `build_quality_screen` |
| `src/trading_agent_framework/fundamentals/{sec,edgar_client,__init__}.py` | modified | `parse_dt` public; payload getters take `as_of`/`max_age_days`; exports only `SecEdgarClient` |
| `src/trading_agent_framework/agents/tools/fundamentals.py` | modified | passes the strategy clock and 30 days to the client |
| `src/trading_agent_framework/utils/helpers.py` | modified | gains `fractional_qty`, `parse_insufficient_buying_power` |
| `src/trading_agent_framework/strategies/cross_momentum/utils.py` | modified | re-exports the two helpers |
| `src/trading_agent_framework/strategies/bill_ackman/parameters.py` | new | `AckmanParams` |
| `src/trading_agent_framework/strategies/bill_ackman/fact_sheet.py` | new | per-company dict for the agents (pure) |
| `src/trading_agent_framework/strategies/bill_ackman/hysteresis.py` | new | fail counters, forced exits, allowed set (pure) |
| `src/trading_agent_framework/strategies/bill_ackman/portfolio.py` | new | weights to `TargetPortfolio` (pure) |
| `src/trading_agent_framework/strategies/bill_ackman/handoff.py` | new | validation, `HandoffRecorder`, the three submit tools |
| `src/trading_agent_framework/strategies/bill_ackman/rebalancer.py` | new | the only order code |
| `src/trading_agent_framework/strategies/bill_ackman/state.py` | new | `StateStore`, `ReviewLog`, `state_path` |
| `src/trading_agent_framework/strategies/bill_ackman/prompts.py` | new | the three system prompts, task prompts, retry prompt |
| `src/trading_agent_framework/strategies/bill_ackman/pipeline.py` | new | `ReviewPipeline` |
| `src/trading_agent_framework/strategies/bill_ackman/agent_bill_ackman.py` + `__init__.py` | new | `BillAckmanStrategy` |
| `src/trading_agent_framework/main.py` | modified | registers `"bill_ackman"` |
| `tests/strategies/bill_ackman/screen/*` | moved from `tests/fundamentals/` | the screen's tests, plus `test_annual_figures.py` cut from `test_sec.py` |
| `tests/strategies/bill_ackman/test_ackman_*.py` | new | one file per new module, plus the end-to-end backtest |
| `scripts/tests/smoke_quality_screen.py`, `scripts/tests/HOWTO.md` | modified | new imports, `SKIP:` without the user agent, HOWTO section |
| `README.md`, `CLAUDE.md` | modified | docs |

---

### Task 1: Move the screen into the strategy package (no behaviour change)

**Files:**
- Move (`git mv`): `src/trading_agent_framework/fundamentals/{quality,screen,splits,annual_store}.py` to `src/trading_agent_framework/strategies/bill_ackman/screen/`
- Move (`git mv`): `tests/fundamentals/{test_quality,test_screen,test_screen_integration,test_splits,test_annual_store,annual_fixtures}.py` to `tests/strategies/bill_ackman/screen/`
- Create: `src/trading_agent_framework/strategies/bill_ackman/__init__.py`, `.../screen/__init__.py`, `.../screen/annual_figures.py` (cut from `fundamentals/sec.py`), `tests/strategies/bill_ackman/screen/test_annual_figures.py` (cut from `tests/fundamentals/test_sec.py`)
- Modify: `src/trading_agent_framework/fundamentals/sec.py`, `src/trading_agent_framework/fundamentals/__init__.py`, `scripts/tests/smoke_quality_screen.py`, `docs/superpowers/specs/2026-10-02-fundamentals-quality-screen-design.md` (a one-line note)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - Package `trading_agent_framework.strategies.bill_ackman.screen` exporting `Candidate`, `QualityScreen`, `ScreenParams`, `ScreenResult`, `build_quality_screen` (same signatures as before the move).
  - Modules `...screen.quality`, `...screen.screen`, `...screen.splits`, `...screen.annual_store`, `...screen.annual_figures` (`annual_figures`, `parse_sic`, `ANNUAL_FLOW_TAGS`, `MIN_FISCAL_YEAR_DAYS`, `MAX_FISCAL_YEAR_DAYS`).
  - `trading_agent_framework.fundamentals.sec.parse_dt(value) -> datetime | None` (was the private `_parse_dt`).
  - `trading_agent_framework.fundamentals` exports only `SecEdgarClient`.
  - Later tasks import the screen from `trading_agent_framework.strategies.bill_ackman.screen` and `ScreenParams` from `...screen.quality`.

This is a pure move: the same tests pass before and after, with the same count.

- [ ] **Step 1: Record the baseline**

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: `1929 passed` (note the exact number: it must be the same at Step 8).

- [ ] **Step 2: Move the files**

```bash
S=src/trading_agent_framework
T=tests
mkdir -p $S/strategies/bill_ackman/screen $T/strategies/bill_ackman/screen
for m in quality screen splits annual_store; do git mv $S/fundamentals/$m.py $S/strategies/bill_ackman/screen/$m.py; done
for f in test_quality test_screen test_screen_integration test_splits test_annual_store annual_fixtures; do git mv $T/fundamentals/$f.py $T/strategies/bill_ackman/screen/$f.py; done
```

- [ ] **Step 3: Cut the screen-only code out of `sec.py`**

This moves `annual_figures`, `parse_sic`, `ANNUAL_FLOW_TAGS`, the fiscal-year constants and their private helpers into `screen/annual_figures.py`, and makes `_parse_dt` public as `parse_dt`. Run from the repository root:

```bash
uv run python - <<'PY'
"""Cut the screen-only code out of fundamentals/sec.py into strategies/bill_ackman/screen/annual_figures.py."""

from pathlib import Path

sec_path = Path("src/trading_agent_framework/fundamentals/sec.py")
text = sec_path.read_text()
start_const = text.index("# The quality screen's annual figures")
end_const = text.index("def parse_company_tickers")
consts = text[start_const:end_const].rstrip() + "\n"
start_fn = text.index("def _tag_rows(")
fns = text[start_fn:]
remaining = (text[:start_const] + text[end_const:start_fn]).rstrip() + "\n"
sec_path.write_text(remaining.replace("_parse_dt", "parse_dt"))  # `parse_dt` becomes public: annual_figures.py imports it

header = '''"""The quality screen's reduction of a SEC company-facts payload to annual figures.

Pure translation (no I/O, no state, no clock): `annual_figures` turns a company-facts payload (about 4 MB)
into the few KB of rows the screen needs, and `parse_sic` reads the industry code from a submissions
payload. A fiscal year is identified by its period END date, never by XBRL's `fy` field: each 10-K repeats
three years of figures, all tagged with the filing's own `fy`.
"""

from __future__ import annotations

from typing import Any

from trading_agent_framework.fundamentals.sec import BALANCE_SHEET_TAGS, INCOME_STATEMENT_TAGS, parse_dt

'''
consts = consts.replace(
    "# The quality screen's annual figures (`annual_figures`). A fiscal year is identified by its period END\n"
    "# date: each 10-K repeats three years of figures, all tagged with the filing's own `fy`.\n",
    "",
)
out = Path("src/trading_agent_framework/strategies/bill_ackman/screen/annual_figures.py")
out.write_text(header + consts + "\n\n" + fns.replace("_parse_dt", "parse_dt"))
PY
```

Check: `grep -c "_parse_dt" src/trading_agent_framework/fundamentals/sec.py` prints `0`, and `grep -n "def annual_figures" src/trading_agent_framework/strategies/bill_ackman/screen/annual_figures.py` finds it.

- [ ] **Step 4: Write the package files**

`src/trading_agent_framework/strategies/bill_ackman/__init__.py` (replaced by the strategy export in Task 11):

```python
"""The Bill Ackman portfolio strategy: a researcher, a short seller and a trader over the fundamentals quality screen."""
```

`src/trading_agent_framework/strategies/bill_ackman/screen/__init__.py`:

```python
"""The fundamentals quality screen: which companies were simple, predictable, cash-generative, lightly indebted
and reasonably priced on a date (`quality.py` pure gates and score, `annual_figures.py` pure SEC reduction,
`annual_store.py` and `splits.py` I/O, `screen.py` wiring)."""

from trading_agent_framework.strategies.bill_ackman.screen.quality import Candidate, ScreenParams, ScreenResult
from trading_agent_framework.strategies.bill_ackman.screen.screen import QualityScreen, build_quality_screen

__all__ = ["Candidate", "QualityScreen", "ScreenParams", "ScreenResult", "build_quality_screen"]
```

`src/trading_agent_framework/fundamentals/__init__.py` (replace the whole file):

```python
"""SEC EDGAR fundamentals: `sec.py` (pure translation) and `edgar_client.py` (cached I/O)."""

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils import get_version

__version__ = get_version("trading_agent_framework")

__all__ = ["SecEdgarClient"]
```

- [ ] **Step 5: Point the moved modules and tests at their new locations**

```bash
uv run python - <<'PY'
"""Point the moved modules and tests at their new locations."""

import re
from pathlib import Path

NEW = "trading_agent_framework.strategies.bill_ackman.screen"
roots = [Path("src/trading_agent_framework/strategies/bill_ackman/screen"), Path("tests/strategies/bill_ackman/screen")]
for root in roots:
    for path in root.glob("*.py"):
        text = original = path.read_text()
        text = re.sub(r"trading_agent_framework\.fundamentals\.(quality|screen|splits|annual_store)\b", NEW + r".\1", text)
        text = text.replace("from trading_agent_framework.fundamentals import annual_store", f"from {NEW} import annual_store")
        text = text.replace(
            "from trading_agent_framework.fundamentals import Candidate, QualityScreen, ScreenParams, ScreenResult, build_quality_screen",
            f"from {NEW} import Candidate, QualityScreen, ScreenParams, ScreenResult, build_quality_screen",
        )
        text = text.replace("from trading_agent_framework.fundamentals import QualityScreen, ScreenParams", f"from {NEW} import QualityScreen, ScreenParams")
        text = text.replace("tests.fundamentals.annual_fixtures", "tests.strategies.bill_ackman.screen.annual_fixtures")
        if path.name == "quality.py":
            text = text.replace(
                "from trading_agent_framework.fundamentals.sec import MAX_FISCAL_YEAR_DAYS, MIN_FISCAL_YEAR_DAYS",
                f"from {NEW}.annual_figures import MAX_FISCAL_YEAR_DAYS, MIN_FISCAL_YEAR_DAYS",
            )
            text = text.replace("(same rules as `sec.py`)", "(same rules as `annual_figures.py`)").replace("(`sec.annual_figures`)", "(`annual_figures.annual_figures`)")
        if path.name == "annual_store.py":
            text = text.replace("from trading_agent_framework.fundamentals import sec\n", f"from {NEW}.annual_figures import annual_figures, parse_sic\n")
            text = text.replace("sec.parse_sic(", "parse_sic(").replace("sec.annual_figures(", "annual_figures(")
            text = text.replace("`sec.annual_figures(...)`", "`annual_figures(...)`").replace("`sec.py` (`annual_figures`)", "`annual_figures.py`")
        if text != original:
            path.write_text(text)
            print("edited", path)
PY
```

- [ ] **Step 6: Move the `annual_figures` / `parse_sic` tests out of `test_sec.py`**

```bash
uv run python - <<'PY'
"""Move the annual_figures / parse_sic tests out of tests/fundamentals/test_sec.py into their own file."""

from pathlib import Path

test_sec = Path("tests/fundamentals/test_sec.py")
lines = test_sec.read_text().splitlines(keepends=True)
first = next(i for i, line in enumerate(lines) if line.startswith("def _annual("))
moved = "".join(lines[first:])
test_sec.write_text("".join(lines[:first]).rstrip() + "\n")
header = '''from __future__ import annotations

from datetime import date, timedelta

import pytest

from trading_agent_framework.strategies.bill_ackman.screen import annual_figures as af


'''
for old, new in (
    ("sec.annual_figures(", "af.annual_figures("),
    ("sec.parse_sic(", "af.parse_sic("),
    ("sec.MIN_FISCAL_YEAR_DAYS", "af.MIN_FISCAL_YEAR_DAYS"),
    ("sec.MAX_FISCAL_YEAR_DAYS", "af.MAX_FISCAL_YEAR_DAYS"),
):
    moved = moved.replace(old, new)
Path("tests/strategies/bill_ackman/screen/test_annual_figures.py").write_text(header + moved)
PY
```

- [ ] **Step 7: Update the smoke script's import**

```bash
sed -i 's/from trading_agent_framework.fundamentals import ScreenParams, build_quality_screen/from trading_agent_framework.strategies.bill_ackman.screen import ScreenParams, build_quality_screen/' scripts/tests/smoke_quality_screen.py
```

- [ ] **Step 8: Lint, and run the whole suite**

Run: `uv run ruff check --fix src tests scripts && uv run ruff check && uv run pytest -q 2>&1 | tail -1`
Expected: `All checks passed!` and the same pass count as Step 1 (`1929 passed`). `ruff --fix` sorts the rewritten import blocks.

- [ ] **Step 9: Check nothing still points at the old locations**

Run: `grep -rn "fundamentals\.\(quality\|screen\|splits\|annual_store\)\|from trading_agent_framework.fundamentals import \(Candidate\|QualityScreen\|ScreenParams\|ScreenResult\|build_quality_screen\)" src tests scripts`
Expected: no output.

- [ ] **Step 10: Note the move in the first spec**

In `docs/superpowers/specs/2026-10-02-fundamentals-quality-screen-design.md`, insert a blank line and then this note right after the first paragraph (the one that ends `... that consumes this screen)`), before `## Problem`:

```markdown
> **Note (moved):** the screen described here now lives in `src/trading_agent_framework/strategies/bill_ackman/screen/`
> (`quality.py`, `screen.py`, `splits.py`, `annual_store.py`, and `annual_figures.py`, which holds the annual-figures
> reduction that used to be in `fundamentals/sec.py`). `fundamentals/` keeps the SEC client and the generic SEC
> translation. See `docs/superpowers/specs/2026-10-02-bill-ackman-strategy-design.md` §1.1.
```

- [ ] **Step 11: Commit**

```bash
git add src/trading_agent_framework/fundamentals src/trading_agent_framework/strategies/bill_ackman tests/fundamentals tests/strategies/bill_ackman scripts/tests/smoke_quality_screen.py docs/superpowers/specs/2026-10-02-fundamentals-quality-screen-design.md
git status --short
git commit -m "refactor: move the quality screen into strategies/bill_ackman/screen

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
`git status --short` must show only unrelated files (for example `TODO.md`) as modified.

---

### Task 2: `QualityScreen.run(top_n=)`, and the screen smoke script's `SKIP` and HOWTO

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/screen/screen.py`
- Test: `tests/strategies/bill_ackman/screen/test_screen.py`
- Modify: `scripts/tests/smoke_quality_screen.py`, `scripts/tests/HOWTO.md`

**Interfaces:**
- Consumes: `QualityScreen.run(symbols, *, as_of, price_of)` from Task 1.
- Produces: `QualityScreen.run(symbols, *, as_of, price_of, top_n: int | None = None) -> ScreenResult`. A `top_n` replaces `params.top_n` for that call only (a negative value raises `ValueError` through `ScreenParams`' own validation); `None` keeps today's behaviour. The pipeline screens the holdings with `top_n=len(holdings)`.

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/bill_ackman/screen/test_screen.py`, insert these three tests immediately before `def test_each_run_logs_one_summary_line(`:

```python
def test_a_top_n_argument_overrides_the_params_for_that_call_only() -> None:
    names = ["AAA", "BBB", "CCC"]
    screen = QualityScreen(FakeStore({name: healthy_figures() for name in names}), FakeSplits(), params=ScreenParams(top_n=2))

    limited = screen.run(names, as_of=AS_OF, price_of=Prices(), top_n=1)
    default = screen.run(names, as_of=AS_OF, price_of=Prices())

    assert len(limited.candidates) == 1
    assert len(default.candidates) == 2  # the override did not stick


def test_a_top_n_of_zero_returns_no_candidates_but_still_reports_rejections() -> None:
    screen = QualityScreen(FakeStore({"AAA": healthy_figures()}), FakeSplits())

    result = screen.run(["AAA", "GONE"], as_of=AS_OF, price_of=Prices(), top_n=0)

    assert result.candidates == []
    assert result.rejections == {"GONE": "no_data"}


def test_a_negative_top_n_argument_is_refused() -> None:
    screen = QualityScreen(FakeStore({"AAA": healthy_figures()}), FakeSplits())

    with pytest.raises(ValueError, match="top_n"):
        screen.run(["AAA"], as_of=AS_OF, price_of=Prices(), top_n=-1)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/screen/test_screen.py -q -k top_n`
Expected: `3 failed, 1 passed` with `TypeError: QualityScreen.run() got an unexpected keyword argument 'top_n'` (one existing test also matches `-k top_n` and passes).

- [ ] **Step 3: Implement**

In `src/trading_agent_framework/strategies/bill_ackman/screen/screen.py`:

1. Add `from dataclasses import replace` to the imports (after `from collections.abc import Callable, Sequence`; `ruff --fix` sorts it).
2. Replace the signature line of `run` with:

```python
    def run(self, symbols: Sequence[str], *, as_of: datetime, price_of: Callable[[str], Decimal | None], top_n: int | None = None) -> ScreenResult:
```

3. In the docstring of `run`, after the sentence ending `... before the price gate.` add:

```
        `top_n`, when given, replaces `params.top_n` for this call only (the holdings of a strategy are screened
        with `top_n=len(holdings)` so none is cut); a negative value raises `ValueError`.
```

4. Replace the line `        params = self.params` (the first line after the `as_of` check) with:

```python
        params = self.params if top_n is None else replace(self.params, top_n=top_n)
```

- [ ] **Step 4: Run the screen tests**

Run: `uv run ruff check --fix src tests && uv run pytest tests/strategies/bill_ackman/screen -q`
Expected: all pass (the three new tests included), no lint errors.

- [ ] **Step 5: Make the smoke script `SKIP` instead of failing without the user agent**

`scripts/tests/run_smoke_tests.sh` already runs every `smoke_*.py`; it reports a script whose output has a line starting `SKIP:` as skipped. In `scripts/tests/smoke_quality_screen.py`, replace

```python
    if not os.environ.get("SEC_EDGAR_USER_AGENT"):
        print(f"FAILED: SEC_EDGAR_USER_AGENT is not set (define it in {ENV_FILE} or in the environment)")
        return 1
```

with

```python
    if not os.environ.get("SEC_EDGAR_USER_AGENT"):
        # `run_smoke_tests.sh` reports a script whose output has a `SKIP:` line as skipped, not failed.
        print(f"SKIP: SEC_EDGAR_USER_AGENT is not set (define it in {ENV_FILE} or in the environment)")
        return 0
```

and in the module docstring replace the two lines

```
`SEC_EDGAR_USER_AGENT` comes from env/.env.alpaca.integration-tests, as in the other smoke scripts
(or from the environment).
```

with

```
`SEC_EDGAR_USER_AGENT` comes from env/.env.alpaca.integration-tests, as in the other smoke scripts
(or from the environment); without it the script prints `SKIP:` and exits 0, so the aggregate runner
`scripts/tests/run_smoke_tests.sh` reports it as skipped.
```

- [ ] **Step 6: Verify the `SKIP` behaviour (no network)**

```bash
env -u SEC_EDGAR_USER_AGENT uv run python - <<'PY'
import os, runpy

os.environ.pop("SEC_EDGAR_USER_AGENT", None)
try:
    runpy.run_path("scripts/tests/smoke_quality_screen.py", run_name="__main__")
except SystemExit as exc:
    print("exit code", exc.code)
PY
```
Expected: a line starting `SKIP: SEC_EDGAR_USER_AGENT is not set` and `exit code 0`. (If the machine has `env/.env.alpaca.integration-tests` with the variable defined, the script loads it and would run for real: temporarily rename that file to check, then rename it back.)

- [ ] **Step 7: Add the HOWTO section**

In `scripts/tests/HOWTO.md`, insert before the line `## Tools`:

```markdown
## Smoke quality screen
Manual run of the fundamentals quality screen (`strategies/bill_ackman/screen/`) against real SEC EDGAR and Yahoo data.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_quality_screen.py

`SEC_EDGAR_USER_AGENT` ("<app name> <contact email>") comes from env/.env.alpaca.integration-tests, as in the
other smoke scripts, or from the environment. Without it the script prints `SKIP:` and exits 0, so
`run_smoke_tests.sh` reports it as skipped. The script is read-only. The first run downloads one SEC company-facts
payload per symbol (about 4 MB each; a full 1,200-symbol universe is about 5 GB and 10 to 20 minutes) and writes
reduced copies under cache/sec/annual/; a second run the same day makes no SEC request and finishes in seconds.
It fails only on things that must always hold: every symbol is either a candidate or rejected with a reason,
an unknown ticker is `no_data`, a bank is rejected, the two Alphabet listings never both become candidates,
at least three symbols are candidates, the known-indebted names (KO, MDLZ, HLT) that are candidates have a
reported debt figure, the utility NEE is never a candidate, and the equipment lessor URI, if a candidate, shows
a plausible operating margin. If it ever fails on the debt check, a debt tag is probably missing from
`screen/annual_figures.py` (bump `annual_store.SCHEMA_VERSION` after fixing it).
```

- [ ] **Step 8: Run the whole suite, then commit**

Run: `uv run ruff check && uv run pytest -q 2>&1 | tail -1`
Expected: no lint errors; `1932 passed` (1929 plus the three new tests).

```bash
git add src/trading_agent_framework/strategies/bill_ackman/screen/screen.py tests/strategies/bill_ackman/screen/test_screen.py scripts/tests/smoke_quality_screen.py scripts/tests/HOWTO.md
git commit -m "feat: QualityScreen.run(top_n=), and the screen smoke script skips without a user agent

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fresh SEC payloads for the drill-down tools

**Files:**
- Modify: `src/trading_agent_framework/fundamentals/edgar_client.py`
- Modify: `src/trading_agent_framework/agents/tools/fundamentals.py`
- Test: `tests/fundamentals/test_edgar_client.py`, `tests/agents/tools/test_fundamentals_tools.py`

**Interfaces:**
- Consumes: `fundamentals/freshness.is_stale(fetched_at, as_of, max_age_days) -> bool` (already in the repo).
- Produces:
  - `SecEdgarClient.get_json(url, cache_key, *, as_of: datetime | None = None, max_age_days: int | None = None) -> dict`
  - `SecEdgarClient.get_company_facts_payload(cik, *, as_of=None, max_age_days=None)` and `get_submissions_payload(cik, *, as_of=None, max_age_days=None)`.
  - With both arguments, a cached file whose modification time is more than `max_age_days` before `as_of` is fetched again and rewritten; a stale file that cannot be refreshed is served with a warning. Without them, behaviour is exactly as before (cached forever). A naive `as_of` raises `ValueError("as_of must be timezone-aware")`.
  - `agents/tools/fundamentals.MAX_PAYLOAD_AGE_DAYS = 30`; every payload read by `fundamentals_tools` passes `as_of=strategy.clock.now()` and `max_age_days=MAX_PAYLOAD_AGE_DAYS`.

- [ ] **Step 1: Write the failing client tests**

In `tests/fundamentals/test_edgar_client.py`, replace the first import lines

```python
import json
from pathlib import Path
```

with

```python
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
```

and append at the end of the file:

```python
# --- freshness of the cached payloads (as_of / max_age_days) ---------------------------------------

_FACTS_FILE = ("companyfacts", "CIK0000320193.json")


def _age_file(path: Path, *, days: float) -> None:
    """Make `path` look like it was written `days` ago (its modification time is what the client reads)."""
    then = (datetime.now(UTC) - timedelta(days=days)).timestamp()
    os.utime(path, (then, then))


def _seed_facts(tmp_path: Path, payload: dict[str, object], *, age_days: float) -> Path:
    path = tmp_path.joinpath(*_FACTS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _age_file(path, days=age_days)
    return path


def test_a_fresh_cached_payload_is_served_without_a_request(tmp_path: Path) -> None:
    calls = []
    _seed_facts(tmp_path, {"v": "cached"}, age_days=5)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    payload = client.get_company_facts_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"v": "cached"}
    assert calls == []


def test_a_stale_cached_payload_is_refetched_and_rewritten(tmp_path: Path) -> None:
    calls = []
    path = _seed_facts(tmp_path, {"v": "cached"}, age_days=40)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    payload = client.get_company_facts_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"v": "new"}
    assert len(calls) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": "new"}


def test_a_file_written_today_is_fresh_for_every_past_as_of(tmp_path: Path) -> None:
    # A backtest asks with a simulated, past `as_of`: the file's own date is today, so it never refetches.
    calls = []
    _seed_facts(tmp_path, {"v": "cached"}, age_days=0)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    payload = client.get_company_facts_payload("0000320193", as_of=datetime(2024, 1, 2, tzinfo=UTC), max_age_days=30)

    assert payload == {"v": "cached"}
    assert calls == []


def test_a_stale_payload_survives_a_failed_refresh_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _seed_facts(tmp_path, {"v": "cached"}, age_days=40)
    client = _client(tmp_path, lambda request: httpx.Response(500))

    with caplog.at_level(logging.WARNING):
        payload = client.get_company_facts_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"v": "cached"}
    assert "could not be refreshed" in caplog.text


def test_without_as_of_and_max_age_an_old_cached_payload_is_still_served(tmp_path: Path) -> None:
    calls = []
    _seed_facts(tmp_path, {"v": "cached"}, age_days=400)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    assert client.get_company_facts_payload("0000320193") == {"v": "cached"}
    assert calls == []


def test_a_naive_as_of_is_refused_when_a_cached_file_exists(tmp_path: Path) -> None:
    _seed_facts(tmp_path, {"v": "cached"}, age_days=1)
    client = _client(tmp_path, lambda request: httpx.Response(200, json={"v": "new"}))

    with pytest.raises(ValueError, match="timezone-aware"):
        client.get_company_facts_payload("0000320193", as_of=datetime(2026, 9, 14), max_age_days=30)


def test_a_corrupt_cached_file_is_refetched_even_with_freshness_arguments(tmp_path: Path) -> None:
    calls = []
    path = _seed_facts(tmp_path, {}, age_days=1)
    path.write_text("{not json", encoding="utf-8")
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    payload = client.get_company_facts_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"v": "new"}
    assert len(calls) == 1


def test_the_submissions_payload_takes_the_same_freshness_arguments(tmp_path: Path) -> None:
    calls = []
    path = tmp_path / "submissions" / "CIK0000320193.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"filings": "old"}), encoding="utf-8")
    _age_file(path, days=40)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"filings": "new"}))

    payload = client.get_submissions_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"filings": "new"}
    assert len(calls) == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/fundamentals/test_edgar_client.py -q`
Expected: `7 failed` (`TypeError: ... got an unexpected keyword argument 'as_of'`, and the naive-`as_of` test failing for the same reason); the `without_as_of_and_max_age` test already passes.

- [ ] **Step 3: Implement the client changes**

In `src/trading_agent_framework/fundamentals/edgar_client.py`:

1. Replace the imports block

```python
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError, FundamentalsNotFoundError
```

with

```python
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError, FundamentalsNotFoundError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "SecEdgarClient")
```

2. Replace the whole `get_json` method (from `    def get_json(` up to, not including, `    def get_text(`) with:

```python
    def get_json(
        self, url: str, cache_key: tuple[str, ...], *, as_of: datetime | None = None, max_age_days: int | None = None
    ) -> dict[str, Any]:
        """The payload at `url`, cached to disk.

        Without `as_of` and `max_age_days` a cached file is served forever (the original behaviour). With both,
        a cached file whose modification time is more than `max_age_days` before `as_of` is stale and is
        fetched again (`freshness.is_stale`, the screen's rule): in a backtest `as_of` is simulated, and a file
        written today is fresh for every past date; live refreshes monthly. A stale file that cannot be
        refreshed is served with a warning rather than failing the caller.
        """
        cache_path = self._cache_path(*cache_key)
        cached = self._read_cached_json(cache_path)
        if cached is not None and not self._cache_is_stale(cache_path, as_of, max_age_days):
            return cached
        try:
            payload = self.fetch_json(url)
        except FundamentalsError as exc:
            if cached is None:
                raise
            logger.log_warning(f"{url} could not be refreshed, using the cached copy: {exc}")
            return cached
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    @staticmethod
    def _read_cached_json(cache_path: Path) -> dict[str, Any] | None:
        if not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            # A corrupt/truncated cache entry (e.g. from an interrupted write) is
            # treated as a cache miss so it self-heals on the next fetch, rather than
            # permanently wedging the tool until someone deletes the file by hand.
            return None

    @staticmethod
    def _cache_is_stale(cache_path: Path, as_of: datetime | None, max_age_days: int | None) -> bool:
        if as_of is None or max_age_days is None:
            return False
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        return is_stale(datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC), as_of, max_age_days)
```

3. Replace the two payload getters

```python
    def get_company_facts_payload(self, cik: str) -> dict[str, Any]:
        return self.get_json(_company_facts_url(cik), ("companyfacts", f"CIK{cik}.json"))
```

```python
    def get_submissions_payload(self, cik: str) -> dict[str, Any]:
        return self.get_json(_submissions_url(cik), ("submissions", f"CIK{cik}.json"))
```

with

```python
    def get_company_facts_payload(self, cik: str, *, as_of: datetime | None = None, max_age_days: int | None = None) -> dict[str, Any]:
        return self.get_json(_company_facts_url(cik), ("companyfacts", f"CIK{cik}.json"), as_of=as_of, max_age_days=max_age_days)
```

```python
    def get_submissions_payload(self, cik: str, *, as_of: datetime | None = None, max_age_days: int | None = None) -> dict[str, Any]:
        return self.get_json(_submissions_url(cik), ("submissions", f"CIK{cik}.json"), as_of=as_of, max_age_days=max_age_days)
```

- [ ] **Step 4: Run the client tests**

Run: `uv run ruff check --fix src tests && uv run pytest tests/fundamentals -q`
Expected: all pass, no lint errors.

- [ ] **Step 5: Write the failing tool test**

In `tests/agents/tools/test_fundamentals_tools.py`:

1. In `_FakeEdgarClient.__init__`, after the line `self.raises: FundamentalsError | None = None` add:

```python
        self.freshness_args: list[tuple[str, object, object]] = []  # (endpoint, as_of, max_age_days) of every payload read
```

2. Replace the two fake getters

```python
    def get_company_facts_payload(self, cik: str) -> dict[str, object]:
        return self.company_facts

    def get_submissions_payload(self, cik: str) -> dict[str, object]:
        return self.submissions
```

with

```python
    def get_company_facts_payload(self, cik: str, *, as_of: object = None, max_age_days: object = None) -> dict[str, object]:
        self.freshness_args.append(("companyfacts", as_of, max_age_days))
        return self.company_facts

    def get_submissions_payload(self, cik: str, *, as_of: object = None, max_age_days: object = None) -> dict[str, object]:
        self.freshness_args.append(("submissions", as_of, max_age_days))
        return self.submissions
```

3. Append at the end of the file:

```python
def test_every_payload_read_passes_the_strategy_clock_and_a_thirty_day_freshness_to_the_client() -> None:
    client = _FakeEdgarClient()
    client.submissions = {
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": ["0000320193-26-000001"],
                "filingDate": ["2026-01-02"],
                "reportDate": ["2025-12-31"],
                "acceptanceDateTime": ["2026-01-02T10:00:00.000Z"],
                "primaryDocument": ["a.htm"],
            }
        }
    }
    tools = _tools(client)
    now = et(2026, 9, 14, 10)

    tools["get_company_facts"]("AAPL")  # type: ignore
    tools["get_income_statement"]("AAPL")  # type: ignore
    tools["get_balance_sheet"]("AAPL")  # type: ignore
    tools["get_filings"]("AAPL")  # type: ignore
    tools["get_filing_document"]("AAPL", "0000320193-26-000001")  # type: ignore

    assert [endpoint for endpoint, _, _ in client.freshness_args] == ["companyfacts", "companyfacts", "companyfacts", "submissions", "submissions"]
    assert all(as_of == now and max_age_days == 30 for _, as_of, max_age_days in client.freshness_args)
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/agents/tools/test_fundamentals_tools.py -q`
Expected: `1 failed, 8 passed` (the new test: the tools do not pass the arguments yet).

- [ ] **Step 7: Pass the arguments from the tools**

In `src/trading_agent_framework/agents/tools/fundamentals.py`:

1. After `MAX_FILINGS_LIMIT = 25` add:

```python
# A cached SEC payload older than this (by the strategy clock, so a backtest never refetches) is fetched again.
MAX_PAYLOAD_AGE_DAYS = 30
```

2. Replace both occurrences of `payload = edgar.get_company_facts_payload(cik)` with

```python
            payload = edgar.get_company_facts_payload(cik, as_of=as_of, max_age_days=MAX_PAYLOAD_AGE_DAYS)
```

3. Replace both occurrences of `submissions = edgar.get_submissions_payload(cik)` with

```python
            submissions = edgar.get_submissions_payload(cik, as_of=as_of, max_age_days=MAX_PAYLOAD_AGE_DAYS)
```

(`as_of` is already defined in each of those four tools as `strategy.clock.now()`.)

- [ ] **Step 8: Run the agent and fundamentals tests, then the suite**

Run: `uv run ruff check --fix src tests && uv run ruff check && uv run pytest tests/agents tests/fundamentals -q && uv run pytest -q 2>&1 | tail -1`
Expected: all pass; `1941 passed` (1932 plus 8 client tests and 1 tool test).

- [ ] **Step 9: Commit**

```bash
git add src/trading_agent_framework/fundamentals/edgar_client.py src/trading_agent_framework/agents/tools/fundamentals.py tests/fundamentals/test_edgar_client.py tests/agents/tools/test_fundamentals_tools.py
git commit -m "feat: SEC payloads for the drill-down tools refresh monthly by the strategy clock

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Move the two order helpers to `utils/helpers.py`

**Files:**
- Modify: `src/trading_agent_framework/utils/helpers.py`, `src/trading_agent_framework/strategies/cross_momentum/utils.py`
- Test: `tests/test_order_helpers.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `trading_agent_framework.utils.helpers.fractional_qty(value: float, decimals: int = 6) -> float` (floors, never rounds up) and `parse_insufficient_buying_power(error: Exception) -> float | None` (the broker's real buying power from an Alpaca "insufficient buying power" error, else `None`). `cross_momentum/utils.py` re-imports both under the same names, so its call sites and tests are untouched. This is a move: the function bodies do not change.

- [ ] **Step 1: Write the tests**

`tests/test_order_helpers.py`:

```python
from __future__ import annotations

import json

import pytest

from trading_agent_framework.utils.helpers import fractional_qty, parse_insufficient_buying_power


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (1.23456789, 6, 1.234567),  # floored, never rounded up
        (1.9999999, 6, 1.999999),
        (5.0, 6, 5.0),
        (0.0000009, 6, 0.0),  # below the precision: nothing to buy
        (12.345, 2, 12.34),
        (-1.5, 6, -1.5),
    ],
)
def test_fractional_qty_floors_to_the_given_number_of_decimals(value: float, decimals: int, expected: float) -> None:
    assert fractional_qty(value, decimals) == expected


def test_fractional_qty_defaults_to_six_decimals() -> None:
    assert fractional_qty(2.0000019) == 2.000001


def test_the_cost_of_a_floored_quantity_never_exceeds_the_budget() -> None:
    price, budget = 33.3333, 1000.0

    assert fractional_qty(budget / price) * price <= budget


def test_parse_insufficient_buying_power_reads_the_brokers_real_figure() -> None:
    error = Exception(json.dumps({"buying_power": "132.45", "code": 40310000, "message": "insufficient buying power"}))

    assert parse_insufficient_buying_power(error) == 132.45


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        json.dumps(["a", "list"]),
        json.dumps({"message": "something else", "buying_power": "5"}),
        json.dumps({"message": "insufficient buying power"}),
        json.dumps({"message": "insufficient buying power", "buying_power": "n/a"}),
    ],
)
def test_parse_insufficient_buying_power_is_none_for_any_other_error(body: str) -> None:
    assert parse_insufficient_buying_power(Exception(body)) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_order_helpers.py -q`
Expected: collection error, `ImportError: cannot import name 'fractional_qty' from 'trading_agent_framework.utils.helpers'`.

- [ ] **Step 3: Move the functions**

Run from the repository root (it cuts both functions out of `cross_momentum/utils.py`, appends them to `utils/helpers.py` with their `json`/`math` imports, and leaves a re-export behind):

```bash
uv run python - <<'PY'
"""Move fractional_qty and parse_insufficient_buying_power to utils/helpers.py; cross_momentum.utils re-exports them."""

from pathlib import Path

utils = Path("src/trading_agent_framework/strategies/cross_momentum/utils.py")
text = utils.read_text()
start = text.index("def fractional_qty(")
end = text.index("def load_cross_momentum_universe")
block = text[start:end].rstrip() + "\n"
utils.write_text(text[:start] + text[end:])

helpers = Path("src/trading_agent_framework/utils/helpers.py")
body = helpers.read_text().replace("import logging\nimport os\n", "import json\nimport logging\nimport math\nimport os\n", 1)
helpers.write_text(body.rstrip() + "\n\n\n" + block)

text = utils.read_text()
marker = "if TYPE_CHECKING:"
assert text.count(marker) == 1
reexport = "from trading_agent_framework.utils.helpers import fractional_qty, parse_insufficient_buying_power  # noqa: F401  (moved to utils/helpers.py; re-exported)\n\n"
utils.write_text(text.replace(marker, reexport + marker, 1))
PY
```

- [ ] **Step 4: Run the tests, lint, and the strategies' tests**

Run: `uv run ruff check --fix src tests && uv run ruff check && uv run pytest tests/test_order_helpers.py tests/strategies -q`
Expected: `14` new tests pass along with every existing strategy test, no lint errors. If `ruff` reports an unused `json` or `math` import in `cross_momentum/utils.py`, remove that import (`ruff --fix` does).

- [ ] **Step 5: Run the whole suite, then commit**

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: `1955 passed`.

```bash
git add src/trading_agent_framework/utils/helpers.py src/trading_agent_framework/strategies/cross_momentum/utils.py tests/test_order_helpers.py
git commit -m "refactor: move fractional_qty and parse_insufficient_buying_power to utils/helpers

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `AckmanParams` and the fact sheet

**Files:**
- Create: `src/trading_agent_framework/strategies/bill_ackman/parameters.py`, `src/trading_agent_framework/strategies/bill_ackman/fact_sheet.py`
- Test: `tests/strategies/bill_ackman/test_ackman_params.py`, `tests/strategies/bill_ackman/test_ackman_fact_sheet.py`

**Interfaces:**
- Consumes: `trading_agent_framework.strategies.bill_ackman.screen` (`Candidate`, `ScreenParams`) from Task 1.
- Produces:
  - `AckmanParams` (frozen dataclass): `screen: ScreenParams`, `research_top_n: int = 5`, `max_positions: int = 5`, `max_weight: float = 0.35`, `min_weight: float = 0.05`, `cash_buffer: float = 0.02`, `forced_exit_fails: int = 2`, `rebalance_band: float = 0.05`, `min_trade_pct: float = 0.005`, `parking_symbol: str = "SHV"`, `max_consecutive_abandoned: int = 3`, `reason_max_chars: int = 300`; property `max_total_weight -> float` (`1 - cash_buffer`); `__post_init__` raises `ValueError("AckmanParams: ...")` for out-of-range values.
  - `fact_sheet(candidate: Candidate, *, price: float | None, price_return_12m: float | None) -> dict[str, Any]`, `unavailable_fact_sheet(symbol, *, reason, price, price_return_12m) -> dict[str, Any]`, `price_return(closes: Sequence[float], lookback: int = 252) -> float | None`, constant `TRADING_DAYS_PER_YEAR = 252`.

- [ ] **Step 1: Write the tests**

`tests/strategies/bill_ackman/test_ackman_params.py`:

```python
from __future__ import annotations

import dataclasses
import math

import pytest

from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.screen import ScreenParams


def test_the_defaults_are_the_spec_values() -> None:
    params = AckmanParams()

    assert params.screen == ScreenParams()
    assert params.screen.top_n == 15
    assert (params.research_top_n, params.max_positions) == (5, 5)
    assert (params.max_weight, params.min_weight, params.cash_buffer) == (0.35, 0.05, 0.02)
    assert (params.forced_exit_fails, params.rebalance_band, params.min_trade_pct) == (2, 0.05, 0.005)
    assert (params.parking_symbol, params.max_consecutive_abandoned, params.reason_max_chars) == ("SHV", 3, 300)


def test_the_largest_total_weight_is_one_minus_the_cash_buffer() -> None:
    assert AckmanParams().max_total_weight == pytest.approx(0.98)
    assert AckmanParams(cash_buffer=0.1).max_total_weight == pytest.approx(0.9)


def test_the_params_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        AckmanParams().max_weight = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"research_top_n": 0},
        {"max_positions": 0},
        {"min_weight": 0.0},
        {"min_weight": -0.01},
        {"max_weight": 1.01},
        {"min_weight": 0.4, "max_weight": 0.35, "rebalance_band": 0.05},
        {"min_weight": 0.04, "rebalance_band": 0.05},  # a chosen stock could sit below the band and never be bought
        {"cash_buffer": -0.01},
        {"cash_buffer": 1.0},
        {"forced_exit_fails": 0},
        {"rebalance_band": 0.0},
        {"rebalance_band": 1.0},
        {"min_trade_pct": -0.001},
        {"min_trade_pct": 1.0},
        {"max_consecutive_abandoned": 0},
        {"reason_max_chars": 0},
        {"parking_symbol": " "},
        {"max_weight": math.nan},
        {"cash_buffer": math.inf},
        {"max_positions": 30, "min_weight": 0.05},  # 30 x 5% cannot fit in 98%
    ],
)
def test_an_out_of_range_value_is_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="AckmanParams"):
        AckmanParams(**overrides)  # type: ignore[arg-type]


def test_boundary_values_are_accepted() -> None:
    AckmanParams(min_weight=0.05, max_weight=0.05, rebalance_band=0.05, cash_buffer=0.0, max_positions=20, forced_exit_fails=1)
```

`tests/strategies/bill_ackman/test_ackman_fact_sheet.py`:

```python
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from trading_agent_framework.strategies.bill_ackman.fact_sheet import fact_sheet, price_return, unavailable_fact_sheet
from trading_agent_framework.strategies.bill_ackman.screen import Candidate


def _candidate(**overrides: object) -> Candidate:
    values: dict[str, object] = {
        "symbol": "AAA",
        "rank": 1,
        "score": 0.8,
        "sic": 5812,
        "market_cap": Decimal("123456789012"),
        "fcf_yield": 0.0456789,
        "fcf_margin": 0.1834567,
        "operating_margin": 0.2512345,
        "operating_margin_stdev": 0.0212345,
        "revenue_growth": 0.0812345,
        "net_debt_to_operating_income": 1.23456,
        "debt_reported": True,
        "fiscal_year_end": date(2025, 12, 31),
        "filed": date(2026, 2, 15),
    }
    values.update(overrides)
    return Candidate(**values)  # type: ignore[arg-type]


def test_a_fact_sheet_has_the_agreed_fields_rounded() -> None:
    sheet = fact_sheet(_candidate(), price=123.456, price_return_12m=0.123456)

    assert sheet == {
        "symbol": "AAA",
        "sic": 5812,
        "market_cap_usd_bn": 123.46,
        "fcf_yield": 0.0457,
        "fcf_margin_5y": 0.1835,
        "operating_margin": 0.2512,
        "operating_margin_stdev": 0.0212,
        "revenue_cagr_5y": 0.0812,
        "net_debt_to_operating_income": 1.23,
        "fiscal_year_end": "2025-12-31",
        "filed": "2026-02-15",
        "price": 123.456,
        "price_return_12m": 0.1235,
    }


def test_a_missing_debt_figure_is_flagged_and_a_reported_one_is_not() -> None:
    assert fact_sheet(_candidate(debt_reported=False), price=1.0, price_return_12m=None)["debt_reported"] is False
    assert "debt_reported" not in fact_sheet(_candidate(), price=1.0, price_return_12m=None)


def test_a_price_fact_that_cannot_be_computed_is_none() -> None:
    sheet = fact_sheet(_candidate(), price=None, price_return_12m=None)

    assert sheet["price"] is None
    assert sheet["price_return_12m"] is None


def test_a_missing_sic_code_is_none() -> None:
    assert fact_sheet(_candidate(sic=None), price=1.0, price_return_12m=None)["sic"] is None


def test_a_holding_the_screen_could_not_describe_carries_only_price_facts_and_the_reason() -> None:
    sheet = unavailable_fact_sheet("hlt", reason="no_data", price=150.5, price_return_12m=-0.05)

    assert sheet == {"symbol": "HLT", "screen_unavailable": "no_data", "price": 150.5, "price_return_12m": -0.05}


def test_price_return_compares_the_last_close_with_the_close_a_year_earlier() -> None:
    closes = [100.0] + [110.0] * 251 + [120.0]  # 253 closes: 252 bars back is the first one

    assert price_return(closes) == pytest.approx(0.2)


@pytest.mark.parametrize("closes", [[], [100.0] * 252])
def test_price_return_is_none_without_enough_history(closes: list[float]) -> None:
    assert price_return(closes) is None


def test_price_return_is_none_when_the_starting_close_is_not_positive() -> None:
    assert price_return([0.0] + [10.0] * 252) is None


def test_price_return_takes_a_shorter_lookback() -> None:
    assert price_return([10.0, 11.0, 12.0, 15.0], lookback=3) == pytest.approx(0.5)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_params.py tests/strategies/bill_ackman/test_ackman_fact_sheet.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bill_ackman.parameters'` (and `...fact_sheet`).

- [ ] **Step 3: Implement `parameters.py`**

```python
"""`AckmanParams`: every threshold of the Bill Ackman strategy in one frozen dataclass."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from trading_agent_framework.strategies.bill_ackman.screen import ScreenParams


@dataclass(frozen=True, slots=True)
class AckmanParams:
    screen: ScreenParams = field(default_factory=ScreenParams)
    research_top_n: int = 5  # ideas the researcher submits
    max_positions: int = 5  # stocks in the target portfolio
    max_weight: float = 0.35  # largest weight of one stock, as a fraction of portfolio value
    min_weight: float = 0.05  # smallest weight of one stock; must be at least `rebalance_band`
    cash_buffer: float = 0.02  # share of portfolio value never invested (fees, fill drift)
    forced_exit_fails: int = 2  # consecutive `fail` verdicts that force a holding out
    rebalance_band: float = 0.05  # drift, as a fraction of portfolio value, that triggers a trade
    min_trade_pct: float = 0.005  # smallest order, as a fraction of portfolio value
    parking_symbol: str = "SHV"  # where unallocated money goes
    max_consecutive_abandoned: int = 3  # abandoned reviews in a row that abort a backtest
    reason_max_chars: int = 300  # longest `reason` a submit tool accepts

    def __post_init__(self) -> None:
        floats = (self.max_weight, self.min_weight, self.cash_buffer, self.rebalance_band, self.min_trade_pct)
        problems = {
            "research_top_n must be at least 1": self.research_top_n < 1,
            "max_positions must be at least 1": self.max_positions < 1,
            "weights, cash_buffer, rebalance_band and min_trade_pct must be finite": not all(math.isfinite(value) for value in floats),
            "min_weight must be above 0 and at most max_weight, which is at most 1": not 0 < self.min_weight <= self.max_weight <= 1,
            "min_weight must be at least rebalance_band (a chosen stock below the band would never be bought)": self.min_weight < self.rebalance_band,
            "cash_buffer must be in [0, 1)": not 0 <= self.cash_buffer < 1,
            "rebalance_band must be in (0, 1)": not 0 < self.rebalance_band < 1,
            "min_trade_pct must be in [0, 1)": not 0 <= self.min_trade_pct < 1,
            "forced_exit_fails must be at least 1": self.forced_exit_fails < 1,
            "max_consecutive_abandoned must be at least 1": self.max_consecutive_abandoned < 1,
            "reason_max_chars must be at least 1": self.reason_max_chars < 1,
            "parking_symbol must not be blank": not self.parking_symbol.strip(),
            "max_positions x min_weight must fit in 1 - cash_buffer": self.max_positions * self.min_weight > 1 - self.cash_buffer + 1e-9,
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ValueError("AckmanParams: " + "; ".join(failed))

    @property
    def max_total_weight(self) -> float:
        """The largest sum of stock weights: everything but the cash buffer."""
        return 1 - self.cash_buffer
```

- [ ] **Step 4: Implement `fact_sheet.py`**

```python
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
```

- [ ] **Step 5: Run the tests and lint**

Run: `uv run ruff check src tests && uv run pytest tests/strategies/bill_ackman/test_ackman_params.py tests/strategies/bill_ackman/test_ackman_fact_sheet.py -q`
Expected: `24 passed` and `10 passed` (34 tests), no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/bill_ackman/parameters.py src/trading_agent_framework/strategies/bill_ackman/fact_sheet.py tests/strategies/bill_ackman/test_ackman_params.py tests/strategies/bill_ackman/test_ackman_fact_sheet.py
git commit -m "feat: AckmanParams and the per-company fact sheet

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Hysteresis and the target portfolio

**Files:**
- Create: `src/trading_agent_framework/strategies/bill_ackman/hysteresis.py`, `src/trading_agent_framework/strategies/bill_ackman/portfolio.py`
- Test: `tests/strategies/bill_ackman/test_ackman_hysteresis.py`, `tests/strategies/bill_ackman/test_ackman_portfolio.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (both modules are pure and import only the standard library).
- Produces:
  - `hysteresis.apply_verdicts(*, holdings: Sequence[str], ranking: Sequence[str], verdicts: Mapping[str, str], fail_counts: Mapping[str, int], forced_exit_fails: int) -> HysteresisOutcome`; `HysteresisOutcome(fail_counts: dict[str, int], forced_exits: list[str], pending: list[str], allowed: list[str])`; constants `SURVIVE = "survive"`, `FAIL = "fail"`. Raises `ValueError` for a holding or ranked symbol without a verdict, or a verdict other than `survive`/`fail`.
  - `portfolio.target_portfolio(weights: Mapping[str, float], *, cash_buffer: float) -> TargetPortfolio`; `TargetPortfolio(weights: dict[str, float], parking_weight: float)`. Raises `ValueError` if the weights sum above `1 - cash_buffer`.

- [ ] **Step 1: Write the tests**

`tests/strategies/bill_ackman/test_ackman_hysteresis.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bill_ackman.hysteresis import apply_verdicts


def _apply(*, holdings=(), ranking=(), verdicts, fail_counts=None, forced_exit_fails: int = 2):
    return apply_verdicts(holdings=list(holdings), ranking=list(ranking), verdicts=verdicts, fail_counts=fail_counts or {}, forced_exit_fails=forced_exit_fails)


def test_a_surviving_holding_resets_its_counter() -> None:
    outcome = _apply(holdings=["HLT"], verdicts={"HLT": "survive"}, fail_counts={"HLT": 1})

    assert outcome.fail_counts == {}
    assert outcome.forced_exits == [] and outcome.pending == []
    assert outcome.allowed == ["HLT"]


def test_a_first_fail_is_pending_and_the_holding_stays_allowed() -> None:
    outcome = _apply(holdings=["HLT"], verdicts={"HLT": "fail"})

    assert outcome.fail_counts == {"HLT": 1}
    assert outcome.pending == ["HLT"]
    assert outcome.forced_exits == []
    assert outcome.allowed == ["HLT"]


def test_a_second_consecutive_fail_forces_the_exit_and_removes_it_from_the_allowed_set() -> None:
    outcome = _apply(holdings=["HLT"], verdicts={"HLT": "fail"}, fail_counts={"HLT": 1})

    assert outcome.forced_exits == ["HLT"]
    assert outcome.pending == []
    assert outcome.allowed == []
    assert outcome.fail_counts == {"HLT": 2}  # kept: if the sell fails the holding is still forced out next time


def test_the_forced_exit_threshold_is_a_parameter() -> None:
    assert _apply(holdings=["HLT"], verdicts={"HLT": "fail"}, forced_exit_fails=1).forced_exits == ["HLT"]
    assert _apply(holdings=["HLT"], verdicts={"HLT": "fail"}, fail_counts={"HLT": 2}, forced_exit_fails=4).pending == ["HLT"]


def test_a_new_candidate_that_fails_is_not_allowed_and_has_no_counter() -> None:
    outcome = _apply(ranking=["NEW"], verdicts={"NEW": "fail"})

    assert outcome.allowed == []
    assert outcome.fail_counts == {}


def test_counters_of_symbols_no_longer_held_are_dropped() -> None:
    outcome = _apply(holdings=["KO"], verdicts={"KO": "survive"}, fail_counts={"GONE": 1, "KO": 1})

    assert outcome.fail_counts == {}


def test_the_allowed_set_lists_ranked_survivors_first_then_other_survivors_then_pending_fails() -> None:
    outcome = _apply(
        holdings=["OLDSURVIVOR", "RANKEDPENDING", "OLDPENDING"],
        ranking=["B", "RANKEDPENDING", "A", "REJECTED"],
        verdicts={
            "A": "survive",
            "B": "survive",
            "REJECTED": "fail",
            "RANKEDPENDING": "fail",
            "OLDSURVIVOR": "survive",
            "OLDPENDING": "fail",
        },
    )

    assert outcome.allowed == ["B", "A", "OLDSURVIVOR", "RANKEDPENDING", "OLDPENDING"]
    assert outcome.pending == ["RANKEDPENDING", "OLDPENDING"]


def test_a_ranked_holding_that_survives_appears_once() -> None:
    outcome = _apply(holdings=["A"], ranking=["A", "B"], verdicts={"A": "survive", "B": "survive"})

    assert outcome.allowed == ["A", "B"]


def test_a_holding_without_a_verdict_is_a_pipeline_bug() -> None:
    with pytest.raises(ValueError, match="HLT"):
        _apply(holdings=["HLT"], verdicts={})


def test_a_ranked_symbol_without_a_verdict_is_a_pipeline_bug() -> None:
    with pytest.raises(ValueError, match="NEW"):
        _apply(ranking=["NEW"], verdicts={})


def test_an_unknown_verdict_value_is_refused() -> None:
    with pytest.raises(ValueError, match="maybe"):
        _apply(holdings=["HLT"], verdicts={"HLT": "maybe"})


def test_several_holdings_are_handled_independently() -> None:
    outcome = _apply(
        holdings=["A", "B", "C"],
        verdicts={"A": "survive", "B": "fail", "C": "fail"},
        fail_counts={"B": 1, "C": 0},
    )

    assert outcome.forced_exits == ["B"]
    assert outcome.pending == ["C"]
    assert outcome.fail_counts == {"B": 2, "C": 1}
    assert outcome.allowed == ["A", "C"]
```

`tests/strategies/bill_ackman/test_ackman_portfolio.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bill_ackman.portfolio import target_portfolio


def test_the_remainder_after_the_stocks_and_the_cash_buffer_goes_to_parking() -> None:
    target = target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02)

    assert target.weights == {"A": 0.35, "B": 0.25}
    assert target.parking_weight == pytest.approx(0.38)


def test_an_empty_portfolio_parks_everything_but_the_cash_buffer() -> None:
    target = target_portfolio({}, cash_buffer=0.02)

    assert target.weights == {}
    assert target.parking_weight == pytest.approx(0.98)


def test_a_fully_invested_portfolio_parks_nothing() -> None:
    target = target_portfolio({"A": 0.35, "B": 0.35, "C": 0.28}, cash_buffer=0.02)

    assert target.parking_weight == pytest.approx(0.0, abs=1e-9)


def test_weights_above_the_investable_share_are_refused() -> None:
    with pytest.raises(ValueError, match="0.98"):
        target_portfolio({"A": 0.35, "B": 0.35, "C": 0.3}, cash_buffer=0.02)


def test_the_input_is_copied() -> None:
    weights = {"A": 0.3}
    target = target_portfolio(weights, cash_buffer=0.02)
    weights["B"] = 0.3

    assert target.weights == {"A": 0.3}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_hysteresis.py tests/strategies/bill_ackman/test_ackman_portfolio.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bill_ackman.hysteresis'` (and `...portfolio`).

- [ ] **Step 3: Implement `hysteresis.py`**

```python
"""Fail counters, forced exits and the trader's allowed set (pure).

A holding that gets the verdict `fail` has its counter raised; at `forced_exit_fails` consecutive fails it is a
forced exit (code sells it whatever the trader submits). A `survive` resets the counter. A new candidate that
fails is simply not allowed and has no counter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SURVIVE = "survive"
FAIL = "fail"


@dataclass(frozen=True, slots=True)
class HysteresisOutcome:
    fail_counts: dict[str, int]  # after this review, for holdings that have failed and not recovered (forced exits included)
    forced_exits: list[str]  # holdings to sell in full, in `holdings` order
    pending: list[str]  # holdings that failed but are below the threshold, in `holdings` order
    allowed: list[str]  # what the trader may choose from: ranked survivors, other surviving holdings, pending fails


def apply_verdicts(
    *,
    holdings: Sequence[str],
    ranking: Sequence[str],
    verdicts: Mapping[str, str],
    fail_counts: Mapping[str, int],
    forced_exit_fails: int,
) -> HysteresisOutcome:
    """Update the counters from today's `verdicts` and derive the forced exits and the allowed set.

    `verdicts` must hold a verdict for every holding and every ranked symbol (a missing one is a pipeline bug and
    raises `ValueError`, as does a verdict other than `survive`/`fail`). `fail_counts` entries for symbols that
    are no longer held are dropped.
    """
    for symbol in [*holdings, *ranking]:
        verdict = verdicts.get(symbol)
        if verdict is None:
            raise ValueError(f"no verdict for {symbol}")
        if verdict not in (SURVIVE, FAIL):
            raise ValueError(f"verdict for {symbol} must be 'survive' or 'fail', got {verdict!r}")

    counts: dict[str, int] = {}
    forced_exits: list[str] = []
    pending: list[str] = []
    for symbol in holdings:
        if verdicts[symbol] == SURVIVE:
            continue
        count = fail_counts.get(symbol, 0) + 1
        counts[symbol] = count
        (forced_exits if count >= forced_exit_fails else pending).append(symbol)

    allowed: list[str] = []
    for symbol in ranking:
        if verdicts[symbol] == SURVIVE and symbol not in allowed:
            allowed.append(symbol)
    for symbol in holdings:
        if verdicts[symbol] == SURVIVE and symbol not in allowed:
            allowed.append(symbol)
    allowed.extend(symbol for symbol in pending if symbol not in allowed)
    return HysteresisOutcome(fail_counts=counts, forced_exits=forced_exits, pending=pending, allowed=allowed)
```

- [ ] **Step 4: Implement `portfolio.py`**

```python
"""The trader's weights to a target portfolio (pure): each stock's weight, and the remainder parked in SHV."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    weights: dict[str, float]  # stock -> fraction of portfolio value
    parking_weight: float  # the parking instrument's share: 1 - cash_buffer - sum(weights)


def target_portfolio(weights: Mapping[str, float], *, cash_buffer: float) -> TargetPortfolio:
    """Nothing is rescaled: a weight the trader chose is the weight targeted. The cash buffer stays as cash.

    Raises `ValueError` when the weights exceed `1 - cash_buffer` (the submit tool already refuses that; this
    keeps the invariant if another caller builds a target).
    """
    total = sum(weights.values())
    investable = 1 - cash_buffer
    if total > investable + _EPS:
        raise ValueError(f"stock weights sum to {total:.4f}, above the investable {investable:.4f}")
    return TargetPortfolio(weights=dict(weights), parking_weight=max(0.0, investable - total))
```

- [ ] **Step 5: Run the tests and lint**

Run: `uv run ruff check src tests && uv run pytest tests/strategies/bill_ackman/test_ackman_hysteresis.py tests/strategies/bill_ackman/test_ackman_portfolio.py -q`
Expected: `12 passed` and `5 passed`, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/bill_ackman/hysteresis.py src/trading_agent_framework/strategies/bill_ackman/portfolio.py tests/strategies/bill_ackman/test_ackman_hysteresis.py tests/strategies/bill_ackman/test_ackman_portfolio.py
git commit -m "feat: fail-counter hysteresis and the target portfolio

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The hand-off: validation, recorder and submit tools

**Files:**
- Create: `src/trading_agent_framework/strategies/bill_ackman/handoff.py`
- Test: `tests/strategies/bill_ackman/test_ackman_handoff.py`

**Interfaces:**
- Consumes: `AckmanParams` from Task 5.
- Produces (all in `handoff.py`):
  - Dataclasses `Idea(symbol, reason)`, `Verdict(symbol, verdict, reason)`, `PortfolioPosition(symbol, weight, reason)`; `HandoffError(ValueError)`; constants `SURVIVE`, `FAIL`, `VERDICTS`.
  - Pure validators `validate_ranking(raw, *, candidates, top_n, reason_max_chars) -> list[Idea]`, `validate_verdicts(raw, *, expected, reason_max_chars) -> list[Verdict]`, `validate_portfolio(raw, *, allowed, max_positions, min_weight, max_weight, max_total_weight, reason_max_chars) -> list[PortfolioPosition]`, all raising `HandoffError` with a message that says what is wrong and what is allowed.
  - `HandoffRecorder(params: AckmanParams)` with `expect_ranking(candidates)`, `expect_verdicts(symbols)`, `expect_portfolio(allowed)` (each arms a stage and clears the previous submission and error), properties `submitted: bool`, `submission` (the armed stage's valid list, else `None`), attribute `last_error: str | None`, and `submit(stage, raw) -> {"status": "recorded"} | {"error": str}`.
  - `submit_tools(recorder) -> dict[str, Callable]` with keys `submit_ranking(ideas)`, `submit_verdicts(verdicts)`, `submit_portfolio(positions)`; each function's `__name__` equals its key and it has a one-line docstring.

- [ ] **Step 1: Write the tests**

`tests/strategies/bill_ackman/test_ackman_handoff.py`:

```python
from __future__ import annotations

import inspect
import typing
from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from trading_agent_framework.strategies.bill_ackman.handoff import (
    HandoffError,
    HandoffRecorder,
    Idea,
    PortfolioPosition,
    Verdict,
    submit_tools,
    validate_portfolio,
    validate_ranking,
    validate_verdicts,
)
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams

CANDIDATES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]


def _idea(symbol: str, reason: str = "cash rich") -> dict[str, Any]:
    return {"symbol": symbol, "reason": reason}


def _ranking(raw: Any, *, candidates: list[str] = CANDIDATES, top_n: int = 5, reason_max_chars: int = 300) -> list[Idea]:
    return validate_ranking(raw, candidates=candidates, top_n=top_n, reason_max_chars=reason_max_chars)


def _portfolio(raw: Any, *, allowed: list[str] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")) -> list[PortfolioPosition]:  # type: ignore[assignment]
    return validate_portfolio(raw, allowed=allowed, max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300)


def _position(symbol: str, weight: Any, reason: str = "best idea") -> dict[str, Any]:
    return {"symbol": symbol, "weight": weight, "reason": reason}


# --- ranking ------------------------------------------------------------------------------------


def test_a_valid_ranking_keeps_the_order_and_normalises_symbols() -> None:
    ideas = _ranking([_idea(" bbb ", "  growing "), _idea("AAA")])

    assert ideas == [Idea("BBB", "growing"), Idea("AAA", "cash rich")]


def test_a_ranking_may_hold_the_top_n_ideas_and_no_more() -> None:
    assert len(_ranking([_idea(s) for s in CANDIDATES[:5]])) == 5
    with pytest.raises(HandoffError, match="at most 5"):
        _ranking([_idea(s) for s in CANDIDATES[:6]])


def test_the_ranking_limit_is_the_number_of_candidates_when_there_are_fewer() -> None:
    assert len(_ranking([_idea("AAA"), _idea("BBB")], candidates=["AAA", "BBB"])) == 2
    with pytest.raises(HandoffError, match="at most 2"):
        _ranking([_idea("AAA"), _idea("BBB"), _idea("CCC")], candidates=["AAA", "BBB"])


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "empty"),
        ("AAA", "list of objects"),
        (None, "list of objects"),
        (["AAA"], "must be an object"),
        ([{"reason": "x"}], "no symbol"),
        ([{"symbol": "  ", "reason": "x"}], "no symbol"),
        ([{"symbol": 5, "reason": "x"}], "no symbol"),
        ([{"symbol": "ZZZ", "reason": "x"}], "not one of the candidates"),
        ([_idea("AAA"), _idea("aaa")], "appears twice"),
        ([{"symbol": "AAA"}], "no reason"),
        ([{"symbol": "AAA", "reason": "   "}], "no reason"),
        ([{"symbol": "AAA", "reason": "x" * 301}], "under 300"),
    ],
)
def test_an_invalid_ranking_is_refused_with_a_message_that_says_why(raw: Any, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        _ranking(raw)


def test_a_reason_of_exactly_the_maximum_length_is_accepted() -> None:
    assert _ranking([_idea("AAA", "x" * 300)])[0].reason == "x" * 300


def test_the_not_a_candidate_message_lists_the_candidates() -> None:
    with pytest.raises(HandoffError, match="AAA, BBB"):
        _ranking([_idea("ZZZ")], candidates=["AAA", "BBB"])


# --- verdicts -----------------------------------------------------------------------------------


def _verdict(symbol: str, verdict: Any = "survive", reason: str = "debt is fine") -> dict[str, Any]:
    return {"symbol": symbol, "verdict": verdict, "reason": reason}


def test_valid_verdicts_cover_exactly_the_asked_symbols() -> None:
    verdicts = validate_verdicts([_verdict("aaa", " FAIL "), _verdict("BBB")], expected=["AAA", "BBB"], reason_max_chars=300)

    assert verdicts == [Verdict("AAA", "fail", "debt is fine"), Verdict("BBB", "survive", "debt is fine")]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([_verdict("AAA")], "no verdict for BBB"),
        ([_verdict("AAA"), _verdict("BBB"), _verdict("CCC")], "CCC was not asked about"),
        ([_verdict("AAA"), _verdict("AAA"), _verdict("BBB")], "appears twice"),
        ([_verdict("AAA", "maybe"), _verdict("BBB")], "'survive' or 'fail'"),
        ([_verdict("AAA", None), _verdict("BBB")], "'survive' or 'fail'"),
        ([_verdict("AAA", reason=""), _verdict("BBB")], "no reason"),
        ([], "no verdict for AAA, BBB"),
        ("nope", "list of objects"),
    ],
)
def test_invalid_verdicts_are_refused(raw: Any, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        validate_verdicts(raw, expected=["AAA", "BBB"], reason_max_chars=300)


# --- portfolio ----------------------------------------------------------------------------------


def test_a_valid_portfolio_is_returned_with_float_weights() -> None:
    positions = _portfolio([_position("aaa", 0.35), _position("BBB", "0.25"), _position("CCC", 0.3)])

    assert positions == [PortfolioPosition("AAA", 0.35, "best idea"), PortfolioPosition("BBB", 0.25, "best idea"), PortfolioPosition("CCC", 0.3, "best idea")]


def test_an_empty_portfolio_is_valid() -> None:
    assert _portfolio([]) == []
    assert _portfolio([], allowed=[]) == []


def test_the_weight_and_total_boundaries_are_inclusive() -> None:
    assert len(_portfolio([_position("AAA", 0.35), _position("BBB", 0.35), _position("CCC", 0.28)])) == 3  # total exactly 0.98
    assert _portfolio([_position("AAA", 0.05)])[0].weight == 0.05
    assert _portfolio([_position("AAA", 0.35)])[0].weight == 0.35


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([_position("AAA", 0.04)], "between 0.05 and 0.35"),
        ([_position("AAA", 0.36)], "between 0.05 and 0.35"),
        ([_position("AAA", -0.1)], "between 0.05 and 0.35"),
        ([_position("AAA", 0.35), _position("BBB", 0.35), _position("CCC", 0.29)], "at most 0.98"),
        ([_position(s, 0.1) for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")], "at most 5 positions"),
        ([_position("ZZZ", 0.2)], "not allowed"),
        ([_position("AAA", 0.2), _position("AAA", 0.2)], "appears twice"),
        ([_position("AAA", None)], "must be a number"),
        ([_position("AAA", "lots")], "must be a number"),
        ([_position("AAA", True)], "must be a number"),
        ([_position("AAA", float("nan"))], "finite"),
        ([_position("AAA", float("inf"))], "finite"),
        ([_position("AAA", 10**400)], "must be a number"),
        ([{"symbol": "AAA", "weight": 0.2}], "no reason"),
        ("AAA", "list of objects"),
    ],
)
def test_an_invalid_portfolio_is_refused(raw: Any, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        _portfolio(raw)


def test_with_nothing_allowed_the_refusal_says_to_submit_an_empty_list() -> None:
    with pytest.raises(HandoffError, match="empty list"):
        _portfolio([_position("AAA", 0.2)], allowed=[])


# --- the recorder and the tools -------------------------------------------------------------------


def _recorder() -> HandoffRecorder:
    return HandoffRecorder(AckmanParams())


def test_a_valid_submission_is_recorded_and_exposed() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)

    result = submit_tools(recorder)["submit_ranking"]([_idea("AAA")])

    assert result == {"status": "recorded"}
    assert recorder.submitted
    assert recorder.submission == [Idea("AAA", "cash rich")]
    assert recorder.last_error is None


def test_an_invalid_submission_returns_the_error_and_records_nothing() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)

    result = submit_tools(recorder)["submit_ranking"]([_idea("ZZZ")])

    assert "not one of the candidates" in result["error"]
    assert not recorder.submitted
    assert recorder.last_error == result["error"]


def test_the_model_can_correct_an_invalid_submission_in_the_same_run() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)
    ranking = submit_tools(recorder)["submit_ranking"]

    assert "error" in ranking([_idea("ZZZ")])
    assert ranking([_idea("AAA")]) == {"status": "recorded"}
    assert recorder.last_error is None


def test_the_first_valid_submission_is_final() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)
    ranking = submit_tools(recorder)["submit_ranking"]
    ranking([_idea("AAA")])

    second = ranking([_idea("BBB")])

    assert second == {"error": "already recorded for this review"}
    assert recorder.submission == [Idea("AAA", "cash rich")]


def test_an_empty_portfolio_counts_as_a_submission() -> None:
    recorder = _recorder()
    recorder.expect_portfolio([])

    assert submit_tools(recorder)["submit_portfolio"]([]) == {"status": "recorded"}
    assert recorder.submitted
    assert recorder.submission == []


def test_a_tool_called_out_of_turn_is_refused() -> None:
    recorder = _recorder()
    tools = submit_tools(recorder)

    assert "not expected" in tools["submit_ranking"]([_idea("AAA")])["error"]  # nothing armed yet
    recorder.expect_verdicts(["AAA"])
    assert "not expected" in tools["submit_portfolio"]([])["error"]  # the verdicts stage is armed, not the portfolio
    assert not recorder.submitted


def test_arming_a_stage_clears_the_previous_submission_and_error() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)
    tools = submit_tools(recorder)
    tools["submit_ranking"]([_idea("ZZZ")])
    tools["submit_ranking"]([_idea("AAA")])

    recorder.expect_verdicts(["AAA"])

    assert not recorder.submitted
    assert recorder.submission is None
    assert recorder.last_error is None


def test_each_stage_validates_with_the_recorders_parameters() -> None:
    recorder = HandoffRecorder(AckmanParams(max_weight=0.2, min_weight=0.05))
    recorder.expect_portfolio(["AAA"])

    result = submit_tools(recorder)["submit_portfolio"]([_position("AAA", 0.25)])

    assert "between 0.05 and 0.2" in result["error"]


def test_an_overflowing_integer_is_refused_with_a_correctable_error() -> None:
    recorder = _recorder()
    recorder.expect_portfolio(["AAA"])
    tools = submit_tools(recorder)

    result = tools["submit_portfolio"]([{"symbol": "AAA", "weight": 10**400, "reason": "x"}])

    assert "error" in result
    assert "must be a number" in result["error"]
    assert not recorder.submitted
    assert recorder.last_error is not None

    # The model can correct it in the same run
    corrected = tools["submit_portfolio"]([_position("AAA", 0.3)])

    assert corrected == {"status": "recorded"}
    assert recorder.submitted


def test_the_tools_are_named_after_their_keys_and_have_one_line_docstrings() -> None:
    tools = submit_tools(_recorder())

    assert set(tools) == {"submit_ranking", "submit_verdicts", "submit_portfolio"}
    for name, tool in tools.items():
        assert tool.__name__ == name
        assert tool.__doc__ is not None and len(tool.__doc__.strip().splitlines()) == 1


def test_the_tools_have_real_annotations_the_agent_layer_can_read() -> None:
    # Regression guard for the no-`from __future__ import annotations` rule of this module.
    for tool in submit_tools(_recorder()).values():
        for parameter in inspect.signature(tool).parameters.values():
            assert not isinstance(parameter.annotation, str)
        assert typing.get_type_hints(tool)["return"] == dict[str, Any]


@pytest.mark.parametrize(("name", "argument"), [("submit_ranking", "ideas"), ("submit_verdicts", "verdicts"), ("submit_portfolio", "positions")])
def test_langchain_builds_an_array_schema_from_each_tool(name: str, argument: str) -> None:
    tool = StructuredTool.from_function(submit_tools(_recorder())[name])

    assert list(tool.args) == [argument]
    assert tool.args[argument]["type"] == "array"
    assert tool.name == name
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_handoff.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bill_ackman.handoff'`.

- [ ] **Step 3: Implement `handoff.py`**

Keep the module free of `from __future__ import annotations` (the agent layer reads real annotations; one test pins it).

```python
"""The hand-off between the three agents: submit tools, their validation and the recorder.

Each agent ends its run by calling one submit tool. A tool validates its argument against what the pipeline
armed the recorder with for this stage, records the first valid submission, and otherwise returns
`{"error": ...}` so the model can correct itself in the same run. Free text from an agent is never trusted.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's
schema from the function's real annotations (as `memory/tools.py` does).
"""

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams

SURVIVE = "survive"
FAIL = "fail"
VERDICTS = (SURVIVE, FAIL)
_EPS = 1e-9

RANKING, VERDICTS_STAGE, PORTFOLIO = "ranking", "verdicts", "portfolio"
_TOOL_NAMES = {RANKING: "submit_ranking", VERDICTS_STAGE: "submit_verdicts", PORTFOLIO: "submit_portfolio"}


class HandoffError(ValueError):
    """A submission that breaks a rule; the message tells the model what is wrong and what is allowed."""


@dataclass(frozen=True, slots=True)
class Idea:
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class Verdict:
    symbol: str
    verdict: str
    reason: str


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    symbol: str
    weight: float
    reason: str


# --- validation (pure) --------------------------------------------------------------------------


def _items(raw: Any, what: str) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list | tuple):
        raise HandoffError(f"{what} must be a list of objects")
    items = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise HandoffError(f"{what} item {index} must be an object, got {type(item).__name__}")
        items.append(item)
    return items


def _symbol(item: Mapping[str, Any], index: int) -> str:
    value = item.get("symbol")
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"item {index} has no symbol")
    return value.strip().upper()


def _reason(item: Mapping[str, Any], symbol: str, max_chars: int) -> str:
    value = item.get("reason")
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"{symbol} has no reason: give one short sentence")
    text = value.strip()
    if len(text) > max_chars:
        raise HandoffError(f"the reason for {symbol} is {len(text)} characters: keep it under {max_chars}")
    return text


def _weight(item: Mapping[str, Any], symbol: str) -> float:
    value = item.get("weight")
    if isinstance(value, bool):
        raise HandoffError(f"the weight for {symbol} must be a number")
    try:
        weight = float(value)  # type: ignore[arg-type]  # an int, a float or a numeric string
    except (TypeError, ValueError, OverflowError):
        raise HandoffError(f"the weight for {symbol} must be a number, a fraction of portfolio value such as 0.25") from None
    if not math.isfinite(weight):
        raise HandoffError(f"the weight for {symbol} must be a finite number")
    return weight


def validate_ranking(raw: Any, *, candidates: Sequence[str], top_n: int, reason_max_chars: int) -> list[Idea]:
    """1 to `top_n` unique ideas (or all candidates if fewer), each from `candidates`, best first."""
    items = _items(raw, "ideas")
    limit = min(top_n, len(candidates))
    if not items:
        raise HandoffError("ideas is empty: submit at least one idea chosen from the candidates")
    if len(items) > limit:
        raise HandoffError(f"submit at most {limit} ideas, got {len(items)}")
    allowed = set(candidates)
    seen: set[str] = set()
    ideas = []
    for index, item in enumerate(items, start=1):
        symbol = _symbol(item, index)
        if symbol not in allowed:
            raise HandoffError(f"{symbol} is not one of the candidates ({', '.join(candidates)})")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        ideas.append(Idea(symbol, _reason(item, symbol, reason_max_chars)))
    return ideas


def validate_verdicts(raw: Any, *, expected: Collection[str], reason_max_chars: int) -> list[Verdict]:
    """Exactly one verdict (`survive` or `fail`) per symbol in `expected`, and no other symbol."""
    items = _items(raw, "verdicts")
    wanted = set(expected)
    seen: set[str] = set()
    verdicts = []
    for index, item in enumerate(items, start=1):
        symbol = _symbol(item, index)
        if symbol not in wanted:
            raise HandoffError(f"{symbol} was not asked about: judge only {', '.join(sorted(wanted))}")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        verdict = item.get("verdict")
        verdict = verdict.strip().lower() if isinstance(verdict, str) else verdict
        if verdict not in VERDICTS:
            raise HandoffError(f"the verdict for {symbol} must be 'survive' or 'fail'")
        verdicts.append(Verdict(symbol, verdict, _reason(item, symbol, reason_max_chars)))
    missing = sorted(wanted - seen)
    if missing:
        raise HandoffError(f"no verdict for {', '.join(missing)}: give one verdict per symbol")
    return verdicts


def validate_portfolio(
    raw: Any,
    *,
    allowed: Collection[str],
    max_positions: int,
    min_weight: float,
    max_weight: float,
    max_total_weight: float,
    reason_max_chars: int,
) -> list[PortfolioPosition]:
    """At most `max_positions` unique stocks from `allowed`, each weight in [min, max], their sum at most `max_total_weight`.

    An empty list is valid: it means everything goes to the parking instrument.
    """
    items = _items(raw, "positions")
    if len(items) > max_positions:
        raise HandoffError(f"submit at most {max_positions} positions, got {len(items)}")
    allowed_set = set(allowed)
    seen: set[str] = set()
    positions = []
    for index, item in enumerate(items, start=1):
        symbol = _symbol(item, index)
        if symbol not in allowed_set:
            raise HandoffError(f"{symbol} is not allowed (choose from {', '.join(sorted(allowed_set)) or 'nothing: submit an empty list'})")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        weight = _weight(item, symbol)
        if not min_weight - _EPS <= weight <= max_weight + _EPS:
            raise HandoffError(f"the weight for {symbol} is {weight:.4f}: it must be between {min_weight} and {max_weight}")
        positions.append(PortfolioPosition(symbol, weight, _reason(item, symbol, reason_max_chars)))
    total = sum(position.weight for position in positions)
    if total > max_total_weight + _EPS:
        raise HandoffError(f"the weights sum to {total:.4f}: they must sum to at most {max_total_weight:.2f}")
    return positions


# --- the recorder ---------------------------------------------------------------------------------


class HandoffRecorder:
    """Holds what the current stage must validate against, and the first valid submission for it."""

    def __init__(self, params: AckmanParams) -> None:
        self._params = params
        self._stage: str | None = None
        self._context: dict[str, Any] = {}
        self._submission: Any = None
        self._submitted = False
        self.last_error: str | None = None

    def expect_ranking(self, candidates: Sequence[str]) -> None:
        self._arm(RANKING, {"candidates": list(candidates)})

    def expect_verdicts(self, symbols: Sequence[str]) -> None:
        self._arm(VERDICTS_STAGE, {"expected": list(symbols)})

    def expect_portfolio(self, allowed: Sequence[str]) -> None:
        self._arm(PORTFOLIO, {"allowed": list(allowed)})

    def _arm(self, stage: str, context: dict[str, Any]) -> None:
        self._stage, self._context = stage, context
        self._submission, self._submitted, self.last_error = None, False, None

    @property
    def submitted(self) -> bool:
        """Whether the armed stage has a valid submission."""
        return self._submitted

    @property
    def submission(self) -> Any:
        """The armed stage's valid submission (a list of `Idea`, `Verdict` or `PortfolioPosition`), else None."""
        return self._submission

    def submit(self, stage: str, raw: Any) -> dict[str, Any]:
        """What a submit tool returns: `{"status": "recorded"}` or `{"error": ...}` (recorded in `last_error`)."""
        if self._stage != stage:
            return self._fail(f"{_TOOL_NAMES[stage]} is not expected at this point of the review")
        if self._submitted:
            return self._fail("already recorded for this review")
        params = self._params
        try:
            if stage == RANKING:
                value = validate_ranking(raw, candidates=self._context["candidates"], top_n=params.research_top_n, reason_max_chars=params.reason_max_chars)
            elif stage == VERDICTS_STAGE:
                value = validate_verdicts(raw, expected=self._context["expected"], reason_max_chars=params.reason_max_chars)
            else:
                value = validate_portfolio(
                    raw,
                    allowed=self._context["allowed"],
                    max_positions=params.max_positions,
                    min_weight=params.min_weight,
                    max_weight=params.max_weight,
                    max_total_weight=params.max_total_weight,
                    reason_max_chars=params.reason_max_chars,
                )
        except HandoffError as exc:
            return self._fail(str(exc))
        self._submission, self._submitted, self.last_error = value, True, None
        return {"status": "recorded"}

    def _fail(self, message: str) -> dict[str, Any]:
        self.last_error = message
        return {"error": message}


# --- the tools ------------------------------------------------------------------------------------


def submit_tools(recorder: HandoffRecorder) -> dict[str, Callable[..., dict[str, Any]]]:
    """The three submit tools, closures over `recorder`, keyed by name. Each agent is given only its own."""

    def submit_ranking(ideas: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit your ranked ideas, best first: a list of objects with symbol and reason."""
        return recorder.submit(RANKING, ideas)

    def submit_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one verdict per symbol you were asked about: objects with symbol, verdict (survive or fail) and reason."""
        return recorder.submit(VERDICTS_STAGE, verdicts)

    def submit_portfolio(positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit the portfolio to hold: objects with symbol, weight (fraction of portfolio value) and reason; an empty list holds nothing."""
        return recorder.submit(PORTFOLIO, positions)

    return {"submit_ranking": submit_ranking, "submit_verdicts": submit_verdicts, "submit_portfolio": submit_portfolio}
```

- [ ] **Step 4: Run the tests and lint**

Run: `uv run ruff check src tests && uv run pytest tests/strategies/bill_ackman/test_ackman_handoff.py -q`
Expected: `59 passed`, no lint errors. (The last three tests build a real LangChain `StructuredTool` from each submit tool: they prove the agent layer can read the schema.)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bill_ackman/handoff.py tests/strategies/bill_ackman/test_ackman_handoff.py
git commit -m "feat: hand-off validation, recorder and the three submit tools

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The rebalancer

**Files:**
- Create: `src/trading_agent_framework/strategies/bill_ackman/rebalancer.py`
- Test: `tests/strategies/bill_ackman/test_ackman_rebalancer.py`

**Interfaces:**
- Consumes: `AckmanParams` (Task 5), `TargetPortfolio` (Task 6), `fractional_qty` and `parse_insufficient_buying_power` from `utils.helpers` (Task 4), the `Strategy` facade (`get_positions`, `get_last_price`, `create_order`, `submit_order`, `broker.get_account()`, `broker.tracker.get_active_orders()`), `entities.enums.OrderSide`.
- Produces:
  - `PlacedOrder(symbol: str, side: str, quantity: float)` (frozen dataclass; `side` is `"buy"` or `"sell"`).
  - `Rebalancer(strategy, params)` with `holdings() -> list[str]` (every long position except the parking instrument, sorted), `current_weights() -> dict[str, float]` (share of portfolio value, 4 decimals, sorted; empty if the portfolio value is not positive) and `rebalance(target: TargetPortfolio, forced_exits: Collection[str] = ()) -> list[PlacedOrder]` (the accepted orders, in submission order).
  - Behaviour: forced exits and stocks missing from the target are sold in full (never more than held and not already being sold); a stock above its target by more than the band (`rebalance_band` of portfolio value) is trimmed; parking above its target is sold down and also sold to fund the buys; buys are sized against `min(buying_power, cash + estimated sell proceeds) - cash_buffer` and clipped to it; the money left goes to the parking instrument up to its target; orders under `min_trade_pct` of portfolio value are skipped; a refused order is logged and does not stop the others, and a buy refused for buying power resyncs the available cash from the broker's figure. Open orders count: a pending buy counts toward the position, a pending sell is deducted.

- [ ] **Step 1: Write the tests**

`tests/strategies/bill_ackman/test_ackman_rebalancer.py` (portfolio value is 10,000 in every test, so the band is 500, the smallest order 50 and the cash reserve 200):

```python
from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.portfolio import target_portfolio
from trading_agent_framework.strategies.bill_ackman.rebalancer import PlacedOrder, Rebalancer
from trading_agent_framework.utils.errors import BrokerError

# Portfolio value 10,000 in every test: the band is 500 (5%), the smallest order 50 (0.5%), the cash reserve 200 (2%).


def _book(
    tmp_path: Path,
    *,
    positions: dict[str, float],
    prices: dict[str, float],
    cash: float,
    portfolio_value: float = 10_000.0,
    buying_power: float = 1_000_000.0,
) -> tuple[Strategy, FakeBroker, Rebalancer]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset(symbol), quantity=Decimal(str(quantity)), side=PositionSide.LONG) for symbol, quantity in positions.items()]
    broker.last_prices = {symbol: Decimal(str(price)) for symbol, price in prices.items()}
    broker.account = AccountBalances(cash=Decimal(str(cash)), portfolio_value=Decimal(str(portfolio_value)), buying_power=Decimal(str(buying_power)))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    return strategy, broker, Rebalancer(strategy, AckmanParams())


def _orders(broker: FakeBroker) -> list[tuple[str, str, float]]:
    return [(order.asset.symbol, order.side.value, float(order.quantity)) for order in broker.submitted]


def test_an_empty_book_buys_the_parking_instrument_up_to_everything_but_the_cash_buffer(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"SHV": 100}, cash=10_000)

    placed = rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02))

    assert _orders(broker) == [("SHV", "buy", 98.0)]
    assert placed == [PlacedOrder("SHV", "buy", 98.0)]


def test_new_stocks_are_funded_by_selling_parking_and_every_sell_goes_out_before_any_buy(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    assert _orders(broker) == [("SHV", "sell", 60.0), ("A", "buy", 70.0), ("B", "buy", 100.0)]


def test_a_forced_exit_is_sold_in_full_and_its_money_is_parked(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000)

    rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert _orders(broker) == [("A", "sell", 100.0), ("SHV", "buy", 98.0)]


def test_a_forced_exit_is_sold_even_if_it_is_in_the_target_weights(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02), forced_exits=["A"])

    assert _orders(broker)[0] == ("A", "sell", 100.0)
    assert not any(symbol == "A" and side == "buy" for symbol, side, _ in _orders(broker))


def test_a_holding_missing_from_the_target_is_sold_in_full(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"B": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=7500)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert _orders(broker) == [("B", "sell", 100.0), ("A", "buy", 70.0), ("SHV", "buy", 63.0)]


def test_a_drift_inside_the_band_does_not_trade(tmp_path: Path) -> None:
    # A is held at 3,300 against a target of 3,500: 200 below, inside the 500 band. Only the parking buy happens.
    _, broker, rebalancer = _book(tmp_path, positions={"A": 66}, prices={"A": 50, "SHV": 100}, cash=6700)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert _orders(broker) == [("SHV", "buy", 63.0)]


def test_a_drift_beyond_the_band_buys_the_difference(tmp_path: Path) -> None:
    # 2,950 against 3,500: 550 below, beyond the band.
    _, broker, rebalancer = _book(tmp_path, positions={"A": 59}, prices={"A": 50, "SHV": 100}, cash=7050)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert ("A", "buy", 11.0) in _orders(broker)


def test_a_drift_of_exactly_the_band_does_not_trade(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 60}, prices={"A": 50, "SHV": 100}, cash=7000)  # 3,000 against 3,500

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert not any(symbol == "A" for symbol, _, _ in _orders(broker))


def test_a_stock_above_its_band_is_trimmed_and_the_proceeds_are_parked(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 90}, prices={"A": 50, "SHV": 100}, cash=5500)  # 4,500 against 3,500

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert _orders(broker) == [("A", "sell", 20.0), ("SHV", "buy", 63.0)]


def test_parking_above_its_target_is_sold_down(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({}, cash_buffer=0.5))  # parking target 5,000, holding 9,800

    assert _orders(broker) == [("SHV", "sell", 48.0)]


def test_a_refused_buy_resyncs_the_available_cash_from_the_brokers_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)
    original = strategy.submit_order

    def submit(order):  # the broker refuses A for buying power and says what it really has
        if order.asset.symbol == "A" and order.side is OrderSide.BUY:
            raise BrokerError(json.dumps({"buying_power": "1000", "message": "insufficient buying power"}))
        return original(order)

    monkeypatch.setattr(strategy, "submit_order", submit)

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    # B is sized against 1,000 (the broker's figure) less the 200 reserve = 800, not against our own estimate.
    assert placed == [PlacedOrder("SHV", "sell", 60.0), PlacedOrder("B", "buy", 32.0)]


def test_a_refused_sell_is_not_counted_as_proceeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"B": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=7500)
    original = strategy.submit_order

    def submit(order):
        if order.asset.symbol == "B":
            raise BrokerError("cannot sell")
        return original(order)

    monkeypatch.setattr(strategy, "submit_order", submit)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    # available = min(bp, 7,500 + 0) - 200 = 7,300: A takes 3,500, the parking buy gets the remaining 3,800
    assert _orders(broker) == [("A", "buy", 70.0), ("SHV", "buy", 38.0)]


def test_a_target_without_a_price_is_skipped_with_a_warning_and_the_others_proceed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"B": 25, "SHV": 100}, cash=200)

    with caplog.at_level(logging.WARNING):
        rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    assert not any(symbol == "A" and side == "buy" for symbol, side, _ in _orders(broker))
    assert ("B", "buy", 100.0) in _orders(broker)
    assert "A" in caplog.text and "No price" in caplog.text


def test_a_failed_price_lookup_does_not_stop_a_forced_exit(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={}, cash=0)
    broker.market_data_error = BrokerError("market data is down")

    placed = rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert placed == [PlacedOrder("A", "sell", 100.0)]  # the quantity needs no price; no parking buy without one


def test_an_order_below_the_minimum_size_is_skipped(tmp_path: Path) -> None:
    # Only 240 in cash: 240 - 200 reserve = 40 available, below the 50 minimum trade.
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"SHV": 100}, cash=240)

    assert rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02)) == []
    assert broker.submitted == []


def test_quantities_are_floored_so_a_cost_never_exceeds_the_money(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 33.33, "SHV": 100}, cash=10_000)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    (_, _, quantity) = next(order for order in _orders(broker) if order[0] == "A")
    assert quantity == 105.010501
    assert quantity * 33.33 <= 3500


def test_buying_power_below_cash_limits_the_buys(tmp_path: Path) -> None:
    # A margin-free account reports less buying power than cash (unsettled funds): the smaller figure sizes the buys.
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "SHV": 100}, cash=10_000, buying_power=2_200)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert _orders(broker)[0] == ("A", "buy", 40.0)  # (2,200 - 200 reserve) / 50


def test_a_portfolio_value_that_is_not_positive_places_nothing(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 1}, prices={"A": 50}, cash=0, portfolio_value=0.0)

    assert rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02)) == []
    assert rebalancer.current_weights() == {}


def test_holdings_exclude_the_parking_instrument_and_empty_positions(tmp_path: Path) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"B": 10, "A": 100, "SHV": 10, "C": 0}, prices={"A": 50, "B": 25, "SHV": 100}, cash=0)

    assert rebalancer.holdings() == ["A", "B"]


def test_current_weights_are_each_stocks_share_of_portfolio_value(tmp_path: Path) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"A": 100, "B": 10, "SHV": 10}, prices={"A": 50, "B": 25, "SHV": 100}, cash=0)

    assert rebalancer.current_weights() == {"A": 0.5, "B": 0.025}


# --- orders still open from an earlier review -----------------------------------------------------------


def test_a_review_does_not_send_again_what_is_already_in_flight(tmp_path: Path) -> None:
    # FakeBroker tracks every submitted order as open and never fills it: the second review finds the first one's orders pending.
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "SHV": 100}, cash=10_000)
    target = target_portfolio({"A": 0.35}, cash_buffer=0.02)

    first = rebalancer.rebalance(target)
    second = rebalancer.rebalance(target)

    assert first == [PlacedOrder("A", "buy", 70.0), PlacedOrder("SHV", "buy", 63.0)]
    assert second == []
    assert len(broker.submitted) == 2


def test_a_forced_exit_is_not_sold_twice_while_the_first_sell_is_open(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000)
    target = target_portfolio({}, cash_buffer=0.02)

    first = rebalancer.rebalance(target, forced_exits=["A"])
    second = rebalancer.rebalance(target, forced_exits=["A"])

    assert first == [PlacedOrder("A", "sell", 100.0), PlacedOrder("SHV", "buy", 98.0)]
    assert second == []


def test_only_the_unfilled_part_of_a_partly_filled_order_counts_as_in_flight(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "SHV": 100}, cash=10_000)
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("A"), side=OrderSide.BUY, quantity=Decimal(70), filled_quantity=Decimal(30)))

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    # 40 shares (2,000) are still to come against a 3,500 target: 1,500 more, i.e. 30 shares
    assert PlacedOrder("A", "buy", 30.0) in placed


def test_a_sell_never_exceeds_what_is_held_and_not_already_being_sold(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000)
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("A"), side=OrderSide.SELL, quantity=Decimal(60)))

    placed = rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert placed[0] == PlacedOrder("A", "sell", 40.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_rebalancer.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bill_ackman.rebalancer'`.

- [ ] **Step 3: Implement `rebalancer.py`**

```python
"""The only module that places orders: sells first, then buys, then the parking instrument (SHV).

Follows the shape of `cross_momentum.rebalance` and the repo's cash-account rules: sell orders are submitted
before any buy, a buy is sized against the SMALLER of `buying_power` and cash plus the estimated proceeds of the
sells submitted in this run, minus the cash buffer (never `buying_power` alone: on a margin account it exceeds
cash), and quantities are floored (`fractional_qty`) so a cost never exceeds the money available.

Orders still open from an earlier review count: a pending buy counts toward the position and a pending sell is
deducted, so a review never sends again what is already in flight. (A backtest fills an order on the next bar, so a
daily review always finds the previous day's orders pending; in paper/live a market order is normally filled by then.)
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from trading_agent_framework.entities.enums import OrderSide
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.portfolio import TargetPortfolio
from trading_agent_framework.utils.errors import TradingFrameworkError
from trading_agent_framework.utils.helpers import fractional_qty, parse_insufficient_buying_power

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


@dataclass(frozen=True, slots=True)
class PlacedOrder:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float


class Rebalancer:
    def __init__(self, strategy: Strategy, params: AckmanParams) -> None:
        self._strategy = strategy
        self._params = params

    # --- what is held ----------------------------------------------------------------------------

    def holdings(self) -> list[str]:
        """The stocks held (every long position except the parking instrument), sorted."""
        parking = self._params.parking_symbol
        return sorted(position.asset.symbol for position in self._strategy.get_positions() if position.quantity > 0 and position.asset.symbol != parking)

    def current_weights(self) -> dict[str, float]:
        """Each held stock's share of portfolio value, rounded to 4 decimals; empty if the portfolio value is not positive."""
        portfolio_value = float(self._strategy.broker.get_account().portfolio_value)
        if portfolio_value <= 0:
            return {}
        parking = self._params.parking_symbol
        weights = {}
        for position in self._strategy.get_positions():
            symbol = position.asset.symbol
            if position.quantity > 0 and symbol != parking:
                weights[symbol] = round(float(position.quantity) * self._price(symbol) / portfolio_value, 4)
        return dict(sorted(weights.items()))

    def _price(self, symbol: str) -> float:
        """The last price, or 0.0 when the lookup fails: a failed quote must not abort a rebalance after sells went out."""
        try:
            return float(self._strategy.get_last_price(symbol) or 0.0)
        except TradingFrameworkError as exc:
            self._strategy.log_warning(f"No price for {symbol}: {exc}")
            return 0.0

    def _in_flight(self) -> tuple[dict[str, float], dict[str, float]]:
        """The quantity still to fill per symbol from open orders: (buys, sells)."""
        buys: dict[str, float] = {}
        sells: dict[str, float] = {}
        for order in self._strategy.broker.tracker.get_active_orders():
            if order.quantity is None:
                continue
            remaining = float(order.quantity - order.filled_quantity)
            book = buys if order.side is OrderSide.BUY else sells
            book[order.asset.symbol] = book.get(order.asset.symbol, 0.0) + remaining
        return buys, sells

    # --- the rebalance -----------------------------------------------------------------------------

    def rebalance(self, target: TargetPortfolio, forced_exits: Collection[str] = ()) -> list[PlacedOrder]:
        """Trade the book toward `target`; returns the orders that were accepted, in submission order."""
        strategy, params = self._strategy, self._params
        account = strategy.broker.get_account()
        portfolio_value = float(account.portfolio_value)
        if portfolio_value <= 0:
            strategy.log_warning(f"Portfolio value is {portfolio_value}: nothing to rebalance")
            return []
        parking = params.parking_symbol
        forced = set(forced_exits)
        targets = {symbol: weight for symbol, weight in target.weights.items() if symbol not in forced and symbol != parking}
        held = {position.asset.symbol: float(position.quantity) for position in strategy.get_positions() if position.quantity > 0}
        incoming, outgoing = self._in_flight()
        prices = {symbol: self._price(symbol) for symbol in dict.fromkeys([*held, *incoming, *outgoing, *targets, parking])}
        band = params.rebalance_band * portfolio_value
        min_trade = params.min_trade_pct * portfolio_value
        reserve = params.cash_buffer * portfolio_value
        placed: list[PlacedOrder] = []
        refusals: list[Exception] = []  # the last refused order's error, read by the buy loop

        def sellable(symbol: str) -> float:
            """What is held and not already being sold."""
            return max(0.0, held.get(symbol, 0.0) - outgoing.get(symbol, 0.0))

        def value_of(symbol: str) -> float:
            """The position's value counting open orders: held, plus buys in flight, minus sells in flight."""
            return max(0.0, held.get(symbol, 0.0) + incoming.get(symbol, 0.0) - outgoing.get(symbol, 0.0)) * prices.get(symbol, 0.0)

        def submit(symbol: str, side: str, quantity: float, why: str) -> bool:
            try:
                strategy.submit_order(strategy.create_order(symbol, quantity, side, time_in_force="day"))
            except TradingFrameworkError as exc:
                refusals[:] = [exc]
                strategy.log_warning(f"{side} {quantity:g} {symbol} ({why}) was refused: {exc}")
                return False
            strategy.log_info(f"{side} {quantity:g} {symbol} @ ${prices[symbol]:.2f} ({why})")
            placed.append(PlacedOrder(symbol, side, quantity))
            return True

        # 1. Sells: forced exits and dropped stocks in full, stocks above their band trimmed.
        proceeds = 0.0
        for symbol in held:
            if symbol == parking:
                continue
            price = prices[symbol]
            if symbol in forced or symbol not in targets:
                quantity, why = sellable(symbol), "forced exit" if symbol in forced else "not in the target portfolio"
            else:
                excess = value_of(symbol) - targets[symbol] * portfolio_value
                if price <= 0 or excess <= band or excess < min_trade:
                    continue
                quantity, why = min(fractional_qty(excess / price), sellable(symbol)), "trim above target"
            if quantity > 0 and submit(symbol, "sell", quantity, why):
                proceeds += quantity * price

        # What the buys need, and what is available for them: the smaller of buying power and cash plus the
        # estimated proceeds of the sells just submitted, less the cash buffer.
        plan: list[tuple[str, float]] = []
        for symbol, weight in targets.items():
            difference = weight * portfolio_value - value_of(symbol)
            if difference > band and difference >= min_trade:
                if prices[symbol] <= 0:
                    strategy.log_warning(f"No price for {symbol}: not buying it this review")
                    continue
                plan.append((symbol, difference))
        available = min(float(account.buying_power), float(account.cash) + proceeds) - reserve

        # The parking instrument gives back what it holds above its target, and funds the buys.
        parking_price = prices[parking]
        parking_value = value_of(parking)
        parking_target = target.parking_weight * portfolio_value
        if parking_price > 0 and sellable(parking) > 0:
            sell_value = 0.0
            excess = parking_value - parking_target
            if excess > band and excess >= min_trade:
                sell_value = excess
            shortfall = sum(difference for _, difference in plan) - available
            if shortfall > 0:
                sell_value = max(sell_value, min(shortfall, parking_value))
            if sell_value >= min_trade or (sell_value > 0 and sell_value >= parking_value):
                quantity = sellable(parking) if sell_value >= parking_value else min(fractional_qty(sell_value / parking_price), sellable(parking))
                if quantity > 0 and submit(parking, "sell", quantity, "above target or funding the buys"):
                    available += quantity * parking_price
                    parking_value -= quantity * parking_price

        # 2. Buys, in the trader's order.
        for symbol, difference in plan:
            if available <= 0:
                strategy.log_warning("No cash left for buying: skipping the remaining stocks")
                break
            price = prices[symbol]
            quantity = fractional_qty(min(difference, available) / price)
            if quantity * price < min_trade:
                continue
            if submit(symbol, "buy", quantity, "toward target"):
                available -= quantity * price
            else:
                # Refused for buying power: continue from the broker's own figure, not our (drifted) estimate.
                real = parse_insufficient_buying_power(refusals[0])
                if real is not None:
                    strategy.log_warning(f"Resyncing available cash to the broker-reported buying power: ${real:.2f}")
                    available = real - reserve

        # 3. Park what the stock buys left, up to the parking target.
        if parking_price > 0 and available > 0 and parking_target - parking_value > band:
            quantity = fractional_qty(min(parking_target - parking_value, available) / parking_price)
            if quantity * parking_price >= min_trade:
                submit(parking, "buy", quantity, "parking")
        return placed
```

- [ ] **Step 4: Run the tests and lint**

Run: `uv run ruff check src tests && uv run pytest tests/strategies/bill_ackman/test_ackman_rebalancer.py -q`
Expected: `24 passed`, no lint errors. (A floating-point weight such as 0.98 - 0.3 floors an order to `67.999999` shares, not `68`: the tests that depend on it compare with a tolerance. That is the intended `fractional_qty` behaviour, one millionth of a share.)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bill_ackman/rebalancer.py tests/strategies/bill_ackman/test_ackman_rebalancer.py
git commit -m "feat: the Ackman rebalancer, the only order code

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: State and the review log

**Files:**
- Create: `src/trading_agent_framework/strategies/bill_ackman/state.py`
- Test: `tests/strategies/bill_ackman/test_ackman_state.py`

**Interfaces:**
- Consumes: `TradingMode` (`config.env`), `ColorLogger` (`utils.log`).
- Produces:
  - `state_path(project_root: Path, mode: TradingMode) -> Path` (`<root>/data/bill_ackman_state_<mode>.json`).
  - `ReviewState(last_review: str | None = None, fail_counts: dict[str, int] = {}, last_ranking: list[str] = [], last_verdicts: dict[str, str] = {}, abandoned_streak: int = 0)` (frozen dataclass).
  - `StateStore(path)` with `load() -> ReviewState` (a missing file, or one that is unreadable or not a valid state, is an empty state; the invalid cases log a warning), `save(state)` (atomic: temporary file in the same directory, then replace; an `OSError` is logged, never raised) and `wipe()` (safe when there is no file).
  - `ReviewLog(path: Path | None)` with `append(record: dict)`: one JSON line (`default=str`, so `Decimal`s become strings); with no path it does nothing; an `OSError` is logged, never raised.

- [ ] **Step 1: Write the tests**

`tests/strategies/bill_ackman/test_ackman_state.py`:

```python
from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.bill_ackman.state import ReviewLog, ReviewState, StateStore, state_path


def _state() -> ReviewState:
    return ReviewState(
        last_review="2026-09-14",
        fail_counts={"HLT": 1},
        last_ranking=["AAA", "BBB"],
        last_verdicts={"AAA": "survive", "HLT": "fail"},
        abandoned_streak=2,
    )


def test_the_state_path_is_per_mode_under_the_projects_data_directory(tmp_path: Path) -> None:
    assert state_path(tmp_path, TradingMode.PAPER) == tmp_path / "data" / "bill_ackman_state_paper.json"
    assert state_path(tmp_path, TradingMode.BACKTESTING).name == "bill_ackman_state_backtesting.json"


def test_a_missing_file_is_an_empty_state(tmp_path: Path) -> None:
    assert StateStore(tmp_path / "state.json").load() == ReviewState()


def test_the_empty_state_has_no_history() -> None:
    state = ReviewState()

    assert (state.last_review, state.fail_counts, state.last_ranking, state.last_verdicts, state.abandoned_streak) == (None, {}, [], {}, 0)


def test_a_state_round_trips(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data" / "state.json")

    store.save(_state())

    assert StateStore(tmp_path / "data" / "state.json").load() == _state()


def test_saving_leaves_no_temporary_file(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")

    store.save(_state())
    store.save(ReviewState())

    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"version": 2}),
        json.dumps({"version": 1, "last_review": 5}),
        json.dumps({"version": 1, "fail_counts": {"HLT": "one"}}),
        json.dumps({"version": 1, "fail_counts": {"HLT": 0}}),
        json.dumps({"version": 1, "fail_counts": []}),
        json.dumps({"version": 1, "last_ranking": "AAA"}),
        json.dumps({"version": 1, "last_ranking": [1, 2]}),
        json.dumps({"version": 1, "last_verdicts": {"AAA": "maybe"}}),
        json.dumps({"version": 1, "abandoned_streak": -1}),
        json.dumps({"version": 1, "abandoned_streak": True}),
    ],
)
def test_a_corrupt_or_wrong_shaped_file_is_an_empty_state_with_a_warning(tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        state = StateStore(path).load()

    assert state == ReviewState()
    assert "state" in caplog.text.lower()


def test_a_state_that_cannot_be_written_is_logged_and_never_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "data"
    blocker.write_text("a file where the directory should be", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        StateStore(blocker / "state.json").save(_state())

    assert "could not be saved" in caplog.text


def test_wipe_removes_the_file_and_is_safe_when_there_is_none(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.save(_state())

    store.wipe()
    store.wipe()

    assert not (tmp_path / "state.json").exists()
    assert store.load() == ReviewState()


def test_the_review_log_appends_one_json_line_per_review(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "logs" / "run" / "reviews.jsonl")

    log.append({"date": "2026-09-14", "orders": [{"symbol": "AAA", "side": "buy", "quantity": 1.5}]})
    log.append({"date": "2026-09-15", "abandoned": True, "stage": "researcher", "error": "no submission"})

    lines = (tmp_path / "logs" / "run" / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["date"] for line in lines] == ["2026-09-14", "2026-09-15"]


def test_the_review_log_writes_decimals_as_strings(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "reviews.jsonl")

    log.append({"market_cap": Decimal("123.45")})

    assert json.loads((tmp_path / "reviews.jsonl").read_text(encoding="utf-8")) == {"market_cap": "123.45"}


def test_a_review_log_without_a_path_does_nothing(tmp_path: Path) -> None:
    ReviewLog(None).append({"date": "2026-09-14"})

    assert list(tmp_path.iterdir()) == []


def test_a_review_log_that_cannot_be_written_is_logged_and_never_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "logs"
    blocker.write_text("a file", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        ReviewLog(blocker / "reviews.jsonl").append({"date": "2026-09-14"})

    assert "could not be written" in caplog.text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_state.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bill_ackman.state'`.

- [ ] **Step 3: Implement `state.py`**

```python
"""What the strategy remembers between reviews, and the per-review log.

`StateStore` keeps the fail counters, yesterday's ranking and verdicts, and the abandoned-review streak in one
small JSON file per mode, written atomically. A missing or corrupt file is an empty state: the strategy then
simply starts its counters again. `ReviewLog` appends one JSON line per review to the run directory.
Neither ever raises on an I/O problem: bookkeeping must not stop a review.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "BillAckmanState")

STATE_VERSION = 1


def state_path(project_root: Path, mode: TradingMode) -> Path:
    """`<project_root>/data/bill_ackman_state_<mode>.json`."""
    return project_root / "data" / f"bill_ackman_state_{mode.value}.json"


@dataclass(frozen=True, slots=True)
class ReviewState:
    last_review: str | None = None  # ISO date of the last completed review
    fail_counts: dict[str, int] = field(default_factory=dict)  # holding -> consecutive fails (always >= 1)
    last_ranking: list[str] = field(default_factory=list)
    last_verdicts: dict[str, str] = field(default_factory=dict)  # symbol -> "survive" | "fail"
    abandoned_streak: int = 0  # abandoned reviews in a row


def _valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return False
    last_review = raw.get("last_review")
    fail_counts, ranking, verdicts = raw.get("fail_counts", {}), raw.get("last_ranking", []), raw.get("last_verdicts", {})
    streak = raw.get("abandoned_streak", 0)
    return (
        (last_review is None or isinstance(last_review, str))
        and isinstance(fail_counts, dict)
        and all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 1 for k, v in fail_counts.items())
        and isinstance(ranking, list)
        and all(isinstance(symbol, str) for symbol in ranking)
        and isinstance(verdicts, dict)
        and all(isinstance(k, str) and v in ("survive", "fail") for k, v in verdicts.items())
        and isinstance(streak, int)
        and not isinstance(streak, bool)
        and streak >= 0
    )


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> ReviewState:
        """The saved state; an empty one when the file is missing, unreadable, or not a valid state (warned about)."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ReviewState()
        except (OSError, ValueError) as exc:
            logger.log_warning(f"state file {self._path} is unreadable, starting from an empty state: {exc}")
            return ReviewState()
        if not _valid(raw):
            logger.log_warning(f"state file {self._path} is not a valid state, starting from an empty state")
            return ReviewState()
        return ReviewState(
            last_review=raw.get("last_review"),
            fail_counts=dict(raw.get("fail_counts", {})),
            last_ranking=list(raw.get("last_ranking", [])),
            last_verdicts=dict(raw.get("last_verdicts", {})),
            abandoned_streak=raw.get("abandoned_streak", 0),
        )

    def save(self, state: ReviewState) -> None:
        """Write the state atomically (temporary file in the same directory, then replace); an I/O error is logged."""
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self._path.parent, prefix=f"{self._path.name}.", suffix=".tmp")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"version": STATE_VERSION, **asdict(state)}, handle)
            os.replace(temporary, self._path)
        except OSError as exc:
            logger.log_warning(f"state could not be saved to {self._path}: {exc}")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def wipe(self) -> None:
        """Delete the state file (a backtest starts clean); safe when there is none."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.log_warning(f"state file {self._path} could not be deleted: {exc}")


class ReviewLog:
    """`reviews.jsonl`: one JSON line per review. With no path (a strategy run outside a runner) it does nothing."""

    def __init__(self, path: Path | None) -> None:
        self._path = path

    def append(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.log_warning(f"review log {self._path} could not be written: {exc}")
```

- [ ] **Step 4: Run the tests and lint**

Run: `uv run ruff check src tests && uv run pytest tests/strategies/bill_ackman/test_ackman_state.py -q`
Expected: `23 passed`, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bill_ackman/state.py tests/strategies/bill_ackman/test_ackman_state.py
git commit -m "feat: persisted review state and the per-review log

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Prompts and the review pipeline

**Files:**
- Create: `src/trading_agent_framework/strategies/bill_ackman/prompts.py`, `src/trading_agent_framework/strategies/bill_ackman/pipeline.py`
- Test: `tests/strategies/bill_ackman/test_ackman_pipeline.py`

**Interfaces:**
- Consumes: from earlier tasks `AckmanParams` (5), `fact_sheet`, `unavailable_fact_sheet`, `price_return` (5), `apply_verdicts` (6), `target_portfolio` (6), `HandoffRecorder`, `Idea`, `Verdict`, `PortfolioPosition`, `FAIL`, `submit_tools` (7), `Rebalancer` (8), `StateStore`, `ReviewState`, `ReviewLog` (9), `Candidate`, `ScreenResult` (1), `QualityScreen.run(..., top_n=)` (2); `AgentHandle.run(task_prompt, *, context, run_id, force_tool) -> AgentRunResult` from the agent layer; `MARKET_TZ`; `AgentError`, `FundamentalsError`, `TradingFrameworkError`.
- Produces:
  - `prompts`: `RESEARCHER_SYSTEM`, `SHORT_SELLER_SYSTEM`, `TRADER_SYSTEM` (strings), `researcher_task(top_n) -> str`, `SHORT_SELLER_TASK`, `TRADER_TASK`, `retry_prompt(tool, error) -> str`.
  - `pipeline`: `QUALITY_REJECTIONS` (frozenset of the screen rejection reasons that make code give a holding the verdict `fail`), protocols `ScreenLike` and `AgentLike`, `ReviewAbandoned(stage, message)`, `ReviewOutcome(completed: bool, abandoned_streak: int)`, and `ReviewPipeline(*, strategy, params, screen, agents, recorder, state, review_log, rebalancer, universe)` with `run() -> ReviewOutcome`.
  - `run()` does, in order: screen the universe and the holdings (`top_n=len(holdings)`); researcher (skipped when there are no candidates); review set = ranked + holdings; verdict `fail` by code for a holding rejected on a quality gate; short seller on the rest (skipped when nothing is left to judge); hysteresis; trader (skipped when nothing is allowed); target and `Rebalancer.rebalance`; save the state and append the review line. A stage that never makes a valid submit call (one run, then one forced retry with the last error) or a `FundamentalsError` from the screen abandons the review: nothing is traded, only `abandoned_streak` moves, and an abandoned line is logged. A `ConfigurationError` from an agent propagates.

- [ ] **Step 1: Write the tests**

`tests/strategies/bill_ackman/test_ackman_pipeline.py` (fake screen and fake agents; each fake agent step calls the real submit tools):

```python
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.bill_ackman.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.pipeline import ReviewOutcome, ReviewPipeline
from trading_agent_framework.strategies.bill_ackman.rebalancer import Rebalancer
from trading_agent_framework.strategies.bill_ackman.screen import Candidate, ScreenResult
from trading_agent_framework.strategies.bill_ackman.state import ReviewLog, ReviewState, StateStore
from trading_agent_framework.utils.errors import AgentError, BrokerError, ConfigurationError, FundamentalsError

Step = Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]


def _candidate(symbol: str, rank: int = 1) -> Candidate:
    return Candidate(
        symbol=symbol,
        rank=rank,
        score=0.8,
        sic=5812,
        market_cap=Decimal("100000000000"),
        fcf_yield=0.05,
        fcf_margin=0.2,
        operating_margin=0.25,
        operating_margin_stdev=0.02,
        revenue_growth=0.08,
        net_debt_to_operating_income=1.0,
        debt_reported=True,
        fiscal_year_end=date(2025, 12, 31),
        filed=date(2026, 2, 15),
    )


# --- fakes -------------------------------------------------------------------------------------------


class FakeScreen:
    """Returns the universe result for a universe call and the holdings result for a call with `top_n`."""

    def __init__(self, candidates: list[Candidate], *, holdings: list[Candidate] | None = None, holding_rejections: dict[str, str] | None = None) -> None:
        self.universe = ScreenResult(candidates=candidates, rejections={})
        self.holdings = ScreenResult(candidates=holdings or [], rejections=holding_rejections or {})
        self.calls: list[tuple[list[str], int | None]] = []
        self.error: FundamentalsError | None = None

    def run(self, symbols, *, as_of, price_of, top_n=None) -> ScreenResult:  # noqa: ANN001
        self.calls.append((list(symbols), top_n))
        if self.error is not None:
            raise self.error
        return self.universe if top_n is None else self.holdings


class FakeAgent:
    """Runs one scripted step per call; a step calls the real submit tools (or raises `AgentError`)."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.calls: list[dict[str, Any]] = []
        self.tools: dict[str, Callable[..., dict[str, Any]]] = {}

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None) -> AgentRunResult:
        self.calls.append({"task": task_prompt, "context": context, "run_id": run_id, "force_tool": force_tool})
        if self.steps:
            self.steps.pop(0)(self.tools, context)
        return AgentRunResult(output="done", tool_calls=[])


def ranks(*symbols: str) -> Step:
    return lambda tools, ctx: tools["submit_ranking"]([{"symbol": symbol, "reason": f"{symbol} is simple"} for symbol in symbols])


def judges(**verdicts: str) -> Step:
    return lambda tools, ctx: tools["submit_verdicts"]([{"symbol": symbol, "verdict": verdict, "reason": f"{verdict} reason"} for symbol, verdict in verdicts.items()])


def holds(**weights: float) -> Step:
    return lambda tools, ctx: tools["submit_portfolio"]([{"symbol": symbol, "weight": weight, "reason": "best idea"} for symbol, weight in weights.items()])


def does_nothing(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    return None


def crashes(message: str) -> Step:
    def step(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        raise AgentError(message)

    return step


@dataclass
class Harness:
    pipeline: ReviewPipeline
    broker: FakeBroker
    screen: FakeScreen
    researcher: FakeAgent
    short_seller: FakeAgent
    trader: FakeAgent
    store: StateStore
    log_path: Path
    orders: list[tuple[str, str, float]] = field(default_factory=list)

    def run(self) -> ReviewOutcome:
        self.broker.submitted.clear()
        outcome = self.pipeline.run()
        self.orders = [(order.asset.symbol, order.side.value, float(order.quantity)) for order in self.broker.submitted]
        return outcome

    def log_lines(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()] if self.log_path.exists() else []


def _harness(
    tmp_path: Path,
    screen: FakeScreen,
    *,
    held: dict[str, float] | None = None,
    params: AckmanParams | None = None,
    state: ReviewState | None = None,
) -> Harness:
    """Portfolio value 10,000. `held` maps a symbol to a number of shares; prices: AAA 50, BBB 25, HHH 50, SHV 100."""
    params = params or AckmanParams()
    prices = {"AAA": 50, "BBB": 25, "CCC": 20, "HHH": 50, "SHV": 100}
    held = held or {}
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    broker.last_prices = {symbol: Decimal(price) for symbol, price in prices.items()}
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset(symbol), quantity=Decimal(shares), side=PositionSide.LONG) for symbol, shares in held.items()]
    invested = sum(shares * prices[symbol] for symbol, shares in held.items())
    broker.account = AccountBalances(cash=Decimal(10_000 - invested), portfolio_value=Decimal(10_000), buying_power=Decimal(1_000_000))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    store = StateStore(tmp_path / "data" / "state.json")
    if state is not None:
        store.save(state)
    recorder = HandoffRecorder(params)
    agents = {"researcher": FakeAgent(), "short_seller": FakeAgent(), "trader": FakeAgent()}
    for agent in agents.values():
        agent.tools = submit_tools(recorder)
    log_path = tmp_path / "logs" / "reviews.jsonl"
    pipeline = ReviewPipeline(
        strategy=strategy,
        params=params,
        screen=screen,
        agents=agents,
        recorder=recorder,
        state=store,
        review_log=ReviewLog(log_path),
        rebalancer=Rebalancer(strategy, params),
        universe=["AAA", "BBB", "CCC"],
    )
    return Harness(pipeline, broker, screen, agents["researcher"], agents["short_seller"], agents["trader"], store, log_path)


# --- the happy path ---------------------------------------------------------------------------------


def test_a_review_runs_the_three_agents_in_order_and_places_the_orders(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA", 1), _candidate("BBB", 2), _candidate("CCC", 3)]))
    h.researcher.steps = [ranks("AAA", "BBB")]
    h.short_seller.steps = [judges(AAA="survive", BBB="fail")]
    h.trader.steps = [holds(AAA=0.3)]

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=True, abandoned_streak=0)
    # 30% in AAA, the rest but the 2% buffer parked (a floating-point weight of 0.68 floors to 67.999999 shares)
    assert h.orders[0] == ("AAA", "buy", 60.0)
    assert h.orders[1][:2] == ("SHV", "buy") and h.orders[1][2] == pytest.approx(68.0, abs=1e-5)
    assert len(h.orders) == 2
    state = h.store.load()
    assert (state.last_review, state.last_ranking, state.abandoned_streak) == ("2026-09-14", ["AAA", "BBB"], 0)
    assert state.last_verdicts == {"AAA": "survive", "BBB": "fail"}


def test_the_review_log_records_the_whole_review_as_one_line(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA", 1), _candidate("BBB", 2)]))
    h.researcher.steps = [ranks("AAA", "BBB")]
    h.short_seller.steps = [judges(AAA="survive", BBB="fail")]
    h.trader.steps = [holds(AAA=0.3)]

    h.run()

    (line,) = h.log_lines()
    assert line["date"] == "2026-09-14" and line["abandoned"] is False
    assert [c["symbol"] for c in line["candidates"]] == ["AAA", "BBB"]
    assert [idea["symbol"] for idea in line["ranking"]] == ["AAA", "BBB"]
    assert line["review_set"] == ["AAA", "BBB"]
    assert {(v["symbol"], v["verdict"], v["source"]) for v in line["verdicts"]} == {("AAA", "survive", "llm"), ("BBB", "fail", "llm")}
    assert line["allowed"] == ["AAA"]
    assert line["portfolio"] == [{"symbol": "AAA", "weight": 0.3, "reason": "best idea"}]
    assert line["targets"]["AAA"] == 0.3 and line["targets"]["SHV"] == pytest.approx(0.68)
    assert line["orders"][0] == {"symbol": "AAA", "side": "buy", "quantity": 60.0}
    assert line["orders"][1]["symbol"] == "SHV" and line["orders"][1]["quantity"] == pytest.approx(68.0, abs=1e-5)
    assert line["forced_exits"] == [] and line["fail_counts_after"] == {}


def test_the_researcher_is_asked_for_the_number_of_ideas_the_candidates_allow(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA"), _candidate("BBB"), _candidate("CCC")]))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    h.run()

    assert "Rank your best 3 ideas" in h.researcher.calls[0]["task"]


def test_the_agents_get_the_context_the_design_promises(tmp_path: Path) -> None:
    state = ReviewState(last_ranking=["OLD"], fail_counts={"HHH": 1})
    screen = FakeScreen([_candidate("AAA", 1)], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100}, state=state)
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive", HHH="survive")]
    h.trader.steps = [holds(AAA=0.3, HHH=0.2)]

    h.run()

    researcher = h.researcher.calls[0]["context"]
    assert researcher["current_datetime"].startswith("2026-09-14T10:00")
    assert [sheet["symbol"] for sheet in researcher["candidates"]] == ["AAA"]
    assert researcher["holdings"] == [{"symbol": "HHH", "weight": 0.5}]
    assert researcher["previous_ranking"] == ["OLD"]
    seller = h.short_seller.calls[0]["context"]["to_judge"]
    assert [entry["fact_sheet"]["symbol"] for entry in seller] == ["AAA", "HHH"]  # the ranked first, then the holdings
    assert seller[0]["researcher_reason"] == "AAA is simple" and seller[0]["held"] is False
    assert seller[1] == {**seller[1], "held": True, "current_weight": 0.5, "fail_count": 1, "researcher_reason": None}
    trader = h.trader.calls[0]["context"]
    by_symbol = {entry["symbol"]: entry for entry in trader["allowed"]}
    assert by_symbol["AAA"]["research_rank"] == 1 and by_symbol["AAA"]["verdict"] == "survive" and by_symbol["AAA"]["pending_fail_count"] == 0
    assert by_symbol["HHH"]["research_rank"] is None and by_symbol["HHH"]["verdict"] == "survive" and by_symbol["HHH"]["pending_fail_count"] == 0
    assert by_symbol["HHH"]["current_weight"] == 0.5
    assert trader["forced_exits"] == []
    assert trader["constraints"]["max_positions"] == 5 and trader["constraints"]["max_total_weight"] == pytest.approx(0.98)


# --- holdings, hysteresis and forced exits ------------------------------------------------------------


def test_a_holding_that_fails_twice_in_a_row_is_sold_even_if_the_trader_would_keep_it(tmp_path: Path) -> None:
    screen = FakeScreen([], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})

    h.short_seller.steps = [judges(HHH="fail")]
    h.trader.steps = [holds(HHH=0.35)]
    first = h.run()
    assert first.completed
    assert h.store.load().fail_counts == {"HHH": 1}
    (pending,) = h.trader.calls[0]["context"]["allowed"]
    assert (pending["symbol"], pending["verdict"], pending["pending_fail_count"]) == ("HHH", "fail", 1)
    assert h.orders[0] == ("HHH", "sell", 30.0)  # trimmed to 35%: it is a pending fail, not yet a forced exit

    h.short_seller.steps = [judges(HHH="fail")]
    h.trader.steps = [holds(HHH=0.35)]  # would keep it, but the trader is not even asked: nothing is allowed
    second = h.run()

    assert second.completed
    assert h.store.load().fail_counts == {"HHH": 2}
    assert len(h.trader.calls) == 1  # only the first day
    # The first day's trim (30 shares) is still open in the fake broker, which never fills: the forced exit sells the rest.
    assert h.orders[0] == ("HHH", "sell", 70.0)
    assert h.log_lines()[-1]["forced_exits"] == ["HHH"]


def test_a_holding_that_recovers_resets_its_counter(tmp_path: Path) -> None:
    screen = FakeScreen([], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.short_seller.steps = [judges(HHH="fail")]
    h.trader.steps = [holds(HHH=0.35)]
    h.run()
    h.short_seller.steps = [judges(HHH="survive")]
    h.trader.steps = [holds(HHH=0.35)]

    h.run()

    assert h.store.load().fail_counts == {}


def test_the_forced_exit_threshold_comes_from_the_parameters(tmp_path: Path) -> None:
    screen = FakeScreen([], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100}, params=AckmanParams(forced_exit_fails=1))
    h.short_seller.steps = [judges(HHH="fail")]

    h.run()

    assert h.orders[0] == ("HHH", "sell", 100.0)
    assert h.trader.calls == []


def test_a_holding_the_screen_rejected_on_quality_fails_by_code_and_never_reaches_the_short_seller(tmp_path: Path) -> None:
    screen = FakeScreen([], holding_rejections={"HHH": "negative_fcf"})
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.trader.steps = [holds(HHH=0.35)]

    h.run()

    assert h.short_seller.calls == []
    (line,) = h.log_lines()
    assert line["verdicts"] == [{"symbol": "HHH", "verdict": "fail", "reason": "screen: negative_fcf", "source": "screen"}]
    assert h.store.load().fail_counts == {"HHH": 1}
    assert h.trader.calls[0]["context"]["allowed"][0]["verdict_reason"] == "screen: negative_fcf"


def test_a_holding_the_screen_could_not_describe_goes_to_the_short_seller_with_a_reduced_sheet(tmp_path: Path) -> None:
    screen = FakeScreen([], holding_rejections={"HHH": "no_data"})
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.short_seller.steps = [judges(HHH="survive")]
    h.trader.steps = [holds(HHH=0.35)]

    h.run()

    (entry,) = h.short_seller.calls[0]["context"]["to_judge"]
    assert entry["fact_sheet"]["screen_unavailable"] == "no_data"
    assert entry["fact_sheet"]["price"] == 50.0
    assert h.log_lines()[0]["verdicts"][0]["source"] == "llm"


def test_the_holdings_are_screened_without_truncation(tmp_path: Path) -> None:
    screen = FakeScreen([_candidate("AAA")], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail", HHH="survive")]
    h.trader.steps = [holds(HHH=0.35)]

    h.run()

    assert screen.calls == [(["AAA", "BBB", "CCC"], None), (["HHH"], 1)]


# --- failures ----------------------------------------------------------------------------------------


def test_an_invalid_submission_is_corrected_inside_the_same_run(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))

    def bad_then_good(tools, ctx) -> None:  # noqa: ANN001
        assert "error" in tools["submit_ranking"]([{"symbol": "ZZZ", "reason": "x"}])
        ranks("AAA")(tools, ctx)

    h.researcher.steps = [bad_then_good]
    h.short_seller.steps = [judges(AAA="fail")]

    assert h.run().completed
    assert len(h.researcher.calls) == 1


def test_a_missing_submission_is_fixed_by_one_forced_retry_in_the_same_run(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [does_nothing, ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    assert h.run().completed

    first, retry = h.researcher.calls
    assert (first["force_tool"], retry["force_tool"]) == (None, "submit_ranking")
    assert first["run_id"] == retry["run_id"]
    assert "ended without calling submit_ranking" in retry["task"]


def test_the_retry_prompt_quotes_the_last_tool_error(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [lambda tools, ctx: tools["submit_ranking"]([{"symbol": "ZZZ", "reason": "x"}]), ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    h.run()

    assert "not one of the candidates" in h.researcher.calls[1]["task"]


def test_a_stage_with_no_valid_submission_after_the_retry_abandons_the_review_and_trades_nothing(tmp_path: Path) -> None:
    before = ReviewState(last_review="2026-09-11", fail_counts={"HHH": 1}, last_ranking=["OLD"], abandoned_streak=1)
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=before)
    h.researcher.steps = [does_nothing, does_nothing]

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=2)
    assert h.orders == []
    assert h.short_seller.calls == [] and h.trader.calls == []
    assert h.store.load() == ReviewState(last_review="2026-09-11", fail_counts={"HHH": 1}, last_ranking=["OLD"], abandoned_streak=2)
    (line,) = h.log_lines()
    assert line["abandoned"] is True and line["stage"] == "researcher" and line["abandoned_streak"] == 2


def test_an_agent_error_on_both_attempts_abandons_the_review_with_its_message(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [crashes("llm down"), crashes("llm down")]

    outcome = h.run()

    assert not outcome.completed
    assert "llm down" in h.log_lines()[0]["error"]


def test_a_later_stage_can_abandon_the_review_too(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [does_nothing, does_nothing]

    outcome = h.run()

    assert not outcome.completed
    assert h.log_lines()[0]["stage"] == "short_seller"
    assert h.orders == []


def test_a_screen_outage_abandons_the_review_before_any_agent_runs(tmp_path: Path) -> None:
    screen = FakeScreen([_candidate("AAA")])
    screen.error = FundamentalsError("sec is down")
    h = _harness(tmp_path, screen)

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=1)
    assert h.researcher.calls == []
    assert h.log_lines()[0]["stage"] == "screen"


def test_a_completed_review_resets_the_abandoned_streak(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]), state=ReviewState(abandoned_streak=2))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    assert h.run().abandoned_streak == 0
    assert h.store.load().abandoned_streak == 0


# --- degenerate days ---------------------------------------------------------------------------------


def test_with_no_candidates_and_no_holdings_no_agent_runs_and_everything_is_parked(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([]))

    outcome = h.run()

    assert outcome.completed
    assert (h.researcher.calls, h.short_seller.calls, h.trader.calls) == ([], [], [])
    assert h.orders == [("SHV", "buy", 98.0)]


def test_when_nothing_survives_the_trader_is_not_asked_and_everything_is_parked(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    assert h.run().completed

    assert h.trader.calls == []
    assert h.orders == [("SHV", "buy", 98.0)]


def test_an_empty_portfolio_from_the_trader_parks_everything(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive")]
    h.trader.steps = [lambda tools, ctx: tools["submit_portfolio"]([])]

    assert h.run().completed
    assert h.orders == [("SHV", "buy", 98.0)]


# --- inputs the design implies but does not spell out ---------------------------------------------------------


def test_a_failed_price_lookup_never_stops_a_review(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.broker.market_data_error = BrokerError("market data is down")
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive")]
    h.trader.steps = [holds(AAA=0.3)]

    outcome = h.run()

    assert outcome.completed
    (sheet,) = h.researcher.calls[0]["context"]["candidates"]
    assert sheet["price"] is None and sheet["price_return_12m"] is None  # the fact sheet says what it could not compute
    assert h.orders == []  # no price, nothing to size an order with


def test_a_configuration_error_from_an_agent_propagates_and_leaves_the_state_alone(tmp_path: Path) -> None:
    before = ReviewState(last_review="2026-09-11", fail_counts={"HHH": 1}, abandoned_streak=1)
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=before)

    def no_model(tools, ctx) -> None:  # noqa: ANN001
        raise ConfigurationError("No model id given and LLM_MODEL is not set")

    h.researcher.steps = [no_model]

    with pytest.raises(ConfigurationError, match="LLM_MODEL"):
        h.run()

    assert h.orders == []
    assert h.store.load() == before  # an unusable LLM setup is a configuration problem, not an abandoned review
    assert h.log_lines() == []


def test_a_position_the_screen_does_not_know_can_be_dropped_by_the_trader_and_is_then_sold(tmp_path: Path) -> None:
    # A dedicated account is assumed, so any position is reviewed; an ETF has no SEC data and reaches the short seller with a reduced sheet.
    screen = FakeScreen([], holding_rejections={"HHH": "no_data"})
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.short_seller.steps = [judges(HHH="survive")]
    h.trader.steps = [holds()]  # the trader chooses to hold nothing

    assert h.run().completed

    assert h.orders[0] == ("HHH", "sell", 100.0)


def test_the_counters_survive_a_restart(tmp_path: Path) -> None:
    first_day = _harness(tmp_path / "one", FakeScreen([], holdings=[_candidate("HHH")]), held={"HHH": 100})
    first_day.short_seller.steps = [judges(HHH="fail")]
    first_day.trader.steps = [holds(HHH=0.35)]
    first_day.run()
    saved = first_day.store.load()
    assert saved.fail_counts == {"HHH": 1}

    # A new process: a new pipeline, a new state file, seeded from what the old one saved on disk.
    next_day = _harness(tmp_path / "two", FakeScreen([], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=saved)
    next_day.short_seller.steps = [judges(HHH="fail")]

    assert next_day.run().completed

    assert next_day.log_lines()[0]["forced_exits"] == ["HHH"]
    assert next_day.trader.calls == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_pipeline.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bill_ackman.pipeline'`.

- [ ] **Step 3: Implement `prompts.py`**

```python
"""The three system prompts and task prompts (English only, written for a local model).

Each starts from the lumibot example's one-sentence role and adds what a local model needs: how to read the
fact sheet, what the verdicts mean, the language rule, and the contract of the single submit tool the agent must
end with. No agent is given an order tool; code places every order.
"""

from __future__ import annotations

_ENGLISH = (
    "Write everything in English: your reasons and every tool argument, even if a source you read drifts into another language."
)

RESEARCHER_SYSTEM = (
    "You are the researcher of a concentrated, long-only stock portfolio in the style of Bill Ackman: own just a few simple, "
    "high-quality companies and put real money behind them. Each day you receive fact sheets for the companies a quantitative "
    "screen selected, the stocks currently held, and yesterday's ranking.\n\n"
    "Your job: find the simple, predictable companies that make lots of cash and trade at a good price, and rank the best "
    "ones, best first.\n"
    "Read the fact sheet first; its numbers are computed by code, so do not recompute them. fcf_yield is free cash flow over "
    "market cap (higher is cheaper). fcf_margin_5y is the five-year average free cash flow over revenue. operating_margin_stdev "
    "is the volatility of the operating margin (lower is more predictable). revenue_cagr_5y is the growth rate. "
    "net_debt_to_operating_income is a debt multiple (lower is safer, negative means net cash); if debt_reported is false the "
    "debt figure is missing, so be suspicious of it. price_return_12m is the share price change over a year.\n"
    "Use the research tools only to check a specific claim, never to browse. Prefer a stock we already hold to a marginally "
    "better new idea unless something changed. You do not trade and have no order tool: a separate trader decides what to "
    "hold.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_ranking exactly once, with your ranked ideas: for each, the symbol (from the candidates "
    "only) and one short reason. If the tool returns an error, read it and call it again with a corrected argument. Once it "
    "returns status recorded, reply with one line and call no other tool."
)

SHORT_SELLER_SYSTEM = (
    "You are a short seller. Attack each idea you are given: too much debt, weak management, strong rivals, or a price that is "
    "too high. Say which ideas survive.\n\n"
    "For each symbol, survive means your attack failed: the company still looks simple, cash-generative, not over-indebted "
    "and not overpriced. fail means your attack succeeded and a prudent investor should not hold it. Start from the fact "
    "sheet (its numbers are computed by code; do not recompute them), then test a specific concern with the filing, news and "
    "price tools: heavy or rising debt, a falling or erratic margin, a new competitor, an accounting problem, a price far "
    "above what the cash flow supports. One or two checks per name is enough. A holding shows a fail_count: a stock that "
    "has already failed once and fails again is sold, so do not fail a holding lightly.\n"
    "Sources can be wrong or stale: do not repeat a figure from a news item as fact when the fact sheet or a filing says "
    "otherwise.\n"
    "Judge every symbol in to_judge and no other symbol.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_verdicts exactly once, with one object per symbol: the symbol, the verdict (survive or "
    "fail) and one short reason. If the tool returns an error, read it and call it again with a corrected argument. Once it "
    "returns status recorded, reply with one line and call no other tool."
)

TRADER_SYSTEM = (
    "You are the trader of a concentrated, long-only stock portfolio. Hold the few ideas that survived, with more money in "
    "the best ones, and let go of a stock that no longer survives the attack.\n\n"
    "You choose only from the allowed list in the context. Each name there has its research rank, the short seller's verdict "
    "and reason, a pending_fail_count (a name that failed once and is still allowed: weigh it against the stronger names), "
    "its current_weight and its fact sheet. Names in forced_exits are sold by code and are not allowed.\n"
    "Size each position as a fraction of portfolio value, for example 0.25, within the minimum and maximum weight given in "
    "the constraints; the weights together must not exceed the maximum total. Do not hold more positions than allowed. Leave "
    "money unallocated when fewer names deserve it: code parks it in short-term Treasuries (SHV), so cash is never a reason "
    "to hold a weak stock. An empty list means hold nothing. Prefer changing little: keep a current holding near its current "
    "weight unless the ranking or the verdict gives a reason to change it. You do not place orders; code does.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_portfolio exactly once, with one object per stock to hold: the symbol, the weight and one "
    "short reason. If the tool returns an error, read it and call it again with a corrected argument. Once it returns status "
    "recorded, reply with one line and call no other tool."
)


def researcher_task(top_n: int) -> str:
    return f"Rank your best {top_n} ideas from the candidates in the context, best first, then submit them. The current datetime is in the context."


SHORT_SELLER_TASK = "Attack each idea in to_judge in the context and submit one verdict per symbol. The current datetime is in the context."

TRADER_TASK = "Choose the portfolio to hold from the allowed list in the context and submit it. The current datetime is in the context."


def retry_prompt(tool: str, error: str) -> str:
    """The corrective turn after a run that never made a valid submit call."""
    return (
        f"Your previous run ended without a valid {tool} call. The last problem was: {error}\n"
        f"Call {tool} now, once, with a valid argument, then stop. Do not do any more research."
    )
```

- [ ] **Step 4: Implement `pipeline.py`**

```python
"""One review: screen, researcher, short seller, trader, rebalancer (see the design spec, section 2).

The agents decide; this module validates (through the `HandoffRecorder`), remembers (`StateStore`) and calls the
`Rebalancer`. A stage that never yields a valid submission abandons the review: nothing is traded and no counter
moves. Free text from an agent is logged and otherwise ignored.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.strategies.bill_ackman.fact_sheet import fact_sheet, price_return, unavailable_fact_sheet
from trading_agent_framework.strategies.bill_ackman.handoff import FAIL, HandoffRecorder, Idea, PortfolioPosition, Verdict
from trading_agent_framework.strategies.bill_ackman.hysteresis import apply_verdicts
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.portfolio import target_portfolio
from trading_agent_framework.strategies.bill_ackman.prompts import SHORT_SELLER_TASK, TRADER_TASK, researcher_task, retry_prompt
from trading_agent_framework.strategies.bill_ackman.rebalancer import Rebalancer
from trading_agent_framework.strategies.bill_ackman.screen import Candidate, ScreenResult
from trading_agent_framework.strategies.bill_ackman.state import ReviewLog, ReviewState, StateStore
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, FundamentalsError, TradingFrameworkError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

# A holding the screen rejects for one of these reasons failed a quality gate: code gives it the verdict `fail`.
# The other reasons (no_data, no_price, no_split_data, duplicate_listing) are data problems, not judgements.
QUALITY_REJECTIONS = frozenset(
    {"insufficient_history", "stale_filing", "operating_loss", "negative_fcf", "shrinking_revenue", "debt_unknown", "too_much_debt", "excluded_sector"}
)


class ScreenLike(Protocol):
    def run(self, symbols: Sequence[str], *, as_of: datetime, price_of: Callable[[str], Decimal | None], top_n: int | None = None) -> ScreenResult: ...


class AgentLike(Protocol):
    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None) -> AgentRunResult: ...


class ReviewAbandoned(Exception):
    """A stage could not produce what the review needs; the review ends with nothing traded."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    completed: bool
    abandoned_streak: int  # abandoned reviews in a row after this one (0 after a completed review)


class ReviewPipeline:
    def __init__(
        self,
        *,
        strategy: Strategy,
        params: AckmanParams,
        screen: ScreenLike,
        agents: Mapping[str, AgentLike],
        recorder: HandoffRecorder,
        state: StateStore,
        review_log: ReviewLog,
        rebalancer: Rebalancer,
        universe: Sequence[str],
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._screen = screen
        self._agents = agents
        self._recorder = recorder
        self._state = state
        self._log = review_log
        self._rebalancer = rebalancer
        self._universe = list(universe)

    # --- one review --------------------------------------------------------------------------------

    def run(self) -> ReviewOutcome:
        state = self._state.load()
        now = self._strategy.clock.now().astimezone(MARKET_TZ)
        record: dict[str, Any] = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, "abandoned": False}
        try:
            self._review(now, state, record)
        except ReviewAbandoned as exc:
            streak = state.abandoned_streak + 1
            self._strategy.log_error(f"review abandoned at the {exc.stage} stage (streak {streak}): {exc}")
            self._state.save(replace(state, abandoned_streak=streak))
            self._log.append({**record, "abandoned": True, "stage": exc.stage, "error": str(exc), "abandoned_streak": streak})
            return ReviewOutcome(completed=False, abandoned_streak=streak)
        return ReviewOutcome(completed=True, abandoned_streak=0)

    def _review(self, now: datetime, state: ReviewState, record: dict[str, Any]) -> None:
        params = self._params
        strategy = self._strategy
        price_of = strategy.get_last_price

        # 1-2. The screen over the universe, and over the holdings (so each holding has metrics or a reason).
        try:
            universe_result = self._screen.run(self._universe, as_of=now, price_of=price_of)
            holdings = self._rebalancer.holdings()
            holdings_result = self._screen.run(holdings, as_of=now, price_of=price_of, top_n=len(holdings)) if holdings else ScreenResult(candidates=[], rejections={})
        except FundamentalsError as exc:
            raise ReviewAbandoned("screen", str(exc)) from exc
        weights = self._rebalancer.current_weights()
        candidates = universe_result.candidates
        sheets = self._fact_sheets(candidates, holdings, holdings_result)
        record.update(candidates=[{"symbol": c.symbol, "rank": c.rank, "score": round(c.score, 4)} for c in candidates], holdings=holdings)

        # 3. Researcher.
        ranking: list[Idea] = []
        if candidates:
            symbols = [c.symbol for c in candidates]
            self._recorder.expect_ranking(symbols)
            context = {
                "current_datetime": now.isoformat(),
                "candidates": [sheets[symbol] for symbol in symbols],
                "holdings": [{"symbol": symbol, "weight": weight} for symbol, weight in weights.items()],
                "previous_ranking": state.last_ranking,
            }
            self._run_stage("researcher", "submit_ranking", researcher_task(min(params.research_top_n, len(symbols))), context)
            ranking = self._recorder.submission
        ranked = [idea.symbol for idea in ranking]
        researcher_reasons = {idea.symbol: idea.reason for idea in ranking}

        # 4-5. The review set, and the verdicts code gives to holdings the screen rejected on quality.
        review_set = list(dict.fromkeys([*ranked, *holdings]))
        verdicts: dict[str, Verdict] = {
            symbol: Verdict(symbol, FAIL, f"screen: {holdings_result.rejections[symbol]}")
            for symbol in holdings
            if holdings_result.rejections.get(symbol) in QUALITY_REJECTIONS
        }
        sources = {symbol: "screen" for symbol in verdicts}

        # 6. Short seller.
        to_judge = [symbol for symbol in review_set if symbol not in verdicts]
        if to_judge:
            self._recorder.expect_verdicts(to_judge)
            context = {
                "current_datetime": now.isoformat(),
                "to_judge": [
                    {
                        "fact_sheet": sheets[symbol],
                        "researcher_reason": researcher_reasons.get(symbol),
                        "held": symbol in holdings,
                        "current_weight": weights.get(symbol, 0.0),
                        "fail_count": state.fail_counts.get(symbol, 0),
                    }
                    for symbol in to_judge
                ],
            }
            self._run_stage("short_seller", "submit_verdicts", SHORT_SELLER_TASK, context)
            for verdict in self._recorder.submission:
                verdicts[verdict.symbol] = verdict
                sources[verdict.symbol] = "llm"

        # 7. Hysteresis.
        outcome = apply_verdicts(
            holdings=holdings,
            ranking=ranked,
            verdicts={symbol: verdict.verdict for symbol, verdict in verdicts.items()},
            fail_counts=state.fail_counts,
            forced_exit_fails=params.forced_exit_fails,
        )

        # 8. Trader.
        positions: list[PortfolioPosition] = []
        if outcome.allowed:
            self._recorder.expect_portfolio(outcome.allowed)
            context = {
                "current_datetime": now.isoformat(),
                "allowed": [
                    {
                        "symbol": symbol,
                        "research_rank": ranked.index(symbol) + 1 if symbol in ranked else None,
                        "verdict": verdicts[symbol].verdict,
                        "verdict_reason": verdicts[symbol].reason,
                        "researcher_reason": researcher_reasons.get(symbol),
                        "pending_fail_count": outcome.fail_counts.get(symbol, 0),
                        "current_weight": weights.get(symbol, 0.0),
                        "fact_sheet": sheets[symbol],
                    }
                    for symbol in outcome.allowed
                ],
                "forced_exits": outcome.forced_exits,
                "constraints": {
                    "max_positions": params.max_positions,
                    "min_weight": params.min_weight,
                    "max_weight": params.max_weight,
                    "max_total_weight": params.max_total_weight,
                    "unallocated_money": f"parked in {params.parking_symbol} by code",
                },
            }
            self._run_stage("trader", "submit_portfolio", TRADER_TASK, context)
            positions = self._recorder.submission

        # 9. Targets and execution.
        target = target_portfolio({position.symbol: position.weight for position in positions}, cash_buffer=params.cash_buffer)
        orders = self._rebalancer.rebalance(target, outcome.forced_exits)

        # 10. Remember and log.
        self._state.save(
            ReviewState(
                last_review=now.date().isoformat(),
                fail_counts=outcome.fail_counts,
                last_ranking=ranked,
                last_verdicts={symbol: verdict.verdict for symbol, verdict in verdicts.items()},
                abandoned_streak=0,
            )
        )
        self._log.append(
            {
                **record,
                "ranking": [asdict(idea) for idea in ranking],
                "review_set": review_set,
                "verdicts": [{**asdict(verdict), "source": sources[verdict.symbol]} for verdict in verdicts.values()],
                "fail_counts_before": state.fail_counts,
                "fail_counts_after": outcome.fail_counts,
                "forced_exits": outcome.forced_exits,
                "allowed": outcome.allowed,
                "portfolio": [asdict(position) for position in positions],
                "targets": {**target.weights, params.parking_symbol: target.parking_weight},
                "orders": [asdict(order) for order in orders],
            }
        )

    # --- helpers -----------------------------------------------------------------------------------

    def _run_stage(self, agent_name: str, tool: str, task: str, context: dict[str, Any]) -> None:
        """Run an agent until it makes a valid `tool` call: once, then once more with a forced call; else abandon."""
        agent = self._agents[agent_name]
        run_id = uuid.uuid4().hex
        prompt, force_tool, error = task, None, ""
        for _ in range(2):
            try:
                result = agent.run(prompt, context=context, run_id=run_id, force_tool=force_tool)
                self._strategy.log_info(f"[{agent_name}] {result.output}")
            except AgentError as exc:
                error = str(exc)
                self._strategy.log_error(f"[{agent_name}] run failed: {exc}")
            if self._recorder.submitted:
                return
            error = self._recorder.last_error or error or f"the agent ended without calling {tool}"
            prompt, force_tool = retry_prompt(tool, error), tool
        raise ReviewAbandoned(agent_name, error)

    def _fact_sheets(self, candidates: Sequence[Candidate], holdings: Sequence[str], holdings_result: ScreenResult) -> dict[str, dict[str, Any]]:
        """A fact sheet per candidate and per holding; a holding the screen could not describe gets the price facts and the reason."""
        sheets: dict[str, dict[str, Any]] = {}
        for candidate in [*candidates, *holdings_result.candidates]:
            if candidate.symbol not in sheets:
                price, change = self._price_facts(candidate.symbol)
                sheets[candidate.symbol] = fact_sheet(candidate, price=price, price_return_12m=change)
        for symbol in holdings:
            if symbol not in sheets:
                price, change = self._price_facts(symbol)
                sheets[symbol] = unavailable_fact_sheet(symbol, reason=holdings_result.rejections.get(symbol, "no_data"), price=price, price_return_12m=change)
        return sheets

    def _price_facts(self, symbol: str) -> tuple[float | None, float | None]:
        """(last price, 12-month return); each None when it cannot be computed (a failed lookup never stops a review)."""
        strategy = self._strategy
        try:
            last = strategy.get_last_price(symbol)
        except TradingFrameworkError:
            last = None
        try:
            bars = strategy.get_historical_prices(symbol, 253)
        except TradingFrameworkError:
            bars = None
        closes = [float(close) for close in bars.df["close"]] if bars is not None else []
        return (float(last) if last is not None else None, price_return(closes))
```

- [ ] **Step 5: Run the tests and lint**

Run: `uv run ruff check src tests && uv run pytest tests/strategies/bill_ackman/test_ackman_pipeline.py -q`
Expected: `25 passed`, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/bill_ackman/prompts.py src/trading_agent_framework/strategies/bill_ackman/pipeline.py tests/strategies/bill_ackman/test_ackman_pipeline.py
git commit -m "feat: the Ackman review pipeline and the agent prompts

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: The strategy, its registration and the README

**Files:**
- Create: `src/trading_agent_framework/strategies/bill_ackman/agent_bill_ackman.py`
- Modify: `src/trading_agent_framework/strategies/bill_ackman/__init__.py`, `src/trading_agent_framework/main.py`, `README.md`
- Test: `tests/strategies/bill_ackman/test_ackman_strategy.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: everything above; `Strategy` (`core`), `YahooBacktestData`, `backtest_window`/`PredefinedWindow`, `build_quality_screen` (Task 1), `FatalStrategyError`/`ConfigurationError`, `market_data_tools` (eager) and `fundamentals_tools`/`news_tools` (imported inside `initialize`).
- Produces:
  - `BillAckmanStrategy(broker, *, mode=TradingMode.PAPER, universe: Sequence[str], settings: AckmanParams | None = None, screen: ScreenLike | None = None, **kwargs)`; class attributes `sleeptime = "1D"` and `parameters` (`backtesting_start`/`backtesting_end` from `backtest_window(PredefinedWindow.BI_MONTH)`, `benchmark_symbol` `"SPY"`, `warmup_trading_days` 10, `budget` 100000); `initialize()`, `on_trading_iteration()`, `_review_log_path()`, `run_backtesting()`; attribute `pipeline: ReviewPipeline | None`.
  - `initialize()` wipes the state file in backtesting mode, builds the screen (`build_quality_screen` unless one was injected), creates the three agents with the tool lists of the spec, and builds the pipeline; a `ConfigurationError` becomes `FatalStrategyError`. `on_trading_iteration()` runs `pipeline.run()` and, in a backtest only, raises `FatalStrategyError` once `abandoned_streak >= settings.max_consecutive_abandoned`.
  - `main.py`: `"bill_ackman": _build_bill_ackman` in `AGENT_STRATEGIES` (loads the `cross_momentum` universe file; prints the standard message and returns `None` when it is missing).

- [ ] **Step 1: Write the tests**

`tests/strategies/bill_ackman/test_ackman_strategy.py`:

```python
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.pipeline import ReviewOutcome
from trading_agent_framework.strategies.bill_ackman.state import ReviewState, StateStore, state_path
from trading_agent_framework.utils.errors import FatalStrategyError

UNIVERSE = ["AAA", "BBB", "CCC"]


class _FakeAgents:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        return object()

    def __getitem__(self, name: str) -> object:
        return object()


class _FakeScreen:
    def run(self, symbols, *, as_of, price_of, top_n=None):  # noqa: ANN001, ANN201
        raise AssertionError("not called in these tests")


class _FakePipeline:
    def __init__(self, outcomes: list[ReviewOutcome]) -> None:
        self.outcomes = list(outcomes)
        self.runs = 0

    def run(self) -> ReviewOutcome:
        self.runs += 1
        return self.outcomes.pop(0)


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _strategy(tmp_path: Path, mode: TradingMode = TradingMode.PAPER, **kwargs: Any) -> tuple[BillAckmanStrategy, _FakeAgents]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    screen = kwargs.pop("screen", _FakeScreen())
    strategy = BillAckmanStrategy(broker, mode=mode, universe=UNIVERSE, project_root=tmp_path, screen=screen, **kwargs)
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def _tool_names(created: dict[str, Any]) -> set[str]:
    return {tool.__name__ for tool in created["tools"]}


def test_the_strategy_runs_once_a_day() -> None:
    assert BillAckmanStrategy.sleeptime == "1D"


def test_the_defaults_use_the_bi_month_window_and_the_spec_budget() -> None:
    parameters = BillAckmanStrategy.parameters

    assert (parameters["backtesting_start"], parameters["backtesting_end"]) == backtest_window(PredefinedWindow.BI_MONTH)
    assert (parameters["benchmark_symbol"], parameters["budget"], parameters["warmup_trading_days"]) == ("SPY", 100000, 10)


def test_initialize_creates_the_three_agents_with_their_own_tools(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    researcher, short_seller, trader = agents.created
    assert [created["name"] for created in agents.created] == ["researcher", "short_seller", "trader"]
    assert {"get_company_facts", "get_filings", "get_bars", "get_last_price", "submit_ranking"} <= _tool_names(researcher)
    assert {"get_company_facts", "search_news", "get_bars", "submit_verdicts"} <= _tool_names(short_seller)
    assert _tool_names(trader) == {"submit_portfolio"}


def test_no_agent_has_an_order_account_indicator_or_memory_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    forbidden = {"submit_order", "cancel_order", "cancel_open_orders", "close_position", "sell_all", "get_positions", "get_account_balance", "get_orders", "remember_decision", "search_memory"}
    for created in agents.created:
        assert not forbidden & _tool_names(created)


def test_each_agent_gets_only_its_own_submit_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    submits = [{name for name in _tool_names(created) if name.startswith("submit_")} for created in agents.created]
    assert submits == [{"submit_ranking"}, {"submit_verdicts"}, {"submit_portfolio"}]


def test_every_prompt_requires_english_and_names_its_submit_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    for created, tool in zip(agents.created, ["submit_ranking", "submit_verdicts", "submit_portfolio"], strict=True):
        assert "English" in created["system_prompt"]
        assert tool in created["system_prompt"]
        assert "exactly once" in created["system_prompt"]


def test_a_backtest_starts_with_no_saved_state_and_paper_keeps_it(tmp_path: Path) -> None:
    for mode, kept in ((TradingMode.BACKTESTING, False), (TradingMode.PAPER, True)):
        StateStore(state_path(tmp_path, mode)).save(ReviewState(fail_counts={"HHH": 1}))
        strategy, _ = _strategy(tmp_path, mode)

        strategy.initialize()

        assert state_path(tmp_path, mode).exists() is kept


def test_a_missing_sec_user_agent_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    strategy, _ = _strategy(tmp_path)

    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_without_an_injected_screen_the_real_one_is_built_and_needs_the_user_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    strategy = BillAckmanStrategy(FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman"), mode=TradingMode.PAPER, universe=UNIVERSE, project_root=tmp_path)
    strategy._agents = cast(AgentManager, _FakeAgents())

    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_a_review_runs_on_every_iteration(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.initialize()
    pipeline = _FakePipeline([ReviewOutcome(True, 0)] * 3)
    strategy.pipeline = pipeline  # type: ignore[assignment]

    for _ in range(3):
        strategy.on_trading_iteration()

    assert pipeline.runs == 3


def test_a_backtest_aborts_after_three_abandoned_reviews_in_a_row(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    strategy.initialize()
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, 1), ReviewOutcome(False, 2), ReviewOutcome(False, 3)])  # type: ignore[assignment]

    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    with pytest.raises(FatalStrategyError, match="3 reviews abandoned in a row"):
        strategy.on_trading_iteration()


def test_the_abort_threshold_comes_from_the_parameters(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING, settings=AckmanParams(max_consecutive_abandoned=1))
    strategy.initialize()
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, 1)])  # type: ignore[assignment]

    with pytest.raises(FatalStrategyError):
        strategy.on_trading_iteration()


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_paper_and_live_never_abort_on_abandoned_reviews(tmp_path: Path, mode: TradingMode) -> None:
    strategy, _ = _strategy(tmp_path, mode)
    strategy.initialize()
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, streak) for streak in range(1, 8)])  # type: ignore[assignment]

    for _ in range(7):
        strategy.on_trading_iteration()


def test_the_review_log_lives_in_the_run_directory_once_there_is_a_run_id(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)

    assert strategy._review_log_path() is None
    strategy.run_id = "2026-09-14_100000_backtesting"

    assert strategy._review_log_path() == tmp_path / "logs" / "bill_ackman" / "backtesting" / "2026-09-14_100000_backtesting" / "reviews.jsonl"


def test_run_backtesting_passes_the_window_the_daily_yahoo_source_and_the_preloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: seen.update(kwargs))

    strategy.run_backtesting()

    start, end = backtest_window(PredefinedWindow.BI_MONTH)
    assert (seen["start"], seen["end"]) == (start, end)
    assert seen["data_source"] is YahooBacktestData
    assert seen["timestep"] == "day" and seen["benchmark"] == "SPY" and seen["warmup_trading_days"] == 10
    assert seen["budget"] == Decimal("100000") and seen["agent_telemetry"] is True
    assert [asset.symbol for asset in seen["preload_assets"]] == ["AAA", "BBB", "CCC", "SHV", "SPY"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_strategy.py -q`
Expected: collection error, `ImportError: cannot import name 'BillAckmanStrategy' from 'trading_agent_framework.strategies.bill_ackman'`.

- [ ] **Step 3: Implement the strategy**

`src/trading_agent_framework/strategies/bill_ackman/agent_bill_ackman.py`:

```python
"""BillAckmanStrategy: a researcher, a short seller and a trader over the fundamentals quality screen.

Port of lumibot's Bill Ackman example, hardened for a local model: code screens the universe and builds the fact
sheets, three agents hand structured results to each other through submit tools, and code keeps the state and
places every order (`ReviewPipeline`, `Rebalancer`). No agent has an order tool.
See docs/superpowers/specs/2026-10-02-bill-ackman-strategy-design.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.bill_ackman.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.pipeline import ReviewPipeline, ScreenLike
from trading_agent_framework.strategies.bill_ackman.prompts import RESEARCHER_SYSTEM, SHORT_SELLER_SYSTEM, TRADER_SYSTEM
from trading_agent_framework.strategies.bill_ackman.rebalancer import Rebalancer
from trading_agent_framework.strategies.bill_ackman.screen import build_quality_screen
from trading_agent_framework.strategies.bill_ackman.state import ReviewLog, StateStore, state_path
from trading_agent_framework.utils.errors import ConfigurationError, FatalStrategyError


class BillAckmanStrategy(Strategy):
    """One review per session: screen, researcher, short seller, trader, rebalancer (see `ReviewPipeline`)."""

    sleeptime = "1D"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.BI_MONTH)[0],
        "backtesting_end": backtest_window(PredefinedWindow.BI_MONTH)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 10,  # the screen needs a last price on the first simulated day
        "budget": 100000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: AckmanParams | None = None,
        screen: ScreenLike | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.universe = list(universe)
        self.settings = settings or AckmanParams()
        self._screen = screen  # injected in tests; the real screen is built in `initialize`
        self.pipeline: ReviewPipeline | None = None

    # --- lifecycle ------------------------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the screen, the three agents and the pipeline (once per run)."""
        state_store = StateStore(state_path(self.project_root, self.trading_mode))
        if self.is_backtesting:
            state_store.wipe()  # one run's counters must not leak into the next; paper and live state is never wiped
        try:
            screen = self._screen if self._screen is not None else build_quality_screen(self.project_root, params=self.settings.screen)
            recorder = HandoffRecorder(self.settings)
            submit = submit_tools(recorder)
            from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
            from trading_agent_framework.agents.tools.news import news_tools

            # No agent gets an order, account, indicator or memory tool: they research and hand over structured results.
            self.agents.create(name="researcher", system_prompt=RESEARCHER_SYSTEM, tools=[*fundamentals_tools(self), *market_data_tools(self), submit["submit_ranking"]])
            self.agents.create(
                name="short_seller",
                system_prompt=SHORT_SELLER_SYSTEM,
                tools=[*fundamentals_tools(self), *news_tools(self), *market_data_tools(self), submit["submit_verdicts"]],
            )
            self.agents.create(name="trader", system_prompt=TRADER_SYSTEM, tools=[submit["submit_portfolio"]])
        except ConfigurationError as exc:
            # The strategy refuses to start rather than fail every day (SEC_EDGAR_USER_AGENT missing, no LLM model).
            raise FatalStrategyError(str(exc)) from exc
        self.pipeline = ReviewPipeline(
            strategy=self,
            params=self.settings,
            screen=screen,
            agents=self.agents,
            recorder=recorder,
            state=state_store,
            review_log=ReviewLog(self._review_log_path()),
            rebalancer=Rebalancer(self, self.settings),
            universe=self.universe,
        )
        self.log_info(f"BillAckmanStrategy initialized: {len(self.universe)} symbols, sleeptime {self.sleeptime}")

    def on_trading_iteration(self) -> None:
        """One review. A backtest aborts when too many reviews in a row were abandoned (a dead LLM or data source)."""
        assert self.pipeline is not None, "initialize() has not run"
        outcome = self.pipeline.run()
        if not outcome.completed and self.is_backtesting and outcome.abandoned_streak >= self.settings.max_consecutive_abandoned:
            raise FatalStrategyError(f"{outcome.abandoned_streak} reviews abandoned in a row, aborting the backtest (see the log for the stage and the error)")

    def _review_log_path(self) -> Path | None:
        """`reviews.jsonl` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "reviews.jsonl"

    # --- backtesting ------------------------------------------------------------------------------------

    def run_backtesting(self):
        """Backtest over the class `parameters` window (`PredefinedWindow.BI_MONTH`) on Yahoo daily bars."""
        symbols = list(dict.fromkeys([*self.universe, self.settings.parking_symbol, self.parameters["benchmark_symbol"]]))
        return super().run_backtesting(
            data_source=YahooBacktestData,  # years of daily history
            timestep="day",
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],  # the screen's daily price lookups hit the cache
            benchmark=self.parameters["benchmark_symbol"],
            budget=Decimal(str(self.parameters["budget"])),
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=True,
        )
```

Replace the whole of `src/trading_agent_framework/strategies/bill_ackman/__init__.py` with:

```python
"""The Bill Ackman portfolio strategy: a researcher, a short seller and a trader over the fundamentals quality screen."""

from trading_agent_framework.strategies.bill_ackman.agent_bill_ackman import BillAckmanStrategy

__all__ = ["BillAckmanStrategy"]
```

- [ ] **Step 4: Register the strategy and extend `tests/test_main.py`**

Run from the repository root (it edits `main.py` and `tests/test_main.py`; both edits are asserted to match exactly once):

```bash
uv run python - <<'PY'
"""Register the strategy in main.py and extend tests/test_main.py."""

from pathlib import Path

main = Path("src/trading_agent_framework/main.py")
text = main.read_text()


def replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, old
    return source.replace(old, new)


text = replace_once(
    text,
    "from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy\n",
    "from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy\nfrom trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy\n",
)
text = replace_once(
    text,
    'AGENT_STRATEGIES: dict[str, StrategyBuilder] = {\n    "cross_momentum": _build_cross_momentum,',
    '''def _build_bill_ackman(broker: Broker, mode: TradingMode) -> Strategy | None:
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run `uv run batch-universe` before executing this strategy.", style="bold red")
        return None
    return BillAckmanStrategy(broker=broker, mode=mode, universe=universe)


AGENT_STRATEGIES: dict[str, StrategyBuilder] = {
    "bill_ackman": _build_bill_ackman,
    "cross_momentum": _build_cross_momentum,''',
)
main.write_text(text)

test_main = Path("tests/test_main.py")
tests = test_main.read_text()
tests = replace_once(
    tests,
    '{"cross_momentum", "news_binary", "vwap_pullback_continuation"}',
    '{"bill_ackman", "cross_momentum", "news_binary", "vwap_pullback_continuation"}',
)
tests = replace_once(
    tests,
    "def test_cross_momentum_builder_returns_none_without_a_universe(",
    '''def test_bill_ackman_builder_returns_the_strategy_with_the_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: ["AAA", "BBB"])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")

    strategy = main_module._build_bill_ackman(broker, TradingMode.BACKTESTING)

    assert isinstance(strategy, BillAckmanStrategy)
    assert strategy.universe == ["AAA", "BBB"]
    assert strategy.is_backtesting


def test_bill_ackman_builder_returns_none_without_a_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: [])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")

    assert main_module._build_bill_ackman(broker, TradingMode.BACKTESTING) is None


def test_cross_momentum_builder_returns_none_without_a_universe(''',
)
tests = replace_once(
    tests,
    "from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy\n",
    "from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy\nfrom trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy\n",
)
test_main.write_text(tests)
PY
```

- [ ] **Step 5: Run the tests and lint**

Run: `uv run ruff check --fix src tests && uv run ruff check && uv run pytest tests/strategies/bill_ackman/test_ackman_strategy.py tests/test_main.py -q`
Expected: `16 passed` and `8 passed` (24 tests), no lint errors.

- [ ] **Step 6: README**

Run from the repository root (the user agent is also needed by this strategy, and a section for it):

```bash
uv run python - <<'PY'
"""README: the user agent is also needed by the Ackman strategy, and a section for the strategy."""

from pathlib import Path

readme = Path("README.md")
text = readme.read_text()


def replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, old
    return source.replace(old, new)


text = replace_once(
    text,
    "`SEC_EDGAR_USER_AGENT` is needed only if a strategy wires in `agents.tools.fundamentals_tools` (SEC company facts/filings).",
    "`SEC_EDGAR_USER_AGENT` is needed if a strategy wires in `agents.tools.fundamentals_tools` (SEC company facts/filings) or uses the fundamentals quality screen (`bill_ackman`).",
)
section = '''#### 📈 `bill_ackman` — Bill Ackman portfolio (researcher, short seller, trader)
| Field | Value |
|---|---|
| **File** | `strategies/bill_ackman/agent_bill_ackman.py` |
| **Model** | from `LLM_MODEL` in the env file (one model for the three agents) |
| **Agents** | `researcher`, `short_seller`, `trader` |
| **Tools** | researcher: SEC fundamentals + market data; short seller: SEC fundamentals + news + market data; trader: none. Each agent ends with one submit tool (`submit_ranking`, `submit_verdicts`, `submit_portfolio`); none can place an order |
| **Asset universe** | the `cross_momentum` universe file, cut to 15 candidates by the quality screen every day, plus SHV |
| **Agent frequency** | once per session |
| **Trading modes** | backtest, paper, live |
| **Benchmark** | SPY |
| **Env file** | `env/.env.bill_ackman.<mode>`: `LLM_*`, `SEC_EDGAR_USER_AGENT`, `ALPACA_NEWS_*` (the short seller's news tool), plus the broker keys in paper/live |

A concentrated long-only portfolio of at most 5 stocks, after lumibot's Bill Ackman example. Each day code screens the universe for simple, cash-generative, lightly indebted, reasonably priced companies (`strategies/bill_ackman/screen/`), the researcher ranks the best 5, the short seller attacks them and every stock already held, and the trader picks the weights. Code applies a hysteresis (a holding that fails the attack on 2 consecutive days is sold), validates every agent output, and places all orders; money not allocated to stocks is parked in SHV. Run `uv run agent bill_ackman backtesting` (default window `PredefinedWindow.BI_MONTH`); each run writes `reviews.jsonl` (one line per daily review) next to its report. The first run downloads about 5 GB of SEC data once.

'''
text = replace_once(text, "#### 📈 `opening_range_breakout`", section + "#### 📈 `opening_range_breakout`")
readme.write_text(text)
PY
```

Check: `git diff --stat README.md` shows one file changed; `uv run agent 2>&1 | grep bill_ackman` lists the strategy.

- [ ] **Step 7: Run the whole suite, then commit**

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: `2155 passed`.

```bash
git add src/trading_agent_framework/strategies/bill_ackman/agent_bill_ackman.py src/trading_agent_framework/strategies/bill_ackman/__init__.py src/trading_agent_framework/main.py README.md tests/strategies/bill_ackman/test_ackman_strategy.py tests/test_main.py
git commit -m "feat: BillAckmanStrategy, registered as bill_ackman

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: The end-to-end backtest test, `CLAUDE.md` and the final check

**Files:**
- Test: `tests/strategies/bill_ackman/test_ackman_backtest.py` (new)
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the whole strategy, `tests.backtesting.fakes.FakeBacktestDataSource`, `tests.fakes` (`FakeBroker`, `FakeClock`, `et`, `weekday_sessions`), `Strategy.run_backtesting` (the real `BacktestBroker` runs underneath).
- Produces: three end-to-end tests that run a real backtest over 5 sessions with a fake screen and fake agents calling the real submit tools, and the documentation of the strategy's rules.

- [ ] **Step 1: Write the end-to-end tests**

`tests/strategies/bill_ackman/test_ackman_backtest.py`:

```python
"""End to end: a backtest over a few sessions with fake agents (they call the real submit tools) and a fake screen."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.bill_ackman.screen import Candidate, ScreenResult

PRIOR = weekday_sessions(date(2026, 9, 9), 3)  # history before the window: the screen needs a last price on the first simulated day
SESSIONS = weekday_sessions(date(2026, 9, 14), 5)


def _bars(closes: list[float]) -> pd.DataFrame:
    assert len(closes) == len(PRIOR) + len(SESSIONS)
    return pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes, "volume": [1000.0] * len(closes)},
        index=pd.DatetimeIndex([session.close for session in [*PRIOR, *SESSIONS]], name="timestamp"),
    )


def _candidate(symbol: str) -> Candidate:
    return Candidate(
        symbol=symbol, rank=1, score=0.9, sic=5812, market_cap=Decimal("100000000000"), fcf_yield=0.05, fcf_margin=0.2, operating_margin=0.25,
        operating_margin_stdev=0.02, revenue_growth=0.08, net_debt_to_operating_income=1.0, debt_reported=True,
        fiscal_year_end=date(2025, 12, 31), filed=date(2026, 2, 15),
    )


class FakeScreen:
    """AAA is the one candidate, every day; a held AAA is described, any other holding has no data."""

    def __init__(self) -> None:
        self.as_ofs: list[Any] = []
        self.prices: list[Decimal | None] = []

    def run(self, symbols, *, as_of, price_of, top_n=None) -> ScreenResult:  # noqa: ANN001
        self.as_ofs.append(as_of)
        if top_n is None:
            self.prices.append(price_of("AAA"))  # the real screen asks for a price; in a backtest this goes through the no-look-ahead gate
            return ScreenResult(candidates=[_candidate("AAA")], rejections={})
        return ScreenResult(candidates=[_candidate(s) for s in symbols if s == "AAA"], rejections={s: "no_data" for s in symbols if s != "AAA"})


class _Handle:
    def __init__(self, tools: dict[str, Callable[..., dict[str, Any]]], script: Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]) -> None:
        self.tools, self.script, self.runs = tools, script, 0

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None) -> AgentRunResult:
        self.runs += 1
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


def _research(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_ranking"]([{"symbol": sheet["symbol"], "reason": "simple and cash rich"} for sheet in context["candidates"]][:1])


def _judge(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_verdicts"]([{"symbol": entry["fact_sheet"]["symbol"], "verdict": "survive", "reason": "the attack failed"} for entry in context["to_judge"]])


def _trade(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_portfolio"]([{"symbol": entry["symbol"], "weight": 0.3, "reason": "best idea"} for entry in context["allowed"]][:1])


class _Manager:
    def __init__(self) -> None:
        self.handles: dict[str, _Handle] = {}
        self.scripts = {"researcher": _research, "short_seller": _judge, "trader": _trade}

    def create(self, *, name: str, system_prompt: str, tools: list[Callable[..., Any]], **_: Any) -> _Handle:
        self.handles[name] = _Handle({tool.__name__: tool for tool in tools}, self.scripts[name])
        return self.handles[name]

    def __getitem__(self, name: str) -> _Handle:
        return self.handles[name]

    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        return {}  # the runner reads per-agent totals at the end of a run


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _run(tmp_path: Path, manager: _Manager, screen: FakeScreen) -> BillAckmanStrategy:
    source = FakeBacktestDataSource()
    source.set_sessions(SESSIONS)
    for symbol, closes in {"AAA": [49.0, 49.5, 50.0, 50.5, 51.5, 51.0, 52.0, 52.0], "SHV": [100.0] * 8, "SPY": [398.0, 399.0, 400.0, 401.0, 402.0, 403.0, 404.0, 405.0]}.items():
        source.set_bars(Asset(symbol), _bars(closes))
    strategy = BillAckmanStrategy(
        FakeBroker(FakeClock(et(2026, 9, 14, 9, 0)), strategy_name="bill_ackman"),
        mode=TradingMode.BACKTESTING,
        universe=["AAA", "BBB"],
        project_root=tmp_path,
        screen=screen,
    )
    strategy._agents = cast(AgentManager, manager)
    Strategy.run_backtesting(strategy, start=SESSIONS[0].open - timedelta(hours=1), end=SESSIONS[-1].close, data_source=source, budget=Decimal(10000), benchmark="SPY")
    return strategy


def _review_lines(tmp_path: Path) -> list[dict[str, Any]]:
    (path,) = tmp_path.rglob("reviews.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_backtest_reviews_every_session_and_buys_the_trader_s_choice(tmp_path: Path) -> None:
    manager, screen = _Manager(), FakeScreen()

    _run(tmp_path, manager, screen)

    lines = _review_lines(tmp_path)
    assert len(lines) == len(SESSIONS)
    assert [line["date"] for line in lines] == [session.open.date().isoformat() for session in SESSIONS]
    assert not any(line["abandoned"] for line in lines)
    assert {order["symbol"] for order in lines[0]["orders"]} == {"AAA", "SHV"}  # 30% AAA, the rest but the 2% buffer parked
    # Nothing is re-sent on the second day (a backtest fills an order on the next bar, so day one's orders are still open).
    # The open-order accounting itself is pinned by the rebalancer's FakeBroker tests (test_ackman_rebalancer.py), where cash
    # cannot mask it; here BacktestBroker's buying power also nets the pending buys.
    assert lines[1]["orders"] == []
    assert all(manager.handles[name].runs == len(SESSIONS) for name in ("researcher", "short_seller", "trader"))
    assert list(tmp_path.rglob("metrics.json"))  # the run completed and wrote its report


def test_the_screen_is_asked_with_a_market_local_as_of_and_gets_no_look_ahead_prices(tmp_path: Path) -> None:
    screen = FakeScreen()

    _run(tmp_path, _Manager(), screen)

    assert all(as_of.tzinfo is not None and as_of.utcoffset() == timedelta(hours=-4) for as_of in screen.as_ofs)
    assert all(price is not None for price in screen.prices)
    # each session's price is the last close at or before that moment: the previous day's close, never the day's own or a later one
    assert screen.prices[:3] == [Decimal("50.0"), Decimal("50.5"), Decimal("51.5")]
    # from the second session on the screen is asked twice a day: the universe, then the holdings
    assert len(screen.as_ofs) > len(SESSIONS)


def test_a_dead_llm_aborts_a_real_backtest_instead_of_writing_a_flat_report(tmp_path: Path) -> None:
    from trading_agent_framework.utils.errors import AgentError, BacktestError

    manager = _Manager()

    def dead(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
        raise AgentError("llm down")

    manager.scripts["researcher"] = dead

    with pytest.raises(BacktestError, match="reviews abandoned in a row"):
        _run(tmp_path, manager, FakeScreen())

    assert manager.handles["researcher"].runs == 6  # three abandoned reviews, each with its one forced retry
    assert list(tmp_path.rglob("metrics.json")) == []
    assert [line["abandoned"] for line in _review_lines(tmp_path)] == [True, True, True]
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_backtest.py -q`
Expected: `3 passed`. These tests need no new production code: if one fails, the failure is a seam bug between the earlier tasks, to be fixed in the module that owns it (do not weaken the test). What they pin: every session has one review line; the first review buys the trader's choice and the second finds those orders still open and sends nothing; the screen is asked with a market-local `as_of` and gets only the prices a backtest allows; a dead LLM aborts the backtest after three abandoned reviews without writing a report.

- [ ] **Step 3: Update `CLAUDE.md`**

Run from the repository root (it rewrites the `strategies/` and `fundamentals/` architecture bullets, fixes two path references in the quality-screen gotcha, and adds three gotchas):

```bash
uv run python - <<'PY'
"""CLAUDE.md: the strategy, the split of `fundamentals/`, and the gotchas of the Ackman strategy."""

from pathlib import Path

path = Path("CLAUDE.md")
lines = path.read_text().splitlines()


def index_of(prefix: str) -> int:
    matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    assert len(matches) == 1, (prefix, matches)
    return matches[0]


# 1. The strategies bullet gains bill_ackman/.
i = index_of("- `strategies/` -- concrete strategies.")
lines[i] += (
    " `bill_ackman/` (`BillAckmanStrategy`, registered as `\"bill_ackman\"`): a daily researcher → short seller → trader pipeline over the fundamentals quality screen. "
    "`screen/` is the screen (`build_quality_screen(project_root).run(symbols, as_of=, price_of=, top_n=)`: `quality.py` pure gates and score, `annual_figures.py` pure SEC reduction, `annual_store.py` and `splits.py` I/O, `screen.py` wiring). "
    "`ReviewPipeline` runs one review: screen → researcher → short seller → hysteresis → trader → `Rebalancer`. The agents hand structured results to each other through three submit tools (`handoff.py`: `submit_ranking`, `submit_verdicts`, `submit_portfolio`, validated by `HandoffRecorder`) and have NO order tool; `Rebalancer` is the only order code (sells first, buys, then SHV); `StateStore` keeps the fail counters, the last ranking and verdicts and the abandoned streak in `data/bill_ackman_state_<mode>.json`; `ReviewLog` appends one line per review to `reviews.jsonl` in the run directory. "
    "Pure: `fact_sheet`, `hysteresis`, `portfolio`, the validators in `handoff`."
)

# 2. The fundamentals bullet now covers only the SEC client and the generic translation.
i = index_of("- `fundamentals/` --")
lines[i] = (
    "- `fundamentals/` -- SEC EDGAR client, trimmed and ported from lumibot's `SECFundamentals`: `sec.py` (**pure**: tag maps, as-of candidate filtering, statement-period matching, filings parsing, URL building, HTML stripping, `parse_dt`) "
    "and `edgar_client.py` (`SecEdgarClient`, the only module allowed to import `httpx` for SEC access -- cached to `<project_root>/cache/sec/`, rate-limited, requires `SEC_EDGAR_USER_AGENT`; `fetch_json` is its uncached path and HTTP 404 raises `FundamentalsNotFoundError`; "
    "the cached payload getters take optional `as_of`/`max_age_days`, so a payload older than max_age_days by the strategy clock is fetched again (the drill-down tools pass 30) -- a backtest never refetches an existing file, live refreshes monthly), plus `freshness.py` (`is_stale`, the cache-age rule). "
    "The quality screen built on it lives in `strategies/bill_ackman/screen/`."
)

# 3. The as-of gotcha now lives next to the screen's new location.
i = index_of("- **The quality screen's as-of rule is stricter than the fundamentals tools'.**")
assert lines[i].count("`quality.assess` treats a row") == 1
lines[i] = lines[i].replace("`quality.assess` treats a row", "`screen/quality.assess` (`strategies/bill_ackman/screen/`) treats a row")
assert lines[i].count("the row shape in `sec.py` change") == 1
lines[i] = lines[i].replace("the row shape in `sec.py` change", "the row shape in `screen/annual_figures.py` change")

# 4. The gotchas of the Ackman strategy, right after it.
new_gotchas = [
    "- **The Ackman agents never trade, and only their submit tools are trusted.** No agent gets an order, account, indicator or memory tool (`BillAckmanStrategy.initialize` builds each agent's tool list explicitly: researcher = SEC fundamentals + market data + `submit_ranking`, short seller = those + news + `submit_verdicts` instead, trader = `submit_portfolio` only). Each submit tool validates against what the pipeline armed the `HandoffRecorder` with for that stage and returns `{\"error\": ...}` on a violation, so the model can correct itself in the same run; the first valid submission is final. Free text from an agent (including its last message) is logged and ignored. `handoff.py` has no `from __future__ import annotations` (the agent layer reads real annotations).",
    "- **An abandoned Ackman review changes nothing.** `ReviewPipeline._run_stage` runs an agent, then once more with `force_tool=<its submit tool>` and a prompt quoting the last tool error; a stage with no valid submission after that (or a `FundamentalsError` from the screen) abandons the review: nothing is traded, no counter moves, `abandoned_streak` goes up and the reason is written to `reviews.jsonl`. A backtest raises `FatalStrategyError` at `max_consecutive_abandoned` (3) abandoned reviews in a row; paper and live log and carry on. A `ConfigurationError` (no model configured, missing `SEC_EDGAR_USER_AGENT`) is raised while `initialize` builds the screen and the agents and becomes `FatalStrategyError`: the strategy refuses to start and the error never reaches the abandonment logic; a `ConfigurationError` raised by an agent run during a review (not expected in practice) propagates and leaves the state untouched.",
    "- **Ackman hysteresis and open orders.** A holding's `fail` verdict raises its counter, `survive` resets it, and at `forced_exit_fails` (2) consecutive fails code sells it whatever the trader submits (it is not in the trader's allowed set; its counter is kept so a failed sell is forced again). A holding the screen rejects on a quality gate gets the verdict `fail` from code and never reaches the short seller; a data rejection (`no_data`, `no_price`, `no_split_data`, `duplicate_listing`) still does, with a reduced fact sheet. `Rebalancer` counts open orders (a pending buy counts toward the position, a pending sell is deducted and a sell never exceeds what is not already being sold): a backtest fills an order on the next bar, so the daily review always meets yesterday's orders pending. `initialize` wipes the state file in backtesting mode only; paper/live counters survive restarts.",
]
i = index_of("- **The quality screen's as-of rule is stricter than the fundamentals tools'.**")
lines[i + 1 : i + 1] = new_gotchas
path.write_text("\n".join(lines) + "\n")
PY
```

Check: `git diff --stat CLAUDE.md` shows one file changed, and `grep -c "Ackman agents never trade" CLAUDE.md` prints `1`.

- [ ] **Step 4: Final verification**

Run each and read the output:

```bash
uv run ruff check
uv run pytest -q 2>&1 | tail -1
uv run agent 2>&1 | grep bill_ackman
git status --short
git log --oneline main..HEAD
```

Expected: `All checks passed!`; `2158 passed`; `  - bill_ackman`; `git status --short` shows only unrelated files (for example `TODO.md`) and nothing under `cache/`, `data/` or `logs/`; one commit per task since the spec.

- [ ] **Step 5: Commit**

```bash
git add tests/strategies/bill_ackman/test_ackman_backtest.py CLAUDE.md
git commit -m "test: end-to-end Ackman backtest, and the strategy's rules in CLAUDE.md

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: A real run (manual, not automated, not part of the commit)**

This needs the local LLM and the network, so it is for the user to run, and the executor should only prepare it:

1. Create `env/.env.bill_ackman.backtesting` (copy `env/.env.example`) with `LLM_BASE_URL`, `LLM_MODEL`, `SEC_EDGAR_USER_AGENT` and `ALPACA_NEWS_API_KEY`/`ALPACA_NEWS_API_SECRET` (the short seller's news tool). The file is git-ignored.
2. Make sure `data/universe/us_stock_universe.json` exists (`uv run batch-universe`).
3. `uv run python scripts/tests/smoke_quality_screen.py` first: it fills `cache/sec/annual/` for a few symbols and proves SEC and Yahoo are reachable.
4. `uv run agent bill_ackman backtesting`. The first run downloads about 5 GB of SEC data once (10 to 20 minutes) before the first review; each simulated day then costs three agent runs (plus up to three forced retries), so the default two-month window takes hours on a local 27B model. Read `reviews.jsonl` in the run directory (`logs/bill_ackman/backtesting/<run>/`) to see each day's ranking, verdicts, forced exits, targets and orders.

## Post-review amendments

The final whole-branch review led to one fix wave after Task 12 (commits `16d9a0f`..`92977b0`). The spec carries the amended rules; this plan's code blocks show the pre-wave code for these points:

- `Rebalancer` sizes buys from `min(buying power re-read after the sells, cash + this run's proceeds + earlier in-flight credit) - reserve`, and exposes `placed` (spec §5).
- `warmup_trading_days` is 260 so the 12-month return exists from the first simulated day (spec §10).
- `ReviewPipeline.run()` abandons a review on a `BrokerError`/`BacktestError` (stage `broker`, or `execution` with the orders already sent logged) (spec §9); a retry no longer reports the first attempt's tool error.
- `Rebalancer.current_weights()` counts open orders (spec §5.1).
