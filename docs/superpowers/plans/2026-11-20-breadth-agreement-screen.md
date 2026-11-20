# Breadth and Agreement Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five candidates (stock breadth, new lows, sector breadth, agreement of 2 and of 3 fast inputs) to the existing `regime screen` / `regime fast` research, run the screen once, and stop before the held-out look.

**Architecture:** The pure maths joins `core/fast_regime.py` (new flags from month lists, per-stock closes, sector closes and the existing flags). The I/O (the research copy of the universe history, the cached stock-closes download, the sector download) lives in a new `strategies/common/regime_breadth_data.py`, because it reuses `download_batch` and the universe history from `strategies/common` and `core` must not import `strategies` at module level; `regime_cli` imports it inside `run_screen`/`run_fast` only. The screen, the selection rule, the variant and the held-out guard are reused unchanged.

**Tech Stack:** Python 3.14, uv, pytest, yfinance through the existing helpers.

**Spec:** `docs/superpowers/specs/2026-11-20-breadth-agreement-screen-design.md` (and `2026-11-19-fast-regime-research-design.md`, whose rule and bar still bind).

## Global Constraints

- Worktree `.worktrees/fast-regime`, branch `feat/fast-regime-research`. No merge, push or PR (the user's explicit instruction).
- Production regime output unchanged; existing tests in `tests/core/test_regime*.py` unchanged and green; the first screen's six candidates, thresholds, selection rule (`select`), bar (`passes`) and windows (`DEV_START 2016-01-04`, `DEV_END 2024-06-28`, `HELD_OUT_START 2024-07-01`) unchanged. Existing `screen()` calls without the new argument keep returning the same 12 rows.
- New names and thresholds (spec, fixed): `stocks_below_20d` (>= 0.75 of eligible stocks with `close[i] < mean(close[i-19..i])`), `new_lows` (>= 0.03 of eligible stocks with `close[i] <= min(close[i-251..i])`), `sectors_below_20d` (>= 8/11 of available sectors below their 20-session mean; available = 20 closes up to `i`), `agree_2` / `agree_3` (>= 2 / >= 3 of `vix9d, vix_jump, vvix, credit, short_drawdown, breadth` true at `i`). Eligible stock: in the list of `i`'s month (`"YYYY-MM"`), a close at `i`, and the window's closes all present; fewer than 100 eligible -> False. A month with no list -> False.
- Sector ETFs: `XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC`; forward fill up to 5 sessions after a sector's first close; leading rows before its first close stay NaN (never trim other series to its start); a gap longer than 5 sessions after the first close raises `RegimeDataError` naming the symbol.
- Stock closes: split- and dividend-adjusted (`closes_by_session`), aligned on SPY's sessions, **no forward fill** (missing = NaN = not eligible that day); download through `batch_us_stock_universe.download_batch` in batches of `HISTORY_BATCH_SIZE`; more than `MAX_ABSENT_FRACTION` (1%) of requested symbols with no data -> `RegimeDataError` naming the count; cache under the cache dir; a request whose end is past the cached end re-downloads the whole range (never appends); reads trim to the requested end.
- Paths: `--universe PATH` (default `<project root>/data/regime_research/us_stock_universe_history.json`) and `--cache-dir PATH` (default `<project root>/cache/regime_research`) on both `screen` and `fast`. A missing file or a month missing from it -> exit 1 with a message naming it; never `ensure_universe_history`.
- No `--held-out` -> nothing (SPY/VIX inputs, sectors, stocks) is downloaded or read past `DEV_END`.
- Suite offline. Gates before each commit: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest`, `uv run pytest --cov` (>= 80%), `uv run interrogate -v` (>= 80%), `uv run pyright <touched files>` (40 pre-existing errors elsewhere; add none); then the implementer runs `/ponytail:ponytail-review` on the staged changes (Skill tool, skill `ponytail:ponytail-review`) before every commit, fixes every "Must fix" and lists the "Should fix" in its report (the user's rule; the last plan skipped it).

## Review Focus

- **Look-ahead through the month list:** the flag at index `i` must use the list of `i`'s own month only, never a later month's.
- **Split artefacts:** an adjusted series with a 2:1 split inside the 252-session window must not flag `new_lows`; an unadjusted one would.
- **Eligibility floor and month gaps:** 99 eligible stocks -> False; a month absent from the lists -> False for every index in it, not an exception and not the previous month's list.
- **Sector availability:** before XLC's 20th close the share is over the 10 available sectors (`count / available >= 8/11`, so 8 of 10 qualifies and 7 of 10 does not); leading NaNs never shorten the session list.
- **Held-out leak via the cache:** a dev-only run after a held-out download must not read rows past `DEV_END` from the cache.

---

### Task 1: Pure breadth and agreement flags in `core/fast_regime.py`

**Files:** Modify `src/trading_agent_framework/core/fast_regime.py`; Test `tests/core/test_fast_regime.py`.

**Interfaces:**
- Produces: `BREADTH_CANDIDATES = ("stocks_below_20d", "new_lows", "sectors_below_20d", "agree_2", "agree_3")`, `ALL_CANDIDATES = CANDIDATES + BREADTH_CANDIDATES`, `SECTORS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC")`, the threshold constants, and
  `breadth_conditions(sessions: Sequence[date], month_lists: Mapping[str, Sequence[str]], stocks: Mapping[str, Sequence[float]], sectors: Mapping[str, Sequence[float]], flags: Mapping[str, Sequence[bool]]) -> dict[str, list[bool]]`: the five flags per index (`stocks`/`sectors` aligned on `sessions`, NaN = no close; `flags` = `conditions()` output, used for the agreement counts).
- Changes: `screen(..., extra_flags: Mapping[str, Sequence[bool]] | None = None)`: with `extra_flags`, the five rows follow the six old candidate rows (prior-day `raw_labels`), 17 rows in all; without it, the same 12 rows as today. `select` ranks names in `ALL_CANDIDATES` (rule unchanged). `variant_names` accepts `ALL_CANDIDATES`. `variant_by_session`'s `flags` argument receives the merged dict (no signature change).

- [ ] **Step 1: Write the failing tests** (small hand-made data, 1-2 tests each): `stocks_below_20d` at exactly 75% true, 74% false; the 100-name floor (99 eligible -> False even at 100% below); a month boundary where the next month's list differs (index `i` uses `i`'s month); a month absent from `month_lists` -> False; a stock with a NaN inside its 20-close window is not eligible that day; `new_lows` at exactly 3% true, below false, and an adjusted series around a 2:1 split (constant adjusted price) does not flag; `sectors_below_20d` share over available sectors only (10 available before XLC has 20 closes) with the 8/11 boundary; `agree_2`/`agree_3` count exactly the six names; `screen` with `extra_flags` returns the 17 names in order and without it the old 12; `select` can pick a breadth name; `variant_names(["agree_2"])` accepted.
- [ ] **Step 2: Run** `uv run pytest tests/core/test_fast_regime.py -q`. Expected: ImportError / failures for the new names.
- [ ] **Step 3: Implement** (pure; no pandas; NaN checks with `math.isnan`; validate that `stocks`/`sectors` series have `len(sessions)` values, `ValueError` otherwise).
- [ ] **Step 4: Run** the new tests, then `uv run pytest tests/core -q`. Expected: all pass, old tests unchanged.
- [ ] **Step 5: Gates, commit** `feat(regime): breadth and agreement candidates for the fast regime screen`.

### Task 2: Breadth research data (`strategies/common/regime_breadth_data.py`)

**Files:** Create `src/trading_agent_framework/strategies/common/regime_breadth_data.py`; Test `tests/strategies/common/test_regime_breadth_data.py`.

**Interfaces:**
- `load_month_lists(path: Path, first: date, last: date) -> dict[str, list[str]]`: the lists of every month from `first`'s to `last`'s, from a `UniverseHistory` file; missing file or any requested month missing -> `RegimeDataError` naming it (the caller asks only for months from `BREADTH_FIRST_MONTH` on; see Task 3).
- `stock_closes(symbols: Sequence[str], sessions: Sequence[date], start: date, end: date, cache_dir: Path, download: DownloadFn | None = None) -> dict[str, list[float]]`: adjusted closes aligned on `sessions` (NaN where absent), via `download_batch` per `HISTORY_BATCH_SIZE`, cached under `cache_dir` with the downloaded range recorded; a cached range covering `[start, end]` is read and trimmed to `end`; otherwise the whole `[start, end]` is downloaded again and the cache replaced; absent symbols counted, more than `MAX_ABSENT_FRACTION` -> `RegimeDataError`.
- `sector_closes(sessions: Sequence[date], start: date, end: date, download: DownloadFn | None = None, max_fill: int = 5) -> dict[str, list[float]]`: the 11 ETFs on `sessions`, the fill and leading-NaN rules of Global Constraints.

- [ ] **Step 1: Write the failing tests** with fake downloads (no network): month lists (all present; one missing month -> error naming `YYYY-MM`; missing file -> error); stock closes (aligned with NaN for a missing day, no fill; second call served from cache without calling the download; a later `end` re-downloads the whole range, not an append; a read with an earlier `end` returns nothing past it; 2 of 100 symbols absent -> error, 1 of 100 -> fine); sectors (XLC starting late keeps leading NaN and full session length; a 3-session hole filled; a 7-session hole raises naming the symbol).
- [ ] **Step 2: Run** — expected ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests and `uv run pytest tests/strategies/common -q`. Expected: all pass.
- [ ] **Step 5: Gates, commit** `feat(regime): research data for breadth: month lists, cached stock closes, sector ETFs`.

### Task 3: Wire the candidates into `regime screen` and `regime fast`

**Files:** Modify `src/trading_agent_framework/core/regime_cli.py`; Test `tests/core/test_regime_cli.py`.

**Interfaces:**
- Consumes Tasks 1-2. `run_screen(start, dev_end, universe: Path, cache_dir: Path)` and `run_fast(replace, same_day, held_out, universe: Path, cache_dir: Path)`: after `research_closes`, load the month lists from `BREADTH_FIRST_MONTH = date(2014, 1, 1)` (a constant in `fast_regime`, the extended history's first month) to the end month (months before it are absent, and `breadth_conditions` gives False there), the stock closes of the union of those lists over `[sessions[0], end]` aligned on `sessions` (the sessions start about 765 calendar days before `DEV_START`, so `new_lows` has its 252 closes from about 2014-12, before the variant's earliest replay window of about 2015-01 and the first scored session 2016-01-04; the spec's 2013-01 download start is therefore not needed — a deviation recorded here), and the sector closes over the sessions; build `extra = breadth_conditions(...)`; `screen(..., extra_flags=extra)` / variant over `conditions(closes) | extra`. Import `regime_breadth_data` inside the functions (comment why: `core` does not import `strategies` at module level). `--universe` and `--cache-dir` flags on both subcommands (defaults in Global Constraints, from `config.env.find_project_root()`). `RegimeDataError` from any loader -> `error: ...`, exit 1.

- [ ] **Step 1: Write the failing tests** with monkeypatched fakes for `research_closes`, `load_month_lists`, `stock_closes`, `sector_closes` (patch the module the CLI imports from): `screen` prints the 17 row names and a verdict; the loaders are called with an end of `dev_end` (and never past it) without `--held-out`; `fast --replace agree_2` runs and prints both scorecards; a missing month from the loader exits 1 with its message; `--universe`/`--cache-dir` reach the loaders.
- [ ] **Step 2: Run** — expected failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/core -q`. Expected: all pass.
- [ ] **Step 5: Gates, commit** `feat(regime): screen and fast read breadth and agreement candidates`.

### Task 4: The runs, then stop

**Files:** Create `docs/superpowers/replays/2026-11-20-breadth-agreement-screen.md` (only on a NO); modify `src/trading_agent_framework/core/CLAUDE.md` and root `CLAUDE.md` (the new flags) in the same commit.

All commands that touch `data/` run against the **main checkout** paths; the research code runs from the worktree with explicit paths. `M=/mnt/data/projets/trading-agent-framework`.

- [ ] **Step 1: Research copy and extension.** `mkdir -p $M/data/regime_research && cp $M/data/universe/us_stock_universe_history.json $M/data/regime_research/`, then from `$M` (main checkout code, its pool cache): `uv run batch-universe --history --start 2014-01 --output-dir data/regime_research`. Then check: every month of the original file is present in the copy with the identical list (a Python one-liner comparing `months[k]` for every original `k`); the copy has months 2014-01..2015-12; `$M/data/universe/us_stock_universe_history.json` is unchanged (`git -C $M status` shows nothing for it; compare its checksum before and after). Any difference -> STOP and report BLOCKED.
- [ ] **Step 2: Screen.** From the worktree: `uv run regime screen --universe $M/data/regime_research/us_stock_universe_history.json --cache-dir $M/cache/regime_research > logs/regime/screen_breadth.txt 2>&1`. Sanity: 17 rows; the first 12 rows equal `logs/regime/screen.txt`'s (same numbers); plausible values. Otherwise STOP and report.
- [ ] **Step 3: Decide by the printed verdict.** (a) `result: NO` -> go to Step 5. (b) Winners `vix_jump, short_drawdown` and same-day yes (the first screen's) -> phase B is not rerun (identical inputs, recorded FAIL); go to Step 5. (c) Otherwise run the printed `next:` command plus the same `--universe`/`--cache-dir` flags, **without** `--held-out`: `> logs/regime/fast_breadth_dev.txt`.
- [ ] **Step 4: If the development check printed PASS: STOP.** Do not run `--held-out`. Report DONE_WITH_CONCERNS to the controller with the development scorecards; the held-out look waits for the user's explicit go-ahead (spec). No record yet.
- [ ] **Step 5: On NO or FAIL, record**: the replay note (question, fixed bar, method, the 17-row table, winners, phase B result or why it was not rerun, the survivor gap and the agreement look-ahead stated, the 2014 extension check), one line in `core/CLAUDE.md`, the two flags in root `CLAUDE.md`'s `regime screen`/`regime fast` lines. Numbers exactly as printed. Gates, commit `docs(regime): breadth and agreement screen results`.
