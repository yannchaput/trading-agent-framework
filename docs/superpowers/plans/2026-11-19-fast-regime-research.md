# Faster Bearish Regime Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Screen seven fast inputs against SPY's falls (phase A), build a faster regime from the winners (phase B), and score it once on a held-out window, without changing the production regime.

**Architecture:** The production state machine in `core/regime.py` is extracted into one shared pure function that both `regime_path` and a new variant use. A new pure module `core/fast_regime.py` computes per-index conditions and per-session labels (prior-day or same-day timing), the screen rows and the selection rule. A research download aligns five extra Yahoo series on SPY's sessions. Two new `uv run regime` subcommands print the screen and the variant's scorecard.

**Tech Stack:** Python 3.14, uv, pytest, yfinance (via the existing `backtesting/data/yahoo.py` helpers), argparse.

**Spec:** `docs/superpowers/specs/2026-11-19-fast-regime-research-design.md`

## Global Constraints

- Worktree `.worktrees/fast-regime`, branch `feat/fast-regime-research`. No merge, push or PR unless the user asks.
- The production regime's output must not change: `regime_path`, `classify_regime`, `regime_by_session`, `uv run regime scorecard` and `recompute` give identical results; every existing `tests/core/test_regime*.py` test stays unchanged and green; `replay disagreements: 0` stays.
- Development window: sessions `2016-01-04` to `2024-06-28`. Held-out window: `2024-07-01` to the last downloaded session. The held-out window is scored only by `regime fast --held-out`, and only when the development check passes.
- Fixed thresholds (spec, not tuned): `vix9d`: `VIX9D / VIX >= 1.0`; `vix_jump`: `VIX[i] / VIX[i-3] - 1 >= 0.20`; `vvix`: `VVIX >= 110`; `credit`: `(HYG/IEF)[i] / (HYG/IEF)[i-10] - 1 <= -0.015`; `short_drawdown`: `SPY[i] <= 0.97 x max(SPY[i-19..i])`; `breadth`: `(RSP/SPY)[i] / (RSP/SPY)[i-10] - 1 <= -0.01`. The four production conditions keep `RegimeParameters()` defaults.
- Development success bar: `mean_on_fall >= -0.03` and `false_bear_days <=` today's regime's on the same window.
- The suite never touches the network (fake downloads/sources, as in `tests/core/`).
- Gates before each commit: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest`, `uv run pytest --cov` (>= 80%), `uv run interrogate -v` (>= 80%), `uv run pyright` on the touched files (the repo has 40 pre-existing errors elsewhere; touched files must add none); then `/ponytail:ponytail-review` on the staged changes.

## Review Focus

- **Timing off-by-one:** a prior-day label for session `j` reads conditions at index `j - 1`; a same-day label reads index `j`. A same-day bug that reads `j + 1` would look like a great signal.
- **The variant with the four production conditions and prior-day timing must equal `regime_by_session` on every session**; any difference means the shared machine or the replay window is wrong.
- **Early indices:** a condition needing `k` closes is False (not an error, not a crash) for `i < k`; ratios never divide by zero (inputs are positive closes, checked once).
- **Gaps in the new inputs:** a NaN left after the 5-session forward fill inside the scored span fails with a clear message naming the symbol; it never becomes a silent False.
- **The held-out guard:** `regime fast --held-out` with a failing development check prints why and exits 1 without scoring the held-out window.

---

### Task 1: Shared state machine in `core/regime.py`

**Files:**
- Modify: `src/trading_agent_framework/core/regime.py`
- Test: `tests/core/test_regime.py`

**Interfaces:**
- Produces: `MachineStep` (frozen dataclass: `regime: int`, `episode_low: float | None`, `blocked: bool`) and `run_machine(spy: Sequence[float], scores: Sequence[int], params: RegimeParameters = _DEFAULTS) -> list[MachineStep]` in `regime.py`: the state machine of `regime_path` (block, bear update, entry, neutral hold), starting fresh at index 0 of the given lists, one step per index. `regime_path` computes its scores and calls `run_machine`; its readings are unchanged.

- [ ] **Step 1: Write the characterization test** in `tests/core/test_regime.py`:

```python
def test_run_machine_reproduces_regime_path() -> None:
    spy = [100 + 15 * math.sin(i / 30) + 0.03 * i for i in range(700)]
    vix = [18 + 12 * math.sin(i / 21) ** 2 for i in range(700)]
    vix3m = [21.0] * 700
    path = regime_path(spy, vix, vix3m, first=DEFAULT.lookback - 1)
    steps = run_machine(spy[DEFAULT.lookback - 1 :], [r.score for r in path])
    assert [(s.regime, s.episode_low, s.blocked) for s in steps] == [(r.regime, r.episode_low, r.blocked) for r in path]
    assert {r.regime for r in path} == {-1, 0, 1}  # the fixture exercises every state
```

(`DEFAULT = RegimeParameters()`; import `math`, `run_machine`, `regime_path` as needed. If the fixture does not reach all three states, adjust its amplitudes until it does, before implementing.)

- [ ] **Step 2: Run it** — `uv run pytest tests/core/test_regime.py -q`. Expected: ImportError on `run_machine`.
- [ ] **Step 3: Implement** `MachineStep` and `run_machine` by moving the loop body's state logic out of `regime_path`; `regime_path` keeps the indicator maths and builds `RegimeReading`s from the steps.
- [ ] **Step 4: Run** `uv run pytest tests/core -q`. Expected: all pass, existing tests unchanged.
- [ ] **Step 5: Gates, review, commit** (add the spec and this plan in the same commit):

```bash
git add src/trading_agent_framework/core/regime.py tests/core/test_regime.py docs/superpowers/specs/2026-11-19-fast-regime-research-design.md docs/superpowers/plans/2026-11-19-fast-regime-research.md
git commit -m "refactor(regime): extract the state machine into run_machine (no behaviour change)"
```

### Task 2: Conditions, labels and the variant regime (`core/fast_regime.py`)

**Files:**
- Create: `src/trading_agent_framework/core/fast_regime.py`
- Test: `tests/core/test_fast_regime.py`

**Interfaces:**
- Consumes: `run_machine`, `RegimeParameters`, `regime_by_session` (Task 1 / existing).
- Produces:
  - `Closes = Mapping[str, Sequence[float]]` keyed `"SPY"`, `"^VIX"`, `"^VIX3M"`, `"^VIX9D"`, `"^VVIX"`, `"HYG"`, `"IEF"`, `"RSP"`, all aligned with one session list.
  - `PRODUCTION = ("inverted", "vix_spike", "below_sma", "off_high")`, `CANDIDATES = ("vix9d", "vix_jump", "vvix", "credit", "short_drawdown", "breadth")`.
  - `conditions(closes: Closes, params: RegimeParameters = RegimeParameters()) -> dict[str, list[bool]]`: one bool per index for every name in `PRODUCTION + CANDIDATES`, using closes up to and including that index, False where the window is not yet full (thresholds in Global Constraints; production ones as in `regime_path`).
  - `raw_labels(sessions: Sequence[date], flags: Sequence[bool], same_day: bool) -> dict[date, int]`: -1 where the flag at index `j` (same-day) or `j - 1` (prior-day) is True, else 0; session 0 is 0 for prior-day.
  - `variant_by_session(sessions: Sequence[date], spy: Sequence[float], flags: Mapping[str, Sequence[bool]], names: Sequence[str], same_day: bool, params: RegimeParameters = RegimeParameters()) -> dict[date, int]`: per session, the score is the count of `names` true at each index; the regime is the last step of `run_machine` over the `params.replay` indices ending at `j - 1` (prior-day) or `j` (same-day); sessions with fewer than `params.min_bars` indices of history get no entry (same coverage as `regime_by_session`).

- [ ] **Step 1: Write the failing tests:**
  - `test_variant_with_production_conditions_equals_today`: on the 700-index fixture of Task 1 (plus constant fillers for the new symbols), `variant_by_session(..., PRODUCTION, same_day=False)` equals `{d: r.regime for d, r in regime_by_session(...).items()}` on every session.
  - `test_same_day_is_prior_day_shifted_by_one`: for the same inputs, `variant(same_day=True)[sessions[j]] == variant(same_day=False)[sessions[j + 1]]` for every `j` both cover.
  - `test_raw_labels_timing`: flags `[F, T, F]` give prior-day `{d0: 0, d1: 0, d2: -1}` and same-day `{d0: 0, d1: -1, d2: 0}`.
  - One test per candidate, on small hand-made series, that the flag turns True exactly at the index the threshold is crossed (e.g. `vix_jump`: VIX `[20, 20, 20, 23.9, 24.0]` -> `[F, F, F, F, T]`), and that indices before the window are False.
- [ ] **Step 2: Run** `uv run pytest tests/core/test_fast_regime.py -q`. Expected: ImportError.
- [ ] **Step 3: Implement** `fast_regime.py` (pure; no pandas; docstrings state units and timing).
- [ ] **Step 4: Run** the new tests and `tests/core -q`. Expected: all pass.
- [ ] **Step 5: Gates, review, commit** — `feat(regime): fast_regime conditions, labels and the variant regime`.

### Task 3: The research download

**Files:**
- Modify: `src/trading_agent_framework/core/regime_data.py` (turn `YahooRegimeData._closes` into a module function `closes_by_session(raw, symbol) -> pd.Series`, kept as the class's implementation; production behaviour unchanged)
- Create: `src/trading_agent_framework/core/fast_regime_data.py`: `RESEARCH_SYMBOLS = ("SPY", "^VIX", "^VIX3M", "^VIX9D", "^VVIX", "HYG", "IEF", "RSP")` and `research_closes(start: date, end: date, download: DownloadFn | None = None, max_fill: int = 5) -> tuple[list[date], dict[str, list[float]]]`
- Test: `tests/core/test_fast_regime_data.py`

**Interfaces:**
- Produces: `research_closes`: one batched download (same call shape as `YahooRegimeData._fetch`), every symbol aligned on SPY's sessions and forward-filled up to `max_fill`; rows before the first session where every symbol has a value are dropped; rows after `end` are dropped; any NaN left in the kept rows raises `RegimeDataError` naming the symbol and its first gap date; an empty download or symbol raises `RegimeDataError`.

- [ ] **Step 1: Write the failing tests** with a fake `download` returning a `group_by="ticker"` frame (build it the way `tests/core/test_regime_data.py` does): alignment on SPY's dates; a 3-day hole in `HYG` is filled; a 7-day hole raises with `"HYG"` in the message; a symbol starting later trims the leading rows; an empty symbol raises.
- [ ] **Step 2: Run** — expected ImportError.
- [ ] **Step 3: Implement**; `YahooRegimeData._closes` calls `closes_by_session`.
- [ ] **Step 4: Run** `tests/core -q` — all pass (production data tests unchanged).
- [ ] **Step 5: Gates, review, commit** — `feat(regime): research download of VIX9D, VVIX, HYG, IEF and RSP`.

### Task 4: Phase A, the screen (`regime screen`)

**Files:**
- Modify: `src/trading_agent_framework/core/fast_regime.py` (screen rows and selection rule)
- Modify: `src/trading_agent_framework/core/regime_cli.py` (subcommand `screen`)
- Test: `tests/core/test_fast_regime.py`, `tests/core/test_regime_cli.py`

**Interfaces:**
- Produces in `fast_regime.py`:
  - `DEV_START = date(2016, 1, 4)`, `DEV_END = date(2024, 6, 28)`, `HELD_OUT_START = date(2024, 7, 1)`.
  - `ScreenRow(name: str, kind: str, card: Scorecard)` with `kind` in `"reference" | "candidate"`.
  - `screen(sessions, closes: Closes, window: Sequence[date], params=RegimeParameters()) -> list[ScreenRow]`: reference rows `today` (production `regime_by_session` regimes), `inverted`, `vix_spike`, `below_sma`, `off_high` (raw prior-day labels); candidate rows `same_day` (`variant_by_session(PRODUCTION, same_day=True)`) and each name in `CANDIDATES` (raw prior-day labels). Each scored with `regime_scorecard.score(window, spy over window, labels)`.
  - `select(rows: Sequence[ScreenRow]) -> tuple[list[str], bool]`: the rule of the spec. Qualify: `false_bear_days <= max(below_sma, off_high false_bear_days)`; winners: the two best `mean_on_fall` among qualifiers (ties: fewer false days, then name order); `same_day` is True when its `mean_on_fall` > `today`'s and its `false_bear_days` <= `today`'s.
- Produces in `regime_cli.py`: `regime screen [--start 2016-01-04] [--dev-end 2024-06-28]`: downloads with `research_closes(start - lookback, dev_end)`, prints one line per row (`name kind mean_on_fall false_bear_days recall changes episodes_caught/total`), then `winners: <a>, <b> | none`, `same-day: yes|no`, and the exact `uv run regime fast ...` command phase B should run (or `result: NO (no input qualifies)` when nothing wins). Exit 0; 1 on `RegimeDataError`.

- [ ] **Step 1: Write the failing tests:** `select` on hand-made `ScreenRow`s: a fast input with too many false days is excluded; the two fastest qualifiers win in order; a tie goes to fewer false days; `same_day` yes/no on both conditions; no qualifier and no same-day -> `([], False)`. CLI: with a fake `research_closes` (monkeypatched in `regime_cli`) on a synthetic 900-session series, `screen` prints every row name and a `winners:` line and exits 0; a `RegimeDataError` exits 1 with `error:`.
- [ ] **Step 2: Run** — expected failures (missing names).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/core -q` — all pass.
- [ ] **Step 5: Gates, review, commit** — `feat(regime): regime screen, phase A of the fast regime research`.

### Task 5: Phase B, the variant (`regime fast`)

**Files:**
- Modify: `src/trading_agent_framework/core/fast_regime.py`, `src/trading_agent_framework/core/regime_cli.py`
- Test: `tests/core/test_fast_regime.py`, `tests/core/test_regime_cli.py`

**Interfaces:**
- Produces in `fast_regime.py`: `variant_names(replace: Sequence[str]) -> list[str]`: `inverted`, `vix_spike`, then two winners replace `below_sma` and `off_high`; one winner replaces `off_high` only; none keeps `PRODUCTION`; a name outside `CANDIDATES` or more than two raises `ValueError`. `passes(card: Scorecard, today: Scorecard) -> bool`: `card.mean_on_fall >= -0.03 and card.false_bear_days <= today.false_bear_days`.
- Produces in `regime_cli.py`: `regime fast [--replace a[,b]] [--same-day] [--held-out]`: prints the variant's names and timing, then the development-window scorecard of the variant and of today's regime side by side (`_print_scorecard` layout for each) and `development check: PASS|FAIL (<reason>)`. With `--held-out`: if the check failed, print `held-out not scored: the development check failed` and exit 1; else print `HELD-OUT (one-time look): ...` and both scorecards on `HELD_OUT_START` to the last session. Bad `--replace` values are a usage error (`parser.error`).

- [ ] **Step 1: Write the failing tests:** `variant_names` cases (0/1/2 winners, unknown name, three names); `passes` boundaries (`-0.03` passes, `-0.0301` fails; equal false days pass). CLI with the fake download: `fast --replace vix9d` prints both scorecards and a `development check:` line; `fast --held-out` on a fixture where the check fails exits 1 and prints no `HELD-OUT`; `--replace nonsense` exits 2.
- [ ] **Step 2: Run** — expected failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/core -q` — all pass.
- [ ] **Step 5: Gates, review, commit** — `feat(regime): regime fast, phase B with the held-out guard`.

### Task 6: The runs and the record

**Files:**
- Create: `docs/superpowers/replays/2026-11-19-fast-regime-research.md`
- Modify: `src/trading_agent_framework/core/CLAUDE.md` (architecture line: `fast_regime.py`, `fast_regime_data.py`, `regime screen|fast`; one line pointing to the record), `CLAUDE.md` (commands: `uv run regime screen` and `uv run regime fast`)

- [ ] **Step 1: Run** `uv run regime screen > logs/regime/screen.txt` (create `logs/regime/`; git-ignored). Expected: exit 0, a `winners:` line.
- [ ] **Step 2: Sanity check** before trusting the screen: the `today` row's `mean_on_fall` and `false_bear_days` must be computed on the development window only (they will differ from the full-window -5.8% / 86); if `today` is missing or the window is wrong, stop and debug.
- [ ] **Step 3: Run phase B exactly as the screen printed it** (or record `NO` and skip to Step 5 if nothing won): `uv run regime fast <printed flags> > logs/regime/fast_dev.txt`.
- [ ] **Step 4: Only if the development check printed PASS**, run the one-time look: `uv run regime fast <same flags> --held-out > logs/regime/fast_held_out.txt`. Never run it twice; never change flags after seeing it.
- [ ] **Step 5: Record** the screen table, the winners, phase B's development scorecard against today's, and (if run) the held-out result, in the replay note and the CLAUDE.md lines. Numbers as read; label 3 held-out falls as weak evidence; no reruns, no new variants.
- [ ] **Step 6: Gates, review, commit** — `docs(regime): fast regime research results`.
