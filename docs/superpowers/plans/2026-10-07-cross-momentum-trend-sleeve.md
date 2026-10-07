# cross_momentum trend sleeve + 5-weekday protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cross_momentum's SHV-only parking with a trend-filtered SHV/GLD/IEF sleeve, and judge it against the current strategy with one backtest per rebalance weekday.

**Architecture:** Part 2 of the spec lands first on `main`: a pure `weekday_protocol.py` (manifests, resume, comparison, verdict) and a thin CLI in `scripts/experiments/` that runs each weekday in its own subprocess, sequentially. The baseline set then runs from `main` while Part 1 is built in a worktree: a pure `sleeve_weights()` in `utils.py`, two strategy methods that fetch the trend assets' completed daily closes, and `rebalance()`'s single SHV block turned into a loop over the sleeve symbols. The candidate set runs from the worktree and `compare` decides KEEP or REJECT.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff, pandas; the framework's `Strategy.run_backtesting` with `YahooBacktestData`.

**Spec:** `docs/superpowers/specs/2026-10-07-cross-momentum-trend-sleeve-design.md`

## Global Constraints

- Python 3.14, package manager `uv` (`uv run pytest`, `uv run ruff check`); never pip.
- The automated test suite never touches the network; `scripts/` is not collected by pytest.
- Tests use hand-written fakes (`SimpleNamespace`, `tests/fakes.py`), never `MagicMock`.
- The sleeve replaces SHV-only parking outright: no enable flag.
- Unchanged: universe, filters, ranking, weights, the three exposure legs and their `min()`, rank-35 hysteresis, trims, `cash_buffer_pct`, `min_trade_pct`, `_REBALANCE_BAND`, rebalance day and time.
- Config: `"parking": {"symbol": "SHV", "trend_assets": ("GLD", "IEF"), "trend_sma_window": 200, "min_trade_pct": 0.01}`.
- A trend asset is on only when its last completed close is **strictly** above the simple average of its last `trend_sma_window` closes; no hysteresis.
- KEEP only if: mean CAGR candidate > baseline; mean alpha ≥ baseline; mean beta ≤ baseline; mean max drawdown ≥ baseline − 0.01; CAGR wins on ≥ 3 of 5 days.
- Weekday runs are sequential (every backtest rewrites `data/cross_momentum_ptf_history_backtesting.json` and `data/cross_momentum_breadth_backtesting.json`).
- Commit messages follow the repo cadence: `Task N: ...`, then a `Review fix: ...` commit if the review finds something. End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A trend asset's live daily bars include today's partial bar → it must be dropped before the SMA test, as for the stocks (Task 5, `test_a_partial_bar_for_today_is_ignored_by_the_trend_test`).
2. A trend asset's fetch returns empty bars rather than raising → its half goes to SHV with a warning, the rebalance continues (Task 5, `test_empty_bars_for_a_trend_asset_send_its_half_to_shv`).
3. The universe file contains a sleeve symbol (e.g. GLD) → it is never scored, ranked or bought as a stock (Task 5, `test_sleeve_symbols_are_removed_from_the_universe`).
4. A run directory in a manifest was deleted or its run failed → `compare` says which run dir, not a bare `KeyError`/`FileNotFoundError` (Task 1, `test_read_run_metrics_names_a_missing_run_dir`).
5. A set interrupted mid-way is resumed after the universe file was rebuilt → the resume is refused, naming the field (Task 1, `test_mismatches_names_every_differing_field`, used by the CLI in Task 2).

---

### Task 1: Pure weekday protocol module

**Files:**
- Create: `src/trading_agent_framework/strategies/cross_momentum/weekday_protocol.py`
- Test: `tests/strategies/test_cross_momentum_weekday_protocol.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Task 2):
  - `DAY_NAMES: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri")`
  - `@dataclass(frozen=True) RunEntry(run_dir: str, commit: str, dirty: bool)`
  - `@dataclass Manifest(label: str, slippage: float, start: str, end: str, universe_sha256: str, runs: dict[int, RunEntry])`
  - `load_manifest(path: Path) -> Manifest`, `save_manifest(path: Path, manifest: Manifest) -> None`
  - `mismatches(a: Manifest, b: Manifest) -> list[str]` (names of differing fields among `slippage`, `start`, `end`, `universe_sha256`)
  - `days_to_run(manifest: Manifest, days: list[int]) -> list[int]`
  - `@dataclass(frozen=True) RunMetrics(cagr, alpha, beta, max_drawdown, sharpe: float)`
  - `read_run_metrics(run_dir: Path) -> RunMetrics` (raises `ValueError` naming the run dir)
  - `compare(baseline: Manifest, candidate: Manifest, baseline_metrics: dict[int, RunMetrics], candidate_metrics: dict[int, RunMetrics]) -> Comparison` (raises `ValueError`)
  - `Comparison.verdict -> str` (`"KEEP"` / `"REJECT"`), `Comparison.failures: list[str]`, `Comparison.warnings: list[str]`
  - `render(comparison: Comparison) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_cross_momentum_weekday_protocol.py`:

```python
"""The 5-weekday A/B protocol: one backtest per rebalance weekday, judged on the means."""

import json

import pytest

from trading_agent_framework.strategies.cross_momentum.weekday_protocol import (
    Manifest,
    RunEntry,
    RunMetrics,
    compare,
    days_to_run,
    load_manifest,
    mismatches,
    read_run_metrics,
    render,
    save_manifest,
)

DAYS = [0, 1, 2, 3, 4]


def _manifest(label="base", *, days=DAYS, commit="abc", dirty=False, **overrides):
    fields = dict(slippage=0.0, start="2016-01-01T00:00:00-05:00", end="2026-10-05T00:00:00-04:00", universe_sha256="u1")
    fields.update(overrides)
    runs = {day: RunEntry(run_dir=f"/runs/{label}/{day}", commit=commit, dirty=dirty) for day in days}
    return Manifest(label=label, runs=runs, **fields)


def _metrics(cagr=0.30, alpha=0.17, beta=0.86, max_drawdown=-0.25, sharpe=1.2):
    return RunMetrics(cagr=cagr, alpha=alpha, beta=beta, max_drawdown=max_drawdown, sharpe=sharpe)


def _same(metrics, days=DAYS):
    return {day: metrics for day in days}


def _verdict(baseline_metrics, candidate_metrics, **candidate_overrides):
    return compare(_manifest("base"), _manifest("cand", **candidate_overrides), baseline_metrics, candidate_metrics)


def test_keep_when_every_criterion_holds():
    result = _verdict(_same(_metrics()), _same(_metrics(cagr=0.31)))

    assert result.verdict == "KEEP"
    assert result.failures == []
    assert result.cagr_wins == 5


def test_an_equal_mean_cagr_is_a_reject():
    result = _verdict(_same(_metrics()), _same(_metrics()))

    assert result.verdict == "REJECT"
    assert any("CAGR" in failure for failure in result.failures)


def test_alpha_equal_passes_and_just_below_rejects():
    assert _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, alpha=0.17))).verdict == "KEEP"
    result = _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, alpha=0.17 - 1e-6)))
    assert result.verdict == "REJECT"
    assert any("alpha" in failure for failure in result.failures)


def test_beta_equal_passes_and_just_above_rejects():
    assert _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, beta=0.86))).verdict == "KEEP"
    result = _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, beta=0.86 + 1e-6)))
    assert result.verdict == "REJECT"
    assert any("beta" in failure for failure in result.failures)


def test_max_drawdown_may_be_one_point_worse_but_no_more():
    assert _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, max_drawdown=-0.26))).verdict == "KEEP"
    result = _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, max_drawdown=-0.2601)))
    assert result.verdict == "REJECT"
    assert any("drawdown" in failure for failure in result.failures)


def test_the_candidate_must_win_on_at_least_three_days():
    baseline = _same(_metrics())
    two_wins = dict(zip(DAYS, [_metrics(cagr=c) for c in (0.40, 0.40, 0.29, 0.29, 0.29)]))  # mean 0.334 > 0.30
    three_wins = dict(zip(DAYS, [_metrics(cagr=c) for c in (0.40, 0.40, 0.31, 0.29, 0.29)]))

    rejected = _verdict(baseline, two_wins)
    assert rejected.verdict == "REJECT"
    assert rejected.cagr_wins == 2
    assert any("wins" in failure for failure in rejected.failures)
    assert _verdict(baseline, three_wins).verdict == "KEEP"


@pytest.mark.parametrize(
    "overrides",
    [{"start": "2017-01-01T00:00:00-05:00"}, {"end": "2026-09-23T00:00:00-04:00"}, {"slippage": 0.0005}, {"universe_sha256": "u2"}],
)
def test_sets_with_a_different_setup_cannot_be_compared(overrides):
    with pytest.raises(ValueError, match=next(iter(overrides))):
        _verdict(_same(_metrics()), _same(_metrics()), **overrides)


def test_sets_with_different_days_cannot_be_compared():
    with pytest.raises(ValueError, match="days"):
        _verdict(_same(_metrics()), _same(_metrics(), [0, 1, 2]), days=[0, 1, 2])


def test_missing_metrics_for_a_recorded_day_cannot_be_compared():
    with pytest.raises(ValueError, match="metrics"):
        _verdict(_same(_metrics()), _same(_metrics(), [0, 1, 2, 3]))


def test_mixed_commits_and_dirty_runs_are_warned_about_without_changing_the_verdict():
    candidate = _manifest("cand", dirty=True)
    candidate.runs[4] = RunEntry(run_dir="/runs/cand/4", commit="def", dirty=True)

    result = compare(_manifest("base"), candidate, _same(_metrics()), _same(_metrics(cagr=0.31)))

    assert result.verdict == "KEEP"
    assert any("cand" in warning and "commits" in warning for warning in result.warnings)
    assert any("cand" in warning and "dirty" in warning for warning in result.warnings)


def test_mismatches_names_every_differing_field():
    assert mismatches(_manifest(), _manifest()) == []
    assert mismatches(_manifest(), _manifest(slippage=0.0005, universe_sha256="u2")) == ["slippage", "universe_sha256"]


def test_days_to_run_skips_recorded_days_and_keeps_the_requested_order():
    assert days_to_run(_manifest(days=[1, 3]), [4, 0, 1, 2, 3]) == [4, 0, 2]
    assert days_to_run(_manifest(days=DAYS), DAYS) == []


def test_a_manifest_round_trips_through_json(tmp_path):
    path = tmp_path / "experiments" / "base.json"
    manifest = _manifest(days=[0, 2])

    save_manifest(path, manifest)

    assert load_manifest(path) == manifest
    assert sorted(json.loads(path.read_text())["runs"]) == ["0", "2"]


def test_read_run_metrics_reads_the_five_figures(tmp_path):
    figures = {"cagr_strategy": 0.32, "alpha": 0.17, "beta": 0.86, "max_drawdown_strategy": -0.25, "sharpe_strategy": 1.25, "other": 1}
    (tmp_path / "metrics.json").write_text(json.dumps(figures))

    assert read_run_metrics(tmp_path) == _metrics(cagr=0.32, alpha=0.17, beta=0.86, max_drawdown=-0.25, sharpe=1.25)


def test_read_run_metrics_names_a_missing_run_dir(tmp_path):
    with pytest.raises(ValueError, match=str(tmp_path / "gone")):
        read_run_metrics(tmp_path / "gone")


def test_read_run_metrics_names_a_missing_figure(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"cagr_strategy": 0.32}))

    with pytest.raises(ValueError, match="beta"):
        read_run_metrics(tmp_path)


def test_render_shows_both_labels_the_verdict_and_the_failures():
    text = render(_verdict(_same(_metrics()), _same(_metrics())))

    assert "base" in text and "cand" in text
    assert "REJECT" in text
    assert "CAGR" in text
    assert "Tue" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_weekday_protocol.py -v`
Expected: collection error, `ModuleNotFoundError: ... weekday_protocol`.

- [ ] **Step 3: Write the module**

Create `src/trading_agent_framework/strategies/cross_momentum/weekday_protocol.py`:

```python
"""The 5-weekday A/B protocol for cross_momentum backtests.

One backtest cannot see an effect smaller than the spread between rebalance weekdays (the same code scored a CAGR
of 32.2% on Tuesday and 29.6% on Wednesday), so a change is judged on the means of one run per weekday. This
module records a set of runs (a manifest) and compares two sets; `scripts/experiments/cross_momentum_weekdays.py`
runs them. Pure apart from the manifest and metrics JSON files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import fmean

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri")

# KEEP thresholds (spec "Success bar"). Max drawdown is negative: the candidate's mean may be this much lower.
MDD_TOLERANCE = 0.01
MIN_CAGR_WINS = 3
_EPSILON = 1e-12  # float slack, so a value equal to its threshold counts as equal

# The setup two sets must share to be compared, and a resumed set must keep
_SETUP_FIELDS = ("slippage", "start", "end", "universe_sha256")

_METRIC_KEYS = {
    "cagr": "cagr_strategy",
    "alpha": "alpha",
    "beta": "beta",
    "max_drawdown": "max_drawdown_strategy",
    "sharpe": "sharpe_strategy",
}


@dataclass(frozen=True)
class RunEntry:
    """One recorded backtest: its absolute run directory and the code it ran."""

    run_dir: str
    commit: str
    dirty: bool


@dataclass
class Manifest:
    """A set of runs of one variant, keyed by `day_of_week` (0 = Monday)."""

    label: str
    slippage: float
    start: str
    end: str
    universe_sha256: str
    runs: dict[int, RunEntry] = field(default_factory=dict)


def load_manifest(path: Path) -> Manifest:
    data = json.loads(Path(path).read_text())
    runs = {int(day): RunEntry(**entry) for day, entry in data.pop("runs").items()}
    return Manifest(runs=runs, **data)


def save_manifest(path: Path, manifest: Manifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: getattr(manifest, name) for name in ("label", *_SETUP_FIELDS)}
    data["runs"] = {str(day): vars(entry) for day, entry in sorted(manifest.runs.items())}
    path.write_text(json.dumps(data, indent=2) + "\n")


def mismatches(a: Manifest, b: Manifest) -> list[str]:
    """The setup fields (slippage, window, universe hash) on which two manifests differ."""
    return [name for name in _SETUP_FIELDS if getattr(a, name) != getattr(b, name)]


def days_to_run(manifest: Manifest, days: list[int]) -> list[int]:
    """The requested days not recorded yet, in the requested order: a set interrupted mid-way resumes."""
    return [day for day in days if day not in manifest.runs]


@dataclass(frozen=True)
class RunMetrics:
    cagr: float
    alpha: float
    beta: float
    max_drawdown: float
    sharpe: float


def read_run_metrics(run_dir: Path) -> RunMetrics:
    """The five compared figures from a run's `metrics.json`."""
    path = Path(run_dir) / "metrics.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"no metrics.json in {run_dir}: the run was deleted or did not finish") from exc
    missing = [key for key in _METRIC_KEYS.values() if data.get(key) is None]
    if missing:
        raise ValueError(f"{path} has no {', '.join(missing)}")
    return RunMetrics(**{name: float(data[key]) for name, key in _METRIC_KEYS.items()})


@dataclass(frozen=True)
class DayRow:
    day: int
    baseline: RunMetrics
    candidate: RunMetrics


@dataclass(frozen=True)
class Comparison:
    baseline_label: str
    candidate_label: str
    rows: list[DayRow]
    baseline_mean: RunMetrics
    candidate_mean: RunMetrics
    cagr_wins: int
    warnings: list[str]
    failures: list[str]

    @property
    def verdict(self) -> str:
        return "REJECT" if self.failures else "KEEP"


def _mean(metrics: list[RunMetrics]) -> RunMetrics:
    return RunMetrics(**{f.name: fmean(getattr(m, f.name) for m in metrics) for f in fields(RunMetrics)})


def _warnings(manifest: Manifest) -> list[str]:
    warnings = []
    commits = sorted({entry.commit for entry in manifest.runs.values()})
    if len(commits) > 1:
        warnings.append(f"{manifest.label}: runs span commits {', '.join(commits)}")
    dirty = [DAY_NAMES[day] for day, entry in sorted(manifest.runs.items()) if entry.dirty]
    if dirty:
        warnings.append(f"{manifest.label}: {', '.join(dirty)} ran on a dirty tree")
    return warnings


def compare(
    baseline: Manifest,
    candidate: Manifest,
    baseline_metrics: dict[int, RunMetrics],
    candidate_metrics: dict[int, RunMetrics],
) -> Comparison:
    """Judge the candidate set against the baseline set (spec "Success bar").

    Raises:
        ValueError: the sets differ in setup or days, or a recorded day has no metrics.
    """
    for name in mismatches(baseline, candidate):
        raise ValueError(f"the two sets differ in {name}: {getattr(baseline, name)!r} vs {getattr(candidate, name)!r}")
    if set(baseline.runs) != set(candidate.runs):
        raise ValueError(f"the two sets cover different days: {sorted(baseline.runs)} vs {sorted(candidate.runs)}")
    for manifest, metrics in ((baseline, baseline_metrics), (candidate, candidate_metrics)):
        if set(metrics) != set(manifest.runs):
            raise ValueError(f"{manifest.label}: metrics for days {sorted(metrics)}, runs for days {sorted(manifest.runs)}")

    rows = [DayRow(day, baseline_metrics[day], candidate_metrics[day]) for day in sorted(baseline.runs)]
    base = _mean([row.baseline for row in rows])
    cand = _mean([row.candidate for row in rows])
    wins = sum(1 for row in rows if row.candidate.cagr > row.baseline.cagr)

    failures = []
    if not cand.cagr > base.cagr + _EPSILON:
        failures.append(f"mean CAGR {cand.cagr:.4f} is not above {base.cagr:.4f}")
    if cand.alpha < base.alpha - _EPSILON:
        failures.append(f"mean alpha {cand.alpha:.4f} is below {base.alpha:.4f}")
    if cand.beta > base.beta + _EPSILON:
        failures.append(f"mean beta {cand.beta:.4f} is above {base.beta:.4f}")
    if cand.max_drawdown < base.max_drawdown - MDD_TOLERANCE - _EPSILON:
        failures.append(f"mean max drawdown {cand.max_drawdown:.4f} is more than {MDD_TOLERANCE:.0%} below {base.max_drawdown:.4f}")
    if wins < MIN_CAGR_WINS:
        failures.append(f"CAGR wins on {wins} of {len(rows)} days, {MIN_CAGR_WINS} needed")

    return Comparison(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        rows=rows,
        baseline_mean=base,
        candidate_mean=cand,
        cagr_wins=wins,
        warnings=_warnings(baseline) + _warnings(candidate),
        failures=failures,
    )


def _cells(base: RunMetrics, cand: RunMetrics) -> str:
    return "".join(f"{getattr(base, f.name):>9.4f} {getattr(cand, f.name):>8.4f}" for f in fields(RunMetrics))


def render(comparison: Comparison) -> str:
    """The per-day table, the means and their deltas, then the verdict, as plain text."""
    names = [f.name for f in fields(RunMetrics)]
    lines = [
        f"baseline = {comparison.baseline_label}, candidate = {comparison.candidate_label} (each cell: baseline candidate)",
        "",
        f"{'day':<6}" + "".join(f"{name:>18}" for name in names),
    ]
    for row in comparison.rows:
        lines.append(f"{DAY_NAMES[row.day]:<6}" + _cells(row.baseline, row.candidate))
    lines.append(f"{'mean':<6}" + _cells(comparison.baseline_mean, comparison.candidate_mean))
    deltas = [getattr(comparison.candidate_mean, name) - getattr(comparison.baseline_mean, name) for name in names]
    lines.append(f"{'delta':<6}" + "".join(f"{delta:>+18.4f}" for delta in deltas))
    lines.append("")
    lines.append(f"CAGR wins: {comparison.cagr_wins} of {len(comparison.rows)}")
    lines.extend(f"warning: {warning}" for warning in comparison.warnings)
    lines.append(f"verdict: {comparison.verdict}")
    lines.extend(f"  - {failure}" for failure in comparison.failures)
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/test_cross_momentum_weekday_protocol.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/trading_agent_framework/strategies/cross_momentum/weekday_protocol.py tests/strategies/test_cross_momentum_weekday_protocol.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/weekday_protocol.py tests/strategies/test_cross_momentum_weekday_protocol.py
git commit -m "Task 1: cross_momentum 5-weekday protocol: manifests, resume, comparison and verdict

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Weekday CLI

**Files:**
- Create: `scripts/experiments/cross_momentum_weekdays.py`
- Modify: `CLAUDE.md` (Commands block)

**Interfaces:**
- Consumes (Task 1): `DAY_NAMES`, `Manifest`, `RunEntry`, `load_manifest`, `save_manifest`, `mismatches`, `days_to_run`, `read_run_metrics`, `compare`, `render`.
- Produces: the commands `run --label <name> [--slippage S] [--days 0,1,2,3,4]` and `compare <baseline.json> <candidate.json>` (exit 0 KEEP, 1 REJECT, 2 cannot compare); manifests at `logs/cross_momentum/experiments/<label>.json`.

The CLI is not covered by pytest (like `scripts/tests/`); its logic lives in Task 1. Each weekday runs in its own subprocess (`one --day D`), so logging handlers, thread pools and the yfinance session never carry from one run to the next.

- [ ] **Step 1: Write the script**

Create `scripts/experiments/cross_momentum_weekdays.py`:

```python
"""Run cross_momentum once per rebalance weekday, or compare two such sets.

    uv run python scripts/experiments/cross_momentum_weekdays.py run --label shv-baseline
    uv run python scripts/experiments/cross_momentum_weekdays.py compare BASELINE.json CANDIDATE.json

`run` records each backtest in logs/cross_momentum/experiments/<label>.json and resumes a set interrupted
mid-way. Days run one after another, each in its own process: every backtest rewrites
data/cross_momentum_*_backtesting.json, so two at once would corrupt each other. Run it from the project root.
The verdict rules are in trading_agent_framework.strategies.cross_momentum.weekday_protocol.
"""

import argparse
import hashlib
import logging
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from trading_agent_framework.config import TradingMode, find_project_root, load_strategy_env
from trading_agent_framework.strategies.cross_momentum.weekday_protocol import (
    DAY_NAMES,
    Manifest,
    RunEntry,
    compare,
    days_to_run,
    load_manifest,
    mismatches,
    read_run_metrics,
    render,
    save_manifest,
)

STRATEGY = "cross_momentum"
UNIVERSE_FILE = Path("data/universe/us_stock_universe.json")
EXPERIMENTS_DIR = Path("logs/cross_momentum/experiments")
RUN_DIR_PREFIX = "RUN_DIR="


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _days(value: str) -> list[int]:
    days = [int(part) for part in value.split(",")]
    if any(day not in range(5) for day in days) or len(set(days)) != len(days):
        raise argparse.ArgumentTypeError("days are distinct integers from 0 (Monday) to 4 (Friday)")
    return days


def _require_project_root() -> Path:
    root = find_project_root()
    if Path.cwd().resolve() != root.resolve():
        sys.exit(f"Run this from the project root ({root}): the strategy reads data/ relative to the working directory.")
    return root


def _one(args: argparse.Namespace) -> int:
    """One backtest with `day_of_week = args.day`; prints its run directory on the last line."""
    from trading_agent_framework.backtesting.placeholder import PlaceholderBroker
    from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy
    from trading_agent_framework.strategies.cross_momentum.utils import load_cross_momentum_universe

    root = _require_project_root()
    logging.basicConfig(level=logging.INFO)
    load_strategy_env(STRATEGY, TradingMode.BACKTESTING.value, root)
    universe = load_cross_momentum_universe()
    if not universe:
        sys.exit("Universe file not found: run `uv run batch-universe` first.")
    strategy = CrossMomentumStrategy(
        broker=PlaceholderBroker(STRATEGY),
        mode=TradingMode.BACKTESTING,
        universe=universe,
        parameters={"day_of_week": args.day},
    )
    result = strategy.run_backtesting(slippage=Decimal(str(args.slippage)))
    print(f"{RUN_DIR_PREFIX}{Path(result.run_dir).resolve()}", flush=True)
    return 0


def _run(args: argparse.Namespace) -> int:
    from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy

    _require_project_root()
    if not UNIVERSE_FILE.exists():
        sys.exit("Universe file not found: run `uv run batch-universe` first.")
    fresh = Manifest(
        label=args.label,
        slippage=args.slippage,
        start=CrossMomentumStrategy.parameters["backtesting_start"].isoformat(),
        end=CrossMomentumStrategy.parameters["backtesting_end"].isoformat(),
        universe_sha256=hashlib.sha256(UNIVERSE_FILE.read_bytes()).hexdigest(),
    )
    path = EXPERIMENTS_DIR / f"{args.label}.json"
    manifest = fresh
    if path.exists():
        manifest = load_manifest(path)
        differing = mismatches(manifest, fresh)
        if differing:
            details = ", ".join(f"{name} {getattr(manifest, name)!r} -> {getattr(fresh, name)!r}" for name in differing)
            sys.exit(f"{path} was recorded with a different setup ({details}); use a new label.")

    days = days_to_run(manifest, args.days)
    if not days:
        print(f"{path}: every requested day is already recorded")
        return 0
    for day in days:
        print(f"[{args.label}] {DAY_NAMES[day]}: running", flush=True)
        command = [sys.executable, __file__, "one", "--day", str(day), "--slippage", str(args.slippage)]
        completed = subprocess.run(command, capture_output=True, text=True)
        run_dirs = [line[len(RUN_DIR_PREFIX):] for line in completed.stdout.splitlines() if line.startswith(RUN_DIR_PREFIX)]
        if completed.returncode != 0 or not run_dirs:
            sys.stderr.write(completed.stderr[-4000:])
            sys.exit(f"[{args.label}] {DAY_NAMES[day]}: the backtest failed (exit {completed.returncode}); the days before it are saved in {path}")
        manifest.runs[day] = RunEntry(run_dir=run_dirs[-1], commit=_git("rev-parse", "HEAD"), dirty=bool(_git("status", "--porcelain")))
        save_manifest(path, manifest)
        print(f"[{args.label}] {DAY_NAMES[day]}: {run_dirs[-1]}", flush=True)
    print(f"Manifest: {path.resolve()}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline = load_manifest(Path(args.baseline))
    candidate = load_manifest(Path(args.candidate))
    try:
        baseline_metrics = {day: read_run_metrics(Path(entry.run_dir)) for day, entry in baseline.runs.items()}
        candidate_metrics = {day: read_run_metrics(Path(entry.run_dir)) for day, entry in candidate.runs.items()}
        comparison = compare(baseline, candidate, baseline_metrics, candidate_metrics)
    except ValueError as exc:
        print(f"Cannot compare: {exc}", file=sys.stderr)
        return 2
    print(render(comparison))
    return 0 if comparison.verdict == "KEEP" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run (or resume) a set of weekday backtests")
    run.add_argument("--label", required=True, help="the set's name: logs/cross_momentum/experiments/<label>.json")
    run.add_argument("--slippage", type=float, default=0.0, help="per-trade slippage passed to run_backtesting (default 0)")
    run.add_argument("--days", type=_days, default=[0, 1, 2, 3, 4], help="comma-separated weekdays, 0 = Monday (default 0,1,2,3,4)")
    run.set_defaults(handler=_run)

    one = commands.add_parser("one", help=argparse.SUPPRESS)
    one.add_argument("--day", type=int, required=True)
    one.add_argument("--slippage", type=float, default=0.0)
    one.set_defaults(handler=_one)

    comparison = commands.add_parser("compare", help="judge a candidate set against a baseline set")
    comparison.add_argument("baseline", help="the baseline manifest (JSON path)")
    comparison.add_argument("candidate", help="the candidate manifest (JSON path)")
    comparison.set_defaults(handler=_compare)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Check the help and the argument validation**

Run: `uv run python scripts/experiments/cross_momentum_weekdays.py run --help`
Expected: usage listing `--label`, `--slippage`, `--days`.

Run: `uv run python scripts/experiments/cross_momentum_weekdays.py run --label x --days 1,7`
Expected: exits 2 with `days are distinct integers from 0 (Monday) to 4 (Friday)`; no `logs/cross_momentum/experiments/x.json` is created.

- [ ] **Step 3: Smoke-test `compare` on two hand-made one-day manifests**

They point at the two existing decade runs (Tuesday `2026-10-07_214307` vs Wednesday `2026-10-07_074747`, both day 1 here on purpose, so this is a mechanics check, not a result).

```bash
S=/tmp/claude-1000/-home-yann-projets-trading-agent-framework/1c83972f-155c-4c2b-83fc-333727ef8fc3/scratchpad
R=$(pwd)/logs/cross_momentum/backtesting
cat > $S/a.json <<EOF
{"label": "a", "slippage": 0.0, "start": "s", "end": "e", "universe_sha256": "u", "runs": {"1": {"run_dir": "$R/2026-10-07_214307_backtesting", "commit": "c", "dirty": false}}}
EOF
cat > $S/b.json <<EOF
{"label": "b", "slippage": 0.0, "start": "s", "end": "e", "universe_sha256": "u", "runs": {"1": {"run_dir": "$R/2026-10-07_074747_backtesting", "commit": "c", "dirty": false}}}
EOF
uv run python scripts/experiments/cross_momentum_weekdays.py compare $S/a.json $S/b.json; echo "exit $?"
```

Expected: a one-row table (`Tue`, CAGR `0.3217  0.2963`), `CAGR wins: 0 of 1`, `verdict: REJECT` with the CAGR and wins failures, `exit 1`.

- [ ] **Step 4: Document the command in `CLAUDE.md`**

In the ```` ```bash ```` Commands block, after the `uv run python scripts/tests/smoke_ibkr_orders.py` line, add:

```bash
uv run python scripts/experiments/cross_momentum_weekdays.py run --label NAME   # cross_momentum backtest on each rebalance weekday, sequentially (~65 min); `compare A.json B.json` gives KEEP/REJECT
```

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check scripts/experiments/cross_momentum_weekdays.py`
Expected: `All checks passed!`

```bash
git add scripts/experiments/cross_momentum_weekdays.py CLAUDE.md
git commit -m "Task 2: cross_momentum weekday CLI: run a set sequentially, one process per day, and compare two sets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Run the baseline set from `main` (operational, ~65 min)

No code. It needs Tasks 1–2 committed and a **clean** tree on `main` (the manifest records `git status`).

- [ ] **Step 1: Check the tree is clean and on main**

Run: `git status --short && git branch --show-current`
Expected: no file listed, `main`.

- [ ] **Step 2: Start the set in the background**

```bash
mkdir -p logs/cross_momentum/experiments
nohup uv run python scripts/experiments/cross_momentum_weekdays.py run --label shv-baseline > logs/cross_momentum/experiments/shv-baseline.out 2>&1 &
```

- [ ] **Step 3: Do not touch `main`'s working tree, `data/` or the universe file while it runs.** Task 4 continues in a worktree. Do not run `uv run batch-universe` until both sets are done: a new universe file changes the hash and `compare` refuses the pair.

- [ ] **Step 4: When it ends, check the output**

Run: `tail -5 logs/cross_momentum/experiments/shv-baseline.out && python3 -m json.tool logs/cross_momentum/experiments/shv-baseline.json`
Expected: `Manifest: .../shv-baseline.json`; five runs (keys "0" to "4"), all `"dirty": false`, one commit. If a day failed, rerun the same command: recorded days are skipped.

---

### Task 4: Sleeve config and pure trend functions (in a worktree)

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/parameters.py:47-53`
- Modify: `src/trading_agent_framework/strategies/cross_momentum/utils.py` (new functions after `breadth_exposure`, line ~336)
- Test: `tests/strategies/test_cross_momentum_sleeve.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Task 5):
  - `trend_reading(closes: list[float], sma_window: int) -> tuple[float, float] | None` (last close, SMA; `None` when unusable)
  - `sleeve_weights(closes_by_asset: dict[str, list[float]], trend_assets: tuple[str, ...], sma_window: int, fallback: str) -> dict[str, float]` (keys in order: `fallback`, then the trend assets)
  - `sleeve_symbols(parking: dict) -> tuple[str, ...]` (`(parking["symbol"], *parking["trend_assets"])`)
  - `CONFIG["parking"]["trend_assets"] == ("GLD", "IEF")`, `CONFIG["parking"]["trend_sma_window"] == 200`

- [ ] **Step 1: Create the worktree** (use superpowers:using-git-worktrees; branch `feature/cross-momentum-trend-sleeve` from `main` with Tasks 1–2). Then copy the gitignored files a backtest and the tests need, and install:

```bash
MAIN=/home/yann/projets/trading-agent-framework
mkdir -p data/universe env
cp $MAIN/data/universe/us_stock_universe.json data/universe/
cp $MAIN/env/.env.cross_momentum.backtesting env/
uv sync
uv run pytest -q
```

Expected: the suite passes before any change.

- [ ] **Step 2: Write the failing tests**

Create `tests/strategies/test_cross_momentum_sleeve.py`:

```python
"""The parking sleeve: GLD and IEF each hold their share while they trend, SHV holds the rest."""

import math

import pytest

from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.strategies.cross_momentum.utils import sleeve_symbols, sleeve_weights, trend_reading

ASSETS = ("GLD", "IEF")
RISING, FALLING = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]  # last close above / below the 3-day SMA (2.0)


def _weights(closes_by_asset, assets=ASSETS):
    return sleeve_weights(closes_by_asset, assets, 3, "SHV")


def test_the_config_names_the_sleeve():
    parking = CONFIG["parking"]
    assert parking["symbol"] == "SHV"
    assert parking["trend_assets"] == ("GLD", "IEF")
    assert parking["trend_sma_window"] == 200
    assert parking["min_trade_pct"] == 0.01


def test_sleeve_symbols_are_the_fallback_then_the_trend_assets():
    assert sleeve_symbols({"symbol": "SHV", "trend_assets": ("GLD", "IEF")}) == ("SHV", "GLD", "IEF")


def test_trend_reading_is_the_last_close_and_the_sma_of_the_last_window_closes():
    assert trend_reading(RISING, 3) == (3.0, 2.0)
    assert trend_reading([100.0, *RISING], 3) == (3.0, 2.0)


@pytest.mark.parametrize("closes", [[1.0, 2.0], [], [1.0, 2.0, math.nan], [math.inf, 2.0, 3.0]])
def test_trend_reading_is_none_when_unusable(closes):
    assert trend_reading(closes, 3) is None


def test_both_trending_split_the_sleeve():
    assert _weights({"GLD": RISING, "IEF": RISING}) == {"SHV": 0.0, "GLD": 0.5, "IEF": 0.5}


def test_one_trending_takes_its_half_and_shv_the_other():
    assert _weights({"GLD": RISING, "IEF": FALLING}) == {"SHV": 0.5, "GLD": 0.5, "IEF": 0.0}


def test_none_trending_is_all_shv():
    assert _weights({"GLD": FALLING, "IEF": FALLING}) == {"SHV": 1.0, "GLD": 0.0, "IEF": 0.0}


def test_a_close_equal_to_its_sma_is_off():
    assert _weights({"GLD": [2.0, 2.0, 2.0], "IEF": RISING}) == {"SHV": 0.5, "GLD": 0.0, "IEF": 0.5}


@pytest.mark.parametrize("gld", [None, [2.0, 3.0], [1.0, 2.0, math.nan]])
def test_missing_short_or_nan_history_is_off(gld):
    closes = {"IEF": RISING} if gld is None else {"GLD": gld, "IEF": RISING}

    assert _weights(closes) == {"SHV": 0.5, "GLD": 0.0, "IEF": 0.5}


def test_no_trend_assets_is_all_shv():
    assert _weights({}, assets=()) == {"SHV": 1.0}


@pytest.mark.parametrize("gld", [RISING, FALLING])
@pytest.mark.parametrize("ief", [RISING, FALLING])
@pytest.mark.parametrize("tlt", [RISING, FALLING])
def test_weights_always_sum_to_one(gld, ief, tlt):
    weights = sleeve_weights({"GLD": gld, "IEF": ief, "TLT": tlt}, ("GLD", "IEF", "TLT"), 3, "SHV")

    assert sum(weights.values()) == pytest.approx(1.0)
    assert list(weights) == ["SHV", "GLD", "IEF", "TLT"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_sleeve.py -v`
Expected: collection error, `ImportError: cannot import name 'sleeve_symbols'`.

- [ ] **Step 4: Update the config**

In `parameters.py`, replace the parking block (lines 47–53) with:

```python
    # ── Parking sleeve ─────────────────────────
    # Capital the exposure legs take out of stocks is parked in this sleeve instead of idle cash. Each trend asset
    # holds 1/len(trend_assets) of it while its last completed close is above its trend_sma_window-day SMA; SHV
    # (`symbol`) holds the rest. Sleeve trades smaller than min_trade_pct of the portfolio are skipped (per-order fees).
    "parking": {
        "symbol": "SHV",
        "trend_assets": ("GLD", "IEF"),
        "trend_sma_window": 200,
        "min_trade_pct": 0.01,
    },
```

- [ ] **Step 5: Add the pure functions**

In `utils.py`, after `breadth_exposure` (before `load_breadth_step`), add:

```python
def sleeve_symbols(parking: dict) -> tuple[str, ...]:
    """Every symbol of the parking sleeve: the fallback (SHV) first, then the trend assets."""
    return (parking["symbol"], *parking["trend_assets"])


def trend_reading(closes: list[float], sma_window: int) -> tuple[float, float] | None:
    """(last close, simple average of the last `sma_window` closes), or None with too few or non-finite values."""
    if len(closes) < sma_window:
        return None
    last = closes[-1]
    sma = sum(closes[-sma_window:]) / sma_window
    if not (math.isfinite(last) and math.isfinite(sma)):
        return None
    return last, sma


def sleeve_weights(
    closes_by_asset: dict[str, list[float]],
    trend_assets: tuple[str, ...],
    sma_window: int,
    fallback: str,
) -> dict[str, float]:
    """Share of the parking sleeve for the fallback and each trend asset; the shares sum to 1.0.

    Each trend asset gets 1/len(trend_assets) while its last close is strictly above its SMA; otherwise (or when
    its closes are missing, too short or non-finite) that share goes to `fallback`. No hysteresis: the weekly
    cadence, the rebalance band and the minimum trade already damp a close hovering at its SMA.
    """
    if not trend_assets:
        return {fallback: 1.0}
    share = 1.0 / len(trend_assets)
    weights = {fallback: 0.0, **{asset: 0.0 for asset in trend_assets}}
    for asset in trend_assets:
        reading = trend_reading(closes_by_asset.get(asset, []), sma_window)
        if reading is not None and reading[0] > reading[1]:
            weights[asset] = share
        else:
            weights[fallback] += share
    return weights
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/test_cross_momentum_sleeve.py -v`
Expected: all PASS.

- [ ] **Step 7: Run the whole cross_momentum suite** (the config change must not break the rebalance tests yet: their fake builds its own `parking` dict)

Run: `uv run pytest tests/strategies/ -q -k cross_momentum`
Expected: all PASS.

- [ ] **Step 8: Lint and commit**

Run: `uv run ruff check src/trading_agent_framework/strategies/cross_momentum tests/strategies/test_cross_momentum_sleeve.py`
Expected: `All checks passed!`

```bash
git add src/trading_agent_framework/strategies/cross_momentum/parameters.py src/trading_agent_framework/strategies/cross_momentum/utils.py tests/strategies/test_cross_momentum_sleeve.py
git commit -m "Task 4: cross_momentum sleeve config and pure trend weights (GLD/IEF above their 200d SMA, SHV otherwise)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Wire the sleeve into the strategy

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (module docstring, imports, class docstring, `__init__`, new `_trend_closes` / `_sleeve_weights`, `rebalance`, `_backtest_preload_assets`)
- Modify: `tests/strategies/test_cross_momentum_rebalance.py` (fake, preload test, new tests)
- Modify: `README.md` (cross_momentum section)

**Interfaces:**
- Consumes (Task 4): `sleeve_symbols(parking)`, `trend_reading(closes, sma_window)`, `sleeve_weights(closes_by_asset, trend_assets, sma_window, fallback)`.
- Produces: `CrossMomentumStrategy._trend_closes(self, symbol: str) -> list[float] | None`, `CrossMomentumStrategy._sleeve_weights(self) -> dict[str, float]`; `rebalance(target, all_ranks)` keeps its signature.

- [ ] **Step 1: Extend the test fake** (existing tests keep passing with `trend_assets=()`, which gives SHV 100%)

In `tests/strategies/test_cross_momentum_rebalance.py`:

Replace the imports block with:

```python
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pandas as pd
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BrokerError

TODAY = date(2026, 10, 7)
RISING, FALLING = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]  # last close above / below the fake's 3-day SMA (2.0)
```

Replace `FakeStrategy.__init__` and add the bar methods (keep every other method):

```python
    def __init__(
        self, *, cash, positions=(), last_prices=None, cash_buffer_pct=0.05, min_trade_pct=0.01, reject=(), price_errors=(), trend_assets=(), bars=None
    ):
        self.parameters = {
            "sell_rank_threshold": 35,
            "cash_buffer_pct": cash_buffer_pct,
            "parking": {"symbol": "SHV", "trend_assets": tuple(trend_assets), "trend_sma_window": 3, "min_trade_pct": min_trade_pct},
        }
        self._cash = cash
        self._positions = list(positions)
        self._last_prices = last_prices or {}
        self._reject = set(reject)
        self._price_errors = set(price_errors)
        self._bars = bars or {}
        self.portfolio_value = cash + sum(p.quantity * self._last_prices[p.asset.symbol] for p in self._positions)
        self.vars = SimpleNamespace(alpaca_rate_limiter=SimpleNamespace(wait=lambda: None))
        self.bar_requests = []
        self.orders = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def get_historical_prices(self, symbol, length, timestep):
        self.bar_requests.append((symbol, length, timestep))
        bars = self._bars.get(symbol)
        if isinstance(bars, Exception):
            raise bars
        return bars

    def _market_date(self):
        return TODAY

    _sleeve_weights = CrossMomentumStrategy._sleeve_weights
    _trend_closes = CrossMomentumStrategy._trend_closes
```

Add after `_orders`:

```python
def _bars(closes, *, today_close=None):
    """Daily bars stamped at the close on the sessions before TODAY, plus a partial bar for TODAY if given."""
    days = [TODAY - timedelta(days=len(closes) - i) for i in range(len(closes))]
    if today_close is not None:
        days.append(TODAY)
        closes = [*closes, today_close]
    index = pd.DatetimeIndex([datetime.combine(day, time(16), tzinfo=MARKET_TZ) for day in days])
    frame = pd.DataFrame({"close": closes}, index=index)
    return SimpleNamespace(empty=frame.empty, pandas_df=frame)


SLEEVE_PRICES = {"AAA": 100.0, "GLD": 10.0, "IEF": 10.0, "SHV": 50.0}


def _sleeve_fake(*, gld=RISING, ief=RISING, **kwargs):
    bars = {"GLD": gld if not isinstance(gld, list) else _bars(gld), "IEF": ief if not isinstance(ief, list) else _bars(ief)}
    kwargs.setdefault("cash", 1000.0)
    kwargs.setdefault("last_prices", dict(SLEEVE_PRICES))
    return FakeStrategy(trend_assets=("GLD", "IEF"), bars=bars, **kwargs)


def _sequence(fake):
    return [(o.symbol, o.side, o.quantity) for o in fake.orders]
```

Replace `test_backtests_preload_the_parking_symbol_once` with:

```python
def test_backtests_preload_the_sleeve_symbols_once():
    fake = SimpleNamespace(parameters={"parking": {"symbol": "SHV", "trend_assets": ("GLD", "IEF")}}, vars=SimpleNamespace(universe=["AAA", "SHV", "BBB"]))

    assets = CrossMomentumStrategy._backtest_preload_assets(fake)

    assert [a.symbol for a in assets] == ["AAA", "SHV", "BBB", "GLD", "IEF"]
    fake.vars.universe = ["AAA"]
    assert CrossMomentumStrategy._backtest_preload_assets(fake) == [Asset(symbol=s) for s in ("AAA", "SHV", "GLD", "IEF")]
```

- [ ] **Step 2: Add the sleeve tests** (append to the same file)

```python
# ── Trend sleeve ───────────────────────────────────────────────────────────────
# With cash 1000 and AAA at 30%, the parking target is 950 - 300 = 650: 325 per trend asset that trends.


def test_idle_cash_with_both_trends_on_is_split_between_gld_and_ief():
    fake = _sleeve_fake()

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _sequence(fake) == [("AAA", "buy", 3.0), ("GLD", "buy", 32.5), ("IEF", "buy", 32.5)]


def test_sleeve_symbols_are_never_sold_as_unranked():
    held = [_held("GLD", 32.5), _held("IEF", 32.5), _held("SHV", 0.5)]
    fake = _sleeve_fake(cash=325.0, positions=held)  # pv 325 + 325 + 325 + 25 = 1000

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert [o for o in fake.orders if o.side == "sell" and o.symbol != "SHV"] == []
    assert not any("not ranked" in message for message in fake.warnings)


def test_a_trend_turning_off_sells_the_whole_holding_and_parks_it_in_shv():
    fake = _sleeve_fake(ief=FALLING, cash=350.0, positions=[_held("GLD", 32.5), _held("IEF", 32.5)])

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "IEF", "sell") == [32.5]
    assert _orders(fake, "GLD", "sell") == [] and _orders(fake, "GLD", "buy") == []
    assert _orders(fake, "SHV", "buy") == [6.5]  # SHV target 325


def test_trend_assets_are_bought_before_shv_and_within_the_cash_left():
    fake = _sleeve_fake(ief=FALLING)

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _sequence(fake) == [("AAA", "buy", 3.0), ("GLD", "buy", 32.5), ("SHV", "buy", 6.5)]
    assert _planned_buy_cost(fake, SLEEVE_PRICES) <= 1000.0 * 0.95 + 1e-6


def test_a_missing_price_for_one_sleeve_asset_skips_only_that_asset():
    fake = _sleeve_fake(price_errors={"GLD"})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _sequence(fake) == [("AAA", "buy", 3.0), ("IEF", "buy", 32.5)]
    assert any("GLD" in message for message in fake.warnings)


def test_a_failed_bars_fetch_for_a_trend_asset_sends_its_half_to_shv():
    fake = _sleeve_fake(gld=BrokerError("no bars"))

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _sequence(fake) == [("AAA", "buy", 3.0), ("IEF", "buy", 32.5), ("SHV", "buy", 6.5)]
    assert any("GLD" in message for message in fake.warnings)


def test_empty_bars_for_a_trend_asset_send_its_half_to_shv():
    fake = _sleeve_fake(gld=_bars([]))

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _sequence(fake) == [("AAA", "buy", 3.0), ("IEF", "buy", 32.5), ("SHV", "buy", 6.5)]
    assert any("GLD" in message for message in fake.warnings)


def test_a_partial_bar_for_today_is_ignored_by_the_trend_test():
    fake = _sleeve_fake(gld=_bars(FALLING, today_close=100.0))  # with today's bar GLD would read as trending

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "GLD", "buy") == []
    assert _orders(fake, "SHV", "buy") == [6.5]


def test_the_trend_assets_are_read_over_the_strategy_history_window():
    fake = _sleeve_fake()

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert fake.bar_requests == [("GLD", 301, "day"), ("IEF", 301, "day")]


def test_the_sleeve_reading_is_logged():
    fake = _sleeve_fake(ief=FALLING)

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert "Sleeve: GLD on (close 3.00 > SMA3 2.00), IEF off (close 1.00 <= SMA3 2.00) -> SHV 50% / GLD 50% / IEF 0%" in fake.infos


def test_sleeve_symbols_are_removed_from_the_universe():
    broker = FakeBroker(FakeClock(et(2026, 10, 6, 7), []))

    strategy = CrossMomentumStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA", "GLD", "SHV", "BBB", "IEF"])

    assert strategy.vars.universe == ["AAA", "BBB"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_rebalance.py -v`
Expected: collection error, `AttributeError: type object 'CrossMomentumStrategy' has no attribute '_sleeve_weights'`.

- [ ] **Step 4: Implement in `agent_cross_momentum.py`**

4a. Module docstring, point 10, becomes:

```
  10. Park the de-risked capital (everything but the cash reserve) in a sleeve: GLD and IEF each take half while
      above their 200-day SMA, SHV holds the rest
```

4b. Add to the `from .utils import (...)` list, in alphabetical order: `sleeve_symbols`, `sleeve_weights`, `trend_reading`.

4c. Class docstring, last sentence, becomes: `Held positions are trimmed to the scaled target, and the capital taken out of stocks is parked in a trend-filtered SHV/GLD/IEF sleeve.`

4d. In `__init__`, replace `self.vars.universe = universe or []` with:

```python
        # A sleeve asset (SHV, GLD, IEF) is never scored or bought as a stock, even if the universe file lists it
        sleeve = set(sleeve_symbols(self.parameters["parking"]))
        self.vars.universe = [symbol for symbol in universe if symbol not in sleeve]
```

4e. Add after `_price_or_zero`:

```python
    def _trend_closes(self, symbol: str) -> list[float] | None:
        """A trend asset's completed daily closes, oldest first, or None (logged) when its bars are unavailable."""
        fallback = self.parameters["parking"]["symbol"]
        self.vars.alpaca_rate_limiter.wait()
        try:
            bars = self.get_historical_prices(symbol, length=_HISTORY_BARS + 1, timestep="day")
        except BrokerError as exc:
            self.log_warning(f"Sleeve: no bars for {symbol} ({exc}): its share goes to {fallback}")
            return None
        if bars is None or bars.empty:
            self.log_warning(f"Sleeve: no bars for {symbol}: its share goes to {fallback}")
            return None
        # Completed sessions only, like the stocks: a bar dated today is partial while the session is open
        return completed_bars(bars.pandas_df, self._market_date()).tail(_HISTORY_BARS)["close"].tolist()

    def _sleeve_weights(self) -> dict[str, float]:
        """This week's share of the parking sleeve per symbol (SHV first, then the trend assets), logged."""
        parking = self.parameters["parking"]
        window = parking["trend_sma_window"]
        trend_assets = tuple(parking["trend_assets"])
        closes_by_asset = {symbol: closes for symbol in trend_assets if (closes := self._trend_closes(symbol)) is not None}
        weights = sleeve_weights(closes_by_asset, trend_assets, window, parking["symbol"])

        readings = []
        for symbol in trend_assets:
            reading = trend_reading(closes_by_asset.get(symbol, []), window)
            if reading is None:
                readings.append(f"{symbol} off (no data)")
            elif weights[symbol] > 0:
                readings.append(f"{symbol} on (close {reading[0]:.2f} > SMA{window} {reading[1]:.2f})")
            else:
                readings.append(f"{symbol} off (close {reading[0]:.2f} <= SMA{window} {reading[1]:.2f})")
        allocation = " / ".join(f"{symbol} {weight:.0%}" for symbol, weight in weights.items())
        self.log_info(f"Sleeve: {', '.join(readings) + ' -> ' if readings else ''}{allocation}")
        return weights
```

4f. In `rebalance`:

Docstring first line becomes `"""Compare holdings to target, apply hysteresis, trim, submit orders, and park the rest in the sleeve.`

Replace

```python
        parking_symbol = self.parameters["parking"]["symbol"]

        sell_threshold = self.parameters["sell_rank_threshold"]
        portfolio_value = float(self.portfolio_value or 1.0)
        min_trade_value = portfolio_value * self.parameters["parking"]["min_trade_pct"]
```

with

```python
        parking = self.parameters["parking"]
        # Trend assets first, SHV last: the order of the sleeve buys once the stock buys are funded
        sleeve_order = (*parking["trend_assets"], parking["symbol"])

        sell_threshold = self.parameters["sell_rank_threshold"]
        portfolio_value = float(self.portfolio_value or 1.0)
        min_trade_value = portfolio_value * parking["min_trade_pct"]
```

Replace

```python
        parking_position = None
        for pos in current_positions:
            symbol = pos.asset.symbol
            if symbol == parking_symbol:
                # The parking sleeve is never ranked: it must not be exited as "not ranked" below
                parking_position = pos
                continue
```

with

```python
        sleeve_positions = {}
        for pos in current_positions:
            symbol = pos.asset.symbol
            if symbol in sleeve_order:
                # The parking sleeve is never ranked: it must not be exited as "not ranked" below
                sleeve_positions[symbol] = pos
                continue
```

Replace everything from `parking_price = self._price_or_zero(parking_symbol)` down to the end of the parking sell `try/except` (the block ending with `self.log_error(f"Failed to submit parking sell order for {parking_symbol}: {e}")`) with:

```python
        weights = self._sleeve_weights()
        sleeve: list[tuple[str, float, float, float]] = []  # (symbol, price, current value, target value), priced only
        for symbol in sleeve_order:
            position = sleeve_positions.get(symbol)
            price = self._price_or_zero(symbol)
            if price <= 0:
                self.log_warning(f"Parking: no price for {symbol} — no {symbol} order this week")
                continue
            value = float(position.quantity) * price if position else 0.0
            symbol_target = parking_target * weights.get(symbol, 0.0)
            sleeve.append((symbol, price, value, symbol_target))
            self.log_info(f"Parking: {symbol} target ${symbol_target:,.0f} (current ${value:,.0f})")
            excess = value - symbol_target
            if value > symbol_target * (1 + _REBALANCE_BAND) and excess >= min_trade_value:
                # A zero target sells the exact holding, so float flooring leaves no dust behind.
                # Guard against a missing position so static analysis does not treat it as a known quantity.
                sell_qty = float(position.quantity) if position is not None and symbol_target == 0 else fractional_qty(excess / price)
                if sell_qty > 0:
                    self.log_info(f"Selling {sell_qty} {symbol} @ ${price:.2f} (parking above target)")
                    try:
                        self.submit_order(self.create_order(symbol, sell_qty, "sell", time_in_force="day"))
                        estimated_sell_proceeds += sell_qty * price
                    except Exception as e:
                        self.log_error(f"Failed to submit parking sell order for {symbol}: {e}")
```

Replace the whole Phase 3 block (from `# Phase 3: Park what the stock buys left, up to the parking target` to its `except` line) with:

```python
        # Phase 3: Park what the stock buys left, up to each sleeve target (the trend assets first, SHV last)
        for symbol, price, value, symbol_target in sleeve:
            if value >= symbol_target * (1 - _REBALANCE_BAND):
                continue
            buy_value = min(symbol_target - value, available_cash)
            if buy_value < min_trade_value:
                continue
            quantity = fractional_qty(buy_value / price)
            if quantity <= 0:
                continue
            available_cash -= quantity * price
            self.log_info(f"Buying {quantity} {symbol} @ ${price:.2f} (parking)")
            try:
                self.submit_order(self.create_order(symbol, quantity, "buy", time_in_force="day"))
            except Exception as e:
                self.log_warning(f"Failed to submit parking buy order for {symbol}: {e}")
```

4g. `_backtest_preload_assets` becomes:

```python
    def _backtest_preload_assets(self) -> list[Asset]:
        """The universe plus the sleeve symbols, each once: what a backtest loads up front."""
        symbols = list(dict.fromkeys([*self.vars.universe, *sleeve_symbols(self.parameters["parking"])]))
        return [Asset(symbol=symbol) for symbol in symbols]
```

and its call site comment in `run_backtesting` becomes `# preload the ticker universe and the sleeve ETFs in memory`.

- [ ] **Step 5: Run the rebalance tests**

Run: `uv run pytest tests/strategies/test_cross_momentum_rebalance.py -v`
Expected: all PASS, the pre-existing ones included (with `trend_assets=()` the sleeve is SHV 100%, as before).

- [ ] **Step 6: README**

In `README.md`, in the `### 📈 cross_momentum` section, after the paragraph starting `Final Version of the **cross_momentum** strategy`, add:

```markdown
Capital the exposure legs take out of stocks is parked in a sleeve: GLD and IEF each take half of it while their last completed close is above their 200-day SMA, and SHV holds the rest (`parameters.py`, `parking`).
```

- [ ] **Step 7: Full suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all PASS, `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py tests/strategies/test_cross_momentum_rebalance.py README.md
git commit -m "Task 5: cross_momentum parks in a trend-filtered SHV/GLD/IEF sleeve

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Candidate set, comparison and decision (operational, ~65 min)

Needs Task 3's baseline set finished and Task 5 (plus any review fix) committed, with a clean worktree.

- [ ] **Step 1: Check the universe file still matches the baseline's**

```bash
MAIN=/home/yann/projets/trading-agent-framework
sha256sum data/universe/us_stock_universe.json $MAIN/data/universe/us_stock_universe.json
```

Expected: both hashes equal. If not, copy `main`'s file again before running.

- [ ] **Step 2: Run the candidate set from the worktree**

```bash
mkdir -p logs/cross_momentum/experiments
nohup uv run python scripts/experiments/cross_momentum_weekdays.py run --label trend-sleeve > logs/cross_momentum/experiments/trend-sleeve.out 2>&1 &
```

When it ends: `tail -5 logs/cross_momentum/experiments/trend-sleeve.out`. Expected: `Manifest: .../trend-sleeve.json`. A failed day: rerun the same command.

- [ ] **Step 3: Check one run actually used the sleeve**

Run: `grep -m3 "Sleeve:" "$(python3 -c "import json;print(json.load(open('logs/cross_momentum/experiments/trend-sleeve.json'))['runs']['1']['run_dir'])")/backtest.log"`
Expected: `Sleeve: GLD ... IEF ... -> SHV ...` lines.

- [ ] **Step 4: Compare**

```bash
uv run python scripts/experiments/cross_momentum_weekdays.py compare $MAIN/logs/cross_momentum/experiments/shv-baseline.json logs/cross_momentum/experiments/trend-sleeve.json
```

Expected: the table, `CAGR wins: n of 5`, `verdict: KEEP` or `verdict: REJECT` with the failed criteria; no warning.

- [ ] **Step 5: Decide with the user**

Show the user the full `compare` output. Then:
- **KEEP:** finish the branch with superpowers:finishing-a-development-branch (merge into `main`).
- **REJECT:** with the user's agreement, delete the branch unmerged (`git worktree remove`, `git branch -D feature/cross-momentum-trend-sleeve`). Tasks 1–2 stay on `main`.

- [ ] **Step 6: Record the result in memory** — update `project_cross_momentum_beta_experiments.md` (the run map and the "Untried" list: the GLD/IEF sleeve is now tried) with both manifests' paths, the five-day means and the verdict, and add the weekday spread (Tue 0.3217 vs Wed 0.2963) as the noise floor to beat.
