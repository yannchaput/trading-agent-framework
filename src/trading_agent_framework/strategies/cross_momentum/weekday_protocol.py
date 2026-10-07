"""The 5-weekday A/B protocol for cross_momentum backtests.

One backtest cannot see an effect smaller than the spread between rebalance weekdays (the same code scored a CAGR
of 32.2% on Tuesday and 29.6% on Wednesday), so a change is judged on the means of one run per weekday. This
module records a set of runs (a manifest) and compares two sets; `scripts/experiments/cross_momentum_weekdays.py`
runs them. Pure apart from the manifest and metrics JSON files.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import fmean

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri")
WEEKDAYS = frozenset(range(len(DAY_NAMES)))

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
    runs = {}
    for key, entry in data.pop("runs").items():
        day = int(key)
        if day not in WEEKDAYS:
            raise ValueError(f"manifest {data.get('label')!r} ({path}): run day {key} is not a weekday from 0 (Monday) to 4 (Friday)")
        runs[day] = RunEntry(**entry)
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
    figures = {name: float(data[key]) for name, key in _METRIC_KEYS.items()}
    for name, value in figures.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} is {value} in {path} (run {run_dir}): the run is unusable")
    return RunMetrics(**figures)


def count_log_lines(run_dir: Path, patterns: dict[str, str]) -> dict[str, int]:
    """Per name, how many lines of the run's `backtest.log` contain that substring."""
    path = Path(run_dir) / "backtest.log"
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"no backtest.log in {run_dir}: the run was deleted or did not finish") from exc
    return {name: sum(1 for line in lines if pattern in line) for name, pattern in patterns.items()}


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
        ValueError: the sets differ in setup or days, either misses a weekday, or a recorded day has no metrics.
    """
    for name in mismatches(baseline, candidate):
        raise ValueError(f"the two sets differ in {name}: {getattr(baseline, name)!r} vs {getattr(candidate, name)!r}")
    if set(baseline.runs) != set(candidate.runs):
        raise ValueError(f"the two sets cover different days: {sorted(baseline.runs)} vs {sorted(candidate.runs)}")
    for manifest in (baseline, candidate):
        if set(manifest.runs) != WEEKDAYS:
            raise ValueError(f"a comparison needs all five weekdays; {manifest.label} covers days {sorted(manifest.runs)}")
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
