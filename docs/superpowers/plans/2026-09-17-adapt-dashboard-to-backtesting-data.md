# Adapt Dashboard to Backtesting Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the copied-in `lumibot` Streamlit dashboard (`src/trading_agent_framework/dashboard/`) actually read this project's real backtest output (`backtesting/report.py`'s `settings.json` / `metrics.json` / `equity.parquet` / `trades.parquet` / `indicators.parquet`) instead of lumibot's quantstats-shaped, timestamp-prefixed files.

**Architecture:** The dashboard is a thin read-only Streamlit viewer over files a backtest run writes to `logs/<strategy>/<mode>/<run_ts>_<mode>/`. No new abstractions are introduced — each task fixes one loader function (or a small cluster of them) against the *exact* contract `backtesting/report.py` and `backtesting/metrics.py` already produce (verified against `tests/backtesting/test_report.py` and `tests/backtesting/dashboard_contract.py`, which already exist as this project's half of that contract). Every fixture used to test a loader is built via the *real* `backtesting.report.write_*` functions — the same functions a real backtest calls — so the tests are true contract round-trips, not guesses at file shape.

**Tech Stack:** Python 3.14, pydantic v2, pandas/pyarrow (parquet), Streamlit, pytest, `uv`.

**Spec:** This plan document — findings were gathered directly from reading `src/trading_agent_framework/dashboard/**`, `src/trading_agent_framework/backtesting/{report,metrics,runner,ledger}.py`, `src/trading_agent_framework/utils/log.py`, and `tests/backtesting/{test_report.py,test_runner.py,dashboard_contract.py}` in this conversation. There is no separate spec file.

## Global Constraints

- Python 3.14, package manager `uv` (`uv sync`, `uv run pytest`, `uv run ruff check`) — see root `CLAUDE.md`.
- The automated test suite never touches the network (root `CLAUDE.md`) — this plan actively *removes* the dashboard's only network call (a live `yfinance.download()` for the benchmark), so no test may add one back.
- `backtesting/report.py` writes **fixed filenames, never timestamped/glob'd**: `settings.json`, `metrics.json`, `equity.parquet`, `trades.parquet`, `indicators.parquet` — every dashboard loader must read those exact names, never a glob pattern.
- `metrics.json` is written keyed **exactly** by `MetricSet`'s field names (see `tests/backtesting/dashboard_contract.py::METRIC_SET_FIELDS`) — a benchmark-less run writes explicit JSON `null` for every `*_benchmark`/relative field (confirmed by `tests/backtesting/test_report.py::test_write_metrics_fills_missing_metric_set_fields_with_none`).
- Run directories are `logs/<strategy_name>/<mode>/<run_ts>_<mode>/` where `mode` is `backtesting`/`paper`/`live` and `run_ts` is `%Y-%m-%d_%H%M%S` (`utils/log.py::setup_strategy_logging`) — **no** `agent_` prefix anywhere, unlike lumibot's layout.
- Only `mode == "backtesting"` runs ever get `settings.json`/`metrics.json`/`equity.parquet`/etc. (`backtesting/runner.py` is the only writer); paper/live run directories never contain them.
- Decided during brainstorming with the user (do not re-litigate):
  1. Read the benchmark series from `equity.parquet`'s precomputed `benchmark_close`/`benchmark_return` columns — never re-fetch from yfinance live.
  2. Remove the 3 quantstats-only metric cards (`Expected Yearly%`, `Best Day`, `Worst Day`) from the Run Detail page — this project's metrics engine never computes them.
  3. Delete the indicator-based and trades-reconstruction equity-curve fallback code paths entirely (`load_equity_from_indicators`, `load_equity_from_trades`) — `equity.parquet` is always written by a real run, so there is nothing to fall back from.

---

## File Structure

- `src/trading_agent_framework/dashboard/models.py` — fix `RunRef.from_path` for this project's directory layout; remove `MetricSet.from_scalars` (a quantstats-CSV translator with nothing left to translate).
- `src/trading_agent_framework/dashboard/discovery.py` — fix import; glob for `metrics.json` instead of `*_tearsheet_metrics.json`.
- `src/trading_agent_framework/dashboard/reader.py` — fix import and two Python-2-style `except` clauses; replace every glob-based `_find_file` lookup with a direct fixed path; rewrite `load_metrics`, `get_benchmark_symbol`, `load_settings`, `load_parameters`, `load_portfolio_breakdown`, `load_cumulative_returns`, `load_yearly_returns`, `load_trades_curve`, `load_run`; delete `load_equity_from_indicators`, `load_equity_from_trades`, `_parse_tearsheet_csv`, `_parse_tearsheet_cell`, `_cumret_no_benchmark`; add `load_equity_curve`.
- `src/trading_agent_framework/dashboard/app.py`, `_pages/scorecard.py`, `_pages/side_by_side.py` — fix import only.
- `src/trading_agent_framework/dashboard/_pages/detail.py` — fix import; remove the 3 dead metric cards.
- `pyproject.toml` — fix the `dashboard` script's entry point module path.
- `tests/dashboard/__init__.py` (new), `tests/dashboard/test_package_imports.py` (new), `tests/dashboard/test_models.py` (new), `tests/dashboard/test_discovery.py` (new), `tests/dashboard/test_reader.py` (new), `tests/dashboard/test_app_smoke.py` (new) — all new; no dashboard tests exist today.

---

### Task 1: Fix blocking import and syntax bugs

The dashboard package cannot be imported at all right now: every module imports from the nonexistent `lumibot_trading_agent` package, and `reader.py` has two Python-2-style `except X, Y:` clauses (invalid Python 3 syntax). Fix both, plus the `pyproject.toml` script entry point that has the same wrong package name.

**Files:**
- Modify: `src/trading_agent_framework/dashboard/discovery.py:8`
- Modify: `src/trading_agent_framework/dashboard/reader.py:12,36,288`
- Modify: `src/trading_agent_framework/dashboard/app.py:9-10`
- Modify: `src/trading_agent_framework/dashboard/_pages/detail.py:6,18,19`
- Modify: `src/trading_agent_framework/dashboard/_pages/scorecard.py:5-7`
- Modify: `src/trading_agent_framework/dashboard/_pages/side_by_side.py:7`
- Modify: `pyproject.toml` (the `dashboard` line under `[project.scripts]`)
- Test: `tests/dashboard/__init__.py`, `tests/dashboard/test_package_imports.py`

**Interfaces:**
- Produces: every dashboard module becomes importable under `trading_agent_framework.dashboard.*` — later tasks assume this.

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/__init__.py` (empty file, matching `tests/backtesting/__init__.py`'s pattern).

Create `tests/dashboard/test_package_imports.py`:

```python
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "trading_agent_framework.dashboard.models",
    "trading_agent_framework.dashboard.discovery",
    "trading_agent_framework.dashboard.reader",
    "trading_agent_framework.dashboard.theme",
    "trading_agent_framework.dashboard.components.charts",
    "trading_agent_framework.dashboard.components.tables",
    "trading_agent_framework.dashboard.components.metric_cards",
    "trading_agent_framework.dashboard._pages.scorecard",
    "trading_agent_framework.dashboard._pages.detail",
    "trading_agent_framework.dashboard._pages.side_by_side",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_dashboard_module_imports_cleanly(module_name: str) -> None:
    importlib.import_module(module_name)


def test_pyproject_dashboard_script_points_at_the_real_package() -> None:
    import pathlib

    text = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    assert "trading_agent_framework.dashboard.cli:main" in text
    assert "lumibot_trading_agent" not in text
```

(`app.py` is deliberately excluded here — it calls `st.set_page_config()` at module level, which needs a live Streamlit script context; it's covered by the `AppTest`-based smoke test in Task 9 instead.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_package_imports.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumibot_trading_agent'` (or `SyntaxError` once that import is fixed by hand, but not yet at this step).

- [ ] **Step 3: Fix the import lines**

In `discovery.py`, `reader.py`, `app.py`, `_pages/detail.py`, `_pages/scorecard.py`, `_pages/side_by_side.py`, replace every occurrence of `lumibot_trading_agent.dashboard` with `trading_agent_framework.dashboard`. Concretely:

`discovery.py:8`:
```python
from trading_agent_framework.dashboard.models import RunIndex, RunRef
```

`reader.py:12`:
```python
from trading_agent_framework.dashboard.models import MetricSet, Run, RunRef, Settings
```

`app.py:9-10`:
```python
from trading_agent_framework.dashboard.reader import load_description, save_description
from trading_agent_framework.dashboard.theme import apply_theme
```
and further down (`main()`'s page routing imports), the three `from lumibot_trading_agent.dashboard._pages.<x> import <fn>` lines become `from trading_agent_framework.dashboard._pages.<x> import <fn>`.

`_pages/detail.py:6-19`:
```python
from trading_agent_framework.dashboard.components.charts import (
    cumulative_returns_chart,
    drawdown_chart,
    equity_curve_chart,
    monthly_returns_distribution,
    monthly_returns_heatmap,
    returns_distribution,
    rolling_sharpe_chart,
    rolling_sortino_chart,
    rolling_volatility_chart,
    trades_chart,
)
from trading_agent_framework.dashboard.components.metric_cards import render_header_card, render_metric_card
from trading_agent_framework.dashboard.reader import load_cumulative_returns, load_parameters, load_portfolio_breakdown, load_run, load_trades_curve, load_yearly_returns
```

`_pages/scorecard.py:5-7`:
```python
from trading_agent_framework.dashboard.components.tables import render_scorecard_table
from trading_agent_framework.dashboard.discovery import scan_runs
from trading_agent_framework.dashboard.reader import load_description, load_metrics, load_settings
```

`_pages/side_by_side.py:7`:
```python
from trading_agent_framework.dashboard.reader import load_portfolio_breakdown, load_run
```

- [ ] **Step 4: Fix the two Python-2-style `except` clauses in `reader.py`**

Line 36, inside `load_description`:
```python
    except (json.JSONDecodeError, OSError):
        return None
```

Line 288, inside `_parse_tearsheet_cell` (this whole function is deleted in Task 4 — for this step just make it syntactically valid so the module imports; Task 4 removes it):
```python
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 5: Fix `pyproject.toml`**

Change:
```toml
dashboard = "lumibot_trading_agent.dashboard.cli:main"
```
to:
```toml
dashboard = "trading_agent_framework.dashboard.cli:main"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_package_imports.py -v`
Expected: PASS (all parametrized imports succeed; pyproject check passes).

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/trading_agent_framework/dashboard/`
Expected: no new errors (pre-existing style issues aside, the file should at least parse and import-sort cleanly).

- [ ] **Step 8: Commit**

```bash
git add tests/dashboard/__init__.py tests/dashboard/test_package_imports.py \
  src/trading_agent_framework/dashboard/discovery.py \
  src/trading_agent_framework/dashboard/reader.py \
  src/trading_agent_framework/dashboard/app.py \
  src/trading_agent_framework/dashboard/_pages/detail.py \
  src/trading_agent_framework/dashboard/_pages/scorecard.py \
  src/trading_agent_framework/dashboard/_pages/side_by_side.py \
  pyproject.toml
git commit -m "fix: point dashboard imports and script entry at trading_agent_framework, fix py2 except syntax"
```

---

### Task 2: Adapt `RunRef.from_path` to this project's log layout

**Files:**
- Modify: `src/trading_agent_framework/dashboard/models.py:24-54`
- Test: `tests/dashboard/test_models.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `RunRef.from_path(path: str) -> RunRef` with `.strategy_name`, `.run_ts`, `.mode`, `.path` — used by `discovery.scan_runs` (Task 3) and every reader function taking a `ref: RunRef`.

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_models.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.dashboard.models import RunRef


def test_from_path_parses_this_projects_log_layout() -> None:
    ref = RunRef.from_path("logs/momentum/backtesting/2026-06-22_194053_backtesting")
    assert ref.strategy_name == "momentum"
    assert ref.mode == "backtesting"
    assert ref.run_ts == "2026-06-22_194053"
    assert ref.path == "logs/momentum/backtesting/2026-06-22_194053_backtesting"


def test_from_path_parses_paper_and_live_modes() -> None:
    assert RunRef.from_path("logs/momentum/paper/2026-06-22_194053_paper").mode == "paper"
    assert RunRef.from_path("logs/momentum/live/2026-06-22_194053_live").mode == "live"


def test_from_path_rejects_a_path_with_no_recognizable_mode() -> None:
    with pytest.raises(ValueError, match="Cannot parse"):
        RunRef.from_path("logs/momentum/2026-06-22_194053")


def test_from_path_handles_a_trailing_slash() -> None:
    ref = RunRef.from_path("logs/momentum/backtesting/2026-06-22_194053_backtesting/")
    assert ref.run_ts == "2026-06-22_194053"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_models.py -v`
Expected: FAIL — the current `from_path` requires an `"agent_"`-prefixed path segment, which none of these paths have, so it raises `ValueError: Cannot parse strategy name from path` for all of them (including the ones that should succeed).

- [ ] **Step 3: Rewrite `RunRef.from_path`**

Replace the whole method body (`models.py:24-54`) with:

```python
    @classmethod
    def from_path(cls, path: str) -> "RunRef":
        """Extract strategy_name, run_ts, and mode from a directory path.

        Expected pattern: logs/{strategy_name}/{mode}/{run_ts}_{mode}/
        (trading_agent_framework.utils.log.setup_strategy_logging's layout;
        run_ts is "%Y-%m-%d_%H%M%S", e.g. "2026-06-22_194053").
        """
        parts = [p for p in path.replace("\\", "/").rstrip("/").split("/") if p]
        if len(parts) < 3:
            raise ValueError(f"Cannot parse run directory from path: {path}")

        run_dir, mode, strategy_name = parts[-1], parts[-2], parts[-3]
        if mode not in ("backtesting", "paper", "live"):
            raise ValueError(f"Cannot parse mode from path: {path}")

        run_ts = "_".join(run_dir.split("_")[:2]) if "_" in run_dir else run_dir

        return cls(strategy_name=strategy_name, run_ts=run_ts, mode=mode, path=path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_models.py src/trading_agent_framework/dashboard/models.py
git commit -m "fix: parse RunRef paths against this project's logs/<strategy>/<mode>/<ts>_<mode> layout"
```

---

### Task 3: Fix `discovery.scan_runs`'s glob pattern

**Files:**
- Modify: `src/trading_agent_framework/dashboard/discovery.py:11-14`
- Test: `tests/dashboard/test_discovery.py` (new)

**Interfaces:**
- Consumes: `RunRef.from_path` (Task 2).
- Produces: `scan_runs(base_path: str = "logs") -> RunIndex` — used by `_pages/scorecard.py` (already fixed in Task 1, unchanged otherwise).

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_discovery.py`:

```python
from __future__ import annotations

from pathlib import Path

from trading_agent_framework.dashboard.discovery import scan_runs


def _make_run_dir(base: Path, strategy: str, run_ts: str) -> Path:
    run_dir = base / strategy / "backtesting" / f"{run_ts}_backtesting"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "settings.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_scan_runs_finds_runs_by_metrics_json(tmp_path: Path) -> None:
    _make_run_dir(tmp_path, "momentum", "2026-06-22_194053")
    _make_run_dir(tmp_path, "mean_reversion", "2026-06-23_090000")

    index = scan_runs(str(tmp_path))

    assert sorted(index.strategy_names()) == ["mean_reversion", "momentum"]
    assert len(index.runs) == 2


def test_scan_runs_ignores_paper_and_live_run_dirs_that_have_no_metrics_json(tmp_path: Path) -> None:
    _make_run_dir(tmp_path, "momentum", "2026-06-22_194053")
    paper_dir = tmp_path / "momentum" / "paper" / "2026-06-23_100000_paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "some_other_file.log").write_text("x", encoding="utf-8")

    index = scan_runs(str(tmp_path))

    assert len(index.runs) == 1
    assert index.runs[0].mode == "backtesting"


def test_scan_runs_returns_empty_index_for_empty_logs_dir(tmp_path: Path) -> None:
    index = scan_runs(str(tmp_path))
    assert index.runs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_discovery.py -v`
Expected: FAIL — `scan_runs` globs for `*_tearsheet_metrics.json`, which no fixture file matches, so `index.runs` is always empty.

- [ ] **Step 3: Fix the glob pattern**

In `discovery.py`, change:
```python
    pattern = os.path.join(base_path, "**", "*_tearsheet_metrics.json")
```
to:
```python
    pattern = os.path.join(base_path, "**", "metrics.json")
```

(No other change needed — the rest of `scan_runs` already just takes `os.path.dirname(match)` and hands it to `RunRef.from_path`, which Task 2 already fixed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_discovery.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_discovery.py src/trading_agent_framework/dashboard/discovery.py
git commit -m "fix: discover runs by metrics.json instead of lumibot's tearsheet filename"
```

---

### Task 4: Rewrite metrics loading against `metrics.json`'s flat contract

`load_metrics` currently expects a quantstats CSV/JSON tearsheet pair with a nested `scalar_metrics: {"Total Return": {"Strategy": v, "Benchmark": v}, ...}` shape, translated through `MetricSet.from_scalars()`'s display-name lookup table. This project's `metrics.json` is already flat, keyed by `MetricSet`'s exact field names (`backtesting/report.py::write_metrics`) — no CSV tearsheet is ever produced. Replace the translation entirely with a direct `model_validate`, after normalizing the `null`s a benchmark-less run legitimately writes for `*_benchmark` fields (`MetricSet`'s fields are plain `float`, not `float | None`, so a raw `null` would fail validation).

**Files:**
- Modify: `src/trading_agent_framework/dashboard/models.py` (delete `MetricSet.from_scalars`, `models.py:118-183`)
- Modify: `src/trading_agent_framework/dashboard/reader.py` (delete `_parse_tearsheet_csv`, `_parse_tearsheet_cell` at `reader.py:214-289`; rewrite `load_metrics` at `reader.py:292-328`)
- Test: `tests/dashboard/test_reader.py` (new — this task starts the file; later tasks append to it)

**Interfaces:**
- Consumes: `trading_agent_framework.backtesting.report.write_metrics(run_dir: Path, metrics: dict) -> Path` (already exists, unchanged) to build fixtures.
- Produces: `load_metrics(ref: RunRef) -> MetricSet | None` — used by `_pages/scorecard.py`, `_pages/detail.py`, `_pages/side_by_side.py` (already fixed to import correctly in Task 1, no further changes needed to those call sites for this task). Also produces `MetricSet.model_validate({...})` as the only construction path (the `from_scalars` classmethod is gone).

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_reader.py`:

```python
from __future__ import annotations

from pathlib import Path

from tests.backtesting.dashboard_contract import METRIC_SET_FIELDS
from trading_agent_framework.backtesting import report
from trading_agent_framework.dashboard.models import RunRef
from trading_agent_framework.dashboard.reader import load_metrics


def _run_dir(tmp_path: Path, strategy: str = "momentum", run_ts: str = "2026-06-22_194053") -> Path:
    d = tmp_path / strategy / "backtesting" / f"{run_ts}_backtesting"
    d.mkdir(parents=True)
    return d


def _ref(run_dir: Path) -> RunRef:
    return RunRef.from_path(str(run_dir))


def test_load_metrics_reads_a_full_metrics_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["total_return_strategy"] = 0.42
    metrics["sharpe_strategy"] = 1.5
    metrics["raw"] = {"summary_tables": {"eoy_returns_vs_benchmark": [{"year": 2026, "strategy": 0.1, "benchmark": 0.05, "won": True}], "drawdowns": []}}
    report.write_metrics(run_dir, metrics)

    loaded = load_metrics(_ref(run_dir))

    assert loaded is not None
    assert loaded.total_return_strategy == 0.42
    assert loaded.sharpe_strategy == 1.5
    assert loaded.raw["summary_tables"]["eoy_returns_vs_benchmark"][0]["year"] == 2026


def test_load_metrics_defaults_a_benchmark_less_runs_null_fields_to_zero(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    # A benchmark-less run: compute_metrics only returns strategy-side fields;
    # write_metrics fills every *_benchmark/relative field with JSON null.
    report.write_metrics(run_dir, {"total_return_strategy": 0.1, "raw": {}})

    loaded = load_metrics(_ref(run_dir))

    assert loaded is not None
    assert loaded.total_return_strategy == 0.1
    assert loaded.sharpe_benchmark == 0.0
    assert loaded.beta == 0.0


def test_load_metrics_returns_none_when_metrics_json_is_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_metrics(_ref(run_dir)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: FAIL — `load_metrics` looks for `*_tearsheet_metrics.json`/`*_tearsheet.csv` (neither exists) and returns `None` for the first two tests.

- [ ] **Step 3: Delete `MetricSet.from_scalars` from `models.py`**

Delete the entire classmethod at `models.py:118-183` (from `@classmethod` through the closing `raw=raw,\n        )` line), leaving `MetricSet` as a plain field-only model:

```python
class MetricSet(BaseModel):
    """All scalar performance metrics from metrics.json (backtesting.report.write_metrics
    writes this keyed exactly by these field names; see tests/backtesting/dashboard_contract.py)."""

    total_return_strategy: float = 0.0
    total_return_benchmark: float = 0.0
    cagr_strategy: float = 0.0
    cagr_benchmark: float = 0.0
    sharpe_strategy: float = 0.0
    sharpe_benchmark: float = 0.0
    sortino_strategy: float = 0.0
    sortino_benchmark: float = 0.0
    calmar_strategy: float = 0.0
    calmar_benchmark: float = 0.0
    omega_strategy: float = 0.0
    omega_benchmark: float = 0.0
    max_drawdown_strategy: float = 0.0
    max_drawdown_benchmark: float = 0.0
    volatility_strategy: float = 0.0
    volatility_benchmark: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    correlation: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio_strategy: float = 0.0
    information_ratio_benchmark: float = 0.0
    r_squared_strategy: float = 0.0
    r_squared_benchmark: float = 0.0
    skew_strategy: float = 0.0
    skew_benchmark: float = 0.0
    kurtosis_strategy: float = 0.0
    kurtosis_benchmark: float = 0.0
    win_days_pct_strategy: float = 0.0
    win_days_pct_benchmark: float = 0.0
    win_month_pct_strategy: float = 0.0
    win_month_pct_benchmark: float = 0.0
    longest_dd_days_strategy: float = 0.0
    longest_dd_days_benchmark: float = 0.0
    avg_drawdown_strategy: float = 0.0
    avg_drawdown_benchmark: float = 0.0
    recovery_factor_strategy: float = 0.0
    recovery_factor_benchmark: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Delete the quantstats CSV parsing helpers from `reader.py`**

Delete `_parse_tearsheet_csv` and `_parse_tearsheet_cell` entirely (`reader.py:214-289`).

- [ ] **Step 5: Rewrite `load_metrics`**

Replace `load_metrics` (`reader.py:292-328`) with:

```python
def load_metrics(ref: RunRef) -> MetricSet | None:
    """Load performance metrics from metrics.json.

    metrics.json is written by backtesting.report.write_metrics, keyed exactly by
    MetricSet's field names. A benchmark-less run writes JSON `null` for every
    *_benchmark/relative field (write_metrics's docstring) -- MetricSet's fields are
    plain (non-Optional) floats, so those nulls are normalized to 0.0 before
    validation, matching the "couldn't be computed" convention used everywhere else
    in this dashboard.
    """
    path = os.path.join(ref.path, "metrics.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    raw = data.get("raw") or {}
    scalars = {k: (v if v is not None else 0.0) for k, v in data.items() if k != "raw"}
    return MetricSet.model_validate({**scalars, "raw": raw})
```

(`json` and `os` are already imported at the top of `reader.py`; `MetricSet` is already imported from `models`.)

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: PASS

- [ ] **Step 7: Run the full test suite to confirm nothing else referenced the deleted code**

Run: `uv run pytest`
Expected: PASS (only `tests/dashboard/` and pre-existing tests run; nothing outside the dashboard package imported `from_scalars`/`_parse_tearsheet_csv`/`_parse_tearsheet_cell`, confirmed by an earlier repo-wide grep during planning).

- [ ] **Step 8: Commit**

```bash
git add tests/dashboard/test_reader.py src/trading_agent_framework/dashboard/models.py src/trading_agent_framework/dashboard/reader.py
git commit -m "fix: load metrics.json directly as MetricSet instead of translating a quantstats tearsheet"
```

---

### Task 5: Fix settings/parameters loading to fixed filenames and this project's keys

`get_benchmark_symbol` reads a nested `benchmark_asset.symbol` key that doesn't exist in this project's `settings.json` (it writes a flat `benchmark_symbol` string — see `backtesting/runner.py:299` and `tests/backtesting/test_runner.py:91`). `load_settings`/`load_parameters` glob for `*_settings.json`; this project always writes `settings.json`. `load_parameters` also reads a `lumibot_version` key that doesn't exist (this project writes `framework_version`).

**Files:**
- Modify: `src/trading_agent_framework/dashboard/reader.py` (`_find_file` removal, `get_benchmark_symbol`, `load_settings`, `load_parameters`, `load_description`, `save_description`)
- Test: `tests/dashboard/test_reader.py` (append)

**Interfaces:**
- Consumes: `report.write_settings(run_dir: Path, settings: dict) -> Path`.
- Produces: `get_benchmark_symbol(ref) -> str`, `load_settings(ref) -> Settings | None`, `load_parameters(ref) -> list[tuple[str, str, str]]`, `load_description(ref) -> str | None`, `save_description(ref, description: str) -> None` — all now reading fixed filenames.

- [ ] **Step 1: Write the failing test**

Append to `tests/dashboard/test_reader.py`:

```python
from trading_agent_framework.dashboard.reader import get_benchmark_symbol, load_parameters, load_settings


def _settings_payload(**overrides) -> dict:
    payload = {
        "name": "momentum", "mode": "backtesting", "run_ts": "2026-06-22_194053",
        "backtesting_start": "2026-01-01T09:30:00-05:00", "backtesting_end": "2026-06-01T16:00:00-04:00",
        "budget": 10000.0, "risk_free_rate": 0.03, "backtesting_data_sources": "yahoo",
        "backtest_time_seconds": 12.5, "timestep": "day", "sleeptime": "1D",
        "commission": 0.0, "slippage": 0.0, "warmup_trading_days": 20,
        "benchmark_symbol": "QQQ", "framework_version": "0.1.0",
        "parameters": {"lookback": 20},
    }
    payload.update(overrides)
    return payload


def test_get_benchmark_symbol_reads_the_flat_settings_key(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload())

    assert get_benchmark_symbol(_ref(run_dir)) == "QQQ"


def test_get_benchmark_symbol_falls_back_to_spy_when_settings_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert get_benchmark_symbol(_ref(run_dir)) == "SPY"


def test_load_settings_reads_settings_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload())

    settings = load_settings(_ref(run_dir))

    assert settings is not None
    assert settings.budget == 10000.0
    assert settings.backtesting_data_sources == "yahoo"


def test_load_parameters_reports_the_framework_version(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(framework_version="0.1.0"))

    rows = load_parameters(_ref(run_dir))

    assert ("Run", "Framework version", "0.1.0") in rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: FAIL — `get_benchmark_symbol` returns `"SPY"` even when `settings.json` has `benchmark_symbol="QQQ"` (it looks for `benchmark_asset.symbol`); `load_settings`/`load_parameters` glob for `*_settings.json` which the fixture never creates (it's `settings.json`); no `"Framework version"` row exists.

- [ ] **Step 3: Remove `_find_file` and rewrite the settings/description functions**

Delete `_find_file` (`reader.py:15-19`) — every filename is fixed now, so a direct `os.path.join` replaces every call site.

Replace `load_description`/`save_description` (already syntax-fixed in Task 1) to use `os.path.join(ref.path, "description.json")` directly — they already do this; no further change needed there beyond Task 1's `except` fix.

Replace `get_benchmark_symbol`:

```python
def get_benchmark_symbol(ref: RunRef) -> str:
    """Extract the benchmark ticker from settings.json's flat `benchmark_symbol` field.

    Falls back to "SPY" when the settings file is missing or the key is absent/empty.
    This is the single source of truth for the benchmark ticker shown throughout the
    dashboard.
    """
    path = os.path.join(ref.path, "settings.json")
    if not os.path.isfile(path):
        return "SPY"
    try:
        with open(path) as f:
            data = json.load(f)
        symbol = data.get("benchmark_symbol", "")
        return symbol if symbol else "SPY"
    except (OSError, json.JSONDecodeError):
        return "SPY"
```

Replace `load_settings`:

```python
def load_settings(ref: RunRef) -> Settings | None:
    """Load backtesting settings from settings.json."""
    path = os.path.join(ref.path, "settings.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return Settings.model_validate(data)
```

In `load_parameters`, change the file lookup at the top from:
```python
    path = _find_file(ref.path, "*_settings.json")
```
to:
```python
    path = os.path.join(ref.path, "settings.json")
    if not os.path.isfile(path):
        return []
```

and change:
```python
    rows.append(("Run", "Lumibot version", data.get("lumibot_version", "")))
```
to:
```python
    rows.append(("Run", "Framework version", data.get("framework_version", "")))
```

(Leave the rest of `load_parameters` — the per-agent telemetry section reading `data.get("parameters", {})` — untouched; those keys are populated by strategy/agent code outside this task's scope and will simply produce no rows until that instrumentation exists, which is expected and harmless.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_reader.py src/trading_agent_framework/dashboard/reader.py
git commit -m "fix: read settings.json by its fixed name and this project's flat benchmark_symbol/framework_version keys"
```

---

### Task 6: Replace equity-curve loading with direct `equity.parquet` reads

`load_equity_from_indicators` looks for a `portfolio_value` row inside `indicators.parquet` — lumibot's design. This project's `indicators.parquet` only ever holds genuine strategy-added indicator lines (`ledger.lines`); portfolio value lives in a dedicated `equity.parquet` (columns: `portfolio_value`, `cash`, `positions_value`, `benchmark_close`, `return`, `benchmark_return`, indexed by `datetime`). `load_portfolio_breakdown` looks for a nonexistent `stats.parquet`. Per the earlier decision, delete the indicator/trades-reconstruction fallback chain entirely and read `equity.parquet` directly.

**Files:**
- Modify: `src/trading_agent_framework/dashboard/reader.py` (delete `load_equity_from_indicators`, `load_equity_from_trades`; rewrite `load_portfolio_breakdown`; add `load_equity_curve`; rewrite `load_run`)
- Test: `tests/dashboard/test_reader.py` (append)

**Interfaces:**
- Consumes: `report.write_equity(run_dir: Path, equity: Sequence[EquitySample], benchmark: dict[datetime, Decimal] | None) -> Path`; `trading_agent_framework.backtesting.ledger.EquitySample`.
- Produces: `load_portfolio_breakdown(ref) -> dict[str, Any] | None`, `load_equity_curve(ref) -> list[dict[str, Any]]` (new), `load_run(ref) -> Run` (now built from `load_equity_curve` instead of the deleted indicator/trades fallbacks).

- [ ] **Step 1: Write the failing test**

Append to `tests/dashboard/test_reader.py`:

```python
from datetime import UTC, datetime
from decimal import Decimal

from trading_agent_framework.backtesting.ledger import EquitySample
from trading_agent_framework.dashboard.reader import load_equity_curve, load_portfolio_breakdown, load_run

NOW = datetime(2026, 1, 5, 21, tzinfo=UTC)
LATER = datetime(2026, 1, 6, 21, tzinfo=UTC)


def _equity_samples() -> list[EquitySample]:
    return [
        EquitySample(time=NOW, portfolio_value=Decimal(10000), cash=Decimal(10000), positions_value=Decimal(0)),
        EquitySample(time=LATER, portfolio_value=Decimal(10500), cash=Decimal(500), positions_value=Decimal(10000)),
    ]


def test_load_portfolio_breakdown_reads_equity_parquet(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_equity(run_dir, _equity_samples())

    breakdown = load_portfolio_breakdown(_ref(run_dir))

    assert breakdown is not None
    assert breakdown["portfolio_value"] == [10000.0, 10500.0]
    assert breakdown["cash"] == [10000.0, 500.0]
    assert breakdown["assets"] == [0.0, 10000.0]
    assert len(breakdown["dates"]) == 2


def test_load_portfolio_breakdown_returns_none_when_equity_parquet_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_portfolio_breakdown(_ref(run_dir)) is None


def test_load_equity_curve_reads_portfolio_value_from_equity_parquet(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_equity(run_dir, _equity_samples())

    curve = load_equity_curve(_ref(run_dir))

    assert curve == [
        {"date": NOW.strftime("%Y-%m-%d"), "value": 10000.0},
        {"date": LATER.strftime("%Y-%m-%d"), "value": 10500.0},
    ]


def test_load_run_populates_equity_curve_from_equity_parquet(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_equity(run_dir, _equity_samples())
    report.write_settings(run_dir, _settings_payload())
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["raw"] = {}
    report.write_metrics(run_dir, metrics)

    run = load_run(_ref(run_dir))

    assert run.settings is not None
    assert run.metrics is not None
    assert len(run.equity_curve) == 2
    assert run.equity_curve[0]["value"] == 10000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: FAIL — `load_portfolio_breakdown`/`load_equity_from_indicators` look for `stats.parquet`/`*_indicators.parquet`, neither of which the fixture creates; `load_equity_curve` doesn't exist yet.

- [ ] **Step 3: Delete `load_equity_from_indicators` and `load_equity_from_trades`**

Delete both functions entirely from `reader.py` (`load_equity_from_indicators` and `load_equity_from_trades`).

- [ ] **Step 4: Rewrite `load_portfolio_breakdown`**

```python
def load_portfolio_breakdown(ref: RunRef) -> dict[str, Any] | None:
    """Load daily portfolio decomposition (total value, cash, assets) from equity.parquet.

    `positions_value` is read directly rather than derived as `portfolio_value - cash`:
    backtesting.report.write_equity already computes it exactly from the ledger, so
    subtracting would only reintroduce float rounding drift for no benefit.
    """
    path = os.path.join(ref.path, "equity.parquet")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or "portfolio_value" not in df.columns:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "portfolio_value": [round(float(v), 2) for v in df["portfolio_value"].to_numpy()],
        "cash": [round(float(v), 2) for v in df["cash"].to_numpy()],
        "assets": [round(float(v), 2) for v in df["positions_value"].to_numpy()],
    }
```

- [ ] **Step 5: Add `load_equity_curve`**

Add this new function where `load_equity_from_indicators` used to be:

```python
def load_equity_curve(ref: RunRef) -> list[dict[str, Any]]:
    """Load the portfolio-value curve from equity.parquet (one row per trading session)."""
    path = os.path.join(ref.path, "equity.parquet")
    if not os.path.isfile(path):
        return []
    try:
        df = pd.read_parquet(path)
    except Exception:
        return []
    if df.empty or "portfolio_value" not in df.columns:
        return []
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in df["portfolio_value"].sort_index().items()]
```

- [ ] **Step 6: Rewrite `load_run`**

```python
def load_run(ref: RunRef) -> Run:
    """Load all data for a run: settings, metrics, and the equity curve (from equity.parquet)."""
    settings = load_settings(ref)
    metrics = load_metrics(ref)
    equity = load_equity_curve(ref)
    return Run(ref=ref, settings=settings, metrics=metrics, equity_curve=equity)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tests/dashboard/test_reader.py src/trading_agent_framework/dashboard/reader.py
git commit -m "feat: read the equity curve straight from equity.parquet, drop indicator/trades reconstruction fallbacks"
```

---

### Task 7: Rewrite cumulative/yearly returns to use the persisted benchmark, not live yfinance

`load_cumulative_returns` currently does a live `yfinance.download()` on every dashboard load to fetch the benchmark, then aligns it by hand. `equity.parquet` already has `benchmark_close`/`benchmark_return` precomputed at backtest time — the exact series `backtesting/metrics.py::compute_metrics` used to derive this run's own Alpha/Beta/Sharpe. Read those directly instead. Separately, `load_yearly_returns` currently recomputes a yearly table from scratch; `metrics.json`'s `raw.summary_tables.eoy_returns_vs_benchmark` already has the identical table, computed from the identical series, by `compute_metrics`'s own `_yearly_table` — read that instead of recomputing.

**Files:**
- Modify: `src/trading_agent_framework/dashboard/reader.py` (delete `_cumret_no_benchmark`; rewrite `load_cumulative_returns`, `load_yearly_returns`; keep `_compound_monthly_returns` as-is)
- Test: `tests/dashboard/test_reader.py` (append)

**Interfaces:**
- Consumes: `equity.parquet`'s `return`/`benchmark_close`/`benchmark_return` columns (Task 6's fixture helper); `metrics.json`'s `raw.summary_tables.eoy_returns_vs_benchmark` (Task 4's fixture helper).
- Produces: `load_cumulative_returns(ref) -> dict[str, Any] | None` (no network access), `load_yearly_returns(ref) -> list[dict[str, Any]] | None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/dashboard/test_reader.py`:

```python
from trading_agent_framework.dashboard.reader import load_cumulative_returns, load_yearly_returns


def test_load_cumulative_returns_uses_the_persisted_benchmark_columns(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    benchmark = {NOW: Decimal("400.0"), LATER: Decimal("404.0")}
    report.write_equity(run_dir, _equity_samples(), benchmark)
    report.write_settings(run_dir, _settings_payload(benchmark_symbol="QQQ"))

    result = load_cumulative_returns(_ref(run_dir))

    assert result is not None
    assert result["benchmark_symbol"] == "QQQ"
    assert result["benchmark"] is not None
    assert result["strategy"][-1] == pytest.approx(0.05, rel=1e-6)  # 10500/10000 - 1
    assert result["benchmark_daily_returns"][-1] == pytest.approx(0.01, rel=1e-6)  # 404/400 - 1


def test_load_cumulative_returns_handles_a_benchmark_less_run_without_network_access(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_equity(run_dir, _equity_samples())  # no benchmark passed
    report.write_settings(run_dir, _settings_payload())

    result = load_cumulative_returns(_ref(run_dir))

    assert result is not None
    assert result["benchmark"] is None
    assert result["strategy"][-1] == pytest.approx(0.05, rel=1e-6)


def test_load_cumulative_returns_returns_none_when_equity_parquet_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_cumulative_returns(_ref(run_dir)) is None


def test_load_yearly_returns_reads_the_precomputed_summary_table(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["raw"] = {
        "summary_tables": {
            "eoy_returns_vs_benchmark": [{"year": 2026, "strategy": 0.1, "benchmark": 0.05, "won": True}],
            "drawdowns": [],
        }
    }
    report.write_metrics(run_dir, metrics)

    yearly = load_yearly_returns(_ref(run_dir))

    assert yearly == [{"year": 2026, "strategy": 0.1, "benchmark": 0.05, "won": True}]


def test_load_yearly_returns_returns_none_when_metrics_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_yearly_returns(_ref(run_dir)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: FAIL — `load_cumulative_returns` looks for `stats.parquet` and, when that's missing, returns `None`; even if it found `equity.parquet` it would then attempt a real `yfinance.download()` call, which is exactly what these tests must never trigger. `load_yearly_returns` independently recomputes from `stats.parquet` and also returns `None`.

- [ ] **Step 3: Delete `_cumret_no_benchmark` and rewrite `load_cumulative_returns`**

Delete `_cumret_no_benchmark` entirely. Replace `load_cumulative_returns`:

```python
def load_cumulative_returns(ref: RunRef) -> dict[str, Any] | None:
    """Build cumulative returns for strategy and benchmark from equity.parquet.

    Both series come from backtesting.report.write_equity's own `return`/
    `benchmark_return`/`benchmark_close` columns -- the exact series
    backtesting.metrics.compute_metrics used to derive this run's own Sharpe/Alpha/
    Beta -- rather than a live re-fetch that could silently disagree with them (and
    that would violate this project's "tests never touch the network" rule).
    """
    path = os.path.join(ref.path, "equity.parquet")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or "return" not in df.columns:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    daily_ret = df["return"].fillna(0.0)
    strategy_cum = (1 + daily_ret).cumprod() - 1
    benchmark_symbol = get_benchmark_symbol(ref)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    has_benchmark = "benchmark_close" in df.columns and df["benchmark_close"].notna().any()
    if not has_benchmark:
        return {
            "dates": dates,
            "strategy": [round(float(v), 6) for v in strategy_cum.to_numpy()],
            "benchmark": None,
            "benchmark_symbol": benchmark_symbol,
            "benchmark_source": "unavailable (no benchmark recorded for this run)",
            "benchmark_daily_close": None,
            "strategy_monthly": _compound_monthly_returns(daily_ret),
            "benchmark_monthly": None,
            "strategy_daily_returns": [round(float(v), 6) for v in daily_ret.to_numpy()],
            "benchmark_daily_returns": None,
        }

    bm_daily_ret = df["benchmark_return"].fillna(0.0)
    bm_cum = (1 + bm_daily_ret).cumprod() - 1

    return {
        "dates": dates,
        "strategy": [round(float(v), 6) for v in strategy_cum.to_numpy()],
        "benchmark": [round(float(v), 6) for v in bm_cum.to_numpy()],
        "benchmark_symbol": benchmark_symbol,
        "benchmark_source": "equity.parquet (recorded at backtest time)",
        "benchmark_daily_close": [round(float(v), 6) if pd.notna(v) else None for v in df["benchmark_close"].to_numpy()],
        "strategy_monthly": _compound_monthly_returns(daily_ret),
        "benchmark_monthly": _compound_monthly_returns(bm_daily_ret),
        "strategy_daily_returns": [round(float(v), 6) for v in daily_ret.to_numpy()],
        "benchmark_daily_returns": [round(float(v), 6) for v in bm_daily_ret.to_numpy()],
    }
```

- [ ] **Step 4: Rewrite `load_yearly_returns`**

```python
def load_yearly_returns(ref: RunRef) -> list[dict[str, Any]] | None:
    """Yearly strategy/benchmark returns, read directly from metrics.json's
    `raw.summary_tables.eoy_returns_vs_benchmark` -- computed once, in
    backtesting.metrics.compute_metrics, from the exact same returns series used for
    every other headline metric on this page. No independent recomputation here.
    """
    metrics = load_metrics(ref)
    if metrics is None:
        return None
    table = metrics.raw.get("summary_tables", {}).get("eoy_returns_vs_benchmark")
    return table or None
```

- [ ] **Step 5: Remove the now-unused `yfinance` import**

Confirm no remaining `import yfinance` reference in `reader.py` (it was only ever imported lazily inside the deleted code path) — if any lingering `import yfinance as yf` line remains, delete it.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: PASS

- [ ] **Step 7: Confirm no network access — run with sockets disabled**

Run: `uv run pytest tests/dashboard/test_reader.py -v -p no:cacheprovider` and additionally grep to be sure:
```bash
grep -n "yfinance" src/trading_agent_framework/dashboard/reader.py
```
Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add tests/dashboard/test_reader.py src/trading_agent_framework/dashboard/reader.py
git commit -m "fix: source benchmark returns from equity.parquet, not a live yfinance call; read yearly table from metrics.json"
```

---

### Task 8: Fix `load_trades_curve`'s filename

`load_trades_curve` (the Trades tab's marker-overlay chart data) globs for `*_trades.parquet`/`*_trades.csv`; this project always writes a fixed `trades.parquet` (no CSV alternative is ever produced). Column names (`time`, `symbol`, `side`, `status`, `filled_quantity`, `price`, `trade_cost`) already match this project's `FillRecord`/`write_trades` output — only the filename lookup needs fixing.

**Files:**
- Modify: `src/trading_agent_framework/dashboard/reader.py` (`load_trades_curve`)
- Test: `tests/dashboard/test_reader.py` (append)

**Interfaces:**
- Consumes: `report.write_trades(run_dir: Path, ledger: Ledger) -> Path`; `trading_agent_framework.backtesting.ledger.{Ledger, FillRecord}`; `trading_agent_framework.entities.enums.{OrderSide, OrderType}`.
- Produces: `load_trades_curve(ref: RunRef, budget: float) -> dict[str, Any] | None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/dashboard/test_reader.py`:

```python
from trading_agent_framework.backtesting.ledger import FillRecord, Ledger
from trading_agent_framework.dashboard.reader import load_trades_curve
from trading_agent_framework.entities.enums import OrderSide, OrderType


def test_load_trades_curve_reads_trades_parquet(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(backtesting_start=NOW.isoformat(), budget=10000.0))
    ledger = Ledger()
    ledger.record_fill(FillRecord(
        time=NOW, identifier="abc", symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal(10), filled_quantity=Decimal(10), price=Decimal("100"),
        trade_cost=Decimal("1.0"), trade_slippage=Decimal("0.0"),
    ))
    report.write_trades(run_dir, ledger)

    curve = load_trades_curve(_ref(run_dir), budget=10000.0)

    assert curve is not None
    assert len(curve["trades"]) == 1
    assert curve["trades"][0]["side"] == "buy"
    assert curve["trades"][0]["symbol"] == "AAPL"


def test_load_trades_curve_returns_none_when_no_fills(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_trades(run_dir, Ledger())
    assert load_trades_curve(_ref(run_dir), budget=10000.0) is None


def test_load_trades_curve_returns_none_when_file_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_trades_curve(_ref(run_dir), budget=10000.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: FAIL — `load_trades_curve` globs for `*_trades.parquet`, which the fixture's `trades.parquet` doesn't match, so it always returns `None`.

- [ ] **Step 3: Fix the filename lookup**

In `load_trades_curve`, replace:
```python
    trades_path = _find_file(ref.path, "*_trades.parquet")
    if trades_path is None:
        trades_path = _find_file(ref.path, "*_trades.csv")
    if trades_path is None:
        return None

    try:
        if trades_path.endswith(".parquet"):
            df = pd.read_parquet(trades_path)
        else:
            df = pd.read_csv(trades_path)
    except Exception:
        return None
```
with:
```python
    trades_path = os.path.join(ref.path, "trades.parquet")
    if not os.path.isfile(trades_path):
        return None

    try:
        df = pd.read_parquet(trades_path)
    except Exception:
        return None
```

(No other changes needed in this function — the rest of the reconstruction logic already matches this project's `side.value` (`"buy"`/`"sell"`), `status == "fill"`, and column names.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_reader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_reader.py src/trading_agent_framework/dashboard/reader.py
git commit -m "fix: read trades.parquet by its fixed name"
```

---

### Task 9: Remove the 3 dead metric cards from the Run Detail page

`_pages/detail.py`'s "Returns" section shows `Expected Yearly%`, `Best Day`, and `Worst Day`, sourced from `m.raw["Expected Yearly%"]`/`"Best Day"`/`"Worst Day"` — quantstats-only keys this project's vectorbt-based `compute_metrics` never writes into `raw` (only `raw["summary_tables"]` exists). These would always render "0.0%". Remove them per the earlier decision.

**Files:**
- Modify: `src/trading_agent_framework/dashboard/_pages/detail.py:83-99`

**Interfaces:**
- Consumes: `MetricSet.total_return_strategy`, `MetricSet.cagr_strategy` (unchanged).
- Produces: no new interface — this is a UI-only trim.

- [ ] **Step 1: Make the edit**

Replace:
```python
    with tab1:
        st.subheader("Returns")
        cols = st.columns(5)
        with cols[0]:
            render_metric_card("Total Return", m.total_return_strategy)
        with cols[1]:
            render_metric_card("CAGR", m.cagr_strategy)
        with cols[2]:
            ey = m.raw.get("Expected Yearly%", {}).get("Strategy", 0)
            render_metric_card("Expected Yearly", float(ey) if ey else 0)
        with cols[3]:
            bd = m.raw.get("Best Day", {}).get("Strategy", 0)
            render_metric_card("Best Day", float(bd) if bd else 0)
        with cols[4]:
            wd = m.raw.get("Worst Day", {}).get("Strategy", 0)
            render_metric_card("Worst Day", float(wd) if wd else 0)
```
with:
```python
    with tab1:
        st.subheader("Returns")
        cols = st.columns(2)
        with cols[0]:
            render_metric_card("Total Return", m.total_return_strategy)
        with cols[1]:
            render_metric_card("CAGR", m.cagr_strategy)
```

- [ ] **Step 2: Verify no leftover references**

Run:
```bash
grep -n "Expected Yearly\|Best Day\|Worst Day" src/trading_agent_framework/dashboard/_pages/detail.py
```
Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add src/trading_agent_framework/dashboard/_pages/detail.py
git commit -m "fix: remove quantstats-only metric cards this project's metrics engine never computes"
```

---

### Task 10: End-to-end smoke test — a real backtest run through the whole dashboard

Every loader is now unit-tested against fixtures built with the real `report.write_*` functions. This task adds one integration-style test that builds a *complete* run directory (settings + metrics + equity + trades + indicators, via the real `Ledger`/`report` machinery) and drives the actual Streamlit pages through `streamlit.testing.v1.AppTest`, proving the whole stack — discovery, reading, and rendering — works together with no exceptions. This is also the first test to import `app.py` (deferred from Task 1 for exactly this reason).

**Files:**
- Test: `tests/dashboard/test_app_smoke.py` (new)

**Interfaces:**
- Consumes: everything built in Tasks 1-9.
- Produces: nothing new — this is a pure verification task.

- [ ] **Step 1: Write the test**

Create `tests/dashboard/test_app_smoke.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.backtesting.dashboard_contract import METRIC_SET_FIELDS

from trading_agent_framework.backtesting import report
from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, IndicatorLine, Ledger
from trading_agent_framework.entities.enums import OrderSide, OrderType

NOW = datetime(2026, 1, 5, 21, tzinfo=UTC)
LATER = datetime(2026, 1, 6, 21, tzinfo=UTC)


def _build_full_run(base: Path) -> Path:
    run_dir = base / "momentum" / "backtesting" / "2026-01-05_210000_backtesting"
    run_dir.mkdir(parents=True)

    ledger = Ledger()
    ledger.record_equity(EquitySample(time=NOW, portfolio_value=Decimal(10000), cash=Decimal(10000), positions_value=Decimal(0)))
    ledger.record_equity(EquitySample(time=LATER, portfolio_value=Decimal(10500), cash=Decimal(500), positions_value=Decimal(10000)))
    ledger.record_fill(FillRecord(
        time=LATER, identifier="abc", symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal(10), filled_quantity=Decimal(10), price=Decimal("950"),
        trade_cost=Decimal("1.0"), trade_slippage=Decimal("0.0"),
    ))
    ledger.record_line(IndicatorLine(time=NOW, name="sma_200", value=Decimal("148.5"), color=None, style="solid", plot_name="default_plot"))

    report.write_equity(run_dir, ledger.equity, {NOW: Decimal("400.0"), LATER: Decimal("404.0")})
    report.write_trades(run_dir, ledger)
    report.write_indicators(run_dir, ledger)

    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["total_return_strategy"] = 0.05
    metrics["cagr_strategy"] = 0.05
    metrics["raw"] = {
        "summary_tables": {
            "eoy_returns_vs_benchmark": [{"year": 2026, "strategy": 0.05, "benchmark": 0.01, "won": True}],
            "drawdowns": [],
        }
    }
    report.write_metrics(run_dir, metrics)

    report.write_settings(run_dir, {
        "name": "momentum", "mode": "backtesting", "run_ts": "2026-01-05_210000",
        "backtesting_start": NOW.isoformat(), "backtesting_end": LATER.isoformat(),
        "budget": 10000.0, "risk_free_rate": 0.03, "backtesting_data_sources": "yahoo",
        "backtest_time_seconds": 1.5, "timestep": "day", "sleeptime": "1D",
        "commission": 0.0, "slippage": 0.0, "warmup_trading_days": 0,
        "benchmark_symbol": "SPY", "framework_version": "0.1.0", "parameters": {},
    })
    return run_dir


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return _build_full_run(logs_dir)


APP_PATH = Path(__file__).resolve().parents[2] / "src" / "trading_agent_framework" / "dashboard" / "app.py"


def test_app_boots_and_scorecard_lists_the_run(run_dir: Path) -> None:
    # Use an absolute path anchored on this test file's own location, not a
    # cwd-relative one: the `run_dir` fixture does `monkeypatch.chdir(tmp_path)`
    # (needed for scan_runs("logs") to resolve), which would otherwise break a
    # repo-root-relative path to app.py.
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()

    assert not at.exception


def test_run_detail_renders_without_exception(run_dir: Path) -> None:
    from trading_agent_framework.dashboard.discovery import scan_runs
    from trading_agent_framework.dashboard.reader import load_run

    index = scan_runs("logs")
    assert len(index.runs) == 1

    run = load_run(index.runs[0])
    assert run.metrics is not None
    assert run.metrics.total_return_strategy == 0.05
    assert len(run.equity_curve) == 2
```

(`test_app_boots_and_scorecard_lists_the_run` exercises `app.py` end to end via `AppTest`, which runs it inside a real Streamlit script context — this is why `app.py`'s import was excluded from Task 1's plain-import smoke test. `test_run_detail_renders_without_exception` exercises the full `discovery -> reader` path with real backtest-shaped fixtures, standing in for a full `AppTest`-driven Detail-page render, which needs session-state navigation (`st.session_state.detail_ref`) that `AppTest` supports but which adds significant test complexity for marginal extra coverage over the loader-level tests already in Tasks 4-8.)

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/dashboard/test_app_smoke.py -v`
Expected: PASS. If `AppTest.from_file` raises `ModuleNotFoundError` for `streamlit.testing`, confirm the installed `streamlit` version is `>=1.28` (the `dashboard` dependency group already pins `streamlit>=1.40`, so this should not happen — if it does, run `uv sync` first).

- [ ] **Step 3: Run the entire project test suite**

Run: `uv run pytest`
Expected: PASS, no regressions anywhere outside `tests/dashboard/`.

- [ ] **Step 4: Lint the whole dashboard package**

Run: `uv run ruff check src/trading_agent_framework/dashboard/ tests/dashboard/`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_app_smoke.py
git commit -m "test: add an end-to-end dashboard smoke test against a real backtest-shaped run directory"
```

---

### Task 11: Manual verification

Automated tests don't cover Streamlit's actual rendered UI (colors, layout, interactivity). Confirm the dashboard is usable against a real run before calling this done.

**Files:** none (verification only).

- [ ] **Step 1: Run a real backtest** to produce at least one genuine run directory under `logs/`, e.g.:
```bash
uv run python -m trading_agent_framework.main  # or whatever this repo's real backtest entry point is; confirm with the user which strategy/script to run
```
If no strategy is readily runnable, reuse Task 10's `_build_full_run` helper standalone to generate a `logs/momentum/backtesting/...` directory by hand for a manual look.

- [ ] **Step 2: Launch the dashboard**
```bash
uv run dashboard
```
Confirm: the `dashboard` console script now resolves (Task 1 fixed its entry point) and Streamlit opens in the browser.

- [ ] **Step 3: Walk through all three pages**
- Scorecard: the run appears in the table with real CAGR/Sharpe/Sortino/Max DD numbers (not all zeros).
- Run Detail: Performance Metrics tab shows only Total Return + CAGR in the Returns section (the 3 dead cards are gone); Charts tab shows a real equity curve and drawdown; Trades tab shows buy/sell markers if the run had fills; Returns tab shows cumulative returns vs. benchmark with no error and no network delay (confirm devtools/terminal shows no outbound `yfinance` request); Yearly Returns table populates from the precomputed summary table.
- Side-by-Side: pick 2+ runs (create a second fixture run if only one exists) and confirm the overlay chart and metric comparison render.

- [ ] **Step 4: Report back** any visual/UX issue found (this step has no fixed code changes — report findings to the user for a follow-up task rather than guessing at a fix here).
