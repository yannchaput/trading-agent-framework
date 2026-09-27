# Dashboard: dark theme, top tabs, Models page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Streamlit dashboard a dark trading theme, a logo with a top row of two tabs (Backtesting = today's pages, Models = vLLM benchmark comparison), with the backtesting sidebar kept on the Backtesting tab.

**Architecture:** A Streamlit-free reader (`dashboard/benchmark_reader.py`) turns `../benchmark-vllm-models/results/<YYYYMMDD-HHMMSS>/` into frozen dataclasses (`dashboard/models.py`). Pure chart/table builders render them; `_pages/models.py` wires them into a page with a cached loader. `app.py` becomes a shell running `st.navigation(position="top")` over two function pages; today's `main()` moves verbatim to `_pages/backtesting.py`. The dark theme is passed by `cli.py` as `--theme.*` flags plus a shared plotly `CHART_TEMPLATE`.

**Tech Stack:** Python 3.14, Streamlit 1.64 (`st.navigation`, `st.logo`, `AppTest`), plotly 5.24, pandas, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-09-27-dashboard-dark-tabs-models-design.md`

## Global Constraints

- Dashboard is **read-only**: never write into the benchmark results directory.
- Results dir: `--benchmark-dir PATH` (CLI flag) wins; default `find_project_root() / ".." / "benchmark-vllm-models" / "results"`, resolved. No env var, no `.env`.
- A benchmark run is complete only when both `meta.json` and `summary.json` exist; others are not listed.
- New benchmark dataclasses are `@dataclass(frozen=True)`; collection fields are tuples (dicts allowed for keyed scores). Function **returns** use concrete `list`; function **parameters** taking collections use `Sequence`.
- `benchmark_reader.py` never imports `streamlit`; caching lives in the page (`st.cache_data`).
- Reader failures raise `BenchmarkReadError`, never a raw `JSONDecodeError`/`KeyError`/`OSError`.
- Theme colours (verbatim): background `#0b0e14`, secondary background `#131722`, text `#d1d4dc`, primary `#22d3ee`, positive `#22c55e`, negative `#ef4444`, borders/grid `#2a2e39`, muted label `#8a8f98`.
- Logo text: `Yann's Trading Bots`.
- `message trace` (`trace` key in `.jsonl`) is never loaded into dataclasses or shown.
- Existing backtesting pages' content and session-state keys (`current_page`, `selected_refs`, `detail_ref`, `compare_refs`, `run_index`, `show_description_editor`, `description_editor_ref`) do not change.
- Tests never touch the network and use hand-built fixtures, no `MagicMock`.
- `AppTest.switch_page` cannot reach a callable `st.Page` (verified on 1.64): the Models page is tested with `AppTest.from_function`; the tab list is tested through the plain-data `NAV_PAGES`.

## Review Focus

- A model in `summary.json` but missing from `meta.json` `models` → still shown, with `served_name`/`vllm_version` `None` (test in Task 1).
- `--benchmark-dir ~/x` or a relative path → expanded and resolved, not taken literally (test in Task 1).
- A scenario in `meta.json` with no score for a model → heatmap cell `—`, no crash; drill-down default still picks a scenario (tests in Tasks 5 and 6).
- A run where no model ran (`ran: false` everywhere) → page shows a warning and the table, no `max()` on an empty list (test in Task 6).
- Switching the run picker to an older run with a different model set → drill-down selectboxes reset (keys scoped per run), no stale `index` error (test in Task 6).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/trading_agent_framework/dashboard/models.py` (modify) | + `BenchmarkRunRef`, `ScenarioScore`, `BenchmarkModel`, `BenchmarkRun`, `Check`, `ScenarioRun`, `ScenarioRuns` |
| `src/trading_agent_framework/dashboard/benchmark_reader.py` (new) | Locate results dir, scan runs, load a run, load one scenario's runs. Streamlit-free. |
| `src/trading_agent_framework/dashboard/cli.py` (modify) | Build the `streamlit run` command: theme flags, `--benchmark-dir` forwarding |
| `src/trading_agent_framework/dashboard/theme.py` (modify) | Dark CSS for `.metric-card`, `.strategy-badge` |
| `src/trading_agent_framework/dashboard/components/charts.py` (modify) | `CHART_TEMPLATE`, colour retune, + 3 benchmark charts |
| `src/trading_agent_framework/dashboard/components/tables.py` (modify) | + `benchmark_summary_frame`, `render_benchmark_summary_table`, `scenario_runs_frame` |
| `src/trading_agent_framework/dashboard/components/metric_cards.py` (modify) | `render_metric_card(..., tone=None)` |
| `src/trading_agent_framework/dashboard/_pages/models.py` (new) | Models tab page |
| `src/trading_agent_framework/dashboard/_pages/backtesting.py` (new) | Today's `app.main()` body, verbatim |
| `src/trading_agent_framework/dashboard/app.py` (modify) | Shell: page config, theme, logo, `st.navigation` |
| `src/trading_agent_framework/dashboard/_pages/side_by_side.py` (modify) | Template + palette |
| `src/trading_agent_framework/dashboard/assets/logo.svg`, `logo-icon.svg` (new) | Logo |
| `tests/dashboard/benchmark_fixtures.py` (new) | Builds a results tree under `tmp_path` |
| `tests/dashboard/test_benchmark_reader.py` (new) | Reader tests |
| `tests/dashboard/test_cli.py` (new) | Command-building tests |
| `tests/dashboard/test_dark_theme.py` (new) | Template/CSS tests |
| `tests/dashboard/test_benchmark_components.py` (new) | Chart/table builder tests |
| `tests/dashboard/test_models_page.py` (new) | Models page `AppTest` tests |
| `tests/dashboard/test_app_smoke.py` (modify) | Navigation tests |
| `tests/dashboard/test_package_imports.py` (modify) | New modules in `MODULES` |
| `CLAUDE.md`, `README.md` (modify) | Docs |

Run all commands from the repo root `/home/yann/projets/trading-agent-framework`.

---

### Task 1: Benchmark types and run reader

**Files:**
- Modify: `src/trading_agent_framework/dashboard/models.py`
- Create: `src/trading_agent_framework/dashboard/benchmark_reader.py`
- Create: `tests/dashboard/benchmark_fixtures.py`
- Create: `tests/dashboard/test_benchmark_reader.py`
- Modify: `tests/dashboard/test_package_imports.py`

**Interfaces:**
- Consumes: `trading_agent_framework.config.env.find_project_root(start: Path | None = None) -> Path` (raises `trading_agent_framework.utils.errors.ConfigurationError`).
- Produces:
  - `models.BenchmarkRunRef(path: Path, run_id: str, started_at: datetime)`, `BenchmarkRunRef.from_path(path: Path) -> BenchmarkRunRef` (raises `ValueError` for a non-`YYYYMMDD-HHMMSS` name)
  - `models.ScenarioScore(passed: int, runs: int, mean_partial: float)`
  - `models.BenchmarkModel(key, display_name, served_name, vllm_version, ran, error, overall, mean_partial, categories, scenarios, runs_passed, runs_total, text_tool_calls, avg_tool_calls, median_run_s, median_tokens_per_s, timeouts, errors)`
  - `models.BenchmarkRun(ref, started_at, finished_at, repeats, timeout_s, scenarios: tuple[str, ...], models: tuple[BenchmarkModel, ...])` with property `categories -> tuple[str, ...]`
  - `benchmark_reader.BENCHMARK_DIR_FLAG = "--benchmark-dir"`
  - `benchmark_reader.BenchmarkReadError(Exception)`
  - `benchmark_reader.split_benchmark_dir(argv: Sequence[str]) -> tuple[list[str], str | None]`
  - `benchmark_reader.resolve_results_dir(argv: Sequence[str], start: Path | None = None) -> Path`
  - `benchmark_reader.scan_benchmark_runs(base: Path) -> list[BenchmarkRunRef]`
  - `benchmark_reader.load_benchmark_run(ref: BenchmarkRunRef) -> BenchmarkRun`
  - `tests/dashboard/benchmark_fixtures.py`: `LATEST`, `OLDER`, `IN_PROGRESS`, `SCENARIOS`, `build_results_tree(base: Path) -> Path`, `write_json(path: Path, data) -> None`, `record(...) -> dict`

- [ ] **Step 1: Write the fixture helper**

Create `tests/dashboard/benchmark_fixtures.py`:

```python
"""Hand-built vLLM benchmark results trees, shaped like the real
../benchmark-vllm-models/results/20260927-132243 run (meta.json, summary.json, <key>.jsonl)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LATEST = "20260927-132243"
OLDER = "20260925-233646"
IN_PROGRESS = "20260928-090000"  # newer than LATEST but has no summary.json yet
SCENARIOS = ["reasoning.rsi_signal", "reasoning.headline_trap", "tools.limit_order"]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _meta(models: list[tuple[str, str, str]], started: str, finished: str) -> dict[str, Any]:
    return {
        "started_at": started,
        "finished_at": finished,
        "models": [
            {"key": key, "display_name": name, "start_function": f"vllmStart{key}", "port": 8000 + i, "served_name": served}
            for i, (key, name, served) in enumerate(models)
        ],
        "vllm_versions": {key: "0.30.0" for key, _, _ in models},
        "repeats": 2,
        "timeout_s": 360.0,
        "scenarios": SCENARIOS,
        "cli_args": {"only": None, "repeats": 2, "scenarios": "*", "timeout": 360.0},
    }


def _summary(key: str, name: str, **fields: Any) -> dict[str, Any]:
    entry = {
        "key": key, "display_name": name, "ran": True, "error": None,
        "overall": 0.5, "mean_partial": 0.9, "categories": {"reasoning": 0.5, "tools": 0.5},
        # Keys sorted alphabetically, like the real file -- NOT in meta.json's scenario order.
        "scenarios": {
            "reasoning.headline_trap": {"passed": 2, "runs": 2, "mean_partial": 1.0},
            "reasoning.rsi_signal": {"passed": 0, "runs": 2, "mean_partial": 0.8},
            "tools.limit_order": {"passed": 1, "runs": 2, "mean_partial": 0.9},
        },
        "runs_passed": 3, "runs_total": 6, "text_tool_calls": 0, "avg_tool_calls": 3.24,
        "median_run_s": 4.9, "median_tokens_per_s": 131.7, "timeouts": 0, "errors": 0,
    }
    entry.update(fields)
    return entry


def record(
    scenario_id: str,
    repeat: int,
    *,
    passed: bool,
    partial: float,
    status: str = "ok",
    error: str | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One <key>.jsonl line, trace included (the reader must drop it)."""
    if checks is None:
        checks = [
            {"type": "called", "passed": passed, "reason": "found" if passed else "no call matching submit_order{'symbol': 'DOCU'}"},
            {"type": "max_tool_calls", "passed": True, "reason": "4 tool calls (max 10)"},
        ]
    return {
        "scenario_id": scenario_id, "category": scenario_id.split(".")[0], "repeat": repeat,
        "status": status, "passed": passed, "partial": partial, "checks": checks,
        "metrics": {
            "total_s": 6.906, "tool_calls": 4, "model_calls": 5, "tool_arg_errors": 0, "text_tool_calls": 0,
            "completion_tokens": 882, "reasoning_chars": 2506, "tokens_per_s": 121.35,
        },
        "error": error,
        "trace": {"sessions": [{"index": 1, "prompt": "p", "messages": [{"role": "human", "content": "x" * 50}]}]},
    }


def _write_jsonl(path: Path, lines: list[dict[str, Any] | str]) -> None:
    path.write_text("\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def build_results_tree(base: Path) -> Path:
    """LATEST (glm + qwen3627b, complete, one malformed .jsonl line, one timeout),
    OLDER (glm + gptoss where gptoss did not run), IN_PROGRESS (no summary.json),
    a non-timestamp folder and a stray file. Returns `base`."""
    latest = base / LATEST
    write_json(latest / "meta.json", _meta(
        [("glm", "GLM-4.7-Flash", "glm-4.7-flash"), ("qwen3627b", "Qwen3.6-27B-AWQ", "qwen3.6-27b-awq")],
        "2026-09-27T13:22:43", "2026-09-27T16:08:15",
    ))
    write_json(latest / "summary.json", [
        _summary("glm", "GLM-4.7-Flash"),
        _summary(
            "qwen3627b", "Qwen3.6-27B-AWQ",
            overall=0.83, mean_partial=0.99, categories={"reasoning": 0.75, "tools": 1.0},
            scenarios={
                "reasoning.headline_trap": {"passed": 2, "runs": 2, "mean_partial": 1.0},
                "reasoning.rsi_signal": {"passed": 1, "runs": 2, "mean_partial": 0.96},
                "tools.limit_order": {"passed": 2, "runs": 2, "mean_partial": 1.0},
            },
            runs_passed=5, text_tool_calls=1, avg_tool_calls=3.51, median_run_s=29.7,
            median_tokens_per_s=44.5, timeouts=1,
        ),
    ])
    _write_jsonl(latest / "qwen3627b.jsonl", [
        record("reasoning.rsi_signal", 2, passed=False, partial=0.0, status="timeout", error="run exceeded 360s", checks=[]),
        record("reasoning.headline_trap", 1, passed=True, partial=1.0),
        "{not json",
        record("reasoning.rsi_signal", 1, passed=True, partial=1.0),
        record("tools.limit_order", 1, passed=True, partial=1.0),
    ])
    _write_jsonl(latest / "glm.jsonl", [
        record("reasoning.rsi_signal", 1, passed=False, partial=0.8),
        record("reasoning.rsi_signal", 2, passed=False, partial=0.8),
    ])
    (latest / "glm.vllm.log").write_text("vllm log\n", encoding="utf-8")

    older = base / OLDER
    write_json(older / "meta.json", _meta(
        [("glm", "GLM-4.7-Flash", "glm-4.7-flash"), ("gptoss", "Gpt-OSS-20b", "gpt-oss-20b")],
        "2026-09-25T23:36:46", "2026-09-26T00:46:00",
    ))
    write_json(older / "summary.json", [
        _summary("glm", "GLM-4.7-Flash"),
        {
            "key": "gptoss", "display_name": "Gpt-OSS-20b", "ran": False, "error": "vLLM server failed to start",
            "overall": None, "mean_partial": None, "categories": {}, "scenarios": {},
            "runs_passed": 0, "runs_total": 0, "text_tool_calls": 0, "avg_tool_calls": None,
            "median_run_s": None, "median_tokens_per_s": None, "timeouts": 0, "errors": 0,
        },
    ])
    _write_jsonl(older / "glm.jsonl", [record("reasoning.rsi_signal", 1, passed=False, partial=0.8)])

    write_json(base / IN_PROGRESS / "meta.json", _meta([("glm", "GLM-4.7-Flash", "glm-4.7-flash")], "2026-09-28T09:00:00", ""))
    write_json(base / "notes" / "meta.json", {})
    write_json(base / "notes" / "summary.json", [])
    (base / "README.txt").write_text("not a run\n", encoding="utf-8")
    return base
```

- [ ] **Step 2: Write the failing reader tests**

Create `tests/dashboard/test_benchmark_reader.py`:

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from tests.dashboard.benchmark_fixtures import IN_PROGRESS, LATEST, OLDER, SCENARIOS, build_results_tree, write_json
from trading_agent_framework.dashboard.benchmark_reader import (
    BenchmarkReadError,
    load_benchmark_run,
    resolve_results_dir,
    scan_benchmark_runs,
    split_benchmark_dir,
)
from trading_agent_framework.dashboard.models import BenchmarkRunRef, ScenarioScore


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return build_results_tree(tmp_path / "results")


def _ref(results: Path, run_id: str) -> BenchmarkRunRef:
    return BenchmarkRunRef.from_path(results / run_id)


# --- scan ------------------------------------------------------------------------------


def test_scan_lists_only_complete_timestamped_runs_newest_first(results: Path) -> None:
    refs = scan_benchmark_runs(results)

    assert [ref.run_id for ref in refs] == [LATEST, OLDER]
    assert IN_PROGRESS not in [ref.run_id for ref in refs]
    assert refs[0].started_at == datetime(2026, 9, 27, 13, 22, 43)


def test_scan_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert scan_benchmark_runs(tmp_path / "nope") == []


def test_run_ref_rejects_a_non_timestamp_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        BenchmarkRunRef.from_path(tmp_path / "notes")


# --- load_benchmark_run ----------------------------------------------------------------------------


def test_load_joins_meta_and_summary_by_key(results: Path) -> None:
    run = load_benchmark_run(_ref(results, LATEST))

    q27 = next(m for m in run.models if m.key == "qwen3627b")
    assert q27.display_name == "Qwen3.6-27B-AWQ"
    assert q27.served_name == "qwen3.6-27b-awq"
    assert q27.vllm_version == "0.30.0"
    assert q27.overall == 0.83
    assert q27.categories == {"reasoning": 0.75, "tools": 1.0}
    assert q27.scenarios["reasoning.rsi_signal"] == ScenarioScore(passed=1, runs=2, mean_partial=0.96)
    assert (q27.runs_passed, q27.runs_total, q27.text_tool_calls, q27.timeouts) == (5, 6, 1, 1)
    assert run.repeats == 2
    assert run.timeout_s == 360.0
    assert run.finished_at == datetime(2026, 9, 27, 16, 8, 15)


def test_scenario_and_category_order_follow_meta_not_summary(results: Path) -> None:
    run = load_benchmark_run(_ref(results, LATEST))

    assert run.scenarios == tuple(SCENARIOS)
    assert run.categories == ("reasoning", "tools")


def test_a_model_that_did_not_run_is_kept_with_its_error(results: Path) -> None:
    run = load_benchmark_run(_ref(results, OLDER))

    gptoss = next(m for m in run.models if m.key == "gptoss")
    assert gptoss.ran is False
    assert gptoss.error == "vLLM server failed to start"
    assert gptoss.overall is None
    assert gptoss.median_run_s is None


def test_a_summary_model_missing_from_meta_has_no_served_name_or_version(results: Path) -> None:
    run_dir = results / LATEST
    summary = json.loads((run_dir / "summary.json").read_text())
    summary.append({**summary[0], "key": "ghost", "display_name": "Ghost"})
    write_json(run_dir / "summary.json", summary)

    ghost = next(m for m in load_benchmark_run(_ref(results, LATEST)).models if m.key == "ghost")
    assert ghost.display_name == "Ghost"
    assert ghost.served_name is None
    assert ghost.vllm_version is None


@pytest.mark.parametrize("content", ["{not json", '{"not": "a list"}'])
def test_a_malformed_summary_raises_benchmark_read_error(results: Path, content: str) -> None:
    (results / LATEST / "summary.json").write_text(content, encoding="utf-8")

    with pytest.raises(BenchmarkReadError):
        load_benchmark_run(_ref(results, LATEST))


def test_a_meta_without_scenarios_raises_benchmark_read_error(results: Path) -> None:
    write_json(results / LATEST / "meta.json", {"models": [], "repeats": 2, "timeout_s": 360.0})

    with pytest.raises(BenchmarkReadError):
        load_benchmark_run(_ref(results, LATEST))


# --- results dir -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "rest", "value"),
    [
        (["--server.port", "8502"], ["--server.port", "8502"], None),
        (["--benchmark-dir", "/data/r", "--server.port", "8502"], ["--server.port", "8502"], "/data/r"),
        (["--benchmark-dir=/data/r"], [], "/data/r"),
    ],
)
def test_split_benchmark_dir(argv: list[str], rest: list[str], value: str | None) -> None:
    assert split_benchmark_dir(argv) == (rest, value)


def test_split_benchmark_dir_without_a_value_is_an_error() -> None:
    with pytest.raises(SystemExit):
        split_benchmark_dir(["--benchmark-dir"])


def test_the_flag_wins_and_is_expanded_and_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_results_dir(["--benchmark-dir", "~/bench/results"]) == (tmp_path / "bench" / "results").resolve()


def test_the_default_sits_next_to_the_project_root_not_the_cwd(tmp_path: Path) -> None:
    project = tmp_path / "trading-agent-framework"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    assert resolve_results_dir([], start=project / "src") == (tmp_path / "benchmark-vllm-models" / "results").resolve()
```

Add to `MODULES` in `tests/dashboard/test_package_imports.py` (after `"trading_agent_framework.dashboard.reader",`):

```python
    "trading_agent_framework.dashboard.benchmark_reader",
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_benchmark_reader.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'BenchmarkRunRef'` (or `ModuleNotFoundError: ...benchmark_reader`).

- [ ] **Step 4: Add the dataclasses to `models.py`**

In `src/trading_agent_framework/dashboard/models.py`, add `from pathlib import Path` to the imports, and append at the end of the file:

```python
# --- vLLM benchmark results (read by benchmark_reader.py) ---------------------------------------


@dataclass(frozen=True)
class BenchmarkRunRef:
    """A benchmark run directory, <results>/<YYYYMMDD-HHMMSS>/. Built without reading any file."""

    path: Path
    run_id: str
    started_at: datetime

    @classmethod
    def from_path(cls, path: Path) -> BenchmarkRunRef:
        try:
            started_at = datetime.strptime(path.name, "%Y%m%d-%H%M%S")
        except ValueError as exc:
            raise ValueError(f"Not a benchmark run directory name: {path.name}") from exc
        return cls(path=path, run_id=path.name, started_at=started_at)


@dataclass(frozen=True)
class ScenarioScore:
    passed: int
    runs: int
    mean_partial: float


@dataclass(frozen=True)
class BenchmarkModel:
    """One model's line of summary.json, joined with its meta.json entry (served name, vLLM version).

    Numeric fields are None when the model did not run (``ran`` is False, ``error`` says why).
    """

    key: str
    display_name: str
    served_name: str | None
    vllm_version: str | None
    ran: bool
    error: str | None
    overall: float | None
    mean_partial: float | None
    categories: dict[str, float]
    scenarios: dict[str, ScenarioScore]
    runs_passed: int | None
    runs_total: int | None
    text_tool_calls: int | None
    avg_tool_calls: float | None
    median_run_s: float | None
    median_tokens_per_s: float | None
    timeouts: int | None
    errors: int | None


@dataclass(frozen=True)
class BenchmarkRun:
    ref: BenchmarkRunRef
    started_at: datetime
    finished_at: datetime | None
    repeats: int
    timeout_s: float
    scenarios: tuple[str, ...]  # meta.json order, e.g. "reasoning.rsi_signal"
    models: tuple[BenchmarkModel, ...]  # summary.json order

    @property
    def categories(self) -> tuple[str, ...]:
        """Scenario id prefixes, in first-seen scenario order."""
        return tuple(dict.fromkeys(scenario.split(".", 1)[0] for scenario in self.scenarios))
```

- [ ] **Step 5: Write `benchmark_reader.py`**

Create `src/trading_agent_framework/dashboard/benchmark_reader.py`:

```python
"""Read-only access to vLLM benchmark results (sibling repo benchmark-vllm-models).

Layout: <results>/<YYYYMMDD-HHMMSS>/ with meta.json, summary.json and one <model key>.jsonl per model.
A run is complete once summary.json exists; until then it is not listed. Streamlit-free, like
reader.py: the Models page adds the caching. Nothing here ever writes into the results directory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_agent_framework.config.env import find_project_root
from trading_agent_framework.dashboard.models import (
    BenchmarkModel,
    BenchmarkRun,
    BenchmarkRunRef,
    ScenarioScore,
)

BENCHMARK_DIR_FLAG = "--benchmark-dir"
DEFAULT_RESULTS_DIR = Path("..") / "benchmark-vllm-models" / "results"


class BenchmarkReadError(Exception):
    """A benchmark run's files are missing or malformed."""


def split_benchmark_dir(argv: Sequence[str]) -> tuple[list[str], str | None]:
    """Remove ``--benchmark-dir PATH`` / ``--benchmark-dir=PATH`` from argv; return (the rest, PATH)."""
    rest: list[str] = []
    value: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == BENCHMARK_DIR_FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{BENCHMARK_DIR_FLAG} needs a path")
            value = argv[i + 1]
            i += 2
            continue
        if arg.startswith(BENCHMARK_DIR_FLAG + "="):
            value = arg.split("=", 1)[1]
        else:
            rest.append(arg)
        i += 1
    return rest, value


def resolve_results_dir(argv: Sequence[str], start: Path | None = None) -> Path:
    """``--benchmark-dir`` if given, else benchmark-vllm-models/results next to the project root."""
    _, value = split_benchmark_dir(argv)
    if value is not None:
        return Path(value).expanduser().resolve()
    return (find_project_root(start) / DEFAULT_RESULTS_DIR).resolve()


def scan_benchmark_runs(base: Path) -> list[BenchmarkRunRef]:
    """Complete runs (meta.json and summary.json present) under `base`, newest first."""
    if not base.is_dir():
        return []
    refs: list[BenchmarkRunRef] = []
    for child in base.iterdir():
        if not (child.is_dir() and (child / "meta.json").is_file() and (child / "summary.json").is_file()):
            continue
        try:
            refs.append(BenchmarkRunRef.from_path(child))
        except ValueError:
            continue
    refs.sort(key=lambda ref: ref.started_at, reverse=True)
    return refs


def load_benchmark_run(ref: BenchmarkRunRef) -> BenchmarkRun:
    """Join meta.json and summary.json on the model key."""
    meta = _read_json(ref.path / "meta.json")
    summary = _read_json(ref.path / "summary.json")
    try:
        return _build_run(ref, meta, summary)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise BenchmarkReadError(f"Malformed benchmark run {ref.run_id}: {exc!r}") from exc


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReadError(f"Cannot read {path.name}: {exc}") from exc


def _build_run(ref: BenchmarkRunRef, meta: dict[str, Any], summary: Any) -> BenchmarkRun:
    if not isinstance(summary, list):
        raise TypeError("summary.json is not a list of models")
    meta_models = {entry["key"]: entry for entry in meta["models"]}
    versions = meta.get("vllm_versions") or {}
    return BenchmarkRun(
        ref=ref,
        started_at=_opt_datetime(meta.get("started_at")) or ref.started_at,
        finished_at=_opt_datetime(meta.get("finished_at")),
        repeats=int(meta["repeats"]),
        timeout_s=float(meta["timeout_s"]),
        scenarios=tuple(str(scenario) for scenario in meta["scenarios"]),
        models=tuple(
            _build_model(entry, meta_models.get(entry["key"], {}), versions.get(entry["key"])) for entry in summary
        ),
    )


def _build_model(entry: dict[str, Any], meta_model: dict[str, Any], vllm_version: str | None) -> BenchmarkModel:
    return BenchmarkModel(
        key=str(entry["key"]),
        display_name=str(entry.get("display_name") or meta_model.get("display_name") or entry["key"]),
        served_name=meta_model.get("served_name"),
        vllm_version=vllm_version,
        ran=bool(entry.get("ran", True)),
        error=entry.get("error"),
        overall=_opt_float(entry.get("overall")),
        mean_partial=_opt_float(entry.get("mean_partial")),
        categories={str(name): float(score) for name, score in (entry.get("categories") or {}).items()},
        scenarios={
            str(scenario_id): ScenarioScore(
                passed=int(score["passed"]), runs=int(score["runs"]), mean_partial=float(score["mean_partial"])
            )
            for scenario_id, score in (entry.get("scenarios") or {}).items()
        },
        runs_passed=_opt_int(entry.get("runs_passed")),
        runs_total=_opt_int(entry.get("runs_total")),
        text_tool_calls=_opt_int(entry.get("text_tool_calls")),
        avg_tool_calls=_opt_float(entry.get("avg_tool_calls")),
        median_run_s=_opt_float(entry.get("median_run_s")),
        median_tokens_per_s=_opt_float(entry.get("median_tokens_per_s")),
        timeouts=_opt_int(entry.get("timeouts")),
        errors=_opt_int(entry.get("errors")),
    )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _opt_datetime(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_benchmark_reader.py tests/dashboard/test_package_imports.py -q`
Expected: all PASS.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/dashboard tests/dashboard
git add src/trading_agent_framework/dashboard/models.py src/trading_agent_framework/dashboard/benchmark_reader.py tests/dashboard/benchmark_fixtures.py tests/dashboard/test_benchmark_reader.py tests/dashboard/test_package_imports.py
git commit -m "feat(dashboard): read vLLM benchmark runs (meta.json + summary.json)"
```

---

### Task 2: Per-scenario drill-down reader

**Files:**
- Modify: `src/trading_agent_framework/dashboard/models.py`
- Modify: `src/trading_agent_framework/dashboard/benchmark_reader.py`
- Modify: `tests/dashboard/test_benchmark_reader.py`

**Interfaces:**
- Consumes: Task 1's `BenchmarkRunRef`, `BenchmarkReadError`, `_opt_float`, `_opt_int`; fixtures `build_results_tree`, `LATEST`.
- Produces:
  - `models.Check(type: str, passed: bool, reason: str)`
  - `models.ScenarioRun(repeat: int, status: str, passed: bool, partial: float, error: str | None, checks: tuple[Check, ...], total_s: float | None, tool_calls: int | None, model_calls: int | None, tokens_per_s: float | None, completion_tokens: int | None)`
  - `models.ScenarioRuns(runs: tuple[ScenarioRun, ...], skipped_lines: int)`
  - `benchmark_reader.load_scenario_runs(ref: BenchmarkRunRef, model_key: str, scenario_id: str) -> ScenarioRuns`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_benchmark_reader.py` (and extend its imports: `load_scenario_runs` from `benchmark_reader`, `Check` from `models`):

```python
# --- load_scenario_runs ------------------------------------------------------------------------


def test_scenario_runs_are_filtered_and_sorted_by_repeat(results: Path) -> None:
    result = load_scenario_runs(_ref(results, LATEST), "qwen3627b", "reasoning.rsi_signal")

    assert [run.repeat for run in result.runs] == [1, 2]
    first = result.runs[0]
    assert (first.status, first.passed, first.partial) == ("ok", True, 1.0)
    assert first.checks[0] == Check(type="called", passed=True, reason="found")
    assert (first.total_s, first.tool_calls, first.model_calls, first.tokens_per_s, first.completion_tokens) == (
        6.906, 4, 5, 121.35, 882,
    )


def test_a_timed_out_run_is_kept_with_its_error(results: Path) -> None:
    result = load_scenario_runs(_ref(results, LATEST), "qwen3627b", "reasoning.rsi_signal")

    timed_out = result.runs[1]
    assert timed_out.status == "timeout"
    assert timed_out.error == "run exceeded 360s"
    assert timed_out.checks == ()


def test_malformed_lines_are_skipped_and_counted(results: Path) -> None:
    result = load_scenario_runs(_ref(results, LATEST), "qwen3627b", "tools.limit_order")

    assert len(result.runs) == 1
    assert result.skipped_lines == 1


def test_the_message_trace_is_not_kept(results: Path) -> None:
    run = load_scenario_runs(_ref(results, LATEST), "glm", "reasoning.rsi_signal").runs[0]

    assert not hasattr(run, "trace")


def test_a_missing_model_file_raises_benchmark_read_error(results: Path) -> None:
    with pytest.raises(BenchmarkReadError):
        load_scenario_runs(_ref(results, LATEST), "nope", "reasoning.rsi_signal")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_benchmark_reader.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'load_scenario_runs'`.

- [ ] **Step 3: Add the dataclasses**

Append to `src/trading_agent_framework/dashboard/models.py`:

```python
@dataclass(frozen=True)
class Check:
    type: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class ScenarioRun:
    """One repeat of one scenario for one model (a <key>.jsonl line, message trace dropped)."""

    repeat: int
    status: str  # "ok", "timeout", "error"
    passed: bool
    partial: float
    error: str | None
    checks: tuple[Check, ...]
    total_s: float | None
    tool_calls: int | None
    model_calls: int | None
    tokens_per_s: float | None
    completion_tokens: int | None


@dataclass(frozen=True)
class ScenarioRuns:
    runs: tuple[ScenarioRun, ...]
    skipped_lines: int  # lines of the whole file that were not valid records
```

- [ ] **Step 4: Implement `load_scenario_runs`**

In `benchmark_reader.py`, add `Check, ScenarioRun, ScenarioRuns` to the `models` import and add after `load_benchmark_run`:

```python
def load_scenario_runs(ref: BenchmarkRunRef, model_key: str, scenario_id: str) -> ScenarioRuns:
    """Every repeat of `scenario_id` in <model_key>.jsonl, sorted by repeat.

    Streams the file line by line; a line that is not a valid record is skipped and counted.
    """
    path = ref.path / f"{model_key}.jsonl"
    runs: list[ScenarioRun] = []
    skipped = 0
    try:
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record["scenario_id"] != scenario_id:
                        continue
                    runs.append(_build_scenario_run(record))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    skipped += 1
    except OSError as exc:
        raise BenchmarkReadError(f"Cannot read {path.name}: {exc}") from exc
    runs.sort(key=lambda run: run.repeat)
    return ScenarioRuns(runs=tuple(runs), skipped_lines=skipped)


def _build_scenario_run(record: dict[str, Any]) -> ScenarioRun:
    metrics = record.get("metrics") or {}
    return ScenarioRun(
        repeat=int(record["repeat"]),
        status=str(record["status"]),
        passed=bool(record["passed"]),
        partial=float(record["partial"]),
        error=record.get("error"),
        checks=tuple(
            Check(type=str(check["type"]), passed=bool(check["passed"]), reason=str(check.get("reason", "")))
            for check in record.get("checks") or []
        ),
        total_s=_opt_float(metrics.get("total_s")),
        tool_calls=_opt_int(metrics.get("tool_calls")),
        model_calls=_opt_int(metrics.get("model_calls")),
        tokens_per_s=_opt_float(metrics.get("tokens_per_s")),
        completion_tokens=_opt_int(metrics.get("completion_tokens")),
    )
```

Note: `"{not json"` raises `JSONDecodeError`; a JSON line that is a list/str raises `TypeError` on `record["scenario_id"]` — both counted.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_benchmark_reader.py -q`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/dashboard tests/dashboard
git add src/trading_agent_framework/dashboard/models.py src/trading_agent_framework/dashboard/benchmark_reader.py tests/dashboard/test_benchmark_reader.py
git commit -m "feat(dashboard): load one scenario's benchmark runs and checks from <model>.jsonl"
```

---

### Task 3: CLI — dark theme flags and `--benchmark-dir` forwarding

**Files:**
- Modify: `src/trading_agent_framework/dashboard/cli.py`
- Create: `tests/dashboard/test_cli.py`

**Interfaces:**
- Consumes: `benchmark_reader.split_benchmark_dir`, `benchmark_reader.BENCHMARK_DIR_FLAG`.
- Produces: `cli.THEME_ARGS: list[str]`, `cli.build_command(python: str, app_file: Path, argv: Sequence[str]) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

from trading_agent_framework.dashboard.cli import THEME_ARGS, build_command

APP = Path("/x/app.py")


def test_the_dark_theme_is_passed_as_flags() -> None:
    command = build_command("py", APP, [])

    assert command[:5] == ["py", "-m", "streamlit", "run", str(APP)]
    assert "--theme.base" in command and command[command.index("--theme.base") + 1] == "dark"
    assert command[command.index("--theme.backgroundColor") + 1] == "#0b0e14"
    assert "--" not in command


def test_user_args_come_after_the_defaults_so_they_override() -> None:
    command = build_command("py", APP, ["--theme.base", "light"])

    assert command[-2:] == ["--theme.base", "light"]
    assert command.index("--theme.base") < len(command) - 2  # the default is still there, earlier


def test_benchmark_dir_is_forwarded_to_the_script_after_a_double_dash() -> None:
    command = build_command("py", APP, ["--benchmark-dir", "/r", "--server.port", "8502"])

    assert command[-3:] == ["--", "--benchmark-dir", "/r"]
    assert command.index("--server.port") < command.index("--")


def test_theme_args_are_flag_value_pairs() -> None:
    assert len(THEME_ARGS) % 2 == 0
    assert all(flag.startswith("--theme.") for flag in THEME_ARGS[::2])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_cli.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'THEME_ARGS'`.

- [ ] **Step 3: Rewrite `cli.py`**

Replace `src/trading_agent_framework/dashboard/cli.py` with:

```python
"""CLI entry point invoked by `uv run dashboard`.

`uv run dashboard [--benchmark-dir PATH] [streamlit options...]`
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from trading_agent_framework.dashboard.benchmark_reader import BENCHMARK_DIR_FLAG, split_benchmark_dir

# Streamlit reads .streamlit/config.toml from the working directory, not the app's, so the dark
# trading theme travels as flags. User-supplied args come later and win.
THEME_ARGS = [
    "--theme.base", "dark",
    "--theme.backgroundColor", "#0b0e14",
    "--theme.secondaryBackgroundColor", "#131722",
    "--theme.textColor", "#d1d4dc",
    "--theme.primaryColor", "#22d3ee",
]


def build_command(python: str, app_file: Path, argv: Sequence[str]) -> list[str]:
    """The `streamlit run` command; --benchmark-dir goes to the app as a script argument (after `--`)."""
    rest, benchmark_dir = split_benchmark_dir(argv)
    # Default to hot-reload on source changes; user-supplied args override.
    command = [python, "-m", "streamlit", "run", str(app_file), *THEME_ARGS, "--server.runOnSave", "true", *rest]
    if benchmark_dir is not None:
        command += ["--", BENCHMARK_DIR_FLAG, benchmark_dir]
    return command


def main():
    import subprocess
    import sys

    app_file = Path(__file__).parent / "app.py"
    try:
        sys.exit(subprocess.run(build_command(sys.executable, app_file, sys.argv[1:])).returncode)
    except KeyboardInterrupt:
        print("Exit gracefully.")


if __name__ == "__main__":
    main()
```

If ruff reformats `THEME_ARGS` one-per-line, accept its formatting (`uv run ruff format` is not configured here; only fix `ruff check` findings).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_cli.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/dashboard tests/dashboard
git add src/trading_agent_framework/dashboard/cli.py tests/dashboard/test_cli.py
git commit -m "feat(dashboard): dark theme flags and --benchmark-dir forwarding in the CLI"
```

---

### Task 4: Dark CSS and shared chart template

**Files:**
- Modify: `src/trading_agent_framework/dashboard/theme.py`
- Modify: `src/trading_agent_framework/dashboard/components/charts.py`
- Modify: `src/trading_agent_framework/dashboard/_pages/side_by_side.py`
- Create: `tests/dashboard/test_dark_theme.py`

**Interfaces:**
- Produces: `theme.THEME_CSS: str`; `charts.CHART_TEMPLATE: go.layout.Template`, `charts.ZERO_LINE_COLOR`, `charts.MODEL_COLORS: list[str]`, `charts.DIVERGING_SCALE`.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_dark_theme.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from trading_agent_framework.dashboard.components import charts
from trading_agent_framework.dashboard.theme import THEME_CSS

DASHBOARD = Path(charts.__file__).resolve().parents[1]


def test_css_has_no_white_background_left() -> None:
    assert "#ffffff" not in THEME_CSS.lower()
    assert "#131722" in THEME_CSS  # metric card background


@pytest.mark.parametrize(
    "build",
    [
        lambda: charts.equity_curve_chart([]),
        lambda: charts.drawdown_chart([]),
        lambda: charts.cumulative_returns_chart({}),
        lambda: charts.trades_chart({}),
    ],
)
def test_charts_use_the_transparent_dark_template(build) -> None:
    template = build().layout.template

    assert template.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert template.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_no_chart_is_left_on_the_white_template() -> None:
    for path in [DASHBOARD / "components" / "charts.py", *(DASHBOARD / "_pages").glob("*.py")]:
        assert "plotly_white" not in path.read_text(encoding="utf-8"), path.name
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_dark_theme.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'THEME_CSS'`.

- [ ] **Step 3: Rewrite `theme.py`**

Replace `src/trading_agent_framework/dashboard/theme.py` with:

```python
"""Dark trading theme: CSS for the custom HTML components.

Streamlit's own colours come from cli.THEME_ARGS; charts use components.charts.CHART_TEMPLATE.
"""

import streamlit as st

THEME_CSS = """
<style>
.metric-card {
    background: #131722; border: 1px solid #2a2e39;
    border-radius: 8px; padding: 16px; text-align: center;
}
.metric-card .label {
    font-size: 0.8rem; color: #8a8f98;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 1.4rem; font-weight: 600; color: #d1d4dc;
}
.metric-card .value.positive { color: #22c55e; }
.metric-card .value.negative { color: #ef4444; }
.strategy-badge {
    display: inline-block; background: rgba(34, 211, 238, 0.12);
    color: #22d3ee; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 500;
}
</style>
"""


def apply_theme():
    """Inject the dark theme CSS."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
```

(The old `.stApp { background-color: #ffffff; }` rule is dropped: the page background now comes from `--theme.backgroundColor`.)

- [ ] **Step 4: Add the template and retune colours in `charts.py`**

In `src/trading_agent_framework/dashboard/components/charts.py`, add `import plotly.io as pio` after `import plotly.graph_objects as go`, and add after the imports:

```python
# Dark template shared by every figure: plotly_dark with transparent backgrounds, so charts sit on
# the page's own background (cli.THEME_ARGS). go.layout.Template copies, plotly_dark is untouched.
GRID_COLOR = "#2a2e39"
ZERO_LINE_COLOR = "#4b5563"
CHART_TEMPLATE = go.layout.Template(pio.templates["plotly_dark"])
CHART_TEMPLATE.layout.paper_bgcolor = "rgba(0,0,0,0)"
CHART_TEMPLATE.layout.plot_bgcolor = "rgba(0,0,0,0)"
CHART_TEMPLATE.layout.font.color = "#d1d4dc"
CHART_TEMPLATE.layout.xaxis.gridcolor = GRID_COLOR
CHART_TEMPLATE.layout.yaxis.gridcolor = GRID_COLOR

# Red -> dark neutral -> green, for signed values (a white midpoint glares on a dark page).
DIVERGING_SCALE = [[0.0, "#ef4444"], [0.5, "#1f2430"], [1.0, "#22c55e"]]
# One colour per series/model, readable on the dark background.
MODEL_COLORS = ["#22d3ee", "#f59e0b", "#a78bfa", "#22c55e", "#f472b6", "#60a5fa"]
```

Then, in the same file:
- replace every `template="plotly_white"` with `template=CHART_TEMPLATE` (15 places: `sed -i 's/template="plotly_white"/template=CHART_TEMPLATE/' src/trading_agent_framework/dashboard/components/charts.py`);
- in `monthly_returns_heatmap`, replace `colorscale="RdBu",` with `colorscale=DIVERGING_SCALE,`;
- in `returns_distribution`, replace `line_color="#9ca3af"` with `line_color=ZERO_LINE_COLOR`, and the comment `# Solid, saturated colors that pop on a white background` with `# Solid, saturated colors that pop on the dark background`;
- replace both `line_color="#d1d5db"` (in `monthly_returns_distribution` and `cumulative_returns_chart`) with `line_color=ZERO_LINE_COLOR`.

- [ ] **Step 5: Update `side_by_side.py`**

In `src/trading_agent_framework/dashboard/_pages/side_by_side.py`:
- add `from trading_agent_framework.dashboard.components.charts import CHART_TEMPLATE, MODEL_COLORS` to the imports;
- replace `colors = ["#0891b2", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad", "#2c3e50"]` with `colors = MODEL_COLORS`;
- replace `template="plotly_white",` with `template=CHART_TEMPLATE,`.

- [ ] **Step 6: Run the dashboard tests**

Run: `uv run pytest tests/dashboard -q`
Expected: all PASS (new theme tests and the existing smoke/chart tests).

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/dashboard tests/dashboard
git add src/trading_agent_framework/dashboard/theme.py src/trading_agent_framework/dashboard/components/charts.py src/trading_agent_framework/dashboard/_pages/side_by_side.py tests/dashboard/test_dark_theme.py
git commit -m "feat(dashboard): dark trading theme for cards and charts"
```

---

### Task 5: Benchmark chart and table builders

**Files:**
- Modify: `src/trading_agent_framework/dashboard/components/charts.py`
- Modify: `src/trading_agent_framework/dashboard/components/tables.py`
- Create: `tests/dashboard/test_benchmark_components.py`

**Interfaces:**
- Consumes: `BenchmarkModel`, `ScenarioRun`, `load_benchmark_run`, `load_scenario_runs`, fixtures; `CHART_TEMPLATE`, `MODEL_COLORS`.
- Produces:
  - `charts.benchmark_category_chart(models: Sequence[BenchmarkModel], categories: Sequence[str], title: str = "Scores by category") -> go.Figure`
  - `charts.benchmark_speed_quality_chart(models: Sequence[BenchmarkModel], title: str = "Quality vs speed") -> go.Figure`
  - `charts.benchmark_scenario_heatmap(models: Sequence[BenchmarkModel], scenarios: Sequence[str], title: str = ...) -> go.Figure`
  - `tables.benchmark_summary_frame(models: Sequence[BenchmarkModel], categories: Sequence[str]) -> pd.DataFrame`
  - `tables.render_benchmark_summary_table(models: Sequence[BenchmarkModel], categories: Sequence[str]) -> None`
  - `tables.scenario_runs_frame(runs: Sequence[ScenarioRun]) -> pd.DataFrame`

Colour rule: callers pass the same model list (the ran models, summary order) to every chart, so model *i* gets `MODEL_COLORS[i % len(MODEL_COLORS)]` everywhere.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_benchmark_components.py`:

```python
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.dashboard.benchmark_fixtures import LATEST, OLDER, SCENARIOS, build_results_tree
from trading_agent_framework.dashboard.benchmark_reader import load_benchmark_run, load_scenario_runs
from trading_agent_framework.dashboard.components.charts import (
    CHART_TEMPLATE,
    benchmark_category_chart,
    benchmark_scenario_heatmap,
    benchmark_speed_quality_chart,
)
from trading_agent_framework.dashboard.components.tables import benchmark_summary_frame, scenario_runs_frame
from trading_agent_framework.dashboard.models import BenchmarkRun, BenchmarkRunRef


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return build_results_tree(tmp_path / "results")


def _run(results: Path, run_id: str = LATEST) -> BenchmarkRun:
    return load_benchmark_run(BenchmarkRunRef.from_path(results / run_id))


def test_category_chart_has_one_horizontal_bar_trace_per_model(results: Path) -> None:
    run = _run(results)
    fig = benchmark_category_chart(run.models, run.categories)

    assert [trace.name for trace in fig.data] == ["GLM-4.7-Flash", "Qwen3.6-27B-AWQ"]
    assert all(trace.orientation == "h" for trace in fig.data)
    assert list(fig.data[1].y) == ["Reasoning", "Tools"]
    assert list(fig.data[1].x) == [75.0, 100.0]
    assert fig.layout.template.layout.paper_bgcolor == CHART_TEMPLATE.layout.paper_bgcolor


def test_speed_quality_chart_plots_run_time_against_overall_score(results: Path) -> None:
    fig = benchmark_speed_quality_chart(_run(results).models)

    (points,) = fig.data
    assert list(points.x) == [4.9, 29.7]
    assert list(points.y) == [50.0, 83.0]
    assert list(points.text) == ["GLM-4.7-Flash", "Qwen3.6-27B-AWQ"]
    assert fig.layout.xaxis.type == "log"


def test_speed_quality_chart_skips_models_without_timings(results: Path) -> None:
    fig = benchmark_speed_quality_chart(_run(results, OLDER).models)

    assert list(fig.data[0].text) == ["GLM-4.7-Flash"]


def test_heatmap_rows_are_scenarios_in_meta_order_and_cells_show_passed_over_runs(results: Path) -> None:
    run = _run(results)
    (heatmap,) = benchmark_scenario_heatmap(run.models, run.scenarios).data

    assert list(heatmap.y) == SCENARIOS
    assert list(heatmap.x) == ["GLM-4.7-Flash", "Qwen3.6-27B-AWQ"]
    assert heatmap.text[0] == ("0/2", "1/2")  # reasoning.rsi_signal
    assert heatmap.z[0] == (80.0, 96.0)


def test_heatmap_shows_a_dash_for_a_scenario_a_model_has_no_score_for(results: Path) -> None:
    run = _run(results)
    glm = dataclasses.replace(run.models[0], scenarios={})
    (heatmap,) = benchmark_scenario_heatmap([glm], run.scenarios).data

    assert heatmap.text[0] == ("—",)
    assert heatmap.z[0] == (None,)


def test_summary_frame_is_sorted_by_overall_with_the_winner_marked(results: Path) -> None:
    run = _run(results)
    frame = benchmark_summary_frame(run.models, run.categories)

    assert list(frame["Model"]) == ["🏆 Qwen3.6-27B-AWQ", "GLM-4.7-Flash"]
    assert list(frame.columns[:4]) == ["Model", "Overall", "Reasoning", "Tools"]
    assert frame.loc[0, "Overall"] == pytest.approx(83.0)
    assert frame.loc[0, "Runs passed"] == "5/6"
    assert frame.loc[0, "Timeouts / errors"] == "1/0"


def test_summary_frame_puts_a_model_that_did_not_run_last_with_empty_scores(results: Path) -> None:
    run = _run(results, OLDER)
    frame = benchmark_summary_frame(run.models, run.categories)

    assert list(frame["Model"]) == ["🏆 GLM-4.7-Flash", "Gpt-OSS-20b (did not run)"]
    assert frame["Overall"].isna().iloc[1]


def test_scenario_runs_frame_has_one_row_per_repeat(results: Path) -> None:
    runs = load_scenario_runs(BenchmarkRunRef.from_path(results / LATEST), "qwen3627b", "reasoning.rsi_signal").runs
    frame = scenario_runs_frame(runs)

    assert list(frame["Repeat"]) == [1, 2]
    assert list(frame["Passed"]) == ["✅", "❌"]
    assert list(frame["Status"]) == ["ok", "timeout"]
    assert frame.loc[0, "Partial %"] == pytest.approx(100.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_benchmark_components.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'benchmark_category_chart'`.

- [ ] **Step 3: Add the chart builders**

In `src/trading_agent_framework/dashboard/components/charts.py`, add to the imports:

```python
from collections.abc import Sequence

from trading_agent_framework.dashboard.models import BenchmarkModel
```

and append at the end of the file:

```python
# --- vLLM benchmark (Models tab) ---------------------------------------------------------------

# Mean partial score, 0-100: red (bad) -> amber -> green (good).
SCORE_SCALE = [[0.0, "#ef4444"], [0.5, "#f59e0b"], [1.0, "#22c55e"]]


def _pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100, 4)


def benchmark_category_chart(
    models: Sequence[BenchmarkModel], categories: Sequence[str], title: str = "Scores by category"
) -> go.Figure:
    """Grouped horizontal bars: one group per category, one bar (trace) per model."""
    labels = [category.capitalize() for category in categories]
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(
            go.Bar(
                y=labels,
                x=[_pct(model.categories.get(category)) for category in categories],
                name=model.display_name,
                orientation="h",
                marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                hovertemplate="%{y}: %{x:.0f}%<extra>" + model.display_name + "</extra>",
            )
        )
    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        barmode="group",
        xaxis=dict(title="Score (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.2),
        height=380,
    )
    return fig


def benchmark_speed_quality_chart(models: Sequence[BenchmarkModel], title: str = "Quality vs speed") -> go.Figure:
    """One labelled point per model: median run time (log x) against overall score."""
    points = [(i, m) for i, m in enumerate(models) if m.median_run_s is not None and m.overall is not None]
    fig = go.Figure(
        go.Scatter(
            x=[m.median_run_s for _, m in points],
            y=[_pct(m.overall) for _, m in points],
            mode="markers+text",
            text=[m.display_name for _, m in points],
            textposition="top center",
            marker=dict(size=14, color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i, _ in points]),
            hovertemplate="%{text}<br>median run %{x:.1f} s<br>overall %{y:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        xaxis=dict(title="Median run time (s, log scale)", type="log"),
        yaxis=dict(title="Overall score (%)", range=[0, 105]),
        showlegend=False,
        height=380,
    )
    return fig


def benchmark_scenario_heatmap(
    models: Sequence[BenchmarkModel],
    scenarios: Sequence[str],
    title: str = "Per-scenario results (passed/runs; colour = mean partial score)",
) -> go.Figure:
    """Rows = scenarios (meta order, so grouped by category), columns = models."""
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for scenario_id in scenarios:
        scores = [model.scenarios.get(scenario_id) for model in models]
        z.append([None if s is None else _pct(s.mean_partial) for s in scores])
        text.append(["—" if s is None else f"{s.passed}/{s.runs}" for s in scores])
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[model.display_name for model in models],
            y=list(scenarios),
            text=text,
            texttemplate="%{text}",
            colorscale=SCORE_SCALE,
            zmin=0,
            zmax=100,
            xgap=2,
            ygap=2,
            colorbar=dict(title="Mean partial %"),
            hovertemplate="%{y}<br>%{x}<br>%{text} passed, mean partial %{z:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        yaxis=dict(autorange="reversed"),
        height=max(300, 28 * len(scenarios) + 120),
    )
    return fig
```

- [ ] **Step 4: Add the table builders**

In `src/trading_agent_framework/dashboard/components/tables.py`, add to the imports:

```python
from collections.abc import Sequence

from trading_agent_framework.dashboard.models import BenchmarkModel, ScenarioRun
```

and append:

```python
# --- vLLM benchmark (Models tab) ---------------------------------------------------------------


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100


def _ratio(a: int | None, b: int | None) -> str:
    return "—" if a is None or b is None else f"{a}/{b}"


def benchmark_summary_frame(models: Sequence[BenchmarkModel], categories: Sequence[str]) -> pd.DataFrame:
    """One row per model, best overall first (marked 🏆); models that did not run go last."""
    ranked = sorted(models, key=lambda m: (not m.ran, -(m.overall or 0.0)))
    rows = []
    for i, model in enumerate(ranked):
        if not model.ran:
            name = f"{model.display_name} (did not run)"
        else:
            name = f"🏆 {model.display_name}" if i == 0 else model.display_name
        row: dict[str, object] = {"Model": name, "Overall": _pct(model.overall)}
        for category in categories:
            row[category.capitalize()] = _pct(model.categories.get(category))
        row |= {
            "Runs passed": _ratio(model.runs_passed, model.runs_total),
            "Text tool calls": model.text_tool_calls,
            "Avg tool calls": model.avg_tool_calls,
            "Median run (s)": model.median_run_s,
            "Tokens/s": model.median_tokens_per_s,
            "Timeouts / errors": _ratio(model.timeouts, model.errors),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def render_benchmark_summary_table(models: Sequence[BenchmarkModel], categories: Sequence[str]) -> None:
    frame = benchmark_summary_frame(models, categories)
    score_columns = ["Overall", *(category.capitalize() for category in categories)]
    column_config = {
        column: st.column_config.ProgressColumn(column, min_value=0, max_value=100, format="%.0f%%")
        for column in score_columns
    }
    column_config |= {
        "Avg tool calls": st.column_config.NumberColumn(format="%.2f"),
        "Median run (s)": st.column_config.NumberColumn(format="%.1f"),
        "Tokens/s": st.column_config.NumberColumn(format="%.1f"),
    }
    st.dataframe(frame, width="stretch", hide_index=True, column_config=column_config)


def scenario_runs_frame(runs: Sequence[ScenarioRun]) -> pd.DataFrame:
    """One row per repeat of a scenario for one model."""
    return pd.DataFrame(
        [
            {
                "Repeat": run.repeat,
                "Status": run.status,
                "Passed": "✅" if run.passed else "❌",
                "Partial %": run.partial * 100,
                "Total (s)": run.total_s,
                "Tool calls": run.tool_calls,
                "Model calls": run.model_calls,
                "Tokens/s": run.tokens_per_s,
            }
            for run in runs
        ]
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_benchmark_components.py -q`
Expected: all PASS. If `heatmap.text[0]` comes back as a list rather than a tuple, compare with `tuple(heatmap.text[0])` (plotly stores 2-D arrays as tuples of tuples in 5.24; adjust the assertion, not the builder).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/dashboard tests/dashboard
git add src/trading_agent_framework/dashboard/components/charts.py src/trading_agent_framework/dashboard/components/tables.py tests/dashboard/test_benchmark_components.py
git commit -m "feat(dashboard): benchmark category/speed/heatmap charts and summary tables"
```

---

### Task 6: Models page

**Files:**
- Create: `src/trading_agent_framework/dashboard/_pages/models.py`
- Modify: `src/trading_agent_framework/dashboard/components/metric_cards.py`
- Create: `tests/dashboard/test_models_page.py`
- Modify: `tests/dashboard/test_package_imports.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 2, 5; `render_metric_card`; `ConfigurationError` from `trading_agent_framework.utils.errors`.
- Produces: `_pages.models.page_models() -> None`; `render_metric_card(label, value, fmt=".2f", is_pct=True, tone: str | None = None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_models_page.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.dashboard.benchmark_fixtures import LATEST, OLDER, build_results_tree, write_json


def _models_script() -> None:
    # Serialized by AppTest.from_function: must be self-contained.
    from trading_agent_framework.dashboard._pages.models import page_models

    page_models()


def _app(results: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    # A real launch gets --benchmark-dir from cli.py after `--`, i.e. in sys.argv.
    monkeypatch.setattr(sys, "argv", ["app.py", "--benchmark-dir", str(results)])
    at = AppTest.from_function(_models_script, default_timeout=30)
    at.run()
    return at


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return build_results_tree(tmp_path / "results")


def test_models_page_renders_the_latest_run(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(results, monkeypatch)

    assert not at.exception
    assert [title.value for title in at.title] == ["Model benchmark"]
    assert len(at.get("plotly_chart")) == 3  # categories, quality vs speed, heatmap
    assert len(at.dataframe) == 2  # summary + drill-down repeats


def test_run_picker_lists_complete_runs_newest_first(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    picker = _app(results, monkeypatch).sidebar.selectbox[0]

    assert picker.value == LATEST
    assert len(picker.options) == 2


def test_drill_down_defaults_to_the_winners_worst_scenario(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(results, monkeypatch)

    model, scenario = at.main.selectbox[0], at.main.selectbox[1]  # at.selectbox would include the sidebar picker
    assert model.value == "qwen3627b"
    assert scenario.value == "reasoning.rsi_signal"
    assert len(at.expander) == 2  # one per repeat
    assert any("1 malformed line" in caption.value for caption in at.caption)


def test_switching_to_an_older_run_warns_about_the_model_that_did_not_run(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(results, monkeypatch)
    at.sidebar.selectbox[0].set_value(OLDER).run()

    assert not at.exception
    assert any("Gpt-OSS-20b did not run" in warning.value for warning in at.warning)
    assert at.main.selectbox[0].value == "glm"


def test_a_run_where_no_model_ran_shows_a_warning_and_the_table(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    summary_path = results / LATEST / "summary.json"
    summary = json.loads(summary_path.read_text())
    write_json(summary_path, [{**entry, "ran": False, "error": "boom"} for entry in summary])

    at = _app(results, monkeypatch)

    assert not at.exception
    assert any("No model ran" in warning.value for warning in at.warning)
    assert len(at.dataframe) == 1


def test_an_unreadable_run_shows_its_error(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (results / LATEST / "summary.json").write_text("{not json", encoding="utf-8")

    at = _app(results, monkeypatch)

    assert not at.exception
    assert any("cannot be read" in error.value for error in at.error)


def test_no_results_shows_where_it_looked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path / "missing", monkeypatch)

    assert not at.exception
    assert any("--benchmark-dir" in info.value for info in at.info)


def test_the_models_sidebar_has_no_backtesting_buttons(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [button.label for button in _app(results, monkeypatch).sidebar.button]

    assert labels == ["🔄 Refresh Data"]
```

Add to `MODULES` in `tests/dashboard/test_package_imports.py`:

```python
    "trading_agent_framework.dashboard._pages.models",
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_models_page.py -q`
Expected: FAIL — every test reports an exception from the script (`ModuleNotFoundError: ...dashboard._pages.models`).

- [ ] **Step 3: Add `tone` to `render_metric_card`**

In `src/trading_agent_framework/dashboard/components/metric_cards.py`, change the signature and colour choice of `render_metric_card`:

```python
def render_metric_card(
    label: str, value: float | str, fmt: str = ".2f", is_pct: bool = True, tone: str | None = None
) -> None:
    """Render a single KPI metric card. Use inside st.columns().

    `tone` ("positive"/"negative") overrides the colour, which otherwise follows the sign of a numeric value.
    """
```

and, just before the `st.markdown(` call, add:

```python
    if tone is not None:
        css_class = tone
```

- [ ] **Step 4: Write the page**

Create `src/trading_agent_framework/dashboard/_pages/models.py`:

```python
"""Models tab: compare the vLLM models of a benchmark-vllm-models run (read-only)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import streamlit as st

from trading_agent_framework.dashboard import benchmark_reader as br
from trading_agent_framework.dashboard.components.charts import (
    benchmark_category_chart,
    benchmark_scenario_heatmap,
    benchmark_speed_quality_chart,
)
from trading_agent_framework.dashboard.components.metric_cards import render_metric_card
from trading_agent_framework.dashboard.components.tables import render_benchmark_summary_table, scenario_runs_frame
from trading_agent_framework.dashboard.models import BenchmarkModel, BenchmarkRun, BenchmarkRunRef, ScenarioRuns
from trading_agent_framework.utils.errors import ConfigurationError


@st.cache_data(show_spinner=False)
def _load_run(path: str) -> BenchmarkRun:
    return br.load_benchmark_run(BenchmarkRunRef.from_path(Path(path)))


@st.cache_data(show_spinner=False)
def _load_scenario_runs(path: str, model_key: str, scenario_id: str) -> ScenarioRuns:
    return br.load_scenario_runs(BenchmarkRunRef.from_path(Path(path)), model_key, scenario_id)


def page_models() -> None:
    """Benchmark run picker in the sidebar; scores, charts, heatmap and drill-down in the page."""
    if st.sidebar.button("🔄 Refresh Data", width="stretch", key="models_refresh"):
        _load_run.clear()
        _load_scenario_runs.clear()
        st.rerun()

    st.title("Model benchmark")
    try:
        results_dir = br.resolve_results_dir(sys.argv[1:])
    except ConfigurationError as exc:
        st.error(f"Cannot locate the benchmark results: {exc}")
        return

    refs = br.scan_benchmark_runs(results_dir)
    if not refs:
        st.info(
            f"No complete benchmark run found in `{results_dir}`. Run the benchmark, or point the "
            "dashboard at its results with `uv run dashboard --benchmark-dir PATH`."
        )
        return

    loaded: dict[str, BenchmarkRun | br.BenchmarkReadError] = {}
    for ref in refs:
        try:
            loaded[ref.run_id] = _load_run(str(ref.path))
        except br.BenchmarkReadError as exc:
            loaded[ref.run_id] = exc

    run_id = st.sidebar.selectbox(
        "Benchmark run",
        [ref.run_id for ref in refs],
        format_func=lambda rid: _run_label(rid, loaded[rid]),
        key="benchmark_run",
    )
    run = loaded[run_id]
    if isinstance(run, br.BenchmarkReadError):
        st.error(f"This benchmark run cannot be read: {run}")
        return

    _render_header(run)
    ran = [model for model in run.models if model.ran]
    if not ran:
        st.warning("No model ran in this benchmark run.")
        render_benchmark_summary_table(run.models, run.categories)
        return

    _render_headline_cards(ran)
    st.subheader("Summary")
    render_benchmark_summary_table(run.models, run.categories)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(benchmark_category_chart(ran, run.categories), width="stretch")
    with right:
        st.plotly_chart(benchmark_speed_quality_chart(ran), width="stretch")
    st.subheader("Per-scenario results")
    st.plotly_chart(benchmark_scenario_heatmap(ran, run.scenarios), width="stretch")
    _render_drill_down(run, ran)


def _run_label(run_id: str, run: BenchmarkRun | br.BenchmarkReadError) -> str:
    stamp = BenchmarkRunRef.from_path(Path(run_id)).started_at.strftime("%Y-%m-%d %H:%M")
    if isinstance(run, br.BenchmarkReadError):
        return f"⚠ {stamp} · unreadable"
    return f"{stamp} · {len(run.models)} models · {run.repeats} repeats"


def _fmt_duration(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def _render_header(run: BenchmarkRun) -> None:
    parts = [f"Started {run.started_at:%Y-%m-%d %H:%M}"]
    if run.finished_at is not None:
        parts.append(f"finished {run.finished_at:%Y-%m-%d %H:%M} ({_fmt_duration(run.finished_at - run.started_at)})")
    parts += [f"{run.repeats} repeats", f"timeout {run.timeout_s:.0f} s"]
    versions = sorted({model.vllm_version for model in run.models if model.vllm_version})
    if versions:
        parts.append("vLLM " + ", ".join(versions))
    st.caption(" · ".join(parts))
    for model in run.models:
        if not model.ran:
            st.warning(f"{model.display_name} did not run: {model.error or 'no error recorded'}")


def _best(models: Sequence[BenchmarkModel]) -> BenchmarkModel:
    return max(models, key=lambda m: m.overall or 0.0)


def _render_headline_cards(ran: Sequence[BenchmarkModel]) -> None:
    best = _best(ran)
    timed = [m for m in ran if m.median_tokens_per_s is not None]
    quick = [m for m in ran if m.median_run_s is not None]
    text_calls = sum(m.text_tool_calls or 0 for m in ran)
    cols = st.columns(4)
    with cols[0]:
        render_metric_card("🏆 Best overall", f"{best.display_name} · {(best.overall or 0.0):.0%}")
    with cols[1]:
        fastest = max(timed, key=lambda m: m.median_tokens_per_s) if timed else None
        render_metric_card(
            "⚡ Fastest (tokens/s)", f"{fastest.display_name} · {fastest.median_tokens_per_s:.0f}" if fastest else "—"
        )
    with cols[2]:
        quickest = min(quick, key=lambda m: m.median_run_s) if quick else None
        render_metric_card(
            "⏱ Lowest median run", f"{quickest.display_name} · {quickest.median_run_s:.1f} s" if quickest else "—"
        )
    with cols[3]:
        render_metric_card("🛠 Text tool calls", str(text_calls), tone="negative" if text_calls else "positive")


def _worst_scenario(model: BenchmarkModel, scenarios: Sequence[str]) -> str:
    """The model's lowest mean-partial scenario (first in meta order on ties)."""
    scored = [s for s in scenarios if s in model.scenarios]
    return min(scored, key=lambda s: model.scenarios[s].mean_partial) if scored else scenarios[0]


def _render_drill_down(run: BenchmarkRun, ran: Sequence[BenchmarkModel]) -> None:
    if not run.scenarios:
        return
    st.subheader("Drill-down")
    names = {model.key: model.display_name for model in ran}
    keys = list(names)
    best = _best(ran)
    left, right = st.columns(2)
    with left:
        # Keys are scoped to the run, so switching runs resets both pickers.
        model_key = st.selectbox(
            "Model", keys, index=keys.index(best.key), format_func=names.__getitem__, key=f"drill_model_{run.ref.run_id}"
        )
    model = next(m for m in ran if m.key == model_key)
    scenarios = list(run.scenarios)
    with right:
        scenario_id = st.selectbox(
            "Scenario",
            scenarios,
            index=scenarios.index(_worst_scenario(model, scenarios)),
            key=f"drill_scenario_{run.ref.run_id}",
        )

    try:
        result = _load_scenario_runs(str(run.ref.path), model_key, scenario_id)
    except br.BenchmarkReadError as exc:
        st.error(f"Cannot load the runs of {names[model_key]}: {exc}")
        return

    if not result.runs:
        st.info("No runs recorded for this model and scenario.")
    else:
        st.dataframe(scenario_runs_frame(result.runs), width="stretch", hide_index=True)
        for scenario_run in result.runs:
            icon = "✅" if scenario_run.passed else "❌"
            label = f"{icon} Repeat {scenario_run.repeat} · {scenario_run.status} · {scenario_run.partial:.0%}"
            with st.expander(label, expanded=not scenario_run.passed):
                if scenario_run.error:
                    st.error(scenario_run.error)
                if not scenario_run.checks:
                    st.caption("No checks recorded.")
                for check in scenario_run.checks:
                    st.markdown(f"{'✅' if check.passed else '❌'} **{check.type}**: `{check.reason}`")
    if result.skipped_lines:
        st.caption(f"{result.skipped_lines} malformed line(s) skipped in {model_key}.jsonl")
```

Note on `_run_label`: `BenchmarkRunRef.from_path(Path(run_id))` only parses the folder name, it reads nothing.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_models_page.py tests/dashboard/test_package_imports.py -q`
Expected: all PASS. If `test_drill_down_...` sees the caption as `"1 malformed line(s) skipped in qwen3627b.jsonl"`, it passes (substring match).

- [ ] **Step 6: Run the whole dashboard suite and lint**

Run: `uv run pytest tests/dashboard -q && uv run ruff check src/trading_agent_framework/dashboard tests/dashboard`
Expected: all PASS, no ruff findings.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/dashboard/_pages/models.py src/trading_agent_framework/dashboard/components/metric_cards.py tests/dashboard/test_models_page.py tests/dashboard/test_package_imports.py
git commit -m "feat(dashboard): Models page comparing vLLM benchmark runs with a per-scenario drill-down"
```

---

### Task 7: Top navigation, logo, Backtesting page

**Files:**
- Create: `src/trading_agent_framework/dashboard/_pages/backtesting.py`
- Modify: `src/trading_agent_framework/dashboard/app.py`
- Create: `src/trading_agent_framework/dashboard/assets/logo.svg`
- Create: `src/trading_agent_framework/dashboard/assets/logo-icon.svg`
- Modify: `tests/dashboard/test_app_smoke.py`
- Modify: `tests/dashboard/test_package_imports.py`

**Interfaces:**
- Consumes: `page_models` (Task 6), `apply_theme` (Task 4).
- Produces: `_pages.backtesting.page_backtesting() -> None`; `app.NAV_PAGES: tuple[tuple[str, str, Callable[[], None]], ...]`; `app.ASSETS: Path`; `app.main() -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_app_smoke.py`:

```python
# --- Top navigation --------------------------------------------------------------------------------


def test_the_tabs_are_backtesting_then_models() -> None:
    from trading_agent_framework.dashboard import app

    assert [(title, url_path) for title, url_path, _ in app.NAV_PAGES] == [
        ("Backtesting", "backtesting"),
        ("Models", "models"),
    ]


def test_the_logo_files_exist_and_carry_the_name() -> None:
    from trading_agent_framework.dashboard import app

    assert "Yann's Trading Bots" in (app.ASSETS / "logo.svg").read_text(encoding="utf-8")
    assert (app.ASSETS / "logo-icon.svg").is_file()


def test_backtesting_is_the_default_tab_with_its_sidebar(run_dir: Path) -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()

    assert not at.exception
    labels = [button.label for button in at.sidebar.button]
    assert "📋 Scorecard" in labels
    assert "🔄 Refresh Data" in labels
```

Add to `MODULES` in `tests/dashboard/test_package_imports.py`:

```python
    "trading_agent_framework.dashboard._pages.backtesting",
    "trading_agent_framework.dashboard.app",
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_app_smoke.py tests/dashboard/test_package_imports.py -q`
Expected: FAIL — `AttributeError: module ... app has no attribute 'NAV_PAGES'` and `ModuleNotFoundError: ..._pages.backtesting`.

- [ ] **Step 3: Create the logo files**

Create `src/trading_agent_framework/dashboard/assets/logo.svg`:

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="300" height="48" viewBox="0 0 300 48" role="img" aria-label="Yann's Trading Bots">
  <g stroke-width="2" stroke-linecap="round">
    <line x1="8" y1="12" x2="8" y2="40" stroke="#22c55e"/>
    <rect x="4" y="18" width="8" height="16" rx="1" fill="#22c55e"/>
    <line x1="22" y1="8" x2="22" y2="36" stroke="#ef4444"/>
    <rect x="18" y="14" width="8" height="14" rx="1" fill="#ef4444"/>
    <line x1="36" y1="4" x2="36" y2="32" stroke="#22d3ee"/>
    <rect x="32" y="8" width="8" height="18" rx="1" fill="#22d3ee"/>
  </g>
  <text x="52" y="31" font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif" font-size="20" font-weight="600" fill="#d1d4dc">Yann's Trading Bots</text>
</svg>
```

Create `src/trading_agent_framework/dashboard/assets/logo-icon.svg`:

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44" role="img" aria-label="Yann's Trading Bots">
  <g stroke-width="2" stroke-linecap="round">
    <line x1="8" y1="12" x2="8" y2="40" stroke="#22c55e"/>
    <rect x="4" y="18" width="8" height="16" rx="1" fill="#22c55e"/>
    <line x1="22" y1="8" x2="22" y2="36" stroke="#ef4444"/>
    <rect x="18" y="14" width="8" height="14" rx="1" fill="#ef4444"/>
    <line x1="36" y1="4" x2="36" y2="32" stroke="#22d3ee"/>
    <rect x="32" y="8" width="8" height="18" rx="1" fill="#22d3ee"/>
  </g>
</svg>
```

- [ ] **Step 4: Move today's `main()` into `_pages/backtesting.py`**

Create `src/trading_agent_framework/dashboard/_pages/backtesting.py` containing, **verbatim** from the current `app.py`: the `from trading_agent_framework.dashboard.reader import load_description, save_description` import, the whole `@st.dialog("Edit Description", width="large") def edit_description_dialog(): ...` function, and the body of `main()` renamed to `page_backtesting`:

```python
"""Backtesting tab: Scorecard / Run Detail / Side-by-Side, driven by the sidebar buttons."""

import streamlit as st

from trading_agent_framework.dashboard.reader import load_description, save_description


@st.dialog("Edit Description", width="large")
def edit_description_dialog():
    ...  # copied unchanged from app.py


def page_backtesting():
    """Backtesting runs, with sidebar navigation between Scorecard, Run Detail and Side-by-Side."""
    st.sidebar.title("📊 Strategy Dashboard")
    ...  # the rest of app.main()'s body, unchanged, down to the page_side_by_side() routing
```

The `...` above stand for the exact current code of `app.py` lines 23–53 (`edit_description_dialog`) and lines 57–130 (`main()` body); copy them without edits. Do not change session-state keys, button keys or labels.

- [ ] **Step 5: Rewrite `app.py` as the shell**

Replace `src/trading_agent_framework/dashboard/app.py` with:

```python
"""Streamlit dashboard: Backtesting and Models tabs.

Run via: uv run dashboard [--benchmark-dir PATH]
Or:      uv run streamlit run src/trading_agent_framework/dashboard/app.py [-- --benchmark-dir PATH]
"""

from pathlib import Path

import streamlit as st

from trading_agent_framework.dashboard._pages.backtesting import page_backtesting
from trading_agent_framework.dashboard._pages.models import page_models
from trading_agent_framework.dashboard.theme import apply_theme

ASSETS = Path(__file__).parent / "assets"

# (title, url path, page function); the first one is the default tab. Plain data, so tests can
# check the tabs without a Streamlit runtime (st.Page cannot be built outside one).
NAV_PAGES = (
    ("Backtesting", "backtesting", page_backtesting),
    ("Models", "models", page_models),
)


def main():
    st.set_page_config(  # must be the first Streamlit call
        page_title="Yann's Trading Bots",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo-icon.svg"), size="large")
    pages = [
        st.Page(page, title=title, url_path=url_path, default=(i == 0))
        for i, (title, url_path, page) in enumerate(NAV_PAGES)
    ]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the dashboard suite**

Run: `uv run pytest tests/dashboard -q`
Expected: all PASS — including the pre-existing `test_app_boots_and_scorecard_lists_the_run` and the Run Detail tests, unchanged (they set `current_page`/`detail_ref` and land on the default Backtesting tab).

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/dashboard tests/dashboard
git add src/trading_agent_framework/dashboard/app.py src/trading_agent_framework/dashboard/_pages/backtesting.py src/trading_agent_framework/dashboard/assets tests/dashboard/test_app_smoke.py tests/dashboard/test_package_imports.py
git commit -m "feat(dashboard): logo and Backtesting/Models top tabs"
```

---

### Task 8: Docs and end-to-end check

**Files:**
- Modify: `CLAUDE.md` (the `dashboard/` bullet under Architecture, and the Commands block)
- Modify: `README.md` (section `### 📊 Strategy Dashboard`)

- [ ] **Step 1: Update `CLAUDE.md`**

In the Commands block, change the dashboard line to:

```bash
uv run dashboard [--benchmark-dir PATH]   # Streamlit dashboard: Backtesting tab (logs/ and memory/ run history) and Models tab (vLLM benchmark results)
```

Replace the `dashboard/` Architecture bullet with:

```markdown
- `dashboard/` -- Streamlit app (`uv run dashboard`, entry `app.py`: dark theme, logo, `st.navigation(position="top")` over two tabs). **Backtesting** (`_pages/backtesting.py`, the sidebar buttons and the scorecard/detail/side-by-side router): `reader.py` reads `logs/<strategy>/<mode>/<ts>_<mode>/` run directories, `memory.sqlite` and `llm_stats.sqlite` read-only into `models.py` types. **Models** (`_pages/models.py`): `benchmark_reader.py` (Streamlit-free) reads the sibling `benchmark-vllm-models/results/<YYYYMMDD-HHMMSS>/` runs (`meta.json` + `summary.json`; `<model>.jsonl` for the per-scenario drill-down, message traces never loaded) -- default next to the project root, override with `uv run dashboard --benchmark-dir PATH` (`cli.py` forwards it after `--`; no env var). The dark theme is `cli.THEME_ARGS` (`--theme.*` flags, since `.streamlit/config.toml` would be read from the cwd) plus `components/charts.CHART_TEMPLATE` -- never `plotly_white`. `components/` (charts/metric cards/tables) render both tabs. Read-only -- never writes back into a run. `AppTest.switch_page` can't reach callable pages: page tests use `AppTest.from_function`, tabs are checked via `app.NAV_PAGES`.
```

- [ ] **Step 2: Update `README.md`**

In `### 📊 Strategy Dashboard`, replace the first two lines and the `**What it shows:**` list intro with:

```markdown
Compare backtesting runs across all strategies, and the local vLLM models benchmarked for the agents, in a dark-themed Streamlit web app with two tabs.

**Run:** `uv run dashboard` (Models tab reads `../benchmark-vllm-models/results` by default; override with `uv run dashboard --benchmark-dir PATH`)

**What it shows:**
- **Backtesting tab** — the three pages below, reached from the sidebar:
```

indent the existing Scorecard / Run Detail / Side-by-Side bullets one level under it, and add after them:

```markdown
- **Models tab** — pick a benchmark run (latest by default): best/fastest model cards, a summary table (overall and per-category scores, runs passed, text tool calls, latency, tokens/s), category bars, a quality-vs-speed scatter, a per-scenario heatmap, and a drill-down listing each repeat's checks with their pass/fail reasons.
```

- [ ] **Step 3: Full test suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all PASS, no findings.

- [ ] **Step 4: Manual check against the real results**

Run: `uv run dashboard` and open the printed URL. Check:
- dark background, logo "Yann's Trading Bots" top-left, tabs **Backtesting | Models** in the top bar;
- Backtesting: sidebar buttons as before; Scorecard lists runs; Run Detail and Side-by-Side charts are dark with readable lines;
- Models: run picker shows `2026-09-27 13:22 · 5 models · 5 repeats` first; 🏆 card is Qwen3.6-27B-AWQ · 94%; heatmap cell Qwen3.6-27B-AWQ × `memory.profit_take_rule` reads `5/5`, GLM-4.7-Flash × `reasoning.rsi_signal` reads `0/5` (matches the README table); drill-down lists checks; sidebar has only the picker and Refresh;
- `uv run dashboard --benchmark-dir /tmp/nope` → Models shows the "No complete benchmark run found in `/tmp/nope`" message.

Use the `run` skill or Chrome tools for the browser check if available; otherwise ask the user to confirm.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: dashboard tabs, Models page and --benchmark-dir"
```
