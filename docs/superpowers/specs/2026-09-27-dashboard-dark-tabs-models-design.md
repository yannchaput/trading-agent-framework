# Dashboard: dark theme, top tabs, Models page — design

Date: 2026-09-27

## 1. Context and goal

The Streamlit dashboard (`uv run dashboard`) only shows backtesting runs, on a white theme. The vLLM
model benchmark (sibling repo `../benchmark-vllm-models`) is only visible as rich-text tables pasted
into `README.md` ("Last benchmark result").

Goal:

- a dark, trading-style theme;
- a logo, with one row below it holding two tabs: **Backtesting** (today's content and layout,
  unchanged) and **Models** (a comparison of the models in a vLLM benchmark run, latest by default);
- the left sidebar keeps today's buttons on the Backtesting tab.

Decided during brainstorming: stay on Streamlit (1.64 installed); top navigation via
`st.navigation(position="top")`; the Models page shows summary views plus a per-run drill-down into
checks (no message traces); a run picker defaulting to the latest run; the backtesting sidebar
buttons appear only on the Backtesting tab; a generated SVG logo reading "Yann's Trading Bots".

Non-goals: rendering message traces from the `.jsonl` files, the `*.vllm.log` files, writing anything
into the benchmark repo, changing the backtesting pages' content.

## 2. Benchmark results format (input, read-only)

`<results>/<YYYYMMDD-HHMMSS>/` per benchmark run:

- `meta.json`: `started_at`, `finished_at`, `models[]` (`key`, `display_name`, `served_name`, ...),
  `vllm_versions{key: version}`, `repeats`, `timeout_s`, `scenarios[]` (ordered ids like
  `reasoning.rsi_signal`).
- `summary.json`: list, one entry per model: `key`, `display_name`, `ran`, `error`, `overall`,
  `mean_partial`, `categories{name: score}`, `scenarios{id: {passed, runs, mean_partial}}`,
  `runs_passed`, `runs_total`, `text_tool_calls`, `avg_tool_calls`, `median_run_s`,
  `median_tokens_per_s`, `timeouts`, `errors`.
- `<key>.jsonl`: one line per run: `scenario_id`, `category`, `repeat`, `status`, `passed`,
  `partial`, `checks[]` (`type`, `passed`, `reason`), `metrics` (`total_s`, `tool_calls`,
  `model_calls`, `tokens_per_s`, `completion_tokens`, ...), `trace` (large; ignored).

A run is **complete** when both `meta.json` and `summary.json` exist; otherwise it is still running
and is skipped.

## 3. Layout and theme

### 3.1 Entry point

`app.py` becomes a thin shell: `st.set_page_config` (first call), `apply_theme()`,
`st.logo(<logo.svg>, icon_image=<logo-icon.svg>)`, then

```python
st.navigation(
    [
        st.Page(page_backtesting, title="Backtesting", url_path="backtesting", default=True),
        st.Page(page_models, title="Models", url_path="models"),
    ],
    position="top",
).run()
```

### 3.2 Backtesting page

New `_pages/backtesting.py` holds today's `main()` body verbatim: sidebar title and buttons
(Scorecard, Run Detail, Side-by-Side, Edit Description, Refresh Data), the `edit_description_dialog`,
and the `current_page` router to `page_scorecard` / `page_detail` / `page_side_by_side`. Session state
keys (`current_page`, `selected_refs`, `detail_ref`, `compare_refs`, `run_index`, ...) are unchanged,
so switching tabs keeps the selection. Only this page renders the backtesting sidebar.

### 3.3 Theme

- Streamlit theme passed by `cli.py` as `--theme.*` flags (a `.streamlit/config.toml` would be read
  from the working directory, not the app's): `base=dark`, `backgroundColor=#0b0e14`,
  `secondaryBackgroundColor=#131722`, `textColor=#d1d4dc`, `primaryColor=#22d3ee`. User-supplied
  arguments still come last and override.
- `theme.py`: dark `.metric-card` (background `#131722`, border `#2a2e39`, label `#8a8f98`, value
  `#d1d4dc`), `.positive` `#22c55e`, `.negative` `#ef4444`, dark `.strategy-badge`.
- Charts: `components/charts.py` gets one `CHART_TEMPLATE` (plotly_dark-based, transparent
  `paper_bgcolor`/`plot_bgcolor`, grid `#2a2e39`) replacing all 15 `template="plotly_white"` uses.
  Colours chosen for white backgrounds are retuned: the zero-line grey, the `#2c3e50` series colour in
  `side_by_side.py`, and the `RdBu` returns heatmap's midpoint.

### 3.4 Logo

`dashboard/assets/logo.svg`: a cyan/green candlestick glyph with the inline text
"Yann's Trading Bots" in light text. `dashboard/assets/logo-icon.svg`: the glyph alone (collapsed
sidebar). Paths resolved from `Path(__file__).parent`, so they work from the installed wheel
(hatchling already ships non-Python files inside `src/trading_agent_framework`; no `pyproject.toml`
change).

## 4. Benchmark reader

### 4.1 Locating results

`benchmark_reader.resolve_results_dir(argv: list[str]) -> Path`, pure:

- `--benchmark-dir PATH` in `argv` wins;
- otherwise `find_project_root() / ".." / "benchmark-vllm-models" / "results"` (from
  `config/env.py`), resolved — not relative to the raw working directory.

`uv run dashboard --benchmark-dir PATH`: `cli.py` pulls `--benchmark-dir PATH` out of its arguments
and forwards it as a script argument (`streamlit run app.py <streamlit args> -- --benchmark-dir PATH`);
the Models page calls `resolve_results_dir(sys.argv[1:])`. No env var and no `.env` file.

### 4.2 Types (in `dashboard/models.py`)

Frozen dataclasses (`@dataclass(frozen=True)`); collection fields are tuples or read-only by
convention (dicts), so a frozen instance is not mutable through a list.

- `BenchmarkRunRef`: `path: Path`, `run_id: str` (folder name), `started_at: datetime` (parsed from
  the folder name).
- `ScenarioScore`: `passed: int`, `runs: int`, `mean_partial: float`.
- `BenchmarkModel`: `key`, `display_name`, `served_name: str | None`, `vllm_version: str | None`,
  `ran: bool`, `error: str | None`, `overall`, `mean_partial`, `categories: dict[str, float]`,
  `scenarios: dict[str, ScenarioScore]`, `runs_passed`, `runs_total`, `text_tool_calls`,
  `avg_tool_calls`, `median_run_s`, `median_tokens_per_s`, `timeouts`, `errors`. Numeric summary
  fields are `float | None` / `int | None` so a model that did not run stays representable.
- `BenchmarkRun`: `ref`, `started_at`, `finished_at`, `repeats`, `timeout_s`,
  `scenarios: tuple[str, ...]` (meta order), `models: tuple[BenchmarkModel, ...]`.
- `Check`: `type: str`, `passed: bool`, `reason: str`.
- `ScenarioRun`: `repeat`, `status`, `passed`, `partial`, `checks: tuple[Check, ...]`, `total_s`,
  `tool_calls`, `model_calls`, `tokens_per_s`, `completion_tokens` (metrics `None` when absent).
- `ScenarioRuns`: `runs: tuple[ScenarioRun, ...]`, `skipped_lines: int`.

### 4.3 Functions (`dashboard/benchmark_reader.py`, read-only)

- `scan_benchmark_runs(base: Path) -> list[BenchmarkRunRef]`: timestamp-named folders (`YYYYMMDD-HHMMSS`)
  containing both `meta.json` and `summary.json`, newest first; `[]` if `base` does not exist.
- `load_benchmark_run(ref) -> BenchmarkRun`: joins `meta.json` and `summary.json` on `key`. Any
  missing file, JSON error or missing required field raises `BenchmarkReadError` (new, in the module),
  never a raw `JSONDecodeError`/`KeyError`.
- `load_scenario_runs(ref, model_key, scenario_id) -> ScenarioRuns`: streams `<model_key>.jsonl`,
  keeps lines with a matching `scenario_id`, drops `trace`, sorted by `repeat`. A line that is not
  valid JSON or lacks required fields is skipped and counted in `skipped_lines`. A missing file
  raises `BenchmarkReadError`.
- Function returns are concrete `list`; tuple fields only where a frozen dataclass holds a collection.

Caching: the page wraps the loaders in `st.cache_data` (the reader module itself stays
Streamlit-free, like `reader.py`). The Models sidebar's Refresh clears that cache.

## 5. Models page (`_pages/models.py`)

### 5.1 Sidebar

- Run picker: `st.selectbox` over complete runs, newest first, labelled
  `YYYY-MM-DD HH:MM · N models · R repeats`. A run whose `load_benchmark_run` raised is labelled
  with ⚠ and selecting it shows the error.
- 🔄 Refresh: clears the benchmark cache, reruns.
- No backtesting buttons.

Empty state (no complete run): an info message naming the resolved directory and the
`--benchmark-dir` flag; no exception.

### 5.2 Content, top to bottom

1. Caption: start → finish (duration), repeats, timeout, vLLM version(s) (shown once if all equal).
   A warning per model with `ran: false`, showing its `error`. Such models appear in the table and
   are excluded from charts and the heatmap.
2. Headline cards (existing `metric_card`): 🏆 best overall (model, score), ⚡ fastest (highest
   median tokens/s), ⏱ lowest median run time, 🛠 text tool calls total (negative colour if > 0).
3. Summary table (`components/tables.py`): one row per model sorted by overall, winner marked 🏆;
   columns overall, reasoning, memory, workflow, tools (`ProgressColumn`, 0–100%), runs passed,
   text tool calls, avg tool calls, median run s, tokens/s, timeouts/errors. Category columns come
   from the data, not hardcoded.
4. Charts (`components/charts.py`):
   - category scores: grouped horizontal bars, one group per category, one bar per model;
   - quality vs speed: scatter, x = median run s (log), y = overall, one labelled point per model.
5. Per-scenario heatmap: rows = scenarios in meta order (category is the id prefix), columns =
   models, colour = `mean_partial` (red → amber → green), text = `passed/runs`.
6. Drill-down: model selectbox and scenario selectbox, defaulting to the best-overall model's lowest
   `mean_partial` scenario. A per-repeat table (repeat, status, passed ✅/❌, partial %, total s, tool
   calls, model calls, tokens/s), then one expander per repeat listing its checks (✅/❌, type,
   reason); failed repeats expanded. Caption "N malformed lines skipped" when `skipped_lines > 0`.
   A `BenchmarkReadError` here shows an error message, not an exception.

## 6. Error handling summary

| Situation | Behaviour |
|---|---|
| Results dir missing / no complete run | Info message with path and `--benchmark-dir` hint |
| In-progress run (no `summary.json`) | Not listed |
| Malformed `meta.json`/`summary.json` | Run listed with ⚠; selecting it shows the error |
| Model `ran: false` | Warning with `error`; in table, out of charts |
| Malformed `.jsonl` line | Skipped, counted, caption shown |
| Missing `<key>.jsonl` | Drill-down error message |

## 7. Testing

- Fixture helper `tests/dashboard/benchmark_fixtures.py` writes, under `tmp_path`: a complete run
  (2 models × 3 scenarios in 2 categories, one `.jsonl` with a malformed line and a `timeout` run),
  an older complete run, an in-progress folder (no `summary.json`), a non-timestamp folder. Shapes
  copied from the real `20260927-132243` run.
- `tests/dashboard/test_benchmark_reader.py`: scan order and filtering, missing dir → `[]`;
  meta/summary join, meta scenario order, `ran: false` kept, malformed JSON → `BenchmarkReadError`;
  scenario filtering, `trace` dropped, malformed line skipped and counted, timeout kept;
  `resolve_results_dir` flag vs default (default based on `find_project_root`).
- `tests/dashboard/test_app_smoke.py`: existing tests pass unchanged (Backtesting is the default
  page — regression guard for moving `main()`); new: Models page renders the fixture without
  exception (argv patched via `monkeypatch`, `at.switch_page`; if `AppTest.switch_page` cannot reach
  a callable `st.Page`, the Models tests run `page_models` through `AppTest.from_function` instead,
  and one app-level test still asserts both tabs are registered), empty-state message, picker lists 2
  runs with newest selected, drill-down shows one expander per repeat, no backtesting buttons in the
  Models sidebar.
- Chart functions: returns a `go.Figure` with the expected trace count and the dark template.
- `cli.py`: theme flags present, `--benchmark-dir` forwarded after `--`, user args still override.
- Manual: `uv run dashboard` on the real results; both tabs, logo, per-tab sidebar, heatmap values
  for Qwen3.6-27B-AWQ match the README table; `uv run pytest tests/dashboard/`, `uv run ruff check`.

## 8. Files

New: `dashboard/_pages/backtesting.py`, `dashboard/_pages/models.py`, `dashboard/benchmark_reader.py`,
`dashboard/assets/logo.svg`, `dashboard/assets/logo-icon.svg`, `tests/dashboard/benchmark_fixtures.py`,
`tests/dashboard/test_benchmark_reader.py`.

Changed: `dashboard/app.py`, `dashboard/cli.py`, `dashboard/theme.py`, `dashboard/models.py`,
`dashboard/components/charts.py`, `dashboard/components/tables.py`, `dashboard/_pages/side_by_side.py`,
`tests/dashboard/test_app_smoke.py`, `CLAUDE.md` (dashboard entry:
tabs, benchmark reader, `--benchmark-dir`).
