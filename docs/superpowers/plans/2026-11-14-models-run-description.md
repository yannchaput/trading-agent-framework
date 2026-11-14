# Models Tab Run Description Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An editable "Description" text area in the Models tab sidebar, under the run picker, saved to `<results>/<run_id>/description.json`.

**Architecture:** Reuse `reader.load_description` / `reader.save_description` (widen their `ref` type to accept a `BenchmarkRunRef`). `_pages/models.py` renders one `st.sidebar.text_area` per run key and saves through an `on_change` callback.

**Tech Stack:** Python 3.14, Streamlit (`AppTest.from_function` for page tests), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-14-models-run-description-design.md`

## Global Constraints

- File format is exactly `{"description": "<text>"}` (what `save_description` already writes).
- Widget: label `"Description"`, `height=150`, `placeholder="Notes about this benchmark run…"`, key `f"benchmark_description_{run_id}"`.
- Saves only on commit (`on_change`), no Save button; empty text saves `""`.
- `benchmark_reader.py` itself stays write-free.
- Start from a clean `_pages/models.py`: the working tree holds a throwaway spike. Run `git checkout src/trading_agent_framework/dashboard/_pages/models.py` first, on a new branch `feat/models-run-description`.
- Gate: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest --cov` (>= 80%), `uv run interrogate -v`. Use `uv sync --extra dashboard` or the page tests are skipped.

## Review Focus

1. Results folder not writable (OSError on save): the user expects a visible message, not a traceback. `_save_description` catches `OSError` and calls `st.toast`. Tested in Task 2.
2. Unreadable run (`BenchmarkReadError`): the description box still shows, rendered before the error return. Tested in Task 2.
3. Switching runs: the second run shows its own text, not the first run's. Tested in Task 2.
4. Malformed existing `description.json`: `load_description` already returns `None`, so the box is empty and the next save overwrites the file. Covered by existing reader tests, no new test needed.
5. Non-ASCII text (é, emoji) round-trips: `json.dump` escapes it and `json.load` restores it. Asserted in the Task 1 round-trip test.

---

### Task 1: Reader accepts a benchmark run

**Files:**
- Modify: `src/trading_agent_framework/dashboard/reader.py` (`load_description`, `save_description` signatures)
- Test: `tests/dashboard/test_reader.py`

**Interfaces:**
- Produces: `load_description(ref: RunRef | BenchmarkRunRef) -> str | None`, `save_description(ref: RunRef | BenchmarkRunRef, description: str) -> None`. Behaviour unchanged.

- [ ] **Step 1: Write the failing test** `test_description_round_trips_on_a_benchmark_run(tmp_path)`: `ref = BenchmarkRunRef.from_path(tmp_path / "20261111-232707")` (directory created by `save_description`); `save_description(ref, "Qwen 3.8 sweep — été ✓")`; assert `load_description(ref) == "Qwen 3.8 sweep — été ✓"` and `json.loads((ref.path / "description.json").read_text()) == {"description": "Qwen 3.8 sweep — été ✓"}`.
- [ ] **Step 2: Run** `uv run pytest tests/dashboard/test_reader.py -k benchmark_run -v`. Expect a pyright-level mismatch only; at runtime it may already pass, since both refs have `.path`. Confirm the type failure with `uv run pyright tests/dashboard/test_reader.py` (argument type error).
- [ ] **Step 3: Widen** both signatures to `RunRef | BenchmarkRunRef`, importing `BenchmarkRunRef` from `dashboard.models`. Use `os.path.join(ref.path, ...)` as now, which works for `Path` too.
- [ ] **Step 4: Run** the test and pyright on both files: PASS, 0 errors.
- [ ] **Step 5: Commit** `feat(dashboard): description helpers accept a benchmark run`.

### Task 2: Description box on the Models page

**Files:**
- Modify: `src/trading_agent_framework/dashboard/_pages/models.py` (in `page_models`, right after the `st.sidebar.selectbox`, before `run = loaded[run_id]`; new `_save_description`)
- Modify: `src/trading_agent_framework/dashboard/benchmark_reader.py` (module docstring: "Nothing here ever writes…" stays true; add that the Models page writes `description.json` into a run folder through `reader.save_description`)
- Modify: `src/trading_agent_framework/dashboard/CLAUDE.md` (Models sentence: mention the sidebar Description box and `description.json`; change "Read-only -- never writes back into a run" to: the only writes into a run are `description.json` (Backtesting and Models) and Backtesting's Decision/Regime tags in `settings.json` (`reader.save_decision` / `save_regime`))
- Test: `tests/dashboard/test_models_page.py`

**Interfaces:**
- Consumes: Task 1's `load_description` / `save_description`.
- Produces: `_save_description(ref: BenchmarkRunRef, key: str) -> None` (private, page-only).

- [ ] **Step 1: Write the failing tests** in `test_models_page.py`, using the `results` fixture and `_app`, with `LATEST` / `OLDER` from `benchmark_fixtures`:
  - `test_description_box_sits_under_the_run_picker`: `at.sidebar.text_area[0].label == "Description"` and its value is `""`.
  - `test_typing_a_description_saves_it_in_the_run_folder`: `at.sidebar.text_area[0].input("vllm 0.33 sweep").run()`; `json.loads((results / LATEST / "description.json").read_text()) == {"description": "vllm 0.33 sweep"}`.
  - `test_an_existing_description_prefills_the_box`: write `{"description": "old note"}` to `results / LATEST / "description.json"` before `_app`; the value is `"old note"`.
  - `test_switching_runs_shows_that_runs_description`: write a note only in `OLDER`; `at.sidebar.selectbox[0].select(OLDER).run()`; `at.sidebar.text_area[0].value == "older note"`; then select `LATEST` again and the value is `""`.
  - `test_unreadable_run_still_shows_the_description_box`: reuse the setup of the existing `test_an_unreadable_run_shows_its_error`; assert `at.error` is non-empty and `len(at.sidebar.text_area) == 1`.
  - `test_a_failed_save_shows_a_toast`: `monkeypatch.setattr("trading_agent_framework.dashboard._pages.models.save_description", raising OSError("read-only"))`, input text and run; `not at.exception` and `"read-only"` appears in `at.toast[0].value`.
- [ ] **Step 2: Run** `uv run pytest tests/dashboard/test_models_page.py -v`. Expect the 6 new tests to FAIL (no text_area) and the old ones to pass.
- [ ] **Step 3: Implement.** Add `st.sidebar.text_area(...)` with the Global Constraints values, `value=load_description(ref) or ""`, `on_change=_save_description`, `args=(ref, key)`, where `ref` is the `BenchmarkRunRef` in `refs` whose `run_id == run_id`. `_save_description` calls `save_description(ref, st.session_state[key])` and on `OSError` calls `st.toast(f"Could not save the description: {exc}")`. Give it a one-line docstring. Then make the two docs edits listed under Files.
- [ ] **Step 4: Run** the full gate (Global Constraints). All pass; report the coverage and interrogate numbers.
- [ ] **Step 5:** Run `/simplify` on the diff, then run the gate again.
- [ ] **Step 6:** Run `/ponytail:ponytail-review` on the staged changes, fix every "Must fix", run the gate again, then commit `feat(dashboard): editable run description in the Models tab` (with the spec and this plan).
