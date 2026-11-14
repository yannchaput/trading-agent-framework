# Models tab: per-run description

Small change. Approved in conversation on 2026-11-14 (look checked on a spike in the live dashboard).

## Intent

Let the user note what each vLLM benchmark run was for, the way Backtesting runs already carry a
description. The note lives with the run, in its results folder.

## What gets built

- In the Models tab sidebar, directly under the "Benchmark run" picker: `st.sidebar.text_area("Description")`,
  height 150, placeholder "Notes about this benchmark run…".
- Widget key `benchmark_description_<run_id>`, so switching runs shows that run's own text (`value=` is read
  only when a key first appears).
- Saves on commit (blur or Ctrl+Enter) through an `on_change` callback with `args=(ref, key)`, which writes
  `{"description": "<text>"}` to `<results>/<run_id>/description.json`. No Save button. Empty text is saved
  as `""`.
- Shown for every listed run, including one whose results are unreadable (notes don't depend on them).
- Reuses `reader.load_description` / `reader.save_description` unchanged in behaviour; their `ref` parameter is
  widened to `RunRef | BenchmarkRunRef` (both only need `.path`), so no cast.

## Constraints

- `benchmark_reader.py` stays read-only. The one write is `description.json`, done by the page through
  `reader.save_description`. Update the "Nothing here ever writes into the results directory" docstring and
  `dashboard/CLAUDE.md` (Models paragraph, and its "Read-only -- never writes back into a run" line, which
  Backtesting's description already contradicts) to name that exception.
- `results/` is gitignored in `benchmark-vllm-models`, and that repo never reads `description.json`.

## Approaches

1. **Always-visible sidebar text area that saves on commit** (picked): what the user asked for, one widget,
   reuses the existing helpers.
2. Backtesting-style button + dialog: more clicks, and the user asked for a field in the left pane.
3. `st.form` with a Save button: an extra click for a single field; commit-on-blur already avoids
   per-keystroke writes.

## Tests

- `tests/dashboard/test_reader.py`: save then load round trip on a `BenchmarkRunRef`.
- `tests/dashboard/test_models_page.py`: the sidebar text area exists under the picker; typing into it writes
  `description.json` in the selected run; an existing file pre-fills it; switching runs shows the other run's
  text.

## Non-goals

Showing the description elsewhere (header, run picker label), editing history, markdown rendering.
