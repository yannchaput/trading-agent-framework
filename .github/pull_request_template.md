## Summary

<!-- What does this PR change, and why? One or two sentences is enough. -->

## Related issue / task

<!-- e.g. "Task 7 of the plan", "Closes #123", or "n/a" -->

## Type of change

- [ ] Feature
- [ ] Bug fix (add a regression test)
- [ ] Refactor (no behaviour change)
- [ ] Docs / tooling
- [ ] Strategy change (affects backtest or live results)

## Changes

<!-- Bullet list of the notable changes, grouped by area (brokers, backtesting, core, strategies, dashboard, ...). -->

-

## Testing

- [ ] `uv sync --extra dashboard` run
- [ ] `uv run ruff check` passes
- [ ] `uv run ruff format --check` passes
- [ ] `uv run pytest --cov` passes, coverage >= 80%
- [ ] New or changed code is covered by tests
- [ ] `uv run pyright` passes (if types were touched)
- [ ] `uv run interrogate -v` >= 80%
- [ ] Manual smoke test run (name the script, or "n/a")

Commands run and their results:

```text
<!-- paste the relevant output summary here -->
```

## Backtest / trading impact

<!-- Delete this section if the PR does not touch strategies, backtesting or the executor. -->

- [ ] No look-ahead: data tools take their cutoff from `strategy.clock.now()`, not wall-clock time
- [ ] No `time.sleep` or `datetime.now` added to strategy or executor code
- [ ] Results compared on the same window (before/after numbers, if any):

## Checklist

- [ ] Docstrings added for new public modules, classes, functions and methods
- [ ] No `# noqa`, `# type: ignore`, `# pragma: no cover` or threshold changes added without discussion
- [ ] No debug `print`, commented-out code or leftover TODOs added
- [ ] Money stays `Decimal` (float only at the documented boundaries)
- [ ] No credentials, account IDs or personal data in the diff, logs or tests
- [ ] Nested `CLAUDE.md` or `README.md` updated if behaviour or config changed
- [ ] Reviewed the diff myself before requesting review

## Notes for reviewers

<!-- Anything risky, non-obvious, or worth checking first. Known limitations or shortcuts (`shortcut:` comments) go here too. -->

