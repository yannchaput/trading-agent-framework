# Bear call spread: hold through bounces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a choice of regime exit (`not_bearish`, `bullish`, `never`) to the bear call spread replay, run A (`never`) as the decision test and B (`bullish`) for information, and record the results.

**Architecture:** One new field on `SpreadParameters`, read by `exit_reason`; one CLI flag that sets it and a summary line that prints it. The replay loop, entries, pricing, costs and verdict are untouched.

**Tech Stack:** Python 3.14, uv, pytest, argparse.

**Spec:** `docs/superpowers/specs/2026-11-19-bear-call-spread-hold-through-bounces-design.md`

## Global Constraints

- Work in the worktree `.worktrees/bear-call-spread`, branch `feat/bear-call-spread-research`. No merge, push or PR.
- The default stays today's rule: `SpreadParameters()` and `bear-call-research` with no new flag must give the same trades as before.
- GO bar unchanged and fixed before any run: closed P&L > 0, at most 2 losing calendar years, drawdown <= 35%; $5,000 start, 2016-2026 (default `--start`), default pricing and costs.
- Exactly three runs: A default, A at `--call-vol-ratio 0.7`, B default. No other variant.
- Gates before each commit: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest`, `uv run pytest --cov` (>= 80%), `uv run interrogate -v` (>= 80%), `uv run pyright`; then `/ponytail:ponytail-review` on the staged changes.

## Review Focus

- `regime_exit="never"` must still close on time, stop and target (only the regime check goes).
- `regime_exit="bullish"` must hold on regime 0 and exit on regime 1.
- The order time -> stop -> target -> regime must hold for every setting.
- An unknown `--regime-exit` value is a usage error (argparse `choices`), not a silent default.
- The default run's trades must not change (the existing `test_exit_reason_order` cases stay as they are).

---

### Task 1: The `regime_exit` setting and the `--regime-exit` flag

**Files:**
- Modify: `src/trading_agent_framework/strategies/bear_call_spread/scripts/rules.py` (`SpreadParameters`, `exit_reason`)
- Modify: `src/trading_agent_framework/strategies/bear_call_spread/scripts/research_cli.py` (`main`, `run`/`_print_report` summary line)
- Modify: `CLAUDE.md` (the `bear-call-research` commands line gains `--regime-exit not_bearish|bullish|never`)
- Add: the spec and this plan (`docs/superpowers/specs/2026-11-19-bear-call-spread-hold-through-bounces-design.md`, this file)
- Test: `tests/strategies/bear_call_spread/test_rules.py`, `tests/strategies/bear_call_spread/test_research_cli.py`

**Interfaces:**
- Produces: `RegimeExit = Literal["not_bearish", "bullish", "never"]` in `rules.py`; `SpreadParameters.regime_exit: RegimeExit = "not_bearish"` (documented in the class docstring's Attributes). `exit_reason` signature unchanged.
- Produces: CLI flag `--regime-exit` with `choices=("not_bearish", "bullish", "never")`, default `SpreadParameters().regime_exit`; `main` builds `SpreadParameters(regime_exit=args.regime_exit)` and passes it to `run`.
- Produces: a printed line `regime exit: <value>` before the `verdict:` line, from the same `params` passed to `simulate`.

- [ ] **Step 1: Write the failing rule tests** in `test_rules.py`, below `test_exit_reason_order` (leave its cases unchanged):

```python
@pytest.mark.parametrize(
    ("setting", "regime", "expected"),
    [
        ("not_bearish", 0, "regime"),
        ("not_bearish", 1, "regime"),
        ("bullish", 0, None),
        ("bullish", 1, "regime"),
        ("never", 0, None),
        ("never", 1, None),
    ],
)
def test_regime_exit_setting(setting: RegimeExit, regime: int, expected: str | None) -> None:
    assert exit_reason(TODAY, EXPIRY, regime, 2.0, 2.0, SpreadParameters(regime_exit=setting)) == expected


@pytest.mark.parametrize("setting", ["bullish", "never"])
def test_other_exits_still_fire_without_the_regime_exit(setting: RegimeExit) -> None:
    params = SpreadParameters(regime_exit=setting)
    assert exit_reason(date(2026, 12, 11), EXPIRY, 1, 1.5, 2.0, params) == "time"
    assert exit_reason(TODAY, EXPIRY, 1, 4.0, 2.0, params) == "stop"
    assert exit_reason(TODAY, EXPIRY, 1, 1.0, 2.0, params) == "target"


def test_the_default_is_todays_rule() -> None:
    assert SpreadParameters().regime_exit == "not_bearish"
```

Import `RegimeExit` with the other names from `rules`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bear_call_spread/test_rules.py -q`
Expected: collection error, `ImportError: cannot import name 'RegimeExit'`.

- [ ] **Step 3: Implement** `RegimeExit` and `SpreadParameters.regime_exit` in `rules.py`; in `exit_reason`, replace the final regime check with: `not_bearish` exits when `regime != -1`, `bullish` when `regime == 1`, `never` never. Update the `exit_reason` docstring (the regime reason now depends on `params.regime_exit`).

- [ ] **Step 4: Run the rule tests**

Run: `uv run pytest tests/strategies/bear_call_spread/test_rules.py -q`
Expected: all pass, including the unchanged `test_exit_reason_order`.

- [ ] **Step 5: Write the failing CLI tests** in `test_research_cli.py`:

```python
def test_regime_exit_flag_reaches_the_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    data = _patch(monkeypatch)
    base = ["--start", data.sessions[550].isoformat()]

    research_cli.main([*base, "--out", str(tmp_path / "default")])
    assert "regime exit: not_bearish" in capsys.readouterr().out
    research_cli.main([*base, "--regime-exit", "never", "--out", str(tmp_path / "never")])
    assert "regime exit: never" in capsys.readouterr().out

    with (tmp_path / "never" / "trades.csv").open() as f:
        assert all(row["reason"] != "regime" for row in csv.DictReader(f))


def test_an_unknown_regime_exit_is_a_usage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    with pytest.raises(SystemExit):
        research_cli.main(["--regime-exit", "sometimes", "--out", str(tmp_path)])
```

(Import `csv` if the file does not already.)

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bear_call_spread/test_research_cli.py -q`
Expected: the first fails on `"regime exit: not_bearish" in ...`; the second may already pass (argparse rejects the unknown flag); that is fine, it pins the `choices` once the flag exists.

- [ ] **Step 7: Implement** the flag in `main` (help: `"when a regime change closes a spread: not_bearish (today's rule), bullish (only a +1 reading) or never"`), pass `SpreadParameters(regime_exit=args.regime_exit)` to `run`, and print `regime exit: {params.regime_exit}` before `verdict:` (pass the params to `_print_report` or print it in `run` right after the report, before `verdict:` is printed; keep one `params` object for `simulate` and the line). Update the root `CLAUDE.md` commands line.

- [ ] **Step 8: Run the bear call spread tests, then the gates**

Run: `uv run pytest tests/strategies/bear_call_spread -q`, then every gate in Global Constraints.
Expected: all pass; coverage >= 80%; interrogate >= 80%; pyright 0 errors.

- [ ] **Step 9: Review and commit**

Run `/ponytail:ponytail-review` on the staged changes, fix any Must fix, re-run the gates, then:

```bash
git add src/trading_agent_framework/strategies/bear_call_spread/scripts/rules.py src/trading_agent_framework/strategies/bear_call_spread/scripts/research_cli.py tests/strategies/bear_call_spread/test_rules.py tests/strategies/bear_call_spread/test_research_cli.py CLAUDE.md docs/superpowers/specs/2026-11-19-bear-call-spread-hold-through-bounces-design.md docs/superpowers/plans/2026-11-19-bear-call-spread-hold-through-bounces.md
git commit -m "feat(bear_call_spread): choose the regime exit (not_bearish, bullish, never)"
```

### Task 2: The three runs and the record

**Files:**
- Modify: `src/trading_agent_framework/strategies/bear_call_spread/CLAUDE.md` (Results; the `regime_exit` default in Gotchas)

**Interfaces:**
- Consumes: `--regime-exit` from Task 1.

- [ ] **Step 1: Run the three replays** (network: Yahoo), each to its own directory under `logs/bear_call_spread/research/` (git-ignored), saving the printed report next to it:

```bash
uv run bear-call-research --regime-exit never --out logs/bear_call_spread/research/hold_A > logs/bear_call_spread/research/hold_A.txt
uv run bear-call-research --regime-exit never --call-vol-ratio 0.7 --out logs/bear_call_spread/research/hold_A_stress > logs/bear_call_spread/research/hold_A_stress.txt
uv run bear-call-research --regime-exit bullish --out logs/bear_call_spread/research/hold_B > logs/bear_call_spread/research/hold_B.txt
```

Expected: each ends with `regime exit: <value>` and `verdict: GO` or `verdict: NO-GO`, and writes the four CSVs.

- [ ] **Step 2: Read the results** from each run: final equity and return, drawdown, closed trades, win rate, losing years, the three PASS/FAIL lines, the verdict; from `trades.csv`, P&L by exit reason; the split of A's trades into those opened inside a 10%+ fall (between an `episodes.csv` peak and trough) and outside one; and from `years.csv`, the 2023-2026 years (compounded return and losing years).

- [ ] **Step 3: Record them** in `bear_call_spread/CLAUDE.md` under Results, dated 2026-11-19: A's verdict decides (stress next to it), B labelled information only, the 2023-2026 line labelled weak evidence (one fall), and the comparison with the earlier real run (-40.9%) and the hindsight ceiling (+101.9%). Add the `regime_exit` default to the Defaults gotcha. Report numbers as read; no reruns, no new variants.

- [ ] **Step 4: Gates, review and commit**

Run the gates in Global Constraints, `/ponytail:ponytail-review` on the staged change, then:

```bash
git add src/trading_agent_framework/strategies/bear_call_spread/CLAUDE.md
git commit -m "docs(bear_call_spread): hold-through-bounces results (A decides, B information)"
```
