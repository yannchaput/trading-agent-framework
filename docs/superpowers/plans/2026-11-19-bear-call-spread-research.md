# bear_call_spread research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uv run bear-call-research` replays $5-wide SPY bear call spreads sold on the bearish regime over 2016-2026
and prints a GO / NO-GO verdict; `bear_put_spread` is deleted and its reusable parts live in `bear_call_spread`.

**Architecture:** Three pure modules in `strategies/bear_call_spread/` (`pricing.py`, `rules.py`, `research.py`) and one
I/O module (`research_cli.py`). Tasks 1-3 build the new package next to the old one (every commit stays green); Task 4
switches the command over and deletes `bear_put_spread`; Task 5 runs it on real data and records the result.

**Tech Stack:** Python 3.14, `uv`, pytest, stdlib only (`math.erf`, `csv`, `argparse`).

**Spec:** `docs/superpowers/specs/2026-11-18-bear-call-spread-research-design.md`

## Global Constraints

- No new dependency; floats for money in this research package (as in `bear_put_spread`), no `Decimal`.
- Tests never touch the network; the CLI test injects a fake `YahooRegimeData`.
- Gates: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest` (coverage >= 80%), `uv run interrogate -v`
  (>= 80%, Google docstrings on every public module/class/function), pyright clean on the new package.
- Every package has `__version__ = get_version("trading_agent_framework")` (`tests/test_package_versions.py`).
- Defaults (verbatim from the spec): equity $5,000; `short_otm` 0.02; `width` 5.0; `min_dte` 30; `exit_dte` 7;
  `stop_loss` 1.0; `take_profit` 0.50; `risk_fraction` 0.10; `call_vol_ratio` 0.80; `call_skew` 1.0; vol floor 0.01;
  costs `half_spread` 0.05, `commission` 1.00, `regulatory` 0.10; drawdown bar 0.35; at most 2 losing years.
- Before each commit: `/ponytail:ponytail-review` on the staged changes; fix every "Must fix".
- `uv sync --extra dashboard`, never plain `uv sync` (it drops ~290 dashboard tests).

## Review Focus

1. An exit and a new entry on the same session: the roll must wait for the next session (Task 3 test).
2. A calendar year with no trade: return 0, counted as not losing (Task 3 test).
3. A deep in-the-money spread at high VIX: `call_spread_mid` stays within `[0, width]` (Task 1 test).
4. A spread still open at the window's end: out of the closed P&L, but inside the last year's return (Task 3 test).
5. A window inside one calendar year: one year result, flagged partial (Task 3 test).

---

### Task 1: Package and call pricing

**Files:**
- Create: `src/trading_agent_framework/strategies/bear_call_spread/__init__.py` (docstring + `__version__`, as
  `bear_put_spread/__init__.py`)
- Create: `src/trading_agent_framework/strategies/bear_call_spread/pricing.py`
- Test: `tests/strategies/bear_call_spread/test_pricing.py`

**Interfaces:**
- Produces: `put_price(spot, strike, vol, years) -> float` (copied unchanged), `call_price(spot, strike, vol, years) ->
  float`, `CallVolModel(call_vol_ratio: float = 0.80, call_skew: float = 1.0)` with `.vol(spot, strike, vix) -> float`,
  `call_spread_mid(spot, short_strike, long_strike, vix, years, model: CallVolModel) -> float`, `CostModel` (copied
  unchanged: `open_cost(contracts)`, `close_cost(contracts)`).

- [ ] **Step 1: Write the failing tests** (copy the `put_price` and `CostModel` tests from
  `tests/strategies/bear_put_spread/test_pricing.py`, then add):

```python
def test_call_by_parity_at_the_money() -> None:
    assert call_price(100, 100, 0.2, 1.0) == pytest.approx(7.9656, abs=1e-4)


def test_call_tends_to_intrinsic() -> None:
    assert call_price(110, 100, 0.2, 1e-6) == pytest.approx(10.0, abs=1e-3)
    assert call_price(90, 100, 0.2, 1e-6) == pytest.approx(0.0, abs=1e-3)


def test_vol_model_matches_the_anchor_snapshot() -> None:  # 2026-11-17 close, Alpaca OPRA
    model = CallVolModel()
    assert model.vol(732.41, 747, 25.46) == pytest.approx(0.184, abs=0.002)
    assert model.vol(732.41, 752, 25.46) == pytest.approx(0.177, abs=0.002)
    assert model.vol(732.41, 730, 25.46) == pytest.approx(0.207, abs=0.002)  # skew also below spot


def test_vol_model_floor() -> None:
    assert CallVolModel().vol(100, 1000, 10) == 0.01


def test_spread_mid_matches_the_anchor_snapshot() -> None:
    assert call_spread_mid(732.41, 747, 752, 25.46, 31 / 365, CallVolModel()) == pytest.approx(2.13, abs=0.10)


def test_spread_mid_within_width_deep_in_the_money_at_high_vix() -> None:
    assert 0 <= call_spread_mid(800, 700, 705, 80, 30 / 365, CallVolModel()) <= 5
    assert 0 <= call_spread_mid(500, 700, 705, 80, 30 / 365, CallVolModel()) <= 5
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/bear_call_spread/test_pricing.py -v` -- Expected: FAIL (module not
  found).
- [ ] **Step 3: Implement `pricing.py`.** `call_price = put_price(...) + spot - strike` (rate 0); `CallVolModel.vol =
  max(0.01, vix / 100 * call_vol_ratio - call_skew * ln(strike / spot))`; `call_spread_mid` = short-leg call minus
  long-leg call, each at its own strike's vol, clamped to `[0, long_strike - short_strike]`. Module docstring states rate
  0, dividend 0, the anchor snapshot and why the skew runs on both sides.
- [ ] **Step 4: Run** the same command -- Expected: PASS.
- [ ] **Step 5: Commit** `feat(bear_call_spread): call pricing anchored on a real SPY snapshot`

### Task 2: Rules

**Files:**
- Create: `src/trading_agent_framework/strategies/bear_call_spread/rules.py`
- Test: `tests/strategies/bear_call_spread/test_rules.py`

**Interfaces:**
- Produces: `SpreadParameters(short_otm=0.02, width=5.0, min_dte=30, exit_dte=7, stop_loss=1.0, take_profit=0.50,
  risk_fraction=0.10)` (frozen); `third_friday(year, month) -> date` and `monthly_expiry(today, params) -> date` (copied
  unchanged); `strikes(spot, params) -> tuple[float, float]` returning `(short, long)`; `should_open(regime, has_open) ->
  bool`; `exit_reason(today, expiry, regime, value, credit, params) -> str | None` (`"time"`, `"stop"`, `"target"`,
  `"regime"`); `contracts_for(equity, credit, width, round_trip_cost, params) -> int`.

- [ ] **Step 1: Write the failing tests** (copy the `third_friday` / `monthly_expiry` tests from
  `tests/strategies/bear_put_spread/test_rules.py`, then add):

```python
P = SpreadParameters()
TODAY, EXPIRY = date(2026, 11, 18), date(2026, 12, 18)


def test_strikes_short_two_percent_above_long_five_higher() -> None:
    assert strikes(732.41, P) == (747.0, 752.0)


def test_should_open_only_bearish_and_flat() -> None:
    assert should_open(-1, False) and not should_open(-1, True) and not should_open(0, False)


@pytest.mark.parametrize(
    ("today", "regime", "value", "expected"),
    [
        (date(2026, 12, 11), -1, 1.0, "time"),  # 7 days left
        (date(2026, 12, 11), -1, 5.0, "time"),  # time beats stop
        (TODAY, -1, 4.0, "stop"),  # 2x the 2.0 credit
        (TODAY, 1, 4.0, "stop"),  # stop beats regime
        (TODAY, -1, 1.0, "target"),  # half the credit
        (TODAY, 1, 1.0, "target"),  # target beats regime
        (TODAY, 0, 2.0, "regime"),
        (TODAY, -1, 2.0, None),
    ],
)
def test_exit_reason_order(today: date, regime: int, value: float, expected: str | None) -> None:
    assert exit_reason(today, EXPIRY, regime, value, 2.0, P) == expected


def test_size_by_max_loss() -> None:
    # max loss per contract = (5 - 2) * 100 + 22.4 (open 12.2 + close 10.2) = 322.4; 10% of 5000 = 500 -> 1
    assert contracts_for(5000, 2.0, 5.0, 22.4, P) == 1
    assert contracts_for(20000, 2.0, 5.0, 22.4, P) == 6


def test_too_small_gives_zero() -> None:
    assert contracts_for(1000, 2.0, 5.0, 22.4, P) == 0
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/bear_call_spread/test_rules.py -v` -- Expected: FAIL.
- [ ] **Step 3: Implement `rules.py`.** Short strike `float(round(spot * (1 + short_otm)))`, long `+ width`. Exit order
  time -> stop (`value >= credit * (1 + stop_loss)`) -> target (`value <= credit * (1 - take_profit)`) -> regime.
  `contracts_for = max(0, floor(risk_fraction * equity / ((width - credit) * 100 + round_trip_cost)))`. Module docstring
  keeps the holiday-expiry note and adds: early assignment of the short call before an ex-dividend date is ignored, the
  time stop closes 7 days before expiry.
- [ ] **Step 4: Run** the same command -- Expected: PASS.
- [ ] **Step 5: Commit** `feat(bear_call_spread): entry, exit, strike, expiry and max-loss sizing rules`

### Task 3: Simulation, years and verdict

**Files:**
- Create: `src/trading_agent_framework/strategies/bear_call_spread/research.py`
- Test: `tests/strategies/bear_call_spread/test_research.py`

**Interfaces:**
- Consumes: Task 1 `call_spread_mid`, `CallVolModel`, `CostModel`; Task 2 everything.
- Produces:
  - `Trade(entry, exit, reason, spy_entry, spy_exit, short_strike, long_strike, expiry, contracts, credit, exit_value,
    margin, costs, pnl, pnl_pct)` -- `reason` adds `"open"`; `margin = width * 100 * contracts`;
    `pnl = (credit - exit_value) * 100 * contracts - costs`.
  - `SimulationResult(trades, equity: list[tuple[date, float]], skipped: int)`.
  - `simulate(sessions, spy, vix, regimes, params, model: CallVolModel, costs, start_equity) -> SimulationResult`.
  - `YearResult(year: int, start_equity, end_equity, ret, trades: int, partial: bool)`;
    `year_results(equity, trades, start_equity) -> list[YearResult]`.
  - `EpisodeResult`, `episode_results(...)`, `max_drawdown(...)` copied unchanged from `bear_put_spread/research.py`.
  - `Verdict(closed_pnl, losing_years, years, drawdown, max_drawdown, max_losing_years)` with properties `pnl_ok`,
    `years_ok` (`losing_years <= max_losing_years`), `drawdown_ok`, `go`;
    `verdict(result, years, max_drawdown_bar=0.35, max_losing_years=2) -> Verdict`.

- [ ] **Step 1: Write the failing tests.** Synthetic sessions: weekdays from 2021-01-04; regimes set by hand; flat VIX
  20; `CallVolModel()`, `CostModel()`, `SpreadParameters()`, equity 5000.

```python
def test_a_drift_down_keeps_the_credit() -> None:
    # SPY 400 falling 0.3%/session for 40 sessions, regime -1 throughout
    result = simulate(...)
    first = result.trades[0]
    assert first.reason == "target" and first.pnl > 0 and first.margin == 500.0


def test_a_rally_hits_the_stop() -> None:
    # SPY 400 rising 1%/session, regime -1 throughout
    first = simulate(...).trades[0]
    assert first.reason == "stop" and first.pnl < 0


def test_equity_after_entry_is_start_minus_round_trip_costs() -> None:
    # one session, regime -1: credit received and the mark cancel at the same mid
    result = simulate(...)
    assert result.equity[0][1] == pytest.approx(5000 - CostModel().open_cost(1) - CostModel().close_cost(1))


def test_the_roll_waits_for_the_next_session() -> None:
    # exit on session k (target), regime still -1: trades[1].entry is session k+1, never k
    ...


def test_open_at_the_end_is_out_of_closed_pnl_but_in_the_last_year() -> None:
    result = simulate(...)  # regime -1 on the last 3 sessions only
    years = year_results(result.equity, result.trades, 5000)
    assert result.trades[-1].reason == "open"
    assert verdict(result, years).closed_pnl == pytest.approx(sum(t.pnl for t in result.trades[:-1]))
    assert years[-1].end_equity == pytest.approx(result.equity[-1][1])


def test_a_year_without_trades_is_not_losing() -> None:
    # regime 0 through 2021, -1 in 2022
    years = year_results(...)
    assert years[0].ret == 0 and years[0].trades == 0


def test_a_window_inside_one_year_is_partial() -> None:
    # sessions 2021-03-01 .. 2021-06-30
    (only,) = year_results(...)
    assert only.partial


def test_verdict_lines() -> None:
    # hand-built SimulationResult and years: 3 losing years -> years_ok False; 2 -> True;
    # drawdown 0.36 -> drawdown_ok False; closed pnl 0 -> pnl_ok False; all good -> go True
    ...
```

  Copy the `episode_results` and `max_drawdown` tests from `tests/strategies/bear_put_spread/test_research_verdict.py`,
  adapted to the new `Trade` fields.
- [ ] **Step 2: Run** `uv run pytest tests/strategies/bear_call_spread/test_research.py -v` -- Expected: FAIL.
- [ ] **Step 3: Implement `research.py`.** Same loop shape as `bear_put_spread.simulate`, with credit signs: entry
  `cash += credit * 100 * n - open_cost(n)`; mark `equity = cash - (mid * 100 * n + close_cost(n))`; exit
  `cash -= mid * 100 * n + close_cost(n)` then `continue` (no same-session re-entry). Size with
  `contracts_for(cash, credit, width, open_cost(1) + close_cost(1), params)` (cash = equity when flat); 0 -> `skipped +=
  1`. `year_results`: group the equity curve by calendar year; start equity = the previous year's last value
  (`start_equity` for the first); `ret = end / start - 1`; `trades` = trades closed in that year; `partial` = the first
  session is after 7 January or the last is before 24 December. `verdict`: `losing_years` = years with `ret < 0`.
- [ ] **Step 4: Run** the same command -- Expected: PASS.
- [ ] **Step 5: Commit** `feat(bear_call_spread): credit-spread simulation, calendar years and GO/NO-GO verdict`

### Task 4: The command, docs, and deleting bear_put_spread

**Files:**
- Create: `src/trading_agent_framework/strategies/bear_call_spread/research_cli.py`
- Create: `src/trading_agent_framework/strategies/bear_call_spread/CLAUDE.md`
- Delete: `src/trading_agent_framework/strategies/bear_put_spread/` and `tests/strategies/bear_put_spread/` (`git rm -r`)
- Modify: `pyproject.toml` (`bear-spread-research` -> `bear-call-research =
  "trading_agent_framework.strategies.bear_call_spread.research_cli:main"`)
- Modify: `CLAUDE.md` (Commands line; the strategy list names `bear_call_spread` as research only)
- Modify: `src/trading_agent_framework/strategies/CLAUDE.md` (index entry `bear_call_spread/`)
- Test: `tests/strategies/bear_call_spread/test_research_cli.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `run(start, end, equity, vol: CallVolModel, costs, out, params=None) -> int`, `main(argv=None) -> int`.
  Flags: `--start` (default `DEFAULT_START`), `--end`, `--equity` (5000.0), `--call-vol-ratio` (0.80), `--call-skew`
  (1.0), `--half-spread` (0.05), `--out` (default `logs/bear_call_spread/research/<YYYY-MM-DD_HHMMSS>/`). No
  `--trigger`.

- [ ] **Step 1: Write the failing tests** -- port from `tests/strategies/bear_put_spread/test_research_cli.py` the fake
  Yahoo `_series`/`_patch` helpers and these tests: prints a verdict and writes the CSVs (now four: `trades.csv`,
  `years.csv`, `episodes.csv`, `equity.csv`); flags reach the models (`--call-vol-ratio 0.7 --call-skew 0.5
  --half-spread 0.1 --equity 8000` reach `run`); a download error exits 1 and writes nothing; an empty window exits 1
  and writes nothing. Add: the report prints one `year` line per calendar year and the three `PASS`/`FAIL` lines
  (`closed trades P&L`, `losing years`, `max drawdown`). Drop the trigger and put-structure tests.
- [ ] **Step 2: Run** `uv run pytest tests/strategies/bear_call_spread/test_research_cli.py -v` -- Expected: FAIL.
- [ ] **Step 3: Implement `research_cli.py`** (same shape as the put CLI: download, `trim`, `regime_by_session`, window,
  `simulate`, `year_results`, `episode_results`, print, write). The report prints trades (with credit and margin),
  years (`year 2020 [partial] +x.x% (n trades)`), SPY falls labelled "information only", the summary lines of the put
  CLI, the three verdict lines and `verdict: GO|NO-GO`.
- [ ] **Step 4: Delete the old package and switch the docs.** `git rm -r` both `bear_put_spread` directories, edit
  `pyproject.toml`, run `uv sync --extra dashboard`. `bear_call_spread/CLAUDE.md`: research only (no `Strategy`), the
  pure modules stage 2 reuses, the modelling choices (adjusted closes, rate 0, call vol anchored on the 2026-11-17
  snapshot, both-sided skew, mid minus/plus costs), the verdict, and a "Bear put spread record" section copied from the
  spec, ending "code on branch `feat/bear-put-spread-research`". Update both index `CLAUDE.md` files.
- [ ] **Step 5: Run the gates** -- `uv run pytest` (all pass), `uv run pytest --cov` (>= 80%), `uv run ruff check`,
  `uv run ruff format --check`, `uv run interrogate -v`, `uv run pyright src/trading_agent_framework/strategies/bear_call_spread`
  -- Expected: all clean; `grep -rn bear_put_spread src tests pyproject.toml CLAUDE.md` -- Expected: only the record
  section in `bear_call_spread/CLAUDE.md`.
- [ ] **Step 6: Commit** `feat(bear_call_spread): bear-call-research command; remove bear_put_spread`

### Task 5: Real run and the recorded result

**Files:**
- Modify: `src/trading_agent_framework/strategies/bear_call_spread/CLAUDE.md` (a "Results" line)

- [ ] **Step 1:** `uv run bear-call-research --out <scratchpad>/default` (2016-01-04 to the latest session).
  Expected: a verdict line; read every year line and the falls.
- [ ] **Step 2:** `uv run bear-call-research --call-vol-ratio 0.70 --out <scratchpad>/stress`. Expected: a verdict line.
- [ ] **Step 3:** Add one "Results" line to `CLAUDE.md` with both verdicts, total return, losing years, max drawdown,
  win rate and the 2020 year return. No other run, no parameter change (spec anti-goal).
- [ ] **Step 4: Commit** `docs(bear_call_spread): 2016-2026 research result`
