# bear_put_spread research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uv run bear-spread-research` replays $5-wide SPY bear put spreads opened on the bearish regime over
2016-2026, with Black-Scholes prices from SPY and VIX, and prints trades, SPY falls, a summary and a GO / NO-GO verdict.

**Architecture:** Three pure modules in `strategies/bear_put_spread/` (`pricing.py`, `rules.py` reused unchanged by
stage 2's paper strategy, `research.py` the daily loop, episodes and verdict) and one I/O module (`research_cli.py`) that
reuses the regime CLI's data path (`YahooRegimeData`, `regime_cli.trim`, `regime_cli.lookback_days`, `regime_by_session`).

**Tech Stack:** Python 3.14 stdlib only (`math.erf`, `csv`, `dataclasses`, `argparse`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-bear-put-spread-research-design.md`

## Global Constraints

- No new dependency; no pandas in the pure modules; the normal CDF is `0.5 * (1 + math.erf(x / math.sqrt(2)))`.
- Read-only: no broker, no orders; writes only `trades.csv`, `episodes.csv`, `equity.csv` in `--out`. Never write a `metrics.json`.
- Tests never touch the network (fake `YahooRegimeData`, as `tests/core/test_regime_cli.py` does).
- Defaults (spec): `long_otm` 0.02, `width` 5.0, expiry DTE in [30, 60] (nearer if two), `exit_dte` 7, `take_profit` 0.80,
  `risk_fraction` 0.15, start equity 1200.0, `vol_scale` 1.0, `skew` 0.8, half bid/ask $0.05/share, commission $1.00 per
  contract per leg on opens only, regulatory $0.10 per contract per leg on opens and closes. Rate 0, dividend 0, years = days/365.
- Floats, not `Decimal`: this is research output, like the regime scorecard, not a ledger.
- Gates on touched code: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest` (cov >= 80%), `uv run interrogate -v`,
  `uv run pyright`. Google docstrings on every public module/class/function. No `noqa`/`type: ignore`/`pragma: no cover`.
- Workflow (user's CLAUDE.md): after a task's tests are green, simplify (code-simplifier agent in subagent-driven mode),
  re-run tests, then run `/ponytail:ponytail-review` on the staged changes before each commit and fix every "Must fix".

## Review Focus

1. VIX at crash levels (80, March 2020) with SPY far below both strikes: the spread's value must stay in [0, width] -> Task 1.
2. A window that starts while the regime is already bearish: a spread opens on the first session -> Task 3.
3. An empty window (`--end` before `--start`, or no session with a reading): `error: ...`, exit 1, nothing written -> Task 5.
4. An account that grew: sizing gives more than 1 contract and costs scale with contracts -> Task 2.
5. The window ends with a spread open: it is marked net of exit costs, reported `open`, and excluded from criterion 1 -> Tasks 3 and 4.

---

### Task 1: pricing

**Files:**
- Create: `src/trading_agent_framework/strategies/bear_put_spread/__init__.py` (module docstring only)
- Create: `src/trading_agent_framework/strategies/bear_put_spread/pricing.py`
- Test: `tests/strategies/bear_put_spread/test_pricing.py` (no `__init__.py` unless sibling test dirs have one; follow `tests/strategies/sector_pods/`)

**Interfaces:**
- Produces:
  - `put_price(spot: float, strike: float, vol: float, years: float) -> float` -- Black-Scholes European put per share, rate 0, dividend 0; `ValueError` if any input is not finite and > 0.
  - `@dataclass(frozen=True, slots=True) class VolModel: vol_scale: float = 1.0; skew: float = 0.8` with `vol(self, spot: float, strike: float, vix: float) -> float` = `vix / 100 * vol_scale + skew * max(0.0, ln(spot / strike))`.
  - `spread_mid(spot: float, long_strike: float, short_strike: float, vix: float, years: float, model: VolModel) -> float` -- per share, `put(long) - put(short)`, each leg with its own `model.vol`.
  - `@dataclass(frozen=True, slots=True) class CostModel: half_spread: float = 0.05; commission: float = 1.00; regulatory: float = 0.10` with `open_cost(self, contracts: int) -> float` = `contracts * 2 * (half_spread * 100 + commission + regulatory)` and `close_cost(self, contracts: int) -> float` = `contracts * 2 * (half_spread * 100 + regulatory)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_at_the_money_put_matches_black_scholes():
    assert put_price(100.0, 100.0, 0.2, 1.0) == pytest.approx(7.96557, abs=1e-4)

def test_put_tends_to_intrinsic_at_expiry():
    assert put_price(90.0, 100.0, 0.3, 1e-6) == pytest.approx(10.0, abs=1e-3)
    assert put_price(110.0, 100.0, 0.3, 1e-6) == pytest.approx(0.0, abs=1e-3)

@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_bad_inputs_raise(bad):
    for args in ((bad, 100, 0.2, 1), (100, bad, 0.2, 1), (100, 100, bad, 1), (100, 100, 0.2, bad)):
        with pytest.raises(ValueError):
            put_price(*args)

def test_skew_raises_vol_below_spot_only():
    m = VolModel()
    assert m.vol(600, 600, 20) == pytest.approx(0.20)
    assert m.vol(600, 660, 20) == pytest.approx(0.20)
    assert m.vol(600, 540, 20) == pytest.approx(0.20 + 0.8 * math.log(600 / 540))

@pytest.mark.parametrize("spot,vix", [(600, 18), (600, 80), (300, 80), (400, 80)])
def test_spread_value_stays_between_zero_and_width(spot, vix):  # Review Focus 1
    v = spread_mid(spot, 588.0, 583.0, vix, 30 / 365, VolModel())
    assert 0.0 <= v <= 5.0

def test_costs_per_contract():
    c = CostModel()
    assert c.open_cost(1) == pytest.approx(12.2)
    assert c.close_cost(1) == pytest.approx(10.2)
    assert c.open_cost(3) == pytest.approx(36.6)
```

- [ ] **Step 2: Run `uv run pytest tests/strategies/bear_put_spread/test_pricing.py -v`** -- Expected: FAIL (module not found).
- [ ] **Step 3: Implement `pricing.py`** per the Interfaces block. `spread_mid` clamps nothing: if the test at VIX 80 shows a tiny negative from float error, clamp to `[0, long - short]` with a one-line comment.
- [ ] **Step 4: Run the same command** -- Expected: all PASS.
- [ ] **Step 5: Commit** `feat(bear_put_spread): Black-Scholes put spread pricing and cost model`

### Task 2: rules

**Files:**
- Create: `src/trading_agent_framework/strategies/bear_put_spread/rules.py`
- Test: `tests/strategies/bear_put_spread/test_rules.py`

**Interfaces:**
- Consumes: nothing from Task 1 (rules take prices as numbers).
- Produces:
  - `@dataclass(frozen=True, slots=True) class SpreadParameters: long_otm: float = 0.02; width: float = 5.0; min_dte: int = 30; max_dte: int = 60; exit_dte: int = 7; take_profit: float = 0.80; risk_fraction: float = 0.15; min_one_fraction: float = 0.20`
  - `third_friday(year: int, month: int) -> date`
  - `monthly_expiry(today: date, params: SpreadParameters) -> date` -- the third Friday with `min_dte <= (expiry - today).days <= max_dte`, the nearer if two. (A [30, 60] window always holds one; raise `ValueError` if a custom window holds none.)
  - `strikes(spot: float, params: SpreadParameters) -> tuple[float, float]` -- `(long, short)`: `long = float(round(spot * (1 - long_otm)))`, `short = long - width`.
  - `should_open(regime: int, has_open: bool) -> bool` -- `regime == -1 and not has_open`.
  - `exit_reason(today: date, expiry: date, regime: int, value: float, width: float, params: SpreadParameters) -> str | None` -- first match: `"time"` if `(expiry - today).days <= exit_dte`; `"target"` if `value >= take_profit * width`; `"regime"` if `regime != -1`; else `None`. `value` is the per-share model mid.
  - `contracts_for(equity: float, debit: float, open_cost_per_contract: float, params: SpreadParameters) -> int` -- `n = floor(risk_fraction * equity / cost)` with `cost = debit * 100 + open_cost_per_contract`; when `n == 0`, return 1 if `cost <= min_one_fraction * equity`, else 0. Never negative.

- [ ] **Step 1: Write the failing tests**

```python
def test_third_fridays():
    assert third_friday(2026, 11) == date(2026, 11, 20)
    assert third_friday(2026, 12) == date(2026, 12, 18)
    assert third_friday(2027, 1) == date(2027, 1, 15)

def test_expiry_takes_the_nearer_of_two():
    assert monthly_expiry(date(2026, 11, 18), SpreadParameters()) == date(2026, 12, 18)  # 30 and 58 days

def test_expiry_skips_one_too_close():
    assert monthly_expiry(date(2026, 11, 19), SpreadParameters()) == date(2027, 1, 15)  # Dec is 29 days

def test_strikes_are_whole_dollars_five_apart():
    assert strikes(600.0, SpreadParameters()) == (588.0, 583.0)
    assert strikes(601.3, SpreadParameters()) == (589.0, 584.0)

def test_should_open():
    assert should_open(-1, False) and not should_open(-1, True) and not should_open(0, False)

def test_exit_priority_time_then_target_then_regime():
    p, exp = SpreadParameters(), date(2026, 12, 18)
    assert exit_reason(date(2026, 12, 11), exp, 0, 4.5, 5.0, p) == "time"
    assert exit_reason(date(2026, 12, 1), exp, 0, 4.0, 5.0, p) == "target"
    assert exit_reason(date(2026, 12, 1), exp, 0, 3.9, 5.0, p) == "regime"
    assert exit_reason(date(2026, 12, 1), exp, -1, 3.9, 5.0, p) is None

def test_sizing():
    p = SpreadParameters()
    assert contracts_for(1200.0, 1.35, 12.2, p) == 1
    assert contracts_for(1200.0, 2.05, 12.2, p) == 1   # 217.2 > 180 but <= 240: the 20% exception
    assert contracts_for(1200.0, 2.30, 12.2, p) == 0   # 242.2 > 240
    assert contracts_for(5000.0, 1.35, 12.2, p) == 5  # Review Focus 4: 750 / 147.2
```

- [ ] **Step 2: Run `uv run pytest tests/strategies/bear_put_spread/test_rules.py -v`** -- Expected: FAIL (module not found).
- [ ] **Step 3: Implement `rules.py`** per the Interfaces block; module docstring says stage 2's strategy calls it unchanged and that holiday-shifted expiries are ignored (one-day error).
- [ ] **Step 4: Run the same command** -- Expected: all PASS.
- [ ] **Step 5: Commit** `feat(bear_put_spread): entry, exit, strike, expiry and sizing rules`

### Task 3: the daily simulation

**Files:**
- Create: `src/trading_agent_framework/strategies/bear_put_spread/research.py`
- Test: `tests/strategies/bear_put_spread/test_research_simulate.py`

**Interfaces:**
- Consumes: Task 1 (`spread_mid`, `VolModel`, `CostModel`), Task 2 (all of `rules.py`).
- Produces:
  - `@dataclass(frozen=True, slots=True) class Trade: entry: date; exit: date | None; reason: str; spy_entry: float; spy_exit: float; long_strike: float; short_strike: float; expiry: date; contracts: int; debit: float; exit_value: float; costs: float; pnl: float; pnl_pct: float` -- `reason` is `"time" | "target" | "regime" | "open"`; `exit` is None and `reason` `"open"` for a spread still open at the last session; `debit`/`exit_value` per share mid; `costs` = open + close cost; `pnl = (exit_value - debit) * 100 * contracts - costs`; `pnl_pct = pnl / equity at entry`.
  - `@dataclass(frozen=True, slots=True) class SimulationResult: trades: list[Trade]; equity: list[tuple[date, float]]; skipped: int`
  - `simulate(sessions: Sequence[date], spy: Sequence[float], vix: Sequence[float], regimes: Mapping[date, int], params: SpreadParameters, model: VolModel, costs: CostModel, start_equity: float) -> SimulationResult`

The loop, per session `d` in order (only the given sessions; the caller passes the window):
1. If a spread is open: value it with `spread_mid(spy[d], long, short, vix[d], (expiry - d).days / 365, model)`; `exit_reason(...)`; on a reason, close at that value (cash += value * 100 * c - close_cost), record the trade, and **do not open again on `d`** (a roll waits for the next session).
2. Else if `should_open(regimes[d], False)`: strikes, expiry, debit = `spread_mid` at `d`; `c = contracts_for(cash, debit, costs.open_cost(1), params)`; 0 -> `skipped += 1`; else cash -= debit * 100 * c + open_cost(c).
3. Append `(d, cash + open mark)` where the open mark is `value * 100 * c - close_cost(c)` (0 without a position).
After the loop an open spread becomes a `Trade` with `exit=None`, `reason="open"`, valued like the last equity point.

- [ ] **Step 1: Write the failing tests** (helper `_days(n)` builds `n` weekdays from 2026-01-05; VIX constant 20, where one $5 spread at SPY 600 costs about $150 with entry costs, under the $180 that 15% of $1,200 allows; at VIX 30 it is borderline)

```python
def test_a_fall_hits_the_target_then_rolls_next_session():
    # SPY 600 falling 1%/session for 30 sessions, regime -1 throughout
    r = simulate(days, spy, vix, {d: -1 for d in days}, SpreadParameters(), VolModel(), CostModel(), 1200.0)
    first, second = r.trades[0], r.trades[1]
    assert first.reason == "target" and first.pnl > 0
    assert second.entry == days[days.index(first.exit) + 1]

def test_a_false_alarm_loses_and_exits_on_regime():
    # SPY flat 600, regime -1 for 5 sessions then 0
    r = simulate(...)
    assert [t.reason for t in r.trades] == ["regime"] and r.trades[0].pnl < 0
    assert r.equity[-1][1] == pytest.approx(1200.0 + r.trades[0].pnl)

def test_a_window_starting_bearish_opens_on_the_first_session():  # Review Focus 2
    assert r.trades[0].entry == days[0]

def test_too_small_an_account_skips_every_bearish_session():
    r = simulate(..., start_equity=100.0)  # 5 bearish sessions
    assert r.trades == [] and r.skipped == 5

def test_an_open_spread_at_the_end_is_marked_net_of_exit_costs():  # Review Focus 5
    t = r.trades[-1]
    assert t.reason == "open" and t.exit is None
    assert t.costs == pytest.approx(CostModel().open_cost(t.contracts) + CostModel().close_cost(t.contracts))
    assert r.equity[-1][1] == pytest.approx(1200.0 + sum(x.pnl for x in r.trades))

def test_equity_has_one_point_per_session():
    assert [d for d, _ in r.equity] == days
```

- [ ] **Step 2: Run `uv run pytest tests/strategies/bear_put_spread/test_research_simulate.py -v`** -- Expected: FAIL.
- [ ] **Step 3: Implement `Trade`, `SimulationResult`, `simulate`** in `research.py` per the loop above.
- [ ] **Step 4: Run the same command** -- Expected: all PASS.
- [ ] **Step 5: Commit** `feat(bear_put_spread): daily spread simulation over regime readings`

### Task 4: episodes, drawdown and verdict

**Files:**
- Modify: `src/trading_agent_framework/strategies/bear_put_spread/research.py`
- Test: `tests/strategies/bear_put_spread/test_research_verdict.py`

**Interfaces:**
- Consumes: Task 3 (`Trade`, `SimulationResult`); `core.regime_scorecard.zigzag` and `Episode`.
- Produces:
  - `@dataclass(frozen=True, slots=True) class EpisodeResult: episode: Episode; trades: int; pnl: float; open: bool`
  - `episode_results(sessions: Sequence[date], spy: Sequence[float], trades: Sequence[Trade], threshold: float = 0.10, attribution: int = 20) -> list[EpisodeResult]` -- `zigzag(sessions, spy, threshold)`; a trade belongs to an episode when `peak <= entry <= sessions[index(trough) + attribution]` (capped at the last session); `open` = no close after the trough is `>= trough close * (1 + threshold)`.
  - `max_drawdown(values: Sequence[float]) -> float` -- largest peak-to-trough fall as a positive fraction; 0 for fewer than 2 values.
  - `@dataclass(frozen=True, slots=True) class Verdict: closed_pnl: float; episodes_won: int; episodes_closed: int; drawdown: float; max_drawdown: float; go: bool` with properties `pnl_ok` (`closed_pnl > 0`), `episodes_ok` (`episodes_closed > 0 and 2 * episodes_won > episodes_closed`), `drawdown_ok` (`drawdown <= max_drawdown`); `go` = all three.
  - `verdict(result: SimulationResult, episodes: Sequence[EpisodeResult], max_drawdown: float = 0.35) -> Verdict` -- `closed_pnl` sums trades with `reason != "open"`; episodes counted only when not `open`; a closed episode with 0 trades is not won; won = `pnl > 0`; drawdown from the whole equity curve.

- [ ] **Step 1: Write the failing tests**

```python
def test_max_drawdown():
    assert max_drawdown([100, 120, 90, 130]) == pytest.approx(0.25)
    assert max_drawdown([100]) == 0.0

def test_a_finished_fall_and_one_still_under_way():
    # closes: 100 -> 85 -> 95 (10%+ rebound) -> 100 -> 88 at the end
    eps = episode_results(days, closes, [trade_in_first_fall_pnl_50])
    assert [(e.open, e.trades, e.pnl) for e in eps] == [(False, 1, 50.0), (True, 0, 0.0)]

def test_verdict_ignores_the_open_fall_and_open_trades():
    v = verdict(result_with_closed_pnl_minus_10_and_open_trade_plus_200, eps)
    assert v.closed_pnl == pytest.approx(-10.0) and not v.pnl_ok and not v.go

def test_a_fall_never_traded_counts_as_not_won():
    v = verdict(result, [won_closed, untraded_closed, untraded_closed])
    assert (v.episodes_won, v.episodes_closed, v.episodes_ok) == (1, 3, False)

def test_no_closed_fall_fails_criterion_two():
    assert not verdict(result, [only_open_episode]).episodes_ok

def test_go_when_all_three_pass():
    assert verdict(profitable_result_dd_20pct, [won, won, lost]).go

def test_drawdown_over_the_bar_fails():
    assert not verdict(result_dd_40pct, [won]).drawdown_ok
```

- [ ] **Step 2: Run `uv run pytest tests/strategies/bear_put_spread/test_research_verdict.py -v`** -- Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces block (reuse `zigzag`; do not copy it).
- [ ] **Step 4: Run the same command** -- Expected: all PASS.
- [ ] **Step 5: Commit** `feat(bear_put_spread): SPY fall attribution, drawdown and GO/NO-GO verdict`

### Task 5: the CLI, the script and the docs

**Files:**
- Create: `src/trading_agent_framework/strategies/bear_put_spread/research_cli.py`
- Create: `src/trading_agent_framework/strategies/bear_put_spread/CLAUDE.md`
- Modify: `pyproject.toml` (`[project.scripts]`: `bear-spread-research = "trading_agent_framework.strategies.bear_put_spread.research_cli:main"`)
- Modify: `src/trading_agent_framework/strategies/CLAUDE.md` (add `bear_put_spread/` to the per-strategy `CLAUDE.md` list, noting it is research only, not a registered strategy)
- Modify: `CLAUDE.md` (Commands: the `uv run bear-spread-research [--start --end --equity --vol-scale --skew --half-spread --out]` line, read-only, downloads Yahoo SPY/^VIX/^VIX3M)
- Test: `tests/strategies/bear_put_spread/test_research_cli.py`

**Interfaces:**
- Consumes: Tasks 1-4; `core.regime_cli.trim`, `core.regime_cli.lookback_days`, `core.regime.regime_by_session`, `core.regime_data.YahooRegimeData`, `RegimeDataError`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`. Flags: `--start` (default `date(2016, 1, 4)`), `--end` (default: the last downloaded session), `--equity` (1200.0), `--vol-scale` (1.0), `--skew` (0.8), `--half-spread` (0.05), `--out` (default `logs/bear_put_spread/research/<YYYY-MM-DD_HHMMSS>/`).

Flow: `YahooRegimeData().series(start - timedelta(days=lookback_days()))` -> `trim(series, end or date.max)` -> `regime_by_session` with default `RegimeParameters()` -> window = sessions in `[start, end]` with a reading -> `simulate` -> `episode_results` over the window -> `verdict` -> print (one line per trade, one per episode, then a summary: total return, final equity, trades, win
rate, average win / average loss, max drawdown, sessions invested, skipped entries) -> write the three CSVs (`csv.DictWriter`, `dataclasses.asdict`; `episodes.csv` flattens `peak`, `trough`, `depth`). `RegimeDataError` or an empty window: print `error: ...`, return 1, write nothing. Exit 0 for GO and NO-GO alike.

Printed lines the tests rely on (rest of the layout is the builder's): one `PASS`/`FAIL` line per criterion, then exactly `verdict: GO` or `verdict: NO-GO`; the open fall on a line containing `still under way`; `skipped: too small` with the count.

- [ ] **Step 1: Write the failing tests** (fake `YahooRegimeData` monkeypatched on `research_cli`, a series like `test_regime_cli._series` with a 15% fall after session 550)

```python
def test_prints_a_verdict_and_writes_three_csvs(tmp_path, monkeypatch, capsys):
    code = research_cli.main(["--start", s[550].isoformat(), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert ("verdict: GO" in out) or ("verdict: NO-GO" in out)
    assert out.count("PASS") + out.count("FAIL") == 3
    assert {p.name for p in tmp_path.iterdir()} == {"trades.csv", "episodes.csv", "equity.csv"}

def test_flags_reach_the_models(...):  # --vol-scale 2 makes the first trade's debit larger than with 1

def test_download_error_exits_1(...):  # fake series() raises RegimeDataError -> "error:" printed, 1, tmp_path empty

def test_empty_window_exits_1(...):  # Review Focus 3: --end before --start -> "error:", 1, tmp_path empty
```

- [ ] **Step 2: Run `uv run pytest tests/strategies/bear_put_spread/test_research_cli.py -v`** -- Expected: FAIL.
- [ ] **Step 3: Implement `research_cli.py`**, add the script to `pyproject.toml`, run `uv sync --extra dashboard` (plain `uv sync` drops the dashboard extra), write the docs: `bear_put_spread/CLAUDE.md` states the stage-1 scope, that `rules.py`/`pricing.py` are pure and reused by stage 2, the conservative modelling choices (adjusted closes, rate 0, mid minus exit costs) and the verdict criteria.
- [ ] **Step 4: Run the same command** -- Expected: all PASS.
- [ ] **Step 5: Run the full gate**: `uv run ruff check && uv run ruff format --check && uv run pytest --cov && uv run interrogate -v && uv run pyright` -- Expected: exit 0, coverage >= 80%, interrogate >= 80%.
- [ ] **Step 6: Commit** `feat(bear_put_spread): bear-spread-research command`

### Task 6: the real run (manual, network)

- [ ] **Step 1:** `uv run bear-spread-research` (2016-01-04 to the latest session, Yahoo download).
- [ ] **Step 2:** Report to the user: the three criteria with their numbers, the verdict, the open fall line, trades and skipped count, and the output directory. Then the two stress runs `--vol-scale 1.2` and `--half-spread 0.10`, reported the same way. No code changes in this task; if the run fails, go to systematic debugging, not a quick fix.
