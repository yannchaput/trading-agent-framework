# Intraday fall signal spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One research script that says whether a 10-minute VIXY/SPY signal, trading SDS or SPXU intraday with a
volatility-sized trailing stop, day limits and SHV overnight, makes money on 2016-01 to 2024-06, then scores the chosen
cell once on 2024-07 onward.

**Architecture:** `scripts/regime/intraday_fall_signal_spike.py`, modes `check` / `dev` / `test`, in the style of
`scripts/regime/fall_intraday_spike.py` (loaded with `importlib` for its session helpers, as `vix_filter_spike.py`
loads its siblings). Signal values per decision time are computed once per session and shared by all cells; every
trading decision goes through one pure function `decide`.

**Tech Stack:** Python 3.14, `uv`, pandas, numpy, yfinance, alpaca-py, python-dotenv (all present).

**Spec:** `docs/superpowers/specs/2026-11-23-intraday-fall-signal-spike-design.md`

## Global Constraints

- No change under `src/` or `tests/`; no new dependency. Stage only this plan's files by path (the user has
  uncommitted `TODO.md` and `bull_bear/parameters.py` edits: never stage them).
- Reuse from `fall_intraday_spike.py`: `regular_session`, `trim_early_closes`, `is_early_close`, `curve_stats`
  semantics (copy is fine where its `START` differs: it does not).
- Alpaca SIP, `Adjustment.ALL`, 1-minute bars **with volume**, `load_strategy_env("bull_bear", "paper")`; cache
  `cache/intraday_fall_signal/<SYMBOL>_<YEAR>.parquet` (current year not cached).
- Yahoo daily: `VIXY` close (SMA50 through D-1), `SHV` open and close.
- `START` 5,000; ETF cost 3 bps per side; SHV 1 bp per side; fractional shares.
- Dev 2016-01-04 to 2024-06-28; held-out 2024-07-01 to yesterday; guard `cache/intraday_fall_signal/test.done`.
- Grid, fixed values, bar and choice rule exactly as the spec (copied in Task 4).
- `uv run ruff check` and `uv run ruff format --check` (whole repo: it also formats Python blocks in this plan and the
  record) exit 0 before every commit; `/ponytail:ponytail-review` on staged changes before every commit.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A session with fewer than 5 earlier sessions in the data (the first week of 2016): no entries, never a crash.
- A VIXY minute with no trade (thin bars): "VIXY at time x" is the last close at or before x, never a later bar.
- An early-close day: last decision 12:40, forced exit at 12:55, D-1..D-5 volume comparisons skip sessions that did
  not reach the clock time.
- A gap through the stop at the next minute's open fills at that open.
- The day-trade cap counts executed round trips only, across session boundaries (3 in any 5 sessions).

---

### Task 1: Data and signal values

**Files:**
- Create: `scripts/regime/intraday_fall_signal_spike.py`

**Interfaces:**
- Produces:
  - `minute_bars(symbol: str, start: date, end: date) -> pd.DataFrame`: columns `open, high, low, close, volume`, New
    York index, regular session, early closes trimmed.
  - `daily_yahoo() -> pd.DataFrame`: date index, columns `vixy_close, vixy_sma50, shv_open, shv_close`;
    `vixy_sma50` on row d = mean of the 50 VIXY closes **before** d.
  - `day_context(prev: list[pd.DataFrame], sma50: float) -> dict | None`: from the VIXY and SPY minute frames of
    D-1..D-5 (`prev[i] = {"spy": df, "vixy": df}`, most recent first) returns `{"vixy_d1_close", "vixy_d1_vwap",
    "spy_d1_close", "vixy_sma50", "vixy_cumvol": list[pd.Series]}` (each D-k's cumulative VIXY volume indexed by
    minutes since 09:30); `None` when fewer than 5 earlier sessions.
  - `observe(spy: pd.DataFrame, vixy: pd.DataFrame, ctx: dict, t: pd.Timestamp) -> dict`: the spec's values at
    decision time `t` from bars that **close by t** (a minute bar stamped m closes at m+1min): `vixy_day, vixy_hour,
    vixy_vs_d1avg, vixy_trend, rvol, spy_day, spy_hour`, plus `vixy_30, spy_30` (moves over the last 30 minutes,
    used by the fade).
  - `decision_times(session: date, early: bool) -> list[pd.Timestamp]`: 09:40 to 15:40 (12:40 on early closes) every
    10 minutes.

- [ ] **Step 1: Write `check()` with these assertions first**

```python
def check() -> None:
    spy, vixy = synth_day(date(2024, 3, 4)), synth_day(date(2024, 3, 4), drift=0.002)
    prev = [{"spy": synth_day(d), "vixy": synth_day(d)} for d in five_days_before(date(2024, 3, 4))]
    ctx = day_context(prev, sma50=10.0)
    t = pd.Timestamp("2024-03-04 11:00", tz=NY)
    base = observe(spy, vixy, ctx, t)
    # 1. no look-ahead: changing any bar that closes after t never changes the values at t
    moved = vixy.copy()
    moved.loc[moved.index >= t, ["open", "high", "low", "close"]] *= 5
    assert observe(spy, moved, ctx, t) == base
    # 2. vixy_hour at 09:50 is clipped at today's open (window 09:30 -> 09:50), never yesterday
    t950 = pd.Timestamp("2024-03-04 09:50", tz=NY)
    assert close_to(observe(spy, vixy, ctx, t950)["vixy_hour"], vixy.close[vixy.index < t950].iloc[-1] / vixy.open.iloc[0] - 1)
    # 3. rvol compares the same clock time: today's volume to 11:00 over the mean D-1..D-5 volume to 11:00
    assert close_to(base["rvol"], vixy.volume[vixy.index < t].sum() / np.mean([p["vixy"].volume[p["vixy"].index.time < t.time()].sum() for p in prev]))
    # 4. a thin minute: drop the 10:59 VIXY bar -> "now" is the 10:58 close
    thin = vixy.drop(pd.Timestamp("2024-03-04 10:59", tz=NY))
    assert close_to(observe(spy, thin, ctx, t)["vixy_day"], thin.close[thin.index < t].iloc[-1] / ctx["vixy_d1_close"] - 1)
    # 5. D-1 VWAP and a missing history
    d1 = prev[0]["vixy"]
    assert close_to(ctx["vixy_d1_vwap"], (d1.close * d1.volume).sum() / d1.volume.sum())
    assert day_context(prev[:4], 10.0) is None
    # 6. decision times: 37 normal, 19 on an early close, first 09:40
    assert len(decision_times(date(2024, 3, 4), False)) == 37 and len(decision_times(date(2024, 11, 29), True)) == 19
    print("self-check OK")
```

`synth_day`, `five_days_before` and `close_to` are small helpers next to `check()` (random-walk minute bars with
volume on one New York session; the previous 5 weekdays; float compare at 1e-9).

- [ ] **Step 2: Run and see it fail**

Run: `uv run python scripts/regime/intraday_fall_signal_spike.py check`
Expected: `NameError` on `day_context`.

- [ ] **Step 3: Implement the functions above.** VWAP uses each minute's close x volume (no typical price);
"VIXY at x" = last close of bars stamped before x; `vixy_hour` reference = the close of the last bar stamped before
`max(t - 60min, 09:30)`, or the session's first open when that is 09:30.

- [ ] **Step 4: Self-check and lint**

Run: `uv run python scripts/regime/intraday_fall_signal_spike.py check && uv run ruff check && uv run ruff format --check`
Expected: `self-check OK`, both ruff exit 0.

- [ ] **Step 5: Smoke real data:** `minute_bars("VIXY", date(2016,1,4), date(2016,1,8))` has 5 sessions starting
09:30 with a non-zero `volume` column; same for SPY, SDS, SPXU. If VIXY has no 2016 bars, stop and report.

- [ ] **Step 6: Commit** spec, plan and script (`git add` by path), message
`research: intraday fall signal spike, spec, plan, data and signal values`.

### Task 2: The pure decision function and one session's simulation

**Files:**
- Modify: `scripts/regime/intraday_fall_signal_spike.py`

**Interfaces:**
- Consumes: `observe`, `decision_times`, `day_context`.
- Produces:
  - `entry_signal(obs: dict, first: bool, cell: dict) -> bool`: the spec's 4 entry conditions; the fast trigger is
    `vixy_day >= V_DAY` when `first` (the 09:40 decision) else `vixy_hour >= V_HOUR`; volume skipped when
    `cell["rvol"] is None`.
  - `decide(obs: dict, first: bool, in_position: bool, minutes_held: float, limits_hit: bool, cap_ok: bool, cell: dict) -> str`:
    `"enter"`, `"exit"` (fade: `minutes_held >= 30` and `vixy_30 < 0` and `spy_30 > 0`) or `"none"`.
  - `atr10(prev_etf: list[pd.DataFrame]) -> float`: mean 10-minute true range over D-1..D-5 of the ETF.
  - `simulate_session(etf: pd.DataFrame, obs_by_t: dict[pd.Timestamp, dict], cell: dict, atr: float, cap_left: int, early: bool) -> list[dict]`:
    the session's trades, each `{"entry", "exit", "mult", "how"}` (`how` in `stop`, `fade`, `close`), costs in `mult`;
    stops on every minute (spec rules, seeded with the entry fill), fade at decision times, forced exit at the open of
    the 15:55 bar (12:55); daily loss limit (closed P&L <= -2% of the day's opening equity or 2 stops) blocks later
    entries; at most `cap_left` trades.
  - `cell` keys: `v_day, v_hour, rvol (None or 1.5), etf ("SDS" | "SPXU"), k (3 | 5)`; fixed `S_CONF = 0.005`.

- [ ] **Step 1: Add to `check()`**

```python
    cell = {"v_day": 0.03, "v_hour": 0.02, "rvol": None, "etf": "SPXU", "k": 3}
    obs = {"vixy_day": 0.04, "vixy_hour": 0.0, "vixy_vs_d1avg": 0.01, "vixy_trend": True, "rvol": 1.0, "spy_day": -0.006, "spy_hour": -0.001, "vixy_30": 0.01, "spy_30": -0.001}
    # 7. the D-1 close trigger counts only at 09:40; later the hour trigger is needed
    assert entry_signal(obs, True, cell) and not entry_signal(obs, False, cell)
    assert entry_signal(obs | {"vixy_hour": 0.025}, False, cell)
    # 8. every confirmation is required; volume only when the cell has it
    for k, v in (("vixy_vs_d1avg", -0.01), ("vixy_trend", False), ("spy_day", -0.004), ("spy_hour", 0.001)):
        assert not entry_signal(obs | {k: v}, True, cell), k
    assert not entry_signal(obs, True, cell | {"rvol": 1.5}) and entry_signal(obs | {"rvol": 1.6}, True, cell | {"rvol": 1.5})
    # 9. fade only after 30 minutes held, and only when VIXY is down AND SPY up
    fading = obs | {"vixy_30": -0.01, "spy_30": 0.002}
    assert decide(fading, False, True, 30, False, True, cell) == "exit"
    assert decide(fading, False, True, 20, False, True, cell) == "none"
    assert decide(obs | {"vixy_30": -0.01}, False, True, 40, False, True, cell) == "none"
    # 10. limits and cap block entries
    assert decide(obs, True, False, 0, True, True, cell) == "none" and decide(obs, True, False, 0, False, False, cell) == "none"
    # 11. simulate_session: signal at 09:40 -> fill at the 09:40 bar's open; a later bar gapping under the stop fills at its open
    # 12. forced exit at the 15:55 bar's open (12:55 on an early close)
    # 13. two stops in a session -> no third entry; cap_left=1 -> one trade only
```

Checks 11-13 build an ETF minute frame and `obs_by_t` by hand (signal on at every decision time) and assert the
trades' entry/exit times, `how` and `mult` with `close_to(..., costs=True)`.

- [ ] **Step 2: Run, see `NameError` on `entry_signal`.**
- [ ] **Step 3: Implement.** `simulate_session` walks the session's minutes once; the decision at `t` acts on the
bar stamped `t`.
- [ ] **Step 4: Self-check and lint** (as Task 1 Step 4).
- [ ] **Step 5: Commit** `research: intraday fall signal spike, decision function and session simulation`.

### Task 3: The account across sessions

**Files:**
- Modify: `scripts/regime/intraday_fall_signal_spike.py`

**Interfaces:**
- Consumes: Task 2's `simulate_session`, `atr10`; Task 1's data.
- Produces:
  - `run_account(sessions: list[date], data: dict, obs: dict[date, dict], cell: dict | None, cap: bool, overnight: str) -> pd.DataFrame`:
    one row per session: `equity` (close of day, after the overnight leg into the next open is applied on the next
    row), `trades`, `stops`, `fades`, `closes`, `cap_skips`, `limit_skips`. `cell=None` = no trades (the SHV-only row).
    `overnight` in `"shv"` (equity x SHV open(d+1) / close(d), 1 bp each side) or `"cash"`. Cap: executed trades in
    the last 5 sessions including today <= 3.
  - `summary(rows) -> dict`: as `fall_intraday_spike.summary` plus `stops, fades, closes, cap_skips, limit_skips`.
  - `signal_gate(sessions, data, obs, setting: dict) -> float`: mean, over sessions with a signal, of the -1 x SPY trade
    from the first signal to the forced exit, costs in (`2 - p / entry` on SPY minute bars).

- [ ] **Step 1: Add to `check()`:** (14) cap: a signal every session, `cap=True` -> trades by session
`[1,1,1,0,0,1,...]` pattern never exceeds 3 in any 5; `cap=False` -> one per session at least. (15) SHV overnight: no
trades, SHV open/close = 100 -> 100.01 every night -> equity x (1.0001 x (1-1e-4)^2) per night. (16) `cell=None` makes
zero trades. (17) `signal_gate` on a synthetic SPY falling 1% after the first signal returns about +1% minus costs.
- [ ] **Step 2: Run, see `NameError` on `run_account`.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Self-check and lint.**
- [ ] **Step 5: Commit** `research: intraday fall signal spike, account replay, cap and SHV overnight`.

### Task 4: Grid, bar, choice and the dev / test modes

**Files:**
- Modify: `scripts/regime/intraday_fall_signal_spike.py`

**Interfaces:**
- Produces: `GRID` (16 cells), `neighbours(name) -> list[str]`, `choose(passing: dict[str, dict], all_names) -> str | None`,
  `report(window: str, only: str | None = None) -> str | None`, `main()`.

Values fixed by the spec: strength {loose `v_day` 0.03 / `v_hour` 0.02; strict 0.05 / 0.03} x volume {None, 1.5} x etf
{SDS, SPXU} x k {3, 5}. Bar (dev, with cap): the cell's signal setting passed the gate (mean > 0); total > SHV-only
total; at most 2 losing calendar years; at least 30 trades. Choice: most passing neighbours (cells differing in one
dimension), then higher total, then smaller drawdown. Held-out GO: total > 0 and > SHV-only.

- [ ] **Step 1: Add to `check()`:** (18) every cell has exactly 4 neighbours; (19) `choose` prefers a cell with 3
passing neighbours over a higher-total cell with 0; ties on neighbours go to higher total; empty -> `None`.
- [ ] **Step 2: Run, see `NameError` on `neighbours`.**
- [ ] **Step 3: Implement `report` and `main`.** `report("dev")` prints the 4 gate means, then per cell (with cap,
without cap, cash overnight) and the reference rows (-1 x SPY per setting, SHV only, SPY buy and hold, half SPY + half
chosen), per-year returns, exit causes, skips, entry-hour histogram, then `chosen: <cell>` or `NO: <reason>`.
`test` refuses if the guard exists, re-derives the choice from dev, writes the guard, then reports the held-out window
for that cell only and prints `GO` or `NO-GO`.
- [ ] **Step 4: Self-check and lint.**
- [ ] **Step 5: Commit** `research: intraday fall signal spike, grid, bar and modes`.

### Task 5: Dev run, record, one held-out run

**Files:**
- Create: `docs/superpowers/replays/2026-11-23-intraday-fall-signal.md`

- [ ] **Step 1:** `uv run python scripts/regime/intraday_fall_signal_spike.py dev > logs/intraday_fall_signal_dev.txt 2>&1`
Expected: gate means, the cell table, `chosen: <cell>` or `NO: <reason>`.
- [ ] **Step 2:** Write the record's dev section (question, fixed bar, tables, verdict); commit it **before** any
held-out run. On NO: stop here.
- [ ] **Step 3:** `uv run python scripts/regime/intraday_fall_signal_spike.py test > logs/intraday_fall_signal_test.txt 2>&1`
Expected: the chosen cell on 2024-07-01 onward, `GO` or `NO-GO`; a second run refuses.
- [ ] **Step 4:** Add the held-out section; commit.
