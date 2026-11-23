# Earnings Short Side Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One research script that tests whether a debit put spread opened after a bad earnings reaction makes money:
step 1 on stock moves (2016-2026), step 2 on real Alpaca option prices (2024-2026), with a pre-registered GO / NO-GO.

**Architecture:** One file, `scripts/earnings/short_side_spike.py`, modes `check` (assert self-check on synthetic
data, no network), `dev`, `test` (one shot, guard file), built like `scripts/regime/fall_intraday_spike.py` (module
constants, plain functions, `check()` of numbered asserts, `main()` on `sys.argv[1]`). It reuses the framework's pure
parts: `reaction_date`, `earnings_events`, `reaction_features`, `older_pages`, `UniverseHistory`, `parse_yahoo_frame`,
`SecEdgarClient`, `in_market_fall`.

**Tech Stack:** Python 3.14, uv, pandas, yfinance, alpaca-py, python-dotenv (all installed).

**Spec:** `docs/superpowers/specs/2026-11-23-earnings-short-side-spike-design.md`

## Global Constraints

- No change under `src/` or `tests/`; no new dependency. Script only, plus a replay record.
- Built in the worktree `.worktrees/earnings-short-side`; git-ignored inputs (`cache/`, `data/universe/`, `env/`) read
  from the main checkout: `MAIN = Path(git rev-parse --git-common-dir).resolve().parent`. Caches the script writes go to
  `MAIN / "cache" / "earnings_short_side"`; SEC cache stays `MAIN / "cache" / "sec"` (`default_client(MAIN)`).
- Credentials: `env/.env.bull_bear.paper` only (`ALPACA_DATA_API_KEY`, `ALPACA_DATA_API_SECRET`,
  `SEC_EDGAR_USER_AGENT`), read with `dotenv_values`; read-only calls, never an order.
- Windows: dev **2016-01-04 to 2023-12-29**; held-out **2024-01-02 to the last complete session**, scored once.
- Account: START **5,000**, **5 slots**, **20%** of equity per new position, idle cash earns nothing.
- Step 1 cost **80 bps per side** (bar), info rows at **10 bps per side**; position value `2 - max(p / entry, 0.9)`.
- Grid: `D` in {0.05, 0.08} x `H` in {5, 10, 20}; fixed `close_location < 0.5`, `rel_volume >= 2.0`, traded close
  `>= 10`, `dollar_volume_20d >= 20_000_000`.
- Step 2: half-spread **0.004 x stock traded open** per leg per side; Tastytrade fees: $1.00 per contract to open capped
  at $10 per leg, $0 to close, **$0.13** per contract per side (clearing $0.10 + regulatory ~$0.03).
- Lint: `uv run ruff check` and `uv run ruff format --check` clean on the script and the Markdown files.

## Review Focus

1. A stock with no bar on its `H`-th session (halt, delisting, data end): exit at its first later bar, else its last
   bar; never a crash or a silent drop. Test in Task 1 (`exit_index`).
2. Share classes on one 8-K (GOOG/GOOGL, BRK-A/B): one event per CIK and accession, the symbol first in the month's
   list; a held CIK blocks its sibling. Test in Task 1 (`dedupe_by_cik`) and Task 2 (`replay_stocks`).
3. A -1x stock that more than doubles: value floors at 0, never negative equity. Test in Task 1 (`trade_value`).
4. A put spread whose computed exit credit is negative (legs near zero minus half-spreads): exit credit floors at 0.
   Test in Task 3 (`spread_cash`).
5. Strike pick where the nearest 10%-lower strike equals the ATM strike (sparse chain): skipped as `no spread`,
   counted. Test in Task 3 (`pick_contracts`).

---

### Task 1: Data, signal and step-1 trade maths, with the self-check

**Files:**
- Create: `scripts/earnings/short_side_spike.py`

**Interfaces:**
- Consumes: `reaction_date(accepted_at, trading_dates)`, `earnings_events(symbol, payloads)` (`earnings_drift/
  events.py`); `reaction_features(stock, benchmark, day)` (`earnings_drift/reaction.py`, needs a tz-aware index and
  lowercase `open high low close volume`); `older_pages(payload, since)` (`fundamentals/sec.py`); `default_client(root)`,
  `client.ticker_to_cik(symbol)` (raises `FundamentalsNotFoundError`), `client.get_submissions_payload(cik)`,
  `client.get_submissions_page(name)`; `UniverseHistory.load(path).on(day)`; `parse_yahoo_frame(raw)` (`traded_close`).
- Produces:
  - `MAIN: Path`, `START = 5000.0`, `SLOTS = 5`, `FRACTION = 0.20`, `COST = 0.008`, `INFO_COST = 0.001`,
    `CAP = 0.9`, `WINDOWS = {"dev": (date(2016, 1, 4), date(2023, 12, 29)), "test": (date(2024, 1, 2), <yesterday>)}`,
    `GRID = [(d, h) for d in (0.05, 0.08) for h in (5, 10, 20)]`, `CELLS = {f"D{d:.0%} H{h}": (d, h)}`.
  - `load_bars(symbol: str) -> pd.DataFrame | None`: the cached parquet `MAIN/cache/universe_history/bars/<SYMBOL>
    .parquet`, index tz-aware session close (as `parse_yahoo_frame`), columns `open high low close volume`
    (split-adjusted, for the features), `tr_open tr_close` (total return: `Adj Close`, `Open x Adj Close / Close`),
    `traded_open traded_close` (`traded_close` from `parse_yahoo_frame`, `traded_open = Open x traded_close / Close`),
    plus `day` (the session `date`) as a column for lookups; None when the file is missing.
  - `Event` = plain dict `{"symbol", "cik", "accession", "reaction_day", "entry_day", "abnormal", "features"}`.
  - `qualifies(f: ReactionFeatures, traded_close: float, drop: float) -> bool`: the spec's rules 1-4.
  - `dedupe_by_cik(events: list[dict], month_list: list[str]) -> list[dict]`: one per `(cik, accession)`, keeping the
    symbol earliest in `month_list`.
  - `exit_index(days: list[date], entry_pos: int, hold: int) -> int`: position of the exit bar in a stock's own bar
    list: the bar on the `hold`-th SPY session counted from entry (entry = 1), else the first later bar, else the last.
  - `trade_value(ratio: float, cost: float) -> float`: value of 1 unit of notional at the exit:
    `max(0.0, (1 - cost) * (2 - max(ratio, CAP)) * (1 - cost))`; the trade return is `trade_value - 1`.
  - `find_events(sessions: list[date], universe: UniverseHistory, bars: dict[str, pd.DataFrame], client) -> list[dict]`:
    every 8-K 2.02 event of a universe symbol (in the list of its reaction day's month) with features, entry day = the
    next SPY session; events are not yet filtered by `qualifies` (the grid filters). Cached as a pickle in the script's
    cache dir keyed by the window, so `dev` and `test` do not re-walk SEC.

- [ ] **Step 1: Write the failing self-check**

`check()` with numbered asserts on synthetic data (no network, no files), `main()` dispatching `check`. Asserts:

```python
# 1. reaction day: a release at 16:05 ET on a Monday reacts Tuesday; at 08:00 ET reacts Monday
# 2. qualifies: abnormal -0.06, close_location 0.3, rel_volume 2.5, traded close 12, dollar volume 25e6 -> True at D=5%,
#    False at D=0.08; close_location 0.5 -> False; rel_volume 1.99 -> False; traded close 9.99 -> False; 19.9e6 -> False
# 3. trade_value: ratio 0.95, cost 0 -> 1.05; ratio 0.70 (crash) -> 1.10 (the 10% cap); ratio 1.10 -> 0.90;
#    ratio 2.5 -> 0.0 (floor); ratio 0.95, cost 0.008 -> 1.05 * 0.992**2
# 4. exit_index: stock days == SPY days, entry_pos 0, hold 5 -> 4; stock missing SPY's 5th session -> first later bar;
#    stock data ends before -> last bar
# 5. dedupe_by_cik: GOOGL and GOOG, same cik and accession, month list [..., "GOOGL", ..., "GOOG"] -> one event, GOOGL
# 6. load_bars on a synthetic raw frame with a 2:1 split after day 1 and a dividend: traded_close day 0 == 2 x Close,
#    tr_close == Adj Close, tr_open == Open x Adj Close / Close, close == Close (split-adjusted, not dividend-adjusted)
# 7. find_events with a fake client (ticker_to_cik, get_submissions_payload, get_submissions_page) and 3 synthetic
#    symbols: an event whose symbol is absent from its reaction month's list is dropped; an after-close release's
#    entry_day is the second SPY session after the release date; an unknown ticker is counted as "no cik"
print("self-check OK")
```

`load_bars` takes the raw frame through a `bars_from_raw(raw: pd.DataFrame) -> pd.DataFrame` helper so assert 6 needs
no file; `load_bars` = read parquet + `bars_from_raw`.

- [ ] **Step 2: Run it to see it fail**

Run: `uv run python scripts/earnings/short_side_spike.py check`
Expected: FAIL (`NameError` / `AssertionError` on the first missing function).

- [ ] **Step 3: Implement the constants and the functions in "Produces"**

`find_events` walks `universe.union(start, end)`: for each symbol with bars, `ticker_to_cik` (unknown ticker -> counted
as `no cik`, skipped), events from the recent submissions payload plus `older_pages(recent, date(2015, 1, 1))`,
`reaction_date` on SPY's sessions, keep the event when the symbol is in `universe.on(reaction_day)` and
`reaction_features(stock, spy, reaction_day)` is not None; then `dedupe_by_cik` per reaction day with that month's
list. Print the counts (symbols, no cik, events, kept).

- [ ] **Step 4: Run the self-check to see it pass, then lint**

Run: `uv run python scripts/earnings/short_side_spike.py check && uv run ruff check scripts/earnings && uv run ruff
format --check scripts/earnings`
Expected: `self-check OK`, both ruff commands exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/earnings/short_side_spike.py
git commit -m "research: earnings short-side spike, data, signal and trade maths"
```

---

### Task 2: Step-1 account replay, the bar, and `dev` mode

**Files:**
- Modify: `scripts/earnings/short_side_spike.py`

**Interfaces:**
- Consumes: Task 1's `find_events`, `qualifies`, `exit_index`, `trade_value`, `load_bars`, constants.
- Produces:
  - `gate_trades(events: list[dict], bars, drop: float, hold: int, cost: float) -> pd.DataFrame`: one row per
    qualifying event, no slot limit (columns `entry_day exit_day ret year in_fall`).
  - `replay_stocks(events, bars, sessions: list[date], drop, hold, cost, falls: set[date]) -> tuple[pd.Series,
    pd.DataFrame]`: daily equity (index = sessions) and the trades (`symbol entry_day exit_day ret in_fall`), plus the
    skip counts in `trades.attrs["skipped"]` (`slots full`, `already held`, `no entry bar`).
  - `summary(equity: pd.Series, trades: pd.DataFrame) -> dict`: total, CAGR, per-year returns, max drawdown, trades,
    win rate, mean trade (the reference script's `summary`, adapted).
  - `passes(s: dict, shv_total: float) -> bool`: total > SHV, at most `MAX_LOSING_YEARS = 2` losing years, trades >= 30.
  - `choose(results: dict[str, dict], passed: set[str]) -> str | None`: most passing neighbours (cells differing in
    exactly one grid value: same `D` other `H`, or same `H` other `D`), then higher total, then smaller drawdown.
  - `report_dev() -> str | None`: prints every table and returns the chosen cell name or None.
  - `falls_before_open(sessions) -> set[date]`: sessions whose open follows an `in_market_fall` close (Yahoo ^VIX,
    ^VIX3M, VIXY from 2008-01-02, as the reference script's `daily_closes` / `fall_days`).

- [ ] **Step 1: Add failing asserts to `check()`**

```python
# 8. replay_stocks, one slot free and two events the same reaction day: the larger drop (-0.09) enters, -0.06 is
#    "slots full"; equity at the entry open sizes 20%; a held CIK skips its sibling ("already held")
# 9. replay_stocks: a position's slot frees at its exit close and is usable at the next open, not the same session
# 10. replay_stocks: a position open at the window's last session is closed at that close
# 11. summary on a 2-year curve: years[2023], years[2024], max_dd match hand values
# 12. passes: total 0.10 vs SHV 0.12 -> False; 3 losing years -> False; 29 trades -> False; otherwise True
# 13. choose: with D5%H5, D5%H10, D8%H10 passing, D5%H10 (2 passing neighbours) wins; a tie goes to the higher total
# 14. gate_trades mean on two hand-built trades (ratios 0.95 and 1.02, cost 0.008) equals the hand mean
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run python scripts/earnings/short_side_spike.py check`
Expected: FAIL on assert 8.

- [ ] **Step 3: Implement the functions in "Produces" and `dev` in `main()`**

Replay order per session: (1) exits due at today's close are applied at the close; (2) at the open, positions are
marked at `tr_open`, equity = cash + marks, new entries from yesterday's reaction day take free slots, largest drop
first, ties by symbol; (3) the close marks every position (for the drawdown). A position's mark is `alloc x
trade_value(tr_close / entry_tr_open, cost)` without the exit cost until it exits. `report_dev` prints, per cell, the
gate row (mean trade at 80 bps, and at 10 bps for information), the account row at 80 bps and the info row at 10 bps,
the per-year gate means and events per year, then SHV and SPY buy and hold
(SHV `yf.download("SHV", auto_adjust=True)`, SPY from the
cache), the chosen cell's P&L split by `in_fall`, and the verdict line. A cell whose gate mean is <= 0 is out; all 6
out prints `NO (sign gate)` and returns None.

- [ ] **Step 4: Run the self-check and lint**

Run: `uv run python scripts/earnings/short_side_spike.py check && uv run ruff check scripts/earnings && uv run ruff
format --check scripts/earnings`
Expected: `self-check OK`, ruff exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/earnings/short_side_spike.py
git commit -m "research: earnings short-side spike, step-1 account replay and dev mode"
```

---

### Task 3: Step 2 (real put spreads), the data probe and `test` mode

**Files:**
- Modify: `scripts/earnings/short_side_spike.py`

**Interfaces:**
- Consumes: Task 2's `replay_stocks`, `summary`, `report_dev`, Task 1's events and bars.
- Produces:
  - `STEP2_START = date(2024, 2, 1)` (Alpaca's option history start), `HALF_SPREAD = 0.004`, `OPEN_FEE = 1.00`,
    `LEG_FEE_CAP = 10.0`, `SIDE_FEE = 0.13`.
  - `put_contracts(underlying: str, expiry_from: date, expiry_to: date) -> list[tuple[str, date, float]]`: `(occ
    symbol, expiry, strike)` of puts, active and inactive (`GetOptionContractsRequest`, `status` both), via alpaca-py
    `TradingClient(key, secret, paper=True)` with the data keys (if the trading endpoint refuses them, the same env
    file's paper `ALPACA_API_KEY` / `ALPACA_API_SECRET`, still read-only); cached to JSON per `(underlying,
    expiry_from)`.
  - `option_bars(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]`: daily bars (`vwap close`,
    date index) via `OptionHistoricalDataClient.get_option_bars`; cached per symbol.
  - `pick_contracts(contracts, entry_day: date, sessions: list[date], hold: int, open_price: float) -> tuple[str, str,
    date] | None`: first expiry at least `hold` sessions after the entry session; long = strike nearest `open_price`,
    short = strike nearest `0.9 x open_price`; None when none or short strike >= long strike (`no spread`).
  - `spread_cash(long_vwap: float, short_vwap: float, half: float, opening: bool) -> float`: per share; opening
    `(long + half) - (short - half)` (debit), closing `max(0.0, (long - half) - (short + half))` (credit).
  - `fees(contracts: int, opening: bool) -> float`: opening `2 x min(OPEN_FEE x n, LEG_FEE_CAP) + 2 x SIDE_FEE x n`,
    closing `2 x SIDE_FEE x n`.
  - `replay_spreads(events, bars, sessions, drop, hold) -> tuple[pd.Series, pd.DataFrame]`: as `replay_stocks`, but a
    skipped candidate (`no contracts`, `no spread`, `no entry bar`, `too expensive`) passes its slot to the next drop of
    the same reaction day; `n = floor(FRACTION x equity / (debit x 100))`, 0 = `too expensive`; exit legs at the
    exit session's VWAP, else the first later bar, else intrinsic `max(K - traded_close_at_expiry, 0)`; daily marks
    from each leg's close, else its last known close; a spread open at the window's end exits at the legs' last
    available VWAP, else intrinsic value.
  - `probe() -> bool`: AAPL, MSFT, NVDA: put contracts expiring in March 2024 exist and one of each has daily bars in
    February 2024. Prints what it found; never looks at a result.

- [ ] **Step 1: Add failing asserts to `check()`**

```python
# 15. pick_contracts: strikes 95, 100, 105, 90 for 2 expiries; open 101, hold 5 -> long 100, short 90, the first expiry
#     >= 5 sessions after entry; an expiry 3 sessions out is passed over; strikes {100} only -> None
# 16. spread_cash: long 3.00, short 1.00, half 0.40 open -> 2.80; close long 0.30, short 0.10, half 0.40 -> 0.0
# 17. fees: 3 contracts open -> 2*3.00 + 2*0.13*3 = 6.78; 15 contracts open -> 2*10 + 2*0.13*15 = 23.90; close 3 -> 0.78
# 18. replay_spreads on synthetic contracts/bars: debit 2.80, equity 5000 -> n = floor(1000 / 280) = 3; a candidate with
#     no contracts gives its slot to the next drop that day; a leg with no exit bar uses its next bar
```

`put_contracts` and `option_bars` are passed into `replay_spreads` as arguments (`contracts_of`, `bars_of` callables) so
assert 18 runs on in-memory fakes.

- [ ] **Step 2: Run it to see it fail**

Run: `uv run python scripts/earnings/short_side_spike.py check`
Expected: FAIL on assert 15.

- [ ] **Step 3: Implement the functions and `test` in `main()`**

`test` order, verbatim from the spec: guard file exists -> exit; `chosen = report_dev()`, None -> exit "held-out stays
unused"; `probe()` False -> print "step 2 impossible with this data", exit **without** writing the guard; write the
guard `MAIN/cache/earnings_short_side/test.done` with the chosen cell; step 1 on the held-out window (chosen cell only,
SHV on the same window): pass = total > 0 and > SHV; pass -> step 2 from `STEP2_START`: pass = total > 0; print
`GO` only when both pass, with mean debit and the skip counts by cause.

- [ ] **Step 4: Run the self-check and lint**

Run: `uv run python scripts/earnings/short_side_spike.py check && uv run ruff check scripts/earnings && uv run ruff
format --check scripts/earnings`
Expected: `self-check OK`, ruff exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/earnings/short_side_spike.py
git commit -m "research: earnings short-side spike, real put spreads, probe and test mode"
```

---

### Task 4: Dev run, record, one held-out run

**Files:**
- Create: `docs/superpowers/replays/2026-11-23-earnings-short-side.md`

- [ ] **Step 1: Run dev**

Run: `uv run python scripts/earnings/short_side_spike.py dev > logs/earnings_short_side_dev.txt 2>&1; tail -60
logs/earnings_short_side_dev.txt` (the first run walks SEC for ~2,000 symbols: allow it to run in the background).
Expected: the event counts, 6 cell rows, the reference rows, and a chosen cell or `NO`.

- [ ] **Step 2: Write the record's dev section and commit it before any held-out run**

The record: question, the fixed bar, the data gaps (survivorship, today's ticker -> CIK map, `no cik` count), the dev
tables (gate, every cell at 80 and 10 bps, per year, references, in/out of fall), the verdict.

```bash
git add docs/superpowers/replays/2026-11-23-earnings-short-side.md
git commit -m "research: earnings short-side spike, dev result (before the held-out run)"
```

- [ ] **Step 3: Stop and ask the user before the held-out run**

The held-out window is scored once and cannot be re-run. Show the dev verdict; run `test` only on the user's go.

- [ ] **Step 4: Run test once, add the held-out section, commit**

Run: `uv run python scripts/earnings/short_side_spike.py test > logs/earnings_short_side_test.txt 2>&1`
Expected: the probe result, then the held-out step 1 and (if it passes) step 2, and `GO` or `NO`.

```bash
git add docs/superpowers/replays/2026-11-23-earnings-short-side.md
git commit -m "research: earnings short-side spike, held-out result"
```
