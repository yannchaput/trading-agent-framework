# Intraday fall signal (VIXY / SPY, 10-minute loop): research spike design

## Intent and why

bull_bear loses money in market falls; the user wants a **separate account that makes money in those falls, profitable
on its own** (a smaller combined drawdown is a bonus, not success). The first attempt, an inverse ETF held open to close
on `in_market_fall` days, was NO at its sign gate (`docs/superpowers/replays/2026-11-23-market-fall-intraday.md`): the
daily signal marks volatile days, not down days. The user's conclusion: **the vehicle is not the problem, the signal is.**

This spike tests a **new intraday signal**, read every 10 minutes, that compares *today* against short windows: VIXY's
move against D-1's close (at the open) and against an hour ago (during the day), against D-1's average price, its 50-day
trend, its volume against the same time of day over D-1 to D-5, and a falling SPY as confirmation. When it fires, the
account holds an inverse S&P ETF until the signal fades (VIXY easing and SPY recovering), a volatility-sized trailing
stop, or 15:55. Idle money sits in SHV overnight.

Path, decided by the user: research -> framework backtest -> Alpaca paper -> Tastytrade margin (or cash) live.

## Goals

- One script, `scripts/regime/intraday_fall_signal_spike.py` (`uv run python`), modes `check` (assert self-check on
  synthetic data), `dev`, `test` (one shot, guard file), in the style of `scripts/regime/fall_intraday_spike.py`.
- Every entry / exit / stop / day-limit decision in **one pure function** with no data access, so a GO can move it
  unchanged into a framework `Strategy`.
- Report per cell and a pre-registered GO / NO-GO; a record in `docs/superpowers/replays/` at the end.

## Non-goals

- No production code, no `Strategy`, nothing under `src/` or `tests/`; no options, shorting or futures.
- No intraday VIX index (no history exists: Alpaca none, Yahoo 60 days); **VIXY is the proxy**. A live strategy may
  later read VIX itself; that is a separate decision.
- No overnight holding of the inverse ETF; no LLM, news or `in_market_fall` gate.
- No parameter search beyond the grid below.

## Anti-goals

- **Searching until something passes.** The 2026-11-20 intraday spike tried about 70 variants on 2016-2024 and its
  pick failed out of sample. Here the grid (16 cells), the bar and the choice rule are fixed in this document before any
  run, and the held-out window is scored once.
- **A flattering result:** costs on every fill, stops filled against the trader, no look-ahead (a decision at time t
  reads only bars closed by t and trades on the next minute's open).
- Network access in the self-check.

## Data

- Alpaca 1-minute bars, SIP feed, `Adjustment.ALL`, **with volume**, regular session (09:30-15:59 New York, early
  closes trimmed at 13:00 as in `fall_intraday_spike.py`), for **SPY, VIXY, SDS (-2x), SPXU (-3x)**; credentials from
  `env/.env.bull_bear.paper` (`ALPACA_DATA_API_*`, read-only). Cache `cache/intraday_fall_signal/<SYMBOL>_<YEAR>.parquet`
  (the existing `cache/fall_intraday/` files have no volume and are not reused).
- Yahoo daily closes: **VIXY** (for its SMA50, from 2015) and **SHV** open and close (overnight carry).
- 10-minute bars are built from the 1-minute bars: blocks 09:30-09:39, 09:40-09:49, ..., 15:50-15:59.
- **Development window: 2016-01-04 to 2024-06-28. Held-out window: 2024-07-01 to the last complete session**, scored
  once. (That window was used once, on 2026-11-20, for a UPRO-only *bullish* rule; no bearish intraday rule has been
  scored on it. The user accepts its reuse for this different question.)

## The decision loop (every 10 minutes)

Decision times t = 09:40, 09:50, ..., 15:40 (the close of each 10-minute block); an order decided at t fills at the
**open of the next 1-minute bar**. No entry after 15:40. Values at t use only bars closed by t:

- `vixy_day` = VIXY now / VIXY's D-1 close - 1 (at 09:40 this is mostly the overnight gap)
- `vixy_hour` = VIXY now / VIXY 60 minutes earlier - 1, the window clipped at today's 09:30 open (never across a night),
  used from 09:50 on
- `vixy_vs_d1avg` = VIXY now / D-1's VWAP (volume-weighted average price of D-1's session) - 1
- `vixy_trend` = VIXY now > its SMA50 (Yahoo daily closes through D-1)
- `rvol` = VIXY volume from 09:30 to t today / mean over D-1..D-5 of VIXY volume from 09:30 to the same clock time
  (sessions shorter than t, i.e. early closes, are left out of the mean)
- `spy_day` = SPY now / SPY's D-1 close - 1; `spy_hour` = SPY now / SPY 60 minutes earlier - 1 (clipped at the open)

### Entry (all must hold, and no position open, and no day limit hit)

1. **Fast trigger:** at **09:40 only**, `vixy_day >= V_DAY` (the opening jump against D-1's close); at every later
   decision time, `vixy_hour >= V_HOUR` only. (Decided by the user: the D-1 close comparison is an opening trigger, so
   entries after 09:40 need fresh VIXY momentum, not a day that is already up.)
2. **Context:** `vixy_vs_d1avg > 0` and `vixy_trend`
3. **Volume:** `rvol >= RVOL` (when the cell has RVOL off, skip)
4. **Movement:** `spy_day <= -S_CONF` and `spy_hour < 0`

Buy the cell's ETF with 100% of the account's equity (fractional shares, 6 decimals).

### Exit (first that applies)

1. **Trailing stop** (resting, checked on every 1-minute bar): `K x ATR` below the ETF's highest price since entry, ATR =
   mean true range of the ETF's 10-minute bars over D-1..D-5. Same fill rules as `fall_intraday_spike.py`: the level comes
   from earlier bars' highs (seeded with the entry fill), a bar opening at or under it fills at its open, else a bar
   trading down to it fills at the level. *Sized from the ETF's own volatility, so it widens on stressed days (the user
   asked for a stop that is not too close on SPXU).*
2. **Signal fades** (checked at each decision time at least 30 minutes after the entry fill): over the last 30 minutes
   VIXY is down **and** SPY is up (both, at the same check). The first possible fade check is therefore the first
   decision time 30 or more minutes after the fill, and its window never starts before the fill.
3. **Forced exit at 15:55** (open of the 15:55 1-minute bar; on early-close days, 12:55).

Re-entry the same day is allowed if the entry conditions hold again and no day limit is hit.

## Day limits (the user chose both)

1. **Day-trade cap (margin account under $25k):** every trade opens and closes the same day, so each is a day trade;
   at most **3 in any 5 consecutive sessions**. An entry that would be the 4th is skipped. Reported both **with the
   cap** (the 5,000 account: this is the verdict case) and **without** (an account over $25k, diagnostic).
2. **Daily loss limit:** after the day's closed P&L reaches **-2% of the day's opening equity**, or after **2 stopped
   trades** that day, no more entries that day.

## Overnight: SHV

The account holds **SHV from each session's close to the next session's open**, every night, whatever the day did
(bought at the close, sold at the open; Yahoo SHV close and open; 1 bp per side). Diagnostic row: the same strategy with
cash overnight, so the SHV leg's cost against its carry is visible (T-bill yields were near zero in 2016-2021). SHV is
never traded intraday, so it adds no day trades.

## Account and costs

Fake account **5,000** (the user's EUR 5,000 in USD, no FX). ETF costs **3 bps per side**; SHV 1 bp per side; commission
0. Margin account (the day-trade cap applies, settlement does not).

## Grid (fixed now: 16 cells)

- **Signal strength** {loose: `V_DAY` 3%, `V_HOUR` 2%; strict: `V_DAY` 5%, `V_HOUR` 3%}
- **Volume** {off, `RVOL` 1.5}
- **Instrument** {SDS, SPXU}
- **Stop** `K` in {3, 5}

Fixed: `S_CONF` 0.5%, the 30-minute fade window, the 60-minute hour window, the daily loss limit (-2% or 2 stops).

Reference rows (not candidates): **-1 x SPY** (the synthetic `2 - p / entry` inverse) under each signal setting,
**SHV only** = the strategy with no entries (SHV every night, cash during the day, so "beats SHV only" means the
trades made money after their costs), SPY buy and hold, and "half SPY + half the chosen cell" (hedge diagnostic).

## Bar (fixed now), dev window, with the day-trade cap

1. **Signal gate (per signal setting: strength x volume, 4 settings):** on each dev session, **one** -1 x SPY trade
   from the session's **first** entry signal to 15:55 (12:55 on early closes), no stop, no fade exit, no day limits, no
   cap (a position held to the forced exit blocks any later entry that day), must have a **positive mean trade after
   costs**. A setting that fails is
   out; if all 4 fail: **NO**, nothing else is judged.
2. **A cell passes** if its signal setting passed the gate and, with the cap: total return > the SHV-only row, at most
   2 losing calendar years (2016-2023 and 2024 to June), and **at least 30 trades**.
3. **Choice:** among passing cells, the one with the most passing neighbours (cells differing in one grid dimension);
   ties by higher dev total, then smaller max drawdown. If none passes: **NO**, the held-out window is not run.
4. **Held-out (run once, the chosen cell, with the cap): GO** if total return > 0 **and** > the SHV-only row on the
   same window. Without-cap and cash-overnight results are reported next to it.

Reported per cell: total, CAGR, per calendar year, max drawdown, trades, win rate, mean trade, trades by exit cause
(stop, fade, 15:55), trades skipped by the cap and by the daily loss limit, entry time-of-day histogram (by hour).

## Constraints

- Python 3.14, `uv`; no new dependency; read-only (Alpaca data, Yahoo); writes only its cache, `logs/` (git-ignored)
  and the record.
- Lint: `ruff check` and `ruff format --check` clean on the script **and on the Markdown files** (ruff formats their
  Python code blocks).
- Self-check covers at least: no look-ahead in every signal value, the opening `vixy_hour` clip, `rvol` same-clock-time
  comparison, the stop's own-bar rule and gap fill, the fade exit, the 15:55 / 12:55 forced exit, the 3-in-5 cap, the
  daily loss limit, and the SHV overnight leg.

## Approaches (decided with the user, 2026-11-23)

- **A. Research script, rules as one pure function (chosen).** Fast to iterate, little thrown away on a NO; the pure
  decision function lifts into a framework `Strategy` after a GO. Its fill model is simpler than `BacktestBroker`, so a
  GO still needs a framework backtest before paper.
- B. Framework strategy backtested directly: one code path to paper and real order handling, but slow per run (minute
  bars through the executor, 16 cells) and a whole strategy to delete on a NO.
- C. Rules module in `src/` with tests plus a research script: no drift, but the full quality gates for code that may be
  dropped; A's pure function gives most of the benefit.

## Decided with the user

Loop every 10 minutes (to adjust later); triggers VIXY vs D-1 close at the open only (09:40) and vs an hour ago
from 09:50 on; VIXY vs
D-1 average; SMA50 trend confirmation (SMA200 dropped: VIXY's roll decay makes it rare and late); VIXY volume; VIXY and
SPY movement as confirmation; exit on VIXY degrading and SPY recovering, with a stop not too close; both day limits;
SHV overnight only.

## My calls (correctable)

D-1 average = D-1 VWAP; volume compared at the same clock time; SPY confirmation as `spy_day <= -0.5%` and `spy_hour < 0`;
fade = both moves over 30 minutes; stop = K x 10-minute ATR over D-1..D-5; ETFs SDS and SPXU (SH dropped: its minute
data starts 2016-06-27 and -1x is covered by the -1 x SPY reference); daily loss limit -2% or 2 stops; SMA50 from Yahoo
daily closes; the held-out boundary 2024-07-01; the
signal gate takes each session's first signal only; "SHV only" = SHV overnight, cash in the day; the fade is checked
only from 30 minutes after the fill.

## Left to the builder

Script layout, cache format, how 10-minute and same-clock-time aggregates are computed (vectorised or looped), table
printing, the self-check's synthetic data.
