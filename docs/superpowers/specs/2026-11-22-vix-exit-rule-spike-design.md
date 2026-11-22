# Spike: a fast exit after a VIX switch -- get back into SPY early in the rebound

Date: 2026-11-22. Status: **closed, NO on dev** (no exit variant passed; the 2008-2015 test window is unused). Type: research spike (one throwaway script, no production code).
Branch: `research/vix-exit-rule`.

## Intent

VIX alone (VIX/VIX3M >= 0.95 or VIXY +10% over 5 sessions) switches defensive fast: on time on all three 2025-2026
holdout falls (-2.0% to -2.6%). It loses money on the way back: on dev 2016-2024 it returned to SPY a median of about
28 sessions after the trough (7 to 62; the dev run below gives 32 over 9 falls), with SPY already +5% to +43% off the low and about half the fall recovered,
because its exit waits for VIX/VIX3M < 0.90, and the VIX level stays high until the rebound is well under way
(user's observation). This spike keeps the VIX entry and tests fast price / volatility exits.

Recon (dev only, scratchpad): about 40 false episodes, typically 3-10 sessions with SPY +1-3% meanwhile, so the exit
also shortens the false switches.

## Goals, non-goals, anti-goals

- Goal: an exit rule (plus the re-arm rule below) whose SPY / SHV portfolio beats VIX alone on money (user's choice a).
- Non-goals: no change to the VIX entry; no strategy integration; no production code; no reuse of `core/regime*.py`.
- Anti-goals: changing a parameter or adding a variant after seeing `dev`; running the 2008-2015 test more than once
  or for a rule that failed `dev`; any verdict from 2025-2026 (spent by `vix_jump_spike.py final`).

## Approaches considered

- **1. A small declared set of exit rules, run on dev, the best one tested once on 2008-2015 (chosen by the user).**
- 2. One rule fixed in advance: cleaner statistically, but a wrong guess ends the work with nothing learned.
- Dropped: a jump model or other fitted exit: persistence is built in, which is the slowness being removed.

## Rules (fixed before running)

Every value used before the open of session t comes from closes up to t-1. Prices: Yahoo, `auto_adjust=True`
(adjusted open, high, low, close).

**Entry trigger** (unchanged): `trigger_t` = VIX/VIX3M (t-1) >= 0.95, or VIXY close (t-1) / close (t-6) - 1 >= +10%.
VIXY starts 2011-01-04; before that its condition is false (ratio only), for every variant and the baseline alike.

**State machine** (all exit variants; state at the window start: normal, armed), each morning:

1. If normal:
   - switch defensive if `armed` and `trigger_t`;
   - else switch defensive if **failed rally**: `armed` is false and SPY close (t-1) < the lowest SPY close of the
     previous defensive episode (from the close before its first session up to its last session's close).
2. If defensive (since at least session t-1): back to normal if the variant's **price exit** fires, or the
   **VIX calm exit** fires, whichever comes first. The calm exit is counted exactly as `run_switch()` in
   `scripts/news/early_regime_spike.py` (the baseline's): the counter resets on the switch session, then counts
   consecutive defensive sessions with VIX/VIX3M (t-1) < 0.90; at 3 the state goes normal.
3. `armed` is updated at the end of each morning, after steps 1 and 2 (so step 1 sees the previous morning's value):
   false after a price exit that morning; otherwise true if `trigger_t` is false; otherwise unchanged. The VIX calm
   exit does not disarm. A failed-rally switch starts a new episode like any other.

The re-arm rule exists because a price exit while VIX/VIX3M is still >= 0.95 would otherwise be undone the next morning.

**Price exits** (each a separate variant; values at t-1):

| Variant | Exit fires when |
|---|---|
| E1 price reclaim | SPY close > the highest of the 5 closes before it (t-6 .. t-2) |
| E2 rebound off the low | SPY close >= 1.04 x the lowest SPY close since the episode began (from the close before its first session) |
| E3 volatility compression | 5-session realised volatility < 20-session realised volatility (std of daily log returns), and SPY close > its 5-session simple average |
| E4 DMI (reference, user named ADX) | +DI(14) > -DI(14), Wilder smoothing on SPY high, low, close |

**Baseline**: VIX alone, unchanged (no price exit, no re-arm, no failed-rally rule).

## Evaluation

- **Portfolio**: SPY when normal, SHV when defensive; the switch trades at the open (the overnight move stays with the
  old position); 10 bp per switch. Same arithmetic as `portfolio()` in `scripts/regime/vix_jump_spike.py`.
- Reported per variant: total return, CAGR, max drawdown, Sharpe (no risk-free), Calmar, switches per year, share of
  sessions defensive; SPY buy and hold as a reference row.
- Diagnostics (no pass/fail): per SPY fall >= 8% (same zigzag), sessions from the trough to the return to SPY and SPY's
  gain from the trough at that point; the entry table from `evaluate()` (to see that the re-arm rule did not damage the
  entry); **whipsaws** = price exits followed by a new defensive switch within 20 sessions where SPY's close before
  the new switch session is lower than its close before the exit session.
- **Development pass bar** (dev 2016-2024, against VIX alone in the same run): Sharpe higher, total return higher, and
  max drawdown no worse than -20%. Among the variants that pass, the chosen one is the highest Sharpe (tie: higher
  Calmar). None pass: verdict NO, the test window stays unused.

## Windows

- Development: 2016-01-04 to 2024-12-31. Indicator warmup uses data before the window.
- Test: 2008-01-02 to 2015-12-31, run **once**, only for the chosen variant (with VIX alone and buy and hold as
  references), nothing changed. Pass: higher Sharpe and higher total return than VIX alone. Known limit: VIXY is missing
  until 2011, so the entry there runs on the ratio alone.
- 2025-2026: not used (spent).

## Constraints

- Code: `scripts/regime/vix_exit_spike.py`, modes `check` and `dev` (`test` was not built: no variant passed dev); it reuses `portfolio`
  from `vix_jump_spike.py` (`market` and `metrics` are its own: `open_close` starts in 2015, the bar needs numbers) and `falls`, `evaluate` from `leaders_crack_spike.py`. A `test` mode with a one-shot guard was planned but not built (no variant passed dev). Yahoo daily data from 2007-06. No new dependency.

## Left to the builder

- Output layout; how the state loop is coded; Wilder smoothing details (first value = simple mean of the first 14).

## Results and verdict (2026-11-22): NO, no variant passes dev

`uv run --with jumpmodels python scripts/regime/vix_exit_spike.py dev`: 2016-01-04 to 2024-12-31, 2,264 sessions, 10 SPY
falls >= 8%. The VIX-alone row reproduces the jump-model spike exactly (+118.8%, Sharpe 1.04, max DD -14.2%).

| Variant (SPY / SHV, 10 bp per switch) | Total | CAGR | Max DD | Sharpe | Calmar | Switches/yr | Defensive | Pass bar |
|---|---|---|---|---|---|---|---|---|
| VIX alone | +118.8% | +9.1% | -14.2% | 1.04 | 0.64 | 11.1 | 40% | - |
| E1 price reclaim | +139.2% | +10.2% | -23.8% | 0.83 | 0.43 | 18.5 | 23% | FAIL |
| E2 rebound off the low (+4%) | +100.0% | +8.0% | -29.7% | 0.67 | 0.27 | 16.0 | 29% | FAIL |
| E3 volatility compression | +149.9% | +10.7% | -21.3% | 0.92 | 0.50 | 18.2 | 26% | FAIL |
| E4 DMI | +107.7% | +8.5% | -15.8% | 0.81 | 0.54 | 18.2 | 27% | FAIL |
| SPY buy and hold | +240.9% | +14.6% | -33.7% | 0.86 | 0.43 | - | - | - |

| Variant | Median sessions trough -> back to SPY (falls counted) | False switches/yr | Price exits | Whipsaws |
|---|---|---|---|---|
| VIX alone | 32 (7 to 62; 9 falls) | 4.2 | - | - |
| E1 | 5 (3 to 10; 8 falls) | 6.5 | 75 | 26 |
| E2 | 5 (2 to 7; 7 falls) | 5.4 | 42 | 21 |
| E3 | 6 (2 to 12; 9 falls) | 6.7 | 58 | 24 |
| E4 | 12 (6 to 23; 9 falls) | 6.7 | 69 | 32 |

- **The exits do what they were built for**: back in SPY a median of 5-6 sessions after the trough (E4, the DMI
  reference, 12) against 32 for VIX alone. The entry is untouched (median -1.6% at the switch, same as before).
- **It does not pay.** Faster returns come with 16-18.5 switches a year (VIX alone 11) and 4 in 10 price exits are
  followed within 20 sessions by a new switch at a lower SPY close (whipsaws); false switches rise from 4.2 to 5.4-6.7
  a year. E1 and E3 beat VIX alone on total return (+139% and +150% against +119%) but lose on Sharpe (0.83, 0.92 against
  1.04) and on drawdown (-23.8%, -21.3% against -14.2%); all four fail the bar, E1 and E3 on two of three counts.
- **ADX / DMI (E4)** was the slowest of the four back in, as expected, and has the most whipsaws (32 of 69).
- Not tested: other parameters (reclaim length, rebound size), combining an exit with a confirmation, or a longer
  re-entry that scales back in. The rules were fixed before the run and not changed after it.
- The 2008-2015 test window and the 2025-2026 holdout were not used.

The falls not counted in the table (E1 and E2: 2022-08; E2: 2020-02) are falls the variant had already exited before the trough, i.e. exits that were too early.
