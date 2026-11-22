# Spike: a fast exit after a VIX switch -- get back into SPY early in the rebound

Date: 2026-11-22. Status: **design, not run**. Type: research spike (one throwaway script, no production code).
Branch: `research/vix-exit-rule`.

## Intent

VIX alone (VIX/VIX3M >= 0.95 or VIXY +10% over 5 sessions) switches defensive fast: on time on all three 2025-2026
holdout falls (-2.0% to -2.6%). It loses money on the way back: on dev 2016-2024 it returned to SPY a median of about
28 sessions after the trough (7 to 62), with SPY already +5% to +43% off the low and about half the fall recovered,
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

- Code: `scripts/regime/vix_exit_spike.py`, modes `dev` and `test`; it reuses `open_close`, `portfolio`, `stats`
  from `vix_jump_spike.py` and `falls`, `evaluate` from `leaders_crack_spike.py`. `test` refuses to run twice: it writes
  `cache/regime/vix_exit_test.done` and exits if the file exists (my call). Yahoo daily data from 2007-06. No new dependency.

## Left to the builder

- Output layout; how the state loop is coded; Wilder smoothing details (first value = simple mean of the first 14).
