# Spike: staged exposure after a VIX switch -- cut the return cost of being defensive

Date: 2026-11-22. Status: **design, not run**. Type: research spike (one throwaway script, no production code).
Branch: `research/vix-exit-rule`.

## Intent

The user wants a defensive switch whose return gets as close as possible to SPY buy and hold (+240.9% on dev
2016-2024), accepting a lower Sharpe and a deeper drawdown than VIX alone (+118.8%, max DD -14.2%). Recon on dev (log
return vs SHV while defensive, VIX alone): **-0.80 lost after troughs** (slow return to SPY), **-0.56 lost on 38 false
switches**, -0.10 switch costs, **+1.02 saved during real falls**. Both leaks come from being 100% out; the fast
all-in exits of `2026-11-22-vix-exit-rule-spike-design.md` cut the after-trough leak but whipsawed. This spike tests
partial exposure: scaling back in by thirds (B) and only half out until the fall is confirmed (C), against a plain
fixed blend (A) that marks the line any smarter rule must beat.

## Goals, non-goals, anti-goals

- Goal: more total return than the A line at the same max drawdown, and more than VIX alone.
- Non-goals: no change to the VIX trigger itself; no strategy integration (partial exposure would mean strategies run
  at reduced size, a later design decision); no production code.
- Anti-goals: changing a tranche size, the 50% floor, the confirmation rule or a signal after seeing `dev`; running
  2008-2015 more than once or for a variant that failed `dev`; any use of 2025-2026 (spent).
- Known bias, stated up front: E1 and E3 were picked as re-entry signals because they had the best dev totals in the
  exit spike. Dev is therefore not clean evidence for B or C; the 2008-2015 test is.

## Approaches considered (user's choice: B and C, against A)

- **A. Fixed blend**: when VIX alone is defensive, keep a fixed share of SPY. Simple, exact, no improvement in the
  trade-off by construction; it is the reference line.
- **B. Staged re-entry**: out fully on the VIX trigger, back in by thirds.
- **C. Graded entry + staged re-entry**: half out on the trigger, fully out only once confirmed, then B's re-entry.

## Rules (fixed before running)

Values used before the open of session t come from closes up to t-1. Exposure `e_t` is the SPY weight (rest SHV),
set at the open. Inputs reused from `scripts/regime/vix_exit_spike.py`: `entry_trigger` (VIX/VIX3M >= 0.95 or VIXY
+10% over 5 sessions; VIXY NaN before 2011 = false), `calm` (VIX/VIX3M (t-1) < 0.90), `exit_signals` E1 (SPY close
above the highest of the 5 closes before it) and E3 (5-session realised vol < 20-session, and SPY close above its
5-session average), all lagged one session.

**A, fixed blend** (reference line): `e = 1 - w x defensive(VIX alone)` for w in 0, 0.25, 0.5, 0.75, 1 (w = 0 is
buy and hold, w = 1 is VIX alone). VIX alone's defensive flag is `run_switch(None, trigger, calm)`, unchanged.

**B, staged re-entry** (one state machine, each morning, in this order):

1. Normal (`e = 1`): on `trigger_t`, start an episode: all three tranches out (`e = 0`), `episode_low = spy_prev[t]`,
   calm counter reset. (No other way in: B leaves the defensive state only through the calm exit, so no re-arm rule.)
2. In an episode, from the morning after the switch session:
   - **new low**: if `spy_prev[t] < episode_low`, every returned tranche goes back out (`e = 0`) and
     `episode_low = spy_prev[t]`; otherwise
   - **tranches**: tranche 1 is back on any morning E1 is true while it is out, tranche 2 on any morning E3 is true
     while it is out (a level, not a crossing: the same rule the first time and after a new low);
   - **calm exit**, checked every morning whatever happened above: counter `+1` on each session with `calm`, reset to
     0 otherwise (counted like `run_switch`); at 3, all tranches return (`e = 1`) and the episode ends. Tranche 3
     returns only this way.
3. `e = (tranches back) / 3`.

With E1 and E3 never firing, B is exactly VIX alone.

**C, graded entry**: run B's machine; the episode is **confirmed** on the first morning (switch session included)
when either `trigger` was also true the previous morning (stressed 2 sessions in a row), or `spy_prev[t] <= 0.97 x`
the highest SPY close of the 60 sessions up to t-1. Once confirmed, it stays confirmed until the episode ends.
`e_C = e_B` when confirmed, else `max(e_B, 0.5)`.

**Portfolio** (all variants, A included): rebalanced to the target at the open. Daily return
`(1 + overnight) x (1 + intraday) x (1 - 0.0010 x |e_t - e_(t-1)|) - 1`, with overnight = old weights on SPY and SHV
close-to-open, intraday = new weights open-to-close; first session = new weights open-to-close. At e in {0, 1} this
is exactly `portfolio()` of `vix_jump_spike.py` (10 bp per full switch). Weight drift inside a day is ignored (my call).

## Evaluation

- Per variant: total return, CAGR, max drawdown, Sharpe, average exposure, turnover per year (sum of |change in e|).
- **A line**: the five A points as (max DD, total); A's total at a given max DD by straight-line interpolation between
  neighbouring points, once sorted by max DD. Max DD better than VIX alone's: compared with VIX alone's total. Worse than buy and hold's: fail.
- Diagnostic (no pass/fail): lost log return vs SHV, weighted by (1 - e), split into false / fall (20 sessions
  before a high to its trough) / after trough (from the session after a trough while e < 1 without a break), the split of the recon above (falls: SPY drops >= 8%, same zigzag).
- **Development pass bar** (dev 2016-01-04 to 2024-12-31): total above the A line at the variant's max DD, and above
  VIX alone. Both pass: the chosen one is the higher total. None pass: verdict NO, the test window stays unused.
- **Test** (2008-01-02 to 2015-12-31, once, chosen variant only, nothing changed): same bar against the A line and VIX
  alone on that window. A guard refuses a second run (`cache/regime/vix_staged_test.done`).

## Constraints

- Code: `scripts/regime/vix_staged_spike.py`, modes `check` (asserts on synthetic series), `dev`, `test`; it imports
  `market`, `entry_trigger`, `exit_signals`, `metrics`, `run_switch`, `falls` from `vix_exit_spike.py`. No new
  dependency. Run with `uv run --with jumpmodels`.
- `check` asserts the state machines on synthetic series and that the new portfolio function equals `portfolio()` at
  e in {0, 1}; `dev` must reproduce VIX alone's +118.8% before its variants are read.

## Left to the builder

- Output layout; how the state machine is coded.
