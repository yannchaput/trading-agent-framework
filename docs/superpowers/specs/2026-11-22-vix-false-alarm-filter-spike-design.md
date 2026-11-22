# Spike: filters to cut VIX alone's false alarms

Date: 2026-11-22. Status: **dev PASS for (b) SPY confirmation; 2008-2015 test pending**. Type: research spike (one throwaway script, no production code).
Branch: `research/vix-exit-rule`.

## Intent

VIX alone (VIX/VIX3M >= 0.95 or VIXY +10% over 5 sessions; calm exit after 3 sessions with the ratio < 0.90) switches
fast but raises 38 false alarms on dev 2016-2024 (4.2 a year, -0.58 log return against SHV). Recon (dev, scratch):
the ratio-only leg fired 12 false and 2 real switches, VIXY only 12 false and 8 real, both together 14 false and 2
real; in 26 of 38 false alarms the ratio never reached 1.0 (4 of 12 real). The filters tried so far waited longer
(2-3 sessions, jump model) and paid in speed. This spike tests one threshold change and three filters that add other
information, each on the entry only, judged on return.

## Goals, non-goals, anti-goals

- Goal: an entry with more total return than VIX alone, for both the 100% SHV and the 50% SPY + 50% SHV defensive mix.
- Non-goals: no change to the calm exit; no strategy integration; no production code.
- Anti-goals: changing a threshold or adding a filter after seeing `dev`; running 2008-2015 more than once or for a
  failed filter; any use of 2025-2026 (spent).
- Known bias: (a) comes from the dev recon above, so dev is not clean evidence for it; 2008-2015 is.

## Variants (fixed before running)

Row t uses closes up to t-1. `T` = VIX alone's trigger (`entry_trigger` in `vix_exit_spike.py`, VIXY NaN before 2011
= false). Each variant replaces `T` in `run_switch(None, trigger, calm)`; the calm exit is unchanged.

| Variant | Trigger |
|---|---|
| VIX alone (baseline) | `T` |
| a. ratio threshold 1.0 | VIX/VIX3M >= **1.0** or VIXY +10% over 5 sessions (my call instead of dropping the ratio leg: VIXY is missing before 2011, so the test window would have no entry until then) |
| b. SPY confirmation | `T` and SPY close <= 0.98 x its highest close of the last 20 sessions |
| c. credit confirmation | `T` and HYG / IEF (closes) below its 20-session simple average |
| d. sector breadth | `T` and at most 4 of the 9 original SPDR sectors (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) close above their 50-session average (my call: the point-in-time stock universe starts in 2016, too late for the test window) |

## Evaluation

- Defensive mixes: w = 0 (100% SHV) and w = 0.5 (50% SPY + 50% SHV); normal = 100% SPY. Portfolio: `portfolio_exposure`
  (`vix_staged_spike.py`), 10 bp per full switch.
- Per variant and mix: total return, CAGR, max DD, Sharpe, switches per year; per variant: false alarms per year, median
  drawdown at the switch and missed falls (`evaluate()` of `leaders_crack_spike.py`).
- `dev` must reproduce VIX alone: +118.8% (w = 0) and +180.4% (w = 0.5).
- **Development pass bar** (dev 2016-01-04 to 2024-12-31): total return above VIX alone at **both** mixes, and max DD no
  worse than buy and hold's (-33.7%). Chosen: the passing variant with the highest total at w = 0.5 (the user's
  posture). None pass: verdict NO, the test window stays unused.
- **Test**: 2008-01-02 to 2015-12-31, once, chosen variant against VIX alone at both mixes, nothing changed; same bar.

## Constraints

- Code: `scripts/regime/vix_filter_spike.py`, modes `check` and `dev` (`test` only if a variant passes); it imports from
  `vix_staged_spike.py`. Extra Yahoo closes: HYG, IEF, the 9 sector ETFs, from 2007-06. No new dependency.

## Left to the builder

- Output layout.

## Dev results (2026-11-22): (b) passes, (a), (c), (d) fail

`uv run --with jumpmodels python scripts/regime/vix_filter_spike.py dev`: 2016-01-04 to 2024-12-31, 2,264 sessions, 10
SPY falls >= 8%. VIX alone reproduces +118.8% (100% SHV) and +180.4% (50/50).

| Variant | Mix | Total | CAGR | Max DD | Sharpe | Switches/yr | False/yr | Median DD at switch | Missed |
|---|---|---|---|---|---|---|---|---|---|
| VIX alone | 100% SHV | +118.8% | +9.1% | -14.2% | 1.04 | 11.1 | 4.2 | -1.6% | 0 |
| VIX alone | 50/50 | +180.4% | +12.1% | -17.9% | 1.04 | 11.1 | 4.2 | -1.6% | 0 |
| a. ratio threshold 1.0 | 100% SHV | +95.1% | +7.7% | -14.2% | 0.86 | 10.2 | 3.8 | -1.7% | 0 |
| a. ratio threshold 1.0 | 50/50 | +164.5% | +11.4% | -17.9% | 0.97 | 10.2 | 3.8 | -1.7% | 0 |
| **b. SPY 2% under 20-day high** | 100% SHV | **+138.7%** | +10.2% | -14.2% | 1.11 | 8.2 | 2.8 | -3.6% | 0 |
| **b. SPY 2% under 20-day high** | 50/50 | **+192.6%** | +12.7% | -17.9% | 1.07 | 8.2 | 2.8 | -3.6% | 0 |
| c. credit HYG/IEF weak | 100% SHV | +76.3% | +6.5% | -19.3% | 0.71 | 8.7 | 3.3 | -4.3% | 1 |
| c. credit HYG/IEF weak | 50/50 | +151.3% | +10.8% | -20.2% | 0.91 | 8.7 | 3.3 | -4.3% | 1 |
| d. sector breadth <= 4 of 9 | 100% SHV | +65.4% | +5.8% | -19.0% | 0.61 | 7.3 | 2.8 | -6.6% | 2 |
| d. sector breadth <= 4 of 9 | 50/50 | +143.0% | +10.4% | -19.5% | 0.86 | 7.3 | 2.8 | -6.6% | 2 |

- **(b) passes** at both mixes with the same max DD as VIX alone and a higher Sharpe: false alarms 4.2 -> 2.8 a year,
  at the price of a later switch (median -3.6% from the high against -1.6%, outside the original "first 3%" goal).
- (a) cuts few false alarms (3.8 a year) and loses return; (c) and (d) switch late (-4.3%, -6.6%) and miss 1-2 falls.
- Chosen for the one-shot test: (b). `test` runs only VIX alone and (b) on 2008-2015 and writes
  `cache/regime/vix_filter_test.done` so it cannot run twice.
