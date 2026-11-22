# Spike: filters to cut VIX alone's false alarms

Date: 2026-11-22. Status: **design, not run**. Type: research spike (one throwaway script, no production code).
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
