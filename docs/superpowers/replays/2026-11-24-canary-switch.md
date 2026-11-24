# Canary switch on bull_bear (Keller BAA): replay record

Design: `docs/superpowers/specs/2026-11-24-canary-switch-spike-design.md`. Script:
`scripts/regime/canary_switch_spike.py` (`run`, 2026-11-24). Raw output: `logs/canary_switch_run.txt` (git-ignored).

## Question

One account that runs bull_bear while Keller's canary (SPY, VWO, VEA, BND, fast momentum 13612W, monthly) is bullish
and holds the BAA defensive basket (top 3 of TIP, DBC, BIL, IEF, TLT, LQD, BND by slow momentum, BIL as the floor)
when any canary is not. Does it cut bull_bear's falls without giving up more than 15% of its gain?

## The bar (fixed before the run)

On both bull_bear windows: a smaller max drawdown than bull_bear alone, and a gain of at least 85% of bull_bear's; and
the defensive basket's compounded return over all bear-mode sessions of the SPY check (2008-2026) at least 0. GO needs
all three. A one-session delay and 10 bps (bull_bear side) + 5 bps (ETF side) per switch were charged.

## Result

| Window | Account | Gain | Max drawdown | Bear-mode sessions |
|---|---|---|---|---|
| 5Y (2021-09-20 to 2026-10-05) | bull_bear alone | +468.5% | -25.2% | |
| | **canary switch, monthly (the bar)** | **+112.8%** | **-25.2%** | 715 of 1,266 |
| | canary switch, Tuesdays (info) | +60.8% | -30.8% | 759 |
| | 70% bull_bear / 30% SHV | +273.5% | -18.2% | |
| | SPY | +90.9% | -24.5% | |
| COVID (2020-01-02 to 2021-05-26) | bull_bear alone | +136.9% | -37.7% | |
| | **canary switch, monthly (the bar)** | **+91.1%** | **-16.7%** | 204 of 353 |
| | canary switch, Tuesdays (info) | +131.7% | -16.7% | 155 |
| | 70% bull_bear / 30% SHV | +89.8% | -27.1% | |
| | SPY | +31.8% | -33.7% | |
| SPY check (2008-08-01 to 2026-11-23) | canary on SPY | +298.8% | -16.9% | 2,522 of 4,607 |
| | SPY | +697.8% | -47.2% | |
| | 60% SPY / 40% IEF | +360.2% | -28.2% | |

| Condition | Value | Result |
|---|---|---|
| 5Y: smaller drawdown | -25.2% vs -25.2% (equal, not smaller) | FAIL |
| 5Y: gain >= 85% of bull_bear's (+398.2%) | +112.8% | FAIL |
| COVID: smaller drawdown | -16.7% vs -37.7% | PASS |
| COVID: gain >= 85% of bull_bear's (+116.4%) | +91.1% | FAIL |
| Defensive basket's bear-mode return >= 0 (SPY check) | +47.9% (+3.0% since 2022-08) | PASS |

## Verdict: NO

The defensive side does what the user asked of it: it makes money in the canary's bear mode (+47.9% over 2008-2026,
and +3.0% since the paper's 2022 out-of-sample start). The switch fails because of how often it sits there and what
bull_bear earns meanwhile.

- **The canary says bear for 56% of the 5Y window** (715 of 1,266 sessions) and 58% of COVID. bull_bear earned most
  of its gains in those months: +17.0% in the 2021-10 to 2022-12 stretch (account +5.5%), +22.1% in 2024-02 to 04
  (account -1.0%), +31.2% in 2026-04 (account +2.5%), +24.6% in 2024-11.
- **It does not cut the worst fall of the 5Y window.** The -25.2% drawdown (2026-07, a momentum crash with the canaries
  still bullish) is the same: the canary uses index, foreign and bond momentum and cannot see it.
- **COVID shows the benefit** (drawdown -16.7% against -37.7%) but at too high a price: +91.1% against +136.9%, 66.5% of
  bull_bear's gain.
- **Information rows:** the Tuesday check fares worse on the 5Y (+60.8%, drawdown -30.8%) and keeps COVID's gain
  (+131.7%) with the same -16.7% drawdown. Neither is judged.

## Known gaps

- bull_bear's curves are flattered by the universe and the window end (the 2Y flaw check of 2026-11-01) and hold
  two index falls (2020, 2022); no held-out window exists. A NO here is not tuned to these curves: the rule is the
  paper's, scored once.
- The replay assumes bull_bear's book can be bought at once on switching back in (one session after the signal).
- A defensive basket that makes money (condition 3) does not by itself make the account worth switching: that needs
  the canary to be right about *when*, which it is not for a stock-momentum book.
