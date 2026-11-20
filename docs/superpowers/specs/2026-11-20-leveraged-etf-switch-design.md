# Leveraged ETF switch rules (UPRO, TQQQ), 1993-2026: research design

## Intent and why

The user wants to hold a 3x ETF (UPRO or TQQQ) when odds are good and limit losses in bear markets, without missing
the rebound. Every faster bear signal tried missed the -3% bar (`2026-11-19-fast-regime-research.md`,
`2026-11-20-breadth-agreement-screen.md`, `2026-11-20-hhmm-regime-paper.md`, `2026-11-20-ms-tvtp-exit.md`), and the
same late signal on SPXU lost -66% (`2026-11-19-bear-call-spread-research.md`). So this study drops the inverse leg
and asks which rule for **how much 3x to hold** hurts least: cash (T-bills) when bearish, never SPXU or SQQQ.

## Goals, non-goals, anti-goals

- **Goal:** compare three rules fixed in advance against three baselines on UPRO and TQQQ, over a window that holds
  long bear markets (2000-2002, 2008), and say which, if any, beats the fixed mix on return per unit of drawdown.
- **Non-goals:** inverse ETFs or shorts (already NO-GO); intraday timing; changing the production regime; a strategy
  in `strategies/` (only if a rule passes, as a later design).
- **Anti-goals:** tuning a window, threshold or target after seeing results; judging on 2016-2026 alone (no long bear
  market in it, and it is already seen).

## Approaches (picked by the user, 2026-11-20)

From the options laid out in the conversation: (1) drop the inverse leg, adopted for every rule; (2) graded exposure
on today's regime; (3) volatility-scaled leverage; (4) 200-day trend filter. The user chose to test 2, 3 and 4 on
both UPRO and TQQQ. Option 1 (inverse leg) is not tested again.

## Data (Yahoo daily closes)

- Underlyings: `^SP500TR` (S&P 500 total return, from 1988) for UPRO; `^NDX` (Nasdaq-100 price, from 1985; dividends
  ignored, about 1%/y, a handicap for the 1x legs only) for TQQQ. Cash: `^IRX` (13-week T-bill) / 252 per session.
- Synthetic 3x, each session: `3 r - 2 rf - ER / 252`, with ER 0.91% (UPRO) and 0.84% (TQQQ); financing at the
  T-bill rate, no extra spread. Validation, reported and not used to adjust anything: CAGR and correlation of the
  synthetic series against real UPRO (from 2009-06-25) and TQQQ (from 2010-02-11).
- 1x legs: the underlying itself (`^SP500TR`, `^NDX`), no fee.
- Regime inputs: `^GSPC` (stands in for SPY: same index, no dividend), `^VIX` (from 1990), `^VIX3M` (from 2006-07).

## Window

1993-01-04 to the last session downloaded, every rule and baseline on the same sessions. Reported: full window,
1993-2015 (unseen by earlier studies) and 2016-2026 (seen). Indicator warm-up uses data before 1993.

## Rules (fixed; nothing tuned)

Each rule gives a target per session: weights in {3x ETF, 1x underlying, T-bills} that sum to 1.

- **R2 graded regime:** today's regime (`core/regime.py` `regime_by_session`, default `RegimeParameters`) on `^GSPC`,
  `^VIX`, `^VIX3M`, for both UPRO and TQQQ. Bullish -> 100% 3x; neutral -> 100% 1x; bearish -> 100% T-bills.
  Before VIX3M exists, `inverted` counts as false (VIX3M passed as 1e9: `regime_path` rejects non-finite closes; my
  call: a handicap that makes the regime slower before 2006, stated in the record).
- **R3 volatility-scaled:** weight in 3x = `min(1, 0.30 / (3 x sigma20))`, rest in T-bills, where `sigma20` is the
  annualised standard deviation of the underlying's last 20 daily returns (`sqrt(252)`). Full 3x only when the
  underlying runs at 10% volatility or less. The target 30% is fixed here, not varied.
- **R4 200-day trend** (Gayed and Bilello, "Leverage for the Long Run", 2016): 100% 3x when the underlying's close is
  above its 200-session simple average, else 100% T-bills (UPRO on `^GSPC`, TQQQ on `^NDX`).

Baselines: **B1** 3x buy-and-hold; **B2** 1x buy-and-hold; **B3** fixed mix 1/3 3x + 2/3 1x, rebalanced on the first
session of each month.

## Execution and costs

- The signal uses closes up to session t; the position changes at the close of t+1 (one full session of lag,
  conservative against a live order near the close).
- Rebalance only when a weight is 10 points or more off its target (R3 only; R2, R4 and B3 move in whole steps).
- Cost 10 bps on the traded fraction of the account (one way), on every change.

## Reported, per instrument and per window

Final multiple, CAGR, max drawdown, Calmar (CAGR / max drawdown), worst calendar year, losing years, share of
sessions with some 3x, number of changes, and **rebound capture**: for each fall of 15% or more in the underlying
(zigzag on closes, same rule as `core/regime_scorecard.zigzag`), the rule's return over the 63 sessions after the
trough, next to B1's over the same sessions; reported as the median of each over falls (no ratio: B1 can be down
over those 63 sessions in a long bear market).

## Bar (fixed before the run)

A rule **passes** for an instrument if, on the full window **and** on each sub-window, its Calmar beats B3's, and its
worst calendar year is better than B1's. Rebound capture is reported, not part of the bar. The run happens once; no
rule, window, target or threshold changes after it.

## Constraints and what's left to the builder

- Research only: one script under `scripts/research/`, not committed (record kept, as before); the production regime
  code is imported, not changed. Network access to Yahoo only.
- A small self-check in the script (synthetic 3x formula on a hand case; one-session lag of positions; Calmar on a
  toy series). The repository suite is unaffected.
- Output: a printed table per instrument and window, saved under `logs/research/leveraged_switch/`, and a record in
  `docs/superpowers/replays/`.
