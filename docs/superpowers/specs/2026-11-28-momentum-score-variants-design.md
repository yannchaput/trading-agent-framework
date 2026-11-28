# Momentum score variants on the rank-only book (TODO Bull Bear item 6)

Date: 2026-11-28. Status: design, rule fixed before any variant is run.

## Intent

Find out whether a different momentum score picks better stocks than today's
`S = 0.5 R(12-1) + 0.3 R(6-1) + 0.2 R(3)` (`strategies/common/scoring.py`). The score feeds both
cross_momentum and bull_bear's shortlist (top 15 plus holdings shown to the judge), so a better score
helps both. The 2026-11-28 overlay check showed the worst falls (2014-16 -64% with SPY +2.9%,
2021-02, 2026-07) are book-specific momentum crashes no market-level layer sees; only a change to which
stocks are in the book can reach them.

## Goal, non-goals

- Goal (user's choice A): **higher CAGR than today, Sharpe no lower.** Drawdown is reported, not judged.
- Non-goals: tuning weights or windows; combining variants (a combination chosen after seeing results is
  tuning); judge replays (rerun noise ~25 log points); changing production code in this spike.

## Variants (as published, no tuning)

All keep today's filters (price, dollar volume, `max_volatility` 0.80 on 20-day vol for new entries,
holdings exempt, `min_trading_days` 250) and today's ranking otherwise.

| Id | Score |
|---|---|
| `today` | 0.5 R(12-1) + 0.3 R(6-1) + 0.2 R(3) |
| `vol_scaled` | today's blend / annualized volatility over 126 sessions |
| `residual` | regress 36 non-overlapping 21-session returns on SPY's; score = sum of the residuals of months 2-12 / their standard deviation (Blitz, Huij and Martens 2011, market-only) |
| `high_52w` | last close / highest close of the last 252 sessions (George and Hwang 2004); ties broken by today's score |
| `r3_skip` | today's blend with R(3) skipping the last 21 sessions, like R(12-1) and R(6-1) |

`residual` needs 757 sessions of history: stocks with less are not ranked (the report prints the ranked count).

## Approaches

1. **Chosen: extend the stress bench.** Reuse `scripts/bull_bear/rank_only_stress.py`'s `simulate`,
   `load_data`, `rebalance_days` and point-in-time ranking: top 10 weekly, 9.8% each, rest in SHV, 5 bps.
   Isolates the score.
2. **Chosen as step 2, only for a variant that passes step 1's holdout:** bull_bear's real book rules
   (holdings kept to rank 35, 4-review cooldown after an exit), to check the gain survives them.
3. Dropped: judge-in-the-loop replay (GPU, noise too large for a score change).

## Windows and rule

- Development: 2005-01-03 to 2015-12-31. Holdout: 2016-01-04 to 2026-11-24, **run once, only for variants
  that pass development**. Each window starts a fresh book at 1.0.
- A variant passes a window when its CAGR is above `today`'s and its daily Sharpe (no risk-free rate,
  same for all) is at least `today`'s, on the same window.
- Kept only if it passes both windows. Several passers: report all; picking one is the user's call.
- Step 2 runs the passers and `today` under the real book rules on both windows; same rule.

## Output

- Script `scripts/bull_bear/score_variants.py` with `--self-check` (hand-worked asserts for each score).
- Report table per window: CAGR, Sharpe, max DD, worst fall in 2008-09 / 2014-16 / 2020 / 2022,
  turnover, mean ranked count. Record in `docs/superpowers/replays/`, a memory note, TODO item 6 updated.

## Known limits

- Candidate pool = today's listings (dead companies missing). They are mostly high-volatility crashers that
  raw momentum favours, so the bias likely flatters `today` against `vol_scaled` and `residual`.
- One market path per window; no significance test beyond the two-window rule.

## Left to the builder

Code layout inside the script, report formatting, how the cooldown is tracked in step 2.

## Results (2026-11-28, script `scripts/bull_bear/score_variants.py`)

Deviation fixed before any result: the cached bars start 2003-06, so `residual` (757 sessions) ranks nothing before
2006-06-02. It is judged against a fresh `today` book from 2006-07-01 to 2015-12-31; the other variants keep 2005.

Sanity: today dev CAGR -2.24%, stress test curve -2.24% (match).

### Development, after r3_skip

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | ranked | turnover |
|---|---|---|---|---|---|---|---|
| today | -2.2% | 0.12 | -78.9% | -71.9% | -36.8% | 607 | 18.4% |
| vol_scaled | 0.9% | 0.18 | -64.3% | -64.3% | -25.9% | 606 | 23.8% |
| residual | 5.6% | 0.35 | -54.0% | -54.0% | -15.7% | 611 | 52.7% |
| today from 2006-07 | -5.8% | 0.02 | -78.9% | -71.9% | -36.8% | 633 | 19.2% |
| high_52w | -8.9% | -0.32 | -76.4% | -64.9% | -18.4% | 606 | 77.0% |
| r3_skip | -1.8% | 0.13 | -76.9% | -69.9% | -35.2% | 607 | 18.9% |

### Holdout 2016-2026

| Book | cagr | sharpe | max_dd | 2015-16 | 2018 Q4 | 2020 | 2022 | ranked | turnover |
|---|---|---|---|---|---|---|---|---|---|
| today | 22.8% | 0.69 | -47.9% | -33.0% | -44.0% | -47.9% | -29.2% | 1033 | 16.8% |
| vol_scaled | 24.1% | 0.78 | -39.8% | -24.9% | -32.5% | -39.8% | -24.9% | 1033 | 22.4% |
| residual | 9.0% | 0.48 | -39.1% | -14.2% | -24.6% | -39.1% | -17.8% | 988 | 54.4% |
| r3_skip | 19.5% | 0.63 | -48.8% | -32.9% | -43.6% | -47.2% | -28.6% | 1033 | 17.3% |

Verdicts: `vol_scaled: KEPT`, `residual: FAILED holdout`, `high_52w: FAILED development`, `r3_skip: FAILED holdout`.

Caveats: `vol_scaled`'s holdout margin is small (CAGR +1.3 points, Sharpe +0.09) on one market path, with more
turnover (22% vs 17% of the book a week, already charged at 5 bps); no significance test; today's-listing candidate
pool (missing dead companies likely flatter `today`); no judge, no real book rules yet (step 2).

## Step 2: bull_bear's real book rules (2026-11-28, `--book-rules vol_scaled`)

### Real book rules, 2005-2015

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | ranked | turnover |
|---|---|---|---|---|---|---|---|
| today | -1.0% | 0.15 | -68.0% | -68.0% | -36.6% | 607 | 5.5% |
| vol_scaled | -1.2% | 0.11 | -66.4% | -66.3% | -26.8% | 606 | 7.5% |

### Real book rules, 2016-2026

| Book | cagr | sharpe | max_dd | 2015-16 | 2018 Q4 | 2020 | 2022 | ranked | turnover |
|---|---|---|---|---|---|---|---|---|---|
| today | 27.7% | 0.79 | -50.0% | -34.1% | -39.8% | -50.0% | -30.3% | 1033 | 4.7% |
| vol_scaled | 28.8% | 0.88 | -42.7% | -22.6% | -31.2% | -42.7% | -26.9% | 1033 | 7.1% |
vol_scaled under the real book rules: FAILED

**Final verdict: no variant is kept.** `vol_scaled` passes 2016-2026 (+1.1 CAGR points, Sharpe 0.88 vs 0.79) but
fails 2005-2015 (CAGR -1.2% vs -1.0%, Sharpe 0.11 vs 0.15). Today's score stays. Under both books it had a smaller
drawdown than today's on every window; the user's goal (choice A) is return, so drawdown does not decide.
