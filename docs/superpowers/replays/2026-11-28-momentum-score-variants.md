# Momentum score variants (TODO Bull Bear item 6)

Spec: `docs/superpowers/specs/2026-11-28-momentum-score-variants-design.md`

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

## Full run log

```
self-check ok

### Development, after today

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | ranked | turnover |
|---|---|---|---|---|---|---|---|
| today | -2.2% | 0.12 | -78.9% | -71.9% | -36.8% | 607 | 18.4% |

### Development, after vol_scaled

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | ranked | turnover |
|---|---|---|---|---|---|---|---|
| today | -2.2% | 0.12 | -78.9% | -71.9% | -36.8% | 607 | 18.4% |
| vol_scaled | 0.9% | 0.18 | -64.3% | -64.3% | -25.9% | 606 | 23.8% |

### Development, after residual

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | ranked | turnover |
|---|---|---|---|---|---|---|---|
| today | -2.2% | 0.12 | -78.9% | -71.9% | -36.8% | 607 | 18.4% |
| vol_scaled | 0.9% | 0.18 | -64.3% | -64.3% | -25.9% | 606 | 23.8% |
| residual | 5.6% | 0.35 | -54.0% | -54.0% | -15.7% | 611 | 52.7% |
| today from 2006-07 | -5.8% | 0.02 | -78.9% | -71.9% | -36.8% | 633 | 19.2% |

### Development, after high_52w

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | ranked | turnover |
|---|---|---|---|---|---|---|---|
| today | -2.2% | 0.12 | -78.9% | -71.9% | -36.8% | 607 | 18.4% |
| vol_scaled | 0.9% | 0.18 | -64.3% | -64.3% | -25.9% | 606 | 23.8% |
| residual | 5.6% | 0.35 | -54.0% | -54.0% | -15.7% | 611 | 52.7% |
| today from 2006-07 | -5.8% | 0.02 | -78.9% | -71.9% | -36.8% | 633 | 19.2% |
| high_52w | -8.9% | -0.32 | -76.4% | -64.9% | -18.4% | 606 | 77.0% |

### Development, after r3_skip

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | ranked | turnover |
|---|---|---|---|---|---|---|---|
| today | -2.2% | 0.12 | -78.9% | -71.9% | -36.8% | 607 | 18.4% |
| vol_scaled | 0.9% | 0.18 | -64.3% | -64.3% | -25.9% | 606 | 23.8% |
| residual | 5.6% | 0.35 | -54.0% | -54.0% | -15.7% | 611 | 52.7% |
| today from 2006-07 | -5.8% | 0.02 | -78.9% | -71.9% | -36.8% | 633 | 19.2% |
| high_52w | -8.9% | -0.32 | -76.4% | -64.9% | -18.4% | 606 | 77.0% |
| r3_skip | -1.8% | 0.13 | -76.9% | -69.9% | -35.2% | 607 | 18.9% |

Passed development: ['vol_scaled', 'residual', 'r3_skip']
Sanity: today dev CAGR -2.24%, stress test curve -2.24%

### Holdout 2016-2026

| Book | cagr | sharpe | max_dd | 2015-16 | 2018 Q4 | 2020 | 2022 | ranked | turnover |
|---|---|---|---|---|---|---|---|---|---|
| today | 22.8% | 0.69 | -47.9% | -33.0% | -44.0% | -47.9% | -29.2% | 1033 | 16.8% |
| vol_scaled | 24.1% | 0.78 | -39.8% | -24.9% | -32.5% | -39.8% | -24.9% | 1033 | 22.4% |
| residual | 9.0% | 0.48 | -39.1% | -14.2% | -24.6% | -39.1% | -17.8% | 988 | 54.4% |
| r3_skip | 19.5% | 0.63 | -48.8% | -32.9% | -43.6% | -47.2% | -28.6% | 1033 | 17.3% |
vol_scaled: KEPT
residual: FAILED holdout
high_52w: FAILED development
r3_skip: FAILED holdout
```
