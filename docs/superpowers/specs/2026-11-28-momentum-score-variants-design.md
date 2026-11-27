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
