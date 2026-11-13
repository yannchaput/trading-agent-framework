# bull_bear judge: strong and weak grounds for an out (replay on the COVID run)

Run replayed: `logs/bull_bear/backtesting/2026-11-09_224910_backtesting`, judge only, arms old-1, old-2, new. Script: `scripts/replays/bull_bear_judge_vetoes.py` (the rule is in its docstring).
Reviews scored (all three arms recorded): 73 of 73. Failed: old-1 0; old-2 0; new 0.

## Verdict: DROP

| Gate | Group | new | old-2 | Pass |
|---|---|---|---|---|
| 1. Weak grounds: out rate at most 2/3 of the rerun's | 87 stocks | 90% | 91% | False |
| 2. Strong grounds: out rate within 10 points of the rerun's | 288 stocks | 95% | 95% | True |
| 3. No harm: book against old-1, 20 sessions, mean per review | 73 reviews | +0.02% | +0.04% | True |

## Not gated

| | new | old-2 (rerun) |
|---|---|---|
| Book minus old-1 book, 5 sessions | +0.04% (t 0.36) | +0.01% (t 0.08) |
| Book minus old-1 book, 20 sessions | +0.02% (t 0.13) | +0.04% (t 0.41) |
| Verdicts that differ from old-1 | 78 of 1213 (6.4%) | 72 of 1213 (5.9%) |
| Stocks in per review | 9.8 | 9.7 |

- old-1 against the logged run (other quantisation, memory tools): 75 of 1213 (6.2%) verdicts differ.
- Stocks new puts in and old-1 out (20 sessions, against the review's mean): n=41, +0.93%.
- Stocks new puts out and old-1 in: n=37, +1.27%.
- Seconds per review: old-1 101, old-2 102, new 103.
