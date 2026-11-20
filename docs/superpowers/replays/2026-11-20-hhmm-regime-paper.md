# Hierarchical HMM bull/bear regime (arXiv:2007.14874), 2016-2026: record and result

> Code not kept: the research script (`scripts/research/hhmm_regime.py`) and its output (`logs/regime/hhmm/`) were not committed; this record is what remains.

Asked by the user on 2026-11-20, after the indicator-based regime screens all failed
(`2026-11-19-fast-regime-research.md`, `2026-11-20-breadth-agreement-screen.md`): test the method of Oelschlaeger and
Adam, "Detecting bearish and bullish markets in financial time series using hierarchical hidden Markov models"
(arXiv:2007.14874, 2020), over 2016-2026.

## The paper

- Two-level HMM. 3 coarse states ("market moods"); each emits the mean log-return of a chunk of 30 trading days.
  Inside each coarse state, a 2-state fine HMM on that chunk's daily log-returns. Scaled t emissions at both scales.
- Maximum likelihood (forward algorithm), several random starts; coarse Viterbi decoding.
- Bullish = the coarse state with the lowest volatility and highest return; bearish = the highest volatility and lowest return.
- Applications: DAX 2000-2020 and S&P 500. S&P 500 stationary shares of the three coarse states: 0.530 / 0.371 / 0.099.
- No out-of-sample or real-time test: every bear period is decoded with the whole sample in view.

## Method (fixed before the run, nothing tuned)

- Data: `^GSPC` daily close log-returns from 2000-01-03 (Yahoo). N = 3, N* = 2, chunks of 30, t emissions at both
  scales, L-BFGS-B, 10 random starts on a cold fit. Coarse states ordered by marginal volatility (fine states at their
  stationary weights): lowest = bullish (1), middle = neutral (0), highest = bearish (-1).
- **paper**: one fit on 2000-01 to 2026-11, coarse Viterbi; a session takes its chunk's state. Look-ahead by
  construction (the upper bound the paper's figures show).
- **realtime**: refit at the first session of each month on returns strictly before it (warm start + 3 random
  starts, 131 fits); session `d` reads the forward-filtered coarse state of the last chunk of returns ending at `d - 1`
  (the chunk grid re-anchored every day so the last chunk is complete). No look-ahead.
- Scored with `core/regime_scorecard.score` on SPY closes, next to today's regime, with the same bar as the earlier
  screens: **mean fall at switch-on >= -3% with no more false-alarm bear days than today's** on the development window
  2016-01-04 to 2024-06-28. The held-out window (2024-07-01 onward) was scored in the same run, because the user asked
  for 2016-2026 and the method has no threshold to tune.
- Self-check: the nested likelihood equals a brute-force forward pass over (coarse, fine) pairs to 1e-8.
- A quick 2-start fit on 2000-2015 gave stationary shares 0.52 / 0.40 / 0.09, close to the paper's S&P 500 figures.

## Result (as printed)

```
development: 2016-01-04 to 2024-06-28 (2136 sessions)
row        mean fall false bear recall changes overstay  per-episode switch-on
today          -5.9%         77   0.63     118        8  -3.6% -7.9% -6.9% -7.6% +0.0% -8.2% -6.5% -5.0% -8.0% -5.6%
paper          -2.4%         16   0.73      12     44.5  +0.0% +0.0% -4.8% +0.0% +0.0% never +0.0% +0.0% +0.0% never
realtime      -13.0%         19   0.06      45        0  never never never -26.7% +0.0% never never -15.6% never never

held-out: 2024-07-01 to 2026-11-20 (602 sessions)
row        mean fall false bear recall changes overstay  per-episode switch-on
today          -5.4%          9   0.35      47      2.5  -8.4% -5.9% -1.5% -5.6%
paper          -4.4%          7   0.51       3        0  never +0.0% never -0.1%
realtime      -11.6%         30   0.00      23        0  never never never never

full: 2016-01-04 to 2026-11-20 (2738 sessions)
today          -5.8%         86   0.55     165      5.5
paper          -3.0%         23   0.67      15       14
realtime      -12.6%         49   0.04      68        0
```

Fitted coarse states (marginal, annualised):

| fit | bullish | neutral | bearish |
|---|---|---|---|
| paper (2000-2026, whole sample) | +17.5% / vol 9.9%, share 0.15 | +18.6% / vol 12.9%, share 0.56 | -25.0% / vol 30.7%, share 0.29 |
| last realtime (to 2026-10) | +16.6% / vol 11.3%, share 0.58 | -1.1% / vol 20.3%, share 0.35 | -41.2% / vol 46.6%, share 0.07 |

## Conclusion

NO. The paper row meets the bar (-2.4%, 16 false bear days) only through look-ahead: it is bearish on the peak day
itself in 7 of 10 falls, because Viterbi labels a 30-day chunk as a whole, so the chunk holding the peak is read
bearish from the crash later in it. In real time the method is far worse than today's regime: it switches on at a
mean of -13.0% (today -5.9%), recall 0.06, and misses 11 of the 14 falls since 2016. It turned bearish only at the
first COVID leg's trough (-26.7%), stayed bearish into the second leg (hence its +0.0%), and caught mid-2022 at -15.6%;
on the held-out window it was never bearish inside a fall.

Why: fitted on real history, the bearish state is a crash state (about 46% annual volatility, 7% of sessions), which
ordinary 8-20% corrections never reach; and a 30-day chunk mean lags by construction. The
whole-sample fit also landed on a different solution (bearish share 0.29), so the states depend on which crashes are
in the sample.

## Limits

- The coarse emission family, the chunk alignment and the number of starts are not stated in the paper's text; t,
  non-overlapping chunks and 10 / 3 starts were chosen before the run.
- Reading "neutral or bearish" as bearish was not run: choosing it after seeing these results would be tuning.
- `^GSPC` returns drive the model; SPY closes are scored (same sessions).
