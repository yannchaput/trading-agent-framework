# MS-TVTP bull/bear exit timing (Haase & Neuenkirch), 2016-2026: record and result

> Code not kept: the research script (`scripts/research/ms_tvtp_exit.py`) and its output (`logs/regime/ms_tvtp/`) were not committed; this record is what remains.

Asked by the user on 2026-11-20, after the hierarchical HMM test (`2026-11-20-hhmm-regime-paper.md`): can the model of
Haase and Neuenkirch, "Predictability of Bull and Bear Markets: A New Look at Forecasting Stock Market Regimes (and
Returns) in the US" (CESifo Working Paper 8828, 2021; International Journal of Forecasting 39(2), 2023), anticipate a
change of regime?

## The paper, and why only the exit was tested

- Weekly S&P 500 log-returns, 2-state Markov-switching model, time-varying transition probabilities through a logit
  of one lagged predictor. Model A adds the predictor and an AR(1) term to the mean; Model B (tested here) puts it in
  the switching equation only: `r_t = mu_S + u_t`, `p_ij,t = exp(v_ij + g_ij z_{t-1}) / (1 + exp(v_ij + g_ij z_{t-1}))`.
- Predictors: observable (lagged return, dividend-price ratio, VIX, term spread 10Y-3M, credit spread BAA-10Y, PMI,
  variance risk premium) or principal components of 115 macro-financial series (including Consensus Economics surveys
  and real-time vintages, not reproducible here). About 60 models pooled by simple average, inverse BIC rank or BMA.
- Out of sample 2004-10 to 2020-07, quarterly expanding refits, one-step-ahead bear probability from the Hamilton
  filter, bearish above 0.5; scored against Lunde-Timmermann dating (10% fall / 15% rise).
- **Entry is not anticipated, by the paper's own Table D1**: every bull-to-bear delay is 0 to +9 weeks after the
  dated start, equal to the constant-transition benchmark (TCTP) in 6 of 7 events; the authors write "The delay of the
  change into a bear market, however, is comparable to the benchmark." The S&P 500 at the best model's entry signal,
  measured from `^GSPC` weekly closes (peak approximated as the high of the 2 weeks before the dated start): -1.9%
  (2007), -6.7% (2010), -1.5% (2011, mid-bounce), -6.6% (2015), -5.6% (2018), -12.2% (2020).
- The claimed gain is the **exit**: bear-to-bull delays of +1 to +3 weeks for the "Best" model, against +6 to +41
  for TCTP. So the user asked to test the exit.

## Method (fixed before the run, nothing tuned)

- Weekly (Friday) `^GSPC` log-returns in percent from 1990-02. Model B, one predictor per model: lagged weekly return,
  VIX, T10Y3M (FRED, lagged 1 session), BAA10Y (FRED, lagged 2 sessions: it publishes about 2 days late), variance risk
  premium `VIX^2 / 12 - sum of the last 22 squared daily returns`. Dividend-price ratio and PMI dropped (no free
  history at hand). Each predictor standardised with its training window's mean and sd.
- Maximum likelihood (L-BFGS-B; 5 random starts cold, warm start + 2 random after; the paper uses EM on the same likelihood),
  expanding window, refit every 13 weeks; between refits the Hamilton filter. Bear state = the higher-sigma state.
  Pooled = simple average of the 5 models (the paper's B-OBS-AVE, minus two predictors); bearish if above 0.5.
  Benchmark: the same model with constant transition probabilities (TCTP).
- The forecast made with Friday's close labels the sessions of the following week (no look-ahead).
- Exit bar on the development window 2016-01-04 to 2024-06-28: the pooled model misses less of the rebound than
  today's regime (mean SPY move from the 8% fall's trough to the first non-bearish session, over the falls bearish at
  the trough), with no more false-alarm bear days and bearish at the trough in at least as many falls. Secondary: SPY
  (adjusted) on non-bearish sessions, T-bills (`^IRX`) on bearish ones, 10 bps per switch.
- Self-check: the filter's likelihood equals a direct product of 2x2 transition matrices; the weekly-to-daily label
  shift is asserted.

## Replication check on the paper's window (2004-10 to 2020-07)

| Bear market (LT dates) | Paper exit Best / Worst / TCTP (weeks) | Ours: pooled / TCTP |
|---|---|---|
| GFC II (to 2009-03-06) | +1 / +44 / +41 | +26 / +41 |
| Flash crash (to 2010-07-02) | +3 / +14 / +14 | +11 / +15 |
| Debt crisis (to 2011-08-19) | +3 / +23 / +23 | +21 / +23 |
| China (to 2016-02-12) | +1 / +5 / +6 | +2 / +5 |
| Q4 2018 (to 2018-12-21) | +3 / +7 / +9 | +5 / +7 |

Our TCTP matches the paper's TCTP within 0-2 weeks; our pooled model sits between the paper's best and worst. Entries
were within 1 week of the paper's best, except 2011 (+12 against +9).

## Result (as printed)

Development window 2016-01-04 to 2024-06-28 (10 falls):

```
row        mean fall false bear recall changes overstay  per-episode switch-on
today          -5.9%         77   0.63     118        8  -3.6% -7.9% -6.9% -7.6% +0.0% -8.2% -6.5% -5.0% -8.0% -5.6%
ms_tvtp        -6.0%        141   0.55      38     15.5  -4.6% never -6.1% -8.6% +0.0% -6.8% -7.9% +0.0% -6.3% never
ms_tctp        -5.2%        155   0.60      27       28  +0.0% never -6.1% -8.6% +0.0% never -7.9% +0.0% +0.0% never

row        bear in fall at trough rebound missed sessions  SPY/T-bill  max DD
today                10        10          +6.4%        9     +168.8%   18.9%
ms_tvtp               8         6         +19.2%     28.5     +123.5%   18.4%
ms_tctp               7         6         +21.2%       62     +132.9%   13.0%
ms_R                  8         7         +10.5%       21     +181.5%   18.1%
ms_VIX                9         9         +19.3%       23      +72.8%   22.1%
ms_TS                 7         6         +15.5%     30.5     +128.4%   13.0%
ms_CS                 7         6         +20.7%       32     +116.3%   17.3%
ms_VP                 8         8         +16.2%     23.5      +82.7%   19.4%
buy-hold                                                      +213.7%   33.7%
```

Held-out window 2024-07-01 to 2026-11-20 (4 falls; the last one, from 2026-10-06, is still open, so a row bearish at
its trough counts as missing nothing, which flatters today's regime there):

```
row        bear in fall at trough rebound missed sessions  SPY/T-bill  max DD
today                 4         4          +3.8%      3.5      +24.5%   10.7%
ms_tvtp               2         2         +17.0%     30.5       -7.2%   17.4%
ms_tctp               1         1         +21.4%       47       +1.0%   17.4%
buy-hold                                                       +29.9%   18.8%
```

**development check: FAIL** (rebound missed +19.2% against +6.4%; 141 false bear days against 77; bearish at the
trough in 6 falls against 10).

## Conclusion

NO. The pooled MS-TVTP model exits a median 28.5 sessions after the trough, against 9 for today's regime, and misses
+19.2% of the rebound against +6.4%. Every single-predictor model also exits later than today's regime; the closest,
lagged return, still misses +10.5%. The predictors do speed the exit relative to the same model without them (28.5
against 62 sessions), which is the paper's claim, but a weekly model that needs weeks of calm returns to switch back
loses to a daily rule that reads VIX. The paper's 1-3 week exits are Table D1's "Best" column: for each event, the
fastest of 30 combinations, a different one each time, which no one could pick in advance.

## Also reviewed, not run

Kraevskiy, Prokhorov and Sokolovskiy, "An early warning system for emerging markets" (arXiv 2404.03319, Emerging
Markets Review 2026): a sequential change-point alarm (Shiryaev-Roberts) on the conditional entropy of AR residuals
given covariates. It gives no direction and no bull/bear label; its real-data evidence is two hand-picked events
(Russia 2022, South Africa 2020) read after the fact with a threshold set knowing them; in simulation it detects a
change 5-15 observations after it happens. Not run: turning it into a bear label needs a threshold and a holding
period, two new choices to tune.

## Limits

- Dividend-price ratio and PMI are missing from the observable set (5 of the paper's 7 predictors), and the
  115-variable factor models (the paper's best) were not reproduced.
- Simple average only; the paper's inverse-rank and BMA pools were not run.
- One random seed; refits every 13 weeks as in the paper.
