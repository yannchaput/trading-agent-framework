# Market regime early detection: work list

Created 2026-11-22. Source: a Gemini Deep Research report (27 studies on regime detection and
forecasting), filtered to papers from 2024 onwards that fit our problem.

**Problem:** detect a regime shift at the very start (one day or less) so the daily trading
strategy can change. Context: US equities, daily bars, causal (no look-ahead) detection.

**Status:** none of these papers has been opened yet. Everything below comes from Gemini's summary
of each paper. Verify each claim against the paper before building on it.

**Already tested and rejected** (see memory notes): HHMM (arXiv 2007.14874, misses 11/14 falls,
needs look-ahead), MS-TVTP weekly exit (~6 weeks after the trough), the current SPY/VIX regime
(too late to gate leverage).

**Finding that shapes this list:** of the 15 papers from 2024 onwards, only one (Yi et al.)
reports a detection lag in days, and it is on Treasuries. No paper on the list shows
one-day-or-less detection on US equities. Faster detection also costs stability: the jump penalty
and label matching in the top papers exist to stop day-to-day flip-flopping, and they delay the
switch.

## Selected (3)

### 1. Shu, Yu & Mulvey (2024): Statistical Jump Model
- Title: *Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach*
- Link: https://arxiv.org/abs/2402.05272 (also Journal of Asset Management)
- Method: 2-state statistical jump model on a rolling 2,000-day window, re-estimated every six
  months, causal daily inference, t+1 forecast using state persistence.
- Why it fits: the only 2024+ paper that is a tradeable daily design; includes 10 bp one-way costs.
- Check: how many days the persistence penalty adds to the switch; whether results are on US
  equity indices; features used.

### 2. Yi, Mehra, Chen & Cartlidge (2026): LLM proposals validated by econometric tests
- Title: *Enhancing Regime Shift Detection Using Unstructured Data: A Study on the Treasury Market*
- Link: https://arxiv.org/abs/2605.30363 (FinLLM@IJCAI)
- Method: an LLM proposes shifts from FOMC minutes; a bootstrap VAR test (14 rate/macro variables)
  and PCMCI validate them.
- Reported: F1 0.82, F2 0.86, mean lag +0.9 day versus +8.2 to +14.8 days for BinSeg, PELT and
  Bai-Perron on 26 anchor points (2010-2024).
- Why it fits: the only paper that measures lag, and "LLM proposes, statistics confirm" matches our
  agent stack and the news and macro research tools.
- Caveats: Treasuries, not equities; anchored on FOMC minutes, so it fires on event days only; no
  chronological train/test split and no trading backtest.
- Check: could the same pattern run on news and earnings events for equities.

### 3. Boukardagha (2026): Wasserstein HMM
- Title: *Explainable Regime-Aware Investing*
- Link: https://arxiv.org/abs/2603.04441 (Columbia University)
- Method: strictly causal rolling HMM (estimated up to t-1), re-estimated daily, number of states
  chosen by one-step predictive likelihood, states matched across re-estimations by 2-Wasserstein
  distance to avoid label permutation.
- Why it fits: daily, causal, and fixes the label switching that breaks rolling HMMs.
- Caveat: the summary gives no results. Our HHMM test showed how a "causal" HMM can hide
  look-ahead, so audit the causality claim before trusting it.

## Theoretical (2)

### 4. arXiv 2610.20366 (2026): Regime detection in latent factor models
- Link: https://arxiv.org/pdf/2610.20366
- Method: joint filtering of latent continuous factors (regime-switching Heston and Vasicek
  models) with quickest detection by optimal stopping.
- Why it matters: it states our exact problem, minimum detection delay at a controlled false-alarm
  rate, and gives a framework for the trigger rule.
- Caveat: continuous-time theory; the paper explicitly leaves trading out of scope.

### 5. Li, Chen, Tao & Ji (2025): Sparse Jump Model with Model Predictive Control
- Title: *Regime-Switching Asset Allocation Using a Framework Combining a Jump Model and Model
  Predictive Control* (Mathematics)
- Link: https://www.mdpi.com/2227-7390/13/17/2837
- Method: sparse jump model with feature selection (AIC, BIC, FTIC) and a rolling multi-horizon
  forecast feeding model predictive control.
- Why it matters: it selects VIX, yield-curve and downside-risk features, close to what the
  framework already computes, and it controlled drawdowns in COVID-19.
- Caveat: an allocation paper, not a detector; compared against an equal-weight portfolio.

## Open points before building anything
- Three arXiv IDs have unusually high sequence numbers (2605.30363, 2610.20366, and 2610.27751
  elsewhere in the report): confirm they resolve to real papers.
- For each paper, record detection lag in days, false-alarm rate, and whether the test is
  out-of-sample on US equities.
- Any detector used to gate a strategy is a new design decision (see the market regime note in the
  root `CLAUDE.md`).
