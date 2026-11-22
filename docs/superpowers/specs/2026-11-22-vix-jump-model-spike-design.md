# Spike: a jump model on VIX inputs -- keep VIX's speed, cut its false switches

Date: 2026-11-22. Status: **closed, NO at penalty 50** (see Results). Type: research spike (one throwaway script, no production code).

## Intent

On 2016-2024, VIX alone (VIX/VIX3M >= 0.95 or VIXY +10% over 5 sessions) already switches defensive at a median
-1.9% from the high of a SPY fall, inside the -3% goal, but it raises 4.2 false switches a year and is defensive 39%
of the time (`2026-11-22-leaders-crack-first-spike-design.md`). This spike tests whether a statistical jump model
(Shu, Yu & Mulvey 2024, arXiv 2402.05272), fed volatility inputs, keeps that speed while its jump penalty suppresses
the short-lived false switches.

## Goals, non-goals, anti-goals

- Goal: a daily `normal` / `defensive` verdict from a 2-state jump model on volatility features (variant B), with the
  paper's own model (variant C) as a reference, both compared with VIX alone.
- Non-goals: no strategy integration, no P&L, no penalty tuning, no combination with leaders or news.
- Anti-goals: changing the penalty or the features after seeing `dev`; touching the holdout unless B passes.

## Approaches considered

- A. VIX proposes, the paper's jump model confirms: the switch waits for the slower detector (the paper reports about
  half a month of latency at the COVID onset), so likely late.
- **B. Jump model on VIX inputs (chosen by the user).** Persistence is added to the fast inputs themselves.
- C. The paper's model as published: kept as a reference row.

## Model (fixed before running)

- Library: `jumpmodels` 0.1.1 (the authors' package), run with `uv run --with jumpmodels`, not a project dependency.
  `JumpModel(n_components=2, jump_penalty=50, cont=False)`; states from `predict_online` (causal, no revision).
- **Features B**: VIX / VIX3M; 5-session log change of VIX; SPY downside deviation (exponentially weighted mean of
  r^2 * 1{r<0}, half-life 10 sessions, square root).
- **Features C** (paper): SPY downside deviation (half-life 10); SPY Sortino ratio = EW mean return / EW downside
  deviation, half-lives 20 and 60. Plain SPY returns (no risk-free rate subtracted, a simplification).
- Preprocessing: clip each feature at 3 standard deviations, then standardise, both fitted on the training window only
  (`jumpmodels.preprocess` `DataClipperStd(mul=3)` and `StandardScalerPD`).
- **Training**: refit every 6 months (first session of January and July), on an expanding window from the first
  session with all features (VIX3M starts in December 2007) up to the refit date; the first refit (January 2016) has
  about 8 years.
- **Bear state**: after each fit, the state whose centre has the higher SPY downside deviation.
- **No look-ahead**: features are shifted by one session, so the state used before the open of session t comes from
  closes up to t-1. `defensive` = bear state; the model's persistence is the only exit rule.
- **Baseline**: VIX alone, unchanged (VIX/VIX3M >= 0.95 or VIXY +10% over 5 sessions; exit after 3 sessions with
  VIX/VIX3M < 0.90).

## Evaluation (same as the leaders spike)

- SPY falls of at least 8% (zigzag on closes) fully inside the window; per fall: early (switched up to 20 sessions
  before the high and still defensive at it), on time (after the high, -3% or better), late, missed.
- False switches per year: starts outside [20 sessions before a high, its trough]. Share of sessions defensive.
- **Development pass bar**: median drawdown at the switch better than -3% (missed = worst), and false switches per
  year **below** VIX alone's on the same window.

## Windows

- Development: 2016-01-04 to 2024-12-31.
- Holdout: 2025-01-02 to 2026-11-20, run once, only if B passes development, with nothing changed.

## Constraints

- Code: `scripts/regime/vix_jump_spike.py`; it reuses `evaluate` and the fall zigzag from
  `scripts/regime/leaders_crack_spike.py` / `scripts/news/early_regime_spike.py`. Yahoo daily closes from 2007-12.

## Results and verdict (2026-11-22): NO at penalty 50 (holdout not run)

`dev` 2016-01-04 to 2024-12-31, 10 SPY falls of at least 8%.

| Fall (high -> trough) | B: jump model, VIX inputs | C: paper's model | VIX alone |
|---|---|---|---|
| 2018-01 (-10%) | missed | missed | on time, -0.7% |
| 2018-09 (-19%) | late, -9.8% (55 sessions) | late, -7.0% (32) | on time, -1.5% |
| 2020-02 COVID (-27%) | late, -10.6% (12) | late, -18.9% (14) | early, 15 sessions |
| 2020-03 second leg (-17%) | early (still defensive) | early (still defensive) | already defensive since 2020-01-28 |
| 2020-09 (-9%) | missed | missed | early, 0 sessions |
| 2022-01 (-13%) | missed | late, -11.7% (36) | late, -5.4% |
| 2022-03 (-21%) | missed | missed | late, -4.7% |
| 2022-08 (-17%) | missed | missed | late, -3.8% |
| 2023-07 (-10%) | missed | missed | on time, -1.7% |
| 2024-07 (-8%) | missed | missed | on time, -2.2% |
| False switches | 0.2/year | 0.2/year | 4.2/year |
| Sessions defensive | 11% | 26% | 40% |

At a penalty of 50 the jump model is far too sticky: it removes the false switches but only enters the bear state in
the largest falls, late. Likely cause (not tested): the training window starts in 2008, so the clipped, standardised
features put the 2008-2009 crisis in the bear state and everything milder in the normal one. The penalty was fixed
before running and was not changed.

## Trade-off curve (dev exploration, declared; holdout still not run)

`uv run --with jumpmodels python scripts/regime/vix_jump_spike.py curve`, same 10 falls and evaluation.

| Variant | Early | On time | Late | Missed | Median at switch* | False/year | Defensive |
|---|---|---|---|---|---|---|---|
| VIX alone, 1 session | 3 | 4 | 3 | 0 | -1.6% | 4.2 | 40% |
| VIX alone, 2 sessions in a row | 2 | 4 | 4 | 0 | -2.2% | 2.8 | 35% |
| VIX alone, 3 sessions in a row | 2 | 0 | 8 | 0 | -5.0% | 2.0 | 32% |
| B, penalty 0 | 1 | 2 | 7 | 0 | -3.8% | 8.0 | 21% |
| B, penalty 5 | 1 | 0 | 6 | 3 | -8.7% | 0.8 | 19% |
| B, penalty 15 | 1 | 0 | 6 | 3 | -9.7% | 0.2 | 17% |
| B, penalty 35 | 2 | 0 | 2 | 6 | -100% | 0.1 | 17% |
| B, penalty 50 | 1 | 0 | 2 | 7 | -100% | 0.2 | 11% |

\* Over all 10 falls; early (including already defensive) = 0%, missed = -100%. Corrected run after the
2026-11-22 review fix in `evaluate` (falls already covered by an older switch were printed as missed).

No point beats VIX alone (1 session) on both speed and false switches. Each false switch removed per year costs
roughly 1.5 to 2 points of drawdown at the switch: in the first sessions a real fall and a blip look the same in
these inputs. Next step: price the trade-off with a SPY / SHV portfolio per variant.

## Cost comparison (dev, declared exploration; holdout still not run)

`uv run --with jumpmodels python scripts/regime/vix_jump_spike.py costs`: SPY when normal, SHV when defensive, the
switch trades at the open (overnight move stays with the old position), 10 bp per switch.

| Variant | Total | CAGR | Max DD | Sharpe | Calmar | Switches/year | Defensive |
|---|---|---|---|---|---|---|---|
| SPY buy and hold | +240.9% | +14.6% | -33.7% | 0.86 | 0.43 | - | - |
| VIX alone, 1 session | +118.8% | +9.1% | **-14.2%** | **1.04** | **0.64** | 11.1 | 40% |
| VIX alone, 2 sessions | +118.8% | +9.1% | -19.6% | 0.99 | 0.46 | 7.8 | 35% |
| VIX alone, 3 sessions | +94.1% | +7.7% | -23.7% | 0.79 | 0.32 | 6.5 | 32% |
| B, penalty 0 | +115.5% | +8.9% | -26.6% | 0.83 | 0.34 | 22.2 | 21% |
| B, penalty 5 | +123.9% | +9.4% | -24.4% | 0.85 | 0.38 | 3.1 | 19% |
| B, penalty 15 | +172.9% | +11.8% | -26.8% | 0.99 | 0.44 | 1.8 | 17% |
| B, penalty 35 | +162.2% | +11.3% | -27.1% | 0.90 | 0.42 | 1.1 | 17% |
| B, penalty 50 | +173.5% | +11.8% | -27.9% | 0.86 | 0.42 | 0.9 | 11% |

Nothing beats buy and hold on return over 2016-2024. VIX alone (1 session) has the best risk-adjusted profile: max
drawdown cut from -33.7% to -14.2% and the best Sharpe and Calmar, at the price of about half the total return
(5.5 points of CAGR) and 40% of sessions out of SPY. The jump models land in between with little risk benefit.
Waiting for confirmation (2-3 sessions) makes VIX alone worse on every measure.

## Holdout (run once, 2026-11-22; nothing changed after it)

`uv run --with jumpmodels python scripts/regime/vix_jump_spike.py final`: 2025-01-02 to 2026-11-20, 474 sessions,
the two variants chosen on dev, 3 SPY falls of at least 8%.

| Fall (high -> trough) | VIX alone, 1 session | B, penalty 15 |
|---|---|---|
| 2025-02-19 -> 2025-04-08 (-18.8%) | on time, -2.6% (4 sessions) | late, -8.8% (16 sessions) |
| 2026-01-27 -> 2026-03-30 (-8.9%) | on time, -2.6% (8 sessions) | missed |
| 2026-10-06 -> 2026-11-20 (-11.5%, current) | on time, -2.0% (9 sessions) | late, -6.0% (31 sessions) |
| False switches | 3.7/year | 0 |

| Portfolio (SPY / SHV, 10 bp per switch) | Total | CAGR | Max DD | Sharpe | Calmar | Switches/year | Defensive |
|---|---|---|---|---|---|---|---|
| SPY buy and hold | +19.3% | +9.9% | -18.8% | 0.63 | 0.53 | - | - |
| VIX alone, 1 session | +4.9% | +2.6% | -9.3% | 0.33 | 0.28 | 14.4 | 36% |
| B, penalty 15 | +19.0% | +9.7% | -9.3% | **0.81** | **1.04** | 1.6 | 13% |

- **Detection**: VIX alone met the -3% goal on all three out-of-sample falls (-2.0% to -2.6%), including the
  current one. The jump model was late or missed.
- **Money**: the opposite. VIX alone's 3.7 false switches a year and its slow return to SPY after V-shaped rebounds
  cost most of the return (+4.9% vs +19.3%). The jump model matched buy and hold's return with half its drawdown.
- The ranking of the two flipped between dev and holdout (dev Sharpe: VIX alone 1.04, B 0.99; holdout: 0.33 vs
  0.81). With 3 falls in 474 sessions, neither result is robust; the holdout is now spent.

## Left to the builder

- Output layout; how the online prediction is run per 6-month block.
