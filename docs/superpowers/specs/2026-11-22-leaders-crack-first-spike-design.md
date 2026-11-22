# Spike: leaders crack first -- early detection of a SPY fall

Date: 2026-11-22. Status: **closed, NO** (see Results). Type: research spike (one throwaway script, no production code).

## Intent

Daily strategies should switch defensive in the first 3% of a SPY fall. Every detector tried so far reacts after
the fall starts (VIX alone: median 5 sessions and -3.8% after the high; news sentiment: NO, see
`2026-11-22-news-vix-early-regime-spike-design.md`). This spike tests the one idea with a structural reason to
*lead* the index: the heaviest stocks carry a large share of SPY, so a fall can start in them while SPY still sits
near its high.

## Goals, non-goals, anti-goals

- Goal: a daily `normal` / `defensive` verdict from leader weakness alone, compared with VIX alone.
- Non-goals: no strategy integration, no orders, no P&L of a defensive switch, no combination with VIX or news
  (a follow-up only if a variant passes), no reuse of `core/regime*.py`.
- Anti-goals: tuning a threshold after looking at the holdout; more than the two variants below.

## Approaches considered

- **1. Drawdown gap (chosen).** Equal-weight leader basket at least 7% below its 60-session high while SPY is
  within 3% of its own. Same rule for both leader sets.
- 2. Relative-strength break (leaders / SPY at an N-session low): fires on ordinary sector rotation; too many
  false switches expected.
- 3. Leader breadth (6 of 10 names down 7%): too coarse with 3 ETFs, so the two sets would not share a rule.

## Leader sets (user's choice: both)

- **L1, stocks**: the first 10 symbols of the point-in-time monthly list for that month in
  `data/universe/us_stock_universe_history.json` (`months`; each list ranked only on data before the month, by
  dollar volume). One ticker per company: GOOG is skipped when GOOGL is in the list (same for any BRK-A / BRK-B
  pair), and the next symbol fills the slot. Members change on the first session of each month. Prices: `Adj Close`
  from `cache/universe_history/bars/<SYMBOL>.parquet`.
- **L2, ETFs**: QQQ, SMH, XLK (Yahoo adjusted closes).

## Rules (fixed before running)

1. **Basket**: equal-weight, rebalanced daily: its daily return is the mean of the members' daily returns (members
   with no bar that day are left out of the mean). The basket level is the cumulative product, starting at 1.
2. **Drawdowns**: `basket_dd` = basket level / its max over the last 60 sessions - 1; `spy_dd` likewise for SPY
   closes.
3. **Crack** on session t (decided before the open, from closes up to t-1): `basket_dd <= -7%` and
   `spy_dd >= -3%`.
4. **Defensive** starts on the first session with a crack.
5. **Exit**: back to `normal` after 3 consecutive sessions with `basket_dd > -3%` (leaders back near their high).
6. **Baseline, VIX alone** (unchanged from the news spike): VIX / VIX3M >= 0.95 or VIXY 5-session return >= +10%
   switches defensive; exit after 3 consecutive sessions with VIX / VIX3M < 0.90.

## Evaluation

- Reference falls: SPY closes, swing high followed by a drop of at least 8% (same zigzag as the news spike).
- Per fall, the first defensive start, classified:
  - **early**: started up to 20 sessions before the high and still defensive at the high (report the lead in
    sessions; counted as 0% drawdown);
  - **on time**: after the high with SPY drawdown at the switch (previous close vs the high) of -3% or better;
  - **late**: after the high, worse than -3%;
  - **missed**: no switch between 20 sessions before the high and the trough.
- False switches: starts not within [20 sessions before a high, its trough] of any fall; reported per year.
- Also: median sessions and median drawdown at the switch, share of sessions defensive per year.
- **Development pass bar**: median drawdown at the switch over all falls (missed = worst) better than -3%, and
  false switches per year no more than VIX alone on the same window. A variant that fails stops there.

## Windows

- Development: 2016-01-04 to 2024-12-31 (first verdict after the 60-session warmup, ~April 2016).
- Holdout: 2025-01-02 to 2026-11-20, run **once**, only for a variant that passed development, with nothing changed.

## Data and constraints

- No new download beyond Yahoo daily closes for SPY, QQQ, SMH, XLK, ^VIX, ^VIX3M, VIXY; stock bars are cached.
- Code in `scripts/regime/leaders_crack_spike.py`; it may import the fall zigzag, market download and switch
  helpers from `scripts/news/early_regime_spike.py` rather than copy them.

## Results and verdict (2026-11-22): NO (both variants fail development; holdout not run)

`dev` 2016-03-31 to 2024-12-31 (2,204 sessions after the warmup), 10 SPY falls of at least 8% fully inside.

| Fall (high -> trough) | L1 stocks top 10 | L2 ETFs | VIX alone |
|---|---|---|---|
| 2018-01 (-10%) | missed | missed | on time, -0.7% |
| 2018-09 (-19%) | **early, 8 sessions before the high** | missed | on time, -1.5% |
| 2020-02 COVID (-27%) | missed | missed | early, 15 sessions |
| 2020-03 second leg (-17%) | missed | missed | already defensive since 2020-01-28 |
| 2020-09 (-9%) | missed | missed | early, 0 sessions |
| 2022-01 (-13%) | **early, 18 sessions before the high** | missed | late, -5.4% |
| 2022-03 (-21%) | already defensive since 2021-12-07 | missed | late, -4.7% |
| 2022-08 (-17%) | missed | missed | late, -3.8% |
| 2023-07 (-10%) | missed | on time, -2.7% | on time, -1.7% |
| 2024-07 (-8%) | on time, -1.4% | on time, -2.8% | on time, -2.2% |
| False switches | 1.3/year | 0.3/year | 4.2/year |
| Sessions defensive | 25% | 6% | 39% |

- **L1** leads in exactly the leader-driven falls (Q4 2018 FAANG, the 2022 growth top, July 2024 AI) and misses
  the rest: macro shocks (Feb 2018, COVID) and joint crashes where SPY falls as fast as the leaders, so the
  "SPY still within 3%" gap never opens. 4 of 10 covered (one of them by the long 2021-12 switch).
- **L2** almost never fires: QQQ, SMH and XLK move too closely with SPY for a 7% vs 3% gap to open (2 of 10).
- **VIX alone on 2016-2024** already switches at a median of -1.6% from the high (better than the -3% goal);
  its real weakness is 4.2 false switches a year and 39% of sessions defensive.
- Not tried, as the spec requires: other thresholds, L1 combined with VIX. Known limit: the dollar-volume list
  includes non-SPY names (BABA, NIO, BHC) and BRK-A.

Correction (2026-11-22 review): `evaluate` first printed a fall as missed when the strategy had been defensive
since before the 20-session early window (VIX alone 2020-03, L1 2022-03); it now counts as early, already defensive.
The table above is the corrected run; no verdict changed.

## Left to the builder

- Output layout; how the share-class rule is coded; warmup handling at the window start.
