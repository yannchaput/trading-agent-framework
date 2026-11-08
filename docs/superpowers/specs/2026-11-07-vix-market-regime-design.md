# A market regime that labels SPY sell-offs while they happen -- design

Date: 2026-11-07. Status: agreed in conversation, builder check done, awaiting the user's approval.

## Why

Every strategy gets a market regime (`core/regime.py`, `strategy.regime`: 1 bullish / 0 neutral / -1 bearish),
refreshed once per session and drawn as coloured bands on the dashboard's Trades and Returns charts. Nothing trades
on it, and that stays true: its only job is to **label market stress accurately on the dashboard**.

Today it does that badly. The 2026-11-07 audit of the 10Y cross_momentum run (`2026-11-07_152938_backtesting`) found
the SMA50/SMA200 crossover with a realized-volatility cap:

- turns neutral only once SPY is already ~5% off its peak, and bearish at -12% to -17% or never (COVID -12.4%, 2025
  tariffs -17.3% four days before the bottom; 2018-02, 2020-09, 2024-08, 2026-03 never);
- reads bearish mostly near bottoms: SPY's next-21-session return averages +4.4% after a bearish day against +1.2%
  after a bullish one;
- labels 25% of the bearish reference days (below) as bearish.

The user wants the bearish readings to be accurate first; the neutral and bullish readings come from the same formula.

## The reference: what "accurate" means

Decided by the user, used as the yardstick for every acceptance number below:

- **Bearish** = every SPY fall of at least 8% from a recent high, from the swing high to the swing low (a zigzag with
  an 8% threshold on daily closes: a swing ends when SPY rebounds 8% off its low). Small falls count: 2024-08 (-8.4%)
  and 2026-03 (-8.9%) are bearish. Falls inside a recovery count too (2023-07-31 -> 2023-10-27, -10%), and 2022 is
  three legs, its +11% and +17% rallies not bearish.
- The episode **ends at the bottom** (the swing low).
- **Neutral** = 4-8% falls on the same zigzag rule (my call; nobody reads them well and the user did not ask about
  them). **Bullish** = everything else.

Over 2016-01-04 .. 2026-10-05 that gives 13 bearish episodes: 2016-01, 2018-02, 2018 Q4, 2020-02 -> 03-12,
2020-03-13 -> 03-23, 2020-09, 2022-01 -> 03, 2022-03 -> 06, 2022-08 -> 10, 2023-07 -> 10, 2024-08, 2025-02 -> 04,
2026-01 -> 03.

A label computed before the open cannot know where a fall will end, so it will switch on some way into each fall and
overstay the bottom by a few sessions. The scorecard measures those lags instead of expecting a perfect match.

## The formula (chosen by the user: "finalist 1")

Each session, from data up to the **previous** completed close, score four conditions, one point each:

| Condition | Meaning |
|---|---|
| VIX close >= VIX3M close | term structure inverted: near-term fear above 3-month fear |
| VIX close >= 1.5 x its median over the last 252 sessions | a VIX spike relative to its own recent normal |
| SPY close < its 50-session SMA | short-term downtrend |
| SPY close <= 0.95 x its highest close over the last 252 sessions | at least 5% off the 52-week high |

States:

- **Bearish** when the score is >= 2.
- **Neutral** when the score is 1; **bullish** when it is 0.
- **Leaving bearish (the "bottom exit")**: once bearish, the state tracks SPY's lowest close since it turned bearish.
  The episode ends when SPY closes 8% or more above that low (the same 8% rebound that ends a reference swing), or
  when the score falls below 2. After an 8%-bounce exit the bottom is presumed in: the state cannot turn bearish again
  until SPY closes below that episode low (the fall resumed) or the score has dropped to 0 (stress cleared).
- **Neutral hold (my call)**: a change between bullish and neutral takes effect only after it has held for 3
  sessions, to calm the flicker (about 260 regime changes in 10 years without it). Changes into and out of bearish are
  never delayed. The builder confirms on the scorecard that this lowers the changes count and leaves bearish recall
  unchanged; if it costs recall, drop it and say so.

The SMA50/SMA200 trend, the 20-day realized volatility and its 80th-percentile cap are removed, with
`RegimeParameters`' current fields. The four thresholds, the 2-point bearish line and the 8% bounce become the new
`RegimeParameters` (round values as above; do not tune them further: the sweep tried 72 combinations on 13 episodes,
so finer tuning would be fitting noise).

### Measured on the spike (2016-01-04 .. 2026-10-05, signal from the prior close)

| | Current formula | Finalist 1 |
|---|---|---|
| Reference bearish days labeled bearish | 25% | 58% |
| Bearish days outside any episode or its overstay | 86 | 86 |
| Mean SPY fall when bearish first appears in an episode | -11% | -5.8% |
| Median sessions still bearish after the bottom | 0 | 6 |
| Regime changes | 74 | ~260 (before the neutral hold) |

Known, accepted limits: short shallow dips (2018-02, 2024-08, 2026-03, 2023) stay partly neutral; the 2020-09 dip is
never bearish; the remaining false alarms are mostly 4-8% pullbacks read one step too dark (2018-03 -> 05 retest,
SVB 2023-03, 2024-04).

## Goals

1. `core/regime.py` implements the formula above, still as a **pure** function (no I/O, no pandas, no `Strategy`):
   SPY closes, VIX closes and VIX3M closes in, a reading out. It stays stateless across calls: each refresh **replays**
   the state machine over a trailing window of sessions (see Constraints), so a live/paper restart and a backtest
   started on any date give the same answer as an uninterrupted run.
2. Every strategy gets it, in every mode, through the existing `Strategy._refresh_regime` path (once per session,
   failures keep the previous value with a warning, never a hook, never `on_bot_crash`). `strategy.regime` keeps its
   meaning and values; nothing trades on it.
3. **SPY, VIX and VIX3M all come from Yahoo** (`SPY`, `^VIX`, `^VIX3M`) in every mode and for every strategy, whatever
   the strategy's own price source (the user's decision: the regime labels the market, so it reads the same for every
   strategy on a given day; Alpaca serves no indices anyway). SPY uses the same adjusted daily closes as the Yahoo
   backtest data (`backtesting/data/yahoo.py`). The regime no longer reads `get_historical_prices(benchmark_symbol,
   ...)`, and no longer follows `benchmark_symbol`: it is always SPY. cross_momentum's `_refresh_regime` override
   (which exists only to feed the regime Yahoo SPY in paper/live) becomes redundant; the builder removes it if nothing
   else needs it.
4. The regime log line shows the score and its four components with their values (VIX, VIX3M, VIX vs its median, SPY
   vs SMA50, SPY vs 52-week high, and the episode low when bearish), so a reader can see why a session read bearish.
5. **Existing backtest runs are recomputed** (the user's decision): a one-off command rewrites the `Regime` line of
   every run under `logs/*/backtesting/*/` with the new formula, over that run's sessions, and the dashboard shows the
   rewritten data with no change to how it reads it. Runs that have no `Regime` line yet get one (my call: so every run
   shows the band).
6. A reproducible, read-only scorecard command (my call) recomputes the reference episodes and Finalist 1's row of
   the table above for the implemented formula, so the numbers here can be checked and re-checked later. The "Current
   formula" column is a historical record: the old classifier is removed, not kept for the scorecard (my call).

State-machine details, as the spike implements them (settled from the spike code):

- After an 8%-bounce exit while the score is still >= 2, the state reads **neutral** (the block forbids bearish; the
  score is not 0, so not bullish).
- The re-entry block clears when the score reaches 0 or SPY closes below the episode low, whichever comes first.
- The scorecard reports an episode that never turns bearish with its full depth as its "fall when bearish appeared".

## Non-goals

- Trading, sizing or gating any strategy on the regime. Still a separate decision.
- Catching cross_momentum's own drawdowns that are not market sell-offs (2019-07, 2020-10, 2021-02 -> 05, 2023-03 ->
  10, 2024-03, 2025-11: momentum crashes with SPY flat or up). A market label must not light up for them.
- The dashboard's user-set `dashboard_regime` run tag (`settings.json`, "Bearish/Neutral/Bullish/All-Weather" on the
  Scorecard page). Different thing, untouched.
- Rewriting `backtest.log`'s old regime lines, `metrics.json`, equity or trades.
- A regime on any index other than SPY. A strategy's `benchmark_symbol` still drives its benchmark curve and metrics,
  not its regime.

## Anti-goals (would make this a failure even if it works)

- **Look-ahead.** A backtest's regime on session D may use only VIX/VIX3M/SPY closes of sessions strictly before D
  (the root `CLAUDE.md` rule: no data tool may use wall-clock time; cut on `strategy.clock.now()`). The recompute
  command obeys the same rule for each session it rewrites.
- Losing a session because Yahoo is down, or making a backtest fail because VIX is missing for a few days.
- A regime that differs between a restarted live process and an uninterrupted one.
- Destroying an existing run's data without a way back.

## Constraints and decisions already made

- Money stays `Decimal`; the regime works on floats from closes as today (it is not money). No new float boundary.
- `classify_regime` keeps rejecting non-finite or non-positive closes.
- **History needed.** The indicators need 252 sessions (VIX median, 52-week high) and the replay needs a trailing
  window on top. Replaying 252 sessions (my call) means about 505 daily closes of SPY, VIX and VIX3M, against 273 today;
  `min_bars`, the backtest warmup floor (`REGIME_WARMUP_SLACK` in `Strategy.run_backtesting`) and the
  `warmup_trading_days` the runs record follow from it. A bear episode older than the replay window starts its replay
  from a non-bearish state; the builder measures on the scorecard how often a 252-session replay disagrees with a
  continuous run over 2016-2026 and reports it (expected: never or almost never, since episodes end on an 8% bounce).
- **Aligning dates.** SPY, VIX and VIX3M are joined on session date; a session missing VIX or VIX3M uses the last
  known value before it (forward fill) up to 5 sessions, beyond which the refresh fails and keeps the previous regime.
- **The recompute uses Yahoo for every run**, including the 11 runs backtested on Alpaca bars (news_binary,
  vwap_pullback_continuation, earnings_drift): since the regime always reads Yahoo SPY, a rewritten run and a fresh
  backtest agree whatever the run's price source, and the command needs no broker keys.
- **Fetching SPY, VIX and VIX3M.** One Yahoo download covering the whole window (backtest: warmup start .. end, cached on disk with
  the other Yahoo data; paper/live: the trailing window, refreshed each session), sliced per refresh to closes strictly
  before the session. Where it lives and how it is cached is the builder's call, inside the layering of the root
  `CLAUDE.md` (`core/regime.py` stays pure; the network sits outside `core/`).
- **The recompute command** keeps a one-time backup of each `indicators.parquet` it rewrites (my call: a sibling file
  the dashboard ignores; a second run never overwrites the first backup), rewrites only the `Regime` rows (every other
  indicator line untouched), stamps them at the same time of day the existing lines use, takes the run's sessions from
  its `equity.parquet` index, and is idempotent. It reports per run: rewritten, added, or skipped with a reason.
- Tests stay off the network (hand-written fakes, as in `tests/fakes.py`); manual checks against real Yahoo go in
  `scripts/tests/` like the other smoke scripts.
- The quality gate in the user's global instructions applies (ruff, format, pyright, pytest, coverage >= 80%,
  interrogate >= 80%).
- Docs: `core/CLAUDE.md`'s regime gotcha and the root `CLAUDE.md` regime bullet are rewritten for the new formula;
  the dashboard's band colours and labels do not change.

## Acceptance

1. The scorecard command, run on the new implementation over 2016-01-04 .. 2026-10-05, reproduces Finalist 1's row
   above within a small tolerance (Yahoo revisions; the neutral hold changes the changes count only).
2. A backtest's regime on any session equals the recompute command's value for that session (same formula, same
   cutoff).
3. After the recompute, the 10Y cross_momentum run's dashboard bands show bearish during 2018 Q4, COVID, the 2022
   legs and the 2025 tariff fall, from the sessions listed by the scorecard.
4. No test touches the network; the gate is green.

## Left to the builder

Module layout and names for the VIX fetch, the scorecard and the recompute commands (a `uv run` entry point or a
`scripts/` file); the backup file name; the exact log line wording; how `RegimeReading` carries the components; task
order.

## Reference material

The throwaway spike that produced the numbers: `scorecard3.py` and `sweep.py` in
`/tmp/claude-1000/-mnt-data-projets/4e393363-f391-4c71-8ab0-ebc0b7f7728b/scratchpad/` (session scratchpad; may be
gone, the formula above is the source of truth).

## Built differently from the design

- The Yahoo fetcher lives in `core/regime_data.py` (it imports `backtesting.data.yahoo`); `core/regime.py` itself stays pure.
- The SPY/VIX/VIX3M download is cached in memory per process and day, not on disk.
- `DAILY_WARMUP_FLOOR` (283) is kept instead of raising it to about 515: the regime no longer reads strategy bars.
