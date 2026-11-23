# sector_pods: a two-speed stress signal for the risk manager and the PM

Date: 2026-11-24. Status: design, awaiting approval.

## Intent

The sector_pods risk manager challenges the pods' views, and today the only market signal it gets is
`regime`, one word (bullish / neutral / bearish / unknown) from the SPY/VIX stress score. That score is
slow: it needs two of four conditions, and two of them (SPY below its SMA50, SPY 5% off its 1-year high)
only trip once a fall is well under way. The prompts list `regime` as a field but never say what to do
with it.

The user wants a more accurate input: **when the early warning is on, the PM should pick more stable
ETFs (consumer staples, healthcare)**. The book stays always invested (98%), so the signal changes the
mix, never the size.

"Two speeds":

- **Early warning** (fast, noisy): the "VIX alone" rule from the 2026-11 research, already coded as
  `utils/helpers.in_market_fall` and called by nothing. A fall starts on a close with VIX/VIX3M >= 0.95
  or VIXY up 10%+ over 5 sessions; it ends after 3 consecutive calm closes (VIX/VIX3M < 0.90). Research
  record: switched at a median -1.6% from the SPY high on 2016-2024, caught 3/3 falls of 2025-2026,
  about 4 false alarms a year, back to normal a median 32 sessions after the trough.
- **Confirmed regime** (slow): today's `strategy.regime`, now shown with its 0-4 score and the four
  conditions, so the agents can tell a bearish 2/4 from a 4/4 and see a neutral one point from bearish.

## Goals

1. Every strategy gets `strategy.market_fall` (the early warning) and `strategy.regime_reading` (the
   full `RegimeReading`), refreshed once per session next to `strategy.regime`, with the same
   no-look-ahead cutoff.
2. The sector_pods risk manager and PM see a `stress` field next to `regime`, and are told how to use it.
3. When the early warning is on, the risk manager flags strong scores on cyclical ETFs and the PM tilts
   toward the defensive list, within the existing caps.

## Non-goals

- No change to the 4-point regime rule, its thresholds, its chart line, or `uv run regime
  recompute|scorecard`.
- No change to `in_market_fall`'s thresholds (fixed by the research).
- No chart line for the early warning (log only).
- The pods' prompts are unchanged: they keep the `regime` label only. The tilt is the risk manager's and
  the PM's job; a pod rates its own sleeve.
- No other strategy reads `market_fall` yet.

## Anti-goals (a failure even if it works)

- **Code forcing weights.** No minimum defensive share, no cap change, no rule in `check_weights` or
  `fill_book`. It stays a prompt nudge; the PM decides.
- **A quiet exposure cut.** The tilt must not turn into a defensive-only book: with 25% per ETF, XLP +
  XLV + XLU top out at 75%, and a book of only those would leave the rest in SHV. The PM keeps other
  ETFs so `fill_book` still reaches 98%.
- **A VIXY problem costing the regime.** VIXY starts in 2011 and may fail to download; neither may stop
  the regime or the session.

## Constraints

- No look-ahead: the closes are those strictly before the session date, the same `RegimeSeries.before`
  cutoff the regime uses (root `CLAUDE.md`, "No data tool may use wall-clock time").
- One batched Yahoo download per day, as today.
- `reviews.jsonl`'s `regime` stays a string: `dashboard/reviews_reader.py` reads that file.
- `RegimeSeries` is built in 6 places (`core/regime_data.py` x2, `core/regime_cli.py`, 4 test files);
  the new column must not force edits there.
- Both clean windows for VIX-based rules (2025-2026 holdout, 2008-2015) are spent: any evaluation is on
  data already seen, and is said so.

## Approaches (picked: A)

- **A. In `core/`, for every strategy (picked).** `YahooRegimeData` adds VIXY to its one batched
  download; `Strategy._refresh_regime` keeps the full reading and computes `market_fall` with
  `in_market_fall`. One download, one look-ahead cutoff, reusable by bull_bear later. Cost: touches
  shared code; every strategy downloads one more symbol and logs one more line.
- **B. In sector_pods only.** `MarketFacts` downloads VIX/VIX3M/VIXY itself. Core untouched, but a second
  download of the same data and a second cutoff to get right. Rejected.
- **Dropped: ratio only, no VIXY.** No new data, but the VIXY leg made 8 of the research's 10 real catches
  against 2 for the ratio; without it the warning is mostly false alarms.
- Effect on the book, picked by the user: **tilt, not swap** (above), over a defensive-only book (cuts
  exposure in warnings, ~4 false alarms a year) and over higher caps for defensives during a warning
  (more concentration, `check_weights` changes).

## Design

### core/regime_data.py

- `SYMBOLS` gains `"VIXY"`. `RegimeSeries` gains `vixy: tuple[float, ...] = ()`. An empty tuple means
  "no VIXY data" so the 6 existing constructors keep working; otherwise it is aligned with `sessions`.
- `_fetch`: VIXY closes reindexed on SPY's sessions, forward-filled with the same `max_fill`; `nan`
  where missing (before 2011, a gap). **An empty or unreadable VIXY is not an error**: it becomes all
  `nan`. SPY, VIX and VIX3M keep their current rules.
- VIXY has had many reverse splits. `parse_yahoo_frame` already replaces `close` with Yahoo's Adj Close
  (split- and dividend-adjusted, the same series the research downloaded with `auto_adjust=True`), so a
  split is not a jump; a test pins this (a reverse split in the fake download must not turn the flag on).
- `before` and `series` slice `vixy` like the other columns (when it is not empty). `before` never raises
  on a `nan` in `vixy`; `in_market_fall` already reads `nan` as "VIXY leg off".

### core/strategy.py

- New attributes next to `regime`: `regime_reading: RegimeReading | None = None` and
  `market_fall: bool | None = None`.
- `_refresh_regime`, on the same `series` it already reads: when `reading` is not None, set
  `regime_reading = reading` and `market_fall = in_market_fall(series.vix, series.vix3m, vixy)`, where
  `vixy` is `series.vixy` or all-`nan` when empty. The 503 rows are well over the 250 `in_market_fall`
  needs. One more log line, e.g. `Market fall warning ON (VIX/VIX3M 0.97, VIXY +12.3% over 5 sessions)`
  / `OFF`.
- Failures keep the previous values, exactly as `regime` does. When `reading` is None (short history),
  `market_fall` stays None too: one rule, both signals appear together.

### strategies/sector_pods

- `pods.py`: `DEFENSIVE_ETFS = ("XLP", "XLV", "XLU")`, each checked to be in `ALL_SYMBOLS`.
- `graph.py` `MarketFacts`: takes `stress: Callable[[], dict | None]` (or the reading and the flag; the
  builder's call) and adds to `base`:

  ```json
  "stress": {"early_warning": true, "regime_score": 2,
             "conditions": {"vix_above_vix3m": true, "vix_spike": false,
                            "spy_below_sma50": true, "spy_5pct_off_high": false},
             "defensive_etfs": ["XLP", "XLV", "XLU"]}
  ```

  `null` when the strategy has no reading yet. `agent_sector_pods.py` wires it from `self.regime_reading`
  and `self.market_fall`.
- The risk manager's and the PM's contexts gain `stress`; the pods' do not. `reviews.jsonl` records
  `stress` next to `regime`.
- `handoff.OBJECTION_KINDS` gains `"stress"`, so the tilt is countable in `reviews.jsonl`; the risk
  prompt's list of kinds (`crowding, macro, drawdown, reversal or other`) gains it too.
- `prompts.py`, risk manager: describe `stress` in the context fields, and add to "Look for":
  *stress: when `stress.early_warning` is true, strong scores on cyclical or high-volatility ETFs
  (`volatility_20d_pct`); suggest the `defensive_etfs` instead. The early warning fires early and often
  falsely (about 4 times a year); weigh it with `regime_score` and `conditions`.*
- `prompts.py`, PM: describe `stress`, and add a rule: *when `stress.early_warning` is true, tilt toward
  `defensive_etfs` up to the caps, but keep other ETFs: the book stays fully invested.*

### Docs

`core/CLAUDE.md` (regime bullet: VIXY, `market_fall`, `regime_reading`), `strategies/sector_pods/CLAUDE.md`
(the `stress` field, the defensive list, the new objection kind; "Regime is context only" stays true in
code), root `CLAUDE.md` regime bullet (one clause).

### Tests

- `regime_data`: VIXY aligned and forward-filled; empty or unreadable VIXY gives `nan`, no error; `before`
  slices it and does not raise on its `nan`; an old-style `RegimeSeries` without VIXY still works.
- `strategy`: `market_fall` and `regime_reading` set on refresh; None with short history; kept on a failed
  refresh; a stressed fake series turns the flag on, a calm one off.
- sector_pods: the `stress` dict built from a reading; `null` without one; present in the risk and PM
  contexts and `reviews.jsonl`, absent from the pods'; `"stress"` accepted as an objection kind;
  `DEFENSIVE_ETFS` in `ALL_SYMBOLS`.

## How we will know it worked

1. **Behaviour (the main check):** in a 2-year backtest, weeks with `stress.early_warning` true hold a
   larger share of XLP + XLV + XLU in `book` than the other weeks, `"stress"` objections appear in
   warning weeks, and every week is still about 98% invested.
2. **Return, read with care:** the same window as the always-invested run (2024-09-03 to 2026-10-05),
   but the baseline arm is **rerun** on today's code and data, not taken from 2026-11-18_080858 (the
   Alpaca news archive changed since). Reruns differ by about 25 log points, so 2-3 runs per arm; a gap
   inside that is "no harm", not a win. Both falls in that window (Apr 2025, Feb 2026) were already used
   to judge VIX alone.

## Left to the builder

- The exact shape of what `MarketFacts` receives (one callable or two).
- Log wording.
- Prompt wording, inside the rules above (no absolutes: describe a balance, as in the existing prompts).
