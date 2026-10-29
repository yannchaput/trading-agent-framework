# bull_bear: the judge sizes each position by the strength of its case

Date: 2026-10-28 · Builds on `2026-10-26-bull-bear-fundamentals-first-design.md` (fundamentals first, news as support).

## Revision 1 (2026-10-29, the user's decisions after the first replay) — supersedes the text below where they conflict

The first replay (108 judge runs on the two runs; strengths recorded for the stocks put in only) gave strong +1.61, standard
-0.01, weak -4.94 pts mean 20-session excess (n 521 / 405 / 64; order the same in both runs), per-review strong - weak
+2.23 (42 reviews, 60 skipped, t 0.69, below the +3.0 bar), per-review strong - standard +2.14 (102 reviews, t 1.03).
The user chose to proceed past the missed bar (option 2) with these changes:

- **Three buckets, then the selection.** The judge rates EVERY stock of the debate `strong`, `standard` or `weak`
  (a strength on every verdict, `in` or `out`), and only then picks the stocks to put in, from the strong and standard
  ones, at most `max_in` (still a ceiling). **A weak stock is always `out`**: a submission that puts a weak stock in is
  rejected (HandoffError naming the stock). A strong or standard stock may still be `out` (more than `max_in` earn a
  place, or the judge's own call). Rating the whole debate keeps `standard` from absorbing the marginal names and gives
  the scorer a strength for every debated stock, not only those put in. (My reading of the user's words; the details
  above are my calls.)
- **Two weights, no `weak_weight`:** `strong_weight` 0.14, `standard_weight` 0.09 (replace `min_weight`/`max_weight`).
  Validation: `0 < standard_weight < strong_weight <= 1`, finite, and the smallest standard weight after scaling,
  `standard_weight x investable / ((max_book_size - 1) x strong_weight + standard_weight)`, must stay above
  `rebalance_band` (6.5% with the defaults).
- **SHV: nothing new.** The existing parking stays as it is (money the weights do not use is parked in SHV through
  `target_portfolio`); no new SHV logic, wording or tests for it. The judge's prompt no longer mentions cash or SHV.
- **State** (version 3): `state.verdicts` keeps the strength of EVERY verdict (`in` and `out`), so the next review shows
  the previous strength of each stock it judged before. A version-2 file still loads without strengths.
- **Logs and scoring:** `reviews.jsonl` carries `strength` on every verdict and `strength_counts` over all verdicts. The
  scorer groups ALL debated stocks by the judge's strength (strong, standard, weak) and reports strong - weak and
  strong - standard spreads, with the same rules as before.
- **Validation:** the first replay no longer gates (the user accepted its evidence). After the new prompt is built, a
  second replay of the same two runs, recording the strength of every stock, runs in the background as non-gating
  evidence (shares of each strength, weak vs the others, strong vs standard among the stocks put in).

Everywhere below, read "strength only on an `in`", "weak_weight" and "cash/SHV in the prompt" in light of this revision.

## Why

Today every stock the judge puts in is sized by inverse volatility alone (`sizing.capped_inverse_volatility`, bounded to
[4%, 20%], summing to 98%). A stock with a compelling bull case and a weak bear case gets the same treatment as one the
judge let in narrowly, and the book is always fully invested once it holds 5 stocks or more.

The decision data of the 9 Qwen 3.6 / 3.8 backtests (study of 2026-10-28, about 6,000 candidate ratings matched to their
20-session forward return from the first open after the review, excess over that review's candidates) shows that the
strength of the debate carries information the sizing throws away:

- Among the stocks the 3.8 judge put in, bull `high` with bear `low` did **+3.1 pts** (496), bull `high` with bear
  `medium` +0.7 (285), bull `medium` about 0 whatever the bear said (+0.1 / +0.0), bear `high` -1.7 to -3.0. The bear's
  rating predicts nothing on its own (t < 0.5 in every run), but alongside the bull's it separates the strong cases.
- The judge puts almost every bull-`high` stock in (about 4.5 a review), then fills the book to about 9.8 stocks with
  `medium` and `low` ones; those fillers did worse than the average candidate in 7 of 8 runs. Weekly equal-weight books
  of the bull-`high` stocks beat the judge's book in 6 of 8 runs (e.g. 10-27 run: +175% vs +87%; crude: about 4.5 names,
  no sizing, no path effects).

The user does not want the bull, or any single agent, to decide the trades: the judge decides, weighing the bull's
conviction against the bear's risk. So the fix is not to build the book from the bull's ratings but to let the judge say
how strong each case is and let that strength, alone, set the position's size.

## Goals

- The judge rates every stock it puts in as `strong`, `standard` or `weak`, from the strength of both cases: the bull's
  conviction against the bear's risk.
- The strength alone sets the stock's weight: three fixed weights, one per strength. Volatility no longer sizes the book.
- Weak cases lower exposure: money a weak book does not use stays in the parking instrument (SHV).
- A held stock whose strength changes is resized at the next review, even when its verdict stays `in`.
- The strength is logged and scored, so a replay and backtests can tell whether it separates forward returns.

## Non-goals

- Changing the researcher's, the bull's or the bear's prompts or ratings.
- Changing what goes into the debate set, `max_book_size` (10, a ceiling), the cooldowns, the forced exits, or `in`/`out`
  as the judge's main decision.
- Making the bear more informative (a separate topic).
- Showing the strength on the dashboard (it can read `targets` as today).

## Anti-goals

- **Code never reads the bull's conviction or the bear's risk to set a weight.** They are the judge's inputs only; the
  judge's strength is the only sizing input. Neither the bull nor the bear decides a trade.
- **No forced full investment.** When the judge's cases are weak, the portfolio holds less stock; code never scales weak
  weights up to fill the book.
- No volatility term in the sizing, not even as a tie-break.

## Decisions (binding)

### The judge's strength

- `submit_verdicts` gains a `strength` per verdict: one of `strong`, `standard`, `weak`. It is **required on every `in`**
  and **not allowed on an `out`**; a submission that breaks either rule is rejected with a message that says which
  stock and what is allowed, like today's checks (`HandoffError`), and the model corrects it in the same run.
- The judge's system prompt explains the three strengths as a balance, never as an absence of risk: weigh the bull's
  conviction against the bear's risk; `strong` when the bull's case clearly outweighs the bear's, `standard` when it
  outweighs it, `weak` when it only narrowly earns a place. It says that every stock carries some risk and that a strong
  case does not need a risk-free stock (the user's concern: wording such as "no material risk" would be read literally
  and a model would almost never rate a stock strong). It says that code sizes each
  position by its strength alone: a strong case gets the largest position, a weak one a small position, and money weak
  cases do not use stays in cash. It does not give the numbers. The evidence rules already in the prompt (fundamentals
  first, news from others never a reason for out) apply to the strength as they do to the verdict.
- A stock the judge judged before shows its previous strength next to its previous verdict (`previous_verdict` gains
  `strength` when the previous verdict was `in`). **Changing the strength needs no `what_changed`**: sizes follow the
  judge's current view. A change of verdict still needs `what_changed`, unchanged.
- `max_in` stays a ceiling, not a target; nothing in the prompt about how many of each strength.

### Sizing

- Three parameters in `BullBearParams` (fractions of portfolio value): `strong_weight` 0.14, `standard_weight` 0.09,
  `weak_weight` 0.04. Ten `standard` stocks make 90% (the user's anchor); `strong` is about 1.5x `standard`; `weak` is the
  current minimum position. They replace `min_weight` and `max_weight` in `BullBearParams`.
- Each stock in the book gets its strength's weight. When the weights sum to more than `investable` (98%), every weight is
  scaled down by the same factor so they sum to `investable` (ratios kept). Otherwise they are used as they are, and
  `target_portfolio` parks the rest in SHV, as it already does.
- The "below 5 stocks the rest is parked" rule disappears as a special case: fixed weights park the rest at any size.
- Resizing is the rebalancer's existing behaviour, best effort: a holding whose current weight is more than
  `rebalance_band` (3%) from its new target is traded. Unscaled, neighbouring strengths differ by 5 pts, above the band,
  so a strength change resizes the stock. When the book is scaled down (weights above 98%) the gap shrinks: in a full
  book of 10 strong stocks (9.80% each), three dropping to standard target 7.06%, a 2.74-pt move that is not traded. That
  miss costs no exposure: a scaled book is at 98% whatever the strengths, so the trade would only move weight between
  holdings; exposure falls only once the weights sum below 98%, where every change is 5 pts. (My call, from the user's
  reason for resizing, "to limit exposure".) No rebalancer change; tests cover the unscaled cases and one scaled miss;
  `bull_bear/CLAUDE.md` says best effort.
- Parameter validation: `0 < weak_weight < standard_weight < strong_weight <= 1`, all finite; and the smallest weight a
  weak stock can get after scaling must stay above `rebalance_band`, or a new weak stock would never be bought. The worst
  case is a full book with one weak stock: `weak_weight x investable / ((max_book_size - 1) x strong_weight + weak_weight)`.
  With the defaults that is 0.04 x 0.98 / 1.30 = 3.02%, just above 3%; the builder keeps this check exact (it replaces
  today's `min_weight must be above rebalance_band` and `max_book_size x min_weight` checks).
- The three weights are mirrored into `strategy.parameters`, so `settings.json` tells runs apart.
- `sizing.capped_inverse_volatility` and its tests are removed if nothing else uses it (only bull_bear does today).

### State, logs, scoring

- State: the judge's latest verdict per stock (`state.verdicts`) gains `strength` for an `in`. `STATE_VERSION` becomes 3;
  a version-2 file loads its verdicts without strengths (so the next review shows no previous strength) and keeps
  everything else; the version-1 path stays as it is. An abandonment at the `execution` stage stores the strengths with
  the verdicts, like today.
- `reviews.jsonl`: each verdict carries `strength` (null on `out`); a `strength_counts` field beside `verdict_counts`;
  `targets` show the strength weights after scaling. `dashboard/reviews_reader.py` keys are untouched.
- The judge's log line names the strength: `[judge] in: MU (strong), LITE (weak); out: ...`.
- `bull-bear-scores` gains a judge-strength section: mean 20-session (and 5-session) excess of `strong`, `standard` and
  `weak` among the `in` stocks, and the per-review spreads strong - weak and strong - standard with their t-stats, by the
  same rules as the existing spreads. A run without strengths shows the section empty, never an error.
- `bull_bear/CLAUDE.md` is updated where it describes the judge, the sizing and the state.

## Validation (the user's rule: replay first, then backtests)

1. **Replay.** Run the new judge (new prompt and handoff) on the saved judge inputs of the two fundamentals-first 3.8
   runs (`2026-10-26_232151` and `2026-10-27_195100`: the judge's context of every review, from `backtest.log`), with an
   empty memory and the previous verdicts as they were logged (without strengths). A throwaway script in the session
   scratchpad, as for the earlier replays, not committed. Score the strengths against the 20-session forward returns as
   above.
   - The bar (my proposal): over the two runs pooled, `strong` beats `standard` and `standard` beats `weak` in mean
     20-session excess (pooled group means), and the mean of the per-review strong - weak spreads (the scorer's rule:
     a review missing either group is skipped) is at least +3 pts (the bull-high gap measured on the same runs). Each
     run's figures are reported too, and the pooled group-mean gap beside the per-review one. Also reported: the share
     of each strength (a judge that almost never says `strong` is a wording failure, whatever the returns), how often the new judge's `in`/`out` differs from the logged verdicts, and rejections or retries.
   - Each review is replayed once per run (temperature 0.3, as the runs): about 108 judge calls, a night of GPU time
     on the user's vLLM server (Qwen 3.8 at `localhost:8010`), started by the builder; the two runs give each date two
     independent judgments. (My call.)
   - A replay that misses the bar allows **one** revision of the judge's strength wording (the user's decision), then
     a second replay by the same rules. The revision is written from what the first replay showed (which strength
     was wrong, and how), and the report says that the second replay is tuned on the same runs, so its pass is
     weaker evidence. A second miss stops the work before any backtest; the branch is not merged.
2. **Backtests.** The number of runs and the window are decided after the replay. Single runs cannot judge this
   change (two identical-code runs differed by 29 pts); the first year of `BI_YEAR` (2024-09 to 2025-09) is out of sample
   for every study so far.

**Order of work (my call).** All work happens on a feature branch, squash-merged into `main` only after the user's go.
The judge's strength (handoff and prompt) is built first, then the replay runs on that code; sizing, the previous
strength with state version 3 (the replay cannot exercise it: the logged runs have no strengths), logs, the scorer and
the `CLAUDE.md` update follow only if the replay meets the bar.

## Left to the builder

- Exact wording of the judge prompt and of the rejection messages, within the decisions above.
- Where the strength-to-weight function lives (a pure function in `sizing.py` replacing the inverse-volatility one is the
  natural place) and its tests.
- Test updates for the removed parameters, the new validation, the state migration and the scorer.
