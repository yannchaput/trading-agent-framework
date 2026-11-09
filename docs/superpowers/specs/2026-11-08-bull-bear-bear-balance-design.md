# bull_bear: the bear rates "high" as a balance, and the judge stops treating it as a veto

Date: 2026-11-08. Branch `worktree-bull-bear-bear-balance`. Evidence: the finished 5Y point-in-time run
`logs/bull_bear/backtesting/2026-11-05_012455_backtesting` (262 reviews, 2021-09-20 to 2026-10-05, +468.5% vs SPY +90.9%).

## Intent

The bear is the bull's counterpart. Its job is to bring balance by arguing, not to get stocks refused: a "high" risk
must mean there is a real reason, not that the fact sheet shows any decline. Today "high" is common enough (15% of
ratings, 11-18% every year) but carries no information, and the judge treats it as a veto.

What the 5Y run shows (20-session forward return, excess over the review's candidate mean, from `bull-bear-scores`):

| bear risk | n | 20d excess | in worst 10% | in best 10% | judge out |
|---|---|---|---|---|---|
| low | 1765 | +0.9% | 9.4% | 12.5% | 7% |
| medium | 1861 | -0.9% | 14.8% | 11.6% | 52% |
| high | 647 | +0.2% | 13.0% | 13.9% | 97% |

- "high" stocks are not worse than average and land in the best decile more often than any other level. 431 of the 647
  were ranked top 10 by momentum.
- The same argument behaves oppositely by level: a runway argument rated medium flags losers (-5.3%, 25% in the worst
  decile); rated high it does not (-1.4%). A revenue-decline argument rated high averaged +2.4% with a 7% worst-decile rate.
- With a bull rated medium, a bear "medium" sends the stock out 48% of the time and those stocks are the real losers
  (-1.2%, 15.1% worst decile); a bear "high" sends it out 92% of the time and those stocks are average (0.0%, 10.5% worst
  decile, below the 12.3% base). The veto removes average stocks while half the true losers stay in.

Root cause: `BEAR_SYSTEM` (`prompts.py`, the "Rate a stock high only on evidence in its fundamentals: a decline in
revenue or operating margin, a fall in free cash flow or a rise in debt" sentence) is meant as a limit but reads as a
sufficient condition: any year-on-year decline qualifies, with no size, no recency and no weighing. In a momentum universe
full of fast-growing, loss-making companies that is most debates, and the market has often already seen the filing and
bid the stock up since. `JUDGE_SYSTEM` then puts out a stock on exactly that evidence.

## Goals

1. **High as a balance.** The bear rates high only when a deterioration is large (a coded field past a stated threshold,
   or a serious company-reported event), recent (the market has not already looked past it) and clearly outweighs the
   strengths the same fact sheet shows.
2. **No veto.** The judge weighs a high risk against the bull's case on the evidence each gives; the bear's level is an
   argument, not a verdict.
3. **Coded fields for "large" and "recent".** Code computes the size of the changes and what happened since the filing;
   the prompt compares them with thresholds. Why code: a local model applies "large" and "recent" inconsistently across
   15-25 stocks and across reruns, but compares a number with a threshold reliably; this project already had to move the
   runway (`runway_months`) and `momentum_exhaustion` (-15 / -10) into coded numbers after prompt words failed. And the
   key fact, how the stock moved since the filing, cannot be computed by the model at all: the fact sheet has the filing
   date but no price history from it.

## Non-goals

- The researcher, the bull's prompt (it only gains the new field descriptions, shared by every agent), sizing, the book
  size, cooldowns and the rebalancer do not change.
- No sector cap (a separate design call; the 5Y study found it not free). No cash sweep (dashboard work elsewhere).
- No change to the "news from others is never high" rule, nor to `momentum_exhaustion`.
- Reviews do not start saving their inputs; the replay rebuilds them from the run log.

## Anti-goals (a failure even if it technically works)

- A toothless bear: high collapses to almost nothing, or every stock comes back low. The bear must still argue.
- More highs, or a high share that swings with reruns as much as before.
- A judge that now keeps the stocks the bear rightly flags (a large, recent, company-reported deterioration).
- Look-ahead: any new field using information dated on or after the review day.

## The change

### A. New fact-sheet fields (code, in `fundamentals.py` / `fact_sheet.py`)

Added to the fundamentals block, each absent when it cannot be computed (same rule as the existing fields):

| field | definition |
|---|---|
| `operating_margin_change_pts` | `operating_margin_pct - operating_margin_prior_year_pct`, in percentage points, one decimal |
| `debt_growth_pct` | `total_debt / total_debt_prior_year - 1`, in percent, one decimal; absent when the prior debt is 0 or missing |
| `days_since_filed` | calendar days from `fundamentals_filed` to the review day |
| `return_since_filed_pct` | the stock's return from the last adjusted close strictly before `fundamentals_filed` to the last close the fact sheet uses; absent when that close is not in the stock's history |

Rules:
- Same point-in-time rule as today: only filings dated before the review day (the existing cutoff); the prices are the
  closes the fact sheet already reads (`completed_bars`), never a later bar.
- `return_since_filed_pct` uses the close **before** the filing day because a filing is dated by day and often lands
  after the close; the filing day's own move is then counted as the market's reaction to it.
- `_FIGURES` (the shared fact-sheet paragraph every agent reads) describes the four fields in one or two sentences each,
  in the same style as the existing ones.
- `fact_sheet_row` stays pure. The filing date lives in the fundamentals, the closes in the momentum row: the builder
  decides how the dated closes reach the computation (the momentum row carries closes without dates today).

### B. The bear's high rule (`BEAR_SYSTEM`)

Replace the sentence that lists the declines with a rule built on the fields. Content to keep, wording left to the
builder (stay close to this, keep the numbers):

- High needs **all three**:
  1. **A large deterioration**, one of: `revenue_growth_pct` of -10 or below; `operating_margin_change_pts` of -10 or
     below **when `operating_margin_prior_year_pct` is -50 or above** (a company with an operating business, not a
     tiny-revenue firm whose margin swings by hundreds of points); `debt_growth_pct` of +50 or above while free cash
     flow is negative; `runway_months` below 12; or a serious fact the company itself reports or files, found in the
     note (a guidance cut, going-concern doubts, a covenant breach, a restatement, an impairment, a share issuance).
  2. **Not already weighed by the market**, for the four fact-sheet triggers only: when `return_since_filed_pct` is +20
     or above, the market has seen that filing and bid the stock up since; the risk can be medium, not high. When
     `return_since_filed_pct` is absent, this condition counts as met (no sign the market looked past it). A serious
     company-reported fact from the note is a discrete event, not a year-on-year drift, and is not subject to this test.
     `days_since_filed` is context for the agents; no rule uses it.
  3. **Balance**: the risk clearly outweighs the strengths the same fact sheet shows (growth, margin, cash flow,
     runway). Write it as a balance, never as an absolute ("no risk", "always").
- A smaller or older decline is still a risk to argue: rate it medium and name it. Medium is where most real risks belong.
- Keep: news from others is never high by itself; `momentum_exhaustion` (-15 / -10); "a fallen price with intact cash
  flow makes a company cheaper, not riskier"; "use none with low when you find no concrete risk"; "do not give every
  stock the same rating".

### C. The judge (`JUDGE_SYSTEM`)

Replace "Put a stock out when the bear names a risk shown in its fundamentals (...) that outweighs the bull's case" with
the same evidence rule but no mapping from the bear's level to a verdict. Content:

- The bear's risk level is its argument, not a verdict. Weigh a high risk the way you weigh a medium one: on the
  evidence the bear gives (the fact sheet's fields, the company's own reported facts), against the bull's case.
- Put a stock out when that evidence outweighs the bull's case. A deterioration the market has already weighed
  (`return_since_filed_pct` of +20 or above) weighs less.
- Unchanged: the strength scale and its balance wording, "a weak stock goes out", `max_in` as a ceiling,
  `previous_verdict` and `what_changed`, news from others is no reason for out, memory tools.

### D. Constraints

- Prompt word bans hold: no `held`, `weight_pct` or `forced` in any prompt (`test_no_prompt_explains_held_or_weight`).
- Prompt length, measured on 2026-11-08 with the Qwen 3.8 server's tokenizer over the 5Y run's 262 logged contexts per
  stage: the four fields cost 49 tokens per stock; the `_FIGURES` descriptions add 77 tokens to every system prompt, the
  bear's rule about +131, the judge's about +26 (draft wording). Largest call (21 stocks): bull 12,315 -> 13,421
  (+9.0%), bear 12,594 -> 13,831 (+9.8%), judge 17,899 -> 19,031 (+6.3%); researcher +126 per call. At 25 stocks the
  fields add about 1,225. The judge's 19k + 45,056 cap is about 64k of the 81,920 window, under the 65,536 compaction
  line. Record the measured judge prompt size in the replay report.
- Prompt tests (`tests/strategies/bull_bear/test_bull_bear_prompts.py`) pin the new rule's key sentences and numbers;
  field tests cover each new field's computation, absence rules and the no-look-ahead cutoff (a filing on the review
  day is not used; the close before the filing day).
- `bull_bear/CLAUDE.md` is updated with the new rule, the fields and why (this document's evidence in two lines).

## Validation: replay first, keep/drop rule fixed now

### Replay

- **Sample, fixed before any run:** 24 reviews of the 5Y run, the review records at indices 5, 16, 27, ... (every 11th
  record of `reviews.jsonl` starting at index 5; 24 indices), regardless of what they contain.
- **Inputs:** each stage's context is rebuilt from the run log (the bull's and bear's "Make the ... case" messages, the
  judge's "Give your verdict" message: all 262 of each are in `backtest.log` in full). The researcher is not rerun; its
  notes are reused. Arm B adds the four new fields to every stock, computed by the new code as of the review date
  (SEC cache and Yahoo closes, filings before the review day); arm A keeps the logged fact sheets as they are.
- **Arms:**
  - **A (rerun arm):** the current prompts on the logged inputs: bull, bear, then judge on A's bull and bear. Measures
    rerun noise against the logged run.
  - **B (change):** the new prompts and fields: bull, bear, then judge on B's bull and bear.
  - Same model and settings as the run (Qwen 3.8 27B AWQ, the run's thinking budgets and output caps, temperature from
    `BullBearParams`). The judge's memory tools start empty in both arms (the run's memory is gone); `previous_verdict`
    stays as logged.
- **Throwaway script** (precedent: the 2026-10-26 replays), not a repo command. Outcomes for scoring: the 20-session
  excess from `agent_scores.forward_returns` (per stock; never compounded into a portfolio path, since 5-session legs
  overlap in holiday weeks).
- **Anti-goal list, chosen by rule before the run, from the whole run (all 262 reviews), not only the sample:** every
  stock whose arm-B fields show `revenue_growth_pct` <= -10, or `operating_margin_change_pts` <= -10 with
  `operating_margin_prior_year_pct` >= -50, with `days_since_filed` <= 45 and `return_since_filed_pct` < 0. If more
  than 12 qualify, take 12 with a fixed seed (`random.Random(20261108).sample`, over the list sorted by review date then
  symbol). The script prints the list before any agent runs. For each listed stock's review not already in the sample,
  only arm B's bear is run (one call per review; its ratings of other stocks are ignored).

### Keep/drop rule

Keep the change (and move on to a backtest) only when **all** of these hold; otherwise drop it and report why:

1. **The bear still argues:** arm B's high share is between 4% and 15% of bear ratings, and its low share is not above
   arm A's by more than 15 points.
2. **No veto:** among stocks the bull rated medium or high, the judge's out rate on bear-high is below 75% in arm B
   (the run's was 90%; report arm A's).
3. **Real risks still flagged:** the bear rates high at least half of the anti-goal list in arm B. With fewer than 6
   listed stocks in the whole run, this test cannot be run: it does not pass, and the report says so (the change is
   not kept on rules 1, 2, 4 and 5 alone without the user's say).
4. **An effect above noise:** the share of verdicts that differ between B and A is larger than the share that differs
   between A and the logged run.
5. **No harm in the flips:** the mean 20d excess of the stocks B puts in and A puts out is at least the mean of the
   stocks A puts in and B puts out. An empty group counts as 0 (the excess is relative to the review's candidate
   mean, so 0 is "average").

Reported but not gating (too few stocks to be reliable on 24 reviews): arm B's high group's worst-decile and
best-decile rates, its 20d excess by level, and the bull x bear table of out rates.

### After a keep, or a drop

- **Keep:** a backtest on the new prompts, results not promised. Which window, and whether to compare against the 5Y
  run or a shorter point-in-time run, is decided with the user after the replay report (a 5Y run took about 63 hours).
  The bear's level table above is rebuilt for the new run by the same throwaway method (per-stock 20d excess from
  `agent_scores.forward_returns`); extending `bull-bear-scores` is not part of this work.
- **Drop:** the branch is not merged, fields included (the fields were only validated together with the prompts). The
  replay report and this document stay on the branch for the record.

### Scope of this work

The four fields with their tests, the bear and judge prompt changes with their tests, the `_FIGURES` descriptions,
the `bull_bear/CLAUDE.md` update, and the throwaway replay script with its report. Not the backtest.

## Calls made while writing this up (not discussed with the user)

- The user approved the three levers and asked for the design without a question round; the thresholds below are mine.
- "Large": revenue -10%, margin -10 points, debt +50% with negative free cash flow, runway under 12 months. Round
  numbers, not fitted to the 5Y data, so the replay is not judged on numbers chosen from it.
- "Already weighed": a rise of 20% or more since the filing. Same reasoning.
- Four fields, not more: `free_cash_flow` change was left out (its period can differ from revenue's, and both values
  are already on the sheet).
- The replay sample (24 reviews, every 11th from index 5), the arms, and the five keep/drop tests and their limits.
- Arm A uses the logged fact sheets; arm B the same plus the four fields, so the only differences are the fields and
  the prompts.
- The judge's memory starts empty in both replay arms.
- The margin trigger needs a prior-year margin of -50% or above. Applied to the 4,273 logged fact sheets, the four
  coded triggers alone hit 14.4% of stocks, and 197 of the 302 margin hits were on companies with a margin below -50%
  in one of the two years (tiny revenue), which would make the trigger noise.
- The "already weighed" test applies to the coded triggers only; a serious company-reported fact from the note is not
  subject to it (the model cannot compute a move since an arbitrary date). A missing `return_since_filed_pct` counts
  as not weighed. `days_since_filed` is shown, not used by a rule.
- The anti-goal list comes from the whole run (up to 12 stocks, fixed seed), with a bear-only replay on reviews outside
  the sample; fewer than 6 means test 3 fails as "not testable". An empty flip group counts as 0 in test 5.
- A drop discards the fields with the prompts. `bull-bear-scores` is not extended.

## Left to the builder

- Exact prompt wording within the content above; field names may change if the tests and `_FIGURES` follow.
- How dated closes reach the `return_since_filed_pct` computation while `fact_sheet_row` stays pure.
- The replay script's structure, how it parses the logged contexts, and how it runs the agents outside a strategy.
- Task breakdown and order (TDD: field tests and prompt tests first).
