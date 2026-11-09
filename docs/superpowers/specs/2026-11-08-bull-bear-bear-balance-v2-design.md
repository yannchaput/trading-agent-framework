# bull_bear bear balance, v2: same change, tests that measure the right thing

Date: 2026-11-08. Branch `worktree-bull-bear-bear-balance-v2`, cut from `worktree-bull-bear-bear-balance` at `537b7ff`.
Builds on `docs/superpowers/specs/2026-11-08-bull-bear-bear-balance-design.md` (v1) and its replay report
`docs/superpowers/replays/2026-11-08-bull-bear-bear-balance.md` (decision DROP).

## Intent

v1's replay dropped the change on two tests. Test 2 (no veto) failed with 3 of 4 stocks out, against a limit of
"under 75%". Test 3 (real risks still flagged) failed with 5 of 12 listed stocks rated high, against a limit of 6.
Checking the outcomes after the drop shows that **the change did what it should and the two tests were wrong**:

| group, whole 5Y run (20-session excess) | n | 20d excess | in worst 10% |
|---|---|---|---|
| every stock | 4273 | 0.0% | 10.0% |
| v1's anti-goal rule (large decline, filed in the last 45 days, down since) | 57 | +3.4% | 10.5% |
| the same, operating businesses only | 19 | +5.0% | 5.3% |
| runway under 12 months | 193 | -3.4% | 16.6% |
| logged bear high | 647 | +0.2% | 12.8% |

- Test 3's list was not a list of real risks: those stocks did better than average. v1's bear rated most of them medium,
  citing a long cash runway, and that call was arguably right.
- Of the coded signals, only a short runway predicts losers. That is the real risk the bear must keep flagging.
- Test 2 hinged on a group the change itself shrinks: under the new bear, a stock the bull likes rarely also gets a
  high. Four stocks cannot measure a judge.
- Within v1's own sample, arm B's highs were rare (5.4%) and were real losers (21 stocks, -8.0% 20d excess, against
  -1.7% for B's medium). The sample is small, so this is a hint, not evidence.

The user chose: keep v1's bear rule and judge wording unchanged, and fix the tests. v2 is a second, pre-registered
replay of the same code on reviews v1 never used.

## Goals

1. Re-test v1's change, prompts and fields byte for byte, on a fresh sample of the 5Y run.
2. Measure the judge's veto on a group large enough to read, by replaying the judge alone on logged arguments.
3. Judge "real risk" by an outcome-validated signal (runway under 12 months) and by whether the bear's high group
   actually does worse, not by a hand-made list.
4. Fix every rule, sample and threshold in this document and in a committed plan file before any agent runs.

## Non-goals

- No prompt, field or code change to the strategy. `BEAR_SYSTEM`, `JUDGE_SYSTEM`, `_FIGURES` and the four fields stay
  as committed at `407a310`. If a test fails, v2 drops: no tuning on the fresh sample.
- No backtest in this work. No change to `bull-bear-scores`.
- The v1 branch is not modified; it stays as the record of v1.

## Anti-goals

- Fitting: any threshold changed after the replay starts, or a sample reused from v1.
- A judge that, freed from the veto, keeps stocks with a short runway.
- A bear whose high group is no worse than its medium group (high has lost its meaning).
- The same failure as v1's test 2: a gating test that hinges on a handful of stocks. Every gating group has a stated
  minimum size; below it the test cannot be run and does not pass.

## The replay

### Reviews

- **Fresh pool:** the 262 reviews of `logs/bull_bear/backtesting/2026-11-05_012455_backtesting`, minus v1's 24 sample
  reviews and v1's 8 anti-goal-only reviews: 230 reviews.
- **Full sample (arms A and B):** 36 reviews, `random.Random(20261109).sample(sorted(pool), 36)`.
- **Judge sample (arms J-old and J-new):** the pool reviews whose logged record has at least one stock with bull
  medium or high and bear high (116 reviews), then 60 of them, `random.Random(20261110).sample(sorted(those), 60)`.
  It may overlap the full sample. A judge-only replay costs about half of a full one.
- The script prints both lists and writes them to the plan file, which is committed before any agent call.

### Arms

- **A** and **B** exactly as in v1: A is the old prompts on the logged inputs (rerun noise), B is the new prompts
  plus the four fields. Each runs bull, bear, then the judge on that arm's own bull and bear.
- **J-old:** the old `JUDGE_SYSTEM` on the logged judge input (the logged bull and bear arguments and logged fact
  sheets).
- **J-new:** the new `JUDGE_SYSTEM` on the same logged bull and bear arguments, with the four fields added to each fact
  sheet as in arm B.
- J-old and J-new isolate the judge: the bear's highs here are the old, frequent ones (15%), which is the case the
  "no veto" wording is for.
- Same model, settings, empty memory and logged `previous_verdict` as v1. Outcomes: per-stock 20d excess from
  `agent_scores.forward_returns`, never compounded.

### Cost

From v1's timings (7.1 to 7.9 minutes per full arm review, judge stage 2 to 4 minutes): about 9 hours for A and B,
about 7 hours for J-old and J-new, so 16 hours on the local server. A run that is stopped resumes, as in v1.

## Keep/drop rule (fixed now)

Keep only when **all** of these hold; otherwise drop and report why.

| # | test | arms | rule | minimum group |
|---|---|---|---|---|
| 1 | The bear still argues | A, B | B's high share is 4-15% of bear ratings, and B's low share is not above A's by more than 15 points (as v1) | none |
| 2 | No veto | J-old, J-new | among stocks with logged bull medium/high and bear high, J-new's out rate is below 75% **and** at least 15 points below J-old's | 30 stocks |
| 3 | The judge still removes real risks | J-new | J-new's out rate on stocks with `runway_months` < 12 is at least 80% | 10 stocks |
| 4 | High means risk | B | the mean 20d excess of B's bear-high group is below that of B's bear-medium group | 20 high stocks |
| 5 | Real risks still flagged | B | B's bear rates high at least half of the stocks with `runway_months` < 12 | 10 stocks |
| 6 | No harm in the flips | A, B | the mean 20d excess of stocks B puts in and A puts out is at least that of stocks A puts in and B puts out; an empty group counts as 0 (as v1) | none |

Below a minimum, the test cannot be run and does not pass, and the report says so.

Reported, not gating:
- v1's "effect above noise" (verdicts B vs A against A vs logged).
- J-old against the logged verdicts (judge rerun noise).
- Flips between J-old and J-new with their 20d excess.
- Bear level tables with deciles, and the bull x bear out-rate tables, for A, B, logged, J-old and J-new.
- v1's anti-goal rule applied to the fresh sample, with B's ratings (for comparison with v1).
- The largest judge prompt, and timings.

### Why these thresholds

- Test 2: in the logged run, the judge puts out 90% of stocks with bull medium/high and bear high, against 39% when
  the bear says medium. Below 75%, and 15 points below a rerun of the old judge, means the wording moved the judge,
  not rerun noise. 30 stocks is about 0.5 per judge-sample review; the pool holds 160 such stocks in 116 reviews.
- Test 3: the logged judge put out every runway-under-12 stock (193 of 193). 80% leaves room for a strong bull case
  without letting the veto's removal open the door to the worst group.
- Test 4: a direction test, not proof. With about 30 high stocks the difference is noisy. It catches a high that has
  lost its meaning, which is the failure v2 must rule out.
- Test 5: in v1's sample (not the v2 sample), B rated 12 of 19 runway-under-12 stocks high, A 18 of 19. Half is the
  floor for "the bear still flags the one coded risk that predicts losers". About 26 such stocks are expected in 36
  reviews.
- Thresholds 4-15%, 15 points and the flip test are v1's, unchanged.

## After the replay

- **Keep:** a backtest is the real check, because v2's tests were chosen knowing the 5Y outcomes. The window is decided
  with the user after the report. A window the agents have not seen (before 2021-09) would need the universe history
  extended back. The branch merges only after the user decides.
- **Drop:** neither branch is merged. The v2 report states which test failed. A third attempt would need a new design.

## Scope of this work

A throwaway replay script for v2 (`scripts/replays/bull_bear_bear_balance_v2.py`, reusing v1's script where it can),
its committed plan file, the replay, and the report at `docs/superpowers/replays/2026-11-08-bull-bear-bear-balance-v2.md`.
No change under `src/` or `tests/`.

## Calls made while writing this up (not discussed with the user)

- The user skipped the question round and approved "keep v1's bear rule and fix the tests". The samples, arms and
  thresholds are mine.
- The outcome study above was read before writing the tests; the tests are fixed before any agent sees the fresh
  sample. The study used the whole 5Y run, which overlaps the fresh sample, so a KEEP still needs a backtest.
- I read v1's runway numbers (12 of 19 rated high by B) before setting test 5's floor at half. That is v1's sample, not
  v2's.
- 36 full reviews, not 24: B's high share (about 5%) needs about 36 reviews to reach 20 high stocks for test 4.
- A judge-only pair of arms instead of a larger full sample: it measures the veto on the old, frequent highs at half
  the cost per review.
- v1's test 4 (effect above noise) is reported, not gating: tests 2 and 6 already require a real and harmless
  change.

## Left to the builder

- The script's structure, and how much of v1's script it imports or copies.
- How J-old and J-new build the judge input from the logged judge context (the logged message holds the bull and bear
  arguments).
- Task breakdown (TDD for the pure parts: the samples, the six tests, the minimum-size rule).
