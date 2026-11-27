# bull_bear paper run: judge vs rank-only top 10 (pre-registration)

TODO Bull Bear item 3. Written 2026-11-27, before any paper-run number was computed.

## Why

Every judge-vs-top-10 figure so far comes from backtests whose dates sit inside Qwen 3.8's training data
(knowledge edge ~2025-04), so a gap there could be memory rather than skill. Paper reviews after the cutoff are
the only truly out-of-sample test. The result decides the strategy: **if the judge does not beat the top 10,
bull_bear switches to rank-only.** Rank-only is the default; the judge must earn its GPU hours.

## Goals / non-goals

- Goal: one fixed rule, one interim read now (not a verdict), one final read at >= 3 months of paper reviews
  (about 13 weekly reviews, first possible date 2027-01-19).
- Non-goal: production code, a CLI or tests. The analysis is a scratch script reusing
  `strategies/bull_bear/batch/agent_scores.py` (`forward_returns`, `scored_reviews`, `newey_west_t`) and
  `scores_cli.py` (`read_reviews`, `fetch_opens`), as in item 1's `leakage_split.py`.
- Anti-goal: moving the rule after seeing the interim numbers. The rule below is frozen by this commit.

## Data

- Source: every `logs/bull_bear/paper/*/reviews.jsonl` (each restart writes its own file), non-abandoned reviews.
- One review per ISO week: the last one (2026-11-23 and 2026-11-24 are both present; 11-24 is kept).
- Shadow book: the first 10 symbols of `debate_set` by `rank`. Judge book: `book`. Both equal weight.
- Entry at the open of the session after the review (same as `bull-bear-scores`), raw forward return.
- **Live runs are excluded** (changed from the conversation's proposal to pool paper and live per date): since
  2026-11-23 the live book is 5 names from a 7-8 name debate set, a different strategy from paper's 10.
- Known confound, accepted: the judge's code changed during the window (`recent_verdicts` removed after
  2026-10-27, news toggled 2026-11-27). The test scores "the judge as it ran on paper", not one fixed version.

## Rule (frozen)

Per weekly review, gap = mean raw return of the judge book - mean raw return of the shadow top 10.

- **Primary:** 5-session gap, Newey-West t (0 lags at 5 sessions, as in `leakage_split.py`).
- **Keep the judge** only if the primary mean > 0 and t >= 1, **and** the judge's compounded weekly book ends
  ahead of the compounded shadow book over the same reviews (chain of 5-session returns).
- Primary passes but compounded shadow ends ahead -> "inconclusive", which means rank-only (the default).
- Anything else -> **switch to rank-only**.
- 20-session gap (lags 3): reported as a check only; it can neither keep nor drop the judge.
- Final read needs >= 13 scored weekly reviews with a 5-session forward window. Before that, numbers are interim.

Expected behaviour (weekly gap sd ~2.2% from the 2Y backtest, n = 13): a judge with no edge passes ~16% of the
time; one with the backtest's +0.95%/week edge passes ~73%.

## Approaches

- Scratch script + this doc (chosen): no code to maintain for a 2-3 use analysis; the doc carries the rule.
- `bull-bear-scores` CLI section with tests: one command in January, but production code the item says it
  does not need; rejected by the user.
- Pre-register only, no interim: avoids any peek, but leaves the pipeline untested until January; rejected.

## Left to the builder

The script's location (session scratchpad, path recorded with the results) and the interim report's format
(append a dated "Interim read" section to this doc, numbers only, no rule change).

## Interim read 2026-11-27 (not a verdict)

Rule frozen before this read: the text above had sha256 `5e2d0e04...c44a` at 2026-11-27T22:50:14+01:00, before any
paper number was computed (first commit made after the read, with `git add -f`: `.git/info/exclude` ignores `docs/`).

6 weekly reviews (2026-10-20 .. 2026-11-24); 11-24 not matured yet. Gap = judge book - rank top 10.

| Review | Judge 5d | Top 10 5d | Gap |
|---|---|---|---|
| 10-20 | +0.44% | +0.64% | -0.20% |
| 10-27 | -13.70% | -12.01% | -1.69% |
| 11-03 | -0.96% | +0.59% | -1.56% |
| 11-10 | -5.85% | -7.41% | +1.56% |
| 11-17 | -13.10% | -7.98% | -5.12% |

- 5 sessions (primary): mean gap -1.40%, t -1.28; compounded judge -29.77% vs top 10 -24.10%. 5/13 reviews.
- 20 sessions (check): mean gap -3.99%, t -2.53, but only 2 reviews with overlapping windows: no weight.

Script: `scripts/bull_bear/paper_judge_vs_top10.py` (reuses `forward_returns`, `newey_west_t`, `fetch_opens`).
Final read: at >= 13 weekly reviews with a matured 5-session window.

## Amendment 2026-11-27 (after the interim read, approved by the user)

The final read scores **exactly the first 13** weekly reviews (2026-10-20 onward), not ">= 13": rerunning at 14 or 15
could otherwise give a different answer. The thresholds are unchanged.
