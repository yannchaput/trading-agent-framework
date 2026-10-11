# bull_bear: realign the debate (held-neutral judge, verdict memory, shared SEC data)

Date: 2026-10-11 · Amends `2026-10-11-bull-bear-veto-judge-design.md` and `2026-10-09-bull-bear-strategy-design.md`
(both kept as history) · Branch `feat/bull-bear-debate-realignment` (worktree `../trading-agent-framework-realign`).

## Goal (the user's definition of the strategy)

- The **researcher** gathers the signals on each candidate: SEC filings and fundamentals, news, ...
- The **bull** and the **bear** receive the same researcher package; the bull argues for each candidate, the bear
  against.
- The **judge** receives both cases and decides, for each candidate, whether it belongs in the portfolio. **Whether a
  stock is held must not change the judge's behaviour.**
- A judge that sells a holding and buys it back soon after is the area to improve.

## Why

Audit of the judge's sells of held stocks (73 sells: 67 drops by the picker in the 1Y run
`logs/bull_bear/backtesting/2026-10-10_085928_backtesting`, 6 vetoes in the cancelled veto-judge run
`2026-10-11_143148_backtesting`). Metric agreed with the user: the sold stock against the stock(s) bought into its
slot (equal weight; the kept book when nothing new was bought), Yahoo closes, 1 and 4 weeks.

- **The sells are a coin flip.** 4 weeks: 51% good calls, -0.8 points mean edge (1 week: 64%, +0.6).
- **Half the sells were bought back** (33 of 67, median 2 weeks, 24 within 4 weeks), and those round trips lost on
  both legs: the sale (-3.6 points vs +2.8 for sells never re-bought) and the re-entry (the stock then lagged the
  candidates not picked by 4.2 points over 4 weeks, n = 24).
- **The worst sells react to fresh news the price already absorbed**: `news_event` sells -6.5 points (n = 16, the
  stock already down 8.6% that week); sells after a >10% weekly drop -9.2 (n = 9); sells whose newest article was from that day or the
  day before -4.3 (n = 33) vs +4.0 when the researcher found no article (n = 18).
- **The SEC data the agents read is wrong.** The agent tools' `debt` takes the current-portion tag first
  (`fundamentals/sec.py` `BALANCE_SHEET_TAGS["debt"]`; tag order only breaks ties in `_candidate_sort_key`): of 172
  cached us-gaap filers, `debt` matches total debt for 26, differs for 57 (median 13% of the total, often 0) and is
  missing for 38 that report a total (approximate counts: "total" uses the screen's tag preference). There is no cash-flow statement, and each field is a single latest value, so
  the researcher prompt asks for a trend the tools cannot show. Ticker `B` returned 2011 filings.
- bill_ackman's **screen** already resolves total debt correctly and reads cash flow
  (`strategies/bill_ackman/screen/annual_figures.py`); its **agents** share the broken tools.
- Code that departs from the goal: the fact sheet shows `held` and `weight_pct` to all four agents; the judge prompt
  says "A held stock you veto is sold"; `build_book` puts kept holdings first and cuts the kept names at `book_size`
  by rank (1-5 kept names cut every review in the veto run), so "keep" is not "in the portfolio"; the veto cap forces
  at least 10 keeps; the judge has no memory of its verdicts.

All samples are small (one run per design, t-stats mostly below 1); the evaluation below is the real test.

## Decisions (binding)

| Topic | Decision |
|---|---|
| SEC data | One definition in `fundamentals/sec.py` (total debt, cash flow), used by the agent tools and by bill_ackman's screen; the screen's output must not change; IFRS in the screen is out of scope |
| Researcher | Same tools, `research_tool_budget` 3, `note_max_chars` 500; statements gain the prior-year period; new switch `research_news` (default on) |
| Fact sheet | Drops `held` and `weight_pct` for all four agents; adds `return_1w_pct` |
| Candidates | The top `candidate_count` (15) by rank, plus holdings ranked up to `retention_rank` (35), minus stocks in cooldown |
| Code rules before the debate | Unchanged: a holding ranked worse than 35 or unranked is a forced exit; `max_volatility` gates new entries only |
| Judge | A verdict `in` or `out`, with a reason, on every candidate; at most `max_book_size` (10) `in`; no veto cap; held-blind |
| Book | Every `in` name; no held-first rule, no rank cut; fewer than 5 names parks the remainder in SHV |
| Verdict memory | Code-injected `previous_verdict` + required `what_changed` on a flip (bill_ackman's pattern), plus the memory tools `search_memory` and `remember_lesson` |
| Feedback | Every `out` verdict of the last 4 reviews, with its outcome since, shown to the judge |
| Cooldown | A holding the judge votes `out` is sold and leaves the candidate list for 4 completed reviews |
| Priced-in news | The bear and the judge are told an event the price already reacted to is not by itself a reason |
| Agents | All four stay; no agent gets an order tool |

## 1. Data: one SEC definition, a fuller researcher package

### `fundamentals/sec.py`

- **Total debt** moves here from `screen/annual_figures.py` (`_debt_sources`, `_summed`), with the same order of
  preference: `LongTermDebt`, `LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities`,
  `LongTermDebtNoncurrent` + `LongTermDebtCurrent`, `LongTermDebtAndCapitalLeaseObligations` +
  `LongTermDebtAndCapitalLeaseObligationsCurrent`, `DebtLongtermAndShorttermCombinedAmount`. It is defined once and
  takes the forms to read, so the screen keeps 10-K/10-K/A only and the agent tools also read 10-Q.
- **Cash flow** tags move here from `ANNUAL_FLOW_TAGS` (`operating_cash_flow`, `capex`), with `ifrs-full`
  equivalents for the agent tools (tag names checked against real payloads during implementation, as for the IFRS
  change).
- `screen/annual_figures.py` imports both instead of defining them. **Gate:** its records do not change. Verified by
  reducing every cached company-facts payload (`cache/sec/companyfacts/`, 183 files; the screen does not keep its
  own payloads) with the old and the new code and comparing the results, plus the screen's unit tests. If any
  record differs, implementation stops for a decision; no `SCHEMA_VERSION` bump otherwise. Optional, by hand: refetch
  every universe payload and compare all records.

### Agent tools (`agents/tools/fundamentals.py`, shared by bull_bear's and bill_ackman's agents)

- `get_balance_sheet`: `debt` is total debt.
- `get_income_statement`: adds `operating_cash_flow`, `capex` and `free_cash_flow` (operating cash flow minus capex).
  Every field carries its period (`start`, `end`) and the same-length period ending about one year earlier
  (`prior_year`), so year-over-year change is visible and a quarter cannot be read as year-to-date (10-Q cash flows
  are year-to-date). No extra tool call.
- Both statements: a result whose newest period ended more than 18 months before `as_of` carries `stale: true`.
  Before adding it, check whether the IFRS change already fixed `B` (Barrick files 40-F).
- `as_of` is still `strategy.clock.now()` (no wall-clock time in a data tool).

### Researcher

- Tools unchanged (`search_news`, `get_income_statement`, `get_balance_sheet`), `research_tool_budget` 3,
  `note_max_chars` 500. The prompt uses the prior-year figures and no longer asks for a trend the tools cannot show.
- `research_news` (default `True`). `False`: the researcher gets no `search_news`, its prompt drops the news
  sentence, and `initialize` skips `_require_news_provider`. This is the switch for the news on/off comparison.

### Fact sheet (`fact_sheet.py`, all four agents)

- Drops `held` and `weight_pct` (and the prompt sentence explaining them).
- Adds `return_1w_pct`: the return over the last 5 sessions.

## 2. Debate, judge and book

### Candidates (code, `debate_set.py`)

- The top `candidate_count` (15) by rank, plus every holding ranked up to `retention_rank` (35), minus the stocks in
  cooldown. Worst case 25 (15 + 10 holdings); typically 15-18.
- Forced exits and the `max_volatility` asymmetry are unchanged. A forced exit starts no cooldown.
- An empty candidate list abandons the review (stage `data`), as a too-small debate set does today.

### Bull and bear

- Same flow, same evidence (fact sheet + note), neither sees the other.
- Bear prompt, new rule: an event the price has already reacted to (see `return_1w_pct`) is not by itself a high
  risk; name the fundamental consequence that would make it one.

### Judge context

Per candidate: the fact sheet, the note, the bull's case, the bear's case, and `previous_verdict` (`verdict`,
`reason`, `date`) when the judge has judged that stock before (its latest verdict on record).

Once per review:

- **Recent verdicts**: every `out` verdict of the last `feedback_reviews` (4) completed reviews, sells and names not
  bought alike: symbol, date, reason, the stock's return from the last close known at that review to the last close
  known now, and the mean return of that review's `in` names over the same span. Prices are closes up to `clock.now()` from the bars the review already loaded; an
  entry without a price is left out, never estimated.
- **Memory tools**: `search_memory` and `remember_lesson` (`memory/tools.py`, through `strategy.memory`; wiped at
  the start of every backtest), with `judge_memory_budget` (6) calls; `submit_verdicts` is exempt. The prompt: look
  up your lessons before deciding; when a recent outcome teaches something, record a lesson.

Judge prompt changes: drop "A held stock you veto is sold"; state the cap ("at most max_book_size in; when more
deserve it, keep the strongest cases"); the priced-in rule as for the bear; start from `previous_verdict` and change
it only for a new, material fact.

### `submit_verdicts` (`handoff.py`)

One object per candidate, exactly once: `symbol`, `verdict` (`in` or `out`), `reason`, and `what_changed` when the
verdict differs from `previous_verdict` (all texts at most `reason_max_chars`). Rejected, with an error the model
can act on: a missing or extra symbol, more than `max_book_size` `in`, a flip without `what_changed`. The existing
forced retry applies; no valid submission after it abandons the review with no order.

### Book and sizing (code)

- The book is every `in` name (`book.py` reduces to that; the held-first rule and the rank cut go).
- Inverse-volatility weights in [`min_weight`, `max_weight`] = [4%, 20%] (`sizing.capped_inverse_volatility`) with
  `total = min(investable, n x max_weight)`; `target_portfolio` parks the remainder in SHV. Zero `in`: everything is
  parked.
- A holding voted `out` is sold (by the `Rebalancer`, with the forced exits) and starts a cooldown of
  `reentry_cooldown_reviews` (4) completed reviews.

### State (`state.py`, `StateStore`)

- `last_completed_review`, `abandoned_streak` (unchanged); `verdicts` (per symbol: latest `verdict`, `reason`,
  `date`); `cooldowns` (symbol to completed reviews left); `history` (the last `feedback_reviews` reviews: date and
  each candidate's verdict). `last_book` goes.
- `STATE_VERSION` bumps; an older file (picker or veto judge) loads as an empty state. Backtests wipe it in
  `initialize` (unchanged); paper/live start without previous verdicts.
- An abandoned review changes nothing (no verdict stored, no cooldown counted down; bill_ackman's rule), except at
  the execution stage, where the judge has decided and orders may be out: its verdicts and the cooldown of each sell
  actually placed are stored (running cooldowns are still not counted down, and no history entry is added).

### Parameters (`BullBearParams`)

| | Field | Default | Meaning |
|---|---|---|---|
| removed | `book_size`, `max_vetoes`, `debate_size` (property) | | the fixed book and the veto cap |
| new | `candidate_count` | 15 | top ranks debated every review |
| new | `max_book_size` | 10 | most `in` verdicts per review |
| new | `reentry_cooldown_reviews` | 4 | completed reviews a judge-sold stock stays out of the candidates |
| new | `feedback_reviews` | 4 | reviews whose `out` verdicts are shown with their outcome |
| new | `judge_memory_budget` | 6 | memory tool calls per judge run |
| new | `research_news` | `True` | the researcher's news tool |
| kept | `retention_rank` | 35 | must stay above `candidate_count` |

`__post_init__` checks every rule: `candidate_count`, `max_book_size`, `feedback_reviews` at least 1;
`reentry_cooldown_reviews` and `judge_memory_budget` at least 0; `retention_rank > candidate_count`;
`max_book_size x min_weight <= investable`; the existing weight, band and buffer rules. The rule that
`book_size x max_weight` reaches the investable total goes (the book may now be smaller than 5).

## 3. Logging, errors, testing, evaluation

### `reviews.jsonl` (one line per review)

Each verdict with `previous_verdict` and `what_changed`; `verdict_counts` (`in`/`out`); `book`; book size and parked
weight; cooldowns started and active; the recent-verdicts block as the judge saw it; memory calls and the text of
every lesson written. `veto_cap` and `judge_verdicts` go. Nothing outside bull_bear reads these fields (the
dashboard does not). `settings.json` carries `research_news` through the parameters.

### Errors

- A rejected `submit_verdicts`: the tool's error, then the forced retry; still invalid, the review is abandoned with
  no order and the rejection is logged.
- A memory database failure surfaces as an agent error and abandons the review (today's path).
- A feedback entry without a price is dropped.

### Tests (hand-written fakes, no network)

- `fundamentals/`: the total-debt resolver (the screen's debt tests move with it), cash flow, `prior_year`, the
  `stale` flag, the `ifrs-full` equivalents.
- bill_ackman: the screen's existing tests pass unchanged.
- bull_bear: fact sheet (no `held`/`weight_pct`, `return_1w_pct`); `submit_verdicts` (cap, every candidate once,
  `what_changed` on a flip); the book from `in` verdicts; sizing under 5 names and with none; the cooldown (starts
  on a sell, counts completed reviews only, removes the stock from the candidates); the feedback block reads no
  price after `clock.now()`; `research_news=False` (no news tool, no provider check); parameters; state loading an
  older file; pipeline tests rewritten for `in`/`out` (the book and veto tests go).

Manual, reported in the merge: the screen-record gate above, `scripts/tests/smoke_quality_screen.py`, and `B`.

### Evaluation (the user runs the backtests)

- Baseline: the picker run `2026-10-10_085928_backtesting` (1Y, 2025-09-29 to 2026-10-05). The veto-judge run was
  cancelled after 8 reviews; it informs the audit only.
- **Run A**: this design with news, same window. **Run B** (later): `research_news=False`.
- Read alpha and beta, not raw CAGR, against the ~3.8-point single-path noise floor; then rerun the sell audit on
  each run (same metric).
- Success: sells beat their replacement in more than half of the cases at 4 weeks with a positive mean edge; no
  re-entry within 4 reviews (by construction); the book holds 5-10 names on most reviews; validation rejections
  abandon a review rarely, not weekly.

### Docs

`strategies/bull_bear/CLAUDE.md` rewritten; `fundamentals/CLAUDE.md` (the single debt definition, cash flow,
`prior_year`, `stale`); `strategies/bill_ackman/CLAUDE.md` (the screen imports the shared definitions; its agents see
the corrected tools); `README.md` where it lists bull_bear's parameters.

## Out of scope

- IFRS (20-F/40-F) filers in bill_ackman's screen (179 of 1,200 universe names; needs FX conversion and ADR ratios).
- SEC filing documents (`get_filings`, `get_filing_document`) for the researcher; a bigger tool budget or note.
- Removing the news tool (decided by run B, not here).
- Changes to cross_momentum, `scoring.py` or the `Rebalancer`.
- bill_ackman's RMD `fcf_yield` of 15.93 (1,593%) on 2021-09-20: a likely screen bug, noted, not investigated.
