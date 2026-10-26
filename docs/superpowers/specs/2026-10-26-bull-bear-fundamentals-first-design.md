# bull_bear: fundamentals first, news as support

Date: 2026-10-26

## Why

Qwen 3.8 is systematically worse than Qwen 3.6 on bull_bear (two 1Y runs at about +44% total return against seven 3.6
runs at +61% to +91%). Replays on 2026-10-26 ruled out reasoning effort for every agent and showed the judge is not the
weak link: given 3.6's bull and bear cases, the 3.8 judge reproduces 3.6's book. The difference comes from the debate,
mostly the bear.

The clearest case is MU on 2026-03-31. It dropped 20% in one week on a news story (Google's TurboQuant memory
compression) after a 666% rally, while its fundamentals were the strongest in the book: revenue up 3x on the year,
$8.5B free cash flow, supply short of demand.

- The 3.8 bear rated MU **high** in 6 of 7 runs, on 3.8's note and on 3.6's, at effort `medium` and `xhigh`. It used
  either `news_event` or `momentum_exhaustion` and always the same argument: TurboQuant is "a structural demand
  threat". The 3.6 bear rated the same news **medium**.
- The current prompts let a bear rate already-priced news high "when you can name its lasting consequence for the
  business". The judge puts a stock out on the same condition. Any confident sentence about the future passes that test.
- MU qualified for `momentum_exhaustion` (1-month −21.9%, drawdown −30.3%), but its whole one-month loss came from
  the last week (−20.4%): one news drop, not an exhausted trend.
- On identical input the 3.8 bear gave 2 to 8 high ratings out of 12 from one run to the next.
- The agents' only source of fundamentals is the researcher's 500-character note. The fact sheet holds prices only,
  so the fundamentals reach the debate only when the researcher writes them down.

MU was out of the 3.8 book on 03-31. By 05-05 it was up 57% on the month.

## What the user decided

- The bull's and the bear's cases rest on the stock's **fundamentals and financial health**. News only supports a
  case; it is never the case by itself.
- The fundamentals are **added to the fact sheet by code** (option B), not left to the researcher's note. Every agent
  sees the same exact figures.
- A **high** risk needs evidence in the fundamentals: a guidance cut, or a revenue or margin decline.
- **Facts the company itself reports or files count as fundamental evidence**, also when they reach the agents through
  the researcher's note: results in an earnings release, guidance, a share issuance (dilution), going-concern doubts,
  a covenant breach, a restatement, an impairment. Third-party news and opinion do not: a competitor's technology,
  analyst actions, headlines, speculation about consequences. (MU's TurboQuant stays news alone; ABVX's planned capital
  raise on 03-31 counts.)
- **SEC cache freshness belongs to the core framework (`fundamentals/`), not to strategy code.**
- **Momentum exhaustion** needs a decline over several weeks, not one week. The test is below.
- From bill_ackman's critic prompt, carried over:
  - "A fallen price with intact cash flow makes a company cheaper, not riskier." Goes into the bear's prompt.
  - "Sources can be wrong or stale: do not repeat a figure from a news item as fact when the fact sheet or a filing
    says otherwise."
- **The judge follows the same rule.**
- The researcher no longer needs to spend tool calls on the income statement, because the fact sheet carries it. It
  keeps its SEC tools for detail, such as reading a filing to explain a figure.
- **Valuation (P/E, FCF yield) is out of scope for now.**
- **Validation:** a replay first, then a 1Y backtest. No window-only backtest before this change.

## Goals

- Every fact sheet carries the stock's latest reported fundamentals against the same period a year earlier.
- A news event alone, however its future consequences are argued, never makes a high bear risk nor, by itself, a
  judge `out`.
- `momentum_exhaustion` is used only for a decline that ran for weeks before the last one.
- The bull argues from the same fundamentals.

## Non-goals

- Valuation fields (they need a split-adjusted market cap; bill_ackman's screen has that machinery; a later step).
- Changes to the pipeline's stages, the handoff tools' schemas, sizing, the book or the ranking.
- Changes to reasoning effort, thinking budgets or output caps. The per-agent effort spec
  (`2026-10-26-bull-bear-reasoning-effort-per-agent-design.md`) is shelved: replays showed no effect from effort.
- Any other strategy's fact sheet or prompts.

## Anti-goals

- **A bear that misses a real deterioration.** A stock whose revenue or margins are falling, or whose guidance was
  cut, while momentum still carries it must still be rated high. The new rules raise the bar for news; they must not
  make the bear unable to rate high at all.
- No code computes a rating or a verdict: the agents still decide. Code supplies figures only.

## Decisions (binding)

### Fact sheet: fundamentals (`strategies/bull_bear/fact_sheet.py` and its caller)

New fields on every debate stock's fact sheet, from SEC company facts **as of the strategy clock** (the same
point-in-time rule as the researcher's SEC tools: only figures filed by `strategy.clock.now()`):

- the latest reported period's **revenue growth** against the same-length period a year earlier, in percent;
- **operating margin** for that period and for the same period a year earlier, in percent (their difference is the
  margin decline);
- **free cash flow** for its period and a year earlier;
- **total debt** at the period end and a year earlier;
- the **period end** and **filing date** of the figures, so every agent sees how recent they are;
- a **stale** flag when the newest period ended more than 548 days before the clock (as `sec.is_stale_statement`);
- when no usable figures exist (no CIK, no XBRL facts, an SEC error), a field saying the fundamentals are unavailable,
  with a short reason, and none of the figure fields.

Rules:

- Reuse `fundamentals/sec.py`: `income_statement` (revenue, operating income, `free_cash_flow`, each with its
  `prior_year`) and `balance_sheet` (`debt`, with its `prior_year`). Same definitions as the researcher's tools, so
  the fact sheet and a tool call never disagree. IFRS filers (20-F / 40-F) come through the same functions.
- Ratios (growth, margins) are currency-free. Money amounts (free cash flow, debt) are given in billions with the
  reporting currency labelled, never converted (as the SEC tools do).
- A figure that cannot be computed (a missing field, no prior year, revenue at or below zero for a margin) is left
  out or `None`; it is never guessed. A partial set is fine.
- An SEC failure for one stock never abandons the review: that stock's fact sheet says the fundamentals are
  unavailable and the review goes on.
- The SEC payload comes through `SecEdgarClient` and its cache; how fresh it must be is the framework's rule (next
  section), not the strategy's. `SEC_EDGAR_USER_AGENT` is already required by bull_bear.

### SEC cache freshness (`fundamentals/`, framework)

- Today the agent tools refetch a cached company-facts or submissions payload only when it is older than 30 days by
  the strategy clock (`agents/tools/fundamentals.py`'s `MAX_PAYLOAD_AGE_DAYS`). In paper/live a 10-Q filed after the
  cached copy was fetched stays invisible for up to 30 days; with the fact sheet as the main evidence, that matters.
- The freshness rule moves into `fundamentals/` as one framework default that every caller of the company-facts and
  submissions payloads uses (the agent tools and bull_bear's fact sheet): a cached payload older than **1 day** by the
  strategy clock is fetched again (the user confirmed 1 day). bull_bear does not set its own value.
- Backtests are unchanged: `as_of` is in the past and a file fetched today is never older than it, so an existing file
  is never refetched (the rule of `freshness.is_stale`).
- Effect on the other strategies (paper/live only; backtests unchanged for all):
  - bill_ackman's researcher and short seller (`fundamentals_tools`): their SEC data is at most 1 day old instead of
    30. Its quality screen is untouched (its own annual store and `max_age_days`, fetched through the uncached path).
  - earnings_drift's agent (`get_filings`, `get_filing_document` from `fundamentals_tools`): its filings list is at
    most 1 day old instead of 30. Its event source keeps its explicit `max_age_days=0` on submissions.
  - insider_trades: untouched (its form 4 source keeps its explicit `max_age_days=0`; it has no SEC agent tools).
  - The only cost is more SEC downloads in paper/live, within the client's rate limit.
- A caller that passes a stricter explicit age (the `max_age_days=0` above) keeps it; the framework default applies to
  every caller that does not.
- `fundamentals/CLAUDE.md` and `agents/CLAUDE.md` describe the rule.
- The fact sheet stays computed by code and agents must not recompute it. `_FACT_SHEET` (the prompt paragraph every
  agent reads) describes the new fields.
- Whether a stock is held stays out of the fact sheet (unchanged rule).

### Fact sheet: momentum exhaustion

- New field `return_1m_skip_1w_pct`: the return over the month leaving out the last week (the 16 sessions before the
  last 5), one decimal like the other returns. `_FACT_SHEET` describes it.
- The bear uses `momentum_exhaustion` only when **both**: `drawdown_from_52w_high_pct` is −15 or below (as today), and
  `return_1m_skip_1w_pct` is −10 or below (the decline had already run 10% or more before the last week). It replaces
  today's "a negative one-month return" condition. A stock still rising is never `momentum_exhaustion` (kept).
- Example for the prompt or its tests: MU on 03-31 had −21.9% over the month, −20.4% in the last week, about −1.9%
  over the three weeks before: a one-week drop, not exhaustion.

### Prompts (`strategies/bull_bear/prompts.py`)

- **Shared (`_FACT_SHEET`)**: describes the new fields. Adds the sources rule: "Sources can be wrong or stale: do not
  repeat a figure from a news item as fact when the fact sheet or a filing says otherwise."
- **Researcher**: the fact sheet now carries the latest figures against a year earlier; the note looks for what
  explains them and what they cannot show (guidance, deals, lawsuits, management, analyst actions, filing text). It
  still must not repeat the fact sheet's numbers. The SEC tools stay available. Both variants (news on and off) change.
- **Bull**: the case rests on the fundamentals and financial health in the fact sheet; news and the note support it.
  `high` stays reserved for the strongest and most durable cases.
- **Bear**:
  - the case rests on fundamentals and financial health; news only supports it;
  - **high** only with evidence in the fundamentals: a revenue or operating margin decline, or a free cash flow or
    debt deterioration, from the fact sheet; or a fact the company itself reports or files, from the note (results in
    an earnings release, guidance cut, share issuance, going-concern doubts, covenant breach, restatement,
    impairment). Third-party news and opinion alone (a competitor's technology, a downgrade, a price-target cut, a
    lawsuit headline, speculation about consequences) is never high, whatever consequence the bear argues;
  - when the fact sheet's fundamentals are unavailable or stale (foreign filers with annual figures only, for example),
    the company-reported facts in the note are the evidence; such a stock can still be rated high;
  - "A fallen price with intact cash flow makes a company cheaper, not riskier.";
  - the new `momentum_exhaustion` test above;
  - the concern labels stay the same set (`valuation, momentum_exhaustion, earnings, fundamentals, news_event, sector,
    none`).
- **Judge**: same rule as the bear. Put a stock out when the bear names a concrete risk shown in the fundamentals (the
  fact sheet's figures, or a fact the company reports or files, from the note) that outweighs the bull's case. A news event, already priced or
  not, is not a reason for `out` by itself, whatever consequence the bear argues: the "unless the bear names its
  lasting consequence for the business" escape is removed. Everything else in the judge's prompt stays (max_in as a
  ceiling, previous_verdict, recent_verdicts, memory).

### Docs

- `strategies/bull_bear/CLAUDE.md`: the fact sheet's new fields and where they come from, the fundamentals-first rule
  and the `momentum_exhaustion` test.

## Constraints

- Repo rules (root and nested `CLAUDE.md`): no data tool or data code may use wall-clock time (SEC figures gated on
  `strategy.clock.now()`); hand-written fakes, no network in tests; `fact_sheet.py` stays pure (the SEC fetch happens
  in its caller or in an injected source, as `sector_of` is injected into `ReviewPipeline` today); TDD; numbered tasks
  with a review after each.
- Prompt tests (`tests/strategies/bull_bear/test_bull_bear_prompts.py`) pin the new rules' key sentences.

## Success criteria

1. **Replay (before the backtest)**, from the saved inputs of the 10-23 run's 2026-03-31 review, with the new fact
   sheet (fundamentals computed as of that date) and the new prompts:
   - the bear does not rate MU high, nor `momentum_exhaustion`, in any of at least 3 runs;
   - the judge puts MU in;
   - **anti-goal check:** at least one stock with a real fundamental decline (falling revenue or operating margin in
     its new fact sheet, or a company-reported deterioration in its note), taken from any saved review, is still rated
     high by the new bear in most runs. If no saved review has such a stock, say so before the backtest;
   - the replay is a throwaway script (as the 2026-10-26 replays: inputs rebuilt from the run's log, closes and an SEC
     fetch as of the review date), not a repo command; reviews do not start saving their inputs. The fact sheet code
     must therefore be callable outside a running strategy (pure, with the SEC source passed in);
   - the number of bear highs across runs on identical input is reported (the old spread was 2 to 8 of 12).
2. **1Y backtest, Qwen 3.8**, compared with the 3.6 runs (+61% to +91%) and the 3.8 runs (+42%, +44.6%); results are
   not promised. The comparison is not like-for-like: the inputs changed. The log also shows whether the 81,920 window
   ended the failed judge compactions ("too long to summarize").

## Calls made while writing this up (not discussed with the user)

- Free cash flow falling or debt rising (from the fact sheet) count as fundamental evidence for a high risk, as part of
  "financial health". The user named a guidance cut and revenue or margin declines.
- A lawsuit headline, like other news, is not high by itself (it was on the old list of concrete risks; bill_ackman
  treats it the same way). A company's own filing about a lawsuit's cost (a charge, a settlement) counts.
- The framework's SEC freshness rule is a plain age limit (1 day, confirmed by the user) rather than refetching on a
  new filing in the submissions feed.
- The replay is a throwaway script and reviews do not start saving their inputs (outside this change).
- The sources rule goes into the shared `_FACT_SHEET` paragraph, so all four agents get it.
- `return_1m_skip_1w_pct` covers the 16 sessions before the last 5 (a 21-session month).
- The replay re-runs the bull, bear and judge on the saved researcher notes. It does not re-run the researcher.

## Left to the builder

- Field names for the fundamentals, within the content above; how the SEC source is injected into the pipeline.
- How a 10-Q's periods are handled (a quarter against a quarter; cash flow is often year-to-date): each figure keeps
  the period `sec.py` gives it, and a figure whose period differs from revenue's says so.
- Exact prompt wording, within the rules above.
- Task breakdown and order.
