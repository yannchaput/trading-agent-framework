# Earnings-reaction and profitability entry gates on the rank-only book

Date: 2026-11-28. Status: design, rules fixed before any gate is run.

## Intent

The 2026-11-28 score-variant study (`docs/superpowers/specs/2026-11-28-momentum-score-variants-design.md`) found no
price-only change to the momentum score that beats today's under bull_bear's book rules: keeping holdings to rank 35
already cuts turnover to ~5% a week, so a reordering of the ranking adds little. What is left is to change **which stocks
may enter** the book, with information prices do not carry yet. Two published signals:

- **Earnings reaction** (Chan, Jegadeesh and Lakonishok 1996): the stock's market-adjusted return around its last
  earnings release. Momentum that the latest results contradict tends to fade.
- **Gross profitability** (Novy-Marx 2013): gross profit / total assets. Unprofitable high-momentum names (often
  pre-revenue biotech) are the crash-prone part of a momentum book.

## Goal, non-goals, anti-goals

- Goal (user's rule A, as before): **higher CAGR than today, Sharpe no lower**, on both windows. Drawdown reported,
  not judged.
- Non-goals: blending the signals into the score, double sorts, tuning cut-offs, combining the two gates (a
  combination chosen after seeing results is tuning), judge replays, any change to `src/`.
- Anti-goal: a gate that "passes" because it was tried several ways. One definition per gate, fixed here.

## Approaches (user picked 1)

1. **Chosen: entry gate.** Today's ranking unchanged; a stock not held is refused entry when its gate signal is weak.
   Holdings are exempt (as for `max_volatility`). No free parameter beyond the published cut-off.
2. Dropped: double sort (top 50 by momentum re-ranked by the signal): changes holdings too and adds a cut-off.
3. Dropped: rank blend: its weight is a tuning knob.

## Book (same as the score-variant step 2)

`scripts/bull_bear/rank_only_stress.py` data and simulator, bull_bear's book rules as in
`score_variants.book_picks`: keep holdings ranked within 35, fill to 10 from the top, 9.8% each, rest in SHV, weekly,
5 bps, no re-entry cooldown (a judge-less book sells only through forced exits). **Ranks are computed on the full
ranking; the gate only removes names from the fill candidates**, so a holding's rank (and the 35 rule) is the same as
in the ungated book. If fewer than 10 names pass, the book holds fewer and the rest is in SHV.

## Gate definitions

All read only information public strictly before the trade day.

**Earnings gate (`earnings`).** Events = 8-K filings with item 2.02 (`strategies/earnings_drift/events.earnings_events`),
reaction session `r` = `events.reaction_date` (first session whose close is after the acceptance time). Reaction
return = stock close(r+1) / close(r-2) - 1 minus SPY's over the same sessions (a 3-session window, -1..+1 around `r`,
which also covers SEC acceptance times trailing the release by hours). Usable only from the session after `r+1`. The
signal is the latest usable event's reaction; an event older than 126 sessions, or none, counts as **missing**.
**Blocked: reaction < 0.** Missing: allowed. Sessions are SPY's trading dates; a stock with no close at `r-2` or `r+1`
has a missing reaction for that event (builder's call).

**Profitability gate (`profitability`).** From SEC company facts (`us-gaap`), annual figures (form 10-K) filed before
the trade day, latest fiscal year: gross profit = `GrossProfit`, else revenue (`Revenues`, else
`RevenueFromContractWithCustomerExcludingAssessedTax`) minus cost (`CostOfRevenue`, else
`CostOfGoodsAndServicesSold`), from the same fiscal period; divided by `Assets` at that period's end. A 10-K older
than 18 months counts as missing. **Blocked: in the bottom third of the trade day's eligible stocks that have a
value.** Missing (banks, insurers and others with no gross profit line): allowed. A fiscal-year figure is a fact whose
period runs 350 to 380 days; when several filings report the same period, the latest one filed before the trade day
wins (`fundamentals.sec.filter_facts_as_of` rule) (builder's calls).

## Windows and rule

| Gate | Development | Holdout |
|---|---|---|
| `earnings` | 2005-01-01 to 2015-12-31 | 2016-01-01 to 2026-11-24 |
| `profitability` | 2012-01-01 to 2018-12-31 | 2019-01-01 to 2026-11-24 |

8-K item 2.02 exists from 2004-08; company-facts coverage in the cache starts around 2012 (median first `filed`).
Each window starts a fresh book at 1.0 and is compared with the ungated book (`today`) on the same window. A gate
passes a window when its CAGR is above `today`'s and its daily Sharpe is at least `today`'s. **The holdout runs only
for a gate that passes development**, in the same invocation. Kept only if it passes both. Two kept: both reported,
the choice is the user's.

Honesty note: 2016-2026 was the score variants' holdout, so this book's path there has been seen; the gate signals
have not.

## Data

- SEC submissions (with the older pages back to 2004, `fundamentals.sec.older_pages`) and company facts for every
  symbol in the stress pool (2,419; 2,412 map to a CIK through today's `company_tickers.json`), fetched once through
  `SecEdgarClient` into `cache/sec/` (rate-limited; user agent from `SEC_EDGAR_USER_AGENT` in the env, never written
  into the repo). Unmapped symbols count as missing.
- Prices: `cache/stress_2005/` only. Nothing written under `data/` or `src/`.

## Output

- Script `scripts/bull_bear/entry_gates.py` with `--self-check` (hand-worked asserts: the reaction window and its
  usable-from session, the 126-session staleness, the gross-profit fallback and same-period rule, the bottom-third cut,
  missing = allowed, a held name never blocked, ranks of holdings unchanged by the gate) and `--fetch` (the SEC
  download, separate from the run).
- Table per window: CAGR, Sharpe, max DD, the stress falls (2008-09, 2015-16, 2018 Q4, 2020, 2022 as each window
  covers), turnover, plus per gate: share of fill candidates blocked, share with a missing signal (per year), and the
  mean number of names held.
- Results appended here, a replay record in `docs/superpowers/replays/`, a memory note; `TODO.md` only if asked.

## Known limits

- Candidate pool = today's listings, mapped by today's ticker-to-CIK file (a reused ticker can point to the wrong
  company's filings before a rename).
- Early company-facts coverage is partial, so the profitability gate is mostly "allowed" in its first years (reported).
- One market path per window; no significance test beyond the two-window rule. Two gates tested = two chances to pass
  by luck.

## Left to the builder

Code layout, how the signals are cached per (symbol, day), report formatting, fetch progress output.

## Results (2026-11-28, script `scripts/bull_bear/entry_gates.py`)

**Verdicts: `earnings`: FAILED holdout. `profitability`: FAILED holdout. No gate is kept; the entry rule stays as today.**

Both passed development and failed the holdout:

- `earnings`: 2005-2015 CAGR 3.7% vs 0.5% (Sharpe 0.28 vs 0.20); 2016-2026 CAGR 23.9% vs 30.7% (Sharpe 0.73 vs 0.84,
  max DD -54.1% vs -46.0%).
- `profitability`: 2012-2018 CAGR 9.3% vs 5.6% (Sharpe 0.44 vs 0.33); 2019-2026 CAGR 37.9% vs 42.1% (Sharpe 0.96 vs
  1.01).

Sanity: the ungated 2005-2015 book reproduces the score-variant study's (CAGR 0.5%, Sharpe 0.20; asserted in the
script). Download: 2,417 symbols, 2,411 mapped to a CIK, 2,407 with company facts, 10 failures.

#### earnings, development 2005-2015

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | turnover | held | blocked | missing |
|---|---|---|---|---|---|---|---|---|---|
| today | 0.5% | 0.20 | -65.8% | -65.8% | -36.6% | 5.6% | 10.0 | 26.2% | 18.8% |
| earnings | 3.7% | 0.28 | -65.9% | -65.9% | -33.4% | 5.5% | 10.0 | 28.8% | 18.8% |
Names with no signal, by year: 2005 19%, 2006 20%, 2007 20%, 2008 21%, 2009 21%, 2010 20%, 2011 21%, 2012 19%, 2013 18%, 2014 17%, 2015 15%

#### earnings, holdout 2016-2026

| Book | cagr | sharpe | max_dd | 2015-16 | 2018 Q4 | 2020 | 2022 | turnover | held | blocked | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| today | 30.7% | 0.84 | -46.0% | -34.1% | -39.8% | -46.0% | -30.3% | 4.7% | 10.0 | 26.6% | 12.8% |
| earnings | 23.9% | 0.73 | -54.1% | -24.8% | -37.1% | -41.3% | -30.6% | 4.9% | 10.0 | 29.0% | 12.8% |
Names with no signal, by year: 2016 14%, 2017 14%, 2018 13%, 2019 13%, 2020 14%, 2021 13%, 2022 12%, 2023 12%, 2024 12%, 2025 13%, 2026 12%

#### profitability, development 2012-2018

| Book | cagr | sharpe | max_dd | 2015-16 | 2018 Q4 | turnover | held | blocked | missing |
|---|---|---|---|---|---|---|---|---|---|
| today | 5.6% | 0.33 | -61.2% | -53.7% | -39.8% | 4.9% | 10.0 | 13.5% | 60.5% |
| profitability | 9.3% | 0.44 | -53.4% | -53.4% | -38.3% | 5.2% | 10.0 | 17.9% | 60.5% |
Names with no signal, by year: 2012 63%, 2013 63%, 2014 61%, 2015 60%, 2016 60%, 2017 60%, 2018 59%

#### profitability, holdout 2019-2026

| Book | cagr | sharpe | max_dd | 2020 | 2022 | turnover | held | blocked | missing |
|---|---|---|---|---|---|---|---|---|---|
| today | 42.1% | 1.01 | -46.0% | -46.0% | -30.3% | 4.4% | 10.0 | 16.4% | 45.8% |
| profitability | 37.9% | 0.96 | -46.3% | -46.0% | -28.8% | 4.5% | 10.0 | 19.1% | 45.9% |
Names with no signal, by year: 2019 47%, 2020 44%, 2021 44%, 2022 46%, 2023 46%, 2024 46%, 2025 46%, 2026 47%

earnings: FAILED holdout
profitability: FAILED holdout

### Deviations from the plan (all fixed before any gate result)

- Gross-profit fallback is decided per (period, filing), not per period: per period would let a `GrossProfit` line
  filed later erase an already public derived figure (look-ahead).
- `--fetch` refreshed cached SEC files older than 7 days.
- The held/blocked self-check tests `gated_picks` on a hand-made order, not `run()` on synthetic frames.

### Found after the results (rule unchanged, nothing rerun)

**This spec's tag list was incomplete.** Before 2018 many companies reported revenue and cost as `SalesRevenueNet` /
`CostOfGoodsSold`, which the fallback does not read. Eligible names on 2015-06-02 (850): 39.6% had a value, 19.3% had
revenue but no cost line, **18.2% used tags outside the list**, 12.1% had no recent 10-K revenue or gross profit, 5.4%
no company facts, 5.3% never filed a 10-K with assets. On 2025-06-03 (1,121): 54.6% had a value, 28.4% revenue but no
cost line, 0% outside the list. So the gap weakens the **development** result (and, little, the start of the holdout);
the 2019-2026 holdout that rejects the gate is mostly unaffected. One day sampled per year.

### Limits

- Candidate pool = today's listings; today's ticker-to-CIK map; dead companies missing.
- Profitability was "missing" (allowed in) for 61% of eligible names in development and 46% in the holdout, so it is
  a partial gate.
- One market path per window, no significance test; 2016-2026 had already been seen for this book (not for these
  signals).
