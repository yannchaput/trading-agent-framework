# bull_bear rank-only stress test, 2005-2026 (TODO item 5)

Written 2026-11-27, before any result of this test. Short design (research script, no production code).

## Why

Rank-only is bull_bear's default if the judge fails the paper test (item 3). Before real money, know the worst fall
of that book through 2008, 2015-16, 2018 Q4, 2020 and 2022. **The user's limit: max drawdown 40%.**

Known before this test: the 2026-11-12 lever study rebuilt the top-10-by-rank book from backtest `reviews.jsonl`
and found max drawdown -46.4% on 2020-01..2021-05 and -34.6% on the 5Y window. Not re-verified, and not mentioned
when the limit was asked for. The limit was set by the user without that figure.

## The book (no LLM)

- Universe: point-in-time monthly lists from `ensure_universe_history` (top 1200 by 63-session dollar volume,
  price and dollar-volume screens), into a **separate** history file and cache dir under the session scratchpad or
  `cache/stress_2005/`; `pool.json` copied from `cache/universe_history/`. The files backtests use are not touched.
- Ranking: `strategies/common/scoring.py` `score_stock` + `rank` with cross_momentum's `CONFIG`, completed
  sessions only; a holding is exempt from `max_volatility`, as in bull_bear.
- Weekly, Tuesday (next session on a holiday): rank on closes through the previous session, trade at that
  session's open. Top 10 by rank, 9.8% each, reset to target every week. No cooldown, sector cap or overlay.
- The rest in SHV; before SHV exists (2007-01) cash at 0%. 5 bps per side on turnover.
- A holding whose prices stop is sold at its last price.
- Period: 2005-01 to 2026-11-24, equity marked at each session's close.

## Rule (fixed)

- **Verdict, at 98% stocks:** max drawdown on daily closes over the whole period <= 40% -> "not rejected"
  (unconfirmed: survivorship makes it optimistic). Worse than 40% -> **rank-only at this size is rejected for
  real money.**
- Report only (builder's proposal, no verdict): the same drawdown at 84% and 70% stocks, to size a follow-up.
  Choosing a size from this table is in-sample and needs the user's decision.

## References and report

- SPY (total return), the equal-weighted universe (same lists, same bias), and Ken French's daily top-decile
  momentum portfolio ("10 Portfolios Prior 12-2", equal- and value-weighted; includes delisted stocks, different
  rules: an outside check, not a like-for-like).
- For each book: CAGR, max drawdown with its dates, and the drawdown in 2008-09, 2015-16, 2018 Q4, 2020, 2022.
- Per year: eligible stocks ranked (the $20M screen is in nominal dollars, so early years are thinner) and the
  share of listed symbols with bars.

## Approaches

- A (chosen): our universe and ranking code on Yahoo, plus French as outside check. Free; survivor-biased.
- B: wait for item 4 (EODHD delisted names). Less biased, much more work; coverage and plan unchecked.
- C: French data only. Quick; does not test our rules or a 10-stock book.

## Left to the builder

Script layout (`scripts/bull_bear/`, with one assert-based self-check of the weekly book and drawdown maths),
download batching and resume. Yahoo rate limits or more than 1% absent symbols stop the history build
(`UniverseHistoryError`): if that limit trips, stop and report instead of raising it.

## Results (2026-11-27)

**Verdict: rank-only at 98% stocks is REJECTED for real money** (max drawdown -78.9% against the 40% limit).
84% (-72.2%) and 70% (-64.0%) stocks are also far past the limit (report-only).

- The worst fall runs 2008-06-17 to 2011-10-03; that peak was regained only on 2020-07-08. SPY's worst was -55.2%.
- CAGR 9.3% against 10.5% for SPY and 10.4% for the equal-weighted universe: over 2005-2026 the book took more
  risk for less return, even with a survivor-biased candidate pool.
- French's top momentum decile (delisted names included, different rules) fell -56% to -58%: the 10-stock book
  fell much further than a diversified momentum portfolio.
- Survivorship: the candidate pool holds only today's listings. Only 32-37% of it has bars in 2005-2009, mostly
  later listings, so that figure does not count the companies that died. How much, and in which direction, their
  absence moves this book is not measured (TODO item 4). The "optimistic" wording of the rule above was an
  assumption; a rejection should not be read as a floor.
- Checks: a separate cost-free weekly recompute from `books.csv` (stocks only) gives -78.5%, over the full period
  and inside 2008-06..2011-10 alike (trough 2011-10-04). The independent review's daily recompute, without costs or
  SHV, gives -78.1% with the simulation's dates. It found no bad data in the names held over 2008-06..2011-10.
- Data: the largest held one-day moves (below) are mostly biotech names and were not checked against a news source.
  The 2192 non-positive price rows are in CBIO and RCAT, never held.
- Cross-check with the 2026-11-12 rebuilt book (-46.4% over 2020-01..2021-05, a different book): -47.9% here.
- Yahoo had no data for 1 of 6,662 symbols (TSE). Equity and books: `cache/stress_2005/equity.csv`, `books.csv`.

| Book | CAGR | Max DD | Peak to trough | 2008-09 | 2015-16 | 2018 Q4 | 2020 | 2022 | Data to |
|---|---|---|---|---|---|---|---|---|---|
| Rank-only 98% stocks | 9.3% | -78.9% | 2008-06-17 to 2011-10-03 | -71.9% | -56.9% | -44.0% | -47.9% | -29.2% | 2026-11-24 |
| Rank-only 84% stocks | 9.3% | -72.2% | 2008-06-17 to 2011-10-03 | -65.5% | -51.0% | -38.6% | -42.3% | -25.4% | 2026-11-24 |
| Rank-only 70% stocks | 9.0% | -64.0% | 2008-06-17 to 2011-10-03 | -57.9% | -44.4% | -32.9% | -36.3% | -21.6% | 2026-11-24 |
| SPY | 10.5% | -55.2% | 2007-10-09 to 2009-03-09 | -51.9% | -13.0% | -19.2% | -33.7% | -24.5% | 2026-11-24 |
| Equal-weighted universe | 10.4% | -55.8% | 2007-10-10 to 2008-11-20 | -53.8% | -22.2% | -21.0% | -41.2% | -25.6% | 2026-11-24 |
| French top decile VW | 14.0% | -56.4% | 2008-06-18 to 2009-03-05 | -56.4% | -17.9% | -28.5% | -32.5% | -29.1% | 2026-09-30 |
| French top decile EW | 12.3% | -58.5% | 2007-10-31 to 2009-03-09 | -57.0% | -27.1% | -28.2% | -42.3% | -24.2% | 2026-09-30 |

| Year | Mean ranked stocks | Pool symbols with bars |
|---|---|---|
| 2005 | 415 | 32% |
| 2006 | 490 | 34% |
| 2007 | 594 | 35% |
| 2008 | 518 | 36% |
| 2009 | 492 | 37% |
| 2010 | 588 | 39% |
| 2011 | 634 | 40% |
| 2012 | 639 | 42% |
| 2013 | 696 | 43% |
| 2014 | 777 | 46% |
| 2015 | 831 | 48% |
| 2016 | 837 | 49% |
| 2017 | 923 | 52% |
| 2018 | 1019 | 55% |
| 2019 | 1024 | 58% |
| 2020 | 918 | 63% |
| 2021 | 1080 | 71% |
| 2022 | 1077 | 73% |
| 2023 | 1151 | 76% |
| 2024 | 1148 | 80% |
| 2025 | 1111 | 87% |
| 2026 | 1084 | 100% |

Largest one-day moves of held stocks:

```
  2018-05-31 MDGL +145.0%
  2018-09-18 VKTX +87.3%
  2019-12-16 AXSM +70.5%
  2023-03-28 VKTX +69.3%
  2021-01-29 NVAX +64.9%
  2016-01-15 SRPT -54.9%
  2022-11-10 VERU -53.6%
  2024-08-15 ASTS +50.7%
  2023-08-01 TGTX -49.3%
  2026-04-28 ERAS -48.3%
  2025-02-18 WGS +47.7%
  2026-06-02 ABVX -44.1%
  2014-10-08 ARWR -43.8%
  2025-04-30 WGS -42.8%
  2008-10-13 SQM +42.6%
  2020-02-19 ENPH +42.4%
  2020-09-08 CODX +42.0%
  2018-06-04 NKTR -41.8%
  2008-10-02 MOS -41.3%
  2014-01-13 ALNY +40.9%
```
