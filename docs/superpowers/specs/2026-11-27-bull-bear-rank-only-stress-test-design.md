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
