# A universe as it stood on each date, for cross_momentum and bull_bear backtests -- design

Date: 2026-11-03. Status: approved in conversation, awaiting a plan.

## Why

cross_momentum and bull_bear trade the universe in `data/universe/us_stock_universe.json`, which
`uv run batch-universe` (`strategies/cross_momentum/batch_us_stock_universe.py`) builds from TODAY's data: common
stocks with a market cap over $2B, a price over $10 and an average daily dollar volume over $20M, the top 1,200 by
market cap. A backtest over the past therefore only trades companies already known to have become big by the build
date. That is look-ahead, and it is large: the 2026-11-01 audit of the 2Y bull_bear run found random picks from the
current list made 4.0x, against 2.55x on a universe rebuilt as it stood on each date, about half the profit.

The goal, in the user's words: avoid the survivor bias in backtests of at most 10 years, without a price source other
than Yahoo.

## Goals

1. cross_momentum's and bull_bear's backtests trade, on each date, the universe the live batch would have built then:
   rebuilt monthly, top 1,200, the live batch's screens (common stocks, traded price at least $10, average daily
   dollar volume at least $20M), computed only from data available before that month.
2. It is the ONLY universe for these two strategies' backtests (no switch back to today's file).
3. The history is built by the existing batch, `uv run batch-universe --history`, and runs automatically as a
   pre-step of each backtest, reusing a cached history when it already covers the window.
4. If the pre-step cannot produce a history covering the window, the backtest stops with an error.
5. Each run records which universe it used, and results say that delisted stocks are still missing.

## Non-goals

- Delisted stocks. Yahoo drops most of them, so stocks delisted inside the window stay absent: the residual
  survivorship bias is accepted and stated, not fixed.
- Any price source other than Yahoo.
- Paper and live: they keep today's monthly `us_stock_universe.json` (correct for them: it is built from today's data).
- bill_ackman, vwap_pullback and earnings_drift: they keep today's file in backtests for now; the design must let them
  adopt the history later without rework.
- A market-cap screen as it stood on each date (Yahoo has no reliable share-count history).
- Windows longer than about 10 years.

## Anti-goals

- Any look-ahead in the monthly lists: a month's list uses no bar dated on or after that month's first session.
- A backtest silently running on today's universe (or on a history that does not cover its window).
- Rebuilding the whole history (thousands of Yahoo downloads) on every backtest.
- Changing paper/live behaviour, or the live batch's output when run without `--history`.

## Constraints

- Python 3.14, `uv`; Yahoo through the existing helpers (`backtesting/data/yahoo.yfinance_download`,
  `extract_ticker_frame`, `parse_yahoo_frame`), which give split- and dividend-adjusted OHLCV plus `traded_close` /
  `traded_volume` (the price and volume actually traded) from an `auto_adjust=False, actions=True` download.
- The thresholds come from cross_momentum's `CONFIG` (`min_price` 10, `min_dollar_volume` 20M), as the live batch's do.
- The automated suite never touches the network (injected download/fetch fakes); real Yahoo only in `scripts/tests/`.
- No wall-clock cutoff inside a backtest's data path: the build date of the history (today) only decides what can be
  downloaded; every list is cut by its own month.
- Backtest start/end are inclusive by calendar date (`backtesting/CLAUDE.md`).

## Decided: how it works

### The history mode of the batch

- `uv run batch-universe --history` (with start/end months; the backtest pre-step calls the same function in-process)
  shares the live batch's pool code and thresholds: `get_ticker_universe()` / `get_extra_universe_symbols()` (NASDAQ
  screener, iShares IWB and IWM, Vanguard VTI), `normalize_symbol`, the common-stock check (`quoteType == EQUITY`).
- **Pool:** those sources WITHOUT the $2B market-cap pre-filter (about 5,000-6,000 currently listed names), common
  stocks only. A company that was big years ago and has shrunk since is still listed, so it is in the pool. The pool
  snapshot (symbols and its date) is saved in the history file and reused by later builds, so NASDAQ/iShares/Vanguard
  and the per-symbol common-stock check are paid once per pool snapshot.
- **Bars:** the pool's Yahoo daily bars over [first month - about 3 months, last month], downloaded in batches and
  cached on disk (`cache/universe_history/`), so a rebuild or an extension downloads only what is missing.
- **Monthly lists:** for each month, on its first session (from the SPY calendar in the same data), rank the pool by
  average daily traded dollar volume (`traded_close x traded_volume`) over the previous 63 sessions (the live batch reads
  Yahoo's `averageVolume`, a 3-month average), using only bars dated before that session; keep stocks whose last traded
  close is at least `min_price` and whose average is at least `min_dollar_volume`; keep the top 1,200 by that average.
  Dollar volume stands in for the market-cap rank and the $2B floor.
- **Output:** `data/universe/us_stock_universe_history.json`: build date, pool date and size, thresholds, top N, and
  one entry per month (`"YYYY-MM": [symbols]`).

### The backtest pre-step

- cross_momentum's and bull_bear's `run_backtesting` ensure, before anything else, that the history covers every month
  of the window: reuse the file if it does; otherwise build or extend it (reusing the cached pool and bars). Any failure
  (Yahoo, NASDAQ, iShares, Vanguard, a month left empty) raises an error that stops the backtest before it starts, with
  a message saying what failed.
- These two strategies' backtests must not require `us_stock_universe.json` (paper/live still do).

### The strategies in a backtest

- At each rebalance, the universe is the list of the month containing the review date. cross_momentum's scan
  (`compute_target_portfolio`, and everything reading `vars.universe` in a backtest) and bull_bear's pipeline (its
  `_universe`) ask for the universe of the current date instead of a fixed list.
- The backtest preloads the union of the window's monthly lists, plus the parking/sleeve ETFs and the benchmark.
- A holding that leaves the universe at a month change is treated as live treats it when the monthly file is refreshed:
  cross_momentum sells it (never requested, sold by design), bull_bear drops it from the ranking and sells it. No new
  rule.
- `settings.json` records the universe mode (`point_in_time`), the history file's build date and pool date.

### Docs

- `strategies/cross_momentum/CLAUDE.md`, `strategies/bull_bear/CLAUDE.md`, `backtesting/CLAUDE.md`, the root
  `CLAUDE.md` Commands line for `batch-universe --history`: which universe each mode uses, the pre-step and its error,
  the remaining delisted-stock bias. The other three strategies' CLAUDE.md files: one line saying they still use
  today's universe in backtests.
- The dashboard's run view shows the universe mode from `settings.json` with the delisted-stock caveat.

## Left to the builder

- Where the shared code lives (the history mode in `batch_us_stock_universe.py`, or a sibling module the batch imports)
  and how the universe-by-date object is shaped (a small class `month -> symbols` that the strategies and the pre-step
  share), as long as the batch stays one command and the other three strategies could adopt it unchanged.
- Batch size, rate-limit handling and the cache file layout (follow the live batch's rate-limit gate).
- How the first session of a month is found and what happens to a month whose first session falls before the cached
  bars allow 63 sessions of history (extend the download back; never shorten the average silently).
- The exact error class (a `BacktestError` subclass is natural) and message.

## How we know it worked

- Unit tests on hand-built bars and fake fetchers, no network: a month's list ignores bars on or after its first
  session; the $10 and $20M screens use traded figures; the ranking keeps the top N by dollar volume; the pool is
  built without the market-cap pre-filter; the pre-step reuses a covering file, builds/extends a non-covering one, and
  raises when the build fails; a backtest asks for the universe of each review date and preloads the union.
- A manual script against real Yahoo (`scripts/tests/smoke_universe_history.py`): NVDA is in the 2017 lists; a stock
  that was large then and is small now appears in its old months and not in a recent one; a month's list has at most
  1,200 symbols, all priced at least $10 then.
- A cross_momentum backtest rerun, compared with the last run on today's universe, to see the gap (the audit expects
  roughly half the profit to go).
