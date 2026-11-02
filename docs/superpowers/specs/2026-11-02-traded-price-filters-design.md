# Momentum filters on traded prices -- design

Date: 2026-11-02. Status: approved in conversation, awaiting a plan.

## Why

cross_momentum (and bull_bear, which ranks with cross_momentum's own scoring and settings) drops a stock whose last
close is under `min_price` ($10) or whose 20-day average dollar volume is under `min_dollar_volume` ($20M)
(`strategies/common/scoring.apply_filters`). Both figures are computed from Yahoo bars that are split- and
dividend-adjusted back from today. In a long backtest that is information from the future:

- **Price.** A stock that later split looks cheap in the past. NVDA traded around $20 in early 2015, but its adjusted
  close is about $0.50 after the 2021 4:1 and 2024 10:1 splits, so the $10 filter drops it until mid-2020. The external
  review of 2026-10-31 measured the effect at +1.4 points a year over 15 years, and almost nothing since 07/2020.
- **Dollar volume.** Yahoo adjusts volume for splits too, so splits cancel out of price x volume. Dividends do not: they
  lower past prices but leave volume alone, so a stock yielding 3% a year looks about 35% less liquid 15 years back and a
  stock near the $20M cut is wrongly dropped in the early years.

The goal, in the user's words: **make the long backtests trustworthy.** A filter must see what a trader saw that day.

## Goals

1. The price filter compares `min_price` with the price the stock actually traded at on that date.
2. The dollar-volume filter compares `min_dollar_volume` with the dollar volume actually traded: the mean of each
   session's traded close x traded volume over the last 20 sessions (fewer if the history is shorter). A last-close x
   average-volume formula would mix the pre- and post-split share counts for 20 sessions after any split.
3. Backtest, paper and live use the same data and the same code path (`YahooBacktestData` and
   `strategies/common/yahoo_daily_bars.YahooDailyBars` share `parse_yahoo_frame` today, and keep sharing it).

## Non-goals

- earnings_drift and vwap_pullback. They have their own `min_price` checks on Alpaca bars (also fully adjusted), but
  separate code and short backtests. Untouched.
- bull_bear's fact sheet. It shows the LLM adjusted returns and closes; whether past prices shown to the LLM should match
  the prices quoted in the news is a separate question. Untouched.
- The survivorship bias of today's universe. Separate, bigger problem (needs a point-in-time universe).
- `batch_us_stock_universe.py`'s own price threshold (it builds today's universe from today's prices: correct as is).

## Anti-goals (a failure even if the filters work)

- **Any change to returns, volatility, ranking maths, the risk overlay, the sleeve or fills.** These must keep reading
  exactly today's adjusted OHLC. The only thing allowed to move a backtest is which stocks pass the two filters.
- **A second Yahoo download.** One batched download per load, as today (1,200-symbol universe, Yahoo rate limits).
- **A stock silently scored on mixed data**, e.g. traded volume with an adjusted close.

## Constraints

- Python 3.14, `uv`, money as `Decimal` except at the documented float boundaries. `Bars.df` is a float64 boundary
  (`entities/bars.py`); the new columns are float64 like the others.
- The automated suite never touches the network: unit tests use injected `download` fakes (both classes already accept
  `download=`). Anything against real Yahoo goes in a manual script under `scripts/tests/`.
- No data path may read wall-clock time for a decision (root `CLAUDE.md`). See the split-history note below: using
  splits after a date to undo Yahoo's adjustment is not look-ahead, since the traded price was known on that date.
- yfinance 1.7.0. Its `auto_adjust=True` is exactly `utils.auto_adjust`: `ratio = Adj Close / Close`; open, high and
  low are multiplied by `ratio`; close becomes `Adj Close`; volume is untouched.

## Decided: how it works

### Data (one download, two extra columns)

- `YahooBacktestData.load`, `YahooBacktestData._fetch` and `YahooDailyBars.bars` download with `auto_adjust=False`
  and `actions=True`, instead of `auto_adjust=True`. That returns `Open/High/Low/Close` adjusted for splits only,
  `Adj Close` (splits and dividends), `Volume` (adjusted for splits), and `Stock Splits` (the ratio on each ex-date,
  0 elsewhere; below 1 for a reverse split).
- `parse_yahoo_frame` rebuilds the adjusted OHLC with yfinance's own arithmetic (`ratio = Adj Close / Close`, open,
  high and low times `ratio`, close = `Adj Close`, volume unchanged), so `open/high/low/close/volume` are the same
  values `auto_adjust=True` gives today.
- It adds two float64 columns:
  - `traded_close` = `Close` (split-adjusted, not dividend-adjusted) x the product of every `Stock Splits` ratio on a
    date strictly after the row's date;
  - `traded_volume` = `Volume` / that same product.
- **Split history must reach today, not the window's end.** Yahoo adjusts for every split up to the day of the
  download, so a backtest ending in 2023 still has NVDA's 2024 split baked into its prices. The download's end is
  therefore today (the day of the run) and the frames are cut back to the requested window after the split factors are
  computed. In paper/live the download already ends today.
- The NaN-row drop applies to the original OHLCV columns as today. `Dividends` and `Stock Splits` are not kept in the
  frame.

### Scoring

- `momentum_inputs` computes `price` (the last traded close) and `avg_dollar_volume` (the mean of each session's traded
  close x traded volume over the last 20 sessions, since a last-close formula mixes share counts after a split) from the
  traded series when it is given them, and from the adjusted series (today's last close x average volume) when it is not. Returns and volatility always come from the adjusted closes.
- cross_momentum (`_compute_indicators_for_ticker`) and bull_bear (`strategies/bull_bear/market_data.DailySeries`,
  then `score_stock`) pass the traded columns through when the frame has both.
- **Fallback:** a frame without both columns (an Alpaca source, a test fake, an older caller) scores exactly as today.
  A frame with only one of them is treated as having neither (anti-goal 3).

## Left to the builder

- The exact API: new optional parameters on `momentum_inputs`/`score_stock`, or a small series type, as long as the
  fallback and anti-goal 3 hold.
- Where the split-factor computation lives (a pure helper next to `parse_yahoo_frame` is the natural place).
- How the download end is extended to today and the frame trimmed, as long as nothing after the backtest's end is
  visible through `bars()` (the `_source_bars` gate checks only rows after the cutoff, so trimming to the window is the
  builder's job). The source of "today" here is the download date, the same date Yahoo adjusted to.
- Whether `MomentumInputs` gets new field names (`traded_price`?) or keeps `price`/`avg_dollar_volume` with the new
  meaning. `price` and `avg_dollar_volume` feed only the filters today (cross_momentum's result dict, `apply_filters`);
  the fact sheet reads returns and closes only. Check every reader before renaming.
- Updating `strategies/common/CLAUDE.md`, `strategies/cross_momentum/CLAUDE.md` and `backtesting/CLAUDE.md` to say
  the filters read traded figures and why.

## How we know it worked

1. Unit tests (no network), on hand-built frames with literal expected values:
   - a 2:1 split mid-frame: `traded_close` before the split is twice `Close`, after it equal; `traded_volume` the
     inverse; a reverse split the other way; two splits multiply;
   - a split after the requested window's end still scales the rows inside the window;
   - the adjusted columns equal yfinance's `auto_adjust` arithmetic on the same input;
   - the filters drop or keep a stock on the traded price and traded dollar volume, and fall back on frames without the
     columns.
2. A manual script, `scripts/tests/smoke_traded_prices.py`, against real Yahoo:
   - NVDA's `traded_close` in early January 2015 is about $20 while its adjusted close is about $0.50;
   - for a handful of symbols, the rebuilt `open/high/low/close/volume` equal a plain `auto_adjust=True` download;
   - a high-yield stock's traded dollar volume in 2012 is above its adjusted one.
3. A cross_momentum backtest on a long window (2011 onward) differs from the previous code's run only through the
   symbols whose filter result changed: same returns for the same holdings, and NVDA eligible from the start.
