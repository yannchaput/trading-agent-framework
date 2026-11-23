# Earnings short side (put spreads after a bad reaction): research spike design

## Intent and why

bull_bear loses money in market falls; the user wants a **separate account that makes money on its own and, ideally,
most when the market falls**. Every index-level attempt failed: the VIX and SPY signals mark volatile days, not lasting
falls, and after a stress signal SPY drifts up (`docs/superpowers/replays/2026-11-23-market-fall-intraday.md`,
`2026-11-23-intraday-fall-signal.md` on branch `research/intraday-fall-signal`); paid hedges (put spreads, inverse
ETFs) on the VIX switch all lost on dev (spec `2026-11-22-vix-put-spread-spike-design.md`).

This spike moves the bet from the index to **single stocks after bad news**: a stock that drops hard on its earnings
release, on heavy volume, without bouncing, is expected to keep drifting down (the short side of post-earnings drift).
The bet does not depend on predicting the market; such stocks tend to fall harder when the market falls, which is the
bonus the user wants. **The vehicle is a debit put spread** (the user's choice: the loss is capped at the debit; no
short selling). It runs **all year**, on every qualifying drop, with no market gate (decided with the user,
2026-11-23).

Path, decided by the user: research -> framework backtest -> Alpaca paper -> Tastytrade **margin** account (a cash
account cannot hold a vertical spread at Tastytrade; a debit spread is fully paid, so the margin account borrows
nothing). Alpaca has no cash / margin choice to make: multi-leg spreads need options **level 3**, on by default in
paper and applied for in the dashboard for live.

## Goals

- One script, `scripts/earnings/short_side_spike.py` (`uv run python`), modes `check` (assert self-check on synthetic
  data, no network), `dev`, `test` (one shot, guard file), in the style of `scripts/regime/fall_intraday_spike.py`.
- **Step 1, stock moves (2016-2026):** does the stock fall after a qualifying drop, after costs, in absolute terms (not
  only against SPY)? A 5-slot account replay with a synthetic -1x stock position per slot.
- **Step 2, real option prices (2024-2026), only after a step-1 GO:** the chosen cell replayed with real put spread
  prices.
- A pre-registered GO / NO-GO and a record in `docs/superpowers/replays/` at the end.

## Non-goals

- No production code, nothing under `src/` or `tests/`; no framework option support (that comes after a GO, in its own
  brainstorm); no short selling; no agent, LLM or news.
- No market-timing gate (`in_market_fall`, regime): the account trades all year.
- No EPS-surprise data: Benzinga's history now starts 2026-04-29, so the signal is the price reaction alone.
- No parameter search beyond the grid below.

## Anti-goals

- **Searching until something passes.** Grid, bar and choice rule are fixed here, before any run; the held-out window
  is scored once.
- **A flattering result:** costs on every fill, entry the session after the signal is known, point-in-time universe,
  no look-ahead in the universe, the events or the features.
- Network access in the self-check.

## Data

- **Universe:** the point-in-time monthly lists, `data/universe/us_stock_universe_history.json` (2016-01 to 2026-11,
  top 1,200 by dollar volume, ranked on data before each month; `UniverseHistory.on(day)` in
  `strategies/common/universe_history.py`). An event counts only if its symbol is in the list of its reaction day's
  month. Known bias, reported in the record: delisted stocks are absent from Yahoo and so from the lists (bankruptcies
  would have been winning shorts, takeovers losing ones).
- **Daily bars:** the cached Yahoo bars `cache/universe_history/bars/<SYMBOL>.parquet` (columns `Open High Low Close
  Adj Close Volume ...`, split-adjusted `Close`, dividend-adjusted `Adj Close`, from 2013). P&L uses total-return prices
  (`Adj Close`, and `Open` scaled by `Adj Close / Close` of the same row), so a short pays the dividends it would owe.
  The reaction features are ratios (drop, range position, relative volume, dollar volume), which a later split leaves
  unchanged, so they use the split-adjusted OHLCV. **Every price level uses the price traded that day** (the `$10`
  floor, step 2's strike pick): the cached `Close` is divided by every later split (NVDA's 2016 close is under $1), so
  a floor on it is look-ahead that drops later winners such as NVDA, CMG or ANET. The traded close is
  `parse_yahoo_frame`'s `traded_close` (`backtesting/data/yahoo.py`), as the universe build uses it; the traded open is
  `Open x traded_close / Close`. SPY from the same cache; **SHV** daily from Yahoo (for the reference row).
- **Earnings events:** SEC 8-K filings carrying item 2.02, through `SecEdgarClient` (`fundamentals/edgar_client.py`)
  and the pure `earnings_events` / `reaction_date` (`strategies/earnings_drift/events.py`), submissions pages back to
  2015 (`older_pages`, as `event_source.py` reads them). Cache: the client's own `cache/sec/`. Known gap, reported: the
  ticker -> CIK map is today's, so renamed or delisted tickers may lose their events.
- **Option prices (step 2 only):** Alpaca historical option bars (credentials `ALPACA_DATA_API_*` from
  `env/.env.bull_bear.paper`, read-only; history from February 2024). **Precondition:** expired contracts and their
  daily bars must be retrievable; if not, step 2 reports that and stops (no modelled prices instead).
- **Windows:** development **2016-01-04 to 2023-12-29**; held-out **2024-01-02 to the last complete session**, scored
  once. Step 2 runs on the held-out window from the first session with option bars.

## Signal (computed at the reaction day's close)

For each event, the reaction day = `reaction_date(accepted_at, trading_dates)` (`earnings_drift`'s rule: a release
after the close reacts the next session). With `reaction_features(stock, spy, day)` (`earnings_drift/reaction.py`), the
event **qualifies** when all hold:

1. `abnormal_pct <= -D` (the day's return minus SPY's; `D` is a grid value)
2. `close_location < 0.5` (closed in the lower half of the day's range: no bounce)
3. `rel_volume >= 2.0` (volume against the 20-session average before the day)
4. traded close `>= 10` and `dollar_volume_20d >= 20,000,000` (`earnings_drift`'s floors, so the options are liquid)
5. one position per company: share classes mapping to the same SEC CIK (GOOG/GOOGL, BRK-A/B, FOX/FOXA) count as one
   symbol, the first in the month's list

## Trade model, step 1 (stock moves)

- **Entry:** the open of the session after the reaction day (the signal is known at that close). **Exit:** the close
  of the `H`-th session counted from the entry session (entry session = 1). No stop: the spread caps the loss.
- **Position:** a synthetic -1x stock **with its gain capped at 10%** (decided by the user), value
  `2 - max(p / entry, 0.9)` on total-return prices: the put spread's short leg 10% lower stops its gain there, so a
  crash past -10% must not count more in step 1 than the spread can pay. The loss stays uncapped (the spread's loss
  stops at its debit), which keeps step 1 on the hard side.
- **Cost (decided by the user): 80 bps per side, the put spread's cost.** A spread round trip pays four option
  half-spreads of about 0.4% of the stock's price each (step 2), about 1.6% per round trip, against about 0.2% for a
  short stock. Judging step 1 at the stock's cost could pass a drift the spread cannot survive and spend the held-out
  window on a certain step-2 failure. Every table also shows the same rows at **10 bps per side** (the short-stock
  cost, single stocks trading wider than ETFs), for information only: the bar reads the 80 bps rows.
- **Account:** fake **5,000**, **5 slots**, each new position sized at **20% of the account's equity at the entry
  open**; idle money earns nothing (the SHV reference row shows what it gave up). When more events qualify on one
  reaction day than there are free slots: **largest drop first** (most negative `abnormal_pct`), ties by symbol. A
  symbol already held is skipped. Slots freed at a close are usable at the next open.
- **Signal gate (per cell, no slot limit):** every qualifying dev event, one -1x trade from entry to exit, equal
  weight.

## Trade model, step 2 (real put spreads)

The chosen cell, same events, slot count and entry rule, in **its own replay** (step 1's schedule is not reused):

- **A skipped trade frees its slot for the next-largest drop of the same reaction day** (a real account would take the
  next affordable candidate), so idle money is not inflated by expensive stocks. Skips are counted by cause.
- **Contracts:** buy the put whose strike is nearest the entry-day traded open (at the money), sell the put whose
  strike is nearest 10% below it, same expiry: the **first listed expiry at least `H` sessions after the entry
  session**.
- **Fills:** each leg at its **daily bar's VWAP** on the entry session and on the exit session, **plus a half-spread of
  0.4% of the stock's traded price per leg per side** against the trader. Alpaca has no historical option quotes, so the
  half-spread is a measured constant: on 2026-11-23 (10:48 ET, OPRA, 15 single stocks, ~2-3 week puts) the median
  half-spread was $0.39 per share at the money, $0.44 about 10% out of the money, about 0.4% of the stock's price (range
  0.17% AAPL to 2.6% ENPH). A round trip on a spread therefore costs about **1.6% of the stock's price** (4 leg-sides).
- **Fees: Tastytrade's, the production broker (user's choice):** $1.00 per contract to open (capped at $10 per leg),
  $0 to close, $0.10 clearing per contract per side, about $0.03 regulatory per contract per side (ORF, TAF). (Alpaca,
  used for paper only, is cheaper: no commission, about $0.05 per contract per side in pass-through fees.)
- **Missing bars:** a leg with no bar on the entry session: the trade is skipped (counted). A leg with no bar on the
  exit session: the first later session with a bar, else intrinsic value at expiry.
- **Sizing:** whole contracts, `floor(20% of equity / (debit x 100))`; zero contracts = skipped (counted, so a
  universe of expensive stocks shows up).

## Grid (fixed now: 6 cells)

Drop threshold `D` in {5%, 8%} x holding period `H` in {5, 10, 20} sessions. Fixed: `close_location < 0.5`,
`rel_volume >= 2`, the liquidity floors, 5 slots, 80 bps per side.

Reference rows (not candidates): SHV buy and hold, SPY buy and hold, the gate's mean trade split by year, and **the
chosen cell's P&L on trades entered while `in_market_fall` was on vs off** (the bonus the user wants: does the account
do best in falls?).

## Bar (fixed now)

1. **Signal gate (dev, per cell):** the mean -1x trade after costs must be **positive**. A cell that fails is out; if
   all 6 fail: **NO**, nothing else is judged.
2. **A cell passes (dev, 5-slot account)** if its gate passed and: total return > SHV buy and hold over the same
   window, at most **2 losing calendar years** of 8, and **at least 30 trades**.
3. **Choice:** among passing cells, the one with the most passing neighbours (cells differing in one grid dimension);
   ties by higher dev total, then smaller max drawdown. None passes: **NO**, held-out not run.
4. **Held-out, step 1 (once, chosen cell):** total return > 0 **and** > SHV buy and hold on the same window.
5. **Step 2 (only if 4 passes):** the chosen cell with real put spreads on the held-out window from the first session
   with option bars: **total return > 0**. GO needs 4 and 5.

**Order of `test` mode:** before the guard file is written, a data probe (no results looked at) checks that Alpaca
returns expired put contracts and their daily bars for a few 2024 dates. If it fails, `test` stops without scoring
anything and the held-out window stays unused; the record says step 2 is impossible with this data, and the user
decides what next. Positions still open at a window's end are closed at its last session's close (step 2: the legs'
last available VWAP, else intrinsic value). Step 2's drawdown uses daily marks (each leg's daily close, else its last
known close).

Reported per cell: total, CAGR, per calendar year, max drawdown, trades, win rate, mean trade, trades skipped (slots
full, already held), events per year; step 2 adds mean debit, skipped for no bar or too expensive.

## Constraints

- Python 3.14, `uv`, no new dependency; read-only (SEC, Yahoo, Alpaca data); writes only caches, `logs/` (git-ignored)
  and the record.
- **Built in the worktree `.worktrees/earnings-short-side`** (the user's main checkout stays untouched). The git-ignored
  inputs (`cache/`, `data/universe/`, `env/`) exist only in the main checkout, so the script resolves them from the
  main checkout's root (`git rev-parse --git-common-dir` -> its parent), not from the worktree.
- SEC requests through `SecEdgarClient` (its rate limit and user agent, `SEC_EDGAR_USER_AGENT`).
- Lint: `ruff check` and `ruff format --check` clean on the script and the Markdown files.
- The self-check covers at least: entry the session after the reaction day, the reaction-day rule for after-close
  releases, universe membership by month, the qualifying rule, largest-drop-first slot filling, the H-session exit,
  total-return pricing, the 10% gain cap, the gate's mean, the choice rule, and step 2's contract pick, fills and
  sizing.

## Approaches (decided with the user, 2026-11-23)

- **A. Research script reusing `earnings_drift`'s pure parts (chosen).** Same method as the last spikes; a stock-move
  gate first, then the account, then real spreads; little thrown away on a NO.
- B. A framework strategy backtested directly: closest to paper, but the framework has no option support; weeks of
  work before knowing there is an edge.
- C. The stock-move gate alone: cheapest, but a GO would not say whether a 5-slot spread account makes money; A
  includes it as its first step.

## Decided with the user

Single-stock bet after a bad earnings reaction; debit put spreads, not shorting (capped loss); Tastytrade margin
account; all year, no market gate; profitable on its own, doing best in falls a bonus; two steps (stock moves 2016-2026,
then real spreads 2024-2026); step 1 at 80 bps per side with its gain capped at 10%; Tastytrade fees in step 2; the
trade model, grid and bar above (played back and approved).

## My calls (correctable)

Thresholds `close_location < 0.5`, `rel_volume >= 2`, `earnings_drift`'s liquidity floors; 10 bps per side as the
short-stock info rows; 5 slots at 20% of equity; largest drop first; ATM long put and ~10% lower short put; VWAP fills
plus a 0.4%-of-stock-price half-spread per leg (one measured snapshot); first expiry at least `H` sessions out; dev
2016-2023 and held-out 2024 on; idle cash earns nothing; step 2 stops rather than modelling prices when expired
contracts cannot be fetched, and that is checked before the held-out window is spent; price floors and strikes on the
traded price; one position per company; a skipped spread passes its slot to the next drop that day; open positions
closed at a window's end.

## Left to the builder

Script layout, how SEC pages are fetched and cached for ~2,000+ symbols, the trading calendar (SPY's bar dates), table
printing, the self-check's synthetic data.
