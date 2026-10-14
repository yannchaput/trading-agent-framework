# dashboard: Live and Paper tabs (realized and unrealized P&L per day)

Date: 2026-10-13 · Branch `main` (shared checkout, no feature branch).

## Goal

Two new dashboard tabs, **Live** and **Paper**, placed before **Backtesting**. Each shows, for one strategy picked in the
sidebar, its realized and unrealized P&L per trading day in a line chart, rebuilt from the strategy's Alpaca account and
fetched on demand.

## Why

Live and paper runs write only a `<mode>.log` into `logs/<strategy>/<mode>/<ts>_<mode>/` -- no `trades.parquet`, no
`equity.parquet` -- so nothing in `logs/` says how a running strategy is doing. The broker account does.

## Scope

In: the two tabs, a Streamlit-free account reader, the daily P&L computation, one chart, one page module, tests, one doc
line in `dashboard/CLAUDE.md`. Out: IBKR strategies (only `BROKER=alpaca` env files are listed); splitting a shared
account between strategies; auto-refresh; writing anything anywhere; changing what live/paper runs record.

## Facts the design rests on

- Each paper strategy has its own Alpaca account (a distinct `ALPACA_API_KEY` per `env/.env.<strategy>.paper`), so the
  account's P&L is the strategy's P&L.
- `cross_momentum` and `news_binary` share one live account. Accepted as is: `news_binary` has not run live since
  2026-09-24 and will not again, so the live account is treated as `cross_momentum`'s; selecting `news_binary` in Live
  shows the same account.
- `env/.env.cross_momentum.paper` sets `BROKER_API_IS_PAPER=false` with a key that differs from the live one: likely a
  misconfiguration, flagged to the user, not fixed here. The reader uses whatever the env file says.
- The positions API (`get_all_positions`) gives today's `unrealized_pl` per symbol only, no history.
- alpaca-py 0.44 has the `TradeActivity` / `NonTradeActivity` models but no `get_account_activities` method:
  activities are fetched with the client's raw `get("/account/activities", ...)`.
- `brokers/alpaca/market_data.ADJUSTMENT` is `Adjustment.ALL`; fills are raw prices, so marking holdings to market needs
  `Adjustment.RAW` closes, built here rather than through `build_bars_request`.

## Design

### Tabs and selection

- `app.NAV_PAGES` becomes `Live`, `Paper`, `Backtesting`, `Models`; **Backtesting stays the default tab** (`default=`
  is set by title, not by position).
- Live and Paper are one page module, `_pages/account.py`, with `page_live()` and `page_paper()` wrapping
  `page_account(mode)`.
- Each tab draws its own sidebar (as Backtesting's multiselect lives only on Backtesting): a `selectbox` of the
  strategies with an `env/.env.<strategy>.<mode>` file whose `BROKER` is `alpaca` (missing `BROKER` means alpaca, as in
  `BrokerSettings`), keyed per mode (`account_strategy_live` / `account_strategy_paper`) so each tab remembers its
  choice, and a **Refresh** button that clears the cached fetch and reruns.
- No matching env file: `st.info` naming the expected file pattern.

### `dashboard/account_reader.py` (no Streamlit)

- `strategies_for_mode(mode, env_dir) -> list[str]`: sorted strategy names from `env_dir/.env.<strategy>.<mode>` files
  with `BROKER=alpaca`; read with `dotenv_values`.
- `load_account_credentials(strategy, mode, env_dir) -> tuple[AlpacaCredentials, AlpacaCredentials]`: the trading and
  data credentials, via `AlpacaCredentials.for_trading(env=values)` / `for_data(env=values)` on the
  `dotenv_values` mapping -- **never `load_dotenv`**, so one strategy's keys never reach `os.environ` and switching
  strategies cannot mix credentials. Blank keys raise the existing `ConfigurationError`.
- Narrow client Protocols, read-only by construction:
  - `AccountTradingClient`: `get(path, data)` and `get_all_positions()` (no order methods).
  - `AccountDataClient`: `get_stock_bars(request)`.
- `fetch_activities(client) -> list[TradeActivity | NonTradeActivity]`: `GET /account/activities` with
  `activity_types=FILL,FEE`, `direction=asc`, `page_size=100`, following `page_token` (the last item's `id`) until a
  short page; each item parsed into the alpaca model by its `activity_type`.
- `fetch_daily_closes(client, symbols, start, end) -> dict[str, dict[date, float]]`: one `StockBarsRequest`, 1Day,
  `feed=IEX`, `adjustment=RAW`.
- `load_account_pnl(trading, data, today) -> AccountPnl`: fetches activities and positions, replays the fills, fetches
  closes for every symbol held at some day's close, and returns the result of `daily_account_pnl` (below).

### Daily P&L (pure)

- The average-cost replay in `reader.load_session_pnl` moves to a shared helper (one pass over time-ordered fills
  `(time, symbol, side, qty, price, fee)` keeping `symbol -> [shares, remaining cost]`, buy fees in cost, the sell
  fee deducted from proceeds, the sold shares leaving at average cost). `load_session_pnl` keeps its exact behaviour and
  output; its 9 existing tests stay unchanged and must pass.
- `daily_account_pnl(fills, fees, closes, positions, today) -> AccountPnl`, days in `MARKET_TZ` (US/Eastern) dates:
  - **realized[d]**: the P&L of the sales on day d, net of their fees, at average cost; `FEE` activities (regulatory
    fees, `net_amount` negative) on day d are added to realized[d].
  - **unrealized[d]** for a past day: sum over the symbols held at d's close of `shares * close[d] - remaining cost`.
  - **unrealized[today]** (amended after the final review): never a bar close. Per position, the replayed shares at
    the position's `current_price` against their replayed (average) cost, so it shares the realized P&L's cost
    basis; Alpaca re-bases overnight positions to compressed FIFO, which would otherwise make Total mix two
    bases. A position the fills do not explain (other quantity, not replayed, no price) keeps the broker's
    `unrealized_pl`. `AccountPnl.broker_unrealized` keeps Alpaca's own sum and the page explains a difference.
  - Days: every trading day from the first fill to today that has a bar for at least one held symbol, plus any day with
    a sale; today always has a row when positions or fills exist.
  - Missing close for a held symbol on a day (halt, delisting): the last earlier close is carried forward; with none,
    the symbol is left out of that day's unrealized and named in `AccountPnl.unpriced`.
  - **Mismatch**: after the replay, the replayed holdings are compared with the live positions (quantity per symbol,
    tolerance 1e-6); every difference -- a split, a transfer, history before the first fill, a sale with no prior buy
    -- goes into `AccountPnl.mismatches` as `(symbol, replayed_qty, broker_qty)`.
- `AccountPnl`: `days: list[{date, realized, unrealized}]`, `realized_total`, `unrealized_now`, `positions:
  list[{symbol, qty, avg_entry_price, current_price, market_value, unrealized_pl, unrealized_plpc}]`, `mismatches`,
  `unpriced`. Floats throughout (the dashboard is display-only, like the existing reader).

### Page

1. Title `Live` / `Paper` and the selected strategy as a caption.
2. Metric cards: cumulative realized, current unrealized, total (realized + unrealized).
3. `components/charts.account_pnl_chart(days)`: one scatter line per day of realized + unrealized (hover gives both
   parts), markers green/red per sign, dotted zero line, `CHART_TEMPLATE`; empty days -> the "No trade data available" annotation like
   `session_pnl_chart`.
4. Caption explaining the points (realized = sales that day, net of fees; unrealized = open positions at that day's
   close, today at the current price).
5. `st.warning` listing `mismatches`, `unmatched_sales` and the `unpriced` symbols when not empty. The caption names the
   account the keys really hit (`BROKER_API_IS_PAPER`), with a warning when that is not the tab's mode.
6. Time window (added 2026-10-14): sidebar **Start date** / **End date** inputs bounded by the first and last day of the
   history, default all of it, remembered per tab. The plot and the Realized card (sum of realized P&L inside the
   window) follow it; Unrealized now stays the current figure and Total is their sum. A start after the end is a
   sidebar error and an empty plot. A caption states the window when it is narrower than the history.
7. Current positions table (`st.dataframe`, USD and % column formats).

Caching: the page calls `load_account_pnl` through an `st.cache_data` wrapper keyed by `(strategy, mode, today)`
(credentials are re-read inside, never part of the cache key or the page). Refresh clears it. No background polling.

### Error handling

| Situation | Tab shows |
|---|---|
| No `BROKER=alpaca` env file for the mode | `st.info` |
| Blank or missing keys (`ConfigurationError`) | `st.error` with the env file and the variable name, never a key value |
| Alpaca API or network error (`APIError`, `requests` errors) | `st.error("Could not reach Alpaca: ...")`; not cached, Refresh retries |
| Positions but no fills | metric cards, positions table, `st.info` that there is no fill history |
| No fills and no positions | `st.info("No trades yet")` |
| Replay/positions mismatch, unpriced symbol | `st.warning`, bars still drawn |

## Testing (TDD, never the network)

- Shared replay helper: the existing `load_session_pnl` tests unchanged and green.
- `daily_account_pnl`: realized across days net of fees and `FEE` activities; unrealized at close; today from the
  positions; carried-forward close and `unpriced`; mismatches (sale with no buy, quantity differing from the broker).
- `account_reader`: fake trading/data clients returning real alpaca models (`TradeActivity`, `NonTradeActivity`,
  `Position` via `tests/fakes.make_alpaca_position`, `BarSet` via `make_alpaca_barset`); activity pagination; the bars
  request is IEX/RAW/1Day; credentials read from a `tmp_path` env file without changing `os.environ`;
  `ConfigurationError` on blank keys; `strategies_for_mode` lists per mode and drops `BROKER=ibkr`.
- `account_pnl_chart`: one lines+markers trace of the daily total, dark template, empty-data annotation.
- Pages: `NAV_PAGES` titles in order `Live, Paper, Backtesting, Models` and Backtesting the default;
  `AppTest.from_function` with a monkeypatched loader for the metric cards, the chart, the warning and the error path.
- Manual: one read-only fetch against a paper account, with the user's go-ahead.

## Docs

`dashboard/CLAUDE.md`: one paragraph for the Live/Paper tabs (env-file discovery, `dotenv_values` not `load_dotenv`,
read-only client Protocols, RAW closes, the shared average-cost helper). Root `CLAUDE.md` command line for
`uv run dashboard` mentions the two new tabs.
