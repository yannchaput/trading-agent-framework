# Breadth and agreement screen: research design

Follows `2026-11-19-fast-regime-research-design.md` (the "fast regime" screen; result NO, record in `docs/superpowers/replays/2026-11-19-fast-regime-research.md`). Same branch `feat/fast-regime-research`, not merged (the user's choice).

## Intent and why

The goal is unchanged: a signal that turns bearish **in the first days of a fall**, so the bear call spread, a SPY short or an inverse ETF enter on time. The first screen showed that single fast inputs either fire too often (VIX9D/VIX 322 false bear days, VVIX 318) or catch too little of a fall (VIX jump recall 0.17), and the variant built from its winners failed the bar (-5.2% at switch-on against -3.0%). Two ideas were left untested:

1. **Market breadth**, the classic early sign: many stocks weaken under the index before the index shows it. Measured two ways (decided with the user): across our point-in-time stock universe, and across the 11 S&P sector ETFs.
2. **Agreement of fast inputs**: a day counts only when several of the six fast inputs fire together, to cut the false alarms that sank each one alone.

## Goals, non-goals, anti-goals

- **Goal:** one more screen, on the same development window with the same selection rule and bar, adding five candidates; if a variant passes the development check, one look at the held-out window.
- **Non-goals:** changing the production regime; trading tests (later, only on a pass); news sentiment and the put/call ratio (later designs); changing the old candidates' thresholds or the selection rule.
- **Anti-goals:** tuning a threshold after seeing a result; looking at the held-out window (2024-07-01 on) before the final pick; a "fast" signal that buys speed with false alarms.

## Constraints

- Everything from the first design still holds: development window 2016-01-04 to 2024-06-28; held-out 2024-07-01 to the last download, scored only by `regime fast --held-out` after a passing development check; `screen` refuses a window reaching the held-out start; no network in the suite; pure functions where the regime code is pure; the production regime's output identical.
- **Same selection rule and bar** (fixed in the first design): a candidate qualifies if its `false_bear_days` is at most the larger of `below_sma`'s and `off_high`'s on the development window; the two qualifying candidates with the best `mean_on_fall` win (ties: fewer false days, then name); `same_day` wins on a strictly better mean with no more false days than `today`. Variant: `inverted` and `vix_spike` plus the winners (one winner replaces `off_high` only). Pass: `mean_on_fall >= -0.03` and `false_bear_days <=` today's.
- **Point-in-time stock lists:** a session's breadth uses only the list of that session's month from `data/universe/us_stock_universe_history.json` (each month's list is ranked on data before the month starts), and only closes up to the flag's index. **Split- and dividend-adjusted closes** (a split must never look like a crash or a new low).
- **Survivor gap (accepted, recorded):** stocks delisted later are absent from Yahoo and from every list (the 2016-01 list has 880 names, not 1,200). Breadth built on survivors understates weakness in a fall; a positive result is conservative, a negative one may be too harsh. The record says so.
- **Development data only without `--held-out`:** stock and sector closes after `DEV_END` are neither downloaded nor read unless `--held-out` is given (the same rule as the first screen's inputs). A local cache must not serve rows after the requested end.

## Approaches (decided with the user, 2026-11-20)

- **A. Extend the existing screen (chosen).** The five candidates join `fast_regime.CANDIDATES`; `regime screen` prints old and new rows in one table under the one rule; `regime fast --replace` and the held-out guard work unchanged with the new names. Costs a stock-breadth data builder with a cache and tests.
- B. A separate `regime breadth` command: rejected, duplicates the selection and printing paths for no new information.
- C. A throwaway spike first: rejected, the selection rule and the held-out guard would be applied by hand.

## The candidates (fixed before any run)

All are raw daily flags at index `i` (closes up to and including `i`), turned into prior-day labels by the existing `raw_labels` (the label of session `j` reads index `j - 1`), and usable by the variant with either timing.

| Name | Bearish at index `i` when |
|---|---|
| `stocks_below_20d` | at least **75%** of the eligible stocks of the month's list close below their 20-session average (`close[i] < mean(close[i-19..i])`) |
| `new_lows` | at least **3%** of the eligible stocks close at their 252-session closing low (`close[i] <= min(close[i-251..i])`) |
| `sectors_below_20d` | at least **8/11** of the available sector ETFs close below their 20-session average (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC; a sector is available at `i` once it has 20 closes, so XLC counts from 2018-07; the share is over the available ones) |
| `agree_2` | at least **2** of the six first-screen inputs (`vix9d`, `vix_jump`, `vvix`, `credit`, `short_drawdown`, `breadth`) fire at `i` |
| `agree_3` | at least **3** of those six fire at `i` |

- **Eligible stock** at `i`: in the list of `i`'s month, with a close at `i` and enough closes for the window (20 for `stocks_below_20d`, 252 for `new_lows`). If fewer than **100** stocks are eligible, the flag is False (too few to read breadth).
- **Missing month list** (before the history's first month): the stock flags are False. *(Decided with the user, 2026-11-20:)* the history is first extended back to **2014-01** with the existing `uv run batch-universe --history --start 2014-01`, so the 503-session replay that the variant needs from 2016-01-04 sees real breadth over 2014-2015 (it covers the Aug 2015 fall that precedes the Jan 2016 episode). If the extension fails or adds months with fewer than 100 eligible names, the record says so and those indices stay False.
- **Agreement uses inputs already scored on this window** (a mild look-ahead, recorded); all six are used, not the ones that looked fast.
- The first screen's six candidates and its rows are printed again unchanged (same thresholds), for context; they compete under the same rule.

## The how

- **Stock closes:** one research download of the union of the lists' symbols over the needed months (about 2,300 symbols from the 2016-2026 file; more after the 2014 extension), daily, adjusted, from 2013-01 (so `new_lows` has 252 closes at 2014-01) to the requested end, through the existing `batch_us_stock_universe.download_batch` (batched, retried, rate-limited). Aligned on SPY's sessions; a stock's missing days stay missing (no forward fill for stocks: a gap means "not eligible that day"). Cached on disk under `cache/` (git-ignored) so the download happens once; the cache stores what was downloaded and the reader trims to the requested end.
- **Sector closes:** the 11 ETFs added to the research download (`fast_regime_data.RESEARCH_SYMBOLS` or a sibling), forward-filled up to 5 sessions like the others; a sector's leading rows before its first close stay "not available" (do not trim all rows to XLC's start).
- **Pure breadth maths:** a function from (sessions, month lists, per-stock closes) to the two stock flags, and one from sector closes to the sector flag; the agreement counts come from the existing `conditions()` flags. All pure and tested on small hand-made data (eligibility, the 100-name floor, the month boundary, a split-adjusted series that must not flag).
- **CLI:** `regime screen` and `regime fast` gain the new candidates with no new flags; their output format is unchanged apart from the extra rows/names. `fast --replace` accepts the new names.
- **Record:** a new replay note `docs/superpowers/replays/2026-11-20-breadth-agreement-screen.md` (the screen table, winners, phase B, held-out if run, the survivor gap and the agreement look-ahead stated), one line in `core/CLAUDE.md`.

## Calls settled after the builder check (2026-11-20)

- **The shared universe file is never written.** The research works on a copy: `data/regime_research/us_stock_universe_history.json` in the **main checkout** (`/mnt/data/projets/trading-agent-framework`, where `data/` and `cache/` live and `data/` is git-ignored), copied from `data/universe/us_stock_universe_history.json`. The 2014-01 extension runs **from the main checkout** with the existing command and its pool cache (pool dated 2026-11-03, so the lists stay comparable): `uv run batch-universe --history --start 2014-01 --output-dir data/regime_research`. After it, every month already in the copy (2016-01 to 2026-11) must be byte-identical to the original's; if any differs, stop and report. The extension re-downloads the pool's bars from about 2013-09 (about an hour); that is accepted.
- **`screen` and `fast` only load the research copy** (path passed in; default the main checkout's `data/regime_research/...`). A month missing from it is an error (exit 1, message names the month), never a call to `ensure_universe_history` (which can start a full rebuild).
- **Stock closes cache:** under the main checkout's `cache/regime_research/` (git-ignored). If a later run needs dates past the cached end (only `--held-out`), the whole range is downloaded again rather than appended, so adjusted closes never join two downloads (a split between them would look like a crash).
- **Absent symbols:** a symbol Yahoo returns nothing for is left out and counted; if more than 1% of the needed symbols are absent (the `MAX_ABSENT_FRACTION` rule of the history build), stop with an error naming the count.
- **The held-out look waits for the user** (decided with the user, 2026-11-20): the runs stop after `regime fast` on the development window; on a PASS the development scorecard goes to the user, and `--held-out` runs only after their explicit go-ahead. On a FAIL the record is written and nothing else runs.
- **If the screen picks the same winners as the first screen** (`vix_jump`, `short_drawdown`, same-day yes), phase B is not rerun: its inputs and output are identical to the recorded first-screen run (-5.2%, FAIL); the record says the new candidates did not win and the result is NO.

## Left to the builder

Module and function names, the cache format and location under `cache/`, how the universe history and the stock closes reach `screen`/`fast` (injected functions so tests stay offline), and the table layout.
