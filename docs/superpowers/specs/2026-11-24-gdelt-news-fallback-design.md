# GDELT news fallback for `search_news` (design)

## Why

Alpaca's news archive now starts on 2026-04-29: any backtest simulating an earlier date gets empty `search_news`
results, silently (the 2026-11-17 bull_bear run had empty news for 6,110 of 6,114 pre-cutoff calls). The GDELT GKG
cache in `cache/gdelt/gkg_stockmarket/` holds stock-market news from 2015 to 2026. Before the cutoff, `search_news`
answers from that cache; after it, from Alpaca as today. Mostly for backtests.

## Goals / non-goals

- Goal: news for simulated dates before 2026-04-29, with the same no-look-ahead guarantee and article shape as today.
- Goal: simple, documented code behind the existing `NewsProvider` seam; `search_news` itself does not change.
- Non-goal: summaries or bodies (GDELT has titles only; `include_content=True` returns titles only).
- Goal (user request): rows with no title (every row before 2019-09-23) get a title made from their URL slug.
- Non-goal: paper/live (they keep Alpaca), a pre-built index, historical company names.
- Anti-goal: anything that can show an article stamped after the clock's `end`.

## Approaches

1. **GDELT provider + date-routing provider, the backtest default (picked).** Two small classes behind
   `NewsProvider`; the tool and its clock gate stay as they are.
2. Date branch inside `search_news`: fewer files, but the tool would choose data sources (the project keeps the clock
   gate in the tool and data in providers) and paper/live would carry GDELT code.
3. Pre-built slim index from a batch step: faster queries, but a rebuild step and more parts. Kept for later if
   per-day reads turn out slow.

The user picked matching rule C and left the how to Claude; Claude picked approach 1.

## Design

One new module, `backtesting/news.py`:

- **`GdeltNewsProvider(root=Path("cache/gdelt/gkg_stockmarket"), names=Path("cache/sec/company_tickers.json"))`**
  - Reads the day files `<root>/<YYYY>/<YYYY-MM-DD>.parquet` covering `[start, end]` (columns `stamp, domain, url,
    title` only), keeps rows with `start <= stamp <= end` (`stamp` is UTC `%Y%m%d%H%M%S`, the earliest time
    the article is known). A missing day file (e.g. the 2025-06-15 to 2025-07-01 gap) is skipped. Day frames are kept
    in an in-memory cache (`functools.lru_cache` on the day), since a backtest asks for the same days many times.
  - **Title from the URL** (user request), when `title` is null: take the URL path's segment with the most `-`/`_`
    separators, drop a file extension (`.html`), split on `-` and `_`, drop all-digit tokens of 5+ digits (article
    ids and dates like `9542091`, `20180306`; short numbers like `430` stay), join with single spaces. A URL with no
    such segment (e.g. `/story/5267827/`) gets no title and the row is dropped. Example:
    `.../9542091_Mkt-takes-U-turn--Sensex-tumbles-430-pts-as-bank-stocks-drag.html` ->
    `Mkt takes U turn Sensex tumbles 430 pts as bank stocks drag`. Matching and dedup then use that title.
  - **Matching (rule C):** with symbols, a title matches a symbol if it has the ticker tag (`(NASDAQ:NVDA)`, any
    exchange) or the company's short name as a whole word, case-insensitive. The short name is SEC's `title` with
    legal suffixes dropped (`Inc`, `Corp`, `Corporation`, `Co`, `Ltd`, `plc`, `Holdings`, `Group`, `Class A`...);
    names shorter than 4 characters are not used (ticker tag only). A small alias dict covers broad ETFs
    (`SPY`/`VOO`/`IVV` -> "S&P 500", `QQQ` -> "Nasdaq", `DIA` -> "Dow"). Without symbols, the US-market filter of the
    sentiment study: US-market words in the title (`wall street`, `S&P`, `Dow`, `Nasdaq`, ...) and no ticker tag.
  - Identical titles keep only the earliest. Sorted newest first by default, oldest first for `sort="asc"`, then cut
    to `limit`.
  - Article shape (same keys as Alpaca's `parse_news`): `id` = url, `headline` = title, `summary` = "",
    `source` = domain, `created_at` = ISO UTC from `stamp`, `symbols` = the requested symbols the title matched
    (empty for a no-symbol search). No `content` key.
- **`CutoffNewsProvider(before, after, cutoff=datetime(2026, 4, 29, tzinfo=UTC))`**: `[start, end]` wholly before
  the cutoff goes to `before` (GDELT), wholly after to `after` (Alpaca). A window spanning it asks both
  (`before` for `[start, cutoff)`, `after` for `[cutoff, end]`), merges, sorts by `created_at` in the asked order and
  cuts to `limit`. `after` is built lazily, so a backtest that never reaches the cutoff needs no `ALPACA_NEWS_*`.
- **Wiring:** `BacktestBroker.news_provider()`'s default (no injected `news_source`) becomes
  `CutoffNewsProvider(GdeltNewsProvider(), <the lazy Alpaca provider it builds today>)`. Injecting `news_source`
  still overrides it (e.g. an Alpaca-only rerun).

## Testing

Unit tests on tiny parquet day files and a tiny `company_tickers.json` in `tmp_path` (no network, no real cache):
slug titles (both separators, id dropped, no-slug URL dropped), ticker-tag and name matches, short-name skip, aliases, no-symbol US filter, `stamp > end` excluded, missing day file,
dedup, sort and limit; the router's three window cases with fake providers. Docs: `backtesting/CLAUDE.md` and the
news line of `agents/CLAUDE.md`.

## Known limits

- SEC names are today's names: older articles under a former name (Facebook for META) match only by ticker tag.
- Slug titles are lower-case or truncated on some sites and rarely carry ticker tags, so before 2019-09-23 matching
  relies on company names (about 78% of slugs are usable).
- Name matching has some false hits (e.g. "Apple" the fruit); accepted.
- Runs on GDELT news are not like-for-like with older Benzinga-news runs (titles only, different sources).

## Left to the builder

The exact suffix list and US-market regex wording (start from `scripts/news/early_regime_spike.py`), the cache size,
and where the default paths are resolved (relative to the working directory, as other caches are).
