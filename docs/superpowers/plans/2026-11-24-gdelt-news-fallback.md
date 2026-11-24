# GDELT News Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In backtests, `search_news` answers from the GDELT cache before 2026-04-29 and from Alpaca after it.

**Architecture:** One new module `backtesting/news.py` with a `GdeltNewsProvider` (reads the cached day parquet
files) and a `CutoffNewsProvider` (routes a window to GDELT or Alpaca by date). `BacktestBroker.news_provider()`'s
default becomes the router. `search_news` and its clock gate do not change.

**Tech Stack:** Python 3.14, pandas/pyarrow (already installed), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-24-gdelt-news-fallback-design.md`

## Global Constraints

- Cutoff: `datetime(2026, 4, 29, tzinfo=UTC)`, a default argument, not an env setting.
- Default paths, relative to the working directory: `cache/gdelt/gkg_stockmarket` and `cache/sec/company_tickers.json`.
- Article keys exactly as Alpaca's `parse_news`: `id`, `headline`, `summary`, `source`, `created_at`, `symbols`; no `content` key.
- No article with `stamp > end` is ever returned (no look-ahead).
- Tests never touch the network or the real `cache/`: build tiny parquet / JSON files in `tmp_path`.
- Google-style docstrings on every public module, class and function; KISS (plain functions, dicts, no new types beyond the two providers).
- Quality gate per the root `CLAUDE.md`: ruff check, ruff format --check, pyright 0 errors, pytest, coverage >= 80%, interrogate >= 80%. No `# noqa` / `# type: ignore`.

## Review Focus

- A window in a non-UTC timezone (the tool passes ET datetimes): 20:00 ET on 03-05 is 01:00 UTC on 03-06, so the 03-06 file must be read (day files are UTC dates). Test in Task 1.
- No cache directory at all: must fail loud (`BrokerError`, which the tool turns into `{"error": ...}`), not return a silent empty list like the Alpaca archive drop did. Test in Task 1.
- A day file with zero rows, or a row with null title AND null url: no crash, row dropped. Test in Task 1.
- A window ending exactly at the cutoff, or starting exactly on it: each article comes from one source only, never twice. Test in Task 3.
- `start=None` (allowed by the `NewsProvider` protocol): GDELT reads only `end`'s UTC day; the router sends it to `after` when `end >= cutoff`. Tests in Tasks 1 and 3.

---

### Task 1: `GdeltNewsProvider` core (day files, slug titles, window, dedup, sort, shape)

**Files:**
- Create: `src/trading_agent_framework/backtesting/news.py`
- Test: `tests/backtesting/test_gdelt_news.py`

**Interfaces:**
- Produces: `slug_title(url: str | None) -> str | None`;
  `GdeltNewsProvider(root: Path = Path("cache/gdelt/gkg_stockmarket"), names: Path = Path("cache/sec/company_tickers.json"))`
  with `get_news(symbols=(), *, start=None, end, limit=10, include_content=False, sort=None) -> list[dict[str, object]]`
  (the `NewsProvider` signature). In this task `symbols` is ignored (every row matches); Task 2 adds matching.

Test helper (in the test file): `_day(root, day: str, rows: list[dict])` writes `root/<YYYY>/<day>.parquet` with
columns `stamp, domain, url, title` (all str, title may be None). Fixture titles and slugs all contain
"Wall Street" (e.g. `wall-street-rallies-on-jobs`), so these no-symbol tests stay valid once Task 2 adds the
US-market filter.

- [ ] **Step 1: Write the failing tests**

```python
def test_slug_title_uses_the_segment_with_most_separators_and_drops_long_ids():
    url = "http://www.ptinews.com/news/9542091_Mkt-takes-U-turn--Sensex-tumbles-430-pts-as-bank-stocks-drag.html"
    assert slug_title(url) == "Mkt takes U turn Sensex tumbles 430 pts as bank stocks drag"
    assert slug_title("http://x.com/news/ap-top-stories/republicans-want-trump-to-back-off/711910290") == "republicans want trump to back off"
    assert slug_title("https://x.com/story/5267827/?cs=233") is None
    assert slug_title(None) is None


def test_a_null_title_is_filled_from_the_url_and_a_row_with_neither_is_dropped(tmp_path):
    ...
    # rows: title None + slug url -> headline is the slug title; title None + url None -> absent


def test_only_rows_inside_the_window_are_returned(tmp_path):
    ...
    # stamps 20240305093000, 20240305150000, 20240305160001; end = 2024-03-05 16:00 UTC -> the third is absent


def test_an_et_window_reads_the_next_utc_day_file(tmp_path):
    ...
    # article in 2024-03-06 file stamped 20240306010000; start/end 2024-03-05 19:00-21:00 America/New_York -> returned


def test_start_none_reads_only_the_end_day(tmp_path): ...


def test_missing_day_files_and_empty_files_are_skipped(tmp_path): ...


def test_a_missing_cache_root_raises_broker_error(tmp_path):
    with pytest.raises(BrokerError, match="GDELT news cache not found"):
        GdeltNewsProvider(root=tmp_path / "nope").get_news(end=datetime(2024, 3, 5, tzinfo=UTC))


def test_identical_titles_keep_the_earliest_then_newest_first_and_limit(tmp_path):
    ...
    # same title at 10:00 and 11:00, other titles at 12:00, 13:00; limit=2 -> [13:00, 12:00]; the duplicate's created_at is 10:00


def test_sort_asc_returns_oldest_first(tmp_path): ...


def test_article_shape_matches_alpaca(tmp_path):
    a = provider.get_news(end=..., include_content=True)[0]
    assert a == {"id": url, "headline": title, "summary": "", "source": domain, "created_at": "2024-03-05T09:30:00+00:00", "symbols": []}
```

- [ ] **Step 2: Run them, expect FAIL** (`ImportError`)

Run: `uv run pytest tests/backtesting/test_gdelt_news.py -v`

- [ ] **Step 3: Implement in `backtesting/news.py`**

- `slug_title`: path segments of `urllib.parse.urlparse(url).path`; pick the one with the most `-`/`_` (none -> `None`); strip a trailing `.html`/`.htm`/`.php`/`.aspx`; split on `[-_]+`; drop tokens matching `^\d{5,}$`; join with `" "`; empty -> `None`.
- `get_news`: raise `BrokerError(f"GDELT news cache not found at {root}")` if `root` is not a directory. Convert `start`/`end` to UTC; days = `start.date()`..`end.date()` (only `end.date()` when `start` is None). For each day, `_day_frame(day)` (cached with `functools.lru_cache(maxsize=64)` on a module-level function of `(root, day)`) reads columns `stamp, domain, url, title` if the file exists, fills null titles with `slug_title(url)`, drops rows still without a title, parses `stamp` with `format="%Y%m%d%H%M%S", utc=True`. Concat, keep `start <= stamp <= end`, sort ascending, `drop_duplicates("title")`, order (`sort == "asc"` ascending, else descending), `head(limit)`, build dicts.

- [ ] **Step 4: Run them, expect PASS**

- [ ] **Step 5: Commit** — `feat(backtesting): GDELT news provider reading the cached day files`

### Task 2: Symbol matching and the no-symbol US-market filter (rule C)

**Files:**
- Modify: `src/trading_agent_framework/backtesting/news.py`
- Test: `tests/backtesting/test_gdelt_news.py`

**Interfaces:**
- Consumes: Task 1's `GdeltNewsProvider`.
- Produces: `short_name(sec_title: str) -> str | None`; `ALIASES: dict[str, str]` = `{"SPY": "S&P 500", "VOO": "S&P 500", "IVV": "S&P 500", "QQQ": "Nasdaq", "DIA": "Dow"}`; `get_news` now filters on `symbols` and fills each article's `symbols` with the requested tickers its title matched.

Test helper: `_names(path, {"NVDA": "NVIDIA CORP", "AAPL": "Apple Inc.", "BP": "BP p.l.c."})` writes SEC's shape `{"0": {"cik_str": 1, "ticker": "NVDA", "title": "NVIDIA CORP"}, ...}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_short_name_drops_legal_suffixes_and_refuses_short_names():
    assert short_name("NVIDIA CORP") == "NVIDIA"
    assert short_name("Apple Inc.") == "Apple"
    assert short_name("Alphabet Inc.") == "Alphabet"
    assert short_name("Super Micro Computer, Inc.") == "Super Micro Computer"
    assert short_name("BP p.l.c.") is None  # under 4 characters


def test_a_symbol_matches_its_ticker_tag_or_its_name_case_insensitive(tmp_path):
    ...
    # titles: "Why Nvidia Stock Was Popping", "Fund buys (NASDAQ:NVDA) shares", "Apple fined" ; symbols=["NVDA"]
    # -> first two, each with symbols == ["NVDA"]


def test_a_name_matches_whole_words_only(tmp_path):
    ...
    # "Pineapple prices rise" is not an AAPL match


def test_a_short_name_matches_by_ticker_tag_only(tmp_path):
    ...
    # "BP sees profit" not matched, "Shares rise (NYSE:BP)" matched


def test_aliases_cover_broad_etfs(tmp_path):
    ...
    # "S&P 500 hits record" matched for SPY, with symbols == ["SPY"]


def test_an_article_lists_every_requested_symbol_it_matched(tmp_path):
    ...
    # "Nvidia and Apple rally" with symbols=["NVDA", "AAPL"] -> symbols == ["NVDA", "AAPL"]


def test_without_symbols_only_us_market_titles_without_ticker_tags(tmp_path):
    ...
    # "Wall Street closes higher" kept; "Nasdaq falls - NVIDIA (NASDAQ:NVDA)" dropped; "2024 Subaru Solterra review" dropped


def test_a_missing_names_file_falls_back_to_ticker_tags_and_aliases(tmp_path): ...
```

- [ ] **Step 2: Run them, expect FAIL**

- [ ] **Step 3: Implement**

- `short_name`: repeatedly strip, case-insensitive, trailing `,`/`.` and suffixes `inc`, `corp`, `corporation`, `co`, `company`, `ltd`, `plc`, `p.l.c`, `holdings`, `group`, `class a`/`b`/`c`, `/de/`, `n.v.`, `s.a.`, `the` prefix; result under 4 characters -> `None`.
- Names are read once (lazily, on the first `get_news`) into `{ticker: short name or None}`; a missing file -> `{}`. For a symbol: patterns = ticker tag `\(\s*[A-Z][A-Z ]*:\s*{ticker}\s*\)` (case-sensitive) plus `\b{re.escape(name)}\b` (`re.I`) for `ALIASES.get(t)` or the SEC short name, whichever exists (alias first).
- Without symbols: copy `US` and `TICKER` from `scripts/news/early_regime_spike.py:25-26`; keep `US` matches without a `TICKER` match.
- Matching runs before dedup and limit.

- [ ] **Step 4: Run the whole file, expect PASS** (Task 1's tests included)

Run: `uv run pytest tests/backtesting/test_gdelt_news.py -v`

- [ ] **Step 5: Commit** — `feat(backtesting): GDELT news matches tickers, company names and US-market titles`

### Task 3: `CutoffNewsProvider`

**Files:**
- Modify: `src/trading_agent_framework/backtesting/news.py`
- Test: `tests/backtesting/test_cutoff_news.py`

**Interfaces:**
- Produces: `GDELT_CUTOFF = datetime(2026, 4, 29, tzinfo=UTC)`;
  `CutoffNewsProvider(before: NewsProvider, after: Callable[[], NewsProvider], cutoff: datetime = GDELT_CUTOFF)` with the `NewsProvider` `get_news` signature. `after` is a factory called at most once, on the first window reaching the cutoff.

Test fakes: `_Recording(articles)` stores each call's kwargs and returns articles whose `created_at` lies in the asked window.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_window_before_the_cutoff_asks_only_before_and_never_builds_after(): ...
def test_a_window_from_the_cutoff_on_asks_only_after(): ...  # start == cutoff
def test_a_window_ending_exactly_at_the_cutoff_asks_after_for_the_cutoff_instant_only(): ...
def test_a_spanning_window_splits_at_the_cutoff_merges_newest_first_and_limits():
    ...
    # before called with end == cutoff - timedelta(microseconds=1), after with start == cutoff; limit 3 of 4 merged


def test_a_spanning_window_with_sort_asc_merges_oldest_first(): ...
def test_start_none_with_end_after_the_cutoff_asks_only_after(): ...
def test_the_after_factory_is_built_once(): ...
def test_a_factory_broker_error_propagates(): ...  # the tool turns it into {"error": ...}
```

- [ ] **Step 2: Run them, expect FAIL**
- [ ] **Step 3: Implement** — `end < cutoff` -> `before`; `start is None or start >= cutoff` -> `after`; else both, each with the same `symbols/limit/include_content/sort`, merged and sorted by `datetime.fromisoformat(a["created_at"])` (reverse unless `sort == "asc"`), cut to `limit`.
- [ ] **Step 4: Run them, expect PASS**
- [ ] **Step 5: Commit** — `feat(backtesting): route news before 2026-04-29 to GDELT, after it to Alpaca`

### Task 4: Wire the default into `BacktestBroker` and document

**Files:**
- Modify: `src/trading_agent_framework/backtesting/broker.py:220-231` (`news_provider`)
- Modify: `tests/backtesting/test_broker_news.py`
- Modify: `src/trading_agent_framework/backtesting/CLAUDE.md`, `src/trading_agent_framework/agents/CLAUDE.md` (the "Research agent tools" gotcha: the `BacktestBroker` default)
- Modify: `src/trading_agent_framework/core/strategy.py:774` and `backtesting/runner.py:135` docstrings ("defaults to an Alpaca provider" -> the GDELT/Alpaca router)

**Interfaces:**
- Consumes: `GdeltNewsProvider()`, `CutoffNewsProvider`, existing `lazy_news_provider(AlpacaCredentials.for_news)` from `brokers/alpaca/news.py`.

- [ ] **Step 1: Update the tests**

```python
def test_the_default_news_source_is_the_cutoff_router_memoized():
    broker = _broker()
    assert isinstance(broker.news_provider(), CutoffNewsProvider)
    assert broker.news_provider() is broker.news_provider()


def test_the_default_builds_alpaca_lazily_only_after_the_cutoff(monkeypatch):
    ...
    # replaces the old "built lazily" test: fake from_credentials; get_news with end before the cutoff
    # monkeypatch.chdir(tmp_path) with an empty tmp_path/cache/gdelt/gkg_stockmarket dir, so the default root
    # resolves there -> built == []; a window after the cutoff -> built once


def test_missing_alpaca_credentials_surface_as_a_broker_error_after_the_cutoff(monkeypatch):
    ...
    # replaces the old one: news_provider() itself no longer raises; get_news(end=2026-05-04) raises BrokerError "no news source available"
```

- [ ] **Step 2: Run, expect FAIL**
- [ ] **Step 3: Implement** — the default branch builds `CutoffNewsProvider(GdeltNewsProvider(), lazy_news_provider(AlpacaCredentials.for_news))` (imports stay deferred inside the method); docstring says so. Update the docs listed above in a line or two each.
- [ ] **Step 4: Run the full gate**

```bash
uv sync --extra dashboard
uv run ruff check && uv run ruff format --check && uv run pyright
uv run pytest --cov
uv run interrogate -v
```
Expected: all green, coverage >= 80%, interrogate >= 80%.

- [ ] **Step 5: Manual check against the real cache (read-only)**

```bash
uv run python -c "
from datetime import datetime, UTC
from trading_agent_framework.backtesting.news import GdeltNewsProvider
p = GdeltNewsProvider()
for s, d in [(['NVDA'], datetime(2024,3,5,21,tzinfo=UTC)), ([], datetime(2024,3,5,21,tzinfo=UTC)), (['AAPL'], datetime(2018,3,6,21,tzinfo=UTC))]:
    print(s, d.date(), [a['headline'] for a in p.get_news(s, end=d, limit=5)])
"
```
Expected: Nvidia headlines, US-market headlines, Apple slug titles; each call under a few seconds.

- [ ] **Step 6: Commit** — `feat(backtesting): default backtest news is GDELT before 2026-04-29, Alpaca after`
