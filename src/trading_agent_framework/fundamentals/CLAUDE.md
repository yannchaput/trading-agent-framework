# fundamentals/ -- SEC EDGAR client

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

## Architecture

- `fundamentals/` -- SEC EDGAR client, trimmed and ported from lumibot's `SECFundamentals`: `sec.py` (**pure**: tag maps, as-of candidate filtering, statement-period matching, filings parsing, URL building, HTML stripping, `parse_dt`) and `edgar_client.py` (`SecEdgarClient`, the only module allowed to import `httpx` for SEC access -- cached to `<project_root>/cache/sec/`, rate-limited, requires `SEC_EDGAR_USER_AGENT`; `fetch_json` is its uncached path and HTTP 404 raises `FundamentalsNotFoundError`; the cached payload getters take optional `as_of`/`max_age_days`, so a payload older than max_age_days by the strategy clock is fetched again (the drill-down tools pass 30) -- a backtest never refetches an existing file, live refreshes monthly), plus `freshness.py` (`is_stale`, the cache-age rule). The quality screen built on it lives in `strategies/bill_ackman/screen/`.
