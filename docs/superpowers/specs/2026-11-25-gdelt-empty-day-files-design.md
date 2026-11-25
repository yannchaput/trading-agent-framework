# GDELT news: an empty day file is a missing day

## Why
17 day files of the GDELT cache (2025-06-15 .. 2025-07-01) are 636-byte parquet files with no columns. `_day_frame`
(`backtesting/news.py`) handles a missing day file but reads every existing one with `columns=["stamp", ...]`, so it
raises `No match for FieldRef.Name(stamp)`. The researcher run fails, the pipeline retries it with `submit_note` forced
and no search, so bull_bear's reviews from 2025-06-17 to 2025-08-05 (run 2026-11-24_235119) got nearly all their notes
written without news, invisibly (`reviews.jsonl` shows no failed note).

## Goal / non-goals
- Goal: a day file without a `stamp` column reads as an empty day, like a missing one.
- Non-goals: no change to the cache files, to good days, or to how gaps are reported.

## Approaches
- **Treat it as a missing day in `_day_frame` (picked):** one check, at the one place that reads day files.
- Rewrite or delete the 17 files: fixes this cache only, and the next download could write empty files again.

## Design
`_day_frame` returns its empty frame when the file is missing or its parquet schema has no `stamp` column (read from the
file's metadata, without loading data). Regression test in `tests/backtesting/test_gdelt_news.py`: a window spanning a
good day and a no-column day file returns the good day's article and does not raise.
