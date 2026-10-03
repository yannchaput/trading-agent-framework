# Year-chunked minute data for long backtests

Date: 2026-10-03. Status: design, awaiting review.

## Problem

`uv run agent vwap_pullback_continuation backtesting` on the 5Y window
(`PredefinedWindow.SEMI_DECADE`, commit `faf0f45`) was OOM-killed three times on
2026-10-03 (python3 at 54-58 GB anon RSS on a 60 GB machine). Because it ran from a
Tabby tab, systemd's `OOMPolicy=stop` then killed the whole Tabby scope.

Two causes, both in `AlpacaBacktestData`:

1. **Peak fetch.** The first `Scanner.prepare_session` keeps `stage1_size = 150`
   survivors -- exactly one Alpaca request (`MAX_SYMBOLS_PER_REQUEST = 150`) -- and
   `_preload(..., "minute")` asks for them over the whole `[warmup_start, end]` window
   (~5.3 years, extended hours included). The SDK materializes ~180M pydantic `Bar`
   objects and `_fetch_many` keeps the parsed result for every chunk until the end.
   The run died before the first `stage 1 for ...` log line.
2. **Unbounded residency.** `_frames` keeps every `(asset, timestep)` frame for the
   whole window, for the whole run. Stage-1 survivors change daily, so over 5 years
   the resident set tends to all 1200 symbols x 5 years of minutes.

## Goal

A backtest whose window is longer than one year never fetches nor keeps more than
about one year of minute bars per symbol. Success criteria:

- The vwap_pullback 5Y backtest runs to completion without OOM, with a memory
  footprint comparable to the YEAR-window run that completed on 2026-10-02.
- A window of one year or less behaves exactly as today (one chunk).
- Daily bars, sessions and every no-look-ahead guarantee are unchanged.

## Decisions taken in brainstorming

| Question | Decision |
|---|---|
| Split unit | Year by year, only when the window is longer than 1Y. |
| Timesteps | Minute only. Daily frames are small (~70 MB for 5Y x 1200) and the regime needs 283 daily bars of lookback. |
| Where | A new wrapper data source (approach A); `AlpacaBacktestData` is not modified. |
| Wiring | vwap_pullback only. news_binary also backtests on minutes but is not wired (YAGNI). |

## Design

### `YearChunkedData` (`backtesting/data/chunked.py`)

A `BacktestDataSource` built like any other source, `(start, end)` first, so it can
be handed to `Strategy.run_backtesting(data_source=...)` as a callable:

```python
YearChunkedData(start, end, *, inner: Callable[[datetime, datetime], BacktestDataSource])
```

`inner` is a source factory (e.g. the `AlpacaBacktestData` class). The wrapper
holds two kinds of inner sources:

- **`_whole`**: `inner(start, end)`, built once. Serves every non-minute `load()`/
  `bars()` call and every `sessions()` call, unchanged.
- **`_minute`**: `inner(fetch_start, chunk_end)` for the current minute chunk only.
  Replaced (the old instance dropped, freeing all its frames) when the chunk changes.

`name` is the inner source's name (as `CachedDataSource` does), so `settings.json`'s
`backtesting_data_sources` still reads `alpaca`.

### Chunks (pure helpers in the same module)

- `CHUNK = timedelta(days=365)`, `OVERLAP = timedelta(days=30)`.
- The window is chunked only when `end - start > CHUNK`; otherwise there is a single
  chunk `[start, end]` and the wrapper's minute path is equivalent to today's.
- Chunk `k` covers `[start + k*CHUNK, min(start + (k+1)*CHUNK, end)]`; the last
  chunk is shorter.
- `chunk_index(cutoff)` = `floor((cutoff - start) / CHUNK)`, clamped to
  `[0, last]` (a cutoff before `start` maps to chunk 0, after `end` to the last).
- Chunk `k` is **fetched** over `[chunk_start - OVERLAP, chunk_end]` (not before
  `start`), so a lookback early in a chunk never needs the previous one. 30 calendar
  days is ~20 sessions; vwap's largest minute lookback is
  `rvol_baseline_sessions + 1` = 11 sessions of regular-hours minutes.

### Which chunk is current

The wrapper keeps `_position`: the latest `cutoff` any `bars()` call (minute or day)
has passed, starting at `start`. A backtest clock only moves forward, and the scanner
reads its daily bars at the session's `now` before preloading minutes, so the
position is already on the right year when the minute preload arrives.

- **`load(assets, start, end, "minute")`**: ignores the given `start`/`end` and
  loads `assets` into the inner for `chunk_index(_position)` (switching chunk first
  if needed). Allowed: `load()` is a pre-warming optimisation, not a contract on the
  window (`data/base.py`). This is how vwap's per-session `_preload` and the runner's
  eager benchmark load stay batched instead of one lazy fetch per asset.
- **`bars(asset, cutoff, length, "minute")`**: updates `_position`, switches to
  `chunk_index(cutoff)` if it is later than the current chunk, then delegates to the
  current minute inner (which fetches lazily for an asset it has not loaded).
- **A cutoff before the current inner's fetch start** (`chunk_start - OVERLAP`)
  raises `BacktestDataError`: going back in time would silently refetch a past year,
  and a backtest clock never does it. A cutoff inside the overlap is served.

Chunk switching is guarded by a `threading.Lock` (cheap; cross_momentum-style thread
fan-out reaches `bars()` concurrently).

### Wiring

`VwapPullbackStrategy.run_backtesting` passes
`data_source=partial(YearChunkedData, inner=AlpacaBacktestData)` instead of
`AlpacaBacktestData`. `Strategy.run_backtesting` already calls a callable source as
`data_source(warmup_start, resolved_end)`, so nothing else changes. vwap's
`_preload` keeps passing the whole window; the wrapper narrows it. It is imported
from its module, like the other sources (`backtesting/data/__init__.py` exports none).

### Memory estimate

Largest single fetch: 150 symbols x ~13 months instead of ~5.3 years (about 4x
smaller). Resident minute frames: at most one chunk (+ overlap) of the symbols seen
during that year -- the YEAR-window footprint. The previous chunk is freed at the
switch.

## Out of scope

- Splitting a single request by symbol count, float32 frames, dropping extended
  hours at fetch time.
- Chunking daily bars; wiring news_binary or other strategies.
- `CachedDataSource` (vwap does not use it).

## Testing

Unit tests in `tests/backtesting/data/test_chunked.py`, no network, with a recording
fake inner source (built per call with its `(start, end)` window, recording `load()`
and `bars()` calls; based on `tests/backtesting/fakes.FakeBacktestDataSource`):

- Window <= 1Y: one minute inner over the whole window; behaviour equals the inner's.
- Window > 1Y: chunk boundaries and the clamped `chunk_index`, including the shorter
  last chunk; the fetch window starts `OVERLAP` before the chunk (never before `start`).
- `load(..., "minute")` with the full window loads only the current chunk's inner.
- A `bars()` cutoff crossing into the next year builds the next inner and drops the
  previous one (the old instance is no longer referenced).
- Daily `load()`/`bars()` and `sessions()` go to the whole-window inner only, and a
  daily `bars()` call advances `_position`.
- A minute cutoff inside the overlap is served; one before the fetch start raises
  `BacktestDataError`.
- `name` mirrors the inner's.

Plus one test that `VwapPullbackStrategy.run_backtesting` passes a `YearChunkedData`
factory. Manual check: rerun the 5Y vwap backtest in a capped scope
(`systemd-run --user --scope -p MemoryMax=40G ...`) and watch its RSS.
