# Year-chunked minute data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A backtest longer than one year never fetches nor keeps more than about one year of minute bars per symbol, so the vwap_pullback 5Y backtest stops getting OOM-killed.

**Architecture:** A new `BacktestDataSource` wrapper, `YearChunkedData` (`backtesting/data/chunked.py`), built from a source factory. Day bars and sessions go to one inner source over the whole window; minute bars go to an inner source built for the current 1-year chunk only (fetched from 30 days before the chunk), replaced when the backtest clock crosses into the next year. `VwapPullbackStrategy.run_backtesting` passes it as its `data_source`.

**Tech Stack:** Python 3.14, pandas, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-10-03-year-chunked-minute-data-design.md`

## Global Constraints

- `CHUNK = timedelta(days=365)`, `OVERLAP = timedelta(days=30)`.
- Chunk only when `end - start > CHUNK`; otherwise one chunk `[start, end]`, equivalent to today.
- Minute only. Day bars, `sessions()` and every no-look-ahead guarantee are unchanged.
- `AlpacaBacktestData` is not modified. The wrapper is imported from its module (`backtesting/data/__init__.py` exports no source).
- Wired into vwap_pullback only (`partial(YearChunkedData, inner=AlpacaBacktestData)`).
- A minute cutoff before the current chunk's fetch start raises `BacktestDataError`.
- Tests never touch the network; use `tests/backtesting/fakes.FakeBacktestDataSource`.
- Run commands with `uv run` (`uv run pytest ...`, `uv run ruff check`).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A cutoff exactly on a chunk boundary belongs to the NEXT chunk (whose fetch window still covers the last 30 days of the previous one) -- pinned in Task 1.
- The last chunk's fetch end is exactly the window's `end` (a bare midnight meaning "through that day"), never a computed `start + k*CHUNK` -- pinned in Task 1.
- A minute `load()` before any `bars()` call targets chunk 0 (position starts at `start`) -- pinned in Task 2.
- A minute `bars()` for an asset never `load()`ed in the current chunk falls through to that chunk's inner source (lazy fetch), not to the whole-window one -- pinned in Task 2.
- A day `bars()` with a cutoff past `end` (the runner's final benchmark read) leaves later minute calls working on the last chunk -- pinned in Task 2.

---

### Task 1: Pure chunk helpers

**Files:**
- Create: `src/trading_agent_framework/backtesting/data/chunked.py`
- Test: `tests/backtesting/data/test_chunked.py`

**Interfaces:**
- Produces: `CHUNK: timedelta`, `OVERLAP: timedelta`, `chunk_count(start: datetime, end: datetime) -> int`, `chunk_index(cutoff: datetime, start: datetime, end: datetime) -> int`, `chunk_window(index: int, start: datetime, end: datetime) -> tuple[datetime, datetime]` (the chunk's FETCH window).

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/data/test_chunked.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta

from trading_agent_framework.backtesting.data.chunked import CHUNK, OVERLAP, chunk_count, chunk_index, chunk_window
from trading_agent_framework.utils.clock import MARKET_TZ

START = datetime(2021, 1, 4, tzinfo=MARKET_TZ)
LONG_END = START + 3 * CHUNK + timedelta(days=100)  # four chunks, the last one 100 days long


def test_a_window_of_one_year_or_less_is_one_chunk() -> None:
    end = START + CHUNK
    assert chunk_count(START, end) == 1
    assert chunk_window(0, START, end) == (START, end)
    assert chunk_index(START - timedelta(days=5), START, end) == 0
    assert chunk_index(end + timedelta(days=5), START, end) == 0


def test_a_longer_window_is_split_year_by_year() -> None:
    assert chunk_count(START, LONG_END) == 4
    assert chunk_window(0, START, LONG_END) == (START, START + CHUNK)  # never fetched before `start`
    assert chunk_window(1, START, LONG_END) == (START + CHUNK - OVERLAP, START + 2 * CHUNK)
    assert chunk_window(3, START, LONG_END) == (START + 3 * CHUNK - OVERLAP, LONG_END)  # the last ends exactly at `end`


def test_chunk_index_is_clamped_and_a_boundary_belongs_to_the_next_chunk() -> None:
    assert chunk_index(START - timedelta(days=1), START, LONG_END) == 0
    assert chunk_index(START + CHUNK - timedelta(seconds=1), START, LONG_END) == 0
    assert chunk_index(START + CHUNK, START, LONG_END) == 1
    assert chunk_index(LONG_END + timedelta(days=10), START, LONG_END) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/backtesting/data/test_chunked.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.backtesting.data.chunked'`

- [ ] **Step 3: Write the helpers**

Create `src/trading_agent_framework/backtesting/data/chunked.py`:

```python
"""Year-chunked minute bars for backtests longer than a year.

A minute frame per symbol over a multi-year window does not fit in memory: the vwap_pullback 5Y backtest
asked Alpaca for 150 symbols x 5 years of minutes in one request and was OOM-killed at 58 GB. Minute bars
are therefore fetched one year (`CHUNK`) at a time, each chunk from `OVERLAP` before its start so a
lookback early in the year never needs the previous one.
See docs/superpowers/specs/2026-10-03-year-chunked-minute-data-design.md.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

CHUNK = timedelta(days=365)
OVERLAP = timedelta(days=30)  # ~20 sessions; vwap's largest minute lookback is 11


def chunk_count(start: datetime, end: datetime) -> int:
    """Pure: how many chunks `[start, end]` splits into (1 when it is a year or less)."""
    if end - start <= CHUNK:
        return 1
    return math.ceil((end - start) / CHUNK)


def chunk_index(cutoff: datetime, start: datetime, end: datetime) -> int:
    """Pure: the chunk `cutoff` falls in, clamped to the window's chunks (a boundary opens the next one)."""
    return max(0, min((cutoff - start) // CHUNK, chunk_count(start, end) - 1))


def chunk_window(index: int, start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Pure: the window chunk `index` is FETCHED over: `OVERLAP` before its start (never before `start`), to its end."""
    chunk_start = start + index * CHUNK
    chunk_end = end if index == chunk_count(start, end) - 1 else start + (index + 1) * CHUNK
    return max(start, chunk_start - OVERLAP), chunk_end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/backtesting/data/test_chunked.py -v`
Expected: 3 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/backtesting/data/chunked.py tests/backtesting/data/test_chunked.py
git add src/trading_agent_framework/backtesting/data/chunked.py tests/backtesting/data/test_chunked.py
git commit -m "feat: year chunk helpers for minute backtest data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `YearChunkedData` data source

**Files:**
- Modify: `src/trading_agent_framework/backtesting/data/chunked.py`
- Test: `tests/backtesting/data/test_chunked.py`

**Interfaces:**
- Consumes: `CHUNK`, `OVERLAP`, `chunk_index`, `chunk_window` from Task 1.
- Produces: `YearChunkedData(start: datetime, end: datetime, *, inner: Callable[[datetime, datetime], BacktestDataSource])`, a `BacktestDataSource` (`load`, `bars`, `sessions`, `name`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/backtesting/data/test_chunked.py`. Update its imports to:

```python
from __future__ import annotations

import gc
import weakref
from datetime import datetime, timedelta

import pytest

from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame
from trading_agent_framework.backtesting.data.chunked import CHUNK, OVERLAP, YearChunkedData, chunk_count, chunk_index, chunk_window
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestDataError
```

Then append:

```python
A = Asset("AAA")
B = Asset("BBB")


class _Factory:
    """Builds a FakeBacktestDataSource per call, recording its window; keeps only weak references, so a test
    can check that a dropped chunk source is really freed. Every source serves `frame` for A at any timestep."""

    def __init__(self, frame=None) -> None:
        self.frame = frame
        self.windows: list[tuple[datetime, datetime]] = []
        self.refs: list[weakref.ref[FakeBacktestDataSource]] = []

    def __call__(self, start: datetime, end: datetime) -> FakeBacktestDataSource:
        source = FakeBacktestDataSource()
        if self.frame is not None:
            source.set_bars(A, self.frame)
        self.windows.append((start, end))
        self.refs.append(weakref.ref(source))
        return source

    def built(self, i: int) -> FakeBacktestDataSource:
        source = self.refs[i]()
        assert source is not None, f"source {i} was freed"
        return source


def test_a_short_window_serves_minutes_from_one_source_over_the_whole_window() -> None:
    end = START + timedelta(days=200)
    frame = make_close_indexed_frame([10.0, 11.0, 12.0], start=START + timedelta(days=1), freq="1min")
    factory = _Factory(frame)
    data = YearChunkedData(START, end, inner=factory)
    data.load([A], START, end, "minute")
    bars = data.bars(A, START + timedelta(days=2), 2, "minute")
    assert factory.windows == [(START, end), (START, end)]  # the whole-window source, then the minute one
    assert factory.built(1).load_windows == [(START, end)]
    assert bars is not None and list(bars.df["close"]) == [11.0, 12.0]


def test_a_minute_load_before_any_bars_call_targets_the_first_chunk() -> None:
    factory = _Factory()
    YearChunkedData(START, LONG_END, inner=factory).load([A], START, LONG_END, "minute")
    assert factory.windows[1] == chunk_window(0, START, LONG_END)


def test_a_minute_load_with_the_whole_window_loads_only_the_current_chunk() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.bars(A, START + CHUNK + timedelta(days=5), 70, "day")  # the scanner's daily read moves the position
    data.load([A, B], START, LONG_END, "minute")
    window = chunk_window(1, START, LONG_END)
    assert factory.windows[1:] == [window]
    assert factory.built(1).load_calls == [(A, B)]
    assert factory.built(1).load_windows == [window]
    assert factory.built(0).load_calls == []  # nothing reached the whole-window source


def test_a_cutoff_in_the_next_year_replaces_the_minute_source_and_frees_the_previous_one() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.bars(A, START + timedelta(days=10), 5, "minute")
    data.bars(A, START + CHUNK + timedelta(days=1), 5, "minute")
    assert factory.windows[1:] == [chunk_window(0, START, LONG_END), chunk_window(1, START, LONG_END)]
    gc.collect()
    assert factory.refs[1]() is None  # chunk 0's frames are gone
    assert factory.built(2).bars_calls == [(A, START + CHUNK + timedelta(days=1), 5, "minute")]


def test_a_minute_bars_call_for_an_asset_never_loaded_falls_through_to_the_chunk_source() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.load([A], START, LONG_END, "minute")
    data.bars(B, START + timedelta(days=3), 5, "minute")
    assert factory.built(1).bars_calls == [(B, START + timedelta(days=3), 5, "minute")]
    assert factory.built(0).bars_calls == []


def test_day_bars_load_and_sessions_go_to_the_whole_window_source() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.load([A], START, LONG_END, "day")
    data.bars(A, START + 2 * CHUNK, 5, "day")
    assert data.sessions(START, LONG_END) == []
    assert factory.windows == [(START, LONG_END)]  # no minute source was ever built
    assert factory.built(0).load_windows == [(START, LONG_END)]
    assert factory.built(0).bars_calls == [(A, START + 2 * CHUNK, 5, "day")]


def test_a_day_read_past_the_end_leaves_minutes_on_the_last_chunk() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.bars(A, LONG_END + timedelta(days=1), 1, "day")  # the runner's final benchmark read
    data.load([A], START, LONG_END, "minute")
    data.bars(A, LONG_END, 5, "minute")
    assert factory.windows[1:] == [chunk_window(3, START, LONG_END)]


def test_a_minute_cutoff_in_the_overlap_is_served_and_one_before_it_raises() -> None:
    factory = _Factory()
    data = YearChunkedData(START, LONG_END, inner=factory)
    data.bars(A, START + CHUNK + timedelta(days=1), 5, "minute")
    data.bars(A, START + CHUNK - timedelta(days=10), 5, "minute")  # inside chunk 1's 30-day overlap
    assert len(factory.windows) == 2  # served by chunk 1, nothing rebuilt
    with pytest.raises(BacktestDataError, match="before"):
        data.bars(A, START + CHUNK - OVERLAP - timedelta(days=1), 5, "minute")


def test_the_name_mirrors_the_inner_source() -> None:
    assert YearChunkedData(START, LONG_END, inner=_Factory()).name == "fake"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/backtesting/data/test_chunked.py -v`
Expected: FAIL with `ImportError: cannot import name 'YearChunkedData'`

- [ ] **Step 3: Write `YearChunkedData`**

In `src/trading_agent_framework/backtesting/data/chunked.py`, replace the imports with:

```python
from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError
```

and append after `chunk_window`:

```python
class YearChunkedData(BacktestDataSource):
    """Minute bars from a source built for the current chunk only; day bars and sessions from one source over
    the whole window. `inner` builds a source for a `(start, end)` window (e.g. the `AlpacaBacktestData` class).

    The current chunk follows the latest `cutoff` any `bars()` call has passed: a backtest clock only moves
    forward, and the scanner reads its daily bars before preloading minutes, so the position is on the right
    year by then. Crossing into the next chunk drops the previous source, and every frame it held.
    """

    def __init__(self, start: datetime, end: datetime, *, inner: Callable[[datetime, datetime], BacktestDataSource]) -> None:
        self._start = start
        self._end = end
        self._inner = inner
        self._whole = inner(start, end)
        self.name = self._whole.name
        self._position = start
        self._chunk = 0
        self._minute: BacktestDataSource | None = None
        self._lock = threading.Lock()  # a strategy may fan bars() out over threads (cross_momentum does)

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        """Day: the given window. Minute: the current chunk's window, whatever is asked (`load()` only pre-warms)."""
        if timestep != "minute":
            self._whole.load(assets, start, end, timestep)
            return
        with self._lock:
            source = self._minute_source(chunk_index(self._position, self._start, self._end))
            fetch_start, fetch_end = chunk_window(self._chunk, self._start, self._end)
        source.load(assets, fetch_start, fetch_end, timestep)

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        with self._lock:
            self._position = max(self._position, cutoff)
            source = self._whole if timestep != "minute" else self._minute_for(cutoff)
        return source.bars(asset, cutoff, length, timestep)

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        return self._whole.sessions(start, end)

    def _minute_for(self, cutoff: datetime) -> BacktestDataSource:
        """The minute source serving `cutoff` (lock held). An earlier chunk is served only within the current fetch window."""
        index = chunk_index(cutoff, self._start, self._end)
        if self._minute is not None and index < self._chunk:
            fetch_start, _ = chunk_window(self._chunk, self._start, self._end)
            if cutoff < fetch_start:
                raise BacktestDataError(
                    f"minute bars requested at {cutoff.isoformat()}, before the current chunk's window "
                    f"(from {fetch_start.isoformat()}): a backtest clock never goes back a year"
                )
            return self._minute
        return self._minute_source(index)

    def _minute_source(self, index: int) -> BacktestDataSource:
        """The source for chunk `index`, built (and the previous one dropped) when it is not the current one (lock held)."""
        if self._minute is None or index > self._chunk:
            self._minute = None  # drop the previous chunk before fetching the next one
            self._minute = self._inner(*chunk_window(index, self._start, self._end))
            self._chunk = index
        return self._minute
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/backtesting/data/test_chunked.py -v`
Expected: 12 passed

- [ ] **Step 5: Run the backtesting suite, lint, commit**

```bash
uv run pytest tests/backtesting -q
uv run ruff check src/trading_agent_framework/backtesting/data/chunked.py tests/backtesting/data/test_chunked.py
git add src/trading_agent_framework/backtesting/data/chunked.py tests/backtesting/data/test_chunked.py
git commit -m "feat: YearChunkedData keeps one year of minute bars at a time

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Wire vwap_pullback and document

**Files:**
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py` (imports; `run_backtesting` at ~line 158)
- Modify: `CLAUDE.md` (the `backtesting/` architecture bullet and the vwap_pullback gotcha)
- Test: `tests/strategies/vwap_pullback/test_vwap_strategy.py`

**Interfaces:**
- Consumes: `YearChunkedData(start, end, *, inner=...)` from Task 2.
- Produces: `VwapPullbackStrategy.run_backtesting()` passes `data_source=partial(YearChunkedData, inner=AlpacaBacktestData)` to `Strategy.run_backtesting`.

- [ ] **Step 1: Write the failing test**

Add to `tests/strategies/vwap_pullback/test_vwap_strategy.py` (imports at the top of the file):

```python
from functools import partial

import pytest

from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.backtesting.data.chunked import YearChunkedData
from trading_agent_framework.core.strategy import Strategy
```

and the test:

```python
def test_run_backtesting_reads_minute_bars_year_by_year(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: captured.update(kwargs))
    _strategy(tmp_path).run_backtesting()
    source = captured["data_source"]
    assert isinstance(source, partial)
    assert source.func is YearChunkedData and source.keywords == {"inner": AlpacaBacktestData}
    assert captured["timestep"] == "minute"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_strategy.py::test_run_backtesting_reads_minute_bars_year_by_year -v`
Expected: FAIL on `assert isinstance(source, partial)` (it is the `AlpacaBacktestData` class)

- [ ] **Step 3: Wire it**

In `agent_vwap_pullback.py`, add `from functools import partial` to the stdlib imports and `from trading_agent_framework.backtesting.data.chunked import YearChunkedData` after the `AlpacaBacktestData` import. In `run_backtesting`, replace:

```python
            data_source=AlpacaBacktestData,  # minute bars with enough history (Yahoo keeps ~30 days of minutes)
```

with:

```python
            # Alpaca: minute bars with enough history (Yahoo keeps ~30 days of minutes), one year at a time
            # (a 5Y window of minutes for 150 symbols in one fetch was OOM-killed at 58 GB)
            data_source=partial(YearChunkedData, inner=AlpacaBacktestData),
```

- [ ] **Step 4: Run the vwap tests to verify they pass**

Run: `uv run pytest tests/strategies/vwap_pullback -q`
Expected: all pass

- [ ] **Step 5: Document it in CLAUDE.md**

In the `backtesting/` architecture bullet, after `` `data/` (`BacktestDataSource` ABC, `CachedDataSource`, `YahooBacktestData` default, `AlpacaBacktestData`) ``, replace that parenthesis with:

```
`data/` (`BacktestDataSource` ABC, `CachedDataSource`, `YahooBacktestData` default, `AlpacaBacktestData`, `YearChunkedData` in `chunked.py`: minute bars one 365-day chunk at a time, each fetched from 30 days before its start, for windows over a year; day bars and sessions from one whole-window source)
```

In the "vwap_pullback never leaves a position without a stop" gotcha, append this sentence:

```
Its backtests read minute bars through `YearChunkedData` (the 5Y window fetched in one go was OOM-killed at 58 GB): the scanner's `_preload` still passes the whole window and the wrapper narrows a minute `load()` to the current year; a minute cutoff before the current chunk's fetch window raises `BacktestDataError`.
```

- [ ] **Step 6: Full suite and lint**

```bash
uv run pytest -q
uv run ruff check
```

Expected: all pass, no lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py tests/strategies/vwap_pullback/test_vwap_strategy.py CLAUDE.md
git commit -m "feat: vwap_pullback backtests read minute bars year by year

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Manual check (needs Alpaca data credentials and network; the human runs or approves it)**

Run the 5Y backtest in a memory-capped scope, so an overrun kills only the backtest, never the terminal:

```bash
systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 uv run agent vwap_pullback_continuation backtesting
```

Watch peak RSS with `ps -C python3 -o rss=,args=` until the first `stage 1 for 2021-09-20` line and a few sessions of ticks appear in the run's `backtest.log`. Expected: no OOM, RSS well under the cap (the YEAR-window run's order of magnitude). Then stop the run (or leave it to finish) and report the observed peak.
