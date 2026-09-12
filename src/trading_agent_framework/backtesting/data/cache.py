"""Read-through parquet cache wrapping any `BacktestDataSource`. First `load()` fetches
and writes; a later `load()` for the same window is network-free -- the case that
matters most when iterating on an agent prompt against a fixed backtest period.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from trading_agent_framework.backtesting.data.base import FULL_HISTORY, BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError


class CachedDataSource(BacktestDataSource):
    """Wraps `inner`, caching each asset's fetched window at
    `<cache_dir>/<inner.name>/<symbol>_<timestep>_<start>_<end>.parquet` plus a
    `.meta.json` sidecar recording provenance (provider, symbol, fetch time, row count)
    -- `inner` may revise its data over time (e.g. Yahoo's retroactive dividend
    adjustments), so a cached file is a frozen, dated snapshot on purpose.
    """

    def __init__(self, inner: BacktestDataSource, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir) / inner.name
        self.name = inner.name

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        for asset in assets:
            self._ensure_cached(asset, start, end, timestep)

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        # bars() alone carries no [start, end] window to cache against -- only load()
        # does -- so a cache miss here falls straight through to the inner source.
        path = self._latest_cache_path(asset, timestep)
        if path is None:
            return self._inner.bars(asset, cutoff, length, timestep)
        return self._read(path, asset, timestep, cutoff, length)

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        return self._inner.sessions(start, end)

    def _cache_path(self, asset: Asset, start: datetime, end: datetime, timestep: str) -> Path:
        stamp = f"{asset.symbol}_{timestep}_{start.date()}_{end.date()}"
        return self._cache_dir / f"{stamp}.parquet"

    def _latest_cache_path(self, asset: Asset, timestep: str) -> Path | None:
        if not self._cache_dir.is_dir():
            return None
        matches = sorted(self._cache_dir.glob(f"{asset.symbol}_{timestep}_*.parquet"))
        return matches[-1] if matches else None

    def _ensure_cached(self, asset: Asset, start: datetime, end: datetime, timestep: str) -> None:
        path = self._cache_path(asset, start, end, timestep)
        if path.exists():
            return
        self._inner.load([asset], start, end, timestep)
        bars = self._inner.bars(asset, end, FULL_HISTORY, timestep)
        if bars is None:
            return
        self._write(path, bars, asset)

    def _write(self, path: Path, bars: Bars, asset: Asset) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            bars.df.to_parquet(path)
            meta_path = path.with_suffix(".meta.json")
            meta_path.write_text(
                json.dumps(
                    {
                        "provider": self.name, "symbol": asset.symbol,
                        "fetched_at": datetime.now().isoformat(), "rows": len(bars.df),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            raise BacktestDataError(f"Failed to write cache file {path}: {exc}") from exc

    def _read(
        self, path: Path, asset: Asset, timestep: str, cutoff: datetime, length: int
    ) -> Bars | None:
        import pandas as pd

        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise BacktestDataError(f"Failed to read cache file {path}: {exc}") from exc
        visible = df[df.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))
