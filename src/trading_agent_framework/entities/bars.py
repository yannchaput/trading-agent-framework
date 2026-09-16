from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trading_agent_framework.entities.asset import Asset

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True, eq=False)
class Bars:
    """OHLCV history for one asset, oldest first, indexed in market time (America/New_York).

    The columns are float64 on purpose, as the codebase's second float boundary: bars feed
    indicator maths, never order sizing. `eq=False` because a generated `__eq__` would compare
    the DataFrames, which raises.
    """

    asset: Asset
    timestep: str
    df: pd.DataFrame

    @property
    def empty(self) -> bool:
        """Lumibot-compatible alias for `df.empty`."""
        return self.df.empty

    @property
    def pandas_df(self) -> pd.DataFrame:
        """Lumibot-compatible alias for `df` (this entity is pandas-only, no polars)."""
        return self.df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, key):
        return self.df[key]
