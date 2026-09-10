from __future__ import annotations

from dataclasses import dataclass

from trading_agent_framework.entities.enums import AssetType


@dataclass(frozen=True, slots=True)
class Asset:
    """A tradable instrument. Frozen because it is used as a dict key / set member."""

    symbol: str
    asset_type: AssetType = AssetType.STOCK

    def __str__(self) -> str:
        return self.symbol
