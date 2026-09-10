from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import AssetType


def test_asset_is_hashable_and_usable_in_a_set() -> None:
    a = Asset(symbol="AAPL")
    b = Asset(symbol="AAPL")
    c = Asset(symbol="MSFT")
    assert {a, b, c} == {a, c}
    assert hash(a) == hash(b)


def test_asset_rejects_mutation() -> None:
    a = Asset(symbol="AAPL")
    with pytest.raises(FrozenInstanceError):
        a.symbol = "MSFT"


def test_asset_default_type_and_str() -> None:
    a = Asset(symbol="AAPL")
    assert a.asset_type == AssetType.STOCK
    assert str(a) == "AAPL"
