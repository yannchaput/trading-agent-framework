from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.entities.asset import Asset


def test_backtest_data_source_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BacktestDataSource()  # ty: ignore[abstract-class-instantiated]


def test_fake_data_source_honours_the_cutoff_and_length_contract() -> None:
    source = FakeBacktestDataSource()
    asset = Asset("AAPL")
    df = make_close_indexed_frame([100.0, 101.0, 102.0], start=datetime(2026, 1, 5, 16, tzinfo=UTC))
    source.set_bars(asset, df)

    # Cutoff before the first bar closes: nothing visible.
    assert source.bars(asset, df.index[0] - pd.Timedelta(seconds=1), 5, "day") is None

    # Cutoff exactly at the first bar's close: exactly that one bar.
    result = source.bars(asset, df.index[0], 5, "day")
    assert result is not None
    assert len(result.df) == 1
    assert result.df["close"].iloc[-1] == 100.0

    # Cutoff after all three: all three, oldest first, length respected.
    result = source.bars(asset, df.index[-1], 2, "day")
    assert result is not None
    assert list(result.df["close"]) == [101.0, 102.0]
