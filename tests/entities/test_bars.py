from __future__ import annotations

import subprocess
import sys

import pandas as pd

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars


def _frame() -> pd.DataFrame:
    index = pd.date_range(
        "2026-09-01", periods=3, freq="1D", tz="America/New_York", name="timestamp"
    )
    close = [1.0, 2.0, 3.0]
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": [10.0] * 3},
        index=index,
    )


def test_bars_hold_asset_timestep_and_frame() -> None:
    df = _frame()
    bars = Bars(asset=Asset("AAPL"), timestep="day", df=df)

    assert bars.asset == Asset("AAPL")
    assert bars.timestep == "day"
    assert bars.df is df


def test_bars_compare_by_identity_not_by_frame() -> None:
    df = _frame()
    first = Bars(asset=Asset("AAPL"), timestep="day", df=df)
    second = Bars(asset=Asset("AAPL"), timestep="day", df=df.copy())

    # A dataclass-generated __eq__ would compare the DataFrames and raise
    # "The truth value of a DataFrame is ambiguous".
    assert first == first
    assert first != second


def test_importing_entities_does_not_import_pandas() -> None:
    """Subprocess: other tests in this session have already imported pandas."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import trading_agent_framework.entities\n"
            "import sys\n"
            "assert 'pandas' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
