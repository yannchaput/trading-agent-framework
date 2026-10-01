from __future__ import annotations

from datetime import date

import pandas as pd

from trading_agent_framework.agents.tools.vix import VixSeries


def _download(symbol, *, start, end, auto_adjust, progress) -> pd.DataFrame:
    assert symbol == "^VIX" and start < "2026-09-01" < end  # the window is padded before `start`
    days = pd.to_datetime(["2026-08-28", "2026-08-31", "2026-09-01"])
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": [15.0, 16.0, 17.0], "Volume": 0}, index=days)


def test_previous_close_is_strictly_before_the_day() -> None:
    vix = VixSeries(download=_download)
    vix.load(date(2026, 9, 1), date(2026, 9, 1))
    assert vix.previous_close(date(2026, 9, 1)) == 16.0  # never the 17.0 printed at that day's own close
    assert vix.previous_close(date(2026, 9, 2)) == 17.0
    assert vix.previous_close(date(2026, 8, 28)) is None


def test_an_empty_download_gives_no_value() -> None:
    vix = VixSeries(download=lambda *a, **k: pd.DataFrame())
    vix.load(date(2026, 9, 1), date(2026, 9, 1))
    assert vix.previous_close(date(2026, 9, 1)) is None
