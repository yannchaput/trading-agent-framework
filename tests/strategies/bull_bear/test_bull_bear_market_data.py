from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.bull_bear.market_data import DataUnavailable, GateBars, YahooBars
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BrokerError, YahooDataError

TODAY = date(2026, 10, 6)


def _bars(symbol: str, sessions: int = 302, last: date = TODAY) -> Bars:
    """Daily bars stamped at the 16:00 close, closes 100, 101, ...; the last one is dated `last` (today: still forming)."""
    dates = [d.date() for d in pd.bdate_range(end=last, periods=sessions)]
    closes = [100.0 + i for i in range(sessions)]
    index = pd.DatetimeIndex([datetime.combine(d, time(16), tzinfo=MARKET_TZ) for d in dates])
    frame = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1e6] * sessions}, index=index)
    return Bars(Asset(symbol), "day", frame)


class FakeSource:
    def __init__(self, answers: list[dict[str, Bars] | Exception]) -> None:
        self.answers = answers
        self.calls: list[tuple[list[str], date]] = []

    def bars(self, symbols, today):  # noqa: ANN001, ANN201
        self.calls.append((list(symbols), today))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _yahoo(source: FakeSource, sleeps: list[float], warnings: list[str]) -> YahooBars:
    return YahooBars(source, today=lambda: TODAY, sleep=sleeps.append, warn=warnings.append, retry_delays=(60.0, 180.0), min_coverage=0.5)


def test_a_usable_batch_gives_300_completed_sessions_per_universe_symbol() -> None:
    source = FakeSource([{"AAA": _bars("AAA"), "BBB": _bars("BBB"), "XXX": _bars("XXX")}])

    series = _yahoo(source, [], []).load(["AAA", "BBB"], held=[])

    assert set(series) == {"AAA", "BBB"}  # a symbol outside the universe is ignored
    assert len(series["AAA"].closes) == 300
    assert series["AAA"].closes[-1] == 400.0  # today's 401 is dropped
    assert series["AAA"].closes[0] == 101.0
    assert series["AAA"].volumes == [1e6] * 300
    assert source.calls == [(["AAA", "BBB"], TODAY)]


def test_a_batch_covering_too_little_of_the_universe_is_retried_then_unavailable() -> None:
    sleeps: list[float] = []
    warnings: list[str] = []
    one = {"AAA": _bars("AAA")}
    source = FakeSource([one, one, one])

    with pytest.raises(DataUnavailable, match="Yahoo covers only 1/4 symbols"):
        _yahoo(source, sleeps, warnings).load(["AAA", "BBB", "CCC", "DDD"], held=[])

    assert sleeps == [60.0, 180.0]
    assert len(warnings) == 2 and "retry in 60s" in warnings[0]


def test_a_failed_download_is_retried_and_a_later_success_is_used() -> None:
    sleeps: list[float] = []
    source = FakeSource([YahooDataError("429"), {"AAA": _bars("AAA")}])

    series = _yahoo(source, sleeps, []).load(["AAA"], held=[])

    assert set(series) == {"AAA"} and sleeps == [60.0]


def test_a_batch_missing_a_held_universe_stock_is_unusable() -> None:
    half = {"AAA": _bars("AAA")}
    source = FakeSource([half, half, half])

    with pytest.raises(DataUnavailable, match="no bars for held BBB"):
        _yahoo(source, [], []).load(["AAA", "BBB"], held=["BBB", "OUTSIDE"])  # OUTSIDE is not in the universe: not required


def test_the_gate_reads_each_symbol_skips_failures_and_drops_todays_bar() -> None:
    warnings: list[str] = []
    answers = {"AAA": _bars("AAA"), "NONE": None}
    requested: list[tuple[str, int]] = []

    def fetch(symbol: str, length: int) -> Bars | None:
        requested.append((symbol, length))
        if symbol == "BAD":
            raise BrokerError("invalid symbol")
        return answers[symbol]

    series = GateBars(fetch, today=lambda: TODAY, warn=warnings.append).load(["AAA", "BAD", "NONE"], held=[])

    assert set(series) == {"AAA"} and series["AAA"].closes[-1] == 400.0
    assert requested == [("AAA", 301), ("BAD", 301), ("NONE", 301)]
    assert any("BAD" in warning for warning in warnings)


def _with_traded_columns(bars: Bars) -> Bars:
    frame = bars.df.copy()
    frame["traded_close"] = [1000.0 + i for i in range(len(frame))]
    frame["traded_volume"] = [3e6 + i for i in range(len(frame))]
    return Bars(bars.asset, "day", frame)


def test_the_series_carries_the_traded_columns_cut_like_the_closes() -> None:
    bars = _with_traded_columns(_bars("AAA", sessions=306))  # today's bar + 305 completed sessions
    source = FakeSource([{"AAA": bars}])

    (series,) = _yahoo(source, [], []).load(["AAA"], held=[]).values()

    assert series.traded_closes is not None and series.traded_volumes is not None
    assert len(series.traded_closes) == len(series.traded_volumes) == len(series.closes) == 300
    assert series.closes[-1] == 404.0  # the last completed session: today's 405 is dropped
    assert series.traded_closes[-1] == 1304.0 and series.traded_volumes[-1] == 3e6 + 304
    assert series.traded_closes[0] == 1005.0  # the same first row as closes[0] == 105.0


def test_a_frame_without_traded_columns_gives_none() -> None:
    source = FakeSource([{"AAA": _bars("AAA")}])

    (series,) = _yahoo(source, [], []).load(["AAA"], held=[]).values()

    assert series.traded_closes is None and series.traded_volumes is None
