from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_agent_framework.strategies.common.universe_history import UniverseHistory, market_day, month_universe, monthly_lists, window_symbols
from trading_agent_framework.utils.errors import BacktestError, FatalStrategyError, UniverseHistoryError

MARKET_TZ = ZoneInfo("America/New_York")


def _weekdays(first: date, last: date) -> list[date]:
    days = [first + timedelta(days=n) for n in range((last - first).days + 1)]
    return [d for d in days if d.weekday() < 5]


# The weekdays 2024-01-29 .. 2024-02-09. February's first session is 02-01, preceded by 01-29, 01-30, 01-31.
SESSIONS = _weekdays(date(2024, 1, 29), date(2024, 2, 9))
JAN_DAYS = [date(2024, 1, 29), date(2024, 1, 30), date(2024, 1, 31)]


def _frame(rows: dict[date, tuple[float, float]]) -> pd.DataFrame:
    """A parse_yahoo_frame-shaped frame: rows {session: (traded_close, traded_volume)}, stamped at 16:00 New York."""
    index = pd.DatetimeIndex([datetime.combine(d, time(16, 0), tzinfo=MARKET_TZ) for d in rows], name="timestamp")
    closes = [c for c, _ in rows.values()]
    volumes = [v for _, v in rows.values()]
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "traded_close": closes,
            "traded_volume": volumes,
        },
        index=index,
    )


def _constant(close: float, volume: float, days: list[date] = JAN_DAYS) -> pd.DataFrame:
    return _frame({d: (close, volume) for d in days})


def _four_symbols() -> dict[str, pd.DataFrame]:
    return {
        "A": _constant(20.0, 2e6),  # 20 * 2e6 = $40M a day: kept
        "B": _constant(5.0, 1e7),  # $50M a day but a $5 price, under the $10 floor
        "C": _constant(50.0, 1e5),  # 50 * 1e5 = $5M a day, under the $20M floor
        "D": _constant(100.0, 1e6),  # 100 * 1e6 = $100M a day: the largest
    }


def _rank(frames: dict[str, pd.DataFrame], top_n: int, months: list[str] | None = None) -> dict:
    return monthly_lists(
        frames,
        SESSIONS,
        months or ["2024-02"],
        min_price=10.0,
        min_dollar_volume=20e6,
        top_n=top_n,
        lookback=3,
    )


def test_a_month_ranks_by_traded_dollar_volume_before_its_first_session() -> None:
    # B fails the price floor ($5 < $10), C the dollar-volume floor ($5M < $20M); D ($100M) ranks above A ($40M).
    assert _rank(_four_symbols(), top_n=2) == {"2024-02": ("D", "A")}
    assert _rank(_four_symbols(), top_n=1) == {"2024-02": ("D",)}


def test_the_first_session_of_the_month_is_excluded() -> None:
    frames = _four_symbols()
    # E: ~$50 a day (50 * 1) over the three January sessions, then 50 * 1e9 = $50B on 02-01. Were the 02-01 row
    # counted, E's mean over its last three rows would be (50 + 50 + 5e10) / 3 ~ $16.7B and it would top the list.
    # Dated before 02-01 only, its mean is $50 < $20M, so it is dropped.
    frames["E"] = _frame({**{d: (50.0, 1.0) for d in JAN_DAYS}, date(2024, 2, 1): (50.0, 1e9)})

    assert "E" not in _rank(frames, top_n=5)["2024-02"]
    assert _rank(frames, top_n=5) == {"2024-02": ("D", "A")}


def test_a_month_without_enough_prior_sessions_is_an_error() -> None:
    # January's first session is 01-29, the first in SESSIONS: zero sessions before it, fewer than the lookback of 3.
    with pytest.raises(UniverseHistoryError):
        _rank(_four_symbols(), top_n=2, months=["2024-01"])


def test_a_month_with_no_session_is_an_error() -> None:
    with pytest.raises(UniverseHistoryError):
        _rank(_four_symbols(), top_n=2, months=["2024-03"])


def test_a_symbol_with_fewer_rows_than_the_lookback_is_not_eligible() -> None:
    frames = _four_symbols()
    # F: $1B a day (100 * 1e7) but only two rows (01-30, 01-31) before 02-01, fewer than the lookback of 3.
    frames["F"] = _constant(100.0, 1e7, days=[date(2024, 1, 30), date(2024, 1, 31)])

    assert _rank(frames, top_n=5) == {"2024-02": ("D", "A")}


def test_the_average_is_over_the_last_lookback_rows_only() -> None:
    frames = _four_symbols()
    # G: a $1T row on 01-26 (outside the last three rows), then $30M (30 * 1e6) a day on 01-29..01-31, so its mean
    # is $30M: below D ($100M) and A ($40M), above the $20M floor, so it ranks last. Were 01-26 counted, G's mean
    # over its last three rows would still be $30M, but over four rows it would be ~$250B and G would come first.
    frames["G"] = _frame({date(2024, 1, 26): (1000.0, 1e9), **{d: (30.0, 1e6) for d in JAN_DAYS}})

    assert _rank(frames, top_n=5) == {"2024-02": ("D", "A", "G")}


# For the staleness tests: the weekdays 2024-01-15 .. 2024-02-09. February's first session is 02-01, and the 13
# sessions before it are 01-15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31. Counting back from 01-31: 31 is the
# 1st-last, 30 the 2nd, 29 the 3rd, 26 the 4th, 25 the 5th, 24 the 6th, 23 the 7th, 22 the 8th, 19 the 9th, 18 the
# 10th and 17 the 11th. A symbol is current when its last bar is on or after the 10th-last prior session (01-18).
LONG_SESSIONS = _weekdays(date(2024, 1, 15), date(2024, 2, 9))


def _rank_long(frames: dict[str, pd.DataFrame]) -> tuple[str, ...]:
    lists = monthly_lists(frames, LONG_SESSIONS, ["2024-02"], min_price=10.0, min_dollar_volume=20e6, top_n=5, lookback=3)
    return lists["2024-02"]


def test_a_symbol_that_stopped_trading_long_before_the_month_is_not_listed() -> None:
    # H: $1B a day (100 * 1e7) on 01-02, 01-03, 01-04: three rows, so it meets the lookback, but its last bar is
    # eleven-plus sessions before 02-01 (the 10th-last prior session is 01-18). Only staleness excludes it.
    # R: $100M a day (100 * 1e6) on the last three sessions before 02-01: current, listed.
    frames = {
        "H": _constant(100.0, 1e7, days=[date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]),
        "R": _constant(100.0, 1e6),
    }

    assert _rank_long(frames) == ("R",)


def test_a_last_bar_on_the_tenth_last_prior_session_is_kept_and_the_eleventh_dropped() -> None:
    frames = {
        # Last bar 01-18, the 10th-last prior session: kept. $200M a day (200 * 1e6).
        "K": _constant(200.0, 1e6, days=[date(2024, 1, 16), date(2024, 1, 17), date(2024, 1, 18)]),
        # Last bar 01-17, the 11th-last prior session: dropped. $300M a day (300 * 1e6).
        "L": _constant(300.0, 1e6, days=[date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)]),
        # Last bar 01-31, the last prior session: kept. $100M a day (100 * 1e6).
        "R": _constant(100.0, 1e6),
    }

    # K ($200M) above R ($100M); L ($300M) would top the list were it not stale.
    assert _rank_long(frames) == ("K", "R")


def test_ties_are_broken_by_symbol() -> None:
    frames = {"Z": _constant(100.0, 1e6), "M": _constant(100.0, 1e6), "D": _constant(100.0, 1e6)}

    assert _rank(frames, top_n=3) == {"2024-02": ("D", "M", "Z")}


def test_each_month_is_ranked_on_its_own_prior_sessions() -> None:
    # March's first session is 03-01 (Friday). SESSIONS here extend through the end of February.
    sessions = _weekdays(date(2024, 1, 29), date(2024, 3, 4))
    feb_days = [d for d in sessions if date(2024, 2, 26) <= d <= date(2024, 2, 29)]
    # X: $100M a day in January, $0.5M (50 * 1e4) a day in late February: in February's list, not March's.
    # Y: the reverse: $0.5M until 01-31, then $100M a day from 02-26..02-29.
    x = _frame({**{d: (100.0, 1e6) for d in JAN_DAYS}, **{d: (50.0, 1e4) for d in feb_days}})
    y = _frame({**{d: (50.0, 1e4) for d in JAN_DAYS}, **{d: (100.0, 1e6) for d in feb_days}})

    lists = monthly_lists(
        {"X": x, "Y": y},
        sessions,
        ["2024-02", "2024-03"],
        min_price=10.0,
        min_dollar_volume=20e6,
        top_n=5,
        lookback=3,
    )

    # February (first session 02-01): last three rows before it are 01-29..31 -> X $100M, Y $0.5M.
    # March (first session 03-01): last three rows before it are 02-27, 02-28, 02-29 -> X $0.5M, Y $100M.
    assert lists == {"2024-02": ("X",), "2024-03": ("Y",)}


def _history(**overrides) -> UniverseHistory:
    fields = {
        "built": date(2024, 4, 2),
        "pool_date": date(2024, 3, 29),
        "top_n": 2,
        "min_price": 10.0,
        "min_dollar_volume": 20e6,
        "lookback": 63,
        "months": {"2024-02": ("D", "A"), "2024-03": ("D", "Q")},
    }
    return UniverseHistory(**{**fields, **overrides})


def test_on_picks_the_month_of_the_day_and_never_another() -> None:
    history = _history()

    assert history.on(date(2024, 2, 20)) == ["D", "A"]
    assert history.on(date(2024, 3, 1)) == ["D", "Q"]
    with pytest.raises(UniverseHistoryError):
        history.on(date(2024, 4, 1))


def test_covers_and_union() -> None:
    history = _history()

    assert history.covers(date(2024, 2, 5), date(2024, 3, 28)) is True
    assert history.covers(date(2024, 1, 31), date(2024, 2, 5)) is False  # January is absent
    assert history.covers(date(2024, 3, 28), date(2024, 4, 2)) is False  # April is absent
    assert history.union(date(2024, 2, 5), date(2024, 3, 28)) == ["A", "D", "Q"]
    assert history.union(date(2024, 2, 5), date(2024, 2, 9)) == ["A", "D"]


def test_union_of_a_range_with_an_absent_month_is_an_error() -> None:
    with pytest.raises(UniverseHistoryError):
        _history().union(date(2024, 1, 31), date(2024, 2, 5))


def test_json_round_trip(tmp_path: Path) -> None:
    history = _history()

    assert UniverseHistory.from_json(history.to_json()) == history
    path = tmp_path / "universe_history.json"
    history.save(path)
    assert UniverseHistory.load(path) == history
    assert UniverseHistory.load(tmp_path / "missing.json") is None


def test_the_error_is_a_backtest_error() -> None:
    assert issubclass(UniverseHistoryError, BacktestError)


def test_the_pool_counts_round_trip_and_default_for_an_old_file(tmp_path: Path) -> None:
    history = _history(pool_size=5400, unanswered=("U1", "U2"), bars_absent=("B1",))

    data = history.to_json()
    assert (data["pool_size"], data["unanswered"], data["bars_absent"]) == (5400, ["U1", "U2"], ["B1"])
    assert UniverseHistory.from_json(data) == history
    path = tmp_path / "universe_history.json"
    history.save(path)
    assert UniverseHistory.load(path) == history
    # A file written before the counts existed loads with their defaults.
    for key in ("pool_size", "unanswered", "bars_absent"):
        del data[key]
    old = UniverseHistory.from_json(data)
    assert (old.pool_size, old.unanswered, old.bars_absent) == (0, (), ())


def test_save_leaves_no_temp_file_and_never_a_partial_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "universe_history.json"
    _history().save(path)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["universe_history.json"]

    def explode(self: Path, target: object) -> None:
        raise OSError("crash between write and rename")

    monkeypatch.setattr(Path, "replace", explode)
    newer = _history(months={"2024-02": ("Z",)})
    with pytest.raises(OSError, match="crash"):
        newer.save(path)
    monkeypatch.undo()

    assert UniverseHistory.load(path) == _history()  # the earlier file is whole and untouched
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["universe_history.json"]  # no .tmp left behind


def _two_month_history() -> UniverseHistory:
    return UniverseHistory(
        built=date(2024, 3, 1), pool_date=date(2024, 3, 1), top_n=3, min_price=10.0, min_dollar_volume=1.0, lookback=63,
        months={"2024-01": ("AAA", "SHV"), "2024-02": ("BBB", "AAA", "SPY")},
    )


def test_market_day_aware_and_naive() -> None:
    late_utc = datetime(2024, 2, 1, 2, 0, tzinfo=ZoneInfo("UTC"))  # still Jan 31 in New York

    assert market_day(late_utc) == date(2024, 1, 31)
    assert market_day(datetime(2024, 2, 1, 2, 0)) == date(2024, 2, 1)  # a naive bound is taken as is


def test_window_symbols_excludes() -> None:
    history = _two_month_history()
    start, end = datetime(2024, 1, 15, tzinfo=MARKET_TZ), datetime(2024, 2, 20, tzinfo=MARKET_TZ)

    assert window_symbols(history, start, end, exclude={"SHV", "SPY"}) == ["AAA", "BBB"]
    assert window_symbols(history, start, end, exclude=()) == ["AAA", "BBB", "SHV", "SPY"]


def test_month_universe_excludes_and_keeps_the_order() -> None:
    assert month_universe(_two_month_history(), date(2024, 2, 9), exclude=["SPY"]) == ["BBB", "AAA"]


def test_month_universe_missing_month_is_fatal() -> None:
    with pytest.raises(FatalStrategyError, match="2024-03"):
        month_universe(_two_month_history(), date(2024, 3, 4), exclude=())
