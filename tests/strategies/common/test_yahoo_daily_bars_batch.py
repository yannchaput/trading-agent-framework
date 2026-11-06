import re
from dataclasses import dataclass
from datetime import date

from trading_agent_framework.strategies.common.yahoo_daily_bars import UnusableBatch, fetch_usable_batch
from trading_agent_framework.utils.errors import YahooDataError

TODAY = date(2026, 11, 6)


class Source:
    """Answers each call with the next entry: a dict of bars (any value) or an exception to raise."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def bars(self, symbols, today):
        self.requests.append((list(symbols), today))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@dataclass
class _Outcome:
    bars: dict | None
    sleeps: list
    retries: list
    error: UnusableBatch | None


def _fetch(source, universe, *, extras=(), held=(), min_coverage=0.5, delays=(60.0,)):
    sleeps, retries = [], []
    kwargs = dict(
        extras=extras,
        today=lambda: TODAY,
        held=lambda: held,
        min_coverage=min_coverage,
        retry_delays=delays,
        sleep=sleeps.append,
        on_retry=lambda problem, delay: retries.append((problem, delay)),
    )
    try:
        bars = fetch_usable_batch(source, universe, **kwargs)
    except UnusableBatch as exc:
        return _Outcome(None, sleeps, retries, exc)
    return _Outcome(bars, sleeps, retries, None)


def test_usable_first_try():
    source = Source({"A": 1, "B": 2})
    out = _fetch(source, ["A", "B"])
    assert out.bars == {"A": 1, "B": 2}
    assert out.sleeps == [] and out.retries == []
    assert source.requests == [(["A", "B"], TODAY)]


def test_retries_then_succeeds():
    source = Source(YahooDataError("429"), {"A": 1, "B": 2})
    out = _fetch(source, ["A", "B"])
    assert out.bars == {"A": 1, "B": 2}
    assert out.retries == [("Yahoo lookup failed (429)", 60.0)]
    assert out.sleeps == [60.0]


def test_raises_after_last_delay():
    source = Source({"A": 1}, {"A": 1}, {"A": 1})
    out = _fetch(source, ["A", "B", "C", "D"], delays=(10.0, 20.0))
    assert re.search(r"Yahoo covers only 1/4 symbols", str(out.error))
    assert out.sleeps == [10.0, 20.0]
    assert [delay for _, delay in out.retries] == [10.0, 20.0]
    assert len(source.requests) == 3


def test_held_outside_universe_ignored():
    source = Source({"A": 1, "B": 2})
    assert _fetch(source, ["A", "B"], held=["GONE"]).bars == {"A": 1, "B": 2}


def test_held_in_universe_missing_is_unusable():
    source = Source({"A": 1, "B": 2, "C": 3, "D": 4})
    out = _fetch(source, ["A", "B", "C", "D", "E"], held=["E"], delays=())
    assert re.search(r"Yahoo has no bars for held E", str(out.error))


def test_extras_not_counted_in_coverage():
    source = Source({"A": 1, "SPY": 2, "QQQ": 3})
    out = _fetch(source, ["A", "B", "C", "D"], extras=["SPY", "QQQ"], delays=())
    assert re.search(r"Yahoo covers only 1/4 symbols", str(out.error))
    assert source.requests == [(["A", "B", "C", "D", "SPY", "QQQ"], TODAY)]
