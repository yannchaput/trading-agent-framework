# tests/strategies/earnings_drift/test_drift_scanner.py
from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.fakes import FakeClock, FakeNewsProvider, FrameDataSource, et, weekday_sessions

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.strategies.earnings_drift.event_source import LoadReport
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.scanner import Scanner
from trading_agent_framework.utils.errors import BrokerError

SESSIONS = weekday_sessions(date(2026, 8, 3), 31)
DATES = [s.open.date() for s in SESSIONS]
TODAY = DATES[-1]
FLAT = (100.0, 100.5, 99.5, 100.0, 1_000_000.0)


def _frame(rows: Sequence[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    sessions = SESSIONS[-len(rows) :]
    return pd.DataFrame(list(rows), columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([s.close for s in sessions], name="timestamp"))


FRAMES = {
    ("AAA", "day"): _frame([FLAT] * 30 + [(106.0, 110.0, 105.0, 109.0, 5_000_000.0)]),
    ("BBB", "day"): _frame([FLAT] * 30 + [(106.0, 110.0, 105.0, 109.0, 5_000_000.0)]),
    ("CCC", "day"): _frame([FLAT] * 30 + [(100.0, 101.0, 99.0, 100.5, 1_500_000.0)]),
    ("SPY", "day"): _frame([(400.0, 401.0, 399.0, 400.0, 5e7)] * 30 + [(400.0, 402.0, 399.0, 400.8, 5e7)]),
}


class StaticEvents:
    """An `EventProvider` over fixed events; counts loads; `report` is what `load` answers."""

    def __init__(self, events: list[EarningsEvent], report: LoadReport | None = None, *, reload_every_cycle: bool = False) -> None:
        self.events, self.report, self.loads, self.reload = events, report or LoadReport(loaded=3), 0, reload_every_cycle
        self.loaded = False
        self.as_ofs: list[datetime] = []
        self.sinces: list[date] = []

    def needs_load(self) -> bool:
        return not self.loaded or self.reload

    def load(self, symbols: Sequence[str], *, as_of: datetime, since: date) -> LoadReport:
        self.loads += 1
        self.as_ofs.append(as_of)
        self.sinces.append(since)
        self.loaded = True
        return self.report

    def discard(self) -> None:
        self.loaded = False

    def all_events(self) -> list[EarningsEvent]:
        return list(self.events) if self.loaded else []


def _event(symbol: str, accepted_at: datetime) -> EarningsEvent:
    return EarningsEvent(symbol, accepted_at, f"acc-{symbol}", f"{symbol}.htm")


def _headline(symbol: str, text: str, when: datetime) -> dict[str, object]:
    return {"headline": text, "created_at": when.isoformat(), "symbols": [symbol]}


RELEASE = et(TODAY.year, TODAY.month, TODAY.day, 7, 0)
NEWS = {
    "AAA": [_headline("AAA", "AAA Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate", RELEASE + timedelta(minutes=1))],
    "CCC": [_headline("CCC", "CCC Q3 EPS $0.52 Beats $0.50 Estimate", RELEASE + timedelta(minutes=1))],
}
EVENTS = [
    _event("AAA", RELEASE),
    _event("BBB", RELEASE),  # no headline: no_surprise_data
    _event("CCC", RELEASE),  # weak reaction
    _event("AAA", RELEASE + timedelta(days=1)),  # reacts tomorrow: ignored today
]


def _scanner(
    tmp_path: Path,
    source: StaticEvents,
    *,
    news: object | None = None,
    frames: dict | None = None,
    now: datetime | None = None,
    params: DriftParams | None = None,
    **scanner_options: Any,
) -> Scanner:
    clock = FakeClock(now or SESSIONS[-1].close, SESSIONS)
    broker = BacktestBroker(
        "earnings_drift",
        data_source=FrameDataSource(frames or FRAMES, SESSIONS),
        clock=clock,
        budget=D("100000"),
        timestep="day",
        news_source=news if news is not None else FakeNewsProvider(NEWS),  # type: ignore[arg-type]
    )
    strategy = Strategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
    return Scanner(strategy, params or DriftParams(), ["AAA", "BBB", "CCC"], source, **scanner_options)


def test_prepare_finds_today_s_candidates_and_rejections(tmp_path: Path) -> None:
    result = _scanner(tmp_path, StaticEvents(EVENTS)).prepare(held=set())
    assert result.today == TODAY and result.trading_dates == DATES and not result.hollow
    assert [c.symbol for c in result.candidates] == ["AAA"]
    candidate = result.candidates[0]
    assert candidate.reaction_day == TODAY
    assert candidate.reaction.abnormal_pct == pytest.approx(0.088)
    assert candidate.surprise.surprise.eps_actual == D("1.52")
    assert candidate.headlines == ((f"{TODAY.isoformat()} 07:01", NEWS["AAA"][0]["headline"]),)
    assert result.rejections == {"BBB": "no_surprise_data", "CCC": "weak_reaction"}


def test_a_held_symbol_is_rejected(tmp_path: Path) -> None:
    result = _scanner(tmp_path, StaticEvents(EVENTS)).prepare(held={"AAA"})
    assert result.candidates == [] and result.rejections["AAA"] == "already_held"


def test_events_load_once_in_a_backtest(tmp_path: Path) -> None:
    source = StaticEvents(EVENTS)
    scanner = _scanner(tmp_path, source)
    scanner.prepare(held=set())
    scanner.prepare(held=set())
    assert source.loads == 1


def test_a_hollow_load_gives_no_candidates_and_is_retried(tmp_path: Path) -> None:
    source = StaticEvents(EVENTS, LoadReport(loaded=5, failed={f"S{i}": "x" for i in range(25)}))
    scanner = _scanner(tmp_path, source)
    result = scanner.prepare(held=set())
    assert result.hollow and result.candidates == []
    assert source.needs_load()  # discarded: the next cycle loads again
    scanner.prepare(held=set())
    assert source.loads == 2


class _BrokenNews:
    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):  # noqa: ANN001, ANN201
        raise BrokerError("news down")


def test_a_news_failure_rejects_the_events_as_no_news(tmp_path: Path) -> None:
    result = _scanner(tmp_path, StaticEvents(EVENTS), news=_BrokenNews()).prepare(held=set())
    assert result.candidates == []
    assert result.rejections == {"AAA": "no_news", "BBB": "no_news", "CCC": "no_news"}


class _NewestFirstNews:
    """Alpaca's shape: the newest `limit` articles tagged with any of `symbols` within `[start, end]`, newest first."""

    def __init__(self, articles: list[dict[str, object]], *, failing: Collection[str] = ()) -> None:
        self.articles, self.failing = articles, set(failing)
        self.calls: list[tuple[tuple[str, ...], datetime | None, datetime, int]] = []

    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):  # noqa: ANN001, ANN201
        self.calls.append((tuple(symbols), start, end, limit))
        if self.failing & set(symbols):
            raise BrokerError("news down for " + ", ".join(sorted(self.failing & set(symbols))))
        wanted = set(symbols)
        rows = []
        for article in self.articles:
            created = datetime.fromisoformat(str(article["created_at"]))
            if wanted & set(article["symbols"]) and (start is None or created >= start) and created <= end:  # type: ignore[call-overload]
                rows.append((created, article))
        return [article for _, article in sorted(rows, key=lambda pair: pair[0], reverse=True)[:limit]]


NOW = SESSIONS[-1].close
FRIDAY_AFTER_CLOSE = et(DATES[-2].year, DATES[-2].month, DATES[-2].day, 16, 30)  # TODAY is a Monday: this one reacts today too


def test_newer_articles_of_other_symbols_do_not_push_the_earnings_headline_out(tmp_path: Path) -> None:
    busy = [_headline("BBB", f"BBB analyst note {i}", RELEASE + timedelta(minutes=10 + i)) for i in range(60)]
    news = _NewestFirstNews([*NEWS["AAA"], *NEWS["CCC"], *busy])
    result = _scanner(tmp_path, StaticEvents(EVENTS), news=news).prepare(held=set())
    assert [c.symbol for c in result.candidates] == ["AAA"]  # a query shared with BBB (limit 50, newest first) lost it


def test_news_is_asked_once_per_event_symbol_around_its_release(tmp_path: Path) -> None:
    events = [_event("AAA", RELEASE), _event("BBB", FRIDAY_AFTER_CLOSE), _event("CCC", RELEASE)]
    news = _NewestFirstNews([])
    _scanner(tmp_path, StaticEvents(events), news=news).prepare(held=set())
    assert news.calls == [
        (("AAA",), RELEASE - timedelta(hours=2), NOW, 50),  # the 24-hour window would end after now: cut at now
        (("BBB",), FRIDAY_AFTER_CLOSE - timedelta(hours=2), FRIDAY_AFTER_CLOSE + timedelta(hours=24), 50),
        (("CCC",), RELEASE - timedelta(hours=2), NOW, 50),
    ]
    assert all(end <= NOW for _, _, end, _ in news.calls)


def test_the_surprise_window_is_a_parameter(tmp_path: Path) -> None:
    news = _NewestFirstNews([])
    scanner = _scanner(tmp_path, StaticEvents([_event("BBB", FRIDAY_AFTER_CLOSE)]), news=news, params=DriftParams(surprise_window_hours=6.0, surprise_lookback_hours=1.0))
    scanner.prepare(held=set())
    assert news.calls == [(("BBB",), FRIDAY_AFTER_CLOSE - timedelta(hours=1), FRIDAY_AFTER_CLOSE + timedelta(hours=6), 50)]


def test_a_news_failure_for_one_symbol_rejects_only_that_symbol(tmp_path: Path) -> None:
    news = _NewestFirstNews([*NEWS["AAA"], *NEWS["CCC"]], failing={"BBB"})
    result = _scanner(tmp_path, StaticEvents(EVENTS), news=news).prepare(held=set())
    assert [c.symbol for c in result.candidates] == ["AAA"]
    assert result.rejections == {"BBB": "no_news", "CCC": "weak_reaction"}


def test_an_article_after_now_is_never_read(tmp_path: Path) -> None:
    late = _headline("AAA", "AAA Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate", NOW + timedelta(minutes=5))
    result = _scanner(tmp_path, StaticEvents([_event("AAA", RELEASE)]), news=_NewestFirstNews([late])).prepare(held=set())
    assert result.candidates == [] and result.rejections == {"AAA": "no_surprise_data"}


def test_no_benchmark_bar_today_gives_no_candidates_and_counts_today(tmp_path: Path) -> None:
    frames = dict(FRAMES)
    frames[("SPY", "day")] = FRAMES[("SPY", "day")].iloc[:-1]
    result = _scanner(tmp_path, StaticEvents(EVENTS), frames=frames).prepare(held=set())
    assert result.candidates == [] and result.trading_dates[-1] == TODAY


# --- A4: the SEC freshness as_of can be the backtest's end ------------------------------------------


def test_events_load_as_of_now_by_default(tmp_path: Path) -> None:
    source = StaticEvents(EVENTS)
    _scanner(tmp_path, source).prepare(held=set())
    assert source.as_ofs == [NOW] and source.sinces == [TODAY - timedelta(days=10)]


def test_events_load_as_of_the_injected_time_but_since_still_counts_from_today(tmp_path: Path) -> None:
    run_end = et(2026, 12, 31, 0, 0)
    source = StaticEvents(EVENTS)
    result = _scanner(tmp_path, source, load_as_of=lambda: run_end).prepare(held=set())
    assert source.as_ofs == [run_end] and source.sinces == [TODAY - timedelta(days=10)]
    assert [c.symbol for c in result.candidates] == ["AAA"]  # every other clock use is still now


def test_a_load_as_of_that_answers_none_falls_back_to_now(tmp_path: Path) -> None:
    source = StaticEvents(EVENTS)
    _scanner(tmp_path, source, load_as_of=lambda: None).prepare(held=set())
    assert source.as_ofs == [NOW]


# --- A5: live dollar volume is IEX's share of the consolidated volume -------------------------------

THIN = 50_000.0  # x 100 = a 5M 20-day dollar volume: under the 20M gate, above 3% of it (0.6M)


def _thin_frames() -> dict:
    frames = dict(FRAMES)
    frames[("AAA", "day")] = _frame([(100.0, 100.5, 99.5, 100.0, THIN)] * 30 + [(106.0, 110.0, 105.0, 109.0, 5 * THIN)])
    return frames


def test_a_thin_name_is_illiquid_on_consolidated_volume(tmp_path: Path) -> None:
    result = _scanner(tmp_path, StaticEvents(EVENTS), frames=_thin_frames(), volume_share=1.0).prepare(held=set())
    assert result.candidates == [] and result.rejections["AAA"] == "illiquid"


def test_the_dollar_volume_gate_scales_with_the_feed_s_volume_share(tmp_path: Path) -> None:
    result = _scanner(tmp_path, StaticEvents(EVENTS), frames=_thin_frames(), volume_share=0.03).prepare(held=set())
    assert [c.symbol for c in result.candidates] == ["AAA"]
    assert result.candidates[0].reaction.dollar_volume_20d == pytest.approx(5_000_000.0)  # the features themselves are not scaled
