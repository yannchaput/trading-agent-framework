"""End to end: a backtest over a few sessions with fake agents (they call the real submit tools) and a fake screen."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.screen import Candidate, ScreenResult

PRIOR = weekday_sessions(date(2026, 9, 9), 3)  # history before the window: the screen needs a last price on the first simulated day
SESSIONS = weekday_sessions(date(2026, 9, 14), 5)


def _bars(closes: list[float]) -> pd.DataFrame:
    assert len(closes) == len(PRIOR) + len(SESSIONS)
    return pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes, "volume": [1000.0] * len(closes)},
        index=pd.DatetimeIndex([session.close for session in [*PRIOR, *SESSIONS]], name="timestamp"),
    )


def _candidate(symbol: str) -> Candidate:
    return Candidate(
        symbol=symbol,
        rank=1,
        sic=5812,
        market_cap=Decimal("100000000000"),
        fcf_yield=0.05,
        fcf_margin=0.2,
        operating_margin=0.25,
        operating_margin_stdev=0.02,
        revenue_growth=0.08,
        net_debt_to_operating_income=1.0,
        debt_reported=True,
        fiscal_year_end=date(2025, 12, 31),
        filed=date(2026, 2, 15),
    )


class FakeScreen:
    """AAA is the one candidate, every day; a held AAA is described, any other holding has no data."""

    def __init__(self) -> None:
        self.as_ofs: list[Any] = []
        self.prices: list[Decimal | None] = []

    def run(self, symbols, *, as_of, price_of, top_n=None, label=None) -> ScreenResult:  # noqa: ANN001
        self.as_ofs.append(as_of)
        if top_n is None:
            self.prices.append(price_of("AAA"))  # the real screen asks for a price; in a backtest this goes through the no-look-ahead gate
            return ScreenResult(candidates=[_candidate("AAA")], rejections={})
        return ScreenResult(candidates=[_candidate(s) for s in symbols if s == "AAA"], rejections={s: "no_data" for s in symbols if s != "AAA"})


class _Handle:
    def __init__(self, tools: dict[str, Callable[..., dict[str, Any]]], script: Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]) -> None:
        self.tools, self.script, self.runs = tools, script, 0

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs += 1
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


def _research(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_ranking"]([{"symbol": sheet["symbol"], "reason": "simple and cash rich"} for sheet in context["candidates"]][:1])


def _judge(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_verdicts"]([{"symbol": entry["fact_sheet"]["symbol"], "verdict": "survive", "reason": "the attack failed"} for entry in context["to_judge"]])


def _trade(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_portfolio"]([{"symbol": entry["symbol"], "weight": 0.3, "reason": "best idea"} for entry in context["allowed"]][:1])


class _Manager:
    def __init__(self) -> None:
        self.handles: dict[str, _Handle] = {}
        self.scripts = {"researcher": _research, "short_seller": _judge, "trader": _trade}

    def create(self, *, name: str, system_prompt: str, tools: list[Callable[..., Any]], **_: Any) -> _Handle:
        self.handles[name] = _Handle({tool.__name__: tool for tool in tools}, self.scripts[name])
        return self.handles[name]

    def __getitem__(self, name: str) -> _Handle:
        return self.handles[name]

    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        return {}  # the runner reads per-agent totals at the end of a run


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _run(tmp_path: Path, manager: _Manager, screen: Any, settings: AckmanParams | None = None) -> BillAckmanStrategy:
    source = FakeBacktestDataSource()
    source.set_sessions(SESSIONS)
    for symbol, closes in {
        "AAA": [49.0, 49.5, 50.0, 50.5, 51.5, 51.0, 52.0, 52.0],
        "BBB": [24.0, 24.5, 25.0, 25.0, 25.5, 26.0, 26.5, 27.0],
        "SHV": [100.0] * 8,
        "SPY": [398.0, 399.0, 400.0, 401.0, 402.0, 403.0, 404.0, 405.0],
    }.items():
        source.set_bars(Asset(symbol), _bars(closes))
    strategy = BillAckmanStrategy(
        FakeBroker(FakeClock(et(2026, 9, 14, 9, 0)), strategy_name="bill_ackman"),
        mode=TradingMode.BACKTESTING,
        universe=["AAA", "BBB"],
        project_root=tmp_path,
        screen=screen,
        settings=settings,
    )
    strategy._agents = cast(AgentManager, manager)
    strategy.sleeptime = "1D"  # these tests count reviews session by session: independent of the strategy's production cadence
    Strategy.run_backtesting(strategy, start=SESSIONS[0].open - timedelta(hours=1), end=SESSIONS[-1].close, data_source=source, budget=Decimal(10000), benchmark="SPY")
    return strategy


def _review_lines(tmp_path: Path) -> list[dict[str, Any]]:
    (path,) = tmp_path.rglob("reviews.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_backtest_reviews_every_session_and_buys_the_trader_s_choice(tmp_path: Path) -> None:
    manager, screen = _Manager(), FakeScreen()

    _run(tmp_path, manager, screen)

    lines = _review_lines(tmp_path)
    assert len(lines) == len(SESSIONS)
    assert [line["date"] for line in lines] == [session.open.date().isoformat() for session in SESSIONS]
    assert not any(line["abandoned"] for line in lines)
    assert {order["symbol"] for order in lines[0]["orders"]} == {"AAA", "SHV"}  # 30% AAA, the rest but the 2% buffer parked
    # Nothing is re-sent on the second day (a backtest fills an order on the next bar, so day one's orders are still open).
    # The open-order accounting itself is pinned by the rebalancer's FakeBroker tests (test_ackman_rebalancer.py), where cash
    # cannot mask it; here BacktestBroker's buying power also nets the pending buys.
    assert lines[1]["orders"] == []
    assert all(manager.handles[name].runs == len(SESSIONS) for name in ("researcher", "short_seller", "trader"))
    assert list(tmp_path.rglob("metrics.json"))  # the run completed and wrote its report


def test_the_screen_is_asked_with_a_market_local_as_of_and_gets_no_look_ahead_prices(tmp_path: Path) -> None:
    screen = FakeScreen()

    _run(tmp_path, _Manager(), screen)

    assert all(as_of.tzinfo is not None and as_of.utcoffset() == timedelta(hours=-4) for as_of in screen.as_ofs)
    assert all(price is not None for price in screen.prices)
    # each session's price is the last close at or before that moment: the previous day's close, never the day's own or a later one
    assert screen.prices[:3] == [Decimal("50.0"), Decimal("50.5"), Decimal("51.5")]
    # from the second session on the screen is asked twice a day: the universe, then the holdings
    assert len(screen.as_ofs) > len(SESSIONS)


def test_a_dead_llm_aborts_a_real_backtest_instead_of_writing_a_flat_report(tmp_path: Path) -> None:
    from trading_agent_framework.utils.errors import AgentError, BacktestError

    manager = _Manager()

    def dead(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
        raise AgentError("llm down")

    manager.scripts["researcher"] = dead

    with pytest.raises(BacktestError, match="reviews abandoned in a row"):
        _run(tmp_path, manager, FakeScreen())

    assert manager.handles["researcher"].runs == 6  # three abandoned reviews, each with its one forced retry
    assert list(tmp_path.rglob("metrics.json")) == []
    assert [line["abandoned"] for line in _review_lines(tmp_path)] == [True, True, True]


class RotatingScreen:
    """AAA is the one candidate for the first three sessions, BBB afterwards; both are described when held."""

    def __init__(self) -> None:
        self.universe_calls = 0

    def run(self, symbols, *, as_of, price_of, top_n=None, label=None) -> ScreenResult:  # noqa: ANN001
        if top_n is None:
            self.universe_calls += 1
            current = "AAA" if self.universe_calls <= 3 else "BBB"
            price_of(current)
            return ScreenResult(candidates=[_candidate(current)], rejections={})
        return ScreenResult(candidates=[_candidate(s) for s in symbols], rejections={})


def test_a_rotation_sells_the_old_stock_and_buys_the_new_one_in_the_same_review(tmp_path: Path) -> None:
    manager = _Manager()

    def trade_ranked(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
        # the first allowed entry the researcher ranked: AAA drops out of the target once it stops being a candidate
        chosen = next(entry["symbol"] for entry in context["allowed"] if entry["research_rank"] is not None)
        tools["submit_portfolio"]([{"symbol": chosen, "weight": 0.9, "reason": "best idea"}])

    manager.scripts["trader"] = trade_ranked

    # One stock at 90% (the default cap is 35%): the SHV left over (8%) is far too small to pay for the switch,
    # so BBB can only be bought in full with the proceeds of the AAA sell submitted in the same review.
    _run(tmp_path, manager, RotatingScreen(), settings=AckmanParams(max_weight=0.9))

    lines = _review_lines(tmp_path)
    assert not any(line["abandoned"] for line in lines)
    rotation = next(line for line in lines if "BBB" in line["targets"])
    orders = {(order["symbol"], order["side"]): order["quantity"] for order in rotation["orders"]}
    assert ("AAA", "sell") in orders
    assert ("BBB", "buy") in orders
    assert orders[("BBB", "buy")] * 26.0 > 8000  # the 9,000 target at the day's 26.0 price, not what a little SHV could fund
