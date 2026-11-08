from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from trading_agent_framework.backtesting import report
from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, Ledger
from trading_agent_framework.dashboard.components.charts import allocation_chart
from trading_agent_framework.dashboard.models import RunRef
from trading_agent_framework.dashboard.reader import SLEEVE_SYMBOLS, load_allocation, yahoo_sleeve_closes
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import OrderSide, OrderType

D5, D6 = date(2026, 1, 5), date(2026, 1, 6)


def _run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "strat" / "backtesting" / "2026-01-07_100000_backtesting"
    d.mkdir(parents=True)
    return d


def _equity(run_dir: Path, rows: list[tuple[int, str, str, str]]) -> None:
    """`(day, portfolio_value, cash, positions_value)` per session close (16:00 New York = 21:00 UTC)."""
    report.write_equity(
        run_dir,
        [EquitySample(time=datetime(2026, 1, day, 21, tzinfo=UTC), portfolio_value=Decimal(pv), cash=Decimal(cash), positions_value=Decimal(pos)) for day, pv, cash, pos in rows],
    )


def _fills(run_dir: Path, fills: list[tuple[int, str, OrderSide, int, str]]) -> None:
    """`(day, symbol, side, qty, price)`, all filled at 11:00 New York."""
    ledger = Ledger()
    for day, symbol, side, qty, price in fills:
        ledger.record_fill(
            FillRecord(
                time=datetime(2026, 1, day, 16, tzinfo=UTC),
                identifier=f"{symbol}{day}{side.value}",
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=Decimal(qty),
                filled_quantity=Decimal(qty),
                price=Decimal(price),
                trade_cost=Decimal(0),
                trade_slippage=Decimal(0),
            )
        )
    report.write_trades(run_dir, ledger)


class _Closes:
    """A recording stand-in for the Yahoo fetch: `calls` lists what `load_allocation` asked for."""

    def __init__(self, prices: dict[str, dict[date, float]]) -> None:
        self.prices = prices
        self.calls: list[tuple[tuple[str, ...], date, date]] = []

    def __call__(self, symbols: Sequence[str], first: date, last: date) -> dict[str, pd.Series]:
        self.calls.append((tuple(symbols), first, last))
        return {s: pd.Series(self.prices[s]) for s in symbols if s in self.prices}


def _ref(run_dir: Path) -> RunRef:
    return RunRef.from_path(str(run_dir))


def test_the_sleeve_is_valued_at_market_and_stocks_are_the_rest_of_the_positions(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _equity(run_dir, [(5, "10000", "4000", "6000"), (6, "10500", "4000", "6500")])
    _fills(run_dir, [(5, "SHV", OrderSide.BUY, 50, "100"), (5, "AAA", OrderSide.BUY, 10, "100")])

    rows = load_allocation(_ref(run_dir), _Closes({"SHV": {D5: 100.0, D6: 110.0}}))

    assert rows is not None
    assert rows[0] == {"date": "2026-01-05", "equity": 10000.0, "stocks_pct": 10.0, "sleeve_pct": 50.0, "cash_pct": 40.0}
    assert rows[1]["sleeve_pct"] == pytest.approx(100 * 5500 / 10500)  # 50 shares at 110
    assert rows[1]["stocks_pct"] == pytest.approx(100 * 1000 / 10500)


def test_every_row_sums_to_one_hundred_percent(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _equity(run_dir, [(5, "10000", "4000", "6000"), (6, "10500", "4000", "6500")])
    _fills(run_dir, [(5, "SHV", OrderSide.BUY, 50, "100"), (5, "GLD", OrderSide.BUY, 5, "100")])

    rows = cast(list[dict[str, Any]], load_allocation(_ref(run_dir), _Closes({"SHV": {D5: 100.0, D6: 111.0}, "GLD": {D5: 100.0, D6: 90.0}})))

    assert all(r["stocks_pct"] + r["sleeve_pct"] + r["cash_pct"] == pytest.approx(100.0) for r in rows)


def test_the_sleeve_never_exceeds_the_positions_the_ledger_holds(tmp_path: Path) -> None:
    # Yahoo's adjusted closes drift from the backtest's: priced too high, the sleeve must stop at the ledger's total.
    run_dir = _run_dir(tmp_path)
    _equity(run_dir, [(5, "10000", "5000", "5000")])
    _fills(run_dir, [(5, "SHV", OrderSide.BUY, 50, "100")])

    rows = cast(list[dict[str, Any]], load_allocation(_ref(run_dir), _Closes({"SHV": {D5: 130.0}})))

    assert (rows[0]["stocks_pct"], rows[0]["sleeve_pct"], rows[0]["cash_pct"]) == (0.0, 50.0, 50.0)


def test_shares_follow_the_fills_through_a_sale(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _equity(run_dir, [(5, "10000", "5000", "5000"), (6, "10000", "8000", "2000")])
    _fills(run_dir, [(5, "IEF", OrderSide.BUY, 50, "100"), (6, "IEF", OrderSide.SELL, 30, "100")])

    rows = cast(list[dict[str, Any]], load_allocation(_ref(run_dir), _Closes({"IEF": {D5: 100.0, D6: 100.0}})))

    assert [r["sleeve_pct"] for r in rows] == [50.0, 20.0]


def test_a_session_without_a_close_uses_the_previous_one(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _equity(run_dir, [(5, "10000", "5000", "5000"), (6, "10000", "5000", "5000")])
    _fills(run_dir, [(5, "SHV", OrderSide.BUY, 50, "100")])

    rows = cast(list[dict[str, Any]], load_allocation(_ref(run_dir), _Closes({"SHV": {D5: 100.0}})))

    assert rows[1]["sleeve_pct"] == 50.0


def test_without_any_sleeve_trade_nothing_is_fetched(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _equity(run_dir, [(5, "10000", "9000", "1000")])
    _fills(run_dir, [(5, "AAA", OrderSide.BUY, 10, "100")])
    fetch = _Closes({})

    rows = cast(list[dict[str, Any]], load_allocation(_ref(run_dir), fetch))

    assert fetch.calls == [] and (rows[0]["stocks_pct"], rows[0]["sleeve_pct"], rows[0]["cash_pct"]) == (10.0, 0.0, 90.0)


def test_only_the_sleeve_symbols_that_traded_are_fetched_over_the_runs_sessions(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _equity(run_dir, [(5, "10000", "5000", "5000"), (6, "10000", "5000", "5000")])
    _fills(run_dir, [(5, "SHV", OrderSide.BUY, 50, "100"), (5, "AAA", OrderSide.BUY, 1, "100")])
    fetch = _Closes({"SHV": {D5: 100.0, D6: 100.0}})

    load_allocation(_ref(run_dir), fetch)

    assert fetch.calls == [(("SHV",), D5, D6)]


def test_a_run_without_equity_has_no_allocation(tmp_path: Path) -> None:
    assert load_allocation(_ref(_run_dir(tmp_path)), _Closes({})) is None


def test_a_run_without_fills_is_all_cash_or_stocks_as_the_ledger_says(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _equity(run_dir, [(5, "10000", "10000", "0")])

    rows = cast(list[dict[str, Any]], load_allocation(_ref(run_dir), _Closes({})))

    assert (rows[0]["stocks_pct"], rows[0]["sleeve_pct"], rows[0]["cash_pct"]) == (0.0, 0.0, 100.0)


def test_the_sleeve_is_shv_ief_and_gld() -> None:
    assert set(SLEEVE_SYMBOLS) == {"SHV", "IEF", "GLD"}


class _FakeSource:
    def bars(self, symbols: Any, today: date) -> dict[str, Bars]:
        index = pd.DatetimeIndex(["2026-01-05 16:00", "2026-01-06 16:00"], tz="America/New_York")
        frame = pd.DataFrame({"close": [100.0, 101.0], "traded_close": [99.0, 99.5]}, index=index)
        return {s: Bars(asset=Asset(s), timestep="day", df=frame) for s in symbols if s != "MISSING"}


def test_yahoo_sleeve_closes_are_the_backtests_adjusted_closes_by_session_date() -> None:
    closes = yahoo_sleeve_closes(("SHV", "MISSING"), D5, D6, source=_FakeSource())

    assert list(closes) == ["SHV"]
    assert closes["SHV"].to_dict() == {D5: 100.0, D6: 101.0}


ROWS = [
    {"date": "2026-01-05", "equity": 10000.0, "stocks_pct": 10.0, "sleeve_pct": 50.0, "cash_pct": 40.0},
    {"date": "2026-01-06", "equity": 10500.0, "stocks_pct": 9.5, "sleeve_pct": 52.5, "cash_pct": 38.0},
]


def test_the_chart_stacks_stocks_the_sleeve_and_cash_up_to_one_hundred_percent() -> None:
    fig = allocation_chart(ROWS)

    traces = cast(Any, fig.data)
    assert [t.name for t in traces] == ["Stocks", "SHV / IEF / GLD", "Cash"]
    assert [list(t.y) for t in traces] == [[10.0, 9.5], [50.0, 52.5], [40.0, 38.0]]
    assert len({t.line.color for t in traces}) == 3  # three distinct colours
    assert {t.stackgroup for t in traces} == {"allocation"}  # one stack: the areas add up
    assert tuple(fig.layout.yaxis.range) == (0, 100)


def test_the_legend_reads_in_stack_order_between_the_title_and_the_toolbar() -> None:
    fig = allocation_chart(ROWS)

    assert fig.layout.legend.traceorder == "normal"
    assert fig.layout.legend.xanchor == "center" and fig.layout.legend.x == 0.5  # the title is at the left, the toolbar at the right


def test_the_chart_without_rows_says_so() -> None:
    fig = allocation_chart([])

    assert not fig.data
    assert fig.layout.annotations[0].text == "No trade data available"
