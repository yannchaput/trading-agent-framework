from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, IndicatorLine, Ledger
from trading_agent_framework.entities.enums import OrderSide, OrderType

NOW = datetime(2026, 1, 5, 16, tzinfo=UTC)


def test_ledger_starts_empty() -> None:
    ledger = Ledger()
    assert ledger.fills == []
    assert ledger.equity == []
    assert ledger.lines == []


def test_ledger_records_a_fill() -> None:
    ledger = Ledger()
    record = FillRecord(
        time=NOW, identifier="abc", symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=Decimal(10), filled_quantity=Decimal(10),
        price=Decimal("150.00"), trade_cost=Decimal("0.15"), trade_slippage=Decimal("0.05"),
    )

    ledger.record_fill(record)

    assert ledger.fills == [record]
    assert ledger.fills[0].status == "fill"  # default


def test_ledger_records_an_equity_sample() -> None:
    ledger = Ledger()
    sample = EquitySample(
        time=NOW, portfolio_value=Decimal(10500), cash=Decimal(500), positions_value=Decimal(10000)
    )

    ledger.record_equity(sample)

    assert ledger.equity == [sample]


def test_ledger_records_an_indicator_line() -> None:
    ledger = Ledger()
    line = IndicatorLine(
        time=NOW, name="sma_200", value=Decimal("148.5"), color=None,
        style="solid", plot_name="default_plot",
    )

    ledger.record_line(line)

    assert ledger.lines == [line]
