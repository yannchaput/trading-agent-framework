"""Records what happened during a backtest run: fills, per-bar equity samples, and
strategy-added indicator lines. Decimal throughout -- the codebase's third float
boundary begins at `backtesting/metrics.py` and `backtesting/report.py`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_agent_framework.entities.enums import OrderSide, OrderType


@dataclass(frozen=True, slots=True)
class FillRecord:
    time: datetime
    identifier: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    filled_quantity: Decimal
    price: Decimal
    trade_cost: Decimal
    trade_slippage: Decimal
    status: str = "fill"
    event_kind: str = "trade"


@dataclass(frozen=True, slots=True)
class EquitySample:
    time: datetime
    portfolio_value: Decimal
    cash: Decimal
    positions_value: Decimal


@dataclass(frozen=True, slots=True)
class IndicatorLine:
    time: datetime
    name: str
    value: Decimal
    color: str | None
    style: str
    plot_name: str


class Ledger:
    """Accumulates fills, equity samples and indicator lines in memory during a run."""

    def __init__(self) -> None:
        self.fills: list[FillRecord] = []
        self.equity: list[EquitySample] = []
        self.lines: list[IndicatorLine] = []

    def record_fill(self, record: FillRecord) -> None:
        self.fills.append(record)

    def record_equity(self, sample: EquitySample) -> None:
        self.equity.append(sample)

    def record_line(self, line: IndicatorLine) -> None:
        self.lines.append(line)
