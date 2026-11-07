"""Records what happened during a backtest run: fills, per-bar equity samples, and strategy-added indicator lines.

Decimal throughout -- the codebase's third float
boundary begins at `backtesting/metrics.py` and `backtesting/report.py`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_agent_framework.entities.enums import OrderSide, OrderType


@dataclass(frozen=True, slots=True)
class FillRecord:
    """One fill (or other order event) in the backtest ledger.

    Attributes:
        time: When it happened.
        identifier: Client order id.
        symbol: Ticker.
        side: Buy or sell.
        order_type: Order type.
        quantity: Ordered quantity.
        filled_quantity: Quantity filled by this event.
        price: Fill price.
        trade_cost: Fees charged.
        trade_slippage: Slippage cost.
        status: Order status after the event.
        event_kind: Kind of event, `trade` for a fill.
    """

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
    """Account equity at one moment.

    Attributes:
        time: When it was sampled.
        portfolio_value: Cash plus positions.
        cash: Settled cash.
        positions_value: Market value of the positions.
    """

    time: datetime
    portfolio_value: Decimal
    cash: Decimal
    positions_value: Decimal


@dataclass(frozen=True, slots=True)
class IndicatorLine:
    """One point of an indicator line plotted by the strategy.

    Attributes:
        time: Timestamp of the point.
        name: Line name.
        value: Value at that time.
        color: Plot colour, or None for the default.
        style: Plot style.
        plot_name: Chart the line belongs to.
    """

    time: datetime
    name: str
    value: Decimal
    color: str | None
    style: str
    plot_name: str


class Ledger:
    """Accumulates fills, equity samples and indicator lines in memory during a run."""

    def __init__(self) -> None:
        """Create an empty ledger."""
        self.fills: list[FillRecord] = []
        self.equity: list[EquitySample] = []
        self.lines: list[IndicatorLine] = []

    def record_fill(self, record: FillRecord) -> None:
        """Append a fill.

        Args:
            record: The fill to store.
        """
        self.fills.append(record)

    def record_equity(self, sample: EquitySample) -> None:
        """Append an equity sample.

        Args:
            sample: The sample to store.
        """
        self.equity.append(sample)

    def record_line(self, line: IndicatorLine) -> None:
        """Append an indicator point.

        Args:
            line: The point to store.
        """
        self.lines.append(line)
