"""Simulated broker fees for backtesting US stocks/ETFs, chosen by `BROKER` (`alpaca` default, `ibkr`).

Pure: no I/O. Rates as of 2026-09-28 (Alpaca Brokerage Fee Schedule rev. 2026-09-17, SEC Fee
Rate Advisory 2026-2). They are constants, so a backtest of an earlier period pays today's
rates (the SEC fee was 0 before 2026-04-04). Every rounding and omission errs toward higher fees.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from trading_agent_framework.config.env import BrokerKind, BrokerSettings
from trading_agent_framework.utils.errors import ConfigurationError

SEC_FEE_RATE = Decimal("0.0000206")  # x sell value
FINRA_TAF_PER_SHARE = Decimal("0.000195")  # x shares sold
FINRA_TAF_MAX_PER_ORDER = Decimal("9.79")
CAT_FEE_PER_SHARE = Decimal("0.000003")  # x shares, buys and sells
IBKR_FIXED_PER_SHARE = Decimal("0.005")
IBKR_FIXED_MIN_PER_ORDER = Decimal("1.00")

_CENT = Decimal("0.01")


def _alpaca_commission(shares: Decimal) -> Decimal:
    return Decimal(0)


def _ibkr_commission(shares: Decimal) -> Decimal:
    # IBKR Pro Fixed; its 1%-of-trade-value cap is deliberately not applied (worst case).
    return max(IBKR_FIXED_MIN_PER_ORDER, IBKR_FIXED_PER_SHARE * shares)


COMMISSIONS: dict[BrokerKind, Callable[[Decimal], Decimal]] = {
    BrokerKind.ALPACA: _alpaca_commission,
    BrokerKind.IBKR: _ibkr_commission,
}


@dataclass(frozen=True, slots=True)
class TradeFees:
    """USD fees for the buy side and the sell side of one `TradingFeeFactory.fees` call."""

    buy: Decimal
    sell: Decimal


class TradingFeeFactory:
    """One broker's fee model: its per-order commission plus the regulatory fees every broker passes on."""

    def __init__(self, broker: BrokerKind) -> None:
        commission = COMMISSIONS.get(broker)
        if commission is None:
            raise ConfigurationError(f"No trading fee schedule for BROKER={broker.value}")
        self._broker = broker
        self._commission = commission

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TradingFeeFactory:
        """The fee model of the broker named by `BROKER`."""
        return cls(BrokerSettings.from_env(env).kind)

    @property
    def broker(self) -> BrokerKind:
        return self._broker

    def fees(self, *, buy_shares: Decimal, sell_shares: Decimal, buy_value: Decimal, sell_value: Decimal) -> TradeFees:
        """Each side with shares is one order; each side's total is rounded up to the cent."""
        for name, value in (
            ("buy_shares", buy_shares),
            ("sell_shares", sell_shares),
            ("buy_value", buy_value),
            ("sell_value", sell_value),
        ):
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative number, got {value}")
        buy = sell = Decimal(0)
        if buy_shares > 0:
            buy = _ceil_cent(self._commission(buy_shares) + CAT_FEE_PER_SHARE * buy_shares)
        if sell_shares > 0:
            sell = _ceil_cent(self._commission(sell_shares) + SEC_FEE_RATE * sell_value + min(FINRA_TAF_PER_SHARE * sell_shares, FINRA_TAF_MAX_PER_ORDER) + CAT_FEE_PER_SHARE * sell_shares)
        return TradeFees(buy=buy, sell=sell)


def _ceil_cent(amount: Decimal) -> Decimal:
    return amount.quantize(_CENT, rounding=ROUND_CEILING)
