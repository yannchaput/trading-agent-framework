from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.brokers import fees
from trading_agent_framework.brokers.fees import TradeFees, TradingFeeFactory
from trading_agent_framework.config.env import BrokerKind
from trading_agent_framework.utils.errors import ConfigurationError

D = Decimal
ALPACA = TradingFeeFactory(BrokerKind.ALPACA)
IBKR = TradingFeeFactory(BrokerKind.IBKR)


def _buy(model: TradingFeeFactory, shares: int, value: int) -> TradeFees:
    return model.fees(buy_shares=D(shares), sell_shares=D(0), buy_value=D(value), sell_value=D(0))


def _sell(model: TradingFeeFactory, shares: int, value: int) -> TradeFees:
    return model.fees(buy_shares=D(0), sell_shares=D(shares), buy_value=D(0), sell_value=D(value))


def test_alpaca_buy_pays_only_the_cat_fee_rounded_up_to_the_cent() -> None:
    assert _buy(ALPACA, 100, 65000) == TradeFees(buy=D("0.01"), sell=D(0))  # CAT 0.0003


def test_alpaca_sell_pays_sec_taf_and_cat() -> None:
    # SEC 65000 * 0.0000206 = 1.339, TAF 100 * 0.000195 = 0.0195, CAT 0.0003 -> 1.3588
    assert _sell(ALPACA, 100, 65000) == TradeFees(buy=D(0), sell=D("1.36"))


def test_ibkr_buy_pays_the_one_dollar_minimum_below_200_shares() -> None:
    assert _buy(IBKR, 100, 65000).buy == D("1.01")  # max(1, 0.5) + CAT 0.0003


def test_ibkr_buy_pays_half_a_cent_a_share_above_200_shares() -> None:
    assert _buy(IBKR, 1000, 650000).buy == D("5.01")  # 5 + CAT 0.003


def test_ibkr_sell_adds_the_regulatory_fees_to_its_commission() -> None:
    assert _sell(IBKR, 100, 65000).sell == D("2.36")  # 1 + 1.339 + 0.0195 + 0.0003


def test_the_taf_is_capped_per_order() -> None:
    # TAF would be 100,000 * 0.000195 = 19.5, capped at 9.79; CAT 0.3; SEC on value 0 is 0
    assert _sell(ALPACA, 100000, 0).sell == D("10.09")


def test_a_side_without_shares_costs_nothing_even_at_ibkr() -> None:
    zero = D(0)
    assert IBKR.fees(buy_shares=zero, sell_shares=zero, buy_value=zero, sell_value=zero) == TradeFees(buy=zero, sell=zero)


def test_both_sides_in_one_call_are_two_orders() -> None:
    result = IBKR.fees(buy_shares=D(10), sell_shares=D(10), buy_value=D(1000), sell_value=D(1000))
    # buy: 1 + 0.00003; sell: 1 + 0.0206 + 0.00195 + 0.00003 = 1.02258
    assert result == TradeFees(buy=D("1.01"), sell=D("1.03"))


@pytest.mark.parametrize("field", ["buy_shares", "sell_shares", "buy_value", "sell_value"])
def test_negative_inputs_are_rejected(field: str) -> None:
    kwargs = {"buy_shares": D(0), "sell_shares": D(0), "buy_value": D(0), "sell_value": D(0), field: D(-1)}
    with pytest.raises(ValueError, match=field):
        ALPACA.fees(**kwargs)


def test_from_env_defaults_to_alpaca() -> None:
    assert TradingFeeFactory.from_env({}).broker is BrokerKind.ALPACA


def test_from_env_reads_broker() -> None:
    assert TradingFeeFactory.from_env({"BROKER": "ibkr"}).broker is BrokerKind.IBKR


def test_from_env_rejects_an_unknown_broker() -> None:
    with pytest.raises(ConfigurationError, match="Unknown BROKER"):
        TradingFeeFactory.from_env({"BROKER": "foo"})


def test_a_broker_without_a_fee_schedule_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(fees.COMMISSIONS, BrokerKind.IBKR)
    with pytest.raises(ConfigurationError, match="BROKER=ibkr"):
        TradingFeeFactory(BrokerKind.IBKR)
