from __future__ import annotations

from decimal import Decimal

import pytest
from tests.fakes import make_ib_summary

from trading_agent_framework.brokers.ibkr import account
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError


def test_parse_account() -> None:
    balances = account.parse_account(make_ib_summary(cash="10000", net_liquidation="25000", buying_power="10000"), "DU123")

    assert balances == AccountBalances(cash=Decimal(10000), portfolio_value=Decimal(25000), buying_power=Decimal(10000))


def test_parse_account_ignores_other_accounts_values() -> None:
    values = make_ib_summary(account="U999", cash="1") + make_ib_summary(account="DU123", cash="10000")

    assert account.parse_account(values, "DU123").cash == Decimal(10000)


def test_parse_account_reports_a_missing_tag() -> None:
    values = [v for v in make_ib_summary() if v.tag != "BuyingPower"]

    with pytest.raises(BrokerError, match="BuyingPower"):
        account.parse_account(values, "DU123")


def test_is_paper_account() -> None:
    assert account.is_paper_account("DU1234567")
    assert not account.is_paper_account("U1234567")


def test_a_paper_cash_account_passes_quietly() -> None:
    assert account.check_account(make_ib_summary(), "DU123", is_paper=True) == []


@pytest.mark.parametrize(("account_id", "is_paper"), [("DU123", False), ("U123", True)])
def test_the_paper_flag_must_match_the_account(account_id: str, is_paper: bool) -> None:
    with pytest.raises(ConfigurationError, match="BROKER_API_IS_PAPER"):
        account.check_account(make_ib_summary(account=account_id), account_id, is_paper=is_paper)


def test_a_margin_account_is_refused_live() -> None:
    values = make_ib_summary(account="U123", cash="10000", buying_power="40000")

    with pytest.raises(ConfigurationError, match="margin"):
        account.check_account(values, "U123", is_paper=False)


def test_a_margin_account_only_warns_on_paper() -> None:
    [warning] = account.check_account(make_ib_summary(cash="10000", buying_power="40000"), "DU123", is_paper=True)

    assert "margin" in warning


def test_a_non_usd_base_currency_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="USD"):
        account.check_account(make_ib_summary(currency="EUR"), "DU123", is_paper=True)


def test_margin_heuristic_exactly_at_threshold_is_not_flagged() -> None:
    # Threshold with cash=10000 is: 10000 * 1.05 + 1 = 10501
    # At the exact threshold (not above), should pass as cash account
    assert account.check_account(make_ib_summary(cash="10000", buying_power="10501"), "DU123", is_paper=True) == []


def test_margin_heuristic_just_below_threshold_is_not_flagged() -> None:
    # Just below threshold (10500 < 10501)
    assert account.check_account(make_ib_summary(cash="10000", buying_power="10500"), "DU123", is_paper=True) == []


def test_margin_heuristic_just_above_threshold_is_flagged_on_paper() -> None:
    # Just above threshold (10502 > 10501)
    [warning] = account.check_account(make_ib_summary(cash="10000", buying_power="10502"), "DU123", is_paper=True)

    assert "margin" in warning


def test_margin_heuristic_just_above_threshold_is_refused_live() -> None:
    # Just above threshold (10502 > 10501) on live account
    with pytest.raises(ConfigurationError, match="margin"):
        account.check_account(make_ib_summary(account="U123", cash="10000", buying_power="10502"), "U123", is_paper=False)
