from __future__ import annotations

import re

import pytest

from trading_agent_framework.strategies.congress_trades import prompts
from trading_agent_framework.strategies.congress_trades.desk import TradeDesk
from trading_agent_framework.strategies.congress_trades.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams

SYSTEMS = {
    "researcher": (prompts.researcher_system("Nancy Pelosi"), ["list_filings", "read_filing", "submit_holdings"]),
    "portfolio": (prompts.portfolio_system("Nancy Pelosi"), ["submit_target"]),
    "trader": (prompts.trader_system(), ["get_positions", "get_account_balance", "get_last_price", "place_order", "check_orders", "submit_trade_report"]),
}


@pytest.mark.parametrize("name", SYSTEMS)
def test_every_prompt_names_the_tools_the_agent_has(name: str) -> None:
    text, tools = SYSTEMS[name]

    for tool in tools:
        assert tool in text


def test_the_prompts_only_name_tools_that_really_exist() -> None:
    from tests.fakes import FakeBroker, FakeClock, et

    from trading_agent_framework.agents.tools.account import account_tools
    from trading_agent_framework.agents.tools.congress import congress_research_tools
    from trading_agent_framework.agents.tools.market_data import market_data_tools
    from trading_agent_framework.core.strategy import Strategy

    strategy = Strategy(FakeBroker(FakeClock(et(2026, 9, 14, 10))))
    real = set(submit_tools(HandoffRecorder(CongressParams())))
    real |= {t.__name__ for t in account_tools(strategy)}
    real |= {t.__name__ for t in market_data_tools(strategy)}
    real |= {t.__name__ for t in congress_research_tools(strategy, object())}  # type: ignore[arg-type]
    real |= {t.__name__ for t in TradeDesk(strategy, CongressParams()).tools()}

    for text, _ in SYSTEMS.values():
        for tool in re.findall(r"\b(?:submit|list|read|get|place|check)_[a-z_]+", text):
            assert tool in real, tool


@pytest.mark.parametrize("name", SYSTEMS)
def test_every_prompt_ends_with_its_submit_contract_and_states_the_language_rule(name: str) -> None:
    text, tools = SYSTEMS[name]

    assert "Write everything in English" in text
    assert text.rstrip().endswith("reply with one line and call no other tool.")
    assert "exactly once" in text and tools[-1] in text


@pytest.mark.parametrize("name", SYSTEMS)
def test_no_prompt_states_how_often_the_strategy_runs(name: str) -> None:
    text = SYSTEMS[name][0].lower()

    for word in ("daily", "every day", "each day", "weekly", "every week", "every morning", "once a day", "hourly"):
        assert word not in text


def test_the_member_is_named_in_the_research_and_portfolio_prompts() -> None:
    assert "Nancy Pelosi" in prompts.researcher_system("Nancy Pelosi")
    assert "Jane Doe" in prompts.portfolio_system("Jane Doe")
    assert "Nancy Pelosi" not in prompts.portfolio_system("Jane Doe")


def test_the_research_prompt_states_the_disclosure_mechanics_and_the_limits() -> None:
    text = prompts.researcher_system("Nancy Pelosi")

    for fragment in ("period_end", "value band", "baseline", "stocks only", "drop true", "You do not trade"):
        assert fragment in text


def test_the_portfolio_prompt_states_the_tier_rule() -> None:
    text = prompts.portfolio_system("Nancy Pelosi")

    assert "must never get a smaller weight than a holding in a lower tier" in text
    assert "weight 0" in text and "stays in cash" in text


def test_the_trading_prompt_states_the_order_rules_the_desk_enforces() -> None:
    text = " ".join(prompts.trader_system().split())

    assert "Send every sell before the first buy" in text
    assert "SMALLER of buying_power and (cash + the proceeds of the sells you placed in this run)" in text
    assert "never against buying_power alone" in text
    assert "Never short" in text
    assert "still working" in text and "do not cancel or repeat it" in text


def test_the_task_prompts_point_at_the_context() -> None:
    for task in (prompts.RESEARCHER_TASK, prompts.PORTFOLIO_TASK, prompts.TRADER_TASK):
        assert "context" in task


def test_the_retry_prompt_quotes_the_error_and_forbids_new_work() -> None:
    text = prompts.retry_prompt("submit_target", "AAA is not one of the holdings")

    assert "submit_target" in text and "AAA is not one of the holdings" in text
    assert "place no new order" in text
    assert "valid argument" in text


def test_the_trading_agent_really_has_the_tools_its_prompt_names() -> None:
    from tests.fakes import FakeBroker, FakeClock, et

    from trading_agent_framework.core.strategy import Strategy

    desk = TradeDesk(Strategy(FakeBroker(FakeClock(et(2026, 9, 14, 10)))), CongressParams())

    assert [t.__name__ for t in desk.tools()] == ["place_order", "check_orders"]
