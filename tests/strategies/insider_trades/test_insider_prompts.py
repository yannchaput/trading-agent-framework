from __future__ import annotations

import re

import pytest

from trading_agent_framework.strategies.common.trade_desk import TradeDesk
from trading_agent_framework.strategies.insider_trades import prompts
from trading_agent_framework.strategies.insider_trades.facts import InsiderFacts
from trading_agent_framework.strategies.insider_trades.handoff import InsiderRecorder, submit_tools
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams

PARAMS = InsiderParams()


class _NoSource:
    def known(self, as_of: object) -> object:
        raise AssertionError("the tools are only listed here, never called")


SYSTEMS = {
    "researcher": (prompts.researcher_system(), ["get_insider_trades", "read_form4", "submit_note"]),
    "trader": (prompts.trader_system(PARAMS), ["get_positions", "get_account_balance", "get_last_price", "submit_target", "place_order", "check_orders", "submit_trade_report"]),
    "trader_retry": (prompts.trader_retry_system(), ["get_positions", "get_account_balance", "get_last_price", "place_order", "check_orders", "submit_trade_report"]),
}


@pytest.mark.parametrize("name", SYSTEMS)
def test_every_prompt_names_the_tools_the_agent_has(name: str) -> None:
    text, tools = SYSTEMS[name]

    for tool in tools:
        assert tool in text


def test_the_prompts_only_name_tools_that_really_exist() -> None:
    from tests.fakes import FakeBroker, FakeClock, et

    from trading_agent_framework.agents.tools.account import account_tools
    from trading_agent_framework.agents.tools.market_data import market_data_tools
    from trading_agent_framework.core.strategy import Strategy
    from trading_agent_framework.strategies.insider_trades.tools import insider_research_tools

    strategy = Strategy(FakeBroker(FakeClock(et(2026, 9, 14, 10))))
    real = set(submit_tools(InsiderRecorder(PARAMS)))
    real |= {t.__name__ for t in account_tools(strategy)}
    real |= {t.__name__ for t in market_data_tools(strategy)}
    real |= {t.__name__ for t in insider_research_tools(strategy, InsiderFacts(strategy, _NoSource(), PARAMS), lambda ticker: None)}
    real |= {t.__name__ for t in TradeDesk(strategy, PARAMS, log_tag="insider_trades").tools()}

    for text, _ in SYSTEMS.values():
        for tool in re.findall(r"\b(?:submit|read|get|place|check)_[a-z_0-9]+", text):
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


def test_the_research_prompt_states_its_one_job_and_the_limits() -> None:
    text = " ".join(prompts.researcher_system().split())

    assert "ONE ticker" in text
    assert text.index("calling get_insider_trades") < text.index("read_form4")
    for fragment in ("never recompute the dollars", "10b5-1 plan stay routine", "an override of one is refused", "only when you can give a reason", "or an empty list", "You do not trade"):
        assert fragment in text


def test_the_trader_prompt_states_the_target_rules_with_the_settings_numbers() -> None:
    params = InsiderParams(watchlist=("AAA", "BBB", "CCC", "DDD"), max_total_weight=0.8, min_weight=0.05, max_position_weight=0.3)
    text = " ".join(prompts.trader_system(params).split())

    assert "EVERY watchlist stock a weight between 0.05 and 0.3" in text
    assert "at most 0.8" in text and "0.2000 each" in text
    assert "A buying stock must never get less than a selling stock" in text
    assert "When no stock has a signal, every weight is the even weight" in text
    assert "names the fact in the stock's note" in text


def test_the_trading_prompts_state_the_order_rules_the_desk_enforces() -> None:
    for text in (prompts.trader_system(PARAMS), prompts.trader_retry_system()):
        text = " ".join(text.split())
        assert "Send every sell before the first buy" in text
        assert "SMALLER of buying_power and (cash + the proceeds of the sells you placed in this run)" in text
        assert "never against buying_power alone" in text
        assert "Never short" in text
        assert "still working" in text and "do not cancel or repeat it" in text


def test_the_retry_system_prompt_has_no_target_to_submit() -> None:
    text = prompts.trader_retry_system()

    assert "submit_target" not in text and "You set no target" in text


def test_the_task_prompts_point_at_the_context() -> None:
    for task in (prompts.RESEARCHER_TASK, prompts.TRADER_TASK, prompts.TRADER_RETRY_TASK):
        assert "context" in task


def test_the_retry_task_asks_for_no_new_target() -> None:
    text = prompts.TRADER_RETRY_TASK

    assert "Set no new target" in text and "remaining shortfalls" in text and "submit_target" not in text


def test_the_retry_prompt_quotes_the_error_and_forbids_new_work() -> None:
    text = prompts.retry_prompt("submit_target", "AAA is not on the watchlist")

    assert "submit_target" in text and "AAA is not on the watchlist" in text
    assert "place no new order" in text
    assert "valid argument" in text
