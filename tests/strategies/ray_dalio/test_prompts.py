from __future__ import annotations

from trading_agent_framework.strategies.ray_dalio.prompts import (
    ANALYSTS,
    CONTEXT,
    DISAGREEMENT,
    LEVERAGE_RULE,
    ORDER_READINESS_RULE,
    SIZING_RULE,
    SYSTEM_PROMPTS,
    TASK_PROMPTS,
    TRADER,
    UNIVERSE,
)


def _context_strings() -> list[str]:
    values: list[str] = []
    for value in CONTEXT.values():
        if isinstance(value, dict):
            values.extend(f"{key} {text}" for key, text in value.items())
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
        elif isinstance(value, str):
            values.append(value)
    return values


def test_universe_is_lumibots_54_symbols_with_shv():
    assert len(UNIVERSE) == 54 and list(UNIVERSE)[-1] == "SHV"


def test_every_symbol_has_a_description_and_the_agents_get_them():
    assert all(isinstance(text, str) and text.strip() for text in UNIVERSE.values())
    assert CONTEXT["universe"] == UNIVERSE


def test_the_funds_the_agents_misread_are_described_right():
    expected = {"TMF": "+3x", "TWM": "-2x", "UGL": "+2x", "GLL": "-2x", "SSO": "+2x", "UPRO": "+3x", "TQQQ": "+3x", "SHV": "1x"}
    for symbol, leverage in expected.items():
        assert UNIVERSE[symbol].startswith(leverage), symbol


def test_every_inverse_fund_says_so():
    for symbol, text in UNIVERSE.items():
        assert text.startswith("-") == ("(inverse)" in text), symbol


def test_the_leverage_rule_points_at_the_descriptions():
    assert "never from memory" in LEVERAGE_RULE


def test_every_agent_has_a_system_and_a_task_prompt():
    assert set(SYSTEM_PROMPTS) == set(TASK_PROMPTS) == {*ANALYSTS, DISAGREEMENT, TRADER}


def test_no_lumibot_tool_name_is_left():
    gone = ("alpaca_news", "account_portfolio", "account_positions", "market_last_price", "orders_submit_order", "list_fred_series", "LumiBot")
    texts = [*SYSTEM_PROMPTS.values(), *TASK_PROMPTS.values(), *_context_strings()]
    for name in gone:
        assert all(name not in text for text in texts), name


def test_only_the_trader_is_told_about_orders_and_sizing():
    assert ORDER_READINESS_RULE in SYSTEM_PROMPTS[TRADER] and SIZING_RULE in TASK_PROMPTS[TRADER]
    others = (*ANALYSTS, DISAGREEMENT)
    assert all("submit_order" not in SYSTEM_PROMPTS[name] for name in others)
    assert all("submit_order" not in TASK_PROMPTS[name] for name in others)
    assert all(ORDER_READINESS_RULE not in SYSTEM_PROMPTS[name] and SIZING_RULE not in TASK_PROMPTS[name] for name in others)


def test_sizing_rule_points_at_spendable():
    assert "spendable" in SIZING_RULE and "buying_power" not in SIZING_RULE
