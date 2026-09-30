from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeNewsProvider
from tests.strategies.vwap_pullback.test_vwap_desk_entries import Rig

from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState
from trading_agent_framework.strategies.vwap_pullback.tools import budgeted_search_news, entry_tools, exit_tools, setup_rows, trade_rows


def _tools(tools: list) -> dict:
    return {tool.__name__: tool for tool in tools}


def test_setup_rows_show_pullback_and_triggered_setups_with_plan_and_headlines(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.state.setups["WATCHED"] = Setup(symbol="WATCHED")
    rig.state.headlines["AAA"] = [{"headline": "AAA beats", "created_at": "2026-09-01T07:00:00-04:00", "source": "b"}]
    rows = setup_rows(rig.desk)
    assert [row["symbol"] for row in rows] == ["AAA"]
    row = rows[0]
    assert row["state"] == "triggered" and row["planned_stop"] == 99.3 and row["r_per_share"] == 0.7
    assert row["headlines"][0]["headline"] == "AAA beats"
    assert "z_rs" not in row and "z_rvol" not in row  # no rule reads them any more: not worth the tokens


def test_trade_rows_show_the_stop_open_r_levels_and_new_headlines(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    rig.state.headlines["AAA"] = [
        {"headline": "before entry", "created_at": "2026-09-01T09:00:00-04:00", "source": "b"},
        {"headline": "after entry", "created_at": "2026-09-01T10:00:30-04:00", "source": "b"},
    ]
    row = trade_rows(rig.desk, rig.clock.now())[0]
    assert row["symbol"] == "AAA" and row["quantity"] == 249 and row["stop_kind"] == "stop" and row["stop_level"] == 99.3
    assert row["unrealised_r"] == 0.0 and row["vwap"] == 99.9 and row["minutes_to_flatten"] == 349
    assert [h["headline"] for h in row["headlines_since_entry"]] == ["after entry"]


def test_search_news_has_a_per_run_budget(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.broker._news_source = FakeNewsProvider()
    search = budgeted_search_news(rig.strategy, 2)
    with agent_call_context(run_id="run-1"):
        assert "error" not in search(symbols="AAA")
        assert "error" not in search(symbols="BBB")
        assert "budget" in search(symbols="CCC")["error"]
    with agent_call_context(run_id="run-2"):
        assert "error" not in search(symbols="AAA")


def test_get_intraday_bars_has_a_per_run_budget(tmp_path: Path) -> None:
    # One response of a local model once asked for 14 bar fetches at once; their results overflowed the 32k context.
    rig = Rig(tmp_path)
    budget = rig.desk.params.bars_calls_per_run
    for tools in (entry_tools(rig.strategy, rig.desk), exit_tools(rig.strategy, rig.desk)):
        bars = _tools(tools)["get_intraday_bars"]
        with agent_call_context(run_id="run-1"):
            for _ in range(budget):
                assert "error" not in bars("AAA")
            assert "budget" in bars("AAA")["error"]
        with agent_call_context(run_id="run-2"):
            assert "error" not in bars("AAA")


def test_entry_tools_delegate_to_the_desk(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    tools = _tools(entry_tools(rig.strategy, rig.desk))
    assert set(tools) == {"get_setups", "get_intraday_bars", "search_news", "enter_long", "pass_on_setup"}
    assert tools["get_setups"]()["setups"][0]["symbol"] == "AAA"
    bars = tools["get_intraday_bars"]("AAA", 100)
    assert len(bars["bars"]) == 1 and bars["bars"][0]["close"] == 100.0
    assert "error" in tools["get_intraday_bars"]("ZZZ")
    assert tools["enter_long"]("AAA", "earnings", "clean")["status"] == "entry submitted"
    assert rig.state.setups["AAA"].state is SetupState.IN_TRADE


def test_exit_tools_delegate_to_the_desk(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    tools = _tools(exit_tools(rig.strategy, rig.desk))
    assert set(tools) == {"get_open_trades", "get_intraday_bars", "search_news", "take_partial_profit", "tighten_stop", "replace_stop_with_trailing", "exit_position", "hold"}
    assert tools["get_open_trades"]()["trades"][0]["symbol"] == "AAA"
    assert tools["hold"]("AAA", "noise")["status"] == "holding"
    for tool in tools.values():
        assert tool.__doc__ and "\n" not in tool.__doc__.strip()
