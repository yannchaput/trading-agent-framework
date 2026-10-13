from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from tests.fakes import make_api_error

from trading_agent_framework.dashboard import account_reader
from trading_agent_framework.dashboard.pnl import AccountPnl, BrokerPosition

KEYS = "ALPACA_API_KEY=k\nALPACA_API_SECRET=s\nALPACA_DATA_API_KEY=dk\nALPACA_DATA_API_SECRET=ds\n"

PNL = AccountPnl(
    days=[{"date": "2026-10-12", "realized": 0.0, "unrealized": 20.0}, {"date": "2026-10-13", "realized": 5.0, "unrealized": 30.0}],
    realized_total=5.0,
    unrealized_now=30.0,
    broker_unrealized=30.0,
    positions=[BrokerPosition("AAA", 10.0, 100.0, 103.0, 1030.0, 30.0, 0.03)],
    mismatches=[],
    unmatched_sales={},
    unpriced=[],
)


def _paper_script() -> None:
    # Serialized by AppTest.from_function: must be self-contained.
    from trading_agent_framework.dashboard._pages.account import page_paper

    page_paper()


def _live_script() -> None:
    from trading_agent_framework.dashboard._pages.account import page_live

    page_live()


def _switching_script() -> None:
    # Stands in for the top navigation: session_state["tab"] picks the tab drawn on this run.
    import streamlit as st

    from trading_agent_framework.dashboard._pages.account import page_account

    page_account(st.session_state.get("tab", "paper"))


@pytest.fixture
def env_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    env = tmp_path / "env"
    env.mkdir()
    (env / ".env.alpha.paper").write_text(KEYS, encoding="utf-8")
    (env / ".env.beta.paper").write_text(KEYS, encoding="utf-8")
    (env / ".env.alpha.live").write_text("BROKER_API_IS_PAPER=false\n" + KEYS, encoding="utf-8")
    monkeypatch.setattr(account_reader, "default_env_dir", lambda: env)
    st.cache_data.clear()
    return env


def _run(script: Callable[[], None], monkeypatch: pytest.MonkeyPatch, result: object) -> tuple[AppTest, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    def fake_load(strategy: str, mode: str, env_dir: Path, today: date) -> AccountPnl:
        calls.append((strategy, mode))
        if isinstance(result, BaseException):
            raise result
        return result  # ty: ignore[invalid-return-type]

    monkeypatch.setattr(account_reader, "load_strategy_pnl", fake_load)
    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    return at, calls


def test_paper_tab_lists_the_paper_strategies_and_shows_the_first_ones_pnl(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run(_paper_script, monkeypatch, PNL)

    assert not at.exception
    assert at.sidebar.selectbox[0].options == ["alpha", "beta"]
    assert calls == [("alpha", "paper")]
    assert [title.value for title in at.title] == ["Paper"]
    assert len(at.get("plotly_chart")) == 1
    assert at.dataframe[0].value["symbol"].tolist() == ["AAA"]
    page_text = " ".join(md.value for md in at.markdown)
    assert "+$5.00" in page_text and "+$30.00" in page_text and "+$35.00" in page_text


def test_choosing_another_strategy_fetches_its_account(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run(_paper_script, monkeypatch, PNL)

    at.sidebar.selectbox[0].select("beta").run()

    assert calls[-1] == ("beta", "paper")


def test_live_lists_the_live_strategies(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run(_live_script, monkeypatch, PNL)

    assert at.sidebar.selectbox[0].options == ["alpha"]
    assert calls == [("alpha", "live")]


def test_each_tab_remembers_its_strategy_across_tab_switches(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Streamlit drops a widget's state on a run that does not draw it (another tab): the page must keep the choice.
    at, _ = _run(_switching_script, monkeypatch, PNL)
    at.sidebar.selectbox[0].select("beta").run()

    at.session_state["tab"] = "live"
    at.run()
    assert at.sidebar.selectbox[0].value == "alpha"

    at.session_state["tab"] = "paper"
    at.run()
    assert at.sidebar.selectbox[0].value == "beta"


def test_mismatches_unmatched_sales_and_unpriced_symbols_are_warned(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    flawed = dataclasses.replace(PNL, mismatches=[("AAA", 10.0, 15.0)], unmatched_sales={"BBB": 5.0}, unpriced=["ZZZ"])
    at, _ = _run(_paper_script, monkeypatch, flawed)

    (warning,) = at.warning
    assert "AAA" in warning.value and "BBB" in warning.value and "ZZZ" in warning.value


def test_an_alpaca_error_is_shown_not_raised(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, _ = _run(_paper_script, monkeypatch, make_api_error(500))

    assert not at.exception
    assert "Could not reach Alpaca" in at.error[0].value


def test_blank_keys_name_the_variable_and_never_a_value(env_dir: Path) -> None:
    (env_dir / ".env.alpha.paper").write_text("ALPACA_API_KEY=\nALPACA_API_SECRET=t0p-s3cret\n", encoding="utf-8")
    at = AppTest.from_function(_paper_script, default_timeout=30)  # the real load_strategy_pnl: fails before any network
    at.run()

    assert not at.exception
    assert "ALPACA_API_KEY" in at.error[0].value
    assert "t0p-s3cret" not in at.error[0].value


def test_no_fill_history_shows_positions_and_says_so(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    at, _ = _run(_paper_script, monkeypatch, dataclasses.replace(PNL, days=[], realized_total=0.0))

    assert len(at.get("plotly_chart")) == 0
    assert any("no fill history" in info.value.lower() for info in at.info)
    assert at.dataframe[0].value["symbol"].tolist() == ["AAA"]


def test_an_empty_account_says_no_trades_yet(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    at, _ = _run(_paper_script, monkeypatch, dataclasses.replace(PNL, days=[], realized_total=0.0, unrealized_now=0.0, positions=[]))

    assert any("No trades yet" in info.value for info in at.info)


def test_a_mode_without_env_files_says_how_to_add_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(account_reader, "default_env_dir", lambda: tmp_path / "nothing")
    at = AppTest.from_function(_paper_script, default_timeout=30)
    at.run()

    assert not at.exception
    assert ".env.<strategy>.paper" in at.info[0].value


def test_a_brokers_unrealized_figure_on_another_cost_basis_is_explained(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    at, _ = _run(_paper_script, monkeypatch, dataclasses.replace(PNL, unrealized_now=0.0, broker_unrealized=-500.0))

    (caption,) = [c.value for c in at.caption if "FIFO" in c.value]
    assert "-$500.00" in caption and "average cost" in caption


def test_the_same_figure_needs_no_explanation(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, _ = _run(_paper_script, monkeypatch, PNL)

    assert not [c for c in at.caption if "FIFO" in c.value]


def test_a_paper_tab_whose_keys_hit_the_live_account_says_so(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (env_dir / ".env.alpha.paper").write_text("BROKER_API_IS_PAPER=false\n" + KEYS, encoding="utf-8")
    at, _ = _run(_paper_script, monkeypatch, PNL)

    (warning,) = at.warning
    assert "LIVE" in warning.value and ".env.alpha.paper" in warning.value
    assert "live Alpaca account" in " ".join(c.value for c in at.caption)


def test_a_live_tab_whose_keys_hit_the_paper_account_says_so(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (env_dir / ".env.alpha.live").write_text(KEYS, encoding="utf-8")  # BROKER_API_IS_PAPER unset: paper
    at, _ = _run(_live_script, monkeypatch, PNL)

    (warning,) = at.warning
    assert "PAPER" in warning.value and ".env.alpha.live" in warning.value


def test_keys_that_match_their_tab_raise_no_account_warning(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paper, _ = _run(_paper_script, monkeypatch, PNL)
    live, _ = _run(_live_script, monkeypatch, PNL)

    assert not paper.warning and not live.warning
