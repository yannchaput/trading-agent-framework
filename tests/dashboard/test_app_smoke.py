from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.backtesting.dashboard_contract import METRIC_SET_FIELDS

from trading_agent_framework.backtesting import report
from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, IndicatorLine, Ledger
from trading_agent_framework.entities.enums import OrderSide, OrderType

NOW = datetime(2026, 1, 5, 21, tzinfo=UTC)
LATER = datetime(2026, 1, 6, 21, tzinfo=UTC)


def _build_full_run(base: Path) -> Path:
    run_dir = base / "momentum" / "backtesting" / "2026-01-05_210000_backtesting"
    run_dir.mkdir(parents=True)

    ledger = Ledger()
    ledger.record_equity(EquitySample(time=NOW, portfolio_value=Decimal(10000), cash=Decimal(10000), positions_value=Decimal(0)))
    ledger.record_equity(EquitySample(time=LATER, portfolio_value=Decimal(10500), cash=Decimal(500), positions_value=Decimal(10000)))
    ledger.record_fill(FillRecord(
        time=LATER, identifier="abc", symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal(10), filled_quantity=Decimal(10), price=Decimal("950"),
        trade_cost=Decimal("1.0"), trade_slippage=Decimal("0.0"),
    ))
    ledger.record_line(IndicatorLine(time=NOW, name="sma_200", value=Decimal("148.5"), color=None, style="solid", plot_name="default_plot"))

    report.write_equity(run_dir, ledger.equity, {NOW: Decimal("400.0"), LATER: Decimal("404.0")})
    report.write_trades(run_dir, ledger)
    report.write_indicators(run_dir, ledger)

    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["total_return_strategy"] = 0.05
    metrics["cagr_strategy"] = 0.05
    metrics["raw"] = {
        "summary_tables": {
            "eoy_returns_vs_benchmark": [{"year": 2026, "strategy": 0.05, "benchmark": 0.01, "won": True}],
            "drawdowns": [],
        }
    }
    report.write_metrics(run_dir, metrics)

    report.write_settings(run_dir, {
        "name": "momentum", "mode": "backtesting", "run_ts": "2026-01-05_210000",
        "backtesting_start": NOW.isoformat(), "backtesting_end": LATER.isoformat(),
        "budget": 10000.0, "risk_free_rate": 0.03, "backtesting_data_sources": "yahoo",
        "backtest_time_seconds": 1.5, "timestep": "day", "sleeptime": "1D",
        "commission": 0.0, "slippage": 0.0, "warmup_trading_days": 0,
        "benchmark_symbol": "SPY", "framework_version": "0.1.0", "parameters": {},
    })
    return run_dir


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return _build_full_run(logs_dir)


APP_PATH = Path(__file__).resolve().parents[2] / "src" / "trading_agent_framework" / "dashboard" / "app.py"


def test_app_boots_and_scorecard_lists_the_run(run_dir: Path) -> None:
    # Use an absolute path anchored on this test file's own location, not a
    # cwd-relative one: the `run_dir` fixture does `monkeypatch.chdir(tmp_path)`
    # (needed for scan_runs("logs") to resolve), which would otherwise break a
    # repo-root-relative path to app.py.
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()

    assert not at.exception


def test_run_detail_renders_without_exception(run_dir: Path) -> None:
    from trading_agent_framework.dashboard.discovery import scan_runs
    from trading_agent_framework.dashboard.reader import load_run

    index = scan_runs("logs")
    assert len(index.runs) == 1

    run = load_run(index.runs[0])
    assert run.metrics is not None
    assert run.metrics.total_return_strategy == 0.05
    assert len(run.equity_curve) == 2


# --- Run Detail: agent telemetry ------------------------------------------------------

_AGENTS = {
    "trader": {
        "model": "qwen3-8b", "calls": 2, "tool_calls": 1, "input_tokens": 230, "output_tokens": 30, "reasoning_tokens": None,
        "total_tokens": 260, "latency_ms_total": 4000.0, "latency_ms_avg": 2000.0,
    }
}


def _detail_page(run_dir: Path, *, agents: dict | None, with_calls_db: bool) -> AppTest:
    import json

    from trading_agent_framework.agents.stats_store import LLMStatsStore, llm_stats_db_path
    from trading_agent_framework.agents.telemetry import CallRecord
    from trading_agent_framework.config.env import TradingMode
    from trading_agent_framework.dashboard.discovery import scan_runs

    if agents is not None:
        settings_path = run_dir / "settings.json"
        settings_path.write_text(json.dumps({**json.loads(settings_path.read_text()), "agents": agents}))
    if with_calls_db:
        store = LLMStatsStore(llm_stats_db_path(run_dir.parents[3], "momentum", TradingMode.BACKTESTING), run_id=run_dir.name)
        for ts, tokens in ((NOW, 120), (LATER, 140)):
            store.record(CallRecord(ts=ts, agent="trader", model="qwen3-8b", input_tokens=tokens - 20, output_tokens=20, reasoning_tokens=None,
                                    total_tokens=tokens, latency_ms=2000.0, tool_calls=0))
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.session_state["current_page"] = "Run Detail"
    at.session_state["detail_ref"] = scan_runs("logs").runs[0]
    at.run()
    return at


def test_run_detail_shows_agent_calls_when_telemetry_was_recorded(run_dir: Path) -> None:
    at = _detail_page(run_dir, agents=_AGENTS, with_calls_db=True)

    assert not at.exception
    assert "Agent calls" in [subheader.value for subheader in at.subheader]


def test_run_detail_explains_missing_per_call_data_when_the_database_is_gone(run_dir: Path) -> None:
    at = _detail_page(run_dir, agents=_AGENTS, with_calls_db=False)

    assert not at.exception
    assert any("No per-call data" in caption.value for caption in at.caption)


def test_run_detail_has_no_agent_section_for_a_run_without_agents(run_dir: Path) -> None:
    at = _detail_page(run_dir, agents=None, with_calls_db=False)

    assert not at.exception
    assert "Agent calls" not in [subheader.value for subheader in at.subheader]
    assert not any("No per-call data" in caption.value for caption in at.caption)


def test_run_detail_header_shows_the_agents_model(run_dir: Path) -> None:
    at = _detail_page(run_dir, agents=_AGENTS, with_calls_db=False)

    assert any("qwen3-8b" in markdown.value for markdown in at.markdown)
