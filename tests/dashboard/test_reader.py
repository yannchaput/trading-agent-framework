from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from tests.backtesting.dashboard_contract import METRIC_SET_FIELDS

from trading_agent_framework.agents.stats_store import LLMStatsStore, llm_stats_db_path
from trading_agent_framework.agents.telemetry import CallRecord
from trading_agent_framework.backtesting import report
from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, Ledger
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.dashboard.models import RunRef
from trading_agent_framework.dashboard.reader import (
    get_benchmark_symbol,
    load_agent_calls,
    load_cumulative_returns,
    load_equity_curve,
    load_indicator_lines,
    load_intraday_exposure,
    load_metrics,
    load_parameters,
    load_portfolio_breakdown,
    load_run,
    load_settings,
    load_trades_curve,
    load_yearly_returns,
    save_decision,
    save_regime,
)
from trading_agent_framework.entities.enums import OrderSide, OrderType


def _run_dir(tmp_path: Path, strategy: str = "momentum", run_ts: str = "2026-06-22_194053") -> Path:
    d = tmp_path / strategy / "backtesting" / f"{run_ts}_backtesting"
    d.mkdir(parents=True)
    return d


def _ref(run_dir: Path) -> RunRef:
    return RunRef.from_path(str(run_dir))


def test_load_metrics_reads_a_full_metrics_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["total_return_strategy"] = 0.42
    metrics["sharpe_strategy"] = 1.5
    metrics["raw"] = {"summary_tables": {"eoy_returns_vs_benchmark": [{"year": 2026, "strategy": 0.1, "benchmark": 0.05, "won": True}], "drawdowns": []}}
    report.write_metrics(run_dir, metrics)

    loaded = load_metrics(_ref(run_dir))

    assert loaded is not None
    assert loaded.total_return_strategy == 0.42
    assert loaded.sharpe_strategy == 1.5
    assert loaded.raw["summary_tables"]["eoy_returns_vs_benchmark"][0]["year"] == 2026


def test_load_metrics_defaults_a_benchmark_less_runs_null_fields_to_zero(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    # A benchmark-less run: compute_metrics only returns strategy-side fields;
    # write_metrics fills every *_benchmark/relative field with JSON null.
    report.write_metrics(run_dir, {"total_return_strategy": 0.1, "raw": {}})

    loaded = load_metrics(_ref(run_dir))

    assert loaded is not None
    assert loaded.total_return_strategy == 0.1
    assert loaded.sharpe_benchmark == 0.0
    assert loaded.beta == 0.0


def test_load_metrics_returns_none_when_metrics_json_is_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_metrics(_ref(run_dir)) is None


def _settings_payload(**overrides) -> dict:
    payload = {
        "name": "momentum",
        "mode": "backtesting",
        "run_ts": "2026-06-22_194053",
        "backtesting_start": "2026-01-01T09:30:00-05:00",
        "backtesting_end": "2026-06-01T16:00:00-04:00",
        "budget": 10000.0,
        "risk_free_rate": 0.03,
        "backtesting_data_sources": "yahoo",
        "backtest_time_seconds": 12.5,
        "timestep": "day",
        "sleeptime": "1D",
        "commission": 0.0,
        "slippage": 0.0,
        "warmup_trading_days": 20,
        "benchmark_symbol": "QQQ",
        "framework_version": "0.1.0",
        "parameters": {"lookback": 20},
    }
    payload.update(overrides)
    return payload


def test_get_benchmark_symbol_reads_the_flat_settings_key(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload())

    assert get_benchmark_symbol(_ref(run_dir)) == "QQQ"


def test_get_benchmark_symbol_falls_back_to_spy_when_settings_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert get_benchmark_symbol(_ref(run_dir)) == "SPY"


def test_load_settings_reads_settings_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload())

    settings = load_settings(_ref(run_dir))

    assert settings is not None
    assert settings.budget == 10000.0
    assert settings.backtesting_data_sources == "yahoo"


def test_load_parameters_reports_the_framework_version(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(framework_version="0.1.0"))

    rows = load_parameters(_ref(run_dir))

    assert ("Run", "Framework version", "0.1.0") in rows


def test_load_parameters_reports_the_strategys_own_parameters(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(parameters={"lookback": 20, "symbol": "AAPL"}))

    rows = load_parameters(_ref(run_dir))

    assert ("Parameters", "lookback", "20") in rows
    assert ("Parameters", "symbol", "AAPL") in rows


def test_load_parameters_json_encodes_nested_parameter_values(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(parameters={"weights": [1, 2, 3]}))

    rows = load_parameters(_ref(run_dir))

    assert ("Parameters", "weights", "[1, 2, 3]") in rows


def test_load_parameters_returns_only_run_rows_when_settings_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_parameters(_ref(run_dir)) == []


def test_load_parameters_returns_empty_list_on_corrupt_settings_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    (run_dir / "settings.json").write_text("{not valid json", encoding="utf-8")

    assert load_parameters(_ref(run_dir)) == []


def test_load_settings_returns_none_on_corrupt_settings_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    (run_dir / "settings.json").write_text("{not valid json", encoding="utf-8")

    assert load_settings(_ref(run_dir)) is None


def test_save_decision_writes_the_leaf_field_and_preserves_other_keys(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload())

    save_decision(_ref(run_dir), "validated")

    settings = load_settings(_ref(run_dir))
    assert settings is not None
    assert settings.dashboard_decision == "validated"
    assert settings.budget == 10000.0  # untouched


def test_save_decision_overwrites_a_previous_decision(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload())

    save_decision(_ref(run_dir), "study")
    save_decision(_ref(run_dir), "discarded")

    assert load_settings(_ref(run_dir)).dashboard_decision == "discarded"


def test_save_regime_writes_the_leaf_field_and_preserves_other_keys(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload())
    save_decision(_ref(run_dir), "study")

    save_regime(_ref(run_dir), ["Bullish", "All-Weather"])

    settings = load_settings(_ref(run_dir))
    assert settings is not None
    assert settings.dashboard_regime == ["Bullish", "All-Weather"]
    assert settings.dashboard_decision == "study"  # the sibling picker is untouched
    assert settings.budget == 10000.0


def test_save_regime_overwrites_and_clears(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload())

    save_regime(_ref(run_dir), ["Bullish"])
    save_regime(_ref(run_dir), ["Bearish", "Neutral"])
    assert load_settings(_ref(run_dir)).dashboard_regime == ["Bearish", "Neutral"]

    save_regime(_ref(run_dir), [])
    assert load_settings(_ref(run_dir)).dashboard_regime == []


NOW = datetime(2026, 1, 5, 21, tzinfo=UTC)
LATER = datetime(2026, 1, 6, 21, tzinfo=UTC)


def _equity_samples() -> list[EquitySample]:
    return [
        EquitySample(time=NOW, portfolio_value=Decimal(10000), cash=Decimal(10000), positions_value=Decimal(0)),
        EquitySample(time=LATER, portfolio_value=Decimal(10500), cash=Decimal(500), positions_value=Decimal(10000)),
    ]


def test_load_portfolio_breakdown_reads_equity_parquet(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_equity(run_dir, _equity_samples())

    breakdown = load_portfolio_breakdown(_ref(run_dir))

    assert breakdown is not None
    assert breakdown["portfolio_value"] == [10000.0, 10500.0]
    assert breakdown["cash"] == [10000.0, 500.0]
    assert breakdown["assets"] == [0.0, 10000.0]
    assert len(breakdown["dates"]) == 2


def test_load_portfolio_breakdown_returns_none_when_equity_parquet_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_portfolio_breakdown(_ref(run_dir)) is None


def test_load_equity_curve_reads_portfolio_value_from_equity_parquet(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_equity(run_dir, _equity_samples())

    curve = load_equity_curve(_ref(run_dir))

    assert curve == [
        {"date": NOW.strftime("%Y-%m-%d"), "value": 10000.0},
        {"date": LATER.strftime("%Y-%m-%d"), "value": 10500.0},
    ]


def test_load_run_populates_equity_curve_from_equity_parquet(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_equity(run_dir, _equity_samples())
    report.write_settings(run_dir, _settings_payload())
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["raw"] = {}
    report.write_metrics(run_dir, metrics)

    run = load_run(_ref(run_dir))

    assert run.settings is not None
    assert run.metrics is not None
    assert len(run.equity_curve) == 2
    assert run.equity_curve[0]["value"] == 10000.0


def test_load_cumulative_returns_uses_the_persisted_benchmark_columns(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    benchmark = {NOW: Decimal("400.0"), LATER: Decimal("404.0")}
    report.write_equity(run_dir, _equity_samples(), benchmark)
    report.write_settings(run_dir, _settings_payload(benchmark_symbol="QQQ"))

    result = load_cumulative_returns(_ref(run_dir))

    assert result is not None
    assert result["benchmark_symbol"] == "QQQ"
    assert result["benchmark"] is not None
    assert result["strategy"][-1] == pytest.approx(0.05, rel=1e-6)  # 10500/10000 - 1
    assert result["benchmark_daily_returns"][-1] == pytest.approx(0.01, rel=1e-6)  # 404/400 - 1


def test_load_cumulative_returns_handles_a_benchmark_less_run_without_network_access(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_equity(run_dir, _equity_samples())  # no benchmark passed
    report.write_settings(run_dir, _settings_payload())

    result = load_cumulative_returns(_ref(run_dir))

    assert result is not None
    assert result["benchmark"] is None
    assert result["strategy"][-1] == pytest.approx(0.05, rel=1e-6)


def test_load_cumulative_returns_returns_none_when_equity_parquet_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_cumulative_returns(_ref(run_dir)) is None


def test_load_yearly_returns_reads_the_precomputed_summary_table(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["raw"] = {
        "summary_tables": {
            "eoy_returns_vs_benchmark": [{"year": 2026, "strategy": 0.1, "benchmark": 0.05, "won": True}],
            "drawdowns": [],
        }
    }
    report.write_metrics(run_dir, metrics)

    yearly = load_yearly_returns(_ref(run_dir))

    assert yearly == [{"year": 2026, "strategy": 0.1, "benchmark": 0.05, "won": True}]


def test_load_yearly_returns_returns_none_when_metrics_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_yearly_returns(_ref(run_dir)) is None


def test_load_trades_curve_reads_trades_parquet(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(backtesting_start=NOW.isoformat(), budget=10000.0))
    ledger = Ledger()
    ledger.record_fill(
        FillRecord(
            time=NOW,
            identifier="abc",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal(10),
            filled_quantity=Decimal(10),
            price=Decimal("100"),
            trade_cost=Decimal("1.0"),
            trade_slippage=Decimal("0.0"),
        )
    )
    report.write_trades(run_dir, ledger)

    curve = load_trades_curve(_ref(run_dir), budget=10000.0)

    assert curve is not None
    assert len(curve["trades"]) == 1
    assert curve["trades"][0]["side"] == "buy"
    assert curve["trades"][0]["symbol"] == "AAPL"


def test_load_trades_curve_returns_none_when_no_fills(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_trades(run_dir, Ledger())
    assert load_trades_curve(_ref(run_dir), budget=10000.0) is None


def test_load_trades_curve_returns_none_when_file_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_trades_curve(_ref(run_dir), budget=10000.0) is None


def _fill(hour: int, minute: int, symbol: str, side: OrderSide, qty: int, price: str, day: int = 5) -> FillRecord:
    # 2026-01-05/06 are winter sessions: market time is UTC-5.
    return FillRecord(
        time=datetime(2026, 1, day, hour + 5, minute, tzinfo=UTC),
        identifier=f"{symbol}{day}{hour}{minute}",
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=Decimal(qty),
        filled_quantity=Decimal(qty),
        price=Decimal(price),
        trade_cost=Decimal(0),
        trade_slippage=Decimal(0),
    )


def _write_fills(run_dir: Path, fills: list[FillRecord]) -> None:
    ledger = Ledger()
    for fill in fills:
        ledger.record_fill(fill)
    report.write_trades(run_dir, ledger)


def test_load_intraday_exposure_reports_the_peak_invested_of_overlapping_positions(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(budget=10000.0))
    _write_fills(
        run_dir,
        [
            _fill(10, 0, "AAA", OrderSide.BUY, 10, "100"),  # 1,000 invested
            _fill(10, 30, "BBB", OrderSide.BUY, 5, "200"),  # 2,000: the peak, two positions
            _fill(11, 0, "AAA", OrderSide.SELL, 10, "110"),
            _fill(12, 0, "BBB", OrderSide.SELL, 5, "190"),
        ],
    )

    assert load_intraday_exposure(_ref(run_dir)) == [
        {"date": "2026-01-05", "peak_invested": 2000.0, "peak_pct": 20.0, "max_positions": 2},
    ]


def test_load_intraday_exposure_measures_a_day_against_the_previous_sessions_close(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(budget=10000.0))
    report.write_equity(
        run_dir,
        [
            EquitySample(time=NOW, portfolio_value=Decimal(8000), cash=Decimal(8000), positions_value=Decimal(0)),
            EquitySample(time=LATER, portfolio_value=Decimal(8000), cash=Decimal(8000), positions_value=Decimal(0)),
        ],
    )
    _write_fills(
        run_dir,
        [
            _fill(10, 0, "AAA", OrderSide.BUY, 10, "100"),
            _fill(10, 5, "AAA", OrderSide.SELL, 10, "100"),
            _fill(10, 0, "AAA", OrderSide.BUY, 20, "100", day=6),  # 2,000 of the 8,000 closing equity of the 5th
            _fill(15, 0, "AAA", OrderSide.SELL, 20, "100", day=6),
        ],
    )

    rows = load_intraday_exposure(_ref(run_dir))

    assert [(r["date"], r["peak_pct"]) for r in rows] == [("2026-01-05", 10.0), ("2026-01-06", 25.0)]


def test_load_intraday_exposure_keeps_a_position_carried_overnight_at_its_cost(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(budget=10000.0))
    _write_fills(
        run_dir,
        [
            _fill(10, 0, "AAA", OrderSide.BUY, 10, "100"),  # held overnight
            _fill(10, 0, "BBB", OrderSide.BUY, 10, "50", day=6),  # 1,000 + 500 the next day
            _fill(11, 0, "AAA", OrderSide.SELL, 5, "120", day=6),  # half of AAA's cost leaves: 500 + 500
        ],
    )

    rows = load_intraday_exposure(_ref(run_dir))

    assert [(r["date"], r["peak_invested"], r["max_positions"]) for r in rows] == [("2026-01-05", 1000.0, 1), ("2026-01-06", 1500.0, 2)]


def test_load_intraday_exposure_shows_a_session_without_trades_as_zero(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(budget=10000.0))
    report.write_equity(run_dir, _equity_samples())  # sessions of the 5th and the 6th
    _write_fills(run_dir, [_fill(10, 0, "AAA", OrderSide.BUY, 10, "100"), _fill(11, 0, "AAA", OrderSide.SELL, 10, "100")])

    rows = load_intraday_exposure(_ref(run_dir))

    assert rows[1] == {"date": "2026-01-06", "peak_invested": 0.0, "peak_pct": 0.0, "max_positions": 0}


def test_load_intraday_exposure_returns_none_without_fills(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_intraday_exposure(_ref(run_dir)) is None
    _write_fills(run_dir, [])
    assert load_intraday_exposure(_ref(run_dir)) is None


# --- agent telemetry ------------------------------------------------------------------

_TRADER = {
    "model": "qwen3-8b",
    "calls": 3,
    "tool_calls": 5,
    "input_tokens": 1234567,
    "output_tokens": 89000,
    "reasoning_tokens": 4500,
    "total_tokens": 1323567,
    "latency_ms_total": 4500.0,
    "latency_ms_avg": 1500.0,
}


def _sections(rows: list[tuple[str, str, str]]) -> list[str]:
    """Section names in order, consecutive duplicates collapsed (the Parameters tab groups by consecutive section)."""
    return [section for i, (section, _, _) in enumerate(rows) if i == 0 or rows[i - 1][0] != section]


def test_load_parameters_shows_one_agents_model_calls_tokens_and_latency(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(agents={"trader": _TRADER}))

    rows = load_parameters(_ref(run_dir))

    assert [row for row in rows if row[0] in {"Model", "Calls", "Tokens", "Latency"}] == [
        ("Model", "Model", "qwen3-8b"),
        ("Calls", "Model calls", "3"),
        ("Calls", "Tool calls", "5"),
        ("Tokens", "Input tokens", "1,234,567"),
        ("Tokens", "Output tokens", "89,000"),
        ("Tokens", "Reasoning tokens", "4,500"),
        ("Tokens", "Total tokens", "1,323,567"),
        ("Latency", "Total latency", "4.5 s"),
        ("Latency", "Avg latency per call", "1.5 s"),
    ]


def test_load_parameters_shows_the_agents_temperature_under_its_model(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(agents={"trader": {**_TRADER, "temperature": 0.3}}))

    rows = load_parameters(_ref(run_dir))

    assert rows[rows.index(("Model", "Model", "qwen3-8b")) + 1] == ("Model", "Temperature", "0.3")


def test_load_parameters_shows_a_none_temperature_as_the_server_default(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(agents={"trader": {**_TRADER, "temperature": None}}))

    assert ("Model", "Temperature", "server default") in load_parameters(_ref(run_dir))


def test_load_parameters_has_no_temperature_row_when_settings_predate_it(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(agents={"trader": _TRADER}))

    assert not any(label == "Temperature" for _, label, _ in load_parameters(_ref(run_dir)))


def test_load_parameters_shows_unreported_tokens_as_a_dash_and_sub_second_latency_in_ms(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    agent = {**_TRADER, "reasoning_tokens": None, "latency_ms_avg": 250.0}
    report.write_settings(run_dir, _settings_payload(agents={"trader": agent}))

    rows = load_parameters(_ref(run_dir))

    assert ("Tokens", "Reasoning tokens", "—") in rows
    assert ("Latency", "Avg latency per call", "250 ms") in rows


def test_load_parameters_keeps_agent_sections_contiguous_between_run_and_parameters(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(agents={"trader": _TRADER}, parameters={"lookback": 20}))

    assert _sections(load_parameters(_ref(run_dir))) == ["Run", "Model", "Calls", "Tokens", "Latency", "Parameters"]


def test_load_parameters_names_the_agent_in_each_section_when_there_are_several(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    researcher = {**_TRADER, "model": "other", "calls": 7}
    report.write_settings(run_dir, _settings_payload(agents={"trader": _TRADER, "researcher": researcher}))

    rows = load_parameters(_ref(run_dir))

    assert ("Calls (trader)", "Model calls", "3") in rows
    assert ("Calls (researcher)", "Model calls", "7") in rows
    assert ("Model (researcher)", "Model", "other") in rows
    assert not any(section == "Calls" for section, _, _ in rows)


def test_load_parameters_without_an_agents_block_has_no_agent_sections(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    report.write_settings(run_dir, _settings_payload(parameters={"lookback": 20}))

    assert _sections(load_parameters(_ref(run_dir))) == ["Run", "Parameters"]


# --- per-call rows from llm_stats.sqlite ----------------------------------------------


def _logs_run_dir(root: Path, strategy: str = "momentum", run: str = "2026-01-05_210000_backtesting") -> Path:
    run_dir = root / "logs" / strategy / "backtesting" / run
    run_dir.mkdir(parents=True)
    return run_dir


def _call(agent: str = "trader", input_tokens: int | None = 100, latency_ms: float = 1500.0, ts: datetime = datetime(2026, 1, 5, 21, tzinfo=UTC)) -> CallRecord:
    return CallRecord(
        ts=ts,
        agent=agent,
        model="qwen3-8b",
        input_tokens=input_tokens,
        output_tokens=20,
        reasoning_tokens=None,
        total_tokens=None if input_tokens is None else input_tokens + 20,
        latency_ms=latency_ms,
        tool_calls=1,
    )


def test_load_agent_calls_returns_this_runs_calls_in_order_from_the_frameworks_own_database(tmp_path: Path) -> None:
    run_dir = _logs_run_dir(tmp_path)
    db = llm_stats_db_path(tmp_path, "momentum", TradingMode.BACKTESTING)
    LLMStatsStore(db, run_id="an_older_run").record(_call(input_tokens=999))
    store = LLMStatsStore(db, run_id=run_dir.name)
    store.record(_call(input_tokens=100, latency_ms=1500.0, ts=datetime(2026, 1, 5, 21, tzinfo=UTC)))
    store.record(_call(agent="researcher", input_tokens=None, latency_ms=250.0, ts=datetime(2026, 1, 6, 21, tzinfo=UTC)))

    df = load_agent_calls(_ref(run_dir))

    assert df is not None
    assert df["agent"].tolist() == ["trader", "researcher"]
    assert df["input_tokens"].tolist()[0] == 100
    assert pd.isna(df["input_tokens"].tolist()[1])
    assert df["latency_ms"].tolist() == [1500.0, 250.0]
    assert df["ts"].tolist() == [pd.Timestamp("2026-01-05T21:00:00+00:00"), pd.Timestamp("2026-01-06T21:00:00+00:00")]


def test_load_agent_calls_is_none_and_creates_nothing_when_the_database_is_missing(tmp_path: Path) -> None:
    run_dir = _logs_run_dir(tmp_path)

    assert load_agent_calls(_ref(run_dir)) is None
    assert not (tmp_path / "memory").exists()


def test_load_agent_calls_is_none_for_a_run_the_database_no_longer_holds(tmp_path: Path) -> None:
    run_dir = _logs_run_dir(tmp_path)
    LLMStatsStore(llm_stats_db_path(tmp_path, "momentum", TradingMode.BACKTESTING), run_id="a_newer_backtest_wiped_this_one").record(_call())

    assert load_agent_calls(_ref(run_dir)) is None


def test_load_agent_calls_is_none_when_the_database_is_corrupt(tmp_path: Path) -> None:
    run_dir = _logs_run_dir(tmp_path)
    db = llm_stats_db_path(tmp_path, "momentum", TradingMode.BACKTESTING)
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not a sqlite database" * 50)

    assert load_agent_calls(_ref(run_dir)) is None


def test_load_indicator_lines_groups_series_by_pane_and_drops_nothing_else(tmp_path: Path) -> None:
    from trading_agent_framework.backtesting.ledger import IndicatorLine

    run_dir = _run_dir(tmp_path)
    ledger = Ledger()
    for minutes, value in ((5, "55"), (0, "50")):  # recorded out of order on purpose
        ledger.record_line(IndicatorLine(NOW + timedelta(minutes=minutes), "Fast", Decimal(value), "#a78bfa", "dashed", "Averages"))
    ledger.record_line(IndicatorLine(NOW, "Slow", Decimal("22"), "#f59e0b", "solid", "Averages"))
    ledger.record_line(IndicatorLine(NOW, "Regime", Decimal("-1"), "#d1d4dc", "solid", "Regime"))
    ledger.record_line(IndicatorLine(NOW, "SMA", Decimal("9"), None, "solid", "default_plot"))
    report.write_indicators(run_dir, ledger)

    panes = load_indicator_lines(_ref(run_dir))

    assert panes is not None and list(panes) == ["Averages", "Regime", "Indicators"]
    fast = next(line for line in panes["Averages"] if line["name"] == "Fast")
    assert fast["values"] == [50.0, 55.0] and fast["dash"] == "dash" and fast["color"] == "#a78bfa"
    assert panes["Regime"][0]["values"] == [-1.0]  # the regime line reaches the chart as its own pane
    assert panes["Indicators"][0]["color"]  # a line without a colour gets a palette one


def test_load_indicator_lines_returns_none_without_a_file_or_lines(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_indicator_lines(_ref(run_dir)) is None
    report.write_indicators(run_dir, Ledger())
    assert load_indicator_lines(_ref(run_dir)) is None
