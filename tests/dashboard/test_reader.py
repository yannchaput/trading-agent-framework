from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.backtesting.dashboard_contract import METRIC_SET_FIELDS

from trading_agent_framework.backtesting import report
from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, Ledger
from trading_agent_framework.dashboard.models import RunRef
from trading_agent_framework.dashboard.reader import (
    get_benchmark_symbol,
    load_cumulative_returns,
    load_equity_curve,
    load_metrics,
    load_parameters,
    load_portfolio_breakdown,
    load_run,
    load_settings,
    load_trades_curve,
    load_yearly_returns,
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
        "name": "momentum", "mode": "backtesting", "run_ts": "2026-06-22_194053",
        "backtesting_start": "2026-01-01T09:30:00-05:00", "backtesting_end": "2026-06-01T16:00:00-04:00",
        "budget": 10000.0, "risk_free_rate": 0.03, "backtesting_data_sources": "yahoo",
        "backtest_time_seconds": 12.5, "timestep": "day", "sleeptime": "1D",
        "commission": 0.0, "slippage": 0.0, "warmup_trading_days": 20,
        "benchmark_symbol": "QQQ", "framework_version": "0.1.0",
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
    ledger.record_fill(FillRecord(
        time=NOW, identifier="abc", symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal(10), filled_quantity=Decimal(10), price=Decimal("100"),
        trade_cost=Decimal("1.0"), trade_slippage=Decimal("0.0"),
    ))
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
