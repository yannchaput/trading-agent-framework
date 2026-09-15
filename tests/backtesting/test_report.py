from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from tests.backtesting.dashboard_contract import METRIC_SET_FIELDS, REQUIRED_SETTINGS_FIELDS

from trading_agent_framework.backtesting import report
from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, IndicatorLine, Ledger
from trading_agent_framework.entities.enums import OrderSide, OrderType

NOW = datetime(2026, 1, 5, 16, tzinfo=UTC)
LATER = datetime(2026, 1, 6, 16, tzinfo=UTC)


def _session_equity() -> list[EquitySample]:
    """What `runner._session_equity_samples` hands `write_equity`: one sample per
    trading session, each stamped at that session's own close (NOT raw ledger samples
    -- see `report.write_equity`'s docstring)."""
    return [
        EquitySample(
            time=NOW, portfolio_value=Decimal(10000), cash=Decimal(10000), positions_value=Decimal(0)
        ),
        EquitySample(
            time=LATER, portfolio_value=Decimal(10500), cash=Decimal(500), positions_value=Decimal(10000)
        ),
    ]


def _ledger() -> Ledger:
    ledger = Ledger()
    for sample in _session_equity():
        ledger.record_equity(sample)
    ledger.record_fill(FillRecord(
        time=LATER, identifier="abc", symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal(10), filled_quantity=Decimal(10), price=Decimal("1000"),
        trade_cost=Decimal("1.0"), trade_slippage=Decimal("0.5"),
    ))
    ledger.record_line(IndicatorLine(
        time=NOW, name="sma_200", value=Decimal("148.5"), color=None, style="solid", plot_name="default_plot"
    ))
    return ledger


def test_write_settings_round_trips_through_json(tmp_path: Path) -> None:
    settings = {
        "name": "momentum", "backtesting_start": NOW.isoformat(), "backtesting_end": LATER.isoformat(),
        "budget": 10000.0, "risk_free_rate": 0.03, "backtesting_data_sources": "yahoo",
        "backtest_time_seconds": 1.5, "parameters": {"lookback": 20},
    }
    path = report.write_settings(tmp_path, settings)
    assert path == tmp_path / "settings.json"
    loaded = json.loads(path.read_text())
    assert REQUIRED_SETTINGS_FIELDS <= loaded.keys()
    assert loaded["name"] == "momentum"


def test_write_metrics_writes_exactly_the_metric_set_field_names(tmp_path: Path) -> None:
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["raw"] = {"summary_tables": {"eoy_returns_vs_benchmark": [], "drawdowns": []}}

    path = report.write_metrics(tmp_path, metrics)
    assert path == tmp_path / "metrics.json"
    loaded = json.loads(path.read_text())

    # metrics.json must contain ONLY MetricSet field names -- MetricSet (unlike
    # Settings) has no extra="allow", so a stray key would be silently dropped by
    # the real dashboard's model_validate() rather than raising, which is worse.
    assert set(loaded.keys()) == METRIC_SET_FIELDS


def test_write_metrics_folds_stray_keys_into_raw_instead_of_dropping_or_leaking_them(tmp_path: Path) -> None:
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["raw"] = {"summary_tables": {"eoy_returns_vs_benchmark": [], "drawdowns": []}}
    metrics["some_future_metric_not_yet_in_metric_set"] = 1.23

    path = report.write_metrics(tmp_path, metrics)
    loaded = json.loads(path.read_text())

    assert set(loaded.keys()) == METRIC_SET_FIELDS
    assert "some_future_metric_not_yet_in_metric_set" not in loaded
    # the stray value must still be recoverable somewhere under raw, not silently dropped.
    assert 1.23 in _flatten_values(loaded["raw"])


def test_write_metrics_fills_missing_metric_set_fields_with_none(tmp_path: Path) -> None:
    # e.g. a benchmark-less run: compute_metrics only returns the strategy-side fields.
    metrics = {"total_return_strategy": 0.1, "raw": {}}

    path = report.write_metrics(tmp_path, metrics)
    loaded = json.loads(path.read_text())

    assert set(loaded.keys()) == METRIC_SET_FIELDS
    assert loaded["total_return_strategy"] == 0.1
    assert loaded["sharpe_benchmark"] is None


def test_write_metrics_sanitizes_inf_and_nan_to_valid_json(tmp_path: Path) -> None:
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["sharpe_strategy"] = float("inf")
    metrics["calmar_strategy"] = float("-inf")
    metrics["sortino_strategy"] = float("nan")
    metrics["raw"] = {"nested_nan": float("nan")}

    path = report.write_metrics(tmp_path, metrics)
    raw_text = path.read_text()

    # The raw file text must never contain the non-standard JSON tokens Python's
    # json.dumps emits by default for inf/nan -- many strict JSON parsers reject them.
    assert "Infinity" not in raw_text
    assert "NaN" not in raw_text

    # And it must actually parse with the standard library's strict json.load.
    loaded = json.loads(raw_text)
    assert loaded["sharpe_strategy"] is None
    assert loaded["calmar_strategy"] is None
    assert loaded["sortino_strategy"] is None
    assert loaded["raw"]["nested_nan"] is None


def _flatten_values(value: object) -> list[object]:
    if isinstance(value, dict):
        result: list[object] = []
        for v in value.values():
            result.extend(_flatten_values(v))
        return result
    if isinstance(value, list):
        result = []
        for v in value:
            result.extend(_flatten_values(v))
        return result
    return [value]


def test_write_equity_produces_a_parquet_file_with_the_expected_columns(tmp_path: Path) -> None:
    path = report.write_equity(tmp_path, _session_equity())
    assert path == tmp_path / "equity.parquet"
    df = pd.read_parquet(path)
    for column in ("portfolio_value", "cash", "positions_value", "return"):
        assert column in df.columns
    assert df["portfolio_value"].iloc[0] == 10000.0
    assert df["portfolio_value"].iloc[1] == 10500.0
    assert df["return"].iloc[1] == pytest.approx(0.05, rel=1e-6)


def test_write_equity_joins_the_benchmark_series_by_session_close(tmp_path: Path) -> None:
    """The benchmark series is keyed by bar CLOSE -- for a daily backtest, exactly the
    session-close timestamps `runner._session_equity_samples` stamps its rows with, so
    every row must get a benchmark value, not a lucky subset. See
    `test_runner.py::test_run_backtest_writes_a_session_cadence_equity_parquet_with_a_
    fully_joined_benchmark` for the same assertion against a REAL `run_backtest` output.
    """
    benchmark = {NOW: Decimal("400.0"), LATER: Decimal("404.0")}
    samples = _session_equity()
    path = report.write_equity(tmp_path, samples, benchmark)
    df = pd.read_parquet(path)

    assert list(df["benchmark_close"]) == [400.0, 404.0]
    assert df["benchmark_close"].notna().sum() == len(samples)
    assert df["benchmark_return"].notna().sum() == len(samples) - 1  # first pct_change is NaN
    assert df["benchmark_return"].iloc[1] == pytest.approx(0.01, rel=1e-6)


def test_write_equity_leaves_benchmark_null_when_a_row_is_not_at_a_benchmark_bar_close(
    tmp_path: Path,
) -> None:
    """Gives the null-count assertions above their teeth: the join is exact-timestamp,
    so a row stamped anywhere other than a benchmark bar close silently yields null --
    which is precisely how the pre-fix raw-ledger version (rows at pre-open/open/
    pre-close instants) produced an almost entirely empty `benchmark_close` column and
    an entirely empty `benchmark_return` one.
    """
    off_session = [
        EquitySample(
            time=NOW - timedelta(hours=6),  # a pre-open clock advance, not a session close
            portfolio_value=Decimal(10000), cash=Decimal(10000), positions_value=Decimal(0),
        ),
        *_session_equity(),
    ]
    path = report.write_equity(tmp_path, off_session, {NOW: Decimal("400.0"), LATER: Decimal("404.0")})
    df = pd.read_parquet(path)

    assert df["benchmark_close"].notna().sum() == 2  # the off-session row got nothing
    # ...and the resulting hole makes pct_change NaN across it, too.
    assert df["benchmark_return"].notna().sum() == 1


def test_write_trades_produces_a_parquet_file_with_the_dashboards_expected_columns(tmp_path: Path) -> None:
    path = report.write_trades(tmp_path, _ledger())
    assert path == tmp_path / "trades.parquet"
    df = pd.read_parquet(path)
    for column in (
        "time", "symbol", "side", "status", "order_type", "quantity", "filled_quantity",
        "price", "trade_cost", "trade_slippage", "identifier", "event_kind",
    ):
        assert column in df.columns
    assert df["status"].iloc[0] == "fill"
    assert df["side"].iloc[0] == "buy"


def test_write_indicators_produces_a_parquet_file_with_the_dashboards_expected_columns(tmp_path: Path) -> None:
    path = report.write_indicators(tmp_path, _ledger())
    assert path == tmp_path / "indicators.parquet"
    df = pd.read_parquet(path)
    for column in ("datetime", "name", "value", "color", "style", "plot_name"):
        assert column in df.columns
    assert df["name"].iloc[0] == "sma_200"


def test_write_metrics_wraps_io_failures(tmp_path: Path) -> None:
    from trading_agent_framework.utils.errors import BacktestDataError

    # run_dir is actually a file, so writing settings.json under it must fail with OSError,
    # which write_metrics/write_settings must wrap rather than let escape raw.
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(BacktestDataError):
        report.write_metrics(blocker, {"raw": {}})


def test_write_settings_wraps_non_finite_floats_instead_of_raising_raw_value_error(
    tmp_path: Path,
) -> None:
    from trading_agent_framework.utils.errors import BacktestDataError

    # settings is an arbitrary caller-supplied dict (e.g. "parameters" from strategy
    # config) -- write_settings must not let json.dumps's raw ValueError on a
    # non-finite float escape unwrapped.
    settings = {"name": "x", "risk_free_rate": float("nan")}
    with pytest.raises(BacktestDataError):
        report.write_settings(tmp_path, settings)


def test_write_equity_with_empty_ledger_produces_a_valid_empty_file(tmp_path: Path) -> None:
    path = report.write_equity(tmp_path, [])
    df = pd.read_parquet(path)
    for column in ("portfolio_value", "cash", "positions_value", "return", "benchmark_close"):
        assert column in df.columns
    assert len(df) == 0


def test_write_trades_with_empty_ledger_produces_a_valid_empty_file(tmp_path: Path) -> None:
    path = report.write_trades(tmp_path, Ledger())
    df = pd.read_parquet(path)
    for column in (
        "time", "symbol", "side", "status", "order_type", "quantity", "filled_quantity",
        "price", "trade_cost", "trade_slippage", "identifier", "event_kind",
    ):
        assert column in df.columns
    assert len(df) == 0


def test_write_indicators_with_empty_ledger_produces_a_valid_empty_file(tmp_path: Path) -> None:
    path = report.write_indicators(tmp_path, Ledger())
    df = pd.read_parquet(path)
    for column in ("datetime", "name", "value", "color", "style", "plot_name"):
        assert column in df.columns
    assert len(df) == 0
