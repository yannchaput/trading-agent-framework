"""Serialises a finished backtest run to disk: settings.json, metrics.json, and the
three parquet time series (equity/trades/indicators). description.json is never
written here -- it is the dashboard's own file, created only when a user adds a
description through the dashboard UI.

This is the codebase's second half of the third float boundary (design spec,
section 7.2, alongside metrics.py): every value here is converted from the ledger's
Decimal to float only at the point it's about to leave the process.

Fixed filenames, no timestamp/glob naming -- the dashboard expects exactly
settings.json / metrics.json / equity.parquet / trades.parquet / indicators.parquet
inside a run directory.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.backtesting.ledger import Ledger
from trading_agent_framework.utils.errors import BacktestDataError

# Mirrors the real dashboard's MetricSet Pydantic model field names (design spec,
# section 6.4/6.5; kept in sync with tests/backtesting/dashboard_contract.py's
# METRIC_SET_FIELDS, minus "raw" which is handled separately below). MetricSet has
# no extra="allow", so write_metrics must emit exactly these fields -- no more, no
# less -- or the real dashboard's model_validate() would silently drop stray keys
# (worse than raising) or fail on missing required ones.
_METRIC_SET_FIELDS = frozenset({
    "total_return_strategy", "total_return_benchmark", "cagr_strategy", "cagr_benchmark",
    "sharpe_strategy", "sharpe_benchmark", "sortino_strategy", "sortino_benchmark",
    "calmar_strategy", "calmar_benchmark", "omega_strategy", "omega_benchmark",
    "max_drawdown_strategy", "max_drawdown_benchmark", "volatility_strategy",
    "volatility_benchmark", "beta", "alpha", "correlation", "treynor_ratio",
    "information_ratio_strategy", "information_ratio_benchmark", "r_squared_strategy",
    "r_squared_benchmark", "skew_strategy", "skew_benchmark", "kurtosis_strategy",
    "kurtosis_benchmark", "win_days_pct_strategy", "win_days_pct_benchmark",
    "win_month_pct_strategy", "win_month_pct_benchmark", "longest_dd_days_strategy",
    "longest_dd_days_benchmark", "avg_drawdown_strategy", "avg_drawdown_benchmark",
    "recovery_factor_strategy", "recovery_factor_benchmark",
})


def _float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _sanitize_json(value: Any) -> Any:
    """Recursively replaces non-finite floats (inf/-inf/nan) with None.

    compute_metrics (metrics.py) can legitimately produce inf/nan -- e.g. a Sharpe
    or Calmar ratio computed against a zero-volatility returns series. Those values
    are numerically correct but are not valid JSON: Python's json.dumps emits the
    non-standard Infinity/-Infinity/NaN tokens for them by default, which strict
    JSON parsers (plausibly including the dashboard's) reject. None -> JSON `null`
    is the most common convention for "this metric couldn't be computed", and is
    always valid JSON, so that's the sentinel used here rather than a large finite
    number (which would misleadingly look like a real value on the dashboard).
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # allow_nan=False is a defensive backstop, not the sanitization mechanism
        # itself: callers must already have replaced inf/nan (see _sanitize_json)
        # before this point; this just turns "we missed one" into a loud, immediate
        # ValueError instead of a silent non-standard JSON token in the output file.
        path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
    except OSError as exc:
        raise BacktestDataError(f"failed to write {path}: {exc}") from exc
    return path


def write_settings(run_dir: Path, settings: dict[str, Any]) -> Path:
    return _write_json(run_dir / "settings.json", settings)


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    """Reshapes `metrics` (compute_metrics's output) into exactly the dashboard's
    MetricSet field names plus `raw`. Fields MetricSet expects but that `metrics`
    doesn't contain (e.g. all the *_benchmark/relative fields on a benchmark-less
    run) are written as `null`. Any stray top-level key that isn't a known MetricSet
    field (e.g. a metric added to compute_metrics before this module is updated to
    match) is folded into `raw` instead of being dropped or leaking as an unknown
    top-level key.
    """
    known = {field: metrics.get(field) for field in _METRIC_SET_FIELDS}
    raw = dict(metrics.get("raw") or {})
    extra = {k: v for k, v in metrics.items() if k not in _METRIC_SET_FIELDS and k != "raw"}
    if extra:
        raw = {**raw, "unmapped_metrics": extra}
    payload = _sanitize_json({**known, "raw": raw})
    return _write_json(run_dir / "metrics.json", payload)


def write_equity(run_dir: Path, ledger: Ledger, benchmark: dict[datetime, Decimal] | None = None) -> Path:
    import pandas as pd

    rows = [
        {
            "datetime": sample.time,
            "portfolio_value": _float(sample.portfolio_value),
            "cash": _float(sample.cash),
            "positions_value": _float(sample.positions_value),
            "benchmark_close": _float(benchmark.get(sample.time)) if benchmark else None,
        }
        for sample in ledger.equity
    ]
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    df["return"] = df["portfolio_value"].pct_change()
    df["benchmark_return"] = df["benchmark_close"].pct_change() if benchmark else pd.Series(dtype="float64")
    return _write_parquet(run_dir / "equity.parquet", df)


def write_trades(run_dir: Path, ledger: Ledger) -> Path:
    import pandas as pd

    rows = [
        {
            "time": f.time, "symbol": f.symbol, "side": f.side.value, "status": f.status,
            "order_type": f.order_type.value, "quantity": _float(f.quantity),
            "filled_quantity": _float(f.filled_quantity), "price": _float(f.price),
            "trade_cost": _float(f.trade_cost), "trade_slippage": _float(f.trade_slippage),
            "identifier": f.identifier, "event_kind": f.event_kind,
        }
        for f in ledger.fills
    ]
    df = pd.DataFrame(rows)
    return _write_parquet(run_dir / "trades.parquet", df)


def write_indicators(run_dir: Path, ledger: Ledger) -> Path:
    import pandas as pd

    rows = [
        {
            "datetime": line.time, "name": line.name, "value": _float(line.value),
            "color": line.color, "style": line.style, "plot_name": line.plot_name,
        }
        for line in ledger.lines
    ]
    df = pd.DataFrame(rows)
    return _write_parquet(run_dir / "indicators.parquet", df)


def _write_parquet(path: Path, df: Any) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except OSError as exc:
        raise BacktestDataError(f"failed to write {path}: {exc}") from exc
    return path
