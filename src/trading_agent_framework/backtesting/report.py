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
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.backtesting.ledger import EquitySample, Ledger
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
        # write_settings has no sanitize-first step of its own (unlike write_metrics),
        # so a non-finite float anywhere in a caller-supplied dict (e.g. "parameters")
        # hits this ValueError -- caught here so it comes out as BacktestDataError
        # like every other failure in this module, not a raw stdlib exception.
        path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
    except (OSError, ValueError) as exc:
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


# Explicit column lists for the three row-list-of-dicts -> DataFrame conversions
# below. Passing `columns=` to the DataFrame constructor (rather than letting it
# infer columns from the dicts) keeps the output schema identical whether `rows`
# is empty or not -- an empty `ledger.equity`/`fills`/`lines` (a genuinely
# completed but trade-less/data-less backtest run) must still produce a
# well-formed, zero-row parquet file with the dashboard's expected columns,
# not a KeyError (write_equity's set_index) or a columns-less frame.
_EQUITY_COLUMNS = ("datetime", "portfolio_value", "cash", "positions_value", "benchmark_close")
_TRADE_COLUMNS = (
    "time", "symbol", "side", "status", "order_type", "quantity", "filled_quantity",
    "price", "trade_cost", "trade_slippage", "identifier", "event_kind",
)
_INDICATOR_COLUMNS = ("datetime", "name", "value", "color", "style", "plot_name")


def write_equity(
    run_dir: Path,
    equity: Sequence[EquitySample],
    benchmark: dict[datetime, Decimal] | None = None,
) -> Path:
    """Write equity.parquet from an ALREADY SESSION-REDUCED equity series.

    `equity` is NOT `ledger.equity` -- it is `runner._session_equity_samples(...)`'s
    output: exactly one sample per trading session, each re-stamped at that session's
    own `close`. Two things depend on that, and both were broken while this function
    took the raw ledger (Critical whole-branch review finding):

    1. **Cadence.** `BacktestBroker.on_advance` samples equity on every clock advance
       (pre-open/open/pre-close/close under `Strategy`'s default timing), so the raw
       ledger is oversampled several-to-one against a session. `metrics.json`'s
       `portfolio_returns` was fixed to run at session cadence (see
       `runner._session_equity_series`); this file's `return` column has to agree with
       it, or the dashboard plots a different return series than the metrics describe.
    2. **The benchmark join.** `benchmark` is keyed by bar-CLOSE timestamp and is
       naturally one row per session -- for a daily backtest, exactly the session-close
       timestamps `_session_equity_samples` stamps its rows with. Joining on those is
       therefore a join on *session identity*. Against raw ledger samples the same
       lookup matched only the handful that happened to land exactly on a 16:00 close
       (and none at all once `minutes_after_closing` moved them), leaving
       `benchmark_close` null on most rows and `benchmark_return` -- a `pct_change`
       across those holes -- null on ALL of them.
    """
    import pandas as pd

    rows = [
        {
            "datetime": sample.time,
            "portfolio_value": _float(sample.portfolio_value),
            "cash": _float(sample.cash),
            "positions_value": _float(sample.positions_value),
            "benchmark_close": _float(benchmark.get(sample.time)) if benchmark else None,
        }
        for sample in equity
    ]
    df = pd.DataFrame(rows, columns=_EQUITY_COLUMNS).set_index("datetime").sort_index()
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
    df = pd.DataFrame(rows, columns=_TRADE_COLUMNS)
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
    df = pd.DataFrame(rows, columns=_INDICATOR_COLUMNS)
    return _write_parquet(run_dir / "indicators.parquet", df)


def _write_parquet(path: Path, df: Any) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except OSError as exc:
        raise BacktestDataError(f"failed to write {path}: {exc}") from exc
    return path
