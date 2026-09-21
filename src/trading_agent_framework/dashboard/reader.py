"""Read backtesting run files into model objects."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from trading_agent_framework.agents.stats_store import llm_stats_db_path
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.dashboard.models import MetricSet, Run, RunRef, Settings


def load_description(ref: RunRef) -> str | None:
    """Load the description from ``description.json`` in the run directory.

    Returns the description string, or None if the file is missing or
    unparseable.
    """
    path = os.path.join(ref.path, "description.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        desc = data.get("description")
        return desc if isinstance(desc, str) else None
    except (json.JSONDecodeError, OSError):
        return None


def save_description(ref: RunRef, description: str) -> None:
    """Write the description to ``description.json`` in the run directory.

    Creates the file if it doesn't exist; overwrites it if it does.
    """
    path = os.path.join(ref.path, "description.json")
    os.makedirs(ref.path, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"description": description}, f, indent=2)


def get_benchmark_symbol(ref: RunRef) -> str:
    """Extract the benchmark ticker from settings.json's flat `benchmark_symbol` field.

    Falls back to "SPY" when the settings file is missing or the key is absent/empty.
    This is the single source of truth for the benchmark ticker shown throughout the
    dashboard.
    """
    path = os.path.join(ref.path, "settings.json")
    if not os.path.isfile(path):
        return "SPY"
    try:
        with open(path) as f:
            data = json.load(f)
        symbol = data.get("benchmark_symbol", "")
        return symbol if symbol else "SPY"
    except (OSError, json.JSONDecodeError):
        return "SPY"


def load_settings(ref: RunRef) -> Settings | None:
    """Load backtesting settings from settings.json."""
    path = os.path.join(ref.path, "settings.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return Settings.model_validate(data)


def load_parameters(ref: RunRef) -> list[tuple[str, str, str]]:
    """Load parameters from settings.json for display in the Parameters tab.

    Returns a list of (section, parameter, value) tuples: "Run" for backtest
    configuration (budget, dates, data source, ...), and "Parameters" for each
    key in settings.json's `parameters` field -- this project's
    `strategy.parameters`, an arbitrary flat mapping the strategy author
    defines (see backtesting/runner.py's write_settings call). Values are
    stringified for display; dicts/lists are JSON-encoded.
    """
    path = os.path.join(ref.path, "settings.json")
    if not os.path.isfile(path):
        return []

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    rows: list[tuple[str, str, str]] = []

    # ── Run config ──
    rows.append(("Run", "Strategy", data.get("name", "")))
    rows.append(("Run", "Data source", data.get("backtesting_data_sources", "")))
    rows.append(("Run", "Framework version", data.get("framework_version", "")))
    budget = data.get("budget", 0)
    rows.append(("Run", "Budget", f"${budget:,.0f}"))
    rf = data.get("risk_free_rate", 0)
    rows.append(("Run", "Risk-free rate", f"{rf * 100:.3f}%"))
    start = data.get("backtesting_start", "")
    end = data.get("backtesting_end", "")
    if start:
        rows.append(("Run", "Backtesting start", start))
    if end:
        rows.append(("Run", "Backtesting end", end))

    # ── Agent telemetry (settings.json's `agents` block, written when telemetry was on) ──
    rows.extend(_agent_rows(data.get("agents") or {}))

    # ── Strategy parameters ──
    strat_params: dict[str, Any] = data.get("parameters") or {}
    for key in sorted(strat_params):
        value = strat_params[key]
        value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        rows.append(("Parameters", key, value_str))

    return rows


def _agent_rows(agents: dict[str, dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Model / Calls / Tokens / Latency rows per agent. Section names carry the agent's name when
    there are several, so each agent's rows stay one contiguous group in the Parameters tab."""
    rows: list[tuple[str, str, str]] = []
    for name, agent in agents.items():
        suffix = f" ({name})" if len(agents) > 1 else ""

        def add(section: str, label: str, value: str, suffix: str = suffix) -> None:
            rows.append((section + suffix, label, value))

        if agent.get("model") is not None:
            add("Model", "Model", str(agent["model"]))
        add("Calls", "Model calls", _fmt_int(agent.get("calls")))
        add("Calls", "Tool calls", _fmt_int(agent.get("tool_calls")))
        add("Tokens", "Input tokens", _fmt_int(agent.get("input_tokens")))
        add("Tokens", "Output tokens", _fmt_int(agent.get("output_tokens")))
        add("Tokens", "Reasoning tokens", _fmt_int(agent.get("reasoning_tokens")))
        add("Tokens", "Total tokens", _fmt_int(agent.get("total_tokens")))
        add("Latency", "Total latency", _fmt_ms(agent.get("latency_ms_total")))
        add("Latency", "Avg latency per call", _fmt_ms(agent.get("latency_ms_avg")))
    return rows


def _fmt_int(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "—"
    ms = float(value)
    return f"{ms / 1000:,.1f} s" if ms >= 1000 else f"{ms:,.0f} ms"


_AGENT_CALL_COLUMNS = "ts, agent, model, input_tokens, output_tokens, reasoning_tokens, total_tokens, latency_ms, tool_calls"


def load_agent_calls(ref: RunRef) -> pd.DataFrame | None:
    """This run's model calls (one row each, oldest first) from `llm_stats.sqlite`, or None.

    The database lives in the project's `memory/<strategy>/<mode>/` -- three levels above the run
    directory's `logs/` -- and is opened read-only so the dashboard never creates it. None means "no
    per-call data": telemetry was off, the file was deleted, a newer backtest wiped this run's rows, or
    the file is unreadable. The per-agent totals in settings.json are unaffected.
    """
    root = Path(ref.path).resolve().parents[3]
    db = llm_stats_db_path(root, ref.strategy_name, TradingMode(ref.mode))
    if not db.is_file():
        return None
    run_id = os.path.basename(os.path.normpath(ref.path))
    try:
        conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
        try:
            df = pd.read_sql_query(f"SELECT {_AGENT_CALL_COLUMNS} FROM llm_calls WHERE run_id = ? ORDER BY id", conn, params=(run_id,))
        finally:
            conn.close()
    except (sqlite3.Error, pd.errors.DatabaseError):
        return None
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_metrics(ref: RunRef) -> MetricSet | None:
    """Load performance metrics from metrics.json.

    metrics.json is written by backtesting.report.write_metrics, keyed exactly by
    MetricSet's field names. A benchmark-less run writes JSON `null` for every
    *_benchmark/relative field (write_metrics's docstring) -- MetricSet's fields are
    plain (non-Optional) floats, so those nulls are normalized to 0.0 before
    validation, matching the "couldn't be computed" convention used everywhere else
    in this dashboard.
    """
    path = os.path.join(ref.path, "metrics.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    raw = data.get("raw") or {}
    scalars = {k: (v if v is not None else 0.0) for k, v in data.items() if k != "raw"}
    return MetricSet.model_validate({**scalars, "raw": raw})


def load_portfolio_breakdown(ref: RunRef) -> dict[str, Any] | None:
    """Load daily portfolio decomposition (total value, cash, assets) from equity.parquet.

    `positions_value` is read directly rather than derived as `portfolio_value - cash`:
    backtesting.report.write_equity already computes it exactly from the ledger, so
    subtracting would only reintroduce float rounding drift for no benefit.
    """
    path = os.path.join(ref.path, "equity.parquet")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or "portfolio_value" not in df.columns:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "portfolio_value": [round(float(v), 2) for v in df["portfolio_value"].to_numpy()],
        "cash": [round(float(v), 2) for v in df["cash"].to_numpy()],
        "assets": [round(float(v), 2) for v in df["positions_value"].to_numpy()],
    }


def load_equity_curve(ref: RunRef) -> list[dict[str, Any]]:
    """Load the portfolio-value curve from equity.parquet (one row per trading session)."""
    path = os.path.join(ref.path, "equity.parquet")
    if not os.path.isfile(path):
        return []
    try:
        df = pd.read_parquet(path)
    except Exception:
        return []
    if df.empty or "portfolio_value" not in df.columns:
        return []
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in df["portfolio_value"].sort_index().items()]


def _compound_monthly_returns(daily_returns: pd.Series) -> list[float] | None:
    """Compound daily returns into monthly returns, matching quantstats behavior.

    Quantstats uses: returns.fillna(0).resample('ME').apply(comp).resample('ME').last()
    where comp = (1+x).prod()-1.
    """
    if daily_returns.empty:
        return None
    monthly = daily_returns.fillna(0.0).resample("ME").apply(lambda x: (1 + x).prod() - 1)
    # quantstats applies .resample('ME').last() again after the apply, which is a no-op
    # in newer pandas but included for compatibility.
    monthly = monthly.resample("ME").last().dropna()
    if monthly.empty:
        return None
    return [round(float(v) * 100, 4) for v in monthly.to_numpy()]


def load_cumulative_returns(ref: RunRef) -> dict[str, Any] | None:
    """Build cumulative returns for strategy and benchmark from equity.parquet.

    Both series come from backtesting.report.write_equity's own `return`/
    `benchmark_return`/`benchmark_close` columns -- the exact series
    backtesting.metrics.compute_metrics used to derive this run's own Sharpe/Alpha/
    Beta -- rather than a live re-fetch that could silently disagree with them (and
    that would violate this project's "tests never touch the network" rule).
    """
    path = os.path.join(ref.path, "equity.parquet")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or "return" not in df.columns:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    daily_ret = df["return"].fillna(0.0)
    strategy_cum = (1 + daily_ret).cumprod() - 1
    benchmark_symbol = get_benchmark_symbol(ref)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    has_benchmark = "benchmark_close" in df.columns and df["benchmark_close"].notna().any()
    if not has_benchmark:
        return {
            "dates": dates,
            "strategy": [round(float(v), 6) for v in strategy_cum.to_numpy()],
            "benchmark": None,
            "benchmark_symbol": benchmark_symbol,
            "benchmark_source": "unavailable (no benchmark recorded for this run)",
            "benchmark_daily_close": None,
            "strategy_monthly": _compound_monthly_returns(daily_ret),
            "benchmark_monthly": None,
            "strategy_daily_returns": [round(float(v), 6) for v in daily_ret.to_numpy()],
            "benchmark_daily_returns": None,
        }

    bm_daily_ret = df["benchmark_return"].fillna(0.0)
    bm_cum = (1 + bm_daily_ret).cumprod() - 1

    return {
        "dates": dates,
        "strategy": [round(float(v), 6) for v in strategy_cum.to_numpy()],
        "benchmark": [round(float(v), 6) for v in bm_cum.to_numpy()],
        "benchmark_symbol": benchmark_symbol,
        "benchmark_source": "equity.parquet (recorded at backtest time)",
        "benchmark_daily_close": [round(float(v), 6) if pd.notna(v) else None for v in df["benchmark_close"].to_numpy()],
        "strategy_monthly": _compound_monthly_returns(daily_ret),
        "benchmark_monthly": _compound_monthly_returns(bm_daily_ret),
        "strategy_daily_returns": [round(float(v), 6) for v in daily_ret.to_numpy()],
        "benchmark_daily_returns": [round(float(v), 6) for v in bm_daily_ret.to_numpy()],
    }


def load_yearly_returns(ref: RunRef) -> list[dict[str, Any]] | None:
    """Yearly strategy/benchmark returns, read directly from metrics.json's
    `raw.summary_tables.eoy_returns_vs_benchmark` -- computed once, in
    backtesting.metrics.compute_metrics, from the exact same returns series used for
    every other headline metric on this page. No independent recomputation here.
    """
    metrics = load_metrics(ref)
    if metrics is None:
        return None
    table = metrics.raw.get("summary_tables", {}).get("eoy_returns_vs_benchmark")
    return table or None


def load_trades_curve(ref: RunRef, budget: float) -> dict[str, Any] | None:
    """Load filled trades from ``trades.parquet`` and reconstruct the
    intra-trade portfolio value curve with buy/sell markers.

    Returns a dict with two keys:

    - ``values``: list of ``{time, portfolio_value}`` dicts — the cumulative
      portfolio value curve, including the starting point (budget at
      backtesting_start).
    - ``trades``: list of ``{time, side, symbol, qty, price, cost,
      portfolio_value}`` dicts — individual filled trades for marker
      overlay.

    Returns None when the trades file is missing or contains no fills.
    """
    trades_path = os.path.join(ref.path, "trades.parquet")
    if not os.path.isfile(trades_path):
        return None

    try:
        df = pd.read_parquet(trades_path)
    except Exception:
        return None

    if df.empty or "status" not in df.columns:
        return None

    # Only filled trades carry price/qty
    fills = df[df["status"] == "fill"].copy()
    if fills.empty:
        return None

    if "time" in fills.columns:
        fills["time"] = pd.to_datetime(fills["time"], utc=True)
        fills = fills.sort_values("time")

    # Fetch backtesting start for the initial anchor point
    settings = load_settings(ref)
    start_dt: pd.Timestamp | None = None
    if settings and settings.backtesting_start:
        start_dt = pd.Timestamp(settings.backtesting_start)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize("UTC")
        else:
            start_dt = start_dt.tz_convert("UTC")

    # Reconstruct cumulative portfolio value from fills
    cash = budget
    positions: dict[str, dict[str, Any]] = {}
    values: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    # Anchor: starting point
    if start_dt is not None:
        values.append({"time": start_dt, "portfolio_value": budget})

    for _, row in fills.iterrows():
        t = row["time"]
        side = str(row.get("side", "buy")).lower()
        qty = float(row.get("filled_quantity", 0) or 0)
        price = float(row.get("price", 0) or 0)
        cost = float(row.get("trade_cost", 0) or 0)
        symbol = str(row.get("symbol", ""))

        if side == "buy":
            cash -= qty * price + cost
            positions.setdefault(symbol, {"qty": 0.0, "mark_price": price})
            positions[symbol]["qty"] += qty
            positions[symbol]["mark_price"] = price
        else:  # sell
            cash += qty * price - cost
            positions.setdefault(symbol, {"qty": 0.0, "mark_price": price})
            positions[symbol]["qty"] -= qty
            positions[symbol]["mark_price"] = price
            if abs(positions[symbol]["qty"]) < 1e-10:
                del positions[symbol]

        pos_val = sum(p["qty"] * p["mark_price"] for p in positions.values())
        pv = cash + pos_val

        values.append({"time": t, "portfolio_value": round(float(pv), 2)})
        trades.append(
            {
                "time": t,
                "side": side,
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "cost": cost,
                "portfolio_value": round(float(pv), 2),
            }
        )

    return {"values": values, "trades": trades}


def load_run(ref: RunRef) -> Run:
    """Load all data for a run: settings, metrics, and the equity curve (from equity.parquet)."""
    settings = load_settings(ref)
    metrics = load_metrics(ref)
    equity = load_equity_curve(ref)
    return Run(ref=ref, settings=settings, metrics=metrics, equity_curve=equity)
