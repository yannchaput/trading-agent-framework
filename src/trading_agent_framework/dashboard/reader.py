"""Read backtesting run files into model objects."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

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
    with open(path) as f:
        data = json.load(f)
    return Settings.model_validate(data)


def load_parameters(ref: RunRef) -> list[tuple[str, str, str]]:
    """Load parameters from settings.json for display in the Parameters tab.

    Returns a list of (section, parameter, value) tuples where section groups
    related fields (e.g. "Model", "Tokens", "Latency"). Values are formatted
    for readability (tokens with commas, latency in seconds, etc.).
    """
    path = os.path.join(ref.path, "settings.json")
    if not os.path.isfile(path):
        return []

    with open(path) as f:
        data = json.load(f)

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

    # ── Per-agent telemetry ──
    strat_params: dict[str, Any] = data.get("parameters", {})

    # Detect the per-agent prefix (e.g. "agent_liquidity_research_")
    # by finding the first key that matches the known suffix pattern
    agent_prefix = ""
    for key in strat_params:
        if key.startswith("agent_") and key.endswith("_calls") and key != "agent_model_calls":
            # Strip the "_calls" suffix to get the prefix
            agent_prefix = key[: -len("calls")]
            break

    def _p(key: str) -> Any:
        """Look up a parameter under the agent prefix, with fallback."""
        full = agent_prefix + key if agent_prefix else None
        if full and full in strat_params:
            return strat_params[full]
        if key in strat_params:
            return strat_params[key]
        return None

    def _fmt_int(val: Any) -> str:
        if val is None:
            return "—"
        return f"{int(val):,}"

    def _fmt_ms(val: Any) -> str:
        if val is None:
            return "—"
        ms = float(val)
        if ms >= 1000:
            return f"{ms / 1000:,.1f} s"
        return f"{ms:,.0f} ms"

    # Model
    model = _p("model")
    if model is not None:
        rows.append(("Model", "Model", str(model)))

    # Calls
    total_calls = strat_params.get("agent_model_calls")
    agent_calls = _p("calls")
    cache_hits = _p("cache_hits")
    tool_calls = _p("tool_calls")

    if total_calls is not None:
        rows.append(("Calls", "Total model calls", _fmt_int(total_calls)))
    if agent_calls is not None:
        rows.append(("Calls", "Agent calls", _fmt_int(agent_calls)))
    if cache_hits is not None:
        rows.append(("Calls", "Cache hits", _fmt_int(cache_hits)))
    if tool_calls is not None:
        rows.append(("Calls", "Tool calls", _fmt_int(tool_calls)))

    # Tokens
    input_tok = _p("input_tokens")
    output_tok = _p("output_tokens")
    total_tok = _p("total_tokens")
    thinking_tok = _p("thinking_tokens")

    if input_tok is not None:
        rows.append(("Tokens", "Input tokens", _fmt_int(input_tok)))
    if output_tok is not None:
        rows.append(("Tokens", "Output tokens", _fmt_int(output_tok)))
    if total_tok is not None:
        rows.append(("Tokens", "Total tokens", _fmt_int(total_tok)))
    if thinking_tok is not None:
        rows.append(("Tokens", "Thinking tokens", _fmt_int(thinking_tok)))

    # Token breakdown (cache utilisation)
    cached_in = _p("cached_input_tokens")
    uncached_in = _p("uncached_input_tokens")
    cache_write = _p("cache_write_input_tokens")
    tool_use_in = _p("tool_use_input_tokens")

    if cached_in is not None:
        rows.append(("Tokens", "Cached input tokens", _fmt_int(cached_in)))
    if uncached_in is not None:
        rows.append(("Tokens", "Uncached input tokens", _fmt_int(uncached_in)))
    if cache_write is not None:
        rows.append(("Tokens", "Cache write input tokens", _fmt_int(cache_write)))
    if tool_use_in is not None:
        rows.append(("Tokens", "Tool-use input tokens", _fmt_int(tool_use_in)))

    # Latency
    total_lat = _p("latency_ms_total")
    avg_lat = _p("latency_ms_avg")
    first_event_lat = _p("first_event_latency_ms_avg")

    if total_lat is not None:
        rows.append(("Latency", "Total latency", _fmt_ms(total_lat)))
    if avg_lat is not None:
        rows.append(("Latency", "Avg latency per call", _fmt_ms(avg_lat)))
    if first_event_lat is not None:
        rows.append(("Latency", "Avg time-to-first-token", _fmt_ms(first_event_lat)))

    return rows


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
