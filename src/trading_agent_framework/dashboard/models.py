"""Pydantic models for the strategy dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class RunRef:
    """Lightweight reference to a backtesting run directory.

    Created during discovery without reading any files.
    """

    strategy_name: str
    run_ts: str  # e.g. "2026-06-22_194053"
    mode: str  # "backtesting", "paper", or "live"
    path: str  # absolute or relative path to the run directory

    @classmethod
    def from_path(cls, path: str) -> "RunRef":
        """Extract strategy_name, run_ts, and mode from a directory path.

        Expected pattern: logs/agent_{strategy_name}/{mode}/{run_ts}_{mode}/
        or legacy:        logs/agent_{strategy_name}/{run_ts}/
        """
        parts = path.replace("\\", "/").rstrip("/").split("/")

        # Find the strategy directory (the one starting with "agent_")
        strategy_idx = None
        for i, part in enumerate(parts):
            if part.startswith("agent_"):
                strategy_idx = i
                break
        if strategy_idx is None:
            raise ValueError(f"Cannot parse strategy name from path: {path}")

        strategy_name = parts[strategy_idx].replace("agent_", "")

        # The run timestamp is the last directory component
        run_dir = parts[-1]
        run_ts = "_".join(run_dir.split("_")[:2]) if "_" in run_dir else run_dir

        # Determine mode from the path
        mode = "backtesting"  # default
        for part in parts[strategy_idx:]:
            if part in ("backtesting", "paper", "live"):
                mode = part

        return cls(strategy_name=strategy_name, run_ts=run_ts, mode=mode, path=path)

    def __repr__(self) -> str:
        return f"RunRef({self.strategy_name}, {self.run_ts}, {self.mode})"


class Settings(BaseModel):
    """Parsed from *_settings.json."""

    name: str = ""
    backtesting_start: datetime | None = None
    backtesting_end: datetime | None = None
    budget: float = 0.0
    risk_free_rate: float = 0.0
    backtesting_data_sources: str = ""
    backtest_time_seconds: float = 0.0
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class MetricSet(BaseModel):
    """All scalar performance metrics from *_tearsheet_metrics.json."""

    total_return_strategy: float = 0.0
    total_return_benchmark: float = 0.0
    cagr_strategy: float = 0.0
    cagr_benchmark: float = 0.0
    sharpe_strategy: float = 0.0
    sharpe_benchmark: float = 0.0
    sortino_strategy: float = 0.0
    sortino_benchmark: float = 0.0
    calmar_strategy: float = 0.0
    calmar_benchmark: float = 0.0
    omega_strategy: float = 0.0
    omega_benchmark: float = 0.0
    max_drawdown_strategy: float = 0.0
    max_drawdown_benchmark: float = 0.0
    volatility_strategy: float = 0.0
    volatility_benchmark: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    correlation: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio_strategy: float = 0.0
    information_ratio_benchmark: float = 0.0
    r_squared_strategy: float = 0.0
    r_squared_benchmark: float = 0.0
    skew_strategy: float = 0.0
    skew_benchmark: float = 0.0
    kurtosis_strategy: float = 0.0
    kurtosis_benchmark: float = 0.0
    win_days_pct_strategy: float = 0.0
    win_days_pct_benchmark: float = 0.0
    win_month_pct_strategy: float = 0.0
    win_month_pct_benchmark: float = 0.0
    longest_dd_days_strategy: float = 0.0
    longest_dd_days_benchmark: float = 0.0
    avg_drawdown_strategy: float = 0.0
    avg_drawdown_benchmark: float = 0.0
    recovery_factor_strategy: float = 0.0
    recovery_factor_benchmark: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_scalars(cls, scalar_metrics: dict[str, dict[str, float]], *, summary_tables: dict[str, Any] | None = None) -> "MetricSet":
        """Build a MetricSet from the scalar_metrics dict in tearsheet_metrics.json.

        The optional *summary_tables* dict (eoy_returns_vs_benchmark, drawdowns) is
        stored as raw["summary_tables"] so the dashboard "Yearly Returns" tab can
        access it.
        """

        def _get(metric_name: str, column: str = "Strategy") -> float:
            entry = scalar_metrics.get(metric_name, {})
            if isinstance(entry, dict):
                val = entry.get(column, 0.0)
            elif isinstance(entry, (int, float)):
                val = entry
            else:
                val = 0.0
            if val is None:
                return 0.0
            return float(val) if isinstance(val, (int, float)) else 0.0

        raw = dict(scalar_metrics)
        if summary_tables:
            raw["summary_tables"] = summary_tables

        return cls(
            total_return_strategy=_get("Total Return"),
            total_return_benchmark=_get("Total Return", "Benchmark"),
            cagr_strategy=_get("CAGR% (Annual Return)"),
            cagr_benchmark=_get("CAGR% (Annual Return)", "Benchmark"),
            sharpe_strategy=_get("Sharpe"),
            sharpe_benchmark=_get("Sharpe", "Benchmark"),
            sortino_strategy=_get("Sortino"),
            sortino_benchmark=_get("Sortino", "Benchmark"),
            calmar_strategy=_get("Calmar"),
            calmar_benchmark=_get("Calmar", "Benchmark"),
            omega_strategy=_get("Omega"),
            omega_benchmark=_get("Omega", "Benchmark"),
            max_drawdown_strategy=_get("Max Drawdown"),
            max_drawdown_benchmark=_get("Max Drawdown", "Benchmark"),
            volatility_strategy=_get("Volatility (ann.)"),
            volatility_benchmark=_get("Volatility (ann.)", "Benchmark"),
            beta=_get("Beta"),
            alpha=_get("Alpha"),
            correlation=_get("Correlation"),
            treynor_ratio=_get("Treynor Ratio"),
            information_ratio_strategy=_get("Information Ratio"),
            information_ratio_benchmark=_get("Information Ratio", "Benchmark"),
            r_squared_strategy=_get("R^2"),
            r_squared_benchmark=_get("R^2", "Benchmark"),
            skew_strategy=_get("Skew"),
            skew_benchmark=_get("Skew", "Benchmark"),
            kurtosis_strategy=_get("Kurtosis"),
            kurtosis_benchmark=_get("Kurtosis", "Benchmark"),
            win_days_pct_strategy=_get("Win Days%"),
            win_days_pct_benchmark=_get("Win Days%", "Benchmark"),
            win_month_pct_strategy=_get("Win Month%"),
            win_month_pct_benchmark=_get("Win Month%", "Benchmark"),
            longest_dd_days_strategy=_get("Longest DD Days"),
            longest_dd_days_benchmark=_get("Longest DD Days", "Benchmark"),
            avg_drawdown_strategy=_get("Avg. Drawdown"),
            avg_drawdown_benchmark=_get("Avg. Drawdown", "Benchmark"),
            recovery_factor_strategy=_get("Recovery Factor"),
            recovery_factor_benchmark=_get("Recovery Factor", "Benchmark"),
            raw=raw,
        )


@dataclass
class Run:
    """A fully loaded backtesting run with all data read from disk."""

    ref: RunRef
    settings: Settings | None = None
    metrics: MetricSet | None = None
    equity_curve: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunIndex:
    """Collection of all discovered runs."""

    runs: list[RunRef] = field(default_factory=list)

    def grouped(self) -> dict[str, list[RunRef]]:
        """Group runs by strategy name."""
        groups: dict[str, list[RunRef]] = {}
        for r in self.runs:
            groups.setdefault(r.strategy_name, []).append(r)
        return groups

    def strategy_names(self) -> list[str]:
        return sorted(set(r.strategy_name for r in self.runs))
