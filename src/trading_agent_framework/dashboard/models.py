"""Pydantic models for the strategy dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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
    def from_path(cls, path: str) -> RunRef:
        """Extract strategy_name, run_ts, and mode from a directory path.

        Expected pattern: logs/{strategy_name}/{mode}/{run_ts}_{mode}/
        (trading_agent_framework.utils.log.setup_strategy_logging's layout;
        run_ts is "%Y-%m-%d_%H%M%S", e.g. "2026-06-22_194053").
        """
        parts = [p for p in path.replace("\\", "/").rstrip("/").split("/") if p]
        if len(parts) < 3:
            raise ValueError(f"Cannot parse run directory from path: {path}")

        run_dir, mode, strategy_name = parts[-1], parts[-2], parts[-3]
        if mode not in ("backtesting", "paper", "live"):
            raise ValueError(f"Cannot parse mode from path: {path}")

        run_ts = "_".join(run_dir.split("_")[:2]) if "_" in run_dir else run_dir

        return cls(strategy_name=strategy_name, run_ts=run_ts, mode=mode, path=path)

    def __repr__(self) -> str:
        return f"RunRef({self.strategy_name}, {self.run_ts}, {self.mode})"


class Settings(BaseModel):
    """Parsed from settings.json."""

    name: str = ""
    backtesting_start: datetime | None = None
    backtesting_end: datetime | None = None
    budget: float = 0.0
    risk_free_rate: float = 0.0
    backtesting_data_sources: str = ""
    backtest_time_seconds: float = 0.0
    parameters: dict[str, Any] = Field(default_factory=dict)
    # Per-agent telemetry totals (written only when the run recorded agent calls); see agents/telemetry.py.
    agents: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Dashboard-only leaf field: "", "discarded", "study", or "validated"; see reader.save_decision.
    dashboard_decision: str = ""

    model_config = {"extra": "allow"}


class MetricSet(BaseModel):
    """All scalar performance metrics from metrics.json (backtesting.report.write_metrics
    writes this keyed exactly by these field names; see tests/backtesting/dashboard_contract.py)."""

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


# --- vLLM benchmark results (read by benchmark_reader.py) ---------------------------------------


@dataclass(frozen=True)
class BenchmarkRunRef:
    """A benchmark run directory, <results>/<YYYYMMDD-HHMMSS>/. Built without reading any file."""

    path: Path
    run_id: str
    started_at: datetime

    @classmethod
    def from_path(cls, path: Path) -> BenchmarkRunRef:
        try:
            started_at = datetime.strptime(path.name, "%Y%m%d-%H%M%S")
        except ValueError as exc:
            raise ValueError(f"Not a benchmark run directory name: {path.name}") from exc
        return cls(path=path, run_id=path.name, started_at=started_at)


@dataclass(frozen=True)
class ScenarioScore:
    passed: int
    runs: int
    mean_partial: float


@dataclass(frozen=True)
class BenchmarkModel:
    """One model's line of summary.json, joined with its meta.json entry (served name, vLLM version).

    Numeric fields are None when the model did not run (``ran`` is False, ``error`` says why).
    """

    key: str
    display_name: str
    served_name: str | None
    vllm_version: str | None
    ran: bool
    error: str | None
    overall: float | None
    mean_partial: float | None
    categories: dict[str, float]
    scenarios: dict[str, ScenarioScore]
    runs_passed: int | None
    runs_total: int | None
    text_tool_calls: int | None
    avg_tool_calls: float | None
    median_run_s: float | None
    median_tokens_per_s: float | None
    timeouts: int | None
    errors: int | None


@dataclass(frozen=True)
class BenchmarkRun:
    ref: BenchmarkRunRef
    started_at: datetime
    finished_at: datetime | None
    repeats: int
    timeout_s: float
    scenarios: tuple[str, ...]  # meta.json order, e.g. "reasoning.rsi_signal"
    models: tuple[BenchmarkModel, ...]  # summary.json order

    @property
    def categories(self) -> tuple[str, ...]:
        """Scenario id prefixes, in first-seen scenario order."""
        return tuple(dict.fromkeys(scenario.split(".", 1)[0] for scenario in self.scenarios))
