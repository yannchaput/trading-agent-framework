"""Batch script: extract drawdown episodes from diagnostics.parquet.

Reads a diagnostics.parquet file produced by CrossMomentumStrategyV23
and prints the worst drawdown episodes with risk snapshots to the console.

Usage:
    uv run python -m lumibot_trading_agent.strategies.candidates.cross_momentum.batch_diagnostics_drawdown_episodes \\
        logs/agent_cross_momentum_v23/backtesting/2024-01-01_120000_backtesting/diagnostics.parquet
"""

import json
import sys
from pathlib import Path

import pandas as pd

from lumibot_trading_agent.strategies.candidates.cross_momentum.risk_diagnostics import (
    compute_baseline_stats,
    identify_drawdown_episodes,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python batch_diagnostics_drawdown_episodes.py <path/to/diagnostics.parquet>")
        raise SystemExit(1)

    diag_path = Path(sys.argv[1])
    if not diag_path.exists():
        print(f"File not found: {diag_path}")
        raise SystemExit(1)

    print(f"Reading diagnostics from: {diag_path}")
    df = pd.read_parquet(diag_path)
    print(f"  {len(df)} rows, {list(df.columns)}")

    # Identify drawdown episodes
    episodes = identify_drawdown_episodes(df, top_n=5)

    if not episodes:
        print("\nNo drawdown episodes found.")
    else:
        print(f"\n{'=' * 80}")
        print(f"Top {len(episodes)} Drawdown Episodes")
        print(f"{'=' * 80}")

        for i, ep in enumerate(episodes, 1):
            dd_pct = ep["drawdown_at_trough"] * 100
            print(f"\n--- Episode {i} ---")
            print(f"  Trough date:        {ep['trough_date']}")
            print(f"  Drawdown:           {dd_pct:.1f}%")
            print(f"  Prior peak date:    {ep['peak_date']}")
            print(f"  Recovery date:      {ep.get('recovery_date', 'Not recovered')}")
            print(f"  Drawdown duration:  {ep['drawdown_duration']} days")
            print(f"  Recovery duration:  {ep.get('recovery_duration', 'N/A')}")

            print("\n  Risk diagnostics before/during trough:")
            print(f"  {'Offset':<8} {'Date':<14} {'Vol20d':>8} {'Vol63d':>8} {'Corr20d':>8} {'Corr63d':>8} {'WCorr63d':>8} {'Beta':>8} {'SectorExp':>10} {'HHI':>8} {'EffN':>8}")
            print(f"  {'-' * 8} {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 8}")

            for offset_label in ["T-60", "T-40", "T-20", "T-10", "T-5", "T+0"]:
                snap = ep["diagnostics_snapshots"].get(offset_label)
                if snap is None:
                    continue
                date_str = snap.get("date", "")[:10]
                vol20 = _fmt(snap.get("portfolio_volatility_20d"))
                vol63 = _fmt(snap.get("portfolio_volatility_63d"))
                corr20 = _fmt(snap.get("average_correlation_20d"))
                corr63 = _fmt(snap.get("average_correlation_63d"))
                wcorr63 = _fmt(snap.get("weighted_correlation_63d"))
                beta = _fmt(snap.get("portfolio_beta_63d"))
                sector_exp = _fmt(snap.get("largest_sector_exposure"))
                hhi = _fmt(snap.get("herfindahl_index"))
                eff_n = _fmt(snap.get("effective_number_positions"))

                print(f"  {offset_label:<8} {date_str:<14} {vol20:>8} {vol63:>8} {corr20:>8} {corr63:>8} {wcorr63:>8} {beta:>8} {sector_exp:>10} {hhi:>8} {eff_n:>8}")

    # Baseline stats
    print(f"\n{'=' * 80}")
    print("Baseline Statistics (median / 90th percentile)")
    print(f"{'=' * 80}")
    stats = compute_baseline_stats(df)
    for key, val in sorted(stats.items()):
        if val is not None:
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        else:
            print(f"  {key}: N/A")

    # Raw JSON for programmatic use
    print(f"\n{'=' * 80}")
    print("Raw episodes (JSON)")
    print(f"{'=' * 80}")
    print(json.dumps(episodes, indent=2, default=str))


def _fmt(value: float | None) -> str:
    """Format a float for table display, returning '    N/A' for None."""
    if value is None:
        return "    N/A"
    return f"{value:8.4f}"


if __name__ == "__main__":
    main()
