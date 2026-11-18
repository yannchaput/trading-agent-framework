"""`uv run bear-spread-research`: replay bear put spreads on the bearish regime over Yahoo history (read-only).

Downloads SPY, ^VIX and ^VIX3M through `YahooRegimeData`, replays `research.simulate` over the window, prints the
trades, the SPY falls, a summary and the GO / NO-GO verdict, and writes `trades.csv`, `episodes.csv` and `equity.csv`.
It places no order and touches no broker.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

from trading_agent_framework.core.regime import RegimeParameters, regime_by_session
from trading_agent_framework.core.regime_cli import DEFAULT_START, lookback_days, trim
from trading_agent_framework.core.regime_data import RegimeDataError, YahooRegimeData
from trading_agent_framework.strategies.bear_put_spread.pricing import CostModel, VolModel
from trading_agent_framework.strategies.bear_put_spread.research import EpisodeResult, SimulationResult, Trade, Verdict, episode_results, simulate, verdict
from trading_agent_framework.strategies.bear_put_spread.rules import SpreadParameters
from trading_agent_framework.strategies.bear_put_spread.triggers import TRIGGERS, trigger_states

OUT_ROOT = Path("logs/bear_put_spread/research")


def _print_report(window: Sequence[date], result: SimulationResult, episodes: Sequence[EpisodeResult], got: Verdict, start_equity: float) -> None:
    for t in result.trades:
        print(f"trade {t.entry} -> {t.exit or 'open'} [{t.reason}] {t.contracts}x {t.long_strike:.0f}/{t.short_strike:.0f} exp {t.expiry} pnl {t.pnl:+.2f} ({t.pnl_pct:+.1%})")
    for e in episodes:
        state = "still under way" if e.open else "closed"
        print(f"fall {e.episode.peak} -> {e.episode.trough} ({e.episode.depth:+.1%}) {state}: {e.trades} trades, pnl {e.pnl:+.2f}")
    closed = [t for t in result.trades if t.reason != "open"]
    wins = [t.pnl for t in closed if t.pnl > 0]
    losses = [t.pnl for t in closed if t.pnl <= 0]
    last = window[-1]
    invested = sum(1 for d in window if any(t.entry <= d <= (t.exit or last) for t in result.trades))
    final = result.equity[-1][1]
    print(f"final equity {final:.2f} ({final / start_equity - 1:+.1%}), {len(closed)} closed trades, win rate {len(wins) / len(closed) if closed else 0.0:.0%}")
    print(f"average win {sum(wins) / len(wins) if wins else 0.0:+.2f}, average loss {sum(losses) / len(losses) if losses else 0.0:+.2f}, invested {invested} of {len(window)} sessions")
    print(f"skipped: too small {result.skipped}")
    for ok, text in (
        (got.pnl_ok, f"closed trades P&L {got.closed_pnl:+.2f} > 0"),
        (got.episodes_ok, f"profit on more than half of closed falls ({got.episodes_won} of {got.episodes_closed})"),
        (got.drawdown_ok, f"max drawdown {got.drawdown:.1%} <= {got.max_drawdown:.0%}"),
    ):
        print(f"{'PASS' if ok else 'FAIL'}: {text}")
    print(f"verdict: {'GO' if got.go else 'NO-GO'}")


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(out: Path, result: SimulationResult, episodes: Sequence[EpisodeResult]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "trades.csv", [dataclasses.asdict(t) for t in result.trades], [f.name for f in dataclasses.fields(Trade)])
    episode_rows = [{"peak": e.episode.peak, "trough": e.episode.trough, "depth": e.episode.depth, "trades": e.trades, "pnl": e.pnl, "open": e.open} for e in episodes]
    _write_csv(out / "episodes.csv", episode_rows, ["peak", "trough", "depth", "trades", "pnl", "open"])
    _write_csv(out / "equity.csv", [{"date": d, "equity": v} for d, v in result.equity], ["date", "equity"])


def run(start: date, end: date | None, equity: float, vol: VolModel, costs: CostModel, out: Path, trigger: str = "regime") -> int:
    """Download, replay, print and write the outputs.

    Args:
        start: First replayed session.
        end: Last replayed session; the last downloaded one when None.
        equity: Starting cash, dollars.
        vol: Volatility model.
        costs: Cost model.
        out: Directory for the CSVs, created only on success.
        trigger: `regime` (the market regime is bearish) or one of `triggers.TRIGGERS`; the window is the same either way.

    Returns:
        The exit code: 0 (GO and NO-GO alike), or 1 when the data cannot be used.
    """
    try:
        series = trim(YahooRegimeData().series(start - timedelta(days=lookback_days())), end or date.max)
    except RegimeDataError as exc:
        print(f"error: {exc}")
        return 1
    readings = regime_by_session(series.sessions, series.spy, series.vix, series.vix3m, RegimeParameters())
    rows = [i for i, d in enumerate(series.sessions) if d >= start and d in readings and (end is None or d <= end)]
    if not rows:
        print(f"error: no session between {start} and {end or 'the last download'} has enough history")
        return 1
    window = [series.sessions[i] for i in rows]
    spy = [series.spy[i] for i in rows]
    if trigger == "regime":
        regimes = {d: readings[d].regime for d in window}
    else:
        states = trigger_states(trigger, series.spy, series.vix)
        regimes = {series.sessions[i]: states[i] for i in rows}
    print(f"trigger: {trigger}")
    params = SpreadParameters()
    result = simulate(window, spy, [series.vix[i] for i in rows], regimes, params, vol, costs, equity)
    episodes = episode_results(window, spy, result.trades)
    _print_report(window, result, episodes, verdict(result, episodes), equity)
    _write_outputs(out, result, episodes)
    print(f"written to {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `bear-spread-research` command.

    Args:
        argv: Arguments; the process's when None.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(prog="bear-spread-research", description="Replay SPY bear put spreads on the bearish regime (read-only).")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START, help="first replayed session")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="last replayed session (default: the last downloaded)")
    parser.add_argument("--trigger", choices=["regime", *TRIGGERS], default="regime", help="what opens a spread: the bearish regime (default) or t1 / t2 / t3 (triggers.py)")
    parser.add_argument("--equity", type=float, default=1200.0, help="starting cash in dollars")
    parser.add_argument("--vol-scale", type=float, default=1.0, help="multiplier on VIX / 100")
    parser.add_argument("--skew", type=float, default=0.8, help="volatility points per unit of ln(spot / strike) below spot")
    parser.add_argument("--half-spread", type=float, default=0.05, help="half bid/ask, dollars per share per leg")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: logs/bear_put_spread/research/<timestamp>)")
    args = parser.parse_args(argv)
    out = args.out or OUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return run(args.start, args.end, args.equity, VolModel(args.vol_scale, args.skew), CostModel(half_spread=args.half_spread), out, args.trigger)
