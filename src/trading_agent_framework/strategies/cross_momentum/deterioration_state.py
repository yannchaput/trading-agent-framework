"""Deterioration state persistence for V2.2 strategies.

Stores per-position deterioration state (normal/exited) as JSON under
.lumibot/memory_<mode>/<strategy>/deterioration_state.json so that
live/paper trading survives crashes and restarts.

Also provides daily rank snapshot persistence for Signal B (rank velocity).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DeteriorationStateManager:
    """Manages deterioration state for each portfolio position.

    Persists to a JSON file so state survives process restarts in
    live and paper trading. In backtesting, memory directories are
    fresh per run so state is effectively ephemeral.

    Usage:
        mgr = DeteriorationStateManager(memory_dir)
        state = mgr.get_position_state("AAPL")       # "normal" | "exited"
        mgr.set_exited("AAPL", {"date": "2026-07-13", "action": "exit", ...})
        mgr.reset_for_friday({"AAPL", "MSFT"})        # after Friday rebalance
    """

    def __init__(self, memory_dir: Path) -> None:
        self._file = memory_dir / "deterioration_state.json"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    # ── Public API ──────────────────────────────────────────────────────

    def get_position_state(self, symbol: str) -> str:
        """Return 'normal' or 'exited' for a symbol. Defaults to 'normal'."""
        pos = self._state.setdefault("positions", {}).get(symbol)
        if pos is None:
            return "normal"
        return pos.get("deterioration_state", "normal")

    def set_exited(self, symbol: str, event: dict) -> None:
        """Mark a position as exited, record the event, and persist."""
        positions = self._state.setdefault("positions", {})
        entry = positions.setdefault(
            symbol,
            {"deterioration_state": "normal", "deterioration_actions": []},
        )
        entry["deterioration_state"] = "exited"
        entry.setdefault("deterioration_actions", []).append(event)
        self._save()

    def reset_for_friday(self, held_symbols: set[str]) -> None:
        """After Friday rebalance: prune non-held symbols, reset held to 'normal'."""
        positions = self._state.setdefault("positions", {})

        # Remove entries for symbols no longer in the portfolio
        stale = [s for s in positions if s not in held_symbols]
        for s in stale:
            del positions[s]

        # Reset all currently-held positions to normal
        for s in held_symbols:
            positions[s] = {"deterioration_state": "normal", "deterioration_actions": []}

        self._date_touch()
        self._save()

    # ── Rank snapshots (Signal B — momentum rank velocity) ──────────────

    def save_rank_snapshot(self, date_str: str, rank_dict: dict[str, int]) -> None:
        """Store a full symbol→rank mapping for a given date.

        Args:
            date_str: ISO date string (e.g. '2026-07-13').
            rank_dict: Mapping from ticker symbol to momentum rank (int).
        """
        snapshots = self._state.setdefault("rank_snapshots", {})
        snapshots[date_str] = rank_dict
        self._save()

    def get_rank_snapshot(self, date_str: str) -> dict[str, int] | None:
        """Retrieve the symbol→rank snapshot for a given date, or None."""
        snapshots = self._state.get("rank_snapshots", {})
        return snapshots.get(date_str)

    def prune_old_snapshots(self, keep_days: int = 20) -> None:
        """Remove rank snapshots older than `keep_days` trading days.

        Kept generous by default to avoid data loss during market holidays.
        """
        from datetime import date, timedelta

        snapshots = self._state.get("rank_snapshots", {})
        if not snapshots:
            return

        cutoff = date.today() - timedelta(days=keep_days)
        stale = [d for d in snapshots if d < str(cutoff)]
        for d in stale:
            del snapshots[d]
        if stale:
            self._save()

    # ── Internal helpers ────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load state from disk, or return a fresh dict if missing/corrupt."""
        if not self._file.exists():
            return {"date": None, "positions": {}}
        try:
            raw = json.loads(self._file.read_text())
            raw.setdefault("positions", {})
            return raw
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Corrupt deterioration state file %s: %s — starting fresh.",
                self._file,
                exc,
            )
            return {"date": None, "positions": {}}

    def _save(self) -> None:
        """Persist current state to disk atomically."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, default=str))
        tmp.replace(self._file)

    def _date_touch(self) -> None:
        """Update the date in the state (kept minimal — caller sets via save)."""
        from datetime import date

        self._state["date"] = str(date.today())
