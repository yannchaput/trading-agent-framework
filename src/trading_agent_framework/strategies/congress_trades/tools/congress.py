"""Plain typed research tools for the congress_trades research agent: which filings are known, and what one contains.

Both tools take their cutoff from `strategy.clock.now()` (the research-tool rule: no data tool may use wall-clock time),
so in a backtest the agent sees only filings dated before the simulated day. A filing counts as known only when its
filing DATE is strictly before the market date of `now` (`CongressSource.known` applies the rule; this module applies it
again to whatever the source returns, so a faulty source cannot leak a filing dated today or later).

The payloads are lean on purpose (they reach the LLM on every call): amounts are whole dollars, dates ISO strings, and
asset names are left out (the ticker identifies the holding; market-data tools check it).

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema from
the function's real annotations (as `memory/tools.py` does).
"""

from collections.abc import Callable
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.memory.tools import RunMemo, current_run_id
from trading_agent_framework.strategies.congress_trades.congress.source import KnownFilings
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import CongressDataError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


class KnownFilingsSource(Protocol):
    def known(self, as_of: datetime) -> KnownFilings: ...


def congress_research_tools(strategy: "Strategy", source: KnownFilingsSource) -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """`list_filings` and `read_filing`, bound to `strategy` and its filings `source`."""
    memo = RunMemo()  # an identical call within one agent run reuses the first result
    cache: dict[date, KnownFilings] = {}  # the parsed filings known on a market date: one lookup serves both tools

    def known_today() -> tuple[date, KnownFilings]:
        today = strategy.clock.now().astimezone(MARKET_TZ).date()
        if today not in cache:
            cache.clear()
            cache[today] = source.known(strategy.clock.now())
        return today, cache[today]

    def list_filings() -> dict[str, Any]:
        """List the known filings: her newest yearly report (the base holdings) and the trade reports filed since."""
        return memo.once(current_run_id(), "list_filings", _list)

    def read_filing(doc_id: str) -> dict[str, Any]:
        """Read one known filing by doc_id: a yearly report's stock holdings or a trade report's stock trades."""
        return memo.once(current_run_id(), ("read_filing", doc_id), lambda: _read(doc_id))

    def _list() -> dict[str, Any]:
        try:
            today, known = known_today()
        except CongressDataError as exc:
            return {"error": str(exc)}
        refs = [ref for ref in known.refs if ref.filed < today]
        return {
            "as_of": today.isoformat(),
            "base_report": known.annual_ref.doc_id,
            "period_end": known.period_end.isoformat(),
            "count": len(refs),
            "filings": [{"doc_id": ref.doc_id, "kind": ref.kind, "filed": ref.filed.isoformat(), "year": ref.year} for ref in refs],
            "unparsed_filings": known.unparsed_filings,
            "skipped_non_stock": known.skipped_non_stock,
        }

    def _read(doc_id: str) -> dict[str, Any]:
        try:
            today, known = known_today()
        except CongressDataError as exc:
            return {"error": str(exc)}
        ref = next((r for r in known.refs if r.doc_id == doc_id.strip()), None)
        if ref is None or ref.filed >= today:
            return {"error": f"{doc_id} is not a known filing: call list_filings for the filings filed before {today.isoformat()}"}
        base = {"doc_id": ref.doc_id, "kind": ref.kind, "filed": ref.filed.isoformat()}
        if ref.doc_id == known.annual_ref.doc_id:
            holdings = [{"ticker": a.ticker, "owner": a.owner, "value_low": int(a.value_low), "value_high": int(a.value_high), "tier": a.tier} for a in known.assets]
            return {**base, "period_end": known.period_end.isoformat(), "count": len(holdings), "holdings": holdings}
        trades = [
            {
                "ticker": t.ticker,
                "side": t.side,
                "owner": t.owner,
                "traded": t.transaction_date.isoformat(),
                "notified": t.notification_date.isoformat(),
                "amount_low": int(t.amount_low),
                "amount_high": int(t.amount_high),
            }
            for t in known.transactions
            if t.doc_id == ref.doc_id
        ]
        return {**base, "count": len(trades), "trades": trades}

    return [list_filings, read_filing]
