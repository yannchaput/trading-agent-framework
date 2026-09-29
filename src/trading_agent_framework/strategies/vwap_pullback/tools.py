"""The two agents' tools (spec §4): lean views over the session, and actions that all go through `Desk`.

No `from __future__ import annotations` on purpose, like `memory/tools.py`: LangChain builds each tool's
schema from the real annotations. Docstrings are one line: each is sent to the model on every call.
Neither agent gets a raw order tool: sizes, prices and stops are the desk's.
"""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.memory.tools import current_run_id
from trading_agent_framework.strategies.vwap_pullback.setups import SetupState, health

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy
    from trading_agent_framework.strategies.vwap_pullback.desk import Desk

# Same values as `prompts.CATALYSTS`; as a Literal, LangChain turns it into an enum in the tool schema, so the
# model is told the allowed values and a wrong one is rejected before `Desk.enter_long` even runs.
Catalyst = Literal["earnings", "guidance", "analyst", "contract_or_product", "sector_or_macro", "none"]
MAX_BARS = 24  # two hours of 5-minute bars: enough context, bounded tokens


def _after(created_at: str, moment: datetime) -> bool:
    """Whether an ISO-8601 `created_at` (a trailing "Z" allowed) is later than `moment`; False when unparseable."""
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")) > moment
    except ValueError:
        return False


def setup_rows(desk: "Desk") -> list[dict[str, Any]]:  # noqa: UP037
    """Pullback and triggered setups: health, stage-2 z-scores, the stop and R an entry would get, headlines."""
    state = desk.state
    rows: list[dict[str, Any]] = []
    for symbol, setup in sorted(state.setups.items()):
        if setup.state not in (SetupState.PULLBACK, SetupState.TRIGGERED):
            continue
        row: dict[str, Any] = {"symbol": symbol, **health(setup)}
        info = state.candidates.get(symbol)
        if info is not None:
            row |= {"z_rs": round(info.z_rs, 2), "z_rvol": round(info.z_rvol, 2)}
        planned = desk.planned_risk(symbol)
        if planned is not None:
            row |= {"planned_stop": float(planned[0]), "r_per_share": float(planned[1])}
        row["headlines"] = state.headlines.get(symbol, [])
        rows.append(row)
    return rows


def trade_rows(desk: "Desk", now: datetime) -> list[dict[str, Any]]:  # noqa: UP037
    """Open trades: size, stop, open profit in R, VWAP, 9-EMA, time left, and headlines since entry."""
    rows: list[dict[str, Any]] = []
    for trade in desk.state.book.open_trades():
        levels = desk.levels(trade.symbol)
        last = Decimal(str(levels.close)) if levels is not None else None
        rows.append({
            "symbol": trade.symbol,
            "quantity": int(trade.quantity),
            "entry_price": float(trade.entry_price) if trade.entry_price is not None else None,
            "last_close": float(last) if last is not None else None,
            "stop_kind": trade.stop_kind,
            "stop_level": float(trade.stop_level),
            "r_per_share": float(trade.r_per_share),
            "unrealised_r": trade.unrealised_r(last) if last is not None else None,
            "tp1_done": trade.tp1_done,
            "vwap": round(levels.vwap, 2) if levels is not None else None,
            "ema9": round(levels.ema, 2) if levels is not None and levels.ema is not None else None,
            "minutes_to_flatten": desk.minutes_to_flatten(now),
            "headlines_since_entry": [h for h in desk.state.headlines.get(trade.symbol, []) if _after(h["created_at"], trade.entered_at)],
        })
    return rows


def budgeted_search_news(strategy: "Strategy", calls_per_run: int) -> Callable[..., dict[str, Any]]:  # noqa: UP037
    """The shared `search_news` tool, refused after `calls_per_run` calls in one agent run."""
    inner = news_tools(strategy)[0]
    # Calls made in the current agent run. A new run id (every agent invocation gets one) resets the count,
    # so the budget is per run; a local model left unbudgeted has looped 100+ times on a search tool.
    usage: dict[str, Any] = {"run_id": None, "count": 0}

    # Same signature and docstring as the wrapped tool, spelled out (not functools.wraps) so LangChain reads
    # a plain function's annotations when it builds the schema.
    def search_news(symbols: str = "", start: str | None = None, end: str | None = None, limit: int = 10, include_content: bool = False) -> dict[str, Any]:
        """Search recent news headlines and summaries, optionally filtered to symbols."""
        run_id = current_run_id()
        if run_id != usage["run_id"]:
            usage["run_id"], usage["count"] = run_id, 0
        if usage["count"] >= calls_per_run:
            return {"error": "news budget for this run is spent; decide with what you have"}
        usage["count"] += 1
        return inner(symbols=symbols, start=start, end=end, limit=limit, include_content=include_content)

    return search_news


def _bars_tool(desk: "Desk") -> Callable[..., dict[str, Any]]:  # noqa: UP037
    """`get_intraday_bars`, shared by both agents: reads the scan's cached contexts, never fetches data."""

    def get_intraday_bars(symbol: str, length: int = 12) -> dict[str, Any]:
        """Get a tracked symbol's recent 5-minute bars with VWAP, oldest first."""
        contexts = desk.state.contexts.get(symbol.strip().upper())
        if not contexts:
            return {"error": f"no intraday bars for {symbol.strip().upper()}"}
        count = min(max(int(length), 1), MAX_BARS)
        return {
            "symbol": symbol.strip().upper(),
            "bars": [
                {"time": c.time.isoformat(), "open": round(c.open, 2), "high": round(c.high, 2), "low": round(c.low, 2), "close": round(c.close, 2), "volume": int(c.volume), "vwap": round(c.vwap, 2)}
                for c in contexts[-count:]
            ],
        }

    return get_intraday_bars


def entry_tools(strategy: "Strategy", desk: "Desk") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """The entry agent's tools: read setups and bars, search news (budgeted), then enter or pass.

    Every action delegates to `desk`, which enforces all the risk rules; the tool docstrings below are
    one line on purpose (they are sent to the model on every call).
    """

    def get_setups() -> dict[str, Any]:
        """List the pullback and triggered setups with their health, planned stop and headlines."""
        return {"setups": setup_rows(desk)}

    def enter_long(symbol: str, catalyst: Catalyst, reason: str) -> dict[str, Any]:
        """Enter a triggered setup; the code sizes the position and places the stop."""
        return desk.enter_long(symbol, catalyst, reason)

    def pass_on_setup(symbol: str, reason: str) -> dict[str, Any]:
        """Record that you pass on a triggered setup."""
        return desk.pass_on_setup(symbol, reason)

    return [get_setups, _bars_tool(desk), budgeted_search_news(strategy, desk.params.news_calls_per_run), enter_long, pass_on_setup]


def exit_tools(strategy: "Strategy", desk: "Desk") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """The exit agent's tools: read trades and bars, search news (budgeted), then one action per trade.

    The stop hand-offs (cancel the working stop, wait, then sell or re-place) all happen inside `desk`;
    each tool gets its own budgeted `search_news`, so the entry and exit agents do not share one budget.
    """

    def get_open_trades() -> dict[str, Any]:
        """List the open trades with their stop, open profit in R, VWAP, 9-EMA and new headlines."""
        return {"trades": trade_rows(desk, strategy.get_datetime())}

    def take_partial_profit(symbol: str, fraction: float) -> dict[str, Any]:
        """Sell part (0.25 to 0.5) of an open trade once; the stop is resized to the rest."""
        return desk.take_partial_profit(symbol, fraction)

    def tighten_stop(symbol: str, stop_price: float) -> dict[str, Any]:
        """Raise an open trade's stop price (a stop only moves up)."""
        return desk.tighten_stop(symbol, stop_price)

    def replace_stop_with_trailing(symbol: str, trail_atr: float) -> dict[str, Any]:
        """Replace an open trade's stop with a trailing stop of trail_atr 5-minute ATRs."""
        return desk.replace_stop_with_trailing(symbol, trail_atr)

    def exit_position(symbol: str, reason: str) -> dict[str, Any]:
        """Sell the whole remaining position now."""
        return desk.exit_position(symbol, reason)

    def hold(symbol: str, reason: str) -> dict[str, Any]:
        """Keep an open trade unchanged this review."""
        return desk.hold(symbol, reason)

    return [
        get_open_trades, _bars_tool(desk), budgeted_search_news(strategy, desk.params.news_calls_per_run),
        take_partial_profit, tighten_stop, replace_stop_with_trailing, exit_position, hold,
    ]
