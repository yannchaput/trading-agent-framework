"""Plain typed research tools for the insider_trades researcher: a watchlist stock's recent insider trades with their labels, and one Form 4.

Both tools read through `InsiderFacts`, which takes its cutoff from `strategy.clock.now()` (the research-tool rule: no data tool may
use wall-clock time) and hides every filing not strictly before the market date, whatever the source returned. The labels are the
code's; the researcher only reads them and the evidence behind them (`rule`, `months_by_year`).

The payloads are lean on purpose (they reach the LLM on every call): dollars are whole `int`s, fractions `float`s rounded to 4
decimals, dates ISO strings. This is a payload boundary only; every figure upstream stays `Decimal`.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema from the
function's real annotations (as `memory/tools.py` does).
"""

from collections import defaultdict
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any

from trading_agent_framework.memory.tools import RunMemo, current_run_id
from trading_agent_framework.strategies.insider_trades.facts import InsiderFacts
from trading_agent_framework.strategies.insider_trades.form4.classify import Labelled
from trading_agent_framework.strategies.insider_trades.form4.signal import InsiderSummary
from trading_agent_framework.utils.errors import InsiderDataError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MAX_TRADES_PER_INSIDER = 10
MAX_FOOTNOTE_CHARS = 500


def _dollars(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _fraction(value: Decimal) -> float:
    return round(float(value), 4)


def _shares(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def _price(value: Decimal) -> float:
    return round(float(value), 4)


def insider_research_tools(strategy: "Strategy", facts: InsiderFacts, on_fetch: Callable[[str], None]) -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """`get_insider_trades` and `read_form4`, bound to `strategy` and its `facts`; `on_fetch(ticker)` follows each successful `get_insider_trades`."""
    memo = RunMemo()  # an identical call within one agent run reuses the first result
    params = facts.params

    def get_insider_trades(ticker: str) -> dict[str, Any]:
        """List a watchlist stock's recent insider trades per insider: dollars bought and sold by label, trade rows with the rule behind each label, and the months the insider traded in each year."""
        return memo.once(current_run_id(), ("get_insider_trades", ticker.strip().upper()), lambda: _trades(ticker.strip().upper()))

    def read_form4(accession_number: str) -> dict[str, Any]:
        """Read one known Form 4 by accession number: owners, plan flag, open-market rows and footnotes."""
        return memo.once(current_run_id(), ("read_form4", accession_number.strip()), lambda: _read(accession_number.strip()))

    def _trades(ticker: str) -> dict[str, Any]:
        if ticker not in {t.strip().upper() for t in params.watchlist}:
            return {"error": f"{ticker} is not on the watchlist: {', '.join(params.watchlist)}"}
        try:
            stock = facts.facts(ticker)
            history = facts.labelled(ticker)
        except InsiderDataError as exc:
            return {"error": str(exc)}
        months = _months_by_year(history)
        insiders = [_insider(summary, months[summary.insider_id]) for summary in stock.insiders]
        on_fetch(ticker)
        return {
            "ticker": ticker,
            "as_of": stock.today.isoformat(),
            "window_start": stock.window_start.isoformat(),
            "insiders": insiders,
            "count": len(insiders),
        }

    def _read(accession: str) -> dict[str, Any]:
        try:
            filing = facts.filing(accession)
        except InsiderDataError as exc:
            return {"error": str(exc)}
        if filing is None:
            return {"error": f"{accession} is not a known Form 4: call get_insider_trades for the filings filed before {facts.today().isoformat()}"}
        return {
            "accession": filing.accession,
            "filed": filing.filed.isoformat(),
            "owners": [{"name": owner.name, "role": owner.role} for owner in filing.owners],
            "attributed_to": filing.attributed_to.name if filing.attributed_to else None,
            "co_filers": list(filing.co_filers),
            "plan": filing.plan,
            "rows": [
                {
                    "code": t.code,
                    "traded": t.trade_date.isoformat(),
                    "shares": _shares(t.shares),
                    "price": _price(t.price),
                    "owned_after": _shares(t.owned_after),
                }
                for t in filing.trades
            ],
            "footnotes": {note_id: text[:MAX_FOOTNOTE_CHARS] for note_id, text in filing.footnotes.items()},
        }

    return [get_insider_trades, read_form4]


def _months_by_year(history: list[Labelled]) -> dict[str, dict[str, list[int]]]:
    """Per insider, the months of each calendar year they traded in at this issuer, over every known year (the habit the label rests on)."""
    seen: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for lt in history:
        seen[lt.trade.owner_cik][lt.trade.trade_date.year].add(lt.trade.trade_date.month)
    return {cik: {str(year): sorted(months) for year, months in sorted(by_year.items())} for cik, by_year in seen.items()}


def _insider(summary: InsiderSummary, months_by_year: dict[str, list[int]]) -> dict[str, Any]:
    newest_first = sorted(summary.labelled, key=lambda lt: (lt.trade.filed, lt.trade.trade_date), reverse=True)
    shown = [
        {
            "accession": lt.trade.accession,
            "filed": lt.trade.filed.isoformat(),
            "traded": lt.trade.trade_date.isoformat(),
            "code": lt.trade.code,
            "shares": _shares(lt.trade.shares),
            "price": _price(lt.trade.price),
            "label": lt.label.value,
            "rule": lt.rule,
        }
        for lt in newest_first[:MAX_TRADES_PER_INSIDER]
    ]
    payload: dict[str, Any] = {
        "insider_id": summary.insider_id,
        "name": summary.name,
        "role": summary.role,
        "bought": {label.value: _dollars(dollars) for label, dollars in summary.bought.items()},
        "sold": {label.value: _dollars(dollars) for label, dollars in summary.sold.items()},
        "max_buy_fraction": _fraction(summary.max_buy_fraction),
        "sell_fraction": _fraction(summary.sell_fraction),
        "months_by_year": months_by_year,
        "trades": shown,
    }
    if len(newest_first) > MAX_TRADES_PER_INSIDER:
        payload["more_trades"] = len(newest_first) - MAX_TRADES_PER_INSIDER
    return payload
