"""Read-only access to the `reviews.jsonl` of a backtesting run (one JSON line per strategy review).

Written by the agent pipelines (`bill_ackman`, `bull_bear`) into the run directory, and shown in the
Agents tab of Run Detail.
The two strategies share some fields (`date`, `abandoned`, `holdings`/`debate_set`, `forced_exits`,
`targets`, `orders`) and differ in the rest, so the table keeps only what both can fill and the page
shows the raw record for the rest. Streamlit-free, like reader.py. Nothing here writes into a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_agent_framework.dashboard.models import RunRef

REVIEWS_FILE = "reviews.jsonl"
COLUMNS = ["Date", "Status", "Stage", "Holdings", "Targets", "Forced exits", "Orders", "Error"]


@dataclass(frozen=True)
class ReviewsFile:
    """The records of one `reviews.jsonl`, plus how many lines could not be read."""

    records: list[dict[str, Any]]
    malformed: int


def has_reviews(ref: RunRef) -> bool:
    """Whether the run wrote a reviews file (only the agent strategies do)."""
    return (Path(ref.path) / REVIEWS_FILE).is_file()


def load_reviews(ref: RunRef) -> ReviewsFile:
    """Every record of the run's reviews file in file order; a line that is not a JSON object is counted, not raised.

    Raises OSError (FileNotFoundError included) when the file cannot be read.
    """
    records: list[dict[str, Any]] = []
    malformed = 0
    text = (Path(ref.path) / REVIEWS_FILE).read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            malformed += 1
    return ReviewsFile(records=records, malformed=malformed)


def reviews_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per review: the fields both strategies' records can fill, as display text."""
    rows = [
        {
            "Date": str(record.get("date", "")),
            "Status": "abandoned" if record.get("abandoned") else "completed",
            "Stage": str(record.get("stage") or ""),
            "Holdings": ", ".join(_holdings(record)),
            "Targets": _targets(record.get("targets")),
            "Forced exits": ", ".join(_symbols(record.get("forced_exits"))),
            "Orders": ", ".join(_order(order) for order in record.get("orders") or [] if isinstance(order, dict)),
            "Error": str(record.get("error") or ""),
        }
        for record in records
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def _holdings(record: dict[str, Any]) -> list[str]:
    if "holdings" in record:  # bill_ackman
        return _symbols(record["holdings"])
    return [stock["symbol"] for stock in record.get("debate_set") or [] if isinstance(stock, dict) and stock.get("held")]  # bull_bear


def _symbols(items: Any) -> list[str]:
    """Symbols from a list of tickers (bill_ackman) or of dicts carrying a `symbol` (bull_bear)."""
    symbols: list[str] = []
    for item in items or []:
        symbol = item.get("symbol") if isinstance(item, dict) else item
        if symbol:
            symbols.append(str(symbol))
    return symbols


def _targets(targets: Any) -> str:
    if not isinstance(targets, dict):
        return ""
    ranked = sorted(targets.items(), key=lambda item: (-float(item[1]), item[0]))
    return ", ".join(f"{symbol} {float(weight) * 100:.1f}%" for symbol, weight in ranked)


def _order(order: dict[str, Any]) -> str:
    return f"{str(order.get('side', '')).upper()} {order.get('symbol', '')} {order.get('quantity', '')}".strip()
