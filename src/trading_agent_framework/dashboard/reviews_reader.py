"""Read-only access to the per-run agent log of a backtesting run (one JSON line per strategy review/run).

`reviews.jsonl` is written by `bill_ackman` and `bull_bear`, `runs.jsonl` (`strategies/common/run_log.py`) by
`insider_trades` and `congress_trades`; the Agents tab of Run Detail shows whichever the run has.
The schemas differ, so the table is generic: one column per key found in the file, in first-seen order, and the page
shows the raw record next to it. Streamlit-free, like reader.py. Nothing here writes into a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_agent_framework.dashboard.models import RunRef

REVIEWS_FILE = "reviews.jsonl"
RUNS_FILE = "runs.jsonl"


@dataclass(frozen=True)
class ReviewsFile:
    """The records of one `reviews.jsonl`, plus how many lines could not be read."""

    records: list[dict[str, Any]]
    malformed: int


def reviews_path(ref: RunRef) -> Path | None:
    """The run's agent log: `reviews.jsonl`, else `runs.jsonl`, else None."""
    for name in (REVIEWS_FILE, RUNS_FILE):
        path = Path(ref.path) / name
        if path.is_file():
            return path
    return None


def has_reviews(ref: RunRef) -> bool:
    """Whether the run wrote an agent log (only the agent strategies do)."""
    return reviews_path(ref) is not None


def load_reviews(ref: RunRef) -> ReviewsFile:
    """Every record of the run's agent log in file order; a line that is not a JSON object is counted, not raised.

    Raises OSError (FileNotFoundError included) when the file cannot be read.
    """
    records: list[dict[str, Any]] = []
    malformed = 0
    path = reviews_path(ref)
    if path is None:
        raise FileNotFoundError(f"{ref.path} has neither {REVIEWS_FILE} nor {RUNS_FILE}")
    text = path.read_text(encoding="utf-8")
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
    """One row per record and one column per key found in any record, in first-seen order.

    A list or dict is shown as compact JSON, a key a record lacks is left empty, and a column whose values are not
    all of one scalar type becomes text (Arrow, which backs `st.dataframe`, rejects mixed columns).
    """
    columns: dict[str, list[Any]] = {}
    for record in records:
        for key in record:
            columns.setdefault(key, [])
    for record in records:
        for key, values in columns.items():
            values.append(_cell(record[key]) if key in record else None)
    return pd.DataFrame({key: _uniform(values) for key, values in columns.items()})


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=str)
    return value


def _uniform(values: list[Any]) -> list[Any]:
    """The values as they are when they share one type (int and float count as one), else as text."""
    kinds = {bool if isinstance(value, bool) else float if isinstance(value, (int, float)) else type(value) for value in values if value is not None}
    if len(kinds) <= 1:
        return values
    return [None if value is None else str(value) for value in values]
