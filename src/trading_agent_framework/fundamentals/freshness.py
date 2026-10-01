"""The one freshness rule of the quality screen's on-disk caches (annual figures, split history).

A cached entry is stale when it was fetched more than `max_age_days` before `as_of`. `as_of` is the
caller's clock, so the same rule covers every mode: in a backtest `as_of` is simulated, and an entry
fetched today is fresh for every past date; in paper/live `as_of` is now, and entries refresh every
`max_age_days`. `fetched_at` is real time, but it is only ever compared with `as_of`, never with data.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def is_stale(fetched_at: datetime, as_of: datetime, max_age_days: int) -> bool:
    return fetched_at < as_of - timedelta(days=max_age_days)
