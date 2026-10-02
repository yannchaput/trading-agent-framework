"""The freshness rule of the quality screen's annual-figures cache.

A cached record is stale when it was fetched more than `max_age_days` before `as_of`. `as_of` is the
caller's clock, so the rule covers every mode: in a backtest `as_of` is simulated, and a record
fetched today is fresh for every past date; in paper/live `as_of` is now, and records refresh every
`max_age_days`. `fetched_at` is real time, but it is only ever compared with `as_of`, never with data.

The rule is used by the annual store and by `SecEdgarClient`'s cached payload getters; split history is judged
by the wall clock instead (`splits.SplitHistory`), because it must match the basis of today's split-adjusted prices.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def is_stale(fetched_at: datetime, as_of: datetime, max_age_days: int) -> bool:
    return fetched_at < as_of - timedelta(days=max_age_days)
