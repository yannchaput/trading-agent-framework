"""TODO item 3: paper-run judge book vs rank-only top 10.

The keep/drop rule is frozen in docs/superpowers/specs/2026-11-27-bull-bear-paper-judge-vs-top10-design.md
(sha256 of the rule text 5e2d0e04...).

Run from the repo root: uv run python scripts/bull_bear/paper_judge_vs_top10.py
"""

import math
import statistics
from datetime import date
from pathlib import Path

from trading_agent_framework.strategies.bull_bear.batch.agent_scores import forward_returns, newey_west_t, scored_reviews
from trading_agent_framework.strategies.bull_bear.batch.scores_cli import fetch_opens, read_reviews

MIN_REVIEWS = 13


def weekly_reviews():
    """Every paper review, the last one of each ISO week."""
    records = [r for path in sorted(Path("logs/bull_bear/paper").glob("*/reviews.jsonl")) for r in scored_reviews(read_reviews(path)[0])]
    by_week = {}
    for review in sorted(records, key=lambda r: r["date"]):
        by_week[date.fromisoformat(review["date"]).isocalendar()[:2]] = review
    return list(by_week.values())


def gaps(reviews, opens, calendar, horizon):
    """[(date, judge mean, top-10 mean)] for reviews whose exit is known."""
    rets = {}
    for row in forward_returns(reviews, opens, calendar, horizons=(horizon,)).rows:
        rets.setdefault(row.review, {})[row.symbol] = row.raw
    out = []
    for review in reviews:
        day = date.fromisoformat(review["date"])
        rows = rets.get(day)
        if not rows:
            continue
        top10 = [e["symbol"] for e in sorted(review["debate_set"], key=lambda e: e["rank"])[:10]]
        book, shadow = [rows[s] for s in review["book"] if s in rows], [rows[s] for s in top10 if s in rows]
        assert len(book) == len(review["book"]) and len(shadow) == len(top10), f"missing opens on {day}"
        out.append((day, statistics.mean(book), statistics.mean(shadow)))
    return out


def main():
    """Print each weekly gap, the t-stats and what the frozen rule says."""
    reviews = weekly_reviews()
    opens, calendar = fetch_opens([reviews], today=date.today())
    print(f"weekly paper reviews: {len(reviews)} ({reviews[0]['date']} .. {reviews[-1]['date']})")
    for horizon in (5, 20):
        rows = gaps(reviews, opens, calendar, horizon)
        print(f"\n== {horizon} sessions: {len(rows)} scored ==")
        for day, j, t in rows:
            print(f"  {day}  judge {j:+7.2%}  top10 {t:+7.2%}  gap {j - t:+7.2%}")
        if len(rows) < 2:
            continue
        diffs = [j - t for _, j, t in rows]
        t_stat = newey_west_t(diffs, lags=max(0, math.ceil(horizon / 5) - 1))
        print(f"  mean gap {statistics.mean(diffs):+.2%}  t {t_stat:+.2f}" if t_stat is not None else f"  mean gap {statistics.mean(diffs):+.2%}  t n/a")
        if horizon == 5:
            judge_c, top_c = math.prod(1 + j for _, j, _ in rows) - 1, math.prod(1 + t for _, _, t in rows) - 1
            print(f"  compounded: judge {judge_c:+.2%}  top10 {top_c:+.2%}")
            passes = statistics.mean(diffs) > 0 and t_stat is not None and t_stat >= 1 and judge_c > top_c
            status = "FINAL" if len(rows) >= MIN_REVIEWS else f"INTERIM ({len(rows)}/{MIN_REVIEWS} reviews, not a verdict)"
            print(f"  rule: {'keep judge' if passes else 'rank-only'}  [{status}]")


if __name__ == "__main__":
    main()
