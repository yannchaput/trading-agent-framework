"""Entry gates (earnings reaction, gross profitability) on the rank-only book, under bull_bear's book rules.

Spec: docs/superpowers/specs/2026-11-28-earnings-profitability-gates-design.md

Run from the repo root: uv run python scripts/bull_bear/entry_gates.py [--self-check | --fetch]
"""

from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from typing import Any

import score_variants
from score_variants import book_picks

from trading_agent_framework.fundamentals.sec import unit_rows

Row = tuple[date, date, date, float]  # period start, period end, filed, value (USD)
Instant = tuple[date, date, float]  # period end, filed, value (USD)
STALE_SESSIONS = 126  # an earnings reaction older than this counts as missing
STALE_DAYS = 548  # 18 months: an annual report older than this counts as missing
ANNUAL_FORMS = frozenset({"10-K"})
REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
COST_TAGS = ("CostOfRevenue", "CostOfGoodsAndServicesSold")


def reaction(closes: dict[date, float], spy: dict[date, float], sessions: list[date], r: date) -> tuple[date, float] | None:
    """(first session it is usable, market-adjusted return) of the release reacting on `r`.

    The return runs from the close two sessions before `r` to the close of the one after (SEC acceptance times can
    trail the release by hours), so it is usable only from the session after that.
    """
    i = bisect_left(sessions, r)
    if i < 2 or i + 2 >= len(sessions):
        return None
    start, end = sessions[i - 2], sessions[i + 1]
    if any(day not in prices for day in (start, end) for prices in (closes, spy)):
        return None
    return sessions[i + 2], (closes[end] / closes[start] - 1) - (spy[end] / spy[start] - 1)


def earnings_value(reactions: list[tuple[date, float]], day: date, sessions: list[date]) -> float | None:
    """The latest reaction usable on `day`; None if there is none or it is older than `STALE_SESSIONS`.

    `reactions` is sorted by usable-from session; both it and `day` are entries of `sessions`.
    """
    count = bisect_right(reactions, day, key=lambda item: item[0])
    if count == 0:
        return None
    usable, value = reactions[count - 1]
    return value if bisect_left(sessions, day) - bisect_left(sessions, usable) <= STALE_SESSIONS else None


def annual_rows(gaap: dict[str, Any], tag: str) -> list[Row]:
    """The 10-K rows of `tag` that cover a fiscal year (350-380 days): quarters and year-to-date rows are left out."""
    rows = []
    for fact in unit_rows(gaap, tag, "USD", ANNUAL_FORMS):
        if fact.get("start"):
            start, end = date.fromisoformat(fact["start"]), date.fromisoformat(fact["end"])
            if 350 <= (end - start).days <= 380:
                rows.append((start, end, date.fromisoformat(fact["filed"]), float(fact["val"])))
    return rows


def instant_rows(gaap: dict[str, Any], tag: str) -> list[Instant]:
    """The 10-K balance-sheet rows of `tag`."""
    return [(date.fromisoformat(f["end"]), date.fromisoformat(f["filed"]), float(f["val"])) for f in unit_rows(gaap, tag, "USD", ANNUAL_FORMS)]


def gross_profit_rows(gaap: dict[str, Any]) -> list[Row]:
    """`GrossProfit` rows, plus revenue minus cost for the same period and filing where it has no such line.

    Decided per filing, never per period: a `GrossProfit` line that first appears in a later filing must not remove
    the figure that was already public.
    """

    def first_of(tags: tuple[str, ...]) -> dict[tuple[date, date, date], float]:
        found: dict[tuple[date, date, date], float] = {}
        for tag in reversed(tags):  # the first tag is written last, so it wins
            found.update({row[:3]: row[3] for row in annual_rows(gaap, tag)})
        return found

    reported = first_of(("GrossProfit",))
    revenue, cost = first_of(REVENUE_TAGS), first_of(COST_TAGS)
    derived = {key: revenue[key] - cost[key] for key in revenue.keys() & cost.keys()}
    return sorted((*key, value) for key, value in (derived | reported).items())


def gpa_value(gross: list[Row], assets: list[Instant], day: date) -> float | None:
    """Gross profit / assets of the latest fiscal year public before `day`; None if missing or over 548 days old."""
    public = [row for row in gross if row[2] < day]
    if not public:
        return None
    _, end, filed, value = max(public, key=lambda row: (row[1], row[2]))
    balance = [(f, v) for e, f, v in assets if e == end and f < day]
    if (day - filed).days > STALE_DAYS or not balance or max(balance)[1] <= 0:
        return None
    return value / max(balance)[1]


def blocked(gate: str, values: dict[str, float | None]) -> frozenset[str]:
    """The names `gate` refuses entry to: a negative earnings reaction, or the bottom third of known profitability.

    A missing value is never blocked.
    """
    known = {symbol: value for symbol, value in values.items() if value is not None}
    if gate == "earnings":
        return frozenset(symbol for symbol, value in known.items() if value < 0)
    assert gate == "profitability", gate
    return frozenset(sorted(known, key=lambda symbol: (known[symbol], symbol))[: len(known) // 3])


def self_check() -> None:
    """Assert the signals, the gates and the blocked fill rule on hand-worked cases."""
    d = [date(2020, 1, 1) + timedelta(days=i) for i in range(200)]
    first = d[:6]
    closes = dict(zip(first, [100.0, 100.0, 100.0, 110.0, 110.0, 110.0], strict=True))
    flat = dict.fromkeys(first, 100.0)
    got = reaction(closes, flat, first, first[2])
    assert got is not None and got[0] == first[4] and abs(got[1] - 0.10) < 1e-12, got
    rising = dict(zip(first, [100.0, 101.0, 102.0, 105.0, 105.0, 105.0], strict=True))
    got = reaction(closes, rising, first, first[2])
    assert got is not None and abs(got[1] - 0.05) < 1e-12, got
    assert reaction(closes, flat, first, first[4]) is None  # no session after r+1
    assert reaction(closes, flat, first, first[1]) is None  # no r-2
    gap = {day: value for day, value in closes.items() if day != first[3]}
    assert reaction(gap, flat, first, first[2]) is None

    events = [(d[4], -0.02), (d[39], 0.03)]
    assert earnings_value(events, d[39], d) == 0.03
    assert earnings_value(events, d[38], d) == -0.02
    assert earnings_value(events, d[3], d) is None
    assert earnings_value([(d[0], 0.01)], d[126], d) == 0.01
    assert earnings_value([(d[0], 0.01)], d[127], d) is None

    def fact(start: str | None, end: str, filed: str, val: float, form: str = "10-K") -> dict[str, Any]:
        row = {"end": end, "filed": filed, "val": val, "form": form}
        return row if start is None else {**row, "start": start}

    def usd(*rows: dict[str, Any]) -> dict[str, Any]:
        return {"units": {"USD": list(rows)}}

    year = ("2010-01-01", "2010-12-31", "2011-02-15")
    gaap = {"GrossProfit": usd(fact(*year, 40), fact("2010-10-01", "2010-12-31", "2011-02-15", 11), fact(*year, 99, "10-Q"))}
    assert annual_rows(gaap, "GrossProfit") == [(date(2010, 1, 1), date(2010, 12, 31), date(2011, 2, 15), 40)]
    derived = {"Revenues": usd(fact(*year, 100)), "CostOfRevenue": usd(fact(*year, 60))}
    assert gross_profit_rows(derived) == [(date(2010, 1, 1), date(2010, 12, 31), date(2011, 2, 15), 40)]
    assert gross_profit_rows({**derived, "GrossProfit": usd(fact(*year, 41))}) == [(date(2010, 1, 1), date(2010, 12, 31), date(2011, 2, 15), 41)]
    new_names = {"RevenueFromContractWithCustomerExcludingAssessedTax": derived["Revenues"], "CostOfGoodsAndServicesSold": derived["CostOfRevenue"]}
    assert gross_profit_rows(new_names) == gross_profit_rows(derived)
    # A GrossProfit tagged only in a later filing must not erase the figure derived from the earlier one (look-ahead).
    later = {**derived, "GrossProfit": usd(fact("2010-01-01", "2010-12-31", "2012-02-15", 50))}
    assert [row[2:] for row in gross_profit_rows(later)] == [(date(2011, 2, 15), 40), (date(2012, 2, 15), 50)]

    k = {
        "GrossProfit": usd(fact(*year, 40), fact("2010-01-01", "2010-12-31", "2012-02-15", 50), fact("2011-01-01", "2011-12-31", "2012-02-15", 60)),
        "Assets": usd(fact(None, "2010-12-31", "2011-02-15", 200), fact(None, "2011-12-31", "2012-02-15", 250)),
    }
    gross, assets = gross_profit_rows(k), instant_rows(k, "Assets")
    assert gpa_value(gross, assets, date(2011, 2, 15)) is None  # filed that day: not yet public
    assert gpa_value(gross, assets, date(2011, 6, 1)) == 0.2
    assert gpa_value(gross, assets, date(2012, 3, 1)) == 60 / 250
    assert gpa_value(gross, assets, date(2013, 9, 1)) is None  # older than 548 days
    assert gpa_value(gross, [(date(2010, 12, 31), date(2011, 2, 15), 0.0)], date(2011, 6, 1)) is None
    assert gpa_value(gross, [], date(2011, 6, 1)) is None

    assert blocked("earnings", {"A": -0.01, "B": 0.0, "C": None}) == {"A"}
    six = {"F": 0.6, "A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "N": None}
    assert blocked("profitability", six) == {"A", "B"}
    assert blocked("profitability", {"C": 0.1, "B": 0.1, "A": 0.1, "D": 0.5, "E": 0.6, "F": 0.7}) == {"A", "B"}

    assert book_picks(["A", "B", "C"], set(), top_n=2, blocked=frozenset({"A"})) == ["B", "C"]
    assert book_picks(["A", "B", "C"], {"A"}, top_n=2, blocked=frozenset({"A", "B"})) == ["A", "C"]
    assert book_picks(["A", "B"], set(), top_n=2, blocked=frozenset({"A", "B"})) == []
    score_variants.self_check()


if __name__ == "__main__":
    self_check()
