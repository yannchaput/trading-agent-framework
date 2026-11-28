"""Entry gates (earnings reaction, gross profitability) on the rank-only book, under bull_bear's book rules.

Spec: docs/superpowers/specs/2026-11-28-earnings-profitability-gates-design.md

Run from the repo root: uv run python scripts/bull_bear/entry_gates.py [--self-check | --fetch] (fetch before the first run)
"""

import statistics
import sys
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd
import score_variants
from dotenv import load_dotenv
from rank_only_stress import CACHE, SLEEVE, TOP_N, WEIGHTS, load_data, simulate
from score_variants import book_picks, passes, ranked, table, window_stats

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient, default_client
from trading_agent_framework.fundamentals.sec import INCOME_STATEMENT_TAGS, older_pages, unit_rows
from trading_agent_framework.strategies.bill_ackman.screen.annual_figures import MAX_FISCAL_YEAR_DAYS, MIN_FISCAL_YEAR_DAYS
from trading_agent_framework.strategies.common.universe_history import month_universe
from trading_agent_framework.strategies.earnings_drift.events import earnings_events, reaction_date
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import FundamentalsError, FundamentalsNotFoundError

Row = tuple[date, date, date, float]  # period start, period end, filed, value (USD)
Instant = tuple[date, date, float]  # period end, filed, value (USD)
Signals = dict[str, list]  # "reactions": (usable from, return), "gross": Row, "assets": Instant
Reader = Callable[[str], tuple[list[Mapping[str, Any]], dict[str, Any]]]  # symbol -> (submissions payloads, us-gaap facts)
SEC_CACHE = Path("cache/sec")
ENV_FILE = "env/.env.earnings_drift_baseline.backtesting"  # holds SEC_EDGAR_USER_AGENT
FILINGS_SINCE = date(2004, 8, 1)  # 8-K item 2.02 starts here
GATES = {  # gate: (development, holdout); company facts are too thin for profitability before 2012
    "earnings": (score_variants.DEV, score_variants.HOLDOUT),
    "profitability": ((date(2012, 1, 1), date(2018, 12, 31)), (date(2019, 1, 1), date(2026, 11, 24))),
}
STALE_SESSIONS = 126  # an earnings reaction older than this counts as missing
STALE_DAYS = 548  # 18 months: an annual report older than this counts as missing
ANNUAL_FORMS = frozenset({"10-K"})
REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
COST_TAGS = tuple(INCOME_STATEMENT_TAGS["cost_of_revenue"])


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
    """The 10-K rows of `tag` that cover a fiscal year: quarters and year-to-date rows are left out."""
    rows = []
    for fact in unit_rows(gaap, tag, "USD", ANNUAL_FORMS):
        if fact.get("start"):
            start, end = date.fromisoformat(fact["start"]), date.fromisoformat(fact["end"])
            if MIN_FISCAL_YEAR_DAYS <= (end - start).days <= MAX_FISCAL_YEAR_DAYS:
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
        return {row[:3]: row[3] for tag in reversed(tags) for row in annual_rows(gaap, tag)}  # the first tag wins

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
    if (day - filed).days > STALE_DAYS or not balance:
        return None
    total = max(balance)[1]  # the latest-filed figure
    return value / total if total > 0 else None


def blocked(gate: str, values: dict[str, float | None]) -> frozenset[str]:
    """The names `gate` refuses entry to: a negative earnings reaction, or the bottom third of known profitability.

    A missing value is never blocked.
    """
    known = {symbol: value for symbol, value in values.items() if value is not None}
    if gate == "earnings":
        return frozenset(symbol for symbol, value in known.items() if value < 0)
    assert gate == "profitability", gate
    return frozenset(sorted(known, key=lambda symbol: (known[symbol], symbol))[: len(known) // 3])


def signals(symbols: list[str], frames: dict[str, pd.DataFrame], sessions: list[date], read: Reader) -> dict[str, Signals]:
    """Each symbol's earnings reactions (sorted by usable-from session) and its annual gross-profit and asset rows.

    A symbol without bars or without cached filings gets empty lists: its signals are missing.
    """
    spy = frames["SPY"]["close"].to_dict()
    found = {}
    for symbol in symbols:
        payloads, gaap = read(symbol)
        closes = frames[symbol]["close"].to_dict() if symbol in frames else {}
        reactions = []
        for event in earnings_events(symbol, payloads):
            first = bisect_left(sessions, event.accepted_at.astimezone(MARKET_TZ).date())
            day = reaction_date(event.accepted_at, sessions[first : first + 2])  # the filing day or the next session
            one = reaction(closes, spy, sessions, day) if day is not None else None
            if one is not None:
                reactions.append(one)
        found[symbol] = {"reactions": sorted(reactions), "gross": gross_profit_rows(gaap), "assets": instant_rows(gaap, "Assets")}
    return found


def cached_reader() -> Reader:
    """A reader of what `fetch` left in `SEC_CACHE`: the SEC client with a transport that answers 404 to any request."""
    offline = httpx.MockTransport(lambda request: httpx.Response(404))
    client = SecEdgarClient("offline cache reader", SEC_CACHE, transport=offline)

    def read(symbol: str) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
        payloads: list[Mapping[str, Any]] = []
        gaap: dict[str, Any] = {}
        try:
            cik = client.ticker_to_cik(symbol)
        except FundamentalsNotFoundError:
            return payloads, gaap
        with suppress(FundamentalsNotFoundError):
            recent = client.get_submissions_payload(cik)
            payloads = [recent, *(client.get_submissions_page(name) for name in older_pages(recent, FILINGS_SINCE))]
        with suppress(FundamentalsNotFoundError):
            gaap = client.get_company_facts_payload(cik).get("facts", {}).get("us-gaap", {})
        return payloads, gaap

    return read


def fetch(symbols: list[str]) -> None:
    """Download each symbol's filing list (with the older pages back to `FILINGS_SINCE`) and company facts into `SEC_CACHE`.

    Cached files younger than 7 days are kept. A failure is printed and counted, never raised.
    """
    load_dotenv(ENV_FILE)
    done = {"mapped": 0, "submissions": 0, "facts": 0}
    failures: list[str] = []
    with default_client(Path(".")) as client:
        for number, symbol in enumerate(symbols, 1):
            now = datetime.now(UTC)
            try:
                cik = client.ticker_to_cik(symbol)
            except FundamentalsNotFoundError:
                failures.append(f"{symbol}: no CIK")
                continue
            done["mapped"] += 1
            try:
                recent = client.get_submissions_payload(cik, as_of=now, max_age_days=7)
                for name in older_pages(recent, FILINGS_SINCE):
                    client.get_submissions_page(name)
                done["submissions"] += 1
            except FundamentalsError as exc:
                failures.append(f"{symbol} submissions: {exc}")
            try:
                client.get_company_facts_payload(cik, as_of=now, max_age_days=7)
                done["facts"] += 1
            except FundamentalsError as exc:
                failures.append(f"{symbol} facts: {exc}")
            if number % 100 == 0:
                print(f"{number}/{len(symbols)} {done}, {len(failures)} failures", flush=True)
    print("\n".join(failures))
    print(f"fetched {len(symbols)} symbols: {done}, {len(failures)} failures", flush=True)


def signal_values(gate: str, symbols: list[str], day: date, series: dict[str, Signals], sessions: list[date]) -> dict[str, float | None]:
    """Each symbol's `gate` signal as known before a trade on `day`; None when missing."""
    empty: Signals = {"reactions": [], "gross": [], "assets": []}
    if gate == "earnings":
        return {s: earnings_value(series.get(s, empty)["reactions"], day, sessions) for s in symbols}
    assert gate == "profitability", gate
    return {s: gpa_value(series.get(s, empty)["gross"], series.get(s, empty)["assets"], day) for s in symbols}


def run(gate: str, apply: bool, first: date, last: date, data: tuple, series: dict[str, Signals], orders: dict) -> tuple[pd.Series, dict[str, float], dict]:
    """A fresh book from `first` to `last`, gated when `apply`: (equity, stats, {year: [names with no signal, names]}).

    `gate` is also given for the ungated book, so that both report the same signal's coverage.
    """
    history, frames, sessions = data
    window = [day for day in sessions if first <= day <= last]
    books: list[list[str]] = []
    refused = [0, 0]  # blocked, non-holdings among the top `retention_rank`
    missing: dict[int, list[int]] = {}

    def picks(day: date, held: set[str]) -> list[str]:
        key = (day, frozenset(held))  # holdings are exempt from the volatility filter, so the order depends on them
        if key not in orders:
            orders[key] = ranked(day, held, month_universe(history, day, exclude={SLEEVE}), frames, sessions, "today")
        order = orders[key]
        values = signal_values(gate, order, day, series, sessions)
        refusals = blocked(gate, values)
        candidates = [s for s in order[: score_variants.BOOK.retention_rank] if s not in held]
        refused[0] += len(refusals & set(candidates))
        refused[1] += len(candidates)
        year = missing.setdefault(day.year, [0, 0])
        year[0] += sum(value is None for value in values.values())
        year[1] += len(values)
        books.append(book_picks(order, held, blocked=refusals if apply else frozenset()))  # only new entries are refused
        return books[-1]

    equity = simulate(frames, window, picks, WEIGHTS[0])
    stats = window_stats(equity)
    stats["turnover"] = statistics.mean(len(set(now) - set(before)) / TOP_N for before, now in pairwise(books))
    stats["held"] = statistics.mean(len(book) for book in books)
    stats["blocked"] = refused[0] / refused[1]
    stats["missing"] = sum(m for m, _ in missing.values()) / sum(n for _, n in missing.values())
    return equity, stats, missing


def compare(gate: str, title: str, window: tuple[date, date], data: tuple, series: dict[str, Signals], orders: dict, curves: dict) -> tuple:
    """Run the ungated and the gated book on `window`, print their table and return (ungated stats, gated stats)."""
    results, coverage = {}, {}
    for name, apply in (("today", False), (gate, True)):
        curves[f"{title} {name}"], results[name], coverage[name] = run(gate, apply, *window, data, series, orders)
    table(title, results)
    by_year = sorted(coverage[gate].items())
    print("Names with no signal, by year (gated book): " + ", ".join(f"{year} {m / n:.0%}" for year, (m, n) in by_year), flush=True)
    return results["today"], results[gate]


def check_baseline(stats: dict[str, float]) -> None:
    """The ungated 2005-2015 book must be the score-variant study's `today` book (`score_variants.py --book-rules`)."""
    curves = pd.read_csv(CACHE / "score_variants_book_rules_equity.csv", index_col=0)
    saved = cast("pd.Series", curves["Real book rules, 2005-2015 today"]).dropna()
    saved.index = pd.Index([date.fromisoformat(str(day)) for day in saved.index])
    expected = window_stats(saved)
    for key, tolerance in (("cagr", 0.001), ("sharpe", 0.005)):
        assert abs(stats[key] - expected[key]) < tolerance, f"ungated {key} {stats[key]:.4f}, score-variant study {expected[key]:.4f}"
    print(f"Baseline check: CAGR {stats['cagr']:.2%} and Sharpe {stats['sharpe']:.2f} match the score-variant study", flush=True)


def main() -> None:
    """For each gate: development, then its holdout only if it passed; print the verdicts and save the curves."""
    data = load_data()
    _, frames, sessions = data
    series = signals(sorted(set(frames) - {"SPY", SLEEVE}), frames, sessions, cached_reader())
    curves: dict[str, pd.Series] = {}
    orders: dict = {}
    verdicts = {}
    for gate, (dev, holdout) in GATES.items():
        base, gated = compare(gate, f"{gate}, development {dev[0].year}-{dev[1].year}", dev, data, series, orders, curves)
        if dev == score_variants.DEV:
            check_baseline(base)
        if not passes(gated, base):
            verdicts[gate] = "FAILED development"
            continue
        base, gated = compare(gate, f"{gate}, holdout {holdout[0].year}-{holdout[1].year}", holdout, data, series, orders, curves)
        verdicts[gate] = "KEPT" if passes(gated, base) else "FAILED holdout"
    print("\n" + "\n".join(f"{gate}: {verdict}" for gate, verdict in verdicts.items()), flush=True)
    pd.DataFrame(curves).sort_index().to_csv(CACHE / "entry_gates_equity.csv")


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

    def filings(*rows: tuple[str, str, str]) -> dict[str, Any]:
        columns = {"form": [r[0] for r in rows], "items": [r[1] for r in rows], "acceptanceDateTime": [r[2] for r in rows]}
        return {"filings": {"recent": {**columns, "accessionNumber": [f"A{i}" for i in range(len(rows))]}}}

    payload = filings(
        ("8-K", "2.02,9.01", "2020-01-10T21:05:00.000Z"),  # 16:05 ET: reacts on the 11th
        ("8-K", "2.02", "2020-01-20T13:00:00.000Z"),  # 08:00 ET: reacts the same day
        ("8-K/A", "2.02", "2020-01-30T13:00:00.000Z"),
        ("10-Q", "", "2020-02-01T13:00:00.000Z"),
    )
    frames = {"A": pd.DataFrame({"close": [100.0 + i for i in range(200)]}, index=d), "SPY": pd.DataFrame({"close": 100.0}, index=d)}
    got_signals = signals(["A", "NOBARS"], frames, d, lambda symbol: ([payload], k) if symbol == "A" else ([], {}))
    # d[10] is 2020-01-11: usable two sessions later, return from close(d[8]) to close(d[11]).
    assert [day for day, _ in got_signals["A"]["reactions"]] == [d[12], d[21]], got_signals["A"]["reactions"]
    assert abs(got_signals["A"]["reactions"][0][1] - (111 / 108 - 1)) < 1e-12
    assert got_signals["A"]["gross"] == gross and got_signals["A"]["assets"] == assets
    assert got_signals["NOBARS"] == {"reactions": [], "gross": [], "assets": []}
    # The rank-1 name had a bad release: the earnings book refuses it entry, but never sells it.
    trade_day, order = d[22], ["BAD", "GOOD", "NONE"]
    series = {"BAD": {"reactions": [(d[12], -0.05)], "gross": [], "assets": []}, "GOOD": {"reactions": [(d[12], 0.02)], "gross": [], "assets": []}}
    values = signal_values("earnings", order, trade_day, series, d)
    assert values == {"BAD": -0.05, "GOOD": 0.02, "NONE": None}
    refusals = blocked("earnings", values)
    assert book_picks(order, set()) == order
    assert book_picks(order, set(), blocked=refusals) == ["GOOD", "NONE"]
    assert book_picks(order, {"BAD"}, blocked=refusals) == order == book_picks(order, {"BAD"})
    assert signal_values("profitability", ["GOOD"], date(2011, 6, 1), {"GOOD": {"reactions": [], "gross": gross, "assets": assets}}, d) == {"GOOD": 0.2}
    score_variants.self_check()


if __name__ == "__main__":
    self_check()
    if "--fetch" in sys.argv:
        fetch(sorted(set(load_data()[1]) - {"SPY", SLEEVE}))
    elif "--self-check" not in sys.argv:
        main()
