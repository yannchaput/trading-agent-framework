"""Spike: does a debit put spread opened after a bad earnings reaction make money?

Design (rules fixed before running): docs/superpowers/specs/2026-11-23-earnings-short-side-spike-design.md

Run:
  uv run python scripts/earnings/short_side_spike.py check   # self-check on synthetic data
  uv run python scripts/earnings/short_side_spike.py dev     # 2016 -> 2023: run, read
  uv run python scripts/earnings/short_side_spike.py test    # 2024 -> last session, ONCE, chosen cell only
"""

import contextlib
import io
import pickle
import subprocess
import sys
from bisect import bisect_left
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import parse_yahoo_frame
from trading_agent_framework.fundamentals.sec import older_pages
from trading_agent_framework.strategies.common.universe_history import UniverseHistory
from trading_agent_framework.strategies.earnings_drift.events import earnings_events, reaction_date
from trading_agent_framework.strategies.earnings_drift.reaction import ReactionFeatures, reaction_features
from trading_agent_framework.utils.errors import FundamentalsNotFoundError


def _main_checkout() -> Path:
    """Root of the main checkout: the git-ignored inputs (cache/, data/universe/, env/) live there, not in a worktree."""
    here = Path(__file__).resolve().parent
    common = subprocess.run(["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, check=True)
    return Path(common.stdout.strip()).parent


MAIN = _main_checkout()
CACHE = MAIN / "cache" / "earnings_short_side"
BARS_DIR = MAIN / "cache" / "universe_history" / "bars"
START = 5000.0  # the fake account
SLOTS = 5
FRACTION = 0.20  # of equity per new position
COST = 0.008  # 80 bps per side: the put spread's cost (the bar reads this)
INFO_COST = 0.001  # 10 bps per side: the short stock's cost, shown for information only
CAP = 0.9  # the spread's short put sits 10% lower, so a short stock's gain is capped at 10%
WINDOWS = {"dev": (date(2016, 1, 4), date(2023, 12, 29)), "test": (date(2024, 1, 2), date.today() - timedelta(days=1))}
GRID = [(d, h) for d in (0.05, 0.08) for h in (5, 10, 20)]
CELLS = {f"D{d:.0%} H{h}": (d, h) for d, h in GRID}
ET = ZoneInfo("America/New_York")


def bars_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Daily bars from a cached Yahoo frame (`auto_adjust=False`), indexed by session close (tz-aware, as `parse_yahoo_frame`).

    Columns: `open high low close volume` (split-adjusted, for the reaction features), `tr_open tr_close` (total
    return: `Adj Close` and `Open x Adj Close / Close`, for P&L), `traded_open traded_close` (what the stock printed
    that day, for price levels and strikes) and `day` (the session date).
    """
    raw = raw.sort_index()
    raw = raw[raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].notna().all(axis=1)]
    parsed = parse_yahoo_frame(raw)
    traded_close = parsed["traded_close"].to_numpy()
    bars = pd.DataFrame(
        {
            "open": raw["Open"].to_numpy(),
            "high": raw["High"].to_numpy(),
            "low": raw["Low"].to_numpy(),
            "close": raw["Close"].to_numpy(),
            "volume": raw["Volume"].to_numpy(),
            "tr_open": (raw["Open"] * raw["Adj Close"] / raw["Close"]).to_numpy(),
            "tr_close": raw["Adj Close"].to_numpy(),
            "traded_open": raw["Open"].to_numpy() * traded_close / raw["Close"].to_numpy(),
            "traded_close": traded_close,
        },
        index=parsed.index,
    )
    bars["day"] = [stamp.date() for stamp in bars.index]
    return bars


def load_bars(symbol: str) -> pd.DataFrame | None:
    """The cached bars of `symbol` (see `bars_from_raw`); None when the cache has none."""
    path = BARS_DIR / f"{symbol}.parquet"
    return bars_from_raw(pd.read_parquet(path)) if path.exists() else None


def qualifies(f: ReactionFeatures, traded_close: float, drop: float) -> bool:
    """The spec's signal: a drop of at least `drop` against SPY, no bounce, heavy volume, liquid and not a penny stock."""
    return f.abnormal_pct <= -drop and f.close_location < 0.5 and f.rel_volume >= 2.0 and traded_close >= 10 and f.dollar_volume_20d >= 20_000_000


def dedupe_by_cik(events: list[dict], month_list: list[str]) -> list[dict]:
    """One event per (cik, accession): of several share classes, the symbol earliest in the month's list."""
    rank = {symbol: i for i, symbol in enumerate(month_list)}
    best: dict[tuple[str, str], dict] = {}
    for event in events:
        key = (event["cik"], event["accession"])
        if key not in best or rank.get(event["symbol"], len(rank)) < rank.get(best[key]["symbol"], len(rank)):
            best[key] = event
    return list(best.values())


def exit_index(days: list[date], sessions: list[date], entry_day: date, hold: int) -> int:
    """Position in a stock's own bar dates `days` of its exit bar.

    The bar on the `hold`-th SPY session counted from `entry_day` (entry = 1), else the first later bar, else the last
    bar. A target past the last session is the last session.
    """
    target = sessions[min(sessions.index(entry_day) + hold - 1, len(sessions) - 1)]
    return min(bisect_left(days, target), len(days) - 1)


def trade_value(ratio: float, cost: float) -> float:
    """Value at the exit of 1 unit of notional in the capped -1x position; `ratio` = exit price / entry price (total return).

    The gain stops at 10% (`CAP`), the value floors at 0 and `cost` is paid on each side.
    """
    return max(0.0, (1 - cost) * (2 - max(ratio, CAP)) * (1 - cost))


def find_events(sessions: list[date], universe: UniverseHistory, bars: dict[str, pd.DataFrame], client, cache: Path | None = None) -> list[dict]:
    """Every item 2.02 event of a universe symbol with its reaction-day features, not yet filtered by `qualifies`.

    An event counts when its symbol is in the list of its reaction day's month; entry day = the next SPY session.
    Share classes on one filing count once. `client` is a `SecEdgarClient`; `bars` must hold "SPY". The result is
    pickled at `cache` (when given) so `dev` and `test` do not walk SEC twice.
    """
    if cache is not None and cache.exists():
        return pickle.loads(cache.read_bytes())
    spy = bars["SPY"]
    next_session = dict(zip(sessions, sessions[1:], strict=False))
    counts = {"symbols": 0, "no cik": 0, "events": 0}
    by_day: dict[date, list[dict]] = {}
    for symbol in universe.union(sessions[0], sessions[-1]):
        stock = bars.get(symbol)
        if stock is None:
            continue
        counts["symbols"] += 1
        try:
            cik = client.ticker_to_cik(symbol)
        except FundamentalsNotFoundError:
            counts["no cik"] += 1
            continue
        recent = client.get_submissions_payload(cik)
        pages = [client.get_submissions_page(name) for name in older_pages(recent, date(2015, 1, 1))]
        traded = dict(zip(stock["day"], stock["traded_close"], strict=True))
        for event in earnings_events(symbol, [recent, *pages]):
            if event.accepted_at.astimezone(ET).date() < sessions[0]:
                continue
            day = reaction_date(event.accepted_at, sessions)
            if day is None or day not in next_session or symbol not in universe.on(day):
                continue
            f = reaction_features(stock, spy, day)
            if f is None:
                continue
            counts["events"] += 1
            by_day.setdefault(day, []).append(
                {
                    "symbol": symbol,
                    "cik": cik,
                    "accession": event.accession_number,
                    "reaction_day": day,
                    "entry_day": next_session[day],
                    "abnormal": f.abnormal_pct,
                    "features": f,
                    "traded_close": traded[day],
                }
            )
    events = [e for day in sorted(by_day) for e in sorted(dedupe_by_cik(by_day[day], universe.on(day)), key=lambda e: e["symbol"])]
    print(f"events: symbols {counts['symbols']} no cik {counts['no cik']} events {counts['events']} kept {len(events)}")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(events))
    return events


def raw_frame(dates: list[date], closes: list[float], volumes: list[float] | None = None, splits: dict[int, float] | None = None, adj_factor: float = 1.0) -> pd.DataFrame:
    """A Yahoo-shaped raw frame (`Open High Low Close Adj Close Volume Dividends Stock Splits`, naive date index)."""
    n = len(dates)
    close = pd.Series(closes, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close.values,
            "High": (close * 1.01).values,
            "Low": (close * 0.99).values,
            "Close": close.values,
            "Adj Close": (close * adj_factor).values,
            "Volume": volumes or [1e6] * n,
            "Dividends": [0.0] * n,
            "Stock Splits": [(splits or {}).get(i, 0.0) for i in range(n)],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def features(**kw: float) -> ReactionFeatures:
    """A ReactionFeatures with the fields the signal reads set from `kw` and the rest neutral."""
    base = dict(abnormal_pct=-0.06, close_location=0.3, rel_volume=2.5, dollar_volume_20d=25e6)
    base.update(kw)
    return ReactionFeatures(gap_pct=0.0, return_pct=-0.06, hold_ratio=None, runup_20d_pct=None, runup_60d_pct=None, atr14_pct=None, close=12.0, reaction_low=11.0, **base)


class FakeClient:
    """The slice of SecEdgarClient `find_events` uses, over in-memory payloads keyed by CIK."""

    def __init__(self, ciks: dict[str, str], payloads: dict[str, dict]) -> None:
        """Hold the symbol -> CIK map and the CIK -> submissions payload map."""
        self.ciks, self.payloads = ciks, payloads

    def ticker_to_cik(self, symbol: str) -> str:
        """The CIK of `symbol`; raises like the real client when unknown."""
        if symbol not in self.ciks:
            raise FundamentalsNotFoundError(f"no cik for {symbol}")
        return self.ciks[symbol]

    def get_submissions_payload(self, cik: str) -> dict:
        """The recent-filings payload of `cik`."""
        return self.payloads[cik]

    def get_submissions_page(self, name: str) -> dict:
        """No older pages in the fake."""
        return {}


def filing(accepted_utc: str, accession: str) -> dict:
    """A submissions payload with one 8-K item 2.02 accepted at `accepted_utc`."""
    recent = {"form": ["8-K"], "items": ["2.02,9.01"], "acceptanceDateTime": [accepted_utc], "accessionNumber": [accession], "primaryDocument": ["x.htm"]}
    return {"filings": {"recent": recent, "files": []}}


def check() -> None:
    """Self-check on synthetic data: no network, no files."""
    et = ZoneInfo("America/New_York")
    # 1. reaction day: a release at 16:05 ET on a Monday reacts Tuesday; at 08:00 ET reacts Monday
    week = [date(2024, 1, 8) + pd.Timedelta(days=i) for i in range(5)]
    assert reaction_date(datetime(2024, 1, 8, 16, 5, tzinfo=et), week) == date(2024, 1, 9)
    assert reaction_date(datetime(2024, 1, 8, 8, 0, tzinfo=et), week) == date(2024, 1, 8)
    # 2. qualifies
    assert qualifies(features(), 12.0, 0.05) and not qualifies(features(), 12.0, 0.08)
    assert not qualifies(features(close_location=0.5), 12.0, 0.05)
    assert not qualifies(features(rel_volume=1.99), 12.0, 0.05)
    assert not qualifies(features(), 9.99, 0.05)
    assert not qualifies(features(dollar_volume_20d=19.9e6), 12.0, 0.05)
    # 3. trade_value: gain capped at 10%, floor at 0, cost on both sides
    assert abs(trade_value(0.95, 0.0) - 1.05) < 1e-12 and abs(trade_value(0.70, 0.0) - 1.10) < 1e-12
    assert abs(trade_value(1.10, 0.0) - 0.90) < 1e-12 and trade_value(2.5, 0.0) == 0.0
    assert abs(trade_value(0.95, 0.008) - 1.05 * 0.992**2) < 1e-12
    # 4. exit_index: the hold-th SPY session counted from entry (entry = 1), else the first later bar, else the last
    sessions = [date(2024, 1, 2) + pd.Timedelta(days=i) for i in range(10)]
    assert exit_index(sessions, sessions, sessions[0], 5) == 4
    missing = [d for i, d in enumerate(sessions) if i != 4]
    assert missing[exit_index(missing, sessions, sessions[0], 5)] == sessions[5]
    assert exit_index(sessions[:3], sessions, sessions[0], 5) == 2
    assert exit_index(sessions, sessions, sessions[8], 5) == 9
    # 5. dedupe_by_cik: one event per (cik, accession), the symbol earliest in the month's list
    pair = [{"symbol": "GOOG", "cik": "1", "accession": "a"}, {"symbol": "GOOGL", "cik": "1", "accession": "a"}, {"symbol": "X", "cik": "2", "accession": "b"}]
    kept = dedupe_by_cik(pair, ["GOOGL", "X", "GOOG"])
    assert [e["symbol"] for e in kept] == ["GOOGL", "X"]
    # 6. bars_from_raw: a 2:1 split on day 2 and a dividend (adj factor 0.9)
    days = [date(2024, 1, 2) + pd.Timedelta(days=i) for i in range(3)]
    raw = raw_frame(days, [50.0, 51.0, 52.0], splits={2: 2.0}, adj_factor=0.9)
    bars = bars_from_raw(raw)
    assert abs(bars.traded_close.iloc[0] - 100.0) < 1e-9 and abs(bars.traded_close.iloc[2] - 52.0) < 1e-9
    assert abs(bars.traded_open.iloc[0] - 100.0) < 1e-9
    assert abs(bars.tr_close.iloc[0] - 45.0) < 1e-9 and abs(bars.tr_open.iloc[0] - 45.0) < 1e-9
    assert abs(bars.close.iloc[0] - 50.0) < 1e-9 and bars.day.iloc[0] == days[0]
    # 7. find_events: universe month, entry day, unknown ticker
    sess = [d.date() for d in pd.bdate_range("2023-01-02", periods=40)]
    prices = [100.0 + (i % 3) for i in range(40)]
    spy = bars_from_raw(raw_frame(sess, prices))
    book = {s: bars_from_raw(raw_frame(sess, prices)) for s in ("AAA", "BBB", "CCC")}
    book["SPY"] = spy
    months = {"2023-01": ("AAA", "BBB", "CCC"), "2023-02": ("AAA",)}
    universe = UniverseHistory(built=date(2023, 3, 1), pool_date=date(2023, 3, 1), top_n=3, min_price=10.0, min_dollar_volume=2e7, lookback=20, months=months)
    client = FakeClient({"AAA": "1", "BBB": "2"}, {"1": filing("2023-02-13T21:05:00.000Z", "a1"), "2": filing("2023-02-14T14:00:00.000Z", "b1")})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        events = find_events(sess, universe, book, client)
    assert [(e["symbol"], e["reaction_day"], e["entry_day"]) for e in events] == [("AAA", date(2023, 2, 14), date(2023, 2, 15))], events
    assert "no cik 1" in out.getvalue(), out.getvalue()
    print("self-check OK")


def main() -> None:
    """Entry point: `check` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
