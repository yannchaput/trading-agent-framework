"""Spike: does a debit put spread opened after a bad earnings reaction make money?

Design (rules fixed before running): docs/superpowers/specs/2026-11-23-earnings-short-side-spike-design.md

Run:
  uv run python scripts/earnings/short_side_spike.py check   # self-check on synthetic data
  uv run python scripts/earnings/short_side_spike.py dev     # 2016 -> 2023: run, read
  uv run python scripts/earnings/short_side_spike.py test    # 2024 -> last session, ONCE, chosen cell only
"""

import contextlib
import io
import json
import math
import os
import pickle
import subprocess
import sys
from bisect import bisect_left
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.models import OptionContractsResponse
from alpaca.trading.requests import GetOptionContractsRequest
from dotenv import dotenv_values

from trading_agent_framework.backtesting.data.yahoo import parse_yahoo_frame
from trading_agent_framework.fundamentals.edgar_client import default_client
from trading_agent_framework.fundamentals.sec import older_pages
from trading_agent_framework.strategies.common.universe_history import UniverseHistory
from trading_agent_framework.strategies.earnings_drift.events import earnings_events, reaction_date
from trading_agent_framework.strategies.earnings_drift.reaction import ReactionFeatures, reaction_features
from trading_agent_framework.utils.errors import FundamentalsNotFoundError
from trading_agent_framework.utils.helpers import in_market_fall


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
STEP2_START = date(2024, 2, 1)  # Alpaca's option bars start in February 2024
HALF_SPREAD = 0.004  # of the stock's traded open, per leg per side (one measured OPRA snapshot)
OPEN_FEE, LEG_FEE_CAP, SIDE_FEE = 1.00, 10.0, 0.13  # Tastytrade: $1 a contract to open (cap $10 a leg), $0 to close, $0.10 clearing + ~$0.03 regulatory a side
MAX_LOSING_YEARS = 2
MIN_TRADES = 30
ENV_FILE = MAIN / "env" / ".env.bull_bear.paper"  # borrowed for its read-only keys and SEC_EDGAR_USER_AGENT


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


def price_book(events: list[dict], bars: dict[str, pd.DataFrame]) -> dict[str, tuple[dict[date, tuple[float, float]], list[date]]]:
    """Per symbol of `events`: its total-return (open, close) by session date, and its bar dates in order."""
    book = {}
    for event in events:
        symbol = event["symbol"]
        if symbol not in book:
            frame = bars[symbol]
            book[symbol] = (dict(zip(frame["day"], zip(frame["tr_open"], frame["tr_close"], strict=True), strict=True)), list(frame["day"]))
    return book


def exit_of(event: dict, book: dict, sessions: list[date], hold: int) -> tuple[float, date, float] | None:
    """(entry total-return open, exit bar date, exit total-return close) of an event's trade; None without an entry bar."""
    prices, days = book[event["symbol"]]
    entry = prices.get(event["entry_day"])
    if entry is None:
        return None
    exit_date = days[exit_index(days, sessions, event["entry_day"], hold)]
    return entry[0], exit_date, prices[exit_date][1]


def qualifying(events: list[dict], sessions: list[date], drop: float) -> list[dict]:
    """The events that qualify at `drop` and enter inside the window `sessions`."""
    window = set(sessions)
    return [e for e in events if e["entry_day"] in window and qualifies(e["features"], e["traded_close"], drop)]


def gate_trades(events: list[dict], bars: dict[str, pd.DataFrame], sessions: list[date], drop: float, hold: int, cost: float, falls: set[date]) -> pd.DataFrame:
    """The signal gate: one capped -1x trade per qualifying event (no slot limit, equal weight), after `cost`."""
    chosen = qualifying(events, sessions, drop)
    book = price_book(chosen, bars)
    rows = []
    for event in chosen:
        leg = exit_of(event, book, sessions, hold)
        if leg is not None:
            entry_day = event["entry_day"]
            rows.append({"entry_day": entry_day, "exit_day": leg[1], "ret": trade_value(leg[2] / leg[0], cost) - 1, "year": entry_day.year, "in_fall": entry_day in falls})
    return pd.DataFrame(rows, columns=["entry_day", "exit_day", "ret", "year", "in_fall"])


def marked(position: dict, cost: float) -> float:
    """A position's value now: its stake after the entry cost, the exit cost not yet paid."""
    return position["alloc"] * trade_value(position["ratio"], cost) / (1 - cost)


def replay_stocks(events: list[dict], bars: dict[str, pd.DataFrame], sessions: list[date], drop: float, hold: int, cost: float, falls: set[date], slots: int = SLOTS) -> tuple[pd.Series, pd.DataFrame]:
    """The step-1 account: daily equity (at each close) and the trades, with the skip counts in `trades.attrs["skipped"]`.

    Per session: positions mark at the open, new entries (the largest drop first, ties by symbol) take free slots at
    20% of the equity at that open (capped by the cash), exits due are paid at the close, the close marks the rest.
    """
    chosen = qualifying(events, sessions, drop)
    book = price_book(chosen, bars)
    by_entry: dict[date, list[dict]] = {}
    for event in chosen:
        by_entry.setdefault(event["entry_day"], []).append(event)
    skipped = {"slots full": 0, "already held": 0, "no entry bar": 0}
    cash, held, trades, equity = START, [], [], []
    for day in sessions:
        for position in held:
            price = book[position["symbol"]][0].get(day)
            position["ratio"] = price[0] / position["entry"] if price else position["ratio"]
        open_equity = cash + sum(marked(p, cost) for p in held)
        for event in sorted(by_entry.get(day, []), key=lambda e: (e["abnormal"], e["symbol"])):
            leg = exit_of(event, book, sessions, hold)
            if len(held) >= slots:
                skipped["slots full"] += 1
            elif any(p["cik"] == event["cik"] for p in held):
                skipped["already held"] += 1
            elif leg is None:
                skipped["no entry bar"] += 1
            else:
                alloc = min(FRACTION * open_equity, cash)
                cash -= alloc
                exit_session = sessions[min(bisect_left(sessions, leg[1]), len(sessions) - 1)]
                held.append({"symbol": event["symbol"], "cik": event["cik"], "alloc": alloc, "entry": leg[0], "ratio": 1.0, "exit_px": leg[2], "exit_session": exit_session, "entry_day": day})
        for position in [p for p in held if p["exit_session"] == day]:
            value = position["alloc"] * trade_value(position["exit_px"] / position["entry"], cost)
            cash += value
            trades.append({"symbol": position["symbol"], "entry_day": position["entry_day"], "exit_day": day, "ret": value / position["alloc"] - 1, "in_fall": position["entry_day"] in falls})
            held.remove(position)
        for position in held:
            price = book[position["symbol"]][0].get(day)
            position["ratio"] = price[1] / position["entry"] if price else position["ratio"]
        equity.append(cash + sum(marked(p, cost) for p in held))
    frame = pd.DataFrame(trades, columns=["symbol", "entry_day", "exit_day", "ret", "in_fall"])
    frame.attrs["skipped"] = skipped
    return pd.Series(equity, index=sessions), frame


def summary(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Total, CAGR, calendar-year returns, max drawdown (negative) and the trade statistics of one replay."""
    curve = pd.concat([pd.Series([START]), equity.reset_index(drop=True)], ignore_index=True)
    total = float(curve.iloc[-1] / START - 1)
    year_end = equity.groupby([d.year for d in equity.index]).last()
    span = (equity.index[-1] - equity.index[0]).days + 1
    return {
        "total": total,
        "cagr": (1 + total) ** (365.25 / span) - 1,
        "years": (year_end / year_end.shift(1, fill_value=START) - 1).to_dict(),
        "max_dd": float((curve / curve.cummax() - 1).min()),
        "trades": len(trades),
        "win_rate": float((trades.ret > 0).mean()) if len(trades) else 0.0,
        "mean_trade": float(trades.ret.mean()) if len(trades) else 0.0,
        "skipped": trades.attrs.get("skipped", {}),
    }


def passes(s: dict, shv_total: float) -> bool:
    """The bar's account test: beats SHV buy and hold, at most 2 losing calendar years, at least 30 trades."""
    return s["total"] > shv_total and sum(y < 0 for y in s["years"].values()) <= MAX_LOSING_YEARS and s["trades"] >= MIN_TRADES


def choose(results: dict[str, dict], passed: set[str]) -> str | None:
    """The passing cell with the most passing neighbours (one grid value apart), then the higher total, then the smaller drawdown."""

    def neighbours(name: str) -> int:
        d, h = CELLS[name]
        return sum(other != name and (CELLS[other][0] == d) != (CELLS[other][1] == h) for other in passed)

    return min(passed, key=lambda n: (-neighbours(n), -results[n]["total"], -results[n]["max_dd"]), default=None)


def fall_days(closes: pd.DataFrame) -> pd.Series:
    """Whether `in_market_fall` is true before each session's open: row d is decided from rows before d (row 0 is False)."""
    vix, vix3m, vixy = closes.vix.tolist(), closes.vix3m.tolist(), closes.vixy.tolist()
    return pd.Series([False] + [in_market_fall(vix[:i], vix3m[:i], vixy[:i]) for i in range(1, len(closes))], index=closes.index)


def falls_before_open(sessions: list[date]) -> set[date]:
    """The sessions whose open follows an `in_market_fall` close (Yahoo ^VIX, ^VIX3M and VIXY from 2008)."""
    import yfinance as yf

    raw = yf.download(["^VIX", "^VIX3M", "VIXY"], start="2008-01-02", progress=False, auto_adjust=True)["Close"]
    raw = raw.rename(columns={"^VIX": "vix", "^VIX3M": "vix3m", "VIXY": "vixy"}).dropna(subset=["vix", "vix3m"])
    raw.index = raw.index.date
    flags = fall_days(raw[["vix", "vix3m", "vixy"]])
    return {d for d in sessions if flags.get(d, False)}


def buy_and_hold(symbol: str, start: date, end: date) -> float:
    """Total return of `symbol` (Yahoo, dividends included) from the first close on or after `start` to the last on or before `end`."""
    import yfinance as yf

    close = yf.download(symbol, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), progress=False, auto_adjust=True)["Close"].squeeze()
    return float(close.iloc[-1] / close.iloc[0] - 1)


def row(s: dict) -> str:
    """One report row for a summary."""
    return f"{s['total']:+8.1%} cagr {s['cagr']:+6.1%} dd {s['max_dd']:6.1%} trades {s['trades']:4d} win {s['win_rate']:4.0%} mean {s['mean_trade']:+.2%} skipped {s['skipped']}"


def window_inputs(window: str) -> tuple[list[date], dict[str, pd.DataFrame], list[dict]]:
    """The window's SPY sessions, the daily bars of its universe and its earnings events (SEC walk cached)."""
    start, end = WINDOWS[window]
    os.environ.setdefault("SEC_EDGAR_USER_AGENT", dotenv_values(ENV_FILE)["SEC_EDGAR_USER_AGENT"] or "")
    spy = load_bars("SPY")
    assert spy is not None
    sessions = [d for d in spy["day"] if start <= d <= end]
    universe = UniverseHistory.load(MAIN / "data" / "universe" / "us_stock_universe_history.json")
    assert universe is not None
    bars = {symbol: frame for symbol in universe.union(sessions[0], sessions[-1]) if (frame := load_bars(symbol)) is not None}
    bars["SPY"] = spy
    events = find_events(sessions, universe, bars, default_client(MAIN), CACHE / f"events_{start}_{end}.pkl")
    return sessions, bars, events


def report_dev() -> str | None:
    """Run the dev window, print every table and return the chosen cell (None: NO)."""
    start, end = WINDOWS["dev"]
    sessions, bars, events = window_inputs("dev")
    falls = falls_before_open(sessions)
    shv, spy = buy_and_hold("SHV", start, end), buy_and_hold("SPY", start, end)
    print(f"dev {sessions[0]} -> {sessions[-1]}, {len(sessions)} sessions, {len(falls)} fall sessions, {len(events)} events")
    print(f"reference: SHV {shv:+.1%}  SPY {spy:+.1%}")
    results, passed, kept = {}, set(), {}
    for name, (drop, hold) in CELLS.items():
        gate = gate_trades(events, bars, sessions, drop, hold, COST, falls)
        info = gate_trades(events, bars, sessions, drop, hold, INFO_COST, falls)
        print(f"\n{name}: gate {len(gate)} trades, mean {gate.ret.mean():+.2%} at 80 bps ({info.ret.mean():+.2%} at 10 bps, info)")
        print("  gate mean by year:", {y: f"{m:+.1%} (n={n})" for y, (m, n) in gate.groupby("year").ret.agg(["mean", "count"]).iterrows()})
        if not gate.ret.mean() > 0:
            print("  OUT: the gate's mean trade is not positive")
            continue
        equity, trades = replay_stocks(events, bars, sessions, drop, hold, COST, falls)
        results[name], kept[name] = summary(equity, trades), trades
        print(f"  account 80 bps: {row(results[name])}")
        print(f"  years: { {y: f'{r:+.1%}' for y, r in results[name]['years'].items()} }")
        print(f"  account 10 bps (info): {row(summary(*replay_stocks(events, bars, sessions, drop, hold, INFO_COST, falls)))}")
        if passes(results[name], shv):
            passed.add(name)
    chosen = choose(results, passed)
    if chosen is None:
        print("\nNO: " + ("every cell fails the gate" if not results else "no cell passes the bar"))
        return None
    by_fall = kept[chosen].groupby("in_fall").ret.agg(["count", "mean", "sum"])
    print(f"\nchosen: {chosen} ({len(passed)} cells pass)\nentered while in_market_fall (True) vs not (False):\n{by_fall}")
    return chosen


def alpaca_keys() -> list[tuple[str, str]]:
    """The read-only Alpaca key pairs of the borrowed env file: the data keys first, the paper trading keys as a fallback."""
    env = dotenv_values(ENV_FILE)
    return [(env["ALPACA_DATA_API_KEY"] or "", env["ALPACA_DATA_API_SECRET"] or ""), (env["ALPACA_API_KEY"] or "", env["ALPACA_API_SECRET"] or "")]


def put_contracts(underlying: str, expiry_from: date, expiry_to: date) -> list[tuple[str, date, float]]:
    """(OCC symbol, expiry, strike) of the puts on `underlying` expiring in the range, active and expired (cached as JSON)."""
    path = CACHE / "contracts" / f"{underlying}_{expiry_from}_{expiry_to}.json"
    if path.exists():
        return [(symbol, date.fromisoformat(expiry), strike) for symbol, expiry, strike in json.loads(path.read_text())]
    found: dict[str, tuple[str, date, float]] | None = None
    for key, secret in alpaca_keys():
        client = TradingClient(key, secret, paper=True)
        try:
            found = {}
            for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
                token = None
                while True:
                    request = GetOptionContractsRequest(
                        underlying_symbols=[underlying], status=status, type=ContractType.PUT, expiration_date_gte=expiry_from, expiration_date_lte=expiry_to, limit=1000, page_token=token
                    )
                    response = client.get_option_contracts(request)
                    assert isinstance(response, OptionContractsResponse)
                    for c in response.option_contracts:
                        found[c.symbol] = (c.symbol, c.expiration_date, float(c.strike_price))
                    token = response.next_page_token
                    if not token:
                        break
            break
        except APIError:
            found = None
    if found is None:
        raise RuntimeError(f"Alpaca refused the option contracts request for {underlying}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([[s, e.isoformat(), k] for s, e, k in found.values()]))
    return list(found.values())


def option_bars(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Daily bars (`vwap`, `close`, indexed by session date) of option contracts; a contract without bars is absent (cached per contract)."""
    out: dict[str, pd.DataFrame] = {}
    missing = []
    for symbol in symbols:
        path = CACHE / "option_bars" / f"{symbol}_{start}_{end}.pkl"
        if path.exists():
            frame = pickle.loads(path.read_bytes())
            if len(frame):
                out[symbol] = frame
        else:
            missing.append(symbol)
    key, secret = alpaca_keys()[0]
    client = OptionHistoricalDataClient(key, secret)
    for i in range(0, len(missing), 20):
        batch = missing[i : i + 20]
        request = OptionBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=datetime.combine(start, time.min, tzinfo=ET), end=datetime.combine(end, time.max, tzinfo=ET))
        result = client.get_option_bars(request)
        assert isinstance(result, BarSet)
        data = result.data
        for symbol in batch:
            bars = data.get(symbol, [])
            frame = pd.DataFrame({"vwap": [float(b.vwap) for b in bars], "close": [float(b.close) for b in bars]}, index=[b.timestamp.astimezone(ET).date() for b in bars])
            path = CACHE / "option_bars" / f"{symbol}_{start}_{end}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(frame))
            if len(frame):
                out[symbol] = frame
    return out


def pick_contracts(contracts: list[tuple[str, date, float]], entry_day: date, sessions: list[date], hold: int, open_price: float) -> tuple[str, str, date] | None:
    """(long put, short put, expiry): the first expiry at least `hold` sessions after the entry session.

    The long put's strike is the nearest `open_price` (at the money), the short put's the nearest 10% below it. None
    without an expiry or when both strikes are the same.
    """
    i = sessions.index(entry_day)
    target = sessions[i + hold] if i + hold < len(sessions) else entry_day + timedelta(days=math.ceil(hold * 7 / 5))
    expiries = sorted({e for _, e, _ in contracts if e >= target})
    if not expiries:
        return None
    chain = [(s, k) for s, e, k in contracts if e == expiries[0]]
    long_put = min(chain, key=lambda c: abs(c[1] - open_price))
    short_put = min(chain, key=lambda c: abs(c[1] - 0.9 * open_price))
    return None if short_put[1] >= long_put[1] else (long_put[0], short_put[0], expiries[0])


def spread_cash(long_vwap: float, short_vwap: float, half: float, opening: bool) -> float:
    """Per share, with a half-spread `half` against the trader on each leg: the debit to open, or the credit (floored at 0) to close."""
    if opening:
        return (long_vwap + half) - (short_vwap - half)
    return max(0.0, (long_vwap - half) - (short_vwap + half))


def fees(contracts: int, opening: bool) -> float:
    """Tastytrade's fees for a two-leg spread of `contracts` contracts: $1 a contract to open (capped at $10 a leg), plus clearing and regulatory fees each side."""
    return (2 * min(OPEN_FEE * contracts, LEG_FEE_CAP) if opening else 0.0) + 2 * SIDE_FEE * contracts


def leg_exit(frame: pd.DataFrame, strike: float, stock: pd.DataFrame, expiry: date, exit_day: date, clipped: bool) -> float:
    """A leg's exit price per share: its VWAP on the exit session, else the first later bar (the last earlier bar at a window's end), else intrinsic value."""
    if exit_day in frame.index:
        return float(frame.loc[exit_day, "vwap"])
    picked = frame[frame.index <= exit_day].tail(1) if clipped else frame[frame.index > exit_day].head(1)
    if len(picked):
        return float(picked["vwap"].iloc[0])
    return max(0.0, strike - float(stock[stock["day"] <= expiry]["traded_close"].iloc[-1]))


def replay_spreads(
    events: list[dict],
    bars: dict[str, pd.DataFrame],
    sessions: list[date],
    drop: float,
    hold: int,
    contracts_of: Callable[[str, date, date], list[tuple[str, date, float]]],
    bars_of: Callable[[list[str], date, date], dict[str, pd.DataFrame]],
    slots: int = SLOTS,
) -> tuple[pd.Series, pd.DataFrame]:
    """The step-2 account with real put spreads: daily equity and the trades (`debit` per share), skips counted by cause.

    Same rules as `replay_stocks`, except that a candidate that cannot be traded (no contracts, no spread, no leg bar on
    the entry session, too expensive) passes its slot to the next-largest drop of the day. Legs fill at the daily VWAP
    plus a half-spread of `HALF_SPREAD` x the stock's traded open; positions mark at each leg's daily close.
    """
    by_entry: dict[date, list[dict]] = {}
    for event in qualifying(events, sessions, drop):
        by_entry.setdefault(event["entry_day"], []).append(event)
    skipped = {"slots full": 0, "already held": 0, "no contracts": 0, "no spread": 0, "no entry bar": 0, "too expensive": 0}
    cash, held, trades, equity = START, [], [], []
    for i, day in enumerate(sessions):
        open_equity = cash + sum(p["mark"] for p in held)
        for event in sorted(by_entry.get(day, []), key=lambda e: (e["abnormal"], e["symbol"])):
            stock = bars[event["symbol"]]
            opened = stock.loc[stock["day"] == day, "traded_open"]
            if len(held) >= slots:
                skipped["slots full"] += 1
                continue
            if any(p["cik"] == event["cik"] for p in held):
                skipped["already held"] += 1
                continue
            if opened.empty:
                skipped["no entry bar"] += 1
                continue
            contracts = contracts_of(event["symbol"], day, day + timedelta(days=75))
            if not contracts:
                skipped["no contracts"] += 1
                continue
            picked = pick_contracts(contracts, day, sessions, hold, float(opened.iloc[0]))
            if picked is None:
                skipped["no spread"] += 1
                continue
            long_sym, short_sym, expiry = picked
            legs = bars_of([long_sym, short_sym], day, expiry)
            if long_sym not in legs or short_sym not in legs or day not in legs[long_sym].index or day not in legs[short_sym].index:
                skipped["no entry bar"] += 1
                continue
            half = HALF_SPREAD * float(opened.iloc[0])
            debit = spread_cash(float(legs[long_sym].loc[day, "vwap"]), float(legs[short_sym].loc[day, "vwap"]), half, True)
            n = math.floor(min(FRACTION * open_equity, cash) / (debit * 100)) if debit > 0 else 0
            while n > 0 and n * debit * 100 + fees(n, True) > cash:
                n -= 1
            if n == 0:
                skipped["too expensive"] += 1
                continue
            cost = n * debit * 100 + fees(n, True)
            cash -= cost
            strikes = {s: k for s, _, k in contracts}
            joined = pd.concat([legs[long_sym]["close"], legs[short_sym]["close"]], axis=1, keys=["long", "short"]).sort_index().ffill()
            exit_index_ = i + hold - 1
            held.append(
                {
                    "symbol": event["symbol"],
                    "cik": event["cik"],
                    "n": n,
                    "cost": cost,
                    "debit": debit,
                    "half": half,
                    "legs": (legs[long_sym], legs[short_sym]),
                    "strikes": (strikes[long_sym], strikes[short_sym]),
                    "expiry": expiry,
                    "value": (joined["long"] - joined["short"]).clip(lower=0),
                    "exit_session": sessions[min(exit_index_, len(sessions) - 1)],
                    "clipped": exit_index_ > len(sessions) - 1,
                    "entry_day": day,
                    "mark": cost,
                }
            )
        for position in [p for p in held if p["exit_session"] == day]:
            long_exit = leg_exit(position["legs"][0], position["strikes"][0], bars[position["symbol"]], position["expiry"], day, position["clipped"])
            short_exit = leg_exit(position["legs"][1], position["strikes"][1], bars[position["symbol"]], position["expiry"], day, position["clipped"])
            value = position["n"] * 100 * spread_cash(long_exit, short_exit, position["half"], False) - fees(position["n"], False)
            cash += value
            trades.append({"symbol": position["symbol"], "entry_day": position["entry_day"], "exit_day": day, "ret": value / position["cost"] - 1, "debit": position["debit"]})
            held.remove(position)
        for position in held:
            known = position["value"].loc[:day]
            position["mark"] = position["n"] * 100 * float(known.iloc[-1]) if len(known) else position["mark"]
        equity.append(cash + sum(p["mark"] for p in held))
    frame = pd.DataFrame(trades, columns=["symbol", "entry_day", "exit_day", "ret", "debit"])
    frame.attrs["skipped"] = skipped
    return pd.Series(equity, index=sessions), frame


def probe() -> bool:
    """Whether Alpaca serves expired put contracts and their daily bars (AAPL, MSFT, NVDA, March 2024 expiries, bars in February 2024).

    Looks at availability only, never at a result. False means step 2 cannot be run with this data.
    """
    ok = True
    for symbol in ("AAPL", "MSFT", "NVDA"):
        frame = load_bars(symbol)
        assert frame is not None
        spot = float(frame[frame["day"] <= date(2024, 2, 15)]["traded_close"].iloc[-1])
        contracts = sorted(put_contracts(symbol, date(2024, 3, 1), date(2024, 3, 29)), key=lambda c: abs(c[2] - spot))[:6]
        found = option_bars([c[0] for c in contracts], date(2024, 2, 12), date(2024, 3, 1)) if contracts else {}
        print(f"probe {symbol}: spot {spot:.0f}, {len(contracts)} near puts expiring in March 2024, {len(found)} with daily bars in February")
        ok = ok and bool(found)
    return ok


TEST_GUARD = CACHE / "test.done"


def report_test(chosen: str) -> bool:
    """The held-out window, once, for the chosen cell: step 1 on stock moves, then (if it passes) step 2 with real put spreads. True = GO."""
    drop, hold = CELLS[chosen]
    sessions, bars, events = window_inputs("test")
    falls = falls_before_open(sessions)
    shv = buy_and_hold("SHV", sessions[0], sessions[-1])
    print(f"held-out {sessions[0]} -> {sessions[-1]}, {len(sessions)} sessions, {len(events)} events; SHV {shv:+.1%}")
    equity, trades = replay_stocks(events, bars, sessions, drop, hold, COST, falls)
    step1 = summary(equity, trades)
    ok1 = step1["total"] > 0 and step1["total"] > shv
    print(f"step 1 ({chosen}, 80 bps): {row(step1)}\n  {'PASS' if ok1 else 'FAIL'}: total > 0 and > SHV")
    if not ok1:
        return False
    window = [d for d in sessions if d >= STEP2_START]
    equity, trades = replay_spreads(events, bars, window, drop, hold, put_contracts, option_bars)
    step2 = summary(equity, trades)
    ok2 = step2["total"] > 0
    print(f"step 2 ({chosen}, real put spreads from {window[0]}): {row(step2)}\n  mean debit {trades.debit.mean():.2f} a share")
    print(f"  {'PASS' if ok2 else 'FAIL'}: total > 0")
    return ok2


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
    # 8. replay_stocks: largest drop first, 20% sizing at the entry open, "slots full", a held CIK skips its sibling
    days = [d.date() for d in pd.bdate_range("2024-01-02", periods=12)]
    fall = [100.0, 100.0, 95.0, 90.0] + [90.0] * 8
    flat = [100.0] * 12
    world = {name: bars_from_raw(raw_frame(days, closes)) for name, closes in (("A", fall), ("A2", flat), ("B", flat), ("X", fall), ("Y", flat), ("Z", flat), ("W", flat[:9] + [100.0, 95.0, 90.0]))}

    def ev(symbol: str, entry_day: date, abnormal: float, cik: str) -> dict:
        return {
            "symbol": symbol,
            "cik": cik,
            "accession": f"acc-{symbol}",
            "reaction_day": entry_day,
            "entry_day": entry_day,
            "abnormal": abnormal,
            "features": features(abnormal_pct=abnormal),
            "traded_close": 20.0,
        }

    equity, trades = replay_stocks([ev("B", days[1], -0.06, "2"), ev("A", days[1], -0.09, "1")], world, days, 0.05, 3, 0.0, set(), slots=1)
    assert trades.attrs["skipped"]["slots full"] == 1 and list(trades.symbol) == ["A"]
    assert abs(equity.iloc[1] - 5000) < 1e-9 and abs(equity.iloc[2] - 5050) < 1e-9 and abs(equity.iloc[-1] - 5100) < 1e-9
    assert abs(trades.ret.iloc[0] - 0.10) < 1e-9
    _, trades = replay_stocks([ev("A", days[1], -0.09, "1"), ev("A2", days[1], -0.07, "1")], world, days, 0.05, 3, 0.0, set(), slots=2)
    assert trades.attrs["skipped"]["already held"] == 1 and list(trades.symbol) == ["A"]
    # 9. a slot freed at an exit close is usable at the next open, not in the same session
    _, trades = replay_stocks([ev("X", days[1], -0.09, "1"), ev("Y", days[3], -0.08, "2"), ev("Z", days[4], -0.07, "3")], world, days, 0.05, 3, 0.0, set(), slots=1)
    assert trades.attrs["skipped"]["slots full"] == 1 and list(trades.symbol) == ["X", "Z"]
    # 10. a position still open at the window's last session is closed at that close
    equity, trades = replay_stocks([ev("W", days[9], -0.09, "4")], world, days, 0.05, 5, 0.0, set())
    assert trades.exit_day.iloc[0] == days[-1] and abs(equity.iloc[-1] - 5100) < 1e-9
    # 11. summary: calendar-year returns and max drawdown
    curve = pd.Series([5200.0, 5500.0, 5300.0, 4950.0], index=[date(2023, 6, 1), date(2023, 12, 29), date(2024, 6, 3), date(2024, 12, 31)])
    s = summary(curve, pd.DataFrame({"ret": [0.1, -0.1]}))
    assert abs(s["years"][2023] - 0.10) < 1e-9 and abs(s["years"][2024] + 0.10) < 1e-9
    assert abs(s["max_dd"] + 0.10) < 1e-9 and abs(s["total"] + 0.01) < 1e-9 and s["trades"] == 2 and s["win_rate"] == 0.5
    # 12. passes: beats SHV, at most 2 losing years, at least 30 trades
    good = {"total": 0.10, "years": {2016: 0.1, 2017: -0.1, 2018: -0.1}, "trades": 30}
    assert passes(good, 0.09) and not passes(good, 0.12) and not passes({**good, "trades": 29}, 0.09)
    assert not passes({**good, "years": {2016: -0.1, 2017: -0.1, 2018: -0.1}}, 0.09)
    # 13. choose: most passing neighbours, then the higher total, then the smaller drawdown
    stats = {name: {"total": 0.1, "max_dd": -0.2} for name in CELLS}
    assert choose(stats, {"D5% H5", "D5% H10", "D8% H10"}) == "D5% H10"
    stats["D8% H5"] = {"total": 0.2, "max_dd": -0.2}
    assert choose(stats, {"D5% H5", "D8% H5"}) == "D8% H5" and choose(stats, set()) is None
    # 14. gate_trades: one capped -1x trade per qualifying event, after costs
    both = {"P1": bars_from_raw(raw_frame(days, [100.0, 100.0, 95.0] + [95.0] * 9)), "P2": bars_from_raw(raw_frame(days, [100.0, 100.0, 102.0] + [102.0] * 9))}
    gate = gate_trades([ev("P1", days[1], -0.09, "1"), ev("P2", days[1], -0.09, "2")], both, days, 0.05, 2, 0.008, {days[1]})
    hand = (trade_value(0.95, 0.008) - 1 + trade_value(1.02, 0.008) - 1) / 2
    assert abs(gate.ret.mean() - hand) < 1e-12 and set(gate.year) == {2024} and gate.in_fall.all()
    # 15. pick_contracts: the first expiry at least `hold` sessions out, long nearest the open, short nearest 10% below
    near, far = days[4], days[8]
    chain = [(f"P{int(k)}_{e}", e, float(k)) for e in (near, far) for k in (95, 100, 105, 90)]
    assert pick_contracts(chain, days[1], days, 5, 101.0) == (f"P100_{far}", f"P90_{far}", far)
    assert pick_contracts([(f"P100_{far}", far, 100.0)], days[1], days, 5, 101.0) is None and pick_contracts([], days[1], days, 5, 101.0) is None
    # 16. spread_cash: the debit pays both half-spreads; the credit floors at 0
    assert abs(spread_cash(3.0, 1.0, 0.4, True) - 2.8) < 1e-12 and spread_cash(0.3, 0.1, 0.4, False) == 0.0
    # 17. fees: Tastytrade's, $1 to open capped at $10 a leg, $0.13 a contract a side
    assert abs(fees(3, True) - 6.78) < 1e-12 and abs(fees(15, True) - 23.90) < 1e-12 and abs(fees(3, False) - 0.78) < 1e-12
    # 18. replay_spreads: 3 contracts at a 2.80 debit; a candidate with no contracts passes its slot on; a missing exit bar uses the next
    chains = {"A": [("LA", days[8], 100.0), ("SA", days[8], 90.0)], "N": []}

    def legs(short_days: list[int]) -> dict[str, pd.DataFrame]:
        long_leg = pd.DataFrame({"vwap": [3.0, 3.0, 4.0, 5.0] + [5.0] * 8, "close": [3.0] * 12}, index=days)
        short_leg = pd.DataFrame({"vwap": [1.0, 1.0, 1.1, 1.0, 1.2] + [1.0] * 7, "close": [1.0] * 12}, index=days).iloc[short_days]
        return {"LA": long_leg, "SA": short_leg}

    def run(short_days: list[int]) -> tuple[pd.Series, pd.DataFrame]:
        def bars_of(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
            return {s: f[(f.index >= start) & (f.index <= end)] for s, f in legs(short_days).items() if s in symbols}

        return replay_spreads([ev("N", days[1], -0.09, "9"), ev("A", days[1], -0.08, "1")], {**world, "N": world["B"], "A": world["B"]}, days, 0.05, 3, lambda s, a, b: chains[s], bars_of, slots=1)

    _, trades = run(list(range(12)))
    cost = 3 * 280 + fees(3, True)
    credit = 3 * 100 * spread_cash(5.0, 1.0, 0.4, False) - fees(3, False)
    assert trades.attrs["skipped"]["no contracts"] == 1 and list(trades.symbol) == ["A"] and abs(trades.debit.iloc[0] - 2.8) < 1e-9
    assert abs(trades.ret.iloc[0] - (credit - cost) / cost) < 1e-9
    _, trades = run([i for i in range(12) if i != 3])
    credit = 3 * 100 * spread_cash(5.0, 1.2, 0.4, False) - fees(3, False)
    assert abs(trades.ret.iloc[0] - (credit - cost) / cost) < 1e-9
    print("self-check OK")


def main() -> None:
    """Entry point: `check`, `dev` or `test` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    if mode == "dev":
        report_dev()
        return
    if mode == "test":
        if TEST_GUARD.exists():
            sys.exit(f"The held-out window was already run ({TEST_GUARD} exists); see the spec. Not re-run.")
        chosen = report_dev()
        if chosen is None:
            sys.exit("No cell passed dev: the held-out window stays unused.")
        if not probe():
            sys.exit("Step 2 is impossible with this data (no expired contracts or bars). The held-out window stays unused; the user decides what next.")
        TEST_GUARD.parent.mkdir(parents=True, exist_ok=True)
        TEST_GUARD.write_text(f"{chosen}\n")
        print("\n" + "=" * 100 + f"\nHELD-OUT (one shot): {chosen}\n")
        print("GO" if report_test(chosen) else "NO")
        return
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
