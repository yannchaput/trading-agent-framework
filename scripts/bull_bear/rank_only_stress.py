"""TODO item 5: stress test of bull_bear's rank-only book, 2005-2026.

Spec: docs/superpowers/specs/2026-11-27-bull-bear-rank-only-stress-test-design.md

Run from the repo root: uv run python scripts/bull_bear/rank_only_stress.py [--self-check]
"""

import io
import math
import shutil
import sys
import zipfile
from bisect import bisect_left
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import requests

from trading_agent_framework.backtesting.data.yahoo import extract_ticker_frame, parse_yahoo_frame, yfinance_download
from trading_agent_framework.strategies.common.batch_us_stock_universe import download_batch, ensure_universe_history
from trading_agent_framework.strategies.common.scoring import MomentumRow, rank, score_stock
from trading_agent_framework.strategies.common.universe_history import UniverseHistory, month_universe
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG

START, END = date(2005, 1, 1), date(2026, 11, 24)
HISTORY_START = date(2003, 10, 1)  # momentum needs a year of closes before 2005
CACHE = Path("cache/stress_2005")
SLEEVE = "SHV"
TOP_N = 10
WEIGHTS = (0.098, 0.084, 0.070)  # the first decides the verdict; the others are report-only
LIMIT = -0.40
CLOSES = 300  # sessions of history handed to the scorer (12-1 momentum needs 274)
WINDOWS = {
    "2008-09": (date(2008, 1, 1), date(2009, 12, 31)),
    "2015-16": (date(2015, 1, 1), date(2016, 12, 31)),
    "2018 Q4": (date(2018, 10, 1), date(2018, 12, 31)),
    "2020": (date(2020, 1, 1), date(2020, 12, 31)),
    "2022": (date(2022, 1, 1), date(2022, 12, 31)),
}
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_Daily_CSV.zip"


def rebalance_days(sessions: list[date]) -> list[date]:
    """Per ISO week, the first session on a Tuesday or later."""
    weeks: dict[tuple[int, int], date] = {}
    for day in sorted(sessions):
        if day.weekday() >= 1:
            weeks.setdefault(day.isocalendar()[:2], day)
    return list(weeks.values())


def simulate(
    prices: dict[str, pd.DataFrame],
    sessions: list[date],
    picks: Callable[[date, set[str]], list[str]],
    weight: float,
    cost: float = 0.0005,
) -> pd.Series:
    """Equity at each session close, from 1.0.

    On a rebalance day every holding is valued at that open (else its last close) and reset to `weight` per pick; the
    rest goes to SHV when it has an open, else stays cash at 0%. `cost` is charged on every dollar traded. A pick
    with no open that day is left out.

    shortcut: a holding whose prices stop keeps its last close until the next rebalance, where it is sold at it.
    """
    opens = {symbol: frame["open"].to_dict() for symbol, frame in prices.items()}
    closes = {symbol: frame["close"].to_dict() for symbol, frame in prices.items()}
    rebalances = set(rebalance_days(sessions))
    cash, shares, last = 1.0, {}, {}
    equity = []
    for day in sessions:
        if day in rebalances:
            held = {symbol for symbol in shares if symbol != SLEEVE}
            wanted = [s for s in picks(day, held) if day in opens.get(s, {})]
            at_open = {s: opens[s][day] if day in opens[s] else last[s] for s in {*shares, *wanted}}
            if day in opens.get(SLEEVE, {}):
                at_open[SLEEVE] = opens[SLEEVE][day]
            old = {s: n * at_open[s] for s, n in shares.items()}
            value = cash + sum(old.values())
            new = {s: weight * value for s in wanted}
            rest = value - sum(new.values())
            if SLEEVE in at_open:
                new[SLEEVE] = rest
            fees = cost * sum(abs(new.get(s, 0.0) - old.get(s, 0.0)) for s in {*new, *old})
            if SLEEVE in new:
                new[SLEEVE] -= fees
                cash = 0.0
            else:
                cash = rest - fees
            shares = {s: dollars / at_open[s] for s, dollars in new.items()}
            last.update({s: at_open[s] for s in shares})
        for symbol in shares:
            if day in closes[symbol]:
                last[symbol] = closes[symbol][day]
        equity.append(cash + sum(n * last[s] for s, n in shares.items()))
    return pd.Series(equity, index=sessions)


def max_drawdown(equity: pd.Series) -> tuple[float, date, date]:
    """(worst fall as a negative fraction, its peak date, its trough date)."""
    falls = equity / equity.cummax() - 1
    trough = falls.idxmin()
    return float(falls.min()), equity.loc[:trough].idxmax(), trough


def eligible(day: date, held: set[str], listed: list[str], frames: dict[str, pd.DataFrame], sessions: list[date], history: int = CLOSES) -> Iterator[tuple[str, MomentumRow, pd.DataFrame]]:
    """(symbol, momentum row, its last `history` sessions) of each listed symbol bull_bear's filters keep for a trade on `day`.

    Only sessions strictly before `day` are read, and a symbol needs a bar on the last of them. The row is scored on the
    last `CLOSES` of them. A held symbol is exempt from max_volatility, as in bull_bear; one that left the list is not seen.
    """
    previous = sessions[bisect_left(sessions, day) - 1]
    exempt = {**CONFIG, "max_volatility": math.inf}
    for symbol in listed:
        frame = frames.get(symbol)
        if frame is None:
            continue
        end = bisect_left(frame.index, day)
        if end == 0 or frame.index[end - 1] != previous:
            continue
        past = frame.iloc[max(0, end - history) : end]
        recent = past.iloc[-CLOSES:]
        row = score_stock(
            symbol,
            recent["close"].tolist(),
            recent["volume"].tolist(),
            exempt if symbol in held else CONFIG,
            traded_closes=recent["traded_close"].tolist(),
            traded_volumes=recent["traded_volume"].tolist(),
        )
        if row is not None:
            yield symbol, row, past


def ranked_symbols(day: date, held: set[str], listed: list[str], frames: dict[str, pd.DataFrame], sessions: list[date]) -> list[str]:
    """The listed symbols bull_bear's ranking keeps for a trade on `day` (see `eligible`), best first."""
    return [ranked.row.symbol for ranked in rank(row for _, row, _ in eligible(day, held, listed, frames, sessions))]


def _load_bars(path: Path) -> pd.DataFrame:
    """A cached Yahoo download, parsed, keyed by session date."""
    frame = parse_yahoo_frame(pd.read_parquet(path))
    frame.index = pd.Index([stamp.date() for stamp in frame.index])
    return frame


def load_data() -> tuple[UniverseHistory, dict[str, pd.DataFrame], list[date]]:
    """The 2003-2026 history, the bars of every listed symbol (plus SPY and SHV) and SPY's sessions."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (CACHE / "pool.json").exists():
        shutil.copy("cache/universe_history/pool.json", CACHE / "pool.json")
    history = ensure_universe_history(HISTORY_START, END, path=CACHE / "history.json", cache_dir=CACHE)
    symbols = [*history.union(START, END), "SPY"]
    frames = {s: _load_bars(CACHE / "bars" / f"{s}.parquet") for s in symbols}
    shv = CACHE / "shv.parquet"
    if not shv.exists():
        raw = download_batch(yfinance_download(), [SLEEVE], date(2003, 6, 1), END, empty_is_failure=True)
        extract_ticker_frame(raw, SLEEVE).to_parquet(shv)
    frames[SLEEVE] = _load_bars(shv)
    assert len(frames[SLEEVE]) > 4000, "SHV has too few bars"
    sessions = [day for day in frames["SPY"].index if day <= END]
    return history, frames, sessions


def french_returns() -> dict[str, pd.Series]:
    """Ken French's top momentum decile, daily, value- and equal-weighted, as fractions (read as data only)."""
    path = CACHE / "french" / "momentum_daily.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(FRENCH_URL, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        lines = archive.read(archive.namelist()[0]).decode("latin-1").splitlines()
    tables: dict[str, dict[date, float]] = {}
    name = None
    for line in lines:
        cells = [cell.strip() for cell in line.split(",")]
        if "Weighted Returns" in line and "Daily" in line:
            name = "French top decile VW" if "Value" in line else "French top decile EW"
            tables[name] = {}
        elif name and len(cells) == 11 and len(cells[0]) == 8 and cells[0].isdigit():
            value = float(cells[10])  # the 10th decile, "Hi PRIOR"
            if value > -99:
                tables[name][date(int(cells[0][:4]), int(cells[0][4:6]), int(cells[0][6:]))] = value / 100
    assert len(tables) == 2 and all(tables.values()), f"unexpected French file layout: {list(tables)}"
    return {name: pd.Series(values).sort_index() for name, values in tables.items()}


def universe_returns(history: UniverseHistory, frames: dict[str, pd.DataFrame], sessions: list[date]) -> pd.Series:
    """The equal-weighted month list, rebalanced daily with no cost: the mean close-to-close return of its members."""
    closes = pd.DataFrame({s: frames[s]["close"] for s in history.union(START, END)}).reindex(sessions)
    returns = closes / closes.shift(1) - 1
    out = {}
    for day in sessions:
        if day >= START:
            members = month_universe(history, day, exclude={SLEEVE})
            out[day] = returns.loc[day, members].mean()
    return pd.Series(out)


def summary(name: str, equity: pd.Series) -> str:
    """One report row: CAGR, max drawdown with its dates, and the drawdown inside each stress window."""
    first_day, last_day = cast("date", equity.index[0]), cast("date", equity.index[-1])
    years = (last_day - first_day).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    dd, peak, trough = max_drawdown(equity)
    inside = []
    for first, last in WINDOWS.values():
        part = cast("pd.Series", equity[(equity.index >= first) & (equity.index <= last)])
        inside.append(f"{max_drawdown(part)[0]:.1%}")
    return f"| {name} | {cagr:.1%} | {dd:.1%} | {peak} to {trough} | {' | '.join(inside)} | {equity.index[-1]} |"


def print_data_checks(frames: dict[str, pd.DataFrame], books: dict[date, list[str]], last_day: date) -> None:
    """Print the 20 largest one-day moves of held stocks and the count of non-positive price rows."""
    held_moves = []
    trade_days = sorted(books)
    for first, following in zip(trade_days, [*trade_days[1:], last_day], strict=True):
        for symbol in books[first]:
            close = frames[symbol]["close"]
            moves = close / close.shift(1) - 1
            held_moves += [(abs(move), day, symbol, move) for day, move in moves[(moves.index > first) & (moves.index <= following)].items()]
    print("\nLargest one-day moves of held stocks:")
    for _, day, symbol, move in sorted(held_moves, reverse=True)[:20]:
        print(f"  {day} {symbol} {move:+.1%}")
    bad = {}
    for symbol, frame in frames.items():
        rows = int((frame[["open", "close", "traded_close"]] <= 0).any(axis=1).sum())
        if rows:
            bad[symbol] = rows
    print(f"Non-positive price rows: {sum(bad.values())} in {len(bad)} symbols {sorted(bad)[:10]}")


def main() -> None:
    """Run the book at the three weights against the references and print the report and the verdict."""
    self_check()
    history, frames, sessions = load_data()
    window = [day for day in sessions if day >= START]
    ranked_count: dict[date, int] = {}
    chosen: dict[date, list[str]] = {}

    def picks(day: date, held: set[str]) -> list[str]:
        # What is held does not depend on the position size, so one ranking per day serves all three weights.
        if day not in chosen:
            ranked = ranked_symbols(day, held, month_universe(history, day, exclude={SLEEVE}), frames, sessions)
            ranked_count[day] = len(ranked)
            chosen[day] = ranked[:TOP_N]
        return chosen[day]

    books = {weight: simulate(frames, window, picks, weight) for weight in WEIGHTS}

    french = french_returns()
    curves: dict[str, pd.Series] = {f"Rank-only {weight * TOP_N:.0%} stocks": books[weight] for weight in WEIGHTS}
    curves["SPY"] = cast("pd.Series", frames["SPY"]["close"].reindex(window))
    curves["Equal-weighted universe"] = cast("pd.Series", (1 + universe_returns(history, frames, sessions)).cumprod())
    for name, returns in french.items():
        curves[name] = cast("pd.Series", (1 + returns[(returns.index >= START) & (returns.index <= END)]).cumprod())

    print(f"| Book | CAGR | Max DD | Peak to trough | {' | '.join(WINDOWS)} | Data to |")
    print("|" + "---|" * (len(WINDOWS) + 5))
    for name, equity in curves.items():
        print(summary(name, equity))

    print("\n| Year | Mean ranked stocks | Pool symbols with bars |")
    print("|---|---|---|")
    years_with_bars = {path.stem: {stamp.year for stamp in pd.read_parquet(path, columns=["Close"]).dropna().index} for path in (CACHE / "bars").glob("*.parquet") if path.stem != "SPY"}
    for year in range(START.year, END.year + 1):
        counts = [n for day, n in ranked_count.items() if day.year == year]
        covered = sum(year in found for found in years_with_bars.values())
        print(f"| {year} | {sum(counts) / len(counts):.0f} | {covered / history.pool_size:.0%} |")

    print_data_checks(frames, chosen, window[-1])

    pd.DataFrame({f"{w:.3f}": books[w] for w in WEIGHTS}).to_csv(CACHE / "equity.csv")
    pd.DataFrame([(day, ",".join(symbols)) for day, symbols in sorted(chosen.items())], columns=["day", "book"]).to_csv(CACHE / "books.csv", index=False)
    worst = max_drawdown(books[WEIGHTS[0]])[0]
    verdict = "not rejected (unconfirmed: survivorship)" if worst >= LIMIT else "rank-only at this size is REJECTED for real money"
    print(f"\nVerdict at {WEIGHTS[0] * TOP_N:.0%} stocks: max drawdown {worst:.1%} vs limit {LIMIT:.0%} -> {verdict}")


def _frame(rows: list[tuple[date, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["day", "open", "close"]).set_index("day")


def self_check() -> None:
    """Assert the weekly book and drawdown maths on hand-worked cases."""
    days = [date(2024, 1, d) for d in (2, 3, 4, 8, 10, 11)]  # Tue, Wed, Thu, Mon, Wed (holiday Tue), Thu
    assert rebalance_days(days) == [date(2024, 1, 2), date(2024, 1, 10)]
    dead = {"A": _frame([(days[0], 10.0, 10.0), (days[1], 10.0, 8.0)])}  # prices stop after day 2
    equity = simulate(dead, days, lambda day, held: ["A"] if day == days[0] else [], weight=0.5, cost=0.0)
    assert [round(v, 6) for v in equity] == [1.0, 0.9, 0.9, 0.9, 0.9, 0.9]  # sold at its last close
    costly = simulate(dead, days[:2], lambda day, held: ["A"], weight=0.5, cost=0.001)
    assert round(costly.iloc[0], 6) == 0.9995
    # With a sleeve: the 50% not in A goes to SHV and earns its return; both trades pay the cost.
    with_sleeve = {"A": _frame([(d, 10.0, 10.0) for d in days[:2]]), "SHV": _frame([(days[0], 100.0, 100.0), (days[1], 100.0, 102.0)])}
    parked = simulate(with_sleeve, days[:2], lambda day, held: ["A"], weight=0.5, cost=0.001)
    assert [round(v, 6) for v in parked] == [0.999, 1.00898]  # 0.499 in SHV after the cost, up 2%
    seen: list[tuple[date, set[str]]] = []
    simulate(dead, days, lambda day, held: seen.append((day, held)) or ["A"], weight=0.5)
    assert seen[1] == (days[4], {"A"})
    dd, peak, trough = max_drawdown(pd.Series([1.0, 2.0, 1.0, 1.5], index=days[:4]))
    assert (dd, peak, trough) == (-0.5, days[1], days[2])
    # Ranking: 300 steady sessions, then a bar on the trade day that crashes. It must not be read.
    history = [date(2023, 1, 1) + timedelta(days=i) for i in range(301)]

    def stock(drift: float, days_: list[date]) -> pd.DataFrame:
        close = [20.0 * (1 + drift) ** i * (1.001 if i % 2 else 1.0) for i in range(len(days_))]
        return pd.DataFrame({"close": close, "volume": 5e6, "traded_close": close, "traded_volume": 5e6}, index=days_)

    frames = {"UP": stock(0.002, history), "FLAT": stock(0.0005, history), "HALT": stock(0.003, history[:-2])}
    frames["UP"].loc[history[-1], ["close", "traded_close"]] = 0.01
    assert ranked_symbols(history[-1], set(), ["FLAT", "UP", "HALT"], frames, history) == ["UP", "FLAT"]  # HALT: no previous-session bar
    wild = stock(0.002, history)
    wild["close"] *= [1.15 if i % 2 and i > 270 else 1.0 for i in range(301)]
    assert ranked_symbols(history[-1], set(), ["WILD"], {"WILD": wild}, history) == []  # above max_volatility
    assert ranked_symbols(history[-1], {"WILD"}, ["WILD"], {"WILD": wild}, history) == ["WILD"]  # held: exempt
    assert ranked_symbols(history[-1], {"UP"}, ["FLAT"], frames, history) == ["FLAT"]  # held but left the list: unranked
    print("self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()
