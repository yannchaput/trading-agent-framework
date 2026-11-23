"""Spike: does an inverse ETF held open-to-close on `in_market_fall` days make money?

Design (rules fixed before running): docs/superpowers/specs/2026-11-23-market-fall-intraday-spike-design.md

Run:
  uv run python scripts/regime/fall_intraday_spike.py check   # self-check on synthetic data
  uv run python scripts/regime/fall_intraday_spike.py dev     # 2016 -> 2024: run, read
  uv run python scripts/regime/fall_intraday_spike.py test    # 2025 -> last session, ONCE, chosen cell only
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trading_agent_framework.utils.helpers import in_market_fall

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "cache" / "fall_intraday"
TEST_GUARD = CACHE / "test.done"
START = 5000.0  # the fake account
COST = 0.0003  # 3 bps per side
NY = "America/New_York"
ENV_STRATEGY, ENV_MODE = "bull_bear", "paper"  # env/.env.bull_bear.paper borrowed for its read-only ALPACA_DATA_API_* keys; no market_fall env exists


def fall_days(closes: pd.DataFrame) -> pd.Series:
    """Whether `in_market_fall` is true before each session's open: row d is decided from rows before d (row 0 is False)."""
    vix, vix3m, vixy = closes.vix.tolist(), closes.vix3m.tolist(), closes.vixy.tolist()
    flags = [False] + [in_market_fall(vix[:i], vix3m[:i], vixy[:i]) for i in range(1, len(closes))]
    return pd.Series(flags, index=closes.index)


def daily_closes() -> pd.DataFrame:
    """Yahoo ^VIX, ^VIX3M and VIXY closes from 2008-01-02 (columns vix, vix3m, vixy; date index). VIXY is NaN before 2011."""
    import yfinance as yf

    raw = yf.download(["^VIX", "^VIX3M", "VIXY"], start="2008-01-02", progress=False, auto_adjust=True)["Close"]
    raw = raw.rename(columns={"^VIX": "vix", "^VIX3M": "vix3m", "VIXY": "vixy"}).dropna(subset=["vix", "vix3m"])
    raw.index = raw.index.date
    return raw[["vix", "vix3m", "vixy"]]


def minute_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Regular-session 1-minute bars (open, high, low, close; New York index), Alpaca SIP, split/dividend adjusted.

    Cached per symbol-year under cache/fall_intraday/. shortcut: the current year is never cached, so a partial year is
    never frozen; fine for a spike, drop the exception if the script is re-run daily.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(start.year, end.year + 1):
        path = CACHE / f"{symbol}_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        df = _download_year(symbol, year)
        if year < date.today().year:
            df.to_parquet(path)
        frames.append(df)
    out = pd.concat(frames)
    return out[(out.index.date >= start) & (out.index.date <= end)]


def _download_year(symbol: str, year: int) -> pd.DataFrame:
    """One calendar year of regular-session minute bars from Alpaca (empty frame when the symbol has none)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from trading_agent_framework.config.env import AlpacaCredentials, load_strategy_env

    load_strategy_env(ENV_STRATEGY, ENV_MODE)
    creds = AlpacaCredentials.for_data()
    client = StockHistoricalDataClient(creds.api_key, creds.api_secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime(year, 1, 1),
        end=datetime(year, 12, 31, 23, 59),
        feed=DataFeed.SIP,
        adjustment=Adjustment.ALL,
    )
    df = client.get_stock_bars(request).df
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz=NY))
    return regular_session(df.droplevel("symbol")[["open", "high", "low", "close"]])


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the 09:30-15:59 New York bars of a UTC-indexed frame (early-close days simply end sooner)."""
    df = df.copy()
    df.index = df.index.tz_convert(NY)
    return df.between_time("09:30", "15:59")


def inverse_of(bars: pd.DataFrame) -> pd.DataFrame:
    """The synthetic -1 x SPY: per session each price p becomes 2 - p / session_open (high and low swap)."""
    out = []
    for _, day in bars.groupby(bars.index.date):
        first = day.open.iloc[0]
        out.append(pd.DataFrame({"open": 2 - day.open / first, "high": 2 - day.low / first, "low": 2 - day.high / first, "close": 2 - day.close / first}))
    return pd.concat(out)


def day_return(bars: pd.DataFrame, stop: float | None, cost: float = COST) -> tuple[float, str]:
    """One session held open to close: the multiplier on the money traded (both costs in) and how it ended.

    The trailing stop sits `stop` below the highest high of the EARLIER bars (seeded with the entry open); a bar that
    opens at or under it fills at its open, else one that trades down to it fills at the level. No re-entry.
    """
    o, h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    entry, exit_price, how, peak = o[0], c[-1], "close", o[0]
    if stop is not None:
        for i in range(len(o)):
            level = peak * (1 - stop)
            if o[i] <= level:
                exit_price, how = o[i], "stop"
                break
            if lo[i] <= level:
                exit_price, how = level, "stop"
                break
            peak = max(peak, h[i])
    return exit_price * (1 - cost) / (entry * (1 + cost)), how


def replay(bars: dict[date, pd.DataFrame], sessions: list[date], traded: set[date], stop: float | None, fraction: float) -> pd.DataFrame:
    """Run the fake account over `sessions`: on each session in `traded` that has bars, `fraction` of it goes in open-to-close.

    One row per session: equity, traded, exit ("close", "stop" or ""), trade_return (on the money traded) and missing
    (a traded session without bars: no trade, equity flat, counted).
    """
    equity, out = START, []
    for d in sessions:
        row = {"traded": False, "exit": "", "trade_return": 0.0, "missing": False}
        if d in traded and d in bars and len(bars[d]):
            mult, how = day_return(bars[d], stop)
            equity *= 1 + fraction * (mult - 1)
            row = {"traded": True, "exit": how, "trade_return": mult - 1, "missing": False}
        elif d in traded:
            row["missing"] = True
        out.append({"equity": equity, **row})
    return pd.DataFrame(out, index=sessions)


def curve_stats(equity: pd.Series) -> tuple[float, float]:
    """Total return and max drawdown (negative) of a daily equity curve that starts at START."""
    curve = pd.concat([pd.Series([START]), equity.reset_index(drop=True)], ignore_index=True)
    return curve.iloc[-1] / START - 1, float((curve / curve.cummax() - 1).min())


def summary(rows: pd.DataFrame) -> dict:
    """Total, CAGR, calendar-year returns, max drawdown and the trade counts of one replay."""
    total, max_dd = curve_stats(rows.equity)
    years_equity = rows.equity.groupby([d.year for d in rows.index]).last()
    years = (years_equity / years_equity.shift(1, fill_value=START) - 1).to_dict()
    span = (rows.index[-1] - rows.index[0]).days + 1
    trades = rows[rows.traded]
    return {
        "total": total,
        "cagr": (1 + total) ** (365.25 / span) - 1,
        "years": years,
        "max_dd": max_dd,
        "trades": len(trades),
        "win_rate": float((trades.trade_return > 0).mean()) if len(trades) else 0.0,
        "mean_trade": float(trades.trade_return.mean()) if len(trades) else 0.0,
        "stops": int((trades.exit == "stop").sum()),
        "closes": int((trades.exit == "close").sum()),
        "missing": int(rows.missing.sum()),
    }


WINDOWS = {"dev": (date(2016, 1, 4), date(2024, 12, 31)), "test": (date(2025, 1, 2), date.today() - timedelta(days=1))}
INSTRUMENTS = ["SH", "SDS", "SPXU", "-1xSPY"]
STOPS = [None, 0.02, 0.04]
CELLS = {f"{inst} {'no stop' if stop is None else f'stop {stop:.0%}'}": (inst, stop) for inst in INSTRUMENTS for stop in STOPS}
MAX_LOSING_YEARS = 2


def by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Split a minute frame into one frame per session date."""
    return {d: g for d, g in df.groupby(df.index.date)}


def load(start: date, end: date) -> tuple[list[date], dict[str, dict[date, pd.DataFrame]], pd.Series]:
    """Sessions (from SPY's minute bars), each instrument's bars by session, and SPY's daily closes."""
    spy = minute_bars("SPY", start, end)
    per = {"-1xSPY": by_day(inverse_of(spy))}
    for symbol in ("SH", "SDS", "SPXU"):
        per[symbol] = by_day(minute_bars(symbol, start, end))
    spy_close = spy.groupby(spy.index.date).close.last()
    return list(spy_close.index), per, spy_close


def line(s: dict) -> str:
    """One report row for a summary."""
    return (
        f"{s['total']:+8.1%} cagr {s['cagr']:+6.1%} dd {s['max_dd']:6.1%} trades {s['trades']:4d} win {s['win_rate']:4.0%} "
        f"mean {s['mean_trade']:+.2%} stop/close {s['stops']}/{s['closes']} missing {s['missing']}"
    )


def report(window: str, only: set[str] | None = None) -> str | None:
    """Run a window and print it; return the chosen cell's name (dev) or the name when it is a GO (test), else None.

    `only` limits the run to those cells (the held-out window), and then only the chosen cell's verdict is printed.
    """
    start, end = WINDOWS[window]
    sessions, per, spy_close = load(start, end)
    flags = fall_days(daily_closes())
    falls = {d for d in sessions if flags.get(d, False)}
    print(f"{window}: {sessions[0]} -> {sessions[-1]}, {len(sessions)} sessions, {len(falls)} fall days")
    spy_curve = spy_close / spy_close.iloc[0] * START
    spy_total, spy_dd = curve_stats(spy_curve)
    print(f"SPY buy and hold: {spy_total:+.1%} dd {spy_dd:.1%}")

    def run(name: str, traded: set[date], fraction: float = 1.0) -> pd.DataFrame:
        inst, stop = CELLS[name]
        return replay(per[inst], sessions, traded, stop, fraction)

    if only is not None:
        (name,) = only
        fall_rows, cash_rows = run(name, falls), run(name, falls, 0.5)
        s = summary(fall_rows)
        print(f"{name} on fall days, unconstrained: {line(s)}\n  years: { {y: f'{r:+.1%}' for y, r in s['years'].items()} }")
        print(f"{name} cash account (50% a day):    {line(summary(cash_rows))}")
        hedge, hedge_dd = curve_stats((spy_curve + fall_rows.equity) / 2)
        print(f"hedge, half SPY + half {name}: {hedge:+.1%} dd {hedge_dd:.1%}")
        go = s["total"] > 0
        print(f"{'GO' if go else 'NO-GO'}: total {s['total']:+.1%} on the held-out window")
        return name if go else None

    gate = np.mean([day_return(per["-1xSPY"][d], None)[0] - 1 for d in falls if d in per["-1xSPY"]])
    print(f"sign gate: -1 x SPY open-to-close on fall days, after costs, mean {gate:+.3%} ({'makes money' if gate > 0 else 'FAILS'})")
    if gate <= 0:
        print("NO: the sign gate failed, nothing else is judged")
        return None
    passing: dict[str, dict] = {}
    for name in CELLS:
        fall, everyday, quiet = (summary(run(name, t)) for t in (falls, set(sessions), set(sessions) - falls))
        print(f"{name:16s} fall days   {line(fall)}\n{'':16s} every day   {line(everyday)}\n{'':16s} no-fall     {line(quiet)}")
        print(f"{'':16s} cash (50%)  {line(summary(run(name, falls, 0.5)))}")
        print(f"{'':16s} years       { {y: f'{r:+.1%}' for y, r in fall['years'].items()} }")
        losing = sum(r < 0 for r in fall["years"].values())
        if not name.startswith("-1x") and fall["total"] > 0 and losing <= MAX_LOSING_YEARS and fall["total"] > everyday["total"]:
            passing[name] = fall
    if not passing:
        print("NO: no ETF cell passed the bar")
        return None
    chosen = max(passing, key=lambda n: (passing[n]["total"], passing[n]["max_dd"]))
    hedge, hedge_dd = curve_stats((spy_curve + run(chosen, falls).equity) / 2)
    print(f"chosen: {chosen} ({passing[chosen]['total']:+.1%}); hedge, half SPY + half {chosen}: {hedge:+.1%} dd {hedge_dd:.1%}")
    return chosen


def close_to(value: float, expected: float, costs: bool = False) -> bool:
    """Compare to `expected`, optionally applying the 3 bps entry and exit cost to it first."""
    if costs:
        expected = expected * (1 - COST) / (1 + COST)
    return abs(value - expected) < 1e-9


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """One New York session of 1-minute bars from (open, high, low, close) rows, starting 09:30."""
    idx = pd.date_range(datetime(2024, 3, 4, 9, 30), periods=len(rows), freq="min", tz=NY)
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def synthetic_closes(n: int) -> pd.DataFrame:
    """Random-walk VIX-like closes with a stressed stretch at rows 150-160 (VIX above VIX3M)."""
    rng = np.random.default_rng(1)
    idx = [date(2020, 1, 1) + pd.Timedelta(days=i) for i in range(n)]
    vix3m = 20 + rng.random(n)
    vix = vix3m * (0.8 + 0.05 * rng.random(n))
    vix[150:160] = vix3m[150:160] * 1.05
    vixy = 20 + rng.random(n)
    return pd.DataFrame({"vix": vix, "vix3m": vix3m, "vixy": vixy}, index=idx)


def check() -> None:
    """Assertions on synthetic data: no look-ahead, stop rules, gap fills, the inverse series."""
    # 1. no look-ahead: changing day k's closes never changes day k's verdict
    closes = synthetic_closes(300)
    base = fall_days(closes)
    for k in range(200, 300, 9):
        moved = closes.copy()
        moved.iloc[k] = moved.iloc[k] * [3.0, 1.0, 2.0]
        assert fall_days(moved).iloc[k] == base.iloc[k]
    assert base.iloc[151:160].any()  # the stressed stretch is seen the day after
    # 2. no stop: open 100 -> close 90, cost 3 bps each side
    bars = frame([(100, 101, 99, 100), (100, 100, 90, 90)])
    mult, exit_ = day_return(bars, None)
    assert exit_ == "close" and close_to(mult, 0.9, costs=True)
    # 3. stop 2%: high 105 on bar 1, bar 2 low 102 -> stop at 105 * 0.98 = 102.9, filled at 102.9
    bars = frame([(100, 105, 100, 104), (104, 104, 102, 103), (103, 120, 103, 120)])
    mult, exit_ = day_return(bars, 0.02)
    assert exit_ == "stop" and close_to(mult, 1.029, costs=True)
    # 4. a bar's own high does not raise its stop: bar 2 high 110 then low 104 -> not stopped (stop is 98 from bar 1)
    bars = frame([(100, 100, 99, 100), (100, 110, 104, 105)])
    assert day_return(bars, 0.02)[1] == "close"
    # 5. gap through: stop 98, next bar opens 95 -> filled at 95
    bars = frame([(100, 100, 99, 99), (95, 96, 94, 95)])
    assert close_to(day_return(bars, 0.02)[0], 0.95, costs=True)
    # 6. inverse: SPY 100 -> 99 gives 1 -> 1.01
    assert close_to(inverse_of(frame([(100, 100, 99, 99)])).iloc[-1]["close"], 1.01)
    # 7. regular session in New York time, whatever the UTC offset: summer 13:30 UTC and winter 14:30 UTC are both 09:30
    for day, open_utc in (("2024-07-01", "13:30"), ("2024-12-02", "14:30")):
        idx = pd.date_range(f"{day} {open_utc}", periods=3, freq="min", tz="UTC").append(
            pd.DatetimeIndex([pd.Timestamp(f"{day} {open_utc}", tz="UTC") - pd.Timedelta(minutes=1), pd.Timestamp(f"{day} {open_utc}", tz="UTC") + pd.Timedelta(hours=7)])
        )
        raw = pd.DataFrame(1.0, index=idx, columns=["open", "high", "low", "close"])
        kept = regular_session(raw)
        assert len(kept) == 3 and str(kept.index[0].time()) == "09:30:00", day  # pre-market and 16:30 bars dropped
    # 8. replay: two traded days (+10% then -10% on the money traded), half the account each day
    d1, d2 = date(2023, 12, 29), date(2024, 1, 2)
    up, down = frame([(100, 100, 100, 110)]), frame([(100, 100, 100, 90)])
    m1, m2 = 1.1 * (1 - COST) / (1 + COST), 0.9 * (1 - COST) / (1 + COST)
    rows = replay({d1: up, d2: down}, [d1, d2], {d1, d2}, None, 0.5)
    assert abs(rows.equity.iloc[-1] - START * (1 + 0.5 * (m1 - 1)) * (1 + 0.5 * (m2 - 1))) < 1e-6
    # 9. a signalled day without bars is untraded and counted; a session off the signal is not traded
    s = summary(replay({d1: up, d2: down}, [d1, d2], {d1, d2}, None, 1.0))
    assert s["trades"] == 2 and s["missing"] == 0
    s = summary(replay({d1: up}, [d1, d2], {d1, d2}, None, 1.0))
    assert s["trades"] == 1 and s["missing"] == 1
    s = summary(replay({d1: up, d2: down}, [d1, d2], {d1}, None, 1.0))
    assert s["trades"] == 1 and s["missing"] == 0
    # 10. summary: calendar-year returns, max drawdown, stops vs closes
    s = summary(replay({d1: up, d2: down}, [d1, d2], {d1, d2}, None, 1.0))
    assert abs(s["years"][2023] - (m1 - 1)) < 1e-9 and abs(s["years"][2024] - (m2 - 1)) < 1e-9
    assert abs(s["max_dd"] - (m2 - 1)) < 1e-9 and s["win_rate"] == 0.5 and s["closes"] == 2 and s["stops"] == 0
    print("self-check OK")


def main() -> None:
    """Entry point: `check`, `dev` or `test` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    if mode == "dev":
        report("dev")
        return
    if mode == "test":
        if TEST_GUARD.exists():
            sys.exit(f"The held-out window was already run ({TEST_GUARD} exists); see the spec. Not re-run.")
        chosen = report("dev")
        if chosen is None:
            sys.exit("No cell passed dev: the held-out window stays unused.")
        print("\n" + "=" * 100 + f"\nHELD-OUT (one shot): {chosen}\n")
        TEST_GUARD.parent.mkdir(parents=True, exist_ok=True)
        TEST_GUARD.write_text(f"{chosen}\n")
        report("test", only={chosen})
        return
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
