"""Spike: a 10-minute VIXY / SPY signal that trades an inverse S&P ETF intraday, with SHV overnight.

Design (rules fixed before running): docs/superpowers/specs/2026-11-23-intraday-fall-signal-spike-design.md

Run:
  uv run python scripts/regime/intraday_fall_signal_spike.py check   # self-check on synthetic data
  uv run python scripts/regime/intraday_fall_signal_spike.py dev     # 2016-01-04 -> 2024-06-28: run, read
  uv run python scripts/regime/intraday_fall_signal_spike.py test    # 2024-07-01 -> last session, ONCE, chosen cell only

A decision at time t (the close of a 10-minute block) reads only 1-minute bars stamped before t and trades at the open
of the bar stamped t.
"""

import importlib.util
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fall = _load("fall_intraday_spike", REPO / "scripts" / "regime" / "fall_intraday_spike.py")
NY, is_early_close, trim_early_closes, regular_session = _fall.NY, _fall.is_early_close, _fall.trim_early_closes, _fall.regular_session
ENV_STRATEGY, ENV_MODE = _fall.ENV_STRATEGY, _fall.ENV_MODE
CACHE = REPO / "cache" / "intraday_fall_signal"
TEST_GUARD = CACHE / "test.done"
SESSION_MINUTES = 390  # 09:30 -> 15:59


def minute_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Regular-session 1-minute bars (open, high, low, close, volume; New York index), Alpaca SIP, adjusted.

    Cached per symbol-year under cache/intraday_fall_signal/. shortcut: the current year is never cached, so a partial
    year is never frozen; fine for a spike.
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
    out = trim_early_closes(pd.concat(frames))
    return out[(out.index.date >= start) & (out.index.date <= end)]


def _download_year(symbol: str, year: int) -> pd.DataFrame:
    """One calendar year of regular-session minute bars with volume from Alpaca (empty frame when there are none)."""
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
    cols = ["open", "high", "low", "close", "volume"]
    if df.empty:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz=NY))
    return regular_session(df.droplevel("symbol")[cols])


def daily_yahoo() -> pd.DataFrame:
    """Yahoo adjusted VIXY close with its SMA50 (the 50 closes BEFORE each day) and SHV open and close, date index."""
    import yfinance as yf

    raw = yf.download(["VIXY", "SHV"], start="2015-06-01", progress=False, auto_adjust=True)
    out = pd.DataFrame(
        {
            "vixy_close": raw["Close"]["VIXY"],
            "vixy_sma50": raw["Close"]["VIXY"].rolling(50).mean().shift(1),
            "shv_open": raw["Open"]["SHV"],
            "shv_close": raw["Close"]["SHV"],
        }
    )
    out.index = out.index.date
    return out


def decision_times(session: date, early: bool) -> list[pd.Timestamp]:
    """Decision times 09:40, 09:50, ... 15:40 (12:40 on early closes), New York time."""
    last = "12:40" if early else "15:40"
    return list(pd.date_range(f"{session} 09:40", f"{session} {last}", freq="10min", tz=NY))


def day_context(prev: list[dict], sma50: float) -> dict | None:
    """What D-1..D-5 give a session: D-1 closes and VIXY VWAP, VIXY's SMA50, each day's cumulative VIXY volume.

    `prev[i]` = {"spy": df, "vixy": df} minute frames of the i+1-th previous session (most recent first); None when
    fewer than 5. `vixy_cumvol` is a list of (cum, reach): cum[m] = VIXY volume of the day's minutes 0..m (since
    09:30), reach = the minute count the day's bars cover (an early close or a thin end reaches less than 390).
    """
    if len(prev) < 5:
        return None
    d1 = prev[0]["vixy"]
    cumvol = []
    for p in prev[:5]:
        v = p["vixy"]
        minute = ((v.index.hour - 9) * 60 + v.index.minute - 30).to_numpy()
        per_minute = np.zeros(SESSION_MINUTES)
        per_minute[minute] = v.volume.to_numpy()
        cumvol.append((np.cumsum(per_minute), int(minute.max()) + 1))
    return {
        "vixy_d1_close": float(d1.close.iloc[-1]),
        "vixy_d1_vwap": float((d1.close * d1.volume).sum() / d1.volume.sum()),
        "spy_d1_close": float(prev[0]["spy"].close.iloc[-1]),
        "vixy_sma50": sma50,
        "vixy_cumvol": cumvol,
    }


def _now_and_before(df: pd.DataFrame, t: pd.Timestamp, minutes: int) -> tuple[float, float]:
    """Last close before t, and its reference `minutes` earlier.

    The reference is the close of the last bar stamped before t - minutes, or the session's first open when that moment
    is the 09:30 open or earlier (a window never crosses a night).
    """
    done = df[df.index < t]
    now = float(done.close.iloc[-1])
    ref_time = t - timedelta(minutes=minutes)
    if ref_time <= t.normalize() + timedelta(hours=9, minutes=30):
        return now, float(df.open.iloc[0])
    return now, float(done[done.index < ref_time].close.iloc[-1])


def observe(spy: pd.DataFrame, vixy: pd.DataFrame, ctx: dict, t: pd.Timestamp) -> dict:
    """The spec's signal values at decision time t, from bars stamped before t only."""
    vixy_now, vixy_hour_ref = _now_and_before(vixy, t, 60)
    spy_now, spy_hour_ref = _now_and_before(spy, t, 60)
    _, vixy_30_ref = _now_and_before(vixy, t, 30)
    _, spy_30_ref = _now_and_before(spy, t, 30)
    elapsed = int((t - t.normalize()).total_seconds() // 60) - (9 * 60 + 30)
    past = [cum[elapsed - 1] for cum, reach in ctx["vixy_cumvol"] if reach >= elapsed]
    today_volume = float(vixy[vixy.index < t].volume.sum())
    return {
        "vixy_day": vixy_now / ctx["vixy_d1_close"] - 1,
        "vixy_hour": vixy_now / vixy_hour_ref - 1,
        "vixy_vs_d1avg": vixy_now / ctx["vixy_d1_vwap"] - 1,
        "vixy_trend": vixy_now > ctx["vixy_sma50"],
        "rvol": today_volume / float(np.mean(past)) if past and np.mean(past) > 0 else 0.0,
        "spy_day": spy_now / ctx["spy_d1_close"] - 1,
        "spy_hour": spy_now / spy_hour_ref - 1,
        "vixy_30": vixy_now / vixy_30_ref - 1,
        "spy_30": spy_now / spy_30_ref - 1,
    }


S_CONF = 0.005  # SPY at least this far under D-1's close
FADE_MINUTES = 30  # a trade can fade only this long after its fill, over a window of this length
DAILY_LOSS = -0.02  # closed P&L of the day, as a share of the day's opening equity, that ends the day's entries
MAX_STOPS = 2  # stopped trades in a day that end the day's entries
COST = 0.0003  # 3 bps per side on the ETF
SHV_COST = 0.0001  # 1 bp per side on SHV
START = _fall.START  # the fake account, 5,000
MAX_DAY_TRADES, CAP_WINDOW = 3, 5  # day-trade cap: at most 3 trades in any 5 consecutive sessions


def entry_signal(obs: dict, first: bool, cell: dict) -> bool:
    """The spec's four entry conditions. `first` = the 09:40 decision, the only one where VIXY's D-1 move counts."""
    fast = obs["vixy_day"] >= cell["v_day"] if first else obs["vixy_hour"] >= cell["v_hour"]
    context = obs["vixy_vs_d1avg"] > 0 and obs["vixy_trend"]
    volume = cell["rvol"] is None or obs["rvol"] >= cell["rvol"]
    movement = obs["spy_day"] <= -S_CONF and obs["spy_hour"] < 0
    return bool(fast and context and volume and movement)


def decide(obs: dict, first: bool, in_position: bool, minutes_held: float, limits_hit: bool, cap_ok: bool, cell: dict) -> str:
    """The one decision function: "enter", "exit" (the signal fades: VIXY down and SPY up over 30 minutes) or "none"."""
    if in_position:
        faded = minutes_held >= FADE_MINUTES and obs["vixy_30"] < 0 and obs["spy_30"] > 0
        return "exit" if faded else "none"
    return "enter" if not limits_hit and cap_ok and entry_signal(obs, first, cell) else "none"


def atr10(prev_etf: list[pd.DataFrame]) -> float:
    """Mean true range of the ETF's 10-minute bars over the given previous sessions (the stop's volatility unit)."""
    ranges = []
    for day in prev_etf:
        bars = day.resample("10min").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        prev_close = bars.close.shift(1).fillna(bars.close)
        ranges.append(pd.concat([bars.high - bars.low, (bars.high - prev_close).abs(), (bars.low - prev_close).abs()], axis=1).max(axis=1))
    return float(pd.concat(ranges).mean())


def simulate_session(etf: pd.DataFrame, obs_by_t: dict[pd.Timestamp, dict], cell: dict, atr: float, cap_left: int, early: bool) -> list[dict]:
    """One session's trades on the ETF's minute bars: [{"entry", "exit", "mult", "how"}], costs in `mult`.

    Per minute bar, in order: forced exit at its open (15:55, 12:55 early); a stop gap (open at or under the level) fills at
    the open; a decision stamped at this bar's time fills at its open (a fade exit, or an entry unless a position just
    closed on this bar); then the resting stop, from the highest high of EARLIER bars, fills at its level when the low
    reaches it. The stop is K x `atr` below the highest price since entry (seeded with the fill).
    """
    forced_at = time(12, 55) if early else time(15, 55)
    first_time = min(obs_by_t) if obs_by_t else None
    trades: list[dict] = []
    position: dict | None = None
    stops, day_mult = 0, 1.0

    def close_out(m: pd.Timestamp, price: float, how: str) -> None:
        nonlocal position, stops, day_mult
        mult = price * (1 - COST) / (position["entry_px"] * (1 + COST))
        trades.append({"entry": position["entry"], "exit": m, "mult": mult, "how": how})
        day_mult *= mult
        stops += how == "stop"
        position = None

    for m, bar in zip(etf.index, etf.itertuples(), strict=True):
        exited_now = False
        if position is not None and m.time() >= forced_at:
            close_out(m, bar.open, "close")
            exited_now = True
        if position is not None and bar.open <= position["peak"] - cell["k"] * atr:
            close_out(m, bar.open, "stop")
            exited_now = True
        if m in obs_by_t and not exited_now:
            obs = obs_by_t[m]
            limits_hit = stops >= MAX_STOPS or day_mult - 1 <= DAILY_LOSS
            held = (m - position["entry"]).total_seconds() / 60 if position else 0.0
            action = decide(obs, m == first_time, position is not None, held, limits_hit, len(trades) < cap_left, cell)
            if action == "exit":
                close_out(m, bar.open, "fade")
            elif action == "enter":
                position = {"entry": m, "entry_px": bar.open, "peak": bar.open}
        if position is not None:
            level = position["peak"] - cell["k"] * atr
            if bar.low <= level:
                close_out(m, level, "stop")
            else:
                position["peak"] = max(position["peak"], bar.high)
    if position is not None:
        close_out(etf.index[-1], float(etf.close.iloc[-1]), "close")
    return trades


def run_account(sessions: list[date], mk: dict, cell: dict | None, cap: bool, overnight: str) -> pd.DataFrame:
    """The fake account over `sessions`: the cell's intraday trades, SHV or cash overnight. `cell=None` never trades.

    Overnight, "shv" = SHV bought at each close and sold at the next open (Yahoo prices, 1 bp each side). With `cap`, the
    day-trade rule: at most 3 trades in any 5 sessions (executed ones; every trade opens and closes the same day).
    Rows (one per session, equity at the close): equity, trades, wins, ret_sum, stops, fades, closes, cap_skips
    (trades the cap blocked: an approximation, re-run uncapped), limit_days (1 when the day's entries ended on the loss
    limit or on two stops).
    """
    equity, out, past_trades = START, [], []
    for i, d in enumerate(sessions):
        if overnight == "shv" and i > 0:
            prev = sessions[i - 1]
            equity *= mk["yahoo"].shv_open[d] / mk["yahoo"].shv_close[prev] * (1 - SHV_COST) / (1 + SHV_COST)
        trades, cap_skips = [], 0
        etf = mk["minute"][cell["etf"]].get(d) if cell else None
        atr, obs = (mk["atr"].get((cell["etf"], d)), mk["obs"].get(d)) if cell else (None, None)
        if etf is not None and atr and obs:
            cap_left = max(0, MAX_DAY_TRADES - sum(past_trades[-(CAP_WINDOW - 1) :])) if cap else 99
            trades = simulate_session(etf, obs, cell, atr, cap_left, is_early_close(d))
            if cap and len(trades) == cap_left:
                cap_skips = len(simulate_session(etf, obs, cell, atr, 99, is_early_close(d))) - len(trades)
        day_mult = float(np.prod([t_["mult"] for t_ in trades])) if trades else 1.0
        equity *= day_mult
        past_trades.append(len(trades))
        how = [t_["how"] for t_ in trades]
        out.append(
            {
                "equity": equity,
                "trades": len(trades),
                "wins": sum(t_["mult"] > 1 for t_ in trades),
                "ret_sum": sum(t_["mult"] - 1 for t_ in trades),
                "stops": how.count("stop"),
                "fades": how.count("fade"),
                "closes": how.count("close"),
                "cap_skips": cap_skips,
                "limit_days": int(how.count("stop") >= MAX_STOPS or day_mult - 1 <= DAILY_LOSS),
            }
        )
    return pd.DataFrame(out, index=sessions)


def summary(rows: pd.DataFrame) -> dict:
    """Total, CAGR, calendar-year returns, max drawdown and the trade counts of one account run."""
    total, max_dd = _fall.curve_stats(rows.equity)
    years_equity = rows.equity.groupby([d.year for d in rows.index]).last()
    years = (years_equity / years_equity.shift(1, fill_value=START) - 1).to_dict()
    span = (rows.index[-1] - rows.index[0]).days + 1
    n = int(rows.trades.sum())
    return {
        "total": total,
        "cagr": (1 + total) ** (365.25 / span) - 1,
        "years": years,
        "max_dd": max_dd,
        "trades": n,
        "win_rate": float(rows.wins.sum() / n) if n else 0.0,
        "mean_trade": float(rows.ret_sum.sum() / n) if n else 0.0,
        "stops": int(rows.stops.sum()),
        "fades": int(rows.fades.sum()),
        "closes": int(rows.closes.sum()),
        "cap_skips": int(rows.cap_skips.sum()),
        "limit_days": int(rows.limit_days.sum()),
    }


def signal_gate(sessions: list[date], mk: dict, setting: dict) -> float:
    """Mean return of the -1 x SPY trade (2 - exit / entry, costs in) from each session's FIRST entry signal to the forced exit.

    No stop, no fade, no day limits, no cap: one trade per session at most. 0.0 when no session has a signal.
    """
    cell = {**setting, "etf": "SPY", "k": 0}
    returns = []
    for d in sessions:
        spy, obs = mk["minute"]["SPY"].get(d), mk["obs"].get(d)
        if spy is None or not obs:
            continue
        first_time = min(obs)
        signal = next((t_ for t_ in sorted(obs) if entry_signal(obs[t_], t_ == first_time, cell)), None)
        if signal is None or signal not in spy.index:
            continue
        forced_at = time(12, 55) if is_early_close(d) else time(15, 55)
        exits = spy[spy.index.time >= forced_at]
        entry = float(spy.open[signal])
        exit_ = float(exits.open.iloc[0]) if len(exits) else float(spy.close.iloc[-1])
        returns.append((2 - exit_ / entry) * (1 - COST) / (1 + COST) - 1)
    return float(np.mean(returns)) if returns else 0.0


def close_to(value: float, expected: float) -> bool:
    """Float compare at 1e-9."""
    return abs(value - expected) < 1e-9


def synth_day(d: date, drift: float = 0.0) -> pd.DataFrame:
    """A random-walk session of 1-minute bars with volume, 09:30-15:59 New York, seeded by the date."""
    rng = np.random.default_rng(d.toordinal())
    idx = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="min", tz=NY)
    close = 20 * np.exp(np.cumsum(rng.normal(drift / 390, 0.001, len(idx))))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) * 1.0005, "low": np.minimum(open_, close) * 0.9995, "close": close, "volume": rng.integers(100, 1000, len(idx)).astype(float)},
        index=idx,
    )


def flat_etf(d: date) -> pd.DataFrame:
    """A session of 1-minute bars flat at 100 (09:30-15:59), for the simulation checks."""
    idx = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="min", tz=NY)
    return pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)


def synthetic_market(n: int) -> dict:
    """N weekday sessions: every symbol flat at 100 (SPY ends at 99), the signal on at every decision time, SHV 100 -> 100.01."""
    days = []
    d = date(2024, 3, 4)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    obs = {"vixy_day": 0.04, "vixy_hour": 0.03, "vixy_vs_d1avg": 0.01, "vixy_trend": True, "rvol": 2.0, "spy_day": -0.006, "spy_hour": -0.001, "vixy_30": 0.01, "spy_30": -0.001}
    minute = {sym: {d: flat_etf(d) for d in days} for sym in ("SPY", "VIXY", "SDS", "SPXU")}
    for d in days:
        minute["SPY"][d].loc[pd.Timestamp(f"{d} 15:55", tz=NY), "open"] = 99.0
    return {
        "sessions": days,
        "minute": minute,
        "obs": {d: {t_: obs for t_ in decision_times(d, False)} for d in days},
        "atr": {(etf, d): 1.0 for etf in ("SDS", "SPXU") for d in days},
        "yahoo": pd.DataFrame({"shv_open": 100.01, "shv_close": 100.0}, index=days),
    }


def five_days_before(d: date) -> list[date]:
    """The five weekdays before d, most recent first."""
    out, cur = [], d
    while len(out) < 5:
        cur -= timedelta(days=1)
        if cur.weekday() < 5:
            out.append(cur)
    return out


def check() -> None:
    """Assertions on synthetic data: no look-ahead, window clips, same-clock volume, thin minutes, D-1 values."""
    day = date(2024, 3, 4)
    spy, vixy = synth_day(day), synth_day(day, drift=0.002)
    prev = [{"spy": synth_day(d), "vixy": synth_day(d)} for d in five_days_before(day)]
    ctx = day_context(prev, sma50=10.0)
    assert ctx is not None
    t = pd.Timestamp("2024-03-04 11:00", tz=NY)
    base = observe(spy, vixy, ctx, t)
    # 1. no look-ahead: changing any bar that closes after t never changes the values at t
    moved = vixy.copy()
    moved.loc[moved.index >= t, ["open", "high", "low", "close"]] *= 5
    assert observe(spy, moved, ctx, t) == base
    # 2. vixy_hour at 09:50 is clipped at today's open (window 09:30 -> 09:50), never yesterday
    t950 = pd.Timestamp("2024-03-04 09:50", tz=NY)
    assert close_to(observe(spy, vixy, ctx, t950)["vixy_hour"], vixy.close[vixy.index < t950].iloc[-1] / vixy.open.iloc[0] - 1)
    # 3. rvol compares the same clock time: today's volume to 11:00 over the mean D-1..D-5 volume to 11:00
    assert close_to(base["rvol"], vixy.volume[vixy.index < t].sum() / np.mean([p["vixy"].volume[p["vixy"].index.time < t.time()].sum() for p in prev]))
    # 4. a thin minute: drop the 10:59 VIXY bar -> "now" is the 10:58 close
    thin = vixy.drop(pd.Timestamp("2024-03-04 10:59", tz=NY))
    assert close_to(observe(spy, thin, ctx, t)["vixy_day"], thin.close[thin.index < t].iloc[-1] / ctx["vixy_d1_close"] - 1)
    # 5. D-1 VWAP and a missing history
    d1 = prev[0]["vixy"]
    assert close_to(ctx["vixy_d1_vwap"], (d1.close * d1.volume).sum() / d1.volume.sum())
    assert day_context(prev[:4], 10.0) is None
    # 6. decision times: 37 normal, 19 on an early close, first 09:40
    assert len(decision_times(date(2024, 3, 4), False)) == 37 and len(decision_times(date(2024, 11, 29), True)) == 19
    assert decision_times(date(2024, 3, 4), False)[0] == pd.Timestamp("2024-03-04 09:40", tz=NY)
    # 7. a D-k session that ended early (12:59) is left out of the volume mean once t is past its end
    short = [{"spy": prev[0]["spy"][prev[0]["spy"].index.hour < 13], "vixy": prev[0]["vixy"][prev[0]["vixy"].index.hour < 13]}, *prev[1:]]
    ctx_short = day_context(short, 10.0)
    assert ctx_short is not None
    t_late = pd.Timestamp("2024-03-04 13:10", tz=NY)
    expected = vixy.volume[vixy.index < t_late].sum() / np.mean([p["vixy"].volume[p["vixy"].index.time < t_late.time()].sum() for p in prev[1:]])
    assert close_to(observe(spy, vixy, ctx_short, t_late)["rvol"], expected)
    # 8. atr10: 10-minute bars with a constant range of 1 give an ATR of 1
    block = pd.DataFrame({"open": 100.5, "high": 101.0, "low": 100.0, "close": 100.5, "volume": 1.0}, index=pd.date_range("2024-03-01 09:30", "2024-03-01 15:59", freq="min", tz=NY))
    assert close_to(atr10([block, block]), 1.0)
    cell = {"v_day": 0.03, "v_hour": 0.02, "rvol": None, "etf": "SPXU", "k": 3}
    obs = {"vixy_day": 0.04, "vixy_hour": 0.0, "vixy_vs_d1avg": 0.01, "vixy_trend": True, "rvol": 1.0, "spy_day": -0.006, "spy_hour": -0.001, "vixy_30": 0.01, "spy_30": -0.001}
    # 9. the D-1 close trigger counts only at 09:40; later the hour trigger is needed
    assert entry_signal(obs, True, cell) and not entry_signal(obs, False, cell)
    assert entry_signal(obs | {"vixy_hour": 0.025}, False, cell)
    # 10. every confirmation is required; volume only when the cell has it
    for key, value in (("vixy_vs_d1avg", -0.01), ("vixy_trend", False), ("spy_day", -0.004), ("spy_hour", 0.001)):
        assert not entry_signal(obs | {key: value}, True, cell), key
    assert not entry_signal(obs, True, cell | {"rvol": 1.5}) and entry_signal(obs | {"rvol": 1.6}, True, cell | {"rvol": 1.5})
    # 11. fade only after 30 minutes held, and only when VIXY is down AND SPY up
    fading = obs | {"vixy_30": -0.01, "spy_30": 0.002}
    assert decide(fading, False, True, 30, False, True, cell) == "exit"
    assert decide(fading, False, True, 20, False, True, cell) == "none"
    assert decide(obs | {"vixy_30": -0.01}, False, True, 40, False, True, cell) == "none"
    # 12. limits and cap block entries
    assert decide(obs, True, False, 0, True, True, cell) == "none" and decide(obs, True, False, 0, False, False, cell) == "none"
    assert decide(obs, True, False, 0, False, True, cell) == "enter"
    # 13. simulate_session: entry at the 09:40 bar's open; a bar opening under the stop fills at its open
    on = obs | {"vixy_hour": 0.03}
    day_ = date(2024, 3, 4)
    all_on = {t_: on for t_ in decision_times(day_, False)}
    first_only = {decision_times(day_, False)[0]: on}
    etf = flat_etf(day_)
    gap = etf.copy()
    gap.loc[pd.Timestamp("2024-03-04 10:15", tz=NY), ["open", "high", "low", "close"]] = 95.0
    trades = simulate_session(gap, first_only, cell, 1.0, 3, False)
    assert len(trades) == 1 and trades[0]["how"] == "stop" and trades[0]["exit"] == pd.Timestamp("2024-03-04 10:15", tz=NY)
    assert close_to(trades[0]["mult"], 0.95 * (1 - COST) / (1 + COST)) and trades[0]["entry"] == pd.Timestamp("2024-03-04 09:40", tz=NY)
    # a bar trading down to the stop (level 97) fills at the level, not at its low
    dip = etf.copy()
    dip.loc[pd.Timestamp("2024-03-04 10:30", tz=NY), "low"] = 96.0
    trades = simulate_session(dip, first_only, cell, 1.0, 3, False)
    assert trades[0]["how"] == "stop" and close_to(trades[0]["mult"], 0.97 * (1 - COST) / (1 + COST))
    # 14. forced exit at the open of the 15:55 bar; 12:55 on an early close
    late = etf.copy()
    late.loc[pd.Timestamp("2024-03-04 15:55", tz=NY), "open"] = 101.0
    trades = simulate_session(late, first_only, cell, 1.0, 3, False)
    assert trades[0]["how"] == "close" and trades[0]["exit"] == pd.Timestamp("2024-03-04 15:55", tz=NY) and close_to(trades[0]["mult"], 1.01 * (1 - COST) / (1 + COST))
    early_day = date(2024, 11, 29)
    early_etf = flat_etf(early_day).between_time("09:30", "12:59")
    early_etf.loc[pd.Timestamp("2024-11-29 12:55", tz=NY), "open"] = 102.0
    trades = simulate_session(early_etf, {decision_times(early_day, True)[0]: on}, cell, 1.0, 3, True)
    assert trades[0]["exit"] == pd.Timestamp("2024-11-29 12:55", tz=NY) and close_to(trades[0]["mult"], 1.02 * (1 - COST) / (1 + COST))
    # 15. two stops end the day's entries; a stop worse than -2% does too; the cap limits trades
    two = etf.copy()
    for stamp in ("2024-03-04 09:50", "2024-03-04 10:10"):
        two.loc[pd.Timestamp(stamp, tz=NY), "low"] = 99.3  # atr 0.2 x k 3 -> stop at 99.4, a 0.6% loss
    assert len(simulate_session(two, all_on, cell, 0.2, 5, False)) == 2
    assert len(simulate_session(two, all_on, cell, 0.2, 1, False)) == 1
    deep = etf.copy()
    deep.loc[pd.Timestamp("2024-03-04 09:50", tz=NY), "low"] = 96.0  # atr 1 x k 3 -> stop at 97, a -3% trade
    assert len(simulate_session(deep, all_on, cell, 1.0, 5, False)) == 1
    # 16. the stop trails: price runs to 110, then a dip to 106 hits the raised level 110 - 3 = 107 (the entry level 97 would not)
    run = etf.copy()
    run.loc[run.index >= pd.Timestamp("2024-03-04 10:00", tz=NY), ["open", "high", "low", "close"]] = 110.0
    run.loc[pd.Timestamp("2024-03-04 10:30", tz=NY), "low"] = 106.0
    trades = simulate_session(run, first_only, cell, 1.0, 3, False)
    assert trades[0]["how"] == "stop" and close_to(trades[0]["mult"], 1.07 * (1 - COST) / (1 + COST))
    # 17. no re-entry on the bar a stop fired: stopped at the 09:50 open (99.0, level 99.4), the next entry is 10:00
    stopped = etf.copy()
    stopped.loc[pd.Timestamp("2024-03-04 09:50", tz=NY), ["open", "high", "low", "close"]] = 99.0
    trades = simulate_session(stopped, all_on, cell, 0.2, 5, False)
    assert trades[0]["how"] == "stop" and trades[1]["entry"] == pd.Timestamp("2024-03-04 10:00", tz=NY)
    # 18. the account: a signal every session, flat prices -> one position held to the close per session
    mk = synthetic_market(8)
    sessions = mk["sessions"]
    capped = run_account(sessions, mk, cell, True, "cash")
    assert list(capped.trades) == [1, 1, 1, 0, 0, 1, 1, 1]  # at most 3 trades in any 5 sessions
    assert capped.cap_skips.sum() > 0
    assert list(run_account(sessions, mk, cell, False, "cash").trades) == [1] * 8
    # 19. no cell = no trades; SHV overnight earns open(d+1)/close(d) less 1 bp each side, cash earns nothing
    none = run_account(sessions, mk, None, True, "shv")
    night = 100.01 / 100.0 * (1 - 1e-4) / (1 + 1e-4)
    assert none.trades.sum() == 0 and abs(none.equity.iloc[-1] - START * night**7) < 1e-6
    assert none.equity.iloc[0] == START and run_account(sessions, mk, None, True, "cash").equity.iloc[-1] == START
    # 20. summary: totals, years, the exit-cause counts
    s_ = summary(capped)
    assert s_["trades"] == 6 and s_["closes"] == 6 and s_["stops"] == 0 and abs(s_["total"] - (capped.equity.iloc[-1] / START - 1)) < 1e-12
    # 21. signal gate: SPY falls 1% from the first signal to the forced exit -> the -1 x SPY trade makes 1% less 6 bps
    assert close_to(signal_gate(sessions, mk, {"v_day": 0.03, "v_hour": 0.02, "rvol": None}), (2 - 0.99) * (1 - COST) / (1 + COST) - 1)
    assert signal_gate(sessions, mk, {"v_day": 0.5, "v_hour": 0.5, "rvol": None}) == 0.0  # no signal: nothing to average
    # 22. the gate keeps the 09:40-only rule: the D-1 move counts at the first decision (hour trigger off), and a D-1 move seen later does not
    assert close_to(signal_gate(sessions, mk, {"v_day": 0.03, "v_hour": 0.5, "rvol": None}), (2 - 0.99) * (1 - COST) / (1 + COST) - 1)
    late_only = {
        **mk,
        "obs": {d: {t_: (obs | {"vixy_day": 0.0, "vixy_hour": 0.0} if t_ == decision_times(d, False)[0] else obs | {"vixy_hour": 0.0}) for t_ in decision_times(d, False)} for d in sessions},
    }
    assert signal_gate(sessions, late_only, {"v_day": 0.03, "v_hour": 0.5, "rvol": None}) == 0.0
    # 23. a -3% stop day is a limit day with a single trade
    hit = synthetic_market(3)
    hit["minute"]["SPXU"][hit["sessions"][0]].loc[pd.Timestamp(f"{hit['sessions'][0]} 09:50", tz=NY), "low"] = 96.0
    rows = run_account(hit["sessions"], hit, cell, False, "cash")
    assert rows.limit_days.iloc[0] == 1 and rows.trades.iloc[0] == 1 and rows.stops.iloc[0] == 1 and rows.limit_days.iloc[1] == 0
    print("self-check OK")


def main() -> None:
    """Entry point: `check`, `dev` or `test` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
