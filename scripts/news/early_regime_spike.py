"""Spike: news proposes, VIX confirms -- early detection of a building SPY fall.

Design: docs/superpowers/specs/2026-11-22-news-vix-early-regime-spike-design.md (rules fixed before running).

Run:
  uv run --with torch --with transformers python scripts/news/early_regime_spike.py score   # FinBERT, needs a free GPU
  uv run python scripts/news/early_regime_spike.py dev        # 2020 -> 2024: run, read, then freeze
  uv run python scripts/news/early_regime_spike.py holdout    # 2025 -> 2026-11-21: run ONCE, after freezing

A daily verdict at 09:00 New York: headlines stamped <= 09:00 that day and market closes up to the previous session.
"""

import glob
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "cache" / "gdelt"
NEWS = ROOT / "gkg_stockmarket"
SCORES = ROOT / "finbert_us_scores.parquet"  # title -> net = p(negative) - p(positive), ProsusAI/finbert
US = re.compile(r"wall street|\bs ?& ?p\b|\bs&p 500\b|\bdow\b|\bnasdaq\b|\bnyse\b|russell 2000|\bu\.?s\.? stocks|american stocks", re.I)
TICKER = re.compile(r"\(\s*[A-Z][A-Z ]+:\s*[A-Z0-9.\-]+\s*\)", re.I)  # (NASDAQ:XYZ), (NYSE:C): single-company items

# Rules (design doc, fixed before running)
DRIFT_ALERT, ALERT_SESSIONS = 1.5, 5  # was 2.0 (99th pct, 9 alerts in 5 years); changed on dev, see spec change log
RATIO_CONFIRM, VIXY_CONFIRM = 0.95, 0.10
RATIO_EXIT, DRIFT_EXIT, EXIT_SESSIONS = 0.90, 0.5, 3
CUSUM_K, CUSUM_H, CUSUM_NORM, CUSUM_MIN = 0.5, 4.0, 252, 60  # variant C (spec): textbook k and h, not fitted
FALL = 0.08
WINDOWS = {"dev": (date(2020, 1, 1), date(2024, 12, 31)), "holdout": (date(2025, 1, 1), date(2026, 11, 21))}


def us_headlines() -> pd.DataFrame:
    """All US-market headlines with a title: columns t (UTC timestamp), title."""
    frames = []
    for p in sorted(glob.glob(f"{NEWS}/*/*.parquet")):
        df = pd.read_parquet(p, columns=["stamp", "title"]) if Path(p).stat().st_size > 1000 else pd.DataFrame()
        if df.empty or df.title.isna().all():
            continue
        df = df.dropna(subset=["title"])
        frames.append(df[df.title.str.contains(US) & ~df.title.str.contains(TICKER)])
    arts = pd.concat(frames, ignore_index=True)
    arts["t"] = pd.to_datetime(arts.stamp, format="%Y%m%d%H%M%S", utc=True)
    return arts[["t", "title"]]


def score() -> None:
    """Score every new distinct US title with FinBERT; append to the cache."""
    import torch
    from transformers import pipeline

    arts = us_headlines()
    done = pd.read_parquet(SCORES) if SCORES.exists() else pd.DataFrame(columns=["title", "net"])
    todo = sorted(set(arts.title) - set(done.title))
    print(f"{arts.t.min():%Y-%m-%d} -> {arts.t.max():%Y-%m-%d}: {len(arts)} US headlines, {len(todo)} titles to score", flush=True)
    if not todo:
        return
    gpu = torch.cuda.is_available()
    torch.set_num_threads(4)
    clf = pipeline("text-classification", model="ProsusAI/finbert", device=0 if gpu else -1, dtype=torch.float16 if gpu else None, top_k=None, truncation=True, max_length=64)
    for i in range(0, len(todo), 50_000):  # checkpoint every 50k titles
        chunk = todo[i : i + 50_000]
        res = clf(chunk, batch_size=64)
        net = [next(x["score"] for x in r if x["label"] == "negative") - next(x["score"] for x in r if x["label"] == "positive") for r in res]
        done = pd.concat([done, pd.DataFrame({"title": chunk, "net": net})], ignore_index=True)
        done.to_parquet(SCORES, index=False)
        print(f"scored {i + len(chunk)}/{len(todo)}", flush=True)


def cut(d: date) -> pd.Timestamp:
    """The 09:00 New York decision time of session d, in UTC."""
    return pd.Timestamp(f"{d} 09:00", tz="America/New_York").tz_convert("UTC")


def ny_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Midnight (naive) of the New York calendar day of `ts`."""
    return ts.tz_convert("America/New_York").tz_localize(None).normalize()


def scored_headlines() -> pd.DataFrame:
    """US-market headlines joined with their cached FinBERT net score."""
    return us_headlines().merge(pd.read_parquet(SCORES), on="title", how="inner")


def news_drift(arts: pd.DataFrame, sessions: list[date]) -> pd.Series:
    """News drift at 09:00 NY of each session (design rule 1)."""
    hourly = arts.set_index("t").net.resample("1h").agg(["sum", "count"])
    csum, ccnt = hourly["sum"].cumsum(), hourly["count"].cumsum()
    daily = arts.assign(day=arts.t.dt.tz_convert("America/New_York").dt.date).groupby("day").net.mean()
    daily.index = pd.to_datetime(daily.index)

    def window_mean(a: pd.Timestamp, b: pd.Timestamp) -> float:  # mean net of hourly bins labelled in [a, b): bins after the cut are excluded
        i, j = csum.index.searchsorted(a, "left") - 1, csum.index.searchsorted(b, "left") - 1
        if i < 0 or j < 0:
            return np.nan
        n = ccnt.iloc[j] - ccnt.iloc[i]
        return (csum.iloc[j] - csum.iloc[i]) / n if n > 0 else np.nan

    first = arts.t.min()
    out = {}
    for d in sessions:
        c = cut(d)
        w0 = c - pd.Timedelta(hours=72)
        if w0 - pd.Timedelta(days=90) < first:  # not enough history for the 30 + 60 day baselines
            continue
        recent, base = window_mean(w0, c), window_mean(w0 - pd.Timedelta(days=30), w0)
        prior = daily[(daily.index < ny_date(w0)) & (daily.index >= ny_date(w0 - pd.Timedelta(days=60)))]
        sd = prior.std()
        # positive drift = more negative news than usual
        out[d] = (recent - base) / sd if sd and sd > 0 and len(prior) >= 30 else np.nan
    return pd.Series(out, dtype=float)


def news_cusum(arts: pd.DataFrame, sessions: list[date]) -> pd.Series:
    """Variant C: CUSUM of the daily news value against its one-year norm (spec rules C1-C3).

    `sessions` must be every session since the titles start (the norm and S carry over), sliced by the caller.
    """
    t = arts.t.sort_values()
    net = arts.net.loc[t.index].to_numpy()
    pos = t.searchsorted([cut(d) for d in sessions], side="right")  # headlines stamped <= each cut
    s = {}
    for i, d in enumerate(sessions[1:], 1):
        chunk = net[pos[i - 1] : pos[i]]  # after the previous session's cut, up to this one
        s[d] = chunk.mean() if len(chunk) else np.nan
    s = pd.Series(s, dtype=float)
    norm = s.shift(1).rolling(CUSUM_NORM, min_periods=CUSUM_MIN)
    x, n = (s - norm.mean()) / norm.std(), norm.count()  # x > 0 = more negative news than the norm
    S, out = 0.0, {}
    for d, v in x.items():
        if not np.isnan(v):  # a session without headlines (GDELT gap) leaves S unchanged
            S = max(0.0, S + v - CUSUM_K)
        out[d] = S if n[d] >= CUSUM_MIN else np.nan
    return pd.Series(out, dtype=float)


def market() -> pd.DataFrame:
    """Daily SPY/VIX/VIX3M/VIXY closes (Yahoo) with the previous-session values known at 09:00."""
    import yfinance as yf

    raw = yf.download(["SPY", "^VIX", "^VIX3M", "VIXY"], start="2019-01-01", end="2026-11-22", progress=False, auto_adjust=True)["Close"]
    m = raw.rename(columns={"^VIX": "vix", "^VIX3M": "vix3m", "SPY": "spy", "VIXY": "vixy"}).dropna(subset=["spy"])
    m.index = m.index.date
    # values known at 09:00 on day t: previous session's closes
    m["ratio_prev"] = (m.vix / m.vix3m).shift(1)
    m["vixy5_prev"] = (m.vixy / m.vixy.shift(5) - 1).shift(1)
    m["spy_prev"] = m.spy.shift(1)
    return m


def falls(spy: pd.Series) -> list[tuple[date, date, float]]:
    """SPY swing highs followed by a drop >= FALL: (high day, trough day, depth)."""
    out, up, hi_d, hi, lo_d, lo = [], True, spy.index[0], spy.iloc[0], None, None
    for d, px in spy.items():
        if up:
            if px > hi:
                hi_d, hi = d, px
            elif px <= hi * (1 - FALL):
                up, lo_d, lo = False, d, px
        else:
            if px < lo:
                lo_d, lo = d, px
            elif px >= lo * (1 + FALL):
                out.append((hi_d, lo_d, lo / hi - 1))
                up, hi_d, hi = True, d, px
    if not up:
        out.append((hi_d, lo_d, lo / hi - 1))
    return out


def run_switch(opener: pd.Series | None, confirm: pd.Series, calm: pd.Series) -> pd.Series:
    """Daily state, True = defensive.

    `opener` True opens a 5-session window; `defensive` starts on a session of the window where `confirm` is
    True (no opener: `confirm` alone switches). Back to normal after 3 consecutive `calm` sessions.
    """
    state, window_left, quiet, out = False, 0, 0, {}
    for d in confirm.index:
        window_left = ALERT_SESSIONS if (opener is not None and opener[d]) else max(window_left - 1, 0)
        if not state:
            if confirm[d] and (opener is None or window_left > 0):
                state, quiet = True, 0
        else:
            quiet = quiet + 1 if calm[d] else 0
            if quiet >= EXIT_SESSIONS:
                state = False
        out[d] = state
    return pd.Series(out)


def evaluate(name: str, df: pd.DataFrame, state: pd.Series, fall_list: list) -> None:
    """Print the switches of one variant against the SPY falls: lag, drawdown, misses, false switches."""
    starts = [d for d, prev in zip(state.index[1:], state.values[:-1], strict=True) if state[d] and not prev]
    if state.iloc[0]:
        starts.insert(0, state.index[0])
    print(f"\n=== {name}: {len(starts)} defensive switches, {state.mean():.0%} of sessions defensive")
    used, lags, dds = set(), [], []
    for hi_d, lo_d, depth in fall_list:
        already = bool(state.get(hi_d, False))
        inside = [s for s in starts if hi_d < s <= lo_d]
        used.update(inside)
        if already:
            print(f"  fall {hi_d} -> {lo_d} ({depth:.1%}): already defensive at the high")
        elif inside:
            s = inside[0]
            dd = df.spy_prev[s] / df.spy[hi_d] - 1
            n = sum(1 for x in df.index if hi_d < x <= s)
            lags.append(n)
            dds.append(dd)
            print(f"  fall {hi_d} -> {lo_d} ({depth:.1%}): defensive {s}, {dd:+.1%} from the high, {n} sessions in {'OK' if dd > -0.03 else 'LATE'}")
        else:
            print(f"  fall {hi_d} -> {lo_d} ({depth:.1%}): MISSED")
    if lags:
        print(f"  detected inside {len(lags)} falls: median {np.median(lags):.0f} sessions after the high (range {min(lags)}-{max(lags)}), median drawdown {np.median(dds):+.1%}")
    false = [s for s in starts if s not in used and not any(h <= s <= lo for h, lo, _ in fall_list)]
    years = max((df.index[-1] - df.index[0]).days / 365.25, 0.1)
    print(f"  false switches (start outside any fall): {len(false)} ({len(false) / years:.1f}/year): {', '.join(map(str, false))}")
    print("  defensive share by year: " + ", ".join(f"{y}: {s.mean():.0%}" for y, s in state.groupby([d.year for d in state.index])))


def main() -> None:
    """Entry point: `score`, `dev` or `holdout` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "score":
        return score()
    start, end = WINDOWS[mode]
    m = market()
    arts = scored_headlines()
    first_news = arts.t.min().date()
    all_sessions = [d for d in m.index if first_news <= d <= end]  # CUSUM carries state from the first headline
    sessions = [d for d in all_sessions if d >= start]
    m["drift"] = news_drift(arts, sessions)
    m["cusum"] = news_cusum(arts, all_sessions)
    df = m.loc[[d for d in sessions if not (np.isnan(m.drift.get(d, np.nan)) or np.isnan(m.cusum.get(d, np.nan)))]]
    vix = (df.ratio_prev >= RATIO_CONFIRM) | (df.vixy5_prev >= VIXY_CONFIRM)
    drift_hi, cusum_hi = df.drift >= DRIFT_ALERT, df.cusum >= CUSUM_H
    years = (df.index[-1] - df.index[0]).days / 365.25
    print(f"{mode}: {df.index[0]} -> {df.index[-1]}, {len(df)} sessions with both news features")
    print(f"VIX stress on {int(vix.sum())} sessions; drift >= {DRIFT_ALERT} on {int(drift_hi.sum())}; CUSUM >= {CUSUM_H} on {int(cusum_hi.sum())}")
    alarms = int((cusum_hi & ~cusum_hi.shift(1, fill_value=False)).sum())
    print(f"CUSUM alarms (crossings of h): {alarms} ({alarms / years:.1f}/year)")
    fl = [f for f in falls(m.spy) if df.index[0] <= f[1] and f[0] <= df.index[-1]]  # falls overlapping the window
    vix_calm = df.ratio_prev < RATIO_EXIT
    evaluate("news drift proposes, VIX confirms", df, run_switch(drift_hi, vix, vix_calm & (df.drift < DRIFT_EXIT)), fl)
    evaluate("VIX proposes, news drift confirms", df, run_switch(vix, drift_hi, vix_calm & (df.drift < DRIFT_EXIT)), fl)
    evaluate("C: VIX proposes, news CUSUM confirms", df, run_switch(vix, cusum_hi, vix_calm & (df.cusum < CUSUM_H / 2)), fl)
    evaluate("CUSUM alone (no VIX)", df, run_switch(None, cusum_hi, df.cusum < CUSUM_H / 2), fl)
    evaluate("baseline: VIX alone", df, run_switch(None, vix, vix_calm), fl)


if __name__ == "__main__":
    main()
