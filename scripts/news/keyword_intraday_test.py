"""Spike: does a burst of falling-market headlines (15-min GDELT stamps) lead SPY's intraday drop?

Run: uv run python scripts/news/keyword_intraday_test.py
Keyword count of fall words in titles, 1-hour rolling, z-scored against the same 15-min slot on the previous
20 weekdays; compared with SPY 5-min bars (Yahoo, last 60 days only).
Result 2026-11-22 (all headlines, unfiltered): 3/10 fall days caught, 22/49 false alarms (12% precision);
the Nov 5 "lead" was midterm-election coverage ("election rout").
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = str(Path(__file__).resolve().parents[2] / "cache" / "gdelt" / "gkg_stockmarket")
FALL = re.compile(
    r"\b(?:"
    + "|".join(
        [
            "fall|falls|fell",
            "plunge|plunges|plunged",
            "tumble|tumbles|tumbled",
            "slump|slumps",
            "slide|slides",
            "rout|crash|selloff|sell-off",
            "dive|dives|sink|sinks",
            "slips?|drops?|dropped|losses|lower",
        ]
    )
    + r")\b",
    re.I,
)

# ---- news: per 15-min stamp, count of fall titles and of all stock-market articles
rows = []
for p in sorted(glob.glob(f"{ROOT}/202[56]/*.parquet")):
    d = p.split("/")[-1][:10]
    if d < "2026-08-01":
        continue
    df = pd.read_parquet(p)
    if df.empty:
        continue
    df["fall"] = df.title.fillna("").map(lambda t: bool(FALL.search(t)))
    g = df.groupby("stamp").agg(n=("fall", "size"), fall=("fall", "sum"), tone=("tone", "mean"))
    rows.append(g)
news = pd.concat(rows)
news.index = pd.to_datetime(news.index, format="%Y%m%d%H%M%S", utc=True)
news = news.reindex(pd.date_range(news.index.min(), news.index.max(), freq="15min", tz="UTC")).fillna({"n": 0, "fall": 0})
news["fall1h"] = news["fall"].rolling(4).sum()  # one hour of fall-title counts, known at the stamp

# z against the same 15-min slot on the previous 20 weekdays
ny = news.index.tz_convert("America/New_York")
news["slot"] = ny.strftime("%H:%M")
news["wk"] = ny.dayofweek < 5
base = news[news.wk]
z = pd.Series(np.nan, index=news.index)
for _, grp in base.groupby("slot"):
    m = grp.fall1h.shift(1).rolling(20, min_periods=10).mean()
    s = grp.fall1h.shift(1).rolling(20, min_periods=10).std()
    z.loc[grp.index] = (grp.fall1h - m) / s.replace(0, np.nan)
news["z"] = z

# ---- price: SPY 5m
spy = yf.download("SPY", period="60d", interval="5m", progress=False, auto_adjust=True)
spy = spy["Close"].squeeze()
spy.index = spy.index.tz_convert("America/New_York")
daily_close = spy.groupby(spy.index.date).last()
prev_close = daily_close.shift(1)


def fmt(t: pd.Timestamp) -> str:
    """New York time of `t` as MM-DD HH:MM, or '-' when missing."""
    return "-" if pd.isna(t) else t.tz_convert("America/New_York").strftime("%m-%d %H:%M")


days = sorted(daily_close.index)
out = []
for d in days[1:]:
    bars = spy[spy.index.date == d]
    pc = prev_close[d]
    ret = bars / pc - 1
    open_t = pd.Timestamp(f"{d} 09:30", tz="America/New_York")
    close_ret = ret.iloc[-1]
    first_m1 = ret[ret <= -0.01].index.min()  # first 5-min bar 1% below prior close
    low = ret.idxmin()
    # news signal window: prior day 16:00 NY (after the close) up to this day's 15:45 NY
    start = pd.Timestamp(f"{daily_close.index[days.index(d) - 1]} 16:00", tz="America/New_York")
    win = news[(news.index >= start.tz_convert("UTC")) & (news.index <= pd.Timestamp(f"{d} 15:45", tz="America/New_York").tz_convert("UTC"))]
    first3 = win.index[win.z >= 3].min() if (win.z >= 3).any() else pd.NaT
    first2 = win.index[win.z >= 2].min() if (win.z >= 2).any() else pd.NaT
    out.append((str(d), round(close_ret * 100, 2), round(ret.min() * 100, 2), fmt(first_m1), fmt(low), fmt(first2), fmt(first3), round(win.z.max(), 1)))
res = pd.DataFrame(out, columns=["day", "close%", "low%", "first -1%", "low at", "news z>=2", "news z>=3", "max z"])
pd.set_option("display.width", 200)
print("SPY 5m days:", len(res), "| news days with z:", int(news.z.notna().sum() / 96))
print("\n=== days with a >=1% fall from prior close at any time, or a news z>=3 ===")
sel = res[(res["low%"] <= -1.0) | (res["news z>=3"] != "-")]
print(sel.to_string(index=False))
big = res["low%"] <= -1.0
sig = res["news z>=3"] != "-"
print(f"\nfall days (low <= -1%): {int(big.sum())}; with news z>=3 that day: {int((big & sig).sum())}")
print(f"non-fall days: {int((~big).sum())}; with news z>=3 (false alarms): {int((~big & sig).sum())}")
