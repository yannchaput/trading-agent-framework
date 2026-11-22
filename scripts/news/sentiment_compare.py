"""Spike: do HF financial-sentiment models flag SPY intraday falls better than a keyword count?

Run: uv run --with torch --with transformers python scripts/news/sentiment_compare.py [us]
     `us` keeps only headlines that name the US market (Wall Street, S&P, Dow, Nasdaq, NYSE, ...).
Window: Aug 1 - Nov 21 2026 headlines from cache/gdelt/gkg_stockmarket/, SPY 5-min bars from Yahoo (last 60
days only). Each distinct title is scored once per model and cached in cache/gdelt/sentiment_scores_2026/
(delete a file to rescore). Scoring needs a free GPU (~1-2 min per model); on CPU it takes hours.
Result 2026-11-22: unfiltered, every signal is below chance; with `us`, FinBERT net share 7/10 falls,
17/49 false alarms (29% precision vs 17% base rate) -- best of 7 signals on 10 falls, not significant.

Same protocol as intraday_test.py: per 15-min GDELT stamp, a 1-hour rolling signal z-scored against the same
time-of-day slot on the previous 20 weekdays; a day "fires" if z >= 3 between the prior close (16:00 NY) and 15:45.
Signals per model: count of negative headlines in the hour (volume-sensitive, like the keyword count) and net
negativity share mean(p_neg - p_pos) (volume-neutral).
"""

import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from transformers import pipeline

ROOT = str(Path(__file__).resolve().parents[2] / "cache" / "gdelt" / "gkg_stockmarket")
OUT = Path(ROOT).parent / "sentiment_scores_2026"
OUT.mkdir(exist_ok=True)
MODELS = {
    "finbert": "ProsusAI/finbert",
    "finsense": "AnkitAI/FinSense-ModernBERT-Financial-News-Sentiment-Analysis",
    "finbert_tone": "yiyanghkust/finbert-tone",
    "finbert_headlines": "HuangYiYang/finbert-financial-news-sentiment",  # raw scores (card's T=2.65 not applied)
    "finbert_peejm": "peejm/finbert-financial-sentiment",
}
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
torch.set_num_threads(8)  # leave cores to the GDELT fetch
GPU = torch.cuda.is_available()

# ---- headlines, Aug 1 -> Nov 21 2026 (Aug-Sep give the 20-weekday baseline)
frames = []
for p in sorted(glob.glob(f"{ROOT}/2026/*.parquet")):
    if p.split("/")[-1][:10] < "2026-08-01":
        continue
    df = pd.read_parquet(p)
    if not df.empty:
        frames.append(df[["stamp", "title", "tone"]])
arts = pd.concat(frames, ignore_index=True)
arts["title"] = arts.title.fillna("").str.strip()
arts = arts[arts.title.str.len() > 0]
titles = arts.title.drop_duplicates().tolist()
print(f"{len(arts)} articles, {len(titles)} distinct titles", flush=True)

# ---- score distinct titles once per model (cached)
for name, repo in MODELS.items():
    path = OUT / f"{name}.parquet"
    if path.exists():
        continue
    model = repo
    tok = None
    if name == "finbert_tone":  # its config.json has no model_type; labels per its card
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

        cfg = BertConfig.from_pretrained(repo, id2label={0: "neutral", 1: "positive", 2: "negative"}, label2id={"neutral": 0, "positive": 1, "negative": 2})
        model = BertForSequenceClassification.from_pretrained(repo, config=cfg)
        tok = BertTokenizer.from_pretrained(repo)
    clf = pipeline("text-classification", model=model, tokenizer=tok, device=0 if GPU else -1, dtype=torch.float16 if GPU else None, top_k=None, truncation=True, max_length=64)
    t0 = time.time()
    res = clf(titles, batch_size=64)
    rows = []
    for t, r in zip(titles, res, strict=True):
        d = {x["label"].lower(): x["score"] for x in r}
        rows.append((t, d.get("negative", 0.0), d.get("positive", 0.0)))
    pd.DataFrame(rows, columns=["title", "p_neg", "p_pos"]).to_parquet(path, index=False)
    print(f"{name}: {len(titles)} titles in {time.time() - t0:.0f} s, labels {sorted(d)}", flush=True)

# ---- optional US-market filter (decided before seeing results): the title must name the US market
US = re.compile(r"wall street|\bs ?& ?p\b|\bs&p 500\b|\bdow\b|\bnasdaq\b|\bnyse\b|russell 2000|\bu\.?s\.? stocks|american stocks", re.I)
if "us" in sys.argv[1:]:
    arts = arts[arts.title.map(lambda t: bool(US.search(t)))]
    print(f"US filter: {len(arts)} articles kept, {arts.title.nunique()} distinct titles", flush=True)

# ---- per-stamp signals
arts["kw"] = arts.title.map(lambda t: bool(FALL.search(t))).astype(float)
arts["gdelt_neg"] = -arts.tone  # GDELT V2Tone (dictionary, full article); sign flipped so higher = more negative
signals = {"keyword_count": ("kw", "count"), "gdelt_tone_share": ("gdelt_neg", "share")}
for name in MODELS:
    s = pd.read_parquet(OUT / f"{name}.parquet")
    arts = arts.merge(s.rename(columns={"p_neg": f"{name}_neg", "p_pos": f"{name}_pos"}), on="title", how="left")
    arts[f"{name}_isneg"] = (arts[f"{name}_neg"] > arts[f"{name}_pos"]) & (arts[f"{name}_neg"] > 0.5)
    arts[f"{name}_net"] = arts[f"{name}_neg"] - arts[f"{name}_pos"]
    signals[f"{name}_neg_count"] = (f"{name}_isneg", "count")
    signals[f"{name}_net_share"] = (f"{name}_net", "share")

idx_all = pd.date_range("2026-08-01", "2026-11-22", freq="15min", tz="UTC")
stamps = pd.to_datetime(arts.stamp, format="%Y%m%d%H%M%S", utc=True)
spy = yf.download("SPY", period="60d", interval="5m", progress=False, auto_adjust=True)["Close"].squeeze()
spy.index = spy.index.tz_convert("America/New_York")
close = spy.groupby(spy.index.date).last()
days = sorted(close.index)[1:]
lows = {d: (spy[spy.index.date == d] / close.shift(1)[d] - 1).min() for d in days}
fall_day = {d: lows[d] <= -0.01 for d in days}


def zscore(col: str, kind: str) -> pd.Series:
    """Hourly rolling signal of `col` z-scored against the same 15-min slot on the previous 20 weekdays."""
    g = arts.groupby(stamps)[col]
    if kind == "count":
        per = g.sum().reindex(idx_all, fill_value=0).astype(float)
        sig = per.rolling(4).sum()
    else:  # mean over the hour, weighted by article count
        s, n = g.sum().reindex(idx_all, fill_value=0), g.size().reindex(idx_all, fill_value=0)
        sig = s.rolling(4).sum() / n.rolling(4).sum().replace(0, np.nan)
    ny = idx_all.tz_convert("America/New_York")
    frame = pd.DataFrame({"sig": sig.values, "slot": ny.strftime("%H:%M"), "wk": ny.dayofweek < 5}, index=idx_all)
    z = pd.Series(np.nan, index=idx_all)
    for _, grp in frame[frame.wk].groupby("slot"):
        past = grp.sig.shift(1).rolling(20, min_periods=10)
        z.loc[grp.index] = (grp.sig - past.mean()) / past.std().replace(0, np.nan)
    return z


rows = []
for label, (col, kind) in signals.items():
    z = zscore(col, kind)
    fired, lead = {}, {}
    for i, d in enumerate(days):
        start = pd.Timestamp(f"{sorted(close.index)[i]} 16:00", tz="America/New_York").tz_convert("UTC")
        end = pd.Timestamp(f"{d} 15:45", tz="America/New_York").tz_convert("UTC")
        w = z[(z.index >= start) & (z.index <= end)]
        hit = w[w >= 3]
        fired[d] = not hit.empty
        if fired[d]:
            lead[d] = hit.index.min().tz_convert("America/New_York").strftime("%m-%d %H:%M")
    tp = sum(fired[d] and fall_day[d] for d in days)
    fp = sum(fired[d] and not fall_day[d] for d in days)
    nf = sum(fall_day.values())
    key = {d: lead.get(d, "-") for d in days if str(d) in ("2026-11-05", "2026-11-16", "2026-11-19", "2026-11-20")}
    rows.append((label, f"{tp}/{nf}", f"{fp}/{len(days) - nf}", f"{tp / max(tp + fp, 1):.0%}", *key.values()))
cols = ["signal", "falls caught", "false alarms", "precision", "11-05 (-4.4%)", "11-16 (-1.9%)", "11-19 (-2.0%)", "11-20 (-3.5%)"]
pd.set_option("display.width", 220)
print(pd.DataFrame(rows, columns=cols).to_string(index=False))
sys.exit(0)
