"""Spike (throwaway): GDELT GKG stock-market articles, one parquet per day, into cache/gdelt/gkg_stockmarket/.

Run: uv run python scripts/news/gdelt_gkg_fetch.py [--start 2015-02-18] [--end YYYY-MM-DD] [--workers 12]

GDELT publishes a GKG file every 15 minutes (~5-17 MB zipped, static downloads, no API quota). For each UTC day
the script downloads the 96 files, keeps only the articles tagged ECON_STOCKMARKET and writes one parquet:

    cache/gdelt/gkg_stockmarket/<YYYY>/<YYYY-MM-DD>.parquet   one row per article
    cache/gdelt/gkg_stockmarket/<YYYY>/<YYYY-MM-DD>.stats.json  {stamp: [all_articles, kept] | null if GDELT has no file}

Columns: stamp (UTC "YYYYMMDDHHMMSS" of the 15-minute file, the earliest moment the article is known),
domain, url, title (only filled from ~2020-03; before that the URL slug is the headline), tone, pos, neg
(GDELT V2Tone), themes (the ECON_/EPU_ theme names of the article).
stats.json holds the per-file article totals, so a tone share can be normalised by the news volume.

Newest day first, resumable (a day already on disk is skipped), today is never fetched (incomplete).
Look-ahead rule for readers: only use rows whose stamp is <= the decision time.
"""

import argparse
import csv
import html
import io
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

URL = "https://data.gdeltproject.org/gdeltv2/{}.gkg.csv.zip"
ROOT = Path(__file__).resolve().parents[2] / "cache" / "gdelt" / "gkg_stockmarket"
FIRST_DAY = date(2015, 2, 19)  # the first GKG 2.1 file is 20150218230000
THEME = "ECON_STOCKMARKET"
TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
THEMES = re.compile(r"\b(?:ECON|EPU)_[A-Z_]+")
csv.field_size_limit(10**9)


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def parse(raw: bytes) -> tuple[int, list[dict]]:
    """Return (all article count, kept articles) of one zipped GKG file."""
    zf = zipfile.ZipFile(io.BytesIO(raw))
    reader = csv.reader(io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8", errors="replace"), delimiter="\t", quoting=csv.QUOTE_NONE)
    total, kept = 0, []
    for r in reader:
        if len(r) < 27:  # malformed line
            continue
        total += 1
        themes = r[8] or r[7]
        if THEME not in themes:
            continue
        tone = (r[15].split(",") + ["", "", ""])[:3]
        m = TITLE.search(r[26])
        kept.append(
            {
                "stamp": r[1],
                "domain": r[3],
                "url": r[4],
                "title": html.unescape(m.group(1)).strip() if m else None,
                "tone": float(tone[0]) if tone[0] else None,
                "pos": float(tone[1]) if tone[1] else None,
                "neg": float(tone[2]) if tone[2] else None,
                "themes": ",".join(sorted(set(THEMES.findall(themes)))),
            }
        )
    return total, kept


def fetch(stamp: str) -> tuple[str, int, list[dict]] | tuple[str, None, None]:
    """Download and parse one file. A 404 is a real gap (None); other errors are retried, then raised."""
    err: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL.format(stamp), headers={"User-Agent": "research-spike"})
            total, kept = parse(urllib.request.urlopen(req, timeout=120).read())
            return stamp, total, kept
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return stamp, None, None
            err = exc
        except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile, ConnectionError) as exc:
            err = exc
        time.sleep(10 * 2**attempt)
    raise RuntimeError(f"{stamp}: giving up after 5 attempts ({err!r})")


def fetch_day(day: date, pool: ThreadPoolExecutor) -> tuple[pd.DataFrame, dict]:
    stamps = [f"{day:%Y%m%d}{h:02d}{m:02d}00" for h in range(24) for m in range(0, 60, 15)]
    rows, stats = [], {}
    for stamp, total, kept in pool.map(fetch, stamps):
        kept = kept or []
        stats[stamp] = None if total is None else [total, len(kept)]
        rows += kept
    return pd.DataFrame(rows), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=FIRST_DAY)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    end = min(args.end, date.today() - timedelta(days=1))

    day, done, t0 = end, 0, time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        while day >= max(args.start, FIRST_DAY):
            path = ROOT / f"{day:%Y}" / f"{day}.parquet"
            if not path.exists():
                df, stats = fetch_day(day, pool)
                path.parent.mkdir(parents=True, exist_ok=True)
                stats_path = path.with_suffix(".stats.json")
                stats_path.write_text(json.dumps(stats))
                df.to_parquet(path.with_suffix(".tmp"), index=False)
                path.with_suffix(".tmp").replace(path)  # parquet last: its presence means the day is complete
                done += 1
                gaps = sum(v is None for v in stats.values())
                log(f"{day} {len(df):>6} articles, {gaps} missing files ({done} days, {(time.time() - t0) / done:.0f} s/day)")
            day -= timedelta(days=1)
    log(f"done: {done} days fetched into {ROOT}")


if __name__ == "__main__":
    main()
