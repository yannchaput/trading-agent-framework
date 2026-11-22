# Spike: news proposes, VIX confirms -- early detection of a building fall

Date: 2026-11-22. Status: **closed, NO** (see Results). Type: research spike (one throwaway script, no production code).

## Intent

Daily strategies should switch defensive at the **early stage of a multi-week fall**, not minutes or hours
after a crash, and not before it. Earlier research (2026-11-19/20, all NO) found price and volatility signals
alone turn bearish around -5.2% into a fall against a -3% goal. This spike asks one question: **does a
multi-day drift in US-market news sentiment, confirmed by VIX/VIXY, get us in earlier, at an acceptable
false-switch rate?**

## Goals

- A daily verdict at 09:00 New York: `normal` or `defensive`.
- Success: defensive within the first 3% of every SPY fall of at least 8% (peak to trough), with few false
  switches per year.
- Show what the news adds: compare against the same market confirmation **without** news.

## Non-goals

- No strategy integration, no orders, no P&L backtest of a defensive switch.
- No reuse of the existing regime code (`core/regime*.py`); the spike stands alone.
- No LLM. No hourly crash alarm.

## Anti-goals (a failure even if it "works")

- Tuning any threshold after looking at the held-out years.
- A detector that is only "VIX is high": if the no-news baseline does as well, news adds nothing.

## Approaches considered

- A. Composite z-score of news drift + VIX level + term structure + VIX/VIXY trend. Simple, but every input
  spikes on the same bad day, so it risks re-learning "VIX is high".
- **B. News proposes, market confirms (chosen).** A news drift raises an alert; it becomes `defensive` only if
  the volatility market confirms within a few sessions. Each signal filters the other's noise.
- C. Learned model (logistic regression or jump model). Too few falls with headline titles (~8-10 since
  2020) to learn weights without overfitting.

## Data

- News: `cache/gdelt/gkg_stockmarket/` (GDELT GKG, ECON_STOCKMARKET). Keep only titles that name the US market
  (regex in `scripts/news/sentiment_compare.py`), **and drop titles with an `(EXCHANGE:TICKER)` tag** such as
  `(NASDAQ:XYZ)` or `(NYSE:C)` (single-company items, not market news). Score each distinct title once with **ProsusAI/finbert**;
  net score = p(negative) - p(positive). Titles exist only from 2019-09-23 (measured); earlier days are out of scope
  (URL slugs are not scored).
- Market: Yahoo daily closes of SPY, ^VIX, ^VIX3M, VIXY. No look-ahead: at 09:00 on day t only closes up to
  t-1 and headlines stamped <= 09:00 on day t are used.

## Rules (fixed before running; my calls, correct them before approval)

1. **News drift** at 09:00 on day t: mean net score of US headlines over the last 72 hours, minus the mean
   over the 30 days before that, divided by the standard deviation of the daily mean net score over the 60
   days before that.
2. **Alert**: news drift >= 2.0. An alert stays open for 5 sessions.
3. **Confirmation** (any one, on t-1 closes): VIX / VIX3M >= 0.95 (term structure flat or inverted), or VIXY
   5-session return >= +10%.
4. **Defensive** starts on the first session with an open alert and a confirmation.
5. **Exit**: back to `normal` after 3 consecutive sessions with VIX / VIX3M < 0.90 and news drift < 0.5.
6. **Baseline (no news)**: rule 3 alone switches defensive; same exit without the news condition.

## Evaluation

- Reference falls: SPY closes, a swing high followed by a drop of at least 8% (computed fresh in the script).
- Per fall: SPY drawdown from the swing high on the day `defensive` starts (target <= 3%), sessions from
  the high, and whether it starts before the trough at all.
- False switches: defensive episodes that start outside any reference fall (from the swing high to the trough).
- Also: share of sessions spent defensive, per year.
- Report the same table for the no-news baseline.

## Windows

- Development: 2020-01-02 (first session with ~90 days of titled news) to 2024-12-31. Run, read, and only then freeze.
- Held-out: 2025-01-01 to 2026-11-21 (includes the April 2025 fall and both 2026 falls, including the current
  one). Run **once** after the rules are frozen; no changes after.

## Constraints

- FinBERT scoring of ~1.3 M US titles: a few minutes on a free GPU, ~6 h on 4 CPU threads. Needs a GPU window
  (vLLM stopped) or an overnight CPU run. Scores cached by title in `cache/gdelt/`.
- Code lives in `scripts/news/`, not in `src/`.

## Change log

- 2026-11-22, before any `dev` run: drop `(EXCHANGE:TICKER)` titles. Reason: 83% of the 1.39 M US-filtered
  titles were ticker-tagged single-company items ("shares bought by ...") that the US regex matched through
  "NYSE"/"Nasdaq" in the tag, drowning the market headlines. Decided from what the titles are, not from any
  result of this spike.

- 2026-11-22, **after reading the `dev` run** (declared tuning on dev): alert threshold 2.0 -> 1.5. Reason: 2.0 was
  the 99th percentile of the drift (9 alerts in 5 years, 5 of 8 falls missed), while the drift peaked at
  +1.1 to +2.1 in the first 15 sessions of every fall. The `dev` run also showed the news peaks on the same
  day as VIX confirmation or later, never before (except COVID), so the goal changes from "earlier than VIX"
  to "news filters VIX false switches without delaying it". The dev numbers after this change are optimistic;
  only the holdout run counts.

- 2026-11-22, after reading the `dev` run (declared tuning on dev): add the reverse ordering, **VIX proposes,
  news confirms** (VIX stress opens a 5-session window; the switch happens on the session the news drift
  reaches 1.5 inside it). This was the user's intended reading of approach B; the `dev` diagnostic (news peaks
  on or after the VIX confirmation day) also favours it. Same thresholds and exit rule. Both orderings and the
  no-news baseline are reported.

## Variant C: cumulative news drift (CUSUM), VIX proposes, news confirms

Added 2026-11-22 after the `dev` runs above showed why the 72 h / 30-day drift misses grinding falls: in a
long fall the 30-day norm sinks with the news, so a new leg never stands out (Jan 2022 peaked at 1.48,
2023 at 1.49, Mar 2022 at 1.12). Chosen by the user over a long-norm level (A) and a two-scale max (B).
Rules fixed before running; `k` and `h` are textbook values, not fitted to the falls.

1. **Daily news value** `s_t` at 09:00 New York of session t: mean FinBERT net score of the filtered US
   headlines stamped after the previous session's 09:00 cut and up to this one (weekend news lands on Monday).
2. **Norm**: mean and standard deviation of `s` over the previous 252 sessions (excluding t), at least 60
   sessions (so the first value is ~Dec 2019 and COVID stays testable). `x_t = (s_t - mean) / sd`;
   positive = more negative news than the norm.
3. **CUSUM**: `S_t = max(0, S_{t-1} + x_t - k)` with `k = 0.5` (tuned to a 1-sd shift). News alarm when
   `S_t >= h = 4` (about one false alarm every ~8 months on independent data; news tone is autocorrelated, so
   expect more). A session with no headlines (GDELT gap) leaves `S` unchanged.
4. **Switch**: VIX stress (rule 3 above) opens a 5-session window; `defensive` starts on the first session in
   the window with `S_t >= h`.
5. **Exit**: 3 consecutive sessions with VIX / VIX3M < 0.90 and `S_t < h / 2`.
6. **Reported** for `dev` and, once frozen, `holdout`: this variant, the CUSUM alone (news alarm without VIX,
   to see its own timing), and the VIX-alone baseline. Per fall: sessions from the swing high and drawdown at
   the switch; the median of both; false switches per year; CUSUM alarms per year.

The `holdout` window has not been run for any variant; it is run once, after this variant is frozen on `dev`.

## Results and verdict (2026-11-22): NO

`dev` only (2020-01-02 to 2024-12-31, 1,258 sessions, 8 SPY falls of at least 8%). `holdout` was deliberately
**not run**: no variant earned it, so 2025-2026 stays untouched for a future detector.

| Variant | Detected after the high | Already defensive | Missed | Median sessions from the high | Median drawdown | False switches |
|---|---|---|---|---|---|---|
| News drift (1.5) proposes, VIX confirms | 3 | 2 | 3 | 5 | -4.2% | 2.8/year |
| VIX proposes, news drift (1.5) confirms | 3 | 2 | 3 | 5 | -4.2% | 2.6/year |
| C: VIX proposes, news CUSUM confirms | 2 | 1 | 5 | 10 | -10.2% | 0.4/year |
| CUSUM alone | 2 | 1 | 5 | 10 | -10.2% | 0.4/year |
| VIX alone (baseline) | 5 | 3 | 0 | 5 | -3.8% | 4.2/year |

Why news cannot do it (diagnostic on the daily news z-score against its one-year norm, noise sd 0.99):

- **No lead**: in the 10 sessions before the high, news is calmer than its norm in 6 of 8 falls.
- **Coincident**: the 72 h drift peaks on the day VIX confirms or later (2022-2024), never before.
- **Too small**: in the first 10 sessions of a fall the news shift is +0.1 to +0.5 sd (COVID +0.84). Detecting a
  ~0.4 sd shift in daily noise at a tolerable false-alarm rate takes ~60 sessions whatever the method.
- The news gate only trades detection for fewer false switches; it never makes the switch earlier.

Only an exogenous shock (COVID, 2020-01-28) moved the news before VIX. Speed limit of the coincident signals:
about 5 sessions and -4% after the high, against the -3% goal.

Kept: the GDELT cache (`cache/gdelt/gkg_stockmarket/`, fetcher `scripts/news/gdelt_gkg_fetch.py`) feeds the
news research tool when Alpaca has no news; FinBERT scores in `cache/gdelt/finbert_us_scores.parquet`.

## Follow-ups (out of this spike's scope)

- Pre-2020 headlines from URL slugs (GDELT has no titles before ~autumn 2019). Validate on 2020-2024 first
  (slug vs title FinBERT score; `dev` alerts from slugs only must match the title run), with a slug variant of
  the ticker filter. If valid, 2015-2019 becomes a second untouched test window (5 falls). Tracked in `TODO.md`.

## Left to the builder

- Plotting, table layout, cache file names.
