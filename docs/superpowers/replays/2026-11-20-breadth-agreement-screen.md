# Breadth and agreement screen (2016-2024): record and result

> Code not kept: the research branch, its scripts and its logs/caches were dropped on 2026-11-20; this record is what remains.

Spec: `docs/superpowers/specs/2026-11-20-breadth-agreement-screen-design.md` (extends `2026-11-19-fast-regime-research-design.md`; first screen: `docs/superpowers/replays/2026-11-19-fast-regime-research.md`). Code: `core/fast_regime.py`, `strategies/common/regime_breadth_data.py` (research data: month lists, cached stock closes, sector ETFs), `uv run regime screen|fast [--universe PATH --cache-dir DIR]`. Raw outputs: `logs/regime/screen_breadth.txt`, `logs/regime/fast_breadth_dev.txt` (git-ignored).

## Question and fixed bar

Can breadth inputs (share of stocks below their 20-day average, new lows, sectors below their 20-day average) or an agreement of two or three fast inputs make a regime that switches on at a mean of **-3% or better** into a fall (`mean_on_fall >= -0.03`) with **no more false-alarm bearish days than today's** on the development window 2016-01-04 to 2024-06-28 (10 falls)? Thresholds and the selection rule are fixed in code before any run; none tuned. Held-out window (2024-07 onward) only after a development PASS and the user's explicit go-ahead.

## Method

- The first screen's 12 rows plus five new candidates: `stocks_below_20d`, `new_lows`, `sectors_below_20d`, `agree_2`, `agree_3`, scored alone as raw daily labels.
- Stock breadth uses the point-in-time universe lists (a research copy of the universe history extended back to 2014-01) and cached Yahoo adjusted closes.
- Same selection rule as the first screen: two qualifying inputs with the best mean fall win; then phase B, today's state machine with `below_sma` and `off_high` replaced by the winners.

## Screen (as printed)

```
screen 2016-01-04 to 2024-06-28 (2136 sessions)
name            kind       mean fall  false bear days  recall  changes  episodes
today           reference      -5.9%               77    0.63      118  10/10
inverted        reference      -8.1%               47    0.20       92  8/10
vix_spike       reference      -8.1%               45    0.29       78  7/10
below_sma       reference      -4.9%              237    0.73      133  10/10
off_high        reference      -4.2%               96    0.70       76  10/10
same_day        candidate      -5.6%               77    0.66      118  10/10
vix9d           candidate      -3.8%              322    0.51      282  10/10
vix_jump        candidate      -3.3%               80    0.17      160  10/10
vvix            candidate      -4.6%              318    0.37      136  10/10
credit          candidate      -7.5%              133    0.23      128  7/10
short_drawdown  candidate      -4.3%              152    0.68      124  10/10
breadth         candidate      -6.1%              251    0.17      153  8/10
stocks_below_20d candidate      -5.6%               67    0.39      116  10/10
new_lows        candidate      -3.8%              102    0.50      212  9/10
sectors_below_20d candidate      -4.6%              200    0.56      186  10/10
agree_2         candidate      -2.6%              291    0.62      244  10/10
agree_3         candidate      -4.1%              132    0.37      180  10/10
winners: vix_jump, new_lows
same-day: yes
```

The first 12 rows equal the first screen's exactly. The winners changed: `new_lows` replaces `short_drawdown`. `agree_2` reached -2.6% but with 291 false bear days, so it does not qualify.

## Phase B: development window (as printed)

Command: `uv run regime fast --replace vix_jump,new_lows --same-day` (with `--universe` and `--cache-dir`).

| | mean fall at switch-on | false-alarm bear days | recall | regime changes | median overstay |
|---|---|---|---|---|---|
| variant (inverted, vix_spike, vix_jump, new_lows; same-day) | -5.9% | 58 | 0.33 | 175 | 0 sessions |
| today | -5.9% | 77 | 0.63 | 118 | 8 sessions |

**development check: FAIL (mean fall at switch-on -5.9% is below -3.0%)**

Held-out window: not run, and not scored.

## Conclusion

NO. At these fixed thresholds, in the combination the rule picked, the variant switches on at the same mean depth as today's regime (-5.9%), with fewer false bear days (58 vs 77) but lower recall (0.33 vs 0.63) and more regime changes (175 vs 118). This does not show that other breadth or agreement signals cannot work.

## Limits

- Survivor gap in stock breadth: stocks delisted later are absent from Yahoo and from every list; the 2014-01 list has 783 names, not 1,200.
- The agreement candidates are built from inputs already scored on this window (a mild look-ahead).
- The 2014-2015 history extension was built as a copy with the same pool; the shared universe file was verified untouched.
- The held-out window (2024-07 onward) was not scored.
