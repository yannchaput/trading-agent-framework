# Faster bearish regime research (2016-2026): record and result

> Code not kept: the research branch, its scripts and its logs/caches were dropped on 2026-11-20; this record is what remains.

Spec: `docs/superpowers/specs/2026-11-19-fast-regime-research-design.md`. Code: `core/fast_regime.py` (pure conditions, labels, variant, screen), `core/fast_regime_data.py` (research download of `^VIX9D`, `^VVIX`, `HYG`, `IEF`, `RSP`), `uv run regime screen|fast`. Raw outputs of the runs: `logs/regime/screen.txt`, `logs/regime/fast_dev.txt` (git-ignored).

## Question and fixed bar

Today's regime turns bearish when SPY is already about -5.8% below the swing high and raises false-alarm bearish days. Can a regime built from faster inputs switch on at a mean of **-3% or better** into a fall (`mean_on_fall >= -0.03`) **with no more false-alarm bearish days than today's regime** on the same window? Thresholds were fixed before any run; none tuned. Development window 2016-01-04 to 2024-06-28 (10 falls); held-out window 2024-07-01 onward (3 falls), to be looked at once and only if the development check passed.

## Method

- Phase A: seven inputs scored alone as raw daily labels (VIX9D/VIX >= 1.0; VIX up 20% over 3 sessions; VVIX >= 110; HYG/IEF down 1.5% over 10 sessions; SPY 3% under its 20-session high; RSP/SPY down 1.0% over 10 sessions; today's regime on same-day closes), next to today's regime and its four conditions as reference rows.
- Selection rule: an input qualifies if its false bear days are at most the larger of `below_sma` and `off_high`; the two qualifying inputs with the best mean fall win. Same-day closes are used if they beat today on mean fall with no more false days.
- Phase B: today's state machine with `below_sma` and `off_high` replaced by the two winners, every condition on same-day closes if same-day won.

## Phase A: the screen (as printed)

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
winners: vix_jump, short_drawdown
same-day: yes
```

Winners: `vix_jump` and `short_drawdown`. Same-day: yes (-5.6% vs -5.9% with 77 false days each). Phase B command: `uv run regime fast --replace vix_jump,short_drawdown --same-day`.

## Phase B: development window (as printed)

| | mean fall at switch-on | false-alarm bear days | recall | regime changes | median overstay |
|---|---|---|---|---|---|
| variant (inverted, vix_spike, vix_jump, short_drawdown; same-day) | -5.2% | 69 | 0.35 | 167 | 1 session |
| today | -5.9% | 77 | 0.63 | 118 | 8 sessions |

Per-episode switch-on depth, variant: -3.6%, -3.9%, -4.8%, -4.7%, -10.9%, -3.4%, -4.4%, -7.7%, -4.0%, -4.7% (10 reference episodes).

**development check: FAIL (mean fall at switch-on -5.2% is below -3.0%)**

## Held-out window

Not run: the development check failed, so by the rule the held-out window stays unseen.

## Conclusion

NO. The fast signal did not reach -3%: the variant switches on at a mean of -5.2% into a fall on the development window (today -5.9%), a gain of 0.7 point. It did have fewer false-alarm bearish days (69 vs 77), but with recall 0.35 vs 0.63: it switched on somewhat earlier and stayed bearish for less of each fall. Regime changes rose from 118 to 167, so it flips more and covers less of each fall. Per-episode, only one fall was caught at -3.4% or better, and the March 2020 second leg was caught at -10.9%. Both winners raised more false alarms than today's regime on their own (vix_jump 80, short_drawdown 152, against today's 77); they qualified only under the looser cap of 237 (below_sma's false days), and vix_jump's recall was 0.17. The held-out period was not looked at. Nothing in this screen supports a trading test of a spread, short or inverse ETF on this signal. At these fixed thresholds, in the one combination the selection rule picked, the cheap price/volatility/credit/breadth inputs did not carry a signal fast enough. The remaining input class is news sentiment (LLM-scored Alpaca/Benzinga headlines), a new design; the put/call ratio would need a different data source (Yahoo has no history; CBOE's own files would be needed). No threshold or combination was retried, as the spec requires.
