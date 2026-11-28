# bull_bear anonymized replay (2026-11-28)

Source `2026-11-27_085328_backtesting`, 2024-09-01..2025-04-30: 35 reviews, 35 done by all arms; failed per arm {'named-1': 0, 'named-2': 0, 'anon': 0}.

## Horizon 20 sessions

| measure | named-1 | named-2 | anon | named - anon (t) |
|---|---|---|---|---|
| A judge in - top 10 | +1.66% | +1.41% | +1.37% | +0.16% (+0.55) n=35 |
| B judge in - out | +5.18% | +4.37% | +4.53% | +0.25% (+0.36) n=35 |
| C bull high - low | +8.46% | +7.13% | +9.22% | -1.43% (-0.83) n=35 |

## Horizon 5 sessions

| measure | named-1 | named-2 | anon | named - anon (t) |
|---|---|---|---|---|
| A judge in - top 10 | +1.21% | +1.04% | +1.22% | -0.10% (-0.78) n=35 |
| B judge in - out | +2.24% | +1.91% | +2.37% | -0.29% (-0.92) n=35 |
| C bull high - low | +3.45% | +3.21% | +1.73% | +1.59% (+1.96) n=35 |

Verdict agreement with named-1: named-2 93% of 570, anon 92% of 570.

Gate (A, 20 sessions): named +1.53%, d +0.16%, t 0.5532664692494935, rerun noise 0.24%; tests [True, False, False, False] -> **NO LEAK DETECTED**

## Reading and limits

Script, method and the rule fixed before any result: `scripts/replays/bull_bear_anonymized.py`. Raw results are not committed.

- Exact figures stay on the fact sheets and in the notes, so recall from the numbers is not excluded.
- A former company name in a note is not scrubbed (seen once: "360 Finance").
- One run, news off, 35 reviews: little power, a modest leak could go undetected.
- Only non-gating hint: bull high - low at 5 sessions is higher named than anon (t 1.96, one of six comparisons), and the order reverses at 20 sessions.
- This makes recall by company name an unlikely cause of the edge. It does not show the edge is real.
