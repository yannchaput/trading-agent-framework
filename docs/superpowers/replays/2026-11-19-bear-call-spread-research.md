# SPY bear call spread research (2016-2026): record and result

Spec: `docs/superpowers/specs/2026-11-18-bear-call-spread-research-design.md`; plans and the hold-through-bounces design: `docs/superpowers/plans/2026-11-19-*` and `docs/superpowers/specs/2026-11-19-*`.

**The code is not kept.** It lived on the branch `feat/bear-call-spread-research` (last commit `e36d3ff`: the pricing model, rules, replay and the `bear-call-research` command, with tests), which was removed on purpose; this note is the record. The earlier bear put spread package stays on `main` (PR #24, `c932e39`).

## Question and fixed bar

Does selling $5-wide SPY bear call spreads when the market regime (`core/regime`, no look-ahead) is bearish make money over 2016-2026, on a small account? Success, in the user's terms: the account is never out of cash, losing months are fine, the year is a gain, one or two losing years are acceptable. The bar, fixed before any run: closed P&L > 0, at most 2 losing calendar years, max drawdown <= 35%; $5,000 start.

## Method

- Sell a call 2% above spot, buy one $5 higher, nearest third Friday at least 30 days out. Exits, first match wins: 7 days to expiry, stop at 2x the credit, target at half the credit, regime no longer bearish. 10% of equity at risk per entry (width minus credit, plus costs).
- Priced with Black-Scholes from SPY and VIX (rate 0, SPY's dividend-adjusted closes), call volatility anchored on one real Alpaca OPRA snapshot (2026-11-17: SPY 732.41, VIX 25.46; at-the-money IV = 0.80 x VIX, skew on both sides of spot; the 747/752 spread models at $2.13 against a real mid of $2.19). Costs: $0.05 half spread per share per leg each way, $1 commission on opens, $0.10 regulatory. Execution at the session's close; early assignment ignored. A stress run at 0.70 x VIX sat next to the default.

## Results

- **Real regime, defaults: NO-GO, all three criteria fail.** $5,000 ends -40.9% (44.1% drawdown), 77 closed trades, win rate 36%, 9 of 11 years lost (2020 +7.7% and 2025 +4.3% the only gains). Stress: -42.1%. Where the loss is (a lead, not a tested cause): 50 regime exits -$3,935 (mean -$79), 22 targets +$2,419, 3 stops -$574, 2 time exits +$45; the regime flips off bearish on a bounce and the spread is bought back at a loss.
- **Hindsight ceiling (opens at each 10%+ fall's peak, bearish to the trough; reads the future, never a strategy): GO.** +101.9% (6.4% drawdown), 7 falls all profitable, 85% win rate, invested 284 of 2,736 sessions, about +7% a year. At 0.70 x VIX: +124.8%. The structure earns money in falls; the entry signal loses it.
- **Hold through bounces (drop the regime exit; run once, decision fixed in advance): NO-GO.** A (`never`): -52.3% (54.6% drawdown, 8 of 11 years lost); the loss moved from regime exits to stops (14, -$3,010) and time exits (12, -$2,168). Stress -47.4%. B (exit only on a bullish +1; information only): -39.6%. Trades opened inside a 10%+ fall made +$424 (23 trades), outside one lost -$3,038 (27 trades). A's account fell to about $2,387 after 2025-04-04 and could not size a contract, so its 2026 is a frozen account, not evidence. 2023-2026 (one fall, weak evidence): A -1.1, -14.9, -16.4, 0.0%; B -0.6, -23.1, -3.2, +1.9%.
- **The same signal on a SPY short or an inverse ETF (2026-11-19, scratch script, 100% of the account, 5 bps per side): all NO-GO.** Short SPY (no borrow cost) -23.5% (30.7% drawdown), SH (-1x) -17.3% (28.7%), SDS (-2x) -41.5% (52.3%), SPXU (-3x) -66.0% (73.8%); all 8 of 11 years lost, 60 entries over 436 sessions. Hindsight ceilings: +181%, +195%, +635%, +1,514%. SPY buy and hold: +324%. Leverage multiplies a bad signal's loss more than a good signal's gain.

## Conclusion

The structure (or the instrument) is not the problem. The entry signal is: 45 of 77 default-run entries opened outside any 10%+ fall and lost -$2,392, and entries that did land in a fall came 6 to 21 days after the peak. The regime exit, the structure and the instrument were each varied once with the bar fixed in advance, and none turned the real signal positive. The next question is the signal itself, with a held-out period set aside (the 2016-2026 window is already seen).
