# bear_put_spread -- stage-1 research of SPY bear put spreads on the bearish regime

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

Design: `docs/superpowers/specs/2026-11-18-bear-put-spread-research-design.md`; plan: `docs/superpowers/plans/2026-11-18-bear-put-spread-research.md`.

## Architecture

- **Research only, not a `Strategy`.** `uv run bear-spread-research` (`research_cli.py`) replays $5-wide SPY bear put spreads opened when `regime_by_session` says bearish (-1), over 2016-2026, and prints trades, SPY falls (10%+, `core/regime_scorecard.zigzag`), a summary and a GO / NO-GO verdict. It places no order and writes only `trades.csv`, `episodes.csv`, `equity.csv` under `logs/bear_put_spread/research/<ts>/` (no `metrics.json`, so the dashboard ignores it). Stage 2 (options in `entities/` and the brokers, a paper `Strategy`) is built only if the verdict is GO; a separate options backtester comes after, not inside `BacktestBroker`.
- `pricing.py` (**pure**): Black-Scholes put (rate 0, no dividend), `VolModel` (VIX / 100 x `vol_scale` + `skew` x ln(spot / strike) below spot), `spread_mid`, `CostModel` (Tastytrade: $0.05 half spread per share per leg every open and close, $1 commission on opens, $0.10 regulatory).
- `rules.py` (**pure**, reused unchanged by stage 2): `SpreadParameters`, `third_friday`, `monthly_expiry`, `strikes`, `should_open`, `exit_reason`, `contracts_for`.
- `triggers.py` (**pure**): `trigger_states(name, spy, vix)` for `--trigger t1|t2|t3`, an alternative to the regime that gives the same -1 / 0 signal, from the closes BEFORE each session. t1: SPY below its 50-day and VIX < 25, hold until back above; t2: SPY below the 50-day and the 50-day below the 200-day, hold until they cross back; t3: new 20-session low and VIX < 25, hold until SPY closes above its 10-day. Fixed in advance, never tuned.
- `research.py` (**pure**): `simulate` (the day loop), `episode_results`, `max_drawdown`, `verdict`.

## Gotchas

- **Defaults come from the user's choices:** long put 2% below spot, short put $5 lower, 15% of equity per entry (1 contract is still taken up to 20%), start equity $1,200, exits at 7 days to expiry / 80% of width / regime no longer bearish, drawdown bar 35%.
- **Expiry is the nearest third Friday at least 30 days away.** Third Fridays are 28-35 days apart, so a fixed 30-60 day window can hold none (2026-12-17).
- **Conservative on purpose:** SPY's dividend-adjusted closes (the ex-dividend drop a put holder gains is lost), rate 0, spreads valued at mid minus exit costs, execution at the session's close. A weak result is read as NO-GO. Skew makes the SOLD put dearer, so it LOWERS the spread's cost: it is realism, not a penalty; `--vol-scale`, `--skew` and `--half-spread` stress the result.
- **The verdict:** closed-trade P&L > 0; profit on more than half of the CLOSED 10%+ SPY falls (a fall never traded counts as not a profit, the fall still under way is printed apart and left out); equity drawdown <= 35%. A trade still open at the end is excluded from the P&L, but can count toward a closed fall whose attribution window (peak to trough + 20 sessions) reaches the last session. A trade counts for the LATEST fall that began before its entry.
- **At $1,200 the account can skip entries in panics:** above a VIX of about 34 one spread costs more than 15% of equity, so the 20% exception takes it (1 contract, up to about $240); only a spread dearer than that is skipped and counted as `skipped: too small`.
- **Results (2026-11-18, 2016-2026, same harness): the regime and t1-t3 are all NO-GO.** Win rate 18-19% (regime), 9-15% (t1), 22-25% (t2), 13-17% (t3); the $1,200 account ends -56%, -56%, -44%, -51%. At $100,000 (no size limit, 15% per trade) each loses 75-97%. Only these four entry rules were tried, and the structure (the $5-wide spread, 30+ days, the exits) was never varied: a trigger or a structure that works would need its own run and the split-window check; nothing here says which of the two is the cause.
