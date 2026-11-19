# bear_call_spread: modelled research of SPY bear call spreads on a bearish regime

## Intent and why

The user wants an options strategy that earns when the market is bearish, on a small account: Alpaca's "options" paper
account first, then their Tastytrade margin account live. The first idea, buying SPY bear put spreads when
`strategy.regime` turns bearish (`bear_put_spread`, branch `feat/bear-put-spread-research`), was NO-GO on every variant
tested over 2016-2026: 18-28% win rates and -1.4% to -2.6% of equity per trade, because it kept paying for protection that
rarely paid out. This design tests the mirror structure: **sell** a SPY bear call spread on the same bearish regime,
collect a credit, and keep it if SPY ends below the short strike. It earns from falling *and* flat markets; its risk is a
sharp rally (a bear-market bounce, March 2020) that costs the full width.

As before this is **stage 1 only**: a read-only research command that answers "does it make money over 2016-2026", with
option prices modelled from SPY and VIX. Stage 2 (options in `entities/` and the broker, a paper `Strategy` calling the
same rules module, later a separate options backtester) only follows a GO.

## What success means (the user's words)

- Failure: the account runs out of money. Losing months are acceptable.
- Success: the year ends in profit; **one or two losing calendar years out of the window are acceptable** as long as the
  whole period is positive.
- The account size can be any realistic amount; the user does not need to stay at $1,200.

## Goals

- `uv run bear-call-research [--start YYYY-MM-DD] [--end YYYY-MM-DD]` replays 2016-01-04 to the latest session and prints
  one line per trade, one per calendar year, the SPY 10%+ falls (information only), a summary and a **GO / NO-GO**
  verdict; writes `trades.csv`, `years.csv`, `episodes.csv`, `equity.csv`.
- The entry/exit/strike/size decisions live in a pure `rules.py` that stage 2's strategy will call unchanged.
- No look-ahead: decisions for session `d` use `regime_by_session`, from closes strictly before `d`.

## Non-goals

- No broker, orders, `Strategy`, options in `entities/`, or change to `BacktestBroker` or the dashboard.
- No other underlyings, no put spreads, no iron condors, no parameter optimisation.
- No historical option-quote calibration (Alpaca has quotes only since February 2024); one live snapshot anchors the
  model (below).
- No alternative triggers: the regime is the only entry. `triggers.py` (t1-t3) is not carried over.

## Anti-goals

- **A flattering result.** Uncertain modelling choices err against the strategy or are flags to stress. For a credit
  spread the dangerous error is an overstated credit, so the call volatility is anchored on real quotes, not on VIX.
- **Rules drifting** between the research and the future paper strategy.
- **Network access in tests.**
- **Searching variants until one passes.** The verdict is read on the defaults below, chosen before any run; the
  `--call-vol-ratio` stress run is reported next to it, not used to pick a winner.

## Constraints

- Python 3.14, `uv`; no new dependency.
- Read-only: downloads Yahoo SPY, `^VIX`, `^VIX3M` through `YahooRegimeData`; writes only its output directory.
- Gates: ruff, ruff format, pytest (>= 80% coverage), interrogate (Google docstrings), pyright clean on the new package.

## Approaches

- **A. A new `bear_call_spread` package that takes over the reusable parts of `bear_put_spread` (chosen).** Its own
  rules, simulation and verdict; pricing, costs, expiry choice, drawdown, fall attribution and data loading move in from
  `bear_put_spread`. **The user asked to delete `bear_put_spread`** and carry over what is needed: its NO-GO results are
  kept as a short section in the new package's `CLAUDE.md`, its spec and plan stay in `docs/superpowers/`, and its code
  stays in git history. (Correction found by the final review: the user chose "no tag, branch only" believing the put
  package was not on `main`; it already is, squashed in with PR #24, `c932e39`. So this branch merges `main` and deletes
  the package there, and its history is `main`'s.)
- **B. Generalise the put harness to any vertical spread.** One loop, but sign conventions, margin and verdicts all
  branch on the spread type, for an idea that may fail too. Rejected.
- **C. A throwaway script first.** Fastest yes/no, but untested and not reusable; a pass would mean building A anyway.
  Rejected.

## Design

### Package layout (`src/trading_agent_framework/strategies/bear_call_spread/`)

- `pricing.py` -- **pure**, moved from `bear_put_spread` and extended: `put_price` (kept, the call is priced by parity),
  `call_price(spot, strike, vol, years) = put_price(...) + spot - strike` (rate 0), `CallVolModel`, `call_spread_mid`,
  `CostModel` (unchanged). The put `VolModel` and `spread_mid` are dropped.
- `rules.py` -- **pure**: `SpreadParameters`, `third_friday` and `monthly_expiry` (moved unchanged, including the walk-forward
  fix), `strikes`, `should_open`, `exit_reason`, `contracts_for`.
- `research.py` -- **pure**: `simulate`, `year_results`, `episode_results` and `max_drawdown` (moved), `Verdict`, `verdict`.
- `research_cli.py` -- I/O, script `bear-call-research` in `pyproject.toml` (replaces `bear-spread-research`).
- `__init__.py` with `__version__` (as every package), `CLAUDE.md`; index entry in `strategies/CLAUDE.md`; the root
  `CLAUDE.md` Commands line is replaced.
- Tests move to `tests/strategies/bear_call_spread/`; tests of moved code move with it, put-only tests are deleted.

### Data and timing

Unchanged from `bear_put_spread`: `YahooRegimeData().series(start - lookback)`, `regime_cli.trim`, `regime_by_session`
with default `RegimeParameters()`, SPY dividend-adjusted closes, VIX close / 100. Every decision for session `d` is taken
from `d`'s regime reading and executed at `d`'s close; positions are marked at every close.

(Dividends: the adjusted close hides the ex-dividend drop, which helps a short call; the error is about 0.1% of SPY a
month and is accepted. Early assignment of the short call before an ex-dividend date is ignored, as the time stop closes
7 days before expiry; builder's note in code.)

### Pricing (`pricing.py`)

- Black-Scholes European call by put-call parity, rate 0, dividend 0, years = calendar days / 365.
- **Call volatility per strike:** `iv(K) = vix / 100 x call_vol_ratio - call_skew x ln(K / S)`, defaults
  `call_vol_ratio` 0.80, `call_skew` 1.0, floored at 0.01 so it stays positive. The slope runs on both sides of spot, so an
  in-the-money short call after a rally (the stop-loss case) is priced dearer than at the money, as in the market; a
  one-sided skew would understate the cost of a stop-loss exit. Anchor (2026-11-17 close, SPY 732.41, VIX 25.46, 18 Dec
  2026 expiry, 31 days, Alpaca OPRA snapshot): at-the-money call IV 0.204 (0.80 x VIX); 730 call 0.208 (model 0.207);
  747 call 0.184 and 752 call 0.178 (model 0.184 and 0.177); 747/752 spread mid **$2.19**, model **$2.13**. Pricing calls flat at
  VIX would give $1.82 and ignore a real effect, so it is not used. This is one day, during a VIX jump; `--call-vol-ratio`
  and `--call-skew` are flags, and the final report includes a stress run at `--call-vol-ratio 0.70` (a lower volatility
  means a smaller credit: against the strategy).
- **Costs:** `CostModel` unchanged (Tastytrade): half bid/ask $0.05/share per leg on every open and close, commission
  $1.00 per contract per leg on opens, regulatory $0.10 per contract per leg on opens and closes. The snapshot's leg
  spreads were $0.03-$0.38, so $0.05 per side is about right for liquid SPY strikes. Opening **receives** mid - costs,
  closing **pays** mid + costs.

### Rules (`rules.py`, defaults in a frozen `SpreadParameters`)

- **Open** when the regime reading is -1 and no spread is open; one spread position at a time; after an exit, a new
  spread may open from the next session while the regime is still -1 (a roll).
- **Strikes:** short call at `round(S x (1 + short_otm))` to a whole dollar, `short_otm` 0.02; long call `width` $5 above.
- **Expiry:** nearest third Friday at least `min_dte` 30 calendar days away (the moved `monthly_expiry`).
- **Close**, checked each session, first match wins, at that close:
  1. time stop: `exit_dte` 7 calendar days or fewer to expiry;
  2. stop loss: spread mid >= credit x (1 + `stop_loss`), `stop_loss` 1.0 (the loss reaches the credit received);
  3. profit target: spread mid <= credit x (1 - `take_profit`), `take_profit` 0.50 (half the credit kept);
  4. regime exit: the reading is no longer -1.
  Stop before target: on a session that could match both (impossible with these defaults) the loss side wins.
  At the window's last session an open spread is marked and reported as open.
- **Size by the worst case:** max loss per contract = (width - credit) x 100 + open costs + close costs.
  contracts = floor(`risk_fraction` x equity / max loss per contract), `risk_fraction` 0.10, equity at the entry
  session. 0 -> no trade, counted as "skipped: too small". The broker's margin, width x 100 per contract, is reported in
  the trade line; with `risk_fraction` <= 1 it can never exceed equity, so no separate buying-power rule.
- Starting equity **$5,000** (`--equity`): one $5-wide contract risks about $300-400, 6-8% of the account. Idle cash
  earns nothing.

### Accounting

- Entry: cash += credit x 100 x n - open costs. Mark: equity = cash - (mid x 100 x n + close costs). Exit: cash -= mid x 100
  x n + close costs. Trade P&L = (credit - exit mid) x 100 x n - all costs.

### Outputs

- **Trades:** entry and exit date, reason, SPY at entry/exit, strikes, expiry, contracts, credit, exit mid, margin held,
  costs, P&L in $ and % of entry equity.
- **Years:** for each calendar year in the window (first and last may be partial, printed as such), equity at the start
  and end from the daily equity curve (open positions marked), return, trades closed. A year with no trade has 0 return.
- **SPY falls of 10%+** (`regime_scorecard.zigzag`, attribution peak to trough + 20 sessions, moved code): printed as
  information, not part of the verdict.
- **Summary:** total return, final equity, trades, win rate, average win and loss, mean P&L per closed trade (% of entry
  equity), max drawdown, sessions invested, skipped entries.

### Verdict (GO only if all three hold, each printed PASS/FAIL)

1. Total P&L of the **closed** trades after costs > 0 (an open trade is printed apart).
2. **At most 2 losing calendar years** (return < 0 on the equity curve) in the window.
3. **Max drawdown of the equity curve <= 35%** -- the line for "the account runs out of money": a strategy that loses a
   third of the account is a failure even if it recovers.

### Errors

As before: `RegimeDataError` or an empty window -> `error: ...`, exit 1; a non-finite or non-positive pricing input
raises `ValueError` and propagates.

### Tests (no network)

- `pricing`: call by parity against a known Black-Scholes value; call -> intrinsic as time -> 0; the vol model on the
  anchor snapshot (747 and 752 IV within 0.002 of 0.184 / 0.177, spread mid within $0.10 of $2.13); spread value
  between 0 and width; costs on the right sides of open and close.
- `rules`: strikes and rounding; each exit reason and their priority; sizing by max loss and the 0-contract skip;
  expiry tests move over.
- `research`: a synthetic path where SPY drifts down (credit kept), one where it rallies (stop loss), the equity curve
  sign conventions, year results incl. a partial year and a no-trade year, the three verdict lines.
- `research_cli`: fake `download`, printed verdict, CSVs in `tmp_path`, exit 1 on `RegimeDataError`.

## Left to the builder

- Internal names, print layout, CSV columns.
- Which `SpreadParameters` / `CallVolModel` / `CostModel` fields are CLI flags beyond `--equity`, `--call-vol-ratio`,
  `--call-skew`, `--half-spread`, `--start`, `--end`, `--out`.

## Calls made in this design (not discussed with the user)

- $5,000 starting equity; 10% of equity at risk per trade by max loss.
- Short call 2% above spot, $5 wide, 30+ days, time stop at 7 days, stop loss at 2x the credit, profit target at half
  the credit, regime exit.
- Call volatility 0.80 x VIX with skew 1.0, anchored on one real snapshot; stress run at 0.70 reported.
- Losing year = negative calendar-year return on the marked equity curve; partial first/last years count.
- 35% max drawdown as the "account runs out" line.
- t1-t3 triggers dropped; SPY falls kept as information only.

## Decided with the user

- Bear call spread on the bearish regime, approach A, `bear_put_spread` deleted with its useful parts moved in.
- Account size free to change (the design's $5,000 is a proposal); failure = the account runs out of money; losing
  months fine; one or two losing calendar years acceptable if the whole period is positive.
- The put code's record is git history (`main` since PR #24, and the branch `feat/bear-put-spread-research`).

## Calls made after the builder check

- Call skew runs on both sides of spot (`- call_skew x ln(K / S)`, no floor at spot), so stop-loss exits on an
  in-the-money short call are not priced too cheaply; checked on the 730 call of the anchor snapshot.

## Bear put spread record (kept in the new `CLAUDE.md`)

Regime trigger, $5-wide, 2% out, 30+ days: win rate 18-19%, $1,200 account -56% (63% drawdown), $100k control -84%.
Triggers t1/t2/t3 and structures S1-S4 (60+ days, no target, $10 wide, at the money): all NO-GO, mean P&L per closed
trade -1.4% to -2.6% of equity; none positive.
