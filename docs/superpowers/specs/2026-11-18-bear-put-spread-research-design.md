# bear_put_spread: modelled research of SPY bear put spreads on a bearish regime

## Intent and why

The user wants an options strategy that profits when the market turns bearish (as in October-November 2026), betting on
direction with little capital. It will run on its own account: Alpaca's "options" paper account (margin x4, shorting
enabled, options level 3, $10,000 today; the strategy is meant for about $1,200) first, then the user's Tastytrade
margin account live. Alpaca's live account stays for
stocks.

Nothing in the framework knows options yet (`AssetType` is `STOCK` only, `BacktestBroker` is a long-only cash account), and
the trigger, `strategy.regime`, has never been traded on; the user's earlier study found a regime overlay did not help
bull_bear. So the first question is whether **"bearish regime -> SPY bear put spread"** makes money at all across several
real downturns. This design is **stage 1 only**: a read-only research command that answers that over 2016-2026, with option
prices modelled from SPY and VIX (no historical option data is available that far back: Alpaca's starts in February 2024).

Roadmap (decided, not in this design): stage 2, only if stage 1 says GO, adds options to `entities/` and the Alpaca broker
and a paper `Strategy` that calls the **same rules module**; later a new options backtester, separate from `BacktestBroker`.

## Goals

- `uv run bear-spread-research [--start YYYY-MM-DD] [--end YYYY-MM-DD]` replays 2016-01-04 to the latest session by
  default, and prints one line per trade, one per bearish SPY episode, a summary, and a **GO / NO-GO** verdict.
- The entry/exit/strike/size decisions live in a pure rules module that stage 2's strategy will call unchanged.
- The regime it trades on is exactly what a live strategy would have seen: `regime_by_session`, from closes strictly
  before each session (no look-ahead).

## Non-goals

- No broker, no orders, no `Strategy`, no options in `entities/`, no change to `BacktestBroker` or the dashboard.
- No interest on idle cash, no other underlyings (QQQ, single stocks), no call spreads, no parameter optimisation.
- No historical option-quote data source.

## Anti-goals

- **A flattering result.** Every modelling choice that is uncertain errs against the strategy (costs) or is exposed as a
  flag the user can stress (volatility, skew). A weak result is read as NO-GO.
- Rules that drift between the research and the future paper strategy: the research loop calls the rules module, it never
  re-implements a decision.
- Any network access in tests.

## Constraints

- Python 3.14, `uv`; no new dependency (the normal CDF is `math.erf`).
- Read-only: downloads Yahoo SPY, `^VIX`, `^VIX3M` through `YahooRegimeData`, writes only its own output directory.
- Project gates: ruff, ruff format, pytest (>= 80% coverage), interrogate (Google docstrings), pyright.

## Approaches

- **A. Research command first, rules shared with a later paper strategy (chosen).** A pure rules module + a pure pricing
  module + a simple daily simulation loop and CLI, like `uv run regime scorecard`. Gives the 10-year yes/no for about a fifth
  of the work and builds no broker option support for an idea that may fail.
- **B. Options inside the existing backtester.** One code path for backtest/paper/live and dashboard reports, but by far the
  largest build, it breaks `BacktestBroker`'s long-only cash invariant, and all of it comes before knowing there is an edge.
  Rejected; the user will build a separate options backtester later if A says GO.
- **C. Paper only + a separate notebook.** Smallest, but the notebook's rules and the strategy's rules drift apart.

## Design

### Where the code lives

`src/trading_agent_framework/strategies/bear_put_spread/` (stage 2's strategy will join it):

- `rules.py` -- **pure**: parameters, entry/exit decisions, strike and expiry choice, sizing.
- `pricing.py` -- **pure**: Black-Scholes puts, the volatility model, the cost model.
- `research.py` -- **pure**: the daily simulation loop over aligned series and regime readings, episode scoring, verdict.
- `research_cli.py` -- I/O: download, call `research.py`, print, write CSVs. Script `bear-spread-research` in
  `pyproject.toml`.
- `CLAUDE.md` -- the strategy's own rules; add it to `strategies/CLAUDE.md`'s index.

### Data

- `YahooRegimeData().series(start - lookback)` then `regime_cli.trim(series, end)`, exactly as `run_scorecard` does (reuse
  `regime_cli.lookback_days`), and `regime_by_session(...)` with default `RegimeParameters()`.
- **SPY prices are the dividend-adjusted closes `RegimeSeries.spy` carries.** Strikes are set as a percentage of that
  close, so moneyness is right; what is lost is the ex-dividend drop during a hold (about 0.1% a month), which a put
  holder would gain: the error is against the strategy. (Builder's call to switch to `traded_close` if it is trivially
  reachable; not required.)
- VIX close / 100 is the base annualised volatility.

### Timing (no look-ahead)

For each session `d` with a regime reading: the reading for `d` uses closes strictly before `d`; every decision for `d` is
taken from that reading and executed at `d`'s close, priced with SPY and VIX closes of `d`. Positions are marked at every
session's close, at model value minus exit costs, for the equity curve. A live strategy would trade earlier in the day; executing at the close loses the
same-day move, a small and unbiased delay.

### Rules (`rules.py`, defaults in a frozen `SpreadParameters` dataclass)

- **Open** when the regime reading is -1 and no spread is open. This includes the session after a profit-target or
  time-stop exit while the regime is still -1 (a roll). One spread position at a time.
- **Strikes**: long put at `round(S x (1 - 0.02))`, rounded to a whole dollar (`long_otm` 0.02); short put a fixed
  `width` of $5 below it. A fixed dollar width keeps the cost per contract about the same at any SPY level (about
  $140-170 with entry costs at VIX 18-28), so the 2016-2026 years stay comparable; the most a contract can pay is
  $500. (The user chose a narrow spread over a percentage-based one because the account will be about $1,200: a spread
  7% wide costs $600-800 per contract at today's SPY.)
- **Expiry**: the monthly expiry (third Friday of the month) whose days to expiry are in [30, 60]; if two qualify, the
  nearer. Holiday-shifted expiries are ignored (one-day error, builder's note in code).
- **Close** (checked each session, first match wins, executed at that close):
  1. time stop: 7 calendar days or fewer to expiry (`exit_dte` 7), so the spread is never held into expiry/assignment;
  2. profit target: model value >= 80% of width (`take_profit` 0.80);
  3. regime exit: the reading is no longer -1.
  At the window's last session an open spread is marked at its model value minus exit costs and reported as open.
- **Size**: contracts = floor(`risk_fraction` x equity / (net debit x 100 + entry costs)), `risk_fraction` 0.15 (the
  user's choice), equity at the entry session: 1 contract at $1,200, more if the account grows. 0 contracts -> no trade,
  counted and printed as "skipped: too small".
- Starting equity **$1,200** (the user's intended live account size; `--equity` flag), cash earns nothing.

### Pricing (`pricing.py`)

- European Black-Scholes put, rate 0 and dividend yield 0, time = calendar days to expiry / 365. (Rate 0 makes puts
  dearer, so a bought spread costs a little more: against the strategy.)
- **Volatility per strike**: `iv(K) = vix / 100 x vol_scale + skew x max(0, ln(S / K))`, `vol_scale` 1.0, `skew` 0.8 (a put
  10% below spot gets about +8 volatility points). Both legs use it at entry and at every mark/exit.
  Note for the user: skew makes the SOLD lower put dearer, which LOWERS the spread's cost; it is realism, not a penalty.
  `--vol-scale` and `--skew` are CLI flags so the result can be stressed both ways.
- **Costs** (`CostModel`, per contract per leg, Tastytrade's schedule since it is the live venue): half bid/ask $0.05 per
  share (= $5) on every open and close, commission $1.00 on opens only, regulatory $0.10 on every open and close. The
  value of a spread is its model mid; entry pays mid + costs, exit receives mid - costs. Costs are flags too.

### Outputs

- **Trades**: entry date, exit date, exit reason, SPY at entry/exit, strikes, expiry, contracts, debit, exit value, costs,
  P&L ($ and % of entry equity).
- **Episodes**: SPY falls of 10% or more found by `regime_scorecard.zigzag(sessions, closes, 0.10)` over the window; for
  each, its peak and trough dates and the summed P&L of the trades whose entry falls between peak and trough + 20 sessions.
  A fall with no such trade (the regime never turned bearish, or every entry was skipped) scores as **not a profit**.
  `zigzag` also records a fall still under way at the window's end: it is **open** when no close after its trough is 10%
  or more above the trough close. An open fall is printed on its own line with its P&L, open trades at model value, and
  is **left out of the verdict** (it is the fall that motivated the idea, so letting it vote would flatter the result).
- **Summary**: total return, final equity, number of trades, win rate, average win/loss, max drawdown of the daily equity
  curve, sessions invested, skipped entries.
- **Verdict** (the user's criteria): GO only if all three hold, each printed with PASS/FAIL:
  1. total P&L after costs of the **closed** trades > 0 (a spread still open at the end is printed apart, outside it);
  2. a profit on more than half of the closed 10%+ SPY episodes in the window (open fall excluded, no-trade falls count
     as not a profit); with no closed episode, FAIL;
  3. max drawdown of the equity curve <= 35% (the user's choice: with 15% per trade, about three losing trades in a row;
     the whole curve's drawdown needs no definition of "false alarm").
- Files: `trades.csv`, `episodes.csv`, `equity.csv` in `--out` (default
  `logs/bear_put_spread/research/<YYYY-MM-DD_HHMMSS>/`). The dashboard's `scan_runs` only picks directories with a
  `metrics.json`, so this directory stays invisible to it; do not write a `metrics.json`.

### Errors

- `RegimeDataError` (download, gap): print `error: ...` and exit 1, like `run_scorecard`.
- No session with a reading in the window: `error: ...`, exit 1.
- A pricing input that is not finite and positive (S, K, vol, time) raises `ValueError` in `pricing.py`; the loop lets it
  propagate (a bad row is a data bug, not something to skip silently).

### Tests (no network)

- `pricing`: known Black-Scholes values; put value -> intrinsic as time -> 0; spread value between 0 and width; costs
  applied on the right sides.
- `rules`: third-Friday expiry selection at month edges and when two qualify; strike rounding; each exit reason and their
  priority; sizing floor and the 0-contract skip; roll after a profit-target exit while still bearish.
- `research`: a short synthetic SPY/VIX path with hand-set regime readings: a fall that pays, a false alarm that loses, the
  equity curve, episode attribution, and the verdict lines.
- `research_cli`: a fake `download` injected into `YahooRegimeData` (as the regime CLI tests do), the printed verdict, the
  CSVs written to `tmp_path`, exit 1 on a `RegimeDataError`.

## Left to the builder

- Internal function and dataclass names, the print layout, CSV column names.
- Whether the simulation is one function or a small class.
- How the CLI's flags map onto `SpreadParameters` / `CostModel` (all their fields may be flags, or only the ones named
  above).

## Calls made in this design (not discussed with the user)

- SPY's dividend-adjusted closes, rate 0, dividend 0 (all against the strategy).
- Execution at the session's close; marking at every close.
- Skew 0.8 and the per-leg costs above (stressable flags).
- Roll on the next session after a profit-target/time-stop exit while bearish.
- Episodes = zigzag falls of 10%+, attributed to trades entered from peak to trough + 20 sessions.
- The 35% criterion read as max drawdown of the whole equity curve.
- "Open fall" = no close after the trough 10% above it; episode attribution window trough + 20 sessions.
- Output under `logs/bear_put_spread/research/<ts>/`.

## Decided with the user (after the builder check)

- Narrow fixed $5-wide spread, 15% of equity per trade, starting equity $1,200, drawdown bar 35%.
- A fall the strategy never traded counts as not a profit; the fall still under way is printed apart, outside the verdict.
