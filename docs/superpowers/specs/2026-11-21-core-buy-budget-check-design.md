# Core buy budget check: one rule for every buy

## Why

A buy that costs more than the account can pay is refused by the broker. In a backtest the refusal comes at fill time,
a session after submission, so an LLM trader never learns it: ray_dalio's 2026-11-21 backtest lost 7 buys this way
(SHV x4, UPRO, UDOW, TMF), each $11-80 short after the price moved between sizing and fill. Every strategy solves
this again on its own, with a different formula and a different reserve:

| Where | Formula | Reserve |
|---|---|---|
| `strategies/common/rebalancer.py:293-297` | min(buying power, cash + this run's sells + earlier open orders' net credit) | 2% of portfolio value (`cash_buffer`) |
| `strategies/common/trade_desk.py:289` | min(buying power, cash + this run's unfilled sells - unfilled buys) | none |
| `strategies/earnings_drift/desk.py:170` | min(buying power, cash + this cycle's unfilled sells - unfilled buys) | none |
| `strategies/cross_momentum/agent_cross_momentum.py:743-751` | cash + unfilled sells + earlier credit (no buying power term) | 5% of that budget |
| `strategies/vwap_pullback/risk.py:94` | min(buying power, cash + pending sell proceeds) | 5% (`cash_buffer=0.95`) |
| `strategies/ray_dalio/prompts.py` `SIZING_RULE` | prose: the LLM does the arithmetic | none |
| `strategies/news_builtin/agent_news_binary.py:145` | prose: the LLM does the arithmetic | 5% |

The user wants one piece of framework code that every buy goes through, so no strategy re-solves it.

## Goals

- One function computes the money a buy may spend; one check, at the chokepoint every order already passes
  (`Strategy.submit_order`), refuses a buy above it with the largest quantity that fits.
- Every strategy keeps a 2% buffer, enforced by the framework. Not configurable: a strategy that wants more cash on
  hand holds it as a sleeve in its own allocation (e.g. SHV); that is the strategy's concern, not the framework's.
- The seven copies above use the shared function instead of their own formula.
- LLM traders read the figure instead of computing it.

## Non-goals

- No change to the brokers (`brokers/`, `backtesting/broker.py`): they model a real broker; the backtest broker's
  fill-time and submission-time refusals stay as the last net.
- No check on sells (a long-only cash account can always sell what it holds; the broker checks quantity).
- No change to a strategy's allocation parameters (bull_bear's and sector_pods' `cash_buffer` also set the parking
  weight; that stays).

## Anti-goals

- Counting a fill twice live (the Rebalancer's cash invariant, see Constraints). Worse than today's rejections.
- Refusing a buy the broker would have accepted with more than the 2% buffer to spare (a check that is too strict
  silently de-invests every strategy).
- A per-strategy knob for the buffer.

## Constraints

- **Read order (live double-count):** read the account FIRST, then the open orders (`broker.tracker.get_active_orders()`).
  A sell filling in between is then counted nowhere (a slightly smaller budget), never twice. The reverse order counts
  it twice: proceeds already in `cash` and the order still open. Same rule as `Rebalancer.rebalance`'s docstring.
  Known gap kept from today: an open BUY filling in between is counted nowhere (budget too large by its cost); the
  `buying_power` term bounds it on a cash account only.
- Count only the part of each open order still to fill (`quantity - filled_quantity`): a filled part is already in
  `cash` (cf. `cross_momentum/utils.unfilled_sell_proceeds`).
- **Backtest:** `get_account().buying_power` already nets every pending order (`BacktestBroker._projection`); `cash` is
  raw. Counting all open orders makes the cash term equal the projection, so the min is consistent.
- **Margin accounts:** never `buying_power` alone (it is a multiple of equity); the cash term bounds it.
- Money is `Decimal` (root `CLAUDE.md`).
- No `datetime.now` / `time.sleep` (go through the strategy's clock if any time is needed; none should be).

## Approaches

1. **Check in `Strategy.submit_order` plus `Strategy.spendable_cash()` (picked).** One rule on the path every order
   takes; the copies call the same method; one extra account read and quote per buy live.
2. Check in the broker template (`brokers/base.py`): covers orders built outside `Strategy`, but puts a strategy policy
   (the buffer) into the layer that models the real broker, and changes backtest fidelity. Dropped.
3. A shared pure function only, no chokepoint: each strategy could still skip it. Ruled out by the user.

## Design

### `Strategy.spendable_cash() -> Decimal` (`core/strategy.py`)

```
spendable = min(buying_power, cash + unfilled open sells - unfilled open buys) - 2% x portfolio_value
```

- One `broker.get_account()` read (cash, buying power, portfolio value), THEN the active orders.
- An open order's value = unfilled quantity x its limit price if it has one, else the last price
  (`get_last_price`). A sell with no price estimate brings no credit; a buy with none costs nothing (and the broker
  remains the net). A stop or trailing-stop SELL brings no credit (it only sells if triggered).
- The 2% is a module constant (`BUY_BUFFER = Decimal("0.02")`), not a parameter.
- May be negative (fully invested book); callers treat <= 0 as "nothing to spend".

### The check in `Strategy.submit_order`

- For a BUY with a quantity: estimated cost = quantity x (limit price, else last price). If no price: submit
  unchecked. If cost > `spendable_cash()`: raise `InsufficientFundsError` (new, `utils/errors.py`, subclass of
  `OrderValidationError`) with `spendable` and `max_quantity` attributes and a message such as
  `buy 24 SHV needs ~$2,449.13, spendable is $2,210.40 (cash after open orders, 2% of the book kept back): at most 21.88 shares`.
  `max_quantity` = `fractional_qty(spendable / price)` (0 when spendable <= 0).
- Sells and notional orders pass as today. `submit_orders` goes straight to `broker.submit_orders` today: it becomes
  `[self.submit_order(o) for o in orders]` so a batch is checked too (each buy sees the ones before it as open orders).
  `close_position` / `sell_all` are sells: unchanged.
- Fees are not estimated: the 2% buffer covers them and fill drift. **Amended 2026-11-23:** the check refuses only above `spendable + 2% x book` (the real
  money). Sizing keeps the 2% back; the check did too, so a last buy sized to exactly `spendable_cash()` was refused on any 1-cent tick of an open order
  (bull_bear paper, SHV: $0.066 short), and the buffer absorbed nothing.

### Callers

- `Rebalancer`: `available` = `strategy.spendable_cash()` read where it reads buying power today (after the sells, after
  the live fill wait). Its own `reserve` subtraction goes (the framework keeps it), its planning loop stays. On an
  `InsufficientFundsError` refusal it resyncs `available` from `exc.spendable` (as it does from Alpaca's
  `parse_insufficient_buying_power` today).
- `trade_desk._check_buy`: its cash clause uses `strategy.spendable_cash()`; its weight and minimum-size checks stay.
- `earnings_drift/desk.py:170`, `cross_momentum` (743-751), `vwap_pullback/risk.py:94`: replace the formula with
  `strategy.spendable_cash()` (passed in where the code is pure, as `risk.plan_entry` takes `cash` today).
- Agent tools: `get_account_balance` (`agents/tools/account.py`) adds `"spendable": float(strategy.spendable_cash())`.
  `SIZING_RULE` (ray_dalio) and the news_builtin sizing sentence become: never buy for more than `spendable` from
  `get_account_balance`, read again after each order.
- Docs: root `CLAUDE.md` "Key patterns" gains the rule; `strategies/CLAUDE.md`'s sizing rule and
  `backtesting/CLAUDE.md`'s line point to `Strategy.spendable_cash`.

### Tests

- `spendable_cash`: buying power binds; cash term binds (margin); unfilled part only; a filled sell not credited;
  stop sell not credited; limit price used; 2% subtracted; read order (a fake broker whose sell fills between the
  account read and the orders read: credited zero times, not twice).
- `submit_order`: a buy over the limit raises `InsufficientFundsError` with the right `max_quantity`; at the limit
  passes; a sell passes; a buy with no price passes.
- The agent tool returns the error as `{"error": ...}` (existing path) and `get_account_balance` has `spendable`.
- Strategy tests that sized to 100% of cash now expect the 2% buffer: update them, never loosen the check.

## Decided after the builder check

- **Larger buffers drop to 2% (user).** cross_momentum's `cash_buffer_pct` (0.05) and vwap_pullback's `cash_buffer`
  (0.95) are deleted, with their validation and tests; news_builtin's prompt loses its 95% sentence. Those three
  invest about 3% more, so their backtests change: expected, not a regression. cross_momentum's parking target
  (`agent_cross_momentum.py:707`, `portfolio_value x (1 - cash_buffer_pct)`) uses `1 - BUY_BUFFER` instead, and its
  buying-power resync (`:809`) the framework figure.
- **A refused buy is retried once at `max_quantity` (final review finding: the binding buy is refused on any uptick
  live).** Code callers size their last buy to exactly `spendable_cash()` at a price read earlier (before the sells
  and the 30 s fill wait, or at earnings_drift's reaction close), so live any uptick refuses it, and a whole target
  stock was dropped. One shared helper, `strategies/common/rebalancer.submit_buy`, used by the Rebalancer,
  cross_momentum and earnings_drift, resends it once at `exc.max_quantity` floored to the caller's precision
  (6 decimals, or whole shares for earnings_drift) when that is still worth the caller's minimum trade, and returns
  the quantity sent so the running budget deducts what was really bought. Refused again, or too small to retry, it is
  skipped as before: the Rebalancer resyncs `available` from `exc.spendable` and goes on with the next symbol.
- **`InsufficientFundsError` subclasses `OrderValidationError` (my call).** The sites that catch `BrokerError` around a
  submit (`cross_momentum:301`, `earnings_drift/desk.py:519`, `vwap_pullback/desk.py:460,607`) then treat it as that
  order being refused (logged, skipped), which is what it is. `news_builtin`'s 3-failures abort counts LLM failures,
  not tool errors (an order error reaches the model as `{"error": ...}`), so it is unaffected.
- **A Rebalancer strategy's `cash_buffer` must be >= `BUY_BUFFER` (my call).** In bull_bear, bill_ackman and
  sector_pods it also shrinks the target weights (they fit in `1 - cash_buffer`); below 2% the targets would always
  be 2% short of the money and the trim step would sell stocks to chase buys that cannot fit. Parameter validation
  requires `cash_buffer >= 0.02` (the default already), so no result changes.

## Left to the builder

Names of private helpers, the exact error wording, where the price lookup helper lives.

## Judging it

ray_dalio rerun on the same window: no "insufficient cash" rejections expected (the 7 shortfalls were $11-80 against a
2% buffer of ~$220). Other strategies: full test suite; a bull_bear backtest should show the same book within noise
(its buffer was already 2%); cross_momentum and vwap_pullback invest ~3% more by design.
