# Bill Ackman: stop the agents from trading on noise

Date: 2026-10-03 · Follows `2026-10-02-bill-ackman-strategy-design.md` (implemented, merged).

## Problem

The first full backtest (`logs/bill_ackman/backtesting/2026-10-02_223452_backtesting`, 2025-09-29 to 2026-09-23,
50 weekly reviews, none abandoned) returned **-6.7%** against **+16.9%** for SPY, with a -28% drawdown against -9%.

An analysis of `reviews.jsonl`, the ledger and Yahoo prices found no code flaw: replaying the trader's weekly target
weights reproduces the run (-6.84% against -6.70%), and the hysteresis arithmetic is correct. The value was lost in
the agent stages. Holding each stage's picks equal-weight for the following week:

| Stage | Return |
|---|---|
| Screen top 15 (code only) | +14.4% |
| Screen top 10 | +11.4% |
| Researcher's top 5 | +8.0% |
| After the short seller (surviving names) | -1.7% |
| Trader's weights (what ran) | -6.8% |

Four defects explain it:

1. **The trader undoes the hysteresis.** A holding at one fail stays on the allowed list by design. In all 31 cases
   the trader dropped it anyway, because its prompt says "let go of a stock that no longer survives the attack". No
   forced exit ever happened. 19 positions were sold and bought back within two weeks.
2. **The short seller's verdicts are noise.** Names it failed rose 2.5% over the next four weeks; names it passed fell
   0.2%. PYPL flipped verdict 16 times in 50 reviews and ADBE 11, on annual fundamentals that barely move. Fails cite
   the week's headlines (an outage, a downgrade). It has no memory of its last verdict, and in three reviews it
   overflowed the 32k context (HTTP 400; the retries recovered).
3. **Sampling is unmanaged.** `ChatOpenAI` is built with no `temperature`, so the server default applies and no run
   records it. Smaller than D1 and D2: Qwen-family servers usually default to about 0.6, not 1.0.
4. **The research stage is narrow and anchored.** `research_top_n = 5`, "prefer a stock we already hold" in the
   prompt, and the previous ranking in the context. ADBE and ACN were in the screen's top 5 in 49 and 46 of 50 reviews
   and fell 33% and 24%.

## Goal and non-goals

Beat SPY with the three agents keeping their roles. The agents keep judging; what changes is that **code enforces the
contract the pipeline already assumed** (two consecutive fails before an exit), the short seller is given memory and a
required thesis, and the research stage is widened. The pipeline's shape, the screen and the rebalancer are unchanged.

Not doing:

- No majority-vote verdicts (triples the slowest stage, which is already 60% of the tokens).
- No change to the screen's score weights or gates. One year that was bad for software cannot justify retuning them.
  Known limit: AKAM, NTAP and IEX (+44% to +69%) ranked 9-11 on the screen and stay out of reach with 8 ideas.
- No fixed sampling `seed` (batched inference is not bit-reproducible, so it would promise determinism we cannot give).
- No removal of the news tools from the short seller. The fix is how it weighs news, not whether it sees it.

## Design

### D1: the trader cannot drop a pending-fail holding (`handoff.py`, `hysteresis.py`, `pipeline.py`, `state.py`, `prompts.py`)

- A holding at its first fail (`HysteresisOutcome.pending`) is **required**: the portfolio must include it at
  `weight >= min_weight`. The trader may shrink it, not drop it. Exit happens only through code, at
  `forced_exit_fails` (2) consecutive fails.
- `HandoffRecorder.expect_portfolio(allowed, required=outcome.pending)`; `validate_portfolio(..., required)` returns
  `{"error": "PYPL failed once and is kept until a second consecutive fail: include it with a weight of at least 0.05"}`.
  The existing correct-and-resubmit path (and the forced retry) handles it. It cannot be unsatisfiable: pending is a
  subset of holdings, holdings number at most `max_positions`, and `AckmanParams` already checks
  `max_positions x min_weight` fits.
- A surviving holding stays droppable by the trader (portfolio management, not noise).
- **Re-entry cooldown.** New `AckmanParams.reentry_cooldown_reviews = 4` (validated >= 0; 0 disables). A forced exit
  sets `cooldowns[symbol] = 4`. A symbol with a cooldown above zero at the start of a review is removed from the
  screen's candidates before the researcher sees them and from `allowed`. Each *completed* review decrements every
  cooldown and drops those that reach 0, so a forced exit at review R excludes the symbol from R+1 to R+4. An
  abandoned review changes nothing (consistent with the existing rule). A holding that is still held (its sell has not
  filled) stays a forced exit; the cooldown never un-forces it.
- `ReviewState` gains `cooldowns: dict[str, int]` (values >= 1). `STATE_VERSION` 1 -> 2 (shared with D2). An old file
  loads as an empty state with the existing warning; backtests wipe the file anyway, paper/live lose their counters once.
- `TRADER_SYSTEM`: replace "let go of a stock that no longer survives the attack" with "a holding with
  pending_fail_count 1 must stay at least at the minimum weight; you may reduce it; code sells it if it fails again".
  The trader context gains `required` (the pending symbols) so the rule is visible before the first submit.
- `reviews.jsonl` gains `required` and `cooldowns`.

### D2: a short seller with memory, a stated thesis and a tool budget (`state.py`, `handoff.py`, `prompts.py`, `agents/manager.py`)

- **Memory.** `ReviewState.last_verdicts` becomes `symbol -> {verdict, reason, concern, date}`. Each `to_judge` item
  gains `previous_verdict` (that object, or null for a name not judged last time).
- **A fail names a concern; a flip says what changed** (`validate_verdicts`). A `fail` requires `concern` in
  `debt | margin | competition | management | accounting | valuation`. A verdict that differs from `previous_verdict`
  requires a non-empty `what_changed` (length-capped like `reason`). Code cannot check truth, but it forces a stated
  thesis, and both fields are logged to `reviews.jsonl` so the next analysis can score verdicts by concern.
- **Prompt rules** added to `SHORT_SELLER_SYSTEM`: a news event alone (an outage, a downgrade, a price-target cut, a
  lawsuit headline) is not a broken thesis, and a fail needs a change in debt, margins, competition, management or
  valuation that the fact sheet or a filing supports; a fallen price with intact cash flow makes a company cheaper, not
  riskier; start from your previous verdict and change it only for something material, saying what.
- **Tool budget (framework, opt-in, generic).** `AgentManager.create(..., exempt_tools: Sequence[str] | None = None)`
  and `AgentHandle.run(..., tool_budget: int | None = None)`. A `wrap_tool_call` middleware counts calls in a run; once
  the budget is spent, every tool not in `exempt_tools` returns
  `{"error": "tool budget of {budget} calls spent; {blocked_tool} was not run. Finish now: call {exempt_tools}."}`
  (or, with no exempt tools, `"... Answer now without calling another tool."`) without running. The template has no
  tool names: `blocked_tool` is read from the refused call, `exempt_tools` and `budget` come from the caller. Both
  options default to `None`: the middleware is installed on every agent but does nothing without a budget, so
  other strategies are unchanged.
  Ackman creates the short seller with `exempt_tools=["submit_verdicts"]` and runs it with
  `tool_budget = 2 x len(to_judge)`, matching the prompt's "one or two checks per name". The budget is per run because
  D4 makes the review set vary from 1 to 13 names.

### D3: sampling temperature (`agents/manager.py`, `parameters.py`, `agent_bill_ackman.py`)

- `AgentManager.create(..., temperature: float | None = None)`. `None` passes nothing (server default, other strategies
  unchanged); a value is passed to `ChatOpenAI`. Ignored for a pre-built `BaseChatModel`, like `timeout_seconds`, and
  documented as such. The per-agent totals in `settings.json["agents"]` record the temperature used.
- `AckmanParams.agent_temperature: float | None = 0.3`, validated in `[0, 2]`, applied to all three agents. Not 0:
  greedy decoding in thinking mode risks repetition loops on a long budgeted run.

### D4: a wider researcher that is not anchored (`parameters.py`, `prompts.py`, `pipeline.py`)

- `research_top_n` 5 -> 8. `max_positions` stays 5 (the trader still builds a concentrated book from up to 8 survivors).
  The short seller's review set grows from about 10 to at most 13 names, about 30-40% more time per review.
- `RESEARCHER_SYSTEM`: remove "Prefer a stock we already hold to a marginally better new idea unless something
  changed". Stop sending `previous_ranking` in the researcher's context (`ReviewState.last_ranking` is still saved and
  logged). Holdings stay in the context with their weights.
- Value-trap and concentration guards, in the prompt only: a high fcf_yield with a large `price_return_12m` fall means
  check why it fell before ranking it high, since cheap alone is not a reason; spread ideas across different businesses
  and industries when quality is similar.

## Files touched

`strategies/bill_ackman/`: `parameters.py`, `handoff.py`, `hysteresis.py`, `state.py`, `pipeline.py`, `prompts.py`,
`agent_bill_ackman.py`. `agents/manager.py` (`temperature`, `exempt_tools`, `tool_budget`). `CLAUDE.md` (the hysteresis,
Ackman-agents and `agents/` notes). No change to `screen/`, `rebalancer.py` or `portfolio.py`.

## Testing

Hand-written fakes, no network, as elsewhere. Pure tests: `validate_portfolio` with a missing or under-weight required
name; `validate_verdicts` for a fail without a concern, an unknown concern, and a flip without `what_changed`;
cooldown set, decrement and expiry in `hysteresis`; state v2 round-trip with cooldowns and verdict reasons, and a v1
file read as empty. Framework: a fake chat model checks `temperature` reaches `ChatOpenAI` (and `None` leaves the
default); a fake tool is blocked past `tool_budget` while the exempt one still runs, and with no budget behaviour is
unchanged. Pipeline: the fake trader drops a pending holding, is told, and resubmits; `previous_verdict` reaches the
short seller's context; the researcher context has no `previous_ranking`; a cooled-down symbol is removed from the
candidates and `allowed`; an 8-name ranking plus holdings sets the budget to twice `to_judge`.

## Verification

Evidence comes from one new full backtest of the same window (`YEAR`, about 5.5 h), compared with the numbers above:
forced exits and kept pending fails in `reviews.jsonl`, verdict flips per symbol, round trips within two weeks,
turnover, tool-budget hits, and the stage-by-stage returns. Only one year of data exists, so a result is evidence
about the mechanism (fewer flips, no dropped pending fails), not proof of edge.

## Risks

- A required pending holding can be held through a real decline for one more week; that is the intended price of the
  two-fail rule.
- A low temperature can reduce the diversity of the researcher's picks; it is a parameter.
- The concern list is a fixed vocabulary; a real thesis outside it (regulatory, litigation) must be filed under the
  closest label or the verdict will be `survive`. Add a label if the log shows it is needed.
