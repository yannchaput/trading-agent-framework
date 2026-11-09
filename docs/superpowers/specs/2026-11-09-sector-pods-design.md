# Sector pods: a LangGraph shared-state strategy

Date: 2026-11-09. Status: design, awaiting approval.

Inspired by lumibot's "Citadel sector pods" example
(https://lumibot.lumiwealth.com/agents_example_citadel_sector_pods.html): five sector pods rank ETFs,
a risk manager challenges them, a portfolio manager allocates. Not affiliated with Citadel.

## 1. Intent and the why

Two goals, equally weighted:

1. **A pattern test.** Agents are orchestrated by a LangGraph `StateGraph` instead of a hard-coded Python
   pipeline (bull_bear's `pipeline.py` style). The user dislikes the hard-coded pipeline; the graph should make
   the flow readable at a glance and run independent agents in parallel. If the style works, it becomes the
   template for future strategies, so the reusable plumbing goes into the framework (`agents/`), not into the
   strategy. The pattern is judged on how it reads and how easy it is to reuse.
2. **A sector-rotation experiment.** First on sector and sub-industry ETFs (this design). If the proof of concept
   succeeds, a later project points the pods at stocks of the point-in-time universe (the monthly top-1,200
   lists), each pod ranking the stocks of its sector.

**Success bar (the user's):** on a 5-year backtest, beat SPY's total return AND have a smaller max drawdown than
SPY. Missing either one means the sector idea does not go on to the universe step.

**Scope of this project: build only.** It ends at a working strategy with a green quality gate (and a short
backtest smoke run proving the graph runs end to end is fine). The user runs and scores the 5-year backtest
themselves; no tuning toward the bar here.

## 2. Goals, non-goals, anti-goals

Goals:
- A strategy `sector_pods` that runs in `live`, `paper` and `backtesting`, rebalancing weekly.
- A graph whose construction reads like its diagram, with the five pods running in parallel.
- Shared state as the only channel between agents (see section 4).
- Reusable graph plumbing in `agents/`, documented, tested, usable by the next strategy without copying.

Non-goals (this project):
- The universe/stock version of the pods (later project; leave room for it, do not build it).
- Agents placing orders. Code trades.
- Using the market regime as a code gate or sizing input (it is context for the agents only).
- A scoring CLI like `bull-bear-scores`, dashboard pages specific to this strategy, a LangGraph checkpointer
  dependency, LangGraph Studio / `langgraph.json`.
- Leveraged or inverse ETFs.

Anti-goals (a failure even if it works):
- Agents reading another agent's output from a prompt assembled by orchestration code outside the graph. The
  context an agent sees comes from the state slots its node declares.
- Copying bull_bear's pipeline, handoff recorder or prompts. Existing shared helpers that are not agent
  orchestration (`Rebalancer`, `target_portfolio`, `JsonStateStore`, `RunLog`, `only`, the news/macro/memory
  tools, `scoring.py` return/volatility maths) are meant to be used.
- A blind or unbounded order: a bad LLM answer must never become a bad order.
- Any look-ahead in a backtest (prices, news, FRED, outcomes).

## 3. The graph

```
START -> load_previous -> market_facts -+-> tech_pod ---------+
                                        +-> financials_pod ---+
                                        +-> healthcare_pod ---+-> risk_manager -> portfolio_manager -> execute -> END
                                        +-> energy_pod -------+
                                        +-> consumer_pod -----+
```

- `load_previous` (code): loads last week's final state from the JSON state file into the `previous` slot.
- `market_facts` (code): computes the facts slot (section 6).
- five pod nodes (agents, parallel fan-out / fan-in).
- `risk_manager` (agent).
- `portfolio_manager` (agent).
- `execute` (code): validated weights -> `Rebalancer`; then the final state is saved for next week and one line is
  appended to the run log.

The builder may merge `load_previous` into `market_facts`, or move `execute` out of the graph (see section 9,
threads), but the diagram above is what the graph construction must read like: one `add_node` per box, one
`add_edge` per arrow, pods added in a loop over the sleeve table.

## 4. Shared state: the pattern

The user's decision, verbatim in spirit: **each node enriches the graph state with its findings. A node receives
the state as its parameter, takes the slot(s) relevant to it, and passes that information to its agent's
`invoke` as part of the message.** The node then returns its findings as a state update. No node reads another
agent's raw text; it reads validated, structured slots.

State slots (a `TypedDict`; names are the builder's call):

| Slot | Written by | Read by |
|---|---|---|
| `date`, `regime`, `holdings` (current weights) | `market_facts` | all agents |
| `previous` (last week's `pods`, `risk`, `pm`, its date) | `load_previous` | each pod (its own previous view), risk, PM |
| `facts` (per-ETF rows, per pod) | `market_facts` | each pod (its sleeve only), risk, PM |
| `outcome` (since last rebalance, section 6) | `market_facts` | PM (pods and risk may see their own part) |
| `pods` (one entry per pod) | each pod | risk, PM |
| `risk` | `risk_manager` | PM |
| `pm` (weights + reasons) | `portfolio_manager` | `execute` |
| `errors` (which node failed and why) | any node | `execute` (logged), run log |

- Parallel pods write to `pods` concurrently, so `pods` (and `errors`) need a merging reducer
  (`Annotated[dict, merge]` / `Annotated[list, operator.add]`).
- A node declares which slots it reads; its agent sees exactly those slots, serialized compactly (JSON, no
  indentation), through the message. `AgentHandle.run(context=...)` already appends a context block to the
  task prompt; reusing it is fine.
- A pod sees only its own sleeve's facts and its own previous view, not the other pods' (pods are
  independent).

## 5. Universe: pods and sleeves

| Pod | Sleeve |
|---|---|
| Technology & Communications | XLK, XLC, SMH, IGV, FDN |
| Financials & Real Estate | XLF, KRE, KIE, IYR |
| Healthcare | XLV, XBI, IHI, XPH, IHF |
| Energy, Utilities, Materials & Industrials | XLE, XOP, OIH, XLU, XLB, XME, XLI, ITA, IYT |
| Consumer & Housing | XLY, XLP, XHB, XRT |

27 ETFs. SHV is the parking asset (`Rebalancer`'s `parking_symbol`), never in a sleeve. SPY is the benchmark and
the relative-return reference. IYR is used instead of XLRE (XLRE starts late 2015). XLC starts June 2018.

- The sleeve table is data (one place), not text inside prompts: prompts are built from it, so the universe
  version later swaps the table for a per-pod stock lookup.
- A sleeve symbol without enough completed daily bars on the review date (fewer than needed for the 200-day
  average) is left out of that week's facts and that pod's task; a pod with no usable symbol is skipped that
  week (its slot says so).

## 6. Market facts (code, no LLM)

Per sleeve ETF, computed from completed daily bars strictly before the review day:
1-week, 1-month, 3-month and 6-month returns; the same returns relative to SPY (or at least 1-month and 3-month
relative); 20-day annualized volatility; percent distance from the 200-day simple moving average. Rounded, a few
numbers per ETF, no raw bars.

Also: `regime` (`strategy.regime`: 1 / 0 / -1, or `None`, shown with a label) as context only; current holdings
and weights.

**Outcome since the last rebalance** (from `previous` and closes since its date): the return of each ETF the PM
weighted last week, the portfolio's return vs SPY's, and per pod how its scores did (for example the mean
1-week relative return of its +scores vs its -scores, or a hit rate). This is the PM's feedback for lessons. The
exact measure is the builder's call; it must use only data before the review day.

## 7. Agents

All agents: research and judgment only; none has order or account tools. Each writes its result with one
validated submit tool (section 8). Model and server settings come from the existing `LLM_*` env (`LLMCredentials`).

**Pod (5 instances, one prompt template filled from the sleeve table).** Reads: its sleeve's facts, its previous
view, date, regime. Tools: `search_news` (Alpaca news, clock-gated) and `get_fred_series` (FRED, clock-gated), with
a small tool budget (about 2 news + 2 FRED; the submit tool exempt). Writes: for every symbol in its sleeve that
week, a conviction score from -2 to +2 and a one-line reason, plus a short sector view. Prompt asks it to justify a
large change from its previous score. Each pod's macro focus (rates and credit for Financials, oil and inflation
for Energy, ...) follows the lumibot prompts' spirit, written fresh.

**Risk manager.** Reads: all pod slots, facts, previous risk notes, regime, holdings. No research tools (it
works from the pods' evidence). Writes: objections, each naming symbol(s), a kind (crowding, macro contradiction,
drawdown risk, reversal risk, other), a severity and a reason; plus a short summary. Crowding = the same bet in
several names/pods (for example XLE and XOP both +2).

**Portfolio manager.** Reads: pods, risk, facts, outcome, previous PM decision, regime, holdings. Tools:
`search_memory` and `remember_lesson` from the existing `memory_tools(strategy.memory)`, subset with `only(...)`,
small budget, submit exempt. The prompt says: search memory for relevant lessons first; write a lesson only when
last week's outcome taught something specific, not every week. Writes: target weights (sleeve symbols only) with a
reason each, which risk objections it accepted or overrode and why, and a short summary. Memory is wiped at the
start of each backtest (existing behaviour) and kept in paper/live.

## 8. Handing results into the state (framework piece)

Each agent ends by calling its submit tool. The submit tool validates the payload against a schema (symbols in
the sleeve / universe, every expected symbol covered, scores in range, weights within the limits below, text
lengths) and, on an invalid payload, returns an `{"error": ...}` the model can fix. A valid submission is recorded
and ends the run (`final_tools`, as the framework already supports). The node returns the recorded value as its
state update.

Failure handling, per agent node:
- No valid submission: one fresh retry with the submit tool forced (`AgentHandle.run(force_tool=...)`); if that
  fails too, the node writes an "unavailable" marker into its slot and an entry into `errors`.
- A pod unavailable (or skipped, or a held ETF missing from the week's facts): risk and PM go on with the pods
  that answered (they see which one is missing). **Keep or trim only:** the PM may weight a held ETF of an
  unanswered sleeve at most at its current weight (keep or cut it), and may not weight any unheld ETF of that
  sleeve. The submit validation enforces it, so a pod failure never sells the sleeve and nothing is bought in an
  unresearched sector. A pod's score does not bind the PM otherwise (a -2 can still be held; the PM explains
  overrides).
- Risk unavailable: the PM goes on without objections.
- PM unavailable: no orders that week, holdings unchanged; the week counts as abandoned.
- In a backtest, 3 abandoned weeks in a row raise `FatalStrategyError` (the framework's way to end a run on
  purpose). Paper/live log and carry on.
- Bound every agent's run: `max_steps`, and (as parameters with defaults) `thinking_token_budget` and
  `max_output_tokens`, using the existing `agents.create` options.

**What goes into `agents/` (the reusable part).** At minimum:
- a way to turn an `AgentHandle` + "slots I read" + "slot I write" + a task + a submit schema into a LangGraph node
  function (with the forced retry and the unavailable marker above);
- the schema-validated submit-tool factory that records one submission per run;
- the reducers the state needs for parallel writes.

LangGraph imports stay deferred inside function bodies, like the rest of `agents/` (a strategy without agents
never imports LangChain). The strategy package keeps only what is specific to it: the sleeve table, the state
`TypedDict` and schemas, prompts, the facts maths, the graph construction, params, persistence.

## 9. Constraints

- **Threads.** The root rule is "strategy code runs on one thread". LangGraph runs parallel nodes on worker
  threads. Pods only call read-only research tools (`search_news`, `get_fred_series`) and the LLM; they never place
  orders or call strategy hooks. The builder must verify, and lock or restructure where needed: the news provider
  and `RunMemo` under concurrent calls, the FRED client, `LLMStatsStore` (SQLite) and telemetry middleware under
  concurrent model calls, `agent_call_context` / `current_run_id` (contextvars) on worker threads, and the
  per-handle tool-budget limiter (one handle per pod is fine). If LangGraph does not run `execute` on the calling
  (strategy) thread, or `Rebalancer` / order-event dispatch is not safe off it, `execute` runs after
  `graph.invoke` returns, on the strategy thread, and the diagram shows the graph ending at the PM. The same
  question applies to `market_facts` calling the price gate. Document the outcome in the strategy's
  `CLAUDE.md` and, if the rule changes, in the root one.
- **No look-ahead.** Facts and outcome from completed bars before the review day; news and FRED through the
  existing clock-gated tools; no `datetime.now` / `time.sleep` in strategy code.
- **Order limits enforced in code** (the submit validation rejects, `execute` re-checks): weights >= 0, only
  sleeve symbols, at most 25% per ETF, at most 40% per pod, sum at most what is investable (1 - cash buffer);
  the rest goes to SHV via `target_portfolio`. With these caps a fully invested book holds at least 4 ETFs; a
  smaller book is only possible by parking the rest in SHV (the risk-off path). The caps are params.
- **Money:** `Rebalancer` and the existing Decimal rules; no new float boundary.
- **Cadence:** `sleeptime = "1D"` with a rebalance weekday and time as params (builder's default; Tuesday 12:00
  ET like bull_bear is fine); other sessions return at once; a week whose run completed is not run again after a
  restart.
- **Data:** backtests on Yahoo daily bars with the 27 ETFs, SHV and SPY preloaded; paper/live daily bars from
  Yahoo the way other ETF/stock strategies do it.
- **Persistence:** the final state of each completed week (pods, risk, pm, date) saved as JSON through a
  `JsonStateStore` subclass at `strategy.state_file("sector_pods")`, `fresh=self.is_backtesting` (wiped at the
  start of a backtest, kept in paper/live). An abandoned week does not overwrite last completed week's state.
  One `RunLog` JSONL line per week in the run directory with the whole final state, the orders placed and
  `errors`.
- **Quality gate** (project and user rules): ruff, ruff format, pytest (no network; hand-written fakes, fake
  agents for graph tests), coverage >= 80% with new code covered, interrogate >= 80%, pyright. Google docstrings.
- Register in `utils/strategy_factory.py` (`Strategies.SECTOR_PODS`), env files per the README convention, nested
  `CLAUDE.md` for the strategy and an `agents/CLAUDE.md` entry for the graph plumbing, root `CLAUDE.md` strategy
  list updated.

## 10. Decided by the user

- LangGraph `StateGraph`, one LangChain agent per node, shared state instead of prompt-chaining.
- Pods run in parallel; flow readable at a glance.
- Nodes take the state, pick their slots, pass them in the agent's invoke message, return findings as a state
  update.
- PM hands over target weights; code executes via `Rebalancer`.
- The five pods and sleeves of section 5 (sub-industry sleeves; XLU and Industrials in the Energy pod).
- Weekly rebalance.
- Agents see last week's state, saved as JSON.
- Code computes prices/facts; agents research news and FRED.
- The market regime is shown to the agents as context.
- The PM has memory tools to learn lessons.
- Success = beat SPY with a lower max drawdown on a 5-year backtest.
- Factor reusable code into `agents/` or other framework packages.
- A missing pod's sleeve (and a held ETF without facts): keep or trim only, no new buys there.
- Scope: build only; the user runs the 5-year backtest.

## 11. Calls made by the designer (check these)

- Strategy name `sector_pods`.
- Score scale -2..+2 per ETF; risk objections typed by kind and severity.
- Caps 25% per ETF / 40% per pod as the code form of "at least 3 positions, no all-in" (instead of a
  min-positions rule).
- Risk manager has no research tools.
- Invalid submission -> one forced retry -> unavailable marker; 3 abandoned weeks in a row end a backtest.
- Persistence through the project's `JsonStateStore` / `state_file` rather than LangGraph checkpointers (no new
  dependency, same in all modes).
- `execute` inside the graph unless the thread check says otherwise.

## 12. Left to the builder

Exact slot and schema names, prompt wording, tool budgets and step limits, thinking/output defaults, the outcome
measure, the facts rounding, the default backtest window (5 years ending near today), file layout inside
`strategies/sector_pods/` and `agents/`, whether pods share one agent factory, and test design.
