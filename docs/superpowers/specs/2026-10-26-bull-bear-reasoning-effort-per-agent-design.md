# bull_bear: reasoning effort per agent, as strategy parameters

Date: 2026-10-26 · Replaces the env-wide effort of `docs/superpowers/plans/2026-10-23-llm-reasoning-effort.md` (kept as history).

## Why

Qwen 3.8 at `LLM_REASONING_EFFORT=medium` is systematically worse than Qwen 3.6 on bull_bear: two 1Y runs at about +44%
total return against seven 3.6 runs at +61% to +91%. What we can measure points at the judge:

- The researcher's notes are not worse at `medium`: on 45 notes for the same stocks and dates, 17.6 dated or numeric
  facts per note at `medium`, 16.6 at `xhigh`, 12.0 on 3.6.
- The bull and the bear score as well as 3.6's or better at `medium` (`bull-bear-scores`, 10-25 run).
- The judge's `in` - `out` spread is lower than 3.6's, and the return gap sits entirely in the holdings the runs did not
  share, where the judge's choices decide. The judge thought 15-25% less at `medium`.

Running every agent at `xhigh` (the 10-26 run, stopped after 2 reviews) took about 65 minutes per review, about 60 hours
for a 1Y run: the researcher, about 37 calls per review, dominates. The next run tests one change: the judge at `xhigh`,
the other three agents at `medium`.

## Goals

- Each bull_bear agent has its own reasoning effort, set in the strategy's parameters and recorded in `settings.json`.
- Defaults: researcher, bull and bear `medium`; judge `xhigh`.
- `LLM_REASONING_EFFORT` is no longer read by any code.

## Non-goals

- Per-agent effort for other strategies: none sends an effort today; they keep sending none.
- Changing the thinking budgets or output caps (`2026-10-26-bull-bear-thinking-budget-design.md`).
- Any change to prompts or to the pipeline.

## Anti-goals

- No warning, error or code path for a leftover `LLM_REASONING_EFFORT`: it is simply ignored (the user's decision).

## Decisions (binding)

### Agent layer (`agents/`)

- `AgentManager.create(..., reasoning_effort: ReasoningEffort | None = None)`. Default `None`: nothing is sent, so the
  server's own default applies (Qwen 3.6 has no effort; Qwen 3.8 uses its own default).
- The effort is a per-agent, per-call setting chosen by the caller, so a pre-built model (`model` given as a
  `BaseChatModel`) gets it too (as the thinking budget).
- `LLMCredentials` loses its `reasoning_effort` field and `config.py` its `_reasoning_effort` reader. No code reads
  `LLM_REASONING_EFFORT`; a value left in an env file is ignored (no log). `ReasoningEffort` stays (the parameter's type).
- Refusal fallback, narrowed: a 400 naming `reasoning_effort` is sent once more without it, with one WARNING (as today),
  and from then on only THAT value is dropped, for every agent that sends it; an agent with another value keeps sending
  its own. (Today the first refusal drops the effort for every agent: with per-agent values, a server refusing `xhigh`
  would also lose the other agents' `medium`.) Any other error propagates.
- The middleware's place in the stack is unchanged (output cap, thinking budget, reasoning effort, compaction,
  telemetry).
- Telemetry summary: each agent's `reasoning_effort` is its value, or `None` when none was set or its value was refused.

### Strategy (`bull_bear`)

- `BullBearParams` gains four fields, typed `ReasoningEffort | None`:
  - `researcher_reasoning_effort = MEDIUM`
  - `bull_reasoning_effort = MEDIUM`
  - `bear_reasoning_effort = MEDIUM`
  - `judge_reasoning_effort = XHIGH`
  `None` means no effort is sent for that agent.
- Comments beside the four fields say why each value is set (from "Why" above): the researcher's notes are as
  fact-dense at `medium` and `xhigh` multiplies a run's time (about 60 hours); the bull and bear score as well as 3.6's at
  `medium`; the judge is where the 3.6/3.8 gap shows and it weighs every candidate at once.
- `initialize` passes each agent its own effort.
- `strategy.parameters` (so `settings.json`) records all four, as strings (`"medium"`, `"xhigh"`) or `None`.
- The forced retry runs through the same agents, so it gets the same effort.

### Files and docs

- Remove `LLM_REASONING_EFFORT` from `env/.env.example` and from the README's env table.
- Update `agents/CLAUDE.md` (the `create()` options paragraph: effort is now a `create()` argument, not an env setting;
  the narrowed refusal rule) and `strategies/bull_bear/CLAUDE.md` (the four parameters and their defaults).
- The dashboard reader keeps reading `reasoning_effort` per agent from `settings.json` (unchanged).

## Constraints

- Repo conventions: hand-written fakes, no network in tests; `agents/manager.py` remains the only module importing
  `langchain`/`langchain_openai`, and only inside method bodies; TDD; numbered tasks with a review after each.
- Existing tests that read the effort from `LLMCredentials` or the env (`tests/agents/test_agent_config.py`,
  `tests/agents/test_agent_manager_reasoning_effort.py`) change with the behaviour. One test pins that
  `LLMCredentials.from_env` ignores `LLM_REASONING_EFFORT`, including an invalid value.

## Success criteria (the next 1Y backtest, Qwen 3.8)

- `settings.json` shows researcher, bull and bear `medium`, judge `xhigh`, both in the strategy parameters and in each
  agent's telemetry.
- A review takes far less than the 65 minutes of the all-`xhigh` run (about 20-25 minutes expected).
- Returns are compared with the 3.6 runs (+61% to +91%) and the two 3.8 `medium` runs (+42%, +44.6%), not promised.

## Calls made while writing this up (not discussed with the user)

- `None` in a `BullBearParams` effort field means no effort is sent for that agent.
- A pre-built model gets the effort (it is now the caller's per-agent choice, not an env setting).
- The narrowed refusal rule (drop only the refused value).
- The user's own `env/.env.bull_bear.backtesting` and `env/.env.bull_bear.paper` are not edited: their
  `LLM_REASONING_EFFORT` line is ignored. The user can delete it.

## Left to the builder

- How the refused values are tracked, within the rule above.
- Task breakdown and order.
