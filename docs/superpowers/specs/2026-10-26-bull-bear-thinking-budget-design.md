# bull_bear: a thinking budget per agent, as a safety net below the output cap

Date: 2026-10-26 · Builds on `2026-10-25-bull-bear-qwen38-robustness-design.md` (judge output cap).

## Why

A reasoning model can think until its output cap is spent. The turn then stops with `finish_reason=length`, no tool call
and nothing recorded, and `ReviewPipeline._run_stage` starts a forced retry: a FRESH run that has lost the run's memory
lookups (judge) or research (researcher). The 10-25 run (Qwen 3.8, effort `medium`, 32k window) had 2 judge calls end
this way; the next run uses effort `xhigh`, which thinks longer, in an 81,920-token window.

vLLM 0.30 (the server in use) accepts `thinking_token_budget` in a chat request: when the reasoning reaches that many
tokens, the server forces the end-of-thinking marker and the model goes on to its answer or tool call. Checked on the
server on 2026-10-26, effort `xhigh`:

- a maths prompt with a 3,000-token cap and no budget: `finish_reason=length`, 3,000 tokens of reasoning, no answer;
  with a budget of 200: reasoning cut, an answer, `finish_reason=stop`;
- a judge-shaped prompt with a `submit_verdicts` tool and a budget of 150: reasoning cut mid-analysis, then a valid
  `submit_verdicts` call (`finish_reason=tool_calls`).

The current server flags (`--reasoning-parser`) are enough; no server change is needed.

## Goals

- A turn that thinks too long still submits, instead of ending with nothing and a forced retry from scratch.
- A healthy turn never reaches the budget: the budget is a safety net, not a control on how long agents think.
- A backtest log shows how often the budget cut a turn.

## Non-goals

- Using the budget to make agents think less, faster or more predictably (the user chose safety net only).
- Per-agent reasoning effort (`LLM_REASONING_EFFORT` stays global and unchanged).
- Changing compaction.

## Anti-goals

- Code never edits what the model wrote. The server ends the thinking; the code only sends the field and logs.
- No in-run correction or retry with thinking off (tried and reverted on 10-19; see `bull_bear/CLAUDE.md`).

## Decisions (binding)

### Three layers, innermost first

1. **Thinking budget** (new): per model call, bounds the reasoning only. Reaching it ends the thinking; the turn goes on.
2. **Output cap** (existing `output_cap_middleware`, unchanged): the backstop for what the budget does not cover: an
   answer that runs on after the thinking ends (seen: the model kept reasoning in its visible answer after a cut), and a
   server that ignores or refuses the budget. It also keeps prompt plus output inside the window.
3. **Compaction** (existing, unchanged): summarizes a long researcher or judge history at 80% of the window (about 65.5k
   tokens now). It guards against long tool loops, not long thinking.

Known gap, accepted: compaction's summary call bypasses every `wrap_model_call` middleware, so it gets neither the
budget nor the cap (as today for the cap and the effort).

### Parameters (`BullBearParams`)

- `thinking_token_budget: int | None = 28672`: the researcher, bull and bear.
- `judge_thinking_token_budget: int | None = 36864`: the judge's own budget, like `judge_max_output_tokens`.
- `None` means no budget for those agents (the field is not sent). `judge_thinking_token_budget=None` means no budget
  for the judge, not the shared one (the same rule as `judge_max_output_tokens`).
- Validation, like the caps: `None` or at least 1.
- Both are recorded in `strategy.parameters` (so in `settings.json`) next to `max_output_tokens` and
  `judge_max_output_tokens`.
- The caps rise to budget + 8,192 (the user chose this over lowering the budgets): `max_output_tokens` 32768 -> 36864,
  `judge_max_output_tokens` 40960 -> 45056. Window check: prompts peaked at 17.4k in the 10-25 run (20.5k in the 10-23 run), and
  20k + 45,056 is about 65k of the 81,920 window.
- **Comments beside the four parameters explain how the values are set** (the user's request):
  - a budget is set high on purpose: it is a safety net, and a healthy turn should never reach it, so a budget below
    what turns normally use would cut good reasoning;
  - each cap is its budget + 8,192 tokens, the room left after a cut for the submission: one run per review covers every
    stock (up to about 25), the judge writes a reason (up to 300 characters) per stock plus a `what_changed` (up to 300)
    on each flip, about 15,000 characters or 4-5k tokens at most; the bull and bear write an argument (up to 300) per
    stock; the rest is margin for text the model writes after the cut;
  - the cap is the backstop for an answer that runs on and for a server that ignores the budget.
  Keep these comments in step with the values.

### Agent layer (`agents/`)

- `AgentManager.create(..., thinking_token_budget: int | None = None)`, validated like `max_output_tokens` (`None` or
  at least 1, else `ValueError`). Default `None`: nothing is sent, so every other strategy is unchanged.
- A `wrap_model_call` middleware sends `thinking_token_budget` in the request body with every agent turn. It is a vLLM
  field, not an OpenAI one: it must reach the request body (for example through `extra_body`, merged with any
  `extra_body` already set), not the OpenAI client's keyword arguments.
- Refusal fallback, like `_reasoning_effort_middleware`: a 400 whose message names `thinking_token_budget` is sent once
  more without it, with one WARNING, and the manager drops the budget for every later call of every agent. Any other
  error propagates.
- A budget at or above its cap is allowed but warned about (the user's request): when `create()` gets both
  `thinking_token_budget` and `max_output_tokens` and the budget is not below the cap, it logs one WARNING naming the
  agent and both values, e.g. `[judge] thinking budget of 45056 tokens is not below the output cap of 45056: the cap stops
  the turn first and the budget never fires`. The agent is still created.
- A pre-built model (`model` given as a `BaseChatModel`) still gets the budget: unlike the effort, the budget is a
  per-call field set by the caller, not read from the environment.
- Detection: after a call returns, when the response reports reasoning tokens and they are within
  `max(64, 1% of the budget)` tokens of the budget (or above it), log one WARNING naming the agent, the budget and the
  reasoning tokens:
  `[judge] thinking budget of 36864 tokens reached (36855 reasoning tokens): the server ended the thinking`.
  No reasoning-token count reported: no line. Why a margin: on a cut turn the server reports slightly FEWER reasoning
  tokens than the budget (measured on 2026-10-26: budget 100 reported 99, budget 500 reported 491; vLLM counts from the
  prompt's open `<think>`, the parser counts generated tokens only), so "at least the budget" would never fire.
- Middleware order: next to the output cap and the reasoning effort, inside the caller's middleware, so a forced retry
  and any model call the caller's middleware makes get the budget too.
- The telemetry summary records each agent's `thinking_token_budget` (the value, or `None` when none was set or the
  server refused it), like `reasoning_effort`.

### Strategy (`bull_bear`)

- `initialize` passes `thinking_token_budget` to the researcher, bull and bear and `judge_thinking_token_budget` to the
  judge, next to their caps.
- The forced retry needs no change: it runs through the same agents, so it gets the same budget.

## Constraints

- Repo conventions (root and nested `CLAUDE.md`): hand-written fakes, no network in tests; `agents/manager.py` remains
  the only module importing `langchain`/`langchain_openai`, and only inside method bodies; TDD; numbered tasks with a
  review after each.
- Update the docs that describe the caps and the agent stack: `strategies/bull_bear/CLAUDE.md` (the "Every agent's
  turns are capped" paragraph) and `agents/CLAUDE.md` (the `create()` options and the middleware order).

## Success criteria (the next 1Y backtest, Qwen 3.8, effort `xhigh`, 81,920 window)

- No settings or refusal warnings for `thinking_token_budget` at start-up.
- Every `thinking budget ... reached` warning is followed by a recorded submission in the same run, not by a forced
  retry.
- Runs ending at the output cap (`finish_reason=length`): rare, and each one explained by an answer that ran on, not by
  thinking.
- Budget warnings are rare. If they are frequent, the budget is cutting normal reasoning and the values are too low.

## Calls made while writing this up (not discussed with the user)

- The budget values (28,672 and 36,864) and the 8,192-token gap to the cap. The user chose raising the caps over
  lowering the budgets.
- The budget is also sent to a pre-built model (it is the caller's choice, not an environment setting).
- Detection compares reported reasoning tokens with the budget; no count reported, no line.
- The detection margin, `max(64, 1% of the budget)`, from the 2026-10-26 measurement.

## Left to the builder

- How the field reaches the request body, and the middleware's exact place in the stack within the rule above.
- Exact log wording, within the example above.
- Task breakdown and order.
