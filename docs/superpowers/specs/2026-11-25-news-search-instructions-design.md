# search_news: instructions apart from facts, titles-only said by the source, labelled budgeted calls

## Why

bull_bear's researcher spends its 7 `search_news` calls badly in backtests before 2026-04-29, where GDELT serves the
news with titles only. Evidence from run `2026-11-24_235119` (1,155 researcher runs, analysed with the log):

- The titles-only note (PR #43) only comes with `include_content=True`, so the model learns "no text" on its second
  call in 74% of runs: the first call is a plain search, the second asks for the text.
- The model ignores the note: 30% of the runs that got it got it 2+ times; 96% of the 1,443 `include_content=True`
  calls came back with it. The note sits last in the answer, after up to 15 articles, as a sibling of the data.
- The model cannot plan: the prompt says "Your tool calls are limited" with no number. Refused runs (about 1.4%)
  spend their 7 calls narrowing windows around one event, asking for the text again and again.

The user wants the model to tell instructions from facts, to know about titles-only from the first answer, and to
know its budget and where it stands in it.

## Goals

1. **The source says titles-only.** `GdeltNewsProvider` marks its articles as titles-only. `search_news` adds the
   titles-only instruction whenever every article it returns is marked, on any call, with or without
   `include_content`. Benzinga (Alpaca) articles are never marked, so they never get it, even with an empty summary.
2. **Instructions apart from facts.** A `search_news` answer has two blocks, instructions first:

   ```json
   {
     "instructions": {
       "tool_call": "search_news call 3 of 7",
       "source": "Titles only: no article text exists for these dates, and include_content cannot add any. Do not search again for the text of these articles: use the titles."
     },
     "facts": {
       "count": 4,
       "articles": [{"id": "...", "headline": "...", "source": "...", "created_at": "...", "symbols": ["VKTX"]}]
     }
   }
   ```

   `instructions` holds only what is addressed to the model; `facts` holds only what it may use. The block is left
   out when it has nothing in it. GDELT articles lose their always-empty `summary` and the marker never reaches
   `facts`.
3. **Opt-in call label from the budget middleware.** A frozen dataclass packs the budget settings given at agent
   creation:

   ```python
   @dataclass(frozen=True)
   class ToolBudget:
       exempt: tuple[str, ...] = ()  # tools the budget never counts or refuses
       label_calls: bool = False  # each counted answer says "<tool> call k of N"
   ```

   `AgentManager.create(..., budget=ToolBudget(...))` replaces `exempt_tools`. With `label_calls=True`, each counted
   call that runs gets `instructions.tool_call = "<tool name> call k of N"` in its answer, N being the run's
   `tool_budget`. Only bull_bear's researcher turns it on.
4. **The budget in the prompt.** bull_bear's researcher prompt states it may call `search_news` at most N times
   (N = `research_tool_budget`), that each answer's `instructions` say which call it was, and that `facts` are what
   the note is written from.

## Non-goals

- No change to the limit values, to which tools are exempt, or to the refusal: a refused call still gets
  `{"error": "tool budget of N calls spent; ..."}`.
- No change to any other tool's answer shape, except the label added to counted calls of an opted-in agent.
- No label for the other budgeted agents (sector_pods, earnings_drift, bill_ackman): `label_calls` stays False.
- No change to the GDELT cache, to `include_content`'s meaning, or to Alpaca's articles.

## Anti-goals (a failure even if it works)

- The titles-only instruction on a Benzinga answer.
- Anything addressed to the model inside `facts`, or data inside `instructions`.
- Breaking the `"error" in result` checks (`_tool_succeeded` in `agents/manager.py`, `news_builtin/grounding.py`,
  `pinned_news.py`): `error` stays a top-level key of an error answer (a label may sit beside it, see below).
- Two parallel calls of one turn getting the same number, or a number depending on which thread finishes first.

## Constraints

- One model turn may hold several `search_news` calls; `ToolNode` runs them on worker threads at once (246 such turns
  in the run above). The base `ToolCallLimitMiddleware` counts in `after_model`, on one thread, before the tools run:
  the numbers are assigned there, by position among the turn's allowed counted calls, after the run's earlier calls.
- `search_news` memoizes identical searches within a run (`RunMemo`) and returns the same dict object: the label is
  added to the tool's message, never to the returned dict. A repeated identical call still gets its own number.
- A tool's dict reaches the model as `json.dumps` text in a `ToolMessage`; the label is added by parsing that JSON.
  An answer that is not a JSON object is left as it is.
- Six strategies use `search_news` (bull_bear, sector_pods, ray_dalio, bill_ackman, earnings_drift, news_builtin):
  the two-block shape and the earlier titles-only instruction change what all of them see in backtests before
  2026-04-29. No prompt refers to the answer's keys today (checked). This is accepted; judge bull_bear with 2-3 runs
  per arm on the same window.
- No wall-clock time anywhere (data-tool rule).

## Approaches considered

| Decision | Considered | Picked, and why |
|---|---|---|
| Titles-only signal | content check (PR #43: no summary/content and `include_content=True`); provider marker | **Provider marker**: known on the first call, never fires on Benzinga's empty summaries |
| Answer shape | flat, instructions first; two blocks | **Two blocks** (user): the separation is structural, not by key names |
| Call label | bull_bear's `pin_search_news` wrapper; generic budget middleware | **Generic middleware, opt-in** (user): one place counts, refuses and labels; numbering by turn order is deterministic |
| Budget settings | more `create` arguments; a dict; a dataclass | **Dataclass** (user): fewer arguments, typo-checked by pyright. The limit stays on `run()` (bill_ackman and sector_pods set it per run); `final_tools` stays its own argument (it also ends runs of agents without a budget) |

## Decided details

- Titles-only instruction text: as in the example above, under `instructions.source`.
- `pin_search_news`'s `symbols_searched` moves into `instructions` (it tells the model how its call was rewritten).
- `instructions` key order: `tool_call` first when present.
- `scripts/tests/smoke_news.py` reads `result["facts"]["articles"]`.

## Calls made after the builder check (author's calls)

- **A counted call that runs and answers an error still gets its label**:
  `{"instructions": {"tool_call": "search_news call 3 of 7"}, "error": "..."}`. The numbering stays continuous and the
  model sees the failed call counted; `"error" in result` still holds. A refusal (the call did not run) is unchanged.
- **An empty answer gets no titles-only instruction**, from either source: with no article there is no title to
  mark, and an empty Benzinga answer must not get it. No change to the `NewsProvider` interface. Empty answers are
  rare where the waste happens: 10 of 899 searches made after a note, 52 of 1,443 `include_content` searches (run
  `2026-11-24_235119`).
- No label when the run has no budget (`run_limit` None); no budget sentence when the researcher runs with news off.

## Left to the builder

- How `GdeltNewsProvider` marks an article (a key the tool removes before building `facts` is the expected form).
- Where `ToolBudget` lives (next to `AgentManager.create`, exported from `agents`) and the 7 call-site edits.
- Exact wording of the prompt sentences.
- Test names and placement; docs to update (`agents/CLAUDE.md`, `backtesting/CLAUDE.md`, bull_bear's `CLAUDE.md` if
  they describe these parts).
