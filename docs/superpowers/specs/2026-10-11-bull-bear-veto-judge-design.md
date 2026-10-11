# bull_bear: the judge vetoes, code builds a fixed-size book

Date: 2026-10-11 · Amends `2026-10-09-bull-bear-strategy-design.md` (kept as history) · Branch
`feat/bull-bear-veto-judge` (fix 5, `max_volatility` for new entries only, is its first commit, 783fe1a).

## Why

Analysis of the 1Y run `logs/bull_bear/backtesting/2026-10-10_085928_backtesting` (53 reviews, weekly
fill-to-fill returns from Yahoo opens, outside the simulator):

- The judge's **entries added nothing**: non-held top-15 stocks it picked returned +1.80%/week, the ones it skipped
  +1.76%. Over the year its picks (equal weight) returned +84.0% against +85.1% for the equal-weight top 15
  (weekly difference t = 0.18); without April they trail by ~16 points.
- Its **exits worked**: held stocks it dropped returned -0.63%/week, the ones it kept +1.18% (n = 67).
- 5-10 picks capped at 20% made the book concentrated (beta 2.38 vs cross_momentum's 1.51 for the same alpha).

So the judge keeps its seat but changes job: from picking 5-10 winners to vetoing the stocks whose bear case wins.
Code fills a book of a fixed size from what survives.

## Decisions (binding)

| Topic | Decision |
|---|---|
| Judge role | Gives a verdict, `keep` or `veto`, with a reason, for **every** debated stock; at most a capped number of vetoes |
| Picker | Replaced, not kept behind a switch: `submit_picks`, `min_picks`, `max_picks`, `shortlist_size` are removed |
| Agents | All four stay (researcher, bull, bear, judge); the researcher, bull and bear are unchanged |
| Judge's view | Merit only: no `in_book` flag, no queue position. The fact sheet is unchanged (`momentum_rank`, `held`, `weight_pct` stay) |
| Book | Exactly `book_size` stocks after every completed review: surviving holdings first, then surviving new names, each group by rank |
| Defaults | `book_size` N = 10, `max_vetoes` k = 5, so the top 15 are debated, as today |
| Numbers | Only N and k are parameters; the debate size is derived (N + k); every rule is checked in `BullBearParams.__post_init__` |
| cross_momentum | No effect: every change is under `strategies/bull_bear/` and `tests/strategies/bull_bear/` |

## Parameters (`BullBearParams`)

| | Field | Default | Meaning |
|---|---|---|---|
| removed | `shortlist_size`, `min_picks`, `max_picks` | | the picker's numbers |
| new | `book_size` | 10 | stocks held after every completed review |
| new | `max_vetoes` | 5 | most vetoes per review |
| derived | `debate_size` (property) | 15 | `book_size + max_vetoes`: the top `debate_size` ranks are debated |
| kept | `retention_rank`, `min_weight`, `max_weight`, `cash_buffer`, `rebalance_band`, `min_trade_pct`, `reason_max_chars`, ... | unchanged | |

Rules (added to the existing ones, which stay unless replaced here):

- `book_size >= 1` and `max_vetoes >= 1`;
- `retention_rank > debate_size` (replaces `retention_rank > shortlist_size`);
- `book_size × min_weight <= 1 − cash_buffer <= book_size × max_weight` (replaces the two `min_picks`/`max_picks`
  rules). Defaults: 0.40 <= 0.98 <= 2.0.

## One review (`ReviewPipeline.run`)

1. **Data and ranking**: unchanged (holdings are scored without `max_volatility`, fix 5).
2. **Debate set**: `build_debate_set(ranked, holdings, debate_size=params.debate_size, retention_rank=...)` (the
   keyword `shortlist_size` is renamed): the top `debate_size` ranks plus every holding ranked within
   `retention_rank`. Forced exits are unchanged.
3. **Guard and cap**: a debate set smaller than `book_size` abandons the review at stage `data` with no order
   ("debate set smaller than book_size"). Otherwise the review's veto cap is
   `cap = min(max_vetoes, len(debate set) − book_size)`, so at least `book_size` stocks always survive.
4. **Fact sheet, researcher, bull, bear**: unchanged, over the same debate set.
5. **Judge**: `submit_verdicts` (below).
6. **Book** (`book.build_book(debate, vetoed, book_size)`, new pure module `book.py`): the survivors are the debated
   stocks not vetoed; holdings first, then new names, each in rank order; the first `book_size` of that order form
   the book, returned as a tuple of symbols in that order (the buy order). With the cap, the result always has
   exactly `book_size` symbols. More surviving holdings than `book_size` cannot happen in normal running (the book
   holds `book_size`); if it does, the best-ranked `book_size` holdings form the book.
7. **Sizing**: `capped_inverse_volatility` over the book, total `1 − cash_buffer`, bounds `[min_weight, max_weight]`;
   the parameter rules make it unable to fail.
8. **Orders**: `Rebalancer.rebalance(target, forced_exits)` unchanged. A vetoed holding is not in the target and is
   sold as "not in the target portfolio".

Example (N = 10, k = 5, holdings H1 at rank 2, H2 at rank 9, H3 at rank 22): the judge vetoes H2 and ranks 1, 5
and 7. The book is H1, H3, then ranks 3, 4, 6, 8, 10, 11, 12, 13; ranks 14 and 15 are not bought; H2 is sold.

## Judge

**Tool** (`handoff.py`): `submit_verdicts(verdicts: list[{symbol, verdict, reason}])` replaces `submit_picks`.
`validate_verdicts` replaces `validate_picks` and refuses, with `{"error": ...}`:

- a list that does not cover every debated symbol exactly once (unknown, missing or duplicate symbols; the same
  `_covering` check the bull and the bear go through);
- a `verdict` other than `keep` or `veto`;
- more vetoes than the review's cap;
- an empty reason or one longer than `reason_max_chars`.

`HandoffRecorder.expect_verdicts(symbols, cap)` replaces `expect_picks`. The result type `Verdicts` holds one
`Verdict(symbol, verdict, reason)` per debated stock, in the submitted order, and exposes the vetoed symbols.
The forced retry and the abandonment after it are unchanged (stage `judge`).

**Context**: `current_datetime` and `stocks` (fact sheet, note, bull case, bear case) unchanged; `constraints`
becomes `{"max_vetoes": cap, "reason_max_chars": ...}`; the `held` list is removed (each fact sheet says `held`).

**Prompt** (`JUDGE_SYSTEM`, `judge_task`), in substance:

- weigh, for every stock, the bull's case against the bear's and give a verdict, keep or veto, with one reason;
- code builds the portfolio from the debated stocks you do not veto; a held stock you veto is sold;
- veto only when the bear's risk is concrete and specific to the stock (an event, a filing, guidance, a lawsuit, a
  balance-sheet problem) and outweighs the bull's case;
- a strong or extended price trend is not a reason to veto by itself: it is the momentum this portfolio buys;
- you may veto none, and at most `max_vetoes` (from the constraints).

The bear's prompt is not changed here.

## State, logs

- `BullBearState.last_picks` becomes `last_book: [{symbol, reason, date}]` (the judge's keep reason), still kept for
  the log and never shown to the agents. A state file with the old `last_picks` key loads; the key is ignored.
  Backtests wipe the state, as before.
- `reviews.jsonl`: `picks` and `drops` are replaced by `verdicts` (every `{symbol, verdict, reason}`),
  `judge_verdicts` (counts, e.g. `{"keep": 13, "veto": 2}`, to spot a judge that always keeps everything),
  `veto_cap` and `book` (symbols in buy order). Every other field is unchanged; an abandonment at stage `execution`
  still records the verdicts, book and targets. The dashboard reads only shared fields and is unaffected.
- Log lines: `[judge] vetoes: SYM (reason), ...; kept N` and `[bull_bear] book: ...` replace the picks line.

## Testing (`tests/strategies/bull_bear/`)

- `test_bull_bear_book.py` (new): holdings first then rank; exactly `book_size`; ranks beyond the book are not
  bought; more surviving holdings than `book_size` keeps the best-ranked.
- `test_bull_bear_handoff.py`: `validate_verdicts` coverage (missing, unknown, duplicate), verdict enum, veto cap,
  reason length; `expect_verdicts`; `submit_verdicts` returns `status recorded`.
- `test_bull_bear_params.py`: the new rules, the removed fields, `debate_size`.
- `test_bull_bear_debate_set.py`: the `debate_size` keyword.
- `test_bull_bear_pipeline.py`: happy path with a verdict script; a vetoed holding is sold; the judge's constraints
  carry the cap (`len(debate) − book_size` when smaller than `max_vetoes`); a debate set smaller than `book_size`
  abandons before any agent; log fields and `last_book`.
- `test_bull_bear_prompts.py`, `test_bull_bear_state.py` (old `last_picks` file), and any strategy/backtest test that
  refers to picks.
- `uv run pytest`, `uv run ruff check` on the changed files, `uv run pyright`; a final `git diff --stat main` shows
  only `strategies/bull_bear/`, `tests/strategies/bull_bear/` and docs.

## Docs

`strategies/bull_bear/CLAUDE.md`: the judge's verdicts, the book rule, the new parameters, the new abandonment reason,
`last_book`. The module docstrings of `pipeline.py`, `debate_set.py`, `agent_bull_bear.py` and `handoff.py`.

## Risks and open points

- A judge that keeps everything turns the strategy into a code-only book (kept holdings first, then the top ranks); `judge_verdicts` makes that visible.
- 15+ reasons per verdict list: each over-long one rejects the submission. The bull and the bear already submit 15+
  arguments under the same 300-character cap with no forced retry in the 53 reviews of run `085928`.
- The comparison with run `085928` is a single 1Y path with a nondeterministic LLM; the book also changes size (5-10
  to 10), so a better or worse result cannot be credited to the veto alone.

## Out of scope

Volatility targeting (fix 4), the bear's prompt (fix 2), the SEC tool's IFRS gap (fix 1, its own TDD task after this
one), any change to cross_momentum or `strategies/common/`.
