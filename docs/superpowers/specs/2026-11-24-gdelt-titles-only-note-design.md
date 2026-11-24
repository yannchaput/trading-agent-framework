# search_news: say when articles have no text

## Why
Since PR #41, backtest `search_news` before 2026-04-29 is served by GDELT: titles only, `summary` always "".
The researcher reads an empty summary as "text not fetched yet", retries with `include_content: True`, then narrows
the window until the 7-call budget refuses it (5.4% of researcher runs on the 11-24 2Y run vs 0.1-0.3% before).

## Goal / non-goals
- Goal: the first result already tells the model there is no text to dig for.
- Non-goals: no change to the budget, the prompt, the tool description or `GdeltNewsProvider`.
- Anti-goal: a note on Benzinga-era results (they carry text).

## Approaches
- **A (picked).** Note in the tool result, provider-agnostic, only when needed.
- B. Tool description: static, wrong after the cutoff.
- C. Researcher prompt: static, bull_bear only.

## Design
In `agents/tools/news.py` `_search`: when `include_content=True`, at least one article is returned and none has a non-empty `summary` or
`content`, add to the result:
`"note": "Titles only: no summary or article text exists for these dates. include_content or a narrower window will not add any."`
Empty results and results with any text are unchanged. Tests in `tests/agents/tools/test_news_tools.py`
(titles-only gets the note; with a summary, with content, and empty get none).

## Judging
It changes what the agent sees: compare 2-3 bull_bear 2Y runs per arm on the same window (noise floor ~50 log points).

Ruling (final review): the note needs `include_content=True`. Without it Alpaca never returns `content`, and 1.9% of Benzinga-era results (30/1,609, run 11-15) had empty summaries, so a note there would be false.
