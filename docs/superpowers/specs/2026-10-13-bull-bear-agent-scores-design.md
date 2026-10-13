# bull_bear: score the bull, the bear and the judge against forward returns

Date: 2026-10-13 · Branch `feat/bull-bear-agent-scores` (worktree `../trading-agent-framework-agent-scores`).

## Goal

A per-run report that scores each debating agent's calls against what the stocks did next, so a run says which
agent's signal is missing or inverted, and runs can be compared side by side (the picker run
`2026-10-10_085928_backtesting`, the in/out Judge run `2026-10-12_005438_backtesting`, and the next no-slippage run).
Read-only: it changes nothing in the strategy and writes nothing into a run directory.

## Why

On the full 1Y runs, the agents' book trailed an equal-weight top-10-by-rank basket of its own candidates by 0.22
points a week in both designs (t -0.45 picker, -0.75 Judge), and the stocks the agents chose lagged the ones they
rejected by 0.40 points a week. The returns come from the momentum ranking; the debate adds no measurable selection
value yet. The report says which of the three agents to work on.

## Scope

In: a pure scoring module and a CLI, both in a new package `strategies/bull_bear/batch/`; a `[project.scripts]` entry
`bull-bear-scores`; tests; two doc lines. Out: feeding scores back to the agents (a separate, later experiment); a
dashboard view; writing into run directories; the researcher (it writes notes, not ratings).

Slippage: `BullBearStrategy.parameters["slippage"]` is already `Decimal("0")` on `main` (c061013), the picker run's
baseline. This branch only adds a test that pins it, and the value the backtest receives.

## Design

### Files

- `strategies/bull_bear/batch/__init__.py`: package docstring (batch tools run by hand over finished runs).
- `strategies/bull_bear/batch/agent_scores.py` (**pure**: no I/O, no network):
  - `forward_returns(reviews, opens, calendar, horizons=(5, 20)) -> list[ForwardReturn]`
  - `score_agents(reviews, returns, horizons=(5, 20)) -> RunScores`
- `strategies/bull_bear/batch/scores_cli.py`: `main(argv=None, *, download=None) -> int`. Reads each run's
  `reviews.jsonl`, fetches prices through `strategies/common/yahoo_daily_bars.YahooDailyBars` (its injectable
  `download`), calls the two functions above and prints the report.
- `pyproject.toml`: `bull-bear-scores = "trading_agent_framework.strategies.bull_bear.batch.scores_cli:main"`.

The strategy package does not import the dashboard (`dashboard/reviews_reader.py` reads the same file): reading a
JSON-lines file is a few lines, and a strategy depending on the UI layer would invert the layering.

### Inputs

- A review record counts when it is not `abandoned` and has a `debate_set`. Its candidates are the `debate_set`
  symbols, the stocks every agent was shown.
- Bull: `bull[*].conviction` in {low, medium, high}. Bear: `bear[*].risk` in {low, medium, high} and
  `bear[*].concern`. A candidate an agent did not rate is left out of that agent's groups.
- Judge: `verdicts[*].verdict` in {in, out}. A picker record (no `verdicts`, has `picks`) puts its `picks` in and the
  rest of the `debate_set` out. A record with neither has no judge calls.

### Forward return

- Prices: daily opens from Yahoo (split- and dividend-adjusted), the calendar from SPY's bars in the same download.
- Entry: the open of the first SPY session strictly after the review date. A Tuesday 12:00 review's market orders
  fill at the next bar's open (`backtesting/fills.py`), so this is the price the book actually paid.
- Exit: the open `h` sessions after the entry session, `h` in {5, 20} (about one and four weeks).
- Raw return `exit / entry - 1`. A candidate with no open on either day is dropped and counted; so is every candidate of
  a review dated before the calendar's first session (the session that followed it is not in the calendar, so its open
  is unknown). An exit past the last session is immature, a count of its own: not yet knowable, not a data problem.
- Excess return: the raw return minus the equal-weight mean raw return of that review's candidates that have a
  return at the same horizon (the candidates move together; excess measures the agent's discrimination, not the
  market).

### Scores

Per agent, per horizon, per run:

| Agent | Groups | Spread (positive = the agent was right) |
|---|---|---|
| Bull | conviction low / medium / high | high - low |
| Bear | risk low / medium / high; concern breakdown | low - high |
| Judge | in / out | in - out |

- Per group, pooled over the run: `n`, mean excess return, hit rate (share with excess > 0).
- Per concern (bear): `n`, mean excess return, hit rate; no t-stat.
- Spread: computed per review (mean excess of group A minus group B in that review; a review missing either group is
  skipped and counted), then averaged across reviews. t-stat across reviews: plain standard error at 5 sessions
  (weekly reviews do not overlap, bar a day in a holiday week), Newey-West with 3 lags (Bartlett weights) at 20
  sessions (four-week windows overlap by three weeks).

### Output

Plain text on stdout, one block per agent, runs as column groups (run directory name), each with a 5d and a 20d
column. A group row shows `n`, mean excess and hit rate; the spread row shows the mean spread and its t-stat. Under
the blocks, each run has three lines: the reviews scored and the malformed lines; the candidate x horizon pairs left
out, without prices and with an exit past the end of the data; and, per agent and horizon, the reviews without a
spread because a group was missing. A t-stat is shown only when the spreads vary beyond float noise. No options
besides the run directories.

### Errors

- A run directory without `reviews.jsonl`: argparse error, exit 2.
- A malformed line in `reviews.jsonl`: skipped and counted.
- `YahooDataError` (outage, empty download): one line on stderr, exit 1.
- A symbol Yahoo has nothing for: its rows are dropped and counted.
- Download window: from 7 calendar days before the first review to 40 calendar days after the last (20 sessions plus
  slack), capped at today.

## Testing

TDD; no test touches the network (repo rule): the CLI tests inject a fake `download`.

- `forward_returns`: entry is the session after the review date; exit is `h` sessions later; excess subtracts the
  review's candidate mean; missing opens are dropped and counted; abandoned reviews are skipped.
- `score_agents`: group stats; per-review spreads with the right sign per agent; a review missing a group is skipped
  and counted; the plain and the Newey-West t-stats on hand-computed series; the picker schema; the concern
  breakdown.
- `scores_cli.main`: two temporary run directories print the bull, bear and judge blocks; a missing `reviews.jsonl`
  exits 2; a failing download exits 1.
- Slippage: `BullBearStrategy.parameters["slippage"] == 0`, and `run_backtesting` passes it on.

## Docs

- Root `CLAUDE.md`, Commands: `uv run bull-bear-scores <run_dir> [<run_dir>...]`.
- `strategies/bull_bear/CLAUDE.md`: the `batch/` package and what the report measures.
