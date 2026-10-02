# HOW TO

## Intent
Those scripts are intended to the framework integration testing. They allow to test every layers with their dependency.
Most of the scripts requires a broker account and shall be run during opening market hours.

## Smoke Strategy Paper
Manual paper-trading sanity check for the strategy framework.

NOT part of the automated test suite (the suite never touches the network).
Run by hand with real *paper* Alpaca credentials:

    uv run python scripts/tests/smoke_strategy_paper.py

Credentials come from env/.env.alpaca.integration-tests, the same git-ignored
file smoke_alpaca_orders.py uses. A script-local always-open clock replaces
Alpaca's calendar so the check also runs outside market hours. Outside regular
hours Alpaca may refuse to replace a queued order; the script then reports FAIL
with the broker's message -- rerun during regular hours before suspecting code.

The strategy (sleeptime 30S):
  1. iteration 1 -- logs cash and portfolio value, submits a 1-share SPY limit
     buy far below market so it rests
  2. iteration 2 -- modifies that order's limit price
  3. iteration 3 -- cancels it, waits for the cancel, then stops the run

PASS requires three iterations, a successful modify, no hook errors,
on_new_order and on_canceled_order to have fired, and on_strategy_end last. The run log lands in
logs/smoke/paper/<timestamp>_paper/paper.log (open it with VS Code's ANSI
Colors plugin). Refuses to run against a live account.

## Smoke Alpaca Orders
Manual paper-trading sanity check for the Alpaca broker order layer.

NOT part of the automated test suite -- the suite never touches the network
(Global Constraint 7). Run this by hand, with real (paper) Alpaca credentials,
whenever you want to verify the whole order lifecycle actually works end to
end against Alpaca's live paper API rather than a fake client:

    uv run python scripts/tests/smoke_alpaca_orders.py

Credentials come from env/.env.alpaca.integration-tests -- a dedicated,
git-ignored credentials file for this script, loaded directly by path rather
than through the strategy-oriented env/.env.{strategy}.{mode} resolver, since
it isn't any particular strategy's runtime configuration.

The script:
  1. submits a 1-share LIMIT buy order for SPY, priced far below market so it
     rests instead of filling
  2. starts the trade-update stream and waits for the broker to observe it go
     `new`
  3. calls pull_orders() and confirms the submitted order is visible there
  4. cancels the order
  5. waits for the stream to report it `canceled`
  6. stops the stream and prints a PASS/FAIL summary

Refuses to run at all against a non-paper (live) account.

## Smoke Alpaca data
Manual paper-account check of the market data layer against Alpaca's live IEX feed.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_alpaca_data.py

Credentials come from env/.env.alpaca.integration-tests, as in smoke_alpaca_orders.py. The
script is read-only (it places no orders) and works at any time of day. It fails only on things
that must always hold: SPY has a last trade, daily bars come back full-length, regular-hours
minute bars stay inside 09:30-16:00, and the indicators return values. It also prints two things
no fake can show: what Alpaca does with an unknown symbol, and how many minute bars the
include_after_hours=False filter keeps.

## Smoke quality screen
Manual run of the fundamentals quality screen (`strategies/bill_ackman/screen/`) against real SEC EDGAR and Yahoo data.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_quality_screen.py

`SEC_EDGAR_USER_AGENT` ("<app name> <contact email>") comes from env/.env.alpaca.integration-tests, as in the
other smoke scripts, or from the environment. Without it the script prints `SKIP:` and exits 0, so
`run_smoke_tests.sh` reports it as skipped. The script is read-only. The first run downloads one SEC company-facts
payload per symbol (about 4 MB each; a full 1,200-symbol universe is about 5 GB and 10 to 20 minutes) and writes
reduced copies under cache/sec/annual/; a second run the same day makes no SEC request and finishes in seconds.
It fails only on things that must always hold: every symbol is either a candidate or rejected with a reason,
an unknown ticker is `no_data`, a bank is rejected, the two Alphabet listings never both become candidates,
at least three symbols are candidates, the known-indebted names (KO, MDLZ, HLT) that are candidates have a
reported debt figure, the utility NEE is never a candidate, and the equipment lessor URI, if a candidate, shows
a plausible operating margin. If it ever fails on the debt check, a debt tag is probably missing from
`screen/annual_figures.py` (bump `annual_store.SCHEMA_VERSION` after fixing it).

## Tools

### Smoke Fundamentals

### Smoke Macro

### Smoke News
Smoke test of the news tool.

    `uv run python scripts/tests/smoke_news.py`