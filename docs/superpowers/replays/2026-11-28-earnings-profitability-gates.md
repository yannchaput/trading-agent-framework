# Earnings and profitability entry gates: run record

Date: 2026-11-28. Spec (rules, results, limits): `docs/superpowers/specs/2026-11-28-earnings-profitability-gates-design.md`.
Commands: `uv run python scripts/bull_bear/entry_gates.py --fetch`, then without arguments. Data: `cache/sec/`,
`cache/stress_2005/` (curves in `entry_gates_equity.csv`).

## Run log (warnings removed)

```text
self-check ok

### earnings, development 2005-2015

| Book | cagr | sharpe | max_dd | 2008-09 | 2015-16 | turnover | held | blocked | missing |
|---|---|---|---|---|---|---|---|---|---|
| today | 0.5% | 0.20 | -65.8% | -65.8% | -36.6% | 5.6% | 10.0 | 26.2% | 18.8% |
| earnings | 3.7% | 0.28 | -65.9% | -65.9% | -33.4% | 5.5% | 10.0 | 28.8% | 18.8% |
Names with no signal, by year: 2005 19%, 2006 20%, 2007 20%, 2008 21%, 2009 21%, 2010 20%, 2011 21%, 2012 19%, 2013 18%, 2014 17%, 2015 15%

### earnings, holdout 2016-2026

| Book | cagr | sharpe | max_dd | 2015-16 | 2018 Q4 | 2020 | 2022 | turnover | held | blocked | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| today | 30.7% | 0.84 | -46.0% | -34.1% | -39.8% | -46.0% | -30.3% | 4.7% | 10.0 | 26.6% | 12.8% |
| earnings | 23.9% | 0.73 | -54.1% | -24.8% | -37.1% | -41.3% | -30.6% | 4.9% | 10.0 | 29.0% | 12.8% |
Names with no signal, by year: 2016 14%, 2017 14%, 2018 13%, 2019 13%, 2020 14%, 2021 13%, 2022 12%, 2023 12%, 2024 12%, 2025 13%, 2026 12%

### profitability, development 2012-2018

| Book | cagr | sharpe | max_dd | 2015-16 | 2018 Q4 | turnover | held | blocked | missing |
|---|---|---|---|---|---|---|---|---|---|
| today | 5.6% | 0.33 | -61.2% | -53.7% | -39.8% | 4.9% | 10.0 | 13.5% | 60.5% |
| profitability | 9.3% | 0.44 | -53.4% | -53.4% | -38.3% | 5.2% | 10.0 | 17.9% | 60.5% |
Names with no signal, by year: 2012 63%, 2013 63%, 2014 61%, 2015 60%, 2016 60%, 2017 60%, 2018 59%

### profitability, holdout 2019-2026

| Book | cagr | sharpe | max_dd | 2020 | 2022 | turnover | held | blocked | missing |
|---|---|---|---|---|---|---|---|---|---|
| today | 42.1% | 1.01 | -46.0% | -46.0% | -30.3% | 4.4% | 10.0 | 16.4% | 45.8% |
| profitability | 37.9% | 0.96 | -46.3% | -46.0% | -28.8% | 4.5% | 10.0 | 19.1% | 45.9% |
Names with no signal, by year: 2019 47%, 2020 44%, 2021 44%, 2022 46%, 2023 46%, 2024 46%, 2025 46%, 2026 47%

earnings: FAILED holdout
profitability: FAILED holdout
```
