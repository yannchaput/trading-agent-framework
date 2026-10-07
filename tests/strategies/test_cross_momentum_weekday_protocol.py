"""The 5-weekday A/B protocol: one backtest per rebalance weekday, judged on the means."""

import json

import pytest

from trading_agent_framework.strategies.cross_momentum.weekday_protocol import (
    Manifest,
    RunEntry,
    RunMetrics,
    compare,
    days_to_run,
    load_manifest,
    mismatches,
    read_run_metrics,
    render,
    save_manifest,
)

DAYS = [0, 1, 2, 3, 4]


def _manifest(label="base", *, days=DAYS, commit="abc", dirty=False, **overrides):
    fields = dict(slippage=0.0, start="2016-01-01T00:00:00-05:00", end="2026-10-05T00:00:00-04:00", universe_sha256="u1")
    fields.update(overrides)
    runs = {day: RunEntry(run_dir=f"/runs/{label}/{day}", commit=commit, dirty=dirty) for day in days}
    return Manifest(label=label, runs=runs, **fields)


def _metrics(cagr=0.30, alpha=0.17, beta=0.86, max_drawdown=-0.25, sharpe=1.2):
    return RunMetrics(cagr=cagr, alpha=alpha, beta=beta, max_drawdown=max_drawdown, sharpe=sharpe)


def _same(metrics, days=DAYS):
    return {day: metrics for day in days}


def _verdict(baseline_metrics, candidate_metrics, **candidate_overrides):
    return compare(_manifest("base"), _manifest("cand", **candidate_overrides), baseline_metrics, candidate_metrics)


def test_keep_when_every_criterion_holds():
    result = _verdict(_same(_metrics()), _same(_metrics(cagr=0.31)))

    assert result.verdict == "KEEP"
    assert result.failures == []
    assert result.cagr_wins == 5


def test_an_equal_mean_cagr_is_a_reject():
    result = _verdict(_same(_metrics()), _same(_metrics()))

    assert result.verdict == "REJECT"
    assert any("CAGR" in failure for failure in result.failures)


def test_alpha_equal_passes_and_just_below_rejects():
    assert _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, alpha=0.17))).verdict == "KEEP"
    result = _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, alpha=0.17 - 1e-6)))
    assert result.verdict == "REJECT"
    assert any("alpha" in failure for failure in result.failures)


def test_beta_equal_passes_and_just_above_rejects():
    assert _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, beta=0.86))).verdict == "KEEP"
    result = _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, beta=0.86 + 1e-6)))
    assert result.verdict == "REJECT"
    assert any("beta" in failure for failure in result.failures)


def test_max_drawdown_may_be_one_point_worse_but_no_more():
    assert _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, max_drawdown=-0.26))).verdict == "KEEP"
    result = _verdict(_same(_metrics()), _same(_metrics(cagr=0.31, max_drawdown=-0.2601)))
    assert result.verdict == "REJECT"
    assert any("drawdown" in failure for failure in result.failures)


def test_the_candidate_must_win_on_at_least_three_days():
    baseline = _same(_metrics())
    two_wins = dict(zip(DAYS, [_metrics(cagr=c) for c in (0.40, 0.40, 0.29, 0.29, 0.29)], strict=True))  # mean 0.334 > 0.30
    three_wins = dict(zip(DAYS, [_metrics(cagr=c) for c in (0.40, 0.40, 0.31, 0.29, 0.29)], strict=True))

    rejected = _verdict(baseline, two_wins)
    assert rejected.verdict == "REJECT"
    assert rejected.cagr_wins == 2
    assert any("wins" in failure for failure in rejected.failures)
    assert _verdict(baseline, three_wins).verdict == "KEEP"


@pytest.mark.parametrize(
    "overrides",
    [{"start": "2017-01-01T00:00:00-05:00"}, {"end": "2026-09-23T00:00:00-04:00"}, {"slippage": 0.0005}, {"universe_sha256": "u2"}],
)
def test_sets_with_a_different_setup_cannot_be_compared(overrides):
    with pytest.raises(ValueError, match=next(iter(overrides))):
        _verdict(_same(_metrics()), _same(_metrics()), **overrides)


def test_sets_with_different_days_cannot_be_compared():
    with pytest.raises(ValueError, match="days"):
        _verdict(_same(_metrics()), _same(_metrics(), [0, 1, 2]), days=[0, 1, 2])


def test_missing_metrics_for_a_recorded_day_cannot_be_compared():
    with pytest.raises(ValueError, match="metrics"):
        _verdict(_same(_metrics()), _same(_metrics(), [0, 1, 2, 3]))


def test_mixed_commits_and_dirty_runs_are_warned_about_without_changing_the_verdict():
    candidate = _manifest("cand", dirty=True)
    candidate.runs[4] = RunEntry(run_dir="/runs/cand/4", commit="def", dirty=True)

    result = compare(_manifest("base"), candidate, _same(_metrics()), _same(_metrics(cagr=0.31)))

    assert result.verdict == "KEEP"
    assert any("cand" in warning and "commits" in warning for warning in result.warnings)
    assert any("cand" in warning and "dirty" in warning for warning in result.warnings)


def test_mismatches_names_every_differing_field():
    assert mismatches(_manifest(), _manifest()) == []
    assert mismatches(_manifest(), _manifest(slippage=0.0005, universe_sha256="u2")) == ["slippage", "universe_sha256"]


def test_days_to_run_skips_recorded_days_and_keeps_the_requested_order():
    assert days_to_run(_manifest(days=[1, 3]), [4, 0, 1, 2, 3]) == [4, 0, 2]
    assert days_to_run(_manifest(days=DAYS), DAYS) == []


def test_a_manifest_round_trips_through_json(tmp_path):
    path = tmp_path / "experiments" / "base.json"
    manifest = _manifest(days=[0, 2])

    save_manifest(path, manifest)

    assert load_manifest(path) == manifest
    assert sorted(json.loads(path.read_text())["runs"]) == ["0", "2"]


def test_read_run_metrics_reads_the_five_figures(tmp_path):
    figures = {"cagr_strategy": 0.32, "alpha": 0.17, "beta": 0.86, "max_drawdown_strategy": -0.25, "sharpe_strategy": 1.25, "other": 1}
    (tmp_path / "metrics.json").write_text(json.dumps(figures))

    assert read_run_metrics(tmp_path) == _metrics(cagr=0.32, alpha=0.17, beta=0.86, max_drawdown=-0.25, sharpe=1.25)


def test_read_run_metrics_names_a_missing_run_dir(tmp_path):
    with pytest.raises(ValueError, match=str(tmp_path / "gone")):
        read_run_metrics(tmp_path / "gone")


def test_read_run_metrics_names_a_missing_figure(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"cagr_strategy": 0.32}))

    with pytest.raises(ValueError, match="beta"):
        read_run_metrics(tmp_path)


def test_render_shows_both_labels_the_verdict_and_the_failures():
    text = render(_verdict(_same(_metrics()), _same(_metrics())))

    assert "base" in text and "cand" in text
    assert "REJECT" in text
    assert "CAGR" in text
    assert "Tue" in text
