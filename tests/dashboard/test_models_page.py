from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.dashboard.benchmark_fixtures import LATEST, OLDER, SCENARIOS, SWITCHED, add_switched_run, build_results_tree, write_json


def _models_script() -> None:
    # Serialized by AppTest.from_function: must be self-contained.
    from trading_agent_framework.dashboard._pages.models import page_models

    page_models()


def _app(results: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    # A real launch gets --benchmark-dir from cli.py after `--`, i.e. in sys.argv.
    monkeypatch.setattr(sys, "argv", ["app.py", "--benchmark-dir", str(results)])
    at = AppTest.from_function(_models_script, default_timeout=30)
    at.run()
    return at


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return build_results_tree(tmp_path / "results")


def test_models_page_renders_the_latest_run(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(results, monkeypatch)

    assert not at.exception
    assert [title.value for title in at.title] == ["Model benchmark"]
    assert len(at.get("plotly_chart")) == 3  # categories, quality vs speed, heatmap
    assert len(at.dataframe) == 2  # summary + drill-down repeats


def test_run_picker_lists_complete_runs_newest_first(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    picker = _app(results, monkeypatch).sidebar.selectbox[0]

    assert picker.value == LATEST
    assert len(picker.options) == 2


def test_drill_down_defaults_to_the_winners_worst_scenario(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(results, monkeypatch)

    model, scenario = at.main.selectbox[0], at.main.selectbox[1]  # at.selectbox would include the sidebar picker
    assert model.value == "qwen3627b"
    assert scenario.value == "reasoning.rsi_signal"
    assert len(at.expander) == 2  # one per repeat
    assert any("1 malformed line" in caption.value for caption in at.caption)


def test_drill_down_falls_back_to_the_first_scenario_when_the_default_model_has_no_scores(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    # The default-selected model (the run's winner) has an empty `scenarios` dict --
    # `_worst_scenario`'s `scored` list is then empty and must fall back to `scenarios[0]`
    # instead of calling `min()` on an empty sequence.
    summary_path = results / LATEST / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary = [{**entry, "scenarios": {}} if entry["key"] == "qwen3627b" else entry for entry in summary]
    write_json(summary_path, summary)

    at = _app(results, monkeypatch)

    assert not at.exception
    model, scenario = at.main.selectbox[0], at.main.selectbox[1]
    assert model.value == "qwen3627b"
    assert scenario.value == SCENARIOS[0]


def test_switching_to_an_older_run_warns_about_the_model_that_did_not_run(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(results, monkeypatch)
    at.sidebar.selectbox[0].set_value(OLDER).run()

    assert not at.exception
    assert any("Gpt-OSS-20b did not run" in warning.value for warning in at.warning)
    assert at.main.selectbox[0].value == "glm"


def test_switching_runs_recomputes_the_default_model_for_the_new_run(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # OLDER has only one *ran* model (glm), so a default-model assertion there can't tell
    # "recomputed correctly" from "there was only one valid choice". SWITCHED has two ran
    # models where gptoss -- not glm, and not LATEST's winner qwen3627b -- wins overall, so
    # landing on "gptoss" here proves the selection was freshly computed for this run.
    add_switched_run(results)
    at = _app(results, monkeypatch)
    at.sidebar.selectbox[0].set_value(SWITCHED).run()

    assert not at.exception
    assert at.main.selectbox[0].value == "gptoss"


def test_a_run_where_no_model_ran_shows_a_warning_and_the_table(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    summary_path = results / LATEST / "summary.json"
    summary = json.loads(summary_path.read_text())
    write_json(summary_path, [{**entry, "ran": False, "error": "boom"} for entry in summary])

    at = _app(results, monkeypatch)

    assert not at.exception
    assert any("No model ran" in warning.value for warning in at.warning)
    assert len(at.dataframe) == 1


def test_an_unreadable_run_shows_its_error(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (results / LATEST / "summary.json").write_text("{not json", encoding="utf-8")

    at = _app(results, monkeypatch)

    assert not at.exception
    assert any("cannot be read" in error.value for error in at.error)


def test_no_results_shows_where_it_looked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path / "missing", monkeypatch)

    assert not at.exception
    assert any("--benchmark-dir" in info.value for info in at.info)


def test_the_models_sidebar_has_no_backtesting_buttons(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [button.label for button in _app(results, monkeypatch).sidebar.button]

    assert labels == ["🔄 Refresh Data"]
