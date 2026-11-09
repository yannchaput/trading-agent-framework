import json
import logging
from pathlib import Path

import pytest

from trading_agent_framework.strategies.sector_pods.store import STATE_VERSION, StateStore, WeekState
from trading_agent_framework.utils.log import ColorLogger

LOG = ColorLogger(logging.getLogger("test.sector_pods.store"), "Store")


def test_store_round_trips_week_state(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json", log=LOG)
    state = WeekState(last_completed="2026-11-03", abandoned_streak=0, previous={"date": "2026-11-03", "pods": {"tech_pod": {"scores": []}}, "risk": {}, "pm": {"weights": []}})

    store.save(state)

    assert StateStore(tmp_path / "state.json", log=LOG).load() == state
    assert json.loads((tmp_path / "state.json").read_text())["version"] == STATE_VERSION
    assert StateStore(tmp_path / "missing.json", log=LOG).load() == WeekState()


@pytest.mark.parametrize("bad", [{"abandoned_streak": -1}, {"abandoned_streak": True}, {"last_completed": 5}, {"last_completed": "soon"}, {"previous": []}])
def test_store_rejects_negative_streak(tmp_path: Path, caplog: pytest.LogCaptureFixture, bad: dict) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": STATE_VERSION, "last_completed": None, "abandoned_streak": 0, "previous": {}} | bad))

    with caplog.at_level(logging.WARNING):
        assert StateStore(path, log=LOG).load() == WeekState()

    assert any("not a valid state" in record.getMessage() for record in caplog.records)


def test_store_fresh_wipes(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    StateStore(path, log=LOG).save(WeekState(last_completed="2026-11-03"))

    store = StateStore(path, log=LOG, fresh=True)

    assert not path.exists() and store.load() == WeekState()
