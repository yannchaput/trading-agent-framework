from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal as D
from pathlib import Path

from trading_agent_framework.strategies.fallout_defensive_swing.rule import Phase, SwingState
from trading_agent_framework.strategies.fallout_defensive_swing.store import StateStore
from trading_agent_framework.utils.log import ColorLogger

SAVED = SwingState(phase=Phase.HOLDING, basis=D("768.00"), sale_prices={"XLP": D("92.28")}, last_sale_session=date(2026, 11, 4), selling=True, peak=D("780.5"))


def _store(path: Path, *, fresh: bool = False) -> StateStore:
    return StateStore(path, log=ColorLogger(logging.getLogger(__name__), "test"), fresh=fresh)


def test_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path / "state.json")
    store.save(SAVED)

    assert store.load() == SAVED


def test_round_trip_of_an_empty_state(tmp_path: Path) -> None:
    store = _store(tmp_path / "state.json")
    store.save(SwingState())

    assert store.load() == SwingState()


def test_a_missing_file_is_an_empty_state(tmp_path: Path) -> None:
    assert _store(tmp_path / "state.json").load() == SwingState()


def test_an_unknown_phase_is_an_empty_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": 1, "phase": "nope", "basis": null, "sale_prices": {}, "last_sale_session": null, "selling": false}')

    assert _store(path).load() == SwingState()


def test_a_bad_amount_is_an_empty_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": 1, "phase": "holding", "basis": "abc", "sale_prices": {}, "last_sale_session": null, "selling": false}')

    assert _store(path).load() == SwingState()


def test_a_fresh_store_ignores_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _store(path).save(SAVED)

    assert _store(path, fresh=True).load() == SwingState()


def test_a_version_1_file_loads_without_a_peak(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": 1, "phase": "holding", "basis": "768", "sale_prices": {}, "last_sale_session": null, "selling": false}')

    assert _store(path).load() == SwingState(phase=Phase.HOLDING, basis=D("768"))
