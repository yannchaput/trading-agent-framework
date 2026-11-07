import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from trading_agent_framework.strategies.common.json_state import (
    InvalidState,
    JsonStateStore,
    atomic_write_json,
    is_int,
    is_weight,
    valid_strings,
    valid_weights,
)
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.utils.log import ColorLogger

log = ColorLogger(logging.getLogger("test_json_state"), "Test")


@dataclass(frozen=True)
class Box:
    n: int = 0


def _decode(raw: dict[str, Any]) -> Box:
    if not is_int(raw.get("n")):
        raise InvalidState
    return Box(n=raw["n"])


def _store(path: Path, **kwargs: Any) -> JsonStateStore[Box]:
    return JsonStateStore(path, version=2, empty=Box, decode=_decode, encode=lambda s: {"n": s.n}, log=log, **kwargs)


def test_missing_file_is_empty(tmp_path: Path) -> None:
    assert _store(tmp_path / "s.json").load() == Box()


def test_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path / "data" / "s.json")
    store.save(Box(n=5))
    assert store.path == tmp_path / "data" / "s.json"
    assert store.load() == Box(n=5)
    assert (tmp_path / "data" / "s.json").read_text() == '{"version": 2, "n": 5}'


def test_unreadable_file_warns_and_is_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "s.json"
    path.write_text("{not json")
    with caplog.at_level(logging.WARNING):
        assert _store(path).load() == Box()
    assert f"state file {path} is unreadable, starting from an empty state: " in caplog.text


def test_wrong_version_is_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "s.json"
    path.write_text('{"version": 1, "n": 5}')
    with caplog.at_level(logging.WARNING):
        assert _store(path).load() == Box()
    assert f"state file {path} is not a valid state, starting from an empty state\x1b" in caplog.messages[-1]


def test_accepts_older_versions(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text('{"version": 1, "n": 5}')
    assert _store(path, accepts={1, 2}).load() == Box(n=5)


def test_decode_error_message_suffix(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "s.json"
    path.write_text('{"version": 2}')  # InvalidState with no text: no suffix
    path2 = tmp_path / "t.json"
    path2.write_text("[1]")  # not a dict
    store = JsonStateStore(path, version=2, empty=Box, decode=lambda raw: Box(n=raw["missing"]), encode=lambda s: {}, log=log)
    with caplog.at_level(logging.WARNING):
        assert store.load() == Box()
    assert "starting from an empty state: 'missing'" in caplog.messages[-1]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert _store(path2).load() == Box()
        assert _store(path).load() == Box()
    assert len(caplog.messages) == 2
    assert all("starting from an empty state\x1b" in m for m in caplog.messages)


def test_save_unserializable_leaves_no_tmp(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = JsonStateStore(tmp_path / "s.json", version=2, empty=Box, decode=_decode, encode=lambda s: {(1, 2): "x"}, log=log)
    with caplog.at_level(logging.WARNING):
        store.save(Box())
    assert "state could not be saved to" in caplog.text
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.iterdir()) == []


def test_fresh_wipes(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    _store(path).save(Box(n=1))
    assert path.exists()
    _store(path, fresh=True)
    assert not path.exists()
    _store(path, fresh=True)  # safe when there is none


def test_wipe_failure_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "s.json"
    path.mkdir()
    (path / "x").write_text("x")  # a non-empty directory: unlink fails
    with caplog.at_level(logging.WARNING):
        _store(path).wipe()
    assert f"state file {path} could not be deleted" in caplog.text


def test_atomic_write_closes_fd_on_fdopen_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    closed: list[int] = []
    real_close = os.close

    def fail_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        raise OSError("no handles")

    def tracking_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    monkeypatch.setattr(os, "close", tracking_close)
    with caplog.at_level(logging.WARNING):
        assert atomic_write_json(tmp_path / "s.json", {"a": 1}, log=log, what="thing") is False
    assert len(closed) == 1
    assert "thing could not be saved to" in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_returns_true(tmp_path: Path) -> None:
    assert atomic_write_json(tmp_path / "s.json", {"a": 1}, log=log) is True


def test_validators() -> None:
    assert is_int(3) and not is_int(True) and not is_int(1.0)
    assert is_weight(0) and is_weight(1) and is_weight(0.5)
    assert not is_weight(float("nan")) and not is_weight(1.5) and not is_weight(True) and not is_weight("0.5")
    assert valid_strings(["a"]) and not valid_strings(["a", 1]) and not valid_strings("a")
    assert valid_weights({"A": 0.5}) and not valid_weights({"A": 2}) and not valid_weights({1: 0.5}) and not valid_weights([])


def test_run_log_callable_path(tmp_path: Path) -> None:
    path = tmp_path / "run" / "x.jsonl"
    holder: list[Path | None] = [None]
    run_log = RunLog(lambda: holder[0])
    run_log.append({"a": 1})  # no path yet: nothing
    holder[0] = path
    run_log.append({"a": 2})
    assert path.read_text() == '{"a": 2}\n'


def test_run_log_unserializable_never_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "x.jsonl"
    with caplog.at_level(logging.WARNING):
        RunLog(path, what="review log").append({("a", "b"): 1})
    assert f"review log {path} could not be written" in caplog.text
    assert not path.exists()
