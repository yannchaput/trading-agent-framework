"""Small JSON state files the strategies keep between runs: an atomic writer, a generic store and the shared validators.

`JsonStateStore` owns the file handling (missing, unreadable or invalid file is an empty state with a warning, atomic save,
wipe); a strategy's store subclasses it with its version, its empty state and the `decode`/`encode` of its dataclass.
Nothing here raises on an I/O problem: bookkeeping must not stop a run.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

from trading_agent_framework.utils.log import ColorLogger


def atomic_write_json(path: Path, payload: Any, *, log: ColorLogger, what: str = "state") -> bool:
    """Write `payload` as JSON to `path` atomically (temporary file in the same directory, then replace). True when written.

    The payload is serialized before anything touches the disk; an I/O or serialization error is logged, the temporary file removed, and False returned.
    """
    temporary: str | None = None
    try:
        text = json.dumps(payload, default=str)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A unique name per write: parallel runs sharing the directory must not share a temp file.
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        try:
            handle = os.fdopen(descriptor, "w", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise
        with handle:
            handle.write(text)
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        log.log_warning(f"{what} could not be saved to {path}: {exc}")
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        return False
    return True


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_weight(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def valid_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def valid_weights(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(k, str) and is_weight(v) for k, v in value.items())


class InvalidState(ValueError):
    """Raised by a store's `decode` when the file's content is not a valid state."""


class JsonStateStore[S]:
    """One state dataclass in one JSON file. `decode` gets the whole file dict (version included) and raises `InvalidState` (or a
    ValueError/KeyError/TypeError/AttributeError) when it is not valid; `encode` gives the fields saved beside the version."""

    def __init__(
        self,
        path: Path,
        *,
        version: int,
        empty: Callable[[], S],
        decode: Callable[[dict[str, Any]], S],
        encode: Callable[[S], dict[str, Any]],
        log: ColorLogger,
        accepts: Collection[int] | None = None,
        fresh: bool = False,
    ) -> None:
        self._path = path
        self._version = version
        self._empty = empty
        self._decode = decode
        self._encode = encode
        self._log = log
        self._accepts = {version} if accepts is None else set(accepts)
        if fresh:
            self.wipe()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> S:
        """The saved state; an empty one when the file is missing, unreadable, or not a valid state (warned about)."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._empty()
        except (OSError, ValueError) as exc:
            self._log.log_warning(f"state file {self._path} is unreadable, starting from an empty state: {exc}")
            return self._empty()
        try:
            if not isinstance(raw, dict) or raw.get("version") not in self._accepts:
                raise InvalidState
            return self._decode(raw)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            detail = str(exc)
            self._log.log_warning(f"state file {self._path} is not a valid state, starting from an empty state" + (f": {detail}" if detail else ""))
            return self._empty()

    def save(self, state: S) -> None:
        """Write the state atomically; an I/O error is logged."""
        atomic_write_json(self._path, {"version": self._version, **self._encode(state)}, log=self._log)

    def wipe(self) -> None:
        """Delete the state file (a backtest starts clean); safe when there is none."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            self._log.log_warning(f"state file {self._path} could not be deleted: {exc}")
