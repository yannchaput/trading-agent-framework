"""`IbkrConnection`: owns an `ib_async.IB` on a dedicated asyncio event-loop thread.

`ib_async` is asyncio-native and runs its event handlers on its loop, while strategy code runs
on the executor thread. Every IB access therefore goes through `call()`, which runs a function
on the loop and blocks the caller for the result -- with a timeout, so a hung Gateway can never
block the strategy forever. This is the only threads/asyncio code in `brokers/ibkr/`.

Never call `call()` from inside a function passed to `call()`: it already runs on the loop
thread and would wait on itself.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from trading_agent_framework.config.env import IbkrSettings
from trading_agent_framework.utils.errors import BrokerError

logger = logging.getLogger(__name__)


def _default_ib_factory() -> Any:
    from ib_async import IB

    return IB()


class IbkrConnection:
    def __init__(
        self,
        settings: IbkrSettings,
        *,
        ib_factory: Callable[[], Any] | None = None,
        call_timeout: float = 30.0,
        connect_timeout: float = 15.0,
        reconnect_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._ib_factory = ib_factory or _default_ib_factory
        self._call_timeout = call_timeout
        self._connect_timeout = connect_timeout
        self._reconnect_attempts = reconnect_attempts
        self._sleep = sleep
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ib: Any = None
        self._has_connected = False
        self.on_reconnect: Callable[[], None] | None = None

    def start(self) -> None:
        """Start the loop thread and create the `IB` object on it."""
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="ibkr-event-loop")
        self._thread.start()
        self._ib = self._run(lambda _: self._ib_factory(), self._call_timeout)

    def connect(self) -> None:
        settings = self._settings
        try:
            self._run(
                lambda ib: ib.connectAsync(settings.host, settings.port, clientId=settings.client_id, timeout=self._connect_timeout),
                self._connect_timeout + 5,
            )
        except BrokerError as exc:
            raise BrokerError(
                f"Could not connect to IB Gateway at {settings.host}:{settings.port} (client id {settings.client_id}): {exc}"
            ) from exc
        self._has_connected = True
        logger.info("Connected to IB Gateway at %s:%s (client id %s)", settings.host, settings.port, settings.client_id)

    def call[T](self, fn: Callable[[Any], T | Awaitable[T]], *, timeout: float | None = None) -> T:
        """Run `fn(ib)` on the loop thread, awaiting it if it returns an awaitable."""
        self._ensure_connected()
        return self._run(fn, timeout if timeout is not None else self._call_timeout)

    def disconnect(self) -> None:
        loop = self._loop
        if loop is None:
            return
        if self._ib is not None:
            try:
                self._run(lambda ib: ib.disconnect() if ib.isConnected() else None, 5.0)
            except BrokerError:
                logger.exception("error disconnecting from IB Gateway")
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(5.0)
        loop.close()
        self._loop = None
        self._thread = None
        self._has_connected = False

    def _ensure_connected(self) -> None:
        if not self._has_connected:
            raise BrokerError("not connected to IB Gateway; call connect() first")
        if self._run(lambda ib: ib.isConnected(), self._call_timeout):
            return
        logger.warning("IB Gateway connection lost; reconnecting")
        last_error: BrokerError | None = None
        for attempt in range(self._reconnect_attempts):
            if attempt:
                self._sleep(2.0**attempt)
            try:
                self.connect()
            except BrokerError as exc:
                last_error = exc
                continue
            if self.on_reconnect is not None:
                self.on_reconnect()  # connected again, so its own call()s do not recurse
            return
        raise BrokerError(f"lost the IB Gateway connection and could not reconnect: {last_error}") from last_error

    def _run[T](self, fn: Callable[[Any], T | Awaitable[T]], timeout: float) -> T:
        loop = self._loop
        if loop is None:
            raise BrokerError("IBKR connection not started; call start() first")

        async def runner() -> T:
            result = fn(self._ib)
            if inspect.isawaitable(result):
                return await result
            return result

        future = asyncio.run_coroutine_threadsafe(runner(), loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise BrokerError(f"IB Gateway did not answer within {timeout}s") from exc
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"IB Gateway call failed: {exc}") from exc
