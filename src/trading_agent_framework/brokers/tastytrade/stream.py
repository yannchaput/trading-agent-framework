"""`TastytradeAccountStream`: the account-streamer websocket thread (order notifications).

Subscribes with the account number and a fresh access token, heartbeats, forwards `Order`
notifications to a callback, and reconnects with backoff -- calling `on_reconnect` (the broker's
`reconcile`) after every subscription but the first. Only feeds its callbacks; never calls
strategy hooks. `websockets` is imported inside `_default_connect` so `brokers` stays import-light.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 20.0  # Tastytrade wants one every 2 s to 1 min
CONNECT_TIMEOUT = 10.0
MAX_BACKOFF = 30.0


class StreamSocket(Protocol):
    """The slice of a websocket the account stream needs."""

    def send(self, message: str) -> None:
        """Send one text message."""
        ...

    def recv(self, timeout: float | None = None) -> str:
        """Receive one text message, waiting up to `timeout` seconds."""
        ...

    def close(self) -> None:
        """Close the socket."""
        ...


def auth_token_value(token: str) -> str:
    """The streamer's `auth-token` field: `Bearer <token>`, as in the REST `Authorization` header.

    Confirmed against the sandbox (2026-10-19): a bare token is refused with "Unknown domain".
    """
    return f"Bearer {token}"


class StreamError(Exception):
    """The streamer refused a request or reported an error: reconnect."""


def _default_connect(url: str) -> StreamSocket:
    """Open a real websocket to `url`."""
    from websockets.sync.client import connect

    from trading_agent_framework.brokers.tastytrade.client import USER_AGENT

    return connect(url, user_agent_header=USER_AGENT, open_timeout=CONNECT_TIMEOUT)  # ty: ignore[invalid-return-type]


class TastytradeAccountStream:
    """Background thread that keeps a Tastytrade account-streamer websocket subscribed and forwards order messages."""

    def __init__(
        self,
        url: str,
        account_number: str,
        token: Callable[[], str],
        on_order: Callable[[Mapping[str, Any]], object],
        on_reconnect: Callable[[], None],
        *,
        connect: Callable[[str], StreamSocket] = _default_connect,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        """Configure the stream.

        Args:
            url: Account-streamer websocket URL.
            account_number: Account to subscribe to.
            token: Returns a fresh session token for each connection.
            on_order: Called with every order message.
            on_reconnect: Called after a reconnect so orders missed meanwhile can be reconciled.
            connect: Opens the websocket for a URL (injected in tests).
            clock: Monotonic time source for the connect deadline and the heartbeat schedule.
            wait: Waits between reconnection attempts and returns True if the stream was stopped meanwhile; defaults to
                waiting on the stream's own stop event.
        """
        self._url = url
        self._account_number = account_number
        self._token = token
        self._on_order = on_order
        self._on_reconnect = on_reconnect
        self._connect = connect
        self._clock = clock
        self._stopped = threading.Event()
        self._subscribed = threading.Event()
        self._last_error: str | None = None
        self._wait = wait if wait is not None else self._stopped.wait
        self._socket: StreamSocket | None = None
        self._thread: threading.Thread | None = None
        self._request_ids = itertools.count(1)

    @property
    def is_running(self) -> bool:
        """Whether the stream thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_subscribed(self) -> bool:
        """Whether the streamer accepted the subscription on the current socket."""
        return self._subscribed.is_set()

    @property
    def last_error(self) -> str | None:
        """Why the last connection failed or dropped (None until one does, and again after a subscription)."""
        return self._last_error

    def wait_subscribed(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the account subscription; return whether it succeeded."""
        return self._subscribed.wait(timeout)

    def start(self) -> None:
        """Start the stream thread; no-op when it is already running."""
        if self.is_running:
            return
        self._stopped.clear()
        self._thread = threading.Thread(target=self.run, name="tastytrade-account-stream", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the stream and join its thread, waiting up to `timeout` seconds."""
        self._stopped.set()
        socket = self._socket
        if socket is not None:
            try:
                socket.close()
            except Exception:
                logger.debug("closing the Tastytrade account-streamer socket failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self) -> None:
        """Connect, subscribe, serve and reconnect with backoff until `stop()` (public so tests can run it inline)."""
        failures = 0
        first_attempt = True
        while not self._stopped.is_set():
            try:
                socket = self._connect(self._url)
                self._socket = socket
                try:
                    self._subscribe(socket)
                    self._last_error = None
                    self._subscribed.set()
                    failures = 0
                    if not first_attempt:
                        self._reconcile()
                    self._serve(socket)
                finally:
                    self._subscribed.clear()
                    self._socket = None
                    try:
                        socket.close()
                    except Exception:
                        logger.debug("closing the Tastytrade account-streamer socket failed", exc_info=True)
            except Exception as exc:
                if self._stopped.is_set():
                    break
                self._last_error = str(exc) or repr(exc)
                logger.warning("Tastytrade account stream disconnected; reconnecting", exc_info=True)
            first_attempt = False
            if self._stopped.is_set():
                break
            failures += 1
            self._wait(min(MAX_BACKOFF, 2.0 ** (failures - 1)))

    def _reconcile(self) -> None:
        """Run the reconnect callback, logging rather than raising its failures."""
        try:
            self._on_reconnect()
        except Exception:
            logger.exception("reconciling Tastytrade orders after a reconnect failed")

    def _send(self, socket: StreamSocket, action: str, value: object = None) -> int:
        """Send an action with a fresh request id and return the id."""
        request_id = next(self._request_ids)
        message: dict[str, Any] = {"action": action}
        if value is not None:
            message["value"] = value
        message["auth-token"] = auth_token_value(self._token())
        message["request-id"] = request_id
        socket.send(json.dumps(message))
        return request_id

    def _subscribe(self, socket: StreamSocket) -> None:
        """Subscribe to the account and wait for the acknowledgement."""
        request_id = self._send(socket, "connect", [self._account_number])
        deadline = self._clock() + CONNECT_TIMEOUT
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise StreamError("no answer to the account-streamer connect request")
            message = self._parse(socket.recv(timeout=remaining))
            if str(message.get("request-id")) == str(request_id):
                if message.get("status") != "ok":
                    raise StreamError(f"account-streamer connect refused: {message.get('message') or message}")
                return
            self._dispatch(message)

    def _serve(self, socket: StreamSocket) -> None:
        """Read messages and send heartbeats until the socket drops or the stream is stopped."""
        next_beat = self._clock() + HEARTBEAT_INTERVAL
        while not self._stopped.is_set():
            remaining = next_beat - self._clock()
            if remaining <= 0:
                self._send(socket, "heartbeat", "")
                next_beat = self._clock() + HEARTBEAT_INTERVAL
                continue
            try:
                raw = socket.recv(timeout=remaining)
            except TimeoutError:
                continue
            self._dispatch(self._parse(raw))

    def _dispatch(self, message: Mapping[str, Any]) -> None:
        """Route one message: order messages go to the order callback."""
        if message.get("type") == "Order":
            data = message.get("data")
            if isinstance(data, Mapping):
                try:
                    self._on_order(data)
                except Exception:
                    logger.exception("applying a Tastytrade order notification failed")
        elif message.get("status") == "error":
            raise StreamError(f"account streamer error: {message.get('message') or message}")

    @staticmethod
    def _parse(raw: object) -> dict[str, Any]:
        """Decode a raw message to a dict; non-JSON payloads become an empty dict."""
        try:
            message = json.loads(raw)  # ty: ignore[invalid-argument-type]
        except TypeError, ValueError:
            logger.warning("unreadable account-streamer message: %r", str(raw)[:200])
            return {}
        return message if isinstance(message, dict) else {}
