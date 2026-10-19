from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from tests.fakes import make_tt_order

from trading_agent_framework.brokers.tastytrade.stream import TastytradeAccountStream

URL = "wss://streamer.cert.tastyworks.com"


def ack(socket: FakeSocket) -> dict[str, Any]:
    return {"action": "connect", "status": "ok", "request-id": socket.sent[-1]["request-id"], "web-socket-session-id": "ws-1"}


def refuse(socket: FakeSocket) -> dict[str, Any]:
    return {"action": "connect", "status": "error", "request-id": socket.sent[-1]["request-id"], "message": "invalid auth token"}


def order_message(**overrides: Any) -> dict[str, Any]:
    return {"type": "Order", "data": make_tt_order(**overrides), "timestamp": 1760536800000}


class FakeSocket:
    """Scripted incoming messages (dicts, raw strings, exceptions, or callables of the socket); then 'closed'."""

    def __init__(self, incoming: list[object]) -> None:
        self.incoming = list(incoming)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    def recv(self, timeout: float | None = None) -> str:
        if not self.incoming:
            raise ConnectionError("closed by the server")
        item = self.incoming.pop(0)
        if callable(item):
            item = item(self)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, str) else json.dumps(item)

    def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(self, sockets: list[FakeSocket | Exception], *, stop_after_waits: int = 1, clock: Callable[[], float] = lambda: 0.0) -> None:
        self.sockets = list(sockets)
        self.orders: list[Mapping[str, Any]] = []
        self.reconnects = 0
        self.waits: list[float] = []
        self._stop_after = stop_after_waits
        self.stream = TastytradeAccountStream(URL, "5WT00001", lambda: "tok", self.orders.append, self._on_reconnect, connect=self._connect, clock=clock, wait=self._wait)
        self.reconnected_after: list[int] = []

    def _connect(self, url: str) -> FakeSocket:
        assert url == URL
        socket = self.sockets.pop(0)
        if isinstance(socket, Exception):
            raise socket
        self.current = socket
        return socket

    def _on_reconnect(self) -> None:
        self.reconnects += 1
        self.reconnected_after.append(len(self.current.sent))

    def _wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        if len(self.waits) >= self._stop_after:
            self.stream._stopped.set()
            return True
        return False


def test_it_subscribes_with_the_account_and_a_bearer_token_and_forwards_orders() -> None:
    socket = FakeSocket([ack, order_message(), {"type": "AccountBalance", "data": {}}])
    harness = Harness([socket])

    harness.stream.run()

    assert socket.sent[0] == {"action": "connect", "value": ["5WT00001"], "auth-token": "Bearer tok", "request-id": 1}
    assert [o["id"] for o in harness.orders] == [1001]
    assert harness.reconnects == 0
    assert socket.closed


def test_a_reconnect_reconciles_after_the_new_subscription() -> None:
    first, second = FakeSocket([ack]), FakeSocket([ack])
    harness = Harness([first, second], stop_after_waits=2)

    harness.stream.run()

    assert harness.reconnects == 1
    assert harness.reconnected_after == [1]  # after the second socket's connect was sent and acknowledged
    assert harness.waits == [1.0, 1.0]  # a successful subscription resets the backoff


def test_failed_connections_back_off_up_to_thirty_seconds() -> None:
    harness = Harness([OSError("refused")] * 7, stop_after_waits=7)

    harness.stream.run()

    assert harness.waits == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_a_refused_subscription_reconnects_without_forwarding_anything(caplog: pytest.LogCaptureFixture) -> None:
    harness = Harness([FakeSocket([refuse, order_message()])])

    with caplog.at_level(logging.WARNING):
        harness.stream.run()

    assert harness.orders == []
    assert "invalid auth token" in caplog.text


def test_the_stream_reports_its_subscription_until_the_socket_drops() -> None:
    during: list[tuple[bool, bool]] = []

    def observe(socket: FakeSocket) -> dict[str, Any]:
        during.append((harness.stream.is_subscribed, harness.stream.wait_subscribed(0)))
        return order_message()

    harness = Harness([FakeSocket([ack, observe])])
    assert not harness.stream.is_subscribed

    harness.stream.run()

    assert during == [(True, True)]
    assert not harness.stream.is_subscribed
    assert not harness.stream.wait_subscribed(0)


def test_a_refused_subscription_is_kept_as_the_last_error_until_a_subscription_succeeds() -> None:
    errors: list[str | None] = []

    def observe(socket: FakeSocket) -> dict[str, Any]:
        errors.append(harness.stream.last_error)
        return order_message()

    harness = Harness([FakeSocket([refuse]), FakeSocket([ack, observe])], stop_after_waits=2)

    harness.stream.run()

    assert errors == [None]
    assert harness.stream.last_error is not None
    assert "closed by the server" in harness.stream.last_error  # the second socket's drop


def test_the_last_error_names_the_refusal() -> None:
    harness = Harness([FakeSocket([refuse])])

    harness.stream.run()

    assert harness.stream.last_error is not None
    assert "invalid auth token" in harness.stream.last_error


def test_a_streamer_error_after_subscribing_reconnects(caplog: pytest.LogCaptureFixture) -> None:
    harness = Harness([FakeSocket([ack, {"status": "error", "message": "token expired"}, order_message()])])

    with caplog.at_level(logging.WARNING):
        harness.stream.run()

    assert harness.orders == []
    assert "token expired" in caplog.text


def test_heartbeats_go_out_on_schedule_with_a_fresh_token() -> None:
    now = [0.0]

    def silence(socket: FakeSocket) -> TimeoutError:
        now[0] += 21.0
        return TimeoutError()

    socket = FakeSocket([ack, silence])
    harness = Harness([socket], clock=lambda: now[0])

    harness.stream.run()

    assert socket.sent[1] == {"action": "heartbeat", "value": "", "auth-token": "Bearer tok", "request-id": 2}


def test_a_failing_order_callback_does_not_drop_the_connection(caplog: pytest.LogCaptureFixture) -> None:
    received: list[Mapping[str, Any]] = []

    def on_order(data: Mapping[str, Any]) -> None:
        if not received:
            received.append(data)
            raise RuntimeError("boom")
        received.append(data)

    socket = FakeSocket([ack, order_message(order_id=1), order_message(order_id=2)])
    harness = Harness([socket])
    harness.stream._on_order = on_order

    with caplog.at_level(logging.ERROR):
        harness.stream.run()

    assert [o["id"] for o in received] == [1, 2]
    assert "boom" in caplog.text


def test_unreadable_messages_are_skipped() -> None:
    harness = Harness([FakeSocket([ack, "not json", order_message()])])

    harness.stream.run()

    assert len(harness.orders) == 1


class BlockingSocket(FakeSocket):
    def __init__(self) -> None:
        super().__init__([ack])
        self._closed = threading.Event()

    def recv(self, timeout: float | None = None) -> str:
        if self.incoming:
            return super().recv(timeout)
        if self._closed.wait(timeout):
            raise ConnectionError("closed")
        raise TimeoutError

    def close(self) -> None:
        self.closed = True
        self._closed.set()


def test_start_and_stop_run_the_thread() -> None:
    socket = BlockingSocket()
    stream = TastytradeAccountStream(URL, "5WT00001", lambda: "tok", lambda data: None, lambda: None, connect=lambda url: socket)

    stream.start()
    deadline = time.monotonic() + 2.0
    while not socket.sent and time.monotonic() < deadline:  # wait until the thread has connected
        time.sleep(0.01)
    assert stream.is_running
    stream.stop(timeout=2.0)

    assert not stream.is_running
    assert socket.closed
