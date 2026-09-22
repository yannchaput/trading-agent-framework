from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest
from tests.fakes import FakeIB

from trading_agent_framework.brokers.ibkr.client import IbkrConnection
from trading_agent_framework.config.env import IbkrSettings
from trading_agent_framework.utils.errors import BrokerError

SETTINGS = IbkrSettings(host="127.0.0.1", port=4002, client_id=7, is_paper=True)


@pytest.fixture
def ib() -> FakeIB:
    return FakeIB()


@pytest.fixture
def connection(ib: FakeIB) -> Iterator[IbkrConnection]:
    conn = IbkrConnection(SETTINGS, ib_factory=lambda: ib, call_timeout=1.0, sleep=lambda seconds: None)
    conn.start()
    yield conn
    conn.disconnect()


def test_connect_uses_the_settings(connection: IbkrConnection, ib: FakeIB) -> None:
    connection.connect()

    assert ib.connect_calls == [("127.0.0.1", 4002, 7)]


def test_call_returns_plain_and_awaited_results(connection: IbkrConnection) -> None:
    connection.connect()

    assert connection.call(lambda ib: ib.managedAccounts()) == ["DU123"]
    assert connection.call(lambda ib: ib.accountSummaryAsync())[0].tag == "TotalCashValue"


def test_call_runs_on_the_loop_thread(connection: IbkrConnection) -> None:
    connection.connect()

    assert connection.call(lambda ib: threading.current_thread().name) == "ibkr-event-loop"


def test_call_timeout_is_a_broker_error(connection: IbkrConnection) -> None:
    connection.connect()

    with pytest.raises(BrokerError, match="IB Gateway did not answer within 0.1s"):
        connection.call(lambda ib: asyncio.sleep(5), timeout=0.1)


def test_call_wraps_exceptions(connection: IbkrConnection) -> None:
    connection.connect()

    def boom(ib: FakeIB) -> None:
        raise RuntimeError("socket closed")

    with pytest.raises(BrokerError, match="IB Gateway call failed: socket closed"):
        connection.call(boom)


def test_a_failed_connect_is_a_broker_error(connection: IbkrConnection, ib: FakeIB) -> None:
    ib.connect_errors = [ConnectionRefusedError("refused")]

    with pytest.raises(BrokerError, match="Could not connect to IB Gateway at 127.0.0.1:4002"):
        connection.connect()


def test_call_before_connect_is_refused(connection: IbkrConnection) -> None:
    with pytest.raises(BrokerError, match="not connected"):
        connection.call(lambda ib: ib.managedAccounts())


def test_a_dropped_connection_reconnects_then_runs_the_reconcile_hook(connection: IbkrConnection, ib: FakeIB) -> None:
    connection.connect()
    reconciled: list[str] = []
    connection.on_reconnect = lambda: reconciled.append("sync")
    ib.disconnect()  # the Gateway restarted

    assert connection.call(lambda ib: ib.managedAccounts()) == ["DU123"]
    assert len(ib.connect_calls) == 2
    assert reconciled == ["sync"]


def test_reconnect_gives_up_after_the_configured_attempts(connection: IbkrConnection, ib: FakeIB) -> None:
    connection.connect()
    ib.disconnect()
    ib.connect_errors = [ConnectionRefusedError("down")] * 3

    with pytest.raises(BrokerError, match="lost the IB Gateway connection"):
        connection.call(lambda ib: ib.managedAccounts())
    assert len(ib.connect_calls) == 4


def test_call_with_reconnect_false_behaves_normally_when_connected(connection: IbkrConnection) -> None:
    connection.connect()

    assert connection.call(lambda ib: ib.managedAccounts(), reconnect=False) == ["DU123"]


def test_call_with_reconnect_false_fails_fast_when_the_connection_is_down(connection: IbkrConnection, ib: FakeIB) -> None:
    connection.connect()
    connect_calls_before = len(ib.connect_calls)
    ib.disconnect()  # the Gateway restarted

    with pytest.raises(BrokerError, match="reconnect was disabled"):
        connection.call(lambda ib: ib.managedAccounts(), reconnect=False)

    # No reconnect attempt was made: no new connect() call, and therefore no `2**attempt`
    # sleep-based delay (the `sleep` test double would otherwise still be a no-op, so this
    # asserts on call counts rather than wall-clock time).
    assert len(ib.connect_calls) == connect_calls_before


def test_disconnect_is_idempotent(ib: FakeIB) -> None:
    conn = IbkrConnection(SETTINGS, ib_factory=lambda: ib)
    conn.start()
    conn.connect()

    conn.disconnect()
    conn.disconnect()

    assert ib.connected is False
