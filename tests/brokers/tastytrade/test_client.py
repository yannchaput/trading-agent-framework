from __future__ import annotations

import json

import httpx
import pytest

from trading_agent_framework.brokers.tastytrade.client import TastytradeHTTPError, TastytradeSession
from trading_agent_framework.config.env import TastytradeSettings
from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError

SETTINGS = TastytradeSettings(client_secret="secret", refresh_token="refresh", account_number="5WT00001", is_paper=True)
ORDERS = "/accounts/5WT00001/orders"


class Api:
    """A scripted Tastytrade: answers /oauth/token itself, then the queued answers per (method, path).

    The last queued answer of a route repeats; an exception in the queue is raised as a transport failure.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.tokens = 0
        self.token_status = 200
        self.routes: dict[tuple[str, str], list[httpx.Response | Exception]] = {}

    def add(self, method: str, path: str, *answers: httpx.Response | Exception) -> None:
        self.routes.setdefault((method, path), []).extend(answers)

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/oauth/token":
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": {"code": "invalid_grant", "message": "Grant is invalid"}})
            self.tokens += 1
            return httpx.Response(200, json={"access_token": f"token-{self.tokens}", "token_type": "Bearer", "expires_in": 900})
        queue = self.routes.get((request.method, request.url.path))
        if not queue:
            return httpx.Response(404, json={"error": {"code": "not_found", "message": "no route"}})
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _session(api: Api, *, clock: Clock | None = None, sleeps: list[float] | None = None) -> TastytradeSession:
    record = sleeps.append if sleeps is not None else (lambda seconds: None)
    return TastytradeSession(SETTINGS, transport=httpx.MockTransport(api), clock=clock or Clock(), sleep=record)


def _ok(data: object) -> httpx.Response:
    return httpx.Response(200, json={"data": data, "context": "/x"})


def test_the_token_is_fetched_once_and_sent_as_a_bearer_with_the_user_agent() -> None:
    api = Api()
    api.add("GET", "/customers/me", _ok({"id": "me"}))
    session = _session(api)

    assert session.get("/customers/me") == {"id": "me"}
    assert session.get("/customers/me") == {"id": "me"}

    token_request, call, _ = api.requests
    assert json.loads(token_request.content) == {"grant_type": "refresh_token", "refresh_token": "refresh", "client_secret": "secret"}
    assert "authorization" not in token_request.headers
    assert call.headers["authorization"] == "Bearer token-1"
    assert call.headers["user-agent"].startswith("trading-agent-framework/")
    assert api.tokens == 1


def test_the_token_is_refreshed_a_minute_before_it_expires() -> None:
    api = Api()
    api.add("GET", "/customers/me", _ok({}))
    clock = Clock()
    session = _session(api, clock=clock)
    session.get("/customers/me")

    clock.now += 839  # 61 s left: still valid
    session.get("/customers/me")
    clock.now += 2  # 59 s left: refreshed before use
    session.get("/customers/me")

    assert api.tokens == 2
    assert api.calls("GET", "/customers/me")[-1].headers["authorization"] == "Bearer token-2"


def test_a_401_refreshes_the_token_and_retries_once() -> None:
    api = Api()
    api.add("GET", "/customers/me", httpx.Response(401, json={"error": {"code": "unauthorized", "message": "expired"}}), _ok({"id": "me"}))

    assert _session(api).get("/customers/me") == {"id": "me"}
    assert api.tokens == 2


def test_a_second_401_is_an_error() -> None:
    api = Api()
    api.add("GET", "/customers/me", httpx.Response(401, json={"error": {"code": "unauthorized", "message": "nope"}}))

    with pytest.raises(TastytradeHTTPError) as excinfo:
        _session(api).get("/customers/me")

    assert excinfo.value.status_code == 401


def test_a_failed_token_request_names_the_variables() -> None:
    api = Api()
    api.token_status = 401

    with pytest.raises(TastytradeHTTPError, match="TASTYTRADE_REFRESH_TOKEN.*sandbox"):
        _session(api).get("/customers/me")


def test_get_retries_a_429_honouring_retry_after() -> None:
    api = Api()
    api.add("GET", "/customers/me", httpx.Response(429, headers={"Retry-After": "2"}, json={"error": {"code": "rate_limited", "message": "slow down"}}), _ok({"id": "me"}))
    sleeps: list[float] = []

    assert _session(api, sleeps=sleeps).get("/customers/me") == {"id": "me"}
    assert sleeps == [2.0]


def test_get_gives_up_after_three_attempts() -> None:
    api = Api()
    api.add("GET", "/customers/me", httpx.Response(503, json={"error": {"code": "unavailable", "message": "down"}}))
    sleeps: list[float] = []

    with pytest.raises(TastytradeHTTPError) as excinfo:
        _session(api, sleeps=sleeps).get("/customers/me")

    assert excinfo.value.status_code == 503
    assert len(api.calls("GET", "/customers/me")) == 3
    assert len(sleeps) == 2


def test_get_retries_a_transport_error() -> None:
    api = Api()
    api.add("GET", "/customers/me", httpx.ConnectTimeout("timed out"), _ok({"id": "me"}))

    assert _session(api).get("/customers/me") == {"id": "me"}


def test_a_post_is_never_retried_and_a_5xx_is_uncertain() -> None:
    api = Api()
    api.add("POST", ORDERS, httpx.Response(503, json={"error": {"code": "unavailable", "message": "down"}}))
    sleeps: list[float] = []

    with pytest.raises(BrokerUncertainError, match="503"):
        _session(api, sleeps=sleeps).post(ORDERS, json={"order-type": "Market"})

    assert len(api.calls("POST", ORDERS)) == 1
    assert sleeps == []


def test_a_post_transport_error_is_uncertain_and_not_raw_httpx() -> None:
    api = Api()
    api.add("POST", ORDERS, httpx.ReadTimeout("timed out"))

    with pytest.raises(BrokerUncertainError):
        _session(api).post(ORDERS, json={})

    assert len(api.calls("POST", ORDERS)) == 1


def test_a_post_429_is_a_certain_error() -> None:
    api = Api()
    api.add("POST", ORDERS, httpx.Response(429, json={"error": {"code": "rate_limited", "message": "slow down"}}))

    with pytest.raises(TastytradeHTTPError) as excinfo:
        _session(api).post(ORDERS, json={})

    assert not isinstance(excinfo.value, BrokerUncertainError)


def test_error_bodies_become_readable_messages() -> None:
    api = Api()
    body = {"error": {"code": "validation_error", "message": "Order is invalid", "errors": [{"code": "insufficient_buying_power", "message": "Not enough buying power"}]}}
    api.add("POST", ORDERS, httpx.Response(422, json=body))

    with pytest.raises(TastytradeHTTPError) as excinfo:
        _session(api).post(ORDERS, json={})

    assert excinfo.value.status_code == 422
    assert f"POST {ORDERS}" in str(excinfo.value)
    assert "insufficient_buying_power: Not enough buying power" in str(excinfo.value)


def test_get_all_follows_pages() -> None:
    api = Api()
    api.add(
        "GET",
        ORDERS,
        httpx.Response(200, json={"data": {"items": [{"id": 1}, {"id": 2}]}, "pagination": {"page-offset": 0, "total-pages": 2}}),
        httpx.Response(200, json={"data": {"items": [{"id": 3}]}, "pagination": {"page-offset": 1, "total-pages": 2}}),
    )

    assert _session(api).get_all(ORDERS, params={"sort": "Desc"}) == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert [r.url.params["page-offset"] for r in api.calls("GET", ORDERS)] == ["0", "1"]
    assert api.calls("GET", ORDERS)[0].url.params["sort"] == "Desc"


def test_get_all_stops_at_max_items() -> None:
    api = Api()
    api.add("GET", ORDERS, httpx.Response(200, json={"data": {"items": [{"id": 1}, {"id": 2}]}, "pagination": {"page-offset": 0, "total-pages": 5}}))

    assert _session(api).get_all(ORDERS, max_items=1) == [{"id": 1}]
    assert len(api.calls("GET", ORDERS)) == 1


def test_get_all_without_pagination_reads_one_page() -> None:
    api = Api()
    api.add("GET", f"{ORDERS}/live", _ok({"items": [{"id": 7}]}))

    assert _session(api).get_all(f"{ORDERS}/live") == [{"id": 7}]


def test_an_empty_delete_returns_an_empty_dict() -> None:
    api = Api()
    api.add("DELETE", f"{ORDERS}/1001", httpx.Response(204))

    assert _session(api).delete(f"{ORDERS}/1001") == {}


def test_a_post_decoding_error_is_uncertain_not_raw_httpx() -> None:
    api = Api()
    api.add("POST", ORDERS, httpx.DecodingError("bad gzip"))

    with pytest.raises(BrokerUncertainError):
        _session(api).post(ORDERS, json={})

    assert len(api.calls("POST", ORDERS)) == 1


def test_get_retries_a_decoding_error_then_raises_a_broker_error() -> None:
    api = Api()
    api.add("GET", "/customers/me", httpx.DecodingError("bad gzip"))

    with pytest.raises(BrokerError) as excinfo:
        _session(api).get("/customers/me")

    assert not isinstance(excinfo.value, httpx.HTTPError)
    assert len(api.calls("GET", "/customers/me")) == 3


def test_a_post_200_with_an_html_body_is_uncertain() -> None:
    api = Api()
    api.add("POST", ORDERS, httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(BrokerUncertainError, match="succeeded"):
        _session(api).post(ORDERS, json={})


def test_a_post_200_with_a_json_list_body_is_uncertain() -> None:
    api = Api()
    api.add("POST", ORDERS, httpx.Response(200, json=[1, 2]))

    with pytest.raises(BrokerUncertainError):
        _session(api).post(ORDERS, json={})


def test_a_get_200_with_an_html_body_is_a_plain_broker_error() -> None:
    api = Api()
    api.add("GET", "/customers/me", httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(BrokerError) as excinfo:
        _session(api).get("/customers/me")

    assert not isinstance(excinfo.value, BrokerUncertainError)
