"""`TastytradeSession`: the only HTTP code of the Tastytrade broker.

One sync `httpx.Client`; OAuth access tokens (15 minutes) refreshed from the refresh token under a
lock, shared by REST calls and the account-streamer thread; GETs retried on 429/5xx/transport
errors; writes never retried (Tastytrade does not deduplicate orders). Raw `httpx` exceptions never
escape: `TastytradeHTTPError` for an error response, `BrokerUncertainError` when a write's outcome
is unknown.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import httpx

from trading_agent_framework.utils import get_version
from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import TastytradeSettings

logger = logging.getLogger(__name__)

USER_AGENT = f"trading-agent-framework/{get_version('trading_agent_framework')}"
TOKEN_REFRESH_MARGIN = 60.0  # seconds before expiry
GET_ATTEMPTS = 3
PAGE_SIZE = 200
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class TastytradeHTTPError(BrokerError):
    """An error response from Tastytrade; `status_code` tells e.g. a 404 (unknown symbol) from a 422 (refused order)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class TastytradeSession:
    def __init__(
        self,
        settings: TastytradeSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 10.0,
    ) -> None:
        self._settings = settings
        self._client = httpx.Client(base_url=settings.api_url, headers={"User-Agent": USER_AGENT}, timeout=timeout, transport=transport)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0

    # --- auth ------------------------------------------------------------------------------

    def access_token(self, *, force: bool = False) -> str:
        """A valid access token: fetched when missing, within a minute of expiry, or when forced."""
        with self._lock:
            if force or self._token is None or self._clock() >= self._expires_at - TOKEN_REFRESH_MARGIN:
                self._token, self._expires_at = self._fetch_token()
            token = self._token
        return token

    def _fetch_token(self) -> tuple[str, float]:
        body = {"grant_type": "refresh_token", "refresh_token": self._settings.refresh_token, "client_secret": self._settings.client_secret}
        try:
            response = self._client.post("/oauth/token", json=body)
        except httpx.HTTPError as exc:
            raise BrokerError(f"Tastytrade token request failed: {exc!r}") from exc
        if not response.is_success:
            environment = "sandbox" if self._settings.is_paper else "production"
            raise TastytradeHTTPError(
                response.status_code,
                f"Tastytrade token request failed: {_describe(response)} (check TASTYTRADE_CLIENT_SECRET / TASTYTRADE_REFRESH_TOKEN, and that they were issued for the {environment})",
            )
        payload = _json(response)
        token = payload.get("access_token")
        if not token:
            raise BrokerError("Tastytrade token response has no access_token")
        return str(token), self._clock() + float(payload.get("expires_in") or 900)

    # --- requests --------------------------------------------------------------------------

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return _data(self._send("GET", path, params=params))

    def post(self, path: str, *, json: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return _write_data(self._send("POST", path, json=json))

    def put(self, path: str, *, json: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return _write_data(self._send("PUT", path, json=json))

    def delete(self, path: str) -> dict[str, Any]:
        return _write_data(self._send("DELETE", path))

    def get_all(self, path: str, *, params: Mapping[str, Any] | None = None, max_items: int | None = None) -> list[dict[str, Any]]:
        """Every `data.items` across pages, or only the first `max_items`."""
        query: dict[str, Any] = {"per-page": PAGE_SIZE, **(params or {})}
        items: list[dict[str, Any]] = []
        offset = 0
        while True:
            body = _json(self._send("GET", path, params={**query, "page-offset": offset}))
            items.extend((body.get("data") or {}).get("items") or [])
            if max_items is not None and len(items) >= max_items:
                return items[:max_items]
            pagination = body.get("pagination") or {}
            if offset >= int(pagination.get("total-pages") or 1) - 1:
                return items
            offset += 1

    def close(self) -> None:
        self._client.close()

    def _send(self, method: str, path: str, *, json: Mapping[str, Any] | None = None, params: Mapping[str, Any] | None = None) -> httpx.Response:
        attempts = GET_ATTEMPTS if method == "GET" else 1
        attempt = 0
        reauthorized = False
        while True:
            attempt += 1
            headers = {"Authorization": f"Bearer {self.access_token()}"}
            try:
                response = self._client.request(method, path, json=json, params=params, headers=headers)
            except httpx.RequestError as exc:  # transport, decoding, redirect loops: never a raw httpx error
                if attempt < attempts:
                    self._sleep(_backoff(attempt, None))
                    continue
                raise BrokerUncertainError(f"Tastytrade {method} {path} failed: {exc!r}") from exc
            if response.status_code == 401 and not reauthorized:
                reauthorized = True  # a 401 was not processed: safe to retry any method once
                attempt -= 1
                self.access_token(force=True)
                continue
            if response.status_code in _RETRYABLE_STATUSES and attempt < attempts:
                self._sleep(_backoff(attempt, response.headers.get("Retry-After")))
                continue
            if response.is_success:
                return response
            message = f"Tastytrade {method} {path}: {_describe(response)}"
            if response.status_code >= 500 and method != "GET":
                raise BrokerUncertainError(message)
            raise TastytradeHTTPError(response.status_code, message)


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(8.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.25)


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        raise BrokerError(f"Tastytrade returned non-JSON ({response.status_code}): {response.text[:200]!r}") from None
    if not isinstance(body, dict):
        raise BrokerError(f"Tastytrade returned unexpected JSON ({response.status_code}): {str(body)[:200]}")
    return body


def _data(response: httpx.Response) -> dict[str, Any]:
    if response.status_code == 204 or not response.content:
        return {}
    data = _json(response).get("data")
    return data if isinstance(data, dict) else {}


def _write_data(response: httpx.Response) -> dict[str, Any]:
    """`_data` for a write that succeeded: an unreadable body means the outcome is unknown, not rejected."""
    try:
        return _data(response)
    except BrokerError as exc:
        raise BrokerUncertainError(f"Tastytrade write succeeded ({response.status_code}) but its response was unreadable: {exc}") from exc


def _describe(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"{response.status_code} {response.text[:200]}"
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return f"{response.status_code} {str(body)[:200]}"
    parts = [f"{e.get('code')}: {e.get('message')}" for e in [error, *(error.get("errors") or [])] if isinstance(e, dict) and e.get("message")]
    return f"{response.status_code} " + ("; ".join(parts) or str(error)[:200])
