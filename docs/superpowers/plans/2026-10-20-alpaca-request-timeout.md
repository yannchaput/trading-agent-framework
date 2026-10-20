# Alpaca Request Timeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every Alpaca REST client a 30-second request timeout, so a stalled HTTP read raises a network error instead of hanging the run forever.

**Architecture:** alpaca-py 0.44's `RESTClient` owns a `requests.Session` (`client._session`) and calls `self._session.request(method, url, **opts)` with no `timeout`, so `requests` waits forever. Each builder in `brokers/alpaca/client.py` mounts a small `HTTPAdapter` subclass on its client's session that fills in a default timeout when none is given. Every caller already turns any client exception into a `BrokerError` (or the news tool's `{"error": ...}`), so no caller changes.

**Tech Stack:** Python 3.14, `uv`, `pytest`, `ruff`, `requests` (already installed through alpaca-py), alpaca-py 0.44.

**Spec:** none (a small change, approved in chat on 2026-10-20). Background: on 2026-10-20 the bull_bear paper run and backtest both hung for hours on a `data.alpaca.markets` socket inside `search_news`; vLLM was idle, and killing the sockets with `ss -K` let both runs carry on.

## Global Constraints

- Timeout: `REQUEST_TIMEOUT_SECONDS = 30.0` (same as `fundamentals/edgar_client.py`), one value for connect and read, for every Alpaca REST call: news, market data, calendar, trading (orders included).
- A stall gives up and carries on: no new retries, no change to any caller (`AlpacaBroker`, `AlpacaMarketData`, `AlpacaNewsProvider`, `AlpacaMarketClock`, `AlpacaBacktestData`, `dashboard/account_reader.py`).
- A timed-out order submit is a failed order, like any other client exception today (`order.set_error`, untracked, re-raised), although Alpaca may have received it. The user accepted that a strategy may buy again on its next tick. No lookup by `client_order_id` (unlike Tastytrade's `BrokerUncertainError`).
- Out of scope: `build_trading_stream` (websocket), the LLM client (no timeout on purpose, `agents/manager.py:397`), and the raw `requests` exception that `AlpacaBroker.pull_order` already lets through on any connection failure (existing behaviour, not made worse).
- `client.py` keeps importing `alpaca` only inside the builder bodies; a module-level `requests` import is fine.
- Tests never touch the network: building an alpaca-py client makes no request, and `HTTPAdapter.send` is monkeypatched.
- Ruff config: `line-length = 200`, rules `E, F, I, UP, B`.
- Before every commit: `uv run pytest && uv run ruff check`, both passing.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Work in a git worktree off `main`** (`superpowers:using-git-worktrees`, branch `fix/alpaca-request-timeout`): a bull_bear paper run and backtest are running from the main checkout, and `backtesting/data/alpaca.py` imports `brokers.alpaca.client` lazily, so switching that checkout's branch mid-run could load new code into them.

## Review Focus

- **A request through a built client's own session** (not the adapter called directly) must carry the timeout: `Session.request` passes `timeout=None` to the adapter by default. Pinned in Task 1 (`test_built_clients_send_requests_with_the_timeout`, through `client._session.request`).
- **A caller that passes its own timeout** keeps it (a future alpaca-py may pass one). Pinned in Task 1 (`test_timeout_adapter_keeps_an_explicit_timeout`).
- **Both URL schemes** are covered: the SDK session mounts `https://` and `http://`, and a test or sandbox base URL may be `http`. Pinned in Task 1 (`test_built_clients_send_requests_with_the_timeout`, asserting the adapter on both prefixes).
- **A large but steadily streaming response** (a 1,200-symbol bars page) is not cut: the `requests` timeout limits the wait between bytes, not the whole transfer. No test (it is `requests`/`urllib3` behaviour); stated in the `brokers/CLAUDE.md` text in Task 1.
- **A timed-out `submit_order`** follows the existing failure path (`set_error`, untracked, re-raised). Already pinned by `tests/brokers/alpaca/test_broker.py::test_submit_order_raises_sets_error_and_tracks_nowhere`, which covers any client exception; no new test.

---

### Task 1: Default request timeout on every built Alpaca REST client

**Files:**
- Modify: `src/trading_agent_framework/brokers/alpaca/client.py`
- Modify: `src/trading_agent_framework/brokers/CLAUDE.md` (Architecture line for `client.py`, one new Gotchas bullet)
- Test: `tests/brokers/alpaca/test_client.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (all in `brokers/alpaca/client.py`):
  - `REQUEST_TIMEOUT_SECONDS: float = 30.0`
  - `class _TimeoutAdapter(requests.adapters.HTTPAdapter)`: `__init__(self, timeout: float, **kwargs: Any) -> None` (stores `self.timeout`, passes `kwargs` to `HTTPAdapter`); `send(self, request, **kwargs)` sets `kwargs["timeout"] = self.timeout` when `kwargs.get("timeout") is None`, then returns `super().send(request, **kwargs)`.
  - `def _with_timeout[T](client: T) -> T` (PEP 695 syntax; ruff's `UP` rules refuse a `TypeVar`): mounts one `_TimeoutAdapter(REQUEST_TIMEOUT_SECONDS)` on `client._session` for `"https://"` and `"http://"` and returns `client`.
  - `build_trading_client`, `build_stock_data_client` and `build_news_client` keep their signatures and return `_with_timeout(<SDK client>)`. `build_trading_stream` is unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/brokers/alpaca/test_client.py` (keep the existing tests: they pass a `MagicMock` class, and `_with_timeout` on a `MagicMock` instance must not break them)

```python
import requests
from requests.adapters import HTTPAdapter

from trading_agent_framework.brokers.alpaca.client import REQUEST_TIMEOUT_SECONDS, _TimeoutAdapter, build_news_client


def _capture_send(monkeypatch) -> list[dict]:
    """Replace the real transport: record the kwargs each send gets and answer 200 with an empty JSON body."""
    calls: list[dict] = []

    def fake_send(self, request, **kwargs):
        calls.append(kwargs)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.request = request
        return response

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    return calls


def test_request_timeout_is_thirty_seconds() -> None:
    assert REQUEST_TIMEOUT_SECONDS == 30.0


@pytest.mark.parametrize("build", [build_trading_client, build_stock_data_client, build_news_client])
def test_built_clients_send_requests_with_the_timeout(monkeypatch, build) -> None:
    calls = _capture_send(monkeypatch)
    client = build(AlpacaCredentials(api_key="key", api_secret="secret", is_paper=True))

    for prefix in ("https://", "http://"):
        adapter = client._session.adapters[prefix]
        assert isinstance(adapter, _TimeoutAdapter)
        assert adapter.timeout == REQUEST_TIMEOUT_SECONDS
    client._session.request("GET", "https://data.alpaca.markets/v1beta1/news")

    assert calls[-1]["timeout"] == REQUEST_TIMEOUT_SECONDS


def test_timeout_adapter_keeps_an_explicit_timeout(monkeypatch) -> None:
    calls = _capture_send(monkeypatch)
    session = requests.Session()
    session.mount("https://", _TimeoutAdapter(REQUEST_TIMEOUT_SECONDS))

    session.request("GET", "https://data.alpaca.markets/v1beta1/news", timeout=5)

    assert calls[-1]["timeout"] == 5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/alpaca/test_client.py -v`
Expected: collection error, `ImportError: cannot import name 'REQUEST_TIMEOUT_SECONDS'`.

- [ ] **Step 3: Implement `REQUEST_TIMEOUT_SECONDS`, `_TimeoutAdapter` and `_with_timeout` in `brokers/alpaca/client.py`**, as given in Interfaces, and wrap the return of the three REST builders in `_with_timeout(...)`. Mount ONE adapter instance on both prefixes. Its default `max_retries` (0) and pool sizes match the `HTTPAdapter` that `requests.Session` mounts, so nothing changes but the timeout. Add one sentence to the module docstring: alpaca-py sets no request timeout, so each REST client gets `REQUEST_TIMEOUT_SECONDS` here, since a stalled read would otherwise block forever.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/alpaca/test_client.py -v`
Expected: all PASS, the 6 existing tests included.

- [ ] **Step 5: Update `src/trading_agent_framework/brokers/CLAUDE.md`**

In Architecture, the `brokers/alpaca/client.py` line becomes:

```
- `brokers/alpaca/client.py` -- trivial factory functions for building Alpaca SDK clients from credentials, so tests can inject fakes instead; every REST client gets a `REQUEST_TIMEOUT_SECONDS` (30 s) request timeout here (the stream is not touched).
```

In Gotchas, add after the **Market data:** bullet:

```
- **Every Alpaca REST client times out after 30 s** (`client.REQUEST_TIMEOUT_SECONDS`, a `_TimeoutAdapter` mounted on the SDK's own `requests` session). alpaca-py sets no timeout, and on 2026-10-20 a stalled `data.alpaca.markets` read hung a bull_bear paper run and a backtest for hours inside `search_news`. The limit is on the wait for the next bytes, not the whole transfer, so a large response that keeps streaming is not cut. A stall now raises like any network error, and the existing `BrokerError` paths take it from there; there are no retries. **Accepted risk:** a timed-out `submit_order` is reported as a failed order (`set_error`, untracked, re-raised) although Alpaca may have received it, so a strategy may buy again on its next tick. There is no lookup by `client_order_id`, unlike Tastytrade's `BrokerUncertainError`.
```

- [ ] **Step 6: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: all tests pass, `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/brokers/alpaca/client.py src/trading_agent_framework/brokers/CLAUDE.md tests/brokers/alpaca/test_client.py
git commit -m "fix(brokers): time out Alpaca REST requests after 30 s

alpaca-py sends every request with no timeout, so a stalled read on
data.alpaca.markets hung a bull_bear paper run and a backtest for hours.
Each built REST client now mounts an adapter with a 30 s default timeout.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Manual check against the real API** (the user runs it, with paper credentials)

Run: `uv run python scripts/tests/smoke_alpaca_data.py`
Expected: same output as before the change (bars, prices, quotes, indicators all reported OK).
