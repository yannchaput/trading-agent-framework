# News-builtin Strategy Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the pasted lumibot `news_builtin` strategy to an LLM-agent `NewsBuiltinStrategy` on this framework (PrebuiltTools + news tool + memory + backtesting), closing the four framework gaps the port exposes.

**Architecture:** A broker-independent `NewsProvider` seam (`Broker.news_provider()`, `BacktestBroker(news_source=)`) lets `search_news` work in every mode; `LLMCredentials` learns to normalize a pasted endpoint URL and to make the API key optional; news payloads get token caps; `main.py` gets a builder registry so a second strategy can be launched. The strategy itself is a small `Strategy` subclass plus a prompts module.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff, LangChain (`create_agent`, `langchain_openai`), alpaca-py (news client), yfinance (Yahoo backtest data).

**Spec:** `docs/superpowers/specs/2026-09-19-news-builtin-strategy-design.md`

## Global Constraints

- Package manager is `uv` (`uv run pytest`, `uv run ruff check`); Python 3.14; ruff line length 200.
- Automated tests never touch the network; use hand-written fakes from `tests/fakes.py` (`FakeBroker`, `FakeClock`, `FakeNewsClient`, `make_alpaca_news_article`, `FakeToolCallingChatModel`, `et`), not `MagicMock`.
- Money is `Decimal`; do not add a fourth float boundary.
- Never let a raw SDK exception escape a tool: tools return `{"error": ...}`.
- No `time.sleep` / `datetime.now` in strategy or executor code; use `strategy.clock` / `strategy.sleep()`.
- `brokers/base.py`, `brokers/news.py` and `backtesting/broker.py` must not import `alpaca` at module level (lazy imports inside method bodies only).
- `agents/` defers every LangChain import to inside method bodies.
- Tool docstrings stay one line (they reach the LLM on every call).
- Commit trailer on every commit: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`. Commit only files the task touched (never `git add -A`); `env/*` is git-ignored except `env/.env.example`.
- Repo cadence: each task is followed by a review-fix commit before the next task starts.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/trading_agent_framework/agents/config.py` | `LLMCredentials`, `normalize_base_url`, `PLACEHOLDER_API_KEY` |
| `src/trading_agent_framework/brokers/news.py` (new) | `NewsProvider` protocol (no alpaca import) |
| `src/trading_agent_framework/brokers/base.py` | `Broker.news_provider()` default hook |
| `src/trading_agent_framework/brokers/alpaca/news.py` (new) | `AlpacaNewsProvider` (I/O wrapper over pure `market_data` news functions) |
| `src/trading_agent_framework/brokers/alpaca/broker.py` | delegate `get_news` to the provider, expose `news_provider()` |
| `src/trading_agent_framework/brokers/alpaca/market_data.py` | `parse_news` content truncation |
| `src/trading_agent_framework/agents/tools/news.py` | provider lookup, content-limit clamp |
| `src/trading_agent_framework/backtesting/broker.py`, `runner.py`, `core/strategy.py` | `news_source=` plumbing |
| `src/trading_agent_framework/strategies/news_builtin/` (new) | `prompts.py`, `agent_news_builtin.py`, `__init__.py` |
| `src/trading_agent_framework/main.py` | builder registry |
| `scripts/tests/smoke_strategy_news.py` (new), `scripts/tests/HOWTO.md` | manual verification |
| `CLAUDE.md`, `README.md`, `env/.env.example` | docs |

---

### Task 1: LLM config — URL normalization and optional API key

**Files:**
- Modify: `src/trading_agent_framework/agents/config.py`
- Modify: `tests/agents/test_agent_config.py`
- Modify: `env/.env.example` (comment lines only)
- Modify (git-ignored, not committed): `env/.env.news_builtin.backtesting`

**Interfaces:**
- Produces: `normalize_base_url(url: str) -> str`; `PLACEHOLDER_API_KEY: str = "not-needed"`; `LLMCredentials.from_env` now normalizes `base_url` and defaults `api_key` to the placeholder.

- [ ] **Step 1: Write the failing tests**

In `tests/agents/test_agent_config.py`, change the import line and replace `test_from_env_missing_or_blank_api_key_raises`, then append the normalization tests:

```python
from trading_agent_framework.agents.config import PLACEHOLDER_API_KEY, LLMCredentials, normalize_base_url
```

```python
@pytest.mark.parametrize("api_key_value", [None, "", "   "])
def test_from_env_missing_or_blank_api_key_falls_back_to_the_placeholder(api_key_value: str | None) -> None:
    env = {"LLM_BASE_URL": "http://localhost:8000/v1"}
    if api_key_value is not None:
        env["LLM_API_KEY"] = api_key_value

    credentials = LLMCredentials.from_env(env)

    assert credentials.api_key == PLACEHOLDER_API_KEY


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:8005//v1/chat/completions", "http://localhost:8005/v1"),
        ("http://localhost:8005/v1/chat/completions", "http://localhost:8005/v1"),
        ("http://localhost:8005/v1/chat/completions/", "http://localhost:8005/v1"),
        ("http://localhost:8005/v1/", "http://localhost:8005/v1"),
        ("http://localhost:8005//v1", "http://localhost:8005/v1"),
        ("  http://localhost:8005/v1  ", "http://localhost:8005/v1"),
        ("https://api.example.com/v1", "https://api.example.com/v1"),
    ],
)
def test_normalize_base_url(raw: str, expected: str) -> None:
    assert normalize_base_url(raw) == expected


def test_from_env_normalizes_the_base_url() -> None:
    env = {"LLM_BASE_URL": "http://localhost:8005//v1/chat/completions"}

    assert LLMCredentials.from_env(env).base_url == "http://localhost:8005/v1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_agent_config.py -v`
Expected: FAIL/ERROR with `ImportError: cannot import name 'PLACEHOLDER_API_KEY'`.

- [ ] **Step 3: Implement**

Replace `src/trading_agent_framework/agents/config.py` with:

```python
"""Env-driven LLM connection credentials for the agent framework (pure, no I/O)."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from trading_agent_framework.utils.errors import ConfigurationError

# The OpenAI SDK needs a non-blank key to build a client; local servers (vLLM, llama.cpp) ignore it.
PLACEHOLDER_API_KEY = "not-needed"

_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def normalize_base_url(url: str) -> str:
    """Reduce a pasted OpenAI-compatible endpoint to the base URL `ChatOpenAI` expects (e.g. `http://host:8005/v1`)."""
    cleaned = url.strip()
    scheme, separator, rest = cleaned.partition("://")
    cleaned = f"{scheme}{separator}{re.sub(r'/{2,}', '/', rest)}".rstrip("/")
    if cleaned.endswith(_CHAT_COMPLETIONS_SUFFIX):
        cleaned = cleaned[: -len(_CHAT_COMPLETIONS_SUFFIX)]
    return cleaned.rstrip("/")


@dataclass(frozen=True, slots=True)
class LLMCredentials:
    base_url: str
    api_key: str = field(repr=False)
    default_model: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMCredentials:
        source = env if env is not None else os.environ

        base_url = source.get("LLM_BASE_URL")
        if not base_url or not base_url.strip():
            raise ConfigurationError("Missing or blank LLM_BASE_URL environment variable")

        api_key = source.get("LLM_API_KEY")
        if not api_key or not api_key.strip():
            api_key = PLACEHOLDER_API_KEY

        default_model = source.get("LLM_MODEL")
        if default_model is not None and not default_model.strip():
            default_model = None

        return cls(base_url=normalize_base_url(base_url), api_key=api_key, default_model=default_model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents -q && uv run ruff check`
Expected: PASS, ruff clean.

- [ ] **Step 5: Update the env files**

1. `env/.env.example`: use the Edit tool to replace the two comment lines above `LLM_API_KEY=` (the ones reading "Most local OpenAI-compatible servers ignore this value, but the OpenAI SDK / requires a non-blank string to construct the client. Use any placeholder.") with:
   `# Optional. Most local OpenAI-compatible servers ignore it; when blank or missing the framework sends a placeholder.`
   and replace the comment line(s) above `LLM_BASE_URL=` so they also say: `A pasted full endpoint (.../v1/chat/completions) is accepted and normalized to the /v1 base.`
   Do not touch any `KEY=`/`SECRET=` value line.
2. `env/.env.news_builtin.backtesting` (git-ignored): run `sed -i 's/^API_BASE_URL=/LLM_BASE_URL=/' env/.env.news_builtin.backtesting` then `grep -c '^LLM_BASE_URL=' env/.env.news_builtin.backtesting` (expect `1`) — never print the credential lines.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/config.py tests/agents/test_agent_config.py env/.env.example
git commit -m "Task 1: LLMCredentials normalizes the base URL and makes the API key optional

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: NewsProvider seam and AlpacaNewsProvider

**Files:**
- Create: `src/trading_agent_framework/brokers/news.py`
- Create: `src/trading_agent_framework/brokers/alpaca/news.py`
- Modify: `src/trading_agent_framework/brokers/base.py`
- Modify: `src/trading_agent_framework/brokers/alpaca/broker.py`
- Create: `tests/brokers/alpaca/test_alpaca_news.py`
- Modify: `tests/brokers/test_base.py`
- Modify: `tests/brokers/alpaca/test_broker_market_data.py`

**Interfaces:**
- Produces:
  - `brokers.news.NewsProvider` protocol: `get_news(self, symbols: Sequence[str] = (), *, start: datetime | None = None, end: datetime, limit: int = 10, include_content: bool = False) -> list[dict[str, object]]`
  - `Broker.news_provider(self) -> NewsProvider | None` (default `None`)
  - `brokers.alpaca.news.AlpacaNewsProvider(client)` with `from_credentials(creds) -> AlpacaNewsProvider` and the same `get_news`; raises `BrokerError("Failed to fetch news: ...")` on client failure.
  - `AlpacaBroker.news_provider()` returns its `AlpacaNewsProvider`, or `None` if built without a news client. `AlpacaBroker.get_news` keeps raising `BrokerError("no news client configured...")` in that case.

- [ ] **Step 1: Write the failing tests**

Create `tests/brokers/alpaca/test_alpaca_news.py`:

```python
from __future__ import annotations

import pytest
from tests.fakes import FakeNewsClient, et, make_alpaca_news_article

from trading_agent_framework.brokers.alpaca import news as news_module
from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.utils.errors import BrokerError

_NOW = et(2026, 9, 14, 10)


def test_get_news_builds_a_request_and_parses_the_response() -> None:
    client = FakeNewsClient()
    client.articles = [make_alpaca_news_article(headline="Rates cut")]
    provider = AlpacaNewsProvider(client)

    articles = provider.get_news(["SPY", "QQQ"], start=None, end=_NOW, limit=5, include_content=False)

    assert articles[0]["headline"] == "Rates cut"
    [request] = client.news_requests
    assert request.symbols == "SPY,QQQ"
    assert request.limit == 5


def test_get_news_wraps_client_failures_as_broker_error() -> None:
    client = FakeNewsClient()
    client.raises = RuntimeError("rate limited")

    with pytest.raises(BrokerError, match="Failed to fetch news"):
        AlpacaNewsProvider(client).get_news(end=_NOW)


def test_from_credentials_builds_the_client_from_the_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNewsClient()
    seen: list[AlpacaCredentials] = []

    def fake_build(creds: AlpacaCredentials) -> FakeNewsClient:
        seen.append(creds)
        return client

    monkeypatch.setattr(news_module, "build_news_client", fake_build)
    creds = AlpacaCredentials(api_key="k", api_secret="s")

    provider = AlpacaNewsProvider.from_credentials(creds)
    provider.get_news(end=_NOW)

    assert seen == [creds]
    assert len(client.news_requests) == 1
```

Append to `tests/brokers/test_base.py` (add `from tests.fakes import FakeBroker, FakeClock, et` to its imports if not already present):

```python
def test_news_provider_defaults_to_none() -> None:
    assert FakeBroker(FakeClock(et(2026, 9, 14, 10))).news_provider() is None
```

Append to `tests/brokers/alpaca/test_broker_market_data.py` (uses its existing `_broker_with_news`, `_NOW`, `AlpacaBroker`, `FakeTradingClient`, `FakeClock` imports):

```python
def test_news_provider_is_available_when_a_news_client_is_configured() -> None:
    broker = _broker_with_news(FakeNewsClient())

    assert broker.news_provider() is not None


def test_news_provider_is_none_without_a_news_client() -> None:
    broker = AlpacaBroker("momentum", FakeTradingClient(), clock=FakeClock(_NOW))

    assert broker.news_provider() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/brokers -q`
Expected: FAIL (`ModuleNotFoundError: ...brokers.alpaca.news`, `AttributeError: ... news_provider`).

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/brokers/news.py`:

```python
"""Broker-agnostic news seam: what `news_tools` needs from a broker. No `alpaca` import, no I/O."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol


class NewsProvider(Protocol):
    """A source of lean news articles (`id`, `headline`, `summary`, `source`, `created_at`, `symbols`, optional `content`)."""

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
    ) -> list[dict[str, object]]: ...
```

Create `src/trading_agent_framework/brokers/alpaca/news.py`:

```python
"""`AlpacaNewsProvider`: Alpaca's news feed behind the broker-agnostic `NewsProvider` seam.

Does the I/O (one `NewsClient.get_news` call) and wraps failures as `BrokerError`; all request
building and response parsing stays in the pure `market_data` module.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, cast

from trading_agent_framework.brokers.alpaca import market_data
from trading_agent_framework.brokers.alpaca.client import build_news_client
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import AlpacaCredentials


class AlpacaNewsProvider:
    def __init__(self, client: market_data.AlpacaNewsClient) -> None:
        self._client = client

    @classmethod
    def from_credentials(cls, creds: AlpacaCredentials) -> AlpacaNewsProvider:
        return cls(cast("market_data.AlpacaNewsClient", build_news_client(creds)))

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
    ) -> list[dict[str, object]]:
        request = market_data.build_news_request(symbols, start=start, end=end, limit=limit, include_content=include_content)
        try:
            response = self._client.get_news(request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch news: {exc}") from exc
        return market_data.parse_news(response)
```

In `src/trading_agent_framework/brokers/base.py`: add `from trading_agent_framework.brokers.news import NewsProvider` next to the other `brokers` imports, and add after `get_tracked_order`:

```python
    def news_provider(self) -> NewsProvider | None:
        """This broker's news source, or `None` when it has none (`news_tools` then reports an error)."""
        return None
```

In `src/trading_agent_framework/brokers/alpaca/broker.py`:
- add imports: `from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider` and `from trading_agent_framework.brokers.news import NewsProvider`;
- in `__init__`, replace `self._news_client = news_client` with
  `self._news_provider = AlpacaNewsProvider(news_client) if news_client is not None else None`;
- delete `_require_news_client` and replace `get_news` with:

```python
    def news_provider(self) -> NewsProvider | None:
        return self._news_provider

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
    ) -> list[dict[str, object]]:
        if self._news_provider is None:
            raise BrokerError(
                "no news client configured; construct the broker with news_client=... "
                "or use AlpacaBroker.from_credentials(...)"
            )
        return self._news_provider.get_news(symbols, start=start, end=end, limit=limit, include_content=include_content)
```

(`from_credentials` still passes `news_client=`; leave it unchanged. `build_news_client` stays imported in `broker.py` because `from_credentials` uses it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/brokers tests/agents -q && uv run ruff check`
Expected: PASS (existing `test_get_news_*` tests keep passing through the delegation), ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers/news.py src/trading_agent_framework/brokers/alpaca/news.py src/trading_agent_framework/brokers/base.py src/trading_agent_framework/brokers/alpaca/broker.py tests/brokers
git commit -m "Task 2: NewsProvider seam, Broker.news_provider(), AlpacaNewsProvider

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: news tool uses the provider seam; token caps

**Files:**
- Modify: `src/trading_agent_framework/brokers/alpaca/market_data.py` (`parse_news`)
- Modify: `src/trading_agent_framework/agents/tools/news.py`
- Modify: `tests/agents/tools/test_news_tools.py` (full replacement below)
- Modify: `tests/brokers/alpaca/test_market_data_parse.py`

**Interfaces:**
- Consumes: `Broker.news_provider()`, `NewsProvider.get_news` (Task 2).
- Produces: `market_data.MAX_NEWS_CONTENT_CHARS = 6000`, `market_data.TRUNCATION_MARKER = "... [truncated]"`; `news.MAX_CONTENT_LIMIT = 5`; `search_news` returns `{"error": "no news provider for this broker"}` when the broker has none, and `{"error": str(exc)}` when the lookup or fetch raises `BrokerError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/brokers/alpaca/test_market_data_parse.py` (add `from tests.fakes import make_alpaca_news_article` and extend the `market_data` import list with `MAX_NEWS_CONTENT_CHARS, TRUNCATION_MARKER` following the file's existing import style):

```python
def test_parse_news_truncates_oversized_content() -> None:
    news_set = NewsSet({"news": [make_alpaca_news_article(content="x" * 7000)], "next_page_token": None})

    [article] = parse_news(news_set)

    assert article["content"] == "x" * MAX_NEWS_CONTENT_CHARS + TRUNCATION_MARKER


def test_parse_news_leaves_short_content_untouched() -> None:
    news_set = NewsSet({"news": [make_alpaca_news_article(content="short body")], "next_page_token": None})

    [article] = parse_news(news_set)

    assert article["content"] == "short body"
```

Replace the whole of `tests/agents/tools/test_news_tools.py` with:

```python
from __future__ import annotations

from datetime import UTC, datetime

from tests.fakes import FakeBroker, FakeClock, FakeNewsClient, FakeTradingClient, et, make_alpaca_news_article

from trading_agent_framework.agents.tools.news import MAX_CONTENT_LIMIT, news_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import BrokerError


def _now() -> datetime:
    return et(2026, 9, 14, 10)


def _alpaca_strategy(news: FakeNewsClient) -> Strategy:
    broker = AlpacaBroker("momentum", FakeTradingClient(), clock=FakeClock(_now()), news_client=news)
    return Strategy(broker)


def _tool(strategy: Strategy):
    [tool] = news_tools(strategy)
    return tool


class _StubProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):
        self.calls.append({"symbols": list(symbols), "start": start, "end": end, "limit": limit, "include_content": include_content})
        return [{"id": 1, "headline": "Stub headline"}]


def _provider_strategy(provider: object = None, *, lookup_error: BrokerError | None = None) -> Strategy:
    class _Broker(FakeBroker):
        def news_provider(self):
            if lookup_error is not None:
                raise lookup_error
            return provider

    return Strategy(_Broker(FakeClock(_now())))


def test_search_news_returns_articles_from_the_broker() -> None:
    news = FakeNewsClient()
    news.articles = [make_alpaca_news_article(headline="Rates cut")]
    tool = _tool(_alpaca_strategy(news))

    result = tool(symbols="SPY,QQQ")

    assert result["count"] == 1
    assert result["articles"][0]["headline"] == "Rates cut"
    [request] = news.news_requests
    assert request.symbols == "SPY,QQQ"


def test_search_news_works_with_any_broker_that_provides_news() -> None:
    provider = _StubProvider()
    tool = _tool(_provider_strategy(provider))

    result = tool(symbols="SPY")

    assert result == {"count": 1, "articles": [{"id": 1, "headline": "Stub headline"}]}
    assert provider.calls[0]["symbols"] == ["SPY"]
    assert provider.calls[0]["end"] == _now()


def test_search_news_defaults_end_to_the_strategy_clock() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    tool()

    [request] = news.news_requests
    # alpaca-py normalises start/end to naive UTC (same as test_market_data_requests.py's
    # test_bars_request_asks_for_adjusted_iex_bars_of_every_symbol)
    assert request.end == _now().astimezone(UTC).replace(tzinfo=None)


def test_search_news_clamps_end_that_is_after_the_clock() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    tool(end=et(2026, 9, 20).isoformat())

    [request] = news.news_requests
    assert request.end == _now().astimezone(UTC).replace(tzinfo=None)


def test_search_news_clamps_limit_to_the_content_cap_when_content_is_requested() -> None:
    provider = _StubProvider()
    tool = _tool(_provider_strategy(provider))

    tool(limit=30, include_content=True)
    tool(limit=30, include_content=False)

    assert provider.calls[0]["limit"] == MAX_CONTENT_LIMIT
    assert provider.calls[1]["limit"] == 30


def test_search_news_returns_an_error_dict_on_broker_failure() -> None:
    news = FakeNewsClient()
    news.raises = RuntimeError("boom")
    tool = _tool(_alpaca_strategy(news))

    result = tool()

    assert "error" in result


def test_search_news_without_a_news_provider_returns_an_error() -> None:
    strategy = Strategy(FakeBroker(FakeClock(_now())))

    assert _tool(strategy)() == {"error": "no news provider for this broker"}


def test_search_news_reports_a_failing_provider_lookup_as_an_error() -> None:
    strategy = _provider_strategy(lookup_error=BrokerError("no news source configured"))

    assert _tool(strategy)() == {"error": "no news source configured"}


def test_search_news_returns_an_error_on_naive_datetime_end() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    # Naive datetime strings (no offset) should return error, not raise TypeError
    result = tool(end="2026-09-20T00:00:00")

    assert "error" in result


def test_search_news_returns_an_error_on_naive_datetime_start() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    # Naive datetime strings (no offset) should return error, not raise TypeError
    result = tool(start="2026-09-10T00:00:00")

    assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_news_tools.py tests/brokers/alpaca/test_market_data_parse.py -q`
Expected: FAIL (`ImportError: MAX_CONTENT_LIMIT`, `MAX_NEWS_CONTENT_CHARS`).

- [ ] **Step 3: Implement**

In `src/trading_agent_framework/brokers/alpaca/market_data.py`, next to `MAX_NEWS_LIMIT = 50` add:

```python
MAX_NEWS_CONTENT_CHARS = 6000
TRUNCATION_MARKER = "... [truncated]"
```

add this helper just above `parse_news`:

```python
def _truncate_content(content: str) -> str:
    """Cap a news article body so a full-content read stays within the LLM token budget."""
    if len(content) <= MAX_NEWS_CONTENT_CHARS:
        return content
    return content[:MAX_NEWS_CONTENT_CHARS] + TRUNCATION_MARKER
```

and change `item["content"] = content` in `parse_news` to `item["content"] = _truncate_content(str(content))`.

Replace `src/trading_agent_framework/agents/tools/news.py` with:

```python
"""Plain typed news tool for a LangChain agent: news search through the broker's `NewsProvider`, gated on the strategy clock."""

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MIN_LIMIT = 1
MAX_LIMIT = 50
MAX_CONTENT_LIMIT = 5  # full articles are large: cap how many one call may return
_DEFAULT_LOOKBACK_DAYS = 7


def news_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """News search tool bound to `strategy`."""

    def search_news(
        symbols: str = "",
        start: str | None = None,
        end: str | None = None,
        limit: int = 10,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Search recent news headlines and summaries, optionally filtered to symbols."""
        try:
            provider = strategy.broker.news_provider()
        except BrokerError as exc:
            return {"error": str(exc)}
        if provider is None:
            return {"error": "no news provider for this broker"}
        try:
            now = strategy.clock.now()
            if end:
                end_parsed = datetime.fromisoformat(end)
                if end_parsed.tzinfo is None:
                    return {"error": "end date must include timezone information"}
                end_dt = min(end_parsed, now)
            else:
                end_dt = now
            if start:
                start_parsed = datetime.fromisoformat(start)
                if start_parsed.tzinfo is None:
                    return {"error": "start date must include timezone information"}
                start_dt = start_parsed
            else:
                start_dt = end_dt - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
        except ValueError as exc:
            return {"error": f"invalid date format: {exc}"}
        max_limit = MAX_CONTENT_LIMIT if include_content else MAX_LIMIT
        clamped_limit = min(max(int(limit), MIN_LIMIT), max_limit)
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        try:
            articles = provider.get_news(symbol_list, start=start_dt, end=end_dt, limit=clamped_limit, include_content=include_content)
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"count": len(articles), "articles": articles}

    return [search_news]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests -q -x && uv run ruff check`
Expected: PASS (full suite: nothing else depended on the removed `AlpacaBroker` check), ruff clean. If `tests/agents/tools/test_agents_tools_lazy_imports.py` fails, read it and adjust only if it asserted the old import (news no longer imports alpaca at all).

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers/alpaca/market_data.py src/trading_agent_framework/agents/tools/news.py tests/agents/tools/test_news_tools.py tests/brokers/alpaca/test_market_data_parse.py
git commit -m "Task 3: search_news goes through Broker.news_provider(); cap article payloads

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `news_source=` plumbing for backtests

**Files:**
- Modify: `src/trading_agent_framework/backtesting/broker.py`
- Modify: `src/trading_agent_framework/backtesting/runner.py`
- Modify: `src/trading_agent_framework/core/strategy.py`
- Create: `tests/backtesting/test_broker_news.py`
- Modify: `tests/core/test_runners.py`

**Interfaces:**
- Consumes: `NewsProvider` (Task 2), `AlpacaNewsProvider.from_credentials`, `AlpacaCredentials.from_env`.
- Produces:
  - `BacktestBroker(..., news_source: NewsProvider | None = None)`; `BacktestBroker.news_provider()` returns the injected source, else lazily builds and memoizes `AlpacaNewsProvider.from_credentials(AlpacaCredentials.from_env())`; a `ConfigurationError` (no Alpaca creds) becomes `BrokerError("no news source available for backtesting: ...")`.
  - `run_backtest(..., news_source: NewsProvider | None = None)` and `Strategy.run_backtesting(..., news_source: NewsProvider | None = None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/test_broker_news.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import et

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
from trading_agent_framework.utils.errors import BrokerError


class _Source:
    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):
        return []


def _broker(**kwargs: object) -> BacktestBroker:
    clock = BacktestClock(start=et(2026, 1, 5, 10), sessions=[])
    return BacktestBroker("news", data_source=FakeBacktestDataSource(), clock=clock, budget=Decimal("10000"), **kwargs)


def test_an_injected_news_source_is_returned_as_is() -> None:
    source = _Source()

    assert _broker(news_source=source).news_provider() is source


def test_the_default_news_source_is_built_lazily_from_env_credentials_and_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    sentinel = _Source()
    built: list[object] = []

    def fake_from_credentials(cls: type, creds: object) -> _Source:
        built.append(creds)
        return sentinel

    monkeypatch.setattr(AlpacaNewsProvider, "from_credentials", classmethod(fake_from_credentials))
    broker = _broker()

    assert built == []  # nothing is constructed until the news tool asks
    assert broker.news_provider() is sentinel
    assert broker.news_provider() is sentinel
    assert len(built) == 1


def test_missing_alpaca_credentials_surface_as_a_broker_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    with pytest.raises(BrokerError, match="no news source available for backtesting"):
        _broker().news_provider()
```

In `tests/core/test_runners.py`, in `test_run_backtesting_runs_end_to_end_via_the_public_api`: add a stub source before the `strategy = _strategy(...)` line and pass it, then assert it landed on the rebound broker:

```python
    class _NewsSource:
        def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):
            return []

    news = _NewsSource()
```

change the call to `strategy.run_backtesting(start=..., end=..., data_source=source, benchmark="SPY", news_source=news,)` (keep the existing arguments) and add after the `isinstance(strategy.broker, BacktestBroker)` assertion:

```python
    assert strategy.broker.news_provider() is news
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtesting/test_broker_news.py tests/core/test_runners.py -q`
Expected: FAIL (`TypeError: unexpected keyword argument 'news_source'`).

- [ ] **Step 3: Implement**

`src/trading_agent_framework/backtesting/broker.py`:
- add `from trading_agent_framework.brokers.news import NewsProvider` with the other `brokers` imports; add `ConfigurationError` to the existing `trading_agent_framework.utils.errors` import and `BrokerError` too if not present;
- add constructor parameter `news_source: NewsProvider | None = None` after `tracker` and `self._news_source = news_source` in `__init__`;
- add this method after `pull_positions`:

```python
    def news_provider(self) -> NewsProvider | None:
        """The injected news source, else an Alpaca provider built from the env credentials on first use."""
        if self._news_source is None:
            # Deferred: importing this module must not pull in `alpaca`, and a backtest that never
            # calls the news tool needs no Alpaca credentials for news.
            from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
            from trading_agent_framework.config.env import AlpacaCredentials

            try:
                self._news_source = AlpacaNewsProvider.from_credentials(AlpacaCredentials.from_env())
            except ConfigurationError as exc:
                raise BrokerError(f"no news source available for backtesting: {exc}") from exc
        return self._news_source
```

`src/trading_agent_framework/backtesting/runner.py`: in the `TYPE_CHECKING` block add `from trading_agent_framework.brokers.news import NewsProvider`; add `news_source: NewsProvider | None = None,` as the last parameter of both `run_backtest(...)` and `_run(...)`; in `run_backtest`'s `_run(...)` call add `news_source=news_source,`; in `_run`, add `news_source=news_source,` to the `BacktestBroker(...)` constructor call. Add one docstring line to `run_backtest`: "`news_source` is handed to the `BacktestBroker` (default: an Alpaca provider built lazily from the env credentials)."

`src/trading_agent_framework/core/strategy.py`: in the `TYPE_CHECKING` block add `from trading_agent_framework.brokers.news import NewsProvider`; add `news_source: NewsProvider | None = None,` after `warmup_trading_days: int = 0,` in `run_backtesting`; add to its Args docstring:

```
            news_source: where the news tool gets historical news (a `NewsProvider`). Defaults to an
                Alpaca provider built lazily from `AlpacaCredentials.from_env()`; the tool's own
                `strategy.clock.now()` cutoff still applies, so no future article leaks.
```

and pass `news_source=news_source,` in the `run_backtest(...)` call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests -q && uv run ruff check`
Expected: PASS, including `tests/backtesting/test_lazy_imports.py` (backtesting must still not import alpaca at import time).

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/backtesting/broker.py src/trading_agent_framework/backtesting/runner.py src/trading_agent_framework/core/strategy.py tests/backtesting/test_broker_news.py tests/core/test_runners.py
git commit -m "Task 4: news_source= plumbing (BacktestBroker, run_backtest, Strategy.run_backtesting)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: `NewsBuiltinStrategy`

**Files:**
- Create: `src/trading_agent_framework/strategies/news_builtin/__init__.py`
- Create: `src/trading_agent_framework/strategies/news_builtin/prompts.py`
- Create: `src/trading_agent_framework/strategies/news_builtin/agent_news_builtin.py`
- Delete: `src/trading_agent_framework/strategies/agent_alpaca_news_builtin.py` (untracked; use `rm`)
- Create: `tests/strategies/test_news_builtin.py`

**Interfaces:**
- Consumes: `PrebuiltTools.all(strategy)`, `news_tools(strategy)`, `Strategy.run_backtesting(...)`, `AgentRunResult.output`, `AgentError`, `ConfigurationError`.
- Produces: `NewsBuiltinStrategy(Strategy)` (constructor `NewsBuiltinStrategy(broker, *, mode, ...)`, agent name `AGENT_NAME = "news_trader"`); `prompts.build_system_prompt(*, symbols, defensive_symbol, news_symbols) -> str`, `prompts.TASK_PROMPT: str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_news_builtin.py`:

```python
from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from tests.fakes import FakeBroker, FakeClock, FakeToolCallingChatModel, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.news_builtin import NewsBuiltinStrategy
from trading_agent_framework.strategies.news_builtin.agent_news_builtin import AGENT_NAME
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, ConfigurationError

_START = et(2026, 9, 14, 9, 0)


class _FakeHandle:
    def __init__(self) -> None:
        self.runs: list[tuple[str, object]] = []
        self.error: Exception | None = None

    def run(self, task_prompt: str, *, context: object = None) -> AgentRunResult:
        self.runs.append((task_prompt, context))
        if self.error is not None:
            raise self.error
        return AgentRunResult(output="Hold SHV.", tool_calls=[])


class _FakeAgents:
    def __init__(self, handle: _FakeHandle) -> None:
        self.handle = handle
        self.created: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeHandle:
        self.created.append(kwargs)
        return self.handle

    def __getitem__(self, name: str) -> _FakeHandle:
        return self.handle


def _strategy(tmp_path: Path, mode: TradingMode) -> tuple[NewsBuiltinStrategy, _FakeAgents, _FakeHandle]:
    strategy = NewsBuiltinStrategy(FakeBroker(FakeClock(_START), strategy_name="news_builtin"), mode=mode, project_root=tmp_path)
    handle = _FakeHandle()
    agents = _FakeAgents(handle)
    strategy._agents = agents  # ty: ignore[invalid-assignment]
    return strategy, agents, handle


def test_backtest_defaults_match_the_original_strategy() -> None:
    assert NewsBuiltinStrategy.backtesting_start == datetime(2025, 1, 1, tzinfo=MARKET_TZ)
    assert NewsBuiltinStrategy.backtesting_end == datetime(2026, 4, 1, tzinfo=MARKET_TZ)
    assert NewsBuiltinStrategy.budget == Decimal("10000")
    assert NewsBuiltinStrategy.benchmark_symbol == "SPY"


@pytest.mark.parametrize(("mode", "expected"), [(TradingMode.BACKTESTING, "1D"), (TradingMode.PAPER, "2H"), (TradingMode.LIVE, "2H")])
def test_initialize_sets_sleeptime_per_mode(tmp_path: Path, mode: TradingMode, expected: str) -> None:
    strategy, _, _ = _strategy(tmp_path, mode)

    strategy.initialize()

    assert strategy.sleeptime == expected


def test_initialize_creates_the_agent_with_prebuilt_memory_and_news_tools(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)

    strategy.initialize()

    [created] = agents.created
    names = {tool.__name__ for tool in created["tools"]}  # ty: ignore[unresolved-attribute]
    assert created["name"] == AGENT_NAME
    assert {"search_news", "remember_decision", "search_memory", "submit_order", "get_positions", "get_bars", "get_last_price"} <= names
    assert "SHV" in created["system_prompt"]  # ty: ignore[unsupported-operator]


def test_backtest_runs_the_agent_on_the_first_and_every_fifth_iteration(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
    strategy.initialize()

    for _ in range(10):
        strategy.on_trading_iteration()

    assert len(handle.runs) == 3  # iterations 1, 5 and 10


def test_paper_runs_the_agent_on_every_iteration(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()

    for _ in range(3):
        strategy.on_trading_iteration()

    assert len(handle.runs) == 3
    _, context = handle.runs[0]
    assert context == {"current_datetime": _START.isoformat()}


def test_an_agent_error_is_logged_and_does_not_stop_the_run(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.error = AgentError("llm exploded")

    with caplog.at_level(logging.ERROR):
        strategy.on_trading_iteration()

    assert "llm exploded" in caplog.text


def test_a_configuration_error_propagates(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.error = ConfigurationError("no model")

    with pytest.raises(ConfigurationError):
        strategy.on_trading_iteration()


def test_a_real_agent_builds_with_every_tool_and_logs_its_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    model = FakeToolCallingChatModel(messages=iter([AIMessage("Hold SHV.")]))
    monkeypatch.setattr(AgentManager, "_resolve_model", lambda self, model_arg, timeout: model)
    strategy = NewsBuiltinStrategy(FakeBroker(FakeClock(_START), strategy_name="news_builtin"), mode=TradingMode.PAPER, project_root=tmp_path)

    with caplog.at_level(logging.INFO):
        strategy.initialize()
        strategy.on_trading_iteration()

    assert AGENT_NAME in strategy.agents
    assert "Hold SHV." in caplog.text


def test_run_backtesting_wires_yahoo_data_preload_and_fees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: captured.update(kwargs))
    strategy, _, _ = _strategy(tmp_path, TradingMode.BACKTESTING)

    strategy.run_backtesting()

    assert captured["data_source"] is YahooBacktestData
    assert [asset.symbol for asset in captured["preload_assets"]] == ["SPY", "QQQ", "SHV"]  # ty: ignore[not-iterable]
    assert captured["commission"] == Decimal("0.001")
    assert captured["warmup_trading_days"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_news_builtin.py -q`
Expected: FAIL (`ModuleNotFoundError: ...strategies.news_builtin`).

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/strategies/news_builtin/prompts.py`:

```python
"""Prompts for the news-builtin agent (v1). Built from the strategy parameters so the instrument list has one source of truth."""

from __future__ import annotations

from collections.abc import Sequence


def build_system_prompt(*, symbols: Sequence[str], defensive_symbol: str, news_symbols: str) -> str:
    allowed = ", ".join([*symbols, defensive_symbol])
    return (
        f"You are a news-driven allocator. Your only allowed instruments are {allowed} "
        f"({defensive_symbol} is the defensive ETF). Never short, never use margin, never trade anything else, "
        "never trade USD or FOREX.\n\n"
        "On every run, follow this workflow:\n"
        "1. Call search_memory to recall recent decisions and the current regime thesis.\n"
        f"2. Scan broad-market news: call search_news with symbols='{news_symbols}', include_content=False and limit=30.\n"
        "3. Pick the single most relevant article. Call search_news again with a narrow start/end window around that "
        "article's created_at (ISO 8601 with timezone), include_content=True and limit=3, to read it in full.\n"
        "4. Compare article timestamps with the current datetime given in the task and ignore stale news.\n"
        f"5. Decide the regime: bullish evidence means hold {' or '.join(symbols)}; negative or unclear evidence means hold {defensive_symbol}.\n"
        "6. Check get_positions and get_orders, then trade only the difference. Sell the current position before buying "
        "the other. Quantity = floor(cash / last price). Keep position sizing reasonable and never place a duplicate order.\n"
        "7. Record the outcome: call remember_decision after every decision, and open_thesis or close_thesis when the "
        "regime call changes."
    )


TASK_PROMPT = "Research current broad-market news and rebalance if needed. The current datetime is in the context below."
```

Create `src/trading_agent_framework/strategies/news_builtin/agent_news_builtin.py`:

```python
"""News-builtin strategy: an LLM agent reads broad-market news and holds SPY/QQQ (bullish) or a defensive ETF.

Port of the lumibot `agent_alpaca_news_builtin` strategy onto this framework: `PrebuiltTools` (trading,
account, market data, indicators, memory) plus the `search_news` tool. News is gated on the strategy clock,
so it works in backtests as well as paper/live.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from trading_agent_framework.agents.tools import PrebuiltTools
from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.news_builtin.prompts import TASK_PROMPT, build_system_prompt
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError

AGENT_NAME = "news_trader"


class NewsBuiltinStrategy(Strategy):
    backtesting_start = datetime(2025, 1, 1, tzinfo=MARKET_TZ)
    backtesting_end = datetime(2026, 4, 1, tzinfo=MARKET_TZ)
    budget = Decimal("10000")
    benchmark_symbol = "SPY"

    parameters = {
        "symbols": ("SPY", "QQQ"),
        "defensive_symbol": "SHV",
        "news_symbols": "SPY,QQQ,DIA,IWM",
        "backtest_every_n_iterations": 5,
        "warmup_trading_days": 0,
    }

    def initialize(self) -> None:
        self.sleeptime = "1D" if self.is_backtesting else "2H"
        self.vars.iteration_count = 0
        self.agents.create(
            name=AGENT_NAME,
            system_prompt=build_system_prompt(
                symbols=self.parameters["symbols"],
                defensive_symbol=self.parameters["defensive_symbol"],
                news_symbols=self.parameters["news_symbols"],
            ),
            tools=[*PrebuiltTools.all(self), *news_tools(self)],
        )
        self.log_info(f"NewsBuiltinStrategy initialized (sleeptime={self.sleeptime})")

    def on_trading_iteration(self) -> None:
        self.vars.iteration_count += 1
        if self.is_backtesting and not self._backtest_iteration_is_due():
            return
        try:
            result = self.agents[AGENT_NAME].run(TASK_PROMPT, context={"current_datetime": self.get_datetime().isoformat()})
        except AgentError as exc:
            # A failed LLM call must not kill a multi-week backtest or a live loop; ConfigurationError still propagates.
            self.log_error(f"[{AGENT_NAME}] run failed: {exc}")
            return
        self.log_info(f"[{AGENT_NAME}] {result.output}")

    def _backtest_iteration_is_due(self) -> bool:
        count = self.vars.iteration_count
        return count == 1 or count % self.parameters["backtest_every_n_iterations"] == 0

    def run_backtesting(self):
        symbols = [*self.parameters["symbols"], self.parameters["defensive_symbol"]]
        return super().run_backtesting(
            data_source=YahooBacktestData,
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],
            commission=Decimal("0.001"),
            warmup_trading_days=self.parameters["warmup_trading_days"],
        )
```

Create `src/trading_agent_framework/strategies/news_builtin/__init__.py`:

```python
from .agent_news_builtin import NewsBuiltinStrategy

__all__ = [
    "NewsBuiltinStrategy",
]
```

Delete the pasted lumibot file: `rm src/trading_agent_framework/strategies/agent_alpaca_news_builtin.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/strategies tests -q && uv run ruff check`
Expected: PASS, ruff clean. If the `caplog` assertions fail because logging propagation is off, inspect `tests/conftest.py` (it resets logging per test) and assert on `strategy._log`'s logger name instead of loosening the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/news_builtin tests/strategies/test_news_builtin.py
git commit -m "Task 5: NewsBuiltinStrategy (LLM agent with prebuilt, memory and news tools)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

(The deleted file was untracked, so it needs no `git rm`.)

---

### Task 6: `main.py` builder registry

**Files:**
- Modify: `src/trading_agent_framework/main.py`
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: `NewsBuiltinStrategy`, `CrossMomentumStrategy`, `load_cross_momentum_universe`.
- Produces: `AGENT_STRATEGIES: dict[str, StrategyBuilder]` where `StrategyBuilder = Callable[[AlpacaBroker, TradingMode], Strategy | None]`; `_build_cross_momentum`, `_build_news_builtin`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main.py`:

```python
from __future__ import annotations

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework import main as main_module
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.news_builtin import NewsBuiltinStrategy


def test_registry_lists_both_strategies() -> None:
    assert set(main_module.AGENT_STRATEGIES) == {"cross_momentum", "news_builtin"}


def test_news_builtin_builder_returns_the_strategy() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="news_builtin")

    strategy = main_module._build_news_builtin(broker, TradingMode.BACKTESTING)  # ty: ignore[invalid-argument-type]

    assert isinstance(strategy, NewsBuiltinStrategy)
    assert strategy.is_backtesting


def test_cross_momentum_builder_returns_none_without_a_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "load_cross_momentum_universe", lambda: [])
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="cross_momentum")

    assert main_module._build_cross_momentum(broker, TradingMode.BACKTESTING) is None  # ty: ignore[invalid-argument-type]


def test_an_unknown_strategy_name_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["agent", "nope", "backtesting"])

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_main.py -q`
Expected: FAIL (`AttributeError: ... '_build_news_builtin'`, registry keys mismatch).

- [ ] **Step 3: Implement**

In `src/trading_agent_framework/main.py`:

1. Add imports: `from collections.abc import Callable`, `from trading_agent_framework.core import Strategy`, `from trading_agent_framework.strategies.news_builtin import NewsBuiltinStrategy`.
2. Replace the `AGENT_STRATEGIES` block with:

```python
StrategyBuilder = Callable[[AlpacaBroker, TradingMode], Strategy | None]


def _build_cross_momentum(broker: AlpacaBroker, mode: TradingMode) -> Strategy | None:
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run batch_stock_universe.py before executing this strategy.", style="bold red")
        return None
    return CrossMomentumStrategy(broker=broker, mode=mode, universe=universe)


def _build_news_builtin(broker: AlpacaBroker, mode: TradingMode) -> Strategy | None:
    return NewsBuiltinStrategy(broker=broker, mode=mode)


# MAPPING OF STRATEGY NAMES TO STRATEGY BUILDERS
AGENT_STRATEGIES: dict[str, StrategyBuilder] = {
    "cross_momentum": _build_cross_momentum,
    "news_builtin": _build_news_builtin,
}
```

3. In `_run_strategy`, replace `strategy_class = AGENT_STRATEGIES[strategy_name]` with `builder = AGENT_STRATEGIES[strategy_name]`, and replace everything from `# TODO: change strategy init argument order...` through the `else:` branch that prints "Universe file not found" (the `universe = ...` / `if universe:` / `else:` block) with:

```python
    strategy = builder(broker, trading_mode)
    if strategy is None:
        return
```

keeping the trailing `# run strategy according to the trading mode` / `strategy.run_strategy()` lines.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests -q && uv run ruff check`
Expected: PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/main.py tests/test_main.py
git commit -m "Task 6: main.py strategy builder registry; register news_builtin

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Smoke script, docs, and end-to-end verification

**Files:**
- Create: `scripts/tests/smoke_strategy_news.py`
- Modify: `scripts/tests/HOWTO.md`
- Modify: `CLAUDE.md`
- Modify: `README.md` (the `news_builtin` entry)

**Interfaces:**
- Consumes: everything above; `load_strategy_env(strategy_name, mode_value, project_root)`.

- [ ] **Step 1: Write the smoke script**

Create `scripts/tests/smoke_strategy_news.py`:

```python
#!/usr/bin/env python3
"""Manual end-to-end check of the news_builtin strategy against a real LLM server and Alpaca news.

NOT part of the automated test suite. Run it by hand:

    uv run python scripts/tests/smoke_strategy_news.py iteration   # one agent iteration (paper account only)
    uv run python scripts/tests/smoke_strategy_news.py backtest    # a 2-week real backtest

LLM_BASE_URL / LLM_MODEL come from env/.env.news_builtin.backtesting. `iteration` takes its Alpaca
credentials from env/.env.alpaca.integration-tests (paper) and refuses to run against a live account,
since the agent may place paper orders. `backtest` only reads news and market data.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config import load_strategy_env
from trading_agent_framework.config.env import AlpacaCredentials, TradingMode
from trading_agent_framework.strategies.news_builtin import NewsBuiltinStrategy
from trading_agent_framework.utils.clock import MARKET_TZ

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"
STRATEGY_NAME = "news_builtin"


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _iteration() -> None:
    if not PAPER_ENV_FILE.is_file():
        raise SmokeTestFailure(f"Paper credentials file not found: {PAPER_ENV_FILE}")
    load_dotenv(PAPER_ENV_FILE, override=True)
    # LLM settings only: override=False keeps the paper Alpaca credentials loaded above.
    load_dotenv(PROJECT_ROOT / "env" / ".env.news_builtin.backtesting", override=False)
    creds = AlpacaCredentials.from_env()
    if not creds.is_paper:
        raise SmokeTestFailure("Refusing to run one agent iteration against a live account.")
    strategy = NewsBuiltinStrategy(AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=False), mode=TradingMode.PAPER)
    strategy.initialize()
    strategy.on_trading_iteration()
    print("PASS: one agent iteration completed (see the log line above for the agent's answer).")


def _backtest() -> None:
    load_strategy_env(STRATEGY_NAME, TradingMode.BACKTESTING.value, PROJECT_ROOT)
    creds = AlpacaCredentials.from_env()
    strategy = NewsBuiltinStrategy(AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=False), mode=TradingMode.BACKTESTING)
    strategy.backtesting_start = datetime(2025, 3, 3, tzinfo=MARKET_TZ)
    strategy.backtesting_end = datetime(2025, 3, 14, tzinfo=MARKET_TZ)
    result = strategy.run_backtesting()
    for name in ("metrics.json", "settings.json", "equity.parquet"):
        if not (result.run_dir / name).is_file():
            raise SmokeTestFailure(f"missing {name} in {result.run_dir}")
    print(f"PASS: backtest report written to {result.run_dir}")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"iteration", "backtest"}:
        print(__doc__)
        return 2
    {"iteration": _iteration, "backtest": _backtest}[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
```

- [ ] **Step 2: Lint and dry-import the script**

Run: `uv run ruff check scripts/tests/smoke_strategy_news.py && uv run python -c "import ast,sys; ast.parse(open('scripts/tests/smoke_strategy_news.py').read())"`
Expected: clean.

- [ ] **Step 3: Document it in `scripts/tests/HOWTO.md`**

Append a section (same style as the others):

```markdown
## Smoke Strategy News
Manual end-to-end check of the `news_builtin` LLM strategy against a real vLLM server and Alpaca news.

NOT part of the automated test suite. Run by hand:

    uv run python scripts/tests/smoke_strategy_news.py iteration   # one agent iteration, paper account only
    uv run python scripts/tests/smoke_strategy_news.py backtest    # a 2-week real backtest (2025-03-03..2025-03-14)

`LLM_BASE_URL` / `LLM_MODEL` come from env/.env.news_builtin.backtesting (the LLM server must be up). `iteration`
loads paper credentials from env/.env.alpaca.integration-tests and refuses to run against a live account, since the
agent may place paper orders. `backtest` only reads news/market data and writes a report under
logs/news_builtin/backtesting/. Expect several LLM calls (iterations 1, 5 and 10 of the backtest).
```

- [ ] **Step 4: Update `CLAUDE.md` and `README.md`**

`CLAUDE.md` (use the Edit tool on the exact existing text):
1. In the "Research agent tools (news/macro/fundamentals)" bullet, replace the sentence beginning "**This clock-gating is uniform across all three, but backtest capability is not**" through "...not a backtest-capable tool that merely happens to gate on the clock like its siblings." with:
   "**All three work in every mode.** `macro_tools`/`fundamentals_tools` call FRED/SEC directly. `news_tools` goes through the broker-agnostic `NewsProvider` seam (`brokers/news.py`): `Broker.news_provider()` returns `None` by default; `AlpacaBroker` returns its `AlpacaNewsProvider` (`brokers/alpaca/news.py`); `BacktestBroker(news_source=...)` takes an injected provider, defaulting lazily to an Alpaca provider built from `AlpacaCredentials.from_env()` (`Strategy.run_backtesting(news_source=...)` mirrors `data_source=`). A future broker (e.g. IBKR) implements `news_provider()` itself. The clock gate stays in the tool, never in a provider. Article payloads are token-capped: `parse_news` truncates content at 6000 chars and `search_news` clamps `limit` to 5 when `include_content=True`."
2. In the Architecture list, add after the `agents/tools/` bullet: "- `strategies/` -- concrete strategies. `cross_momentum/` (no LLM) and `news_builtin/` (`NewsBuiltinStrategy`: an LLM agent built from `PrebuiltTools.all(self)` + `news_tools(self)`; prompts in `prompts.py`). `main.py`'s `AGENT_STRATEGIES` maps a strategy name to a builder `(broker, mode) -> Strategy | None`."
3. In the `agents/` architecture bullet, extend the `config.py` description: "`LLMCredentials.from_env` (`LLM_BASE_URL` is normalized -- a pasted `.../v1/chat/completions` becomes `.../v1`; `LLM_API_KEY` is optional and defaults to a placeholder)".

`README.md`: in the `news_builtin` entry (around the "#### 📈 `news_builtin`" heading) update: **File** to `strategies/news_builtin/agent_news_builtin.py`; **Tools** to "`PrebuiltTools.all(strategy)` + `news_tools(strategy)` (`search_news`)"; the run line to `uv run python -m trading_agent_framework.main news_builtin backtesting`; and add one sentence: "Needs `LLM_BASE_URL` and `LLM_MODEL` (OpenAI-compatible server, e.g. vLLM) plus Alpaca credentials in `env/.env.news_builtin.<mode>`; defensive ETF is `SHV`."

- [ ] **Step 5: Full verification**

Run: `uv run pytest -q && uv run ruff check`
Expected: all tests pass, ruff clean. Paste the summary line in your report.

- [ ] **Step 6: Manual end-to-end run (only if the LLM server is reachable)**

Check first: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8005/v1/models` (expect `200`). If it is not reachable, skip this step and say so explicitly in the report — do not claim end-to-end verification.

If reachable: `uv run python scripts/tests/smoke_strategy_news.py backtest`. Expected: `PASS: backtest report written to ...`. Then read the run's `logs/news_builtin/backtesting/*/backtesting.log` and check that (a) the agent called `search_news` with results (not `{"error": ...}`), (b) `remember_decision` ran, (c) no article was dated after the simulated date. Report anything odd (looping tool calls, empty answers, `<think>` tags in output) and note prompt tweaks needed in `prompts.py` rather than changing the structure.

- [ ] **Step 7: Commit**

```bash
git add scripts/tests/smoke_strategy_news.py scripts/tests/HOWTO.md CLAUDE.md README.md
git commit -m "Task 7: smoke script, docs for the news provider seam and news_builtin strategy

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** §3.1 LLM config → Task 1; §3.2 news seam + BacktestBroker/run_backtest/Strategy plumbing → Tasks 2 and 4; §3.3 tool + token caps → Task 3; §3.4 CLAUDE.md → Task 7; §4 strategy + prompts → Task 5; §5 main.py → Task 6; §6 tests → per task, manual smoke → Task 7; §7 file list covered; §8 risks are surfaced in Task 7 Step 6.
- **Deviations from the spec (deliberate):** (1) `BacktestBroker.news_provider()` converts a missing-credentials `ConfigurationError` into `BrokerError` and `search_news` catches a `BrokerError` from the lookup, so a missing Alpaca key gives the agent a soft error instead of crashing the run (spec was silent). (2) The strategy sets `backtesting_start/end/budget/benchmark_symbol` as class attributes per spec §4.1, unlike `CrossMomentumStrategy` which keeps them in `parameters`.
- **Type consistency:** `NewsProvider.get_news` signature identical in `brokers/news.py`, `AlpacaNewsProvider`, `_StubProvider`, `_NewsSource`. `MAX_CONTENT_LIMIT` (tool) and `MAX_NEWS_CONTENT_CHARS`/`TRUNCATION_MARKER` (`market_data`) are named identically wherever tests import them. Agent name constant `AGENT_NAME = "news_trader"`.
