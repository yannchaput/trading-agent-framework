# Tastytrade Broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Tastytrade a third paper/live broker (`BROKER=tastytrade`) next to Alpaca and IBKR, over its REST API and account streamer, with market data, calendar and news from Alpaca, a $0-commission backtest fee schedule, a refusal to run strategies against the sandbox, and manual sandbox scripts.

**Architecture:** A new package `brokers/tastytrade/`: pure translation modules (`values.py`, `account.py`, `orders.py`), one HTTP module (`client.py`, `TastytradeSession`: OAuth token refresh, retries, pagination), one websocket module (`stream.py`, `TastytradeAccountStream` thread), an event applier feeding `OrderTracker` (`events.py`), and the wiring (`broker.py`, `TastytradeBroker`), which delegates market data, clock and news to the existing `AlpacaMarketData`/`AlpacaMarketClock`/`lazy_news_provider` exactly as `IbkrBroker` does. Config, fees, the factory and `main.py` get small additions.

**Tech Stack:** Python 3.14, `uv`, `pytest`, `ruff`, `httpx` 0.28 (already a direct dependency), `websockets` 16.1 (already installed through alpaca-py; made direct in Task 8), `python-dotenv`.

**Spec:** `docs/superpowers/specs/2026-10-15-tastytrade-broker-design.md`

## Global Constraints

- Env variable names are exactly: `BROKER`, `BROKER_API_IS_PAPER`, `TASTYTRADE_CLIENT_SECRET`, `TASTYTRADE_REFRESH_TOKEN`, `TASTYTRADE_ACCOUNT_NUMBER` (all three required, blank = missing), plus the existing `ALPACA_DATA_API_*`, `ALPACA_DATA_IS_PAPER`, `ALPACA_NEWS_API_*`.
- `BROKER_API_IS_PAPER` true (default) = sandbox: `https://api.cert.tastyworks.com` / `wss://streamer.cert.tastyworks.com`; false = production: `https://api.tastyworks.com` / `wss://streamer.tastyworks.com`.
- `User-Agent` is `trading-agent-framework/<package version>` on every HTTP request and the websocket handshake.
- Strategies never run against the sandbox: `main.py` refuses `BROKER=tastytrade` with `BROKER_API_IS_PAPER` not false, in `paper` and `live` modes. `build_broker` itself still builds sandbox brokers.
- Backtest commission for `BROKER=tastytrade` is `$0`; regulatory fees unchanged.
- Order writes (POST/PUT/DELETE) are never retried automatically. An uncertain submission is looked up by `external-identifier`, never resent.
- `external-identifier = "{strategy_name}:{identifier}"`, `source = "trading-agent-framework"`.
- `OrderType.TRAIL`, `TimeInForce.OPG/CLS/FOK`, `MARKET`+`GTC`, extended-hours `IOC`, fractional non-market orders and notional non-market orders raise `OrderValidationError`.
- `values.py`, `account.py` and `orders.py` are pure: no I/O, no state, no client instances. Money is `Decimal`, parsed with `Decimal(str(raw))`, sent as strings. No new float boundary.
- `httpx` is imported only by `brokers/tastytrade/client.py`; `websockets` only inside `stream._default_connect`. `import trading_agent_framework.brokers` and `import trading_agent_framework.brokers.factory` import neither, nor any `brokers.tastytrade` module.
- No raw `httpx`/`websockets` exception escapes the broker: `BrokerError` (incl. `TastytradeHTTPError`, `BrokerUncertainError`), `OrderValidationError` or `ConfigurationError` only. `order.set_error(...)` happens BEFORE raising on a failed submit.
- Broker/stream/event code never calls strategy hooks; it only feeds `OrderTracker`.
- Tests: hand-written fakes (no `MagicMock`), no network. HTTP tests use `httpx.MockTransport`.
- Ruff config: `line-length = 200`, rules `E, F, I, UP, B`.
- Before every commit: `uv run pytest && uv run ruff check`, both passing.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never edit the user's real `env/.env.*` files (they hold secrets). Only `env/.env.example` is edited. The user adds `TASTYTRADE_*` to `env/.env.alpaca.integration-tests` themselves.

## Review Focus

- **A `Cancelled` for the original id arriving before the replace `PUT` returns** must not cancel the modified order (it stays working). Pinned in Task 7 (`test_a_cancel_of_the_replaced_id_racing_the_put_is_ignored`) and Task 9 (`test_modify_ignores_the_original_ids_cancel`).
- **A sandbox $2 limit that is already `Filled` in the POST response** must become `FILL` exactly once, even when the stream delivers the same snapshot again. Pinned in Task 9 (`test_an_order_filled_in_the_post_response_fills_once`).
- **The access token expiring during a long run** (15-minute lifetime) must be refreshed before the next request, not after a failure. Pinned in Task 6 (`test_the_token_is_refreshed_a_minute_before_it_expires`).
- **Class-share symbols** (`BRK.B`) must reach Tastytrade as `BRK/B` and come back as `BRK.B` (positions, orders). Pinned in Task 4 (`test_class_shares_use_tastytrades_slash_symbol`) and Task 5 (`test_parse_order_and_position_map_class_share_symbols_back`).
- **A submission that timed out but did reach Tastytrade** must be adopted as submitted, tracked, and not sent twice. Pinned in Task 9 (`test_an_uncertain_submission_found_at_tastytrade_is_adopted_not_resent`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `scripts/tests/probe_tastytrade_orders.py` | create | one-off sandbox probe settling the open API questions (Task 1) |
| `src/trading_agent_framework/utils/errors.py` | modify | `BrokerUncertainError` |
| `src/trading_agent_framework/config/env.py` | modify | `BrokerKind.TASTYTRADE`, `TastytradeSettings` |
| `src/trading_agent_framework/config/__init__.py` | modify | re-export `TastytradeSettings` |
| `src/trading_agent_framework/brokers/fees.py` | modify | `$0` Tastytrade commission |
| `src/trading_agent_framework/main.py` | modify | sandbox refusal |
| `src/trading_agent_framework/brokers/tastytrade/__init__.py` | create | package docstring |
| `src/trading_agent_framework/brokers/tastytrade/values.py` | create | pure `decimal_of`/`optional_decimal` |
| `src/trading_agent_framework/brokers/tastytrade/account.py` | create | pure balances parsing and account check |
| `src/trading_agent_framework/brokers/tastytrade/orders.py` | create | pure conform rules, order JSON, status map, parsing |
| `src/trading_agent_framework/brokers/tastytrade/client.py` | create | `TastytradeSession`, `TastytradeHTTPError` |
| `src/trading_agent_framework/brokers/tastytrade/events.py` | create | `TastytradeOrderEvents` |
| `src/trading_agent_framework/brokers/tastytrade/stream.py` | create | `TastytradeAccountStream` |
| `src/trading_agent_framework/brokers/tastytrade/broker.py` | create | `TastytradeBroker` |
| `src/trading_agent_framework/brokers/factory.py` | modify | `_build_tastytrade` |
| `src/trading_agent_framework/brokers/__init__.py` | modify | lazy `TastytradeBroker` export |
| `tests/fakes.py` | modify | `make_tt_order`/`make_tt_fill`/`make_tt_position`, `FakeTastytradeSession` |
| `tests/brokers/tastytrade/*.py` | create | unit tests per module |
| `scripts/tests/smoke_tastytrade_account.py`, `scripts/tests/smoke_tastytrade_orders.py` | create | manual sandbox checks |
| `scripts/tests/HOWTO.md`, `env/.env.example`, `README.md`, `CLAUDE.md`, `src/trading_agent_framework/brokers/CLAUDE.md`, `TODO.md` | modify | docs |

---

### Task 1: Sandbox probe (human in the loop)

**Files:**
- Create: `scripts/tests/probe_tastytrade_orders.py`
- Modify: `docs/superpowers/specs/2026-10-15-tastytrade-broker-design.md` (section 11, after the user runs the probe)

**Interfaces:**
- Consumes: nothing from this plan (standalone `httpx` script; only `config.env.find_project_root`).
- Produces: recorded findings in spec section 11 that Tasks 4 and 5 read (fractional precision, symbol mapping, fixture field names).

- [ ] **Step 1: Write the probe script**

```python
"""Manual one-off probe of the Tastytrade SANDBOX: settles what the API docs leave open before the
Tastytrade broker's order code is written (docs/superpowers/specs/2026-10-15-tastytrade-broker-design.md,
section 11).

NOT part of the automated suite and not a smoke test (`run_smoke_tests.sh` only runs smoke_*.py).
Run it once by hand, then paste its whole output into the implementation session:

    uv run python scripts/tests/probe_tastytrade_orders.py

Credentials: TASTYTRADE_CLIENT_SECRET, TASTYTRADE_REFRESH_TOKEN and TASTYTRADE_ACCOUNT_NUMBER (sandbox ones)
in env/.env.alpaca.integration-tests. Always talks to the sandbox (api.cert.tastyworks.com).

Mostly dry-runs. It places a few sandbox orders and cleans them up: a 0.5-share AAPL market buy (the
sandbox fills market orders at $1) sold back to test fractional sells, and a $5 AAPL limit buy (never
fills in the sandbox) that it replaces at $6 and then cancels.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from trading_agent_framework.config.env import find_project_root

ENV_FILE = find_project_root() / "env" / ".env.alpaca.integration-tests"
SANDBOX_API = "https://api.cert.tastyworks.com"
USER_AGENT = "trading-agent-framework-probe/1.0"
SYMBOL = "AAPL"
PREFIX = "probe:"


def _show(title: str, response: httpx.Response) -> Any:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    print(f"\n=== {title}: HTTP {response.status_code}")
    print(json.dumps(body, indent=2)[:4000])
    return body


def _data(body: Any) -> dict[str, Any]:
    data = body.get("data") if isinstance(body, dict) else None
    return data if isinstance(data, dict) else {}


def _leg(action: str, quantity: str | None = None) -> dict[str, Any]:
    leg: dict[str, Any] = {"instrument-type": "Equity", "symbol": SYMBOL, "action": action}
    if quantity is not None:
        leg["quantity"] = quantity
    return leg


def _market(action: str, quantity: str, external: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"time-in-force": "Day", "order-type": "Market", "legs": [_leg(action, quantity)]}
    if external is not None:
        body["external-identifier"] = external
    return body


def _notional(action: str, value: str, effect: str) -> dict[str, Any]:
    return {"time-in-force": "Day", "order-type": "Notional Market", "value": value, "value-effect": effect, "legs": [_leg(action)]}


def _limit(price: str, quantity: str = "1", external: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"time-in-force": "Day", "order-type": "Limit", "price": price, "price-effect": "Debit", "legs": [_leg("Buy to Open", quantity)]}
    if external is not None:
        body["external-identifier"] = external
    return body


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"SKIP: credentials file not found: {ENV_FILE}")
        return 0
    load_dotenv(ENV_FILE, override=True)
    secret, refresh, account = (os.environ.get(name, "").strip() for name in ("TASTYTRADE_CLIENT_SECRET", "TASTYTRADE_REFRESH_TOKEN", "TASTYTRADE_ACCOUNT_NUMBER"))
    if not (secret and refresh and account):
        print(f"SKIP: TASTYTRADE_CLIENT_SECRET / TASTYTRADE_REFRESH_TOKEN / TASTYTRADE_ACCOUNT_NUMBER not set in {ENV_FILE}")
        return 0
    with httpx.Client(base_url=SANDBOX_API, headers={"User-Agent": USER_AGENT}, timeout=20.0) as client:
        response = client.post("/oauth/token", json={"grant_type": "refresh_token", "refresh_token": refresh, "client_secret": secret})
        token_body = response.json() if response.content else {}
        print(f"=== token: HTTP {response.status_code}; fields (access_token hidden): {sorted(token_body)}; expires_in={token_body.get('expires_in')}")
        if response.status_code != 200:
            print(json.dumps(token_body, indent=2))
            return 1
        client.headers["Authorization"] = f"Bearer {token_body['access_token']}"
        orders = f"/accounts/{account}/orders"
        dry = f"{orders}/dry-run"

        print("\n##### 1. Account, instruments, precisions")
        _show("customers/me/accounts", client.get("/customers/me/accounts"))
        for symbol in ("AAPL", "BRK.B", "BRK%2FB"):
            _show(f"instrument {symbol}", client.get(f"/instruments/equities/{symbol}"))
        _show("quantity decimal precisions", client.get("/instruments/quantity-decimal-precisions"))

        print("\n##### 2. Dry runs (nothing placed)")
        _show("market buy 0.5 (decimal quantity)", client.post(dry, json=_market("Buy to Open", "0.5")))
        for value in ("1.00", "5.00", "10.00"):
            _show(f"notional buy ${value}", client.post(dry, json=_notional("Buy to Open", value, "Debit")))
        _show("limit buy 10 @ 2.00 (fee-calculation)", client.post(dry, json=_limit("2.00", "10")))
        for length in (32, 64, 128, 256):
            _show(f"external-identifier of {length} chars", client.post(dry, json=_limit("2.00", external="x" * length)))

        print("\n##### 3. Fractional sells (real sandbox orders)")
        _show("market buy 0.5 (fills at $1)", client.post(orders, json=_market("Buy to Open", "0.5", f"{PREFIX}fraction")))
        time.sleep(3)
        _show("positions", client.get(f"/accounts/{account}/positions"))
        _show("dry-run market sell 0.5", client.post(dry, json=_market("Sell to Close", "0.5")))
        _show("dry-run notional sell $0.25", client.post(dry, json=_notional("Sell to Close", "0.25", "Credit")))
        _show("market sell 0.5 (cleanup)", client.post(orders, json=_market("Sell to Close", "0.5", f"{PREFIX}fraction-close")))

        print("\n##### 4. Replace semantics (real sandbox orders)")
        placed = _show("limit buy 1 @ 5.00 (rests)", client.post(orders, json=_limit("5.00", external=f"{PREFIX}replace")))
        order_id = (_data(placed).get("order") or {}).get("id")
        if order_id is not None:
            time.sleep(2)
            replacement = {"time-in-force": "Day", "order-type": "Limit", "price": "6.00", "price-effect": "Debit", "external-identifier": f"{PREFIX}replace"}
            _show("replace @ 6.00 (PUT without legs)", client.put(f"{orders}/{order_id}", json=replacement))
            time.sleep(2)
            _show("order history (newest first)", client.get(orders, params={"per-page": 5, "sort": "Desc"}))
        live = _show("live orders", client.get(f"{orders}/live"))
        for item in _data(live).get("items") or []:
            if str(item.get("external-identifier", "")).startswith(PREFIX) and item.get("cancellable"):
                _show(f"cancel {item['id']} (cleanup)", client.delete(f"{orders}/{item['id']}"))
    print("\nDone. Paste this whole output into the implementation session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lint it and check it skips cleanly without credentials**

Run: `uv run ruff check scripts/tests/probe_tastytrade_orders.py && uv run python scripts/tests/probe_tastytrade_orders.py | head -3`
Expected: ruff passes; the script either prints `SKIP: ...` (no `TASTYTRADE_*` yet) or starts printing `=== token: HTTP 200`.

- [ ] **Step 3: Commit**

```bash
git add scripts/tests/probe_tastytrade_orders.py
git commit -m "chore(tastytrade): sandbox probe for the open API questions (Task 1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: STOP and ask the user to run the probe**

Tell the user: "Add `TASTYTRADE_CLIENT_SECRET`, `TASTYTRADE_REFRESH_TOKEN` and `TASTYTRADE_ACCOUNT_NUMBER` (sandbox credentials) to `env/.env.alpaca.integration-tests`, run `uv run python scripts/tests/probe_tastytrade_orders.py`, and paste the whole output here." Wait for the output. Do not continue to Task 2 before it arrives.

- [ ] **Step 5: Record the findings in spec section 11**

Append to `docs/superpowers/specs/2026-10-15-tastytrade-broker-design.md` under `## 11. Probe findings`, one bullet each, quoting the HTTP status and the relevant fields from the output:

1. Account record: `margin-or-cash`, `is-closed`, and whether items are wrapped as `{"account": {...}, "authority-level": ...}`.
2. `is-fractional-quantity-eligible` for AAPL; which of `BRK.B` / `BRK%2FB` returned 200.
3. Equity quantity decimal precision (from `quantity-decimal-precisions`).
4. Dry-run of a 0.5-share market buy: accepted or the error.
5. Smallest accepted notional buy, or the minimum the error states.
6. `fee-calculation` of the 10-share limit buy.
7. Longest accepted `external-identifier`.
8. Fractional sells: was the 0.5 buy filled and shown in positions (with `quantity` format), was the 0.5 market sell accepted, was the notional sell accepted.
9. Replace: does the PUT response carry a new `id`, `replaces-order-id`, and the same `external-identifier`? What did the original become in the order history?
10. Field names actually present on an order snapshot (`legs[].fills[]` keys: `fill-id`, `quantity`, `fill-price`; `reject-reason`; `replacing-order-id`) and on a position (`quantity`, `quantity-direction`, `average-open-price`).

- [ ] **Step 6: Apply the decision table**

| Finding | Action in later tasks |
|---|---|
| `BRK%2FB` is 200 and `BRK.B` is 404 | keep `to_tastytrade_symbol`/`from_tastytrade_symbol` as written in Tasks 4-5 |
| `BRK.B` is 200 | make both functions return the symbol unchanged and change their tests to assert identity |
| Equity precision differs from 5 decimals | set `FRACTIONAL_QUANTUM` in Task 4 to that precision and update the `2.12345` test values |
| Fill/position/order field names differ from the `make_tt_*` fixtures in Task 5 | change the fixtures and the parsing keys in Task 5 to the real names |
| PUT returns the same id | nothing (Task 7's `finish_replace` handles it) |
| 0.5-share market buy or sell refused, notional sell refused, PUT drops `external-identifier`, or longest accepted `external-identifier` is under 64 chars | STOP: report it to the user; the conform rules or identity scheme need a design decision before Task 4 |

- [ ] **Step 7: Commit the findings**

```bash
git add docs/superpowers/specs/2026-10-15-tastytrade-broker-design.md
git commit -m "docs(tastytrade): record sandbox probe findings (Task 1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Config, uncertain-error type, fees and the sandbox refusal

**Files:**
- Modify: `src/trading_agent_framework/utils/errors.py` (after `class BrokerError`)
- Modify: `src/trading_agent_framework/config/env.py` (`BrokerKind`, new `TastytradeSettings` after `IbkrSettings`)
- Modify: `src/trading_agent_framework/config/__init__.py`
- Modify: `src/trading_agent_framework/brokers/fees.py`
- Modify: `src/trading_agent_framework/main.py:19-47`
- Modify: `env/.env.example`, `README.md` (env table, fees paragraph)
- Test: `tests/test_errors.py`, `tests/config/test_env.py`, `tests/config/test_config_public_api.py`, `tests/brokers/test_fees.py`, `tests/test_main.py`

**Interfaces:**
- Produces: `BrokerUncertainError(BrokerError)`; `BrokerKind.TASTYTRADE = "tastytrade"`; `TastytradeSettings(client_secret: str, refresh_token: str, account_number: str, is_paper: bool)` with properties `api_url: str`, `streamer_url: str` and `TastytradeSettings.from_env(is_paper: bool, env: Mapping[str, str] | None = None)`; `COMMISSIONS[BrokerKind.TASTYTRADE]`; `main._refuse_tastytrade_sandbox()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_errors.py`:

```python
def test_an_uncertain_broker_error_is_a_broker_error() -> None:
    from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError

    assert issubclass(BrokerUncertainError, BrokerError)
```

Append to `tests/config/test_env.py` (add `TastytradeSettings` to the existing `from trading_agent_framework.config.env import (...)` block):

```python
# --- Tastytrade ----------------------------------------------------------------------

_TT_ENV = {"TASTYTRADE_CLIENT_SECRET": "secret", "TASTYTRADE_REFRESH_TOKEN": "refresh", "TASTYTRADE_ACCOUNT_NUMBER": "5WT00001"}


def test_broker_settings_accept_tastytrade() -> None:
    assert BrokerSettings.from_env({"BROKER": " Tastytrade "}).kind is BrokerKind.TASTYTRADE


def test_tastytrade_settings_read_the_three_required_variables() -> None:
    assert TastytradeSettings.from_env(True, _TT_ENV) == TastytradeSettings(client_secret="secret", refresh_token="refresh", account_number="5WT00001", is_paper=True)


def test_tastytrade_sandbox_and_production_hosts() -> None:
    sandbox = TastytradeSettings.from_env(True, _TT_ENV)
    production = TastytradeSettings.from_env(False, _TT_ENV)

    assert (sandbox.api_url, sandbox.streamer_url) == ("https://api.cert.tastyworks.com", "wss://streamer.cert.tastyworks.com")
    assert (production.api_url, production.streamer_url) == ("https://api.tastyworks.com", "wss://streamer.tastyworks.com")


@pytest.mark.parametrize("variable", sorted(_TT_ENV))
def test_tastytrade_settings_reject_a_missing_or_blank_variable(variable: str) -> None:
    for broken in ({k: v for k, v in _TT_ENV.items() if k != variable}, {**_TT_ENV, variable: "  "}):
        with pytest.raises(ConfigurationError, match=variable):
            TastytradeSettings.from_env(True, broken)


def test_tastytrade_secrets_stay_out_of_repr() -> None:
    text = repr(TastytradeSettings.from_env(True, _TT_ENV))

    assert "secret" not in text
    assert "refresh" not in text
    assert "5WT00001" in text
```

Append to `tests/config/test_config_public_api.py` inside `test_config_reexports_match_source_module`:

```python
    assert config.TastytradeSettings is env_module.TastytradeSettings
```

Append to `tests/brokers/test_fees.py`:

```python
TASTYTRADE = TradingFeeFactory(BrokerKind.TASTYTRADE)


def test_tastytrade_charges_no_commission_only_regulatory_fees() -> None:
    assert _buy(TASTYTRADE, 100, 65000) == TradeFees(buy=D("0.01"), sell=D(0))  # CAT only
    assert _sell(TASTYTRADE, 100, 65000) == TradeFees(buy=D(0), sell=D("1.36"))  # SEC + TAF + CAT, as Alpaca
```

Append to `tests/test_main.py`:

```python
@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_a_strategy_never_runs_against_the_tastytrade_sandbox(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: TradingMode) -> None:
    from rich.console import Console

    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", lambda *a, **k: pytest.fail("a sandbox broker was built for a strategy"))
    monkeypatch.setenv("BROKER", "tastytrade")
    monkeypatch.delenv("BROKER_API_IS_PAPER", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        main_module._run_strategy(Console(width=500), mode, Strategies.NEWS_BINARY)

    assert excinfo.value.code == 1
    assert "Tastytrade sandbox" in capsys.readouterr().out


def test_live_tastytrade_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console
    from tests.fakes import FakeBroker, FakeClock, et

    built: list[str] = []
    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", lambda name: built.append(name) or FakeBroker(FakeClock(et(2026, 10, 15, 10)), strategy_name=name))
    monkeypatch.setattr(main_module, "build_strategy", lambda strategy, broker, mode: None)
    monkeypatch.setenv("BROKER", "tastytrade")
    monkeypatch.setenv("BROKER_API_IS_PAPER", "false")

    main_module._run_strategy(Console(width=500), TradingMode.LIVE, Strategies.NEWS_BINARY)

    assert built == ["news_binary"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_errors.py tests/config tests/brokers/test_fees.py tests/test_main.py -q`
Expected: FAIL / errors: `cannot import name 'BrokerUncertainError'`, `cannot import name 'TastytradeSettings'`, `BrokerKind has no attribute TASTYTRADE`.

- [ ] **Step 3: Implement**

`src/trading_agent_framework/utils/errors.py`, right after `class BrokerError`:

```python
class BrokerUncertainError(BrokerError):
    """A broker write whose outcome is unknown (timeout, transport failure, 5xx): it may or may not have happened."""
```

`src/trading_agent_framework/config/env.py`: add the constants after `IBKR_LIVE_PORT`:

```python
TASTYTRADE_API_URLS = {True: "https://api.cert.tastyworks.com", False: "https://api.tastyworks.com"}  # keyed by is_paper (sandbox)
TASTYTRADE_STREAMER_URLS = {True: "wss://streamer.cert.tastyworks.com", False: "wss://streamer.tastyworks.com"}
```

extend `BrokerKind` (and its docstring's broker list):

```python
    ALPACA = "alpaca"
    IBKR = "ibkr"
    TASTYTRADE = "tastytrade"
```

and add after `IbkrSettings`:

```python
@dataclass(frozen=True, slots=True)
class TastytradeSettings:
    """Tastytrade OAuth credentials and the account to trade. `is_paper` picks the sandbox hosts:
    sandbox and production credentials are separate OAuth applications and never work against each other."""

    client_secret: str = field(repr=False)
    refresh_token: str = field(repr=False)
    account_number: str
    is_paper: bool

    @property
    def api_url(self) -> str:
        return TASTYTRADE_API_URLS[self.is_paper]

    @property
    def streamer_url(self) -> str:
        return TASTYTRADE_STREAMER_URLS[self.is_paper]

    @classmethod
    def from_env(cls, is_paper: bool, env: Mapping[str, str] | None = None) -> TastytradeSettings:
        source = env if env is not None else os.environ
        _reject_renamed(source)
        values: dict[str, str] = {}
        for name in ("TASTYTRADE_CLIENT_SECRET", "TASTYTRADE_REFRESH_TOKEN", "TASTYTRADE_ACCOUNT_NUMBER"):
            value = (source.get(name) or "").strip()
            if not value:
                raise ConfigurationError(f"Missing or blank {name} environment variable")
            values[name] = value
        return cls(
            client_secret=values["TASTYTRADE_CLIENT_SECRET"],
            refresh_token=values["TASTYTRADE_REFRESH_TOKEN"],
            account_number=values["TASTYTRADE_ACCOUNT_NUMBER"],
            is_paper=is_paper,
        )
```

`src/trading_agent_framework/config/__init__.py`: add `TastytradeSettings` to the import list and to `__all__`.

`src/trading_agent_framework/brokers/fees.py`: update the module docstring's first line to `` chosen by `BROKER` (`alpaca` default, `ibkr`, `tastytrade`)``, and add:

```python
def _tastytrade_commission(shares: Decimal) -> Decimal:
    return Decimal(0)  # $0 to open and close stocks/ETFs


COMMISSIONS: dict[BrokerKind, Callable[[Decimal], Decimal]] = {
    BrokerKind.ALPACA: _alpaca_commission,
    BrokerKind.IBKR: _ibkr_commission,
    BrokerKind.TASTYTRADE: _tastytrade_commission,
}
```

`src/trading_agent_framework/main.py`: change the import to `from trading_agent_framework.config.env import BrokerKind, BrokerSettings, TradingMode`, add above `_run_strategy`:

```python
def _refuse_tastytrade_sandbox() -> None:
    """Strategies never run against the Tastytrade sandbox: its fills are synthetic (market orders at $1)."""
    settings = BrokerSettings.from_env()
    if settings.kind is BrokerKind.TASTYTRADE and settings.is_paper:
        raise ConfigurationError(
            "BROKER=tastytrade with BROKER_API_IS_PAPER=true is the Tastytrade sandbox, which fills every market order at $1 "
            "and resets daily: forward-test on Alpaca paper (BROKER=alpaca) and use BROKER=tastytrade with "
            "BROKER_API_IS_PAPER=false for live trading only"
        )
```

and call it first inside the existing `try` of the non-backtesting branch:

```python
        try:
            _refuse_tastytrade_sandbox()
            broker = build_broker(strategy_name)
        except (ConfigurationError, BrokerError) as exc:
```

`env/.env.example`: change the broker-selection comment to `# Which broker trades: alpaca (default), ibkr or tastytrade.`, add to the `BROKER_API_IS_PAPER` comment `# With BROKER=tastytrade, true is the sandbox, which strategies refuse to run against (synthetic fills): use it only for scripts/tests.`, and add a block after the IBKR block:

```bash
# --- Tastytrade trading (required when BROKER=tastytrade) ---------------------------
# OAuth application client secret and a grant's refresh token (never expires), from
# https://my.tastytrade.com (production) or the sandbox account tool (sandbox credentials are separate).
# TASTYTRADE_ACCOUNT_NUMBER: the account to trade; it must be a cash account for live trading.
TASTYTRADE_CLIENT_SECRET=
TASTYTRADE_REFRESH_TOKEN=
TASTYTRADE_ACCOUNT_NUMBER=
```

`README.md`: in the env table add after the IBKR row:

```markdown
| `TASTYTRADE_CLIENT_SECRET`, `TASTYTRADE_REFRESH_TOKEN`, `TASTYTRADE_ACCOUNT_NUMBER` | Tastytrade trading | `BROKER=tastytrade`, live |
```

and in the backtest fees paragraph change `(`alpaca` or `ibkr`;` to `` (`alpaca`, `ibkr` or `tastytrade`; `` and `(Alpaca $0, IBKR Pro Fixed max($1, $0.005/share))` to `(Alpaca and Tastytrade $0, IBKR Pro Fixed max($1, $0.005/share))`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_errors.py tests/config tests/brokers/test_fees.py tests/test_main.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/utils/errors.py src/trading_agent_framework/config src/trading_agent_framework/brokers/fees.py src/trading_agent_framework/main.py env/.env.example README.md tests/test_errors.py tests/config tests/brokers/test_fees.py tests/test_main.py
git commit -m "feat: BROKER=tastytrade settings, \$0 fees and the sandbox refusal (Task 2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pure number parsing and account module

**Files:**
- Create: `src/trading_agent_framework/brokers/tastytrade/__init__.py`
- Create: `src/trading_agent_framework/brokers/tastytrade/values.py`
- Create: `src/trading_agent_framework/brokers/tastytrade/account.py`
- Create: `tests/brokers/tastytrade/__init__.py` (empty)
- Test: `tests/brokers/tastytrade/test_account.py`

**Interfaces:**
- Produces: `values.decimal_of(raw: object, what: str) -> Decimal` (raises `BrokerError` on missing/non-numeric/non-finite); `values.optional_decimal(raw: object, what: str) -> Decimal | None`; `account.parse_balances(data: Mapping[str, Any]) -> AccountBalances`; `account.check_account(items: Sequence[Mapping[str, Any]], account_number: str, *, is_paper: bool) -> list[str]` (warnings; raises `ConfigurationError`).

- [ ] **Step 1: Write the failing tests**

`tests/brokers/tastytrade/test_account.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.brokers.tastytrade import account
from trading_agent_framework.brokers.tastytrade.values import decimal_of, optional_decimal
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError

D = Decimal
BALANCES = {"account-number": "5WT00001", "cash-balance": "1000.50", "net-liquidating-value": "2500.25", "equity-buying-power": "1000.50", "margin-equity": "2500.25"}


def _item(number: str = "5WT00001", kind: str = "Cash", closed: bool = False) -> dict[str, object]:
    return {"account": {"account-number": number, "margin-or-cash": kind, "is-closed": closed, "nickname": "Individual"}, "authority-level": "owner"}


# --- values ----------------------------------------------------------------------------


def test_decimal_of_parses_strings_and_numbers_exactly() -> None:
    assert decimal_of("150.10", "price") == D("150.10")
    assert decimal_of(3, "quantity") == D(3)


@pytest.mark.parametrize(("raw", "message"), [(None, "missing"), ("abc", "not a number"), ("NaN", "not a finite number")])
def test_decimal_of_rejects_bad_values(raw: object, message: str) -> None:
    with pytest.raises(BrokerError, match=f"Tastytrade price is {message}"):
        decimal_of(raw, "price")


def test_optional_decimal_passes_none_through() -> None:
    assert optional_decimal(None, "price") is None
    assert optional_decimal("1.5", "price") == D("1.5")


# --- balances --------------------------------------------------------------------------


def test_balances_map_cash_net_liquidating_value_and_equity_buying_power() -> None:
    assert account.parse_balances(BALANCES) == AccountBalances(cash=D("1000.50"), portfolio_value=D("2500.25"), buying_power=D("1000.50"))


def test_negative_cash_is_kept() -> None:
    assert account.parse_balances({**BALANCES, "cash-balance": "-12.50"}).cash == D("-12.50")


@pytest.mark.parametrize("key", ["cash-balance", "net-liquidating-value", "equity-buying-power"])
def test_a_missing_balance_field_is_a_broker_error(key: str) -> None:
    with pytest.raises(BrokerError, match=key):
        account.parse_balances({k: v for k, v in BALANCES.items() if k != key})


# --- account check ---------------------------------------------------------------------


def test_a_cash_account_passes_without_warnings() -> None:
    assert account.check_account([_item(), _item("5WT00002")], "5WT00001", is_paper=False) == []


def test_an_account_the_login_does_not_own_is_refused_listing_the_owned_ones() -> None:
    with pytest.raises(ConfigurationError, match=r"TASTYTRADE_ACCOUNT_NUMBER=5WT99999 .*5WT00001, 5WT00002"):
        account.check_account([_item(), _item("5WT00002")], "5WT99999", is_paper=False)


def test_a_closed_account_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="closed"):
        account.check_account([_item(closed=True)], "5WT00001", is_paper=True)


def test_a_margin_account_is_refused_live() -> None:
    with pytest.raises(ConfigurationError, match="margin account"):
        account.check_account([_item(kind="Margin")], "5WT00001", is_paper=False)


def test_a_margin_account_only_warns_in_the_sandbox() -> None:
    [warning] = account.check_account([_item(kind="Margin")], "5WT00001", is_paper=True)

    assert "margin" in warning


def test_unwrapped_account_items_are_accepted_too() -> None:
    assert account.check_account([{"account-number": "5WT00001", "margin-or-cash": "Cash", "is-closed": False}], "5WT00001", is_paper=False) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/tastytrade/test_account.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.brokers.tastytrade'`.

- [ ] **Step 3: Implement**

`src/trading_agent_framework/brokers/tastytrade/__init__.py`:

```python
"""Tastytrade broker: REST API + account streamer for orders, account and positions (see `broker.py`)."""
```

`src/trading_agent_framework/brokers/tastytrade/values.py`:

```python
"""Number parsing shared by the pure Tastytrade modules: the API sends money and quantities as strings.

Pure: no I/O, no state.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from trading_agent_framework.utils.errors import BrokerError


def decimal_of(raw: object, what: str) -> Decimal:
    """`raw` as an exact `Decimal`; `BrokerError` naming `what` when it is missing or not a finite number."""
    if raw is None or isinstance(raw, bool):
        raise BrokerError(f"Tastytrade {what} is missing")
    try:
        value = Decimal(str(raw))
    except InvalidOperation:
        raise BrokerError(f"Tastytrade {what} is not a number: {raw!r}") from None
    if not value.is_finite():
        raise BrokerError(f"Tastytrade {what} is not a finite number: {raw!r}")
    return value


def optional_decimal(raw: object, what: str) -> Decimal | None:
    return None if raw is None else decimal_of(raw, what)
```

`src/trading_agent_framework/brokers/tastytrade/account.py`:

```python
"""Pure Tastytrade account translation: balances, and the start-up account check.

No I/O: the broker passes in what `GET /accounts/{a}/balances` and `GET /customers/me/accounts` returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trading_agent_framework.brokers.tastytrade.values import decimal_of
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.errors import ConfigurationError


def parse_balances(data: Mapping[str, Any]) -> AccountBalances:
    return AccountBalances(
        cash=decimal_of(data.get("cash-balance"), "balance cash-balance"),
        portfolio_value=decimal_of(data.get("net-liquidating-value"), "balance net-liquidating-value"),
        buying_power=decimal_of(data.get("equity-buying-power"), "balance equity-buying-power"),
    )


def check_account(items: Sequence[Mapping[str, Any]], account_number: str, *, is_paper: bool) -> list[str]:
    """Refuse an account this login doesn't own, a closed one, and a live margin account; returns warnings.

    `items` are `GET /customers/me/accounts`'s `data.items`: `{"account": {...}, "authority-level": ...}`.
    """
    accounts = [item.get("account", item) for item in items]
    found = next((a for a in accounts if a.get("account-number") == account_number), None)
    if found is None:
        owned = ", ".join(sorted(str(a.get("account-number")) for a in accounts)) or "none"
        raise ConfigurationError(f"TASTYTRADE_ACCOUNT_NUMBER={account_number} is not an account of this Tastytrade login (its accounts: {owned})")
    if found.get("is-closed"):
        raise ConfigurationError(f"Tastytrade account {account_number} is closed")
    warnings: list[str] = []
    if found.get("margin-or-cash") == "Margin":
        if not is_paper:
            raise ConfigurationError(f"Tastytrade account {account_number} is a margin account; this framework expects a cash account (no margin, no shorting)")
        warnings.append(f"Tastytrade sandbox account {account_number} is a margin account (accepted in the sandbox only)")
    return warnings
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/tastytrade/test_account.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/brokers/tastytrade tests/brokers/tastytrade
git commit -m "feat(tastytrade): pure balances parsing and account check (Task 3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Pure order rules and order JSON

**Files:**
- Create: `src/trading_agent_framework/brokers/tastytrade/orders.py`
- Test: `tests/brokers/tastytrade/test_orders_build.py`

**Interfaces:**
- Consumes: nothing beyond entities.
- Produces (in `orders`): `SOURCE = "trading-agent-framework"`; `FRACTIONAL_QUANTUM: Decimal`; `external_identifier_for(strategy_name: str, identifier: str) -> str`; `identifier_from_external(external: str | None, strategy_name: str) -> str | None`; `to_tastytrade_symbol(symbol: str) -> str`; `from_tastytrade_symbol(symbol: str) -> str`; `conform_order(order: Order, *, fractional_eligible: bool, last_price: Decimal | None) -> Order`; `round_price(price: Decimal) -> Decimal`; `build_order_json(order: Order) -> dict[str, Any]`; `build_replace_json(order: Order) -> dict[str, Any]`.

Before starting: read spec section 11 (Task 1's findings) and apply its decision table to `FRACTIONAL_QUANTUM` and the symbol functions.

- [ ] **Step 1: Write the failing tests**

`tests/brokers/tastytrade/test_orders_build.py`:

```python
from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from trading_agent_framework.brokers.tastytrade import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

D = Decimal
AAPL = Asset("AAPL")


def _order(quantity: str | None = "10", **overrides: object) -> Order:
    fields: dict[str, object] = {"strategy_name": "s", "asset": AAPL, "side": OrderSide.BUY, "quantity": D(quantity) if quantity is not None else None, "client_order_id": "s:abc"}
    fields.update(overrides)
    return Order(**fields)  # ty: ignore[invalid-argument-type]


def _conform(order: Order, *, eligible: bool = False, last_price: str | None = None) -> Order:
    return orders.conform_order(order, fractional_eligible=eligible, last_price=D(last_price) if last_price is not None else None)


# --- identifiers and symbols -------------------------------------------------------------


def test_external_identifier_round_trip() -> None:
    external = orders.external_identifier_for("s", "abc")

    assert external == "s:abc"
    assert orders.identifier_from_external(external, "s") == "abc"
    assert orders.identifier_from_external(external, "other") is None
    assert orders.identifier_from_external("s:", "s") is None
    assert orders.identifier_from_external(None, "s") is None


def test_class_shares_use_tastytrades_slash_symbol() -> None:
    assert orders.to_tastytrade_symbol("BRK.B") == "BRK/B"
    assert orders.from_tastytrade_symbol("BRK/B") == "BRK.B"
    assert orders.build_order_json(_order(asset=Asset("BRK.B")))["legs"][0]["symbol"] == "BRK/B"


# --- conform: unsupported orders -------------------------------------------------------------


def test_trailing_stops_are_rejected() -> None:
    with pytest.raises(OrderValidationError, match="no trailing stops"):
        _conform(_order(order_type=OrderType.TRAIL, trail_percent=D(5)), eligible=True)


@pytest.mark.parametrize("tif", [TimeInForce.OPG, TimeInForce.CLS, TimeInForce.FOK])
def test_unsupported_time_in_force_is_rejected(tif: TimeInForce) -> None:
    with pytest.raises(OrderValidationError, match="time_in_force"):
        _conform(_order(order_type=OrderType.LIMIT, limit_price=D(150), time_in_force=tif))


def test_a_gtc_market_order_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="GTC market"):
        _conform(_order(time_in_force=TimeInForce.GTC))


def test_an_extended_hours_ioc_order_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="extended-hours"):
        _conform(_order(order_type=OrderType.LIMIT, limit_price=D(150), time_in_force=TimeInForce.IOC, extended_hours=True))


# --- conform: fractional quantities ------------------------------------------------------------


def test_a_whole_quantity_is_unchanged() -> None:
    assert _conform(_order("10")).quantity == D(10)


def test_an_eligible_fractional_market_order_keeps_its_fraction_floored_to_the_precision() -> None:
    assert _conform(_order("2.123456"), eligible=True).quantity == D("2.12345")


def test_a_non_eligible_fractional_market_order_is_floored_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        conformed = _conform(_order("2.7"))

    assert conformed.quantity == D(2)
    assert "whole shares" in caplog.text


def test_a_non_eligible_fraction_below_one_share_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="below one whole share"):
        _conform(_order("0.5"))


@pytest.mark.parametrize("eligible", [True, False])
@pytest.mark.parametrize(("order_type", "prices"), [(OrderType.LIMIT, {"limit_price": D(150)}), (OrderType.STOP, {"stop_price": D(140)}), (OrderType.STOP_LIMIT, {"stop_price": D(140), "stop_limit_price": D(139)})])
def test_a_fractional_non_market_order_is_rejected(eligible: bool, order_type: OrderType, prices: dict[str, Decimal]) -> None:
    with pytest.raises(OrderValidationError, match="only supported for MARKET"):
        _conform(_order("4.57", order_type=order_type, **prices), eligible=eligible)


# --- conform: notional ---------------------------------------------------------------------------


def test_an_eligible_notional_market_order_is_unchanged() -> None:
    conformed = _conform(_order(None, notional=D(250)), eligible=True)

    assert (conformed.notional, conformed.quantity) == (D(250), None)


def test_a_non_eligible_notional_order_becomes_whole_shares_at_the_last_price(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        conformed = _conform(_order(None, notional=D(1000)), last_price="150")

    assert (conformed.quantity, conformed.notional) == (D(6), None)
    assert "whole shares" in caplog.text


def test_a_non_eligible_notional_below_one_share_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="below one whole share"):
        _conform(_order(None, notional=D(100)), last_price="150")


def test_a_non_eligible_notional_without_a_price_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="no last price"):
        _conform(_order(None, notional=D(1000)))


def test_a_notional_limit_order_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="notional is only supported for MARKET"):
        _conform(_order(None, notional=D(1000), order_type=OrderType.LIMIT, limit_price=D(150)), eligible=True)


# --- order JSON ----------------------------------------------------------------------------------


def test_a_market_buy() -> None:
    assert orders.build_order_json(_order()) == {
        "time-in-force": "Day",
        "source": "trading-agent-framework",
        "external-identifier": "s:abc",
        "order-type": "Market",
        "legs": [{"instrument-type": "Equity", "symbol": "AAPL", "action": "Buy to Open", "quantity": "10"}],
    }


def test_a_limit_sell_rounds_its_price_and_takes_a_credit() -> None:
    body = orders.build_order_json(_order(side=OrderSide.SELL, order_type=OrderType.LIMIT, limit_price=D("150.126")))

    assert (body["order-type"], body["price"], body["price-effect"]) == ("Limit", "150.13", "Credit")
    assert body["legs"][0]["action"] == "Sell to Close"


def test_a_price_below_one_dollar_keeps_four_decimals() -> None:
    assert orders.round_price(D("0.12345")) == D("0.1235")


def test_a_stop_order_sends_only_a_trigger() -> None:
    body = orders.build_order_json(_order(side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=D(140)))

    assert (body["order-type"], body["stop-trigger"]) == ("Stop", "140.00")
    assert "price" not in body
    assert "price-effect" not in body


def test_a_stop_limit_order_uses_the_stop_limit_price_as_its_price() -> None:
    body = orders.build_order_json(_order(side=OrderSide.SELL, order_type=OrderType.STOP_LIMIT, stop_price=D(140), stop_limit_price=D(139)))

    assert (body["order-type"], body["stop-trigger"], body["price"], body["price-effect"]) == ("Stop Limit", "140.00", "139.00", "Credit")


def test_a_notional_order_sends_a_value_and_no_leg_quantity() -> None:
    body = orders.build_order_json(_order(None, notional=D("250.009")))

    assert (body["order-type"], body["value"], body["value-effect"]) == ("Notional Market", "250.00", "Debit")
    assert "quantity" not in body["legs"][0]


def test_a_fractional_quantity_is_sent_as_a_plain_decimal_string() -> None:
    assert orders.build_order_json(_order("4.57"))["legs"][0]["quantity"] == "4.57"


@pytest.mark.parametrize(("tif", "expected"), [(TimeInForce.DAY, "Ext"), (TimeInForce.GTC, "GTC Ext")])
def test_extended_hours_time_in_force(tif: TimeInForce, expected: str) -> None:
    body = orders.build_order_json(_order(order_type=OrderType.LIMIT, limit_price=D(150), time_in_force=tif, extended_hours=True))

    assert body["time-in-force"] == expected


def test_a_limit_order_without_a_limit_price_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="needs limit_price"):
        orders.build_order_json(_order(order_type=OrderType.LIMIT))


def test_an_order_without_a_client_order_id_sends_no_external_identifier() -> None:
    assert "external-identifier" not in orders.build_order_json(_order(client_order_id=None))


def test_a_replace_body_drops_the_legs_and_keeps_prices_and_identifier() -> None:
    body = orders.build_replace_json(_order(order_type=OrderType.LIMIT, limit_price=D(151)))

    assert "legs" not in body
    assert (body["price"], body["external-identifier"]) == ("151.00", "s:abc")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/tastytrade/test_orders_build.py -q`
Expected: FAIL with `ImportError: cannot import name 'orders'`.

- [ ] **Step 3: Implement**

`src/trading_agent_framework/brokers/tastytrade/orders.py`:

```python
"""Pure translation between our entities and Tastytrade's REST JSON (dasherized keys).

No I/O, no state, no client instances: the broker passes in what it fetched (fractional
eligibility, a last price). Money stays `Decimal`, sent and parsed as strings.
"""

from __future__ import annotations

import logging
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any

from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

logger = logging.getLogger(__name__)

SOURCE = "trading-agent-framework"
# Smallest fractional step of an eligible equity (probe: GET /instruments/quantity-decimal-precisions).
FRACTIONAL_QUANTUM = Decimal("0.00001")

_PENNY = Decimal("0.01")
_SUB_PENNY = Decimal("0.0001")

# Ours -> (regular hours, extended hours); None: Tastytrade has no extended-hours variant.
_TIME_IN_FORCE: dict[TimeInForce, tuple[str, str | None]] = {
    TimeInForce.DAY: ("Day", "Ext"),
    TimeInForce.GTC: ("GTC", "GTC Ext"),
    TimeInForce.IOC: ("IOC", None),
}


# --- identifiers and symbols -------------------------------------------------------------


def external_identifier_for(strategy_name: str, identifier: str) -> str:
    return f"{strategy_name}:{identifier}"


def identifier_from_external(external: str | None, strategy_name: str) -> str | None:
    prefix = f"{strategy_name}:"
    if external and external.startswith(prefix) and len(external) > len(prefix):
        return external[len(prefix) :]
    return None


def to_tastytrade_symbol(symbol: str) -> str:
    """Class shares: our (Alpaca's) `BRK.B` is Tastytrade's `BRK/B`."""
    return symbol.replace(".", "/")


def from_tastytrade_symbol(symbol: str) -> str:
    return symbol.replace("/", ".")


# --- conform -------------------------------------------------------------------------------


def conform_order(order: Order, *, fractional_eligible: bool, last_price: Decimal | None) -> Order:
    """Apply Tastytrade's order rules: reject what it can't take, floor what it can't split.

    `fractional_eligible` is the equity's `is-fractional-quantity-eligible`; `last_price` is only
    needed to turn a non-eligible notional order into whole shares.
    """
    if order.order_type is OrderType.TRAIL:
        raise OrderValidationError("Tastytrade has no trailing stops; use a STOP order")
    time_in_force = _TIME_IN_FORCE.get(order.time_in_force)
    if time_in_force is None:
        raise OrderValidationError(f"Tastytrade does not support time_in_force={order.time_in_force}")
    if order.extended_hours and time_in_force[1] is None:
        raise OrderValidationError(f"Tastytrade has no extended-hours {order.time_in_force} orders")
    if order.order_type is OrderType.MARKET and order.time_in_force is TimeInForce.GTC:
        raise OrderValidationError("Tastytrade does not accept GTC market orders")
    if order.notional is not None:
        return _conform_notional(order, fractional_eligible=fractional_eligible, last_price=last_price)
    if order.quantity is None:
        raise OrderValidationError(f"order {order.identifier} has no quantity")
    if order.quantity % 1 == 0:
        return order
    if order.order_type is not OrderType.MARKET:
        raise OrderValidationError(f"fractional quantity is only supported for MARKET orders (order_type={order.order_type}): a fractional stop or limit would leave the remainder behind")
    if fractional_eligible:
        floored = order.quantity.quantize(FRACTIONAL_QUANTUM, rounding=ROUND_FLOOR)
        if floored != order.quantity:  # keep 4.57 as "4.57", not "4.57000"
            order.quantity = floored
        if order.quantity <= 0:
            raise OrderValidationError(f"order {order.identifier}: quantity rounds to zero at Tastytrade's precision {FRACTIONAL_QUANTUM}")
        return order
    order.quantity = _whole_shares(order, order.quantity, f"quantity {order.quantity}")
    return order


def _conform_notional(order: Order, *, fractional_eligible: bool, last_price: Decimal | None) -> Order:
    if order.order_type is not OrderType.MARKET:
        raise OrderValidationError(f"notional is only supported for MARKET orders (order_type={order.order_type})")
    if fractional_eligible:
        return order
    assert order.notional is not None
    if last_price is None or last_price <= 0:
        raise OrderValidationError(f"{order.asset.symbol} can't be traded by dollar amount at Tastytrade and has no last price to convert {order.notional} into shares")
    order.quantity = _whole_shares(order, order.notional / last_price, f"notional {order.notional} at {last_price}")
    order.notional = None
    return order


def _whole_shares(order: Order, shares: Decimal, what: str) -> Decimal:
    whole = shares.to_integral_value(rounding=ROUND_FLOOR)
    if whole < 1:
        raise OrderValidationError(f"order {order.identifier}: {what} is below one whole share of {order.asset.symbol}, which Tastytrade trades in whole shares only")
    logger.warning("Tastytrade trades %s in whole shares only: %s -> %s shares", order.asset.symbol, what, whole)
    return whole


# --- order JSON ----------------------------------------------------------------------------


def round_price(price: Decimal) -> Decimal:
    return price.quantize(_PENNY if price >= 1 else _SUB_PENNY, rounding=ROUND_HALF_UP)


def _price(price: Decimal) -> str:
    return format(round_price(price), "f")


def _require(value: Decimal | None, name: str, order: Order) -> Decimal:
    if value is None:
        raise OrderValidationError(f"{order.order_type.value} order {order.identifier} needs {name}")
    return value


def _pricing(order: Order, effect: str) -> dict[str, Any]:
    match order.order_type:
        case OrderType.MARKET:
            return {"order-type": "Market"}
        case OrderType.LIMIT:
            return {"order-type": "Limit", "price": _price(_require(order.limit_price, "limit_price", order)), "price-effect": effect}
        case OrderType.STOP:
            return {"order-type": "Stop", "stop-trigger": _price(_require(order.stop_price, "stop_price", order))}
        case OrderType.STOP_LIMIT:
            return {
                "order-type": "Stop Limit",
                "stop-trigger": _price(_require(order.stop_price, "stop_price", order)),
                "price": _price(_require(order.stop_limit_price, "stop_limit_price", order)),
                "price-effect": effect,
            }
    raise OrderValidationError(f"Tastytrade cannot place {order.order_type} orders")


def build_order_json(order: Order) -> dict[str, Any]:
    """The `POST /accounts/{a}/orders` body for an already-conformed `order`."""
    buy = order.side is OrderSide.BUY
    effect = "Debit" if buy else "Credit"
    regular, extended = _TIME_IN_FORCE[order.time_in_force]
    leg: dict[str, Any] = {"instrument-type": "Equity", "symbol": to_tastytrade_symbol(order.asset.symbol), "action": "Buy to Open" if buy else "Sell to Close"}
    body: dict[str, Any] = {"time-in-force": extended if order.extended_hours and extended else regular, "source": SOURCE}
    if order.client_order_id:
        body["external-identifier"] = order.client_order_id
    if order.notional is not None:
        body.update({"order-type": "Notional Market", "value": format(order.notional.quantize(_PENNY, rounding=ROUND_FLOOR), "f"), "value-effect": effect})
    else:
        leg["quantity"] = format(_require(order.quantity, "quantity", order), "f")
        body.update(_pricing(order, effect))
    body["legs"] = [leg]
    return body


def build_replace_json(order: Order) -> dict[str, Any]:
    """The `PUT /accounts/{a}/orders/{id}` body: the order without its legs (they can't change on a replace)."""
    body = build_order_json(order)
    del body["legs"]
    return body
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/tastytrade/test_orders_build.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/brokers/tastytrade/orders.py tests/brokers/tastytrade/test_orders_build.py
git commit -m "feat(tastytrade): pure order rules and order JSON (Task 4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Pure status map and order/position/fill parsing

**Files:**
- Modify: `src/trading_agent_framework/brokers/tastytrade/orders.py` (append)
- Modify: `tests/fakes.py` (append Tastytrade JSON builders)
- Test: `tests/brokers/tastytrade/test_orders_parse.py`

**Interfaces:**
- Consumes: `values.decimal_of`, `values.optional_decimal` (Task 3); `orders.identifier_from_external`, `orders.from_tastytrade_symbol` (Task 4).
- Produces (in `orders`): `TERMINAL_STATUSES: frozenset[str]`; `CANCELED_STATUSES: frozenset[str]`; `map_status(status: str, *, has_fills: bool) -> OrderStatus`; `Fill(fill_id: str, quantity: Decimal, price: Decimal)` (frozen dataclass); `parse_fills(data: Mapping[str, Any]) -> list[Fill]`; `is_parsable_order(data: Mapping[str, Any]) -> bool`; `parse_order(data: Mapping[str, Any], strategy_name: str) -> Order`; `parse_position(data: Mapping[str, Any], strategy_name: str) -> Position | None`.
- Produces (in `tests/fakes.py`): `make_tt_fill(fill_id="f1", quantity="10", price="150.0") -> dict`; `make_tt_order(*, order_id=1001, status="Live", action="Buy to Open", symbol="AAPL", quantity="10", fills=(), external="s:abc", **overrides) -> dict`; `make_tt_position(*, symbol="AAPL", quantity="10", direction="Long", average_open_price="150.0", instrument_type="Equity") -> dict`.

If Task 1's findings list different JSON field names, use the real names in both the fixtures and the parsing below.

- [ ] **Step 1: Add the fixtures to `tests/fakes.py`**

Append (add `Any` and `Sequence` to the existing `typing`/`collections.abc` imports if missing):

```python
# --- Tastytrade JSON (dasherized, as the REST API and the account streamer send it) --------


def make_tt_fill(fill_id: str = "f1", quantity: str = "10", price: str = "150.0") -> dict[str, Any]:
    return {"fill-id": fill_id, "quantity": quantity, "fill-price": price, "filled-at": "2026-10-15T14:00:00.000+00:00", "destination-venue": "TEST"}


def make_tt_order(
    *,
    order_id: int = 1001,
    status: str = "Live",
    action: str = "Buy to Open",
    symbol: str = "AAPL",
    quantity: str | None = "10",
    fills: Sequence[dict[str, Any]] = (),
    external: str | None = "s:abc",
    **overrides: Any,
) -> dict[str, Any]:
    """A Tastytrade order snapshot; `overrides` use the API's dasherized keys (pass them with `**{"order-type": ...}`)."""
    leg: dict[str, Any] = {"instrument-type": "Equity", "symbol": symbol, "action": action, "fills": list(fills)}
    order: dict[str, Any] = {
        "id": order_id,
        "account-number": "5WT00001",
        "time-in-force": "Day",
        "order-type": "Limit",
        "price": "150.0",
        "price-effect": "Debit" if action.startswith("Buy") else "Credit",
        "status": status,
        "underlying-symbol": symbol,
        "underlying-instrument-type": "Equity",
        "cancellable": True,
        "editable": True,
        "edited": False,
        "source": "trading-agent-framework",
        "legs": [leg],
    }
    if quantity is not None:
        leg["quantity"] = leg["remaining-quantity"] = quantity
        order["size"] = quantity
    if external is not None:
        order["external-identifier"] = external
    order.update(overrides)
    return order


def make_tt_position(*, symbol: str = "AAPL", quantity: str = "10", direction: str = "Long", average_open_price: str = "150.0", instrument_type: str = "Equity") -> dict[str, Any]:
    return {
        "account-number": "5WT00001",
        "symbol": symbol,
        "instrument-type": instrument_type,
        "underlying-symbol": symbol,
        "quantity": quantity,
        "quantity-direction": direction,
        "average-open-price": average_open_price,
        "close-price": "151.0",
        "multiplier": 1,
    }
```

- [ ] **Step 2: Write the failing tests**

`tests/brokers/tastytrade/test_orders_parse.py`:

```python
from __future__ import annotations

import logging
from decimal import Decimal

import pytest
from tests.fakes import make_tt_fill, make_tt_order, make_tt_position

from trading_agent_framework.brokers.tastytrade import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce
from trading_agent_framework.utils.errors import BrokerError

D = Decimal


@pytest.mark.parametrize(
    ("status", "has_fills", "expected"),
    [
        ("Received", False, OrderStatus.SUBMITTED),
        ("Routed", False, OrderStatus.SUBMITTED),
        ("In Flight", False, OrderStatus.SUBMITTED),
        ("Contingent", False, OrderStatus.SUBMITTED),
        ("Live", False, OrderStatus.NEW),
        ("Live", True, OrderStatus.PARTIAL_FILL),
        ("Replace Requested", False, OrderStatus.NEW),
        ("Cancel Requested", False, OrderStatus.CANCELLING),
        ("Filled", True, OrderStatus.FILL),
        ("Cancelled", False, OrderStatus.CANCELED),
        ("Removed", False, OrderStatus.CANCELED),
        ("Partially Removed", True, OrderStatus.CANCELED),
        ("Expired", False, OrderStatus.EXPIRED),
        ("Rejected", False, OrderStatus.ERROR),
    ],
)
def test_status_map(status: str, has_fills: bool, expected: OrderStatus) -> None:
    assert orders.map_status(status, has_fills=has_fills) is expected


def test_an_unknown_status_is_unknown_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert orders.map_status("Teleported", has_fills=False) is OrderStatus.UNKNOWN
    assert "Teleported" in caplog.text


def test_parse_a_live_limit_buy_of_ours() -> None:
    order = orders.parse_order(make_tt_order(), "s")

    assert (order.identifier, order.client_order_id) == ("abc", "s:abc")
    assert (order.asset, order.side, order.order_type, order.time_in_force) == (Asset("AAPL"), OrderSide.BUY, OrderType.LIMIT, TimeInForce.DAY)
    assert (order.quantity, order.limit_price, order.status) == (D(10), D(150), OrderStatus.NEW)
    assert order.raw["id"] == 1001


def test_another_strategys_order_is_identified_by_its_tastytrade_id() -> None:
    assert orders.parse_order(make_tt_order(external="other:xyz"), "s").identifier == "1001"
    assert orders.parse_order(make_tt_order(external=None), "s").client_order_id is None


def test_parse_a_stop_limit_sell() -> None:
    order = orders.parse_order(make_tt_order(action="Sell to Close", **{"order-type": "Stop Limit", "price": "149.5", "stop-trigger": "150"}), "s")

    assert (order.side, order.order_type, order.stop_price, order.stop_limit_price, order.limit_price) == (OrderSide.SELL, OrderType.STOP_LIMIT, D(150), D("149.5"), None)


def test_parse_a_notional_order() -> None:
    order = orders.parse_order(make_tt_order(quantity=None, **{"order-type": "Notional Market", "value": "250.0", "value-effect": "Debit"}), "s")

    assert (order.order_type, order.notional, order.quantity) == (OrderType.MARKET, D(250), None)


def test_fills_set_the_filled_quantity_and_average_price() -> None:
    order = orders.parse_order(make_tt_order(status="Filled", fills=[make_tt_fill("f1", "4", "150"), make_tt_fill("f2", "6", "151")]), "s")

    assert (order.filled_quantity, order.avg_fill_price, order.status) == (D(10), D("150.6"), OrderStatus.FILL)


def test_a_rejected_order_carries_its_reason() -> None:
    order = orders.parse_order(make_tt_order(status="Rejected", **{"reject-reason": "Insufficient buying power"}), "s")

    assert (order.status, order.error_message) == (OrderStatus.ERROR, "Insufficient buying power")


def test_an_extended_hours_order() -> None:
    order = orders.parse_order(make_tt_order(**{"time-in-force": "Ext"}), "s")

    assert (order.time_in_force, order.extended_hours) == (TimeInForce.DAY, True)


def test_parse_order_and_position_map_class_share_symbols_back() -> None:
    assert orders.parse_order(make_tt_order(symbol="BRK/B"), "s").asset == Asset("BRK.B")
    position = orders.parse_position(make_tt_position(symbol="BRK/B"), "s")
    assert position is not None
    assert position.asset == Asset("BRK.B")


def test_unsupported_orders_are_not_parsable() -> None:
    assert orders.is_parsable_order(make_tt_order())
    assert not orders.is_parsable_order(make_tt_order(**{"order-type": "Marketable Limit"}))
    option = make_tt_order()
    option["legs"][0]["instrument-type"] = "Equity Option"
    assert not orders.is_parsable_order(option)
    with pytest.raises(BrokerError, match="Marketable Limit"):
        orders.parse_order(make_tt_order(**{"order-type": "Marketable Limit"}), "s")


def test_parse_fills_reads_every_leg_fill_and_rejects_a_fill_without_an_id() -> None:
    assert orders.parse_fills(make_tt_order(fills=[make_tt_fill("f1", "4", "150.5")])) == [orders.Fill("f1", D(4), D("150.5"))]
    broken = make_tt_order(fills=[{"quantity": "1", "fill-price": "1"}])
    with pytest.raises(BrokerError, match="fill-id"):
        orders.parse_fills(broken)


def test_parse_a_fractional_long_position() -> None:
    position = orders.parse_position(make_tt_position(quantity="10.5", average_open_price="150.25"), "s")

    assert position is not None
    assert (position.strategy_name, position.quantity, position.side, position.avg_fill_price) == ("s", D("10.5"), PositionSide.LONG, D("150.25"))


def test_a_short_position_has_a_positive_quantity() -> None:
    position = orders.parse_position(make_tt_position(quantity="5", direction="Short"), "s")

    assert position is not None
    assert (position.quantity, position.side) == (D(5), PositionSide.SHORT)


def test_closed_and_non_equity_positions_are_dropped() -> None:
    assert orders.parse_position(make_tt_position(quantity="0", direction="Zero"), "s") is None
    assert orders.parse_position(make_tt_position(instrument_type="Equity Option"), "s") is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/tastytrade/test_orders_parse.py -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'map_status'`.

- [ ] **Step 4: Implement (append to `orders.py`)**

Add to the imports of `orders.py`:

```python
from collections.abc import Mapping
from dataclasses import dataclass

from trading_agent_framework.brokers.tastytrade.values import decimal_of, optional_decimal
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce
from trading_agent_framework.entities.position import Position
from trading_agent_framework.utils.errors import BrokerError, OrderValidationError
```

(merge with the existing `enums`/`errors` imports), then append:

```python
# --- statuses ------------------------------------------------------------------------------

TERMINAL_STATUSES: frozenset[str] = frozenset({"Filled", "Cancelled", "Rejected", "Expired", "Removed", "Partially Removed"})
# Statuses that end an order without (more) fills; Alpaca's `expired` maps to a cancel the same way.
CANCELED_STATUSES: frozenset[str] = frozenset({"Cancelled", "Removed", "Partially Removed", "Expired"})

_SUBMITTED = frozenset({"Received", "Routed", "In Flight", "Contingent"})
_WORKING = frozenset({"Live", "Replace Requested"})
_STATUSES: dict[str, OrderStatus] = {
    "Cancel Requested": OrderStatus.CANCELLING,
    "Filled": OrderStatus.FILL,
    "Cancelled": OrderStatus.CANCELED,
    "Removed": OrderStatus.CANCELED,
    "Partially Removed": OrderStatus.CANCELED,
    "Expired": OrderStatus.EXPIRED,
    "Rejected": OrderStatus.ERROR,
}


def map_status(status: str, *, has_fills: bool) -> OrderStatus:
    if status in _SUBMITTED:
        return OrderStatus.SUBMITTED
    if status in _WORKING:
        return OrderStatus.PARTIAL_FILL if has_fills else OrderStatus.NEW
    mapped = _STATUSES.get(status)
    if mapped is None:
        logger.warning("Unknown Tastytrade order status %r", status)
        return OrderStatus.UNKNOWN
    return mapped


# --- parsing -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    quantity: Decimal
    price: Decimal


def parse_fills(data: Mapping[str, Any]) -> list[Fill]:
    fills: list[Fill] = []
    for leg in data.get("legs") or []:
        for fill in leg.get("fills") or []:
            fill_id = fill.get("fill-id")
            if fill_id is None:
                raise BrokerError(f"Tastytrade order {data.get('id')} has a fill without a fill-id")
            fills.append(Fill(str(fill_id), abs(decimal_of(fill.get("quantity"), "fill quantity")), abs(decimal_of(fill.get("fill-price"), "fill price"))))
    return fills


_ORDER_TYPES: dict[str, OrderType] = {
    "Market": OrderType.MARKET,
    "Notional Market": OrderType.MARKET,
    "Limit": OrderType.LIMIT,
    "Stop": OrderType.STOP,
    "Stop Limit": OrderType.STOP_LIMIT,
}
_TIME_IN_FORCE_BACK: dict[str, tuple[TimeInForce, bool]] = {
    "Day": (TimeInForce.DAY, False),
    "Ext": (TimeInForce.DAY, True),
    "GTC": (TimeInForce.GTC, False),
    "GTC Ext": (TimeInForce.GTC, True),
    "IOC": (TimeInForce.IOC, False),
}


def is_parsable_order(data: Mapping[str, Any]) -> bool:
    """A single-leg equity order of a type we model (manual option or `Marketable Limit` orders are not)."""
    legs = data.get("legs") or []
    return len(legs) == 1 and legs[0].get("instrument-type") == "Equity" and data.get("order-type") in _ORDER_TYPES


def parse_order(data: Mapping[str, Any], strategy_name: str) -> Order:
    """An `Order` from a Tastytrade order snapshot (REST or account streamer)."""
    raw_type = str(data.get("order-type"))
    order_type = _ORDER_TYPES.get(raw_type)
    legs = data.get("legs") or []
    if order_type is None or not legs:
        raise BrokerError(f"Tastytrade order {data.get('id')} ({raw_type}, {len(legs)} legs) is not a single-leg order this framework models")
    leg = legs[0]
    time_in_force, extended = _TIME_IN_FORCE_BACK.get(str(data.get("time-in-force")), (TimeInForce.DAY, False))
    side = OrderSide.BUY if str(leg.get("action", "")).startswith("Buy") else OrderSide.SELL
    asset = Asset(from_tastytrade_symbol(str(leg.get("symbol") or data.get("underlying-symbol"))))
    if raw_type == "Notional Market":
        order = Order(strategy_name, asset, side, order_type, notional=abs(decimal_of(data.get("value"), "order value")), time_in_force=time_in_force, extended_hours=extended)
    else:
        quantity = abs(decimal_of(leg.get("quantity", data.get("size")), "order quantity"))
        order = Order(strategy_name, asset, side, order_type, quantity=quantity, time_in_force=time_in_force, extended_hours=extended)
    price = optional_decimal(data.get("price"), "order price")
    trigger = optional_decimal(data.get("stop-trigger"), "order stop-trigger")
    if order_type is OrderType.LIMIT:
        order.limit_price = abs(price) if price is not None else None
    elif order_type is OrderType.STOP:
        order.stop_price = abs(trigger) if trigger is not None else None
    elif order_type is OrderType.STOP_LIMIT:
        order.stop_price = abs(trigger) if trigger is not None else None
        order.stop_limit_price = abs(price) if price is not None else None
    external = data.get("external-identifier")
    order.identifier = identifier_from_external(external, strategy_name) or str(data.get("id"))
    order.client_order_id = str(external) if external else None
    fills = parse_fills(data)
    if fills:
        order.filled_quantity = sum((f.quantity for f in fills), Decimal(0))
        order.avg_fill_price = sum((f.quantity * f.price for f in fills), Decimal(0)) / order.filled_quantity
    order.status = map_status(str(data.get("status", "")), has_fills=bool(fills))
    if order.status is OrderStatus.ERROR:
        order.error_message = str(data.get("reject-reason") or "rejected by Tastytrade")
    order.update_raw(dict(data))
    return order


def parse_position(data: Mapping[str, Any], strategy_name: str) -> Position | None:
    """An equity `Position` (shorts have a positive quantity and `side=SHORT`); None for closed or non-equity ones."""
    if data.get("instrument-type") != "Equity":
        return None
    quantity = abs(decimal_of(data.get("quantity"), "position quantity"))
    direction = data.get("quantity-direction")
    if direction == "Zero" or quantity == 0:
        return None
    return Position(
        strategy_name,
        Asset(from_tastytrade_symbol(str(data.get("symbol")))),
        quantity,
        PositionSide.SHORT if direction == "Short" else PositionSide.LONG,
        avg_fill_price=optional_decimal(data.get("average-open-price"), "position average-open-price"),
        raw=dict(data),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/tastytrade -q`
Expected: PASS.

- [ ] **Step 6: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/brokers/tastytrade/orders.py tests/fakes.py tests/brokers/tastytrade/test_orders_parse.py
git commit -m "feat(tastytrade): pure status map and order/position parsing (Task 5)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `TastytradeSession` (HTTP, auth, retries)

**Files:**
- Create: `src/trading_agent_framework/brokers/tastytrade/client.py`
- Test: `tests/brokers/tastytrade/test_client.py`

**Interfaces:**
- Consumes: `TastytradeSettings` (Task 2); `BrokerUncertainError` (Task 2).
- Produces: `client.USER_AGENT: str`; `client.TastytradeHTTPError(status_code: int, message: str)` (a `BrokerError`, attribute `status_code`); `client.TastytradeSession(settings, *, transport: httpx.BaseTransport | None = None, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, timeout: float = 10.0)` with `access_token(*, force: bool = False) -> str`, `get(path, *, params=None) -> dict`, `post(path, *, json=None) -> dict`, `put(path, *, json=None) -> dict`, `delete(path) -> dict`, `get_all(path, *, params=None, max_items: int | None = None) -> list[dict]`, `close() -> None`. `get/post/put/delete` return the response's `data` object (`{}` for an empty body).

- [ ] **Step 1: Write the failing tests**

`tests/brokers/tastytrade/test_client.py`:

```python
from __future__ import annotations

import json

import httpx
import pytest

from trading_agent_framework.brokers.tastytrade.client import TastytradeHTTPError, TastytradeSession
from trading_agent_framework.config.env import TastytradeSettings
from trading_agent_framework.utils.errors import BrokerUncertainError

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/tastytrade/test_client.py -q`
Expected: FAIL with `ModuleNotFoundError: ... tastytrade.client`.

- [ ] **Step 3: Implement**

`src/trading_agent_framework/brokers/tastytrade/client.py`:

```python
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
        return _data(self._send("POST", path, json=json))

    def put(self, path: str, *, json: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return _data(self._send("PUT", path, json=json))

    def delete(self, path: str) -> dict[str, Any]:
        return _data(self._send("DELETE", path))

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
            except httpx.TransportError as exc:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/tastytrade/test_client.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/brokers/tastytrade/client.py tests/brokers/tastytrade/test_client.py
git commit -m "feat(tastytrade): TastytradeSession with token refresh and safe retries (Task 6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `TastytradeOrderEvents`

**Files:**
- Create: `src/trading_agent_framework/brokers/tastytrade/events.py`
- Test: `tests/brokers/tastytrade/test_events.py`

**Interfaces:**
- Consumes: `orders.parse_fills`, `orders.CANCELED_STATUSES` (Task 5); `OrderTracker`.
- Produces: `TastytradeOrderEvents(tracker: OrderTracker)` with `register(order: Order, tt_id: str, *, snapshot: Mapping[str, Any] | None = None) -> None`, `tt_id_of(order: Order) -> str | None`, `begin_replace(order: Order) -> str | None`, `abort_replace(tt_id: str) -> None`, `finish_replace(order: Order, new_tt_id: str) -> None`, `apply(data: Mapping[str, Any]) -> bool`.

- [ ] **Step 1: Write the failing tests**

`tests/brokers/tastytrade/test_events.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest
from tests.fakes import make_tt_fill, make_tt_order

from trading_agent_framework.brokers.tastytrade.events import TastytradeOrderEvents
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order

D = Decimal


@pytest.fixture
def tracker() -> OrderTracker:
    return OrderTracker()


@pytest.fixture
def events(tracker: OrderTracker) -> TastytradeOrderEvents:
    return TastytradeOrderEvents(tracker)


@pytest.fixture
def seen(tracker: OrderTracker) -> list[OrderEvent]:
    received: list[OrderEvent] = []
    tracker.listeners.append(lambda order, event: received.append(event))
    return received


def _tracked(tracker: OrderTracker, events: TastytradeOrderEvents, *, notional: str | None = None, register: bool = True) -> Order:
    if notional is None:
        order = Order("s", Asset("AAPL"), OrderSide.BUY, OrderType.LIMIT, quantity=D(10), limit_price=D(150), identifier="abc", client_order_id="s:abc")
    else:
        order = Order("s", Asset("AAPL"), OrderSide.BUY, OrderType.MARKET, notional=D(notional), identifier="abc", client_order_id="s:abc")
    order.status = OrderStatus.SUBMITTED
    tracker.track_unprocessed(order)
    if register:
        events.register(order, "1001")
    return order


def test_live_makes_a_submitted_order_new(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)

    assert events.apply(make_tt_order(status="Live"))
    assert order.status is OrderStatus.NEW
    assert seen == [OrderEvent.NEW]
    assert not events.apply(make_tt_order(status="Live"))  # already new: nothing to do


def test_an_order_that_is_not_ours_is_ignored(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    _tracked(tracker, events)

    assert not events.apply(make_tt_order(order_id=9999, external="other:xyz"))


def test_an_unregistered_id_falls_back_to_the_external_identifier(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, register=False)

    assert events.apply(make_tt_order(order_id=1001, status="Live"))
    assert order.status is OrderStatus.NEW


def test_a_partial_fill_then_the_rest(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)
    first = make_tt_fill("f1", "4", "150")

    events.apply(make_tt_order(status="Live", fills=[first]))
    assert (order.status, order.filled_quantity) == (OrderStatus.PARTIAL_FILL, D(4))

    final = make_tt_order(status="Filled", fills=[first, make_tt_fill("f2", "6", "151")])
    events.apply(final)
    assert (order.status, order.filled_quantity, len(order.transactions)) == (OrderStatus.FILL, D(10), 2)

    assert not events.apply(final)  # a replayed snapshot changes nothing
    assert order.filled_quantity == D(10)
    assert seen == [OrderEvent.PARTIALLY_FILLED, OrderEvent.FILLED]


@pytest.mark.parametrize("status", ["Cancelled", "Removed", "Partially Removed", "Expired"])
def test_ending_statuses_cancel(tracker: OrderTracker, events: TastytradeOrderEvents, status: str) -> None:
    order = _tracked(tracker, events)

    events.apply(make_tt_order(status=status))

    assert order.status is OrderStatus.CANCELED


def test_a_rejection_carries_its_reason(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)

    events.apply(make_tt_order(status="Rejected", **{"reject-reason": "Insufficient buying power"}))

    assert (order.status, order.error_message) == (OrderStatus.ERROR, "Insufficient buying power")
    assert seen == [OrderEvent.ERROR]


def test_cancel_requested_marks_the_order_cancelling(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)

    events.apply(make_tt_order(status="Cancel Requested"))

    assert order.status is OrderStatus.CANCELLING


def test_a_cancel_of_the_replaced_id_racing_the_put_is_ignored(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.apply(make_tt_order(status="Live"))

    assert events.begin_replace(order) == "1001"
    events.apply(make_tt_order(status="Cancelled"))  # the original's cancel arrives before the PUT returns
    assert order.status is OrderStatus.NEW

    events.finish_replace(order, "1002")
    assert events.tt_id_of(order) == "1002"
    events.apply(make_tt_order(order_id=1002, status="Filled", fills=[make_tt_fill("f1", "10", "151")]))
    assert order.status is OrderStatus.FILL


def test_fills_on_a_replaced_id_still_count(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.begin_replace(order)
    events.finish_replace(order, "1002")

    events.apply(make_tt_order(status="Cancelled", fills=[make_tt_fill("f9", "2", "150")]))

    assert (order.filled_quantity, order.status) == (D(2), OrderStatus.PARTIAL_FILL)


def test_an_aborted_replace_listens_to_the_original_again(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    tt_id = events.begin_replace(order)
    assert tt_id is not None
    events.abort_replace(tt_id)

    events.apply(make_tt_order(status="Cancelled"))

    assert order.status is OrderStatus.CANCELED


def test_a_replace_that_keeps_its_id_is_not_superseded(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.begin_replace(order)
    events.finish_replace(order, "1001")

    events.apply(make_tt_order(status="Cancelled"))

    assert order.status is OrderStatus.CANCELED


def test_a_notional_orders_fills_wait_for_filled(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, notional="250")

    events.apply(make_tt_order(status="Live", quantity=None, fills=[make_tt_fill("f1", "1.2", "150")]))
    assert (order.status, order.filled_quantity) == (OrderStatus.NEW, D(0))  # Live counts, the fill waits

    events.apply(make_tt_order(status="Filled", quantity=None, fills=[make_tt_fill("f1", "1.2", "150"), make_tt_fill("f2", "0.46", "151")]))
    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D("1.66"))


def test_registering_with_a_snapshot_marks_its_fills_as_counted(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, register=False)
    snapshot = make_tt_order(status="Live", fills=[make_tt_fill("f1", "4", "150")])
    order.filled_quantity = D(4)  # as parse_order sets it for an adopted order
    events.register(order, "1001", snapshot=snapshot)

    events.apply(snapshot)

    assert order.filled_quantity == D(4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/tastytrade/test_events.py -q`
Expected: FAIL with `ModuleNotFoundError: ... tastytrade.events`.

- [ ] **Step 3: Implement**

`src/trading_agent_framework/brokers/tastytrade/events.py`:

```python
"""`TastytradeOrderEvents`: applies Tastytrade order snapshots (account streamer or REST) to `OrderTracker`.

Feeds the tracker only -- never strategy hooks. Snapshots are full order objects, so applying one
twice is harmless: fills are counted once by `fill-id`. Orders are found by Tastytrade order id
(falling back to `external-identifier`). A replace gives the order a new id; the old one is
"superseded" from the moment the replace is sent: its fills still count, its statuses are ignored,
so the original's `Cancelled` never cancels the modified order.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from trading_agent_framework.brokers.tastytrade import orders
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.enums import OrderEvent, OrderStatus
from trading_agent_framework.entities.order import Order

_NEW_FROM = frozenset({OrderStatus.UNPROCESSED, OrderStatus.SUBMITTED})


class TastytradeOrderEvents:
    def __init__(self, tracker: OrderTracker) -> None:
        self._tracker = tracker
        # Reentrant, held across a whole apply(): the stream thread and the strategy thread
        # (submit, reconcile) may apply the same snapshot concurrently.
        self._lock = threading.RLock()
        self._current: dict[str, str] = {}  # our identifier -> current Tastytrade id
        self._orders: dict[str, Order] = {}  # every registered Tastytrade id -> our order
        self._superseded: set[str] = set()
        self._seen_fills: set[str] = set()

    def register(self, order: Order, tt_id: str, *, snapshot: Mapping[str, Any] | None = None) -> None:
        """Map `order` to its Tastytrade id; `snapshot` marks its fills as already counted (an adopted order)."""
        with self._lock:
            self._current[order.identifier] = tt_id
            self._orders[tt_id] = order
            if snapshot is not None:
                self._seen_fills.update(f.fill_id for f in orders.parse_fills(snapshot))

    def tt_id_of(self, order: Order) -> str | None:
        with self._lock:
            return self._current.get(order.identifier)

    def begin_replace(self, order: Order) -> str | None:
        """Supersede the current id before a replace is sent; returns it (None when unknown)."""
        with self._lock:
            tt_id = self._current.get(order.identifier)
            if tt_id is not None:
                self._superseded.add(tt_id)
            return tt_id

    def abort_replace(self, tt_id: str) -> None:
        with self._lock:
            self._superseded.discard(tt_id)

    def finish_replace(self, order: Order, new_tt_id: str) -> None:
        with self._lock:
            old = self._current.get(order.identifier)
            self._current[order.identifier] = new_tt_id
            self._orders[new_tt_id] = order
            if new_tt_id == old:  # replaced in place: still the live id
                self._superseded.discard(new_tt_id)

    def apply(self, data: Mapping[str, Any]) -> bool:
        """Apply one order snapshot; True when it changed one of our orders."""
        tt_id = str(data.get("id", ""))
        status = str(data.get("status", ""))
        with self._lock:
            order = self._orders.get(tt_id)
            if order is None:
                external = data.get("external-identifier")
                order = self._tracker.get_tracked_order_by_client_order_id(str(external)) if external else None
                if order is None:
                    return False
            changed = self._apply_fills(order, data, status)
            if tt_id in self._superseded:
                return changed
            return self._apply_status(order, data, status) or changed

    def _apply_fills(self, order: Order, data: Mapping[str, Any], status: str) -> bool:
        if order.notional is not None and status != "Filled":
            return False  # a notional order has no quantity to compare fills against: wait for Filled
        new = [f for f in orders.parse_fills(data) if f.fill_id not in self._seen_fills]
        self._seen_fills.update(f.fill_id for f in new)
        for index, fill in enumerate(new):
            if order.notional is not None:
                complete = index == len(new) - 1
            else:
                complete = order.quantity is not None and order.filled_quantity + fill.quantity >= order.quantity
            event = OrderEvent.FILLED if complete else OrderEvent.PARTIALLY_FILLED
            self._tracker.process_trade_event(order, event, price=fill.price, filled_quantity=fill.quantity)
        if new:
            order.update_raw(dict(data))
        return bool(new)

    def _apply_status(self, order: Order, data: Mapping[str, Any], status: str) -> bool:
        if status == "Live" and order.status in _NEW_FROM:
            event = OrderEvent.NEW
        elif status in orders.CANCELED_STATUSES and order.is_active():
            event = OrderEvent.CANCELED
        elif status == "Rejected" and order.is_active():
            order.set_error(str(data.get("reject-reason") or "rejected by Tastytrade"))
            event = OrderEvent.ERROR
        elif status == "Cancel Requested" and order.is_active() and order.status is not OrderStatus.CANCELLING:
            order.status = OrderStatus.CANCELLING
            order.update_raw(dict(data))
            return True
        else:
            return False
        order.update_raw(dict(data))
        self._tracker.process_trade_event(order, event)
        return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/tastytrade/test_events.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/brokers/tastytrade/events.py tests/brokers/tastytrade/test_events.py
git commit -m "feat(tastytrade): order snapshots feed the OrderTracker, replaces included (Task 7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `TastytradeAccountStream`

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Create: `src/trading_agent_framework/brokers/tastytrade/stream.py`
- Test: `tests/brokers/tastytrade/test_stream.py`

**Interfaces:**
- Consumes: `client.USER_AGENT` (Task 6), imported inside `_default_connect`.
- Produces: `stream.StreamSocket` (Protocol: `send(str)`, `recv(timeout: float | None = None) -> str`, `close()`); `stream.StreamError`; `stream.TastytradeAccountStream(url: str, account_number: str, token: Callable[[], str], on_order: Callable[[Mapping[str, Any]], object], on_reconnect: Callable[[], None], *, connect: Callable[[str], StreamSocket] = _default_connect, clock: Callable[[], float] = time.monotonic, wait: Callable[[float], bool] | None = None)` with `start()`, `stop(timeout: float = 5.0)`, `run()`, property `is_running: bool`.

- [ ] **Step 1: Make `websockets` a direct dependency**

Run: `uv add "websockets>=16.1,<17"`
Expected: `pyproject.toml` lists `websockets>=16.1,<17`; `uv.lock` unchanged apart from the direct-dependency marker (16.1.1 already locked).

- [ ] **Step 2: Write the failing tests**

`tests/brokers/tastytrade/test_stream.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/tastytrade/test_stream.py -q`
Expected: FAIL with `ModuleNotFoundError: ... tastytrade.stream`.

- [ ] **Step 4: Implement**

`src/trading_agent_framework/brokers/tastytrade/stream.py`:

```python
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
    def send(self, message: str) -> None: ...

    def recv(self, timeout: float | None = None) -> str: ...

    def close(self) -> None: ...


class StreamError(Exception):
    """The streamer refused a request or reported an error: reconnect."""


def _default_connect(url: str) -> StreamSocket:
    from websockets.sync.client import connect

    from trading_agent_framework.brokers.tastytrade.client import USER_AGENT

    return connect(url, user_agent_header=USER_AGENT, open_timeout=CONNECT_TIMEOUT)  # ty: ignore[invalid-return-type]


class TastytradeAccountStream:
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
        self._url = url
        self._account_number = account_number
        self._token = token
        self._on_order = on_order
        self._on_reconnect = on_reconnect
        self._connect = connect
        self._clock = clock
        self._stopped = threading.Event()
        self._wait = wait if wait is not None else self._stopped.wait
        self._socket: StreamSocket | None = None
        self._thread: threading.Thread | None = None
        self._request_ids = itertools.count(1)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stopped.clear()
        self._thread = threading.Thread(target=self.run, name="tastytrade-account-stream", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
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
                    failures = 0
                    if not first_attempt:
                        self._reconcile()
                    self._serve(socket)
                finally:
                    self._socket = None
                    try:
                        socket.close()
                    except Exception:
                        logger.debug("closing the Tastytrade account-streamer socket failed", exc_info=True)
            except Exception:
                if self._stopped.is_set():
                    break
                logger.warning("Tastytrade account stream disconnected; reconnecting", exc_info=True)
            first_attempt = False
            if self._stopped.is_set():
                break
            failures += 1
            self._wait(min(MAX_BACKOFF, 2.0 ** (failures - 1)))

    def _reconcile(self) -> None:
        try:
            self._on_reconnect()
        except Exception:
            logger.exception("reconciling Tastytrade orders after a reconnect failed")

    def _send(self, socket: StreamSocket, action: str, value: object = None) -> int:
        request_id = next(self._request_ids)
        message: dict[str, Any] = {"action": action}
        if value is not None:
            message["value"] = value
        message["auth-token"] = f"Bearer {self._token()}"
        message["request-id"] = request_id
        socket.send(json.dumps(message))
        return request_id

    def _subscribe(self, socket: StreamSocket) -> None:
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
        try:
            message = json.loads(raw)  # ty: ignore[invalid-argument-type]
        except (TypeError, ValueError):
            logger.warning("unreadable account-streamer message: %r", str(raw)[:200])
            return {}
        return message if isinstance(message, dict) else {}
```

Note on `test_heartbeats_go_out_on_schedule_with_a_fresh_token`: the heartbeat dict is built in insertion order `action, value, auth-token, request-id`, and the test compares dicts, so key order does not matter.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/tastytrade/test_stream.py -q`
Expected: PASS.

- [ ] **Step 6: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add pyproject.toml uv.lock src/trading_agent_framework/brokers/tastytrade/stream.py tests/brokers/tastytrade/test_stream.py
git commit -m "feat(tastytrade): account-streamer thread with heartbeats and reconnects (Task 8)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `TastytradeBroker`, factory wiring and lazy import

**Files:**
- Create: `src/trading_agent_framework/brokers/tastytrade/broker.py`
- Modify: `src/trading_agent_framework/brokers/factory.py`, `src/trading_agent_framework/brokers/__init__.py`
- Modify: `tests/fakes.py` (append `FakeTastytradeSession`)
- Test: `tests/brokers/tastytrade/test_broker.py`, `tests/brokers/test_factory.py`, `tests/brokers/test_lazy_imports.py`

**Interfaces:**
- Consumes: everything above: `account.*`, `orders.*`, `TastytradeSession`/`TastytradeHTTPError`, `TastytradeOrderEvents`, `TastytradeAccountStream`/`StreamSocket`, `TastytradeSettings`, `AlpacaMarketData`, `AlpacaMarketClock`, `lazy_news_provider`.
- Produces: `TastytradeBroker(strategy_name, session, *, account_number: str, streamer_url: str, market_data: AlpacaMarketData, tracker=None, clock=None, is_paper=True, news_provider_factory=None, stream_connect=None)`; `TastytradeBroker.from_settings(strategy_name, settings, *, data, news=None, session=None, market_data=None)`; `account_number` property; `configure_account()`; `reconcile()`; `brokers.TastytradeBroker` lazy export; `factory.BUILDERS[BrokerKind.TASTYTRADE]`; `tests.fakes.FakeTastytradeSession`.

- [ ] **Step 1: Add `FakeTastytradeSession` to `tests/fakes.py`**

```python
class FakeTastytradeSession:
    """Hand-written stand-in for `TastytradeSession`: canned answers per (method, path), every call recorded.

    An answer is a dict (returned), a callable `(json, params) -> dict` (called), or an exception (raised).
    `get_all` reads the answer's `items`. A missing route is a 404 `TastytradeHTTPError`, like the real API.
    """

    def __init__(self) -> None:
        self.responses: dict[tuple[str, str], Any] = {}
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.closed = False

    def _answer(self, method: str, path: str, json: Any = None, params: Any = None) -> Any:
        from trading_agent_framework.brokers.tastytrade.client import TastytradeHTTPError

        self.calls.append((method, path, json, params))
        if (method, path) not in self.responses:
            raise TastytradeHTTPError(404, f"Tastytrade {method} {path}: 404 not_found: no route")
        answer = self.responses[(method, path)]
        if callable(answer):
            answer = answer(json, params)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def posted(self, path: str) -> list[Any]:
        return [json for method, p, json, _ in self.calls if method == "POST" and p == path]

    def access_token(self, *, force: bool = False) -> str:
        return "token"

    def get(self, path: str, *, params: Any = None) -> dict[str, Any]:
        return self._answer("GET", path, None, params)

    def post(self, path: str, *, json: Any = None) -> dict[str, Any]:
        return self._answer("POST", path, json)

    def put(self, path: str, *, json: Any = None) -> dict[str, Any]:
        return self._answer("PUT", path, json)

    def delete(self, path: str) -> dict[str, Any]:
        return self._answer("DELETE", path) or {}

    def get_all(self, path: str, *, params: Any = None, max_items: int | None = None) -> list[dict[str, Any]]:
        items = list(self._answer("GET", path, None, params).get("items") or [])
        return items[:max_items] if max_items is not None else items

    def close(self) -> None:
        self.closed = True
```

- [ ] **Step 2: Write the failing tests**

`tests/brokers/tastytrade/test_broker.py`:

```python
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from tests.fakes import (
    FakeClock,
    FakeStockHistoricalDataClient,
    FakeTastytradeSession,
    FakeTradingClient,
    et,
    make_alpaca_trade,
    make_tt_fill,
    make_tt_order,
    make_tt_position,
)

from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker
from trading_agent_framework.brokers.tastytrade.client import TastytradeHTTPError
from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError, ConfigurationError, OrderValidationError

D = Decimal
ACCOUNT = "5WT00001"
ORDERS = f"/accounts/{ACCOUNT}/orders"
AAPL, MSFT = Asset("AAPL"), Asset("MSFT")
SETTINGS = TastytradeSettings(client_secret="secret", refresh_token="refresh", account_number=ACCOUNT, is_paper=True)
ACCOUNTS = {"items": [{"account": {"account-number": ACCOUNT, "margin-or-cash": "Cash", "is-closed": False}, "authority-level": "owner"}]}


@pytest.fixture
def session() -> FakeTastytradeSession:
    session = FakeTastytradeSession()
    session.responses[("GET", "/instruments/equities/AAPL")] = {"symbol": "AAPL", "is-fractional-quantity-eligible": True}
    session.responses[("GET", "/instruments/equities/MSFT")] = {"symbol": "MSFT", "is-fractional-quantity-eligible": False}
    session.responses[("GET", "/customers/me/accounts")] = ACCOUNTS
    session.responses[("POST", ORDERS)] = accept()
    return session


def _market_data() -> AlpacaMarketData:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 100.5), "MSFT": make_alpaca_trade("MSFT", 100.5)}
    return AlpacaMarketData(data, FakeTradingClient())


@pytest.fixture
def broker(session: FakeTastytradeSession) -> TastytradeBroker:
    return TastytradeBroker("s", session, account_number=ACCOUNT, streamer_url="wss://streamer.cert.tastyworks.com", market_data=_market_data(), clock=FakeClock(et(2026, 10, 15, 10)))


@pytest.fixture
def seen(broker: TastytradeBroker) -> list[OrderEvent]:
    received: list[OrderEvent] = []
    broker.tracker.listeners.append(lambda order, event: received.append(event))
    return received


def accept(*, order_id: int = 1001, status: str = "Routed", fills: tuple[dict[str, Any], ...] = ()) -> Any:
    """A POST answer echoing the submitted order back, as Tastytrade does."""

    def answer(json: dict[str, Any], params: Any) -> dict[str, Any]:
        leg = json["legs"][0]
        return {"order": make_tt_order(order_id=order_id, status=status, external=json["external-identifier"], symbol=leg["symbol"], action=leg["action"], quantity=leg.get("quantity"), fills=fills, **{"order-type": json["order-type"]}), "buying-power-effect": {}}

    return answer


def _buy(quantity: str = "10", **overrides: object) -> Order:
    fields: dict[str, object] = {"strategy_name": "s", "asset": AAPL, "side": OrderSide.BUY, "quantity": D(quantity)}
    fields.update(overrides)
    return Order(**fields)  # ty: ignore[invalid-argument-type]


def _limit(price: str = "150") -> Order:
    return _buy(order_type=OrderType.LIMIT, limit_price=D(price))


# --- submit ----------------------------------------------------------------------------------


def test_submit_posts_the_order_tagged_with_its_external_identifier(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())

    [body] = session.posted(ORDERS)
    assert body["external-identifier"] == f"s:{order.identifier}" == order.client_order_id
    assert (body["order-type"], body["price"], body["legs"][0]["quantity"]) == ("Limit", "150.00", "10")
    assert order.status is OrderStatus.SUBMITTED
    assert broker.tracker.get_tracked_order(order.identifier) is order


def test_eligibility_is_looked_up_once_per_symbol(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    broker.submit_order(_buy())
    broker.submit_order(_buy())

    assert [c for c in session.calls if c[1] == "/instruments/equities/AAPL"] == [("GET", "/instruments/equities/AAPL", None, None)]


def test_an_unknown_symbol_is_an_order_validation_error(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    with pytest.raises(OrderValidationError, match="does not know the US stock ZZZZ"):
        broker.submit_order(_buy(asset=Asset("ZZZZ")))

    assert session.posted(ORDERS) == []


def test_a_rejection_sets_the_error_before_raising_and_untracks(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = TastytradeHTTPError(422, f"Tastytrade POST {ORDERS}: 422 insufficient_buying_power: Not enough buying power")
    order = _buy()

    with pytest.raises(BrokerError, match="insufficient_buying_power"):
        broker.submit_order(order)

    assert order.status is OrderStatus.ERROR
    assert "insufficient_buying_power" in (order.error_message or "")
    assert broker.tracker.get_tracked_order(order.identifier) is None


def test_an_order_filled_in_the_post_response_fills_once(broker: TastytradeBroker, session: FakeTastytradeSession, seen: list[OrderEvent]) -> None:
    session.responses[("POST", ORDERS)] = accept(status="Filled", fills=(make_tt_fill("f1", "1", "2.00"),))
    order = broker.submit_order(_buy("1", order_type=OrderType.LIMIT, limit_price=D(2)))

    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D(1))
    # the account streamer then delivers the same snapshot
    broker._events.apply(make_tt_order(status="Filled", external=order.client_order_id, quantity="1", fills=[make_tt_fill("f1", "1", "2.00")]))

    assert order.filled_quantity == D(1)
    assert seen.count(OrderEvent.FILLED) == 1


def test_an_uncertain_submission_found_at_tastytrade_is_adopted_not_resent(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1001, status="Live", external="s:fixed")]}
    order = _buy(client_order_id="s:fixed")

    result = broker.submit_order(order)

    assert result.status is OrderStatus.NEW
    assert broker.tracker.get_tracked_order(order.identifier) is order
    assert len(session.posted(ORDERS)) == 1


def test_an_uncertain_submission_not_found_is_an_error_and_not_resent(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("POST", ORDERS)] = BrokerUncertainError("read timeout")
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", ORDERS)] = {"items": []}
    order = _buy()

    with pytest.raises(BrokerError, match="not resent"):
        broker.submit_order(order)

    assert order.status is OrderStatus.ERROR
    assert broker.tracker.get_tracked_order(order.identifier) is None
    assert len(session.posted(ORDERS)) == 1
    [(_, _, _, params)] = [c for c in session.calls if c[:2] == ("GET", ORDERS)]
    assert params["start-date"] == "2026-10-15"


def test_a_non_eligible_notional_order_becomes_whole_shares_at_alpacas_price(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    broker.submit_order(Order("s", MSFT, OrderSide.BUY, notional=D(1000)))

    [body] = session.posted(ORDERS)
    assert (body["order-type"], body["legs"][0]["quantity"]) == ("Market", "9")  # 1000 / 100.5


# --- cancel and modify ------------------------------------------------------------------------


def test_cancel_deletes_the_current_tastytrade_id(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("DELETE", f"{ORDERS}/1001")] = {}
    order = broker.submit_order(_limit())

    broker.cancel_order(order)

    assert session.calls[-1][:2] == ("DELETE", f"{ORDERS}/1001")


def test_modify_replaces_the_order_and_returns_the_same_order(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = make_tt_order(order_id=1002, status="Contingent", external=order.client_order_id, **{"replaces-order-id": 1001})

    result = broker.modify_order(order, limit_price=D(151))

    assert result is order
    assert order.limit_price == D(151)
    [(_, _, body, _)] = [c for c in session.calls if c[0] == "PUT"]
    assert "legs" not in body
    assert body["price"] == "151.00"
    assert broker._events.tt_id_of(order) == "1002"
    # OrderEvent.MODIFIED goes through the tracker, which notifies no listener for it (tracker.py), as for IBKR.


def test_modify_ignores_the_original_ids_cancel(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    broker._events.apply(make_tt_order(status="Live", external=order.client_order_id))
    session.responses[("PUT", f"{ORDERS}/1001")] = make_tt_order(order_id=1002, status="Contingent", external=order.client_order_id)
    broker.modify_order(order, limit_price=D(151))

    broker._events.apply(make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id))

    assert order.status is OrderStatus.NEW


def test_a_failed_modify_restores_the_prices_and_the_original_id(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = TastytradeHTTPError(422, "422 order_not_editable: no")

    with pytest.raises(BrokerError, match="order_not_editable"):
        broker.modify_order(order, limit_price=D(151))

    assert order.limit_price == D(150)
    broker._events.apply(make_tt_order(order_id=1001, status="Cancelled", external=order.client_order_id))
    assert order.status is OrderStatus.CANCELED


def test_an_uncertain_modify_found_at_tastytrade_succeeds(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("PUT", f"{ORDERS}/1001")] = BrokerUncertainError("read timeout")
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=1002, status="Contingent", external=order.client_order_id, **{"replaces-order-id": 1001})]}

    broker.modify_order(order, limit_price=D(151))

    assert broker._events.tt_id_of(order) == "1002"


def test_a_market_order_cannot_be_modified(broker: TastytradeBroker) -> None:
    order = broker.submit_order(_buy())

    with pytest.raises(OrderValidationError, match="can be modified"):
        broker.modify_order(order, limit_price=D(1))


# --- positions, account, orders -----------------------------------------------------------------


def test_close_position_sells_a_fractional_holding_of_an_eligible_stock_in_full(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(quantity="4.57")]}

    order = broker.close_position(AAPL)

    assert order is not None
    [body] = session.posted(ORDERS)
    assert (body["order-type"], body["legs"][0]["action"], body["legs"][0]["quantity"]) == ("Market", "Sell to Close", "4.57")


def test_close_position_floors_a_non_eligible_holding(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(symbol="MSFT", quantity="4.57")]}

    broker.close_position(MSFT)

    assert session.posted(ORDERS)[0]["legs"][0]["quantity"] == "4"


def test_close_position_below_one_non_eligible_share_sends_nothing(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(symbol="MSFT", quantity="0.5")]}

    assert broker.close_position(MSFT) is None
    assert session.posted(ORDERS) == []


def test_close_position_without_a_position_returns_none(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": []}

    assert broker.close_position(AAPL) is None


def test_close_all_positions_cancels_live_orders_first(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = {"items": [make_tt_order(order_id=7), make_tt_order(order_id=8, cancellable=False)]}
    session.responses[("DELETE", f"{ORDERS}/7")] = {}
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(quantity="3")]}

    closed = broker.close_all_positions()

    assert [c[1] for c in session.calls if c[0] == "DELETE"] == [f"{ORDERS}/7"]
    assert len(closed) == 1


def test_pull_positions_keeps_equities(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/positions")] = {"items": [make_tt_position(), make_tt_position(symbol="AAPL  261218C00200000", instrument_type="Equity Option")]}

    [position] = broker.pull_positions()

    assert (position.asset, position.quantity, position.side) == (AAPL, D(10), PositionSide.LONG)


def test_get_account_parses_the_balances(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"/accounts/{ACCOUNT}/balances")] = {"cash-balance": "1000", "net-liquidating-value": "2500", "equity-buying-power": "1000"}

    assert broker.get_account() == AccountBalances(cash=D(1000), portfolio_value=D(2500), buying_power=D(1000))


def test_pull_orders_parses_the_newest_and_skips_unsupported_ones(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", ORDERS)] = {"items": [make_tt_order(order_id=2), make_tt_order(order_id=1, **{"order-type": "Marketable Limit"})]}

    [order] = broker.pull_orders(limit=10)

    assert order.identifier == "abc"
    [(_, _, _, params)] = [c for c in session.calls if c[:2] == ("GET", ORDERS)]
    assert params == {"sort": "Desc"}


def test_pull_order_finds_ours_by_external_identifier(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", ORDERS)] = {"items": [make_tt_order(order_id=2, external="s:other"), make_tt_order(order_id=1)]}

    order = broker.pull_order("abc")

    assert order is not None
    assert order.raw["id"] == 1
    assert broker.pull_order("missing") is None


def test_sync_open_orders_adopts_only_this_strategys_working_orders(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    session.responses[("GET", f"{ORDERS}/live")] = {
        "items": [
            make_tt_order(order_id=1, external="s:mine", fills=[make_tt_fill("f1", "4", "150")]),
            make_tt_order(order_id=2, external="other:theirs"),
            make_tt_order(order_id=3, external="s:done", status="Filled"),
            make_tt_order(order_id=4, external="s:old", **{"replacing-order-id": 5}),
        ]
    }

    [adopted] = broker.sync_open_orders()

    assert (adopted.identifier, adopted.filled_quantity) == ("mine", D(4))
    assert broker._events.tt_id_of(adopted) == "1"
    broker._events.apply(make_tt_order(order_id=1, external="s:mine", fills=[make_tt_fill("f1", "4", "150")]))
    assert adopted.filled_quantity == D(4)  # its existing fill is not counted twice
    assert broker.sync_open_orders() == []  # already tracked


def test_reconcile_re_reads_active_orders(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    order = broker.submit_order(_limit())
    session.responses[("GET", f"{ORDERS}/live")] = {"items": []}
    session.responses[("GET", f"{ORDERS}/1001")] = make_tt_order(order_id=1001, status="Filled", external=order.client_order_id, fills=[make_tt_fill("f1", "10", "150")])

    broker.reconcile()

    assert order.status is OrderStatus.FILL


# --- lifecycle --------------------------------------------------------------------------------


def test_from_settings_checks_the_account(session: FakeTastytradeSession) -> None:
    broker = TastytradeBroker.from_settings("s", SETTINGS, data=AlpacaCredentials(api_key="k", api_secret="s"), session=session, market_data=_market_data())

    assert broker.account_number == ACCOUNT
    assert broker.is_paper
    assert not session.closed


def test_from_settings_closes_the_session_when_the_check_fails(session: FakeTastytradeSession) -> None:
    session.responses[("GET", "/customers/me/accounts")] = {"items": []}

    with pytest.raises(ConfigurationError, match="TASTYTRADE_ACCOUNT_NUMBER"):
        TastytradeBroker.from_settings("s", SETTINGS, data=AlpacaCredentials(api_key="k", api_secret="s"), session=session, market_data=_market_data())

    assert session.closed


def test_stop_stream_closes_the_session(broker: TastytradeBroker, session: FakeTastytradeSession) -> None:
    broker.stop_stream()

    assert session.closed


def test_market_data_comes_from_alpaca(broker: TastytradeBroker) -> None:
    assert broker.get_last_price(AAPL) == D("100.5")
```

Append to `tests/brokers/test_factory.py`:

```python
def test_the_tastytrade_builder_passes_settings_and_alpaca_data_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker

    captured: dict[str, object] = {}

    def fake_from_settings(cls, strategy_name, settings, *, data, news, session=None, market_data=None):
        captured.update(settings=settings, data=data, news=news)
        return _fake_broker(strategy_name)

    monkeypatch.setattr(TastytradeBroker, "from_settings", classmethod(fake_from_settings))
    env = {
        "BROKER": "tastytrade",
        "BROKER_API_IS_PAPER": "false",
        "TASTYTRADE_CLIENT_SECRET": "secret",
        "TASTYTRADE_REFRESH_TOKEN": "refresh",
        "TASTYTRADE_ACCOUNT_NUMBER": "5WT00001",
        "ALPACA_DATA_API_KEY": "dk",
        "ALPACA_DATA_API_SECRET": "ds",
    }

    factory.build_broker("s", env=env)

    settings = captured["settings"]
    assert (settings.account_number, settings.is_paper) == ("5WT00001", False)
    assert captured["data"].api_key == "dk"
```

Append to `tests/brokers/test_lazy_imports.py`:

```python
def test_importing_brokers_and_the_factory_does_not_import_tastytrade_or_websockets() -> None:
    code = (
        "import trading_agent_framework.brokers\n"
        "import trading_agent_framework.brokers.factory\n"
        "import sys\n"
        "assert 'websockets' not in sys.modules\n"
        "assert not [m for m in sys.modules if m.startswith('trading_agent_framework.brokers.tastytrade')]\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_tastytrade_broker_is_a_lazy_export() -> None:
    import trading_agent_framework.brokers as brokers
    from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker

    assert brokers.TastytradeBroker is TastytradeBroker
    assert "TastytradeBroker" in dir(brokers)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/tastytrade/test_broker.py tests/brokers/test_factory.py tests/brokers/test_lazy_imports.py -q`
Expected: FAIL with `ModuleNotFoundError: ... tastytrade.broker`.

- [ ] **Step 4: Implement**

`src/trading_agent_framework/brokers/tastytrade/broker.py`:

```python
"""`TastytradeBroker`: Tastytrade (REST API + account streamer) for trading, account and positions;
Alpaca for market data, the calendar and news.

Wiring only: translation lives in the pure `orders`/`account` modules, HTTP in `TastytradeSession`,
the websocket in `TastytradeAccountStream`, and event handling in `TastytradeOrderEvents`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn
from urllib.parse import quote

from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.alpaca.news import lazy_news_provider
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.brokers.tastytrade import account, orders
from trading_agent_framework.brokers.tastytrade.client import TastytradeHTTPError, TastytradeSession
from trading_agent_framework.brokers.tastytrade.events import TastytradeOrderEvents
from trading_agent_framework.brokers.tastytrade.stream import StreamSocket, TastytradeAccountStream
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError, OrderValidationError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings

logger = logging.getLogger(__name__)

_MODIFIABLE = frozenset({OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT})


class TastytradeBroker(Broker):
    name: ClassVar[str] = "tastytrade"

    def __init__(
        self,
        strategy_name: str,
        session: TastytradeSession,
        *,
        account_number: str,
        streamer_url: str,
        market_data: AlpacaMarketData,
        tracker: OrderTracker | None = None,
        clock: MarketClock | None = None,
        is_paper: bool = True,
        news_provider_factory: Callable[[], NewsProvider] | None = None,
        stream_connect: Callable[[str], StreamSocket] | None = None,
    ) -> None:
        super().__init__(
            strategy_name,
            tracker,
            clock=clock if clock is not None else AlpacaMarketClock(market_data.calendar_client),
            is_paper=is_paper,
        )
        self._session = session
        self._account_number = account_number
        self._market_data = market_data
        self._news_provider: NewsProvider | None = None
        self._news_provider_factory = news_provider_factory
        self._events = TastytradeOrderEvents(self.tracker)
        self._eligible: dict[str, bool] = {}
        connect: dict[str, Any] = {"connect": stream_connect} if stream_connect is not None else {}
        self._stream = TastytradeAccountStream(streamer_url, account_number, session.access_token, self._events.apply, self.reconcile, **connect)

    @classmethod
    def from_settings(
        cls,
        strategy_name: str,
        settings: TastytradeSettings,
        *,
        data: AlpacaCredentials,
        news: Callable[[], AlpacaCredentials] | None = None,
        session: TastytradeSession | None = None,
        market_data: AlpacaMarketData | None = None,
    ) -> TastytradeBroker:
        """Open the session and check the account; any failure closes the session before raising."""
        market_data = market_data if market_data is not None else AlpacaMarketData.from_credentials(data)
        session = session if session is not None else TastytradeSession(settings)
        try:
            broker = cls(
                strategy_name,
                session,
                account_number=settings.account_number,
                streamer_url=settings.streamer_url,
                market_data=market_data,
                is_paper=settings.is_paper,
                news_provider_factory=lazy_news_provider(news) if news is not None else None,
            )
            broker.configure_account()
        except BaseException:
            session.close()
            raise
        return broker

    # --- account ---------------------------------------------------------------------

    @property
    def account_number(self) -> str:
        return self._account_number

    def _path(self, suffix: str) -> str:
        return f"/accounts/{self._account_number}{suffix}"

    def configure_account(self) -> None:
        """Tastytrade cannot switch margin off via the API: check the account instead."""
        items = self._session.get("/customers/me/accounts").get("items") or []
        for warning in account.check_account(items, self._account_number, is_paper=self.is_paper):
            logger.warning(warning)
        logger.info("Tastytrade account %s checked (sandbox=%s)", self._account_number, self.is_paper)

    def get_account(self) -> AccountBalances:
        return account.parse_balances(self._session.get(self._path("/balances")))

    # --- orders ----------------------------------------------------------------------

    def _fractional_eligible(self, asset: Asset) -> bool:
        symbol = asset.symbol
        if symbol not in self._eligible:
            path = f"/instruments/equities/{quote(orders.to_tastytrade_symbol(symbol), safe='')}"
            try:
                data = self._session.get(path)
            except TastytradeHTTPError as exc:
                if exc.status_code == 404:
                    raise OrderValidationError(f"Tastytrade does not know the US stock {symbol}") from exc
                raise
            self._eligible[symbol] = bool(data.get("is-fractional-quantity-eligible"))
        return self._eligible[symbol]

    def _conform_order(self, order: Order) -> Order:
        eligible = self._fractional_eligible(order.asset)
        last_price = self._market_data.get_last_price(order.asset) if order.notional is not None and not eligible else None
        return orders.conform_order(order, fractional_eligible=eligible, last_price=last_price)

    def _submit_order(self, order: Order) -> Order:
        if not order.client_order_id:
            order.client_order_id = orders.external_identifier_for(self.strategy_name, order.identifier)
        body = orders.build_order_json(order)
        external = order.client_order_id
        self.tracker.track_unprocessed(order)  # before posting: the stream can report the order before the call returns
        try:
            placed: Mapping[str, Any] | None = self._session.post(self._path("/orders"), json=body).get("order")
        except BrokerUncertainError as exc:
            placed = self._find_order(lambda item: item.get("external-identifier") == external)
            if placed is None:
                self._fail(order, f"submission outcome unknown ({exc}); Tastytrade does not show it, so it was not resent -- check the account before resubmitting", cause=exc)
            logger.warning("Order %s: the submission outcome was unknown, but Tastytrade has it (id %s)", order.identifier, placed.get("id"))
        except Exception as exc:
            self._fail(order, str(exc), cause=exc)
        if not placed or placed.get("id") is None:
            self._fail(order, "Tastytrade accepted the order but returned no order id")
        self._events.register(order, str(placed["id"]))
        order.update_raw(dict(placed))
        if order.status is OrderStatus.UNPROCESSED:
            order.status = OrderStatus.SUBMITTED  # from here on, TastytradeOrderEvents owns its status
        self._events.apply(placed)  # a sandbox $2 limit is already Filled here
        return order

    def _fail(self, order: Order, message: str, *, cause: BaseException | None = None) -> NoReturn:
        order.set_error(message)  # BEFORE raising -- lumibot's contract
        self.tracker.untrack(order)
        raise BrokerError(f"Tastytrade order {order.identifier} failed: {message}") from cause

    def _find_order(self, matches: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any] | None:
        """The first live, then today's, order matching `matches`; None when absent or every lookup failed."""
        today = self.clock.now().date().isoformat()
        for path, params in ((self._path("/orders/live"), None), (self._path("/orders"), {"start-date": today, "sort": "Desc"})):
            try:
                items = self._session.get_all(path, params=params)
            except BrokerError:
                logger.exception("Looking up Tastytrade orders at %s failed", path)
                continue
            found = next((item for item in items if matches(item)), None)
            if found is not None:
                return found
        return None

    def _tt_id(self, order: Order) -> str:
        tt_id = self._events.tt_id_of(order)
        if tt_id is None:
            raise BrokerError(f"no Tastytrade order id for {order.identifier}")
        return tt_id

    def cancel_order(self, order: Order) -> None:
        self._session.delete(self._path(f"/orders/{self._tt_id(order)}"))

    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Tastytrade replaces the order (new id, same `external-identifier`); the same `Order` comes back."""
        if order.notional is not None or order.order_type not in _MODIFIABLE:
            raise OrderValidationError(f"only limit, stop and stop-limit orders can be modified (order {order.identifier} is {order.order_type.value})")
        tt_id = self._tt_id(order)
        previous = (order.limit_price, order.stop_price, order.stop_limit_price)
        if limit_price is not None:
            if order.order_type is OrderType.STOP_LIMIT:
                order.stop_limit_price = limit_price
            else:
                order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        body = orders.build_replace_json(order)
        self._events.begin_replace(order)  # the original's Cancelled may arrive before the PUT returns
        try:
            replacement: Mapping[str, Any] | None = self._session.put(self._path(f"/orders/{tt_id}"), json=body)
        except BrokerUncertainError:
            external = order.client_order_id
            replacement = self._find_order(lambda item: str(item.get("replaces-order-id")) == tt_id and item.get("external-identifier") == external)
            if replacement is None:
                self._restore(order, tt_id, previous)
                raise
        except Exception:
            self._restore(order, tt_id, previous)
            raise
        if replacement and replacement.get("id") is not None:
            self._events.finish_replace(order, str(replacement["id"]))
        else:
            self._events.abort_replace(tt_id)
        self.tracker.process_trade_event(order, OrderEvent.MODIFIED)
        return order

    def _restore(self, order: Order, tt_id: str, previous: tuple[Decimal | None, Decimal | None, Decimal | None]) -> None:
        self._events.abort_replace(tt_id)
        order.limit_price, order.stop_price, order.stop_limit_price = previous

    def pull_order(self, identifier: str) -> Order | None:
        external = orders.external_identifier_for(self.strategy_name, identifier)
        items = self._session.get_all(self._path("/orders"), params={"sort": "Desc"}, max_items=1000)
        found = next((item for item in items if item.get("external-identifier") == external or str(item.get("id")) == identifier), None)
        return orders.parse_order(found, self.strategy_name) if found is not None and orders.is_parsable_order(found) else None

    def pull_orders(self, limit: int = 100) -> list[Order]:
        """The newest `limit` orders of the account (equity orders this framework models)."""
        items = self._session.get_all(self._path("/orders"), params={"sort": "Desc"}, max_items=limit)
        return [orders.parse_order(item, self.strategy_name) for item in items if orders.is_parsable_order(item)]

    def pull_positions(self) -> list[Position]:
        items = self._session.get(self._path("/positions")).get("items") or []
        parsed = (orders.parse_position(item, self.strategy_name) for item in items)
        return [p for p in parsed if p is not None]

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        position = next((p for p in self.pull_positions() if p.asset == asset), None)
        if position is None:
            return None
        quantity = position.quantity * fraction
        if not self._fractional_eligible(asset) and quantity.to_integral_value(rounding=ROUND_FLOOR) < 1:
            logger.warning("close_position(%s, %s): below one whole share, nothing sent", asset.symbol, fraction)
            return None
        side = OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY
        return self.submit_order(Order(self.strategy_name, asset, side, OrderType.MARKET, quantity=quantity))

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        if cancel_orders:
            for item in self._session.get_all(self._path("/orders/live")):
                if item.get("cancellable"):
                    try:
                        self._session.delete(self._path(f"/orders/{item['id']}"))
                    except BrokerError:
                        logger.exception("Cancelling Tastytrade order %s failed", item.get("id"))
        closed = (self.close_position(p.asset) for p in self.pull_positions())
        return [o for o in closed if o is not None]

    def sync_open_orders(self) -> list[Order]:
        adopted: list[Order] = []
        for item in self._session.get_all(self._path("/orders/live")):
            identifier = orders.identifier_from_external(item.get("external-identifier"), self.strategy_name)
            if identifier is None or item.get("status") in orders.TERMINAL_STATUSES or item.get("replacing-order-id") is not None or not orders.is_parsable_order(item):
                continue
            if self.tracker.get_tracked_order(identifier) is not None:
                continue
            order = orders.parse_order(item, self.strategy_name)
            self._events.register(order, str(item["id"]), snapshot=item)
            self.tracker.track_unprocessed(order)
            adopted.append(order)
        return adopted

    def reconcile(self) -> None:
        """After a stream reconnect: adopt open orders, then re-read every active order we track."""
        self.sync_open_orders()
        for order in self.tracker.get_active_orders():
            tt_id = self._events.tt_id_of(order)
            if tt_id is None:
                continue
            try:
                self._events.apply(self._session.get(self._path(f"/orders/{tt_id}")))
            except BrokerError:
                logger.exception("Re-reading Tastytrade order %s failed", tt_id)

    # --- market data, clock, news (Alpaca) ---------------------------------------------

    def news_provider(self) -> NewsProvider | None:
        if self._news_provider is None and self._news_provider_factory is not None:
            self._news_provider = self._news_provider_factory()
        return self._news_provider

    def get_last_price(self, asset: Asset) -> Decimal | None:
        return self._market_data.get_last_price(asset)

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        return self._market_data.get_last_prices(assets)

    def get_quote(self, asset: Asset) -> Quote | None:
        return self._market_data.get_quote(asset)

    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        return self._market_data.get_bars(assets, length, timestep, end=self.clock.now(), include_after_hours=include_after_hours)

    # --- stream ----------------------------------------------------------------------

    def start_stream(self) -> None:
        self._stream.start()

    def stop_stream(self, timeout: float = 5.0) -> None:
        self._stream.stop(timeout)
        self._session.close()
```

Note: `test_an_uncertain_submission_found_at_tastytrade_is_adopted_not_resent` expects `NEW` because the found snapshot is `Live` and `_submit_order` applies it after registering.

`src/trading_agent_framework/brokers/factory.py`: update the docstring's `BROKER=alpaca|ibkr` to `BROKER=alpaca|ibkr|tastytrade`, add `TastytradeSettings` to the `config.env` import, and add:

```python
def _build_tastytrade(strategy_name: str, settings: BrokerSettings, env: Mapping[str, str] | None) -> Broker:
    from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker

    return TastytradeBroker.from_settings(
        strategy_name,
        TastytradeSettings.from_env(settings.is_paper, env),
        data=AlpacaCredentials.for_data(env),
        news=lambda: AlpacaCredentials.for_news(env),
    )


BUILDERS: dict[BrokerKind, BrokerBuilder] = {
    BrokerKind.ALPACA: _build_alpaca,
    BrokerKind.IBKR: _build_ibkr,
    BrokerKind.TASTYTRADE: _build_tastytrade,
}
```

`src/trading_agent_framework/brokers/__init__.py`: add `from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker` under `TYPE_CHECKING`, `"TastytradeBroker"` to `__all__`, `"TastytradeBroker": (".tastytrade.broker", "TastytradeBroker"),` to `_LAZY`, and a docstring sentence: "`TastytradeBroker` lives in `brokers.tastytrade` (httpx, websockets); lazy for the same reason."

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers -q`
Expected: PASS.

- [ ] **Step 6: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/brokers tests/fakes.py tests/brokers
git commit -m "feat: BROKER=tastytrade builds a checked TastytradeBroker (Task 9)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Sandbox smoke scripts and documentation

**Files:**
- Create: `scripts/tests/smoke_tastytrade_account.py`, `scripts/tests/smoke_tastytrade_orders.py`
- Modify: `scripts/tests/HOWTO.md`, `CLAUDE.md`, `src/trading_agent_framework/brokers/CLAUDE.md`, `README.md`, `TODO.md`

**Interfaces:**
- Consumes: `TastytradeBroker.from_settings`, `TastytradeSettings.from_env`, `AlpacaCredentials.for_data`, `broker.tracker.listeners`.

- [ ] **Step 1: Write `scripts/tests/smoke_tastytrade_account.py`**

```python
"""Manual smoke test: authenticate against the Tastytrade SANDBOX, check the account, print balances and positions.

Read-only: places no orders. Credentials in env/.env.alpaca.integration-tests: TASTYTRADE_CLIENT_SECRET,
TASTYTRADE_REFRESH_TOKEN, TASTYTRADE_ACCOUNT_NUMBER (sandbox ones) and ALPACA_DATA_API_KEY /
ALPACA_DATA_API_SECRET. Always uses the sandbox, whatever BROKER and BROKER_API_IS_PAPER say.

    uv run python scripts/tests/smoke_tastytrade_account.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker
from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings, find_project_root
from trading_agent_framework.entities.asset import Asset

STRATEGY_NAME = "smoke_tastytrade"
ENV_FILE = find_project_root() / "env" / ".env.alpaca.integration-tests"


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"SKIP: credentials file not found: {ENV_FILE}")
        return 0
    load_dotenv(ENV_FILE, override=True)
    if not os.environ.get("TASTYTRADE_REFRESH_TOKEN", "").strip():
        print(f"SKIP: no TASTYTRADE_* credentials in {ENV_FILE}")
        return 0
    broker = TastytradeBroker.from_settings(STRATEGY_NAME, TastytradeSettings.from_env(is_paper=True), data=AlpacaCredentials.for_data())
    try:
        print(f"Sandbox account {broker.account_number}: {broker.get_account()}")
        for position in broker.pull_positions():
            print(f"  {position.asset.symbol}: {position.quantity} {position.side.value} @ {position.avg_fill_price}")
        print(f"SPY last price (Alpaca IEX): {broker.get_last_price(Asset('SPY'))}")
        print(f"Next session: {broker.clock.next_session()}")
    finally:
        broker.stop_stream()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write `scripts/tests/smoke_tastytrade_orders.py`**

```python
"""Manual smoke test: order round trip against the Tastytrade SANDBOX, with the account streamer running.

Built on the sandbox's simulated fills (they ignore the market): limit orders at $3 or more rest
forever, limit orders under $3 fill at once, market orders fill at $1. The sandbox clears trades and
positions every 24 hours. Same credentials as smoke_tastytrade_account.py; always the sandbox.

  1. $5 limit buy of 1 AAPL           -> rests (NEW)
  2. modify it to $6                   -> still working at $6; the original id's Cancelled must not cancel it
  3. cancel it                         -> CANCELED
  4. $2 limit buy of 1 AAPL           -> FILL; the position appears
  5. close the AAPL position (market) -> FILL; the position is gone

    uv run python scripts/tests/smoke_tastytrade_orders.py
"""

from __future__ import annotations

import os
import time
from decimal import Decimal

from dotenv import load_dotenv

from trading_agent_framework.brokers.tastytrade.broker import TastytradeBroker
from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings, find_project_root
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order

STRATEGY_NAME = "smoke_tastytrade"
ENV_FILE = find_project_root() / "env" / ".env.alpaca.integration-tests"
AAPL = Asset("AAPL")


def _wait_for(order: Order, statuses: set[OrderStatus], seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while order.status not in statuses and time.monotonic() < deadline:
        time.sleep(0.2)
    print(f"  {order.identifier}: {order.status}")
    return order.status in statuses


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"SKIP: credentials file not found: {ENV_FILE}")
        return 0
    load_dotenv(ENV_FILE, override=True)
    if not os.environ.get("TASTYTRADE_REFRESH_TOKEN", "").strip():
        print(f"SKIP: no TASTYTRADE_* credentials in {ENV_FILE}")
        return 0
    broker = TastytradeBroker.from_settings(STRATEGY_NAME, TastytradeSettings.from_env(is_paper=True), data=AlpacaCredentials.for_data())
    results: list[tuple[str, bool]] = []
    events: list[tuple[str, OrderEvent]] = []

    def check(name: str, ok: bool) -> None:
        results.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    broker.tracker.listeners.append(lambda order, event: events.append((order.identifier, event)))
    try:
        broker.sync_open_orders()
        broker.start_stream()
        time.sleep(2)  # let the streamer subscribe

        print("1) limit buy 1 AAPL @ $5 (rests in the sandbox)")
        resting = broker.submit_order(Order(STRATEGY_NAME, AAPL, OrderSide.BUY, OrderType.LIMIT, quantity=Decimal(1), limit_price=Decimal(5)))
        check("a $5 limit rests (NEW)", _wait_for(resting, {OrderStatus.NEW}))

        print("2) modify to $6")
        broker.modify_order(resting, limit_price=Decimal(6))
        time.sleep(3)  # the original's Cancelled arrives on the stream meanwhile
        check("the modified order is still working at $6", resting.status is OrderStatus.NEW and resting.limit_price == Decimal(6))
        check("no cancel event reached the modified order", (resting.identifier, OrderEvent.CANCELED) not in events)

        print("3) cancel")
        broker.cancel_order(resting)
        check("cancel -> CANCELED", _wait_for(resting, {OrderStatus.CANCELED}))

        print("4) limit buy 1 AAPL @ $2 (fills at once in the sandbox)")
        filled = broker.submit_order(Order(STRATEGY_NAME, AAPL, OrderSide.BUY, OrderType.LIMIT, quantity=Decimal(1), limit_price=Decimal(2)))
        check("a $2 limit fills (FILL)", _wait_for(filled, {OrderStatus.FILL}))
        position = next((p for p in broker.pull_positions() if p.asset == AAPL), None)
        check("the AAPL position appears", position is not None and position.quantity >= 1)

        print("5) close the AAPL position at market (fills at $1 in the sandbox)")
        close = broker.close_position(AAPL)
        check("the close fills (FILL)", close is not None and _wait_for(close, {OrderStatus.FILL}))
        time.sleep(2)
        check("the AAPL position is gone", all(p.asset != AAPL for p in broker.pull_positions()))
    finally:
        broker.stop_stream()
    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Lint and check both scripts skip cleanly**

Run: `uv run ruff check scripts/tests && uv run python scripts/tests/smoke_tastytrade_account.py | tail -1`
Expected: ruff passes; the script prints `SKIP: ...` without credentials, or `OK` with them.

- [ ] **Step 4: Update the docs**

`scripts/tests/HOWTO.md`: add a section before `## Tools`:

```markdown
## Tastytrade sandbox
Manual checks of `TastytradeBroker` against the Tastytrade sandbox (`api.cert.tastyworks.com`). NOT part
of the automated suite. Credentials go in env/.env.alpaca.integration-tests: `TASTYTRADE_CLIENT_SECRET`,
`TASTYTRADE_REFRESH_TOKEN`, `TASTYTRADE_ACCOUNT_NUMBER` (sandbox ones) plus the `ALPACA_DATA_*` keys. The
scripts always use the sandbox, whatever `BROKER` and `BROKER_API_IS_PAPER` say, and print `SKIP:` without
credentials.

Getting sandbox credentials: open the sandbox account tool (developer.tastytrade.com, Sandbox page), create
a sandbox user and its funded account, create an OAuth application (keep its client secret), then create a
grant for it: the grant's refresh token never expires. Sandbox and production credentials never work
against each other.

The sandbox's fills are synthetic (market orders fill at $1, limits under $3 fill at once, limits at $3 or
more never fill) and it clears trades and positions every 24 hours, which is why strategies refuse to run
against it (`uv run agent ... paper` with `BROKER=tastytrade` exits): forward-test on Alpaca paper.

    uv run python scripts/tests/smoke_tastytrade_account.py   # read-only: auth, account check, balances, positions
    uv run python scripts/tests/smoke_tastytrade_orders.py    # rest, modify, cancel, fill, close; PASS/FAIL per step
    uv run python scripts/tests/probe_tastytrade_orders.py    # one-off API probe (Task 1 of the Tastytrade plan); not run by run_smoke_tests.sh
```

`CLAUDE.md` (root): add to the Commands block after the IBKR lines:

```bash
uv run python scripts/tests/smoke_tastytrade_account.py  # manual Tastytrade sandbox smoke test: account (read-only)
uv run python scripts/tests/smoke_tastytrade_orders.py   # manual Tastytrade sandbox smoke test: orders and the account streamer
```

and change the `brokers/CLAUDE.md` bullet in the nested-docs list to `` `brokers/CLAUDE.md` -- `Broker` template, `OrderTracker`, fees, Alpaca, IBKR and Tastytrade brokers, market-data, IBKR and Tastytrade gotchas. ``

`src/trading_agent_framework/brokers/CLAUDE.md`: change the title to `# brokers/ -- orders, trackers, fees, Alpaca, IBKR and Tastytrade`; in Architecture, change the `factory.py` bullet's broker list to `(`alpaca` default, `ibkr`, `tastytrade`)`, the `__init__.py` bullet to include `TastytradeBroker` (no `httpx`/`websockets`), and add:

```markdown
- `brokers/tastytrade/` -- `TastytradeBroker` over Tastytrade's REST API and account streamer: **pure** `values.py`/`account.py`/`orders.py`; `client.py` (`TastytradeSession`, the only HTTP code: OAuth token refresh under a lock, GET retries, writes never retried, `TastytradeHTTPError`/`BrokerUncertainError`); `stream.py` (`TastytradeAccountStream`, the only websocket code: heartbeats, reconnect then `reconcile`); `events.py` (`TastytradeOrderEvents`, feeds `OrderTracker` only); `broker.py` (wiring; market data, clock and news come from Alpaca).
```

and add to Gotchas:

```markdown
- **Tastytrade has no trailing stops** (`OrderType.TRAIL` is rejected, so `earnings_drift` can't trade there) and takes fractional quantities on market orders of `is-fractional-quantity-eligible` stocks only: a fractional limit/stop is rejected (a floored stop would leave the remainder unprotected), a non-eligible fraction or notional is floored to whole shares with a warning.
- **Tastytrade order identity.** `external-identifier` carries `{strategy}:{identifier}`, but events find orders by Tastytrade's order id: a modify is a cancel-and-replace with a new id, and the old id is superseded from the moment the PUT is sent (its fills count, its `Cancelled` is ignored). `modify_order` returns the same `Order`.
- **Tastytrade does not deduplicate orders.** Writes are never retried; a submission whose outcome is unknown (`BrokerUncertainError`) is looked up by `external-identifier` and adopted if found, otherwise reported as an error -- never resent.
- **The Tastytrade sandbox fills are synthetic** (market at $1, limits under $3 fill at once, $3 or more never fill; daily reset), so `main.py` refuses to run a strategy with `BROKER=tastytrade` and `BROKER_API_IS_PAPER` not false. The sandbox is for `scripts/tests/smoke_tastytrade_*.py` only.
- **`httpx` stays in `brokers/tastytrade/client.py` and `websockets` inside `stream._default_connect`**; `import trading_agent_framework.brokers` imports neither.
```

`README.md`: add a section after the IBKR section:

```markdown
## 📈 Tastytrade

Set `BROKER=tastytrade` and `BROKER_API_IS_PAPER=false` in the strategy's **live** env file, with
`TASTYTRADE_CLIENT_SECRET`, `TASTYTRADE_REFRESH_TOKEN` and `TASTYTRADE_ACCOUNT_NUMBER`. Tastytrade handles
orders, account and positions; prices, bars, the market calendar and news still come from Alpaca
(`ALPACA_DATA_*`, `ALPACA_NEWS_*`). Backtests are unaffected (Yahoo or Alpaca data; `BROKER=tastytrade` in a
backtest env file only picks the $0-commission fee schedule).

1. Create an OAuth application and a grant at my.tastytrade.com; the grant's refresh token never expires.
2. Use a **cash** account: the broker refuses to trade live on a margin account.
3. There is no Tastytrade paper trading here: the sandbox's fills are synthetic (market orders fill at $1),
   so strategies refuse to run against it. Forward-test on Alpaca paper; the sandbox is for the manual
   checks in `scripts/tests/HOWTO.md`.
4. No trailing stops (`earnings_drift` can't run on Tastytrade). Fractional quantities work on market orders
   of eligible stocks; other fractions and dollar amounts are rounded down to whole shares.
```

`TODO.md`: add a line `* Dashboard: Live/Paper tabs list Alpaca strategies only; add Tastytrade accounts`.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add scripts/tests CLAUDE.md src/trading_agent_framework/brokers/CLAUDE.md README.md TODO.md
git commit -m "docs: Tastytrade sandbox smoke scripts, setup and broker notes (Task 10)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ask the user to run the sandbox smoke tests**

Tell the user: "Run `uv run python scripts/tests/smoke_tastytrade_account.py` and `uv run python scripts/tests/smoke_tastytrade_orders.py` and paste the output." Any FAIL goes through superpowers:systematic-debugging before a fix.
