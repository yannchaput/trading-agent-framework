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
