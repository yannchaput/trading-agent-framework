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
from decimal import Decimal
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
    print(json.dumps(body, indent=2)[:20000])
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


def _order_id(body: Any) -> str | None:
    """Order id from a POST response (data.order.id) or a PUT response (data.id or data.order.id)."""
    data = _data(body)
    for candidate in (data.get("order"), data):
        if isinstance(candidate, dict) and candidate.get("id") is not None:
            return str(candidate["id"])
    return None


def _remember(known: list[str], body: Any) -> None:
    order_id = _order_id(body)
    if order_id is not None:
        known.append(order_id)


def _aapl_quantity(client: httpx.Client, account: str) -> Decimal | None:
    """Net AAPL position (Long minus Short), or None when the positions call fails."""
    try:
        response = client.get(f"/accounts/{account}/positions")
        response.raise_for_status()
        items = _data(response.json()).get("items") or []
    except (httpx.HTTPError, ValueError) as error:
        print(f"WARNING: could not read positions: {error}")
        return None
    net = Decimal(0)
    for item in items:
        if item.get("symbol") == SYMBOL:
            quantity = Decimal(str(item.get("quantity", "0")))
            net += -quantity if item.get("quantity-direction") == "Short" else quantity
    return net


def _cleanup(client: httpx.Client, orders: str, account: str, known_order_ids: list[str], baseline: Decimal | None) -> None:
    """Cancel every order this run placed, sweep leftover probe orders, then sell back only the AAPL shares the run added."""
    print("\n##### Cleanup")
    cancelled: set[str] = set()
    for order_id in dict.fromkeys(known_order_ids):
        try:
            response = client.delete(f"{orders}/{order_id}")
        except httpx.HTTPError as error:
            print(f"cancel {order_id}: transport error {error} (ignored)")
            continue
        if response.is_success:
            cancelled.add(order_id)
        print(f"cancel known order {order_id}: HTTP {response.status_code}" + ("" if response.is_success else " (ignored)"))
    try:
        live = client.get(f"{orders}/live")
        live.raise_for_status()
        items = _data(live.json()).get("items") or []
    except (httpx.HTTPError, ValueError) as error:
        print(f"WARNING: could not list live orders: {error}")
        items = []
    for item in items:
        item_id = str(item.get("id"))
        if str(item.get("external-identifier", "")).startswith(PREFIX) and item.get("cancellable") and item_id not in cancelled:
            try:
                _show(f"cancel {item_id} (prefix sweep)", client.delete(f"{orders}/{item_id}"))
            except httpx.HTTPError as error:
                print(f"cancel {item_id}: transport error {error} (ignored)")
    if baseline is None:
        print("WARNING: baseline AAPL position unknown; skipping the cleanup sell (check positions by hand)")
        return
    current = _aapl_quantity(client, account)
    if current is None:
        print("WARNING: positions unreadable after the run; skipping the cleanup sell (check positions by hand)")
        return
    excess = current - baseline
    print(f"AAPL position: baseline={baseline} now={current} excess={excess}")
    if excess > 0:
        try:
            _show("cleanup market sell of the excess", client.post(orders, json=_market("Sell to Close", str(excess), f"{PREFIX}cleanup")))
        except httpx.HTTPError as error:
            print(f"WARNING: cleanup sell failed: {error}; check the AAPL position by hand")
    elif excess < 0:
        print("WARNING: AAPL position is below the baseline; nothing sold (check positions by hand)")


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

        baseline = _aapl_quantity(client, account)  # captured before any real order; None = unknown
        known_order_ids: list[str] = []
        failed = False
        try:
            print("\n##### 3. Fractional sells (real sandbox orders)")
            _remember(known_order_ids, _show("market buy 0.5 (fills at $1)", client.post(orders, json=_market("Buy to Open", "0.5", f"{PREFIX}fraction"))))
            time.sleep(3)
            _show("positions", client.get(f"/accounts/{account}/positions"))
            _show("dry-run market sell 0.5", client.post(dry, json=_market("Sell to Close", "0.5")))
            _show("dry-run notional sell $0.25", client.post(dry, json=_notional("Sell to Close", "0.25", "Credit")))
            _remember(known_order_ids, _show("market sell 0.5 (cleanup)", client.post(orders, json=_market("Sell to Close", "0.5", f"{PREFIX}fraction-close"))))

            print("\n##### 4. Replace semantics (real sandbox orders)")
            placed = _show("limit buy 1 @ 5.00 (rests)", client.post(orders, json=_limit("5.00", external=f"{PREFIX}replace")))
            _remember(known_order_ids, placed)
            order_id = _order_id(placed)
            if order_id is not None:
                time.sleep(2)
                replacement = {"time-in-force": "Day", "order-type": "Limit", "price": "6.00", "price-effect": "Debit", "external-identifier": f"{PREFIX}replace"}
                replaced = _show("replace @ 6.00 (PUT without legs)", client.put(f"{orders}/{order_id}", json=replacement))
                _remember(known_order_ids, replaced)
                time.sleep(2)
                _show("order history (newest first)", client.get(orders, params={"per-page": 5, "sort": "Desc"}))
        except httpx.HTTPError as error:
            failed = True
            print(f"\nERROR: {type(error).__name__}: {error}")
        finally:
            _cleanup(client, orders, account, known_order_ids, baseline)
    if failed:
        print("\nAborted after an HTTP error; cleanup ran. Paste this whole output into the implementation session.")
        return 1
    print("\nDone. Paste this whole output into the implementation session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
