"""Live and Paper tabs: one strategy's realized and unrealized P&L per day, rebuilt from its Alpaca account."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime

import pandas as pd
import streamlit as st
from alpaca.common.exceptions import APIError
from requests import RequestException

from trading_agent_framework.dashboard import account_reader
from trading_agent_framework.dashboard.components.charts import account_pnl_chart, signed_usd
from trading_agent_framework.dashboard.components.metric_cards import render_metric_card
from trading_agent_framework.dashboard.pnl import AccountPnl
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError


@st.cache_data(show_spinner="Fetching the account from Alpaca...", ttl="5m", max_entries=32)
def _account_pnl(strategy: str, mode: str, today: date) -> AccountPnl:
    # Keyed by name, mode and day only: the keys are re-read from the env file inside, never cached or shown.
    return account_reader.load_strategy_pnl(strategy, mode, account_reader.default_env_dir(), today)


def _problems(pnl: AccountPnl) -> list[str]:
    problems = [f"{symbol}: the fills give {replayed:g} shares, the broker holds {held:g}" for symbol, replayed, held in pnl.mismatches]
    problems += [f"{symbol}: {qty:g} shares sold with no recorded buy (their P&L is missing)" for symbol, qty in pnl.unmatched_sales.items()]
    problems += [f"{symbol}: no daily close, left out of the unrealized P&L before today" for symbol in pnl.unpriced]
    return problems


def _render_metrics(pnl: AccountPnl) -> None:
    total = pnl.realized_total + pnl.unrealized_now
    cards = [("Realized", pnl.realized_total), ("Unrealized now", pnl.unrealized_now), ("Total", total)]
    for column, (label, value) in zip(st.columns(len(cards)), cards, strict=True):
        with column:
            render_metric_card(label, signed_usd(value), tone="positive" if value >= 0 else "negative")


def _render_positions(pnl: AccountPnl) -> None:
    st.subheader("Open positions")
    if not pnl.positions:
        st.caption("No open position.")
        return
    frame = pd.DataFrame([asdict(p) for p in pnl.positions])
    frame["unrealized_plpc"] = frame["unrealized_plpc"] * 100
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "qty": st.column_config.NumberColumn("Qty", format="%g"),
            "avg_entry_price": st.column_config.NumberColumn("Avg cost", format="$%.2f"),
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_value": st.column_config.NumberColumn("Value", format="$%,.2f"),
            "unrealized_pl": st.column_config.NumberColumn("Unrealized", format="$%,.2f"),
            "unrealized_plpc": st.column_config.NumberColumn("Unrealized %", format="%.2f%%"),
        },
    )


def page_account(mode: str) -> None:
    """One tab: the sidebar picks the strategy, the page shows its account's P&L per day and its open positions."""
    st.title(mode.capitalize())
    strategies = account_reader.strategies_for_mode(mode, account_reader.default_env_dir())
    if not strategies:
        st.info(f"No strategy trades {mode} on Alpaca: add an env/.env.<strategy>.{mode} file with BROKER=alpaca.")
        return

    # The choice is kept under a plain session key: Streamlit drops the widget's own state on a run that does not
    # draw it, i.e. whenever another tab is open.
    remembered = f"account_strategy_{mode}"
    chosen = st.session_state.get(remembered)
    strategy = st.sidebar.selectbox(
        "Strategy",
        strategies,
        index=strategies.index(chosen) if chosen in strategies else 0,
        key=f"account_strategy_select_{mode}",
    )
    st.session_state[remembered] = strategy
    if st.sidebar.button("Refresh", icon=":material/refresh:", width="stretch", key=f"account_refresh_{mode}"):
        _account_pnl.clear()
    st.caption(f"{strategy} · {mode} Alpaca account")

    try:
        pnl = _account_pnl(strategy, mode, datetime.now(MARKET_TZ).date())
    except ConfigurationError as exc:
        st.error(f"env/.env.{strategy}.{mode}: {exc}")
        return
    except (APIError, RequestException) as exc:
        st.error(f"Could not reach Alpaca: {exc}")
        return

    if not pnl.days and not pnl.positions:
        st.info("No trades yet.")
        return

    _render_metrics(pnl)
    if pnl.days:
        st.plotly_chart(account_pnl_chart(pnl.days), width="stretch")
        st.caption("Realized: the P&L of that day's sales, net of fees, against the average cost. Unrealized (hatched): the open positions at that day's close; today, at the current price.")
    else:
        st.info("No fill history in this account: only today's unrealized P&L is known.")
    problems = _problems(pnl)
    if problems:
        st.warning("The fills do not fully explain the account:\n\n" + "\n".join(f"- {p}" for p in problems))
    _render_positions(pnl)


def page_live() -> None:
    page_account("live")


def page_paper() -> None:
    page_account("paper")
