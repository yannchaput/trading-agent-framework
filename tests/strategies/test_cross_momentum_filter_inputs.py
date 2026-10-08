"""Yahoo as the source of every input of cross_momentum's `apply_filters` in paper/live.

Alpaca's IEX feed carries ~5% of consolidated volume, so `min_dollar_volume` was ~20x stricter live than in a
Yahoo backtest (300 symbols passed instead of ~1100), and a thin stock's IEX bars skip the days without an IEX
print. Live now reads price, average dollar volume, volatility and trading days from Yahoo, exactly as the
backtest's own Yahoo bars give them; the filter itself is unchanged. Scores, ranks and weights still use Alpaca.
"""

from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.strategies.cross_momentum.utils import annualized_volatility
from trading_agent_framework.strategies.cross_momentum.yahoo_filter_inputs import FilterInputs, YahooFilterSource, filter_inputs
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import FilterDataError

TODAY = date(2026, 10, 7)
YESTERDAY = date(2026, 10, 6)


def _frame(sessions, *, close=50.0, volume=1_000_000.0, last=YESTERDAY, closes=None) -> pd.DataFrame:
    """A Yahoo daily frame (Close, Volume) with one row per business day, ending on `last`."""
    index = pd.bdate_range(end=last, periods=sessions)
    return pd.DataFrame({"Close": closes if closes is not None else [close] * sessions, "Volume": [volume] * sessions}, index=index)


# --- filter_inputs (pure) ---------------------------------------------------------------------------------


def test_the_price_is_the_last_completed_close():
    frame = _frame(300, closes=[100.0 + i for i in range(300)])

    assert filter_inputs(frame, TODAY).price == pytest.approx(399.0)


def test_a_bar_dated_today_is_partial_and_ignored():
    frame = _frame(301, last=TODAY, closes=[100.0] * 300 + [1.0])  # today's forming bar crashed

    inputs = filter_inputs(frame, TODAY)

    assert inputs.price == pytest.approx(100.0)
    assert inputs.trading_days == 300


def test_the_dollar_volume_is_the_20_session_average_volume_times_the_last_close():
    frame = _frame(300, close=40.0)
    frame.iloc[-20:, frame.columns.get_loc("Volume")] = 3_000_000.0  # the older sessions stay at 1M

    assert filter_inputs(frame, TODAY).avg_dollar_volume == pytest.approx(3_000_000.0 * 40.0)


def test_the_volatility_is_the_backtests_20_day_annualized_log_return_volatility():
    closes = [100.0 + (i % 7) * 1.5 for i in range(300)]

    inputs = filter_inputs(_frame(300, closes=closes), TODAY)

    assert inputs.volatility == pytest.approx(annualized_volatility(closes, 20))


def test_trading_days_counts_the_completed_sessions_up_to_the_scan_history():
    assert filter_inputs(_frame(320), TODAY).trading_days == 300
    assert filter_inputs(_frame(260), TODAY).trading_days == 260


def test_rows_with_a_missing_close_or_volume_are_dropped():
    frame = _frame(300)
    frame.iloc[10, frame.columns.get_loc("Close")] = float("nan")
    frame.iloc[11, frame.columns.get_loc("Volume")] = float("nan")

    assert filter_inputs(frame, TODAY).trading_days == 298


def test_nothing_is_returned_without_a_full_volume_window():
    assert filter_inputs(_frame(19), TODAY) is None
    assert filter_inputs(_frame(0), TODAY) is None


# --- YahooFilterSource ------------------------------------------------------------------------------------


def _batch(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """What `yfinance.download(symbols, group_by="ticker")` returns: (ticker, field) columns."""
    return pd.concat(frames, axis=1)


class FakeDownload:
    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, symbols, **kwargs):
        self.calls.append((list(symbols), kwargs))
        if self.raises is not None:
            raise self.raises
        return self.result


def test_the_source_returns_each_symbols_inputs_from_one_batch():
    download = FakeDownload(_batch({"AAA": _frame(300, close=50.0), "BBB": _frame(300, close=20.0, volume=4_000_000.0)}))

    result = YahooFilterSource(download=download).inputs(["AAA", "BBB"], TODAY)

    assert result["AAA"].price == pytest.approx(50.0)
    assert result["BBB"].avg_dollar_volume == pytest.approx(80_000_000.0)
    assert len(download.calls) == 1
    assert download.calls[0][0] == ["AAA", "BBB"]


def test_the_batch_ends_after_today_and_reaches_back_over_the_scan_history():
    download = FakeDownload(_batch({"AAA": _frame(300)}))

    YahooFilterSource(download=download).inputs(["AAA"], TODAY)

    kwargs = download.calls[0][1]
    assert kwargs["end"] == "2026-10-08"  # yfinance's end is exclusive: today's forming bar comes back, then is dropped
    assert date.fromisoformat(kwargs["start"]) <= TODAY - timedelta(days=440)  # 300 sessions plus holidays


def test_a_symbol_yahoo_has_no_data_for_is_left_out():
    result = YahooFilterSource(download=FakeDownload(_batch({"AAA": _frame(300)}))).inputs(["AAA", "ZZZ"], TODAY)

    assert set(result) == {"AAA"}


def test_a_symbol_with_too_little_history_for_the_volume_window_is_left_out():
    download = FakeDownload(_batch({"AAA": _frame(300), "NEW": _frame(10)}))

    assert set(YahooFilterSource(download=download).inputs(["AAA", "NEW"], TODAY)) == {"AAA"}


def test_a_failed_download_raises_a_filter_data_error():
    source = YahooFilterSource(download=FakeDownload(raises=RuntimeError("429 Too Many Requests")))

    with pytest.raises(FilterDataError, match="429"):
        source.inputs(["AAA"], TODAY)


def test_an_empty_download_raises_a_filter_data_error():
    with pytest.raises(FilterDataError):
        YahooFilterSource(download=FakeDownload(pd.DataFrame())).inputs(["AAA"], TODAY)


# --- strategy wiring --------------------------------------------------------------------------------------


def _inputs(*, price=50.0, avg_dollar_volume=50_000_000.0, volatility=0.3, trading_days=300) -> FilterInputs:
    return FilterInputs(price=price, avg_dollar_volume=avg_dollar_volume, volatility=volatility, trading_days=trading_days)


class FakeFilterSource:
    def __init__(self, inputs=None, raises=None):
        self._inputs = inputs or {}
        self._raises = raises
        self.calls: list[tuple[list[str], date]] = []

    def inputs(self, symbols, today):
        self.calls.append((list(symbols), today))
        if self._raises is not None:
            raise self._raises
        return self._inputs


def _daily_bars(sessions: int, last: date, volume: float) -> Bars:
    dates = [d.date() for d in pd.bdate_range(end=last, periods=sessions)]
    closes = [100.0 + i for i in range(sessions)]
    index = pd.DatetimeIndex([datetime.combine(d, time(0), tzinfo=MARKET_TZ) for d in dates])
    frame = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": [volume] * sessions}, index=index)
    return Bars(Asset("AAA"), "day", frame)


class FakeIndicatorStrategy:
    """Just enough of `Strategy` for `_compute_indicators_for_ticker`."""

    _compute_indicators_for_ticker = CrossMomentumStrategy._compute_indicators_for_ticker
    _market_date = CrossMomentumStrategy._market_date

    def __init__(self, *, filter_source=None, sessions=301, min_trading_days=250):
        self.parameters = {"min_trading_days": min_trading_days, "skip_days": 21, "volatility_window": 20}
        self.vars = SimpleNamespace(alpaca_rate_limiter=SimpleNamespace(wait=lambda: None), filter_source=filter_source, filter_inputs={})
        self._bars = _daily_bars(sessions, date(2026, 10, 6), 1_000_000.0)

    def get_historical_prices(self, ticker, length, timestep):
        return self._bars

    def get_datetime(self):
        return datetime(2026, 10, 7, 12, 0, tzinfo=MARKET_TZ)

    def log_error(self, *args, **kwargs): ...

    def log_warning(self, *args, **kwargs): ...


def test_a_thin_stocks_short_iex_history_does_not_fail_min_trading_days_when_yahoo_is_the_source():
    # 281 IEX bars is plenty for the 12-1 momentum but under this min_trading_days; Yahoo's own count is what counts
    with_yahoo = FakeIndicatorStrategy(filter_source=FakeFilterSource(), sessions=281, min_trading_days=300)

    assert CrossMomentumStrategy._compute_indicators_for_ticker(with_yahoo, "AAA") is not None


def test_a_backtest_still_rejects_on_its_own_bar_count():
    backtest = FakeIndicatorStrategy(filter_source=None, sessions=281, min_trading_days=300)

    assert CrossMomentumStrategy._compute_indicators_for_ticker(backtest, "AAA") is None


class FakeScanStrategy:
    """Just enough of `Strategy` for `compute_target_portfolio`; each ticker's Alpaca-side indicators are canned."""

    compute_target_portfolio = CrossMomentumStrategy.compute_target_portfolio
    _market_date = CrossMomentumStrategy._market_date

    def _load_filter_inputs(self):
        return CrossMomentumStrategy._load_filter_inputs(self)

    def __init__(self, indicators, filter_source):
        self.parameters = dict(CONFIG)
        self.vars = SimpleNamespace(
            universe=list(indicators), target_closes={}, breadth=None, breadth_step=None, filter_source=filter_source, filter_inputs={}
        )
        self._indicators = indicators
        self.scanned: list[str] = []
        self.errors: list[str] = []
        self.infos: list[str] = []

    def get_datetime(self):
        return datetime(2026, 10, 7, 12, 0, tzinfo=MARKET_TZ)

    def _compute_indicators_for_ticker(self, ticker):
        self.scanned.append(ticker)
        return self._indicators[ticker]

    def log_info(self, message, *args, **kwargs):
        self.infos.append(message)

    def log_warning(self, *args, **kwargs): ...

    def log_error(self, message, *args, **kwargs):
        self.errors.append(message)


def _indicator(symbol, *, price=50.0, avg_dollar_volume=50_000_000.0, volatility=0.3, trading_days=300):
    closes = [float(i) for i in range(1, 301)]
    return {
        "symbol": symbol,
        "price": price,
        "avg_dollar_volume": avg_dollar_volume,
        "volatility": volatility,
        "trading_days": trading_days,
        "ret_12_1m": 0.2,
        "ret_6_1m": 0.1,
        "ret_3m": 0.05,
        "atr": None,
        "closes": closes,
        "close_series": pd.Series(closes),
    }


def _selected(fake) -> set[str]:
    target, _ = fake.compute_target_portfolio()
    return {entry["symbol"] for entry in target}


def test_the_filter_is_evaluated_on_yahoo_inputs_not_on_the_alpaca_ones():
    # Alpaca-side values all fail the filter (price 5, dollar volume 1M, volatility 2.0, 100 days); Yahoo's pass
    indicators = {"AAA": _indicator("AAA", price=5.0, avg_dollar_volume=1e6, volatility=2.0, trading_days=100)}
    fake = FakeScanStrategy(indicators, FakeFilterSource({"AAA": _inputs()}))

    assert _selected(fake) == {"AAA"}


@pytest.mark.parametrize(
    "yahoo",
    [
        _inputs(price=9.99),  # min_price 10
        _inputs(avg_dollar_volume=19_999_999.0),  # min_dollar_volume 20M
        _inputs(volatility=0.81),  # max_volatility 0.80
        _inputs(trading_days=249),  # min_trading_days 250
    ],
)
def test_each_filter_gate_rejects_on_the_yahoo_value(yahoo):
    # Alpaca-side values all pass: only Yahoo's can reject
    fake = FakeScanStrategy({"AAA": _indicator("AAA")}, FakeFilterSource({"AAA": yahoo}))

    assert _selected(fake) == set()


def test_the_filter_runs_before_any_alpaca_request():
    indicators = {"PASS": _indicator("PASS"), "FAIL": _indicator("FAIL"), "NODATA": _indicator("NODATA")}
    source = FakeFilterSource({"PASS": _inputs(), "FAIL": _inputs(price=1.0)})
    fake = FakeScanStrategy(indicators, source)

    fake.compute_target_portfolio()

    assert fake.scanned == ["PASS"]  # a symbol Yahoo rejects, or has nothing for, never costs an Alpaca call


def test_the_scan_loads_yahoo_inputs_once_for_the_whole_universe():
    universe = [f"S{i}" for i in range(10)]
    source = FakeFilterSource({symbol: _inputs() for symbol in universe})
    fake = FakeScanStrategy({symbol: _indicator(symbol) for symbol in universe}, source)

    fake.compute_target_portfolio()

    assert source.calls == [(universe, date(2026, 10, 7))]


def test_a_yahoo_outage_holds_the_book_instead_of_scoring_on_nothing():
    fake = FakeScanStrategy({"AAA": _indicator("AAA"), "BBB": _indicator("BBB")}, FakeFilterSource(raises=FilterDataError("429")))

    assert fake.compute_target_portfolio() == ([], {})
    assert fake.scanned == []  # nothing scored: a held name must not be sold as "not ranked"
    assert any("holding current positions" in message for message in fake.errors)


def test_thin_yahoo_coverage_is_treated_as_an_outage():
    universe = [f"S{i}" for i in range(10)]
    source = FakeFilterSource({symbol: _inputs() for symbol in universe[:4]})  # 40% < 50%
    fake = FakeScanStrategy({symbol: _indicator(symbol) for symbol in universe}, source)

    assert fake.compute_target_portfolio() == ([], {})
    assert fake.scanned == []
    assert any("4/10" in message for message in fake.errors)


def test_half_coverage_is_enough():
    universe = [f"S{i}" for i in range(10)]
    source = FakeFilterSource({symbol: _inputs() for symbol in universe[:5]})
    fake = FakeScanStrategy({symbol: _indicator(symbol) for symbol in universe}, source)

    fake.compute_target_portfolio()

    assert sorted(fake.scanned) == sorted(universe[:5])
    assert not any("Yahoo" in message for message in fake.errors)


def test_a_backtest_filters_on_its_own_indicators():
    indicators = {"OK": _indicator("OK"), "CHEAP": _indicator("CHEAP", price=5.0)}
    fake = FakeScanStrategy(indicators, None)

    assert _selected(fake) == {"OK"}
    assert sorted(fake.scanned) == ["CHEAP", "OK"]


# --- construction -----------------------------------------------------------------------------------------


def _built(mode: TradingMode, **kwargs):
    broker = FakeBroker(FakeClock(et(2026, 10, 6, 7), []))
    return CrossMomentumStrategy(broker, mode=mode, universe=["AAA"], **kwargs)


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_paper_and_live_build_a_yahoo_filter_source(mode):
    assert isinstance(_built(mode).vars.filter_source, YahooFilterSource)


def test_a_backtest_builds_no_filter_source():
    assert _built(TradingMode.BACKTESTING).vars.filter_source is None


def test_an_injected_filter_source_is_kept():
    source = FakeFilterSource()

    assert _built(TradingMode.PAPER, filter_source=source).vars.filter_source is source
