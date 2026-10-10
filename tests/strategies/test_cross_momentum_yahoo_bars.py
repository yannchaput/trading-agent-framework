"""Yahoo as the source of cross_momentum's daily bars in paper/live.

Alpaca's IEX bars carry ~5% of consolidated volume and skip a thin stock's quiet days, and their closes are the
last IEX trade rather than the official close, so the unchanged `apply_filters` passed 307 of the 1200-symbol
universe live against ~1,140 in a Yahoo backtest, and ranks were computed on other prices than the backtest's.
Paper/live now read every daily bar the decision uses (the stocks, SPY for the risk overlay, GLD/IEF for the
sleeve trend) from one Yahoo batch per scan, so the live decision runs the backtest's own code on the backtest's
own data. Orders and the sizing price (`get_last_price`) still go through Alpaca.
"""

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.backtesting.data.yahoo import parse_yahoo_frame
from trading_agent_framework.config import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.common.yahoo_daily_bars import YahooDailyBars
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import YahooDataError

TODAY = date(2026, 10, 7)
YESTERDAY = date(2026, 10, 6)


def _yahoo_frame(sessions, *, close=50.0, volume=1_000_000.0, last=YESTERDAY, closes=None) -> pd.DataFrame:
    """One ticker as yfinance returns it: Open/High/Low/Close/Volume, a naive session-date index."""
    index = pd.bdate_range(end=last, periods=sessions)
    values = closes if closes is not None else [close] * sessions
    return pd.DataFrame(
        {"Open": values, "High": [v + 1 for v in values], "Low": [v - 1 for v in values], "Close": values, "Volume": [volume] * sessions}, index=index
    )


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


# --- YahooDailyBars ---------------------------------------------------------------------------------------


def test_each_symbol_comes_back_as_bars_in_the_shape_the_strategy_reads():
    closes = [100.0 + i for i in range(300)]
    source = YahooDailyBars(download=FakeDownload(_batch({"AAA": _yahoo_frame(300, closes=closes)})))

    bars = source.bars(["AAA"], TODAY)["AAA"]

    assert isinstance(bars, Bars)
    assert bars.timestep == "day"
    assert list(bars.pandas_df.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.pandas_df["close"].tolist() == closes
    assert bars.pandas_df.index[-1] == pd.Timestamp(datetime(2026, 10, 6, 16), tz=MARKET_TZ)  # at the close, as in a backtest


def test_the_bars_are_exactly_the_backtests_parse_of_the_same_download():
    frame = _yahoo_frame(300, closes=[100.0 + i for i in range(300)])

    bars = YahooDailyBars(download=FakeDownload(_batch({"AAA": frame}))).bars(["AAA"], TODAY)["AAA"]

    pd.testing.assert_frame_equal(bars.pandas_df, parse_yahoo_frame(frame))


def test_a_timezone_aware_yahoo_index_reads_the_same_sessions():
    frame = _yahoo_frame(300)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(MARKET_TZ)

    bars = YahooDailyBars(download=FakeDownload(_batch({"AAA": frame}))).bars(["AAA"], TODAY)["AAA"]

    assert bars.pandas_df.index[-1].date() == YESTERDAY
    assert len(bars) == 300


def test_a_symbol_whose_data_cannot_be_parsed_is_left_out_not_raised():
    broken = _yahoo_frame(300).astype(object)
    broken.iloc[5, broken.columns.get_loc("Close")] = "n/a"

    result = YahooDailyBars(download=FakeDownload(_batch({"AAA": _yahoo_frame(300), "BAD": broken}))).bars(["AAA", "BAD"], TODAY)

    assert set(result) == {"AAA"}


def test_one_batch_serves_every_symbol():
    download = FakeDownload(_batch({"AAA": _yahoo_frame(300), "BBB": _yahoo_frame(300, close=20.0)}))

    result = YahooDailyBars(download=download).bars(["AAA", "BBB"], TODAY)

    assert set(result) == {"AAA", "BBB"}
    assert len(download.calls) == 1
    assert download.calls[0][0] == ["AAA", "BBB"]


def test_the_batch_is_adjusted_ends_after_today_and_reaches_back_over_the_scan_history():
    download = FakeDownload(_batch({"AAA": _yahoo_frame(300)}))

    YahooDailyBars(download=download).bars(["AAA"], TODAY)

    kwargs = download.calls[0][1]
    assert kwargs["auto_adjust"] is True  # split- and dividend-adjusted, like the backtest and Alpaca's Adjustment.ALL
    assert kwargs["end"] == "2026-10-08"  # yfinance's end is exclusive: today's forming bar comes back, the strategy drops it
    assert date.fromisoformat(kwargs["start"]) <= TODAY - timedelta(days=440)  # 301 sessions plus holidays


def test_rows_with_a_missing_price_or_volume_are_dropped():
    frame = _yahoo_frame(300)
    frame.iloc[10, frame.columns.get_loc("Close")] = float("nan")
    frame.iloc[11, frame.columns.get_loc("Volume")] = float("nan")

    bars = YahooDailyBars(download=FakeDownload(_batch({"AAA": frame}))).bars(["AAA"], TODAY)["AAA"]

    assert len(bars) == 298


def test_a_symbol_yahoo_has_no_data_for_is_left_out():
    result = YahooDailyBars(download=FakeDownload(_batch({"AAA": _yahoo_frame(300)}))).bars(["AAA", "ZZZ"], TODAY)

    assert set(result) == {"AAA"}


def test_a_symbol_with_only_missing_rows_is_left_out():
    dead = _yahoo_frame(300)
    dead[["Open", "High", "Low", "Close", "Volume"]] = float("nan")

    result = YahooDailyBars(download=FakeDownload(_batch({"AAA": _yahoo_frame(300), "DEAD": dead}))).bars(["AAA", "DEAD"], TODAY)

    assert set(result) == {"AAA"}


def test_a_failed_download_raises_a_yahoo_data_error():
    source = YahooDailyBars(download=FakeDownload(raises=RuntimeError("429 Too Many Requests")))

    with pytest.raises(YahooDataError, match="429"):
        source.bars(["AAA"], TODAY)


def test_an_empty_download_raises_a_yahoo_data_error():
    with pytest.raises(YahooDataError):
        YahooDailyBars(download=FakeDownload(pd.DataFrame())).bars(["AAA"], TODAY)


# --- strategy wiring --------------------------------------------------------------------------------------


class FakeBarsSource:
    def __init__(self, bars=None, raises=None):
        self._bars = bars or {}
        self._raises = raises
        self.calls: list[tuple[list[str], date]] = []

    def bars(self, symbols, today):
        self.calls.append((list(symbols), today))
        if self._raises is not None:
            raise self._raises
        return {symbol: self._bars[symbol] for symbol in symbols if symbol in self._bars}


def _bars(symbol="AAA", sessions=301, *, last=YESTERDAY, close=50.0, volume=1_000_000.0) -> Bars:
    """Bars as the source hands them over: lowercase OHLCV, midnight-ET index."""
    frame = _yahoo_frame(sessions, last=last, close=close, volume=volume)
    frame.columns = [c.lower() for c in frame.columns]
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(MARKET_TZ)
    return Bars(Asset(symbol), "day", frame)


def _built(mode: TradingMode, **kwargs):
    broker = FakeBroker(FakeClock(et(2026, 10, 7, 7), []))
    return CrossMomentumStrategy(broker, mode=mode, universe=["AAA", "BBB"], **kwargs)


class AlpacaRecorder:
    """Stands in for the framework's Alpaca-backed `Strategy.get_historical_prices`."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def method(self):
        def get_historical_prices(strategy, asset, length, timestep="day", **kwargs):
            self.calls.append((str(asset), length))

        return get_historical_prices


@pytest.fixture
def alpaca(monkeypatch) -> AlpacaRecorder:
    recorder = AlpacaRecorder()
    monkeypatch.setattr(Strategy, "get_historical_prices", recorder.method())
    return recorder


def test_paper_and_live_build_a_yahoo_bars_source():
    for mode in (TradingMode.PAPER, TradingMode.LIVE):
        assert isinstance(_built(mode).vars.bars_source, YahooDailyBars)


def test_a_backtest_builds_no_bars_source():
    assert _built(TradingMode.BACKTESTING).vars.bars_source is None


def test_an_injected_bars_source_is_kept():
    source = FakeBarsSource()

    assert _built(TradingMode.PAPER, bars_source=source).vars.bars_source is source


def test_daily_bars_come_from_the_yahoo_batch_without_an_alpaca_request(alpaca):
    strategy = _built(TradingMode.PAPER, bars_source=FakeBarsSource({"AAA": _bars("AAA", 301)}))
    strategy.vars.yahoo_bars = {"AAA": _bars("AAA", 301)}

    result = strategy.get_historical_prices("AAA", 100, "day")

    assert len(result) == 100
    assert result.pandas_df["close"].iloc[-1] == 50.0
    assert alpaca.calls == []


def test_a_symbol_missing_from_the_yahoo_batch_has_no_bars_and_no_alpaca_fallback(alpaca):
    strategy = _built(TradingMode.PAPER, bars_source=FakeBarsSource())

    assert strategy.get_historical_prices("ZZZ", 100, "day") is None
    assert alpaca.calls == []


def test_a_backtest_still_reads_its_bars_from_the_framework(alpaca):
    strategy = _built(TradingMode.BACKTESTING)

    strategy.get_historical_prices("AAA", 100, "day")

    assert alpaca.calls == [("AAA", 100)]


def test_the_scan_decides_on_yahoo_bars_end_to_end(alpaca):
    strategy = _built(TradingMode.PAPER, bars_source=FakeBarsSource())
    strategy.vars.yahoo_bars = {"AAA": _bars("AAA", 301, close=50.0, volume=2_000_000.0)}
    strategy.vars.universe = ["AAA"]

    result = strategy._compute_indicators_for_ticker("AAA")

    assert result is not None
    assert result["price"] == 50.0
    assert result["avg_dollar_volume"] == pytest.approx(2_000_000.0 * 50.0)  # consolidated volume, not IEX's 5%
    assert result["trading_days"] == 300
    assert alpaca.calls == []


class FakeScanStrategy:
    """Just enough of `Strategy` for the Yahoo step of `compute_target_portfolio`."""

    compute_target_portfolio = CrossMomentumStrategy.compute_target_portfolio
    _market_date = CrossMomentumStrategy._market_date

    def _load_yahoo_bars(self):
        return CrossMomentumStrategy._load_yahoo_bars(self)

    def _fetch_yahoo_bars(self):
        return CrossMomentumStrategy._fetch_yahoo_bars(self)

    def __init__(self, universe, bars_source, held=()):
        self.parameters = dict(CONFIG)
        self._held = list(held)
        self.vars = SimpleNamespace(
            universe=universe, target_closes={}, breadth=None, breadth_step=None, bars_source=bars_source, yahoo_bars={}
        )
        self.scanned: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.slept: list[float] = []

    def get_datetime(self):
        return datetime(2026, 10, 7, 12, 0, tzinfo=MARKET_TZ)

    def sleep(self, seconds):
        self.slept.append(seconds)

    def get_positions(self):
        return [SimpleNamespace(asset=Asset(symbol), quantity=1.0) for symbol in self._held]

    def _compute_indicators_for_ticker(self, ticker):
        self.scanned.append(ticker)
        return None

    def log_info(self, *args, **kwargs): ...

    def log_warning(self, message, *args, **kwargs):
        self.warnings.append(message)

    def log_error(self, message, *args, **kwargs):
        self.errors.append(message)


def _universe(count=10):
    return [f"S{i}" for i in range(count)]


def test_the_scan_loads_one_batch_for_the_universe_spy_and_the_sleeve_trend_assets():
    universe = _universe()
    source = FakeBarsSource({symbol: _bars(symbol) for symbol in universe})
    fake = FakeScanStrategy(universe, source)

    fake.compute_target_portfolio()

    assert source.calls == [([*universe, "SPY", "GLD", "IEF"], date(2026, 10, 7))]
    assert set(fake.vars.yahoo_bars) == set(universe)


def test_a_yahoo_outage_holds_the_book_instead_of_scoring_on_nothing():
    fake = FakeScanStrategy(_universe(2), FakeBarsSource(raises=YahooDataError("429")))

    assert fake.compute_target_portfolio() == ([], {})
    assert fake.scanned == []  # nothing scored: a held name must not be sold as "not ranked"
    assert any("holding current positions" in message for message in fake.errors)


def test_thin_yahoo_coverage_of_the_universe_is_treated_as_an_outage():
    universe = _universe()
    source = FakeBarsSource({symbol: _bars(symbol) for symbol in universe[:4]} | {"SPY": _bars("SPY")})  # 4/10 stocks, SPY answered
    fake = FakeScanStrategy(universe, source)

    assert fake.compute_target_portfolio() == ([], {})
    assert fake.scanned == []
    assert any("4/10" in message for message in fake.errors)


def test_half_the_universe_is_enough():
    universe = _universe()
    fake = FakeScanStrategy(universe, FakeBarsSource({symbol: _bars(symbol) for symbol in universe[:5]}))

    fake.compute_target_portfolio()

    assert sorted(fake.scanned) == sorted(universe)  # a symbol without bars is still visited and skipped one by one
    assert not any("Yahoo" in message for message in fake.errors)


def test_a_backtest_does_not_touch_yahoo():
    fake = FakeScanStrategy(["AAA"], None)

    fake.compute_target_portfolio()

    assert fake.scanned == ["AAA"]
    assert fake.vars.yahoo_bars == {}


def test_a_held_stock_missing_from_the_yahoo_batch_holds_the_book():
    # 9/10 coverage is plenty, but S9 is held: scanning without it would sell it as "not ranked"
    universe = _universe()
    fake = FakeScanStrategy(universe, FakeBarsSource({symbol: _bars(symbol) for symbol in universe[:9]}), held=["S9", "S0"])

    assert fake.compute_target_portfolio() == ([], {})
    assert fake.scanned == []
    assert any("S9" in message and "holding current positions" in message for message in fake.errors)


def test_held_stocks_that_all_have_bars_let_the_scan_run():
    universe = _universe()
    fake = FakeScanStrategy(universe, FakeBarsSource({symbol: _bars(symbol) for symbol in universe[:9]}), held=["S0", "S8"])

    fake.compute_target_portfolio()

    assert sorted(fake.scanned) == sorted(universe)


def test_a_held_name_outside_the_universe_does_not_block_the_scan():
    # never requested from Yahoo: the rebalance sells it as "not ranked" by design (it left the universe)
    universe = _universe()
    fake = FakeScanStrategy(universe, FakeBarsSource({symbol: _bars(symbol) for symbol in universe}), held=["GONE", "SHV"])

    fake.compute_target_portfolio()

    assert sorted(fake.scanned) == sorted(universe)


class SequenceSource:
    """Answers each call with the next response: a {symbol: bars} dict, or an exception to raise."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    def bars(self, symbols, today):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return {symbol: response[symbol] for symbol in symbols if symbol in response}


def _full(universe):
    return {symbol: _bars(symbol) for symbol in universe}


def test_a_failed_yahoo_download_is_retried_before_the_week_is_given_up():
    universe = _universe()
    source = SequenceSource(YahooDataError("429"), _full(universe))
    fake = FakeScanStrategy(universe, source)

    fake.compute_target_portfolio()

    assert source.calls == 2
    assert fake.slept == [60.0]  # through Strategy.sleep, so the clock (and order hooks) drive the wait
    assert sorted(fake.scanned) == sorted(universe)
    assert not any("Yahoo" in message for message in fake.errors)
    assert any("429" in message and "retry" in message for message in fake.warnings)


def test_a_thin_batch_is_retried_too():
    universe = _universe()
    source = SequenceSource(_full(universe[:3]), _full(universe))
    fake = FakeScanStrategy(universe, source)

    fake.compute_target_portfolio()

    assert source.calls == 2
    assert sorted(fake.scanned) == sorted(universe)


def test_a_batch_missing_a_held_stock_is_retried_too():
    universe = _universe()
    source = SequenceSource(_full(universe[:9]), _full(universe))
    fake = FakeScanStrategy(universe, source, held=["S9"])

    fake.compute_target_portfolio()

    assert source.calls == 2
    assert sorted(fake.scanned) == sorted(universe)


def test_yahoo_gets_three_attempts_then_the_book_is_held_with_one_error():
    source = SequenceSource(YahooDataError("429"))
    fake = FakeScanStrategy(_universe(), source)

    assert fake.compute_target_portfolio() == ([], {})
    assert source.calls == 3
    assert fake.slept == [60.0, 180.0]
    assert len(fake.errors) == 1 and "429" in fake.errors[0] and "holding current positions" in fake.errors[0]


# --- market regime ----------------------------------------------------------------------------------------


def _spy_uptrend() -> Bars:
    frame = _yahoo_frame(301, closes=[100 * 1.001**i for i in range(301)])
    frame.columns = [c.lower() for c in frame.columns]
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(MARKET_TZ)
    return Bars(Asset("SPY"), "day", frame)


def test_the_regime_reads_spy_from_yahoo_before_any_scan_has_loaded_bars(alpaca):
    # The executor refreshes the regime before the session opens, when the scan's batch (`vars.yahoo_bars`) is empty.
    source = FakeBarsSource({"SPY": _spy_uptrend()})
    strategy = _built(TradingMode.PAPER, bars_source=source)
    assert strategy.vars.yahoo_bars == {}

    strategy._refresh_regime()

    assert strategy.regime == 1
    assert source.calls == [(["SPY"], YESTERDAY + timedelta(days=1))]
    assert alpaca.calls == []  # not Alpaca's IEX bars: the regime reads the data the decisions use


def test_the_regime_fetch_leaves_no_bars_behind(alpaca):
    strategy = _built(TradingMode.PAPER, bars_source=FakeBarsSource({"SPY": _spy_uptrend()}))

    strategy._refresh_regime()

    assert strategy.vars.yahoo_bars == {}


def test_a_failed_yahoo_lookup_keeps_the_previous_regime_and_says_why(alpaca, caplog):
    strategy = _built(TradingMode.PAPER, bars_source=FakeBarsSource(raises=YahooDataError("429")))
    strategy.regime = -1

    with caplog.at_level("WARNING"):
        strategy._refresh_regime()

    assert strategy.regime == -1
    assert "429" in caplog.text
    assert "fewer than" not in caplog.text  # a lookup failure is not a shortage of history


def test_a_backtest_regime_still_reads_the_framework_bars(alpaca):
    strategy = _built(TradingMode.BACKTESTING)

    strategy._refresh_regime()

    assert alpaca.calls == [("SPY", 273)]
