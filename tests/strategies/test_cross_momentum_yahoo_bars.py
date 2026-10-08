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

from trading_agent_framework.config import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.strategies.cross_momentum.yahoo_daily_bars import YahooDailyBars
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
    assert bars.pandas_df.index[-1] == pd.Timestamp(YESTERDAY, tz=MARKET_TZ)  # live Alpaca bars are stamped at midnight ET


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

    def __init__(self, universe, bars_source):
        self.parameters = dict(CONFIG)
        self.vars = SimpleNamespace(
            universe=universe, target_closes={}, breadth=None, breadth_step=None, bars_source=bars_source, yahoo_bars={}
        )
        self.scanned: list[str] = []
        self.errors: list[str] = []

    def get_datetime(self):
        return datetime(2026, 10, 7, 12, 0, tzinfo=MARKET_TZ)

    def _compute_indicators_for_ticker(self, ticker):
        self.scanned.append(ticker)
        return None

    def log_info(self, *args, **kwargs): ...

    def log_warning(self, *args, **kwargs): ...

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
