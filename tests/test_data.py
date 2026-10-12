from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from mcp_server import data


def _recent_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": range(100, 100 + len(index)),
            "High": range(101, 101 + len(index)),
            "Low": range(99, 99 + len(index)),
            "Close": range(100, 100 + len(index)),
            "Volume": [1000] * len(index),
        },
        index=index,
    )


def _recent_sessions(count: int = 5) -> pd.DatetimeIndex:
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    return pd.bdate_range(end=today, periods=count)


def test_history_cache_second_run_makes_no_yahoo_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    symbols = ["AAPL", "SPY", "QQQ"]
    index = _recent_sessions()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        frames = {symbol: _recent_frame(index) for symbol in symbols}
        result = pd.concat(frames, axis=1)
        result.columns.names = ["Ticker", "Price"]
        return result

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))

    first = data.history_cache(["AAPL"])
    second = data.history_cache(["AAPL"])

    assert set(first) == set(symbols)
    assert set(second) == set(symbols)
    assert len(calls) == 1
    assert calls[0]["tickers"] == symbols
    assert all((tmp_path / f"{symbol}.parquet").exists() for symbol in symbols)


def test_get_ohlcv_batch_makes_one_iex_request(monkeypatch):
    requested_symbols = ["AAPL", "MSFT"]
    index = _recent_sessions(3)
    bars = pd.concat(
        {symbol: _recent_frame(index).rename(columns=str.lower) for symbol in requested_symbols},
        names=["symbol", "timestamp"],
    )
    call_count = 0
    captured = {}

    class FakeClient:
        def __init__(self, key, secret):
            assert key == "paper-key"
            assert secret == "paper-secret"

        def get_stock_bars(self, request):
            nonlocal call_count
            call_count += 1
            captured["request"] = request
            return types.SimpleNamespace(df=bars)

    class FakeRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    alpaca_modules = {
        "alpaca": types.ModuleType("alpaca"),
        "alpaca.data": types.ModuleType("alpaca.data"),
        "alpaca.data.enums": types.SimpleNamespace(DataFeed=types.SimpleNamespace(IEX="iex")),
        "alpaca.data.historical": types.SimpleNamespace(StockHistoricalDataClient=FakeClient),
        "alpaca.data.requests": types.SimpleNamespace(StockBarsRequest=FakeRequest),
        "alpaca.data.timeframe": types.SimpleNamespace(TimeFrame=types.SimpleNamespace(Day="1Day")),
    }
    for name, module in alpaca_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ALPACA_KEY_A", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_A", "paper-secret")

    result = data.get_ohlcv_batch(["AAPL", "MSFT"], 30)

    assert list(result.columns) == ["symbol", "timestamp", *data.OHLCV_COLUMNS]
    assert set(result["symbol"]) == set(requested_symbols)
    assert call_count == 1
    assert captured["request"].symbol_or_symbols == requested_symbols
    assert captured["request"].feed == "iex"


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(columns=data.OHLCV_COLUMNS), "no recent bar"),
        (
            _recent_frame(
                pd.DatetimeIndex([pd.Timestamp.now().normalize() - pd.Timedelta(days=20)])
            ),
            "stale",
        ),
        (
            _recent_frame(pd.DatetimeIndex([_recent_sessions(3)[0], _recent_sessions(3)[-1]])),
            "missing daily bars",
        ),
    ],
)
def test_data_quality_errors_are_clear(frame, message):
    with pytest.raises(data.DataQualityError, match=message):
        data.check_data(frame)


def test_old_historical_gaps_do_not_reject_recent_data():
    historical = _recent_frame(
        pd.DatetimeIndex(["2001-09-10", "2001-09-17"])
    )
    recent = _recent_frame(_recent_sessions(24))
    frame = pd.concat([historical, recent]).sort_index()

    data.check_data(frame)