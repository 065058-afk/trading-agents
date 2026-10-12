"""Yahoo history caching and batched Alpaca daily bars."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
MAX_STALE_DAYS = 5
GAP_CHECK_DAYS = 30
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.-]*$")


class DataQualityError(ValueError):
	"""Raised when requested market data is missing or not usable."""


def _normalize_symbols(
	symbols: Sequence[str], *, include_benchmarks: bool = False
) -> list[str]:
	normalized = [str(symbol).strip().upper() for symbol in symbols]
	if include_benchmarks:
		normalized.extend(("SPY", "QQQ"))
	if not normalized:
		raise ValueError("at least one ticker symbol is required")
	unique: list[str] = []
	for symbol in normalized:
		if not _SYMBOL_PATTERN.fullmatch(symbol):
			raise ValueError(f"Invalid ticker symbol: {symbol!r}")
		if symbol not in unique:
			unique.append(symbol)
	return unique


def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
	normalized = frame.copy()
	normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
	missing = [column for column in OHLCV_COLUMNS if column not in normalized.columns]
	if missing:
		raise DataQualityError(f"OHLCV data is missing required columns: {', '.join(missing)}")
	normalized = normalized.loc[:, OHLCV_COLUMNS]
	index = pd.to_datetime(normalized.index)
	if index.tz is not None:
		index = index.tz_convert("America/New_York").tz_localize(None)
	normalized.index = index.normalize()
	normalized.index.name = "timestamp"
	normalized = normalized.dropna(subset=list(OHLCV_COLUMNS), how="all")
	return normalized[~normalized.index.duplicated(keep="last")].sort_index()


def _ticker_frame(download: pd.DataFrame, symbol: str) -> pd.DataFrame:
	if isinstance(download.columns, pd.MultiIndex):
		for level in range(download.columns.nlevels):
			if symbol in download.columns.get_level_values(level):
				download = download.xs(symbol, axis=1, level=level, drop_level=True)
				break
	return _normalize_ohlcv(download)


def _market_holidays(start: date, end: date) -> set[date]:
	federal = USFederalHolidayCalendar().holidays(
		start=pd.Timestamp(start), end=pd.Timestamp(end), return_name=True
	)
	holidays = {
		day.date()
		for day, name in federal.items()
		if name not in {"Columbus Day", "Veterans Day"}
	}
	holidays.update(day.date() for day in GoodFriday.dates(start, end))
	return holidays


def _latest_expected_session(today: date | None = None) -> date:
	now = datetime.now(ZoneInfo("America/New_York"))
	current = today or now.date()
	if today is None and now.time() < time(17):
		current -= timedelta(days=1)
	holidays = _market_holidays(current - timedelta(days=14), current)
	while current.weekday() >= 5 or current in holidays:
		current -= timedelta(days=1)
	return current


def _check_recent_data(symbol: str, frame: pd.DataFrame) -> None:
	if frame.empty:
		raise DataQualityError(f"{symbol}: no recent bar was returned")

	latest = pd.Timestamp(frame.index.max()).date()
	age = (datetime.now(ZoneInfo("America/New_York")).date() - latest).days
	if age > MAX_STALE_DAYS:
		raise DataQualityError(
			f"{symbol}: latest bar is stale ({latest.isoformat()}, {age} days old)"
		)

	last_expected_session = min(latest, _latest_expected_session())
	first_check_date = max(
		frame.index.min().date(), last_expected_session - timedelta(days=GAP_CHECK_DAYS)
	)
	holidays = _market_holidays(first_check_date, last_expected_session)
	expected = {
		day.date()
		for day in pd.date_range(first_check_date, last_expected_session, freq="B")
		if day.date() not in holidays
	}
	actual = {pd.Timestamp(day).date() for day in frame.index}
	gaps = sorted(expected - actual)
	if gaps:
		dates = ", ".join(day.isoformat() for day in gaps[:5])
		suffix = " ..." if len(gaps) > 5 else ""
		raise DataQualityError(f"{symbol}: missing daily bars on {dates}{suffix}")


def history_cache(symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
	"""Return cached daily history from 2000 onward, fetching only missing dates."""
	requested = _normalize_symbols(symbols, include_benchmarks=True)
	CACHE_DIR.mkdir(parents=True, exist_ok=True)

	cached: dict[str, pd.DataFrame] = {}
	missing: list[str] = []
	starts: list[date] = []
	for symbol in requested:
		path = CACHE_DIR / f"{symbol}.parquet"
		if path.exists():
			cached[symbol] = _normalize_ohlcv(pd.read_parquet(path))
			if not cached[symbol].empty:
				latest = pd.Timestamp(cached[symbol].index.max()).date()
				if latest < _latest_expected_session():
					missing.append(symbol)
					starts.append(latest + timedelta(days=1))
				continue
		missing.append(symbol)
		starts.append(date(2000, 1, 1))

	if missing:
		import yfinance as yf

		downloaded = yf.download(
			tickers=missing,
			start=min(starts).isoformat(),
			interval="1d",
			group_by="ticker",
			auto_adjust=False,
			progress=False,
			threads=True,
		)
		for symbol in missing:
			new_rows = _ticker_frame(downloaded, symbol)
			existing = cached.get(symbol, pd.DataFrame(columns=OHLCV_COLUMNS))
			if not existing.empty and not new_rows.empty:
				new_rows = new_rows.loc[new_rows.index > existing.index.max()]
			combined = pd.concat((existing, new_rows))
			combined = combined[~combined.index.duplicated(keep="last")].sort_index()
			if combined.empty:
				raise DataQualityError(f"{symbol}: Yahoo returned no daily bars")
			_check_recent_data(symbol, combined)
			combined.to_parquet(CACHE_DIR / f"{symbol}.parquet")
			cached[symbol] = combined

	for symbol in requested:
		if symbol not in cached or cached[symbol].empty:
			raise DataQualityError(f"{symbol}: no cached history or recent bar")
		_check_recent_data(symbol, cached[symbol])
	return {symbol: cached[symbol] for symbol in requested}


def _bars_by_symbol(bars: pd.DataFrame, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
	if bars.empty:
		return {}
	if isinstance(bars.index, pd.MultiIndex):
		symbol_level = next(
			(level for level, name in enumerate(bars.index.names) if name and name.lower() == "symbol"),
			0,
		)
		return {
			symbol: bars.xs(symbol, level=symbol_level).copy()
			for symbol in symbols
			if symbol in bars.index.get_level_values(symbol_level)
		}
	if len(symbols) == 1:
		return {symbols[0]: bars}
	if "symbol" in bars.columns:
		return {symbol: rows.drop(columns="symbol") for symbol, rows in bars.groupby("symbol")}
	return {}


def _frames_for_check(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
	if frame.empty:
		return {}
	if isinstance(frame.index, pd.MultiIndex):
		symbol_level = next(
			(level for level, name in enumerate(frame.index.names) if name and name.lower() == "symbol"),
			0,
		)
		symbols = frame.index.get_level_values(symbol_level).unique()
		return _bars_by_symbol(frame, [str(symbol) for symbol in symbols])
	if "symbol" in frame.columns:
		return {
			str(symbol).upper(): rows.drop(columns="symbol").copy()
			for symbol, rows in frame.groupby("symbol")
		}
	return {"dataframe": frame.copy()}


def check_data(df: pd.DataFrame) -> None:
	"""Raise DataQualityError if any represented symbol has empty, stale, or gapped bars."""
	if df.empty:
		raise DataQualityError("no recent bar was returned for any symbol")
	grouped = _frames_for_check(df)
	if not grouped:
		raise DataQualityError("no recent bar was returned for any symbol")
	for symbol, frame in grouped.items():
		if "timestamp" in frame.columns:
			frame = frame.set_index("timestamp")
		_check_recent_data(symbol, _normalize_ohlcv(frame))


def _ohlcv_table(bars: pd.DataFrame, requested: Sequence[str]) -> pd.DataFrame:
	if isinstance(bars.index, pd.MultiIndex):
		index_names = list(bars.index.names)
		reset = bars.reset_index()
		reset_names = list(reset.columns[: len(index_names)])
		symbol_level = next(
			(level for level, name in enumerate(index_names) if name and name.lower() == "symbol"),
			0,
		)
		timestamp_level = next((level for level in range(len(index_names)) if level != symbol_level), None)
		if timestamp_level is None:
			raise DataQualityError("Alpaca bars are missing timestamps")
			reset = reset.rename(
				columns={reset_names[symbol_level]: "symbol", reset_names[timestamp_level]: "timestamp"}
			)
	else:
		reset = bars.copy()
		if "timestamp" not in reset.columns:
			reset = reset.reset_index()
			reset = reset.rename(columns={reset.columns[0]: "timestamp"})
		if "symbol" not in reset.columns:
			if len(requested) != 1:
				raise DataQualityError("Alpaca bars are missing symbol identifiers")
			reset["symbol"] = requested[0]

	reset.columns = [str(column).strip().lower() for column in reset.columns]
	columns = ["symbol", "timestamp", *OHLCV_COLUMNS]
	missing = [column for column in columns if column not in reset.columns]
	if missing:
		raise DataQualityError(f"Alpaca bars are missing required columns: {', '.join(missing)}")
	table = reset.loc[:, columns].copy()
	table["symbol"] = table["symbol"].astype(str).str.upper()
	table["timestamp"] = pd.to_datetime(table["timestamp"])
	return table.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def get_ohlcv_batch(
	symbols: Sequence[str], days: int, *, student: str = "A"
) -> pd.DataFrame:
	"""Fetch daily IEX bars in one request and return one table for all symbols."""
	requested = _normalize_symbols(symbols)
	if isinstance(days, bool) or not isinstance(days, int) or days < 1:
		raise ValueError("days must be a positive integer")
	if not student or not student.isalpha():
		raise ValueError("student must identify the Alpaca credential suffix")

	key = os.environ.get(f"ALPACA_KEY_{student}")
	secret = os.environ.get(f"ALPACA_SECRET_{student}")
	if not key or not secret:
		raise ValueError(f"Missing ALPACA_KEY_{student} or ALPACA_SECRET_{student} environment variable")

	from alpaca.data.enums import DataFeed
	from alpaca.data.historical import StockHistoricalDataClient
	from alpaca.data.requests import StockBarsRequest
	from alpaca.data.timeframe import TimeFrame

	client = StockHistoricalDataClient(key, secret)
	request = StockBarsRequest(
		symbol_or_symbols=requested,
		timeframe=TimeFrame.Day,
		start=datetime.now(timezone.utc) - timedelta(days=days),
		feed=DataFeed.IEX,
	)
	bars = client.get_stock_bars(request).df
	grouped = _bars_by_symbol(bars, requested)
	for symbol in requested:
		if symbol not in grouped:
			raise DataQualityError(f"{symbol}: no recent bar was returned")
	table = _ohlcv_table(bars, requested)
	check_data(table)
	return table
