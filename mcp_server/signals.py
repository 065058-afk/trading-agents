"""Daily trading signals built from cached OHLCV and free market-data APIs."""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

warnings.filterwarnings("ignore", category=Warning, module=r"^(numba|pandas_ta)(\.|$)")

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

from mcp_server.data import history_cache

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs" / "signals"
FINNHUB_URL = "https://finnhub.io/api/v1"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
RANDOM_SEED = 42
ATR_SIZE_FLOOR = 0.25
GAP = 5
FEATURE_COLUMNS = (
	"roc5",
	"roc20",
	"rsi14",
	"macd_histogram",
	"adx",
	"distance_ema20",
	"distance_ema50",
	"natr",
	"bb_width",
	"bbp",
	"cmf",
	"mfi",
	"obv_slope",
	"return_zscore20",
)
STRONG_NEGATIVE_EVENTS = {
	"bankruptcy",
	"fraud",
	"default",
	"delisting",
	"going_concern",
	"criminal_investigation",
	"accounting_irregularity",
	"restatement",
}
FINNHUB_NEWS_WORKERS = 5
LOGGER = logging.getLogger(__name__)


class FinnhubAuthenticationError(RuntimeError):
	"""Raised when Finnhub rejects the configured API key."""


def _check_finnhub_response(response: requests.Response) -> None:
	if getattr(response, "status_code", 200) in {401, 403}:
		raise FinnhubAuthenticationError(
			"Finnhub rejected FINNHUB_API_KEY; verify the key in the project .env file"
		)
	response.raise_for_status()


class SignalScores(dict[str, float]):
	"""Score mapping; news results also carry per-symbol veto metadata."""

	def __init__(self, *args: Any, vetoes: set[str] | None = None, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self.vetoes = vetoes or set()


def _symbols(symbols: Sequence[str]) -> list[str]:
	result = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
	if not result or any(not symbol for symbol in result):
		raise ValueError("at least one non-empty ticker symbol is required")
	return result


def _cutoff(as_of: date | datetime | pd.Timestamp | None) -> pd.Timestamp | None:
	if as_of is None:
		return None
	cutoff = pd.Timestamp(as_of)
	if cutoff.tzinfo is not None:
		cutoff = cutoff.tz_convert("America/New_York").tz_localize(None)
	return cutoff.normalize()


def _load_history(
	symbols: Sequence[str],
	as_of: date | datetime | pd.Timestamp | None = None,
	source: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
	frames: Mapping[str, pd.DataFrame]
	if source is None:
		frames = history_cache(list(symbols))
	else:
		frames = source
	cutoff = _cutoff(as_of)
	result: dict[str, pd.DataFrame] = {}
	for symbol, source in frames.items():
		frame = source.copy()
		frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
		frame.index = pd.to_datetime(frame.index)
		if frame.index.tz is not None:
			frame.index = frame.index.tz_convert("America/New_York").tz_localize(None)
		frame.index = frame.index.normalize()
		frame = frame.sort_index()
		if cutoff is not None:
			frame = frame.loc[frame.index <= cutoff]
		result[str(symbol).upper()] = frame
	return result


def _last(series: pd.Series | None) -> float:
	if series is None or series.empty:
		return math.nan
	value = pd.to_numeric(series, errors="coerce").iloc[-1]
	return float(value) if pd.notna(value) and np.isfinite(value) else math.nan


def _score(value: float) -> float:
	return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else 0.0


def _zero_scores(symbols: Sequence[str]) -> dict[str, float]:
	return {symbol: 0.0 for symbol in symbols}


def signal_trend(
	symbols: Sequence[str],
	*,
	as_of: date | datetime | pd.Timestamp | None = None,
	history: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
	"""Score EMA20/EMA50 alignment, filtered by ADX and ATR-sized with a 0.25 floor."""
	requested = _symbols(symbols)
	frames = _load_history(requested, as_of, history)
	scores = _zero_scores(requested)
	for symbol in requested:
		frame = frames.get(symbol)
		if frame is None or len(frame) < 60:
			continue
		close = frame["close"].astype(float)
		atr = ta.atr(frame["high"], frame["low"], close, length=14)
		adx_frame = ta.adx(frame["high"], frame["low"], close, length=14)
		if atr is None or adx_frame is None:
			continue
		adx_column = next((column for column in adx_frame if str(column).startswith("ADX_")), None)
		atr_value = _last(atr)
		adx_value = _last(adx_frame[adx_column]) if adx_column else math.nan
		ema20 = _last(ta.ema(close, length=20))
		ema50 = _last(ta.ema(close, length=50))
		price = _last(close)
		if not all(np.isfinite(value) for value in (atr_value, adx_value, ema20, ema50, price)) or atr_value <= 0:
			continue
		alignment = 0.5 * (
			math.tanh((price - ema20) / atr_value)
			+ math.tanh((price - ema50) / atr_value)
		)
		adx_filter = float(np.clip((adx_value - 10.0) / 20.0, 0.0, 1.0))
		atr_fraction = atr_value / price if price > 0 else math.inf
		atr_size = max(ATR_SIZE_FLOOR, min(1.0, 0.02 / atr_fraction))
		scores[symbol] = _score(alignment * adx_filter * atr_size)
	return scores


def _rank_centered(values: Mapping[str, float]) -> dict[str, float]:
	valid = {symbol: value for symbol, value in values.items() if np.isfinite(value)}
	if not valid:
		return {}
	if len(valid) == 1:
		symbol, value = next(iter(valid.items()))
		return {symbol: math.tanh(value / 0.05)}
	ranks = pd.Series(valid, dtype=float).rank(method="average", pct=True)
	return {symbol: float(2.0 * ranks[symbol] - 1.0) for symbol in valid}


def _rotation_scores(
	requested: Sequence[str], frames: Mapping[str, pd.DataFrame]
) -> dict[str, float]:
	spy = frames.get("SPY")
	if spy is None or len(spy) < 61:
		return _zero_scores(requested)
	spy_close = spy["close"].astype(float)
	spy20 = _last(spy_close.pct_change(20))
	spy60 = _last(spy_close.pct_change(60))
	rel20: dict[str, float] = {}
	rel60: dict[str, float] = {}
	closeness: dict[str, float] = {}
	for symbol in requested:
		frame = frames.get(symbol)
		if frame is None or len(frame) < 61:
			continue
		close = frame["close"].astype(float)
		ret20 = _last(close.pct_change(20))
		ret60 = _last(close.pct_change(60))
		high = _last(close.rolling(252, min_periods=1).max())
		price = _last(close)
		if all(np.isfinite(value) for value in (ret20, ret60, high, price, spy20, spy60)) and high > 0:
			rel20[symbol] = ret20 - spy20
			rel60[symbol] = ret60 - spy60
			closeness[symbol] = price / high
	rank20 = _rank_centered(rel20)
	rank60 = _rank_centered(rel60)
	rank_high = _rank_centered(closeness)
	result = _zero_scores(requested)
	for symbol in requested:
		if symbol in rank20 and symbol in rank60 and symbol in rank_high:
			result[symbol] = _score(0.4 * rank20[symbol] + 0.4 * rank60[symbol] + 0.2 * rank_high[symbol])
	return result


def _rotation_week(as_of: date | datetime | pd.Timestamp | None = None) -> str:
	current = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="America/New_York")
	return current.strftime("%G-W%V")


def signal_rotation(
	symbols: Sequence[str],
	*,
	as_of: date | datetime | pd.Timestamp | None = None,
	history: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
	"""Rank 20/60-day relative returns and 52-week-high proximity; persist weekly."""
	requested = _symbols(symbols)
	week = _rotation_week(as_of)
	cache_path = CACHE_DIR / "rotation_weekly.json"
	if as_of is None and history is None and cache_path.exists():
		try:
			cached = json.loads(cache_path.read_text(encoding="utf-8"))
			if cached.get("week") == week and cached.get("symbols") == requested:
				return {symbol: _score(cached["scores"].get(symbol, 0.0)) for symbol in requested}
		except (OSError, ValueError, TypeError):
			pass
	frames = _load_history([*requested, "SPY"], as_of, history)
	scores = _rotation_scores(requested, frames)
	if as_of is None and history is None:
		cache_path.parent.mkdir(parents=True, exist_ok=True)
		temporary_path = cache_path.with_suffix(".tmp")
		temporary_path.write_text(
			json.dumps({"week": week, "symbols": requested, "scores": scores}), encoding="utf-8"
		)
		temporary_path.replace(cache_path)
	return scores


def signal_breakout(
	symbols: Sequence[str],
	*,
	as_of: date | datetime | pd.Timestamp | None = None,
	history: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
	"""Reward a 20-day closing high with above-average volume and ATR expansion."""
	requested = _symbols(symbols)
	frames = _load_history(requested, as_of, history)
	scores = _zero_scores(requested)
	for symbol in requested:
		frame = frames.get(symbol)
		if frame is None or len(frame) < 45:
			continue
		close = frame["close"].astype(float)
		volume = frame["volume"].astype(float)
		atr = ta.atr(frame["high"], frame["low"], close, length=14)
		if atr is None:
			continue
		prior_high = _last(close.shift(1).rolling(20, min_periods=20).max())
		average_volume = _last(volume.shift(1).rolling(20, min_periods=20).mean())
		atr_value = _last(atr)
		atr_baseline = _last(atr.shift(1).rolling(20, min_periods=20).mean())
		price = _last(close)
		current_volume = _last(volume)
		if not all(np.isfinite(value) for value in (prior_high, average_volume, atr_value, atr_baseline, price, current_volume)):
			continue
		if price > prior_high and current_volume > average_volume and atr_value > atr_baseline:
			volume_expansion = min(current_volume / max(average_volume, 1.0) - 1.0, 1.0)
			atr_expansion = min(atr_value / max(atr_baseline, 1e-9) - 1.0, 1.0)
			scores[symbol] = _score(0.8 + 0.1 * volume_expansion + 0.1 * atr_expansion)
	return scores


def signal_dip(
	symbols: Sequence[str],
	*,
	as_of: date | datetime | pd.Timestamp | None = None,
	history: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
	"""Score active two-day, 2-ATR pullbacks above EMA50 until a defined exit."""
	requested = _symbols(symbols)
	frames = _load_history(requested, as_of, history)
	scores = _zero_scores(requested)
	for symbol in requested:
		frame = frames.get(symbol)
		if frame is None or len(frame) < 60:
			continue
		close = frame["close"].astype(float)
		high = frame["high"].astype(float)
		atr = ta.atr(frame["high"], frame["low"], close, length=14)
		rsi2 = ta.rsi(close, length=2)
		ema50 = ta.ema(close, length=50)
		if atr is None or rsi2 is None or ema50 is None:
			continue
		active_entry: int | None = None
		active_strength = 0.0
		start = max(2, len(frame) - 8)
		for index in range(start, len(frame)):
			if active_entry is not None and index > active_entry:
				bounce = close.iloc[index] > high.iloc[index - 1]
				rsi_recovery = rsi2.iloc[index] >= 50.0
				time_stop = index - active_entry >= 5
				if bounce or rsi_recovery or time_stop:
					active_entry = None
					active_strength = 0.0
			if index < len(frame) - 1 and active_entry is not None:
				continue
			atr_value = float(atr.iloc[index])
			if (
				close.iloc[index] < close.iloc[index - 1]
				and close.iloc[index - 1] < close.iloc[index - 2]
				and close.iloc[index] > ema50.iloc[index]
				and np.isfinite(atr_value)
				and atr_value > 0
			):
				drop = float(close.iloc[index - 2] - close.iloc[index])
				if drop >= 2.0 * atr_value:
					active_entry = index
					active_strength = float(np.clip(0.8 + 0.1 * (drop / (2.0 * atr_value) - 1.0), 0.8, 1.0))
		if active_entry is not None:
			scores[symbol] = _score(active_strength)
	return scores


def _ml_features(frame: pd.DataFrame) -> pd.DataFrame:
	close = frame["close"].astype(float)
	high = frame["high"].astype(float)
	low = frame["low"].astype(float)
	volume = frame["volume"].astype(float)
	macd = ta.macd(close)
	adx = ta.adx(high, low, close, length=14)
	bands = ta.bbands(close, length=20)
	obv = ta.obv(close, volume)
	if any(value is None for value in (macd, adx, bands, obv)):
		return pd.DataFrame(index=frame.index, columns=FEATURE_COLUMNS, dtype=float)
	macd_hist = next(column for column in macd if str(column).startswith("MACDh_"))
	adx_col = next(column for column in adx if str(column).startswith("ADX_"))
	bb_width_col = next(column for column in bands if str(column).startswith("BBB_"))
	bbp_col = next(column for column in bands if str(column).startswith("BBP_"))
	roc5 = ta.roc(close, length=5)
	roc20 = ta.roc(close, length=20)
	rsi14 = ta.rsi(close, length=14)
	ema20 = ta.ema(close, length=20)
	ema50 = ta.ema(close, length=50)
	natr = ta.natr(high, low, close, length=14)
	cmf = ta.cmf(high, low, close, volume, length=20)
	mfi = ta.mfi(high, low, close, volume, length=14)
	returns = close.pct_change()
	rolling_mean = returns.rolling(20, min_periods=20).mean()
	rolling_std = returns.rolling(20, min_periods=20).std()
	features = pd.DataFrame(
		{
			"roc5": roc5,
			"roc20": roc20,
			"rsi14": rsi14,
			"macd_histogram": macd[macd_hist],
			"adx": adx[adx_col],
			"distance_ema20": close / ema20 - 1.0,
			"distance_ema50": close / ema50 - 1.0,
			"natr": natr,
			"bb_width": bands[bb_width_col],
			"bbp": bands[bbp_col],
			"cmf": cmf,
			"mfi": mfi,
			"obv_slope": obv.diff(10) / volume.rolling(10, min_periods=10).sum(),
			"return_zscore20": (returns - rolling_mean) / rolling_std.replace(0.0, np.nan),
		},
		index=frame.index,
	)
	return features.replace([np.inf, -np.inf], np.nan).loc[:, FEATURE_COLUMNS]


def _walk_forward_folds(
	panel: pd.DataFrame,
) -> tuple[list[dict[str, float | int | None]], int]:
	from lightgbm import LGBMRegressor

	dates = sorted(pd.Timestamp(value) for value in panel["session"].unique())
	metrics: list[dict[str, float | int | None]] = []
	if len(dates) < 100:
		return metrics, 1
	test_start = max(80, int(len(dates) * 0.55))
	chunks = [chunk for chunk in np.array_split(np.array(dates[test_start:], dtype=object), 3) if len(chunk)]
	fold_ics: list[float] = []
	for fold_number, chunk in enumerate(chunks, start=1):
		fold_start = pd.Timestamp(chunk[0])
		start_position = dates.index(fold_start)
		purge_position = max(0, start_position - GAP)
		train_cutoff = dates[purge_position]
		train = panel.loc[panel["session"] < train_cutoff]
		valid = panel.loc[panel["session"].isin(chunk)]
		if len(train) < 60 or valid.empty:
			continue
		model = LGBMRegressor(
			objective="regression",
			n_estimators=60,
			learning_rate=0.04,
			num_leaves=7,
			max_depth=3,
			min_child_samples=10,
			reg_lambda=1.0,
			random_state=RANDOM_SEED,
			n_jobs=1,
			verbosity=-1,
		)
		model.fit(train.loc[:, FEATURE_COLUMNS], train["target"])
		predictions = model.predict(valid.loc[:, FEATURE_COLUMNS])
		actual = valid["target"].to_numpy(dtype=float)
		ic = float(pd.Series(predictions).corr(pd.Series(actual), method="spearman"))
		hit_rate = float(np.mean(np.sign(predictions) == np.sign(actual)))
		metrics.append({"fold": fold_number, "ic": ic if np.isfinite(ic) else None, "hit_rate": hit_rate, "train_rows": int(len(train)), "test_rows": int(len(valid))})
		if np.isfinite(ic):
			fold_ics.append(ic)
	return metrics, (1 if not fold_ics or float(np.mean(fold_ics)) >= 0 else -1)


def _record_ml_run(
	metrics: list[dict[str, float | int | None]], sign: int, model_path: Path
) -> None:
	LOG_DIR.mkdir(parents=True, exist_ok=True)
	log_path = LOG_DIR / "ml_walk_forward.jsonl"
	with log_path.open("a", encoding="utf-8") as log_file:
		for metric in metrics:
			log_file.write(json.dumps({**metric, "signal_sign": sign, "model": model_path.name}) + "\n")
		log_file.write(json.dumps({"event": "signal_sign_selected", "signal_sign": sign, "model": model_path.name}) + "\n")


def signal_ml(
	symbols: Sequence[str],
	*,
	as_of: date | datetime | pd.Timestamp | None = None,
	history: Mapping[str, pd.DataFrame] | None = None,
	persist: bool = True,
	walk_forward: bool = True,
) -> dict[str, float]:
	"""Train a seeded LightGBM walk-forward model on 14 causal features and predict."""
	from lightgbm import LGBMRegressor
	import joblib

	requested = _symbols(symbols)
	frames = _load_history(requested, as_of, history)
	blocks: list[pd.DataFrame] = []
	current_features: dict[str, pd.Series] = {}
	validation_scale_values: list[float] = []
	for symbol in requested:
		frame = frames.get(symbol)
		if frame is None or len(frame) < 80:
			continue
		features = _ml_features(frame)
		target = frame["close"].astype(float).shift(-GAP) / frame["close"].astype(float) - 1.0
		block = features.copy()
		block["target"] = target
		block["symbol"] = symbol
		block["session"] = block.index
		blocks.append(block.replace([np.inf, -np.inf], np.nan))
		current_row = features.dropna(subset=list(FEATURE_COLUMNS)).tail(1)
		if not current_row.empty:
			current_features[symbol] = current_row.iloc[-1]
	if not blocks or not current_features:
		return _zero_scores(requested)
	panel = pd.concat(blocks, ignore_index=True)
	panel = panel.dropna(subset=[*FEATURE_COLUMNS, "target"])
	if panel.empty:
		return _zero_scores(requested)
	metrics, sign = _walk_forward_folds(panel) if walk_forward else ([], 1)
	for metric in metrics:
		if metric["ic"] is not None:
			validation_scale_values.append(float(metric["ic"]))
	train = panel.loc[:, [*FEATURE_COLUMNS, "target"]]
	if len(train) < 60:
		return _zero_scores(requested)
	model = LGBMRegressor(
		objective="regression",
		n_estimators=60,
		learning_rate=0.04,
		num_leaves=7,
		max_depth=3,
		min_child_samples=10,
		reg_lambda=1.0,
		random_state=RANDOM_SEED,
		n_jobs=1,
		verbosity=-1,
	)
	model.fit(train.loc[:, FEATURE_COLUMNS], train["target"])
	if persist:
		MODEL_DIR.mkdir(parents=True, exist_ok=True)
		version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
		model_path = MODEL_DIR / f"lightgbm_signal_{version}.joblib"
		joblib.dump({"model": model, "features": list(FEATURE_COLUMNS), "signal_sign": sign}, model_path)
		_record_ml_run(metrics, sign, model_path)
	all_targets = panel["target"].abs().to_numpy(dtype=float)
	scale = float(np.nanmedian(all_targets)) if len(all_targets) else 0.02
	if not np.isfinite(scale) or scale < 1e-4:
		scale = 0.02
	scores = _zero_scores(requested)
	for symbol, features in current_features.items():
		prediction = float(model.predict(features.loc[list(FEATURE_COLUMNS)].to_frame().T)[0])
		scores[symbol] = _score(math.tanh(sign * prediction / scale))
	return scores


def _fetch_news_headlines(symbols: Sequence[str], student: str) -> dict[str, list[str]]:
	finnhub_key = os.environ.get("FINNHUB_API_KEY")
	alpaca_key = os.environ.get(f"ALPACA_KEY_{student}")
	alpaca_secret = os.environ.get(f"ALPACA_SECRET_{student}")
	if not finnhub_key:
		raise ValueError("FINNHUB_API_KEY is required for news signals")
	if not alpaca_key or not alpaca_secret:
		raise ValueError(f"ALPACA_KEY_{student} and ALPACA_SECRET_{student} are required for Alpaca news")

	from alpaca.data.historical.news import NewsClient
	from alpaca.data.requests import NewsRequest

	start = datetime.now(timezone.utc) - timedelta(days=3)
	news_client = NewsClient(alpaca_key, alpaca_secret)
	response = news_client.get_news(
		NewsRequest(symbols=",".join(symbols), start=start, sort="desc", limit=100, exclude_contentless=True)
	)
	alpaca_items = _alpaca_news_items(response)
	combined = {symbol: [] for symbol in symbols}
	for item in alpaca_items or []:
		headline = str(item.get("headline", "") if isinstance(item, Mapping) else getattr(item, "headline", "") or "").strip()
		item_symbols = item.get("symbols", []) if isinstance(item, Mapping) else getattr(item, "symbols", [])
		for symbol in item_symbols or []:
			if symbol in combined and headline:
				combined[symbol].append(headline)

	date_from = start.date().isoformat()
	date_to = datetime.now(timezone.utc).date().isoformat()
	def fetch_finnhub(symbol: str) -> tuple[str, list[str], bool]:
		result = requests.get(
			f"{FINNHUB_URL}/company-news",
			headers={"X-Finnhub-Token": finnhub_key},
			params={"symbol": symbol, "from": date_from, "to": date_to},
		)
		try:
			_check_finnhub_response(result)
		except FinnhubAuthenticationError:
			return symbol, [], True
		headlines = [
			str(item.get("headline", "")).strip()
			for item in result.json()
			if str(item.get("headline", "")).strip()
		]
		return symbol, headlines, False

	with ThreadPoolExecutor(max_workers=FINNHUB_NEWS_WORKERS) as executor:
		for symbol, headlines, auth_rejected in executor.map(fetch_finnhub, symbols):
			if auth_rejected:
				LOGGER.warning(
					"Finnhub rejected FINNHUB_API_KEY; continuing with Alpaca headlines only"
				)
				break
			combined[symbol].extend(headlines)
	return {symbol: list(dict.fromkeys(headlines))[:20] for symbol, headlines in combined.items()}


def _alpaca_news_items(response: Any) -> list[Any]:
	if isinstance(response, dict):
		data = response.get("data", response)
	else:
		data = getattr(response, "data", None)
	if isinstance(data, dict):
		return list(data.get("news", []))
	return list(getattr(response, "news", []) or [])


def _normalize_groq_assessment(parsed: Any) -> dict[str, Any]:
	if not isinstance(parsed, dict):
		parsed = {}
	try:
		sentiment = float(parsed.get("sentiment", 0.0))
		confidence = float(parsed.get("confidence", 0.0))
	except (TypeError, ValueError):
		return _neutral_groq_assessment()
	if not np.isfinite(sentiment) or not np.isfinite(confidence):
		return _neutral_groq_assessment()
	sentiment = float(np.clip(sentiment, -1.0, 1.0))
	confidence = float(np.clip(confidence, 0.0, 1.0))
	flags = parsed.get("event_flags", [])
	if not isinstance(flags, list):
		flags = []
	return {
		"sentiment": sentiment,
		"confidence": confidence,
		"event_flags": [str(flag).strip().lower().replace(" ", "_") for flag in flags],
		"veto": bool(parsed.get("veto", False)),
	}


def _neutral_groq_assessment() -> dict[str, Any]:
	return {"sentiment": 0.0, "confidence": 0.0, "event_flags": [], "veto": False}


def _neutral_groq_assessments(
	headlines_by_symbol: Mapping[str, Sequence[str]], reason: str
) -> dict[str, dict[str, Any]]:
	LOGGER.warning("Groq %s; using neutral news assessments for this run", reason)
	return {symbol: _neutral_groq_assessment() for symbol in headlines_by_symbol}


def _ask_groq_batch(
	headlines_by_symbol: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
	api_key = os.environ.get("GROQ_API_KEY")
	if not api_key:
		raise ValueError("GROQ_API_KEY is required for news signals")
	model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
	symbols = list(headlines_by_symbol)
	assessment_schema = {
		"type": "object",
		"properties": {
			"sentiment": {"type": "number"},
			"event_flags": {"type": "array", "items": {"type": "string"}},
			"confidence": {"type": "number"},
			"veto": {"type": "boolean"},
		},
		"required": ["sentiment", "event_flags", "confidence", "veto"],
		"additionalProperties": False,
	}
	response_schema = {
		"type": "object",
		"properties": {
			"assessments": {
				"type": "object",
				"properties": {symbol: assessment_schema for symbol in symbols},
				"required": symbols,
				"additionalProperties": False,
			}
		},
		"required": ["assessments"],
		"additionalProperties": False,
	}
	prompt = {
		"stocks": {
			symbol: list(headlines)[:1]
			for symbol, headlines in headlines_by_symbol.items()
		},
		"instructions": (
			"For each stock, assess only its supplied headlines for near-term impact. Return one "
			"JSON object with an 'assessments' object keyed by the exact stock symbol. Each value "
			"must contain sentiment (number -1 to 1), event_flags (array of short snake_case "
			"strings), confidence (number 0 to 1), and veto (boolean). Use neutral sentiment and "
			"zero confidence if evidence is insufficient. Set veto true only for strongly negative "
			"events such as fraud, bankruptcy, default, delisting, or a criminal investigation."
		),
	}
	try:
		result = requests.post(
			GROQ_URL,
			headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
			json={
				"model": model,
				"temperature": 0,
				"max_completion_tokens": 4096,
				"response_format": {
					"type": "json_schema",
					"json_schema": {
						"name": "stock_news_assessments",
						"strict": True,
						"schema": response_schema,
					},
				},
				"messages": [
					{"role": "system", "content": "Return only valid JSON matching the requested fields."},
					{"role": "user", "content": json.dumps(prompt)},
				],
			},
		)
	except requests.exceptions.RequestException as error:
		return _neutral_groq_assessments(
			headlines_by_symbol, f"request failed ({type(error).__name__})"
		)
	if result.status_code == 404:
		return _neutral_groq_assessments(
			headlines_by_symbol,
			f"model '{model}' was not found; check GROQ_MODEL",
		)
	if not 200 <= result.status_code < 300:
		reasons = {
			400: "rejected the request (HTTP 400)",
			401: "rejected GROQ_API_KEY (HTTP 401)",
			403: "denied this request (HTTP 403)",
			429: "rate limit reached (HTTP 429)",
		}
		reason = reasons.get(result.status_code, f"request failed (HTTP {result.status_code})")
		return _neutral_groq_assessments(headlines_by_symbol, reason)
	try:
		result.raise_for_status()
		content = result.json()["choices"][0]["message"]["content"]
		parsed = json.loads(content)
	except (KeyError, IndexError, TypeError, ValueError):
		return _neutral_groq_assessments(headlines_by_symbol, "returned an invalid response")
	assessments = parsed.get("assessments", {}) if isinstance(parsed, dict) else {}
	if not isinstance(assessments, dict):
		return _neutral_groq_assessments(headlines_by_symbol, "returned an invalid assessment object")
	return {
		symbol: _normalize_groq_assessment(assessments.get(symbol, {}))
		for symbol in headlines_by_symbol
	}


def _ask_groq(symbol: str, headlines: Sequence[str]) -> dict[str, Any]:
	"""Assess one symbol through the shared batch-capable Groq API."""
	return _ask_groq_batch({symbol: headlines})[symbol]


def signal_news_tilt(
	symbols: Sequence[str],
	*,
	student: str = "A",
	headlines_by_symbol: Mapping[str, Sequence[str]] | None = None,
) -> SignalScores:
	"""Return a capped headline tilt; strongly negative events are marked in ``.vetoes``."""
	requested = _symbols(symbols)
	if headlines_by_symbol is None:
		headlines_by_symbol = _fetch_news_headlines(requested, student)
	scores: dict[str, float] = {}
	vetoes: set[str] = set()
	assessments = _ask_groq_batch(headlines_by_symbol)
	for symbol in requested:
		assessment = assessments.get(symbol, _neutral_groq_assessment())
		sentiment = float(np.clip(assessment["sentiment"], -1.0, 1.0))
		confidence = float(np.clip(assessment["confidence"], 0.0, 1.0))
		flags = set(assessment.get("event_flags", []))
		strong_negative = (
			bool(assessment.get("veto"))
			or sentiment <= -0.75 and confidence >= 0.7
			or bool(flags & STRONG_NEGATIVE_EVENTS) and confidence >= 0.6
		)
		if strong_negative:
			vetoes.add(symbol)
		scores[symbol] = _score(float(np.clip(sentiment * confidence * 0.2, -0.2, 0.2)))
	return SignalScores(scores, vetoes=vetoes)


def earnings_guard(symbols: Sequence[str], days: int) -> list[str]:
	"""Return requested symbols with Finnhub earnings dates in the next ``days`` days."""
	requested = _symbols(symbols)
	if isinstance(days, bool) or not isinstance(days, int) or days < 0:
		raise ValueError("days must be a non-negative integer")
	token = os.environ.get("FINNHUB_API_KEY")
	if not token:
		raise ValueError("FINNHUB_API_KEY is required for the earnings calendar")
	today = date.today()
	result = requests.get(
		f"{FINNHUB_URL}/calendar/earnings",
		headers={"X-Finnhub-Token": token},
		params={
			"from": today.isoformat(),
			"to": (today + timedelta(days=days)).isoformat(),
		},
	)
	_check_finnhub_response(result)
	calendar = result.json().get("earningsCalendar", [])
	upcoming = {str(item.get("symbol", "")).upper() for item in calendar}
	return [symbol for symbol in requested if symbol in upcoming]


def regime_state(
	*,
	as_of: date | datetime | pd.Timestamp | None = None,
	history: Mapping[str, pd.DataFrame] | None = None,
) -> str:
	"""Classify QQQ as risk_on, risk_off, or neutral from moving averages and volatility."""
	frames = _load_history(["QQQ"], as_of, history)
	frame = frames.get("QQQ")
	if frame is None or len(frame) < 20:
		return "neutral"
	close = frame["close"].astype(float)
	returns = close.pct_change().dropna()
	volatility = float(returns.tail(20).std() * math.sqrt(252)) if len(returns) >= 20 else 0.0
	if volatility >= 0.40:
		return "risk_off"
	if len(close) < 50:
		return "neutral"
	price = _last(close)
	ma50 = _last(close.rolling(50).mean())
	if not all(np.isfinite(value) for value in (price, ma50)):
		return "neutral"
	if price < ma50:
		return "risk_off"
	ma200 = _last(close.rolling(200).mean())
	if np.isfinite(ma200) and price > ma50 and price > ma200:
		return "risk_on"
	return "neutral"