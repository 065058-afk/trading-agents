"""FastMCP stdio server for trading data, signals, portfolio, and paper orders."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from mcp_server import data as market_data
from mcp_server import signals as signal_engine

load_dotenv(ROOT / ".env")
CONFIG_DIR = ROOT / "config"
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
STUDENT_CHOICES = {"A", "B", "C"}
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]*$")

mcp = FastMCP("trading-agents", instructions="Paper-only trading tools. Set MCP_DRY_RUN=1 to disable all external calls.")


def _dry_run(dry_run: bool = False) -> bool:
	return dry_run or os.environ.get("MCP_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
	result = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
	if not result or any(not _SYMBOL_RE.fullmatch(symbol) for symbol in result):
		raise ValueError("symbols must contain at least one valid ticker")
	return result


def _config(student: str) -> dict[str, Any]:
	student = student.strip().upper()
	if student not in STUDENT_CHOICES:
		raise ValueError("student must be A, B, or C")
	override_json = os.environ.get("MCP_STUDENT_CONFIG_JSON")
	if override_json:
		try:
			overrides = json.loads(override_json)
		except json.JSONDecodeError as error:
			raise ValueError("Invalid in-memory student config override") from error
		result = overrides.get(student) if isinstance(overrides, Mapping) else None
		if isinstance(result, dict):
			return result
	path = CONFIG_DIR / f"student_{student}.yaml"
	with path.open(encoding="utf-8") as config_file:
		result = yaml.safe_load(config_file)
	if not isinstance(result, dict):
		raise ValueError(f"Invalid student configuration: {path.name}")
	return result


def _account_client(student: str):
	student = student.strip().upper()
	if student not in STUDENT_CHOICES:
		raise ValueError("student must be A, B, or C")
	key = os.environ.get(f"ALPACA_KEY_{student}")
	secret = os.environ.get(f"ALPACA_SECRET_{student}")
	if not key or not secret:
		raise ValueError(f"Missing ALPACA_KEY_{student} or ALPACA_SECRET_{student} environment variable")
	from alpaca.trading.client import TradingClient

	return TradingClient(key, secret, paper=True)


def _sample_history(symbols: Sequence[str], count: int = 330) -> dict[str, pd.DataFrame]:
	requested = list(dict.fromkeys([*symbols, "SPY", "QQQ"]))
	end = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
	dates = pd.bdate_range(end=end, periods=count)
	frames: dict[str, pd.DataFrame] = {}
	for symbol in requested:
		seed = sum((index + 1) * ord(char) for index, char in enumerate(symbol))
		rng = np.random.default_rng(seed)
		returns = rng.normal(0.00025, 0.012, count)
		close = 100.0 * np.exp(np.cumsum(returns))
		frames[symbol] = pd.DataFrame(
			{
				"open": close * (1.0 + rng.normal(0.0, 0.001, count)),
				"high": close * 1.012,
				"low": close * 0.988,
				"close": close,
				"volume": rng.integers(100_000, 500_000, count),
			},
			index=dates,
		)
	return frames


def _history(symbols: Sequence[str], *, dry_run: bool = False) -> dict[str, pd.DataFrame]:
	requested = list(dict.fromkeys([*symbols, "SPY", "QQQ"]))
	if _dry_run(dry_run):
		return _sample_history(requested)
	return market_data.history_cache(requested)


def _json_number(value: Any) -> float | None:
	try:
		number = float(value)
	except (TypeError, ValueError):
		return None
	return number if math.isfinite(number) else None


def _field(value: Any, name: str, default: Any = None) -> Any:
	if isinstance(value, Mapping):
		return value.get(name, default)
	return getattr(value, name, default)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
	result = frame.copy()
	if isinstance(result.index, pd.MultiIndex):
		result = result.reset_index()
	elif "timestamp" not in result.columns:
		result = result.reset_index().rename(columns={result.index.name or "index": "timestamp"})
	result.columns = [str(column).strip().lower() for column in result.columns]
	if "timestamp" in result.columns:
		result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce").map(_timestamp_text)
	rows = result.replace({np.nan: None}).to_dict(orient="records")
	return [{str(key): value for key, value in row.items()} for row in rows]


def _timestamp_text(value: Any) -> str | None:
	if pd.isna(value):
		return None
	return pd.Timestamp(value).isoformat()


def _market_frames(symbols: Sequence[str], *, dry_run: bool = False) -> dict[str, pd.DataFrame]:
	requested = _normalize_symbols(symbols)
	frames = _history(requested, dry_run=dry_run)
	return {symbol: frames[symbol] for symbol in requested if symbol in frames}


def _signal_call(name: str, symbols: Sequence[str], as_of: str | None = None) -> dict[str, float]:
	requested = _normalize_symbols(symbols)
	is_dry = _dry_run()
	if name == "signal_ml" and is_dry:
		return {symbol: 0.0 for symbol in requested}
	frames = _market_frames(requested, dry_run=True) if is_dry else None
	function = getattr(signal_engine, name)
	kwargs: dict[str, Any] = {"as_of": as_of}
	if frames is not None:
		kwargs["history"] = frames
	if name == "signal_ml":
		kwargs["persist"] = not is_dry
	return {symbol: float(score) for symbol, score in function(requested, **kwargs).items()}


def _capped_group_allocation(
	scores: Mapping[str, float], symbols: Sequence[str], budget: float, cap: float
) -> dict[str, float]:
	remaining = float(budget)
	active = {
		symbol: max(0.01, 1.0 + float(scores.get(symbol, 0.0)))
		for symbol in symbols
	}
	allocation: dict[str, float] = {}
	while active and remaining > 1e-10:
		total = sum(active.values())
		capped = [symbol for symbol, score in active.items() if remaining * score / total >= cap]
		if not capped:
			for symbol, score in active.items():
				allocation[symbol] = allocation.get(symbol, 0.0) + remaining * score / total
			break
		for symbol in capped:
			allocation[symbol] = allocation.get(symbol, 0.0) + cap
			remaining -= cap
			del active[symbol]
	return allocation


def _blend_from_scores(
	student: str,
	symbols: Sequence[str],
	score_map: Mapping[str, Mapping[str, float]],
	regime: str,
	earnings: Sequence[str],
	config_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	config = dict(config_override) if config_override is not None else _config(student)
	high_beta = [str(symbol).upper() for symbol in config["high_beta"]]
	diversified = [str(symbol).upper() for symbol in config["diversified"]]
	requested = list(dict.fromkeys([*high_beta, *diversified])) if not symbols else _normalize_symbols(symbols)
	configured = config.get("signal_weights", {})
	weights = {name: float(configured.get(name, 0.0)) for name in ("trend", "rotation", "breakout", "dip", "ml", "news")}
	if regime == "risk_on":
		for name in ("trend", "breakout", "rotation"):
			weights[name] *= 4.0 / 3.0
		weights["dip"] *= 2.0 / 3.0
	elif regime == "risk_off":
		weights["dip"] *= 4.0 / 3.0
		weights["breakout"] *= 2.0 / 3.0
		weights["trend"] *= 2.0 / 3.0
	total_weight = sum(weights.values())
	if total_weight <= 0:
		raise ValueError("configured signal weights must sum to a positive value")
	weights = {name: value / total_weight for name, value in weights.items()}
	guarded = {str(symbol).upper() for symbol in earnings}
	combined: dict[str, float] = {}
	for symbol in requested:
		combined[symbol] = sum(
			weights[name] * float(score_map.get(name, {}).get(symbol, 0.0))
			for name in weights
		)
	low, high = map(float, config.get("high_beta_share_range", [0.35, 0.55]))
	base_share = float(config.get("high_beta_share_base", 0.45))
	high_beta_share = high if regime == "risk_on" else low if regime == "risk_off" else float(np.clip(base_share, low, high))
	eligible = {symbol: score for symbol, score in combined.items() if symbol not in guarded}
	high_names = [symbol for symbol in high_beta if symbol in eligible]
	div_names = [symbol for symbol in diversified if symbol in eligible]
	high_capacity = len(high_names) * float(config["max_position_pct_high_beta"])
	diversified_capacity = len(div_names) * float(config["max_position_pct_diversified"])
	feasible_low = max(low, 1.0 - diversified_capacity)
	feasible_high = min(high, high_capacity)
	if feasible_low > feasible_high + 1e-9:
		raise ValueError(
			"Cannot reach 90-100% gross exposure within the high-beta share range "
			"after earnings exclusions and position caps"
		)
	high_beta_share = float(np.clip(high_beta_share, feasible_low, feasible_high))
	positions = _capped_group_allocation(
		eligible, high_names, high_beta_share, float(config["max_position_pct_high_beta"])
	)
	positions.update(
		_capped_group_allocation(
			eligible, div_names, 1.0 - high_beta_share, float(config["max_position_pct_diversified"])
		)
	)
	targets = {symbol: round(positions.get(symbol, 0.0), 8) for symbol in requested}
	gross = sum(targets.values())
	if not 0.90 <= gross <= 1.000001:
		raise ValueError(
			f"Cannot reach 90-100% gross exposure under score, earnings, and position-cap constraints (got {gross:.1%})"
		)
	return {
		"student": student,
		"regime": regime,
		"adjusted_signal_weights": weights,
		"high_beta_share": high_beta_share,
		"earnings_guard": sorted(guarded),
		"signal_scores": {name: dict(values) for name, values in score_map.items()},
		"target_weights": targets,
		"gross_exposure": round(gross, 8),
	}


def _account_snapshot(student: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	client = _account_client(student)
	account: Any = client.get_account()
	positions: Any = client.get_all_positions()
	if isinstance(account, Mapping):
		account = account.get("account", account)
	if isinstance(positions, Mapping):
		positions = positions.get("positions", [])
	account_dict = {
		"status": str(_field(account, "status", "unknown")),
		"currency": str(_field(account, "currency", "USD")),
		"equity": _json_number(_field(account, "equity")) or 0.0,
		"portfolio_value": _json_number(_field(account, "portfolio_value")) or 0.0,
		"buying_power": _json_number(_field(account, "buying_power")) or 0.0,
		"cash": _json_number(_field(account, "cash")) or 0.0,
		"trading_blocked": bool(_field(account, "trading_blocked", False)),
	}
	position_list = [
		{
			"symbol": str(_field(position, "symbol", "")).upper(),
			"qty": _json_number(_field(position, "qty")) or 0.0,
			"side": str(_field(position, "side", "long")),
			"avg_entry_price": _json_number(_field(position, "avg_entry_price")) or 0.0,
			"market_value": _json_number(_field(position, "market_value")) or 0.0,
			"current_price": _json_number(_field(position, "current_price")) or 0.0,
			"qty_available": (
				_json_number(_field(position, "qty_available"))
				if _field(position, "qty_available") is not None
				else _json_number(_field(position, "qty")) or 0.0
			),
		}
		for position in positions
	]
	return account_dict, position_list


def _demo_account(student: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	capital = float(_config(student).get("capital", 100_000))
	return (
		{
			"status": "dry_run",
			"currency": "USD",
			"equity": capital,
			"portfolio_value": capital,
			"buying_power": capital,
			"cash": capital,
			"trading_blocked": False,
		},
		[],
	)


def _positions_for(student: str, dry_run: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	if _dry_run(dry_run):
		if os.environ.get(f"ALPACA_KEY_{student}") and os.environ.get(f"ALPACA_SECRET_{student}"):
			return _account_snapshot(student)
		return _demo_account(student)
	return _account_snapshot(student)


def _coerce_weights(value: Mapping[str, Any]) -> dict[str, float]:
	result = {str(symbol).upper(): float(weight) for symbol, weight in value.items()}
	if not result:
		raise ValueError("target_weights cannot be empty")
	if any(not math.isfinite(weight) for weight in result.values()):
		raise ValueError("target weights must be finite numbers")
	return result


def _serialize_order(order: Any) -> dict[str, Any]:
	return {
		"id": str(getattr(order, "id", "")),
		"symbol": str(getattr(order, "symbol", "")),
		"side": str(getattr(order, "side", "")),
		"qty": _json_number(getattr(order, "qty", None)),
		"client_order_id": str(getattr(order, "client_order_id", "")),
		"status": str(getattr(order, "status", "")),
	}


@mcp.tool(description="Fetch one batched daily Alpaca IEX OHLCV table for all requested symbols.")
def get_ohlcv_batch(symbols: list[str], days: int, student: str) -> dict[str, Any]:
	requested = _normalize_symbols(symbols)
	if _dry_run():
		frames = _market_frames(requested, dry_run=True)
		combined = pd.concat(
			{symbol: frame.tail(max(1, min(days, 20))) for symbol, frame in frames.items()},
			names=["symbol", "timestamp"],
		)
		return {"rows": _records(combined), "count": len(combined), "feed": "DRY_RUN_IEX"}
	table = market_data.get_ohlcv_batch(requested, days, student=student)
	return {"rows": _records(table), "count": len(table), "feed": "IEX"}


@mcp.tool(description="Refresh local Yahoo daily history cache for symbols plus SPY and QQQ.")
def refresh_history(symbols: list[str]) -> dict[str, Any]:
	requested = _normalize_symbols(symbols)
	frames = _history(requested, dry_run=_dry_run())
	return {
		"tickers": {
			symbol: {
				"rows": int(len(frame)),
				"first_date": frame.index.min().date().isoformat() if not frame.empty else None,
				"last_date": frame.index.max().date().isoformat() if not frame.empty else None,
			}
			for symbol, frame in frames.items()
		}
	}


@mcp.tool(
	description="Compute latest pandas_ta features and common trend/volatility indicators.",
	structured_output=False,
)
def compute_indicators(symbols: list[str]) -> dict[str, Any]:
	requested = _normalize_symbols(symbols)
	frames = _market_frames(requested, dry_run=_dry_run())
	output: dict[str, Any] = {}
	for symbol, frame in frames.items():
		close = frame["close"].astype(float)
		high = frame["high"].astype(float)
		low = frame["low"].astype(float)
		volume = frame["volume"].astype(float)
		previous_close = close.shift(1)
		true_range = pd.concat(
			[(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
			axis=1,
		).max(axis=1)
		atr = true_range.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
		delta = close.diff()
		average_gain = delta.clip(lower=0).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
		average_loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
		rsi = 100.0 - 100.0 / (1.0 + average_gain / average_loss.replace(0.0, np.nan))
		output[symbol] = {
			"as_of": frame.index[-1].date().isoformat(),
			"close": _json_number(close.iloc[-1]),
			"ema20": _json_number(close.ewm(span=20, adjust=False).mean().iloc[-1]),
			"ema50": _json_number(close.ewm(span=50, adjust=False).mean().iloc[-1]),
			"atr14": _json_number(atr.iloc[-1]),
			"rsi14": _json_number(rsi.iloc[-1]),
			"roc5": _json_number(close.pct_change(5).iloc[-1]),
			"roc20": _json_number(close.pct_change(20).iloc[-1]),
			"volume_ratio20": _json_number(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]),
		}
	return output


def _register_price_signal(name: str):
	def run(symbols: list[str], as_of: str | None = None) -> dict[str, float]:
		return _signal_call(name, symbols, as_of)

	run.__name__ = name
	run.__doc__ = f"Compute {name.replace('signal_', '')} scores in [-1, 1]."
	mcp.tool(description=run.__doc__)(run)
	return run


signal_trend = _register_price_signal("signal_trend")
signal_rotation = _register_price_signal("signal_rotation")
signal_breakout = _register_price_signal("signal_breakout")
signal_dip = _register_price_signal("signal_dip")
signal_ml = _register_price_signal("signal_ml")


@mcp.tool(description="Fetch Finnhub and Alpaca headlines for symbols; student selects Alpaca credentials.")
def news_fetch(symbols: list[str], student: str) -> dict[str, list[str]]:
	requested = _normalize_symbols(symbols)
	if _dry_run():
		return {symbol: [f"Dry-run sample headline for {symbol}"] for symbol in requested}
	return signal_engine._fetch_news_headlines(requested, student)


@mcp.tool(description="Blend the student's configured signals, regime weights, earnings guard, and position caps.")
def blend_scores(
	student: str,
	symbols: list[str] | None = None,
	dry_run: bool | None = None,
	config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
	config = dict(config_override) if config_override is not None else _config(student)
	requested = list(dict.fromkeys([*config["high_beta"], *config["diversified"]])) if symbols is None else _normalize_symbols(symbols)
	is_dry = _dry_run() if dry_run is None else dry_run
	if is_dry:
		score_map = {
			name: {
				symbol: float(((sum(ord(char) for char in symbol) + signal_index * 13) % 101) / 100.0)
				for symbol in requested
			}
			for signal_index, name in enumerate(("trend", "rotation", "breakout", "dip", "ml", "news"))
		}
		score_map["ml"] = signal_engine.signal_ml(
			requested,
			history=_market_frames(requested, dry_run=True),
			persist=False,
			walk_forward=False,
		)
		regime = "risk_on"
		earnings: list[str] = []
	else:
		score_map = {
			"trend": signal_engine.signal_trend(requested),
			"rotation": signal_engine.signal_rotation(requested),
			"breakout": signal_engine.signal_breakout(requested),
			"dip": signal_engine.signal_dip(requested),
			"ml": signal_engine.signal_ml(requested),
			"news": signal_engine.signal_news_tilt(requested, student=student),
		}
		regime = signal_engine.regime_state()
		earnings = signal_engine.earnings_guard(
			requested, int(config["risk"].get("earnings_guard_days", 2))
		)
	return _blend_from_scores(student, requested, score_map, regime, earnings, config)


@mcp.tool(description="Validate basket size, high-beta count, position caps, long-only gross exposure, and account buying power.")
def validate_portfolio(
	student: str,
	target_weights: dict[str, float],
	dry_run: bool = False,
	required_buying_power: float | None = None,
	allow_reduced_exposure: bool = False,
) -> dict[str, Any]:
	config = _config(student)
	weights = _coerce_weights(target_weights)
	high_beta = {str(symbol).upper() for symbol in config["high_beta"]}
	checks: dict[str, bool] = {}
	count = len(weights)
	checks["stock_count_10_to_15"] = 10 <= count <= 15
	checks["high_beta_at_least_30_percent"] = len(set(weights) & high_beta) / max(count, 1) >= 0.30
	checks["long_only"] = all(0.0 <= weight for weight in weights.values())
	checks["position_caps"] = all(
		weight <= float(config["max_position_pct_high_beta"] if symbol in high_beta else config["max_position_pct_diversified"]) + 1e-9
		for symbol, weight in weights.items()
	)
	gross = sum(weights.values())
	checks["gross_exposure_90_to_100_percent"] = (
		0.0 <= gross <= 1.000001
		if allow_reduced_exposure
		else 0.90 <= gross <= 1.000001
	)
	if not all(checks.values()):
		failed = [name for name, passed in checks.items() if not passed]
		raise ValueError(f"Portfolio validation failed: {', '.join(failed)}")
	account, positions = _positions_for(student, dry_run)
	portfolio_value = float(account["portfolio_value"])
	current_by_symbol = {position["symbol"]: abs(float(position["market_value"])) for position in positions}
	needed = (
		float(required_buying_power)
		if required_buying_power is not None
		else sum(
			max(0.0, weight * portfolio_value - current_by_symbol.get(symbol, 0.0))
			for symbol, weight in weights.items()
		)
	)
	available = float(account["buying_power"])
	checks["sufficient_buying_power"] = needed <= available + max(0.01, abs(available) * 1e-9)
	if not checks["sufficient_buying_power"]:
		raise ValueError(
			f"Portfolio validation failed: insufficient buying power (need ${needed:,.2f}, "
			f"available ${available:,.2f})"
		)
	return {"valid": True, "checks": checks, "gross_exposure": gross, "required_buying_power": needed}


@mcp.tool(description="Diff target weights against current positions and return buy/sell quantities.")
def portfolio_diff(
	student: str,
	target_weights: dict[str, float],
	dry_run: bool = False,
) -> dict[str, Any]:
	weights = _coerce_weights(target_weights)
	account, positions = _positions_for(student, dry_run)
	is_dry = _dry_run(dry_run)
	if is_dry:
		frames = _market_frames(list(weights), dry_run=True)
		prices = {symbol: float(frame["close"].iloc[-1]) for symbol, frame in frames.items()}
	else:
		table = market_data.get_ohlcv_batch(list(weights), 5, student=student)
		prices = {
			symbol: float(rows.sort_values("timestamp").iloc[-1]["close"])
			for symbol, rows in table.groupby("symbol")
		}
	current_positions = {position["symbol"]: position for position in positions}
	portfolio_value = float(account["portfolio_value"])
	diffs = []
	for symbol, weight in weights.items():
		price = prices.get(symbol)
		if not price or price <= 0:
			continue
		target_qty = weight * portfolio_value / price
		position = current_positions.get(symbol, {})
		quantity = float(position.get("qty", 0.0))
		is_short = str(position.get("side", "long")).lower() == "short" or quantity < 0.0
		current_qty = -abs(quantity) if is_short else max(0.0, quantity)
		delta = target_qty - current_qty
		if delta > 1e-6:
			diffs.append({"symbol": symbol, "side": "buy", "qty": round(delta, 6)})
		elif delta < -1e-6 and not is_short:
			held_qty = max(0.0, quantity)
			available_raw = position.get("qty_available")
			available_qty = held_qty if available_raw is None else max(0.0, float(available_raw))
			sell_qty = min(-delta, held_qty, available_qty)
			if sell_qty > 1e-6:
				diffs.append({"symbol": symbol, "side": "sell", "qty": round(sell_qty, 6)})
	return {"orders": diffs, "count": len(diffs), "dry_run": is_dry}


def _filled_order_values(order: Any) -> dict[str, Any]:
	filled_qty = float(order.filled_qty or 0.0)
	filled_at = order.filled_at
	if filled_qty <= 0.0 or filled_at is None:
		raise ValueError("order has no fill")
	return {
		"symbol": str(order.symbol).upper(),
		"side": str(order.side).lower(),
		"qty": filled_qty,
		"filled_at": pd.Timestamp(filled_at),
	}


def _entry_dates_and_partial_sells(orders: Sequence[Any], symbols: Sequence[str]) -> tuple[dict[str, pd.Timestamp], set[str]]:
	ordered: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
	for order in orders:
		try:
			item = _filled_order_values(order)
		except (AttributeError, TypeError, ValueError):
			continue
		if item["symbol"] in ordered and pd.notna(item["filled_at"]):
			ordered[item["symbol"]].append(item)
	entry_dates: dict[str, pd.Timestamp] = {}
	partial_sales: set[str] = set()
	for symbol, items in ordered.items():
		items.sort(key=lambda item: item["filled_at"])
		net_qty = 0.0
		entry: pd.Timestamp | None = None
		for item in items:
			if item["side"] == "buy":
				if net_qty <= 1e-9:
					entry = item["filled_at"]
				net_qty += item["qty"]
			elif item["side"] == "sell":
				net_qty -= item["qty"]
				if entry is not None and net_qty > 1e-9:
					partial_sales.add(symbol)
				if net_qty <= 1e-9:
					entry = None
		if entry is not None:
			entry_dates[symbol] = entry
	return entry_dates, partial_sales


def _mock_positions_for_exits(student: str) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
	frame = _sample_history(["AAPL"])["AAPL"]
	entry_date = frame.index[-15]
	entry_price = float(frame["close"].iloc[-15])
	position = {
		"symbol": "AAPL",
		"qty": 10.0,
		"qty_available": 10.0,
		"avg_entry_price": entry_price,
		"market_value": float(frame["close"].iloc[-1]) * 10.0,
		"current_price": float(frame["close"].iloc[-1]),
		"opened_at": entry_date.isoformat(),
	}
	return [position], {"AAPL": frame}


@mcp.tool(description="Return sell orders triggered by ATR trailing stop, partial profit, or configured time stop.")
def check_exits(
	student: str,
	dry_run: bool = False,
	positions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
	config = _config(student)
	is_dry = _dry_run(dry_run)
	orders_history: Any = []
	if positions is not None:
		current_positions = positions
		symbols = [str(position["symbol"]).upper() for position in current_positions]
		frames = _market_frames(symbols, dry_run=True)
		entry_dates = {
			str(position["symbol"]).upper(): pd.Timestamp(position["opened_at"])
			for position in current_positions
			if position.get("opened_at")
		}
		partial_sales: set[str] = set()
	else:
		client: Any | None = None
		if is_dry:
			_, positions_snapshot = _positions_for(student, dry_run=True)
		else:
			client = _account_client(student)
			raw_positions: Any = client.get_all_positions()
			if isinstance(raw_positions, Mapping):
				raw_positions = raw_positions.get("positions", [])
			positions_snapshot = [
				{
					"symbol": str(_field(position, "symbol", "")).upper(),
					"qty": float(_field(position, "qty", 0.0)),
					"qty_available": float(_field(position, "qty_available", None)) if _field(position, "qty_available", None) is not None else None,
					"avg_entry_price": float(_field(position, "avg_entry_price", 0.0)),
					"market_value": float(_field(position, "market_value", 0.0) or 0.0),
					"current_price": float(_field(position, "current_price", 0.0) or 0.0),
					"side": str(_field(position, "side", "long")),
				}
				for position in raw_positions
			]
		current_positions = [
			{**position, "opened_at": None}
			for position in positions_snapshot
		]
		if not current_positions:
			return {"orders": [], "count": 0, "dry_run": is_dry}
		symbols = [position["symbol"] for position in current_positions]
		if is_dry:
			frames = _market_frames(symbols, dry_run=True)
			entry_dates = {}
			partial_sales = set()
		else:
			assert client is not None
			bars = market_data.get_ohlcv_batch(symbols, 90, student=student)
			frames = {
				symbol: rows.sort_values("timestamp").set_index("timestamp").loc[:, list(OHLCV_COLUMNS)]
				for symbol, rows in bars.groupby("symbol")
			}
			from alpaca.trading.enums import QueryOrderStatus
			from alpaca.trading.requests import GetOrdersRequest

			orders_history = client.get_orders(
				GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500, symbols=symbols)
			)
			if isinstance(orders_history, Mapping):
				orders_history = orders_history.get("orders", [])
			orders_history = list(orders_history)
			entry_dates, partial_sales = _entry_dates_and_partial_sells(orders_history, symbols)
	orders: list[dict[str, Any]] = []
	risk = config.get("risk", {})
	for position in current_positions:
		symbol = str(position["symbol"]).upper()
		frame = frames.get(symbol)
		if frame is None or frame.empty:
			continue
		frame = frame.sort_index()
		close = frame["close"].astype(float)
		atr = getattr(ta, "atr")(frame["high"], frame["low"], close, length=14)
		if atr is None or atr.dropna().empty:
			continue
		current_price = float(position.get("current_price") or close.iloc[-1])
		atr_value = float(atr.dropna().iloc[-1])
		entry = float(position["avg_entry_price"])
		entry_date = entry_dates.get(symbol)
		observed = frame if entry_date is None else frame.loc[frame.index >= entry_date.tz_localize(None) if entry_date.tzinfo else entry_date]
		high_water = float(observed["high"].max()) if not observed.empty else current_price
		stop_price = high_water - float(risk.get("atr_trailing_stop", 2.5)) * atr_value
		position_qty = float(position["qty"])
		if str(position.get("side", "long")).lower() == "short" or position_qty <= 0.0:
			continue
		available_raw = position.get("qty_available")
		available_qty = position_qty if available_raw is None else max(0.0, float(available_raw))
		qty = min(available_qty, position_qty)
		if qty <= 0.0:
			continue
		if current_price <= stop_price:
			orders.append({"symbol": symbol, "side": "sell", "qty": qty, "reason": "atr_trailing_stop"})
			continue
		held_sessions = 0 if entry_date is None else int((frame.index >= (entry_date.tz_localize(None) if entry_date.tzinfo else entry_date).normalize()).sum())
		if entry_date is not None and held_sessions >= int(risk.get("time_stop_days", 5)):
			orders.append({"symbol": symbol, "side": "sell", "qty": qty, "reason": "time_stop"})
			continue
		partial_target = entry + float(risk.get("partial_profit_atr", 2.0)) * atr_value
		if symbol not in partial_sales and current_price >= partial_target and qty > 1:
			orders.append({"symbol": symbol, "side": "sell", "qty": round(qty / 2.0, 6), "reason": "partial_profit"})
	return {"orders": orders, "count": len(orders), "dry_run": is_dry}


@mcp.tool(description="Place paper market orders. Real submissions require TRADING_ENABLED=true; dry_run never submits.")
def place_orders(
	student: str,
	orders: list[dict[str, Any]],
	dry_run: bool = True,
) -> dict[str, Any]:
	normalized = []
	seen_client_order_ids: set[str] = set()
	for order in orders:
		symbol = str(order.get("symbol", "")).strip().upper()
		side = str(order.get("side", "")).strip().lower()
		qty = float(order.get("qty", 0.0))
		if not _SYMBOL_RE.fullmatch(symbol) or side not in {"buy", "sell"} or not math.isfinite(qty) or qty <= 0:
			raise ValueError("each order must have a valid symbol, buy/sell side, and positive quantity")
		client_order_id = str(
			order.get("client_order_id") or f"{date.today():%Y%m%d}-{symbol}-{side}-{uuid.uuid4().hex[:8]}"
		)
		if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", client_order_id):
			raise ValueError("client_order_id must be 1-48 alphanumeric, underscore, or hyphen characters")
		if client_order_id in seen_client_order_ids:
			raise ValueError("client_order_id values must be unique within an order batch")
		seen_client_order_ids.add(client_order_id)
		normalized.append({"symbol": symbol, "side": side, "qty": qty, "client_order_id": client_order_id})
	if _dry_run(dry_run):
		return {"submitted": False, "dry_run": True, "orders": normalized}
	if os.environ.get("TRADING_ENABLED", "").strip().lower() != "true":
		raise PermissionError("Real paper order submission is disabled; set TRADING_ENABLED=true to enable it")
	client = _account_client(student)
	from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
	from alpaca.trading.requests import MarketOrderRequest

	submitted = []
	for order in normalized:
		request = MarketOrderRequest(
			symbol=order["symbol"],
			qty=order["qty"],
			side=OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL,
			type=OrderType.MARKET,
			time_in_force=TimeInForce.DAY,
			client_order_id=order["client_order_id"],
		)
		submitted.append(_serialize_order(client.submit_order(request)))
	return {"submitted": True, "dry_run": False, "orders": submitted}


@mcp.tool(description="Read a sanitized paper account summary and current positions.")
def get_account(student: str, dry_run: bool = False) -> dict[str, Any]:
	account, positions = _positions_for(student, dry_run)
	return {"account": account, "positions": positions, "dry_run": _dry_run(dry_run)}


def main() -> None:
	mcp.run(transport="stdio")


if __name__ == "__main__":
	main()
