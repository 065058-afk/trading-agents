"""Walk-forward comparison of the current student strategy and a legacy baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from mcp_server.data import history_cache
from mcp_server import signals

EXPERIMENTS_DIR = ROOT / "experiments"
BENCHMARKS = ("SPY", "QQQ")
RISK_OFF_MULTIPLIERS = {
	"trend": 1.5,
	"rotation": 1.25,
	"breakout": 0.25,
	"dip": 0.5,
	"ml": 0.5,
	"news": 0.25,
}
NEUTRAL_MULTIPLIERS = {
	"trend": 1.15,
	"rotation": 1.1,
	"breakout": 0.8,
	"dip": 0.9,
	"ml": 0.9,
	"news": 0.5,
}


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
	result = frame.copy()
	result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
	index = pd.DatetimeIndex(pd.to_datetime(result.index))
	if index.tz is not None:
		index = index.tz_convert("America/New_York").tz_localize(None)
	result.index = index.normalize()
	return result.sort_index()


def _load_frames(symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
	frames = history_cache([*symbols, *BENCHMARKS])
	return {symbol: _normalize_frame(frames[symbol]) for symbol in [*symbols, *BENCHMARKS]}


def _target_dates(frames: Mapping[str, pd.DataFrame], years: int) -> tuple[pd.DatetimeIndex, int]:
	common: pd.DatetimeIndex | None = None
	for frame in frames.values():
		frame_dates = pd.DatetimeIndex(frame.index)
		common = frame_dates if common is None else common.intersection(frame_dates)
	if common is None or len(common) < 300:
		raise ValueError("At least 300 common daily bars are required for a walk-forward backtest")
	common = common.sort_values()
	start_index = max(252, len(common) - years * 252)
	if start_index >= len(common) - 30:
		raise ValueError("Not enough cached history remains for the requested backtest period")
	return common, start_index


def _market_arrays(
	frames: Mapping[str, pd.DataFrame],
	symbols: Sequence[str],
	dates: pd.DatetimeIndex,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
	arrays: dict[str, np.ndarray] = {}
	for field in ("open", "high", "low", "close", "volume"):
		arrays[field] = np.column_stack(
			[frames[symbol].reindex(dates)[field].to_numpy(dtype=float) for symbol in symbols]
		)
	atr = np.column_stack(
			[
				getattr(ta, "atr")(
					frames[symbol]["high"], frames[symbol]["low"], frames[symbol]["close"], length=14
				)
				.reindex(dates)
				.to_numpy(dtype=float)
				for symbol in symbols
			]
	)
	return arrays, atr


def _weighted_score(
	symbol: str,
	weights: Mapping[str, float],
	signal_values: Mapping[str, Mapping[str, float]],
) -> float:
	return float(
		sum(weight * signal_values[name].get(symbol, 0.0) for name, weight in weights.items())
	)


def _capped_group_allocation(
	scores: Mapping[str, float], symbols: Sequence[str], budget: float, cap: float
) -> dict[str, float]:
	active = {symbol: max(0.0, scores.get(symbol, 0.0)) for symbol in symbols}
	active = {symbol: score for symbol, score in active.items() if score > 0}
	allocations: dict[str, float] = {}
	remaining = budget
	while active and remaining > 1e-10:
		total_score = sum(active.values())
		capped = [
			symbol
			for symbol, score in active.items()
			if remaining * score / total_score >= cap - allocations.get(symbol, 0.0)
		]
		if not capped:
			for symbol, score in active.items():
				allocations[symbol] = allocations.get(symbol, 0.0) + remaining * score / total_score
			remaining = 0.0
			break
		for symbol in capped:
			capacity = max(0.0, cap - allocations.get(symbol, 0.0))
			allocations[symbol] = allocations.get(symbol, 0.0) + capacity
			remaining -= capacity
			del active[symbol]
	return allocations


def _portfolio_weights(
	scores: Mapping[str, float],
	high_beta: Sequence[str],
	diversified: Sequence[str],
	config: Mapping[str, Any],
	regime: str,
) -> np.ndarray:
	all_symbols = list(dict.fromkeys([*high_beta, *diversified]))
	share = float(config.get("high_beta_share_base", 0.45))
	if regime == "risk_off":
		share *= 0.75
	share = float(np.clip(share, 0.0, 1.0))
	weights = _capped_group_allocation(
		scores,
		high_beta,
		share,
		float(config.get("max_position_pct_high_beta", 0.15)),
	)
	weights.update(
		_capped_group_allocation(
			scores,
			diversified,
			1.0 - share,
			float(config.get("max_position_pct_diversified", 0.10)),
		)
	)
	return np.array([weights.get(symbol, 0.0) for symbol in all_symbols], dtype=float)


def _regime_weights(config: Mapping[str, Any], regime: str, use_engines: bool) -> dict[str, float]:
	configured = config.get("signal_weights", {})
	base = {
		name: float(configured.get(name, 0.0))
		for name in ("trend", "rotation", "breakout", "dip", "ml", "news")
	}
	if not use_engines:
		legacy_total = base["trend"] + base["rotation"]
		if legacy_total <= 0:
			raise ValueError("trend and rotation weights must sum to a positive value")
		return {"trend": base["trend"] / legacy_total, "rotation": base["rotation"] / legacy_total}
	multipliers = RISK_OFF_MULTIPLIERS if regime == "risk_off" else NEUTRAL_MULTIPLIERS if regime == "neutral" else {}
	adapted = {name: weight * multipliers.get(name, 1.0) for name, weight in base.items()}
	total = sum(adapted.values())
	if total <= 0:
		raise ValueError("configured signal weights must sum to a positive value")
	return {name: weight / total for name, weight in adapted.items()}


def _signals_at(
	symbols: Sequence[str],
	frames: Mapping[str, pd.DataFrame],
	as_of: pd.Timestamp,
	config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float], str]:
	trend = signals.signal_trend(symbols, as_of=as_of, history=frames)
	rotation = signals.signal_rotation(symbols, as_of=as_of, history=frames)
	breakout = signals.signal_breakout(symbols, as_of=as_of, history=frames)
	dip = signals.signal_dip(symbols, as_of=as_of, history=frames)
	ml = signals.signal_ml(symbols, as_of=as_of, history=frames, persist=False)
	regime = signals.regime_state(as_of=as_of, history=frames)
	# Historical headline archives are not available; using today's news would leak future data.
	values = {
		"trend": trend,
		"rotation": rotation,
		"breakout": breakout,
		"dip": dip,
		"ml": ml,
		"news": {symbol: 0.0 for symbol in symbols},
	}
	full_weights = _regime_weights(config, regime, use_engines=True)
	legacy_weights = _regime_weights(config, regime, use_engines=False)
	full_scores = {
		symbol: _weighted_score(symbol, full_weights, values) for symbol in symbols
	}
	legacy_scores = {
		symbol: _weighted_score(symbol, legacy_weights, values) for symbol in symbols
	}
	return full_scores, legacy_scores, regime


def _target_maps(
	frames: Mapping[str, pd.DataFrame],
	symbols: Sequence[str],
	high_beta: Sequence[str],
	diversified: Sequence[str],
	config: Mapping[str, object],
	dates: pd.DatetimeIndex,
	start_index: int,
	rebalance_days: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, str]]:
	full_targets: dict[int, np.ndarray] = {}
	legacy_targets: dict[int, np.ndarray] = {}
	regimes: dict[int, str] = {}
	for execution_index in range(start_index, len(dates), rebalance_days):
		signal_date = dates[execution_index - 1]
		full_scores, legacy_scores, regime = _signals_at(symbols, frames, signal_date, config)
		full_targets[execution_index] = _portfolio_weights(
			full_scores, high_beta, diversified, config, regime
		)
		legacy_targets[execution_index] = _portfolio_weights(
			legacy_scores, high_beta, diversified, config, "neutral"
		)
		regimes[execution_index] = regime
	return full_targets, legacy_targets, regimes


def _simulate(
	dates: pd.DatetimeIndex,
	arrays: Mapping[str, np.ndarray],
	atr: np.ndarray,
	targets: Mapping[int, np.ndarray],
	start_index: int,
	cost_bps: float,
	risk_config: Mapping[str, Any],	rebalance_days: int,
	use_risk_rules: bool,
) -> tuple[pd.Series, pd.Series, pd.Series]:
	symbol_count = arrays["close"].shape[1]
	weights = np.zeros(symbol_count, dtype=float)
	cash_weight = 1.0
	position_state: dict[int, dict[str, float | int | bool]] = {}
	nav = 1.0
	peak_nav = 1.0
	cooldown_until = -1
	breaker_armed = True
	equity: list[float] = []
	daily_returns: list[float] = []
	turnovers: list[float] = []
	stop_multiple = float(risk_config.get("atr_trailing_stop", 2.5))
	partial_multiple = float(risk_config.get("partial_profit_atr", 2.0))
	time_stop = int(risk_config.get("time_stop_days", 5))
	drawdown_limit = float(risk_config.get("drawdown_breaker_pct", 0.06))
	cost_rate = cost_bps / 10_000.0

	for index in range(start_index, len(dates)):
		previous_nav = nav
		risk_exit_turnover = 0.0
		if index > start_index:
			previous_close = arrays["close"][index - 1]
			close = arrays["close"][index]
			port_factor = cash_weight
			next_cash_value = cash_weight
			next_risky_values = np.zeros(symbol_count, dtype=float)
			for symbol_index, weight in enumerate(weights):
				if weight <= 0:
					continue
				close_factor = close[symbol_index] / previous_close[symbol_index]
				position_factor = close_factor
				retained_factor = close_factor
				state = position_state.get(symbol_index)
				if use_risk_rules and state is not None:
					bar_open = arrays["open"][index, symbol_index]
					bar_high = arrays["high"][index, symbol_index]
					bar_low = arrays["low"][index, symbol_index]
					atr_prior = atr[index - 1, symbol_index]
					held_days = int(state["held_days"]) + 1
					stop = (
						float(state["high_water"])
						- stop_multiple * atr_prior
						if np.isfinite(atr_prior) and atr_prior > 0
						else -np.inf
					)
					stop_fill = bar_open if bar_open <= stop else stop if bar_low <= stop else None
					if stop_fill is not None:
						position_factor = stop_fill / previous_close[symbol_index]
						retained_factor = 0.0
						risk_exit_turnover += weight * position_factor
						position_state.pop(symbol_index, None)
					elif held_days >= time_stop:
						retained_factor = 0.0
						risk_exit_turnover += weight * position_factor
						position_state.pop(symbol_index, None)
					elif (
						not bool(state["partial_taken"])
						and np.isfinite(float(state["entry_atr"]))
						and float(state["entry_atr"]) > 0
					):
						target = float(state["entry_price"]) + partial_multiple * float(state["entry_atr"])
						if bar_high >= target:
							fill = bar_open if bar_open >= target else target
							fill_factor = fill / previous_close[symbol_index]
							risk_exit_turnover += 0.5 * weight * fill_factor
							position_factor = 0.5 * (fill_factor + close_factor)
							retained_factor = 0.5 * close_factor
							state["partial_taken"] = True
						state["held_days"] = held_days
						state["high_water"] = max(float(state["high_water"]), bar_high)
					else:
						state["held_days"] = held_days
						state["high_water"] = max(float(state["high_water"]), bar_high)
				port_factor += weight * position_factor
				next_risky_values[symbol_index] = weight * retained_factor
				next_cash_value += weight * (position_factor - retained_factor)
			port_factor -= cost_rate * risk_exit_turnover
			if port_factor <= 0 or not np.isfinite(port_factor):
				raise ValueError(f"Portfolio value became invalid on {dates[index].date()}")
			nav *= port_factor
			weights = next_risky_values / port_factor
			cash_weight = next_cash_value / port_factor
		peak_nav = max(peak_nav, nav)

		turnover = risk_exit_turnover
		force_cash = False
		if use_risk_rules and not breaker_armed and nav >= peak_nav:
			breaker_armed = True
		if use_risk_rules and breaker_armed and nav / peak_nav - 1.0 <= -drawdown_limit:
			force_cash = True
			cooldown_until = index + rebalance_days
			breaker_armed = False
		elif index in targets:
			target_weights = targets[index].copy()
			if index < cooldown_until:
				target_weights[:] = 0.0
			rebalance_turnover = float(np.abs(target_weights - weights).sum())
			turnover += rebalance_turnover
			nav *= max(0.0, 1.0 - cost_rate * rebalance_turnover)
			old_weights = weights.copy()
			weights = target_weights
			cash_weight = max(0.0, 1.0 - float(weights.sum()))
			if use_risk_rules:
				for symbol_index, target_weight in enumerate(weights):
					if target_weight <= 0:
						position_state.pop(symbol_index, None)
					elif old_weights[symbol_index] <= 0 or symbol_index not in position_state:
						atr_at_entry = atr[max(0, index - 1), symbol_index]
						position_state[symbol_index] = {
							"entry_price": float(arrays["close"][index, symbol_index]),
							"entry_atr": float(atr_at_entry) if np.isfinite(atr_at_entry) else 0.0,
							"high_water": float(arrays["close"][index, symbol_index]),
							"held_days": 0,
							"partial_taken": False,
						}
		if force_cash:
			breaker_turnover = float(weights.sum())
			turnover += breaker_turnover
			nav *= max(0.0, 1.0 - cost_rate * breaker_turnover)
			weights[:] = 0.0
			cash_weight = 1.0
			position_state.clear()
		peak_nav = max(peak_nav, nav)
		turnovers.append(turnover)
		equity.append(nav)
		daily_returns.append(nav / previous_nav - 1.0 if previous_nav else 0.0)

	index = dates[start_index:]
	return (
		pd.Series(equity, index=index, name="equity"),
		pd.Series(daily_returns, index=index, name="daily_return"),
		pd.Series(turnovers, index=index, name="turnover"),
	)


def _metrics(equity: pd.Series, daily_returns: pd.Series) -> dict[str, float]:
	if equity.empty:
		return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0}
	periods = max(len(equity) - 1, 1)
	ending = float(equity.iloc[-1])
	cagr = ending ** (252.0 / periods) - 1.0 if ending > 0 else -1.0
	standard_deviation = float(daily_returns.std(ddof=1))
	sharpe = (
		float(daily_returns.mean() / standard_deviation * np.sqrt(252.0))
		if standard_deviation > 0
		else 0.0
	)
	drawdown = equity / equity.cummax() - 1.0
	return {
		"cagr": float(cagr),
		"sharpe": sharpe,
		"max_drawdown": float(drawdown.min()),
		"hit_rate": float((daily_returns > 0).mean()),
	}


def _format_summary(
	student: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	metrics: Mapping[str, Mapping[str, float]],
	cost_bps: float,
	rebalance_days: int,
) -> str:
	rows = []
	for name in ("full_strategy", "legacy_baseline", "SPY", "equal_weight"):
		result = metrics[name]
		rows.append(
			f"| {name.replace('_', ' ')} | {result['cagr']:.2%} | {result['sharpe']:.2f} | "
			f"{result['max_drawdown']:.2%} | {result['hit_rate']:.2%} |"
		)
	full = metrics["full_strategy"]
	return "\n".join(
		[
			f"# Student {student} walk-forward backtest",
			"",
			f"Period: {start.date()} to {end.date()} ({(end - start).days} calendar days).",
			f"Rebalance interval: {rebalance_days} sessions. Transaction cost: {cost_bps:.1f} bps per unit of turnover.",
			"Historical news headlines are not archived, so news tilt is neutral to avoid look-ahead.",
			"Equal-weight is a buy-and-hold basket initialized equally across configured stocks.",
			"The legacy baseline uses trend+rotation with static weights and no risk overlays.",
			"",
			"| Portfolio | CAGR | Sharpe | Max drawdown | Hit rate |",
			"|---|---:|---:|---:|---:|",
			*rows,
			"",
			f"Full strategy CAGR difference: {full['cagr'] - metrics['SPY']['cagr']:+.2%} vs SPY; "
			f"{full['cagr'] - metrics['equal_weight']['cagr']:+.2%} vs equal-weight.",
		]
	)


def run_backtest(
	student: str,
	*,
	years: int = 5,
	rebalance_days: int = 5,
	cost_bps: float = 5.0,
	output_dir: Path = EXPERIMENTS_DIR,
) -> tuple[pd.DataFrame, str]:
	config_path = ROOT / "config" / f"student_{student}.yaml"
	with config_path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	high_beta = [str(symbol).upper() for symbol in config["high_beta"]]
	diversified = [str(symbol).upper() for symbol in config["diversified"]]
	symbols = list(dict.fromkeys([*high_beta, *diversified]))
	frames = _load_frames(symbols)
	dates, start_index = _target_dates(frames, years)
	frames = {
		symbol: frame.loc[frame.index >= dates[max(0, start_index - 6 * 252)]]
		for symbol, frame in frames.items()
	}
	dates = dates[dates >= frames[symbols[0]].index.min()]
	start_index = max(252, len(dates) - years * 252)
	arrays, atr = _market_arrays(frames, symbols, dates)
	full_targets, legacy_targets, regimes = _target_maps(
		frames,
		symbols,
		high_beta,
		diversified,
		config,
		dates,
		start_index,
		rebalance_days,
	)
	risk_config = config.get("risk", {})
	full_equity, full_returns, full_turnover = _simulate(
		dates,
		arrays,
		atr,
		full_targets,
		start_index,
		cost_bps,
		risk_config,
		rebalance_days,
		use_risk_rules=True,
	)
	legacy_equity, legacy_returns, legacy_turnover = _simulate(
		dates,
		arrays,
		atr,
		legacy_targets,
		start_index,
		cost_bps,
		risk_config,
		rebalance_days,
		use_risk_rules=False,
	)
	close = arrays["close"]
	# SPY is appended to the frame map but not the strategy asset vector.
	spy_close = frames["SPY"].reindex(dates)["close"].to_numpy(dtype=float)
	spy_equity = pd.Series(spy_close[start_index:] / spy_close[start_index], index=dates[start_index:])
	asset_prices = np.column_stack(
		[frames[symbol].reindex(dates)["close"].to_numpy(dtype=float) for symbol in symbols]
	)
	equal_weight_values = (asset_prices[start_index:] / asset_prices[start_index]).mean(axis=1)
	equal_weight_equity = pd.Series(equal_weight_values, index=dates[start_index:])
	benchmark_returns = {
		"full_strategy": full_returns,
		"legacy_baseline": legacy_returns,
		"SPY": spy_equity.pct_change().fillna(0.0),
		"equal_weight": equal_weight_equity.pct_change().fillna(0.0),
	}
	equities = {
		"full_strategy": full_equity,
		"legacy_baseline": legacy_equity,
		"SPY": spy_equity,
		"equal_weight": equal_weight_equity,
	}
	metric_table = {
		name: _metrics(equities[name], benchmark_returns[name]) for name in equities
	}
	regime_series = pd.Series(index=dates[start_index:], dtype="object")
	for index, regime in regimes.items():
		if index < len(dates):
			regime_series.loc[dates[index]] = regime
	regime_series = regime_series.ffill().fillna("neutral")
	output = pd.DataFrame(
		{
			"full_strategy_equity": full_equity,
			"full_strategy_return": full_returns,
			"full_strategy_turnover": full_turnover,
			"legacy_baseline_equity": legacy_equity,
			"legacy_baseline_return": legacy_returns,
			"legacy_baseline_turnover": legacy_turnover,
			"spy_equity": spy_equity,
			"spy_return": benchmark_returns["SPY"],
			"equal_weight_equity": equal_weight_equity,
			"equal_weight_return": benchmark_returns["equal_weight"],
			"regime": regime_series,
		},
		index=dates[start_index:],
	)
	output.index.name = "date"
	start_date, end_date = dates[start_index], dates[-1]
	summary = _format_summary(student, start_date, end_date, metric_table, cost_bps, rebalance_days)
	output_dir.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	base_name = f"backtest_student_{student}_{stamp}"
	output.to_csv(output_dir / f"{base_name}.csv", float_format="%.8f")
	(output_dir / f"{base_name}.md").write_text(summary + "\n", encoding="utf-8")
	return output, summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Walk-forward student strategy backtest.")
	parser.add_argument("--student", required=True, choices=("A", "B", "C"))
	parser.add_argument("--years", type=int, default=5, help="Years of out-of-sample daily results")
	parser.add_argument("--rebalance-days", type=int, default=5, help="Trading sessions between rebalances")
	parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in basis points")
	args = parser.parse_args()
	if args.years < 1 or args.rebalance_days < 1 or args.cost_bps < 0:
		parser.error("years and rebalance-days must be positive; cost-bps cannot be negative")
	_, summary = run_backtest(
		args.student,
		years=args.years,
		rebalance_days=args.rebalance_days,
		cost_bps=args.cost_bps,
	)
	print(summary)
	print(f"\nSaved CSV and Markdown summary under {EXPERIMENTS_DIR}.")


if __name__ == "__main__":
	main()