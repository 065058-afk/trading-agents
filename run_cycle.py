"""Run one guarded, paper-only-or-dry trading cycle for a student."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from crew.crew import LOG_ROOT, ROOT, _call_tool, _read_config, run_crew

MAX_ORDER_NOTIONAL_USD = 10_000.0
MAX_TOTAL_EXPOSURE = 1.0
MAX_ORDERS_PER_DAY = 10
MCP_TOOL_TIMEOUT_SECONDS = 120
EASTERN = ZoneInfo("America/New_York")


def _today() -> str:
	return datetime.now(EASTERN).date().isoformat()


def _state_path(student: str) -> Path:
	return LOG_ROOT / student / "state.json"


def _safe_error_message(error: Exception, credentials: Mapping[str, str] | None = None) -> str:
	message = str(error)
	secret_values = set((credentials or {}).values())
	for name in (
		f"ALPACA_KEY_{student}" for student in ("A", "B", "C")
	):
		if os.environ.get(name):
			secret_values.add(os.environ[name])
	for name in ("ALPACA_SECRET_A", "ALPACA_SECRET_B", "ALPACA_SECRET_C", "GROQ_API_KEY", "OLLAMA_API_KEY", "FINNHUB_API_KEY"):
		if os.environ.get(name):
			secret_values.add(os.environ[name])
	for secret in secret_values:
		if secret:
			message = message.replace(secret, "[REDACTED]")
	return message or "No additional details supplied."


def _load_state(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {"peak_equity": None, "runs": {}}
	try:
		state = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as error:
		raise RuntimeError(f"Could not read runner state at {path}: {error}") from error
	if not isinstance(state, dict) or not isinstance(state.get("runs", {}), dict):
		raise RuntimeError(f"Invalid runner state at {path}")
	return {"peak_equity": state.get("peak_equity"), "runs": state.get("runs", {})}


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temp_path = path.with_suffix(".json.tmp")
	try:
		temp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
		temp_path.replace(path)
	except OSError as error:
		raise RuntimeError(f"Could not persist runner state at {path}: {error}") from error


def _trading_client(student: str, credentials: Mapping[str, str] | None = None) -> Any:
	key_name = f"ALPACA_KEY_{student}"
	secret_name = f"ALPACA_SECRET_{student}"
	key = (credentials or {}).get(key_name) or os.environ.get(key_name)
	secret = (credentials or {}).get(secret_name) or os.environ.get(secret_name)
	if not key or not secret:
		raise RuntimeError(f"Missing {key_name} or {secret_name} environment variable")
	from alpaca.trading.client import TradingClient

	return TradingClient(key, secret, paper=True)


def _account_equity(client: Any | None, config: Mapping[str, Any]) -> float:
	if client is None:
		value = float(config.get("capital", 100_000.0))
	else:
		account = client.get_account()
		raw_equity = account.get("equity") if isinstance(account, Mapping) else getattr(account, "equity", None)
		if raw_equity is None:
			raise RuntimeError("Alpaca account response did not include equity")
		try:
			value = float(raw_equity)
		except (TypeError, ValueError) as error:
			raise RuntimeError("Alpaca returned non-numeric account equity") from error
	if not math.isfinite(value) or value <= 0.0:
		raise RuntimeError("Alpaca returned invalid account equity")
	return value


def _field(value: Any, name: str, default: Any = None) -> Any:
	return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _alpaca_snapshot(client: Any) -> dict[str, Any]:
	account = client.get_account()
	positions = client.get_all_positions()
	if isinstance(positions, Mapping):
		positions = positions.get("positions", [])
	raw_equity = _field(account, "equity")
	try:
		equity = float(raw_equity)
	except (TypeError, ValueError) as error:
		raise RuntimeError("Alpaca account response did not include numeric equity") from error
	if not math.isfinite(equity) or equity <= 0.0:
		raise RuntimeError("Alpaca returned invalid account equity")
	account_data = {
		"equity": equity,
		"portfolio_value": float(_field(account, "portfolio_value", equity) or 0.0),
		"buying_power": float(_field(account, "buying_power", 0.0) or 0.0),
		"cash": float(_field(account, "cash", 0.0) or 0.0),
	}
	position_data = []
	for position in positions:
		quantity = float(_field(position, "qty", 0.0) or 0.0)
		available = _field(position, "qty_available")
		position_data.append(
			{
				"symbol": str(_field(position, "symbol", "")).upper(),
				"qty": quantity,
				"qty_available": quantity if available is None else float(available),
				"side": str(_field(position, "side", "long")),
				"avg_entry_price": float(_field(position, "avg_entry_price", 0.0) or 0.0),
				"market_value": float(_field(position, "market_value", 0.0) or 0.0),
				"current_price": float(_field(position, "current_price", 0.0) or 0.0),
			}
		)
	return {"account": account_data, "positions": position_data}


def _long_quantity(position: Mapping[str, Any]) -> float:
	quantity = float(position.get("qty", 0.0) or 0.0)
	if quantity <= 0.0 or str(position.get("side", "long")).lower() == "short":
		return 0.0
	return quantity


def _daily_order_cap(target_symbols: set[str], positions: list[Mapping[str, Any]]) -> tuple[int, str, int]:
	target_count = len(target_symbols)
	held_symbols = {
		str(position.get("symbol", "")).upper()
		for position in positions
		if _long_quantity(position) > 0.0
	}
	held_target_count = len(target_symbols & held_symbols)
	if held_target_count == 0 or held_target_count < target_count * 0.5:
		return max(20, target_count + 5), "expanded", held_target_count
	return MAX_ORDERS_PER_DAY, "normal", held_target_count


def _net_orders(
	orders: list[Mapping[str, Any]], positions: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
	net_by_symbol: dict[str, float] = {}
	for order in orders:
		symbol = str(order.get("symbol", "")).upper()
		side = str(order.get("side", "")).lower()
		quantity = float(order.get("qty", 0.0))
		if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", symbol) or side not in {"buy", "sell"}:
			raise RuntimeError("MCP returned an invalid order")
		if not math.isfinite(quantity) or quantity <= 0.0:
			raise RuntimeError(f"Missing valid price or quantity for {symbol} order")
		net_by_symbol[symbol] = net_by_symbol.get(symbol, 0.0) + (quantity if side == "buy" else -quantity)
	positions_by_symbol = {
		str(position.get("symbol", "")).upper(): position for position in positions
	}
	netted: list[dict[str, Any]] = []
	for symbol, net_quantity in net_by_symbol.items():
		if net_quantity > 1e-6:
			netted.append({"symbol": symbol, "side": "buy", "qty": net_quantity})
		elif net_quantity < -1e-6:
			position = positions_by_symbol.get(symbol, {})
			held_quantity = _long_quantity(position)
			available_raw = position.get("qty_available")
			available_quantity = held_quantity if available_raw is None else max(0.0, float(available_raw))
			sell_quantity = min(-net_quantity, held_quantity, available_quantity)
			if sell_quantity > 1e-6:
				netted.append({"symbol": symbol, "side": "sell", "qty": sell_quantity})
	return netted


def _bounded_orders(
	orders: Sequence[Mapping[str, Any]],
	prices: Mapping[str, float],
	student: str,
	run_date: str,
	remaining_slots: int,
) -> list[dict[str, Any]]:
	bounded: list[dict[str, Any]] = []
	for order in orders:
		if len(bounded) >= remaining_slots:
			break
		symbol = str(order.get("symbol", "")).upper()
		side = str(order.get("side", "")).lower()
		price = float(prices.get(symbol, 0.0))
		quantity = float(order.get("qty", 0.0))
		if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", symbol) or side not in {"buy", "sell"}:
			raise RuntimeError("MCP returned an invalid order")
		if not math.isfinite(price) or price <= 0.0 or not math.isfinite(quantity) or quantity <= 0.0:
			raise RuntimeError(f"Missing valid price or quantity for {symbol} order")
		quantity = min(quantity, MAX_ORDER_NOTIONAL_USD / price)
		quantity = math.floor(quantity * 1_000_000.0) / 1_000_000.0
		if quantity <= 0.0:
			continue
		bounded.append(
				{
					"symbol": symbol,
					"side": side,
					"qty": quantity,
					"client_order_id": f"cycle-{student}-{run_date.replace('-', '')}-{side}-{symbol}",
				}
			)
	return bounded


def _apply_approval_edit(
	proposed: list[dict[str, Any]], decision: Mapping[str, Any]
) -> list[dict[str, Any]]:
	action = decision.get("action")
	if action == "approve":
		return proposed
	if action == "reject":
		return []
	if action != "edit" or not isinstance(decision.get("orders"), list):
		raise ValueError("Approval decision must be approve, reject, or edit with orders")
	allowed = {str(order["client_order_id"]): order for order in proposed}
	result: list[dict[str, Any]] = []
	seen: set[str] = set()
	for edited in decision["orders"]:
		if not isinstance(edited, Mapping):
			raise ValueError("Edited orders must be objects")
		client_order_id = str(edited.get("client_order_id", ""))
		original = allowed.get(client_order_id)
		if original is None or client_order_id in seen:
			raise ValueError("Edits may only retain or remove proposed orders")
		symbol = str(edited.get("symbol", "")).upper()
		side = str(edited.get("side", "")).lower()
		quantity = float(edited.get("qty", 0.0))
		if symbol != original["symbol"] or side != original["side"]:
			raise ValueError("Edits cannot change an order's symbol or side")
		if not math.isfinite(quantity) or quantity <= 0.0 or quantity > float(original["qty"]):
			raise ValueError("Edited quantity must be positive and no greater than proposed")
		seen.add(client_order_id)
		result.append({**original, "qty": quantity})
	return result


def _execution_handler(
	student: str,
	run_date: str,
	dry_run: bool,
	breaker_active: bool,
	target_symbols: set[str],
	approval_handler: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
	event_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
):
	def execute(context: dict[str, Any]) -> dict[str, Any]:
		tool_map = context["tool_map"]
		account = _call_tool(tool_map, "get_account", student=student, dry_run=dry_run)
		account_positions = account.get("positions", []) if isinstance(account, Mapping) else []
		if not isinstance(account_positions, list):
			account_positions = []
		daily_order_cap, cap_kind, held_target_count = _daily_order_cap(target_symbols, account_positions)
		print(
			f"Daily order cap applied: {daily_order_cap} ({cap_kind}; "
			f"{held_target_count}/{len(target_symbols)} target positions held)."
		)
		exits = _call_tool(tool_map, "check_exits", student=student, dry_run=dry_run)
		exit_orders = exits.get("orders", []) if isinstance(exits, dict) else []
		portfolio_diff: dict[str, Any] = {"orders": []}
		if context["risk_approved"]:
			portfolio_diff = _call_tool(
				tool_map,
				"portfolio_diff",
				student=student,
				target_weights=context["target_weights"],
				dry_run=dry_run,
			)
		diff_orders = portfolio_diff.get("orders", []) if isinstance(portfolio_diff, dict) else []
		orders_to_net = list(diff_orders) + list(exit_orders)
		if breaker_active or not context["risk_approved"]:
			orders_to_net = [
				order for order in orders_to_net if str(order.get("side", "")).lower() == "sell"
			]
		netted_orders = _net_orders(orders_to_net, account_positions)
		sells = [order for order in netted_orders if order["side"] == "sell"]
		buys = [order for order in netted_orders if order["side"] == "buy"]
		prices = context["latest_prices"]
		bounded_sells = _bounded_orders(
			sells, prices, student, run_date, daily_order_cap
		)
		bounded_buys = _bounded_orders(
			buys, prices, student, run_date, daily_order_cap - len(bounded_sells)
		)
		proposed_orders = [*bounded_sells, *bounded_buys]
		if event_callback is not None:
			event_callback(
				"orders_proposed",
				{
					"orders": proposed_orders,
					"risk_approved": bool(context["risk_approved"]),
					"validation": context["validation"],
				},
			)
		if approval_handler is not None:
			decision = approval_handler({"orders": proposed_orders, "mode": "dry" if dry_run else "live"})
			proposed_orders = _apply_approval_edit(proposed_orders, decision)
			bounded_sells = [order for order in proposed_orders if order["side"] == "sell"]
			bounded_buys = [order for order in proposed_orders if order["side"] == "buy"]
			exits["approval"] = {"action": decision.get("action", "")}
		sell_result = (
			_call_tool(tool_map, "place_orders", student=student, orders=bounded_sells, dry_run=dry_run)
			if bounded_sells
			else {"submitted": False, "orders": []}
		)
		buy_result = (
			_call_tool(tool_map, "place_orders", student=student, orders=bounded_buys, dry_run=dry_run)
			if bounded_buys
			else {"submitted": False, "orders": []}
		)
		result = {
			"approved": bool(context["risk_approved"]),
			"validation": context["validation"],
			"portfolio_diff": portfolio_diff,
			"exits": exits,
			"sell_orders": sell_result,
			"buy_orders": buy_result,
			"order_count": len(bounded_sells) + len(bounded_buys),
			"proposed_orders": proposed_orders,
			"breaker_active": breaker_active,
			"mode": "dry" if dry_run else "live",
			"daily_order_cap": {
				"limit": daily_order_cap,
				"applied": cap_kind,
				"held_target_positions": held_target_count,
				"target_positions": len(target_symbols),
			},
		}
		result["account"] = account
		if event_callback is not None:
			event_callback("execution_complete", result)
		return result

	return execute


def run_cycle(
	student: str,
	mode: str,
	*,
	force_clock: bool = False,
	config_override: Mapping[str, Any] | None = None,
	credential_override: Mapping[str, str] | None = None,
	approval_handler: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
	event_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
	mcp_timeout_seconds: int = MCP_TOOL_TIMEOUT_SECONDS,
) -> int:
	student = student.upper()
	try:
		if student not in {"A", "B", "C"}:
			raise ValueError("student must be A, B, or C")
		if mode not in {"dry", "live"}:
			raise ValueError("mode must be dry or live")
		if force_clock and mode != "dry":
			raise ValueError("--force-clock is available only in dry mode")

		load_dotenv(ROOT / ".env")
		config = _read_config(student)
		if config_override:
			for key in ("high_beta", "diversified"):
				if key in config_override:
					values = config_override[key]
					if not isinstance(values, list) or not values or any(
						not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", value)
						for value in values
					):
						raise ValueError(f"Invalid {key} override")
					config[key] = list(dict.fromkeys(value.upper() for value in values))
			if set(config["high_beta"]) & set(config["diversified"]):
				raise ValueError("high_beta and diversified symbols must not overlap")
			if "signal_weights" in config_override:
				weights = config_override["signal_weights"]
				allowed_signals = {"trend", "rotation", "breakout", "dip", "ml", "news"}
				if not isinstance(weights, Mapping) or set(weights) - allowed_signals:
					raise ValueError("Invalid signal_weights override")
				parsed_weights = {name: float(value) for name, value in weights.items()}
				if not parsed_weights or any(not math.isfinite(value) or value < 0 for value in parsed_weights.values()) or sum(parsed_weights.values()) <= 0:
					raise ValueError("Signal weights must be finite, non-negative, and sum to a positive value")
				config["signal_weights"] = parsed_weights
			if "llm" in config_override:
				llm_override = config_override["llm"]
				provider_order = llm_override.get("provider_order") if isinstance(llm_override, Mapping) else None
				if not isinstance(provider_order, list) or not provider_order or any(
					name not in config["llm"]["providers"] for name in provider_order
				):
					raise ValueError("Invalid LLM provider selection")
				config["llm"] = {**config["llm"], "provider_order": provider_order}
		trading_enabled = os.environ.get("TRADING_ENABLED", "").strip().lower() == "true"
		dry_run = mode == "dry" or not trading_enabled
		if mode == "live" and not trading_enabled:
			print("TRADING_ENABLED is not true; treating --mode live as dry mode.")

		client: Any | None = None
		if not (dry_run and force_clock):
			client = (
				_trading_client(student, credential_override)
				if credential_override is not None
				else _trading_client(student)
			)
		if not force_clock:
			if client is None:
				raise RuntimeError("Alpaca credentials are required to check the market clock")
			clock = client.get_clock()
			if not bool(clock.is_open):
				reason = "Market is closed; no cycle was run."
				if event_callback is not None:
					event_callback(
						"cycle_skipped",
						{"reason": reason, "account_snapshot": _alpaca_snapshot(client)},
					)
				print(reason)
				return 0

		run_date = _today()
		state_path = _state_path(student)
		state = _load_state(state_path)
		recorded_run = state["runs"].get(run_date, {})
		if (
			not dry_run
			and recorded_run.get("status") == "completed"
			and recorded_run.get("mode") == "live"
		):
			reason = f"A run for student {student} is already recorded for {run_date}; skipping."
			if event_callback is not None:
				if client is None:
					try:
						client = _trading_client(student, credential_override)
					except RuntimeError:
						pass
				account_snapshot = _alpaca_snapshot(client) if client is not None else None
				event_callback(
					"cycle_skipped",
					{"reason": reason, "run_date": run_date, "account_snapshot": account_snapshot},
				)
			print(reason)
			return 0

		equity = _account_equity(None if dry_run else client, config)
		peak_equity = float(state.get("peak_equity") or equity)
		if not math.isfinite(peak_equity) or peak_equity <= 0.0:
			raise RuntimeError("Stored peak equity is invalid")
		drawdown_limit = float(config.get("risk", {}).get("drawdown_breaker_pct", 0.06))
		if not 0.0 < drawdown_limit < 1.0:
			raise RuntimeError("Configured drawdown_breaker_pct must be between zero and one")
		breaker_active = equity < peak_equity * (1.0 - drawdown_limit)
		target_scale = 0.5 if breaker_active else 1.0
		if breaker_active:
			print("Drawdown circuit breaker active: target exposure halved; new buys blocked for today.")

		crew_result = run_crew(
			student,
			"dry" if dry_run else "paper",
			config_override=config,
			credential_override=credential_override,
			mcp_timeout_seconds=max(MCP_TOOL_TIMEOUT_SECONDS, int(mcp_timeout_seconds)),
			target_scale=target_scale,
			max_total_exposure=MAX_TOTAL_EXPOSURE,
			allow_reduced_exposure=breaker_active,
			cycle_event_callback=event_callback,
			execution_handler=_execution_handler(
				student,
				run_date,
				dry_run,
				breaker_active,
				{str(symbol).upper() for symbol in [*config["high_beta"], *config["diversified"]]},
				approval_handler,
				event_callback,
			),
		)
		if not dry_run:
			state["peak_equity"] = max(peak_equity, equity)
			state["runs"][run_date] = {
				"status": "completed",
				"mode": "live",
				"equity": equity,
				"drawdown_breaker": breaker_active,
				"order_count": int(crew_result.get("execution", {}).get("order_count", 0)),
				"daily_order_cap": crew_result.get("execution", {}).get("daily_order_cap"),
			}
			_write_state(state_path, state)
		if event_callback is not None:
			event_callback("cycle_complete", crew_result)
		print(f"Cycle complete for student {student} ({'dry' if dry_run else 'live'}).")
		return 0
	except Exception as error:
		detail = _safe_error_message(error, credential_override)
		if event_callback is not None:
			event_callback(
				"cycle_failed",
				{"reason": f"{type(error).__name__}: {detail}", "exception_type": type(error).__name__},
			)
		print(f"run_cycle failed: {type(error).__name__}: {detail}")
		return 1


def main() -> int:
	parser = argparse.ArgumentParser(description="Run one guarded trading cycle.")
	parser.add_argument("--student", required=True, choices=("A", "B", "C"))
	parser.add_argument("--mode", choices=("dry", "live"), default="dry")
	parser.add_argument("--force-clock", action="store_true", help="Bypass the market clock in dry mode only.")
	args = parser.parse_args()
	return run_cycle(args.student, args.mode, force_clock=args.force_clock)


if __name__ == "__main__":
	raise SystemExit(main())
