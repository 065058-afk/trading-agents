"""Run one guarded, paper-only-or-dry trading cycle for a student."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from crew.crew import LOG_ROOT, ROOT, _call_tool, _read_config, run_crew

MAX_ORDER_NOTIONAL_USD = 10_000.0
MAX_TOTAL_EXPOSURE = 1.0
MAX_ORDERS_PER_DAY = 10
EASTERN = ZoneInfo("America/New_York")


def _today() -> str:
	return datetime.now(EASTERN).date().isoformat()


def _state_path(student: str) -> Path:
	return LOG_ROOT / student / "state.json"


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


def _trading_client(student: str) -> Any:
	key_name = f"ALPACA_KEY_{student}"
	secret_name = f"ALPACA_SECRET_{student}"
	key = os.environ.get(key_name)
	secret = os.environ.get(secret_name)
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


def _bounded_orders(
	orders: list[Mapping[str, Any]],
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


def _execution_handler(
	student: str,
	run_date: str,
	dry_run: bool,
	breaker_active: bool,
):
	def execute(context: dict[str, Any]) -> dict[str, Any]:
		tool_map = context["tool_map"]
		exits = _call_tool(tool_map, "check_exits", student=student, dry_run=dry_run)
		exit_orders = exits.get("orders", []) if isinstance(exits, dict) else []
		exit_symbols = {str(order.get("symbol", "")).upper() for order in exit_orders}
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
		sells = [
			order
			for order in diff_orders
			if str(order.get("side", "")).lower() == "sell"
			and str(order.get("symbol", "")).upper() not in exit_symbols
		]
		sells.extend(exit_orders)
		buys = [] if breaker_active or not context["risk_approved"] else [
			order for order in diff_orders if str(order.get("side", "")).lower() == "buy"
		]
		prices = context["latest_prices"]
		bounded_sells = _bounded_orders(
			sells, prices, student, run_date, MAX_ORDERS_PER_DAY
		)
		bounded_buys = _bounded_orders(
			buys, prices, student, run_date, MAX_ORDERS_PER_DAY - len(bounded_sells)
		)
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
		return {
			"approved": bool(context["risk_approved"]),
			"validation": context["validation"],
			"portfolio_diff": portfolio_diff,
			"exits": exits,
			"sell_orders": sell_result,
			"buy_orders": buy_result,
			"order_count": len(bounded_sells) + len(bounded_buys),
			"breaker_active": breaker_active,
			"mode": "dry" if dry_run else "live",
		}

	return execute


def run_cycle(student: str, mode: str, *, force_clock: bool = False) -> int:
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
		trading_enabled = os.environ.get("TRADING_ENABLED", "").strip().lower() == "true"
		dry_run = mode == "dry" or not trading_enabled
		if mode == "live" and not trading_enabled:
			print("TRADING_ENABLED is not true; treating --mode live as dry mode.")

		client: Any | None = None
		if not (dry_run and force_clock):
			client = _trading_client(student)
		if not force_clock:
			if client is None:
				raise RuntimeError("Alpaca credentials are required to check the market clock")
			clock = client.get_clock()
			if not bool(clock.is_open):
				print("Market is closed; no cycle was run.")
				return 0

		run_date = _today()
		state_path = _state_path(student)
		state = _load_state(state_path)
		if state["runs"].get(run_date, {}).get("status") == "completed":
			print(f"A run for student {student} is already recorded for {run_date}; skipping.")
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
			target_scale=target_scale,
			max_total_exposure=MAX_TOTAL_EXPOSURE,
			allow_reduced_exposure=breaker_active,
			execution_handler=_execution_handler(student, run_date, dry_run, breaker_active),
		)
		state["peak_equity"] = max(peak_equity, equity)
		state["runs"][run_date] = {
			"status": "completed",
			"mode": "dry" if dry_run else "live",
			"equity": equity,
			"drawdown_breaker": breaker_active,
			"order_count": int(crew_result.get("execution", {}).get("order_count", 0)),
		}
		_write_state(state_path, state)
		print(f"Cycle complete for student {student} ({'dry' if dry_run else 'live'}).")
		return 0
	except Exception as error:
		print(f"run_cycle failed: {type(error).__name__}: {error}")
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
