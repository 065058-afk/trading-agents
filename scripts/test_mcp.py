"""Start the trading MCP server in isolated dry-run mode and exercise every tool."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
import uuid

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _payload(result: Any) -> Any:
	structured = getattr(result, "structuredContent", None)
	if structured is not None:
		return structured
	for item in result.content:
		text = getattr(item, "text", None)
		if text:
			try:
				return json.loads(text)
			except json.JSONDecodeError:
				return text
	return None


def _safe_error(message: str) -> str:
	for name, value in os.environ.items():
		if value and (
			name.startswith("ALPACA_KEY_")
			or name.startswith("ALPACA_SECRET_")
			or name in {"GROQ_API_KEY", "OLLAMA_API_KEY", "FINNHUB_API_KEY"}
		):
			message = message.replace(value, "[REDACTED]")
	return " ".join(message.split())


def _one_line_json(value: Any) -> str:
	return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


def _summary(name: str, payload: Any) -> str:
	if name == "blend_scores" and isinstance(payload, dict):
		targets = payload.get("target_weights", {})
		weights = list(targets.values()) if isinstance(targets, dict) else []
		try:
			weight_sum = sum(float(weight) for weight in weights)
		except (TypeError, ValueError):
			weight_sum = float("nan")
		return (
			f"targets={len(weights)}, weight_sum={weight_sum:.6f}, "
			f"high_beta_share={payload.get('high_beta_share', 'n/a')}"
		)
	if name == "validate_portfolio" and isinstance(payload, dict):
		failures = payload.get("failures", [])
		if not isinstance(failures, list):
			failures = []
		details = "; ".join(
			f"{item.get('rule', 'unknown')}: value={_one_line_json(item.get('value'))}, "
			f"limit={_one_line_json(item.get('limit'))}"
			for item in failures
			if isinstance(item, dict)
		) or "none"
		return f"valid={payload.get('valid', 'n/a')}, failures={details}"
	if isinstance(payload, dict):
		return f"returned object with {len(payload)} fields ({', '.join(sorted(map(str, payload.keys()))[:8])})"
	if isinstance(payload, list):
		return f"returned list with {len(payload)} items"
	return f"returned {type(payload).__name__}"


def _validation_symbols(config: dict[str, Any], symbols: list[str]) -> list[str]:
	high_beta = [str(symbol).upper() for symbol in config["high_beta"]]
	diversified = [str(symbol).upper() for symbol in config["diversified"]]
	if len(high_beta) / max(len(symbols), 1) >= 0.30:
		return symbols
	max_count = min(15, int(len(high_beta) / 0.30))
	diversified_count = max_count - len(high_beta)
	if len(high_beta) + diversified_count < 10 or diversified_count < 0:
		raise ValueError("Student configuration cannot form a valid 10-15 stock test basket")
	return [*high_beta, *diversified[:diversified_count]]


async def _run(student: str) -> None:
	with (ROOT / "config" / f"student_{student}.yaml").open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	symbols = list(dict.fromkeys([*config["high_beta"], *config["diversified"]]))
	environment = os.environ.copy()
	environment["MCP_DRY_RUN"] = "1"
	environment["TRADING_ENABLED"] = "false"
	for student_suffix in ("A", "B", "C"):
		environment[f"ALPACA_KEY_{student_suffix}"] = ""
		environment[f"ALPACA_SECRET_{student_suffix}"] = ""
	server = StdioServerParameters(
		command=sys.executable,
		args=[str(ROOT / "mcp_server" / "server.py")],
		cwd=ROOT,
		env=environment,
	)
	failures: list[str] = []
	expected_tools = {
		"get_ohlcv_batch",
		"refresh_history",
		"compute_indicators",
		"signal_trend",
		"signal_rotation",
		"signal_breakout",
		"signal_dip",
		"news_fetch",
		"start_signal_ml",
		"get_signal_ml_result",
		"start_signal_news_tilt",
		"get_signal_news_tilt_result",
		"signal_regime_state",
		"signal_earnings_guard",
		"blend_scores",
		"validate_portfolio",
		"portfolio_diff",
		"check_exits",
		"place_orders",
		"get_account",
	}
	async with stdio_client(server) as (read_stream, write_stream):
		async with ClientSession(
			read_stream,
			write_stream,
		) as session:
			await session.initialize()
			listed = await session.list_tools()
			available = {tool.name for tool in listed.tools}
			missing = expected_tools - available
			if missing:
				raise RuntimeError(f"Server is missing tools: {', '.join(sorted(missing))}")
			print("Tools:", ", ".join(sorted(available)))

			async def call(name: str, arguments: dict[str, Any]) -> Any:
				started = asyncio.get_running_loop().time()
				last_error = ""
				for attempt in (1, 2):
					try:
						result = await session.call_tool(name, arguments=arguments)
						if result.isError:
							messages = [getattr(item, "text", "") for item in result.content]
							last_error = _safe_error(" ".join(text for text in messages if text) or "unspecified MCP error")
						else:
							payload = _payload(result)
							elapsed = asyncio.get_running_loop().time() - started
							label = "WARNING retry succeeded" if attempt == 2 else "OK"
							print(f"{label} {name}: {_summary(name, payload)}; elapsed={elapsed:.2f}s", flush=True)
							return payload
					except Exception as error:
						last_error = _safe_error(f"{type(error).__name__}: {error}")
				if not last_error:
					last_error = "unspecified MCP error"
				elapsed = asyncio.get_running_loop().time() - started
				failures.append(f"{name}: {last_error}")
				print(f"FAILED {name} after retry: {last_error}; elapsed={elapsed:.2f}s", flush=True)
				raise RuntimeError(f"{name} failed after retry: {last_error}")

			student_arg = {"student": student}
			cycle_id = uuid.uuid4().hex
			await call("get_ohlcv_batch", {**student_arg, "symbols": symbols[:2], "days": 5})
			await call("refresh_history", {"symbols": symbols[:2]})
			await call("compute_indicators", {"symbols": symbols[:2]})
			await call("news_fetch", {**student_arg, "symbols": symbols, "cycle_id": cycle_id})
			for name in ("signal_trend", "signal_rotation", "signal_breakout", "signal_dip"):
				await call(name, {**student_arg, "symbols": symbols, "cycle_id": cycle_id})
			ml_job = await call("start_signal_ml", {**student_arg, "symbols": symbols, "cycle_id": cycle_id, "dry_run": True})
			news_job = await call("start_signal_news_tilt", {**student_arg, "symbols": symbols, "cycle_id": cycle_id, "dry_run": True})
			await call("signal_regime_state", {**student_arg, "symbols": symbols, "cycle_id": cycle_id})
			await call("signal_earnings_guard", {**student_arg, "symbols": symbols, "cycle_id": cycle_id})

			async def poll_job(job: Any, tool_name: str) -> Any:
				if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
					raise RuntimeError(f"{tool_name} did not return a job id")
				while True:
					result = await call(tool_name, {"job_id": job["job_id"]})
					if isinstance(result, dict) and result.get("status") == "completed":
						return result
					if isinstance(result, dict) and result.get("status") == "failed":
						raise RuntimeError(f"{tool_name} failed: {result.get('error', 'unknown error')}")
					if not isinstance(result, dict) or result.get("status") != "running":
						raise RuntimeError(f"{tool_name} returned an invalid job status")
					await asyncio.sleep(0.25)

			await poll_job(ml_job, "get_signal_ml_result")
			await poll_job(news_job, "get_signal_news_tilt_result")
			blend = await call("blend_scores", {**student_arg, "symbols": symbols, "cycle_id": cycle_id})
			if not isinstance(blend, dict) or not isinstance(blend.get("target_weights"), dict):
				raise RuntimeError("blend_scores returned no target_weights; refusing to continue")
			validation_symbols = _validation_symbols(config, symbols)
			validation_blend = blend
			if validation_symbols != symbols:
				validation_blend = await call(
					"blend_scores",
					{**student_arg, "symbols": validation_symbols, "cycle_id": cycle_id},
				)
			if isinstance(validation_blend, dict) and isinstance(validation_blend.get("target_weights"), dict):
				target_weights = validation_blend["target_weights"]
				await call("validate_portfolio", {**student_arg, "target_weights": target_weights, "dry_run": True})
				await call("portfolio_diff", {**student_arg, "target_weights": target_weights, "dry_run": True})
			else:
				print("WARNING skipping validation and portfolio_diff: blend_scores returned no target_weights", flush=True)
			await call("check_exits", {**student_arg, "dry_run": True})
			await call(
				"place_orders",
				{**student_arg, "orders": [{"symbol": symbols[0], "side": "buy", "qty": 1}], "dry_run": True},
			)
			await call("get_account", {**student_arg, "dry_run": True})
	if failures:
		raise RuntimeError("MCP tool failures:\n" + "\n".join(failures))
	print(f"Dry-run completed: all {len(expected_tools)} tools called for student {student}.")


def main() -> None:
	parser = argparse.ArgumentParser(description="Exercise every MCP tool using mock data and no account/network actions.")
	parser.add_argument("--student", required=True, choices=("A", "B", "C"))
	args = parser.parse_args()
	asyncio.run(_run(args.student))


if __name__ == "__main__":
	main()