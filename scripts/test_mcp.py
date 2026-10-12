"""Start the trading MCP server in isolated dry-run mode and exercise every tool."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

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
		"signal_ml",
		"news_fetch",
		"blend_scores",
		"validate_portfolio",
		"portfolio_diff",
		"check_exits",
		"place_orders",
		"get_account",
	}
	async with stdio_client(server) as (read_stream, write_stream):
		async with ClientSession(read_stream, write_stream) as session:
			await session.initialize()
			listed = await session.list_tools()
			available = {tool.name for tool in listed.tools}
			missing = expected_tools - available
			if missing:
				raise RuntimeError(f"Server is missing tools: {', '.join(sorted(missing))}")
			print("Tools:", ", ".join(sorted(available)))

			async def call(name: str, arguments: dict[str, Any]) -> Any:
				print(f"Calling {name}...", flush=True)
				try:
					result = await asyncio.wait_for(
						session.call_tool(name, arguments=arguments), timeout=60
					)
				except BaseException as error:
					message = f"{type(error).__name__}: {error}"
					failures.append(f"{name}: {message}")
					print(f"FAILED {name}: {message}", flush=True)
					return None
				if result.isError:
					messages = [getattr(item, "text", "") for item in result.content]
					message = " ".join(text for text in messages if text) or "unspecified MCP error"
					failures.append(f"{name}: {message}")
					print(f"FAILED {name}: {message}", flush=True)
					return None
				payload = _payload(result)
				print(f"Passed {name}.", flush=True)
				return payload

			student_arg = {"student": student}
			await call("get_ohlcv_batch", {**student_arg, "symbols": symbols[:2], "days": 5})
			await call("refresh_history", {"symbols": symbols[:2]})
			await call("compute_indicators", {"symbols": symbols[:2]})
			for name in ("signal_trend", "signal_rotation", "signal_breakout", "signal_dip", "signal_ml"):
				await call(name, {"symbols": symbols[:2]})
			await call("news_fetch", {**student_arg, "symbols": symbols[:2]})
			blend = await call("blend_scores", {**student_arg, "symbols": symbols})
			validation_symbols = _validation_symbols(config, symbols)
			validation_blend = blend
			if validation_symbols != symbols:
				validation_blend = await call(
					"blend_scores", {**student_arg, "symbols": validation_symbols}
				)
			if not isinstance(validation_blend, dict) or "target_weights" not in validation_blend:
				raise RuntimeError("blend_scores did not return target_weights")
			target_weights = validation_blend["target_weights"]
			await call("validate_portfolio", {**student_arg, "target_weights": target_weights, "dry_run": True})
			await call("portfolio_diff", {**student_arg, "target_weights": target_weights, "dry_run": True})
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