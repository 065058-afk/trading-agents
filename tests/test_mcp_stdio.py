from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml
import mcp.client.stdio as stdio
from mcp import ClientSession, StdioServerParameters
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _payload(result: Any) -> Any:
	if result.isError:
		messages = [getattr(item, "text", "") for item in result.content]
		raise AssertionError("MCP tool error: " + " ".join(messages))
	if result.structuredContent is not None:
		return result.structuredContent
	for item in result.content:
		text = getattr(item, "text", "")
		if text:
			return json.loads(text)
	return None


@pytest.mark.parametrize("student", ("A", "B", "C"))
def test_blend_scores_stdio_is_fast_and_stdout_is_jsonrpc(monkeypatch, tmp_path, student):
	asyncio.run(_run_blend_scores_stdio_test(monkeypatch, tmp_path, student))


async def _run_blend_scores_stdio_test(monkeypatch, tmp_path, student):
	with (ROOT / "config" / f"student_{student}.yaml").open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	symbols = list(dict.fromkeys([*config["high_beta"], *config["diversified"]]))
	assert len(symbols) == 15
	cycle_id = uuid.uuid4().hex
	stdout_file = tmp_path / "server-stdout.log"
	stdout_file.write_text("", encoding="utf-8")

	original_text_stream = stdio.TextReceiveStream

	class TeeTextReceiveStream:
		def __init__(self, stream, *, encoding="utf-8", errors="strict"):
			self.stream = original_text_stream(stream, encoding=encoding, errors=errors)

		async def __aiter__(self):
			async for chunk in self.stream:
				with stdout_file.open("a", encoding="utf-8", newline="") as capture:
					capture.write(chunk)
				yield chunk

	monkeypatch.setattr(stdio, "TextReceiveStream", TeeTextReceiveStream)
	environment = os.environ.copy()
	environment["TRADING_ENABLED"] = "false"
	environment["MCP_DRY_RUN"] = "1"
	environment["MCP_CYCLE_CACHE_DIR"] = str(tmp_path / "cycle-cache")
	server = StdioServerParameters(
		command=sys.executable,
		args=[str(ROOT / "mcp_server" / "server.py")],
		cwd=ROOT,
		env=environment,
	)

	async with stdio.stdio_client(server) as (read_stream, write_stream):
		async with ClientSession(
			read_stream,
			write_stream,
		) as session:
			await session.initialize()

			async def call(name: str, **arguments: Any) -> Any:
				response = await session.call_tool(name, arguments=arguments)
				return _payload(response)

			await call("news_fetch", student=student, symbols=symbols, cycle_id=cycle_id)
			for name in ("signal_trend", "signal_rotation", "signal_breakout", "signal_dip"):
				await call(name, student=student, symbols=symbols, cycle_id=cycle_id)
			ml_job = await call(
				"start_signal_ml", student=student, symbols=symbols, cycle_id=cycle_id, dry_run=True
			)
			news_job = await call(
				"start_signal_news_tilt", student=student, symbols=symbols, cycle_id=cycle_id, dry_run=True
			)
			await call("signal_regime_state", student=student, symbols=symbols, cycle_id=cycle_id)
			await call("signal_earnings_guard", student=student, symbols=symbols, cycle_id=cycle_id)

			async def poll(job: dict[str, Any], tool_name: str) -> None:
				while True:
					result = await call(tool_name, job_id=job["job_id"])
					if result["status"] == "completed":
						return
					assert result["status"] == "running", result
					await asyncio.sleep(0.2)

			await poll(ml_job, "get_signal_ml_result")
			await poll(news_job, "get_signal_news_tilt_result")
			blend = await call("blend_scores", student=student, symbols=symbols, cycle_id=cycle_id)

	assert len(blend["target_weights"]) == 15
	for number, line in enumerate(stdout_file.read_text(encoding="utf-8").splitlines(), 1):
		message = json.loads(line)
		assert isinstance(message, dict) and message.get("jsonrpc") == "2.0", number
		assert {"method", "result", "error"} & message.keys(), number