"""Runnable CrewAI trading workflow backed by the local MCP stdio server."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import requests
import yaml
from crewai import Crew, LLM, Process
from crewai.tools import tool
from crewai_tools.adapters.mcp_adapter import CrewAIToolAdapter
from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from pydantic import ValidationError

if __package__ in {None, ""}:
	ROOT = Path(__file__).resolve().parents[1]
	if str(ROOT) not in sys.path:
		sys.path.insert(0, str(ROOT))
else:
	ROOT = Path(__file__).resolve().parents[1]

from crew.agents import AgentReport, create_agents
from crew.tasks import create_phase_one_tasks, create_phase_three_tasks, create_risk_task

load_dotenv(ROOT / ".env")
CONFIG_DIR = ROOT / "config"
LOG_ROOT = ROOT / "logs"
SERVER_PATH = ROOT / "mcp_server" / "server.py"
LOGGER = logging.getLogger("trading_crew")
GROQ_TOKENS_PER_MINUTE = 8_000
LLM_MAX_TOKENS = 1_024
QWEN3_MAX_TOKENS = 2_048
MCP_TOOL_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ProviderSelection:
	name: str
	model: str
	base_url: str
	api_key: str | None


class TokenPerMinuteLimiter:
	"""Conservatively reserve prompt and completion tokens in a rolling minute."""

	def __init__(self, limit: int = GROQ_TOKENS_PER_MINUTE, state_path: Path | None = None):
		self.limit = limit
		self.state_path = state_path
		self._usage: deque[tuple[float, int]] = deque()
		self._blocked_until = 0.0
		self._lock = threading.Lock()

	def reserve(self, prompt: str, completion_tokens: int) -> int:
		reservation = math.ceil(len(prompt.encode("utf-8")) / 3) + completion_tokens
		if reservation > self.limit:
			raise ValueError("LLM prompt exceeds the per-minute token budget")
		if self.state_path is not None:
			return self._reserve_shared(reservation)
		while True:
			with self._lock:
				now = time.monotonic()
				while self._usage and self._usage[0][0] <= now - 60.0:
					self._usage.popleft()
				used = sum(tokens for _, tokens in self._usage)
				wait = max(0.0, self._blocked_until - now)
				if used + reservation > self.limit and self._usage:
					wait = max(wait, self._usage[0][0] + 60.0 - now)
				if wait <= 0.0:
					self._usage.append((now, reservation))
					return reservation
			time.sleep(max(wait, 0.01))

	def wait_for_retry_after(self, seconds: float) -> None:
		if self.state_path is not None:
			self._block_shared(seconds)
			return
		with self._lock:
			self._blocked_until = max(self._blocked_until, time.monotonic() + max(0.0, seconds))

	def _reserve_shared(self, reservation: int) -> int:
		assert self.state_path is not None
		self.state_path.parent.mkdir(parents=True, exist_ok=True)
		while True:
			connection = sqlite3.connect(self.state_path, timeout=60.0)
			try:
				self._initialize_shared(connection)
				connection.commit()
				connection.execute("BEGIN IMMEDIATE")
				now = time.time()
				connection.execute("DELETE FROM token_usage WHERE requested_at <= ?", (now - 60.0,))
				used = int(connection.execute("SELECT COALESCE(SUM(tokens), 0) FROM token_usage").fetchone()[0])
				oldest = connection.execute("SELECT MIN(requested_at) FROM token_usage").fetchone()[0]
				blocked_until = float(
					connection.execute("SELECT blocked_until FROM limiter_state WHERE id = 1").fetchone()[0]
				)
				wait = max(0.0, blocked_until - now)
				if used + reservation > self.limit and oldest is not None:
					wait = max(wait, float(oldest) + 60.0 - now)
				if wait <= 0.0:
					connection.execute(
						"INSERT INTO token_usage(requested_at, tokens) VALUES (?, ?)", (now, reservation)
					)
					connection.commit()
					return reservation
				connection.commit()
			finally:
				connection.close()
			time.sleep(max(wait, 0.01))

	def _block_shared(self, seconds: float) -> None:
		assert self.state_path is not None
		self.state_path.parent.mkdir(parents=True, exist_ok=True)
		connection = sqlite3.connect(self.state_path, timeout=60.0)
		try:
			self._initialize_shared(connection)
			connection.commit()
			connection.execute("BEGIN IMMEDIATE")
			blocked_until = time.time() + max(0.0, seconds)
			connection.execute(
				"UPDATE limiter_state SET blocked_until = MAX(blocked_until, ?) WHERE id = 1",
				(blocked_until,),
			)
			connection.commit()
		finally:
			connection.close()

	@staticmethod
	def _initialize_shared(connection: sqlite3.Connection) -> None:
		connection.execute(
			"CREATE TABLE IF NOT EXISTS token_usage (requested_at REAL NOT NULL, tokens INTEGER NOT NULL)"
		)
		connection.execute(
			"CREATE TABLE IF NOT EXISTS limiter_state (id INTEGER PRIMARY KEY CHECK (id = 1), blocked_until REAL NOT NULL)"
		)
		connection.execute("INSERT OR IGNORE INTO limiter_state(id, blocked_until) VALUES (1, 0)")


def _shared_limiter_path(api_key: str | None) -> Path:
	fingerprint = hashlib.sha256((api_key or "no-key").encode("utf-8")).hexdigest()[:24]
	return Path(tempfile.gettempdir()) / f"trading-agents-groq-tpm-{fingerprint}.sqlite3"


def _read_config(student: str) -> dict[str, Any]:
	student = student.upper()
	if student not in {"A", "B", "C"}:
		raise ValueError("student must be A, B, or C")
	with (CONFIG_DIR / f"student_{student}.yaml").open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	if not isinstance(config, dict):
		raise ValueError(f"Invalid config/student_{student}.yaml")
	return config


def _model_available(provider: str, model: str, base_url: str, api_key: str | None) -> bool:
	if provider == "groq":
		url = base_url.rstrip("/") + "/models"
		headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
	else:
		probe_base = base_url.rstrip("/")
		if probe_base.endswith("/v1"):
			probe_base = probe_base[:-3]
		url = probe_base + "/api/tags"
		headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
	try:
		response = requests.get(url, headers=headers, timeout=4)
		if not response.ok:
			return False
		payload = response.json()
		if provider == "groq":
			available = {str(item.get("id", "")) for item in payload.get("data", [])}
		else:
			available = {
				str(item.get("name", item.get("model", "")))
				for item in payload.get("models", [])
			}
		return not available or model in available or any(item.split(":")[0] == model.split(":")[0] for item in available)
	except (requests.RequestException, ValueError, TypeError):
		return False


def select_provider(config: Mapping[str, Any]) -> ProviderSelection:
	"""Select the first reachable provider in the configured order without exposing credentials."""
	settings = config.get("llm", config)
	provider_order = list(settings.get("provider_order", ["groq", "ollama_cloud", "ollama_local"]))
	providers = settings.get("providers")
	if not isinstance(providers, Mapping):
		providers = {name: settings.get(name, {}) for name in provider_order}

	for name in provider_order:
		provider = providers.get(name, {})
		model = str(
			(os.environ.get("GROQ_MODEL", "").strip() or provider.get("model", ""))
			if name == "groq"
			else provider.get("model", "")
		).strip()
		base_url = str(provider.get("base_url", "")).strip()
		if not model or not base_url:
			continue
		key_env = provider.get("api_key_env") or {
			"groq": "GROQ_API_KEY",
			"ollama_cloud": "OLLAMA_API_KEY",
		}.get(name)
		api_key = os.environ.get(str(key_env)) if key_env else None
		if key_env and not api_key:
			continue
		if _model_available(name, model, base_url, api_key):
			return ProviderSelection(name, model, base_url, api_key)
	raise RuntimeError("No configured LLM provider is reachable (Groq, Ollama Cloud, or local Ollama)")


def _create_llm(
	selection: ProviderSelection,
	temperature: float,
	timeout: int,
	max_tokens: int = LLM_MAX_TOKENS,
	structured: bool = True,
) -> LLM:
	options: dict[str, Any] = {"response_format": AgentReport} if structured else {}
	if selection.name in {"groq", "ollama_cloud"}:
		base_url = selection.base_url
		if selection.name == "ollama_cloud" and not base_url.rstrip("/").endswith("/v1"):
			base_url = base_url.rstrip("/") + "/v1"
		if selection.name == "groq" and "gpt-oss" in selection.model.lower():
			options["reasoning_effort"] = "low"
		return LLM(
			model=f"openai/{selection.model}",
			api_key=selection.api_key,
			base_url=base_url,
			temperature=temperature,
			timeout=timeout,
			max_tokens=max_tokens,
			**options,
		)
	if selection.name == "ollama_local" and selection.model.lower().startswith("qwen3"):
		options["additional_params"] = {"extra_body": {"think": False}}
	return LLM(
		model=f"ollama_chat/{selection.model}",
		base_url=selection.base_url,
		temperature=temperature,
		timeout=timeout,
		max_tokens=max_tokens,
		**options,
	)


def _provider_candidates(
	primary: ProviderSelection | None, settings: Mapping[str, Any]
) -> list[ProviderSelection]:
	if primary is None:
		return []
	candidates = [primary]
	if primary.name == "groq":
		fallback_model = os.environ.get("GROQ_MODEL_FALLBACK", "").strip()
		if fallback_model and fallback_model != primary.model:
			candidates.append(replace(primary, model=fallback_model))
	providers = settings.get("providers")
	if not isinstance(providers, Mapping):
		providers = {name: settings.get(name, {}) for name in settings.get("provider_order", [])}
	cloud = providers.get("ollama_cloud", {})
	cloud_model = str(cloud.get("model", "")).strip()
	cloud_url = str(cloud.get("base_url", "")).strip()
	cloud_key_env = str(cloud.get("api_key_env") or "OLLAMA_API_KEY")
	cloud_key = os.environ.get(cloud_key_env)
	if cloud_model and cloud_url and cloud_key:
		cloud_selection = ProviderSelection("ollama_cloud", cloud_model, cloud_url, cloud_key)
		if all((item.name, item.model) != (cloud_selection.name, cloud_selection.model) for item in candidates):
			candidates.append(cloud_selection)
	return candidates


def _http_status(error: Exception) -> int | None:
	for value in (
		getattr(error, "status_code", None),
		getattr(error, "status", None),
		getattr(getattr(error, "response", None), "status_code", None),
	):
		try:
			if value is not None:
				return int(value)
		except (TypeError, ValueError):
			continue
	return None


def _retry_after_seconds(error: Exception) -> float | None:
	response = getattr(error, "response", None)
	headers = getattr(response, "headers", None) or getattr(error, "headers", None) or {}
	value = headers.get("retry-after") or headers.get("Retry-After")
	if value is None:
		return None
	try:
		return max(0.0, float(value))
	except (TypeError, ValueError):
		try:
			retry_at = parsedate_to_datetime(str(value))
			if retry_at.tzinfo is None:
				retry_at = retry_at.replace(tzinfo=timezone.utc)
			return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
		except (TypeError, ValueError, OverflowError):
			return None


def _short_llm_error(error: Exception, provider: str) -> str:
	status = _http_status(error)
	return f"{provider} HTTP {status}" if status is not None else f"{provider} {type(error).__name__}"


def _is_structured_output_error(error: Exception) -> bool:
	message = str(error).lower()
	return isinstance(error, ValidationError) or type(error).__name__ == "LengthFinishReasonError" or any(
		phrase in message
		for phrase in (
			"json_validate_failed",
			"failed to generate json",
			"max completion tokens reached before generating a valid document",
		)
	)


def _parse_agent_report(result: Any) -> dict[str, Any]:
	if isinstance(result, AgentReport):
		return result.model_dump(mode="json")
	parsed = getattr(result, "pydantic", None)
	if isinstance(parsed, AgentReport):
		return parsed.model_dump(mode="json")
	if isinstance(result, Mapping):
		return AgentReport.model_validate(result).model_dump(mode="json")
	raw = getattr(result, "raw", result)
	if isinstance(raw, str):
		return AgentReport.model_validate_json(raw).model_dump(mode="json")
	raise ValueError("LLM returned no structured AgentReport")


def _call_agent_report(
	selection: ProviderSelection | None,
	settings: Mapping[str, Any],
	prompt: str,
	limiter: TokenPerMinuteLimiter,
	llm_cache: dict[tuple[str, str], LLM],
) -> tuple[dict[str, Any] | None, str | None, str | None]:
	candidates = _provider_candidates(selection, settings)
	if not candidates:
		return None, "No configured LLM provider is reachable", None
	last_reason = "No configured LLM provider is reachable"
	for candidate_index, candidate in enumerate(candidates):
		key = (candidate.name, candidate.model)
		max_tokens = int(settings.get("max_tokens", LLM_MAX_TOKENS))
		if candidate.name == "ollama_local" and candidate.model.lower().startswith("qwen3"):
			max_tokens = max(max_tokens, QWEN3_MAX_TOKENS)
		try:
			llm = llm_cache.get(key)
			if llm is None:
				llm = _create_llm(
					candidate,
					float(settings.get("temperature", 0.1)),
					int(settings.get("timeout_seconds", 30)),
					max_tokens,
				)
				llm_cache[key] = llm
		except Exception as error:
			last_reason = _short_llm_error(error, candidate.name)
			continue
		attempts = 2 if candidate_index == 0 else 1
		for attempt in range(attempts):
			try:
				limiter.reserve(prompt, max_tokens)
				result = llm.call(prompt, response_model=AgentReport)
				return _parse_agent_report(result), None, candidate.name
			except Exception as error:
				status = _http_status(error)
				last_reason = _short_llm_error(error, candidate.name)
				if _is_structured_output_error(error):
					try:
						limiter.reserve(prompt, max_tokens)
						plain_llm = _create_llm(
							candidate,
							float(settings.get("temperature", 0.1)),
							int(settings.get("timeout_seconds", 30)),
							max_tokens,
							False,
						)
						plain_text = plain_llm.call(prompt)
						return _parse_agent_report(plain_text), None, candidate.name
					except Exception as fallback_error:
						last_reason = _short_llm_error(fallback_error, candidate.name)
						if isinstance(fallback_error, ValidationError):
							warning_report = AgentReport(
								status="warning",
								decision="not_applicable",
								summary="LLM commentary did not match the required schema.",
								blockers=["Authoritative MCP results are retained; commentary was not used."],
							)
							return warning_report.model_dump(mode="json"), None, candidate.name
						break
				if attempt == 0 and attempts == 2 and status in {400, 429}:
					if status == 429:
						limiter.wait_for_retry_after(_retry_after_seconds(error) or 1.0)
					continue
				break
	return None, last_reason, None


def _decode_tool_result(value: Any) -> Any:
	if isinstance(value, str):
		try:
			return json.loads(value)
		except json.JSONDecodeError:
			return value
	if hasattr(value, "raw") and isinstance(value.raw, str):
		return _decode_tool_result(value.raw)
	if isinstance(value, dict) and set(value) == {"result"}:
		return _decode_tool_result(value["result"])
	return value


def _call_tool(tool_map: Mapping[str, Any], name: str, **arguments: Any) -> Any:
	tool = tool_map.get(name)
	if tool is None:
		raise RuntimeError(f"MCP server does not expose required tool {name}")
	return _decode_tool_result(tool.run(**arguments))



def _dry_blend_scores(
	student: str, symbols: list[str], config_override: Mapping[str, Any] | None = None
) -> dict[str, Any]:
	from mcp_server.server import blend_scores

	return blend_scores(
		student,
		symbols,
		dry_run=True,
		config_override=dict(config_override) if config_override is not None else None,
	)


def _create_risk_validation_tool(mcp_tool: Any, breaker_allows_reduced_exposure: bool = False) -> Any:
	@tool("validate_portfolio")
	def validate_portfolio(
		student: str,
		target_weights_json: str,
		dry_run: bool,
		allow_reduced_exposure: bool = False,
	) -> str:
		"""Validate JSON-encoded target weights with the local MCP portfolio validator."""
		weights = json.loads(target_weights_json)
		if not isinstance(weights, dict):
			raise ValueError("target_weights_json must encode a JSON object")
		result = mcp_tool.run(
			student=student,
			target_weights=weights,
			dry_run=dry_run,
			allow_reduced_exposure=(breaker_allows_reduced_exposure and allow_reduced_exposure),
		)
		return json.dumps(_decode_tool_result(result), ensure_ascii=True, default=str)

	return validate_portfolio


def _latest_bars(payload: Any) -> dict[str, Any]:
	if not isinstance(payload, dict):
		return {}
	rows = payload.get("rows", [])
	latest: dict[str, dict[str, Any]] = {}
	for row in rows:
		if not isinstance(row, dict) or "symbol" not in row:
			continue
		symbol = str(row["symbol"]).upper()
		if symbol not in latest or str(row.get("timestamp", "")) > str(latest[symbol].get("timestamp", "")):
			latest[symbol] = row
	return latest


def _latest_prices(payload: Any) -> dict[str, float]:
	return {
		symbol: float(row["close"])
		for symbol, row in _latest_bars(payload).items()
		if row.get("close") is not None and float(row["close"]) > 0.0
	}


def _evidence_text(payload: Any) -> str:
	return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _agent_report(task: Any) -> dict[str, Any]:
	output = getattr(task, "output", None)
	if output is None:
		return {"status": "warning", "summary": "Task returned no output.", "decision": "not_applicable", "observations": [], "blockers": []}
	pydantic_output = getattr(output, "pydantic", None)
	if isinstance(pydantic_output, AgentReport):
		return pydantic_output.model_dump(mode="json")
	json_dict = getattr(output, "json_dict", None)
	if isinstance(json_dict, dict):
		return json_dict
	return {"raw": str(getattr(output, "raw", output))}


def _persist_task_outputs(tasks: list[Any], evidence: Mapping[str, Any], provider: ProviderSelection) -> None:
	for task in tasks:
		path = Path(task.output_file) if task.output_file else None
		if path is None:
			continue
		path.parent.mkdir(parents=True, exist_ok=True)
		name = task.name or "Agent"
		report = _agent_report(task)
		tool_data = evidence.get(name, {})
		path.write_text(
			"\n".join(
				[
					f"# {name}",
					"",
					f"LLM provider: {provider.name} ({provider.model})",
					"",
					"## Structured agent output",
					"```json",
					json.dumps(report, indent=2, ensure_ascii=True, default=str),
					"```",
					"",
					"## Authoritative MCP evidence",
					"```json",
					json.dumps(tool_data, indent=2, ensure_ascii=True, default=str),
					"```",
					"",
				],
			),
			encoding="utf-8",
		)


def _crew_kickoff(agents: list[Any], tasks: list[Any], inputs: dict[str, Any]) -> Any:
	crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False, memory=False)
	return crew.kickoff(inputs=inputs)


def _scale_target_weights(
	weights: Mapping[str, float], target_scale: float, max_total_exposure: float
) -> dict[str, float]:
	gross_exposure = sum(float(weight) for weight in weights.values())
	cap_scale = min(1.0, max_total_exposure / gross_exposure) if gross_exposure > 0.0 else 1.0
	combined_scale = min(target_scale, cap_scale)
	return {symbol: float(weight) * combined_scale for symbol, weight in weights.items()}


def _compact_json(payload: Any) -> str:
	return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)


def _score_summary(scores: Any) -> dict[str, list[dict[str, Any]]]:
	if not isinstance(scores, Mapping):
		return {"top": [], "bottom": []}
	rows = []
	for symbol, raw_score in scores.items():
		try:
			score = float(raw_score)
			if math.isfinite(score):
				rows.append({"symbol": str(symbol).upper(), "score": round(score, 6)})
		except (TypeError, ValueError):
			continue
	rows.sort(key=lambda row: (-row["score"], row["symbol"]))
	return {"top": rows[:5], "bottom": sorted(rows, key=lambda row: (row["score"], row["symbol"]))[:5]}


def _compact_headlines(headlines: Any) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	if not isinstance(headlines, Mapping):
		return rows
	for symbol in sorted(headlines):
		items = headlines[symbol]
		if isinstance(items, list):
			for headline in items[:1]:
				rows.append({"symbol": str(symbol).upper(), "headline": str(headline)[:240]})
				if len(rows) == 5:
					return rows
	return rows


def _target_rows(target_weights: Mapping[str, Any]) -> list[dict[str, Any]]:
	return [
		{"symbol": symbol, "weight": round(float(weight), 6)}
		for symbol, weight in sorted(target_weights.items(), key=lambda item: (-float(item[1]), str(item[0])))
		if float(weight) > 0.0
	]


def _validation_summary(validation: Any) -> dict[str, Any]:
	if not isinstance(validation, Mapping):
		return {"valid": False}
	checks = validation.get("checks", {})
	failures = validation.get("failures", [])
	return {
		"valid": validation.get("valid") is True,
		"checks": dict(checks) if isinstance(checks, Mapping) else {},
		"failures": list(failures) if isinstance(failures, list) else [],
		"error": str(validation.get("error", ""))[:160] or None,
	}


def _order_rows(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
	orders = execution.get("proposed_orders")
	if not isinstance(orders, list):
		orders = []
		for key in ("sell_orders", "buy_orders"):
			result = execution.get(key, {})
			if isinstance(result, Mapping) and isinstance(result.get("orders"), list):
				orders.extend(result["orders"])
	return [
		{
			"symbol": str(order.get("symbol", "")).upper(),
			"side": str(order.get("side", "")).lower(),
			"qty": round(float(order.get("qty", 0.0)), 6),
		}
		for order in orders
		if isinstance(order, Mapping)
	]


def _write_commentary_report(
	run_dir: Path,
	stamp: str,
	name: str,
	provider: str,
	evidence: Mapping[str, Any],
	*,
	report: Mapping[str, Any] | None = None,
	unavailable: str | None = None,
) -> Path:
	lines = [f"# {name}", "", f"LLM provider: {provider}", ""]
	if report is None:
		lines.extend(["## Commentary", "", f"LLM commentary unavailable: {unavailable or 'provider unavailable'}", ""])
	else:
		lines.extend(["## Summary", "", str(report.get("summary", "")), ""])
		observations = report.get("observations", [])
		if isinstance(observations, list) and observations:
			lines.extend(["## Observations", *[f"- {str(item)}" for item in observations[:5]], ""])
		blockers = report.get("blockers", [])
		if isinstance(blockers, list) and blockers:
			lines.extend(["## Blockers", *[f"- {str(item)}" for item in blockers[:5]], ""])
	lines.extend(["## Tool summary", "", "```json", json.dumps(evidence, indent=2, ensure_ascii=True, default=str), "```", ""])
	path = run_dir / f"{name}{stamp}.md"
	path.write_text("\n".join(lines), encoding="utf-8")
	return path


def _write_tool_reports(
	run_dir: Path,
	stamp: str,
	refresh: Any,
	bars: Any,
	indicators: Any,
	scores: Mapping[str, Any],
) -> list[Path]:
	latest = _latest_bars(bars)
	refresh_status = refresh.get("status", "available") if isinstance(refresh, Mapping) else "available"
	bar_rows = [
		{"symbol": symbol, "timestamp": str(row.get("timestamp", "")), "close": row.get("close")}
		for symbol, row in sorted(latest.items())[:5]
	]
	indicator_count = len(indicators) if isinstance(indicators, Mapping) else 0
	data_path = run_dir / f"Data{stamp}.md"
	data_path.write_text(
		"\n".join(
			[
				"# Data",
				"",
				"Source: local MCP tool output; no LLM call.",
				f"- Refresh status: {refresh_status}",
				f"- Latest-bar symbols: {len(latest)}",
				f"- Indicator symbols: {indicator_count}",
				"- Latest-bar sample (up to five symbols):",
				*_markdown_rows(bar_rows, ("symbol", "timestamp", "close")),
				"",
			]
		),
		encoding="utf-8",
	)
	paths = [data_path]
	for agent_name, signal_name in (
		("Trend", "trend"),
		("Rotation", "rotation"),
		("Breakout", "breakout"),
		("Dip", "dip"),
		("ML", "ml"),
	):
		summary = _score_summary(scores.get(signal_name, {}))
		path = run_dir / f"{agent_name}{stamp}.md"
		path.write_text(
			"\n".join(
				[
					f"# {agent_name}",
					"",
					"Source: MCP signal tool output; no LLM call.",
					"## Top five scores",
					*_markdown_rows(summary["top"], ("symbol", "score")),
					"## Bottom five scores",
					*_markdown_rows(summary["bottom"], ("symbol", "score")),
					"",
				]
			),
			encoding="utf-8",
		)
		paths.append(path)
	return paths


def _markdown_rows(rows: list[Mapping[str, Any]], columns: tuple[str, ...]) -> list[str]:
	if not rows:
		return ["- No rows returned."]
	return [
		"| " + " | ".join(columns) + " |",
		"| " + " | ".join("---" for _ in columns) + " |",
		*["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows],
	]


def _zero_news_scores(
	student: str, symbols: list[str], blend: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
	from mcp_server.server import _blend_from_scores

	raw_scores = blend.get("signal_scores", {})
	score_map = {
		str(name): dict(values)
		for name, values in raw_scores.items()
		if isinstance(values, Mapping)
	} if isinstance(raw_scores, Mapping) else {}
	score_map["news"] = {symbol: 0.0 for symbol in symbols}
	return _blend_from_scores(
		student,
		symbols,
		score_map,
		str(blend.get("regime", "neutral")),
		blend.get("earnings_guard", []),
		config,
	)


def _append_unavailable(path: Path, reason: str) -> None:
	with path.open("a", encoding="utf-8") as report_file:
		report_file.write(f"\nLLM commentary unavailable: {reason}\n")


def _run_commentary(
	run_dir: Path,
	stamp: str,
	name: str,
	instruction: str,
	evidence: Mapping[str, Any],
	selection: ProviderSelection | None,
	settings: Mapping[str, Any],
	limiter: TokenPerMinuteLimiter,
	llm_cache: dict[tuple[str, str], LLM],
) -> tuple[dict[str, Any] | None, str | None, str]:
	prompt = (
		f"{instruction}\nReturn one JSON object: status, decision, summary, observations, blockers. "
		f"Keep summary under 50 words and observations/blockers to three each. Evidence: {_compact_json(evidence)}"
	)
	report, unavailable, provider = _call_agent_report(selection, settings, prompt, limiter, llm_cache)
	path = _write_commentary_report(
		run_dir,
		stamp,
		name,
		provider or (selection.name if selection else "unavailable"),
		evidence,
		report=report,
		unavailable=unavailable,
	)
	return report, unavailable, str(path)


def run_crew(
	student: str,
	mode: str,
	*,
	target_scale: float = 1.0,
	max_total_exposure: float = 1.0,
	allow_reduced_exposure: bool = False,
	execution_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
	config_override: Mapping[str, Any] | None = None,
	credential_override: Mapping[str, str] | None = None,
	cycle_event_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
	mcp_timeout_seconds: int = MCP_TOOL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
	student = student.upper()
	if mode not in {"dry", "paper"}:
		raise ValueError("mode must be dry or paper")
	if not 0.0 <= target_scale <= 1.0 or not 0.0 < max_total_exposure <= 1.0:
		raise ValueError("target_scale and max_total_exposure must be within their supported ranges")
	if mode == "paper" and os.environ.get("TRADING_ENABLED", "").strip().lower() != "true":
		raise PermissionError("Paper execution requires TRADING_ENABLED=true")
	config = dict(config_override) if config_override is not None else _read_config(student)
	symbols = list(dict.fromkeys([*config["high_beta"], *config["diversified"]]))
	now = datetime.now()
	run_dir = LOG_ROOT / student / now.strftime("%Y%m%d_%H%M%S")
	run_dir.mkdir(parents=True, exist_ok=True)
	stamp = now.strftime("%d%m%Y-%H-%M")

	llm_config = config.get("llm", {})
	try:
		selection: ProviderSelection | None = select_provider(llm_config)
	except RuntimeError:
		selection = None
	llm_cache: dict[tuple[str, str], LLM] = {}
	limiter_path = _shared_limiter_path(selection.api_key) if selection and selection.name == "groq" else None
	limiter = TokenPerMinuteLimiter(state_path=limiter_path)
	(run_dir / "llm_provider.md").write_text(
		f"Provider attempted: {selection.name if selection else 'unavailable'}\n"
		f"Model: {selection.model if selection else 'unavailable'}\n"
		f"Token budget: {GROQ_TOKENS_PER_MINUTE} per minute\n",
		encoding="utf-8",
	)
	if selection is not None:
		print(f"LLM provider: {selection.name} ({selection.model})")
	else:
		print("LLM provider unavailable; using deterministic tool reports.")

	server_env = os.environ.copy()
	if config_override is not None:
		server_env["MCP_STUDENT_CONFIG_JSON"] = json.dumps({student: config})
	if credential_override is not None:
		for credential_name in (f"ALPACA_KEY_{student}", f"ALPACA_SECRET_{student}"):
			credential_value = credential_override.get(credential_name)
			if credential_value:
				server_env[credential_name] = credential_value
	if mode == "dry":
		server_env["MCP_DRY_RUN"] = "1"
	else:
		server_env.pop("MCP_DRY_RUN", None)
	server_params = StdioServerParameters(
		command=sys.executable,
		args=[str(SERVER_PATH)],
		cwd=ROOT,
		env=server_env,
	)
	mcp_timeout_seconds = max(MCP_TOOL_TIMEOUT_SECONDS, int(mcp_timeout_seconds))
	with MCPAdapt(
		server_params,
		CrewAIToolAdapter(),
		connect_timeout=mcp_timeout_seconds,
		client_session_timeout_seconds=mcp_timeout_seconds,
	) as tool_collection:
		tool_map = {tool.name: tool for tool in tool_collection}
		if cycle_event_callback is not None:
			cycle_event_callback("stage", {"name": "Refreshing market data"})
		# The MCP tools are the only source for numeric strategy and account values.
		refresh = _call_tool(tool_map, "refresh_history", symbols=symbols)
		market_symbols = list(dict.fromkeys([*symbols, "SPY"]))
		bars = _call_tool(tool_map, "get_ohlcv_batch", symbols=market_symbols, days=30, student=student)
		indicators = _call_tool(tool_map, "compute_indicators", symbols=symbols)
		headlines = _call_tool(tool_map, "news_fetch", symbols=symbols, student=student)
		for signal_name in ("signal_trend", "signal_rotation", "signal_breakout", "signal_dip", "signal_ml"):
			_call_tool(tool_map, signal_name, symbols=symbols)
		blend = _call_tool(
			tool_map,
			"blend_scores",
			student=student,
			symbols=symbols,
			dry_run=(mode == "dry"),
		)
		if not isinstance(blend, dict) or not isinstance(blend.get("target_weights"), dict):
			raise RuntimeError("blend_scores did not return MCP target_weights")
		scores = blend.get("signal_scores", {})
		if not isinstance(scores, Mapping):
			scores = {}
		deterministic_paths = _write_tool_reports(run_dir, stamp, refresh, bars, indicators, scores)
		llm_failures: list[str] = []
		mode_dry = mode == "dry"
		if cycle_event_callback is not None:
			cycle_event_callback("stage", {"name": "Analyzing signals and news"})
		news_evidence = {
			"headlines": _compact_headlines(headlines),
			"news_scores": _score_summary(scores.get("news", {})),
		}
		news_report, news_error, news_path = _run_commentary(
			run_dir,
			stamp,
			"News",
			"Summarize the supplied headlines and news signal; flag material event risk without inventing facts.",
			news_evidence,
			selection,
			llm_config,
			limiter,
			llm_cache,
		)
		if news_error:
			llm_failures.append(news_error)
			blend = _zero_news_scores(student, symbols, blend, config)
			scores = blend.get("signal_scores", {})
			if not isinstance(scores, Mapping):
				scores = {}

		target_weights = _scale_target_weights(
			blend["target_weights"], target_scale, max_total_exposure
		)
		blend["target_weights"] = target_weights
		blend["gross_exposure"] = round(sum(target_weights.values()), 8)
		target_rows = _target_rows(target_weights)
		portfolio_evidence = {
			"regime": blend.get("regime"),
			"high_beta_share": blend.get("high_beta_share"),
			"gross_exposure": blend.get("gross_exposure"),
			"final_targets": target_rows,
		}
		portfolio_report, portfolio_error, portfolio_path = _run_commentary(
			run_dir,
			stamp,
			"PortfolioManager",
			"Summarize the final MCP targets and regime. Do not alter the allocation.",
			portfolio_evidence,
			selection,
			llm_config,
			limiter,
			llm_cache,
		)
		if portfolio_error:
			llm_failures.append(portfolio_error)

		try:
			validation = _call_tool(
				tool_map,
				"validate_portfolio",
				student=student,
				target_weights=target_weights,
				dry_run=mode_dry,
				allow_reduced_exposure=allow_reduced_exposure,
			)
		except Exception as error:
			validation = {"valid": False, "error": f"{type(error).__name__}: portfolio validation rejected the plan"}
		validation_passed = isinstance(validation, dict) and validation.get("valid") is True
		risk_evidence = {
			"validation": _validation_summary(validation),
			"final_targets": target_rows,
		}
		if cycle_event_callback is not None:
			cycle_event_callback("stage", {"name": "Risk review"})
		risk_report, risk_error, risk_path = _run_commentary(
			run_dir,
			stamp,
			"RiskOfficer",
			"Review MCP validation. Approve only when valid is true; reject if any validation check failed.",
			risk_evidence,
			selection,
			llm_config,
			limiter,
			llm_cache,
		)
		if risk_error:
			llm_failures.append(risk_error)
			risk_report = {
				"status": "warning",
				"decision": "approve" if validation_passed else "reject",
				"summary": "Deterministic MCP validation used because LLM commentary is unavailable.",
				"observations": [],
				"blockers": [] if validation_passed else ["MCP portfolio validation failed."],
			}
		risk_approved = validation_passed and risk_report.get("decision") == "approve"

		execution: dict[str, Any]
		if execution_handler is not None:
			execution = execution_handler(
				{
					"tool_map": tool_map,
					"student": student,
					"target_weights": target_weights,
					"validation": validation,
					"risk_approved": risk_approved,
					"mode": mode,
					"dry_run": mode_dry,
					"latest_prices": _latest_prices(bars),
				}
			)
		elif not risk_approved:
			execution = {"approved": False, "submitted": False, "reason": "RiskOfficer rejected or MCP validation failed", "orders": []}
		else:
			portfolio_diff = _call_tool(
				tool_map,
				"portfolio_diff",
				student=student,
				target_weights=target_weights,
				dry_run=mode_dry,
			)
			exits = _call_tool(tool_map, "check_exits", student=student, dry_run=mode_dry)
			exit_orders = exits.get("orders", []) if isinstance(exits, dict) else []
			exit_symbols = {str(order.get("symbol", "")).upper() for order in exit_orders}
			diff_orders = portfolio_diff.get("orders", []) if isinstance(portfolio_diff, dict) else []
			orders = [
				order
				for order in diff_orders
				if str(order.get("symbol", "")).upper() not in exit_symbols
			]
			orders.extend(exit_orders)
			placed = _call_tool(
				tool_map,
				"place_orders",
				student=student,
				orders=orders,
				dry_run=mode_dry,
			)
			execution = {
				"approved": True,
				"validation": validation,
				"portfolio_diff": portfolio_diff,
				"exits": exits,
				"orders": placed,
				"mode": mode,
			}

		if cycle_event_callback is not None:
			cycle_event_callback("stage", {"name": "Writing run report"})
		order_rows = _order_rows(execution)
		execution_evidence = {
			"mode": execution.get("mode", mode),
			"approved": bool(execution.get("approved", risk_approved)),
			"orders": order_rows,
		}
		execution_report, execution_error, execution_path = _run_commentary(
			run_dir,
			stamp,
			"Execution",
			"Report the supplied order list and whether execution is dry or live; never alter orders.",
			execution_evidence,
			selection,
			llm_config,
			limiter,
			llm_cache,
		)
		if execution_error:
			llm_failures.append(execution_error)
		report_evidence = {
			"mode": mode,
			"provider": selection.name if selection else "unavailable",
			"validation": _validation_summary(validation),
			"risk_decision": risk_report.get("decision"),
			"final_targets": target_rows,
			"orders": order_rows,
		}
		reporter_report, reporter_error, reporter_path = _run_commentary(
			run_dir,
			stamp,
			"Reporter",
			"Summarize the validated targets and order outcomes using only this evidence.",
			report_evidence,
			selection,
			llm_config,
			limiter,
			llm_cache,
		)
		if reporter_error:
			llm_failures.append(reporter_error)
		if len(llm_failures) == 5:
			for path in deterministic_paths:
				_append_unavailable(path, llm_failures[0])
		provider_name = selection.name if selection else "unavailable"
	return {
		"student": student,
		"mode": mode,
		"provider": provider_name,
		"risk_approved": risk_approved,
		"run_dir": str(run_dir),
		"execution": execution,
		"validation": validation,
		"target_weights": target_weights,
		"latest_prices": _latest_prices(bars),
		"ohlcv_rows": bars.get("rows", []) if isinstance(bars, dict) else [],
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Run the student trading crew against the local MCP stdio server.")
	parser.add_argument("--student", required=True, choices=("A", "B", "C"))
	parser.add_argument("--mode", choices=("dry", "paper"), default="dry")
	args = parser.parse_args()
	result = run_crew(args.student, args.mode)
	print(
		f"Crew complete for student {args.student}: risk_approved={result['risk_approved']}, "
		f"mode={args.mode}, logs={result['run_dir']}"
	)


if __name__ == "__main__":
	main()