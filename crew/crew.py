"""Runnable CrewAI trading workflow backed by the local MCP stdio server."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import requests
import yaml
from crewai import Crew, LLM, Process
from crewai.tools import tool
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv
from mcp import StdioServerParameters

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


@dataclass(frozen=True)
class ProviderSelection:
	name: str
	model: str
	base_url: str
	api_key: str | None


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
	"""Select the first reachable, configured provider without exposing credentials."""
	settings = config.get("llm", config)
	provider_order = settings.get("provider_order", ["groq", "ollama_cloud", "ollama_local"])
	providers = settings.get("providers")
	if not isinstance(providers, Mapping):
		providers = {name: settings.get(name, {}) for name in provider_order}
	for name in provider_order:
		provider = providers.get(name, {})
		model = str(provider.get("model", "")).strip()
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
	selection: ProviderSelection, temperature: float, timeout: int, max_tokens: int = 384
) -> LLM:
	if selection.name in {"groq", "ollama_cloud"}:
		base_url = selection.base_url
		if selection.name == "ollama_cloud" and not base_url.rstrip("/").endswith("/v1"):
			base_url = base_url.rstrip("/") + "/v1"
		return LLM(
			model=f"openai/{selection.model}",
			api_key=selection.api_key,
			base_url=base_url,
			temperature=temperature,
			timeout=timeout,
			max_tokens=max_tokens,
			response_format=AgentReport,
		)
	return LLM(
		model=f"ollama_chat/{selection.model}",
		base_url=selection.base_url,
		temperature=temperature,
		timeout=timeout,
		max_tokens=max_tokens,
		response_format=AgentReport,
	)


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


def _dry_blend_scores(student: str, symbols: list[str]) -> dict[str, Any]:
	previous_dry_run = os.environ.get("MCP_DRY_RUN")
	os.environ["MCP_DRY_RUN"] = "1"
	try:
		from mcp_server.server import blend_scores

		return blend_scores(student, symbols)
	finally:
		if previous_dry_run is None:
			os.environ.pop("MCP_DRY_RUN", None)
		else:
			os.environ["MCP_DRY_RUN"] = previous_dry_run


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


def run_crew(
	student: str,
	mode: str,
	*,
	target_scale: float = 1.0,
	max_total_exposure: float = 1.0,
	allow_reduced_exposure: bool = False,
	execution_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
	student = student.upper()
	if mode not in {"dry", "paper"}:
		raise ValueError("mode must be dry or paper")
	if not 0.0 <= target_scale <= 1.0 or not 0.0 < max_total_exposure <= 1.0:
		raise ValueError("target_scale and max_total_exposure must be within their supported ranges")
	if mode == "paper" and os.environ.get("TRADING_ENABLED", "").strip().lower() != "true":
		raise PermissionError("Paper execution requires TRADING_ENABLED=true")
	config = _read_config(student)
	symbols = list(dict.fromkeys([*config["high_beta"], *config["diversified"]]))
	now = datetime.now()
	run_dir = LOG_ROOT / student / now.strftime("%Y%m%d_%H%M%S")
	run_dir.mkdir(parents=True, exist_ok=True)
	stamp = now.strftime("%d%m%Y-%H-%M")

	llm_config = config.get("llm", {})
	selection = select_provider(llm_config)
	llm = _create_llm(
		selection,
		float(llm_config.get("temperature", 0.1)),
		int(llm_config.get("timeout_seconds", 60)),
		int(llm_config.get("max_tokens", 384)),
	)
	(run_dir / "llm_provider.md").write_text(
		f"Provider used: {selection.name}\nModel: {selection.model}\n",
		encoding="utf-8",
	)
	print(f"LLM provider: {selection.name} ({selection.model})")

	server_env = os.environ.copy()
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
	with MCPServerAdapter(server_params) as tool_collection:
		tool_map = {tool.name: tool for tool in tool_collection}
		# The MCP tools are the only source for numeric strategy and account values.
		refresh = _call_tool(tool_map, "refresh_history", symbols=symbols)
		bars = _call_tool(tool_map, "get_ohlcv_batch", symbols=symbols, days=30, student=student)
		indicators = _call_tool(tool_map, "compute_indicators", symbols=symbols)
		headlines = _call_tool(tool_map, "news_fetch", symbols=symbols, student=student)
		blend = (
			_dry_blend_scores(student, symbols)
			if mode == "dry"
			else _call_tool(tool_map, "blend_scores", student=student, symbols=symbols)
		)
		if not isinstance(blend, dict) or not isinstance(blend.get("target_weights"), dict):
			raise RuntimeError("blend_scores did not return MCP target_weights")
		target_weights = _scale_target_weights(
			blend["target_weights"], target_scale, max_total_exposure
		)
		blend["target_weights"] = target_weights
		blend["gross_exposure"] = round(sum(target_weights.values()), 8)
		mode_dry = mode == "dry"
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

		validation_tool = tool_map.get("validate_portfolio")
		if validation_tool is None:
			raise RuntimeError("MCP server does not expose required tool validate_portfolio")
		agents = create_agents(
			llm,
			risk_tools=[_create_risk_validation_tool(validation_tool, allow_reduced_exposure)],
		)
		phase_one_tasks = create_phase_one_tasks(agents, run_dir, stamp)
		scores = blend.get("signal_scores", {})
		phase_one_inputs = {
			"data_evidence": _evidence_text({"refresh": refresh, "latest_bars": _latest_bars(bars), "indicators": indicators}),
			"trend_evidence": _evidence_text(scores.get("trend", {})),
			"rotation_evidence": _evidence_text(scores.get("rotation", {})),
			"breakout_evidence": _evidence_text(scores.get("breakout", {})),
			"dip_evidence": _evidence_text(scores.get("dip", {})),
			"ml_evidence": _evidence_text(scores.get("ml", {})),
			"news_evidence": _evidence_text({"headlines": headlines, "scores": scores.get("news", {})}),
			"portfolio_evidence": _evidence_text(blend),
		}
		phase_one_agents = [agents[name] for name in ("Data", "Trend", "Rotation", "Breakout", "Dip", "ML", "News", "PortfolioManager")]
		_crew_kickoff(phase_one_agents, phase_one_tasks, phase_one_inputs)
		phase_one_evidence = {
			"Data": {"refresh": refresh, "latest_bars": _latest_bars(bars), "indicators": indicators},
			"Trend": scores.get("trend", {}),
			"Rotation": scores.get("rotation", {}),
			"Breakout": scores.get("breakout", {}),
			"Dip": scores.get("dip", {}),
			"ML": scores.get("ml", {}),
			"News": {"headlines": headlines, "scores": scores.get("news", {})},
			"PortfolioManager": blend,
		}
		_persist_task_outputs(phase_one_tasks, phase_one_evidence, selection)

		risk_task = create_risk_task(
			agents["RiskOfficer"], run_dir, stamp, validation_passed, allow_reduced_exposure
		)
		risk_input = {
			"student": student,
			"target_weights_json": _evidence_text(target_weights),
			"dry_run": mode_dry,
			"allow_reduced_exposure": allow_reduced_exposure,
			"risk_evidence": _evidence_text({"validation": validation, "target_weights": target_weights}),
		}
		_crew_kickoff([agents["RiskOfficer"]], [risk_task], risk_input)
		risk_report = _agent_report(risk_task)
		risk_approved = validation_passed and risk_report.get("decision") == "approve"
		_persist_task_outputs([risk_task], {"RiskOfficer": {"validation": validation, "target_weights": target_weights}}, selection)

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

		phase_three_tasks = create_phase_three_tasks(agents, run_dir, stamp)
		phase_three_inputs = {
			"execution_evidence": _evidence_text(execution),
			"report_evidence": _evidence_text({"blend": blend, "validation": validation, "risk": risk_report, "execution": execution, "provider": selection.name}),
		}
		_crew_kickoff(
			[agents["Execution"], agents["Reporter"]],
			phase_three_tasks,
			phase_three_inputs,
		)
		_persist_task_outputs(
			phase_three_tasks,
			{"Execution": execution, "Reporter": {"validation": validation, "risk": risk_report, "execution": execution}},
			selection,
		)
	return {
		"student": student,
		"mode": mode,
		"provider": selection.name,
		"risk_approved": risk_approved,
		"run_dir": str(run_dir),
		"execution": execution,
		"validation": validation,
		"target_weights": target_weights,
		"latest_prices": _latest_prices(bars),
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