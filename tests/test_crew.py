from __future__ import annotations

import json
from types import SimpleNamespace

from crew.agents import AgentReport, create_agents
from crew.crew import (
	ProviderSelection,
	_create_llm,
	_create_risk_validation_tool,
	_read_config,
	select_provider,
)
from crew.tasks import create_phase_one_tasks, create_phase_three_tasks, create_risk_task
from crewai import LLM
from crewai.tools import tool


def test_provider_falls_back_from_groq_to_ollama_cloud(monkeypatch):
	monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
	monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test-key")
	responses = {
		"https://api.groq.test/openai/v1/models": SimpleNamespace(ok=False),
		"https://ollama.test/api/tags": SimpleNamespace(
			ok=True,
			json=lambda: {"models": [{"name": "gpt-oss:20b"}]},
		),
	}
	monkeypatch.setattr(
		"crew.crew.requests.get",
		lambda url, **kwargs: responses[url],
	)
	settings = {
		"provider_order": ["groq", "ollama_cloud", "ollama_local"],
		"providers": {
			"groq": {"model": "openai/gpt-oss-20b", "base_url": "https://api.groq.test/openai/v1"},
			"ollama_cloud": {
				"model": "gpt-oss:20b",
				"base_url": "https://ollama.test/v1",
				"api_key_env": "OLLAMA_API_KEY",
			},
			"ollama_local": {"model": "qwen2.5:3b", "base_url": "http://localhost:11434"},
		},
	}

	selection = select_provider(settings)

	assert selection.name == "ollama_cloud"
	assert selection.model == "gpt-oss:20b"
	assert selection.api_key == "ollama-test-key"


def test_provider_reads_flat_student_config_and_groq_env_key(monkeypatch):
	monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
	monkeypatch.setattr("crew.crew._model_available", lambda *args: args[0] == "groq")
	settings = {
		"llm": {
			"provider_order": ["groq"],
			"groq": {
				"model": "openai/gpt-oss-20b",
				"base_url": "https://api.groq.test/openai/v1",
			},
		}
	}

	selection = select_provider(settings)

	assert selection.name == "groq"
	assert selection.api_key == "groq-test-key"


def test_student_configs_prefer_installed_local_free_model():
	for student in ("A", "B", "C"):
		llm_config = _read_config(student)["llm"]
		assert llm_config["provider_order"][0] == "ollama_local"
		assert llm_config["providers"]["ollama_local"]["model"] == "llama3.2:3b"
		assert llm_config["max_tokens"] == 1024


def test_native_groq_llm_uses_model_id_verbatim():
	selection = ProviderSelection(
		"groq",
		"openai/gpt-oss-20b",
		"https://api.groq.com/openai/v1",
		"test-only",
	)
	llm = _create_llm(selection, 0.1, 30)

	assert type(llm).__name__ == "OpenAICompletion"
	assert llm.model == "openai/gpt-oss-20b"
	assert llm.response_format is AgentReport


def test_ollama_cloud_adds_openai_compatible_api_path():
	selection = ProviderSelection(
		"ollama_cloud",
		"gpt-oss:20b",
		"https://ollama.com",
		"test-only",
	)
	llm = _create_llm(selection, 0.1, 30)

	assert llm.model == "gpt-oss:20b"
	assert str(llm.base_url) == "https://ollama.com/v1"


def test_llm_caps_completion_tokens():
	selection = ProviderSelection("groq", "openai/gpt-oss-20b", "https://api.groq.test/v1", "test-only")
	llm = _create_llm(selection, 0.1, 30, max_tokens=256)

	assert llm.max_tokens == 256


def test_phase_layout_parallelizes_signal_tasks_and_names_agent_files(tmp_path):
	llm = LLM(model="openai/gpt-oss-20b", api_key="test-only", base_url="https://example.test/v1")
	agents = create_agents(llm)
	phase_one = create_phase_one_tasks(agents, tmp_path, "13102026-12-30")
	phase_three = create_phase_three_tasks(agents, tmp_path, "13102026-12-30")

	assert [task.name for task in phase_one] == [
		"Data", "Trend", "Rotation", "Breakout", "Dip", "ML", "News", "PortfolioManager"
	]
	assert all(task.async_execution for task in phase_one[1:7])
	assert not phase_one[0].async_execution and not phase_one[-1].async_execution
	assert [task.name for task in phase_three] == ["Execution", "Reporter"]
	output_file = phase_one[1].output_file
	assert output_file is not None
	assert (tmp_path / "Trend13102026-12-30.md").name == output_file.split("\\")[-1]


def test_risk_task_guardrail_requires_rejection_on_failed_mcp_validation(tmp_path):
	llm = LLM(model="openai/gpt-oss-20b", api_key="test-only", base_url="https://example.test/v1")
	risk_agent = create_agents(llm)["RiskOfficer"]
	task = create_risk_task(risk_agent, tmp_path, "13102026-12-30", validation_passed=False)

	assert task.guardrail_max_retries == 1
	assert task.output_pydantic is AgentReport
	assert task.output_file is not None
	assert task.output_file.endswith("RiskOfficer13102026-12-30.md")
	assert "Call the validate_portfolio MCP tool" in task.description
	assert "target_weights_json" in task.description


def test_risk_officer_receives_the_mcp_validation_tool():
	@tool("validate_portfolio")
	def validate_portfolio(student: str, target_weights: dict[str, float], dry_run: bool = False) -> str:
		"""Validate a target portfolio for a student."""
		return "valid"

	llm = LLM(model="openai/gpt-oss-20b", api_key="test-only", base_url="https://example.test/v1")
	agents = create_agents(llm, risk_tools=[validate_portfolio])

	assert agents["RiskOfficer"].tools == [validate_portfolio]
	assert agents["Trend"].tools == []


def test_risk_validation_tool_uses_groq_compatible_schema_and_calls_mcp():
	calls = []

	class MCPValidator:
		def run(self, **arguments):
			calls.append(arguments)
			return {"valid": True}

	proxy = _create_risk_validation_tool(MCPValidator())
	schema = proxy.args_schema.model_json_schema()

	assert schema["properties"]["target_weights_json"]["type"] == "string"
	result = proxy.run(
		student="A",
		target_weights_json='{"AAPL": 1.0}',
		dry_run=True,
	)
	assert json.loads(result) == {"valid": True}
	assert calls == [{"student": "A", "target_weights": {"AAPL": 1.0}, "dry_run": True}]