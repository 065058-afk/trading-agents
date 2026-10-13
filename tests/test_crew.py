from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from crew.agents import AgentReport, create_agents
from crew.crew import (
	ProviderSelection,
	TokenPerMinuteLimiter,
	_call_agent_report,
	_create_llm,
	_dry_blend_scores,
	_create_risk_validation_tool,
	_retry_after_seconds,
	_score_summary,
	_shared_limiter_path,
	_short_llm_error,
	_read_config,
	_scale_target_weights,
	select_provider,
	_zero_news_scores,
)
from crew.tasks import create_phase_one_tasks, create_phase_three_tasks, create_risk_task
from crewai import LLM
from crewai.tools import tool
from mcp_server import server as mcp_server


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


def test_provider_uses_groq_model_environment_override_first(monkeypatch):
	monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
	monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
	monkeypatch.setattr("crew.crew._model_available", lambda provider, *args: provider == "groq")
	settings = {
		"provider_order": ["ollama_local", "groq"],
		"providers": {
			"groq": {"model": "openai/gpt-oss-20b", "base_url": "https://api.groq.test/v1"},
			"ollama_local": {"model": "qwen3:0.6b", "base_url": "http://localhost:11434"},
		},
	}

	selection = select_provider(settings)

	assert selection.name == "groq"
	assert selection.model == "openai/gpt-oss-120b"


def test_provider_prefers_local_ollama_when_available_even_with_groq_key(monkeypatch):
	monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
	monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test-key")
	monkeypatch.setattr(
		"crew.crew._model_available",
		lambda provider, *args: provider == "ollama_local" or provider == "groq",
	)
	settings = {
		"provider_order": ["ollama_local", "groq"],
		"providers": {
			"groq": {"model": "openai/gpt-oss-20b", "base_url": "https://api.groq.test/v1"},
			"ollama_local": {"model": "qwen3:0.6b", "base_url": "http://localhost:11434"},
		},
	}

	selection = select_provider(settings)

	assert selection.name == "ollama_local"
	assert selection.model == "qwen3:0.6b"


def test_student_configs_prefer_installed_local_free_model():
	for student in ("A", "B", "C"):
		llm_config = _read_config(student)["llm"]
		assert llm_config["provider_order"][0] == "ollama_local"
		assert llm_config["providers"]["ollama_local"]["model"] == "qwen3:0.6b"
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


def test_local_qwen3_disables_thinking_to_avoid_json_truncation():
	selection = ProviderSelection("ollama_local", "qwen3:0.6b", "http://localhost:11434", None)

	llm = _create_llm(selection, 0.1, 30)

	assert llm.additional_params["extra_body"]["think"] is False


def test_llm_caps_completion_tokens():
	selection = ProviderSelection("groq", "openai/gpt-oss-20b", "https://api.groq.test/v1", "test-only")
	llm = _create_llm(selection, 0.1, 30, max_tokens=256)

	assert llm.max_tokens == 256


def test_groq_gpt_oss_uses_low_reasoning_effort():
	selection = ProviderSelection("groq", "openai/gpt-oss-20b", "https://api.groq.test/v1", "test-only")

	llm = _create_llm(selection, 0.1, 30)

	assert llm.reasoning_effort == "low"
	assert llm.max_tokens == 1024

	plain_llm = _create_llm(selection, 0.1, 30, structured=False)
	assert plain_llm.response_format is None


class MockHTTPError(Exception):
	def __init__(self, status_code, retry_after=None, message="mock provider failure"):
		super().__init__(message)
		self.status_code = status_code
		self.response = SimpleNamespace(
			status_code=status_code,
			headers={"retry-after": str(retry_after)} if retry_after is not None else {},
		)


class FakeLimiter:
	def __init__(self, *args, **kwargs):
		self.reservations = []
		self.waits = []

	def reserve(self, prompt, completion_tokens):
		self.reservations.append((prompt, completion_tokens))

	def wait_for_retry_after(self, seconds):
		self.waits.append(seconds)


class FakeLLM:
	def __init__(self, responses):
		self.responses = list(responses)
		self.calls = 0
		self.response_models = []

	def call(self, prompt, *, response_model=None):
		self.calls += 1
		self.response_models.append(response_model)
		response = self.responses.pop(0)
		if isinstance(response, Exception):
			raise response
		return response


def _fallback_settings():
	return {
		"temperature": 0.1,
		"timeout_seconds": 5,
		"provider_order": ["groq", "ollama_cloud"],
		"providers": {
			"groq": {"model": "openai/gpt-oss-20b", "base_url": "https://api.groq.test/v1"},
			"ollama_cloud": {
				"model": "gpt-oss:20b",
				"base_url": "https://ollama.test/v1",
				"api_key_env": "OLLAMA_API_KEY",
			},
		},
	}


def test_groq_429_retries_after_header_then_switches_to_fallback_model(monkeypatch):
	monkeypatch.setenv("GROQ_MODEL_FALLBACK", "openai/gpt-oss-120b")
	monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test-key")
	selection = ProviderSelection("groq", "openai/gpt-oss-20b", "https://api.groq.test/v1", "groq-test-key")
	models = {
		("groq", "openai/gpt-oss-20b"): FakeLLM([MockHTTPError(429, 7), MockHTTPError(429, 7)]),
		("groq", "openai/gpt-oss-120b"): FakeLLM([
			AgentReport(status="complete", decision="not_applicable", summary="Fallback used.")
		]),
	}
	monkeypatch.setattr(
		"crew.crew._create_llm",
		lambda candidate, *args: models[(candidate.name, candidate.model)],
	)
	limiter = FakeLimiter()

	report, unavailable, provider = _call_agent_report(
		selection, _fallback_settings(), "short prompt", limiter, {}
	)

	assert report["summary"] == "Fallback used."
	assert unavailable is None
	assert provider == "groq"
	assert models[("groq", "openai/gpt-oss-20b")].calls == 2
	assert models[("groq", "openai/gpt-oss-120b")].calls == 1
	assert limiter.waits == [7.0]
	assert len(limiter.reservations) == 3


def test_groq_400_retries_then_falls_back_to_ollama_cloud(monkeypatch):
	monkeypatch.setenv("GROQ_MODEL_FALLBACK", "openai/gpt-oss-120b")
	monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test-key")
	selection = ProviderSelection("groq", "openai/gpt-oss-20b", "https://api.groq.test/v1", "groq-test-key")
	models = {
		("groq", "openai/gpt-oss-20b"): FakeLLM([MockHTTPError(400), MockHTTPError(400)]),
		("groq", "openai/gpt-oss-120b"): FakeLLM([MockHTTPError(400)]),
		("ollama_cloud", "gpt-oss:20b"): FakeLLM([
			AgentReport(status="complete", decision="not_applicable", summary="Cloud fallback used.")
		]),
	}
	monkeypatch.setattr(
		"crew.crew._create_llm",
		lambda candidate, *args: models[(candidate.name, candidate.model)],
	)

	report, unavailable, provider = _call_agent_report(
		selection, _fallback_settings(), "short prompt", FakeLimiter(), {}
	)

	assert report["summary"] == "Cloud fallback used."
	assert unavailable is None
	assert provider == "ollama_cloud"
	assert [model.calls for model in models.values()] == [2, 1, 1]


def test_groq_schema_failures_retry_without_strict_response_model(monkeypatch):
	monkeypatch.setenv("GROQ_MODEL_FALLBACK", "openai/gpt-oss-120b")
	selection = ProviderSelection("groq", "openai/gpt-oss-20b", "https://api.groq.test/v1", "groq-test-key")
	provider_model = FakeLLM([
		MockHTTPError(400, message="json_validate_failed: failed to generate JSON"),
		'{"status":"complete","decision":"approve","summary":"Recovered via plain text","observations":[],"blockers":[]}',
	])
	monkeypatch.setattr(
		"crew.crew._create_llm",
		lambda candidate, *args: provider_model,
	)

	report, unavailable, provider = _call_agent_report(
		selection, _fallback_settings(), "short prompt", FakeLimiter(), {}
	)

	assert report["summary"] == "Recovered via plain text"
	assert unavailable is None
	assert provider == "groq"
	assert provider_model.calls == 2
	assert provider_model.response_models == [AgentReport, None]


def test_length_finish_error_retries_without_structured_output(monkeypatch):
	selection = ProviderSelection("ollama_local", "qwen3:0.6b", "http://localhost:11434", None)
	length_error = type("LengthFinishReasonError", (Exception,), {})("completion reached its token limit")
	provider_model = FakeLLM([
		length_error,
		'{"status":"complete","decision":"approve","summary":"Recovered from truncation","observations":[],"blockers":[]}',
	])
	monkeypatch.setattr("crew.crew._create_llm", lambda candidate, *args: provider_model)

	report, unavailable, provider = _call_agent_report(
		selection, _fallback_settings(), "short prompt", FakeLimiter(), {}
	)

	assert report["summary"] == "Recovered from truncation"
	assert unavailable is None
	assert provider == "ollama_local"
	assert provider_model.response_models == [AgentReport, None]


def test_invalid_plain_text_report_degrades_to_warning(monkeypatch):
	selection = ProviderSelection("ollama_local", "qwen3:0.6b", "http://localhost:11434", None)
	with pytest.raises(ValidationError) as validation_error:
		AgentReport.model_validate({})
	provider_model = FakeLLM([validation_error.value, "This is not a valid report."])
	monkeypatch.setattr("crew.crew._create_llm", lambda candidate, *args: provider_model)

	report, unavailable, provider = _call_agent_report(
		selection, _fallback_settings(), "short prompt", FakeLimiter(), {}
	)

	assert report["status"] == "warning"
	assert report["decision"] == "not_applicable"
	assert "MCP results are retained" in report["blockers"][0]
	assert unavailable is None
	assert provider == "ollama_local"


def test_agent_report_uses_configured_completion_token_limit(monkeypatch):
	selection = ProviderSelection("groq", "openai/gpt-oss-20b", "https://api.groq.test/v1", "groq-test-key")
	settings = _fallback_settings()
	settings["max_tokens"] = 768
	created_with = []

	def create_llm(candidate, temperature, timeout, max_tokens):
		created_with.append(max_tokens)
		return FakeLLM([
			AgentReport(status="complete", decision="approve", summary="Token cap respected.")
		])

	monkeypatch.setattr("crew.crew._create_llm", create_llm)
	limiter = FakeLimiter()

	report, unavailable, provider = _call_agent_report(
		selection, settings, "short prompt", limiter, {}
	)

	assert report["summary"] == "Token cap respected."
	assert unavailable is None
	assert provider == "groq"
	assert created_with == [768]
	assert limiter.reservations == [("short prompt", 768)]


def test_local_qwen3_uses_higher_completion_limit(monkeypatch):
	selection = ProviderSelection("ollama_local", "qwen3:0.6b", "http://localhost:11434", None)
	settings = _fallback_settings()
	settings["max_tokens"] = 1024
	created_with = []

	def create_llm(candidate, temperature, timeout, max_tokens):
		created_with.append(max_tokens)
		return FakeLLM([
			AgentReport(status="complete", decision="approve", summary="Qwen output fits.")
		])

	monkeypatch.setattr("crew.crew._create_llm", create_llm)
	limiter = FakeLimiter()

	_report, unavailable, provider = _call_agent_report(
		selection, settings, "short prompt", limiter, {}
	)

	assert unavailable is None
	assert provider == "ollama_local"
	assert created_with == [2048]
	assert limiter.reservations == [("short prompt", 2048)]


def test_retry_after_reads_numeric_header_and_short_error_hides_key():
	error = MockHTTPError(429, 4, "request failed with groq-secret")

	assert _retry_after_seconds(error) == 4.0
	assert _short_llm_error(error, "groq") == "groq HTTP 429"
	assert "groq-secret" not in _short_llm_error(error, "groq")


def test_token_limiter_waits_for_expired_minute_budget(monkeypatch):
	current_time = [0.0]
	sleeps = []
	monkeypatch.setattr("crew.crew.time.monotonic", lambda: current_time[0])

	def advance(seconds):
		sleeps.append(seconds)
		current_time[0] += seconds

	monkeypatch.setattr("crew.crew.time.sleep", advance)
	limiter = TokenPerMinuteLimiter(limit=4)

	limiter.reserve("", 4)
	limiter.reserve("", 4)

	assert sleeps == [60.0]


def test_shared_token_limiter_coordinates_separate_instances(monkeypatch, tmp_path):
	current_time = [1_000.0]
	sleeps = []
	monkeypatch.setattr("crew.crew.time.time", lambda: current_time[0])

	def advance(seconds):
		sleeps.append(seconds)
		current_time[0] += seconds

	monkeypatch.setattr("crew.crew.time.sleep", advance)
	state_path = tmp_path / "shared-tpm.sqlite3"
	first = TokenPerMinuteLimiter(limit=4, state_path=state_path)
	second = TokenPerMinuteLimiter(limit=4, state_path=state_path)

	first.reserve("", 4)
	second.reserve("", 4)

	assert sleeps == [60.0]
	assert "groq-secret" not in str(_shared_limiter_path("groq-secret"))


def test_score_summary_keeps_only_five_top_and_bottom_rows():
	summary = _score_summary({f"SYM{index}": index / 10 for index in range(20)})

	assert len(summary["top"]) == 5
	assert len(summary["bottom"]) == 5
	assert summary["top"][0] == {"symbol": "SYM19", "score": 1.9}
	assert summary["bottom"][0] == {"symbol": "SYM0", "score": 0.0}


def test_zero_news_scores_recomputes_targets_from_tool_scores():
	config = _read_config("A")
	symbols = [*config["high_beta"], *config["diversified"]]
	scores = {signal: {symbol: 0.5 for symbol in symbols} for signal in config["signal_weights"]}
	scores["news"] = {symbol: 1.0 for symbol in symbols}
	blend = {
		"signal_scores": scores,
		"regime": "risk_on",
		"earnings_guard": [],
	}

	result = _zero_news_scores("A", symbols, blend, config)

	assert set(result["signal_scores"]["news"].values()) == {0.0}
	assert 0.9 <= sum(result["target_weights"].values()) <= 1.0


def test_run_crew_survives_all_llm_calls_failing(monkeypatch, tmp_path, capsys):
	config = _read_config("A")
	symbols = [*config["high_beta"], *config["diversified"]]
	scores = {signal: {symbol: 0.5 for symbol in symbols} for signal in config["signal_weights"]}
	scores["news"] = {symbol: 1.0 for symbol in symbols}
	blend = mcp_server._blend_from_scores("A", symbols, scores, "risk_on", [], config)
	monkeypatch.setattr("crew.crew.LOG_ROOT", tmp_path / "logs")
	monkeypatch.setattr("crew.crew.select_provider", lambda settings: ProviderSelection(
		"groq", "openai/gpt-oss-20b", "https://api.groq.test/v1", "never-log-this-key"
	))
	monkeypatch.setenv("GROQ_API_KEY", "never-log-this-key")
	monkeypatch.delenv("GROQ_MODEL_FALLBACK", raising=False)
	monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
	llm = FakeLLM([MockHTTPError(429, 0, "never-log-this-key") for _ in range(10)])
	monkeypatch.setattr("crew.crew._create_llm", lambda *args: llm)
	monkeypatch.setattr("crew.crew.TokenPerMinuteLimiter", FakeLimiter)
	monkeypatch.setattr("crew.crew._dry_blend_scores", lambda *args: blend)
	calls = []

	class FakeMCPTool:
		def __init__(self, name):
			self.name = name

		def run(self, **arguments):
			calls.append((self.name, arguments))
			if self.name == "refresh_history":
				return {"status": "ok"}
			if self.name == "get_ohlcv_batch":
				return {"rows": [{"symbol": symbol, "timestamp": "2026-10-13", "close": 100.0} for symbol in arguments["symbols"]]}
			if self.name == "compute_indicators":
				return {symbol: {"close": 100.0} for symbol in arguments["symbols"]}
			if self.name == "news_fetch":
				return {symbol: [f"Sample headline for {symbol}"] for symbol in arguments["symbols"]}
			if self.name == "validate_portfolio":
				return {"valid": True, "checks": {"gross_exposure": True}}
			raise AssertionError(f"unexpected MCP tool: {self.name}")

	class FakeAdapter:
		def __init__(self, server_params):
			self.tools = [FakeMCPTool(name) for name in (
				"refresh_history", "get_ohlcv_batch", "compute_indicators", "news_fetch", "validate_portfolio"
			)]

		def __enter__(self):
			return self.tools

		def __exit__(self, *args):
			return False

	monkeypatch.setattr("crew.crew.MCPServerAdapter", FakeAdapter)
	execution_contexts = []
	order = {"symbol": "AAPL", "side": "buy", "qty": 1.0}

	def execution_handler(context):
		execution_contexts.append(context)
		return {
			"approved": context["risk_approved"],
			"mode": "dry",
			"proposed_orders": [order],
			"order_count": 1,
		}

	result = __import__("crew.crew", fromlist=["run_crew"]).run_crew(
		"A",
		"dry",
		config_override=config,
		execution_handler=execution_handler,
	)

	assert result["risk_approved"] is True
	assert "never-log-this-key" not in capsys.readouterr().out
	assert execution_contexts[0]["risk_approved"] is True
	assert calls.count(("validate_portfolio", {
		"student": "A",
		"target_weights": execution_contexts[0]["target_weights"],
		"dry_run": True,
		"allow_reduced_exposure": False,
	})) == 1
	assert set(blend["signal_scores"]["news"].values()) == {1.0}
	assert llm.calls == 10
	zero_news_blend = _zero_news_scores("A", symbols, blend, config)
	assert execution_contexts[0]["target_weights"] == pytest.approx(zero_news_blend["target_weights"])
	for agent_name in ("Data", "Trend", "Rotation", "Breakout", "Dip", "ML", "News", "PortfolioManager", "RiskOfficer", "Execution", "Reporter"):
		reports = list((tmp_path / "logs" / "A").rglob(f"{agent_name}*.md"))
		assert len(reports) == 1
		content = reports[0].read_text(encoding="utf-8")
		assert "LLM commentary unavailable:" in content
		assert "never-log-this-key" not in content


def test_target_weight_scaling_enforces_total_exposure_cap():
	weights = _scale_target_weights({"AAA": 0.7, "BBB": 0.5}, 1.0, 0.9)
	breaker_weights = _scale_target_weights(weights, 0.5, 0.9)

	assert sum(weights.values()) == pytest.approx(0.9)
	assert sum(breaker_weights.values()) == pytest.approx(0.45)


def test_dry_blend_uses_local_mcp_logic_with_dry_flag(monkeypatch):
	calls = []

	def fake_blend_scores(student, symbols, *, dry_run, config_override):
		calls.append((student, symbols, dry_run, config_override))
		return {"target_weights": {"AAA": 1.0}}

	monkeypatch.setattr("mcp_server.server.blend_scores", fake_blend_scores)
	monkeypatch.delenv("MCP_DRY_RUN", raising=False)

	result = _dry_blend_scores("A", ["AAA"])

	assert result == {"target_weights": {"AAA": 1.0}}
	assert calls == [("A", ["AAA"], True, None)]


def test_phase_layout_limits_llm_tasks_and_names_agent_files(tmp_path):
	llm = LLM(model="openai/gpt-oss-20b", api_key="test-only", base_url="https://example.test/v1")
	agents = create_agents(llm)
	phase_one = create_phase_one_tasks(agents, tmp_path, "13102026-12-30")
	phase_three = create_phase_three_tasks(agents, tmp_path, "13102026-12-30")

	assert [task.name for task in phase_one] == ["News", "PortfolioManager"]
	assert all(not task.async_execution for task in phase_one)
	assert all(task.guardrail_max_retries == 0 for task in [*phase_one, *phase_three])
	assert [task.name for task in phase_three] == ["Execution", "Reporter"]
	output_file = phase_one[0].output_file
	assert output_file is not None
	assert (tmp_path / "News13102026-12-30.md").name == output_file.split("\\")[-1]


def test_risk_task_guardrail_requires_rejection_on_failed_mcp_validation(tmp_path):
	llm = LLM(model="openai/gpt-oss-20b", api_key="test-only", base_url="https://example.test/v1")
	risk_agent = create_agents(llm)["RiskOfficer"]
	task = create_risk_task(risk_agent, tmp_path, "13102026-12-30", validation_passed=False)

	assert task.guardrail_max_retries == 0
	assert task.output_pydantic is AgentReport
	assert task.output_file is not None
	assert task.output_file.endswith("RiskOfficer13102026-12-30.md")
	assert "precomputed MCP validation" in task.description
	assert "Do not call tools" in task.description


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
	assert calls == [
		{
			"student": "A",
			"target_weights": {"AAPL": 1.0},
			"dry_run": True,
			"allow_reduced_exposure": False,
		}
	]