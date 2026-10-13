"""CrewAI task phases, structured-output retry, and agent artifact paths."""

from pathlib import Path
from typing import Any, Callable, Tuple

from crewai import Agent, Task

from crew.agents import AgentReport


def _guardrail(task_output: Any) -> Tuple[bool, Any]:
	try:
		parsed = task_output.pydantic
		if not isinstance(parsed, AgentReport):
			parsed = AgentReport.model_validate_json(task_output.raw)
		if not parsed.summary.strip():
			return False, "summary must be a non-empty string"
		return True, parsed.model_dump_json()
	except Exception:
		return False, "Return one valid JSON object matching AgentReport; do not include markdown fences."


def _make_task(
	name: str,
	agent: Agent,
	description: str,
	output_file: Path,
	*,
	async_execution: bool = False,
	context: list[Task] | None = None,
	guardrail: Callable[[Any], Tuple[bool, Any]] = _guardrail,
) -> Task:
	return Task(
		name=name,
		description=description,
		expected_output=(
			"A JSON object matching AgentReport with status, decision, concise summary, "
			"observations, and blockers. Include no unsupported numeric claims."
		),
		agent=agent,
		context=context,
		async_execution=async_execution,
		output_pydantic=AgentReport,
		guardrail=guardrail,
		guardrail_max_retries=0,
		output_file=str(output_file),
		create_directory=True,
	)


def create_phase_one_tasks(
	agents: dict[str, Agent], run_dir: Path, stamp: str
) -> list[Task]:
	return [
		_make_task(
			"News",
			agents["News"],
			"Summarize supplied headlines and news scores; flag event risk. Evidence: {news_evidence}",
			run_dir / f"News{stamp}.md",
		),
		_make_task(
			"PortfolioManager",
			agents["PortfolioManager"],
			"Summarize the final MCP target allocation and regime. Do not change the plan. Evidence: {portfolio_evidence}",
			run_dir / f"PortfolioManager{stamp}.md",
		),
	]


def create_risk_task(
	agent: Agent,
	run_dir: Path,
	stamp: str,
	validation_passed: bool,
	allow_reduced_exposure: bool = False,
) -> Task:
	def risk_guardrail(task_output: Any) -> Tuple[bool, Any]:
		passed, content = _guardrail(task_output)
		if not passed:
			return passed, content
		parsed = AgentReport.model_validate_json(str(content))
		if not validation_passed and parsed.decision != "reject":
			return False, "MCP portfolio validation failed; set decision to reject and identify the blocker."
		return True, parsed.model_dump_json()

	return _make_task(
		"RiskOfficer",
		agent,
		"Review the precomputed MCP validation in {risk_evidence}. Reject if it is invalid or any check failed; "
		"otherwise approve. Do not call tools or alter targets.",
		run_dir / f"RiskOfficer{stamp}.md",
		guardrail=risk_guardrail,
	)


def create_phase_three_tasks(
	agents: dict[str, Agent], run_dir: Path, stamp: str
) -> list[Task]:
	execution = _make_task(
		"Execution",
		agents["Execution"],
		"Report the MCP-generated orders and their dry-run/submission status. Do not add or change orders. "
		"Evidence: {execution_evidence}",
		run_dir / f"Execution{stamp}.md",
	)
	reporter = _make_task(
		"Reporter",
		agents["Reporter"],
		"Summarize this run, decisions, validation, and execution outcomes. All numbers must come from the supplied MCP results. "
		"Evidence: {report_evidence}",
		run_dir / f"Reporter{stamp}.md",
		context=[execution],
	)
	return [execution, reporter]