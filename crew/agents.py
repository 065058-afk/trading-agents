"""CrewAI role definitions and structured agent output model."""

from __future__ import annotations

from typing import Any, Literal

from crewai import Agent
from pydantic import BaseModel, Field


class AgentReport(BaseModel):
	"""One structured qualitative report; numeric evidence is supplied by MCP tools."""

	status: Literal["complete", "blocked", "warning"]
	decision: Literal["approve", "reject", "not_applicable"]
	summary: str
	observations: list[str] = Field(default_factory=list)
	blockers: list[str] = Field(default_factory=list)


AGENT_PROFILES = {
	"Data": (
		"Market Data Specialist",
		"Explain data coverage and highlight data quality warnings using only supplied MCP results.",
		"You prepare factual market-data briefs. You never estimate or invent prices, dates, or statistics.",
	),
	"Trend": (
		"Trend Analyst",
		"Interpret tool-produced trend scores qualitatively without recalculating them.",
		"You analyze trend outputs and cite no numbers not present in the supplied tool data.",
	),
	"Rotation": (
		"Sector and Relative-Strength Analyst",
		"Interpret tool-produced rotation scores qualitatively without recalculating them.",
		"You explain relative-strength context without inventing returns or rankings.",
	),
	"Breakout": (
		"Breakout Analyst",
		"Interpret tool-produced breakout scores and note whether setups are present.",
		"You only describe results supplied by the MCP signal tool.",
	),
	"Dip": (
		"Pullback Analyst",
		"Interpret tool-produced dip scores and note active pullback candidates.",
		"You only describe results supplied by the MCP signal tool.",
	),
	"ML": (
		"Machine Learning Analyst",
		"Interpret model scores and their signal direction from tool-produced data.",
		"You do not extrapolate model output or create new quantitative forecasts.",
	),
	"News": (
		"News Analyst",
		"Summarize supplied headlines and tool-produced news tilt, highlighting event risks.",
		"You cannot create sentiment scores; use only the supplied headline and MCP news-signal results.",
	),
	"PortfolioManager": (
		"Portfolio Manager",
		"Summarize the MCP-generated target portfolio and identify concentration or cash implications.",
		"Target weights are authoritative MCP output. Never modify or invent portfolio weights.",
	),
	"RiskOfficer": (
		"Risk Officer",
		"Review the proposed portfolio and MCP validation result; reject any failed validation.",
		"You must reject when any validation check fails. You cannot override buying-power, count, cap, or exposure failures.",
	),
	"Execution": (
		"Execution Specialist",
		"Report the tool-produced execution plan and whether orders were submitted or dry-run only.",
		"Never create or alter orders. Describe only the supplied diff, exit, and place-order results.",
	),
	"Reporter": (
		"Trading Run Reporter",
		"Produce a concise run summary with decisions, blockers, and next steps from supplied results.",
		"All numerical facts must be copied from MCP evidence; do not calculate or infer numbers.",
	),
}


def create_agents(llm, *, risk_tools: list[Any] | None = None) -> dict[str, Agent]:
	"""Build all project agents around the selected low-temperature structured LLM."""
	return {
		name: Agent(
			role=role,
			goal=goal,
			backstory=backstory,
			llm=llm,
			tools=risk_tools or [] if name == "RiskOfficer" else [],
			verbose=False,
			allow_delegation=False,
			max_iter=1,
		)
		for name, (role, goal, backstory) in AGENT_PROFILES.items()
	}