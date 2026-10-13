from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from web import app as web_app


@pytest.fixture
def client():
	with TestClient(web_app.app) as test_client:
		yield test_client
	with web_app._sessions_lock:
		web_app._sessions.clear()


def bootstrap(client):
	response = client.get("/api/bootstrap")
	assert response.status_code == 200
	return response.json()["csrf_token"]


def test_credentials_are_only_in_server_session_memory(client):
	csrf = bootstrap(client)
	key = "test-session-key-never-returned"
	secret = "test-session-secret-never-returned"

	response = client.post(
		"/api/session/credentials",
		json={"student": "A", "key": key, "secret": secret},
		headers={"X-CSRF-Token": csrf},
	)

	assert response.status_code == 200
	assert key not in response.text and secret not in response.text
	assert key not in client.cookies.get(web_app.SESSION_COOKIE, "")
	assert secret not in client.cookies.get(web_app.SESSION_COOKIE, "")
	session = web_app._sessions[client.cookies.get(web_app.SESSION_COOKIE)]
	assert session.credentials == {"ALPACA_KEY_A": key, "ALPACA_SECRET_A": secret}

	cleared = client.delete("/api/session/credentials", headers={"X-CSRF-Token": csrf})
	assert cleared.status_code == 200
	assert session.credentials == {}


def test_run_uses_session_credentials_config_and_approval_callback(monkeypatch, client, tmp_path):
	csrf = bootstrap(client)
	client.post(
		"/api/session/credentials",
		json={"student": "A", "key": "session-key", "secret": "session-secret"},
		headers={"X-CSRF-Token": csrf},
	)
	orders = [
		{
			"symbol": "AAPL",
			"side": "buy",
			"qty": 2.0,
			"client_order_id": "cycle-A-20261013-buy-AAPL",
		}
	]
	captured = {}

	def fake_run_cycle(student, mode, **kwargs):
		captured.update(student=student, mode=mode, **kwargs)
		captured["credential_override"] = dict(kwargs["credential_override"])
		event_callback = kwargs["event_callback"]
		event_callback("orders_proposed", {"orders": orders, "risk_approved": True, "validation": {"valid": True}})
		decision = kwargs["approval_handler"]({"orders": orders, "mode": mode})
		execution = {
			"approved": decision["action"] != "reject",
			"proposed_orders": orders,
			"sell_orders": {"orders": []},
			"buy_orders": {"orders": decision.get("orders", orders)},
			"account": {"account": {"equity": 100_000}, "positions": []},
		}
		event_callback("execution_complete", execution)
		event_callback(
			"cycle_complete",
			{
				"run_dir": str(tmp_path),
				"provider": "ollama_local",
				"execution": execution,
				"ohlcv_rows": [],
				"target_weights": {"AAPL": 1.0},
			},
		)
		return 0

	monkeypatch.setattr(web_app, "run_cycle", fake_run_cycle)
	response = client.post(
		"/api/runs",
		json={
			"student": "A",
			"mode": "dry",
			"approval_mode": "manual",
			"force_clock": True,
			"config": {
				"high_beta": ["MU", "NVDA", "TSLA", "COIN", "UAL"],
				"diversified": ["AAPL", "JPM", "JNJ", "PG", "XOM"],
				"signal_weights": {"trend": 0.2, "rotation": 0.2, "breakout": 0.15, "dip": 0.15, "ml": 0.2, "news": 0.1},
				"provider": "ollama_local",
			},
		},
		headers={"X-CSRF-Token": csrf},
	)
	assert response.status_code == 200
	job_id = response.json()["id"]
	for _ in range(100):
		job_response = client.get(f"/api/runs/{job_id}")
		if job_response.json()["status"] == "awaiting_approval":
			break
		time.sleep(0.02)
	else:
		pytest.fail("Cycle did not reach manual approval")

	job = job_response.json()
	assert job["proposed_orders"] == orders
	assert "session-key" not in job_response.text
	assert "session-secret" not in job_response.text
	edited_order = {**orders[0], "qty": 1.0}
	approved = client.post(
		f"/api/runs/{job_id}/approval",
		json={"action": "edit", "orders": [edited_order]},
		headers={"X-CSRF-Token": csrf},
	)
	assert approved.status_code == 200

	for _ in range(100):
		job_response = client.get(f"/api/runs/{job_id}")
		if job_response.json()["status"] == "completed":
			break
		time.sleep(0.02)
	assert job_response.json()["status"] == "completed"
	completed = job_response.json()
	assert completed["proposed_orders"] or completed["execution"]["buy_orders"]["orders"]
	assert captured["student"] == "A"
	assert captured["mode"] == "dry"
	assert captured["credential_override"] == {"ALPACA_KEY_A": "session-key", "ALPACA_SECRET_A": "session-secret"}
	assert captured["config_override"]["signal_weights"]["trend"] == 0.2
	assert captured["config_override"]["llm"]["provider_order"][0] == "ollama_local"


def test_post_dry_cycle_poll_returns_previous_results_when_skipped(monkeypatch, client, tmp_path):
	csrf = bootstrap(client)
	fresh_account_snapshot = {
		"account": {"equity": 125_000.0, "portfolio_value": 125_000.0},
		"positions": [{"symbol": "NVDA", "qty": 3.0, "market_value": 900.0}],
	}
	log_root = tmp_path / "logs"
	prior_dir = log_root / "A" / "20261013_120733"
	prior_dir.mkdir(parents=True)
	proposal = [{"symbol": "AAPL", "side": "buy", "qty": 1.0, "client_order_id": "previous-buy-AAPL"}]
	previous_execution = {
		"approved": True,
		"validation": {"valid": True},
		"order_count": 1,
		"proposed_orders": proposal,
		"buy_orders": {"orders": proposal, "submitted": False},
		"sell_orders": {"orders": [], "submitted": False},
		"account": {"account": {"equity": 100_000}, "positions": []},
	}
	(prior_dir / "Execution13102026-12-07.md").write_text("# Previous execution\n", encoding="utf-8")
	(prior_dir / "web_summary.json").write_text(
		json.dumps({"status": "completed", "mode": "dry", "provider": "ollama_local", "execution": previous_execution, "proposed_orders": proposal, "target_weights": {"AAPL": 1.0}, "ohlcv_rows": [], "curve": [], "curve_reason": "Previous curve data unavailable."}),
		encoding="utf-8",
	)
	monkeypatch.setattr(web_app, "LOG_ROOT", log_root)
	assert web_app._latest_completed_snapshot("A") is not None

	def fake_run_cycle(student, mode, **kwargs):
		kwargs["event_callback"](
			"cycle_skipped",
			{
				"reason": "A run for student A is already recorded for 2026-10-13; skipping.",
				"account_snapshot": fresh_account_snapshot,
			},
		)
		return 0

	monkeypatch.setattr(web_app, "run_cycle", fake_run_cycle)
	config = {
		"high_beta": ["MU", "NVDA", "TSLA", "COIN", "UAL"],
		"diversified": ["AAPL", "JPM", "JNJ", "PG", "XOM"],
		"signal_weights": {"trend": 0.2, "rotation": 0.2, "breakout": 0.15, "dip": 0.15, "ml": 0.2, "news": 0.1},
		"provider": "ollama_local",
	}
	started = client.post(
		"/api/runs",
		json={"student": "A", "mode": "dry", "approval_mode": "auto", "force_clock": True, "config": config},
		headers={"X-CSRF-Token": csrf},
	)
	assert started.status_code == 200
	job_id = started.json()["id"]
	for _ in range(100):
		response = client.get(f"/api/runs/{job_id}")
		job = response.json()
		if job["status"] in {"completed", "skipped", "failed"}:
			break
		time.sleep(0.02)

	assert response.status_code == 200
	assert job["status"] == "skipped"
	assert "already recorded" in job["reason"]
	assert job["previous_run"] is True, {
		"status": job["status"],
		"stage": job["stage"],
		"reason": job["reason"],
		"logs": job["logs"],
		"orders": job["proposed_orders"],
	}
	assert job["proposed_orders"]
	assert job["execution"]["order_count"] == 1
	assert job["execution"]["account"] == fresh_account_snapshot
	assert job["positions"] == fresh_account_snapshot["positions"]
	assert job["provider"] == "ollama_local"
	assert job["logs"] == ["Execution13102026-12-07.md"]
	assert "Previous completed run" in job["stage"]


def test_order_results_heading_names_dry_and_live_modes(client):
	html = client.get("/").text

	assert "Proposed orders (dry run)" in html
	assert "Executed orders" in html
	assert "updateOrderResultsHeading(job.execution?.mode||job.mode)" in html


def test_failed_run_exposes_exception_type_and_message_without_secret(monkeypatch, client):
	csrf = bootstrap(client)
	monkeypatch.setattr(web_app, "LOG_ROOT", Path("logs"))

	def fake_run_cycle(student, mode, **kwargs):
		kwargs["event_callback"]("cycle_failed", {"reason": "ValueError: safe diagnostic"})
		return 1

	monkeypatch.setattr(web_app, "run_cycle", fake_run_cycle)
	config = {
		"high_beta": ["MU", "NVDA", "TSLA", "COIN", "UAL"],
		"diversified": ["AAPL", "JPM", "JNJ", "PG", "XOM"],
		"signal_weights": {"trend": 0.2, "rotation": 0.2, "breakout": 0.15, "dip": 0.15, "ml": 0.2, "news": 0.1},
		"provider": "ollama_local",
	}
	started = client.post(
		"/api/runs",
		json={"student": "A", "mode": "dry", "approval_mode": "auto", "force_clock": True, "config": config},
		headers={"X-CSRF-Token": csrf},
	)
	job_id = started.json()["id"]
	for _ in range(100):
		job = client.get(f"/api/runs/{job_id}").json()
		if job["status"] == "failed":
			break
		time.sleep(0.02)
	assert job["status"] == "failed"
	assert job["reason"] == "ValueError: safe diagnostic"


def test_portfolio_curve_normalizes_proposal_and_spy():
	rows = [
		{"symbol": "AAA", "timestamp": "2026-01-01T00:00:00Z", "close": 10},
		{"symbol": "AAA", "timestamp": "2026-01-02T00:00:00Z", "close": 11},
		{"symbol": "SPY", "timestamp": "2026-01-01T00:00:00Z", "close": 100},
		{"symbol": "SPY", "timestamp": "2026-01-02T00:00:00Z", "close": 105},
	]

	curve = web_app._portfolio_curve(rows, {"AAA": 1.0})

	assert curve == [
		{"date": "2026-01-01", "portfolio": 100.0, "spy": 100.0},
		{"date": "2026-01-02", "portfolio": 110.0, "spy": 105.0},
	]


def test_snapshot_does_not_write_outside_student_log_root(monkeypatch, tmp_path):
	log_root = tmp_path / "logs"
	foreign_dir = tmp_path / "test-fixture-run"
	foreign_dir.mkdir()
	monkeypatch.setattr(web_app, "LOG_ROOT", log_root)
	job = web_app.RunJob("job", "A", "dry", "auto", {}, {}, run_dir=str(foreign_dir))

	web_app._store_completed_snapshot(
		job,
		{"execution": {"proposed_orders": []}, "target_weights": {}, "ohlcv_rows": []},
	)

	assert job.run_dir is None
	assert not (log_root / "A" / foreign_dir.name / "web_summary.json").exists()


def test_job_serialization_handles_jobs_created_before_latest_prices_field():
	job = SimpleNamespace(
		job_id="old-job",
		student="A",
		mode="dry",
		approval_mode="auto",
		status="completed",
		stage="Cycle complete",
		error=None,
		reason=None,
		previous_run=False,
		proposed_orders=[],
		execution=None,
		curve=[],
		curve_reason=None,
		provider=None,
		logs=[],
		updated_at=0.0,
		condition=threading.Condition(),
	)

	assert web_app._job_json(job)["latest_prices"] == {}
