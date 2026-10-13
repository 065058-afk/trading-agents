from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import run_cycle


class FakeClient:
	def __init__(self, *, is_open: bool = True, equity: float = 100_000.0):
		self.clock = SimpleNamespace(is_open=is_open)
		self.account = SimpleNamespace(equity=equity)
		self.account_calls = 0
		self.clock_calls = 0

	def get_clock(self):
		self.clock_calls += 1
		return self.clock

	def get_account(self):
		self.account_calls += 1
		return self.account


class FakeTool:
	def __init__(self, name, calls, *, exit_orders=None, diff_orders=None):
		self.name = name
		self.calls = calls
		self.exit_orders = exit_orders or []
		self.diff_orders = diff_orders or []

	def run(self, **arguments):
		self.calls.append((self.name, arguments))
		if self.name == "check_exits":
			return {"orders": self.exit_orders}
		if self.name == "portfolio_diff":
			return {"orders": self.diff_orders}
		if self.name == "place_orders":
			return {
				"submitted": not arguments["dry_run"],
				"dry_run": arguments["dry_run"],
				"orders": arguments["orders"],
			}
		raise AssertionError(f"unexpected tool {self.name}")


def install_cycle_fakes(monkeypatch, tmp_path, *, is_open=True, equity=100_000.0, diff_orders=None):
	client = FakeClient(is_open=is_open, equity=equity)
	calls = []
	tools = {
		"check_exits": FakeTool(
			"check_exits",
			calls,
			exit_orders=[{"symbol": "MSFT", "side": "sell", "qty": 1.0}],
		),
		"portfolio_diff": FakeTool("portfolio_diff", calls, diff_orders=diff_orders or [
			{"symbol": "AAPL", "side": "buy", "qty": 200.0},
			{"symbol": "MSFT", "side": "sell", "qty": 2.0},
		]),
		"place_orders": FakeTool("place_orders", calls),
	}
	crew_calls = []

	def fake_run_crew(student, mode, **kwargs):
		crew_calls.append((student, mode, kwargs))
		context = {
			"tool_map": tools,
			"student": student,
			"target_weights": {"AAPL": 0.5, "MSFT": 0.5},
			"validation": {"valid": True},
			"risk_approved": True,
			"mode": mode,
			"dry_run": mode == "dry",
			"latest_prices": {"AAPL": 100.0, "MSFT": 100.0},
		}
		execution = kwargs["execution_handler"](context)
		return {"execution": execution}

	monkeypatch.setattr(run_cycle, "LOG_ROOT", tmp_path / "logs")
	monkeypatch.setattr(run_cycle, "_today", lambda: date(2026, 10, 13).isoformat())
	monkeypatch.setattr(run_cycle, "_trading_client", lambda student: client)
	monkeypatch.setattr(run_cycle, "run_crew", fake_run_crew)
	return client, calls, crew_calls


def test_market_closed_exits_cleanly_without_running_crew(monkeypatch, tmp_path, capsys):
	_, _, crew_calls = install_cycle_fakes(monkeypatch, tmp_path, is_open=False)

	result = run_cycle.run_cycle("A", "dry")

	assert result == 0
	assert crew_calls == []
	assert "Market is closed" in capsys.readouterr().out


def test_force_clock_bypasses_clock_only_in_dry_mode(monkeypatch, tmp_path):
	client, _, crew_calls = install_cycle_fakes(monkeypatch, tmp_path, is_open=False)
	client_constructions = []
	monkeypatch.setattr(
		run_cycle,
		"_trading_client",
		lambda student: client_constructions.append(student) or client,
	)

	assert run_cycle.run_cycle("A", "dry", force_clock=True) == 0
	assert client_constructions == []
	assert client.clock_calls == 0
	assert client.account_calls == 0
	assert crew_calls
	assert run_cycle.run_cycle("A", "live", force_clock=True) == 1


def test_market_clock_api_failure_exits_nonzero_without_running_crew(monkeypatch, tmp_path, capsys):
	class BrokenClockClient(FakeClient):
		def get_clock(self):
			raise RuntimeError("clock unavailable")

	client = BrokenClockClient()
	crew_calls = []
	monkeypatch.setattr(run_cycle, "LOG_ROOT", tmp_path / "logs")
	monkeypatch.setattr(run_cycle, "_trading_client", lambda student: client)
	monkeypatch.setattr(run_cycle, "run_crew", lambda *args, **kwargs: crew_calls.append(args))

	result = run_cycle.run_cycle("A", "dry")

	assert result == 1
	assert crew_calls == []
	assert "clock unavailable" in capsys.readouterr().out


def test_trading_enabled_false_forces_live_request_to_dry(monkeypatch, tmp_path):
	monkeypatch.setenv("TRADING_ENABLED", "false")
	_, calls, crew_calls = install_cycle_fakes(monkeypatch, tmp_path)

	result = run_cycle.run_cycle("A", "live")

	assert result == 0
	assert crew_calls[0][1] == "dry"
	placements = [arguments for name, arguments in calls if name == "place_orders"]
	assert placements
	assert all(arguments["dry_run"] for arguments in placements)


def test_completed_run_for_date_is_skipped(monkeypatch, tmp_path):
	client, _, crew_calls = install_cycle_fakes(monkeypatch, tmp_path)
	state_path = tmp_path / "logs" / "A" / "state.json"
	state_path.parent.mkdir(parents=True)
	state_path.write_text(
		json.dumps({"peak_equity": 100_000.0, "runs": {"2026-10-13": {"status": "completed"}}}),
		encoding="utf-8",
	)

	result = run_cycle.run_cycle("A", "dry")

	assert result == 0
	assert client.account_calls == 0
	assert crew_calls == []


def test_drawdown_breaker_halves_targets_and_blocks_buys(monkeypatch, tmp_path):
	monkeypatch.setenv("TRADING_ENABLED", "true")
	_, calls, crew_calls = install_cycle_fakes(monkeypatch, tmp_path, equity=90_000.0)
	state_path = tmp_path / "logs" / "A" / "state.json"
	state_path.parent.mkdir(parents=True)
	state_path.write_text(json.dumps({"peak_equity": 100_000.0, "runs": {}}), encoding="utf-8")

	result = run_cycle.run_cycle("A", "live")

	assert result == 0
	kwargs = crew_calls[0][2]
	assert kwargs["target_scale"] == 0.5
	assert kwargs["allow_reduced_exposure"] is True
	placements = [arguments for name, arguments in calls if name == "place_orders"]
	assert [order["side"] for arguments in placements for order in arguments["orders"]] == ["sell"]
	updated_state = json.loads(state_path.read_text(encoding="utf-8"))
	assert updated_state["peak_equity"] == 100_000.0
	assert updated_state["runs"]["2026-10-13"]["drawdown_breaker"] is True


def test_order_caps_and_sell_first_submission(monkeypatch, tmp_path):
	monkeypatch.setenv("TRADING_ENABLED", "true")
	many_buys = [
		{"symbol": f"SYM{index}", "side": "buy", "qty": 500.0}
		for index in range(12)
	]
	client = FakeClient()
	calls = []
	tools = {
		"check_exits": FakeTool(
			"check_exits", calls, exit_orders=[{"symbol": "MSFT", "side": "sell", "qty": 500.0}]
		),
		"portfolio_diff": FakeTool("portfolio_diff", calls, diff_orders=many_buys),
		"place_orders": FakeTool("place_orders", calls),
	}

	def run_with_large_diff(student, mode, **kwargs):
		context = {
			"tool_map": tools,
			"target_weights": {},
			"validation": {"valid": True},
			"risk_approved": True,
			"latest_prices": {**{f"SYM{index}": 50.0 for index in range(12)}, "MSFT": 50.0},
		}
		return {"execution": kwargs["execution_handler"](context)}

	monkeypatch.setattr(run_cycle, "LOG_ROOT", tmp_path / "logs")
	monkeypatch.setattr(run_cycle, "_today", lambda: date(2026, 10, 13).isoformat())
	monkeypatch.setattr(run_cycle, "_trading_client", lambda student: client)
	monkeypatch.setattr(run_cycle, "run_crew", run_with_large_diff)

	result = run_cycle.run_cycle("A", "live")

	assert result == 0
	placements = [(name, args) for name, args in calls if name == "place_orders"]
	assert len(placements) == 2
	assert all(order["side"] == "sell" for order in placements[0][1]["orders"])
	assert len(placements[1][1]["orders"]) == 9
	assert all(order["side"] == "buy" for order in placements[1][1]["orders"])
	all_orders = [order for _, arguments in placements for order in arguments["orders"]]
	assert len(all_orders) == run_cycle.MAX_ORDERS_PER_DAY
	assert all(order["qty"] * 50.0 <= run_cycle.MAX_ORDER_NOTIONAL_USD for order in all_orders)
	assert len({order["client_order_id"] for order in all_orders}) == len(all_orders)
