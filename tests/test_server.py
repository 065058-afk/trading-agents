from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mcp_server import server


def test_registry_exposes_all_required_tools():
	expected = {
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
	assert expected <= set(server.mcp._tool_manager._tools)


def test_dry_run_history_refresh_never_calls_yahoo(monkeypatch):
	monkeypatch.setenv("MCP_DRY_RUN", "1")
	monkeypatch.setattr(
		server.market_data,
		"history_cache",
		lambda symbols: (_ for _ in ()).throw(AssertionError("network history accessed")),
	)

	result = server.refresh_history(["AAPL"])

	assert set(result["tickers"]) == {"AAPL", "SPY", "QQQ"}


def test_config_reads_in_memory_student_override(monkeypatch):
	override = {"student": "A", "high_beta": ["AAA"], "diversified": ["BBB"]}
	monkeypatch.setenv("MCP_STUDENT_CONFIG_JSON", json.dumps({"A": override}))

	assert server._config("A") == override


def test_account_snapshot_preserves_zero_available_quantity(monkeypatch):
	client = SimpleNamespace(
		get_account=lambda: SimpleNamespace(equity="1000", portfolio_value="1000"),
		get_all_positions=lambda: [
			SimpleNamespace(symbol="AAA", qty="5", qty_available="0", side="long")
		],
	)
	monkeypatch.setattr(server, "_account_client", lambda student: client)

	_, positions = server._account_snapshot("A")

	assert positions[0]["qty"] == 5.0
	assert positions[0]["qty_available"] == 0.0


def test_compute_indicators_returns_compact_finite_values(monkeypatch):
	monkeypatch.setenv("MCP_DRY_RUN", "1")
	result = server.compute_indicators(["AAPL"])

	assert set(result) == {"AAPL"}
	assert {"close", "ema20", "ema50", "atr14", "rsi14", "roc5", "roc20", "volume_ratio20"} <= set(result["AAPL"])
	assert all(
		value is None or np.isfinite(value)
		for key, value in result["AAPL"].items()
		if key != "as_of"
	)


def test_blend_is_long_only_capped_and_in_configured_share_range(monkeypatch):
	config = server._config("B")
	symbols = [*config["high_beta"], *config["diversified"]]
	scores = {name: {symbol: 0.5 for symbol in symbols} for name in config["signal_weights"]}
	result = server._blend_from_scores("B", symbols, scores, "risk_on", [])
	weights = result["target_weights"]

	assert 0.9 <= result["gross_exposure"] <= 1.0
	assert config["high_beta_share_range"][0] <= result["high_beta_share"] <= config["high_beta_share_range"][1]
	assert all(0.0 <= weight <= 1.0 for weight in weights.values())
	assert all(weights[symbol] <= config["max_position_pct_high_beta"] for symbol in config["high_beta"])
	assert all(weights[symbol] <= config["max_position_pct_diversified"] for symbol in config["diversified"])


def test_dry_blend_retrains_ml_without_persisting_a_model(monkeypatch):
	monkeypatch.setenv("MCP_DRY_RUN", "1")
	config = server._config("A")
	requested = [*config["high_beta"], *config["diversified"]]
	calls = []

	def fake_signal_ml(symbols, *, history, persist, walk_forward):
		calls.append((list(symbols), set(history), persist, walk_forward))
		return {symbol: 0.0 for symbol in symbols}

	monkeypatch.setattr(server.signal_engine, "signal_ml", fake_signal_ml)
	server.blend_scores("A", requested)

	assert calls == [(requested, set(requested), False, False)]


def test_blend_scores_reuses_component_signals_from_same_cycle(monkeypatch):
	monkeypatch.setenv("MCP_DRY_RUN", "1")
	monkeypatch.setattr(server, "_SIGNAL_SCORE_CACHE", {})
	config = server._config("A")
	requested = [*config["high_beta"], *config["diversified"]]
	calls = []

	def fake_signal_call(name, symbols, as_of=None):
		calls.append(name)
		return {symbol: 0.25 for symbol in symbols}

	monkeypatch.setattr(server, "_signal_call", fake_signal_call)
	for name in ("signal_trend", "signal_rotation", "signal_breakout", "signal_dip", "signal_ml"):
		getattr(server, name)(requested)
	monkeypatch.setattr(
		server.signal_engine,
		"signal_ml",
		lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ML was recomputed")),
	)

	result = server.blend_scores("A", requested, dry_run=True)

	assert calls == ["signal_trend", "signal_rotation", "signal_breakout", "signal_dip", "signal_ml"]
	assert all(score == 0.25 for score in result["signal_scores"]["ml"].values())


def test_earnings_guard_symbols_get_no_weight():
	config = server._config("A")
	symbols = [*config["high_beta"], *config["diversified"]]
	scores = {name: {symbol: 1.0 for symbol in symbols} for name in config["signal_weights"]}
	guarded = symbols[:2]
	result = server._blend_from_scores("A", symbols, scores, "risk_on", guarded)
	assert all(result["target_weights"][symbol] == 0.0 for symbol in guarded)
	assert result["high_beta_share"] == pytest.approx(0.45)
	assert result["gross_exposure"] == pytest.approx(1.0)


def test_validate_portfolio_rejects_bad_count_and_insufficient_buying_power(monkeypatch):
	config = server._config("A")
	symbols = [*config["high_beta"], *config["diversified"]]
	valid_weights = {symbol: 0.95 / len(symbols) for symbol in symbols}
	monkeypatch.setattr(server, "_positions_for", lambda student, dry_run: (server._demo_account(student)[0], []))

	bad_count = server.validate_portfolio(
		"A", {symbols[0]: 1.2}, dry_run=True, required_buying_power=200_000
	)
	assert bad_count["valid"] is False
	bad_count_rules = {failure["rule"] for failure in bad_count["failures"]}
	assert {
		"stock_count_10_to_15",
		"position_caps",
		"sum_of_weights",
		"gross_exposure_90_to_100_percent",
		"sufficient_buying_power",
	} <= bad_count_rules
	assert all({"rule", "value", "limit"} <= set(failure) for failure in bad_count["failures"])

	low_buying_power = server.validate_portfolio(
		"A", valid_weights, dry_run=True, required_buying_power=200_000
	)
	assert low_buying_power["valid"] is False
	assert "sufficient_buying_power" in {
		failure["rule"] for failure in low_buying_power["failures"]
	}


def test_validate_portfolio_allows_reduced_exposure_only_when_requested():
	config = server._config("A")
	symbols = [*config["high_beta"][:5], *config["diversified"][:5]]
	weights = {symbol: 0.05 for symbol in symbols}

	result = server.validate_portfolio("A", weights, dry_run=True)
	assert result["valid"] is False
	assert "gross_exposure_90_to_100_percent" in {
		failure["rule"] for failure in result["failures"]
	}

	result = server.validate_portfolio("A", weights, dry_run=True, allow_reduced_exposure=True)
	assert result["valid"] is True
	assert result["gross_exposure"] == pytest.approx(0.5)


def test_validate_portfolio_rejects_low_high_beta_count_and_position_cap(monkeypatch):
	config = server._config("A")
	high_beta = config["high_beta"]
	diversified = config["diversified"]
	monkeypatch.setattr(server, "_positions_for", lambda student, dry_run: (server._demo_account(student)[0], []))
	too_few_high_beta = {high_beta[0]: 0.095, **{symbol: 0.095 for symbol in diversified[:9]}}
	result = server.validate_portfolio("A", too_few_high_beta, dry_run=True)
	assert "high_beta_at_least_30_percent" in {
		failure["rule"] for failure in result["failures"]
	}

	over_cap = {high_beta[0]: 0.16}
	over_cap.update({symbol: 0.79 / 9.0 for symbol in [*high_beta[1:], *diversified[:5]]})
	result = server.validate_portfolio("A", over_cap, dry_run=True)
	assert "position_caps" in {failure["rule"] for failure in result["failures"]}


def test_validate_portfolio_cap_tolerance_is_one_e_minus_six(monkeypatch):
	config = server._config("A")
	symbols = [*config["high_beta"], *config["diversified"]]
	high_beta_symbol = config["high_beta"][0]
	high_beta_weight = config["max_position_pct_high_beta"] + 5e-7
	other_weight = (0.95 - high_beta_weight) / (len(symbols) - 1)
	weights = {symbol: other_weight for symbol in symbols}
	weights[high_beta_symbol] = high_beta_weight
	monkeypatch.setattr(server, "_positions_for", lambda student, dry_run: (server._demo_account(student)[0], []))

	result = server.validate_portfolio("A", weights, dry_run=True)

	assert result["valid"] is True
	assert result["checks"]["position_caps"] is True


def test_portfolio_diff_never_sells_more_than_held_or_opens_a_short(monkeypatch):
	positions = [
		{"symbol": "AAA", "qty": 2.0, "qty_available": 5.0, "side": "long"},
		{"symbol": "BBB", "qty": 3.0, "qty_available": 3.0, "side": "short"},
	]
	monkeypatch.setattr(
		server,
		"_positions_for",
		lambda student, dry_run: ({"portfolio_value": 1_000.0}, positions),
	)
	monkeypatch.setattr(
		server,
		"_market_frames",
		lambda symbols, dry_run: {
			symbol: pd.DataFrame({"close": [10.0]}) for symbol in symbols
		},
	)

	result = server.portfolio_diff("A", {"AAA": 0.0, "BBB": 0.0}, dry_run=True)

	assert result["orders"] == [
		{"symbol": "AAA", "side": "sell", "qty": 2.0},
		{"symbol": "BBB", "side": "buy", "qty": 3.0},
	]
	assert all(order["side"] != "sell" or order["qty"] <= 2.0 for order in result["orders"])


def test_check_exits_respects_held_and_available_quantities(monkeypatch):
	position = {
		"symbol": "AAA",
		"qty": 3.0,
		"qty_available": 2.0,
		"side": "long",
		"avg_entry_price": 10.0,
		"current_price": 10.0,
		"market_value": 30.0,
	}
	frame = pd.DataFrame(
		{"high": [100.0] * 20, "low": [9.0] * 20, "close": [10.0] * 20},
		index=pd.date_range("2026-01-01", periods=20),
	)
	monkeypatch.setattr(server, "_positions_for", lambda student, dry_run: ({}, [position]))
	monkeypatch.setattr(server, "_market_frames", lambda symbols, dry_run: {"AAA": frame})
	monkeypatch.setattr(server.ta, "atr", lambda high, low, close, length: pd.Series(1.0, index=close.index))

	result = server.check_exits("A", dry_run=True)

	assert result["orders"] == [
		{"symbol": "AAA", "side": "sell", "qty": 2.0, "reason": "atr_trailing_stop"}
	]
	position["qty_available"] = 0.0
	assert server.check_exits("A", dry_run=True)["orders"] == []


def test_place_orders_dry_run_never_requires_credentials_or_submits(monkeypatch):
	monkeypatch.delenv("ALPACA_KEY_A", raising=False)
	monkeypatch.delenv("ALPACA_SECRET_A", raising=False)
	monkeypatch.delenv("TRADING_ENABLED", raising=False)
	result = server.place_orders(
		"A",
		[{"symbol": "AAPL", "side": "buy", "qty": 1, "client_order_id": "cycle-A-20261013-buy-AAPL"}],
		dry_run=True,
	)

	assert result["submitted"] is False
	assert result["dry_run"] is True
	assert result["orders"][0]["client_order_id"] == "cycle-A-20261013-buy-AAPL"
	second = server.place_orders("A", [{"symbol": "AAPL", "side": "buy", "qty": 1}], dry_run=True)
	assert second["orders"][0]["client_order_id"] != result["orders"][0]["client_order_id"]


def test_place_orders_refuses_real_submission_without_explicit_enable(monkeypatch):
	monkeypatch.setenv("TRADING_ENABLED", "false")
	with pytest.raises(PermissionError, match="TRADING_ENABLED=true"):
		server.place_orders("A", [{"symbol": "AAPL", "side": "buy", "qty": 1}], dry_run=False)


def test_account_client_uses_student_keys_and_paper_mode(monkeypatch):
	from alpaca.trading import client as trading_client_module

	created = {}

	class FakeTradingClient:
		def __init__(self, key, secret, paper):
			created.update(key=key, secret=secret, paper=paper)

	monkeypatch.setenv("ALPACA_KEY_A", "key-A")
	monkeypatch.setenv("ALPACA_SECRET_A", "secret-A")
	monkeypatch.setenv("ALPACA_KEY_B", "key-B")
	monkeypatch.setenv("ALPACA_SECRET_B", "secret-B")
	monkeypatch.setattr(trading_client_module, "TradingClient", FakeTradingClient)

	server._account_client("B")

	assert created == {"key": "key-B", "secret": "secret-B", "paper": True}


def test_exit_state_ignores_unfilled_orders():
	orders = [
		SimpleNamespace(
			symbol="AAA",
			side="buy",
			filled_qty="10",
			filled_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
		),
		SimpleNamespace(
			symbol="AAA",
			side="buy",
			filled_qty="0",
			filled_at=None,
		),
		SimpleNamespace(
			symbol="AAA",
			side="sell",
			filled_qty="2",
			filled_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
		),
	]
	entries, partial_sales = server._entry_dates_and_partial_sells(orders, ["AAA"])

	assert entries["AAA"] == pd.Timestamp("2026-10-01", tz="UTC")
	assert partial_sales == {"AAA"}