from __future__ import annotations

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

	with pytest.raises(ValueError, match="stock_count_10_to_15"):
		server.validate_portfolio("A", {symbols[0]: 0.95}, dry_run=True)
	with pytest.raises(ValueError, match="insufficient buying power"):
		server.validate_portfolio("A", valid_weights, dry_run=True, required_buying_power=200_000)


def test_validate_portfolio_rejects_low_high_beta_count_and_position_cap(monkeypatch):
	config = server._config("A")
	high_beta = config["high_beta"]
	diversified = config["diversified"]
	monkeypatch.setattr(server, "_positions_for", lambda student, dry_run: (server._demo_account(student)[0], []))
	too_few_high_beta = {high_beta[0]: 0.095, **{symbol: 0.095 for symbol in diversified[:9]}}
	with pytest.raises(ValueError, match="high_beta_at_least_30_percent"):
		server.validate_portfolio("A", too_few_high_beta, dry_run=True)

	over_cap = {high_beta[0]: 0.16}
	over_cap.update({symbol: 0.79 / 9.0 for symbol in [*high_beta[1:], *diversified[:5]]})
	with pytest.raises(ValueError, match="position_caps"):
		server.validate_portfolio("A", over_cap, dry_run=True)


def test_place_orders_dry_run_never_requires_credentials_or_submits(monkeypatch):
	monkeypatch.delenv("ALPACA_KEY_A", raising=False)
	monkeypatch.delenv("ALPACA_SECRET_A", raising=False)
	monkeypatch.delenv("TRADING_ENABLED", raising=False)
	result = server.place_orders("A", [{"symbol": "AAPL", "side": "buy", "qty": 1}], dry_run=True)

	assert result["submitted"] is False
	assert result["dry_run"] is True
	assert result["orders"][0]["client_order_id"].startswith("202")
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