from __future__ import annotations

import numpy as np
import pandas as pd

import backtest


def _frames(symbols, count=340):
	dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=count)
	frames = {}
	for offset, symbol in enumerate(symbols):
		rng = np.random.default_rng(1000 + offset)
		returns = rng.normal(0.0004, 0.01, count)
		close = 100.0 * np.exp(np.cumsum(returns))
		frames[symbol] = pd.DataFrame(
			{
				"open": close * 0.999,
				"high": close * 1.01,
				"low": close * 0.99,
				"close": close,
				"volume": rng.integers(100_000, 300_000, count),
			},
			index=dates,
		)
	return frames


def test_portfolio_weights_respect_group_and_position_limits():
	high_beta = ["A", "B", "C"]
	diversified = ["D", "E", "F", "G", "H", "I"]
	config = {
		"high_beta_share_base": 0.45,
		"max_position_pct_high_beta": 0.15,
		"max_position_pct_diversified": 0.10,
	}
	scores = {symbol: 1.0 for symbol in [*high_beta, *diversified]}
	weights = backtest._portfolio_weights(scores, high_beta, diversified, config, "risk_on")

	assert np.isclose(weights[:3].sum(), 0.45)
	assert np.isclose(weights[3:].sum(), 0.55)
	assert np.all(weights[:3] <= 0.15)
	assert np.all(weights[3:] <= 0.10)


def test_atr_trailing_stop_exits_at_stop_instead_of_later_close():
	dates = pd.bdate_range("2025-01-01", periods=5)
	arrays = {
		"open": np.array([[100.0], [100.0], [100.0], [95.0], [80.0]]),
		"high": np.array([[100.0], [100.0], [115.0], [100.0], [85.0]]),
		"low": np.array([[100.0], [100.0], [99.0], [94.0], [78.0]]),
		"close": np.array([[100.0], [100.0], [110.0], [80.0], [85.0]]),
		"volume": np.full((5, 1), 100_000.0),
	}
	atr = np.full((5, 1), 5.0)
	risk = {
		"atr_trailing_stop": 2.5,
		"partial_profit_atr": 100.0,
		"time_stop_days": 20,
		"drawdown_breaker_pct": 0.9,
	}
	targets = {1: np.array([1.0])}
	full, _, _ = backtest._simulate(dates, arrays, atr, targets, 1, 0.0, risk, 5, True)
	legacy, _, _ = backtest._simulate(dates, arrays, atr, targets, 1, 0.0, risk, 5, False)
	full_with_costs, _, turnover = backtest._simulate(
		dates, arrays, atr, targets, 1, 100.0, risk, 5, True
	)

	assert full.loc[dates[3]] > legacy.loc[dates[3]]
	assert np.isclose(full.loc[dates[3]], 0.95)
	assert turnover.loc[dates[3]] > 0
	assert full_with_costs.loc[dates[3]] < full.loc[dates[3]]


def test_drawdown_breaker_cools_down_then_allows_reentry():
	dates = pd.bdate_range("2025-01-01", periods=7)
	prices = np.array([100.0, 100.0, 90.0, 95.0, 100.0, 110.0, 120.0])[:, None]
	arrays = {
		"open": prices.copy(),
		"high": prices * 1.01,
		"low": prices * 0.99,
		"close": prices.copy(),
		"volume": np.full_like(prices, 100_000.0),
	}
	risk = {
		"atr_trailing_stop": 100.0,
		"partial_profit_atr": 100.0,
		"time_stop_days": 100,
		"drawdown_breaker_pct": 0.06,
	}
	targets = {0: np.array([1.0]), 2: np.array([1.0]), 4: np.array([1.0])}
	equity, _, turnover = backtest._simulate(
		dates, arrays, np.full_like(prices, 1.0), targets, 0, 0.0, risk, 2, True
	)

	assert turnover.loc[dates[2]] > 0
	assert turnover.loc[dates[4]] > 0
	assert equity.loc[dates[5]] > equity.loc[dates[4]]


def test_run_backtest_writes_curve_and_summary(monkeypatch, tmp_path):
	config_path = backtest.ROOT / "config" / "student_A.yaml"
	import yaml

	with config_path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	symbols = [*config["high_beta"], *config["diversified"], "SPY", "QQQ"]
	frames = _frames(symbols)
	monkeypatch.setattr(backtest, "history_cache", lambda requested: frames)

	def score(requested, *, as_of=None, history=None):
		return {symbol: 0.5 for symbol in requested}

	monkeypatch.setattr(backtest.signals, "signal_trend", score)
	monkeypatch.setattr(backtest.signals, "signal_rotation", score)
	monkeypatch.setattr(backtest.signals, "signal_breakout", score)
	monkeypatch.setattr(backtest.signals, "signal_dip", score)
	monkeypatch.setattr(
		backtest.signals,
		"signal_ml",
		lambda requested, *, as_of=None, history=None, persist=True: score(
			requested, as_of=as_of, history=history
		),
	)
	monkeypatch.setattr(
		backtest.signals, "regime_state", lambda *, as_of=None, history=None: "risk_on"
	)

	curve, summary = backtest.run_backtest(
		"A", years=1, rebalance_days=21, cost_bps=5, output_dir=tmp_path
	)

	assert not curve.empty
	assert "Full strategy" in summary
	assert "SPY" in summary
	assert "equal weight" in summary
	assert list(tmp_path.glob("backtest_student_A_*.csv"))
	assert list(tmp_path.glob("backtest_student_A_*.md"))