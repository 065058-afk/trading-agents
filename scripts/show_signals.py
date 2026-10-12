"""Print the current signal table for a configured student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import pandas as pd
import yaml

from mcp_server.data import history_cache
from mcp_server.signals import (
	FinnhubAuthenticationError,
	earnings_guard,
	regime_state,
	signal_breakout,
	signal_dip,
	signal_ml,
	signal_news_tilt,
	signal_rotation,
	signal_trend,
)


def main() -> None:
	parser = argparse.ArgumentParser(description="Show configured student trading signals.")
	parser.add_argument("--student", required=True, choices=("A", "B", "C"))
	args = parser.parse_args()
	config_path = ROOT / "config" / f"student_{args.student}.yaml"
	with config_path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	symbols = list(dict.fromkeys([*config["high_beta"], *config["diversified"]]))
	history_cache([*symbols, "SPY", "QQQ"])

	trend = signal_trend(symbols)
	rotation = signal_rotation(symbols)
	breakout = signal_breakout(symbols)
	dip = signal_dip(symbols)
	ml = signal_ml(symbols)
	news = signal_news_tilt(symbols, student=args.student)
	try:
		earnings: set[str] | None = set(earnings_guard(symbols, 7))
	except FinnhubAuthenticationError as error:
		print(f"Warning: {error}; earnings status is unavailable.")
		earnings = None
	state = regime_state()

	table = pd.DataFrame(
		{
			"Symbol": symbols,
			"Trend": [trend[symbol] for symbol in symbols],
			"Rotation": [rotation[symbol] for symbol in symbols],
			"Breakout": [breakout[symbol] for symbol in symbols],
			"Dip": [dip[symbol] for symbol in symbols],
			"ML": [ml[symbol] for symbol in symbols],
			"News tilt": [news[symbol] for symbol in symbols],
			"News veto": [symbol in news.vetoes for symbol in symbols],
			"Earnings <= 7d": [symbol in earnings if earnings is not None else "Unknown" for symbol in symbols],
		}
	)
	print(f"Student {args.student} | regime: {state}")
	print(table.to_string(index=False, float_format=lambda value: f"{value:+.3f}"))


if __name__ == "__main__":
	main()