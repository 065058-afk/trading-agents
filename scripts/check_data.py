"""Run cache and Alpaca data checks for one configured student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import yaml

from mcp_server.data import get_ohlcv_batch, history_cache


def main() -> None:
	parser = argparse.ArgumentParser(description="Check cached and Alpaca market data for a student.")
	parser.add_argument("--student", required=True, choices=("A", "B", "C"))
	parser.add_argument("--days", type=int, default=30, help="Calendar days of Alpaca history to request")
	args = parser.parse_args()

	config_path = ROOT / "config" / f"student_{args.student}.yaml"
	with config_path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	symbols = list(dict.fromkeys([*config["high_beta"], *config["diversified"], "SPY", "QQQ"]))
	history = history_cache(symbols)
	bars = get_ohlcv_batch(symbols, args.days, student=args.student)
	latest = bars["timestamp"].max().date().isoformat()
	print(
		f"Student {args.student}: {len(history)} cached tickers; "
		f"{len(bars)} Alpaca bars across {bars['symbol'].nunique()} tickers; latest {latest}."
	)


if __name__ == "__main__":
	main()