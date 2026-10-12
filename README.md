# trading-agents

## Market data

`mcp_server/data.py` caches daily Yahoo OHLCV parquet files in `data/cache/`, including SPY and QQQ, and fetches incremental history from 2000 onward. `get_ohlcv_batch` requests all symbols together from Alpaca's free IEX feed and returns one table; set `ALPACA_KEY_A` and `ALPACA_SECRET_A` (or pass another student suffix) in the environment. `check_data(df)` rejects missing, gapped, or stale bars. Run `python scripts/check_data.py --student A` to check a student's configured stocks and `pytest tests/test_data.py` to run the data-layer tests.

## Signals

`mcp_server/signals.py` provides bounded trend, rotation, breakout, dip, LightGBM, and news scores, plus earnings guards and QQQ regime classification. ML runs are retrained with a five-session walk-forward purge and stored as versioned files in `models/`; fold metrics and selected signal direction are appended under `logs/signals/`. News tilt is capped at 0.2 and strongly negative events are exposed through the result's `vetoes` set. Run `python scripts/show_signals.py --student A` to print that student's signal table and current regime. Tests run with `pytest tests/test_signals.py`.