# trading-agents

## Market data

`mcp_server/data.py` caches daily Yahoo OHLCV parquet files in `data/cache/`, including SPY and QQQ, and fetches incremental history from 2000 onward. `get_ohlcv_batch` requests all symbols together from Alpaca's free IEX feed and returns one table; set `ALPACA_KEY_A` and `ALPACA_SECRET_A` (or pass another student suffix) in the environment. `check_data(df)` rejects missing, gapped, or stale bars. Run `python scripts/check_data.py --student A` to check a student's configured stocks and `pytest tests/test_data.py` to run the data-layer tests.