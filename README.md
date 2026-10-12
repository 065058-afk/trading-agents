# trading-agents

## Market data

`mcp_server/data.py` caches daily Yahoo OHLCV parquet files in `data/cache/`, including SPY and QQQ, and fetches incremental history from 2000 onward. `get_ohlcv_batch` requests all symbols together from Alpaca's free IEX feed; set `ALPACA_KEY_A` and `ALPACA_SECRET_A` (or pass another student suffix) in the environment. Both paths reject missing, gapped, or stale recent data. Run the data-layer tests with `pytest tests/test_data.py`.