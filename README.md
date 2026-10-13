# trading-agents

## Market data

`mcp_server/data.py` caches daily Yahoo OHLCV parquet files in `data/cache/`, including SPY and QQQ, and fetches incremental history from 2000 onward. `get_ohlcv_batch` requests all symbols together from Alpaca's free IEX feed and returns one table; set `ALPACA_KEY_A` and `ALPACA_SECRET_A` (or pass another student suffix) in the environment. `check_data(df)` rejects missing, gapped, or stale bars. Run `python scripts/check_data.py --student A` to check a student's configured stocks and `pytest tests/test_data.py` to run the data-layer tests.

## Signals

`mcp_server/signals.py` provides bounded trend, rotation, breakout, dip, LightGBM, and news scores, plus earnings guards and QQQ regime classification. ML runs are retrained with a five-session walk-forward purge and stored as versioned files in `models/`; fold metrics and selected signal direction are appended under `logs/signals/`. News tilt is capped at 0.2 and strongly negative events are exposed through the result's `vetoes` set. Run `python scripts/show_signals.py --student A` to print that student's signal table and current regime. Tests run with `pytest tests/test_signals.py`.

## Backtest

Run `python backtest.py --student A` for a five-year walk-forward comparison. The full strategy uses regime-adaptive configured weights, weekly default rebalancing, transaction costs, and the student's ATR trailing stop, partial-profit, time-stop, and drawdown rules. The legacy comparator uses trend and rotation only, static weights, and no risk overlays. Reports are written to `experiments/` and compare against SPY and an equal-weight buy-and-hold basket. Historical headlines are not cached, so news is held neutral to avoid look-ahead. Use `--years`, `--rebalance-days`, and `--cost-bps` to change the test window and assumptions.

## MCP Server

`mcp_server/server.py` exposes the data, signal, portfolio, exit, account, and paper-order tools over FastMCP stdio. Account tools require a student key pair (`ALPACA_KEY_<student>` and `ALPACA_SECRET_<student>`); `place_orders` never submits unless `TRADING_ENABLED=true`, and all clients are paper-only. Run `python scripts/test_mcp.py --student A` to launch an isolated `MCP_DRY_RUN=1` server and call every tool without provider or account side effects.

## CrewAI Agents

`crew/agents.py` defines the Data, signal, news, portfolio, risk, execution, and reporting agents. They use one configured low-temperature LLM and structured `AgentReport` output; numeric evidence comes from MCP tools.

## CrewAI Tasks

`crew/tasks.py` defines the asynchronous signal/news analysis tasks followed by portfolio, risk, execution, and reporting tasks. Each task writes a timestamped Markdown report under `logs/<student>/<run folder>/`; malformed structured output gets one retry, and RiskOfficer must use `validate_portfolio`.

## Crew Runner

`crew/crew.py` selects Groq, Ollama Cloud, or local Ollama in configured order, connects to `mcp_server/server.py` over stdio, and enforces the validator result before execution. Run `python -m crew.crew --student A --mode dry` for a non-submitting run; use `--mode paper` only with `TRADING_ENABLED=true`.

## Daily Cycle Runner

`run_cycle.py` checks Alpaca's market clock, skips a completed student/date cycle, refreshes data and retrains the ML signal through the crew, then validates and executes sells before buys. It writes peak equity and daily run status to `logs/<student>/state.json`, limits each order to $10,000, total target exposure to 100%, and each cycle to 10 orders. Set `TRADING_ENABLED=true` for `--mode live`; otherwise live requests are forced to dry mode. `--force-clock` bypasses the market clock for dry tests only. Run `python run_cycle.py --student A --mode dry`.