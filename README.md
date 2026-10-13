# trading-agents

## Market data

`mcp_server/data.py` caches daily Yahoo OHLCV parquet files in `data/cache/`, including SPY and QQQ, and fetches incremental history from 2000 onward. `get_ohlcv_batch` requests all symbols together from Alpaca's free IEX feed and returns one table; set `ALPACA_KEY_A` and `ALPACA_SECRET_A` (or pass another student suffix) in the environment. `check_data(df)` rejects missing, gapped, or stale bars. Run `python scripts/check_data.py --student A` to check a student's configured stocks and `pytest tests/test_data.py` to run the data-layer tests.

## Signals

`mcp_server/signals.py` provides bounded trend, rotation, breakout, dip, LightGBM, and news scores, plus earnings guards and QQQ regime classification. ML runs are retrained with a five-session walk-forward purge and stored as versioned files in `models/`; fold metrics and selected signal direction are appended under `logs/signals/`. News tilt is capped at 0.2 and strongly negative events are exposed through the result's `vetoes` set. Run `python scripts/show_signals.py --student A` to print that student's signal table and current regime. Tests run with `pytest tests/test_signals.py`.

## Backtest

Run `python backtest.py --student A` for a five-year walk-forward comparison. The full strategy uses regime-adaptive configured weights, weekly default rebalancing, transaction costs, and the student's ATR trailing stop, partial-profit, time-stop, and drawdown rules. The legacy comparator uses trend and rotation only, static weights, and no risk overlays. Reports are written to `experiments/` and compare against SPY and an equal-weight buy-and-hold basket. Historical headlines are not cached, so news is held neutral to avoid look-ahead. Use `--years`, `--rebalance-days`, and `--cost-bps` to change the test window and assumptions.

## MCP Server

`mcp_server/server.py` exposes the data, signal, portfolio, exit, account, and paper-order tools over FastMCP stdio. With student credentials, dry-run tools read the paper account without submitting orders; credential-free dry tests use synthetic account data. Target diffs and exit signals are based on current positions, and sells are limited to available long shares. `place_orders` never submits unless `TRADING_ENABLED=true`, and all clients are paper-only. Run `python scripts/test_mcp.py --student A` to launch an isolated `MCP_DRY_RUN=1` server and call every tool without provider or account side effects.

## CrewAI Agents

`crew/agents.py` defines the Data, signal, news, portfolio, risk, execution, and reporting agents. They use one configured low-temperature LLM and structured `AgentReport` output; numeric evidence comes from MCP tools.

## CrewAI Tasks

`crew/tasks.py` defines the asynchronous signal/news analysis tasks followed by portfolio, risk, execution, and reporting tasks. Each task writes a timestamped Markdown report under `logs/<student>/<run folder>/`; malformed structured output gets one retry, and RiskOfficer must use `validate_portfolio`.

## Crew Runner

`crew/crew.py` connects to `mcp_server/server.py` over stdio and uses MCP outputs as the numeric authority. Data and signal Markdown reports are written directly from tool results; only News, PortfolioManager, RiskOfficer, Execution, and Reporter use structured LLM commentary, sequentially, with compact evidence and a 160-token output cap. Groq calls are limited to 8,000 estimated tokens per minute, retry once on HTTP 400/429 (respecting `Retry-After` for 429), then fall back to `GROQ_MODEL_FALLBACK` and Ollama Cloud. Set `GROQ_MODEL` to select the primary Groq model. If commentary is unavailable, deterministic MCP validation and orders remain in force, with news scores set to zero. Run `python -m crew.crew --student A --mode dry` for a non-submitting run; use `--mode paper` only with `TRADING_ENABLED=true`.

## Daily Cycle Runner

`run_cycle.py` checks Alpaca's market clock, skips a completed student/date cycle, refreshes data and retrains the ML signal through the crew, then validates and executes sells before buys. It nets buys and sells to one order per symbol and caps sells at the currently available long quantity. Each order is limited to $10,000; total target exposure is limited to 100%. The daily cap is `max(20, configured stock count + 5)` when no target stocks are held or fewer than half are held, then returns to 10; the applied cap is printed and stored in `logs/<student>/state.json`. Skipped runs fetch a fresh Alpaca equity and position snapshot for the web console. Set `TRADING_ENABLED=true` for `--mode live`; otherwise live requests are forced to dry mode. `--force-clock` bypasses the market clock for dry tests only. Run `python run_cycle.py --student A --mode dry`.

## Web Console

`web/app.py` provides a local FastAPI dashboard for running the same cycle, editing per-run student symbols and signal weights, reviewing positions/orders/normalized portfolio-vs-SPY history, and optionally approving, editing downward, or rejecting capped orders. Alpaca credentials are held only in an in-memory browser session and are never written to config or logs. Start it with `python -m uvicorn web.app:app --host 127.0.0.1 --port 8000` and open `http://127.0.0.1:8000`.