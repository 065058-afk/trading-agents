Project: CrewAI trading system for Alpaca PAPER trading, 3 students (A, B, C), Python 3.11,
free tools only. Architecture: CrewAI agents act as an MCP client; a local MCP server
(FastMCP, stdio) hosts all tools. Never use Alpaca's own MCP server.
Rules: read keys only from env vars ALPACA_KEY_<student>, ALPACA_SECRET_<student>,
GROQ_API_KEY, FINNHUB_API_KEY, TRADING_ENABLED. Never print or log keys. Use paper trading
only. Fetch OHLCV for all symbols in ONE Alpaca call. Save every agent's output as an md
file in logs/<student>/<run folder>/<agentName>DDMMYYYY-HH-MM.md. Student settings and
weights come from config/student_<X>.yaml. Write clear comments and a short README section
per module, and add tests.