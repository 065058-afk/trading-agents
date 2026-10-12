# PortfolioManager

LLM provider: groq (openai/gpt-oss-20b)

## Structured agent output
```json
{
  "status": "complete",
  "decision": "approve",
  "summary": "The MCP-generated target portfolio assigns weights to 15 equities, with a risk_on regime, high_beta_share of 0.55, and gross exposure of 1.00000001, indicating a fully invested position.",
  "observations": [
    "Portfolio fully invested with gross exposure ~1.0, indicating no cash position.",
    "High beta share of 0.55 reflects moderate risk exposure.",
    "Regime is risk_on, aligning with aggressive weighting.",
    "Target weights span 15 equities, with no single position exceeding 12.1% (MU).",
    "No concentration risk beyond typical diversification."
  ],
  "blockers": []
}
```

## Authoritative MCP evidence
```json
{
  "student": "A",
  "regime": "risk_on",
  "adjusted_signal_weights": {
    "trend": 0.23529411764705882,
    "rotation": 0.23529411764705882,
    "breakout": 0.1764705882352941,
    "dip": 0.08823529411764705,
    "ml": 0.17647058823529413,
    "news": 0.08823529411764706
  },
  "high_beta_share": 0.55,
  "earnings_guard": [],
  "signal_scores": {
    "trend": {
      "MU": 0.61,
      "NVDA": 0.95,
      "TSLA": 0.05,
      "COIN": 0.95,
      "UAL": 0.24,
      "AAPL": 0.84,
      "JPM": 0.29,
      "JNJ": 0.24,
      "PG": 0.5,
      "XOM": 0.42,
      "NEE": 0.14,
      "CAT": 0.14,
      "HD": 0.39,
      "GOOGL": 0.73,
      "PLD": 0.22
    },
    "rotation": {
      "MU": 0.74,
      "NVDA": 0.07,
      "TSLA": 0.18,
      "COIN": 0.07,
      "UAL": 0.37,
      "AAPL": 0.97,
      "JPM": 0.42,
      "JNJ": 0.37,
      "PG": 0.63,
      "XOM": 0.55,
      "NEE": 0.27,
      "CAT": 0.27,
      "HD": 0.52,
      "GOOGL": 0.86,
      "PLD": 0.35
    },
    "breakout": {
      "MU": 0.87,
      "NVDA": 0.2,
      "TSLA": 0.31,
      "COIN": 0.2,
      "UAL": 0.5,
      "AAPL": 0.09,
      "JPM": 0.55,
      "JNJ": 0.5,
      "PG": 0.76,
      "XOM": 0.68,
      "NEE": 0.4,
      "CAT": 0.4,
      "HD": 0.65,
      "GOOGL": 0.99,
      "PLD": 0.48
    },
    "dip": {
      "MU": 1.0,
      "NVDA": 0.33,
      "TSLA": 0.44,
      "COIN": 0.33,
      "UAL": 0.63,
      "AAPL": 0.22,
      "JPM": 0.68,
      "JNJ": 0.63,
      "PG": 0.89,
      "XOM": 0.81,
      "NEE": 0.53,
      "CAT": 0.53,
      "HD": 0.78,
      "GOOGL": 0.11,
      "PLD": 0.61
    },
    "ml": {
      "MU": 0.12,
      "NVDA": 0.46,
      "TSLA": 0.57,
      "COIN": 0.46,
      "UAL": 0.76,
      "AAPL": 0.35,
      "JPM": 0.81,
      "JNJ": 0.76,
      "PG": 0.01,
      "XOM": 0.94,
      "NEE": 0.66,
      "CAT": 0.66,
      "HD": 0.91,
      "GOOGL": 0.24,
      "PLD": 0.74
    },
    "news": {
      "MU": 0.25,
      "NVDA": 0.59,
      "TSLA": 0.7,
      "COIN": 0.59,
      "UAL": 0.89,
      "AAPL": 0.48,
      "JPM": 0.94,
      "JNJ": 0.89,
      "PG": 0.14,
      "XOM": 0.06,
      "NEE": 0.79,
      "CAT": 0.79,
      "HD": 0.03,
      "GOOGL": 0.37,
      "PLD": 0.87
    }
  },
  "target_weights": {
    "MU": 0.12094717,
    "NVDA": 0.1084951,
    "TSLA": 0.09886194,
    "COIN": 0.1084951,
    "UAL": 0.11320069,
    "AAPL": 0.04642269,
    "JPM": 0.0459691,
    "JNJ": 0.04448623,
    "PG": 0.04426816,
    "XOM": 0.04718157,
    "NEE": 0.04152048,
    "CAT": 0.04152048,
    "HD": 0.04629185,
    "GOOGL": 0.04844637,
    "PLD": 0.04389308
  },
  "gross_exposure": 1.00000001
}
```
