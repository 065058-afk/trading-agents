# PortfolioManager

LLM provider: ollama_local (qwen3:0.6b)

## Structured agent output
```json
{
  "status": "complete",
  "decision": "approve",
  "summary": "The target portfolio has weights of AAPL:0.0438, CAT:0.0422, COIN:0.11, GOOGL:0.0507, HD:0.0440, JNJ:0.0441, JPM:0.0448, MU:0.1256, NEE:0.0416, NVDA:0.1086, PG:0.0483, PLD:0.0437, TSLA:0.0954, UAL:0.1106, XOM:0.0469, with a risk regime of 'risk_on' and a total portfolio value of 100% allocated. Concentration is evident in high-beta stocks and low-beta stocks, and cash implications are clear from the weights. Blockers include no unsupported claims.",
  "observations": [
    "Concentration is evident in high-beta stocks and low-beta stocks, and cash implications are clear from the weights.",
    "The target portfolio is well-structured with a risk regime of 'risk_on' and a total portfolio value of 100% allocated.",
    "The weights and regime are supported by the MCP tools, ensuring no unsupported claims."
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
      "MU": 0.0981515940549893,
      "NVDA": 0.14822658345702194,
      "TSLA": 0.030549470744153574,
      "COIN": 0.24276523655900883,
      "UAL": 0.23843334802114774,
      "AAPL": -0.6129231540574913,
      "JPM": 0.11292423734427291,
      "JNJ": 0.2135094456799466,
      "PG": 0.2626194033995097,
      "XOM": 0.3955583788807045,
      "NEE": 0.2408826906304718,
      "CAT": 0.33979803703393285,
      "HD": 0.014630416362525432,
      "GOOGL": 0.1372853860615015,
      "PLD": 0.2347818028833604
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
    "MU": 0.12555668,
    "NVDA": 0.10858089,
    "TSLA": 0.09540119,
    "COIN": 0.10989107,
    "UAL": 0.11057016,
    "AAPL": 0.04379804,
    "JPM": 0.04479053,
    "JNJ": 0.04405523,
    "PG": 0.04825078,
    "XOM": 0.04691921,
    "NEE": 0.04162194,
    "CAT": 0.04216984,
    "HD": 0.04403374,
    "GOOGL": 0.05070459,
    "PLD": 0.04365608
  },
  "gross_exposure": 0.99999997
}
```
