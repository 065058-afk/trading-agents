# PortfolioManager

LLM provider: ollama_local (qwen3:0.6b)

## Structured agent output
```json
{
  "status": "complete",
  "decision": "approve",
  "summary": "The target portfolio has weights concentrated in AMD, AMZN, COP, CSCO, LIN, MA, MRK, and SO, with a regime of 'risk_on'. The signal scores indicate positive signals in breakout, dip, ml, news, rotation, and trend, suggesting a positive regime.",
  "observations": [
    "The target weights are primarily in key sectors with positive signal scores, indicating a positive regime. The portfolio is structured to align with the signals provided.",
    "The regime is defined as risk on, which suggests a focus on those specific sectors."
  ],
  "blockers": []
}
```

## Authoritative MCP evidence
```json
{
  "student": "C",
  "regime": "risk_on",
  "adjusted_signal_weights": {
    "trend": 0.23529411764705882,
    "rotation": 0.23529411764705882,
    "breakout": 0.1764705882352941,
    "dip": 0.08823529411764705,
    "ml": 0.17647058823529413,
    "news": 0.08823529411764706
  },
  "high_beta_share": 0.6,
  "earnings_guard": [],
  "signal_scores": {
    "trend": {
      "AMD": 0.08,
      "TER": 0.33,
      "DAL": 0.07,
      "FCX": 0.23,
      "CSCO": 0.94,
      "MA": 0.41,
      "MRK": 0.32,
      "WMT": 0.46,
      "COP": 0.24,
      "SO": 0.61,
      "UNP": 0.41,
      "AMZN": 0.07,
      "LIN": 0.25,
      "O": 0.79,
      "TMO": 0.38
    },
    "rotation": {
      "AMD": 0.21,
      "TER": 0.46,
      "DAL": 0.2,
      "FCX": 0.36,
      "CSCO": 0.06,
      "MA": 0.54,
      "MRK": 0.45,
      "WMT": 0.59,
      "COP": 0.37,
      "SO": 0.74,
      "UNP": 0.54,
      "AMZN": 0.2,
      "LIN": 0.38,
      "O": 0.92,
      "TMO": 0.51
    },
    "breakout": {
      "AMD": 0.34,
      "TER": 0.59,
      "DAL": 0.33,
      "FCX": 0.49,
      "CSCO": 0.19,
      "MA": 0.67,
      "MRK": 0.58,
      "WMT": 0.72,
      "COP": 0.5,
      "SO": 0.87,
      "UNP": 0.67,
      "AMZN": 0.33,
      "LIN": 0.51,
      "O": 0.04,
      "TMO": 0.64
    },
    "dip": {
      "AMD": 0.47,
      "TER": 0.72,
      "DAL": 0.46,
      "FCX": 0.62,
      "CSCO": 0.32,
      "MA": 0.8,
      "MRK": 0.71,
      "WMT": 0.85,
      "COP": 0.63,
      "SO": 1.0,
      "UNP": 0.8,
      "AMZN": 0.46,
      "LIN": 0.64,
      "O": 0.17,
      "TMO": 0.77
    },
    "ml": {
      "AMD": 0.09562086887052446,
      "TER": -0.10983377859648416,
      "DAL": 0.3449283806028057,
      "FCX": -0.10983377859648416,
      "CSCO": 0.25509992427186934,
      "MA": 0.17481527899374877,
      "MRK": 0.1462666823482277,
      "WMT": -0.13251565562547815,
      "COP": 0.1377521670830355,
      "SO": 0.1906253317108384,
      "UNP": 0.22419282789569925,
      "AMZN": 0.1507379032327157,
      "LIN": 0.1728113961948305,
      "O": 0.16735328712242606,
      "TMO": 0.11601994463309626
    },
    "news": {
      "AMD": 0.73,
      "TER": 0.98,
      "DAL": 0.72,
      "FCX": 0.88,
      "CSCO": 0.58,
      "MA": 0.05,
      "MRK": 0.97,
      "WMT": 0.1,
      "COP": 0.89,
      "SO": 0.25,
      "UNP": 0.05,
      "AMZN": 0.72,
      "LIN": 0.9,
      "O": 0.43,
      "TMO": 0.02
    }
  },
  "target_weights": {
    "AMD": 0.14999999700000008,
    "TER": 0.14999999700000008,
    "DAL": 0.14999999700000008,
    "FCX": 0.14999999700000008,
    "CSCO": 0.035370399292592024,
    "MA": 0.03675046926499062,
    "MRK": 0.037003379259932426,
    "WMT": 0.036418959271620824,
    "COP": 0.03529267929414642,
    "SO": 0.041002679179946426,
    "UNP": 0.03697168926056622,
    "AMZN": 0.03179668936406622,
    "LIN": 0.03565881928682362,
    "O": 0.03787438924251222,
    "TMO": 0.035859859282802824
  },
  "gross_exposure": 1.0
}
```
