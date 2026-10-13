# PortfolioManager

LLM provider: ollama_local (qwen3:0.6b)

## Structured agent output
```json
{
  "status": "complete",
  "decision": "approve",
  "summary": "The target portfolio has weights of 3.5% for AMT, 3.5% for CVX, 3.59% for DUK, 3.68% for HON, 3.84% for KO, 3.48% for LLY, 12.53% for LRCX, 3.08% for MCD, 3.43% for META, 12.96% for MPWR, 3.12% for MSFT, 13.14% for PLTR, 13.08% for RCL, 3.77% for V, and 13.28% for WDC. The regime is 'risk_on', indicating a focus on risk mitigation. The weights are authoritative and show a balanced allocation across various assets, with a low concentration in certain sectors.",
  "observations": [
    "Observation 1: The target weights are all within the 3.5% range, suggesting a balanced yet conservative allocation across different assets.",
    "Observation 2: The portfolio's concentration in certain assets (e.g., LRCX, MCD) indicates a focus on specific sectors, which aligns with the 'risk_on' regime.",
    "Observation 3: The low concentration in AMT and CVX suggests a conservative approach, which is consistent with the 'risk_on' strategy.",
    "Observations: The summary highlights a balanced and conservative approach, with a focus on risk mitigation through the specified regime."
  ],
  "blockers": []
}
```

## Authoritative MCP evidence
```json
{
  "student": "B",
  "regime": "risk_on",
  "adjusted_signal_weights": {
    "trend": 0.23529411764705882,
    "rotation": 0.23529411764705882,
    "breakout": 0.1764705882352941,
    "dip": 0.08823529411764705,
    "ml": 0.17647058823529413,
    "news": 0.08823529411764706
  },
  "high_beta_share": 0.65,
  "earnings_guard": [],
  "signal_scores": {
    "trend": {
      "LRCX": 0.1,
      "MPWR": 0.23,
      "WDC": 0.2,
      "PLTR": 0.19,
      "RCL": 0.23,
      "MSFT": 0.11,
      "V": 0.86,
      "LLY": 0.39,
      "KO": 0.53,
      "CVX": 0.39,
      "DUK": 0.26,
      "HON": 0.27,
      "MCD": 0.1,
      "META": 0.93,
      "AMT": 0.24
    },
    "rotation": {
      "LRCX": 0.23,
      "MPWR": 0.36,
      "WDC": 0.33,
      "PLTR": 0.32,
      "RCL": 0.36,
      "MSFT": 0.24,
      "V": 0.99,
      "LLY": 0.52,
      "KO": 0.66,
      "CVX": 0.52,
      "DUK": 0.39,
      "HON": 0.4,
      "MCD": 0.23,
      "META": 0.05,
      "AMT": 0.37
    },
    "breakout": {
      "LRCX": 0.36,
      "MPWR": 0.49,
      "WDC": 0.46,
      "PLTR": 0.45,
      "RCL": 0.49,
      "MSFT": 0.37,
      "V": 0.11,
      "LLY": 0.65,
      "KO": 0.79,
      "CVX": 0.65,
      "DUK": 0.52,
      "HON": 0.53,
      "MCD": 0.36,
      "META": 0.18,
      "AMT": 0.5
    },
    "dip": {
      "LRCX": 0.49,
      "MPWR": 0.62,
      "WDC": 0.59,
      "PLTR": 0.58,
      "RCL": 0.62,
      "MSFT": 0.5,
      "V": 0.24,
      "LLY": 0.78,
      "KO": 0.92,
      "CVX": 0.78,
      "DUK": 0.65,
      "HON": 0.66,
      "MCD": 0.49,
      "META": 0.31,
      "AMT": 0.63
    },
    "ml": {
      "LRCX": 0.1874024343015294,
      "MPWR": -0.17067355980986837,
      "WDC": 0.1537976586239627,
      "PLTR": 0.11639404046379324,
      "RCL": -0.10012993678971265,
      "MSFT": -0.056983186429118426,
      "V": -0.06810839424121067,
      "LLY": -0.044530080769980517,
      "KO": 0.09607690557344845,
      "CVX": 0.047779491079712647,
      "DUK": 0.30536856662363693,
      "HON": 0.44780800923922487,
      "MCD": -0.12185952234709727,
      "META": 0.18214316721434237,
      "AMT": 0.13100743406158963
    },
    "news": {
      "LRCX": 0.75,
      "MPWR": 0.88,
      "WDC": 0.85,
      "PLTR": 0.84,
      "RCL": 0.88,
      "MSFT": 0.76,
      "V": 0.5,
      "LLY": 0.03,
      "KO": 0.17,
      "CVX": 0.03,
      "DUK": 0.91,
      "HON": 0.92,
      "MCD": 0.75,
      "META": 0.57,
      "AMT": 0.89
    }
  },
  "target_weights": {
    "LRCX": 0.12534807,
    "MPWR": 0.12963183,
    "WDC": 0.13281068,
    "PLTR": 0.13136196,
    "RCL": 0.13084745,
    "MSFT": 0.03124925,
    "V": 0.03773581,
    "LLY": 0.03484441,
    "KO": 0.03835046,
    "CVX": 0.03525205,
    "DUK": 0.0359406,
    "HON": 0.0367757,
    "MCD": 0.03075667,
    "META": 0.0343366,
    "AMT": 0.03475846
  },
  "gross_exposure": 1.0
}
```
