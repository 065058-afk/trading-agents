# PortfolioManager

LLM provider: ollama_local

## Summary

Final regime with 0.099973% PLTR, 0.096577% RCL, 0.095878% KO, 0.094464% V, 0.085367% DUK, 0.081815% MSFT, 0.07914% MCD, 0.078737% LRCX, 0.077719% AMT, 0.074713% WDC, 0.069009% HON, 0.066608% META

## Observations
- The regime reduces risk exposure by 0.35% in high beta share
- Gross exposure is 99.999998% of the total
- The weights align with the final targets

## Blockers
- The high beta share allocation is critical for risk mitigation
- Gross exposure must remain at 99.999998%
- The weights must be adjusted for market volatility

## Tool summary

```json
{
  "regime": "risk_off",
  "high_beta_share": 0.35,
  "gross_exposure": 0.99999998,
  "final_targets": [
    {
      "symbol": "PLTR",
      "weight": 0.099973
    },
    {
      "symbol": "RCL",
      "weight": 0.096577
    },
    {
      "symbol": "KO",
      "weight": 0.095878
    },
    {
      "symbol": "V",
      "weight": 0.094464
    },
    {
      "symbol": "DUK",
      "weight": 0.085367
    },
    {
      "symbol": "MSFT",
      "weight": 0.081815
    },
    {
      "symbol": "MCD",
      "weight": 0.07914
    },
    {
      "symbol": "LRCX",
      "weight": 0.078737
    },
    {
      "symbol": "AMT",
      "weight": 0.077719
    },
    {
      "symbol": "WDC",
      "weight": 0.074713
    },
    {
      "symbol": "HON",
      "weight": 0.069009
    },
    {
      "symbol": "META",
      "weight": 0.066608
    }
  ]
}
```
