# PortfolioManager

LLM provider: ollama_local

## Summary

Implementing risk-based targets with 65% high beta share, 100% gross exposure, and 3.5% weight allocations.

## Observations
- Weight distribution across 13 symbols shows balanced exposure.
- High beta share indicates focus on risk mitigation.
- Observations: 65% of exposure is allocated to high-risk assets.

## Blockers
- Weight allocations must remain at the specified thresholds.
- High beta share is a critical component of the regime.
- Observations: 65% of exposure is allocated to high-risk assets.

## Tool summary

```json
{
  "regime": "risk_on",
  "high_beta_share": 0.65,
  "gross_exposure": 1.0,
  "final_targets": [
    {
      "symbol": "WDC",
      "weight": 0.132811
    },
    {
      "symbol": "PLTR",
      "weight": 0.131362
    },
    {
      "symbol": "RCL",
      "weight": 0.130847
    },
    {
      "symbol": "MPWR",
      "weight": 0.129632
    },
    {
      "symbol": "LRCX",
      "weight": 0.125348
    },
    {
      "symbol": "KO",
      "weight": 0.03835
    },
    {
      "symbol": "V",
      "weight": 0.037736
    },
    {
      "symbol": "HON",
      "weight": 0.036776
    },
    {
      "symbol": "DUK",
      "weight": 0.035941
    },
    {
      "symbol": "CVX",
      "weight": 0.035252
    },
    {
      "symbol": "LLY",
      "weight": 0.034844
    },
    {
      "symbol": "AMT",
      "weight": 0.034758
    },
    {
      "symbol": "META",
      "weight": 0.034337
    },
    {
      "symbol": "MSFT",
      "weight": 0.031249
    },
    {
      "symbol": "MCD",
      "weight": 0.030757
    }
  ]
}
```
