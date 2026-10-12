# Student A walk-forward backtest

Period: 2025-10-10 to 2026-10-12 (367 calendar days).
Rebalance interval: 63 sessions. Transaction cost: 5.0 bps per unit of turnover.
Historical news headlines are not archived, so news tilt is neutral to avoid look-ahead.
Equal-weight is a buy-and-hold basket initialized equally across configured stocks.
The legacy baseline uses trend+rotation with static weights and no risk overlays.

| Portfolio | CAGR | Sharpe | Max drawdown | Hit rate |
|---|---:|---:|---:|---:|
| full strategy | 3.03% | 0.61 | -4.16% | 3.97% |
| legacy baseline | 54.96% | 1.91 | -11.71% | 52.78% |
| SPY | 19.19% | 1.44 | -9.13% | 52.38% |
| equal weight | 44.88% | 1.77 | -12.74% | 54.76% |

Full strategy CAGR difference: -16.17% vs SPY; -41.86% vs equal-weight.
