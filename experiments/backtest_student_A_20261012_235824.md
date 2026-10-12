# Student A walk-forward backtest

Period: 2025-10-10 to 2026-10-12 (367 calendar days).
Rebalance interval: 5 sessions. Transaction cost: 5.0 bps per unit of turnover.
Historical news headlines are not archived, so news tilt is neutral to avoid look-ahead.
Equal-weight is a buy-and-hold basket initialized equally across configured stocks.
The legacy baseline uses trend+rotation with static weights and no risk overlays.

| Portfolio | CAGR | Sharpe | Max drawdown | Hit rate |
|---|---:|---:|---:|---:|
| full strategy | 13.29% | 1.15 | -7.27% | 21.43% |
| legacy baseline | 31.05% | 1.45 | -14.05% | 51.59% |
| SPY | 19.19% | 1.44 | -9.13% | 52.38% |
| equal weight | 44.88% | 1.77 | -12.74% | 54.76% |

Full strategy CAGR difference: -5.90% vs SPY; -31.59% vs equal-weight.
