# Student B walk-forward backtest

Period: 2021-10-05 to 2026-10-12 (1833 calendar days).
Rebalance interval: 5 sessions. Transaction cost: 5.0 bps per unit of turnover.
Historical news headlines are not archived, so news tilt is neutral to avoid look-ahead.
Equal-weight is a buy-and-hold basket initialized equally across configured stocks.
The legacy baseline uses trend+rotation with static weights and no risk overlays.

| Portfolio | CAGR | Sharpe | Max drawdown | Hit rate |
|---|---:|---:|---:|---:|
| full strategy | 11.97% | 0.78 | -28.03% | 48.25% |
| legacy baseline | 21.05% | 1.13 | -32.36% | 52.38% |
| SPY | 12.43% | 0.77 | -25.36% | 53.57% |
| equal weight | 26.17% | 1.15 | -25.61% | 53.89% |

Full strategy CAGR difference: -0.46% vs SPY; -14.19% vs equal-weight.
