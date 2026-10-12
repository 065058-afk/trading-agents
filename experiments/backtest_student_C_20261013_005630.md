# Student C walk-forward backtest

Period: 2021-10-05 to 2026-10-12 (1833 calendar days).
Rebalance interval: 5 sessions. Transaction cost: 5.0 bps per unit of turnover.
Historical news headlines are not archived, so news tilt is neutral to avoid look-ahead.
Equal-weight is a buy-and-hold basket initialized equally across configured stocks.
The legacy baseline uses trend+rotation with static weights and no risk overlays.

| Portfolio | CAGR | Sharpe | Max drawdown | Hit rate |
|---|---:|---:|---:|---:|
| full strategy | 3.03% | 0.27 | -29.63% | 50.95% |
| legacy baseline | 7.48% | 0.52 | -30.23% | 52.46% |
| SPY | 12.43% | 0.77 | -25.36% | 53.57% |
| equal weight | 15.96% | 0.89 | -23.38% | 53.17% |

Full strategy CAGR difference: -9.40% vs SPY; -12.93% vs equal-weight.
