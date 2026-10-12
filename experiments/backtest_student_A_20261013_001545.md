# Student A walk-forward backtest

Period: 2022-04-12 to 2026-10-12 (1644 calendar days).
Rebalance interval: 5 sessions. Transaction cost: 5.0 bps per unit of turnover.
Historical news headlines are not archived, so news tilt is neutral to avoid look-ahead.
Equal-weight is a buy-and-hold basket initialized equally across configured stocks.
The legacy baseline uses trend+rotation with static weights and no risk overlays.

| Portfolio | CAGR | Sharpe | Max drawdown | Hit rate |
|---|---:|---:|---:|---:|
| full strategy | 17.43% | 1.02 | -21.63% | 48.63% |
| legacy baseline | 21.10% | 1.05 | -26.62% | 52.17% |
| SPY | 13.67% | 0.84 | -19.88% | 53.76% |
| equal weight | 29.27% | 1.14 | -29.63% | 54.12% |

Full strategy CAGR difference: +3.76% vs SPY; -11.84% vs equal-weight.
