# Experiment matrix and interpretation protocol

| Group | Policies / perturbation | Hypothesis | Key diagnostics |
|---|---|---|---|
| E0 | IYW vs DCA | establish cash-flow-matched reference | terminal wealth, TWR, fees |
| E1 | relative_only vs regret_only vs combined_fixed | separate CVaR effects | rel/abs loss CVaR, decision regret, infeasible rate |
| E2 | combined_fixed vs combined_dynamic | incremental effect of review time | horizon histogram, reviews, fees, wealth |
| E3 | combined_dynamic vs preference_robust | sensitivity to risk aversion | CVaR, relative growth, cash held |
| E4 | mixed/crash/break/correlated, 30+ seeds | stress and distribution shift | distribution of paired OOS differences |
| E5 | scenario counts, lookbacks, CVaR budgets | robustness of model assumptions | constraint rejection, sensitivity |
| E6 | real point-in-time equity universe and IYW | external validity | rolling OOS relative performance |
| E7 | oracle menu expansion and horizon normalization | oracle fairness and sensitivity | regret stability and selection changes |

The CLI implements E0-E3 and single-run E4. E5-E7 are research protocols, **not yet automated**. Report negative and null findings. Paired block resampling must be conducted on pre-registered OOS daily return differences and account for overlapping tests. Synthetic seeds are independent replicates; dates inside one backtest are not independent.

## Paper figure captions

- `twr_nav.png`: growth of one initial time-weighted unit, removing mechanical effect of contributions. Not the same as investor money-weighted IRR.
- `wealth.png`: terminal wealth with identical contributions. The frictionless matched IYW reference may outperform a fee-bearing strategy partly because of different fee assumptions.
- `drawdown.png`: drawdown of time-weighted unit NAV. Does not imply max regret or CVaR constraint satisfaction.

## Mandatory limitations table for publication

Document survivorship, delisting, total-return treatment, FX, historical membership, transaction slippage, price impact, hyperparameter selection, independent holdout, multiple testing, scenario miscalibration, bootstrap dependence, and constraints infeasible under Only-In. Do not call a bootstrap hindsight comparator a feasible trading strategy.
