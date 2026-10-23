# V4 replacement: explicit cash wait and infeasibility diagnostics

This is a full drop-in replacement of the uploaded V3 project. Keep the existing folder name so commands remain unchanged.

## Fixes
- Cash wait (`stock weights=0`, `IYW weight=0`) is now a candidate action in scenario scoring and the scenario hindsight oracle. This changes regret definitions and potentially backtest results versus V3.
- Distinguishes `feasible_optimum`, `cash_wait_feasible`, and `infeasible_fallback` in decision CSVs.
- On infeasibility, executes no buy, flags `feasible=False`, records `minimum_total_violation` and metrics of the least-violating *hypothetical* candidate, separately from actual action metrics.
- Records candidate counts and feasible counts. Fallback actual CVaR metrics are NaN, not fabricated zeros.
- Summary adds `cash_wait_feasible`, `infeasible_fallbacks`, `has_infeasible_fallback`.
- Adds three tests; 9/9 pass in the build environment.

## Windows Anaconda Prompt
```
cd C:\lsg\hedge_fund_iyw_benchmark\ADTPA_IYW_Regret_Experiments_V3
python -m pytest -q tests
python -m v3.cli --n 700 --d 6 --scenarios 100 --out results\experiment_v4
```

## Limitations (important for scientific interpretation)
- A forced cash fallback is NOT a feasible portfolio risk solution and must not be presented as evidence of regret optimization success.
- `minimum_total_violation` is a sum of positive exceedances (in return units), not a distance to a mathematically proven feasible set.
- CVaR is calculated from synthetic bootstrap scenarios; regret is scenario-hindsight relative to the enumerated actions, not implementable policy regret.
- `preference_uncertainty` remains 0 by default; preference_robust can coincide with combined_dynamic.
- Baseline IYW strategy pays transaction fees while the matched IYW reference is frictionless. This is an intentional-but-asymmetric reporting convention inherited from V3 and should be harmonized before publication.
- The model uses next-day execution of decisions, but candidate risk estimation assumes decision-day execution. This approximation and any contribution/fee mismatches need future audit.
- This version is for exploratory research, not live trading.
