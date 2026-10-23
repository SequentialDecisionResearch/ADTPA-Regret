# ADTPA-Regret — computational audit for portfolio decision validation

Companion research materials for **Adaptive Simulation Budgets for Portfolio Decision Validation: A Paired Comparison and Feasibility Audit** (working paper, October 2026).

The paper studies a four-checkpoint Monte Carlo gate at 6, 12, 24, and 48 holdout replications, evaluated on 66 frozen portfolio states and 30 holdout seed streams. Reported agreement with a finite 192-replication reference is **82.73%** for the adaptive gate and **73.46%** for a retrospectively mean-budget-matched state-independent 6/48 mixture. These are conditional simulation results, not evidence of trading alpha.

## Code and outputs

- `budget_matched_multistage_audit.py` and `results/budget_matched_multistage_outputs/`: replication costs, agreement, matched-mixture contrasts.
- `multi_seed_stability_audit.py` and `results/multi_seed_stability_outputs/`: finite-reference stability.
- `decision_regret_budget_frontier_audit.py` and `results/decision_regret_budget_frontier_outputs/`: independently evaluated score regret.
- `common_support_regret_feasibility_audit.py` and `results/common_support_audit_outputs/`: common-support and coverage audit.
- `budget_invariant_feasibility_audit.py` and `results/budget_invariant_feasibility_outputs/`: all-replication versus fixed-prefix eligibility.
- `real_market_ablation.py` and `results/real_market_ablation_outputs/`: historical contextual checks, where available.

The original experiment directory names are retained under `results/`. Source scripts are retained at repository root to preserve their relative imports. Additional research-prototype modules are included for dependency context.

## Environment

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Read each script's `--help` or source before running; several audit scripts support `--run`. **These commands have not been rerun or independently verified for this curated release.** The included CSV files are archived experiment outputs, not newly regenerated results.

## Data availability and limitations

The source archive includes a Yahoo Finance-derived frozen adjusted-close market panel. **That price panel is intentionally omitted from this public-upload package pending review of redistribution terms.** Re-running market-dependent experiments may therefore require reconstructing the original input snapshot from an authorized source and verifying its checksum and adjustment conventions. Some scripts may expect the original `real_market_oos_outputs/frozen_adjusted_close.csv` relative path; the absence of this file is deliberate. `stock_universe.csv` is also omitted pending provenance review.

The 30 seed streams reuse one fixed historical state panel and are not independent market histories. The score-regret contrast is conditional on observed common support (965 of 1,980 pairs), in engine score units rather than investment returns. The 192-replication reference is not an economic oracle. No claim of general stopping optimality or investable outperformance is made.

## Release checklist

Before calling this a fully reproducible release: (1) verify dataset redistribution permissions and document an authorized reconstruction path; (2) run tests and all reported audit commands in a clean environment; (3) verify table and figure regeneration; (4) record exact package versions and a release tag. Until then, this is a **source-and-results archive with a known data dependency**, not a certified end-to-end reproduction.

## License

Repository owner selected MIT for the repository. Ensure all authors and code rightsholders consent before distributing third-party or previously supplied code under that license. Third-party data are not automatically covered by MIT.
