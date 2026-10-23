# Real-market ablation (NAV-corrected V3)

This package is self-contained. The original frozen adjusted-close snapshot and its SHA-256 manifest are included. No download or manual file edits are required.

From Anaconda Prompt:

```bat
cd /d C:\lsg\hedge_fund_iyw_benchmark\ADTPA_IYW_Regret_Experiments_V3
python real_market_ablation.py --run
```

Outputs are in `real_market_ablation_outputs/`. Existing `real_market_oos_outputs/` and `real_market_oos_outputs_before_nav_fix/` are not modified by this command.

Comparison: Fixed-6 (one predeclared look at 6), Fixed-12 (one look at 12), Adaptive 6->12 (two looks). Three action-pair comparisons; t-based Bonferroni intervals approximate, not distribution-free. All variants reuse the same 2015-2024 frozen Yahoo snapshot, fixed ex-post stock universe, engine settings and random seed. The Fixed-6 one-look t threshold differs from the adaptive first-look threshold, intentionally reflecting the number of looks. This comparison therefore varies both sampling budget and multiplicity correction; it is not a pure threshold-controlled isolation of stopping.

The program independently checks accounting and prefix invariance, and requires Fixed-12 and Adaptive to reproduce original decision/trace/wealth baselines. Outputs include daily paths, decisions, traces, summary, comparisons, and experiment manifest.

Caveats: one retrospective market path, selection/survivorship bias, daily adjusted-close execution assumption, bootstrap model misspecification. A single path cannot establish statistical superiority of a trading strategy. No live trading claims.
