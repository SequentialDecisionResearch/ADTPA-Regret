"""30-seed paired ADTPA V4 holdout-gate validation.

Place alongside long_run_gate_pilot.py in the V4 project root and run:
    python multiseed_gate_validation.py
    python multiseed_gate_validation.py --seeds 22 42 --fresh
    python multiseed_gate_validation.py --seed-start 1 --count 30

Reuses the pilot's actual engine.run integration, gate, ledger, and prefix checks.
No edits to V4 source. Synthetic data only; NOT a temporal out-of-sample test.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import long_run_gate_pilot as pilot

POLICIES = ('iyw', 'combined_dynamic', 'fixed_gate', 'adaptive_gate')
METRICS = ('terminal_wealth', 'relative_terminal', 'max_drawdown', 'trades', 'fees')
COMPARISONS = (
    ('fixed_gate', 'combined_dynamic'),
    ('adaptive_gate', 'combined_dynamic'),
    ('fixed_gate', 'iyw'),
    ('adaptive_gate', 'iyw'),
    ('adaptive_gate', 'fixed_gate'),
)


def fingerprint():
    """Resume only when relevant implementation and settings are identical."""
    h = hashlib.sha256()
    for f in (Path(__file__), Path(pilot.__file__), Path(pilot.eng.__file__)):
        h.update(f.read_bytes())
    for val in (pilot.BASE_SCENARIOS, pilot.TRAIN_N, pilot.HOLDOUT_N,
                pilot.FIXED_REPS, pilot.ADAPTIVE_FIRST, pilot.ALPHA,
                pilot.MIN_ADVANTAGE):
        h.update(repr(val).encode('ascii'))
    return h.hexdigest()


def save_csv_atomic(frame, target):
    target = Path(target)
    tmp = target.with_name(target.name + '.tmp')
    frame.to_csv(tmp, index=False)
    os.replace(tmp, target)


def run_seed(seed, out, save_daily=False):
    p = pilot.synthetic(n=700, d=6, seed=seed, regime='mixed')
    p = p[[c for c in p.columns if c != 'IYW'] + ['IYW']]
    cfg = pilot.eng.Config(seed=seed, scenarios=pilot.BASE_SCENARIOS, start=233)
    rows = []
    for policy in POLICIES:
        trace = []
        mode = 'fixed' if policy == 'fixed_gate' else 'adaptive'
        if policy.endswith('_gate'):
            def wrapper(prices, t, shares, iyw_shares, cash, config, pol, decision_number):
                return pilot.choose_gate(prices, t, shares, iyw_shares, cash,
                                         config, pol, decision_number, mode, trace)
            pilot.eng.choose = wrapper
            engine_policy = 'combined_dynamic'
        else:
            pilot.eng.choose = pilot.original_choose
            engine_policy = policy
        start = time.perf_counter()
        try:
            daily, decisions = pilot.eng.run(p, engine_policy, cfg)
        finally:
            pilot.eng.choose = pilot.original_choose
        elapsed = time.perf_counter() - start
        error = pilot.ledger_check(p, daily, decisions, cfg, engine_policy, trace)
        checks = pilot.prefix_test(p, cfg, mode, trace) if trace else 0
        if policy.endswith('_gate') and checks == 0:
            raise AssertionError('No prefix checks for gate policy')
        row = pilot.summary(seed, policy, daily, decisions, trace, error)
        row.update(prefix_checks=checks, runtime_seconds=elapsed,
                   gate_trace_rows=len(trace),
                   feasibility_abstentions=sum(r['gate_action'] == 'infeasible_fallback' for r in trace))
        rows.append(row)
        if save_daily:
            daily.to_csv(out / f'seed{seed}_{policy}_daily.csv')
            decisions.to_csv(out / f'seed{seed}_{policy}_decisions.csv', index=False)
        if trace:
            save_csv_atomic(pd.DataFrame(trace), out / f'seed{seed}_{policy}_gate_trace.csv')
        print(f'  {policy:18s} wealth={row["terminal_wealth"]:,.2f} '
              f'trades={row["trades"]:2d} changes={row["changed_decisions"]:2d} '
              f'reps={row["holdout_evaluations"]:3d} elapsed={elapsed:.1f}s', flush=True)
    return pd.DataFrame(rows)


def percentile_ci(x, rng, draws=10000):
    """Paired-across-seeds percentile bootstrap CI of mean difference."""
    a = np.asarray(x, dtype=float)
    n = len(a)
    if n < 2:
        return np.nan, np.nan
    # Chunked sampling avoids excessive memory when scaled up.
    means = []
    remaining = draws
    while remaining:
        k = min(1000, remaining)
        means.append(a[rng.integers(0, n, size=(k, n))].mean(axis=1))
        remaining -= k
    return tuple(np.quantile(np.concatenate(means), [0.025, 0.975]))


def aggregate(all_rows, output):
    if all_rows.empty:
        return
    wide = all_rows.pivot(index='seed', columns='policy', values=list(METRICS) +
                          ['changed_decisions', 'holdout_evaluations', 'runtime_seconds',
                           'max_accounting_error', 'prefix_checks'])
    complete = wide.dropna(subset=[('terminal_wealth', p) for p in POLICIES]).index
    wide = wide.loc[complete].sort_index()
    if len(wide) == 0:
        return
    details = pd.DataFrame({'seed': complete})
    for metric in METRICS + ('changed_decisions', 'holdout_evaluations', 'runtime_seconds',
                             'max_accounting_error', 'prefix_checks'):
        for policy in POLICIES:
            details[f'{policy}_{metric}'] = wide[(metric, policy)].to_numpy()
    rng = np.random.default_rng(20261022)
    summary = []
    for a, b in COMPARISONS:
        for metric in METRICS:
            x = (wide[(metric, a)] - wide[(metric, b)]).to_numpy(dtype=float)
            lo, hi = percentile_ci(x, rng)
            summary.append(dict(comparison=f'{a} - {b}', metric=metric,
                                seeds=len(x), mean_difference=float(x.mean()),
                                median_difference=float(np.median(x)),
                                std_difference=float(x.std(ddof=1)) if len(x)>1 else np.nan,
                                ci95_mean_lower=lo, ci95_mean_upper=hi,
                                positive_fraction=float((x>0).mean()),
                                negative_fraction=float((x<0).mean()),
                                tie_fraction=float((x==0).mean())))
            details[f'{a}_minus_{b}_{metric}'] = x
    save_csv_atomic(details, output / 'paired_seed_details.csv')
    save_csv_atomic(pd.DataFrame(summary), output / 'paired_comparison_summary.csv')
    policy_summary = (all_rows[all_rows.seed.isin(complete)]
                      .groupby('policy', as_index=False)
                      .agg(seeds=('seed','nunique'),
                           mean_wealth=('terminal_wealth','mean'),
                           median_wealth=('terminal_wealth','median'),
                           mean_relative=('relative_terminal','mean'),
                           mean_drawdown=('max_drawdown','mean'),
                           mean_trades=('trades','mean'),
                           mean_fees=('fees','mean'),
                           mean_changed=('changed_decisions','mean'),
                           mean_holdout_reps=('holdout_evaluations','mean'),
                           total_runtime_seconds=('runtime_seconds','sum'),
                           max_accounting_error=('max_accounting_error','max'),
                           total_prefix_checks=('prefix_checks','sum')))
    save_csv_atomic(policy_summary, output / 'policy_summary.csv')
    print('\n=== POLICY SUMMARY ===', flush=True)
    print(policy_summary.to_string(index=False, float_format=lambda x: f'{x:.5f}'), flush=True)
    print('\n=== PAIRED TERMINAL WEALTH DIFFERENCES ===', flush=True)
    s = pd.DataFrame(summary)
    t = s[s.metric == 'terminal_wealth']
    print(t[['comparison','seeds','mean_difference','median_difference',
             'ci95_mean_lower','ci95_mean_upper','positive_fraction']]
          .to_string(index=False, float_format=lambda x: f'{x:.5f}'), flush=True)
    print(f'\nCompleted seeds: {len(complete)}. Output: {output.resolve()}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, nargs='+', help='Explicit market seeds')
    ap.add_argument('--seed-start', type=int, default=1, help='First market seed (default 1)')
    ap.add_argument('--count', type=int, default=30, help='Number of seeds (default 30)')
    ap.add_argument('--out', default='multiseed_gate_outputs')
    ap.add_argument('--fresh', action='store_true', help='Discard saved per-seed summaries')
    ap.add_argument('--save-daily', action='store_true', help='Save large daily/decision CSVs')
    args = ap.parse_args()
    seeds = sorted(set(args.seeds if args.seeds is not None else
                       range(args.seed_start, args.seed_start + args.count)))
    if not seeds or any(x < 0 for x in seeds):
        ap.error('Supply at least one nonnegative seed')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = out / 'run_manifest.json'
    sig = fingerprint()
    manifest = {'fingerprint':sig, 'settings':dict(base_scenarios=pilot.BASE_SCENARIOS,
                training_scenarios=pilot.TRAIN_N, holdout_scenarios=pilot.HOLDOUT_N,
                fixed_repetitions=pilot.FIXED_REPS, adaptive_first=pilot.ADAPTIVE_FIRST,
                alpha=pilot.ALPHA, min_advantage=pilot.MIN_ADVANTAGE),
                'synthetic_regime':'mixed', 'n_days':700, 'n_stocks':6}
    if stamp.exists() and not args.fresh:
        prior = json.loads(stamp.read_text(encoding='utf-8'))
        if prior != manifest:
            raise RuntimeError('Code/settings changed since prior run. Use --fresh or --out NEW_DIR')
    if args.fresh:
        for seed in seeds:
            (out / f'seed{seed}_summary.csv').unlink(missing_ok=True)
    stamp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Requested seeds:',seeds,flush=True)
    print('Using tested V4 pilot execution, ledger and prefix checks.',flush=True)
    print('Resume is enabled; each seed is checkpointed independently.',flush=True)
    completed=[]
    try:
        for k,seed in enumerate(seeds,1):
            path = out / f'seed{seed}_summary.csv'
            if path.exists():
                cached = pd.read_csv(path)
                if set(cached.policy) == set(POLICIES) and len(cached)==len(POLICIES):
                    print(f'[{k}/{len(seeds)}] seed={seed} cached',flush=True)
                    completed.append(cached)
                    continue
                raise RuntimeError(f'Incomplete cached summary: {path}; rerun with --fresh')
            print(f'[{k}/{len(seeds)}] seed={seed} starting',flush=True)
            data = run_seed(seed,out,save_daily=args.save_daily)
            save_csv_atomic(data,path)
            completed.append(data)
            aggregate(pd.concat(completed,ignore_index=True),out)
    finally:
        pilot.eng.choose = pilot.original_choose
    print('\nWARNING: Results reflect synthetic paths, not genuine time-series OOS.',flush=True)
    print('WARNING: Adaptive interim intervals are not anytime-valid.',flush=True)
    print('WARNING: Across-seed bootstrap CI assumes sampled seeds represent the target generator.',flush=True)

if __name__ == '__main__':
    main()
