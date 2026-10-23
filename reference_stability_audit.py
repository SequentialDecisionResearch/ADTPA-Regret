"""Reference stability audit on identical fixed12 historical decision states.

One-file add-on. Run from existing V3 project root:
    python reference_stability_audit.py --run
    python reference_stability_audit.py --run --max-states 2   # smoke test

No baseline files are written. Fixed24/48/96 use the same candidate selection,
training scenarios, and holdout seed prefix as Fixed12, with a single prespecified
look at their respective budgets. Student-t Bonferroni critical values are
approximate, not distribution-free. Fixed12 retains the original two-look-
adjusted threshold to reproduce the historical paired baseline. Differences in
critical values, eligibility, and stochastic Monte Carlo scores can affect the
selected action. High-budget references are not market ground truth.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import long_run_gate_pilot as pilot
import sequential_gate_validation as sequential
import real_market_oos_validation as market
import real_market_ablation as ablation
import paired_gate_audit as paired

BUDGETS = (24, 48, 96)
VARIANTS = ('fixed6', 'fixed12', 'adaptive6to12', 'fixed24', 'fixed48', 'fixed96')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_fixed_budget(n):
    if n not in BUDGETS:
        raise ValueError('Unrecognized fixed reference budget')
    source = inspect.getsource(pilot.choose_gate)
    needle1 = 'maxreps=FIXED_REPS'
    needle2 = "checkpoint=ADAPTIVE_FIRST if mode=='adaptive' else FIXED_REPS"
    if source.count(needle1) != 1 or source.count(needle2) != 1 or source.count('z*') != 3:
        raise RuntimeError('Gate source changed; refusing fragile substitution')
    source = source.replace(needle1, f'maxreps={n}')
    source = source.replace(needle2, f'checkpoint={n}')
    source = source.replace('z*', 'reference_critical(len(collected))*')
    # Single-look, 3-pair two-sided Bonferroni t approximation.
    def reference_critical(k):
        if k != n:
            raise AssertionError(f'Expected exactly {n} repetitions, received {k}')
        return float(student_t.ppf(1 - pilot.ALPHA / (2 * sequential.PAIRS), df=k - 1))
    namespace = dict(vars(pilot))
    namespace['reference_critical'] = reference_critical
    exec(compile(source, f'<fixed{n}_reference>', 'exec'), namespace)
    return namespace['choose_gate'], reference_critical(n)


def capture_fixed12_states(p, cfg):
    captured = []
    reference_trace = []
    original = pilot.eng.choose
    def wrapper(prices, t, shares, iyw_shares, cash, config, policy, decision_number):
        captured.append(dict(t=int(t), q=np.asarray(shares, float).copy(),
                             iq=float(iyw_shares), cash=float(cash),
                             decision_number=int(decision_number)))
        return sequential.adjusted_choose_gate(prices, t, shares, iyw_shares, cash,
                                                config, policy, decision_number, 'fixed', reference_trace)
    pilot.eng.choose = wrapper
    try:
        daily, decisions = pilot.eng.run(p, 'combined_dynamic', cfg)
    finally:
        pilot.eng.choose = original
    if len(captured) != len(decisions) or len(captured) != len(reference_trace):
        raise AssertionError('Captured state counts do not match fixed12 decisions')
    baseline = Path('real_market_ablation_outputs')
    for filename in ('fixed12_daily.csv', 'fixed12_decisions.csv', 'fixed12_gate_trace.csv'):
        if not (baseline / filename).is_file():
            raise FileNotFoundError(f'Missing {baseline / filename}; run real_market_ablation.py --run first')
    ablation.assert_same('fixed12 daily', daily.reset_index(),
                         pd.read_csv(baseline / 'fixed12_daily.csv'),
                         ['date', 'wealth', 'cash', 'fees', 'traded', 'nav', 'iyw_nav'])
    ablation.assert_same('fixed12 decisions', decisions,
                         pd.read_csv(baseline / 'fixed12_decisions.csv'),
                         ['date', 'horizon', 'feasible', 'action', 'iyw_new_weight', 'stock_new_weights'])
    ablation.assert_same('fixed12 trace', pd.DataFrame(reference_trace),
                         pd.read_csv(baseline / 'fixed12_gate_trace.csv'),
                         ['date', 'reps', 'reason', 'gate_action', 'changed'])
    return captured


def run(out, max_states):
    out = Path(out)
    forbidden = [market.DATA_FOLDER, Path('real_market_ablation_outputs'),
                 Path('paired_gate_audit_outputs'), Path('real_market_oos_outputs'),
                 Path('real_market_oos_outputs_before_nav_fix')]
    if out.resolve() in [x.resolve() for x in forbidden]:
        raise ValueError('Refusing to overwrite baseline outputs')
    p, data_manifest = market.load_snapshot(market.DATA_FOLDER)
    start = int(p.index.searchsorted(pd.Timestamp(market.EVALUATION_START)))
    cfg = pilot.eng.Config(seed=42, scenarios=pilot.BASE_SCENARIOS, start=start)
    captured = capture_fixed12_states(p, cfg)
    if max_states:
        captured = captured[:max_states]
    fixed6_fn, fixed6_crit = ablation.build_fixed6()
    high = {f'fixed{n}': build_fixed_budget(n) for n in BUDGETS}
    funcs = {'fixed6': (fixed6_fn, 'fixed'),
             'fixed12': (sequential.adjusted_choose_gate, 'fixed'),
             'adaptive6to12': (sequential.adjusted_choose_gate, 'adaptive')}
    funcs.update({name: (fn, 'fixed') for name, (fn, _) in high.items()})
    rows = []
    for idx, state in enumerate(captured):
        result = {}
        for name in VARIANTS:
            fn, mode = funcs[name]
            dec, tr = paired.evaluate(fn, mode, p, cfg, state)
            result[name] = (paired.ident(dec), tr)
        date = str(p.index[state['t']].date())
        for name in VARIANTS:
            ident, trace = result[name]
            row = dict(date=date, decision_number=state['decision_number'],
                       variant=name, action=ident[0], horizon=ident[1],
                       weights=';'.join(f'{v:.16g}' for v in ident[2]),
                       iyw_weight=ident[3], feasible=ident[4],
                       reps=int(trace['reps']), reason=str(trace['reason']),
                       state_cash=state['cash'])
            for ref in ('fixed12', 'fixed24', 'fixed48', 'fixed96'):
                row[f'same_as_{ref}'] = paired.same(ident, result[ref][0])
            rows.append(row)
        if (idx + 1) % 5 == 0 or idx + 1 == len(captured):
            print(f'Reference states evaluated: {idx+1}/{len(captured)}', flush=True)
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise AssertionError('No states')
    summary = []
    for name in VARIANTS:
        x = detail.loc[detail.variant == name]
        entry = dict(variant=name, states=len(x), total_reps=int(x.reps.sum()),
                     mean_reps=float(x.reps.mean()))
        for ref in ('fixed12', 'fixed24', 'fixed48', 'fixed96'):
            entry[f'agreement_{ref}'] = float(x[f'same_as_{ref}'].mean())
            entry[f'disagreements_{ref}'] = int((~x[f'same_as_{ref}']).sum())
        summary.append(entry)
    summary = pd.DataFrame(summary)
    # Compare references with each other: a large n need not imply convergence.
    refs = ('fixed12', 'fixed24', 'fixed48', 'fixed96')
    matrix = pd.DataFrame(index=refs, columns=refs, dtype=float)
    for a in refs:
        for b in refs:
            xa = detail[detail.variant == a].sort_values('decision_number')
            matrix.loc[a, b] = float(xa[f'same_as_{b}'].mean())
    # Guard against inadvertently changing earlier paired-audit outcomes.
    if max_states == 0:
        prior = Path('paired_gate_audit_outputs/paired_state_decisions.csv')
        if prior.exists():
            old = pd.read_csv(prior)
            for name in ('fixed6', 'fixed12', 'adaptive6to12'):
                a = detail[detail.variant == name].sort_values('decision_number').reset_index(drop=True)
                b = old[old.variant == name].sort_values('decision_number').reset_index(drop=True)
                if len(a) != len(b):
                    raise AssertionError(f'{name}: paired audit state count changed')
                for field in ('date', 'action', 'horizon', 'feasible', 'reps', 'reason'):
                    if not np.array_equal(a[field].astype(str), b[field].astype(str)):
                        raise AssertionError(f'{name}: mismatch against previous paired audit: {field}')
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / 'reference_state_decisions.csv', index=False)
    summary.to_csv(out / 'reference_stability_summary.csv', index=False)
    matrix.to_csv(out / 'reference_agreement_matrix.csv', index_label='reference')
    disagreements = detail[(detail.variant != 'fixed96') & (~detail.same_as_fixed96)]
    disagreements.to_csv(out / 'reference_disagreements_vs_fixed96.csv', index=False)
    manifest = dict(experiment='same-state high-budget reference stability audit',
                    anchor='fixed12 rollout, no alternative policy rollout',
                    snapshot_sha256=data_manifest['sha256'], script_sha256=sha256(__file__),
                    states=len(captured), max_states=max_states,
                    budgets=list(BUDGETS), fixed6_critical=fixed6_crit,
                    fixed12_critical=sequential.critical(12),
                    adaptive_critical_6=sequential.critical(6),
                    adaptive_critical_12=sequential.critical(12),
                    high_budget_criticals={name: crit for name, (_, crit) in high.items()},
                    limitations=['High-budget reference is not a truth label',
                                 'Single historical path; fixed12-visited states only',
                                 'Single-look critical values for 24/48/96 differ from two-look fixed12',
                                 'Holdout all-rep feasibility may change with budget',
                                 'Same seed prefixes across budgets; not independent reference samples',
                                 'Monte Carlo score t approximation, not an anytime-valid guarantee',
                                 'Ex-post stock universe and historical adjusted prices'])
    (out / 'reference_stability_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\nREFERENCE STABILITY AUDIT (fixed12 historical states)')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nHIGH-BUDGET REFERENCE AGREEMENT MATRIX')
    print(matrix.to_string(float_format=lambda x: f'{x:.4f}'))
    print('PASS: frozen snapshot and fixed12 baseline reproduced; original files unchanged.')
    print('Outputs:', out.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--out', default='reference_stability_audit_outputs')
    parser.add_argument('--max-states', type=int, default=0,
                        help='0=all states; small positive integer for smoke test')
    args = parser.parse_args()
    if args.max_states < 0:
        parser.error('--max-states must be nonnegative')
    run(args.out, args.max_states)
