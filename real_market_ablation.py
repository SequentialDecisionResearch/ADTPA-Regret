"""Prespecified real-market gate budget ablation: fixed6, fixed12, adaptive6to12.

Usage: python real_market_ablation.py --run
Reuses verified frozen historical data, never downloads, never alters baseline outputs.
Research caveat: fixed6 is a lower-budget policy with different statistical power;
Bonferroni thresholds are approximate t-based and not anytime-valid.
"""
import argparse
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import long_run_gate_pilot as pilot
import sequential_gate_validation as sequential
import real_market_oos_validation as market

VARIANTS = ('fixed6', 'fixed12', 'adaptive6to12')


def build_fixed6():
    """Use identical gate logic, but stop at 6 and calibrate one-look family of 3 contrasts.

    Importantly, critical value differs from adaptive first look: fixed6 has one
    predeclared look, whereas adaptive has two. We do NOT simply truncate adaptive.
    """
    src = inspect.getsource(pilot.choose_gate)
    assert src.count('maxreps=FIXED_REPS') == 1
    assert src.count("checkpoint=ADAPTIVE_FIRST if mode=='adaptive' else FIXED_REPS") == 1
    assert src.count('z*') == 3
    src = src.replace('maxreps=FIXED_REPS', 'maxreps=6')
    src = src.replace("checkpoint=ADAPTIVE_FIRST if mode=='adaptive' else FIXED_REPS", 'checkpoint=6')
    src = src.replace('z*', 'fixed6_critical(len(collected))*')
    from scipy.stats import t as student_t
    def fixed6_critical(n):
        if n != 6:
            raise AssertionError('fixed6 has exactly one look at n=6')
        return float(student_t.ppf(1 - pilot.ALPHA / (2 * sequential.PAIRS), df=5))
    ns = dict(vars(pilot))
    ns['fixed6_critical'] = fixed6_critical
    exec(compile(src, '<fixed6_prespecified_gate>', 'exec'), ns)
    return ns['choose_gate'], fixed6_critical(6)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_same(label, a, b, fields):
    if len(a) != len(b):
        raise AssertionError(f'{label}: row count differs {len(a)} vs {len(b)}')
    for col in fields:
        if col not in a or col not in b:
            raise AssertionError(f'{label}: missing column {col}')
        x, y = a[col].to_numpy(), b[col].to_numpy()
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            if not np.allclose(x.astype(float), y.astype(float), rtol=1e-10, atol=1e-7, equal_nan=True):
                raise AssertionError(f'{label}: numeric mismatch in {col}')
        else:
            if not np.array_equal(a[col].fillna('<NA>').astype(str), b[col].fillna('<NA>').astype(str)):
                raise AssertionError(f'{label}: mismatch in {col}')


def run(out):
    out = Path(out)
    data_dir = market.DATA_FOLDER
    p, data_manifest = market.load_snapshot(data_dir)
    start = int(p.index.searchsorted(pd.Timestamp(market.EVALUATION_START)))
    cfg = pilot.eng.Config(seed=42, scenarios=pilot.BASE_SCENARIOS, start=start)
    out.mkdir(parents=True, exist_ok=True)
    fixed6_fn, fixed6_crit = build_fixed6()
    summaries = []
    baselines = {'fixed12':'fixed_gate', 'adaptive6to12':'adaptive_gate'}
    for variant in VARIANTS:
        trace = []
        fn = fixed6_fn if variant == 'fixed6' else sequential.adjusted_choose_gate
        mode = 'adaptive' if variant == 'adaptive6to12' else 'fixed'
        def wrapper(prices, t, shares, iyw_shares, cash, config, policy, decision_number):
            return fn(prices,t,shares,iyw_shares,cash,config,policy,decision_number,mode,trace)
        pilot.eng.choose = wrapper
        try:
            daily, decisions = pilot.eng.run(p, 'combined_dynamic', cfg)
        finally:
            pilot.eng.choose = pilot.original_choose
        err = pilot.ledger_check(p, daily, decisions, cfg, 'combined_dynamic', trace)
        checks = pilot.prefix_test(p, cfg, variant, trace)
        s = pilot.summary(42, variant, daily, decisions, trace, err)
        s['nav_max_drawdown'] = float((daily.nav/daily.nav.cummax()-1).min())
        s['terminal_nav'] = float(daily.nav.iloc[-1])
        s['prefix_checks'] = checks
        s['data_sha256'] = data_manifest['sha256']
        s['gate_reason_counts'] = json.dumps(pd.Series([x['reason'] for x in trace]).value_counts().to_dict(),sort_keys=True)
        if variant in baselines:
            old = baselines[variant]
            old_folder = data_dir if (data_dir/f'{old}_daily.csv').exists() else Path('real_market_oos_outputs_before_nav_fix')
            if not (old_folder/f'{old}_daily.csv').exists():
                raise RuntimeError(f'Missing original baseline {old} for exact reproduction check')
            d0 = pd.read_csv(old_folder/f'{old}_daily.csv')
            c0 = pd.read_csv(old_folder/f'{old}_decisions.csv')
            t0 = pd.read_csv(old_folder/f'{old}_gate_trace.csv')
            assert_same(variant+' daily',daily.reset_index(),d0,['date','wealth','cash','fees','traded','nav','iyw_nav']) if old_folder == data_dir else assert_same(variant+' daily',daily.reset_index(),d0,['date','wealth','cash','fees','traded'])
            assert_same(variant+' decisions',decisions,c0,['date','horizon','feasible','action','iyw_new_weight','stock_new_weights'])
            assert_same(variant+' trace',pd.DataFrame(trace),t0,['date','reps','reason','gate_action','changed'])
            s['reproduces_original'] = True
        else:
            s['reproduces_original'] = None
        summaries.append(s)
        daily.to_csv(out/f'{variant}_daily.csv')
        decisions.to_csv(out/f'{variant}_decisions.csv',index=False)
        pd.DataFrame(trace).to_csv(out/f'{variant}_gate_trace.csv',index=False)
        print(f'{variant:16s} wealth={s["terminal_wealth"]:,.2f} NAV_DD={s["nav_max_drawdown"]:.2%} '
              f'reps={s["holdout_evaluations"]} decisions={s["decisions"]} '
              f'ledger_err={err:.8f} prefix={checks}',flush=True)
    tab = pd.DataFrame(summaries)
    tab.to_csv(out/'ablation_summary.csv',index=False)
    reference = tab.set_index('policy')
    ref = reference.loc['fixed12']
    comparisons=[]
    for name in ('fixed6','adaptive6to12'):
        r=reference.loc[name]
        comparisons.append({'comparison':f'{name} minus fixed12',
                            'terminal_wealth_delta':float(r.terminal_wealth-ref.terminal_wealth),
                            'nav_drawdown_delta_pp':float((r.nav_max_drawdown-ref.nav_max_drawdown)*100),
                            'holdout_reps_saved':int(ref.holdout_evaluations-r.holdout_evaluations),
                            'holdout_reps_saved_pct':float(100*(1-r.holdout_evaluations/ref.holdout_evaluations))})
    pd.DataFrame(comparisons).to_csv(out/'ablation_comparisons.csv',index=False)
    manifest={'source_snapshot_sha256':data_manifest['sha256'], 'experiment':'prespecified_gate_budget_ablation',
              'fixed6_critical':fixed6_crit,
              'fixed12_critical':sequential.critical(12),
              'adaptive_critical_6':sequential.critical(6),
              'adaptive_critical_12':sequential.critical(12),
              'fixed6_looks':[6], 'fixed12_looks':[12], 'adaptive_looks':[6,12],
              'pairs':sequential.PAIRS, 'alpha':pilot.ALPHA,
              'fixed6_familywise_approximation':'single-look Bonferroni t; not exact under arbitrary distributions',
              'adaptive_familywise_approximation':'two-look Bonferroni t; not anytime-valid',
              'script_sha256':sha256(__file__), 'config':vars(cfg),
              'limitations':['single retrospective path','ex-post stock universe','no real-world causal inference',
                             'different stopping rules imply different test multiplicity correction',
                             'Holdout repetitions are Monte Carlo evaluations, not independent market paths']}
    (out/'ablation_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print('PASS: fixed12/adaptive reproduce prior baseline; ablation files saved:',out.resolve(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--out',default='real_market_ablation_outputs')
    args=parser.parse_args()
    run(args.out)
