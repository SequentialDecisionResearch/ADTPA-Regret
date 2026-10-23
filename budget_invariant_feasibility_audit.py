"""Final bounded diagnostic: all-rep vs fixed-prefix eligibility on frozen states.

Run: python budget_invariant_feasibility_audit.py --run
Smoke: python budget_invariant_feasibility_audit.py --run --seeds 1 --max-states 2 --out feasibility_smoke

No edits to source files. Fixed-prefix eligibility uses ONLY the first six
holdout replications for every budget; ALL available replications still inform
score comparisons. Original all-rep eligibility is preserved as comparator.
The test does not validate true chance constraints, returns, or generalization.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import long_run_gate_pilot as pilot
import sequential_gate_validation as sequential
import real_market_oos_validation as market
import reference_stability_audit as reference
import paired_gate_audit as paired
import multi_seed_stability_audit as multi

BUDGETS=(6,12,48,96)
PREFIX=6
SEED_STEP=multi.SEED_STEP
SEED_START=3000


def make_gate(budget,prefix):
    src=inspect.getsource(pilot.choose_gate)
    replacements={
        'maxreps=FIXED_REPS':f'maxreps={budget}',
        "checkpoint=ADAPTIVE_FIRST if mode=='adaptive' else FIXED_REPS":f'checkpoint={budget}',
        'z*':'audit_critical(len(collected))*',
    }
    if prefix:
        replacements['all(r[a][1] for r in collected)']=f'all(r[a][1] for r in collected[:{PREFIX}])'
    for a,b in replacements.items():
        expect=3 if a=='z*' else 1
        if src.count(a)!=expect:raise RuntimeError(f'Unexpected source pattern: {a}: {src.count(a)}')
        src=src.replace(a,b)
    def audit_critical(n):
        if n!=budget:raise AssertionError('Unexpected checkpoint')
        return float(student_t.ppf(1-pilot.ALPHA/(2*sequential.PAIRS),df=n-1))
    ns=dict(vars(pilot),audit_critical=audit_critical)
    exec(compile(src,f'<budget_{budget}_prefix_{prefix}>','exec'),ns)
    return ns['choose_gate']


def one_seed(seed_idx,states,prices,cfg,gates,out):
    original=pilot.original_scenarios
    offset=(SEED_START+seed_idx+1)*SEED_STEP
    def shifted(p,t,h,config,seed):
        return original(p,t,h,config,seed+(offset if seed>=202000003 else 0))
    pilot.original_scenarios=shifted
    rows=[]
    try:
        for i,st in enumerate(states):
            for budget in BUDGETS:
                for rule in ('all_reps','fixed_prefix6'):
                    dec,tr=paired.evaluate(gates[(budget,rule)],'fixed',prices,cfg,st)
                    rows.append(dict(seed_index=seed_idx,state_index=i,budget=budget,rule=rule,
                                     signature=multi.sig_string(dec),action=str(dec['action']),
                                     feasible=bool(dec.get('feasible',False)),
                                     fallback=str(dec['action'])=='infeasible_fallback',
                                     reps=int(tr['reps']),reason=str(tr['reason'])))
            if (i+1)%10==0 or i+1==len(states):
                print(f'Seed {seed_idx+1}: {i+1}/{len(states)} states',flush=True)
    finally:
        pilot.original_scenarios=original
    df=pd.DataFrame(rows)
    tmp=out/f'seed_{seed_idx:03d}.tmp'
    df.to_csv(tmp,index=False)
    os.replace(tmp,out/f'seed_{seed_idx:03d}.csv')
    return df


def summarize(data,out):
    per=data.groupby(['seed_index','rule','budget'],as_index=False).agg(
        fallback_rate=('fallback','mean'),feasible_rate=('feasible','mean'),
        mean_reps=('reps','mean'))
    per.to_csv(out/'per_seed_feasibility.csv',index=False)
    summary=per.groupby(['rule','budget'],as_index=False).agg(
        seeds=('seed_index','nunique'),fallback_rate=('fallback_rate','mean'),
        feasible_rate=('feasible_rate','mean'),mean_reps=('mean_reps','mean'))
    summary.to_csv(out/'feasibility_by_budget.csv',index=False)
    print('\nFEASIBILITY BY BUDGET')
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    wide=data.pivot(index=['seed_index','state_index','budget'],columns='rule',values=['signature','fallback'])
    both=wide.dropna().copy()
    both.columns=['_'.join(c) for c in both.columns]
    both['changed']=both['signature_all_reps']!=both['signature_fixed_prefix6']
    both['rescued']=both['fallback_all_reps'].astype(bool)&~both['fallback_fixed_prefix6'].astype(bool)
    both['lost']=~both['fallback_all_reps'].astype(bool)&both['fallback_fixed_prefix6'].astype(bool)
    comparison=both.reset_index().groupby('budget').agg(
        n=('changed','size'),decision_changed=('changed','sum'),
        fallback_rescued=('rescued','sum'),fallback_new=('lost','sum')).reset_index()
    comparison['decision_changed_rate']=comparison['decision_changed']/comparison['n']
    comparison.to_csv(out/'paired_rule_comparison.csv',index=False)
    print('\nPAIRED RULE COMPARISON')
    print(comparison.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    # Within each rule, budget transitions: are fallback statuses nested?
    transitions=[]
    for rule in ('all_reps','fixed_prefix6'):
        f=data[data.rule==rule].pivot(index=['seed_index','state_index'],columns='budget',values='fallback')
        for a,b in zip(BUDGETS[:-1],BUDGETS[1:]):
            x=f[a].astype(bool);y=f[b].astype(bool)
            transitions.append(dict(rule=rule,from_budget=a,to_budget=b,
                                    fallback_both=int((x&y).sum()),
                                    newly_fallback=int((~x&y).sum()),
                                    recovered_from_fallback=int((x&~y).sum()),
                                    neither_fallback=int((~x&~y).sum())))
    trans=pd.DataFrame(transitions)
    trans.to_csv(out/'fallback_transitions.csv',index=False)
    print('\nFALLBACK TRANSITIONS')
    print(trans.to_string(index=False))
    print('\nPASS: fixed-prefix feasibility diagnostic complete; original files unchanged.')
    print('Outputs:',out.resolve())


def run(args):
    out=Path(args.out)
    forbidden={'real_market_ablation_outputs','budget_matched_multistage_outputs',
               'decision_regret_budget_frontier_outputs','common_support_audit_outputs'}
    if out.name in forbidden:raise ValueError('Refusing to overwrite earlier results')
    prices,manifest=market.load_snapshot(market.DATA_FOLDER)
    start=int(prices.index.searchsorted(pd.Timestamp(market.EVALUATION_START)))
    cfg=pilot.eng.Config(seed=42,scenarios=pilot.BASE_SCENARIOS,start=start)
    states=reference.capture_fixed12_states(prices,cfg)
    if args.max_states:states=states[:args.max_states]
    design=dict(snapshot_sha256=manifest['sha256'],seed_start=SEED_START,
                seed_step=SEED_STEP,seeds=args.seeds,states=len(states),
                budgets=list(BUDGETS),fixed_prefix=PREFIX,
                training_unchanged=True,threshold='single-look Bonferroni Student-t',
                caveats=['Frozen fixed12-visited states, not independent market paths',
                         'Eligibility is not a chance constraint confidence bound',
                         'Fixed-prefix isolates budget-dependent eligibility only',
                         'Training feasibility still affects the candidate set',
                         'No claim of financial performance or statistical optimality'],
                source_hashes={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
                               ('long_run_gate_pilot.py','reference_stability_audit.py',
                                'multi_seed_stability_audit.py')})
    out.mkdir(parents=True,exist_ok=True)
    path=out/'design.json'
    if path.exists():
        if json.loads(path.read_text())!=design:raise RuntimeError('Design mismatch; use a new --out')
        if not args.resume:raise RuntimeError('Output exists; use --resume or new --out')
    else:
        if list(out.glob('seed_*.csv')):raise RuntimeError('Orphan seed files')
        path.write_text(json.dumps(design,indent=2),encoding='utf-8')
    gates={(n,r):make_gate(n,r=='fixed_prefix6') for n in BUDGETS for r in ('all_reps','fixed_prefix6')}
    frames=[]
    for seed_idx in range(args.seeds):
        dest=out/f'seed_{seed_idx:03d}.csv'
        if args.resume and dest.is_file():
            frame=pd.read_csv(dest)
            if len(frame)!=len(states)*len(BUDGETS)*2:raise RuntimeError(f'Incomplete {dest}')
            print(f'Reusing {dest.name}',flush=True)
        else:frame=one_seed(seed_idx,states,prices,cfg,gates,out)
        frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(out/'all_decisions.csv',index=False)
    summarize(data,out)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--max-states',type=int,default=0)
    parser.add_argument('--out',default='budget_invariant_feasibility_outputs')
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args()
    if not a.run:parser.print_help()
    elif a.seeds<1 or a.max_states<0:parser.error('Invalid seed/state count')
    else:run(a)
