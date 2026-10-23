"""Multi-seed reference stability audit; additive, read-only V3 experiment.

Run from the existing V3 project directory:
    python multi_seed_stability_audit.py --run
    python multi_seed_stability_audit.py --run --seeds 2 --max-states 2

Uses the 66 frozen fixed12-visited historical states. Training candidates and
portfolio states are held fixed; ONLY holdout Monte Carlo RNG streams vary.
For each seed, all budgets use nested common holdout prefixes. Single-look
fixed budgets use three-pair Bonferroni Student-t thresholds; adaptive uses
its original two-look thresholds. These are approximations, not guarantees.

This script does NOT edit project sources, frozen prices, or prior outputs.
Intermediate per-seed files are written atomically; --resume skips completed
seeds if their recorded design matches. Run time may be substantial.
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
import real_market_ablation as ablation
import paired_gate_audit as paired
import reference_stability_audit as reference

VARIANTS = ('fixed6','fixed12','adaptive6to12','fixed24','fixed48','fixed96','fixed192')
REFS = ('fixed48','fixed96','fixed192')
SEED_STEP = 10000019


def fixed192_gate():
    source = inspect.getsource(pilot.choose_gate)
    a = 'maxreps=FIXED_REPS'
    b = "checkpoint=ADAPTIVE_FIRST if mode=='adaptive' else FIXED_REPS"
    if source.count(a)!=1 or source.count(b)!=1 or source.count('z*')!=3:
        raise RuntimeError('Gate source changed; refusing dynamic patch')
    source=source.replace(a,'maxreps=192').replace(b,'checkpoint=192')
    source=source.replace('z*','large_critical(len(collected))*')
    def large_critical(n):
        if n!=192: raise AssertionError('Expected 192 replications')
        return float(student_t.ppf(1-pilot.ALPHA/(2*sequential.PAIRS),df=n-1))
    ns=dict(vars(pilot));ns['large_critical']=large_critical
    exec(compile(source,'<fixed192_gate>','exec'),ns)
    return ns['choose_gate']


def decision_signature(d):
    return (str(d['action']),int(d['horizon']),
            tuple(float(v) for v in np.asarray(d['weights'],float)),
            float(d['iyw_weight']),bool(d.get('feasible',True)))


def sig_string(d):
    # Exact canonical action signature (not a float-rounded display field).
    sig=decision_signature(d)
    return json.dumps([sig[0],sig[1],list(sig[2]),sig[3],sig[4]],separators=(',',':'))


def generate_seed(seed_index, states, prices, cfg, functions, out):
    # candidate_eval obtains original_scenarios through this global; the
    # engine's original choose/scenarios_at is NOT modified.
    original=pilot.original_scenarios
    offset=(seed_index+1)*SEED_STEP
    def independent_holdout(prices_, t, h, config, seed):
        # Training stream (~101m) stays fixed. Holdout stream (~202m) shifts.
        if seed >= 202000003:
            seed += offset
        return original(prices_,t,h,config,seed)
    pilot.original_scenarios=independent_holdout
    rows=[]
    try:
        for i,st in enumerate(states):
            decisions={}
            for name,(fn,mode) in functions.items():
                d,tr=paired.evaluate(fn,mode,prices,cfg,st)
                decisions[name]=(d,tr)
            for name in VARIANTS:
                d,tr=decisions[name]
                row=dict(seed_index=seed_index,seed_offset=offset,
                         state_index=i,date=str(prices.index[st['t']].date()),
                         decision_number=st['decision_number'],variant=name,
                         signature=sig_string(d),action=str(d['action']),
                         horizon=int(d['horizon']),feasible=bool(d.get('feasible',True)),
                         reps=int(tr['reps']),reason=str(tr['reason']))
                for ref in REFS:
                    row['same_'+ref]=paired.same(decision_signature(d),decision_signature(decisions[ref][0]))
                rows.append(row)
            if (i+1)%10==0 or i+1==len(states):
                print(f'Seed {seed_index+1}: states {i+1}/{len(states)}',flush=True)
    finally:
        pilot.original_scenarios=original
    frame=pd.DataFrame(rows)
    target=out/f'seed_{seed_index:03d}.csv'
    temp=out/f'seed_{seed_index:03d}.tmp'
    frame.to_csv(temp,index=False)
    os.replace(temp,target)
    return frame


def summarise(data, out, nseeds, nstates):
    # Pairwise budget comparisons are matched within (seed,state).
    per_seed=[]
    for (seed,name),x in data.groupby(['seed_index','variant'],sort=True):
        r=dict(seed_index=int(seed),variant=name,states=len(x),
               mean_reps=float(x.reps.mean()),total_reps=int(x.reps.sum()))
        for ref in REFS:
            r['agreement_'+ref]=float(x['same_'+ref].mean())
        per_seed.append(r)
    per_seed=pd.DataFrame(per_seed)
    per_seed.to_csv(out/'per_seed_summary.csv',index=False)
    agg=per_seed.groupby('variant',sort=False).agg(
        seeds=('seed_index','nunique'),mean_reps=('mean_reps','mean'),
        sd_reps=('mean_reps','std'),
        agreement_fixed48=('agreement_fixed48','mean'),
        agreement_fixed96=('agreement_fixed96','mean'),
        agreement_fixed192=('agreement_fixed192','mean')).reset_index()
    agg=agg.set_index('variant').loc[list(VARIANTS)].reset_index()
    agg.to_csv(out/'multi_seed_summary.csv',index=False)

    # Cross-seed modal decision fraction for each state and each variant.
    stability=[]
    for (state,name),x in data.groupby(['state_index','variant'],sort=True):
        counts=x.signature.value_counts()
        stability.append(dict(state_index=int(state),date=str(x.date.iloc[0]),
                              variant=name,n_seeds=len(x),unique_decisions=len(counts),
                              modal_fraction=float(counts.iloc[0]/len(x))))
    stability=pd.DataFrame(stability)
    stability.to_csv(out/'cross_seed_state_stability.csv',index=False)
    modal=stability.groupby('variant').agg(
        mean_modal_fraction=('modal_fraction','mean'),
        stable_states=('unique_decisions',lambda x:int((x==1).sum())),
        states=('state_index','count')).reset_index()
    modal=modal.set_index('variant').loc[list(VARIANTS)].reset_index()
    modal.to_csv(out/'cross_seed_stability_summary.csv',index=False)

    # Independent seed-to-seed agreement for the 96 and 192 budgets.
    # The nested within-seed 96/192 agreement is distinct from this quantity.
    cross=[]
    for name in ('fixed12','fixed48','fixed96','fixed192','adaptive6to12'):
        group=data[data.variant==name]
        for state,x in group.groupby('state_index'):
            values=x.sort_values('seed_index').signature.to_list()
            m=len(values)
            matches=sum(values[i]==values[j] for i in range(m) for j in range(i+1,m))
            cross.append(dict(variant=name,state_index=int(state),
                              pairwise_seed_agreement=(matches/(m*(m-1)/2) if m>1 else np.nan)))
    cross=pd.DataFrame(cross)
    cross.to_csv(out/'independent_seed_pairwise_agreement.csv',index=False)
    print('\nMULTI-SEED STABILITY AUDIT')
    print(agg.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nCROSS-SEED MODAL STABILITY')
    print(modal.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nIndependent-seed pairwise agreement:')
    print(cross.groupby('variant').pairwise_seed_agreement.mean().to_string(float_format=lambda x:f'{x:.4f}'))
    print(f'PASS: {nseeds} seed streams x {nstates} frozen states; no original output files changed.')
    print('Outputs:',out.resolve())


def run(args):
    out=Path(args.out)
    forbidden=[market.DATA_FOLDER,Path('real_market_oos_outputs'),
               Path('real_market_ablation_outputs'),Path('paired_gate_audit_outputs'),
               Path('reference_stability_audit_outputs')]
    if out.resolve() in [x.resolve() for x in forbidden]:
        raise ValueError('Refusing to overwrite existing experiment outputs')
    prices,manifest=market.load_snapshot(market.DATA_FOLDER)
    start=int(prices.index.searchsorted(pd.Timestamp(market.EVALUATION_START)))
    cfg=pilot.eng.Config(seed=42,scenarios=pilot.BASE_SCENARIOS,start=start)
    states=reference.capture_fixed12_states(prices,cfg)
    if args.max_states: states=states[:args.max_states]
    fixed6,_=ablation.build_fixed6()
    funcs={
        'fixed6':(fixed6,'fixed'),
        'fixed12':(sequential.adjusted_choose_gate,'fixed'),
        'adaptive6to12':(sequential.adjusted_choose_gate,'adaptive'),
    }
    for n in (24,48,96): funcs[f'fixed{n}']=(reference.build_fixed_budget(n)[0],'fixed')
    funcs['fixed192']=(fixed192_gate(),'fixed')
    design=dict(experiment='multi-seed holdout stability; frozen fixed12 states',
                snapshot_sha256=manifest['sha256'],
                n_seeds=args.seeds,n_states=len(states),seed_step=SEED_STEP,
                variants=list(VARIANTS),
                holdout_seed_only=True,training_candidates_fixed=True,
                nested_prefix_within_seed=True,
                source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                               for name in ('long_run_gate_pilot.py','sequential_gate_validation.py',
                                            'reference_stability_audit.py','real_market_ablation.py')},
                caveats=['Reference is not market truth',
                         'Fixed12-visited historical states only',
                         'Student-t intervals are approximate',
                         'Different thresholds for two-look and single-look rules',
                         'Training candidates are fixed across seeds',
                         'Seed replicates are Monte Carlo randomness, not independent markets'])
    out.mkdir(parents=True,exist_ok=True)
    mpath=out/'multi_seed_manifest.json'
    if mpath.exists():
        existing=json.loads(mpath.read_text(encoding='utf-8'))
        if existing!=design:
            raise RuntimeError('Existing output manifest differs; use a new --out folder')
        if not args.resume:
            raise RuntimeError('Output folder already initialized. Use --resume or another --out')
    else:
        if any(out.glob('seed_*.csv')):
            raise RuntimeError('Seed CSVs exist without matching manifest')
        mpath.write_text(json.dumps(design,indent=2),encoding='utf-8')
    frames=[]
    for seed in range(args.seeds):
        path=out/f'seed_{seed:03d}.csv'
        if args.resume and path.exists():
            frame=pd.read_csv(path)
            if len(frame)!=len(states)*len(VARIANTS) or set(frame.variant)!=set(VARIANTS):
                raise AssertionError(f'Corrupted/incomplete seed file: {path}')
            print(f'Seed {seed+1}/{args.seeds}: loaded checkpoint',flush=True)
        else:
            print(f'Seed {seed+1}/{args.seeds}: running',flush=True)
            frame=generate_seed(seed,states,prices,cfg,funcs,out)
        frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(out/'all_seed_state_decisions.csv',index=False)
    summarise(data,out,args.seeds,len(states))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true',required=True)
    ap.add_argument('--seeds',type=int,default=30)
    ap.add_argument('--max-states',type=int,default=0,
                    help='0=all 66; positive value for a smoke test')
    ap.add_argument('--out',default='multi_seed_stability_outputs')
    ap.add_argument('--resume',action='store_true',help='resume completed seeds in same output folder')
    args=ap.parse_args()
    if args.seeds<2 or args.max_states<0: ap.error('Require --seeds >= 2 and --max-states >= 0')
    run(args)
