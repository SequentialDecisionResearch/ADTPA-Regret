"""Budget-matched, multi-stage gate experiment; additive read-only V3 script.

Run: python budget_matched_multistage_audit.py --run
Smoke: python budget_matched_multistage_audit.py --run --seeds 2 --max-states 2 --out multistage_smoke
Resume: python budget_matched_multistage_audit.py --run --resume

New independent holdout seeds (indices 1000+), 66 frozen fixed12 states.
A prespecified four-look adaptive gate (6,12,24,48) with Bonferroni-t
thresholds is compared with original adaptive, fixed 6/7/12/24/48/96/192.
Budget-matched randomized fixed-6/12 and fixed-6/48 mixtures are evaluated
from independently randomized assignment per (seed,state), not outcome-based
selection. Mixtures match realized mean replication counts of adaptives at the
aggregate level; matching is post-hoc and descriptive, not confirmatory.
No project source files are changed. High-budget reference is not ground truth.
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
import multi_seed_stability_audit as multi

STAGES=(6,12,24,48)
VARIANTS=('fixed6','fixed7','fixed12','fixed24','fixed48','fixed96','fixed192',
          'adaptive6to12','adaptive6to48')
REFS=('fixed96','fixed192')
SEED_START=1000
SEED_STEP=multi.SEED_STEP


def fixed7_gate():
    src=inspect.getsource(pilot.choose_gate)
    a='maxreps=FIXED_REPS'
    b="checkpoint=ADAPTIVE_FIRST if mode=='adaptive' else FIXED_REPS"
    if src.count(a)!=1 or src.count(b)!=1 or src.count('z*')!=3:
        raise RuntimeError('Unexpected gate source; fail closed')
    src=src.replace(a,'maxreps=7').replace(b,'checkpoint=7')
    src=src.replace('z*','crit7(len(collected))*')
    def crit7(n):
        if n!=7: raise AssertionError('fixed7 checkpoint mismatch')
        return float(student_t.ppf(1-pilot.ALPHA/(2*sequential.PAIRS),df=n-1))
    ns=dict(vars(pilot),crit7=crit7)
    exec(compile(src,'<fixed7_gate>','exec'),ns)
    return ns['choose_gate']


def multistage_gate():
    src=inspect.getsource(pilot.choose_gate)
    replacements={
        'maxreps=FIXED_REPS':'maxreps=48',
        "checkpoint=ADAPTIVE_FIRST if mode=='adaptive' else FIXED_REPS":'checkpoint=6',
        'if rep+1 not in (checkpoint,maxreps):':'if rep+1 not in (6,12,24,48):',
        "if mode=='adaptive' and rep+1==checkpoint:":"if mode=='adaptive' and rep+1 in (6,12,24):",
        'z*':'multi_critical(len(collected))*',
    }
    for a,b in replacements.items():
        expected=3 if a=='z*' else 1
        if src.count(a)!=expected:
            raise RuntimeError(f'Gate source changed: {a!r} ({src.count(a)} vs {expected})')
        src=src.replace(a,b)
    def multi_critical(n):
        if n not in STAGES: raise AssertionError(f'Invalid checkpoint {n}')
        # Four predeclared looks, three pairwise contrasts, two-sided bound.
        return float(student_t.ppf(1-pilot.ALPHA/(2*len(STAGES)*sequential.PAIRS),df=n-1))
    ns=dict(vars(pilot),multi_critical=multi_critical)
    exec(compile(src,'<adaptive6to48_gate>','exec'),ns)
    return ns['choose_gate']


def generate_seed(seed_index, states, prices, cfg, funcs, out):
    original=pilot.original_scenarios
    # Independent of earlier multi-seed audit's indices 0..29.
    offset=(SEED_START+seed_index+1)*SEED_STEP
    def holdout(prices_,t,h,config,seed):
        if seed>=202000003: seed+=offset
        return original(prices_,t,h,config,seed)
    pilot.original_scenarios=holdout
    rows=[]
    try:
        for i,st in enumerate(states):
            results={name:paired.evaluate(fn,mode,prices,cfg,st)
                     for name,(fn,mode) in funcs.items()}
            for name in VARIANTS:
                dec,tr=results[name]
                row=dict(seed_index=seed_index,seed_offset=offset,state_index=i,
                         date=str(prices.index[st['t']].date()),
                         decision_number=int(st['decision_number']),variant=name,
                         signature=multi.sig_string(dec),action=str(dec['action']),
                         horizon=int(dec['horizon']),feasible=bool(dec.get('feasible',True)),
                         reps=int(tr['reps']),reason=str(tr['reason']))
                for ref in REFS:
                    row['same_'+ref]=paired.same(multi.decision_signature(dec),
                                                multi.decision_signature(results[ref][0]))
                rows.append(row)
            if (i+1)%10==0 or i+1==len(states):
                print(f'Seed {seed_index+1}: states {i+1}/{len(states)}',flush=True)
    finally:
        pilot.original_scenarios=original
    df=pd.DataFrame(rows)
    tmp=out/f'seed_{seed_index:03d}.tmp'
    df.to_csv(tmp,index=False)
    os.replace(tmp,out/f'seed_{seed_index:03d}.csv')
    return df


def summarize(data,out):
    byseed=data.groupby(['seed_index','variant'],sort=True).agg(
        mean_reps=('reps','mean'),agreement_fixed96=('same_fixed96','mean'),
        agreement_fixed192=('same_fixed192','mean')).reset_index()
    byseed.to_csv(out/'per_seed_summary.csv',index=False)
    overall=byseed.groupby('variant').agg(
        seeds=('seed_index','nunique'),mean_reps=('mean_reps','mean'),
        sd_reps=('mean_reps','std'),agreement_fixed96=('agreement_fixed96','mean'),
        agreement_fixed192=('agreement_fixed192','mean')).reindex(VARIANTS).reset_index()
    overall.to_csv(out/'multistage_summary.csv',index=False)
    print('\nMULTI-STAGE REFERENCE AUDIT')
    print(overall.to_string(index=False,float_format=lambda x:f'{x:.4f}'))

    # Exact fixed budget 7 is the natural direct comparator for 6->12.
    # Additional randomized fixed mixtures provide equal-mean-cost comparisons
    # with the multi-stage method; allocation is independent of outcomes.
    wide=data.pivot(index=['seed_index','state_index'],columns='variant',
                    values=['reps','same_fixed96','same_fixed192'])
    wide=wide.sort_index()
    rows=[]
    for target,low,high in [('adaptive6to12','fixed6','fixed12'),
                            ('adaptive6to48','fixed6','fixed48')]:
        cost=float(wide['reps'][target].mean())
        lc=float(wide['reps'][low].mean());hc=float(wide['reps'][high].mean())
        if not lc<=cost<=hc: raise AssertionError('Budget cannot be matched')
        prob=(cost-lc)/(hc-lc)
        for ref in REFS:
            actual=float(wide['same_'+ref][target].mean())
            expected=(1-prob)*float(wide['same_'+ref][low].mean())+prob*float(wide['same_'+ref][high].mean())
            # This is an analytically expected random fixed allocation; avoids
            # noise from a single random assignment and matches mean cost exactly.
            rows.append(dict(adaptive=target,reference=ref,
                             mean_budget=cost,comparator=f'random({low},{high})',
                             high_budget_probability=prob,
                             adaptive_agreement=actual,
                             expected_mixture_agreement=expected,
                             improvement=actual-expected))
    matches=pd.DataFrame(rows)
    matches.to_csv(out/'budget_matched_comparisons.csv',index=False)
    print('\nEQUAL EXPECTED BUDGET (fixed-mixture baseline)')
    print(matches.to_string(index=False,float_format=lambda x:f'{x:.4f}'))

    # Per-seed paired differences for a descriptive sampling interval. Seed
    # replicates share market states and are NOT independent market paths.
    uncertainty=[]
    for _,r in matches.iterrows():
        target=r.adaptive;ref=r.reference;low,high=r.comparator.removeprefix('random(').removesuffix(')').split(',')
        p=float(r.high_budget_probability)
        for seed,g in wide.groupby(level=0):
            a=float(g['same_'+ref][target].mean())
            b=(1-p)*float(g['same_'+ref][low].mean())+p*float(g['same_'+ref][high].mean())
            uncertainty.append(dict(seed_index=int(seed),adaptive=target,reference=ref,
                                    difference=a-b))
    unc=pd.DataFrame(uncertainty)
    unc.to_csv(out/'per_seed_budget_matched_differences.csv',index=False)
    ci=[]
    for (a,r),g in unc.groupby(['adaptive','reference']):
        v=g.difference.to_numpy(float)
        mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)))
        q=float(student_t.ppf(.975,len(v)-1))
        ci.append(dict(adaptive=a,reference=r,seeds=len(v),mean_difference=mean,
                       descriptive_ci_low=mean-q*se,descriptive_ci_high=mean+q*se))
    pd.DataFrame(ci).to_csv(out/'budget_matched_descriptive_intervals.csv',index=False)
    print('\nDescriptive seed-level paired intervals (not market-generalization CIs)')
    print(pd.DataFrame(ci).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('PASS: independent holdout streams; original project files unchanged.')
    print('Outputs:',out.resolve())


def run(args):
    out=Path(args.out)
    forbidden=[market.DATA_FOLDER,Path('real_market_oos_outputs'),
               Path('real_market_ablation_outputs'),Path('multi_seed_stability_outputs'),
               Path('reference_stability_audit_outputs')]
    if out.resolve() in [p.resolve() for p in forbidden]:
        raise ValueError('Refusing to overwrite existing experiment results')
    prices,manifest=market.load_snapshot(market.DATA_FOLDER)
    start=int(prices.index.searchsorted(pd.Timestamp(market.EVALUATION_START)))
    cfg=pilot.eng.Config(seed=42,scenarios=pilot.BASE_SCENARIOS,start=start)
    states=reference.capture_fixed12_states(prices,cfg)
    if args.max_states: states=states[:args.max_states]
    f6,_=ablation.build_fixed6()
    funcs={'fixed6':(f6,'fixed'),'fixed7':(fixed7_gate(),'fixed'),
           'fixed12':(sequential.adjusted_choose_gate,'fixed'),
           'adaptive6to12':(sequential.adjusted_choose_gate,'adaptive'),
           'adaptive6to48':(multistage_gate(),'adaptive')}
    for n in (24,48,96):funcs[f'fixed{n}']=(reference.build_fixed_budget(n)[0],'fixed')
    funcs['fixed192']=(multi.fixed192_gate(),'fixed')
    source_files=('long_run_gate_pilot.py','sequential_gate_validation.py',
                  'reference_stability_audit.py','real_market_ablation.py',
                  'multi_seed_stability_audit.py')
    design=dict(experiment='independent-seed equal-budget multistage gate',
                snapshot_sha256=manifest['sha256'],n_seeds=args.seeds,
                n_states=len(states),seed_start=SEED_START,seed_step=SEED_STEP,
                variants=list(VARIANTS),stages=list(STAGES),
                multistage_critical='t Bonferroni alpha/(2*4 looks*3 pairs)',
                candidate_training_fixed=True,holdout_nested_within_seed=True,
                seed_streams_independent_of_prior_30=True,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in source_files},
                caveats=['New seeds avoid reusing previous 30 audit streams',
                         'Only fixed12 historical states; not independent markets',
                         'Reference not market truth; feasibility all-rep attrition',
                         'Different look-adjusted critical values across methods',
                         'Bonferroni-t approximations require distributional assumptions',
                         'Posthoc expected-cost mixture comparison is descriptive',
                         'Multi-stage gate is a new exploratory method, not prespecified prior to earlier results'])
    out.mkdir(parents=True,exist_ok=True)
    mpath=out/'multistage_manifest.json'
    if mpath.exists():
        if json.loads(mpath.read_text(encoding='utf-8'))!=design:
            raise RuntimeError('Output manifest mismatch; choose new --out')
        if not args.resume:raise RuntimeError('Output exists; use --resume or new --out')
    else:
        if any(out.glob('seed_*.csv')):raise RuntimeError('Orphan seed files')
        mpath.write_text(json.dumps(design,indent=2),encoding='utf-8')
    frames=[]
    for seed in range(args.seeds):
        path=out/f'seed_{seed:03d}.csv'
        if args.resume and path.exists():
            frame=pd.read_csv(path)
            if len(frame)!=len(states)*len(VARIANTS) or set(frame.variant)!=set(VARIANTS):
                raise AssertionError('Invalid seed checkpoint: '+str(path))
            print(f'Seed {seed+1}/{args.seeds}: loaded checkpoint',flush=True)
        else:
            print(f'Seed {seed+1}/{args.seeds}: running',flush=True)
            frame=generate_seed(seed,states,prices,cfg,funcs,out)
        frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(out/'all_seed_state_decisions.csv',index=False)
    summarize(data,out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true',required=True)
    ap.add_argument('--seeds',type=int,default=30)
    ap.add_argument('--max-states',type=int,default=0)
    ap.add_argument('--out',default='budget_matched_multistage_outputs')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.seeds<2 or args.max_states<0:ap.error('Require --seeds >=2, --max-states >=0')
    run(args)
