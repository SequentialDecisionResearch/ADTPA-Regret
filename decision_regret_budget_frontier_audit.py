"""Independent-score Decision Regret & Budget Frontier Audit (additive V3 script).

Requires completed budget_matched_multistage_outputs/all_seed_state_decisions.csv.
Run: python decision_regret_budget_frontier_audit.py --run
Smoke: python decision_regret_budget_frontier_audit.py --run --seeds 2 --max-states 2 --eval-reps 3 --out regret_smoke
Resume: python decision_regret_budget_frontier_audit.py --run --resume

A *new, independent* Monte Carlo evaluation stream estimates frozen candidate
scores; original selection/holdout streams are not reused. Score-regret is a
simulation diagnostic, NOT realized trading P&L, causal regret, or an unbiased
estimate of true optimal regret. Candidate selection and feasibility filtering
make the best-score proxy optimistic; report both raw and feasible-only scores.
No source, market snapshot, or previous output is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import long_run_gate_pilot as pilot
import real_market_oos_validation as market
import reference_stability_audit as reference
import budget_matched_multistage_audit as previous

VARIANTS=previous.VARIANTS
EVAL_SEED_BASE=801000000
EVAL_SEED_STEP=10000019
EVAL_REP_STEP=1000003


def write_atomic(frame,path):
    temp=path.with_suffix('.tmp')
    frame.to_csv(temp,index=False)
    os.replace(temp,path)


def frozen_candidates(prices,cfg,st):
    training=pilot.candidate_eval(prices,st['t'],st['q'].copy(),st['iq'],st['cash'],
                                  cfg,st['decision_number'],pilot.TRAIN_N,101000003)
    frozen={}
    for action in pilot.ACTIONS:
        eligible=[(i,c) for i,c in enumerate(training) if c['action']==action and c['feasible']]
        if eligible:
            i,c=max(eligible,key=lambda z:z[1]['score'])
            frozen[action]=(i,pilot.identity(c))
    return training,frozen


def independent_evaluation(prices,cfg,st,seed_index,reps,training,frozen):
    # Independent of training 101m, selection 202m+offset, and other eval seeds.
    action_data={a:[] for a in frozen}
    for rep in range(reps):
        shift=EVAL_SEED_BASE+(seed_index+1)*EVAL_SEED_STEP+rep*EVAL_REP_STEP
        evaluated=pilot.candidate_eval(prices,st['t'],st['q'].copy(),st['iq'],st['cash'],
                                       cfg,st['decision_number'],pilot.HOLDOUT_N,shift)
        if len(evaluated)!=len(training):
            raise AssertionError('Candidate count changed')
        for a,(idx,identity) in frozen.items():
            c=evaluated[idx]
            if pilot.identity(c)!=identity:
                raise AssertionError('Frozen candidate identity changed: '+a)
            action_data[a].append((float(c['score']),bool(c['feasible'])))
    scores={a:float(np.mean([x[0] for x in obs])) for a,obs in action_data.items()}
    feasible={a:bool(all(x[1] for x in obs)) for a,obs in action_data.items()}
    return scores,feasible


def generate_seed(seed_index,states,prices,cfg,source,out,eval_reps):
    seed_rows=source[source.seed_index==seed_index]
    rows=[]
    for i,st in enumerate(states):
        selected=seed_rows[seed_rows.state_index==i]
        if len(selected)!=len(VARIANTS) or set(selected.variant)!=set(VARIANTS):
            raise AssertionError(f'Missing/duplicate selection seed={seed_index} state={i}')
        training,frozen=frozen_candidates(prices,cfg,st)
        scores,feasible=independent_evaluation(prices,cfg,st,seed_index,eval_reps,training,frozen)
        if scores:
            best=max(scores.values())
            feasible_scores={a:s for a,s in scores.items() if feasible[a]}
            best_feasible=max(feasible_scores.values()) if feasible_scores else np.nan
        else:
            best=best_feasible=np.nan
        ref_action=str(selected.set_index('variant').loc['fixed192','action'])
        ref_score=scores.get(ref_action,np.nan)
        for _,r in selected.iterrows():
            a=str(r.action)
            score=scores.get(a,np.nan)
            valid=np.isfinite(score)
            # Scores are candidate utility scores (NOT realized dollars).
            # The unconstrained max is an optimistic in-sample action oracle.
            regret=max(0.,best-score) if valid else np.nan
            feas_regret=max(0.,best_feasible-score) if valid and feasible.get(a,False) and np.isfinite(best_feasible) else np.nan
            rows.append(dict(seed_index=seed_index,state_index=i,date=str(prices.index[st['t']].date()),
                             variant=r.variant,reps=int(r.reps),action=a,
                             same_fixed192=bool(r.same_fixed192),same_fixed96=bool(r.same_fixed96),
                             candidate_score=score,reference_score=ref_score,
                             signed_reference_score_gap=(ref_score-score if valid and np.isfinite(ref_score) else np.nan),
                             oracle_score=best,score_regret=regret,
                             feasible_oracle_score=best_feasible,feasible_score_regret=feas_regret,
                             evaluation_feasible=feasible.get(a,False),
                             score_covered=valid,eval_reps=eval_reps,
                             score_unit='engine_candidate_score'))
        if (i+1)%10==0 or i+1==len(states):
            print(f'Evaluation seed {seed_index+1}: states {i+1}/{len(states)}',flush=True)
    df=pd.DataFrame(rows)
    write_atomic(df,out/f'eval_seed_{seed_index:03d}.csv')
    return df


def summarize(data,out):
    per_seed=data.groupby(['seed_index','variant'],sort=True).agg(
        mean_budget=('reps','mean'),reference_agreement=('same_fixed192','mean'),
        mean_score_regret=('score_regret','mean'),
        mean_feasible_regret=('feasible_score_regret','mean'),
        mean_reference_score_gap=('signed_reference_score_gap','mean'),
        score_coverage=('score_covered','mean'),
        evaluation_feasible_rate=('evaluation_feasible','mean')).reset_index()
    write_atomic(per_seed,out/'per_seed_regret_summary.csv')
    summary=per_seed.groupby('variant').agg(
        seeds=('seed_index','nunique'),mean_reps=('mean_budget','mean'),
        agreement_fixed192=('reference_agreement','mean'),
        mean_score_regret=('mean_score_regret','mean'),
        mean_feasible_regret=('mean_feasible_regret','mean'),
        mean_reference_score_gap=('mean_reference_score_gap','mean'),
        score_coverage=('score_coverage','mean'),
        eval_feasible_rate=('evaluation_feasible_rate','mean')).reindex(VARIANTS).reset_index()
    write_atomic(summary,out/'regret_budget_frontier.csv')
    print('\nDECISION REGRET AND BUDGET FRONTIER (independent evaluation scores)')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))

    # Analytical equal-expected-budget random fixed allocations, state-independent.
    # Compute regret from seed-state paired records, with coverage carefully tracked.
    wide=data.pivot(index=['seed_index','state_index'],columns='variant',
                    values=['reps','score_regret','feasible_score_regret','signed_reference_score_gap','score_covered'])
    comparisons=[];seed_differences=[]
    for target,low,high in [('adaptive6to12','fixed6','fixed12'),
                            ('adaptive6to48','fixed6','fixed48')]:
        cost=float(wide['reps'][target].mean())
        p=(cost-float(wide['reps'][low].mean()))/(float(wide['reps'][high].mean())-float(wide['reps'][low].mean()))
        if not 0<=p<=1:raise AssertionError('Invalid matched-budget mixture')
        for metric in ('score_regret','feasible_score_regret','signed_reference_score_gap'):
            # Per-row paired comparisons only where all three outcomes defined.
            x=wide[metric][[target,low,high]].apply(pd.to_numeric,errors='coerce').dropna()
            if x.empty:continue
            diff=x[target]-((1-p)*x[low]+p*x[high])
            for seed,g in diff.groupby(level=0):
                seed_differences.append(dict(seed_index=int(seed),adaptive=target,metric=metric,
                                             mean_difference=float(g.mean()),matched_states=len(g)))
            by_seed=diff.groupby(level=0).mean()
            m=float(by_seed.mean());sd=float(by_seed.std(ddof=1)) if len(by_seed)>1 else np.nan
            q=float(student_t.ppf(.975,len(by_seed)-1)) if len(by_seed)>1 else np.nan
            margin=q*sd/np.sqrt(len(by_seed)) if len(by_seed)>1 else np.nan
            comparisons.append(dict(adaptive=target,metric=metric,reference='fixed192',
                                    comparator=f'random({low},{high})',mean_budget=cost,
                                    high_budget_probability=p,matched_seed_states=len(x),
                                    adaptive_mean=float(x[target].mean()),
                                    mixture_expected_mean=float(((1-p)*x[low]+p*x[high]).mean()),
                                    difference_adaptive_minus_mixture=m,
                                    descriptive_ci_low=m-margin,descriptive_ci_high=m+margin,
                                    seeds=len(by_seed)))
    comp=pd.DataFrame(comparisons)
    write_atomic(comp,out/'budget_matched_regret_comparisons.csv')
    write_atomic(pd.DataFrame(seed_differences),out/'per_seed_regret_differences.csv')
    print('\nEQUAL-BUDGET SCORE COMPARISONS (negative regret difference favors adaptive)')
    print(comp.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    # Distribution of regrets conditional on disagreement, not just agreement rate.
    disag=data[~data.same_fixed192]
    diag=disag.groupby('variant').agg(disagreements=('state_index','count'),
        mean_score_regret=('score_regret','mean'),
        median_score_regret=('score_regret','median'),
        mean_signed_reference_gap=('signed_reference_score_gap','mean')).reindex(VARIANTS).reset_index()
    write_atomic(diag,out/'disagreement_severity.csv')
    print('\nDISAGREEMENT SEVERITY')
    print(diag.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nPASS: independent evaluation streams; prior results unchanged.')
    print('Outputs:',out.resolve())


def run(args):
    if args.eval_reps<2 or args.seeds<2 or args.max_states<0:
        raise ValueError('Require eval-reps >=2, seeds >=2, max-states >=0')
    out=Path(args.out)
    source_path=Path(args.input)
    if not source_path.exists():
        raise FileNotFoundError('Run budget_matched_multistage_audit.py --run first; missing '+str(source_path))
    if out.resolve()==source_path.parent.resolve():
        raise ValueError('Output must be separate from selection results')
    src=pd.read_csv(source_path)
    if src.duplicated(['seed_index','state_index','variant']).any():
        raise AssertionError('Duplicate decision keys')
    prices,manifest=market.load_snapshot(market.DATA_FOLDER)
    start=int(prices.index.searchsorted(pd.Timestamp(market.EVALUATION_START)))
    cfg=pilot.eng.Config(seed=42,scenarios=pilot.BASE_SCENARIOS,start=start)
    states=reference.capture_fixed12_states(prices,cfg)
    if args.max_states:states=states[:args.max_states]
    available=sorted(src.seed_index.unique().tolist())
    if available[:args.seeds]!=list(range(args.seeds)):
        raise AssertionError('Selection seed indices must be contiguous from zero')
    src=src[src.seed_index<args.seeds]
    src=src[src.state_index<len(states)]
    if len(src)!=args.seeds*len(states)*len(VARIANTS):
        raise AssertionError('Incomplete selection records')
    design=dict(experiment='independent evaluation score regret budget frontier',
        input_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        snapshot_sha256=manifest['sha256'],seeds=args.seeds,states=len(states),
        eval_reps=args.eval_reps,eval_seed_base=EVAL_SEED_BASE,
        eval_seed_step=EVAL_SEED_STEP,eval_rep_step=EVAL_REP_STEP,
        variants=list(VARIANTS),
        caveats=['Same frozen historical states, not new market paths',
                 'Candidate score is NOT realized return or dollar wealth',
                 'Max-over-actions score oracle has finite-sample selection optimism',
                 'Evaluation uses independent RNG from decision and training streams',
                 'Fallback decisions may have undefined candidate scores',
                 'Feasibility can vary across evaluation reps; both unrestricted and feasible-only diagnostics',
                 'Descriptive seed intervals are not market-generalization confidence intervals'])
    out.mkdir(parents=True,exist_ok=True)
    mp=out/'regret_manifest.json'
    if mp.exists():
        if json.loads(mp.read_text(encoding='utf-8'))!=design:
            raise RuntimeError('Manifest mismatch; choose new --out')
        if not args.resume:raise RuntimeError('Output already exists; use --resume')
    else:
        if list(out.glob('eval_seed_*.csv')):raise RuntimeError('Orphan checkpoint files')
        mp.write_text(json.dumps(design,indent=2),encoding='utf-8')
    frames=[]
    for seed in range(args.seeds):
        path=out/f'eval_seed_{seed:03d}.csv'
        if args.resume and path.exists():
            frame=pd.read_csv(path)
            if len(frame)!=len(states)*len(VARIANTS):
                raise AssertionError('Invalid checkpoint '+str(path))
            print(f'Evaluation seed {seed+1}/{args.seeds}: loaded checkpoint',flush=True)
        else:
            print(f'Evaluation seed {seed+1}/{args.seeds}: running',flush=True)
            frame=generate_seed(seed,states,prices,cfg,src,out,args.eval_reps)
        frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    write_atomic(data,out/'all_decision_regret_records.csv')
    summarize(data,out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true',required=True)
    ap.add_argument('--input',default='budget_matched_multistage_outputs/all_seed_state_decisions.csv')
    ap.add_argument('--out',default='decision_regret_budget_frontier_outputs')
    ap.add_argument('--seeds',type=int,default=30)
    ap.add_argument('--max-states',type=int,default=0)
    ap.add_argument('--eval-reps',type=int,default=24)
    ap.add_argument('--resume',action='store_true')
    run(ap.parse_args())
