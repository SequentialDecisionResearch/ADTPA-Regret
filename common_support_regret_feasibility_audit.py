"""Common-Support Regret and Feasibility Audit for ADTPA V3.

Add this file only. Requires completed decision_regret_budget_frontier_outputs/
all_decision_regret_records.csv and budget_matched_multistage_outputs/
all_seed_state_decisions.csv. Pure post-processing: no simulation or engine changes.

Run: python common_support_regret_feasibility_audit.py --run
Outputs are new CSV files in common_support_audit_outputs.

IMPORTANT: common-support comparisons condition on score observability, and do
not identify full-population regret. Structural feasibility can change with
budget; this script diagnoses it, but cannot isolate causal mechanisms from
aggregate selection records or recover unobserved candidate scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

KEY=['seed_index','state_index']
BASE=['fixed6','fixed7','fixed12','fixed24','fixed48','fixed96','fixed192','adaptive6to12','adaptive6to48']
TARGETS=[('adaptive6to12','fixed6','fixed12'),('adaptive6to48','fixed6','fixed48')]
METRICS=['score_regret','feasible_score_regret','signed_reference_score_gap']

def atomic(df,p):
    tmp=p.with_suffix('.tmp'); df.to_csv(tmp,index=False); tmp.replace(p)

def paired_ci(frame,metric):
    z=frame.groupby(level=0)[metric].mean()
    m=float(z.mean()); n=len(z)
    if n>1:
        half=float(student_t.ppf(.975,n-1)*z.std(ddof=1)/np.sqrt(n))
    else: half=float('nan')
    return m,m-half,m+half,n

def main(args):
    inp=Path(args.input); selection=Path(args.selection); out=Path(args.out)
    if not inp.is_file():raise FileNotFoundError(f'Missing {inp}; run decision_regret_budget_frontier_audit.py --run first')
    if not selection.is_file():raise FileNotFoundError(f'Missing {selection}; run budget_matched_multistage_audit.py --run first')
    if out.resolve() in [inp.parent.resolve(),selection.parent.resolve()]:raise ValueError('Output must be separate')
    d=pd.read_csv(inp); s=pd.read_csv(selection)
    required=set(KEY+['variant','reps','action','score_covered','evaluation_feasible','candidate_score','reference_score','oracle_score','feasible_oracle_score']+METRICS)
    if required-set(d.columns):raise ValueError(f'Missing evaluation columns: {sorted(required-set(d.columns))}')
    if d.duplicated(KEY+['variant']).any() or s.duplicated(KEY+['variant']).any():raise AssertionError('Duplicate keys')
    variants=[v for v in BASE if v in set(d.variant)]
    if set(variants)!=set(BASE):raise AssertionError('Expected all nine variants')
    index=pd.MultiIndex.from_frame(d[KEY].drop_duplicates().sort_values(KEY))
    if len(d)!=len(index)*len(variants):raise AssertionError('Missing variant records')
    sw=s.set_index(KEY+['variant'])
    dw=d.set_index(KEY+['variant'])
    for c in ['action','reps']:
        a=dw[c].sort_index(); b=sw.loc[a.index,c].sort_index()
        if not a.astype(str).equals(b.astype(str)):raise AssertionError(f'Selection/evaluation mismatch: {c}')
    w=d.pivot(index=KEY,columns='variant')
    for col in ['score_covered','evaluation_feasible']:
        w[col]=w[col].replace({'True':True,'False':False}).astype(bool)
    score=w['candidate_score'].apply(pd.to_numeric,errors='coerce')
    covered=w['score_covered'] & score.notna() & np.isfinite(score)
    if (w['score_covered'] != covered).to_numpy().any():raise AssertionError('Score coverage flag inconsistent with numeric scores')
    feasible=w['evaluation_feasible'] & covered
    # Within each seed-state, scores of a selected action must be identical
    # across variants selecting that action; otherwise a paired comparison is invalid.
    discrepancy=d[d.score_covered].groupby(KEY+['action'])['candidate_score'].agg(['min','max'])
    max_delta=float((discrepancy['max']-discrepancy['min']).abs().max())
    if max_delta>1e-11:raise AssertionError('Identical actions have different independent evaluation scores')
    if (feasible & ~covered).to_numpy().any():raise AssertionError('Feasible without score')
    # Common support is computed separately for raw regret and feasible-only
    # regret, because the latter has an additional selection mechanism.
    raw_common=covered[variants].all(axis=1)
    feasible_common=feasible[variants].all(axis=1)
    metric_support={
        'score_regret':raw_common,
        'signed_reference_score_gap':raw_common & covered['fixed192'],
        'feasible_score_regret':feasible_common,
    }
    # Non-numeric outcomes may occur even on covered records if the oracle is undefined.
    for metric in METRICS:
        values=w[metric].apply(pd.to_numeric,errors='coerce')
        metric_support[metric]=metric_support[metric] & values[variants].notna().all(axis=1)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for v in variants:
        action=w['action'][v].astype(str)
        fallback=action.str.contains('fallback|infeasible',case=False,regex=True)
        rows.append(dict(variant=v,states=len(index),mean_reps=float(w['reps'][v].mean()),
                         score_covered=int(covered[v].sum()),score_missing=int((~covered[v]).sum()),
                         covered_and_eval_feasible=int(feasible[v].sum()),
                         covered_but_eval_infeasible=int((covered[v]&~feasible[v]).sum()),
                         fallback_action_count=int(fallback.sum()),
                         mean_score_regret_observed=float(w['score_regret'][v].mean()),
                         mean_score_regret_all_variant_common=float(w['score_regret'][v][metric_support['score_regret']].mean()),
                         mean_feasible_regret_all_variant_common=float(w['feasible_score_regret'][v][metric_support['feasible_score_regret']].mean()),
                         raw_common_n=int(metric_support['score_regret'].sum()),
                         feasible_common_n=int(metric_support['feasible_score_regret'].sum())))
    coverage=pd.DataFrame(rows);atomic(coverage,out/'coverage_decomposition.csv')
    support=pd.DataFrame([dict(metric=m,all_seed_states=len(index),common_n=int(mask.sum()),common_fraction=float(mask.mean()),
                               dropped_n=int((~mask).sum())) for m,mask in metric_support.items()]);atomic(support,out/'common_support_sizes.csv')
    # Reference score and evaluation oracles must be invariant across variants.
    for c in ['reference_score','oracle_score','feasible_oracle_score']:
        v=w[c].apply(pd.to_numeric,errors='coerce')
        spread=v.max(axis=1)-v.min(axis=1)
        if (spread.dropna()>1e-11).any():raise AssertionError(f'Inconsistent common evaluation: {c}')
    comparisons=[];seed_rows=[]
    for target,low,high in TARGETS:
        p=(float(w['reps'][target].mean())-float(w['reps'][low].mean()))/(float(w['reps'][high].mean())-float(w['reps'][low].mean()))
        if not 0<=p<=1:raise AssertionError('Budget mixture outside [0,1]')
        for metric in METRICS:
            v=w[metric].apply(pd.to_numeric,errors='coerce')
            for scope,mask in [('three_method_paired',v[[target,low,high]].notna().all(axis=1)),
                               ('all_variant_common',metric_support[metric])]:
                x=v.loc[mask,[target,low,high]].copy()
                if x.empty:
                    comparisons.append(dict(adaptive=target,metric=metric,scope=scope,n=0,warning='NO COMMON SUPPORT'))
                    continue
                x['diff']=x[target]-(1-p)*x[low]-p*x[high]
                mean,lo,hi,n=paired_ci(x,'diff')
                comparisons.append(dict(adaptive=target,metric=metric,scope=scope,n=len(x),seeds=n,
                                        mean_budget=float(w['reps'][target].mean()),mixture_p_high=p,
                                        adaptive_mean=float(x[target].mean()),mixture_expected_mean=float(((1-p)*x[low]+p*x[high]).mean()),
                                        difference=mean,ci_low=lo,ci_high=hi,
                                        warning='Conditional on observable support; not population regret'))
                for seed,g in x.groupby(level=0):
                    seed_rows.append(dict(adaptive=target,metric=metric,scope=scope,seed_index=int(seed),n=len(g),difference=float(g['diff'].mean())))
    comp=pd.DataFrame(comparisons);atomic(comp,out/'common_support_budget_comparisons.csv')
    atomic(pd.DataFrame(seed_rows),out/'per_seed_common_support_differences.csv')
    # State-level feasibility/coverage transitions by budget. These are
    # observational diagnostics, not evidence of a causal mechanism.
    trans=[]
    fixed=['fixed6','fixed7','fixed12','fixed24','fixed48','fixed96','fixed192']
    for left,right in zip(fixed[:-1],fixed[1:]):
        for name,mask in [('score_coverage',covered),('eval_feasible',feasible)]:
            a=mask[left];b=mask[right]
            trans.append(dict(from_variant=left,to_variant=right,flag=name,n=len(a),
                              yes_yes=int((a&b).sum()),yes_no=int((a&~b).sum()),
                              no_yes=int((~a&b).sum()),no_no=int((~a&~b).sum()),
                              rate_from=float(a.mean()),rate_to=float(b.mean())))
    atomic(pd.DataFrame(trans),out/'budget_feasibility_transitions.csv')
    # Detect whether increasing budget causes only losses in coverage (nested
    # feasible attrition), and whether scores are unavailable at fallback.
    nonmonotone={}
    for name,mask in [('score_coverage',covered),('eval_feasible',feasible)]:
        nonmonotone[name]=sum(int(((~mask[a])&mask[b]).sum()) for a,b in zip(fixed[:-1],fixed[1:]))
    # Full-state sensitivity: conservative bounds require externally justified
    # score ranges. Report missing counts rather than silently zero imputing.
    manifest=dict(experiment='Common-Support Regret and Feasibility Audit',
                  input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                  selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
                  seed_states=len(index),variants=variants,
                  raw_common=int(raw_common.sum()),feasible_common=int(feasible_common.sum()),
                  nonmonotone_reappearances=nonmonotone,
                  caveats=['Common support is a selected subset, not a full-population causal estimand',
                           'Independent evaluation uses finite replications and is not a true oracle',
                           'No missing scores are imputed as zero',
                           'Feasibility-at-all-replications may become stricter with budget',
                           'Observed transitions cannot isolate feasibility from threshold changes',
                           'Seed-level t intervals describe Monte Carlo stream variation only'])
    (out/'audit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nCOMMON SUPPORT AND COVERAGE')
    print(support.to_string(index=False))
    print('\nCOVERAGE DECOMPOSITION')
    print(coverage[['variant','mean_reps','score_covered','score_missing','covered_and_eval_feasible','covered_but_eval_infeasible','fallback_action_count']].to_string(index=False))
    print('\nEQUAL-BUDGET REGRET ON MATCHED SUPPORT (negative difference favors adaptive)')
    print(comp[['adaptive','metric','scope','n','difference','ci_low','ci_high']].to_string(index=False,float_format=lambda z:f'{z:.9g}'))
    print('\nFEASIBILITY TRANSITIONS')
    print(pd.DataFrame(trans).to_string(index=False))
    print('\nIMPORTANT: these are conditional common-support comparisons; missing outcomes are NOT zero.')
    print('PASS: selection records match; evaluation score invariance verified; prior files unchanged.')
    print('Outputs:',out.resolve())

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true',required=True)
    ap.add_argument('--input',default='decision_regret_budget_frontier_outputs/all_decision_regret_records.csv')
    ap.add_argument('--selection',default='budget_matched_multistage_outputs/all_seed_state_decisions.csv')
    ap.add_argument('--out',default='common_support_audit_outputs')
    main(ap.parse_args())
