"""ADTPA V4 small-sample executable holdout-gate pilot.

Run from the V4 project directory: python long_run_gate_pilot.py
No changes to v3.engine or any project source. Synthetic research only.
"""
import inspect
from dataclasses import replace
from statistics import NormalDist
from pathlib import Path
import numpy as np
import pandas as pd
import v3.engine as eng
from legacy_adtpa.data import synthetic

SEEDS=(22,42)
BASE_SCENARIOS=100
TRAIN_N=300
HOLDOUT_N=250
FIXED_REPS=12
ADAPTIVE_FIRST=6
ALPHA=0.05
MIN_ADVANTAGE=0.0
ACTIONS=('stock_buy','iyw_buy','cash_wait')
OUT=Path('gate_pilot_outputs')
OUT.mkdir(exist_ok=True)

# Fail closed on unrecognized source structure; diagnostics only in an in-memory copy.
source=inspect.getsource(eng.choose)
needle='    return best\n'
if source.count(needle)!=1 or "diagnostics.append(record)" not in source:
    raise RuntimeError('Unsupported engine.choose source; no files modified')
ns=dict(vars(eng))
exec(source.replace(needle,"    best['all_candidates'] = diagnostics\n    return best\n"),ns)
audit_choose=ns['choose']
original_choose=eng.choose
original_scenarios=eng.scenarios_at
z=NormalDist().inv_cdf(1-ALPHA/4) # 2-sided Bonferroni for 2 challenger comparisons


def identity(c):
    return (str(c['action']),int(c['horizon']),tuple(np.asarray(c['weights'],float).tolist()),float(c['iyw_weight']))


def candidate_eval(p,t,q,iq,cash,cfg,decision_number,n,seed_shift):
    def shifted(prices,tt,h,config,seed):
        assert tt==t and tt<len(prices)
        return original_scenarios(prices,tt,h,config,seed+seed_shift)
    ns['scenarios_at']=shifted
    out=audit_choose(p,t,q,iq,cash,replace(cfg,scenarios=n),'combined_dynamic',decision_number)
    if 'all_candidates' not in out:
        raise RuntimeError('No feasible candidate set; fail closed')
    return out['all_candidates']


def choose_gate(p,t,q,iq,cash,cfg,policy,decision_number,mode,trace):
    # Deterministic seed families indexed by decision date; no information from later dates.
    training=candidate_eval(p,t,q,iq,cash,cfg,decision_number,TRAIN_N,101000003)
    frozen={}
    for a in ACTIONS:
        eligible=[(i,c) for i,c in enumerate(training) if c['action']==a and c['feasible']]
        if eligible:
            i,c=max(eligible,key=lambda x:x[1]['score'])
            frozen[a]=(i,identity(c),c)
    baseline=original_choose(p,t,q,iq,cash,cfg,policy,decision_number)
    if 'iyw_buy' not in frozen:
        # If the default fails training feasibility, abstain: do not force an unsafe buy.
        dec=dict(baseline)
        dec.update(weights=np.zeros_like(q),iyw_weight=0.,horizon=20,feasible=False,
                   action='infeasible_fallback',status='gate_abstain')
        trace.append(dict(date=str(p.index[t].date()),mode=mode,decision_number=decision_number,
                          original_action=baseline.get('action',''),gate_action='infeasible_fallback',
                          changed=True,reps=0,reason='IYW training infeasible'))
        return dec
    collected=[]
    maxreps=FIXED_REPS
    checkpoint=ADAPTIVE_FIRST if mode=='adaptive' else FIXED_REPS
    decision='iyw_buy'; reason='default_no_significant_challenger'
    for rep in range(maxreps):
        holdout=candidate_eval(p,t,q,iq,cash,cfg,decision_number,HOLDOUT_N,202000003+1000003*rep)
        if len(holdout)!=len(training):
            raise AssertionError('Candidate count changed')
        # Assert *all* identities, not merely candidate indices.
        for j,(a,b) in enumerate(zip(training,holdout)):
            if identity(a)!=identity(b):
                raise AssertionError('Candidate identity changed at index %d'%j)
        collected.append({a:(float(holdout[i]['score']),bool(holdout[i]['feasible']))
                          for a,(i,_,_) in frozen.items()})
        if rep+1 not in (checkpoint,maxreps):
            continue
        # Conservative eligibility: candidate must pass every holdout feasibility check.
        eligible=[a for a in ACTIONS if a in frozen and all(r[a][1] for r in collected)]
        if 'iyw_buy' not in eligible:
            decision='infeasible_fallback';reason='IYW holdout infeasible';break
        lower={}
        for a in eligible:
            if a=='iyw_buy':continue
            dif=np.asarray([r[a][0]-r['iyw_buy'][0] for r in collected])
            se=float(dif.std(ddof=1)/np.sqrt(len(dif)))
            lower[a]=float(dif.mean()-z*se)
        winners=[a for a in lower if lower[a]>MIN_ADVANTAGE]
        if winners:
            # Require challenger to dominate all other eligible candidates too.
            strong=[]
            for a in winners:
                if all(float(np.mean([r[a][0]-r[b][0] for r in collected]))-
                       z*float(np.std([r[a][0]-r[b][0] for r in collected],ddof=1))/np.sqrt(len(collected))
                       >MIN_ADVANTAGE for b in eligible if b!=a):
                    strong.append(a)
            if len(strong)==1:
                decision=strong[0];reason='significant_challenger';break
        if mode=='adaptive' and rep+1==checkpoint:
            # Only stop early if every challenger is convincingly inferior.
            if len(eligible)==1 or all(float(np.mean([r[a][0]-r['iyw_buy'][0] for r in collected]))+
               z*float(np.std([r[a][0]-r['iyw_buy'][0] for r in collected],ddof=1))/np.sqrt(len(collected))
               < -MIN_ADVANTAGE for a in eligible if a!='iyw_buy'):
                reason='early_default';break
        if rep+1==maxreps:
            reason='default_after_max_budget'
    if decision=='infeasible_fallback':
        dec=dict(baseline)
        dec.update(weights=np.zeros_like(q),iyw_weight=0.,horizon=20,feasible=False,
                   action='infeasible_fallback',status='gate_abstain')
    else:
        i,ident,c=frozen[decision]
        dec=dict(c)
        dec.update(weights=np.asarray(c['weights'],float).copy(),iyw_weight=float(c['iyw_weight']),
                   action=decision,status='holdout_gate',candidate_count=len(training),
                   feasible_candidate_count=sum(bool(x['feasible']) for x in training),
                   rejected_count=sum(not bool(x['feasible']) for x in training),
                   minimum_total_violation=min(x['total_violation'] for x in training),
                   least_violating_action='',least_violating_horizon=np.nan,
                   least_violating_rel_cvar=np.nan,least_violating_abs_cvar=np.nan,
                   least_violating_regret_cvar=np.nan)
        assert identity(dec)==ident
    changed=(identity(dec)!=identity(baseline)) if dec.get('action')!='infeasible_fallback' else True
    trace.append(dict(date=str(p.index[t].date()),mode=mode,decision_number=decision_number,
                      original_action=baseline.get('action',''),gate_action=dec['action'],
                      original_horizon=int(baseline['horizon']),gate_horizon=int(dec['horizon']),
                      changed=changed,reps=len(collected),reason=reason,
                      weights=';'.join(repr(float(v)) for v in dec['weights'])))
    return dec


def ledger_check(p,d,dec,cfg,policy,trace=None):
    """Independent next-day execution reconstruction from decision log."""
    n=p.shape[1]-1;q=np.zeros(n);iq=0.;cash=cfg.initial;fees=0.;pending=None
    by_date={str(row['date']):row for _,row in dec.iterrows()}
    exact={r['date']:r for r in (trace or []) if 'weights' in r}
    max_err=0.
    for t in range(cfg.start,len(p)):
        dt=str(p.index[t].date()); px=p.iloc[t,:n].to_numpy(float);bp=float(p.iloc[t,-1])
        if (t-cfg.start)%cfg.contribution_every==0:cash+=cfg.contribution
        if pending is not None:
            w,iw,feasible=pending
            budget=max(0.,(cash-cfg.min_cash_reserve)*cfg.invest_fraction)
            if not feasible:budget=0.
            purchase=budget*(w.sum()+iw)
            spend=purchase/(1+cfg.cost_bps/10000)
            fees+=purchase-spend
            if w.sum()>0:q+=spend*w/px
            if iw>0:iq+=spend*iw/bp
            cash-=purchase
            pending=None
        if policy=='iyw':
            spend=cash/(1+cfg.cost_bps/10000);fees+=cash-spend;iq+=spend/bp;cash=0.
        if dt in by_date:
            row=by_date[dt]
            w=np.asarray([float(x) for x in (exact[dt]['weights'] if dt in exact else row['stock_new_weights']).split(';')]);iw=float(row['iyw_new_weight'])
            pending=(w,iw,bool(row['feasible']))
        expected=float(q@px+iq*bp+cash)
        actual=float(d.iloc[t-cfg.start]['wealth'])
        err=abs(expected-actual)
        max_err=max(max_err,err)
        if err>0.03 or abs(float(d.iloc[t-cfg.start]['cash'])-cash)>0.03 or abs(float(d.iloc[t-cfg.start]['fees'])-fees)>0.03:
            raise AssertionError('Accounting mismatch on %s: expected %.8f, actual %.8f'%(dt,expected,actual))
        if cash < -1e-7 or (q < -1e-9).any() or iq < -1e-9:
            raise AssertionError('Only-in violation')
    return max_err


def prefix_test(p,cfg,mode,trace):
    """Verify scenario generation is invariant to replacing all future rows."""
    dates=[str(x['date']) for x in trace[:min(3,len(trace))]]
    for dt in dates:
        t=int(p.index.get_loc(pd.Timestamp(dt)))
        for h in cfg.horizons:
            a=original_scenarios(p,t,h,replace(cfg,scenarios=40),cfg.seed+123+h)
            prefix=p.iloc[:t+1].copy()
            b=original_scenarios(prefix,t,h,replace(cfg,scenarios=40),cfg.seed+123+h)
            if not np.array_equal(a,b):
                raise AssertionError('Future-data leakage in scenarios_at at '+dt)
    return len(dates)*len(cfg.horizons)


def summary(seed,policy,d,dec,trace,err):
    wealth=float(d['wealth'].iloc[-1]); bench=float(d['iyw_wealth'].iloc[-1])
    dd=float((d['wealth']/d['wealth'].cummax()-1).min())
    return dict(seed=seed,policy=policy,terminal_wealth=wealth,benchmark_wealth=bench,
                relative_terminal=wealth/bench-1,max_drawdown=dd,
                decisions=len(dec),trades=int(d['traded'].sum()),fees=float(d['fees'].iloc[-1]),
                changed_decisions=sum(bool(r['changed']) for r in trace),
                holdout_evaluations=sum(int(r['reps']) for r in trace),
                max_accounting_error=err)


def main():
    summaries=[]
    for seed in SEEDS:
        print('\n=== SEED %d ==='%seed,flush=True)
        p=synthetic(n=700,d=6,seed=seed,regime='mixed')
        p=p[[c for c in p.columns if c!='IYW']+['IYW']]
        cfg=eng.Config(seed=seed,scenarios=BASE_SCENARIOS,start=233)
        for policy in ('iyw','combined_dynamic','fixed_gate','adaptive_gate'):
            trace=[]
            if policy.endswith('_gate'):
                mode='fixed' if policy=='fixed_gate' else 'adaptive'
                def wrapper(prices,t,shares,iyw_shares,cash,config,pol,decision_number):
                    return choose_gate(prices,t,shares,iyw_shares,cash,config,pol,decision_number,mode,trace)
                eng.choose=wrapper
                engine_policy='combined_dynamic'
            else:
                eng.choose=original_choose
                engine_policy=policy
            try:
                d,dec=eng.run(p,engine_policy,cfg)
            finally:
                eng.choose=original_choose
            err=ledger_check(p,d,dec,cfg,engine_policy,trace)
            if policy.endswith('_gate'):
                checks=prefix_test(p,cfg,policy,trace)
                print(policy,'prefix checks:',checks,'changed:',sum(r['changed'] for r in trace),
                      'holdout reps:',sum(r['reps'] for r in trace),flush=True)
            else:checks=0
            label=policy
            s=summary(seed,label,d,dec,trace,err)
            s['prefix_checks']=checks
            summaries.append(s)
            d.to_csv(OUT/('seed%d_%s_daily.csv'%(seed,label)))
            dec.to_csv(OUT/('seed%d_%s_decisions.csv'%(seed,label)),index=False)
            if trace:pd.DataFrame(trace).to_csv(OUT/('seed%d_%s_gate_trace.csv'%(seed,label)),index=False)
            print(label,'wealth=%.2f'%s['terminal_wealth'],'trades=',s['trades'],
                  'max_accounting_error=%.8f'%err,flush=True)
    table=pd.DataFrame(summaries)
    table.to_csv(OUT/'pilot_summary.csv',index=False)
    print('\n=== PILOT SUMMARY ===')
    print(table.to_string(index=False))
    print('\nSaved results in:',OUT.resolve())
    print('WARNING: Synthetic paths, no genuine temporal out-of-sample validation.')

if __name__=='__main__':
    try:main()
    finally:eng.choose=original_choose
