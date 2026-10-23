"""Research simulation: synthetic or user-supplied *total return indices*; never broker execution."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Config:
    start: int = 252
    contribution: float = 500.0
    contribution_every: int = 21
    initial: float = 100000.0
    cost_bps: float = 5.0
    lookback: int = 126
    max_weight_new: float = 0.4
    scenarios: int = 160
    seed: int = 42
    horizons: tuple = (5, 10, 20, 40, 60)
    review_cost: float = 0.00015
    risk_budget_rel: float = 0.12
    risk_budget_abs: float = 0.18
    regret_budget: float = 0.14
    alpha: float = 0.90
    risk_aversion: float = 1.0
    preference_uncertainty: float = 0.0
    invest_fraction: float = 1.0
    min_cash_reserve: float = 0.0


def cvar(losses, alpha=0.9):
    x=np.asarray(losses,float)
    if not 0<alpha<1 or not len(x) or not np.isfinite(x).all(): raise ValueError('invalid CVaR')
    # empirical upper tail with fractional mass at quantile; coherent sample CVaR
    z=np.sort(x)[::-1]; k=(1-alpha)*len(z); full=int(np.floor(k)); frac=k-full
    return float((z[:full].sum()+(frac*z[full] if full<len(z) else 0))/k)


def candidate_weights(n, cap):
    if n*cap < 1-1e-10: raise ValueError('infeasible single-asset cap')
    out=[np.zeros(n),np.ones(n)/n]
    for i in range(n):
        w=np.zeros(n); w[i]=min(cap,1.0); remaining=1-w[i]
        others=[j for j in range(n) if j!=i]
        if others:
            w[others]=remaining/len(others)
        if np.max(w)<=cap+1e-9: out.append(w)
    return out


def scenarios_at(prices, t, h, cfg, seed):
    # only data through t; bootstrap daily paired log returns preserving cross-sectional correlation
    arr=np.log(prices.iloc[max(0,t-cfg.lookback):t+1].to_numpy(float))
    r=np.diff(arr,axis=0)
    rng=np.random.default_rng(seed)
    draws=r[rng.integers(len(r),size=(cfg.scenarios,h))]
    return np.expm1(draws.sum(axis=1))


def choose(prices,t,shares,iyw_shares,cash,config,policy,decision_number):
    cfg=config; n=prices.shape[1]-1
    px=prices.iloc[t,:n].to_numpy(float); bp=float(prices.iloc[t,-1])
    current=shares*px
    budget=max(0.0,(cash-cfg.min_cash_reserve)*cfg.invest_fraction)
    if policy=='iyw': return {'horizon':20,'weights':np.zeros(n),'iyw_weight':1.,'metrics':{},'feasible':True}
    hs=(20,) if policy in ('fixed','relative_only','regret_only','combined_fixed','dca') else cfg.horizons
    if policy=='dca': return {'horizon':20,'weights':np.ones(n)/n,'iyw_weight':0.,'metrics':{},'feasible':True}
    candidates=candidate_weights(n,cfg.max_weight_new)
    # include investing new cash into IYW and waiting in cash; no sales of legacy shares
    actions=[(w,0.) for w in candidates if w.sum()>0]+[(np.zeros(n),1.),(np.zeros(n),0.)]
    best=None; diagnostics=[]; best_fallback=None
    for h in hs:
        scen=scenarios_at(prices,t,h,cfg,cfg.seed+decision_number*1009+h)
        stock=scen[:,:n]; b=scen[:,-1]
        current_wealth=current.sum()+iyw_shares*bp+cash
        if current_wealth<=0: raise ValueError('nonpositive wealth')
        realized=[]
        for w,wi in actions:
            buy=(budget/(1+cfg.cost_bps/10000))*(1-wi) if w.sum()>0 else 0.
            ibuy=(budget/(1+cfg.cost_bps/10000))*wi
            # weight sums to one except cash action
            fees=(buy+ibuy)*cfg.cost_bps/10000
            available=cash-buy-ibuy-fees
            if available< -1e-7: raise AssertionError('negative cash')
            gross=(current[None,:]*(1+stock)).sum(axis=1)+(iyw_shares*bp)*(1+b)+available+buy*(1+stock@w)+ibuy*(1+b)
            strat=gross/current_wealth-1
            # matched cashflow IYW benchmark at t, no future info
            bench=b
            realized.append((strat,bench))
        oracle=np.max(np.column_stack([v[0] for v in realized]),axis=1)
        for i,((w,wi),(ret,bench)) in enumerate(zip(actions,realized)):
            # oracle is scenario hindsight upper bound, not implementable policy regret
            reg=oracle-ret
            rel=cvar(bench-ret,cfg.alpha); ab=cvar(-ret,cfg.alpha); rg=cvar(reg,cfg.alpha)
            active=policy not in ('regret_only',)
            reg_active=policy in ('regret_only','combined_fixed','combined_dynamic','preference_robust')
            feasible=((not active or (rel<=cfg.risk_budget_rel and ab<=cfg.risk_budget_abs)) and
                      (not reg_active or rg<=cfg.regret_budget))
            preference=cfg.risk_aversion+(cfg.preference_uncertainty if policy=='preference_robust' else 0)
            # compare per-day estimated excess reward; review penalty discourages excessive checking
            score=float(np.mean(ret-bench)/h-cfg.review_cost/h - preference*(rel+ab)/h - (rg/h if reg_active else 0))
            violation_rel=max(0.,rel-cfg.risk_budget_rel) if active else 0.
            violation_abs=max(0.,ab-cfg.risk_budget_abs) if active else 0.
            violation_regret=max(0.,rg-cfg.regret_budget) if reg_active else 0.
            total_violation=violation_rel+violation_abs+violation_regret
            action='cash_wait' if w.sum()==0 and wi==0 else ('iyw_buy' if wi>0 else 'stock_buy')
            record={'horizon':h,'weights':w,'iyw_weight':wi,'score':score,'feasible':feasible,
                    'action':action,'total_violation':total_violation,
                    'violation_rel':violation_rel,'violation_abs':violation_abs,
                    'violation_regret':violation_regret,
                    'metrics':{'rel_cvar':rel,'abs_cvar':ab,'regret_cvar':rg,'mean_excess':float(np.mean(ret-bench)),
                               'hindsight_regret_mean':float(np.mean(reg))}}
            diagnostics.append(record)
            if feasible and (best is None or score>best['score']): best=record
            if best_fallback is None or (total_violation,-score)<(best_fallback['total_violation'],-best_fallback['score']):
                best_fallback=record
    if best is None:
        # No feasible candidate: execute cash wait, but report the least-violating
        # hypothetical candidate and never label the fallback as feasible.
        return {'horizon':min(hs),'weights':np.zeros(n),'iyw_weight':0.,
                'action':'infeasible_fallback','status':'infeasible_fallback',
                'score':np.nan,'feasible':False,'metrics':{
                    'rel_cvar':np.nan,'abs_cvar':np.nan,'regret_cvar':np.nan,
                    'mean_excess':np.nan,'hindsight_regret_mean':np.nan},
                'candidate_count':len(diagnostics),'feasible_candidate_count':0,
                'rejected_count':len(diagnostics),
                'minimum_total_violation':best_fallback['total_violation'],
                'least_violating_action':best_fallback['action'],
                'least_violating_horizon':best_fallback['horizon'],
                'least_violating_rel_cvar':best_fallback['metrics']['rel_cvar'],
                'least_violating_abs_cvar':best_fallback['metrics']['abs_cvar'],
                'least_violating_regret_cvar':best_fallback['metrics']['regret_cvar']}
    best=best.copy()
    best['status']='cash_wait_feasible' if best['action']=='cash_wait' else 'feasible_optimum'
    best['candidate_count']=len(diagnostics)
    best['feasible_candidate_count']=sum(x['feasible'] for x in diagnostics)
    best['rejected_count']=best['candidate_count']-best['feasible_candidate_count']
    best['minimum_total_violation']=min(x['total_violation'] for x in diagnostics)
    best['least_violating_action']=best_fallback['action']
    best['least_violating_horizon']=best_fallback['horizon']
    best['least_violating_rel_cvar']=best_fallback['metrics']['rel_cvar']
    best['least_violating_abs_cvar']=best_fallback['metrics']['abs_cvar']
    best['least_violating_regret_cvar']=best_fallback['metrics']['regret_cvar']
    return best


def run(prices, policy, cfg=Config()):
    p=prices.copy(); p.index=pd.to_datetime(p.index)
    if p.columns[-1]!='IYW' or p.shape[1]<3: raise ValueError('IYW must be last, at least two stocks')
    if p.isna().any().any() or (p<=0).any().any() or p.index.has_duplicates: raise ValueError('bad data')
    if cfg.start<cfg.lookback+1 or cfg.start>=len(p)-2: raise ValueError('start outside valid window')
    n=p.shape[1]-1; q=np.zeros(n); iq=0.; cash=cfg.initial; bq=0.; bc=cfg.initial
    next_review=cfg.start; pending=None; rows=[]; decisions=[]; ndec=0; cumfees=0.
    nav=1.; bnav=1.; prev=None; bprev=None
    for t in range(cfg.start,len(p)):
        px=p.iloc[t,:n].to_numpy(float); bp=float(p.iloc[t,-1]); dt=p.index[t]
        before=float(q@px+iq*bp+cash); bbefore=float(bq*bp+bc)
        contribution=cfg.contribution if (t-cfg.start)%cfg.contribution_every==0 else 0.
        cash+=contribution; bc+=contribution
        # same contribution dates; benchmark buys IYW immediately
        bq+=bc/bp; bc=0.
        traded=0; fee=0.
        if pending is not None:
            w=np.asarray(pending['weights']); iw=float(pending['iyw_weight'])
            budget=max(0.,(cash-cfg.min_cash_reserve)*cfg.invest_fraction)
            if not pending.get('feasible',True): budget=0.
            purchase=budget*(w.sum()+iw)
            # execution t+1; fixed budget including transaction fees
            spend=purchase/(1+cfg.cost_bps/10000)
            fee=purchase-spend
            if w.sum()>0: q+=spend*w/px
            if iw>0: iq+=spend*iw/bp
            cash-=purchase; cumfees+=fee; traded=int(purchase>1e-10)
            pending=None
        if policy=='iyw':
            spend=cash/(1+cfg.cost_bps/10000); fee+=cash-spend; iq+=spend/bp; cumfees+=cash-spend; cash=0.
        if t>=next_review and t<len(p)-1 and policy!='iyw':
            dec=choose(p,t,q,iq,cash,cfg,policy,ndec); ndec+=1
            pending=dec; next_review=t+int(dec['horizon'])
            decisions.append({'date':str(dt.date()),'policy':policy,'horizon':dec['horizon'],
                              'feasible':dec.get('feasible',True),'status':dec.get('status','baseline'),
                              'action':dec.get('action','baseline'),
                              'candidate_count':dec.get('candidate_count',np.nan),
                              'feasible_candidate_count':dec.get('feasible_candidate_count',np.nan),
                              'rejected_count':dec.get('rejected_count',np.nan),
                              'minimum_total_violation':dec.get('minimum_total_violation',np.nan),
                              'least_violating_action':dec.get('least_violating_action',''),
                              'least_violating_horizon':dec.get('least_violating_horizon',np.nan),
                              'least_violating_rel_cvar':dec.get('least_violating_rel_cvar',np.nan),
                              'least_violating_abs_cvar':dec.get('least_violating_abs_cvar',np.nan),
                              'least_violating_regret_cvar':dec.get('least_violating_regret_cvar',np.nan),
                              'score':dec.get('score',np.nan),
                              'iyw_new_weight':dec['iyw_weight'], 'stock_new_weights':';'.join(f'{v:.4f}' for v in dec['weights']),
                              **dec.get('metrics',{})})
        wealth=float(q@px+iq*bp+cash); bw=float(bq*bp+bc)
        # Time-weighted NAV: external contribution is not investment return.
        # Evaluate after trades so today's transaction costs are recognized today.
        if prev is None:
            nav *= wealth / (cfg.initial + contribution)
            bnav *= bw / (cfg.initial + contribution)
        else:
            nav *= wealth / (prev + contribution)
            bnav *= bw / (bprev + contribution)
        if cash < -1e-6 or np.min(q)<-1e-10 or iq< -1e-10: raise AssertionError('only-in invariant')
        rows.append({'date':dt,'wealth':wealth,'iyw_wealth':bw,'nav':nav,'iyw_nav':bnav,
                     'contribution':contribution,'cash':cash,'fees':cumfees,'traded':traded,
                     'stock_qty_min':float(q.min()),'iyw_qty':iq})
        prev=wealth; bprev=bw
    return pd.DataFrame(rows).set_index('date'),pd.DataFrame(decisions)
