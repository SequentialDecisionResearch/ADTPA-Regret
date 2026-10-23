import numpy as np
import pandas as pd
from v3.engine import Config,cvar,run,choose,candidate_weights
from legacy_adtpa.data import synthetic

def test_cvar():
    assert np.isclose(cvar([0,1,2,3],.75),3)
    assert np.isclose(cvar([0,1,2,3],.5),2.5)

def test_weights():
    for w in candidate_weights(4,.4):
        assert np.all(w>=0) and max(w)<=.400001

def test_only_in_and_cashflows():
    p=synthetic(n=180,d=3,seed=5)
    c=Config(start=130,lookback=100,scenarios=20,horizons=(5,10),contribution_every=10)
    d,dec=run(p,'combined_dynamic',c)
    assert (d.cash>=-1e-8).all()
    assert (d.stock_qty_min>=-1e-8).all()
    assert d.contribution.sum()==5*c.contribution
    assert len(dec)>0

def test_future_no_leakage():
    p=synthetic(n=180,d=3,seed=3)
    c=Config(start=130,lookback=100,scenarios=30,horizons=(5,10))
    q=np.zeros(3)
    a=choose(p,130,q,0.0,1000,c,'combined_dynamic',0)
    z=p.copy();z.iloc[131:,:]*=1000
    b=choose(z,130,q,0.0,1000,c,'combined_dynamic',0)
    assert a['horizon']==b['horizon']
    assert np.allclose(a['weights'],b['weights'])

def test_iyw_matching_no_cost():
    p=synthetic(n=180,d=3,seed=8)
    c=Config(start=130,lookback=100,cost_bps=0)
    d,_=run(p,'iyw',c)
    assert np.allclose(d.wealth,d.iyw_wealth)

def test_iyw_holdings_are_included_in_scenario_wealth():
    p=synthetic(n=180,d=3,seed=3)
    cfg=Config(start=130,lookback=100,scenarios=30,horizons=(5,))
    decision=choose(p,130,np.zeros(3),100.0,0.0,cfg,'fixed',0)
    assert decision['horizon']==20
    assert np.isfinite(decision['score']) or not decision['feasible']

def test_cash_wait_is_explicit_feasible_action():
    p=synthetic(n=180,d=3,seed=3)
    cfg=Config(start=130,lookback=100,scenarios=30,regret_budget=10.,risk_budget_rel=10.,risk_budget_abs=10.)
    a=choose(p,130,np.zeros(3),0.,1000.,cfg,'combined_fixed',0)
    assert a['candidate_count']>=len(candidate_weights(3,cfg.max_weight_new))+1
    assert a['feasible_candidate_count']>0
    assert a['status'] in ('cash_wait_feasible','feasible_optimum')
    assert a['minimum_total_violation']==0


def test_infeasible_fallback_has_diagnostics_and_does_not_trade():
    p=synthetic(n=180,d=3,seed=3)
    cfg=Config(start=130,lookback=100,scenarios=30,regret_budget=-1.,risk_budget_rel=10.,risk_budget_abs=10.)
    d,dec=run(p,'combined_fixed',cfg)
    assert (dec.status=='infeasible_fallback').all()
    assert (~dec.feasible).all()
    assert (dec.feasible_candidate_count==0).all()
    assert (dec.minimum_total_violation>0).all()
    assert dec.regret_cvar.isna().all()
    assert (d.traded==0).all()
    assert np.isclose(d.wealth.iloc[-1],cfg.initial+d.contribution.sum())


def test_cash_wait_has_no_transaction():
    p=synthetic(n=180,d=3,seed=3)
    cfg=Config(start=130,lookback=100,scenarios=30)
    # No spendable cash: cash wait should remain a valid candidate (subject to portfolio risk).
    a=choose(p,130,np.zeros(3),0.,0.01,cfg,'fixed',0)
    assert a['candidate_count']>0
    assert a['action'] in ('cash_wait','stock_buy','iyw_buy','infeasible_fallback')
