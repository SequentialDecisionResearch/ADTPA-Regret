import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .engine import run, Config, cvar

def summary(d,dec):
    a=d.nav.pct_change().dropna(); b=d.iyw_nav.pct_change().dropna(); x=a-b
    dd=d.nav/d.nav.cummax()-1; bdd=d.iyw_nav/d.iyw_nav.cummax()-1
    return {'terminal_wealth':float(d.wealth.iloc[-1]),'matched_iyw_wealth':float(d.iyw_wealth.iloc[-1]),
            'relative_terminal_wealth':float(d.wealth.iloc[-1]/d.iyw_wealth.iloc[-1]-1),
            'twr_return':float(d.nav.iloc[-1]/d.nav.iloc[0]-1),
            'iyw_twr_return':float(d.iyw_nav.iloc[-1]/d.iyw_nav.iloc[0]-1),
            'annualized_tracking_error':float(x.std()*np.sqrt(252)),
            'relative_loss_cvar90_daily':cvar((-x).values,.9),
            'absolute_loss_cvar90_daily':cvar((-a).values,.9),
            'max_drawdown':float(dd.min()),'iyw_max_drawdown':float(bdd.min()),
            'sharpe_zero_rf':float(a.mean()/a.std()*np.sqrt(252)) if a.std()>0 else None,
            'decision_count':len(dec),'infeasible_decisions':int((~dec.feasible).sum()) if len(dec) else 0,
            'cash_wait_feasible':int((dec.status=='cash_wait_feasible').sum()) if len(dec) else 0,
            'infeasible_fallbacks':int((dec.status=='infeasible_fallback').sum()) if len(dec) else 0,
            'has_infeasible_fallback':bool((dec.status=='infeasible_fallback').any()) if len(dec) else False,
            'trades':int(d.traded.sum()),'fees':float(d.fees.iloc[-1])}

def plot(dfs,out):
    for col,filename,title in [('nav','twr_nav.png','Time-weighted growth (not dollar wealth)'),
                               ('wealth','wealth.png','Account wealth with identical cash contributions')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for name,d in dfs.items(): ax.plot(d.index,d[col],label=name,linewidth=1)
        if col=='nav': ax.plot(next(iter(dfs.values())).index,next(iter(dfs.values())).iyw_nav,label='IYW matched',color='black',linewidth=2)
        else: ax.plot(next(iter(dfs.values())).index,next(iter(dfs.values())).iyw_wealth,label='IYW matched',color='black',linewidth=2)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout();fig.savefig(out/filename,dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for name,d in dfs.items(): ax.plot(d.index,d.nav/d.nav.cummax()-1,label=name,linewidth=1)
    ax.set_title('Strategy drawdown on time-weighted NAV');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'drawdown.png',dpi=140);plt.close(fig)

def evaluate(p,out,policies,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dfs={}; results=[]
    for name in policies:
        d,dec=run(p,name,cfg);dfs[name]=d
        d.to_csv(out/f'{name}_daily.csv');dec.to_csv(out/f'{name}_decisions.csv',index=False)
        results.append({'policy':name,**summary(d,dec)})
    metrics=pd.DataFrame(results);metrics.to_csv(out/'metrics.csv',index=False)
    plot(dfs,out)
    with open(out/'interpretation.md','w') as f:
        f.write('# Automated descriptive interpretation\n\n')
        f.write('Exploratory diagnostics only; no statistically significant superiority is claimed.\n\n')
        for _,r in metrics.iterrows():
            f.write(f"- **{r['policy']}**: terminal relative wealth {r['relative_terminal_wealth']:.2%}; "
                    f"max drawdown {r['max_drawdown']:.2%}; relative daily CVaR90 {r['relative_loss_cvar90_daily']:.3%}; "
                    f"{r['infeasible_decisions']} infeasible reviews.\n")
        f.write('\nIMPORTANT: Infeasible fallbacks are forced cash waits, NOT feasible risk-controlled decisions. '
                'Do not interpret their realized return as an optimizer success. '
                'Selected-action CVaR columns are intentionally missing for fallback decisions; '
                'least-violating hypothetical candidate metrics are recorded separately.\n')
        f.write('\nRegret-CVaR in decisions is *scenario hindsight* and cannot be equated to implementable policy regret. '
                'Use held-out chronological evaluation, independent seeds, and multiple-testing controls before inference.\n')
    return metrics
