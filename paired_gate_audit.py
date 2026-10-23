"""Paired gate audit on identical real-market decision states (read-only baseline).

Place this ONE file in the V3 project root and run:
    python paired_gate_audit.py --run

States are captured from a fresh, reproducible fixed12 reference rollout. All
three gates then receive identical date, holdings, cash, decision index and
historical price prefix. No baseline files are modified. Reference agreement
is a *surrogate*, not ground truth or a statistical error-rate guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

import real_market_oos_validation as market
import real_market_ablation as ablation
import sequential_gate_validation as sequential
import long_run_gate_pilot as pilot

VARIANTS = ('fixed6', 'fixed12', 'adaptive6to12')


def ident(dec):
    return (str(dec['action']), int(dec['horizon']),
            tuple(float(x) for x in np.asarray(dec['weights'], float)),
            float(dec['iyw_weight']), bool(dec.get('feasible', True)))


def same(a, b):
    if a[0] != b[0] or a[1] != b[1] or a[4] != b[4]:
        return False
    return np.allclose(a[2], b[2], rtol=0, atol=1e-12) and abs(a[3]-b[3]) <= 1e-12


def evaluate(fn, mode, p, cfg, state):
    trace=[]
    dec=fn(p, state['t'], state['q'].copy(), state['iq'], state['cash'],
           cfg, 'combined_dynamic', state['decision_number'], mode, trace)
    if len(trace)!=1:
        raise AssertionError('Expected exactly one gate trace row')
    tr=trace[0]
    return dec, tr


def run(out, max_states=0):
    out=Path(out)
    if out.resolve() in (market.DATA_FOLDER.resolve(), Path('real_market_ablation_outputs').resolve()):
        raise ValueError('Output must not overwrite baseline directories')
    p, manifest=market.load_snapshot(market.DATA_FOLDER)
    start=int(p.index.searchsorted(pd.Timestamp(market.EVALUATION_START)))
    cfg=pilot.eng.Config(seed=42, scenarios=pilot.BASE_SCENARIOS, start=start)
    fixed6_fn, fixed6_crit=ablation.build_fixed6()
    captured=[]
    reference_trace=[]
    original=pilot.eng.choose

    def capture_wrapper(prices,t,shares,iyw_shares,cash,config,policy,decision_number):
        captured.append({'t':int(t),'q':np.asarray(shares,float).copy(),
                         'iq':float(iyw_shares),'cash':float(cash),
                         'decision_number':int(decision_number)})
        return sequential.adjusted_choose_gate(prices,t,shares,iyw_shares,cash,
                                                config,policy,decision_number,'fixed',reference_trace)
    pilot.eng.choose=capture_wrapper
    try:
        daily, decisions=pilot.eng.run(p,'combined_dynamic',cfg)
    finally:
        pilot.eng.choose=original
    if len(captured)!=len(decisions) or len(captured)!=len(reference_trace):
        raise AssertionError('Reference state capture inconsistent')
    # Fail closed if original ablation baseline cannot be reproduced.
    prior=Path('real_market_ablation_outputs')
    if not (prior/'fixed12_daily.csv').exists():
        raise RuntimeError('Missing prior fixed12 ablation baseline; run real_market_ablation.py --run first')
    ablation.assert_same('reference daily',daily.reset_index(),
                         pd.read_csv(prior/'fixed12_daily.csv'),
                         ['date','wealth','cash','fees','traded','nav','iyw_nav'])
    ablation.assert_same('reference decisions',decisions,
                         pd.read_csv(prior/'fixed12_decisions.csv'),
                         ['date','horizon','feasible','action','iyw_new_weight','stock_new_weights'])
    ablation.assert_same('reference gate trace',pd.DataFrame(reference_trace),
                         pd.read_csv(prior/'fixed12_gate_trace.csv'),
                         ['date','reps','reason','gate_action','changed'])

    if max_states:
        captured=captured[:max_states]
    rows=[]
    for k,st in enumerate(captured):
        evaluated={}
        for name in VARIANTS:
            fn=fixed6_fn if name=='fixed6' else sequential.adjusted_choose_gate
            mode='adaptive' if name=='adaptive6to12' else 'fixed'
            evaluated[name]=evaluate(fn,mode,p,cfg,st)
        ref=ident(evaluated['fixed12'][0])
        date=str(p.index[st['t']].date())
        for name in VARIANTS:
            dec,tr=evaluated[name]
            key=ident(dec)
            rows.append({'anchor':'fixed12', 'date':date, 'decision_number':st['decision_number'],
                         'variant':name, 'action':key[0], 'horizon':key[1],
                         'weights':';'.join(f'{v:.16g}' for v in key[2]),
                         'iyw_weight':key[3], 'feasible':key[4],
                         'reps':int(tr['reps']), 'reason':tr['reason'],
                         'same_action_as_fixed12':key[0]==ref[0],
                         'same_action_horizon_as_fixed12':key[:2]==ref[:2],
                         'same_full_decision_as_fixed12':same(key,ref),
                         'same_full_decision_as_adaptive':same(key,ident(evaluated['adaptive6to12'][0])),
                         'state_cash':st['cash'],
                         'state_wealth':float(st['q'] @ p.iloc[st['t'],:-1].to_numpy(float)+
                                              st['iq']*float(p.iloc[st['t'],-1])+st['cash'])})
        if (k+1)%10==0:
            print(f'Paired states evaluated: {k+1}/{len(captured)}',flush=True)
    df=pd.DataFrame(rows)
    if df.empty: raise AssertionError('No states evaluated')
    summary=[]
    for name in VARIANTS:
        x=df[df.variant==name]
        summary.append({'variant':name,'states':len(x),'total_reps':int(x.reps.sum()),
                        'mean_reps':float(x.reps.mean()),
                        'action_agreement_fixed12':float(x.same_action_as_fixed12.mean()),
                        'action_horizon_agreement_fixed12':float(x.same_action_horizon_as_fixed12.mean()),
                        'full_decision_agreement_fixed12':float(x.same_full_decision_as_fixed12.mean()),
                        'full_decision_agreement_adaptive':float(x.same_full_decision_as_adaptive.mean()),
                        'reasons':json.dumps(x.reason.value_counts().to_dict(),sort_keys=True)})
    tab=pd.DataFrame(summary)
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'paired_state_decisions.csv',index=False)
    tab.to_csv(out/'paired_state_summary.csv',index=False)
    disagreements=df[(df.variant!='fixed12') & (~df.same_full_decision_as_fixed12)]
    disagreements.to_csv(out/'paired_state_disagreements.csv',index=False)
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'paired_state_manifest.json').write_text(json.dumps({
        'experiment':'same-state paired gate comparison',
        'anchor':'fixed12 rollout; every alternative evaluated at exactly its states',
        'snapshot_sha256':manifest['sha256'],'script_sha256':script_hash,
        'states':len(captured),'max_states':max_states,
        'fixed6_critical':fixed6_crit,
        'fixed12_critical':sequential.critical(12),
        'adaptive_critical_6':sequential.critical(6),
        'adaptive_critical_12':sequential.critical(12),
        'limitations':['single retrospective path and ex-post universe',
                       'fixed12 agreement is not an independent truth label',
                       'conditional on states visited by fixed12 only',
                       'Monte Carlo repetitions are not independent market paths',
                       'different multiplicity adjustments across gate rules',
                       'no counterfactual terminal wealth for off-policy gate decisions']},indent=2),encoding='utf-8')
    print('\nPAIRED GATE AUDIT (fixed12 reference states)')
    print(tab.drop(columns=['reasons']).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('PASS: fixed12 original ablation reproduced; identical states and holdout seed families compared.')
    print('Outputs:',out.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--out',default='paired_gate_audit_outputs')
    parser.add_argument('--max-states',type=int,default=0,help='For smoke tests only; 0=all states')
    args=parser.parse_args()
    if args.max_states<0: parser.error('--max-states must be >= 0')
    run(args.out,args.max_states)
