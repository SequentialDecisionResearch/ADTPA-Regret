"""Fixed-state bootstrap audit for ADTPA; does not modify engine.py."""
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import v3.engine as engine
from legacy_adtpa.data import synthetic

TARGET = '2020-02-17'
SEED = 22
REPS = 100
SCENARIO_COUNTS = (100, 500, 1000)

prices = synthetic(n=700, d=6, seed=SEED, regime='mixed')
prices = prices[[c for c in prices.columns if c != 'IYW'] + ['IYW']]
base_cfg = engine.Config(seed=SEED, scenarios=100, start=233)

# Capture state exactly at the target review, before the target decision.
original_choose = engine.choose
state = {}

def capture_choose(p, t, shares, iyw_shares, cash, cfg, policy, decision_number):
    if str(p.index[t].date()) == TARGET:
        state.update(t=t, shares=shares.copy(), iyw_shares=float(iyw_shares),
                     cash=float(cash), decision_number=int(decision_number))
    return original_choose(p, t, shares, iyw_shares, cash, cfg, policy, decision_number)

engine.choose = capture_choose
try:
    engine.run(prices, 'combined_dynamic', base_cfg)
finally:
    engine.choose = original_choose

if not state:
    raise RuntimeError('Target review date not found in baseline run')

# Make a copy of choose() in memory that exposes its full candidate diagnostics.
source = inspect.getsource(original_choose)
old = '    return best\n'
if source.count(old) != 1:
    raise RuntimeError('Unexpected choose() source; cannot expose candidates safely')
namespace = dict(vars(engine))
exec(source.replace(old, "    best['all_candidates'] = diagnostics\n    return best\n"), namespace)
audit_choose = namespace['choose']

rows = []
for n in SCENARIO_COUNTS:
    cfg = replace(base_cfg, scenarios=n)
    for rep in range(REPS):
        # Keep market path, portfolio state and cfg.seed unchanged.
        # Change only the bootstrap seed used by scenarios_at.
        original_scenarios = engine.scenarios_at
        def scenarios_with_rep(p, t, h, config, seed, *, _rep=rep, _orig=original_scenarios):
            return _orig(p, t, h, config, seed + 100000 * _rep)
        namespace['scenarios_at'] = scenarios_with_rep
        result = audit_choose(prices, state['t'], state['shares'],
                              state['iyw_shares'], state['cash'], cfg,
                              'combined_dynamic', state['decision_number'])
        candidates = result['all_candidates']
        best = {}
        for action in ('stock_buy', 'iyw_buy', 'cash_wait'):
            feasible = [x for x in candidates if x['action'] == action and x['feasible']]
            best[action] = max(feasible, key=lambda x: x['score']) if feasible else None
        rows.append(dict(scenarios=n, repetition=rep, winner=result['action'],
                         horizon=result['horizon'],
                         **{a: (best[a]['score'] if best[a] else np.nan) for a in best}))
    print('Completed', n, 'scenarios', flush=True)

out = pd.DataFrame(rows)
filename = 'seed22_fixed_state_convergence.csv'
out.to_csv(filename, index=False)
print('\nFIXED STATE:', TARGET, 'decision_number:', state['decision_number'],
      'cash:', round(state['cash'], 2), 'iyw_shares:', round(state['iyw_shares'], 6),
      'stock_shares:', np.round(state['shares'], 6).tolist())
for n, d in out.groupby('scenarios'):
    diff = d['stock_buy'] - d['iyw_buy']
    print('\nSCENARIOS:', n)
    print('Winners:', d['winner'].value_counts().to_dict())
    print('Stock-IYW score gap: mean=', diff.mean(),
          'std=', diff.std(), 'positive_fraction=', (diff > 0).mean(),
          '5%=', diff.quantile(.05), '95%=', diff.quantile(.95))
    print('Missing scores:', d[['stock_buy','iyw_buy','cash_wait']].isna().sum().to_dict())
print('\nSaved:', filename)
