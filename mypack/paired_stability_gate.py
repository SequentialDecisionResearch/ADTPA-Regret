"""ADTPA fixed-state paired Monte Carlo decision gate (diagnostic only).

Standalone: does not edit v3.engine or any V4 project source file.
The gate compares *action-level optimized scores* across common bootstrap
scenario draws. This controls Monte Carlo noise, NOT in-sample candidate
selection optimism or misspecification of the return-generating process.
"""
import inspect
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pandas as pd
import v3.engine as engine
from legacy_adtpa.data import synthetic

TARGET = '2020-02-17'
SEED = 22
POLICY = 'combined_dynamic'
BASE_SCENARIOS = 100
EVAL_SCENARIOS = 1000
REPS = 200
ALPHA = 0.05
DEFAULT_ACTION = 'iyw_buy'  # benchmark-preserving fallback for this diagnostic
ACTIONS = ('stock_buy', 'iyw_buy', 'cash_wait')

prices = synthetic(n=700, d=6, seed=SEED, regime='mixed')
prices = prices[[c for c in prices.columns if c != 'IYW'] + ['IYW']]
base_cfg = engine.Config(seed=SEED, scenarios=BASE_SCENARIOS, start=233)

# Capture pre-decision state by replaying the original policy, unchanged.
state = {}
original_choose = engine.choose

def capture_choose(p, t, shares, iyw_shares, cash, cfg, policy, decision_number):
    if str(p.index[t].date()) == TARGET:
        state.update(t=t, shares=shares.copy(), iyw_shares=float(iyw_shares),
                     cash=float(cash), decision_number=int(decision_number))
    return original_choose(p, t, shares, iyw_shares, cash, cfg, policy, decision_number)

engine.choose = capture_choose
try:
    engine.run(prices, POLICY, base_cfg)
finally:
    engine.choose = original_choose
if not state:
    raise RuntimeError('Target review date was not reached')

# Instrument a copy of choose() in memory to expose diagnostics.
source = inspect.getsource(original_choose)
old = '    return best\n'
if source.count(old) != 1:
    raise RuntimeError('Unexpected engine.choose() source; stop without modifying it')
namespace = dict(vars(engine))
exec(source.replace(old, "    best['all_candidates'] = diagnostics\n    return best\n"), namespace)
audit_choose = namespace['choose']
original_scenarios = engine.scenarios_at
cfg = replace(base_cfg, scenarios=EVAL_SCENARIOS)

rows = []
for rep in range(REPS):
    # All actions at the same horizon see the SAME bootstrap scenarios.
    # Different repetitions get independent pseudorandom seeds.
    def paired_scenarios(p, t, h, config, seed, *, _rep=rep):
        return original_scenarios(p, t, h, config, seed + 1000003 * (_rep + 1))
    namespace['scenarios_at'] = paired_scenarios
    result = audit_choose(prices, state['t'], state['shares'], state['iyw_shares'],
                          state['cash'], cfg, POLICY, state['decision_number'])
    candidates = result['all_candidates']
    best = {}
    for action in ACTIONS:
        feasible = [x for x in candidates if x['action'] == action and x['feasible']]
        best[action] = max(feasible, key=lambda x: x['score']) if feasible else None
    row = {'rep': rep, 'winner': result['action'], 'winner_horizon': result['horizon']}
    for action in ACTIONS:
        row[action] = best[action]['score'] if best[action] else np.nan
        row[action + '_horizon'] = best[action]['horizon'] if best[action] else np.nan
    rows.append(row)
    if (rep + 1) % 25 == 0:
        print('Completed', rep + 1, '/', REPS, 'paired repetitions', flush=True)

out = pd.DataFrame(rows)
filename = 'seed22_paired_stability_gate.csv'
out.to_csv(filename, index=False)
print('\nFIXED STATE:', TARGET, 'decision_number:', state['decision_number'],
      'cash:', state['cash'], 'iyw_shares:', state['iyw_shares'],
      'stock_shares:', np.round(state['shares'], 6).tolist())
print('Scenarios per repetition:', EVAL_SCENARIOS, 'repetitions:', REPS)
print('Winners:', out['winner'].value_counts().to_dict())
print('Missing action scores:', out[list(ACTIONS)].isna().sum().to_dict())

# Conservative simultaneous two-sided normal intervals for three pairwise
# differences. Because the unit of analysis is a paired repetition, score
# covariance is retained. This is a *Monte Carlo* uncertainty interval only.
comparisons = (('stock_buy', 'iyw_buy'), ('stock_buy', 'cash_wait'),
               ('iyw_buy', 'cash_wait'))
z = NormalDist().inv_cdf(1 - ALPHA / (2 * len(comparisons)))
print('\nPAIRED ACTION-LEVEL SCORE DIFFERENCES')
intervals = {}
for a, b in comparisons:
    d = (out[a] - out[b]).dropna()
    if len(d) < 2:
        print(a, '-', b, ': insufficient feasible paired observations')
        intervals[(a, b)] = (np.nan, np.nan)
        continue
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    lo, hi = mean - z * se, mean + z * se
    intervals[(a, b)] = (lo, hi)
    print('%s - %s: n=%d mean=%+.10g SE=%.5g simultaneous_95%%_CI=[%+.10g, %+.10g] positive_fraction=%.3f'
          % (a, b, len(d), mean, se, lo, hi, float((d > 0).mean())))

# A change from the default is allowed only when the challenger beats
# BOTH other actions with positive simultaneous lower bounds.
def lower_bound(a, b):
    if (a, b) in intervals:
        return intervals[(a, b)][0]
    if (b, a) in intervals:
        return -intervals[(b, a)][1]
    raise KeyError((a, b))

allowed = []
for action in ACTIONS:
    if action == DEFAULT_ACTION:
        continue
    if all(np.isfinite(lower_bound(action, other)) and
           lower_bound(action, other) > 0 for other in ACTIONS if other != action):
        allowed.append(action)
if len(allowed) > 1:
    raise RuntimeError('Inconsistent simultaneous comparisons')
decision = allowed[0] if allowed else DEFAULT_ACTION
print('\nGATE DEFAULT:', DEFAULT_ACTION)
print('GATE DECISION:', decision)
print('Gate changed action:', decision != DEFAULT_ACTION)
print('Saved:', filename)
print('\nLIMITATIONS: Scores re-optimize portfolios/horizons in each repetition.')
print('This is not a selection-adjusted holdout test of a frozen candidate.')
print('The intervals quantify simulation error only, not future-market risk.')
