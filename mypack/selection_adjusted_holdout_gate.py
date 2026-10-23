"""Independent selection/evaluation bootstrap gate for ADTPA V4.

No project source files are modified. Uses a temporary in-memory copy of choose()
that exposes its candidate diagnostics. Candidate identity is frozen by candidate
index and validated by action/horizon across training and holdout calls.

Inference targets the model's conditional expected optimized-score contrast for
training-selected fixed candidates, not actual future investment returns.
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
TRAIN_SCENARIOS = 2000
HOLDOUT_SCENARIOS = 1000
HOLDOUT_REPS = 200
ALPHA = 0.05
MIN_ADVANTAGE = 0.0  # score units; raise this for an economic materiality gate
DEFAULT = 'iyw_buy'
ACTIONS = ('stock_buy', 'iyw_buy', 'cash_wait')

prices = synthetic(n=700, d=6, seed=SEED, regime='mixed')
prices = prices[[c for c in prices.columns if c != 'IYW'] + ['IYW']]
base_cfg = engine.Config(seed=SEED, scenarios=100, start=233)
state = {}
original_choose = engine.choose

def capture(p, t, shares, iyw_shares, cash, cfg, policy, decision_number):
    if str(p.index[t].date()) == TARGET:
        state.update(t=int(t), shares=shares.copy(), iyw_shares=float(iyw_shares),
                     cash=float(cash), decision_number=int(decision_number))
    return original_choose(p, t, shares, iyw_shares, cash, cfg, policy, decision_number)

engine.choose = capture
try:
    engine.run(prices, POLICY, base_cfg)
finally:
    engine.choose = original_choose
if not state:
    raise RuntimeError('Target date not reached')

source = inspect.getsource(original_choose)
needle = '    return best\n'
if source.count(needle) != 1:
    raise RuntimeError('Unexpected choose() source; stopped without changing project files')
namespace = dict(vars(engine))
exec(source.replace(needle, "    best['all_candidates'] = diagnostics\n    return best\n"), namespace)
audit_choose = namespace['choose']
original_scenarios = engine.scenarios_at

def evaluate(n, seed_shift):
    # Common scenario draws within each horizon for all actions.
    def independent_scenarios(p, t, h, config, seed):
        return original_scenarios(p, t, h, config, seed + seed_shift)
    namespace['scenarios_at'] = independent_scenarios
    cfg = replace(base_cfg, scenarios=n)
    result = audit_choose(prices, state['t'], state['shares'], state['iyw_shares'],
                          state['cash'], cfg, POLICY, state['decision_number'])
    return result['all_candidates']

# Select candidates using TRAINING scenarios only; never use holdout to select.
training = evaluate(TRAIN_SCENARIOS, 91234567)
selected = {}
for action in ACTIONS:
    feasible = [(i, c) for i, c in enumerate(training)
                if c['action'] == action and c['feasible']]
    if not feasible:
        raise RuntimeError('No feasible training candidate for ' + action)
    i, c = max(feasible, key=lambda x: x[1]['score'])
    selected[action] = {'index': i, 'horizon': c['horizon'],
                        'train_score': float(c['score'])}

print('FIXED STATE:', TARGET, 'decision_number:', state['decision_number'],
      'cash:', state['cash'], 'iyw_shares:', state['iyw_shares'],
      'stock_shares:', np.round(state['shares'], 6).tolist())
print('Training scenarios:', TRAIN_SCENARIOS, 'holdout scenarios/repetition:',
      HOLDOUT_SCENARIOS, 'holdout repetitions:', HOLDOUT_REPS)
print('\nFROZEN TRAINING-SELECTED CANDIDATES')
for action, s in selected.items():
    print(action, 'candidate_index=', s['index'], 'horizon=', s['horizon'],
          'training_score=', s['train_score'])

rows = []
for rep in range(HOLDOUT_REPS):
    # Holdout seed family disjoint from training seed family.
    candidates = evaluate(HOLDOUT_SCENARIOS, 201000000 + 1000003 * rep)
    if len(candidates) != len(training):
        raise RuntimeError('Candidate count changed; index matching is unsafe')
    # Verify candidate enumeration is invariant in action and horizon.
    for i, (tr, ho) in enumerate(zip(training, candidates)):
        if tr['action'] != ho['action'] or tr['horizon'] != ho['horizon']:
            raise RuntimeError('Candidate ordering changed at index %d' % i)
    row = {'rep': rep}
    for action, s in selected.items():
        candidate = candidates[s['index']]
        row[action] = float(candidate['score'])
        row[action + '_feasible'] = bool(candidate['feasible'])
    rows.append(row)
    if (rep + 1) % 25 == 0:
        print('Completed', rep + 1, '/', HOLDOUT_REPS, 'holdout repetitions', flush=True)

out = pd.DataFrame(rows)
filename = 'seed22_selection_adjusted_holdout.csv'
out.to_csv(filename, index=False)
print('\nHOLDOUT FEASIBILITY COUNTS')
for action in ACTIONS:
    print(action, int(out[action + '_feasible'].sum()), '/', HOLDOUT_REPS)

# Score itself includes nonlinear CVaR and regret terms. Resample whole
# independently evaluated score vectors, NOT individual scenario returns.
# Bonferroni simultaneous normal intervals across all 3 action comparisons.
comparisons = (('stock_buy', 'iyw_buy'), ('stock_buy', 'cash_wait'),
               ('iyw_buy', 'cash_wait'))
z = NormalDist().inv_cdf(1 - ALPHA / (2 * len(comparisons)))
intervals = {}
print('\nINDEPENDENT HOLDOUT: FROZEN-CANDIDATE PAIRED SCORE DIFFERENCES')
for a, b in comparisons:
    d = out[a] - out[b]
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    lo, hi = mean - z * se, mean + z * se
    intervals[(a, b)] = (lo, hi)
    print('%s - %s: mean=%+.10g SE=%.5g simultaneous_95%%_CI=[%+.10g, %+.10g] positive_fraction=%.3f'
          % (a, b, mean, se, lo, hi, float((d > 0).mean())))

def lower(a, b):
    if (a, b) in intervals:
        return intervals[(a, b)][0]
    return -intervals[(b, a)][1]

# Conservative gate: every frozen candidate must be feasible in every holdout
# repetition to be eligible; if baseline fails, abstain rather than force buy.
fully_feasible = {a: bool(out[a + '_feasible'].all()) for a in ACTIONS}
if not fully_feasible[DEFAULT]:
    decision = 'ABSTAIN: default failed holdout feasibility'
else:
    challengers = [a for a in ACTIONS if a != DEFAULT and fully_feasible[a]
                   and all(lower(a, b) > MIN_ADVANTAGE for b in ACTIONS if b != a)]
    if len(challengers) > 1:
        raise RuntimeError('Conflicting significant winners')
    decision = challengers[0] if challengers else DEFAULT
print('\nGATE DEFAULT:', DEFAULT)
print('GATE DECISION:', decision)
print('MIN_ADVANTAGE:', MIN_ADVANTAGE)
print('Saved:', filename)
print('\nLIMITATIONS: Candidate index matching assumes stable enumeration within each action/horizon.')
print('Check portfolio weight identity in engine.choose() before production use.')
print('Holdout inference targets model Monte Carlo uncertainty only, not market risk.')
print('Training and holdout use the same historical return distribution; no time-series OOS validation.')
