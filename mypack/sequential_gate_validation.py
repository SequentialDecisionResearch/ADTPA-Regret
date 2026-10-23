"""Conservative two-look, multiplicity-adjusted ADTPA gate experiment.

Place alongside long_run_gate_pilot.py and multiseed_gate_validation.py.
Examples:
  python sequential_gate_validation.py --seeds 22 42
  python sequential_gate_validation.py --seed-start 1 --count 30

This is a PRE-SPECIFIED two-look Student-t APPROXIMATION, not a distribution-free
confidence sequence or an anytime-valid guarantee. No changes to V4 source.
Outputs default to a separate folder; prior results remain intact.
"""
import argparse
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import long_run_gate_pilot as pilot
import multiseed_gate_validation as multi

# 2 predeclared looks (6,12), 3 unordered action-pair contrasts per look.
# Bonferroni two-sided intervals: family-wise error <= 0.05 IF the individual
# Student-t intervals are valid. Monte Carlo score differences need not be normal.
LOOKS = (pilot.ADAPTIVE_FIRST, pilot.FIXED_REPS)
PAIRS = len(pilot.ACTIONS) * (len(pilot.ACTIONS) - 1) // 2
ALPHA = pilot.ALPHA


def critical(n):
    if n not in LOOKS:
        raise AssertionError(f'Unexpected checkpoint {n}; expected {LOOKS}')
    if n < 2:
        raise AssertionError('At least two independent repetitions needed')
    return float(student_t.ppf(1 - ALPHA / (2 * len(LOOKS) * PAIRS), df=n-1))


# Keep the tested gate implementation byte-for-byte except for its critical
# value. Replacing all three occurrences protects challenger/default contrasts.
_original_source = inspect.getsource(pilot.choose_gate)
if _original_source.count('z*') != 3:
    raise RuntimeError('Unexpected gate source: expected exactly 3 z* comparisons')
_adjusted_source = _original_source.replace('z*', 'critical(len(collected))*')
_namespace = dict(vars(pilot))
_namespace['critical'] = critical
exec(compile(_adjusted_source, '<sequential_gate_adjustment>', 'exec'), _namespace)
adjusted_choose_gate = _namespace['choose_gate']


def adjusted_fingerprint():
    h = hashlib.sha256()
    h.update(multi._original_fingerprint().encode('ascii'))
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--out', default='sequential_gate_outputs')
    args, other = parser.parse_known_args()
    # Preserve all existing multi-seed flags and parser behavior.
    sys.argv = [sys.argv[0], '--out', args.out] + other
    print('Statistical design: two looks', LOOKS, '; pairwise contrasts:', PAIRS)
    print('t critical at first look:', critical(LOOKS[0]),
          '; final look:', critical(LOOKS[-1]))
    print('IMPORTANT: approximate t inference, not anytime-valid or distribution-free.')
    pilot.choose_gate = adjusted_choose_gate
    multi._original_fingerprint = multi.fingerprint
    multi.fingerprint = adjusted_fingerprint
    try:
        multi.main()
    finally:
        pilot.choose_gate = original_gate
        multi.fingerprint = multi._original_fingerprint


original_gate = pilot.choose_gate
if __name__ == '__main__':
    main()
