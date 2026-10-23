"""Compare frozen original outputs with rerun after NAV-only engine change.
Usage: python nav_regression_test.py --old real_market_oos_outputs_before_nav_fix --new real_market_oos_outputs
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

POLICIES = ('iyw','combined_dynamic','fixed_gate','adaptive_gate')

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(old,new):
    for name in ('frozen_adjusted_close.csv','data_manifest.json'):
        assert digest(old/name)==digest(new/name), f'Snapshot changed: {name}'
    for policy in POLICIES:
        a=pd.read_csv(old/f'{policy}_daily.csv')
        b=pd.read_csv(new/f'{policy}_daily.csv')
        assert a.shape==b.shape and a.columns.tolist()==b.columns.tolist(), f'{policy}: daily schema changed'
        for col in a.columns:
            if col in ('nav','iyw_nav'): continue
            if pd.api.types.is_numeric_dtype(a[col]):
                assert np.allclose(a[col],b[col],rtol=1e-11,atol=1e-8,equal_nan=True), f'{policy}: {col} changed'
            else:
                assert a[col].fillna('<NA>').equals(b[col].fillna('<NA>')), f'{policy}: {col} changed'
        for name in ('decisions','gate_trace'):
            x=old/f'{policy}_{name}.csv'; y=new/f'{policy}_{name}.csv'
            assert x.exists()==y.exists(),f'{policy}: {name} existence changed'
            if x.exists(): assert digest(x)==digest(y),f'{policy}: {name} changed'
        # Independent NAV recurrence from wealth and contributions, including transaction fees.
        for wealth,nav in [('wealth','nav'),('iyw_wealth','iyw_nav')]:
            expected=[]; value=1.; prev=None
            for w,c in zip(b[wealth],b['contribution']):
                value*=w/((prev if prev is not None else 100000.)+c)
                expected.append(value); prev=w
            assert np.allclose(b[nav],expected,rtol=1e-11,atol=1e-11), f'{policy}: {nav} incorrect'
        print(f'PASS {policy}: wealth, ledger, decisions, traces unchanged; NAV verified')
    print('PASS: frozen data, wealth, trades, decisions, gate traces unchanged; NAV independently verified.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--old',default='real_market_oos_outputs_before_nav_fix')
    parser.add_argument('--new',default='real_market_oos_outputs')
    args=parser.parse_args()
    verify(Path(args.old),Path(args.new))
