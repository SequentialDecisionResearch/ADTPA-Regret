import argparse
from pathlib import Path
import pandas as pd
from .engine import Config
from .report import evaluate
from legacy_adtpa.data import synthetic,load_csv

POLICIES=['iyw','dca','fixed','relative_only','regret_only','combined_fixed','combined_dynamic','preference_robust']
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data',help='Actual daily TOTAL-RETURN index panel (date, stocks..., IYW last)')
    a.add_argument('--out',default='results/demo')
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--n',type=int,default=700)
    a.add_argument('--d',type=int,default=6)
    a.add_argument('--scenarios',type=int,default=100)
    a.add_argument('--policies',nargs='+',choices=POLICIES,default=POLICIES)
    a.add_argument('--regime',choices=['mixed','crash','break','correlated'],default='mixed')
    args=a.parse_args()
    p=load_csv(args.data) if args.data else synthetic(n=args.n,d=args.d,seed=args.seed,regime=args.regime)
    p=p[[x for x in p.columns if x!='IYW']+['IYW']]
    cfg=Config(seed=args.seed,scenarios=args.scenarios,start=max(127,min(252,len(p)//3)))
    result=evaluate(p,args.out,args.policies,cfg)
    print(result.to_string(index=False))
if __name__=='__main__':main()
