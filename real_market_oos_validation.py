"""Frozen-universe historical walk-forward ADTPA validation.

Run in V4 project root beside engine.py (in v3/), long_run_gate_pilot.py,
multiseed_gate_validation.py, sequential_gate_validation.py.

Install: python -m pip install yfinance scipy
First:   python real_market_oos_validation.py --download-only
Then:    python real_market_oos_validation.py --run
Repeat:  python real_market_oos_validation.py --run  # reuses immutable snapshot

IMPORTANT: Retrospectively adjusted Yahoo data and a 2026-selected universe are
not point-in-time survivorship-free. Results are historical walk-forward evidence,
NOT an unbiased prospective strategy-performance estimate.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Six long-lived stocks; FIX BEFORE OBSERVING RESULTS. IYW must be last.
STOCKS = ('MSFT', 'AAPL', 'AMZN', 'GOOGL', 'NVDA', 'AMD')
TICKERS = STOCKS + ('IYW',)
START = '2015-01-01'       # warmup begins here
END_EXCLUSIVE = '2025-01-01'  # excludes all 2025/2026 observations
EVALUATION_START = '2018-01-02'  # fixed before running
DATA_FOLDER = Path('real_market_oos_outputs')
DATA_FILE = 'frozen_adjusted_close.csv'
MANIFEST_FILE = 'data_manifest.json'
POLICIES = ('iyw', 'combined_dynamic', 'fixed_gate', 'adaptive_gate')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_record():
    return {'tickers': list(TICKERS), 'start_inclusive': START,
            'end_exclusive': END_EXCLUSIVE,
            'evaluation_start': EVALUATION_START,
            'data_kind': 'Yahoo Finance yfinance auto_adjust=True Close; retrospective adjusted close',
            'missing_policy': 'fail_closed_no_forward_fill',
            'currency': 'USD', 'universe_selection': 'fixed_2026_survivors_ex_post',
            'research_limitations': ['survivorship bias', 'retrospective corporate-action adjustment',
                                     'daily close-to-next-close execution model']}


def download_snapshot(folder):
    import yfinance as yf
    folder.mkdir(parents=True, exist_ok=True)
    data_path = folder / DATA_FILE
    manifest_path = folder / MANIFEST_FILE
    if data_path.exists() or manifest_path.exists():
        raise RuntimeError('Snapshot exists: refusing overwrite. Use --run to reuse it.')
    frames = []
    for ticker in TICKERS:
        print('Downloading', ticker, flush=True)
        frame = yf.download(ticker, start=START, end=END_EXCLUSIVE,
                            interval='1d', auto_adjust=True, actions=False,
                            repair=False, progress=False, threads=False)
        if frame is None or frame.empty:
            raise RuntimeError(f'Empty download: {ticker}')
        close = frame['Close']
        if isinstance(close, pd.DataFrame):
            if close.shape[1] != 1:
                raise RuntimeError(f'Ambiguous Close columns for {ticker}')
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors='coerce').rename(ticker)
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        if close.index.has_duplicates:
            raise RuntimeError(f'Duplicate dates: {ticker}')
        frames.append(close)
    joined = pd.concat(frames, axis=1, join='outer').sort_index()
    validate(joined)
    # Explicitly store sufficient digits for near-lossless reproducibility.
    joined.to_csv(data_path, index_label='Date', float_format='%.15g')
    manifest = {'protocol': config_record(),
                'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                'yfinance_version': yf.__version__, 'python': platform.python_version(),
                'pandas_version': pd.__version__, 'numpy_version': np.__version__,
                'rows': len(joined), 'first_date': str(joined.index[0].date()),
                'last_date': str(joined.index[-1].date()), 'sha256': sha256(data_path),
                'source': 'Yahoo Finance via yfinance; research/personal-use terms apply'}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Snapshot saved:', data_path.resolve())
    print('SHA256:', manifest['sha256'])


def validate(prices):
    if list(prices.columns) != list(TICKERS):
        raise ValueError(f'Columns/order mismatch: {list(prices.columns)}')
    if prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError('Missing/duplicate/nonmonotone dates')
    if prices.isna().any().any():
        bad = prices.isna().sum()
        raise ValueError('Missing prices; no imputation allowed: ' + str(bad[bad > 0].to_dict()))
    if not np.isfinite(prices.to_numpy(float)).all() or (prices <= 0).any().any():
        raise ValueError('Nonfinite/nonpositive prices')
    if prices.index[0] > pd.Timestamp(START) + pd.Timedelta(days=10):
        raise ValueError('Insufficient warmup start')
    if prices.index[-1] < pd.Timestamp(END_EXCLUSIVE) - pd.Timedelta(days=10):
        raise ValueError('Incomplete history at end')
    if len(prices) < 1000:
        raise ValueError('Insufficient history')


def load_snapshot(folder):
    data_path = folder / DATA_FILE
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding='utf-8'))
    if manifest['protocol'] != config_record():
        raise RuntimeError('Protocol changed since download; do not reuse snapshot')
    if sha256(data_path) != manifest['sha256']:
        raise RuntimeError('Snapshot hash mismatch: data changed')
    p = pd.read_csv(data_path, index_col='Date', parse_dates=['Date'])
    validate(p)
    return p, manifest


def run_backtest(folder):
    import long_run_gate_pilot as pilot
    import sequential_gate_validation as sequential
    from dataclasses import replace
    p, manifest = load_snapshot(folder)
    # Fixed date maps to first trading session >= EVALUATION_START.
    start = int(p.index.searchsorted(pd.Timestamp(EVALUATION_START)))
    if start < 252 or start >= len(p) - 100:
        raise RuntimeError('Invalid evaluation start / insufficient warmup')
    cfg = pilot.eng.Config(seed=42, scenarios=pilot.BASE_SCENARIOS, start=start)
    print('Evaluation:', str(p.index[start].date()), 'to', str(p.index[-1].date()),
          '| rows:', len(p), '| start index:', start, flush=True)
    results = []
    for policy in POLICIES:
        trace = []
        if policy.endswith('_gate'):
            mode = 'fixed' if policy == 'fixed_gate' else 'adaptive'
            def wrapper(prices, t, shares, iyw_shares, cash, config, pol, decision_number):
                return sequential.adjusted_choose_gate(prices, t, shares, iyw_shares,
                                                        cash, config, pol, decision_number,
                                                        mode, trace)
            pilot.eng.choose = wrapper
            engine_policy = 'combined_dynamic'
        else:
            pilot.eng.choose = pilot.original_choose
            engine_policy = policy
        try:
            daily, decisions = pilot.eng.run(p, engine_policy, cfg)
        finally:
            pilot.eng.choose = pilot.original_choose
        err = pilot.ledger_check(p, daily, decisions, cfg, engine_policy, trace)
        checks = pilot.prefix_test(p, cfg, policy, trace) if trace else 0
        if policy.endswith('_gate') and checks == 0:
            raise AssertionError('No prefix checks')
        summary = pilot.summary(42, policy, daily, decisions, trace, err)
        summary['prefix_checks'] = checks
        summary['data_sha256'] = manifest['sha256']
        results.append(summary)
        daily.to_csv(folder / f'{policy}_daily.csv')
        decisions.to_csv(folder / f'{policy}_decisions.csv', index=False)
        if trace:
            pd.DataFrame(trace).to_csv(folder / f'{policy}_gate_trace.csv', index=False)
        print(f'{policy:18s} wealth={summary["terminal_wealth"]:,.2f} '
              f'drawdown={summary["max_drawdown"]:.2%} '
              f'holdout_reps={summary["holdout_evaluations"]} '
              f'ledger_err={err:.8f} prefix_checks={checks}', flush=True)
    pd.DataFrame(results).to_csv(folder / 'real_market_policy_summary.csv', index=False)
    protocol = {'data_manifest': manifest, 'engine_config': vars(cfg),
                'sequential_looks': list(sequential.LOOKS),
                'sequential_critical': [sequential.critical(x) for x in sequential.LOOKS],
                'scripts_sha256': {x: sha256(x) for x in
                  ('real_market_oos_validation.py', 'long_run_gate_pilot.py',
                   'multiseed_gate_validation.py', 'sequential_gate_validation.py')
                  if Path(x).exists()},
                'limitations': ['one historical path, not 30 independent samples',
                                'survivorship-biased preselected universe',
                                'retrospectively adjusted prices may embed future dividend/split information',
                                'no claim of anytime-valid inference']}
    (folder / 'experiment_manifest.json').write_text(json.dumps(protocol, indent=2, default=str), encoding='utf-8')
    print('DONE. Outputs:', folder.resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--download-only', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--out', default=str(DATA_FOLDER))
    args = parser.parse_args()
    folder = Path(args.out)
    if args.download_only:
        download_snapshot(folder)
    else:
        run_backtest(folder)


if __name__ == '__main__':
    main()
