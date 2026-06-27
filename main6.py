"""
v6 — Binary AR-first cascade + JM-only regime (assignment-compliant combined model).

Pipeline:
  1. Compute AR + AR z-score (sector PCA fragility filter)
  2. Train JM on 5 SPY microstructure features only (responsive, like v2)
  3. Binary cascade at backtest (strict 0/1, no partial sizing):
       AR veto (z > exit_high, or high + rising | persist) → 0
       JM bear                                             → 0
       JM bull and no AR veto                              → 1

Run:  python main6.py  →  ./output_v6/
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)

from src.data_loader import load_etf_data
from src.absorption_ratio import compute_ar, compute_ar_zscore
from src.features import build_features_jm_only
from src.jump_model import run_walk_forward_cv
from src.backtest import (
    run_backtest,
    run_backtest_cascade,
    cascade_cell_counts,
    plot_results,
)
from src.config import OOS_START, OOS_END, LAMBDA_CANDIDATES

_OOS = {'oos_start': OOS_START, 'oos_end': OOS_END}

DATA_PATH   = './Data/etf_ohlcv_20160301_20251231.csv'
OUTPUT_PATH = './output_v6/'

SECTOR_TICKERS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

AR_WINDOW        = 126
AR_N_COMPONENTS  = 2
AR_ZSCORE_WINDOW = 126

# AR-first cascade thresholds
AR_EXIT_HIGH     = 1.0
AR_ENTRY_LOW     = 0.0
AR_SLOPE_DAYS    = 2
AR_EXIT_PERSIST  = 3

LOOKBACK_YEARS     = 3
LOOKBACK_DAYS      = LOOKBACK_YEARS * 252
JM_N_STATES  = 2
JM_MAX_ITER  = 300
JM_TOL       = 1e-4
JM_N_INIT    = 10

V1_METRICS_PATH = './output_v1/metrics.csv'
V2_METRICS_PATH = './output_v2/metrics.csv'
V5_METRICS_PATH = './output_v5/metrics.csv'

print('=== v6: AR-first cascade + JM-only ===')
print('=== Step 1: Loading data ===')
adj, log_rets = load_etf_data(DATA_PATH)
print(f'Loaded: {adj.shape[0]} trading days, {adj.shape[1]} tickers')

print('\n=== Step 2: Absorption Ratio (fragility filter — computed first) ===')
ar = compute_ar(
    log_rets, SECTOR_TICKERS, W=AR_WINDOW, n_components=AR_N_COMPONENTS,
)
ar_zscore = compute_ar_zscore(ar, fast_window=15, slow_window=AR_ZSCORE_WINDOW)
print(f'AR computed: {ar.notna().sum()} valid observations')

print('\n=== Step 3: JM-only feature matrix (5 features, no AR) ===')
features_jm = build_features_jm_only(adj, log_rets)
print(
    f'Feature matrix: {features_jm.shape}, '
    f'from {features_jm.index[0].date()} to {features_jm.index[-1].date()}'
)

print('\n=== Step 4: Walk-forward CV → JM regime ===')
regime = run_walk_forward_cv(
    feature_df=features_jm,
    spy_returns=log_rets['SPY'],
    lookback_days=LOOKBACK_DAYS,
    lambda_candidates=LAMBDA_CANDIDATES,
    n_states=JM_N_STATES,
    max_iter=JM_MAX_ITER,
    tol=JM_TOL,
    n_init=JM_N_INIT,
)
n_switches = int((regime.diff().abs().dropna() > 0).sum())
print(f'Regime series: {len(regime)} days OOS, bull%={regime.mean():.1%}')
print(f'  Switches: {n_switches}')

print('\n=== Step 5: Backtest (v6 binary cascade) ===')
print(f'Evaluation window: {OOS_START} -> {OOS_END or "last available"}')

# JM-only binary baseline (no AR cascade)
jm_ret, bnh_ret, jm_m, bnh_m = run_backtest(regime, log_rets['SPY'], **_OOS)

# v6 combined model
cascade_ret, _, cascade_m, _, position = run_backtest_cascade(
    regime,
    log_rets['SPY'],
    ar_zscore=ar_zscore,
    ar_exit_high=AR_EXIT_HIGH,
    ar_entry_low=AR_ENTRY_LOW,
    ar_slope_days=AR_SLOPE_DAYS,
    exit_persist_days=AR_EXIT_PERSIST,
    strategy_label='v6 AR-first Cascade',
    **_OOS,
)

cells = cascade_cell_counts(
    regime, ar_zscore,
    ar_exit_high=AR_EXIT_HIGH,
    ar_entry_low=AR_ENTRY_LOW,
    ar_slope_days=AR_SLOPE_DAYS,
    exit_persist_days=AR_EXIT_PERSIST,
)
print('\n=== Cascade cell counts (lagged) ===')
print(cells.to_string())
print('\n=== Position diagnostics ===')
print(f'  Days at 1.0: {(position == 1.0).sum()}')
print(f'  Days at 0.0: {(position == 0.0).sum()}')
print(f'  Pct invested: {(position > 0).mean():.1%}')
print(
    f'  Params: exit_high={AR_EXIT_HIGH}, entry_low={AR_ENTRY_LOW}, '
    f'slope_days={AR_SLOPE_DAYS}, persist={AR_EXIT_PERSIST}'
)

metrics_df = pd.DataFrame([jm_m, cascade_m, bnh_m]).set_index('Label')
print('\n=== Performance Metrics — v6 ===')
print(metrics_df.to_string())

compare_rows = []
for path, label, pattern in [
    (V1_METRICS_PATH, 'v1 JM (AR in features)', 'JM Strategy'),
    (V2_METRICS_PATH, 'v2 JM-only binary', 'JM Strategy'),
    (V5_METRICS_PATH, 'v5 hybrid (graded)', 'Hybrid'),
]:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        row = df.loc[df.index.str.contains(pattern, case=False, na=False)]
        if len(row):
            r = row.iloc[0].copy()
            r.name = label
            compare_rows.append(r)
compare_rows.append(cascade_m)
if compare_rows:
    print('\n=== Cross-version comparison ===')
    print(pd.DataFrame(compare_rows)[['CAGR', 'Sharpe', 'Max DD', 'Pct Invested']].to_string())

os.makedirs(OUTPUT_PATH, exist_ok=True)
regime.to_csv(OUTPUT_PATH + 'regime_series.csv')
position.to_csv(OUTPUT_PATH + 'position_series.csv')
cells.to_csv(OUTPUT_PATH + 'cascade_cell_counts.csv')
metrics_df.to_csv(OUTPUT_PATH + 'metrics.csv')
print(f'\nSaved to {OUTPUT_PATH}')

plot_results(
    jm_ret,
    bnh_ret,
    regime,
    gated_returns=cascade_ret,
    gated_label='v6 AR-first Cascade',
    save_path=OUTPUT_PATH + 'equity_curves.png',
    **_OOS,
)
print('Saved: equity_curves.png')
