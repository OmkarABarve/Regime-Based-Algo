"""
v5 — Hybrid: v1 sticky JM regime (7 features incl. AR) + v4 graded AR overlay.

JM trained on full feature set (like v1) for ~7 regime switches.
AR z-score overlay applied separately (like v4) for fast exits and graded
re-entry — AR in JM features drives stickiness; overlay handles 2025-style
lag without waiting for JM bear flip.

Run:  python main5.py  →  ./output_v5/

Tuning: raise bull_floor or AR_EXIT_HIGH if CAGR is too low vs v1.
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
from src.features import build_features
from src.jump_model import run_walk_forward_cv
from src.backtest import run_backtest, run_backtest_asymmetric, plot_results
from src.config import OOS_START, OOS_END, LAMBDA_CANDIDATES
from src.utils import compute_metrics_full_sample

_OOS = {'oos_start': OOS_START, 'oos_end': OOS_END}

DATA_PATH   = './Data/etf_ohlcv_20160301_20251231.csv'
OUTPUT_PATH = './output_v5/'

SECTOR_TICKERS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

AR_WINDOW        = 126
AR_N_COMPONENTS  = 2
AR_ZSCORE_WINDOW = 126

# v4 overlay on v1 regime — looser exit / higher floor for return retention
AR_EXIT_HIGH       = 1.25
AR_ENTRY_LOW       = 0.0
AR_SLOPE_DAYS      = 2
AR_EXIT_PERSIST    = 3
BULL_FLOOR         = 0.75

LOOKBACK_YEARS     = 3
LOOKBACK_DAYS      = LOOKBACK_YEARS * 252
JM_N_STATES  = 2
JM_MAX_ITER  = 300
JM_TOL       = 1e-4
JM_N_INIT    = 10

V1_METRICS_PATH = './output_v1/metrics.csv'
V4_METRICS_PATH = './output_v4/metrics.csv'

print('=== Step 1: Loading data ===')
adj, log_rets = load_etf_data(DATA_PATH)
print(f'Loaded: {adj.shape[0]} trading days, {adj.shape[1]} tickers')

print('\n=== Step 2: Computing Absorption Ratio ===')
ar = compute_ar(
    log_rets, SECTOR_TICKERS, W=AR_WINDOW, n_components=AR_N_COMPONENTS,
)
ar_zscore = compute_ar_zscore(ar, fast_window=15, slow_window=AR_ZSCORE_WINDOW)
print(f'AR computed: {ar.notna().sum()} valid observations')

print('\n=== Step 3: Building feature matrix (7 features, v1-style) ===')
features = build_features(adj, log_rets, ar, ar_zscore)
print(
    f'Feature matrix: {features.shape}, '
    f'from {features.index[0].date()} to {features.index[-1].date()}'
)

print('\n=== Step 4: Walk-forward cross-validation ===')
regime = run_walk_forward_cv(
    feature_df=features,
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

print('\n=== Step 5: Backtest (v5 hybrid) ===')
print(f'Evaluation window: {OOS_START} -> {OOS_END or "last available"}')

jm_ret, bnh_ret, jm_m, bnh_m = run_backtest(regime, log_rets['SPY'], **_OOS)

hybrid_ret, _, hybrid_m, _, position = run_backtest_asymmetric(
    regime,
    log_rets['SPY'],
    ar_zscore=ar_zscore,
    ar_exit_high=AR_EXIT_HIGH,
    ar_entry_low=AR_ENTRY_LOW,
    ar_slope_days=AR_SLOPE_DAYS,
    exit_persist_days=AR_EXIT_PERSIST,
    bull_floor=BULL_FLOOR,
    strategy_label='v5 Hybrid (v1 regime + v4 overlay)',
    **_OOS,
)

print('\n=== Position diagnostics (v5) ===')
print(f'  Days at 1.0: {(position == 1.0).sum()}')
print(f'  Days at {BULL_FLOOR}: {(position == BULL_FLOOR).sum()}')
print(f'  Days at 0.0: {(position == 0.0).sum()}')
print(f'  Pct invested: {(position > 0).mean():.1%}')
print(
    f'  Params: exit_high={AR_EXIT_HIGH}, entry_low={AR_ENTRY_LOW}, '
    f'slope_days={AR_SLOPE_DAYS}, persist={AR_EXIT_PERSIST}, floor={BULL_FLOOR}'
)

metrics_df = pd.DataFrame([jm_m, hybrid_m, bnh_m]).set_index('Label')
bnh_check = compute_metrics_full_sample(log_rets['SPY'], OOS_START, OOS_END)

print('\n=== Performance Metrics — v5 ===')
print(metrics_df.to_string())

compare_rows = []
if os.path.exists(V1_METRICS_PATH):
    v1 = pd.read_csv(V1_METRICS_PATH, index_col=0)
    row = v1.loc[v1.index.str.contains('JM Strategy', case=False, na=False)]
    if len(row):
        row = row.iloc[0].copy()
        row.name = 'v1 JM (binary, sticky)'
        compare_rows.append(row)
if os.path.exists(V4_METRICS_PATH):
    v4 = pd.read_csv(V4_METRICS_PATH, index_col=0)
    row = v4.loc[v4.index.str.contains('Asymmetric', case=False, na=False)]
    if len(row):
        row = row.iloc[0].copy()
        row.name = 'v4 Asymmetric (JM-only regime)'
        compare_rows.append(row)
compare_rows.append(hybrid_m)
compare_df = pd.DataFrame(compare_rows)
if len(compare_df):
    print('\n=== v1 vs v4 vs v5 (strategy rows) ===')
    print(compare_df[['CAGR', 'Sharpe', 'Max DD', 'Pct Invested']].to_string())

os.makedirs(OUTPUT_PATH, exist_ok=True)
regime.to_csv(OUTPUT_PATH + 'regime_series.csv')
position.to_csv(OUTPUT_PATH + 'position_series.csv')
metrics_df.to_csv(OUTPUT_PATH + 'metrics.csv')
print(f'\nSaved to {OUTPUT_PATH}')

plot_results(
    jm_ret,
    bnh_ret,
    regime,
    gated_returns=hybrid_ret,
    gated_label='v5 Hybrid',
    save_path=OUTPUT_PATH + 'equity_curves.png',
    **_OOS,
)
print('Saved: equity_curves.png')
