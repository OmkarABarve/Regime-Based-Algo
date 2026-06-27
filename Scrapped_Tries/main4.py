"""
v4 — JM-only regime + graded asymmetric AR overlay.

JM fit on SPY microstructure only (no AR in features). AR z-score is a
one-sided overlay: fast exit on high/rising AR; JM bull re-enters at
bull_floor (0.5) minimum; full 1.0 when AR calms.

Run:  python main4.py  →  ./output_v4/

Other versions: v1–v3 main1–3 | v5 hybrid main5.py / main.py (v1 regime + v4 overlay)
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
from src.backtest import run_backtest, run_backtest_asymmetric, plot_results
from src.config import OOS_START, OOS_END, LAMBDA_CANDIDATES
from src.utils import compute_metrics_full_sample

_OOS = {'oos_start': OOS_START, 'oos_end': OOS_END}

DATA_PATH   = './Data/etf_ohlcv_20160301_20251231.csv'
OUTPUT_PATH = './output_v4/'

SECTOR_TICKERS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

AR_WINDOW        = 126
AR_N_COMPONENTS  = 2
AR_ZSCORE_WINDOW = 126

AR_EXIT_HIGH       = 1.0
AR_ENTRY_LOW       = 0.0
AR_SLOPE_DAYS      = 2
AR_EXIT_PERSIST    = 3
BULL_FLOOR         = 0.5

LOOKBACK_YEARS     = 3
LOOKBACK_DAYS      = LOOKBACK_YEARS * 252
JM_N_STATES  = 2
JM_MAX_ITER  = 300
JM_TOL       = 1e-4
JM_N_INIT    = 10

print('=== Step 1: Loading data ===')
adj, log_rets = load_etf_data(DATA_PATH)
print(f'Loaded: {adj.shape[0]} trading days, {adj.shape[1]} tickers')

print('\n=== Step 2: Computing Absorption Ratio ===')
ar = compute_ar(
    log_rets, SECTOR_TICKERS, W=AR_WINDOW, n_components=AR_N_COMPONENTS,
)
ar_zscore = compute_ar_zscore(ar, fast_window=15, slow_window=AR_ZSCORE_WINDOW)
print(f'AR computed: {ar.notna().sum()} valid observations')

print('\n=== Step 3: Building JM-only feature matrix ===')
features_jm = build_features_jm_only(adj, log_rets)
print(
    f'Feature matrix: {features_jm.shape}, '
    f'from {features_jm.index[0].date()} to {features_jm.index[-1].date()}'
)

print('\n=== Step 4: Walk-forward cross-validation ===')
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
print(f'Regime series: {len(regime)} days OOS, bull%={regime.mean():.1%}')
print(f'  Switches: {(regime.diff().abs().dropna() > 0).sum()}')

print('\n=== Step 5: Backtest (v4) ===')
print(f'Evaluation window: {OOS_START} -> {OOS_END or "last available"}')

jm_ret, bnh_ret, jm_m, bnh_m = run_backtest(regime, log_rets['SPY'], **_OOS)

asym_ret, _, asym_m, _, position = run_backtest_asymmetric(
    regime,
    log_rets['SPY'],
    ar_zscore=ar_zscore,
    ar_exit_high=AR_EXIT_HIGH,
    ar_entry_low=AR_ENTRY_LOW,
    ar_slope_days=AR_SLOPE_DAYS,
    exit_persist_days=AR_EXIT_PERSIST,
    bull_floor=BULL_FLOOR,
    strategy_label='JM + AR Asymmetric (v4)',
    **_OOS,
)

print('\n=== Position diagnostics ===')
print(f'  Days at 1.0: {(position == 1.0).sum()}')
print(f'  Days at 0.5: {(position == 0.5).sum()}')
print(f'  Days at 0.0: {(position == 0.0).sum()}')
print(f'  Pct invested: {(position > 0).mean():.1%}')

metrics_df = pd.DataFrame([jm_m, asym_m, bnh_m]).set_index('Label')
print('\n=== Performance Metrics — v4 ===')
print(metrics_df.to_string())

os.makedirs(OUTPUT_PATH, exist_ok=True)
regime.to_csv(OUTPUT_PATH + 'regime_series.csv')
position.to_csv(OUTPUT_PATH + 'position_series.csv')
metrics_df.to_csv(OUTPUT_PATH + 'metrics.csv')
print(f'\nSaved to {OUTPUT_PATH}')

plot_results(
    jm_ret,
    bnh_ret,
    regime,
    gated_returns=asym_ret,
    gated_label='JM + AR Asymmetric (v4)',
    save_path=OUTPUT_PATH + 'equity_curves.png',
    **_OOS,
)
