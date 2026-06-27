"""
v_final — Full PIT data + v5 hybrid model.

Data:
  - ETF CSV: SPY trade/benchmark, SPY-based JM vol features
  - S&P 500/400/600 PIT: stock-level AR (rolling PCA) + breadth

Pipeline (same structure as v5):
  1. Load ETF + PIT panels, align calendars
  2. PIT AR + z-score (cached to output_v_final/pit_ar_series.csv)
  3. PIT breadth (% stocks with positive 20d rolling return)
  4. 7-feature JM walk-forward CV
  5. v5 asymmetric AR overlay backtest

Run:  python main_v_final.py  →  ./output_v_final/
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
from src.pit_loader import load_pit_log_returns, pit_breadth_zscore
from src.absorption_ratio import (
    compute_ar_from_panel_cached,
    compute_ar_zscore,
)
from src.features import build_features_pit
from src.jump_model import run_walk_forward_cv
from src.backtest import run_backtest, run_backtest_asymmetric, plot_results
from src.config import OOS_START, OOS_END, LAMBDA_CANDIDATES

_OOS = {'oos_start': OOS_START, 'oos_end': OOS_END}

ETF_PATH      = './Data/etf_ohlcv_20160301_20251231.csv'
OUTPUT_PATH   = './output_v_final/'
AR_CACHE_PATH = './output_v_final/pit_ar_series.csv'

AR_WINDOW        = 126
AR_N_COMPONENTS  = 2
AR_ZSCORE_WINDOW = 126
AR_MIN_NAMES     = 50

AR_EXIT_HIGH     = 1.25
AR_ENTRY_LOW     = 0.0
AR_SLOPE_DAYS    = 2
AR_EXIT_PERSIST  = 3
BULL_FLOOR       = 0.75

LOOKBACK_YEARS = 3
LOOKBACK_DAYS  = LOOKBACK_YEARS * 252
JM_N_STATES    = 2
JM_MAX_ITER    = 300
JM_TOL         = 1e-4
JM_N_INIT      = 10

V5_METRICS_PATH = './output_v5/metrics.csv'

print('=== v_final: PIT AR + PIT breadth + v5 hybrid ===')

print('\n=== Step 1: ETF data (SPY trade + JM vol) ===')
adj, etf_rets = load_etf_data(ETF_PATH)
print(f'ETF loaded: {adj.shape[0]} days, {adj.shape[1]} tickers')

print('\n=== Step 2: PIT data (S&P 500 + 400 + 600, full universe) ===')
pit_rets = load_pit_log_returns(top_n_by_mcap=None)
print(f'PIT panel: {pit_rets.shape[0]} days x {pit_rets.shape[1]} stocks')

common_idx = etf_rets.index.intersection(pit_rets.index)
adj = adj.loc[common_idx]
etf_rets = etf_rets.loc[common_idx]
pit_rets = pit_rets.reindex(common_idx)
print(
    f'Common calendar: {common_idx[0].date()} -> {common_idx[-1].date()} '
    f'({len(common_idx)} days)'
)

print('\n=== Step 3: Absorption Ratio from PIT universe ===')
ar = compute_ar_from_panel_cached(
    pit_rets,
    cache_path=AR_CACHE_PATH,
    W=AR_WINDOW,
    n_components=AR_N_COMPONENTS,
    min_names=AR_MIN_NAMES,
)
ar_zscore = compute_ar_zscore(ar, fast_window=15, slow_window=AR_ZSCORE_WINDOW)

print('\n=== Step 4: PIT breadth feature ===')
breadth = pit_breadth_zscore(pit_rets)

print('\n=== Step 5: Feature matrix (7 features, PIT AR + PIT breadth) ===')
features = build_features_pit(adj, etf_rets, ar, ar_zscore, breadth)
print(
    f'Feature matrix: {features.shape}, '
    f'from {features.index[0].date()} to {features.index[-1].date()}'
)

print('\n=== Step 6: Walk-forward cross-validation ===')
regime = run_walk_forward_cv(
    feature_df=features,
    spy_returns=etf_rets['SPY'],
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

print('\n=== Step 7: Backtest (v5 hybrid overlay) ===')
print(f'Evaluation window: {OOS_START} -> {OOS_END or "last available"}')

jm_ret, bnh_ret, jm_m, bnh_m = run_backtest(regime, etf_rets['SPY'], **_OOS)

hybrid_ret, _, hybrid_m, _, position = run_backtest_asymmetric(
    regime,
    etf_rets['SPY'],
    ar_zscore=ar_zscore,
    ar_exit_high=AR_EXIT_HIGH,
    ar_entry_low=AR_ENTRY_LOW,
    ar_slope_days=AR_SLOPE_DAYS,
    exit_persist_days=AR_EXIT_PERSIST,
    bull_floor=BULL_FLOOR,
    strategy_label='v_final PIT Hybrid',
    **_OOS,
)

print('\n=== Position diagnostics (v_final) ===')
print(f'  Days at 1.0: {(position == 1.0).sum()}')
print(f'  Days at {BULL_FLOOR}: {(position == BULL_FLOOR).sum()}')
print(f'  Days at 0.0: {(position == 0.0).sum()}')
print(f'  Pct invested: {(position > 0).mean():.1%}')
print(
    f'  Params: exit_high={AR_EXIT_HIGH}, entry_low={AR_ENTRY_LOW}, '
    f'slope_days={AR_SLOPE_DAYS}, persist={AR_EXIT_PERSIST}, floor={BULL_FLOOR}'
)

metrics_df = pd.DataFrame([jm_m, hybrid_m, bnh_m]).set_index('Label')
print('\n=== Performance Metrics — v_final ===')
print(metrics_df.to_string())

if os.path.exists(V5_METRICS_PATH):
    v5 = pd.read_csv(V5_METRICS_PATH, index_col=0)
    row = v5.loc[v5.index.str.contains('Hybrid', case=False, na=False)]
    if len(row):
        print('\n=== v5 vs v_final (hybrid rows) ===')
        compare = pd.DataFrame([
            row.iloc[0],
            hybrid_m,
        ], index=['v5 ETF hybrid', 'v_final PIT hybrid'])
        print(compare[['CAGR', 'Sharpe', 'Max DD', 'Pct Invested']].to_string())

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
    gated_label='v_final PIT Hybrid',
    save_path=OUTPUT_PATH + 'equity_curves.png',
    **_OOS,
)
print('Saved: equity_curves.png')
