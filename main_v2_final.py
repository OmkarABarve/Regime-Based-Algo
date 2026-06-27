"""
v2_final — Full PIT data + v2 2x2 grid model.

Data:
  - ETF CSV: SPY trade/benchmark, SPY-based JM vol features
  - S&P 500/400/600 PIT: stock-level AR (fragility axis) + PIT breadth (JM feature)

Pipeline (same structure as v2):
  1. Load ETF + PIT panels, align calendars
  2. PIT AR + z-score (reuses cache from output_v_final/ if present)
  3. PIT breadth for JM-only features (AR excluded from JM)
  4. Walk-forward JM on 5 features
  5. 2x2 grid backtest: JM bull/bear x AR stable/fragile -> 0 / 0.5 / 1.0

Run:  python main_v2_final.py  ->  ./output_v2_final/
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
from src.absorption_ratio import compute_ar_from_panel_cached, compute_ar_zscore
from src.features import build_features_jm_only_pit
from src.jump_model import run_walk_forward_cv
from src.backtest import (
    run_backtest,
    run_backtest_2x2,
    grid_cell_counts,
    plot_results,
)
from src.config import OOS_START, OOS_END, LAMBDA_CANDIDATES

_OOS = {'oos_start': OOS_START, 'oos_end': OOS_END}

ETF_PATH      = './Data/etf_ohlcv_20160301_20251231.csv'
OUTPUT_PATH   = './output_v2_final/'
AR_CACHE_PATH = './output_v_final/pit_ar_series.csv'

AR_WINDOW        = 126
AR_N_COMPONENTS  = 2
AR_ZSCORE_WINDOW = 126
AR_MIN_NAMES     = 50
AR_THRESHOLD     = 1.0

LOOKBACK_YEARS = 3
LOOKBACK_DAYS  = LOOKBACK_YEARS * 252
JM_N_STATES    = 2
JM_MAX_ITER    = 300
JM_TOL         = 1e-4
JM_N_INIT      = 10

V2_METRICS_PATH = './output_v2/metrics.csv'

print('=== v2_final: PIT AR + PIT breadth + v2 2x2 grid ===')

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

print('\n=== Step 4: PIT breadth + JM-only features (no AR in JM) ===')
breadth = pit_breadth_zscore(pit_rets)
features_jm = build_features_jm_only_pit(etf_rets, breadth)
print(
    f'Feature matrix: {features_jm.shape}, '
    f'from {features_jm.index[0].date()} to {features_jm.index[-1].date()}'
)

print('\n=== Step 5: Walk-forward cross-validation ===')
regime = run_walk_forward_cv(
    feature_df=features_jm,
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

print('\n=== Step 6: Backtest (2x2 grid) ===')
print(f'Evaluation window: {OOS_START} -> {OOS_END or "last available"}')

jm_ret, bnh_ret, jm_m, bnh_m = run_backtest(regime, etf_rets['SPY'], **_OOS)

grid_ret, _, grid_m, _, position = run_backtest_2x2(
    regime,
    etf_rets['SPY'],
    ar_zscore=ar_zscore,
    ar_threshold=AR_THRESHOLD,
    **_OOS,
)

cells = grid_cell_counts(regime, ar_zscore, ar_threshold=AR_THRESHOLD)
print('\n=== 2x2 Grid cell occupancy (lagged) ===')
print(cells.to_string())
print(f'\nDays at 0.5 exposure: {(position == 0.5).sum()}')
print(f'Days at 1.0 exposure: {(position == 1.0).sum()}')
print(f'Days at 0.0 exposure: {(position == 0.0).sum()}')

metrics_df = pd.DataFrame([jm_m, grid_m, bnh_m]).set_index('Label')
print('\n=== Performance Metrics — v2_final ===')
print(metrics_df.to_string())

if os.path.exists(V2_METRICS_PATH):
    v2 = pd.read_csv(V2_METRICS_PATH, index_col=0)
    row = v2.loc[v2.index.str.contains('2', case=False, na=False) & v2.index.str.contains('Grid', case=False, na=False)]
    if len(row) == 0:
        row = v2.loc[v2.index.str.contains('Grid', case=False, na=False)]
    if len(row):
        print('\n=== v2 vs v2_final (grid rows) ===')
        compare = pd.DataFrame([
            row.iloc[0],
            grid_m,
        ], index=['v2 ETF 2x2', 'v2_final PIT 2x2'])
        print(compare[['CAGR', 'Sharpe', 'Max DD', 'Pct Invested']].to_string())

os.makedirs(OUTPUT_PATH, exist_ok=True)
regime.to_csv(OUTPUT_PATH + 'regime_series.csv')
position.to_csv(OUTPUT_PATH + 'position_series.csv')
metrics_df.to_csv(OUTPUT_PATH + 'metrics.csv')
cells.to_csv(OUTPUT_PATH + 'grid_cell_counts.csv')
print(f'\nSaved to {OUTPUT_PATH}')

plot_results(
    jm_ret,
    bnh_ret,
    regime,
    gated_returns=grid_ret,
    gated_label='v2_final PIT 2x2 Grid',
    save_path=OUTPUT_PATH + 'equity_curves.png',
    **_OOS,
)
print('Saved: equity_curves.png')
