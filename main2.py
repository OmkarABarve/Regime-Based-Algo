"""
v2 backup — 2×2 AR × JM grid (JM-only features + separate AR axis at backtest).

Run:  python main2.py
Snapshots: src/backtest2.py, src/features2.py, output_v2/
At runtime main2 imports live src/backtest.py (same as backtest2 when saved).

JM is fit on SPY microstructure features only (no AR). AR z-score provides
a separate fragility axis at backtest time:

    bull + stable  → 1.0    bull + fragile → 0.5
    bear + stable  → 0.0    bear + fragile → 0.0

Other backups: v1 → main1.py, output_v1/  |  current → python main.py
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
    run_backtest_2x2,
    grid_cell_counts,
    plot_results,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH   = './Data/etf_ohlcv_20160301_20251231.csv'
OUTPUT_PATH = './output/'

SECTOR_TICKERS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

AR_WINDOW        = 126
AR_N_COMPONENTS  = 2
AR_ZSCORE_WINDOW = 126
AR_THRESHOLD     = 1.0

LOOKBACK_YEARS     = 3
LOOKBACK_DAYS      = LOOKBACK_YEARS * 252
LAMBDA_CANDIDATES  = [5, 10, 15, 20, 25, 30]
JM_N_STATES  = 2
JM_MAX_ITER  = 300
JM_TOL       = 1e-4
JM_N_INIT    = 10

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print('=== Step 1: Loading data ===')
adj, log_rets = load_etf_data(DATA_PATH)
print(f'Loaded: {adj.shape[0]} trading days, {adj.shape[1]} tickers')

# ---------------------------------------------------------------------------
# 2. Absorption Ratio (fragility axis — separate from JM)
# ---------------------------------------------------------------------------
print('\n=== Step 2: Computing Absorption Ratio ===')
ar = compute_ar(
    log_rets,
    SECTOR_TICKERS,
    W=AR_WINDOW,
    n_components=AR_N_COMPONENTS,
)
ar_zscore = compute_ar_zscore(ar, fast_window=15, slow_window=AR_ZSCORE_WINDOW)
print(f'AR computed: {ar.notna().sum()} valid observations')

# ---------------------------------------------------------------------------
# 3. JM feature matrix (no AR — independent trend axis)
# ---------------------------------------------------------------------------
print('\n=== Step 3: Building JM-only feature matrix ===')
features_jm = build_features_jm_only(adj, log_rets)
print(
    f'Feature matrix: {features_jm.shape}, '
    f'from {features_jm.index[0].date()} to {features_jm.index[-1].date()}'
)

# ---------------------------------------------------------------------------
# 4. Walk-forward CV → regime series
# ---------------------------------------------------------------------------
print('\n=== Step 4: Walk-forward cross-validation (this may take a while) ===')
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
print(
    f'Regime series: {len(regime)} days OOS, '
    f'bull%={regime.mean():.1%}'
)

# ---------------------------------------------------------------------------
# 5. Backtest — 2×2 grid + binary JM baseline
# ---------------------------------------------------------------------------
print('\n=== Step 5: Backtest ===')

# Binary JM baseline (bear → cash, bull → full) for comparison
strat_ret, bnh_ret, strat_m, bnh_m = run_backtest(regime, log_rets['SPY'])

# 2×2 grid: JM trend × AR fragility
grid_ret, _, grid_m, _, position = run_backtest_2x2(
    regime,
    log_rets['SPY'],
    ar_zscore=ar_zscore,
    ar_threshold=AR_THRESHOLD,
)

cell_counts = grid_cell_counts(regime, ar_zscore, ar_threshold=AR_THRESHOLD)
print('\n=== 2×2 Grid cell occupancy (lagged) ===')
print(cell_counts.to_string())
print(f'\nDays at 0.5 exposure: {(position == 0.5).sum()}')
print(f'Days at 1.0 exposure: {(position == 1.0).sum()}')
print(f'Days at 0.0 exposure: {(position == 0.0).sum()}')

# ---------------------------------------------------------------------------
# 6. Print metrics
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame([strat_m, grid_m, bnh_m]).set_index('Label')
print('\n=== Performance Metrics ===')
print(metrics_df.to_string())

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
print('\nRegime series:')
print(f'  Start: {regime.index[0].date()}')
print(f'  End:   {regime.index[-1].date()}')
print(f'  Length: {len(regime)} days')
print(f'  Bull %: {regime.mean():.1%}')
print(f'  Number of switches: {(regime.diff().abs().dropna() > 0).sum()}')

print(f'\n=== Step 7: Saving outputs to {OUTPUT_PATH} ===')
os.makedirs(OUTPUT_PATH, exist_ok=True)
regime.to_csv(OUTPUT_PATH + 'regime_series.csv')
position.to_csv(OUTPUT_PATH + 'position_series.csv')
metrics_df.to_csv(OUTPUT_PATH + 'metrics.csv')
cell_counts.to_csv(OUTPUT_PATH + 'grid_cell_counts.csv')
print('Saved: regime_series.csv, position_series.csv, metrics.csv, grid_cell_counts.csv')

# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------
print('\n=== Step 8: Plotting ===')
plot_results(
    strat_ret,
    bnh_ret,
    regime,
    gated_returns=grid_ret,
    gated_label='2×2 Grid',
    save_path=OUTPUT_PATH + 'equity_curves.png',
)
print('Saved: equity_curves.png')
