"""
v1 backup — JM with AR inside features + optional AR gate overlay.

Run:  python main1.py
New:  python main.py  (2×2 grid, JM-only features + separate AR axis)
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')   # non-interactive backend: saves to file, no popup window

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)

from src.data_loader import load_etf_data
from src.absorption_ratio import compute_ar, compute_ar_zscore
from src.features import build_features
from src.jump_model import run_walk_forward_cv
from src.backtest import run_backtest, plot_results
from src.utils import compute_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH   = './Data/etf_ohlcv_20160301_20251231.csv'
OUTPUT_PATH = './output/'

SECTOR_TICKERS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BROAD_TICKERS  = ['SPY', 'QQQ', 'RSP']
ALL_TICKERS    = SECTOR_TICKERS + BROAD_TICKERS

# AR parameters
AR_WINDOW        = 126   # half-year window — recovers ~1yr of feature history vs 252
AR_N_COMPONENTS  = 2
AR_ZSCORE_WINDOW = 126   # half-year window — same rationale

# JM parameters
LOOKBACK_YEARS     = 3
LOOKBACK_DAYS      = LOOKBACK_YEARS * 252
#LAMBDA_CANDIDATES  = [5, 10, 15, 20, 25, 30, 35, 40, 50,60, 70, 80, 100, 120, 150]
LAMBDA_CANDIDATES = [5, 10, 15, 20, 25, 30]
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
# 2. Compute Absorption Ratio
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
# 3. Build feature matrix
# ---------------------------------------------------------------------------
print('\n=== Step 3: Building feature matrix ===')
features = build_features(adj, log_rets, ar, ar_zscore)
print(
    f'Feature matrix: {features.shape}, '
    f'from {features.index[0].date()} to {features.index[-1].date()}'
)

# ---------------------------------------------------------------------------
# 4. Walk-forward CV → regime series
# ---------------------------------------------------------------------------
print('\n=== Step 4: Walk-forward cross-validation (this may take a while) ===')
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
print(
    f'Regime series: {len(regime)} days OOS, '
    f'bull%={regime.mean():.1%}'
)

# ---------------------------------------------------------------------------
# 5. Backtest
# ---------------------------------------------------------------------------
print('\n=== Step 5: Backtest ===')

# Base 2-state binary strategy (no AR gating)
strat_ret, bnh_ret, strat_m, bnh_m = run_backtest(regime, log_rets['SPY'])

# AR-gated strategy: halve exposure when regime is bull but AR z-score is high
gated_ret, _, gated_m, _ = run_backtest(
    regime,
    log_rets['SPY'],
    ar_zscore=ar_zscore,
    ar_threshold=2.0,
)

# ---------------------------------------------------------------------------
# 6. Print metrics
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame([strat_m, gated_m, bnh_m]).set_index('Label')
print('\n=== Performance Metrics ===')
print(metrics_df.to_string())

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
print("Regime series:")
print(f"  Start: {regime.index[0].date()}")
print(f"  End:   {regime.index[-1].date()}")
print(f"  Length: {len(regime)} days")
print(f"  Bull %: {regime.mean():.1%}")
print(f"  Number of switches: {(regime.diff().abs().dropna() > 0).sum()}")
print(f"\nFeature matrix:")
print(f"  Start: {features.index[0].date()}")
print(f"  End:   {features.index[-1].date()}")
print(f'\n=== Step 7: Saving outputs to {OUTPUT_PATH} ===')
os.makedirs(OUTPUT_PATH, exist_ok=True)
regime.to_csv(OUTPUT_PATH + 'regime_series.csv')
metrics_df.to_csv(OUTPUT_PATH + 'metrics.csv')
print('Saved: regime_series.csv, metrics.csv')

# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------
print('\n=== Step 8: Plotting ===')
plot_results(
    strat_ret,
    bnh_ret,
    regime,
    gated_returns=gated_ret,
    gated_label='JM + AR Gate',
    save_path=OUTPUT_PATH + 'equity_curves.png',
)
print('Saved: equity_curves.png')
