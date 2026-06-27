import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute_ar(
    log_rets: pd.DataFrame,
    sector_tickers: list,
    W: int = 252,
    n_components: int = 2
) -> pd.Series:
    """
    Compute the Absorption Ratio (Kritzman, Li, Page & Rigobon 2011).

    PCA is fit on raw log-returns (NOT standardised) over a rolling window.
    AR_t = sum of explained variance ratios for the top n_components eigenvectors.

    Parameters
    ----------
    log_rets : pd.DataFrame
        Wide-format log returns with at least sector_tickers columns.
    sector_tickers : list
        Tickers to include (typically 10 sector ETFs).
    W : int
        Rolling window length in trading days.
    n_components : int
        Number of eigenvectors to include in the numerator.

    Returns
    -------
    pd.Series with name='AR', DatetimeIndex aligned to log_rets.
    """
    sector_rets = log_rets[sector_tickers]
    ar_values = {}

    for t in range(W, len(sector_rets)):
        window = sector_rets.iloc[t - W: t]

        # Drop columns with any NaN in this window
        window = window.dropna(axis=1)

        if window.shape[1] < 5:
            ar_values[sector_rets.index[t]] = np.nan
            continue

        pca = PCA()
        pca.fit(window)
        ar_t = pca.explained_variance_ratio_[:n_components].sum()
        ar_values[sector_rets.index[t]] = ar_t

    return pd.Series(ar_values, name='AR')


def compute_ar_from_panel(
    log_rets: pd.DataFrame,
    W: int = 126,
    n_components: int = 2,
    min_names: int = 50,
) -> pd.Series:
    """
    Absorption Ratio on an arbitrary return panel (e.g. S&P 1500 PIT stocks).

    Same PCA logic as ``compute_ar`` but without sector-ETF column selection.
    Columns with any NaN inside the rolling window are dropped for that day.
    """
    ar_values = {}
    idx = log_rets.index

    for t in range(W, len(log_rets)):
        window = log_rets.iloc[t - W: t].dropna(axis=1, how='any')
        if window.shape[1] < min_names:
            ar_values[idx[t]] = np.nan
            continue
        pca = PCA()
        pca.fit(window.values)
        ar_values[idx[t]] = pca.explained_variance_ratio_[:n_components].sum()

    return pd.Series(ar_values, name='AR')


def compute_ar_from_panel_cached(
    log_rets: pd.DataFrame,
    cache_path: str,
    W: int = 126,
    n_components: int = 2,
    min_names: int = 50,
) -> pd.Series:
    """
    PIT-panel AR with optional disk cache (CSV + JSON metadata).

    Cache is reused when path exists and metadata matches panel bounds and params.
    """
    meta_path = cache_path.replace('.csv', '_meta.json')
    panel_start = log_rets.index.min()
    panel_end = log_rets.index.max()
    n_cols = log_rets.shape[1]

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        meta_ok = (
            meta.get('W') == W
            and meta.get('n_components') == n_components
            and meta.get('min_names') == min_names
            and meta.get('n_cols') == n_cols
            and pd.Timestamp(meta.get('panel_start')) == panel_start
            and pd.Timestamp(meta.get('panel_end')) == panel_end
        )
        if meta_ok:
            ar = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze('columns')
            ar.name = 'AR'
            print(
                f'  AR cache HIT: {cache_path} '
                f'({ar.notna().sum()} valid days)'
            )
            return ar.reindex(log_rets.index)

    print(
        f'  AR cache MISS — computing PCA on {n_cols} names '
        f'({panel_start.date()} -> {panel_end.date()})...'
    )
    t0 = time.perf_counter()
    ar = compute_ar_from_panel(
        log_rets, W=W, n_components=n_components, min_names=min_names,
    )
    elapsed = time.perf_counter() - t0
    print(f'  PIT AR computed in {elapsed:.1f}s ({ar.notna().sum()} valid days)')

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    ar.to_csv(cache_path)
    meta = {
        'W': W,
        'n_components': n_components,
        'min_names': min_names,
        'n_cols': n_cols,
        'panel_start': str(panel_start.date()),
        'panel_end': str(panel_end.date()),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return ar


def compute_ar_zscore(
    ar_series: pd.Series,
    fast_window: int = 15,
    slow_window: int = 252
) -> pd.Series:
    """
    Compute the AR z-score following Kritzman eq.(2):

        delta_AR = (AR_fast - AR_slow) / sigma_slow

    A high positive value indicates a short-term fragility spike (bearish).
    A high negative value indicates a fragility drop (bullish).

    Parameters
    ----------
    ar_series : pd.Series
        Raw AR values (name='AR').
    fast_window : int
        Short-term rolling mean window.
    slow_window : int
        Long-term rolling mean and std window.

    Returns
    -------
    pd.Series with name='AR_zscore'.
    """
    ar_fast = ar_series.rolling(fast_window).mean()
    ar_slow = ar_series.rolling(slow_window).mean()
    sigma_slow = ar_series.rolling(slow_window).std()

    zscore = (ar_fast - ar_slow) / sigma_slow
    zscore.name = 'AR_zscore'
    return zscore
