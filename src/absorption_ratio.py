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
