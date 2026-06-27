"""
Load S&P 500 / 400 / 600 point-in-time constituent data.

Each file contains daily prices for stocks that were index members during
the PIT snapshot period. Used by v_final for stock-level AR and breadth.
"""

import numpy as np
import pandas as pd

DEFAULT_PIT_PATHS = {
    'sp500': './Data/sp500_pit_20160301_20251231.csv',
    'sp400': './Data/sp400_pit_20160301_20251231.csv',
    'sp600': './Data/sp600_pit_20160301_20251231.csv',
}


def load_pit_log_returns(
    paths: dict = None,
    top_n_by_mcap: int = None,
) -> pd.DataFrame:
    """
    Load PIT stock panels and return a wide log-return matrix (dates × stocks).

    Parameters
    ----------
    paths : dict, optional
        Mapping of label → CSV path. Defaults to S&P 500/400/600 files.
    top_n_by_mcap : int, optional
        If set, keep only the top-N names by market cap each day (speeds up PCA).

    Returns
    -------
    pd.DataFrame
        Daily log returns, DatetimeIndex, one column per stock ID.
    """
    if paths is None:
        paths = DEFAULT_PIT_PATHS

    chunks = []
    for path in paths.values():
        df = pd.read_csv(path, parse_dates=['DATE'])
        df = df.dropna(subset=['Price'])
        df = df[df['Price'] > 0]
        chunks.append(df[['DATE', 'ID', 'Price', 'Market_Cap']])

    raw = pd.concat(chunks, ignore_index=True)
    raw = raw.sort_values(['ID', 'DATE'])
    # Same ID can appear in multiple index files on one date — keep last row.
    raw = raw.drop_duplicates(subset=['DATE', 'ID'], keep='last')
    raw['log_ret'] = raw.groupby('ID')['Price'].transform(
        lambda s: np.log(s / s.shift(1))
    )

    log_rets = raw.pivot(index='DATE', columns='ID', values='log_ret')
    mcap = raw.pivot(index='DATE', columns='ID', values='Market_Cap')
    log_rets.index = pd.to_datetime(log_rets.index)
    mcap.index = pd.to_datetime(mcap.index)
    log_rets = log_rets.sort_index()
    mcap = mcap.sort_index()

    if top_n_by_mcap is not None:
        masked = log_rets.copy()
        for dt in log_rets.index:
            row_mcap = mcap.loc[dt].dropna()
            if row_mcap.empty:
                continue
            keep = row_mcap.nlargest(top_n_by_mcap).index
            drop = [c for c in masked.columns if c not in keep]
            masked.loc[dt, drop] = np.nan
        log_rets = masked

    return log_rets


def pit_breadth_zscore(
    log_rets: pd.DataFrame,
    ret_window: int = 20,
    z_window: int = 126,
) -> pd.Series:
    """
    Fraction of stocks with positive rolling mean return, z-scored over time.

    Replaces the RSP/SPY ETF breadth proxy when using the full PIT universe.
    """
    rolling = log_rets.rolling(ret_window, min_periods=max(ret_window // 2, 5)).mean()
    pct_pos = (rolling > 0).mean(axis=1)
    z = (pct_pos - pct_pos.rolling(z_window).mean()) / pct_pos.rolling(z_window).std()
    z.name = 'pit_breadth'
    return z
