import numpy as np
import pandas as pd


def build_features(
    adj: pd.DataFrame,
    log_rets: pd.DataFrame,
    ar: pd.Series,
    ar_zscore: pd.Series
) -> pd.DataFrame:
    """
    Build the 7-feature matrix used by the Jump Model.

    All features are computed at the close of day t using only information
    available at or before t. No shift(-1) is applied here.
    The returned DataFrame contains RAW unscaled features.
    StandardScaler must be applied inside the CV loop in jump_model.py.

    Parameters
    ----------
    adj : pd.DataFrame
        Wide-format adjusted close prices (T x 13).
    log_rets : pd.DataFrame
        Wide-format log returns (T x 13).
    ar : pd.Series
        Raw Absorption Ratio series (name='AR').
    ar_zscore : pd.Series
        AR z-score series (name='AR_zscore').

    Returns
    -------
    pd.DataFrame with columns:
        ['dd_10', 'sortino_20', 'sortino_60', 'ar', 'ar_zscore',
         'vol_ratio', 'breadth']
    Rows with any NaN have been dropped.
    """
    spy_excess = log_rets['SPY']

    # 1. EWM downside deviation, halflife=10, annualised
    downside_10 = spy_excess.clip(upper=0)
    dd_10 = (
        np.sqrt(
            downside_10.pow(2).ewm(halflife=10, adjust=False).mean()
        ) * np.sqrt(252)
    )

    # 2. EWM Sortino ratio, halflife=20
    ewm_ret_20 = spy_excess.ewm(halflife=20, adjust=False).mean() * 252
    ewm_dd_20 = (
        np.sqrt(
            spy_excess.clip(upper=0).pow(2)
            .ewm(halflife=20, adjust=False).mean()
        ) * np.sqrt(252)
    )
    sortino_20 = ewm_ret_20 / ewm_dd_20.replace(0, np.nan)

    # 3. EWM Sortino ratio, halflife=60
    ewm_ret_60 = spy_excess.ewm(halflife=60, adjust=False).mean() * 252
    ewm_dd_60 = (
        np.sqrt(
            spy_excess.clip(upper=0).pow(2)
            .ewm(halflife=60, adjust=False).mean()
        ) * np.sqrt(252)
    )
    sortino_60 = ewm_ret_60 / ewm_dd_60.replace(0, np.nan)

    # 4 & 5. AR and AR_zscore are passed in directly
    ar_feat = ar.rename('ar')
    arz_feat = ar_zscore.rename('ar_zscore')

    # 6. Short-term / long-term vol ratio
    vol_5 = log_rets['SPY'].rolling(5).std() * np.sqrt(252)
    vol_63 = log_rets['SPY'].rolling(63).std() * np.sqrt(252)
    vol_ratio = vol_5 / vol_63.replace(0, np.nan)

    # 7. Equal-weight breadth (RSP/SPY ratio, z-scored over 126-day window)
    raw_breadth = adj['RSP'] / adj['SPY']
    breadth = (
        (raw_breadth - raw_breadth.rolling(126).mean())
        / raw_breadth.rolling(126).std()
    )

    feature_df = pd.concat(
        [dd_10, sortino_20, sortino_60, ar_feat, arz_feat, vol_ratio, breadth],
        axis=1
    )
    feature_df.columns = [
        'dd_10', 'sortino_20', 'sortino_60',
        'ar', 'ar_zscore', 'vol_ratio', 'breadth'
    ]
    feature_df = feature_df.dropna()
    return feature_df
