import numpy as np
import pandas as pd


def annualise_return(daily_returns: pd.Series, periods: int = 252) -> float:
    """Compound annualised return."""
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / periods
    return total ** (1 / n_years) - 1


def annualise_vol(daily_returns: pd.Series, periods: int = 252) -> float:
    """Annualised volatility."""
    return daily_returns.std() * np.sqrt(periods)


def sharpe_ratio(daily_returns: pd.Series, periods: int = 252) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate)."""
    if daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(periods)


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    cum = (1 + daily_returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return dd.min()


def calmar_ratio(daily_returns: pd.Series, periods: int = 252) -> float:
    """Calmar ratio: CAGR / abs(max drawdown)."""
    mdd = abs(max_drawdown(daily_returns))
    if mdd == 0:
        return 0.0
    ann_ret = annualise_return(daily_returns, periods)
    return ann_ret / mdd


def compute_metrics(daily_returns: pd.Series, label: str = 'Strategy') -> pd.Series:
    """
    Compute a standard set of performance metrics.

    Returns
    -------
    pd.Series with keys:
        Label, CAGR, Volatility, Sharpe, Max DD, Calmar, Pct Invested
    """
    return pd.Series({
        'Label':        label,
        'CAGR':         annualise_return(daily_returns),
        'Volatility':   annualise_vol(daily_returns),
        'Sharpe':       sharpe_ratio(daily_returns),
        'Max DD':       max_drawdown(daily_returns),
        'Calmar':       calmar_ratio(daily_returns),
        'Pct Invested': (daily_returns != 0).mean(),
    })


def slice_oos(
    series: pd.Series,
    oos_start: str = None,
    oos_end: str = None,
) -> pd.Series:
    """
    Restrict a time series to the common out-of-sample evaluation window.

    Parameters
    ----------
    series : pd.Series
        DatetimeIndex series to clip.
    oos_start : str or None
        Inclusive start date (YYYY-MM-DD).
    oos_end : str or None
        Inclusive end date (YYYY-MM-DD). None keeps through last observation.

    Returns
    -------
    pd.Series
        Clipped copy (may be empty if window does not overlap).
    """
    out = series.copy()
    if oos_start is not None:
        out = out.loc[out.index >= pd.Timestamp(oos_start)]
    if oos_end is not None:
        out = out.loc[out.index <= pd.Timestamp(oos_end)]
    return out


def compute_metrics_full_sample(
    spy_returns: pd.Series,
    oos_start: str = None,
    oos_end: str = None,
    label: str = 'Buy & Hold (OOS window)',
) -> pd.Series:
    """
    Buy-and-hold metrics on a fixed SPY sample (independent of regime length).

    Use with ``OOS_START`` / ``OOS_END`` from ``src.config`` so B&H benchmarks
    match across all main*.py versions.

    Parameters
    ----------
    spy_returns : pd.Series
        Daily SPY log returns (full history).
    oos_start, oos_end : str or None
        Evaluation window passed to ``slice_oos``.
    label : str
        Row label for the metrics table.

    Returns
    -------
    pd.Series
        Same schema as ``compute_metrics``.
    """
    windowed = slice_oos(spy_returns.dropna(), oos_start, oos_end)
    if windowed.empty:
        raise ValueError(
            f'No SPY observations in OOS window {oos_start!r} → {oos_end!r}'
        )
    return compute_metrics(windowed, label)
