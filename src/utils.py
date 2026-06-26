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
