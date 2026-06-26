import numpy as np
import pandas as pd

ALL_TICKERS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY',
               'SPY', 'QQQ', 'RSP']


def load_etf_data(path: str) -> tuple:
    """
    Load ETF OHLCV data and return (adj_prices, log_returns).

    Parameters
    ----------
    path : str
        Path to the CSV file with columns:
        Ticker, ETF_Group, Date, Open, High, Low, Close, Adj_Close, Volume

    Returns
    -------
    adj : pd.DataFrame
        Wide-format adjusted close prices, shape (T, 13), DatetimeIndex.
    log_rets : pd.DataFrame
        Wide-format log returns, shape (T, 13), DatetimeIndex.
        First row will be NaN (no prior price).
    """
    df = pd.read_csv(path, parse_dates=['Date'])

    adj = (
        df.pivot(index='Date', columns='Ticker', values='Adj_Close')
        .sort_index()
    )

    # Drop rows where every column is NaN (weekends / holidays in raw data)
    adj = adj.dropna(how='all')

    # Forward-fill at most 1 day for genuine corporate-action gaps
    adj = adj.ffill(limit=1)

    # Drop any rows that still have NaNs after the single-day fill
    adj = adj.dropna(how='any')

    # Validate all required tickers are present
    missing = [t for t in ALL_TICKERS if t not in adj.columns]
    if missing:
        raise ValueError(f"Missing tickers in data: {missing}")

    # Keep only the 13 required tickers in a consistent column order
    adj = adj[ALL_TICKERS]

    log_rets = np.log(adj / adj.shift(1))

    return adj, log_rets
