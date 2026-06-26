import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.utils import compute_metrics, max_drawdown


def run_backtest(
    regime_series: pd.Series,
    spy_returns: pd.Series,
    ar_zscore: pd.Series = None,
    ar_threshold: float = 1.0,
) -> tuple:
    """
    Run a regime-following backtest against SPY buy-and-hold, with optional
    AR-gated position sizing.

    The regime at close of day t drives the position held from open of t+1,
    implemented as regime_series.shift(1).

    When ``ar_zscore`` is supplied, the absorption-ratio z-score acts as a
    fragility gate. On any day where the regime says "bull" (position == 1)
    but the (lagged) AR z-score is elevated (> ``ar_threshold``), the position
    is halved to 0.5. This yields three exposure levels:

        0.0  → bear / cash
        0.5  → bull but fragile (high systemic risk)
        1.0  → bull and stable (low systemic risk)

    No look-ahead: ``ar_zscore`` is shifted by one day before being used,
    exactly like the regime signal.

    Parameters
    ----------
    regime_series : pd.Series
        Daily regime labels (1=bull/invested, 0=bear/cash).
    spy_returns : pd.Series
        Daily SPY log returns.
    ar_zscore : pd.Series, optional
        Daily absorption-ratio z-score. If provided, enables AR gating.
    ar_threshold : float, default 1.0
        Z-score level above which a bull-regime position is halved to 0.5.

    Returns
    -------
    strategy_returns : pd.Series
    bnh_returns      : pd.Series
    strategy_metrics : pd.Series   (from compute_metrics)
    bnh_metrics      : pd.Series   (from compute_metrics)
    """
    # Align on common dates
    common = regime_series.index.intersection(spy_returns.index)
    regime_aligned = regime_series.loc[common]
    spy_aligned = spy_returns.loc[common]

    # Regime at close t → position active from open t+1
    position = regime_aligned.shift(1).fillna(0)

    label = 'JM Strategy'
    if ar_zscore is not None:
        # Lag the AR z-score by one day so the gate uses only info known
        # at close of t to size the position held from open of t+1.
        ar_shifted = ar_zscore.reindex(common).shift(1)
        fragile = (position == 1) & (ar_shifted > ar_threshold)
        position = position.where(~fragile, 0.5)
        label = 'JM + AR Gate'

    strategy_returns = position * spy_aligned
    bnh_returns = spy_aligned.copy()

    strategy_metrics = compute_metrics(strategy_returns, label)
    bnh_metrics = compute_metrics(bnh_returns, 'Buy & Hold')

    return strategy_returns, bnh_returns, strategy_metrics, bnh_metrics


def plot_results(
    strategy_returns: pd.Series,
    bnh_returns: pd.Series,
    regime_series: pd.Series,
    gated_returns: pd.Series = None,
    gated_label: str = 'JM + AR Gate',
    save_path: str = None,
) -> None:
    """
    Produce a 3-panel figure:
      1. Equity curves (strategy=blue, buy-and-hold=grey, optional AR-gated
         strategy=darkorange) with bear shading
      2. Drawdown curves
      3. Regime step chart

    Parameters
    ----------
    strategy_returns : pd.Series
    bnh_returns      : pd.Series
    regime_series    : pd.Series  (1=bull, 0=bear)
    gated_returns    : pd.Series or None — if provided, the AR-gated strategy
                       returns are overlaid on the equity and drawdown panels
    gated_label      : str — legend label for the AR-gated series
    save_path        : str or None — if provided, save the figure to this path
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ---- Equity curves ------------------------------------------------
    ax = axes[0]
    cum_strat = (1 + strategy_returns).cumprod()
    cum_bnh = (1 + bnh_returns).cumprod()

    ax.plot(cum_strat.index, cum_strat.values, color='steelblue',
            linewidth=1.5, label='JM Strategy')
    ax.plot(cum_bnh.index, cum_bnh.values, color='grey',
            linewidth=1.5, label='Buy & Hold', alpha=0.8)

    if gated_returns is not None:
        cum_gated = (1 + gated_returns).cumprod()
        ax.plot(cum_gated.index, cum_gated.values, color='darkorange',
                linewidth=1.5, label=gated_label)

    _shade_bear(ax, regime_series)

    ax.set_title('Equity Curves — JM Strategy vs Buy & Hold', fontsize=13)
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # ---- Drawdown -------------------------------------------------------
    ax = axes[1]
    dd_strat = _drawdown_series(strategy_returns)
    dd_bnh = _drawdown_series(bnh_returns)

    ax.plot(dd_strat.index, dd_strat.values, color='steelblue',
            linewidth=1.5, label='JM Strategy')
    ax.plot(dd_bnh.index, dd_bnh.values, color='grey',
            linewidth=1.5, label='Buy & Hold', alpha=0.8)

    if gated_returns is not None:
        dd_gated = _drawdown_series(gated_returns)
        ax.plot(dd_gated.index, dd_gated.values, color='darkorange',
                linewidth=1.5, label=gated_label)

    _shade_bear(ax, regime_series)

    ax.set_title('Drawdown', fontsize=13)
    ax.set_ylabel('Drawdown')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # ---- Regime signal --------------------------------------------------
    ax = axes[2]
    ax.step(regime_series.index, regime_series.values,
            where='post', color='darkgreen', linewidth=1.2, label='Regime')
    ax.fill_between(regime_series.index, 0, regime_series.values,
                    step='post', alpha=0.25, color='darkgreen')

    ax.set_title('Regime Signal (1=Bull, 0=Bear)', fontsize=13)
    ax.set_ylabel('Regime')
    ax.set_xlabel('Date')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drawdown_series(daily_returns: pd.Series) -> pd.Series:
    cum = (1 + daily_returns).cumprod()
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max


def _shade_bear(ax: plt.Axes, regime_series: pd.Series) -> None:
    """Shade bear-regime (regime=0) periods on an axis."""
    in_bear = False
    bear_start = None

    for date, val in regime_series.items():
        if val == 0 and not in_bear:
            bear_start = date
            in_bear = True
        elif val == 1 and in_bear:
            ax.axvspan(bear_start, date, alpha=0.2, color='red', linewidth=0)
            in_bear = False

    if in_bear and bear_start is not None:
        ax.axvspan(bear_start, regime_series.index[-1],
                   alpha=0.2, color='red', linewidth=0)
