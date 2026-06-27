import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.utils import compute_metrics, max_drawdown, slice_oos

# 2×2 grid: (JM bull/bear, AR stable/fragile) → SPY weight
DEFAULT_GRID_ALLOC = {
    (1, False): 1.0,  # bull + stable
    (1, True):  0.5,  # bull + fragile
    (0, False): 0.0,  # bear + stable
    (0, True):  0.0,  # bear + fragile
}


def _apply_oos_window(
    strategy_returns: pd.Series,
    bnh_returns: pd.Series,
    oos_start: str = None,
    oos_end: str = None,
) -> tuple:
    """Clip strategy and B&H return series to the common evaluation window."""
    if oos_start is None and oos_end is None:
        return strategy_returns, bnh_returns
    return (
        slice_oos(strategy_returns, oos_start, oos_end),
        slice_oos(bnh_returns, oos_start, oos_end),
    )


def run_backtest(
    regime_series: pd.Series,
    spy_returns: pd.Series,
    ar_zscore: pd.Series = None,
    ar_threshold: float = 1.0,
    oos_start: str = None,
    oos_end: str = None,
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
    oos_start, oos_end : str or None
        If set, clip returns to this evaluation window before metrics
        (see ``src.config.OOS_START``).

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

    strategy_returns, bnh_returns = _apply_oos_window(
        strategy_returns, bnh_returns, oos_start, oos_end
    )

    strategy_metrics = compute_metrics(strategy_returns, label)
    bnh_metrics = compute_metrics(bnh_returns, 'Buy & Hold')

    return strategy_returns, bnh_returns, strategy_metrics, bnh_metrics


def run_backtest_2x2(
    regime_series: pd.Series,
    spy_returns: pd.Series,
    ar_zscore: pd.Series,
    ar_threshold: float = 1.0,
    grid_alloc: dict = None,
    oos_start: str = None,
    oos_end: str = None,
) -> tuple:
    """
    Run a 2×2 AR × JM grid backtest with graded position sizing.

    JM (trend) and AR (fragility) are combined independently:

        |              | AR stable | AR fragile |
        |--------------|-----------|------------|
        | JM bull      | 1.0       | 0.5        |
        | JM bear      | 0.0       | 0.0        |

    No look-ahead: both regime and AR z-score are shifted by one day before
    sizing the position held from open of t+1.

    Parameters
    ----------
    regime_series : pd.Series
        Daily JM regime labels (1=bull, 0=bear). Should be fit on JM-only
        features (no AR columns) for a clean 2×2 split.
    spy_returns : pd.Series
        Daily SPY log returns.
    ar_zscore : pd.Series
        Daily absorption-ratio z-score (fragility signal).
    ar_threshold : float, default 1.0
        Values above this mark AR as fragile.
    grid_alloc : dict, optional
        Mapping ``(jm_bull: int, ar_fragile: bool)`` → position weight.
        Defaults to ``DEFAULT_GRID_ALLOC``.
    oos_start, oos_end : str or None
        Common evaluation window clip (see ``src.config``).

    Returns
    -------
    strategy_returns, bnh_returns, strategy_metrics, bnh_metrics
    position_series : pd.Series — daily position weights (for diagnostics)
    """
    if grid_alloc is None:
        grid_alloc = DEFAULT_GRID_ALLOC

    common = regime_series.index.intersection(spy_returns.index)
    regime_aligned = regime_series.loc[common]
    spy_aligned = spy_returns.loc[common]

    regime_lag = regime_aligned.shift(1).fillna(0).astype(int)
    ar_shifted = ar_zscore.reindex(common).shift(1)
    ar_fragile = (ar_shifted > ar_threshold).fillna(False)

    position = pd.Series(0.0, index=common)
    bull = regime_lag == 1
    position[bull & ~ar_fragile] = grid_alloc[(1, False)]
    position[bull & ar_fragile] = grid_alloc[(1, True)]

    strategy_returns = position * spy_aligned
    bnh_returns = spy_aligned.copy()

    strategy_returns, bnh_returns = _apply_oos_window(
        strategy_returns, bnh_returns, oos_start, oos_end
    )
    position = slice_oos(position, oos_start, oos_end)

    strategy_metrics = compute_metrics(strategy_returns, '2×2 Grid')
    bnh_metrics = compute_metrics(bnh_returns, 'Buy & Hold')

    return strategy_returns, bnh_returns, strategy_metrics, bnh_metrics, position


def grid_cell_counts(
    regime_series: pd.Series,
    ar_zscore: pd.Series,
    ar_threshold: float = 1.0,
) -> pd.Series:
    """Count days in each 2×2 cell (after the 1-day lag)."""
    common = regime_series.index.intersection(ar_zscore.index)
    regime_lag = regime_series.loc[common].shift(1).fillna(0).astype(int)
    ar_shifted = ar_zscore.reindex(common).shift(1)
    ar_fragile = ar_shifted > ar_threshold

    labels = []
    for date in common:
        jm = int(regime_lag.loc[date])
        frag = bool(ar_fragile.loc[date]) if pd.notna(ar_fragile.loc[date]) else False
        if jm == 1 and not frag:
            labels.append('bull_stable')
        elif jm == 1 and frag:
            labels.append('bull_fragile')
        elif jm == 0 and not frag:
            labels.append('bear_stable')
        else:
            labels.append('bear_fragile')

    counts = pd.Series(labels).value_counts()
    counts.index.name = 'cell'
    return counts


def compute_asymmetric_position(
    regime_series: pd.Series,
    ar_zscore: pd.Series,
    ar_exit_high: float = 1.0,
    ar_entry_low: float = 0.0,
    ar_slope_days: int = 5,
    exit_persist_days: int = 3,
) -> pd.Series:
    """
    Two-stage asymmetric position sizing: JM primary, AR one-sided overlay.

    Signals at close of day t set position held from open of t+1.
    AR may accelerate exits (1→0) but never force re-entry without JM bull
    and AR below the entry threshold (dual-threshold hysteresis).

    Rules (evaluated in order each day):
      1. JM bearish                          → 0
      2. JM bullish, AR high and (rising or  → 0
         persisted ``exit_persist_days``)
      3. JM bullish and AR below entry low   → 1
      4. Otherwise                           → hold prior position

    Parameters
    ----------
    regime_series : pd.Series
        JM regime (1=bull, 0=bear), fit on JM-only features.
    ar_zscore : pd.Series
        Absorption ratio z-score (fragility).
    ar_exit_high : float
        Exit / stay-flat threshold (upper hysteresis band).
    ar_entry_low : float
        Re-entry threshold (lower hysteresis band); must be < ``ar_exit_high``.
    ar_slope_days : int
        Lookback for AR rising trigger (exit accelerator).
    exit_persist_days : int
        Consecutive days AR > ``ar_exit_high`` required for persistence exit.

    Returns
    -------
    pd.Series
        Daily position weights in {0.0, 1.0}, index aligned to overlap.
    """
    common = regime_series.index.intersection(ar_zscore.index)
    regime = regime_series.reindex(common).astype(float)
    ar = ar_zscore.reindex(common)

    position = pd.Series(0.0, index=common)
    prev_pos = 0.0

    for t in range(len(common)):
        if t == 0:
            position.iloc[t] = 0.0
            prev_pos = 0.0
            continue

        sig_idx = t - 1
        jm_bull = regime.iloc[sig_idx] == 1

        if not jm_bull:
            new_pos = 0.0
        else:
            ar_val = ar.iloc[sig_idx]
            if pd.notna(ar_val) and _ar_exit_trigger(
                ar, sig_idx, ar_exit_high, ar_slope_days, exit_persist_days
            ):
                new_pos = 0.0
            elif pd.notna(ar_val) and ar_val < ar_entry_low:
                new_pos = 1.0
            else:
                new_pos = prev_pos

        position.iloc[t] = new_pos
        prev_pos = new_pos

    return position


def _ar_exit_trigger(
    ar: pd.Series,
    idx: int,
    exit_high: float,
    slope_days: int,
    persist_days: int,
) -> bool:
    """True when AR z-score signals fragility (high + rising or persistent)."""
    ar_val = ar.iloc[idx]
    if pd.isna(ar_val) or ar_val <= exit_high:
        return False

    rising = False
    if idx >= slope_days and pd.notna(ar.iloc[idx - slope_days]):
        rising = ar_val > ar.iloc[idx - slope_days]

    persist = False
    if idx >= persist_days - 1:
        window = ar.iloc[idx - persist_days + 1: idx + 1]
        persist = (window > exit_high).all()

    return rising or persist


def run_backtest_asymmetric(
    regime_series: pd.Series,
    spy_returns: pd.Series,
    ar_zscore: pd.Series,
    ar_exit_high: float = 1.0,
    ar_entry_low: float = 0.0,
    ar_slope_days: int = 5,
    exit_persist_days: int = 3,
    oos_start: str = None,
    oos_end: str = None,
) -> tuple:
    """
    Backtest the two-stage asymmetric JM + AR fragility overlay.

    Parameters
    ----------
    oos_start, oos_end : str or None
        Common evaluation window clip (see ``src.config``).

    Returns
    -------
    strategy_returns, bnh_returns, strategy_metrics, bnh_metrics, position
    """
    common = regime_series.index.intersection(spy_returns.index)
    spy_aligned = spy_returns.loc[common]

    position = compute_asymmetric_position(
        regime_series,
        ar_zscore,
        ar_exit_high=ar_exit_high,
        ar_entry_low=ar_entry_low,
        ar_slope_days=ar_slope_days,
        exit_persist_days=exit_persist_days,
    )
    position = position.reindex(common).fillna(0.0)

    strategy_returns = position * spy_aligned
    bnh_returns = spy_aligned.copy()

    strategy_returns, bnh_returns = _apply_oos_window(
        strategy_returns, bnh_returns, oos_start, oos_end
    )
    position = slice_oos(position, oos_start, oos_end)

    strategy_metrics = compute_metrics(strategy_returns, 'JM + AR Asymmetric')
    bnh_metrics = compute_metrics(bnh_returns, 'Buy & Hold')

    return strategy_returns, bnh_returns, strategy_metrics, bnh_metrics, position


def plot_results(
    strategy_returns: pd.Series,
    bnh_returns: pd.Series,
    regime_series: pd.Series,
    gated_returns: pd.Series = None,
    gated_label: str = 'JM + AR Gate',
    save_path: str = None,
    oos_start: str = None,
    oos_end: str = None,
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
    oos_start, oos_end : str or None — clip all series to evaluation window
    """
    strategy_returns = slice_oos(strategy_returns, oos_start, oos_end)
    bnh_returns = slice_oos(bnh_returns, oos_start, oos_end)
    regime_series = slice_oos(regime_series, oos_start, oos_end)
    if gated_returns is not None:
        gated_returns = slice_oos(gated_returns, oos_start, oos_end)

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
