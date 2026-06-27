"""
Statistical Jump Model — Shu, Yu & Mulvey (2024).

Minimises:
    J(S, θ) = Σ_t 0.5 ||x_t - θ_{s_t}||² + λ Σ_{t=1}^{T-1} 1(s_{t-1} ≠ s_t)

The DP solve is a Viterbi-style forward pass + backward traceback.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


from src.config import LAMBDA_CANDIDATES as DEFAULT_LAMBDA_CANDIDATES

LAMBDA_CANDIDATES = DEFAULT_LAMBDA_CANDIDATES


# ---------------------------------------------------------------------------
# Core DP
# ---------------------------------------------------------------------------

def _dp_solve(X: np.ndarray, centroids: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Viterbi-style forward pass + backward traceback.

    Parameters
    ----------
    X          : (T, D) standardised feature matrix
    centroids  : (2, D) centroid matrix
    lambda_    : scalar jump penalty

    Returns
    -------
    S : (T,) int array with values in {0, 1}
    """
    T = len(X)
    K = 2
    INF = 1e18

    cost = np.full((T, K), INF)
    prev = np.zeros((T, K), dtype=np.int32)

    # t=0: no jump term
    for k in range(K):
        cost[0, k] = 0.5 * np.sum((X[0] - centroids[k]) ** 2)

    # Forward pass
    for t in range(1, T):
        for k in range(K):
            fit_cost = 0.5 * np.sum((X[t] - centroids[k]) ** 2)
            stay = cost[t - 1, k]
            jump = cost[t - 1, 1 - k] + lambda_
            if stay <= jump:
                cost[t, k] = stay + fit_cost
                prev[t, k] = k
            else:
                cost[t, k] = jump + fit_cost
                prev[t, k] = 1 - k

    # Backward traceback
    S = np.zeros(T, dtype=np.int32)
    S[T - 1] = int(np.argmin(cost[T - 1]))
    for t in range(T - 2, -1, -1):
        S[t] = prev[t + 1, S[t + 1]]

    return S


# ---------------------------------------------------------------------------
# JumpModel class
# ---------------------------------------------------------------------------

class JumpModel:
    """
    Jump Model for regime detection.

    After fitting, regime labels follow the convention:
        1 = bull  (invested)
        0 = bear  (cash)

    Bull/bear assignment is determined by the dd_10 (feature index 0) centroid:
    the state with the LOWER dd_10 centroid is labelled bull.
    """

    def __init__(
        self,
        lambda_: float,
        n_states: int = 2,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
    ):
        self.lambda_ = lambda_
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.centroids_: np.ndarray | None = None
        self.states_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _kmeans_pp_init(self, X: np.ndarray) -> np.ndarray:
        """k-means++ initialisation. Returns centroids of shape (2, D)."""
        T = len(X)
        idx0 = np.random.randint(T)
        centroids = [X[idx0].copy()]

        dists = np.sum((X - centroids[0]) ** 2, axis=1)
        probs = dists / dists.sum()
        idx1 = np.random.choice(T, p=probs)
        centroids.append(X[idx1].copy())

        return np.array(centroids)

    # ------------------------------------------------------------------
    # Single restart
    # ------------------------------------------------------------------

    def _fit_single(self, X: np.ndarray):
        """One random restart. Returns (objective, states, centroids)."""
        centroids = self._kmeans_pp_init(X)
        prev_obj = np.inf

        for _ in range(self.max_iter):
            S = _dp_solve(X, centroids, self.lambda_)

            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_states):
                mask = S == k
                if mask.sum() == 0:
                    new_centroids[k] = centroids[k]
                else:
                    new_centroids[k] = X[mask].mean(axis=0)

            # Vectorised objective
            fit_costs = 0.5 * np.sum((X - new_centroids[S]) ** 2, axis=1).sum()
            jump_term = self.lambda_ * np.sum(S[1:] != S[:-1])
            obj = fit_costs + jump_term

            centroids = new_centroids

            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj

        return obj, S, centroids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> 'JumpModel':
        """
        Fit with n_init random restarts; keep the lowest-objective solution.

        Bull/bear labelling: state with lower dd_10 centroid (index 0) = bull.
        After fitting, self.states_ uses 1=bull, 0=bear.
        """
        best_obj = np.inf
        best_S = None
        best_c = None

        for _ in range(self.n_init):
            obj, S, c = self._fit_single(X)
            if obj < best_obj:
                best_obj = obj
                best_S = S.copy()
                best_c = c.copy()

        # Bull = state with lower dd_10 centroid (feature index 0)
        if best_c[0][0] <= best_c[1][0]:
            bull_state = 0
        else:
            bull_state = 1
            best_S = 1 - best_S   # flip so that bull_state becomes 0 after remap

        # Remap: 1 = bull, 0 = bear
        self.states_ = (best_S == bull_state).astype(np.int32)
        self.centroids_ = best_c
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Online inference using fitted centroids + DP.

        Returns regime array (1=bull, 0=bear).
        """
        S = _dp_solve(X, self.centroids_, self.lambda_)

        if self.centroids_[0][0] <= self.centroids_[1][0]:
            bull_state = 0
        else:
            bull_state = 1

        return (S == bull_state).astype(np.int32)


# ---------------------------------------------------------------------------
# Monthly walk-forward cross-validation
# ---------------------------------------------------------------------------

def run_walk_forward_cv(
    feature_df: pd.DataFrame,
    spy_returns: pd.Series,
    lookback_days: int = 1260,
    lambda_candidates: list = None,
    n_states: int = 2,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
) -> pd.Series:
    """
    Monthly walk-forward cross-validation for λ selection.

    For each calendar month start in the OOS period:
      1. Slice validation window: feature_df[t - lookback_days : t]
      2. For each λ: fit JumpModel on scaled window, compute Sharpe of
         regime-shifted strategy on the validation window
      3. Select λ* = argmax Sharpe
      4. Refit on validation window with λ*
      5. Predict regime for each day in the upcoming calendar month

    Parameters
    ----------
    feature_df : pd.DataFrame
        Raw (unscaled) feature matrix, DatetimeIndex.
    spy_returns : pd.Series
        SPY log returns aligned to (or broader than) feature_df.
    lookback_days : int
        Length of the rolling validation window in trading days.
    lambda_candidates : list
        Values of λ to search over.
    n_states, max_iter, tol, n_init : JumpModel hyper-parameters.

    Returns
    -------
    pd.Series with DatetimeIndex, name='regime', values in {0, 1}.
    regime at close of day t — position to hold from open of t+1.
    """
    if lambda_candidates is None:
        lambda_candidates = LAMBDA_CANDIDATES

    # Align spy_returns to feature_df index
    spy_ret = spy_returns.reindex(feature_df.index)

    feasible_start = feature_df.index[lookback_days]
    feasible_end = feature_df.index[-1]

    monthly_dates = pd.date_range(
        start=feasible_start,
        end=feasible_end,
        freq='MS',
    )

    all_regimes: dict = {}

    for month_start in monthly_dates:
        # Index position of month_start in feature_df
        loc = feature_df.index.searchsorted(month_start)
        if loc < lookback_days:
            continue

        # Validation window ends just before month_start
        val_start_loc = loc - lookback_days
        val_end_loc = loc   # exclusive

        X_val_raw = feature_df.iloc[val_start_loc:val_end_loc].values
        val_index = feature_df.index[val_start_loc:val_end_loc]
        spy_val = spy_ret.iloc[val_start_loc:val_end_loc]

        scaler = StandardScaler()
        X_val = scaler.fit_transform(X_val_raw)

        # Grid-search λ on validation window using regime separation score.
        # Score = |mean(bull_rets) - mean(bear_rets)| / pooled_std.
        # This is return-distribution agnostic, avoiding bull-market bias
        # that plagues Sharpe/Calmar on trending training windows.
        best_lambda = lambda_candidates[0]
        best_score  = -np.inf

        for lam in lambda_candidates:
            model = JumpModel(
                lambda_=lam,
                n_states=n_states,
                max_iter=max_iter,
                tol=tol,
                n_init=n_init,
            )
            model.fit(X_val)
            states = model.states_

            bull_rets = spy_val.values[states == 1]
            bear_rets = spy_val.values[states == 0]

            # Need at least a few days in each regime to compute a meaningful score
            if len(bull_rets) < 5 or len(bear_rets) < 5:
                score = -np.inf
            else:
                pooled_std = np.sqrt(
                    (bull_rets.var() * len(bull_rets) + bear_rets.var() * len(bear_rets))
                    / (len(bull_rets) + len(bear_rets))
                )
                if pooled_std == 0:
                    score = -np.inf
                else:
                    score = abs(bull_rets.mean() - bear_rets.mean()) / pooled_std

            if score > best_score:
                best_score  = score
                best_lambda = lam

        print(f"  {month_start.date()}  ->  lambda* = {best_lambda:>4},  Sep = {best_score:.4f}")

        # Refit with best λ on the same validation window
        best_model = JumpModel(
            lambda_=best_lambda,
            n_states=n_states,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
        )
        best_model.fit(X_val)

        # Determine the next month's date range
        next_month_start = month_start + pd.DateOffset(months=1)
        oos_mask = (
            (feature_df.index >= month_start) &
            (feature_df.index < next_month_start)
        )
        oos_index = feature_df.index[oos_mask]

        if len(oos_index) == 0:
            continue

        # Online inference: for each OOS day, use all data up to (and including) that day
        for day in oos_index:
            day_loc = feature_df.index.get_loc(day)
            # Inference window: the lookback_days up to and including this day
            inf_start = max(0, day_loc - lookback_days + 1)
            X_inf_raw = feature_df.iloc[inf_start: day_loc + 1].values
            X_inf = scaler.transform(X_inf_raw)
            regime_inf = best_model.predict(X_inf)
            # The regime for 'day' is the last element
            all_regimes[day] = int(regime_inf[-1])

    regime_series = pd.Series(all_regimes, name='regime')
    regime_series.index = pd.DatetimeIndex(regime_series.index)
    regime_series = regime_series.sort_index()
    return regime_series
