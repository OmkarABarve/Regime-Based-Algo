"""
Shared evaluation window for cross-version comparison.

All main*.py scripts slice strategy and Buy & Hold metrics/plots to this
window so B&H Max DD and CAGR are comparable across v1/v2/v3 runs.

OOS_START is set to 2022-07-01 — the latest native start among feature sets
(7-feature matrix with AR). JM-only runs begin earlier but are clipped here.
"""

OOS_START = '2022-07-01'
OOS_END = None  # None → use last available date in the backtest

# Jump Model λ grid — monthly walk-forward CV picks one value per month.
# Lower λ → fewer jump penalties → more regime switches (more reactive).
# Higher λ → stickier regimes (fewer switches).
# Shu et al. use a wider range up to ~150; this grid extends down to 3 for
# comparison runs. Very low λ (1–2) can flicker and overfit the validation window.
LAMBDA_CANDIDATES = [3, 5, 10, 15, 20, 25, 30, 40, 50]
