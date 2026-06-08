"""
Phase 2 – Polynomial spline baseline.

Fits piecewise cubic splines to each normalised trajectory as a
non-ML baseline for reconstruction quality.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import splrep, splev

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def fit_spline(
    pos: np.ndarray,
    n_knots: int = config.SPLINE_N_KNOTS,
    degree: int = config.SPLINE_DEGREE,
) -> np.ndarray:
    """
    Fit a piecewise polynomial spline to a (T, D) trajectory and return
    the reconstructed trajectory of the same shape.
    """
    T, D = pos.shape
    t = np.linspace(0, 1, T)
    # Interior knots evenly spaced
    knots = np.linspace(0, 1, n_knots + 2)[1:-1]
    reconstructed = np.zeros_like(pos)
    for d in range(D):
        tck = splrep(t, pos[:, d], t=knots, k=degree)
        reconstructed[:, d] = splev(t, tck)
    return reconstructed


def evaluate_spline_baseline(trials: list[dict], n_knots: int = config.SPLINE_N_KNOTS) -> dict:
    """
    Evaluate spline reconstruction MSE across all trials.

    Returns dict with per-trial MSEs and overall statistics.
    """
    mses = []
    for trial in trials:
        pos = trial["pos_norm"]
        recon = fit_spline(pos, n_knots=n_knots)
        mse = float(np.mean((pos - recon) ** 2))
        mses.append(mse)

    mses = np.array(mses)
    result = {
        "mean_mse": float(np.mean(mses)),
        "std_mse": float(np.std(mses)),
        "median_mse": float(np.median(mses)),
        "per_trial_mse": mses,
    }
    print(
        f"Spline baseline ({n_knots} knots): "
        f"MSE = {result['mean_mse']:.6f} ± {result['std_mse']:.6f}"
    )
    return result
