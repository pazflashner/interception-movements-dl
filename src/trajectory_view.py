"""Consistent spatial view used by the final study.

The tracker records x/y/z, but the experimental movement occurs in the table
plane.  The final model therefore uses lateral x and forward y while retaining
the untouched three-dimensional trials for audit plots and future reruns.
"""
from __future__ import annotations

import numpy as np


MODEL_AXES = (0, 1)
MODEL_AXIS_NAMES = ("x_lateral", "y_forward")


def project_trial_to_table_plane(trial: dict) -> dict:
    """Return a shallow trial copy whose normalized trajectory is x-y only."""
    projected = dict(trial)
    projected["pos_norm"] = np.asarray(trial["pos_norm"])[:, MODEL_AXES].copy()
    if "vel_norm" in trial:
        projected["vel_norm"] = np.asarray(trial["vel_norm"])[:, MODEL_AXES].copy()
        projected["speed_norm"] = np.linalg.norm(projected["vel_norm"], axis=1)
    return projected


def project_trials_to_table_plane(trials: list[dict]) -> list[dict]:
    """Project a trial collection without modifying the cached source data."""
    return [project_trial_to_table_plane(trial) for trial in trials]
