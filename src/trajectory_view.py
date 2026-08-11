"""Consistent spatial view used by the final study.

The tracker records x/y/z, but the experimental movement occurs in the table
plane.  The final model therefore uses lateral x and forward y while retaining
the untouched three-dimensional trials for audit plots and future reruns.
"""
from __future__ import annotations

import numpy as np


MODEL_AXES = (0, 1)
MODEL_AXIS_NAMES = ("x_lateral", "y_forward")

WINDOW_KEYS = {
    "movement_only": "pos_movement_norm",
    "go_to_arrival": "pos_go_to_arrival_norm",
}


def select_trial_window(trial: dict, window_mode: str) -> dict:
    """Select one cached temporal representation without mutating the trial."""
    if window_mode not in WINDOW_KEYS:
        raise ValueError(f"unknown window mode: {window_mode!r}")
    key = WINDOW_KEYS[window_mode]
    if key not in trial:
        raise KeyError(f"trial lacks {key}; rebuild the canonical dataset")
    selected = dict(trial)
    selected["pos_norm"] = np.asarray(trial[key]).copy()
    selected["vel_norm"] = np.gradient(selected["pos_norm"], axis=0)
    selected["speed_norm"] = np.linalg.norm(selected["vel_norm"], axis=1)
    selected["window_mode"] = window_mode
    selected["segment_start_idx"] = (
        int(trial["move_start_idx"])
        if window_mode == "movement_only"
        else int(trial["go_signal_idx"])
    )
    return selected


def select_trials_window(trials: list[dict], window_mode: str) -> list[dict]:
    return [select_trial_window(trial, window_mode) for trial in trials]


def project_trial_to_table_plane(trial: dict) -> dict:
    """Return a shallow trial copy whose normalized trajectory is x-y only."""
    projected = dict(trial)
    projected["pos_norm"] = np.asarray(trial["pos_norm"])[:, MODEL_AXES].copy()
    for key in WINDOW_KEYS.values():
        if key in trial:
            projected[key] = np.asarray(trial[key])[:, MODEL_AXES].copy()
    if "vel_norm" in trial:
        projected["vel_norm"] = np.asarray(trial["vel_norm"])[:, MODEL_AXES].copy()
        projected["speed_norm"] = np.linalg.norm(projected["vel_norm"], axis=1)
    return projected


def project_trials_to_table_plane(trials: list[dict]) -> list[dict]:
    """Project a trial collection without modifying the cached source data."""
    return [project_trial_to_table_plane(trial) for trial in trials]
