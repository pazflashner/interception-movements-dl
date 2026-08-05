"""
Feature extraction from preprocessed trials.

Extracts kinematic summary features used for baseline analyses
and VAE evaluation probes.

Units
-----
Timing features are in **seconds** and speeds in **mm/s**, not in frames or
"per normalised frame". Temporal normalisation resamples every trial onto the
same 0-100 % axis, so a gradient taken on ``pos_norm`` is a shape derivative
whose scale depends on how long the movement lasted. Re-attaching the movement
duration converts it back to a physical velocity, which is what makes speed
comparable across trials and subjects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


#: Kinematic features that are defined for a *generated* trajectory too — i.e.
#: everything except trial identifiers and task metadata. The generative-fidelity
#: tests compare empirical against generated samples over exactly these.
KINEMATIC_FEATURES = [
    "initiation_time_s",
    "movement_time_s",
    "peak_speed_mm_s",
    "time_to_peak_speed",
    "path_length",
    "straight_line_dist",
    "curvature_index",
    "max_lateral_deviation",
    "end_x",
    "end_y",
    "end_z",
]


def features_from_arrays(
    pos: np.ndarray,
    movement_time_s: float,
    initiation_time_s: float,
) -> dict:
    """
    Kinematic features from a raw (T, 3) trajectory plus its timing.

    Split out from ``compute_trial_features`` so that trajectories *generated*
    by the decoder are described by exactly the same code as recorded ones —
    otherwise a generative-fidelity comparison is partly measuring a difference
    between two feature implementations.
    """
    vel = np.gradient(pos, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # speed is |Δpos| per normalised frame; the movement spans movement_time_s
    # over len(speed) - 1 intervals, so dividing by that step recovers mm/s.
    dt_phys = movement_time_s / max(len(speed) - 1, 1)
    peak_speed = float(np.max(speed) / dt_phys) if dt_phys > 0 else float("nan")
    ttp = int(np.argmax(speed))
    time_to_peak = ttp / max(len(speed) - 1, 1)

    diffs = np.diff(pos, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    start, end = pos[0], pos[-1]
    straight_dist = float(np.linalg.norm(end - start))
    curvature_index = path_length / max(straight_dist, 1e-6)

    if straight_dist > 1e-6:
        direction = (end - start) / straight_dist
        projections = np.dot(pos - start, direction)[:, None] * direction + start
        max_lat_dev = float(np.max(np.linalg.norm(pos - projections, axis=1)))
    else:
        max_lat_dev = 0.0

    return {
        "initiation_time_s": float(initiation_time_s),
        "movement_time_s": float(movement_time_s),
        "peak_speed_mm_s": peak_speed,
        "time_to_peak_speed": time_to_peak,
        "path_length": path_length,
        "straight_line_dist": straight_dist,
        "curvature_index": curvature_index,
        "max_lateral_deviation": max_lat_dev,
        "end_x": float(end[0]),
        "end_y": float(end[1]),
        "end_z": float(end[2]),
    }


def compute_trial_features(trial: dict, fs: float = config.RECORDING_HZ) -> dict:
    """
    Compute scalar kinematic features from a preprocessed trial dict.

    Returns the entries of ``KINEMATIC_FEATURES`` plus trial identifiers and
    task metadata.
    """
    pos = trial["pos_norm"]             # (T, 3)
    meta = trial["metadata"]

    # Timing, in seconds. Reaction time is measured from the go-signal
    # (object starts moving), not object appearance, so the randomised
    # foreperiod is excluded; falls back to the stimulus marker if absent.
    ref = trial.get("go_signal_idx", trial["stim_onset_idx"])
    init_time = (trial["move_start_idx"] - ref) / fs
    move_time = (trial["move_end_idx"] - trial["move_start_idx"]) / fs

    kin = features_from_arrays(pos, move_time, init_time)

    features = {
        "trial_id": meta.get("trial_id", ""),
        "subject": meta.get("subject", ""),
        "condition": meta.get("condition"),
        "sp": meta.get("sp"),
        "side": meta.get("side"),
        "rep": meta.get("rep"),
        "starting_side": meta.get("starting_side"),
        "starting_position_mm": meta.get("starting_position_mm"),
        **kin,
    }
    return features


def extract_features_dataframe(trials: list[dict]) -> pd.DataFrame:
    """Extract features for all trials and return a DataFrame."""
    rows = [compute_trial_features(t) for t in trials]
    return pd.DataFrame(rows)
