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


def compute_trial_features(trial: dict, fs: float = config.RECORDING_HZ) -> dict:
    """
    Compute scalar kinematic features from a preprocessed trial dict.

    Returns dict with:
        - initiation_time_s: stimulus onset → movement start (reaction time)
        - movement_time_s: duration of the movement
        - peak_speed_mm_s: max speed during movement, in physical units
        - time_to_peak_speed: normalised time of peak speed (0–1)
        - path_length: total Euclidean path length of movement
        - straight_line_dist: Euclidean distance start → end
        - curvature_index: path_length / straight_line_dist
        - max_lateral_deviation: maximum deviation from the straight line
        - end_x, end_y, end_z: endpoint position
    """
    pos = trial["pos_norm"]             # (T, 3)
    speed = trial["speed_norm"]         # (T,) mm per normalised frame
    meta = trial["metadata"]

    # Timing, in seconds
    init_time = (trial["move_start_idx"] - trial["stim_onset_idx"]) / fs
    move_time = (trial["move_end_idx"] - trial["move_start_idx"]) / fs

    # Speed metrics. speed_norm is |Δpos| per normalised frame; the movement
    # spans move_time seconds over len(speed) - 1 intervals, so dividing by that
    # step recovers mm/s.
    dt_phys = move_time / max(len(speed) - 1, 1)
    peak_speed = float(np.max(speed) / dt_phys) if dt_phys > 0 else float("nan")
    ttp = int(np.argmax(speed))
    time_to_peak = ttp / max(len(speed) - 1, 1)

    # Path geometry
    diffs = np.diff(pos, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(np.sum(segment_lengths))

    start, end = pos[0], pos[-1]
    straight_dist = float(np.linalg.norm(end - start))
    curvature_index = path_length / max(straight_dist, 1e-6)

    # Lateral deviation from straight line
    if straight_dist > 1e-6:
        direction = (end - start) / straight_dist
        projections = np.dot(pos - start, direction)[:, None] * direction + start
        deviations = np.linalg.norm(pos - projections, axis=1)
        max_lat_dev = float(np.max(deviations))
    else:
        max_lat_dev = 0.0

    features = {
        "trial_id": meta.get("trial_id", ""),
        "subject": meta.get("subject", ""),
        "condition": meta.get("condition"),
        "sp": meta.get("sp"),
        "side": meta.get("side"),
        "rep": meta.get("rep"),
        "starting_side": meta.get("starting_side"),
        "starting_position_mm": meta.get("starting_position_mm"),
        "initiation_time_s": float(init_time),
        "movement_time_s": float(move_time),
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
    return features


def extract_features_dataframe(trials: list[dict]) -> pd.DataFrame:
    """Extract features for all trials and return a DataFrame."""
    rows = [compute_trial_features(t) for t in trials]
    return pd.DataFrame(rows)
