"""
Feature extraction from preprocessed trials.

Extracts kinematic summary features used for baseline analyses
and VAE evaluation probes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_trial_features(trial: dict) -> dict:
    """
    Compute scalar kinematic features from a preprocessed trial dict.

    Returns dict with:
        - initiation_time: frames from stimulus onset to movement start
        - movement_time: frames of movement duration
        - peak_speed: max speed during movement
        - time_to_peak_speed: normalised time of peak speed (0–1)
        - path_length: total Euclidean path length of movement
        - straight_line_dist: Euclidean distance start → end
        - curvature_index: path_length / straight_line_dist
        - max_lateral_deviation: maximum deviation from the straight line
        - end_x, end_y, end_z: endpoint position
    """
    pos = trial["pos_norm"]             # (T, 3)
    speed = trial["speed_norm"]         # (T,)
    meta = trial["metadata"]

    # Timing (in raw frame counts)
    init_time = trial["move_start_idx"] - trial["stim_onset_idx"]
    move_time = trial["move_end_idx"] - trial["move_start_idx"]

    # Speed metrics
    peak_speed = float(np.max(speed))
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
        "initiation_time_frames": init_time,
        "movement_time_frames": move_time,
        "peak_speed": peak_speed,
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
