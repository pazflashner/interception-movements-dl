"""
Feature extraction from preprocessed trials.

Extracts kinematic summary features used for baseline analyses
and VAE evaluation probes.

Units
-----
Timing features are in **seconds**. Spatial quantities remain in the raw
tracker coordinate unit because the supplied CSV specification does not state
whether x/y/z are centimetres or millimetres. Temporal normalisation resamples
every trial onto the same 0-100 % axis, so a gradient taken on ``pos_norm`` is a shape derivative
whose scale depends on how long the movement lasted. Re-attaching the movement
duration converts it back to a physical velocity, which is what makes speed
comparable across trials and subjects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocessing import lowpass_filter


#: Kinematic features that are defined for a *generated* trajectory too — i.e.
#: everything except trial identifiers and task metadata. The generative-fidelity
#: tests compare empirical against generated samples over exactly these.
KINEMATIC_FEATURES = [
    "initiation_time_s",
    "movement_time_s",
    "peak_speed_tracker_units_s",
    "time_to_peak_speed",
    "path_length",
    "straight_line_dist",
    "curvature_index",
    "max_lateral_deviation",
    "n_submovements",
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
    Kinematic features from a raw (T, D) trajectory plus its timing.

    Split out from ``compute_trial_features`` so that trajectories *generated*
    by the decoder are described by exactly the same code as recorded ones —
    otherwise a generative-fidelity comparison is partly measuring a difference
    between two feature implementations.
    """
    sample_hz = (len(pos) - 1) / max(float(movement_time_s), 1e-3)
    cutoff = min(config.LOWPASS_CUTOFF_HZ, 0.45 * sample_hz)
    pos = lowpass_filter(np.asarray(pos, dtype=float), cutoff=cutoff, fs=sample_hz)
    vel = np.gradient(pos, 1.0 / sample_hz, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # speed is |Δpos| per normalised frame; the movement spans movement_time_s
    # over len(speed) - 1 intervals, so dividing by that step recovers mm/s.
    # The physical movement duration restores tracker units per second after
    # temporal normalization to a fixed number of samples.
    peak_speed = float(np.max(speed))
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

    # This is only a speed-peak heuristic. The final study uses Prof. Friedman's
    # minimum-jerk decomposition for submovement claims.
    smax = float(np.max(speed))
    if smax > 0:
        min_distance = max(1, int(round(0.050 * sample_hz)))
        peaks, _ = find_peaks(speed, prominence=0.10 * smax, distance=min_distance)
        n_submovements = float(max(len(peaks), 1))
    else:
        n_submovements = 1.0

    return {
        "initiation_time_s": float(initiation_time_s),
        "movement_time_s": float(movement_time_s),
        "peak_speed_tracker_units_s": peak_speed,
        # Backward-compatible alias for legacy scripts. It must not be used in
        # corrected reports because the physical position unit is unconfirmed.
        "peak_speed_mm_s": peak_speed,
        "time_to_peak_speed": time_to_peak,
        "path_length": path_length,
        "straight_line_dist": straight_dist,
        "curvature_index": curvature_index,
        "max_lateral_deviation": max_lat_dev,
        "speed_peak_count": n_submovements,
        # Legacy alias retained for old result readers. Do not label this as
        # minimum-jerk submovement count in new reports.
        "n_submovements": n_submovements,
        "end_x": float(end[0]),
        "end_y": float(end[1]) if len(end) > 1 else 0.0,
        "end_z": float(end[2]) if len(end) > 2 else 0.0,
    }


def compute_trial_features(trial: dict, fs: float = config.RECORDING_HZ) -> dict:
    """
    Compute scalar kinematic features from a preprocessed trial dict.

    Returns the entries of ``KINEMATIC_FEATURES`` plus trial identifiers and
    task metadata.
    """
    # Behavioral truth is always computed from finger movement onset to
    # arrival, regardless of which temporal window the CVAE receives.
    pos = trial.get("pos_movement_norm", trial["pos_norm"])
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
        "responseText": meta.get("responseText", ""),
        "successful": meta.get("successful", np.nan),
        **kin,
    }
    return features


def movement_from_generated_window(
    pos: np.ndarray,
    movement_time_s: float,
    initiation_time_s: float,
    window_mode: str,
    target_len: int = config.NORMALISED_LENGTH,
) -> np.ndarray:
    """Recover the generated movement interval from a model-window trajectory."""
    pos = np.asarray(pos, dtype=float)
    if window_mode == config.WINDOW_MOVEMENT_ONLY:
        return pos
    if window_mode != config.WINDOW_GO_TO_ARRIVAL:
        raise ValueError(f"unknown window mode: {window_mode!r}")
    total = max(float(movement_time_s) + float(initiation_time_s), 1e-6)
    onset_fraction = float(np.clip(initiation_time_s / total, 0.0, 0.98))
    phase = np.linspace(0.0, 1.0, len(pos))
    movement_phase = np.linspace(onset_fraction, 1.0, target_len)
    movement = np.column_stack([
        np.interp(movement_phase, phase, pos[:, axis]) for axis in range(pos.shape[1])
    ])
    return movement - movement[0]


def features_from_generated_window(
    pos: np.ndarray,
    movement_time_s: float,
    initiation_time_s: float,
    window_mode: str,
) -> dict:
    movement = movement_from_generated_window(
        pos, movement_time_s, initiation_time_s, window_mode
    )
    return features_from_arrays(movement, movement_time_s, initiation_time_s)


def extract_features_dataframe(trials: list[dict]) -> pd.DataFrame:
    """Extract features for all trials and return a DataFrame."""
    rows = [compute_trial_features(t) for t in trials]
    return pd.DataFrame(rows)
