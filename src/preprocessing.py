"""
Preprocessing pipeline.

1. Identify movement onset (stimulus marker = 5) and movement offset.
2. Low-pass Butterworth filter (10 Hz cutoff at 240 Hz).
3. Temporal normalisation to T=100 frames via cubic spline interpolation.
4. Spatial normalisation (subtract initial position).
5. Velocity / acceleration computation on filtered data.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Low-pass Butterworth filter ───────────────────────────────────────────────
def _butter_lowpass(cutoff: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype="low", analog=False)


def lowpass_filter(
    signal: np.ndarray,
    cutoff: float = config.LOWPASS_CUTOFF_HZ,
    fs: float = config.RECORDING_HZ,
    order: int = config.LOWPASS_ORDER,
) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter along axis 0."""
    if len(signal) < 3 * (order + 1):
        return signal  # too short to filter
    b, a = _butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, signal, axis=0)


# ── Movement segmentation ────────────────────────────────────────────────────
def find_stimulus_onset(markers: np.ndarray) -> Optional[int]:
    """Return the frame index where marker == STIMULUS_ONSET_MARKER."""
    idxs = np.where(markers == config.STIMULUS_ONSET_MARKER)[0]
    return int(idxs[0]) if len(idxs) > 0 else None


def find_movement_window(
    pos: np.ndarray,
    stim_onset_idx: int,
    vel_threshold: float = 5.0,
    fs: float = config.RECORDING_HZ,
) -> tuple[int, int]:
    """
    Determine movement start and end indices based on velocity threshold.

    Movement start: first frame after stimulus onset where speed exceeds
    *vel_threshold* mm/s.
    Movement end: last frame (after start) where speed exceeds threshold,
    plus a small buffer.
    """
    dt = 1.0 / fs
    vel = np.gradient(pos, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    search_region = speed[stim_onset_idx:]
    above = np.where(search_region > vel_threshold)[0]

    if len(above) == 0:
        # Fallback: use entire post-stimulus region
        return stim_onset_idx, len(speed) - 1

    start = stim_onset_idx + above[0]
    end = stim_onset_idx + above[-1] + 1
    end = min(end + int(0.05 * fs), len(speed) - 1)  # 50 ms buffer
    return int(start), int(end)


# ── Temporal normalisation ────────────────────────────────────────────────────
def normalise_temporal(
    pos: np.ndarray, target_len: int = config.NORMALISED_LENGTH
) -> np.ndarray:
    """
    Resample a (N, D) position array to *target_len* frames using cubic
    spline interpolation (0–100 % of movement).
    """
    n = pos.shape[0]
    if n < 4:
        # Not enough points for cubic spline – use linear
        t_old = np.linspace(0, 1, n)
        t_new = np.linspace(0, 1, target_len)
        return np.column_stack(
            [np.interp(t_new, t_old, pos[:, d]) for d in range(pos.shape[1])]
        )

    t_old = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, target_len)
    cs = CubicSpline(t_old, pos, axis=0)
    return cs(t_new)


# ── Spatial normalisation ─────────────────────────────────────────────────────
def normalise_spatial(pos: np.ndarray) -> np.ndarray:
    """Subtract the initial position so every trial starts at the origin."""
    return pos - pos[0]


# ── Velocity / acceleration ──────────────────────────────────────────────────
def compute_velocity(pos: np.ndarray, fs: float = config.RECORDING_HZ) -> np.ndarray:
    """Central-difference velocity from position (N, D) → (N, D)."""
    return np.gradient(pos, 1.0 / fs, axis=0)


def compute_speed(vel: np.ndarray) -> np.ndarray:
    """Scalar speed from velocity array."""
    return np.linalg.norm(vel, axis=1)


# ── Full single-trial pipeline ───────────────────────────────────────────────
def preprocess_trial(
    df_trial: pd.DataFrame,
    filter_cutoff: float = config.LOWPASS_CUTOFF_HZ,
) -> Optional[dict]:
    """
    Run the complete preprocessing pipeline on one trial DataFrame.

    Returns a dict with:
        - pos_raw: (N, 3) raw positions
        - pos_filtered: (N, 3) low-pass filtered positions
        - pos_norm: (T, 3) temporally + spatially normalised trajectory
        - vel_norm: (T, 3) velocity on normalised trajectory
        - speed_norm: (T,) speed profile
        - stim_onset_idx: index of stimulus onset
        - move_start_idx, move_end_idx: movement window in raw data
        - metadata: dict of trial-level metadata
    """
    pos_raw = df_trial[["x", "y", "z"]].values.astype(float)
    markers = df_trial["marker"].values

    # 1. Stimulus onset
    stim_idx = find_stimulus_onset(markers)
    if stim_idx is None:
        return None  # skip trials without stimulus marker

    # 2. Low-pass filter
    pos_filtered = lowpass_filter(pos_raw, cutoff=filter_cutoff)

    # 3. Movement window
    move_start, move_end = find_movement_window(pos_filtered, stim_idx)
    movement = pos_filtered[move_start : move_end + 1]

    if len(movement) < 4:
        return None  # too short

    # 4. Temporal normalisation
    pos_norm = normalise_temporal(movement)

    # 5. Spatial normalisation
    pos_norm = normalise_spatial(pos_norm)

    # 6. Velocity on normalised trajectory
    # After normalisation the "sampling rate" is T points over the movement
    vel_norm = np.gradient(pos_norm, axis=0)
    speed_norm = np.linalg.norm(vel_norm, axis=1)

    meta_cols = [
        "subject", "condition", "sp", "side", "rep",
        "starting_position_mm", "starting_side", "trial_id",
    ]
    metadata = {c: df_trial.iloc[0][c] for c in meta_cols if c in df_trial.columns}

    return {
        "pos_raw": pos_raw,
        "pos_filtered": pos_filtered,
        "pos_norm": pos_norm,
        "vel_norm": vel_norm,
        "speed_norm": speed_norm,
        "stim_onset_idx": stim_idx,
        "move_start_idx": move_start,
        "move_end_idx": move_end,
        "metadata": metadata,
    }


# ── Batch preprocessing ──────────────────────────────────────────────────────
def preprocess_dataset(dataset: pd.DataFrame) -> list[dict]:
    """
    Apply *preprocess_trial* to every trial in the dataset.
    Returns a list of result dicts (trials that fail preprocessing are skipped).
    """
    from tqdm import tqdm

    results = []
    trial_ids = dataset["trial_id"].unique()

    for tid in tqdm(trial_ids, desc="Preprocessing trials"):
        df_trial = dataset[dataset["trial_id"] == tid].sort_values("frame")
        out = preprocess_trial(df_trial)
        if out is not None:
            results.append(out)

    print(f"Successfully preprocessed {len(results)} / {len(trial_ids)} trials.")
    return results
