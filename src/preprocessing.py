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


def find_movement_onset(
    pos: np.ndarray,
    search_start_idx: int,
    arrival_idx: int,
    vel_threshold: float = config.ONSET_SPEED_THRESHOLD,
    fs: float = config.RECORDING_HZ,
) -> int:
    """
    First frame at/after the go-signal where finger speed exceeds *vel_threshold*.

    Only the movement ONSET is detected from speed. The movement END is the
    recorded arrival (``arrival_idx``, passed in from ``pressedTime``/CSV end) —
    never the last frame above threshold — so an isolated sensor-jitter blip late
    in a long recording can no longer stretch the window (the old bug that turned
    a 0.6 s reach into a 9.7 s "movement"). Falls back to the go-signal if the
    finger never crosses the threshold.
    """
    dt = 1.0 / fs
    vel = np.gradient(pos, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    region = speed[search_start_idx : arrival_idx + 1]
    above = region > vel_threshold
    sustain = max(int(config.ONSET_SUSTAIN_FRAMES), 1)
    if len(above) >= sustain:
        hits = np.convolve(above.astype(np.int8), np.ones(sustain, dtype=np.int8), mode="valid")
        candidates = np.where(hits == sustain)[0]
        if candidates.size:
            return int(search_start_idx + candidates[0])
    return int(search_start_idx)


def regularise_frame_grid(df_trial: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return position/marker samples on a contiguous frame-counter grid.

    The receive-time column is deliberately ignored. Duplicate frame counters
    are averaged, missing counters are linearly interpolated, and markers are
    retained with a maximum aggregation. This makes every index step exactly
    1/240 s while preserving the timing encoded by the device frame counter.
    """
    work = df_trial[["frame", "x", "y", "z", "marker"]].copy()
    work = work.apply(pd.to_numeric, errors="coerce").dropna(subset=["frame", "x", "y", "z"])
    if len(work) < 4:
        raise ValueError("fewer than four finite tracker samples")

    work["frame"] = np.rint(work["frame"]).astype(np.int64)
    grouped = work.groupby("frame", sort=True, as_index=True).agg(
        x=("x", "mean"), y=("y", "mean"), z=("z", "mean"), marker=("marker", "max")
    )
    observed = grouped.index.to_numpy(dtype=np.int64)
    if observed[-1] <= observed[0]:
        raise ValueError("non-increasing frame-counter span")

    grid = np.arange(observed[0], observed[-1] + 1, dtype=np.int64)
    pos = np.column_stack([
        np.interp(grid, observed, grouped[c].to_numpy(dtype=float)) for c in ("x", "y", "z")
    ])
    marker = np.zeros(len(grid), dtype=float)
    marker[observed - observed[0]] = grouped["marker"].fillna(0).to_numpy(dtype=float)
    quality = {
        "n_rows_observed": int(len(work)),
        "n_unique_frames": int(len(observed)),
        "n_duplicate_rows": int(len(work) - len(observed)),
        "n_interpolated_frames": int(len(grid) - len(observed)),
        "frame_start": int(grid[0]),
        "frame_end": int(grid[-1]),
    }
    return pos, marker, grid, quality


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
def _nan(v) -> bool:
    """True if *v* is None or a NaN float (missing .mat metadata)."""
    return v is None or bool(pd.isna(v))


def preprocess_trial(
    df_trial: pd.DataFrame,
    filter_cutoff: float = config.LOWPASS_CUTOFF_HZ,
) -> Optional[dict]:
    """
    Run the complete preprocessing pipeline on one trial DataFrame.

    Segmentation is event-based (see config): the go-signal (object starts
    moving) is the behavioural zero-time, and the window END is the recorded
    finger arrival. Invalid trials are dropped and reported by returning a small
    ``{"valid": False, "drop_reason": ...}`` dict instead of the processed one.

    A kept trial's dict has:
        - pos_raw / pos_filtered: (N, 3) raw and low-pass-filtered positions
        - pos_norm: (T, 3) movement (onset->arrival) resampled + origin-aligned
        - vel_norm / speed_norm: derivatives of pos_norm
        - stim_onset_idx: object appears (marker == 5)
        - go_signal_idx:  object starts moving (reaction/wait time is measured
                          from here, so the randomised foreperiod is removed)
        - move_start_idx: finger movement onset
        - move_end_idx:   finger arrival (interception)
        - metadata: trial-level metadata incl. responseText / successful
    """
    row0 = df_trial.iloc[0]
    resp = str(row0.get("responseText", "") or "")
    go_s = row0.get("go_signal_s", np.nan)
    arrival_s = row0.get("arrival_s", np.nan)
    afe = row0.get("arrival_window_end_s", np.nan)

    def drop(reason: str) -> dict:
        return {"valid": False, "drop_reason": reason,
                "trial_id": row0.get("trial_id", "")}

    # ── Outcome / timing filters (see config + jason_clarifications.md) ──
    if config.DROP_TOO_EARLY and resp == "Too early":
        return drop("too_early_label")
    if _nan(arrival_s):
        return drop("timeout_no_arrival")              # finger never intercepted
    if (not _nan(afe)) and (arrival_s - afe) > config.LATE_ARRIVAL_CUTOFF_S:
        return drop("too_late")                        # disengaged / skip-to-next
    if arrival_s > config.MAX_TRIAL_DURATION_S:
        return drop("timeout_long")                    # backstop when afe missing
    if (not config.KEEP_NOT_FIXATING) and resp.startswith("Not fixating"):
        return drop("not_fixating")

    try:
        pos_raw, markers, frame_values, frame_quality = regularise_frame_grid(df_trial)
    except ValueError as exc:
        return drop(f"invalid_frame_counter:{exc}")

    stim_idx = find_stimulus_onset(markers)            # object appears (marker==5)
    if stim_idx is None:
        return drop("no_stimulus_marker")

    pos_filtered = lowpass_filter(pos_raw, cutoff=filter_cutoff)
    n = len(pos_filtered)

    # Go-signal (object starts moving) in finger frames, synced to marker==5.
    go_idx = stim_idx if _nan(go_s) else stim_idx + int(round(go_s * config.RECORDING_HZ))
    go_idx = int(min(max(go_idx, 0), n - 1))

    # END = arrival: the recording already stops at interception, so the last
    # frame is the arrival (timeouts, which run to the 10 s cap, were dropped).
    arrival_idx = n - 1
    if arrival_idx - go_idx < 4:
        return drop("window_too_short")

    move_start = find_movement_onset(pos_filtered, go_idx, arrival_idx)
    movement = pos_filtered[move_start : arrival_idx + 1]
    go_to_arrival = pos_filtered[go_idx : arrival_idx + 1]
    if len(movement) < 4:
        return drop("movement_too_short")

    # Store both pre-specified representations in one canonical trial cache.
    # ``pos_norm`` remains the movement-only view for backward compatibility;
    # callers select the comparison view explicitly via trajectory_view.py.
    pos_movement_norm = normalise_spatial(normalise_temporal(movement))
    pos_go_to_arrival_norm = normalise_spatial(normalise_temporal(go_to_arrival))
    pos_norm = pos_movement_norm

    # Velocity on the normalised trajectory: mm per *normalised frame*, not mm/s —
    # its physical scale depends on the movement duration, which resampling
    # removed. move_start_idx / move_end_idx / go_signal_idx restore it; see
    # features.compute_trial_features and vae_model.encode_timing, the single
    # source of truth for the timing channels.
    vel_norm = np.gradient(pos_norm, axis=0)
    speed_norm = np.linalg.norm(vel_norm, axis=1)

    meta_cols = [
        "subject", "condition", "sp", "side", "rep",
        "starting_position_mm", "starting_side", "trial_id",
        "responseText", "successful", "go_signal_s", "arrival_s",
        "arrival_window_end_s", "target_speed_screen_s",
        "target_motion_onset_s", "stimulus_name",
    ]
    metadata = {c: df_trial.iloc[0][c] for c in meta_cols if c in df_trial.columns}

    return {
        "valid": True,
        "pos_raw": pos_raw,
        "pos_filtered": pos_filtered,
        "pos_norm": pos_norm,
        "pos_movement_norm": pos_movement_norm,
        "pos_go_to_arrival_norm": pos_go_to_arrival_norm,
        "vel_norm": vel_norm,
        "speed_norm": speed_norm,
        "stim_onset_idx": stim_idx,
        "go_signal_idx": go_idx,
        "move_start_idx": move_start,
        "move_end_idx": arrival_idx,
        "segment_start_idx": move_start,
        "window_mode": config.WINDOW_MOVEMENT_ONLY,
        "frame_values": frame_values,
        "frame_quality": frame_quality,
        "position_unit": config.POSITION_UNIT,
        "metadata": metadata,
    }


# ── Batch preprocessing ──────────────────────────────────────────────────────
def preprocess_dataset(dataset: pd.DataFrame, return_audit: bool = False):
    """
    Apply *preprocess_trial* to every trial and return the kept trials, printing
    a breakdown of why the rest were dropped (too_early / too_late / timeout / …).
    """
    from collections import Counter

    from tqdm import tqdm

    results = []
    reasons: Counter = Counter()
    trial_ids = dataset["trial_id"].unique()

    for tid in tqdm(trial_ids, desc="Preprocessing trials"):
        df_trial = dataset[dataset["trial_id"] == tid].sort_values("frame")
        out = preprocess_trial(df_trial)
        if out is None or not out.get("valid", False):
            reasons[out["drop_reason"] if out else "none"] += 1
            continue
        results.append(out)

    print(f"Kept {len(results)} / {len(trial_ids)} trials.")
    if reasons:
        print("Dropped:")
        for r, c in reasons.most_common():
            print(f"  {c:5d}  {r}")
    if return_audit:
        return results, dict(reasons)
    return results
