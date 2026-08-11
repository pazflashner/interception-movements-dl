"""
Data loading utilities.

Reads raw CSV trial files from the Dropbox-downloaded folder structure,
parses filenames for trial metadata, and filters for condition 2
(free eye movements).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Filename parser ───────────────────────────────────────────────────────────
_FILENAME_RE = re.compile(
    r"^li_(?P<condition>\d+)_(?P<sp>\d+)_(?P<side>\d+)_(?P<rep>\d+)\.csv$"
)


def parse_filename(fname: str) -> Optional[dict]:
    """Extract trial metadata from a filename like ``li_2_2_1_14.csv``."""
    m = _FILENAME_RE.match(fname)
    if m is None:
        return None
    d = {k: int(v) for k, v in m.groupdict().items()}
    d["starting_position_mm"] = config.STARTING_POSITIONS.get(d["sp"])
    d["speed_range"] = config.SPEED_RANGES.get(d["sp"])
    d["starting_side"] = "left" if d["side"] == 1 else "right"
    return d


# ── Single-file loader ───────────────────────────────────────────────────────
def load_trial_csv(path: Path) -> pd.DataFrame:
    """Load a single trial CSV (9 columns, no header)."""
    df = pd.read_csv(path, header=None, names=config.CSV_COLUMNS)
    return df


# ── Trial metadata from the paired trialinfo_*.mat ────────────────────────────
def _mat_scalar(obj, field: str) -> Optional[float]:
    """First element of a struct field as float, or None if absent/empty."""
    try:
        v = np.array(getattr(obj, field)).flatten()
        return float(v[0]) if v.size else None
    except Exception:
        return None


def object_motion_onset_s(
    dot_array: np.ndarray, stimulus_hz: float = config.STIMULUS_HZ
) -> Optional[float]:
    """
    Seconds from object appearance to the object *starting to move* (the
    go-signal), from the object trajectory (``dotArray``). The object holds still
    for a randomised foreperiod, then moves; we return first-moving-frame / rate.
    """
    if dot_array is None or dot_array.ndim != 2 or len(dot_array) < 2:
        return None
    step = np.linalg.norm(np.diff(dot_array, axis=0), axis=1)
    moving = np.where(step > 1e-6)[0]
    # diff[i] compares samples i and i+1; movement is first present at i+1.
    return float((moving[0] + 1) / stimulus_hz) if moving.size else None


def object_motion_speed_screen_s(
    dot_array: np.ndarray,
    stimulus_hz: float = config.STIMULUS_HZ,
    screen_width_px: float = 1920.0,
) -> Optional[float]:
    """Median executed target speed in screen widths per second."""
    if dot_array is None or dot_array.ndim != 2 or len(dot_array) < 2:
        return None
    step = np.linalg.norm(np.diff(dot_array, axis=0), axis=1)
    moving = step[step > 1e-6]
    return float(np.median(moving) * stimulus_hz / screen_width_px) if moving.size else None


def load_trial_metadata(mat_path: Path) -> dict:
    """
    Per-trial fields we need from ``trialinfo_*.mat`` (the CSV alone has none of
    these). Missing / empty fields come back as None:

        responseText / successful  – task outcome label
        go_signal_s                – object appearance -> object starts moving
        arrival_s                  – object appearance -> finger interception (pressedTime)
        arrival_window_end_s       – when the object leaves the centre (success window closes)
    """
    try:
        m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["thistrial"]
    except Exception:
        return {}
    resp = getattr(m, "responseText", "")
    successful = _mat_scalar(m, "successful")
    start_t = _mat_scalar(m, "starttime")
    pressed_t = _mat_scalar(m, "pressedTime")
    arrival_s = (pressed_t - start_t) if (pressed_t is not None and start_t is not None) else None
    tr = getattr(m, "thisresponse", None)
    afe = _mat_scalar(tr, "arrivalFeedbackEnd") if tr is not None else None
    dot_array = np.array(getattr(m, "dotArray", []))
    go_s = object_motion_onset_s(dot_array)
    target_speed = object_motion_speed_screen_s(dot_array)
    stimulus = getattr(m, "thisstimulus", None)
    stimulus_name = str(getattr(stimulus, "dotsFilename", "")) if stimulus is not None else ""
    return {
        "responseText": str(resp) if resp is not None else "",
        "successful": int(successful) if successful is not None else None,
        "go_signal_s": go_s,
        "arrival_s": arrival_s,
        "arrival_window_end_s": afe,
        "target_speed_screen_s": target_speed,
        "target_motion_onset_s": go_s,
        "stimulus_name": stimulus_name,
    }


# ── Dataset loader ────────────────────────────────────────────────────────────
def load_dataset(
    raw_dir: Optional[Path] = None,
    condition: int = config.CONDITION_FREE_EYE,
) -> pd.DataFrame:
    """
    Walk *raw_dir* (one sub-directory per subject), load all trials for
    the requested condition, and return a single DataFrame with metadata
    columns attached.

    Returns
    -------
    pd.DataFrame
        Columns: subject, condition, sp, side, rep, starting_position_mm,
        speed_range, starting_side, frame, x, y, z, time, marker, trial_id
    """
    raw_dir = Path(raw_dir or config.DATA_RAW_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Download data from Dropbox and place subject folders under data/raw/"
        )

    records: list[pd.DataFrame] = []
    subject_dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    for subj_dir in tqdm(subject_dirs, desc="Loading subjects"):
        subject_id = subj_dir.name
        csv_files = sorted(subj_dir.glob("li_*.csv"))

        for csv_path in csv_files:
            meta = parse_filename(csv_path.name)
            if meta is None:
                continue
            if meta["condition"] != condition:
                continue

            df = load_trial_csv(csv_path)
            # Drop rotation columns
            df = df.drop(columns=["rot1", "rot2", "rot3"])
            # Attach filename metadata
            df["subject"] = subject_id
            for k, v in meta.items():
                if k not in df.columns:
                    df[k] = v if not isinstance(v, tuple) else str(v)
            # Join the paired .mat: outcome label + event timings (go-signal,
            # arrival, success-window). "li_2_1_1_1" -> "trialinfo_2_1_1_1.mat".
            mat_path = csv_path.parent / f"trialinfo_{csv_path.stem[len('li_'):]}.mat"
            tmeta = load_trial_metadata(mat_path) if mat_path.exists() else {}
            for k, v in tmeta.items():
                df[k] = v if v is not None else np.nan
            df["trial_id"] = f"{subject_id}_{csv_path.stem}"
            records.append(df)

    if not records:
        raise ValueError("No trial files found – check raw_dir and condition filter.")

    dataset = pd.concat(records, ignore_index=True)
    print(
        f"Loaded {dataset['trial_id'].nunique()} trials from "
        f"{dataset['subject'].nunique()} subjects (condition={condition})."
    )
    return dataset


# ── Stimulus loader ───────────────────────────────────────────────────────────
def load_stimulus_trajectory(path: Path) -> np.ndarray:
    """Load a stimulus/target trajectory file (assumed 60 Hz)."""
    return np.loadtxt(path, delimiter=",")


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(f"\nSubjects: {sorted(df['subject'].unique())}")
    print(f"Trials per subject:\n{df.groupby('subject')['trial_id'].nunique()}")
