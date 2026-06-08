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
            # Attach metadata
            df["subject"] = subject_id
            for k, v in meta.items():
                if k not in df.columns:
                    df[k] = v if not isinstance(v, tuple) else str(v)
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
