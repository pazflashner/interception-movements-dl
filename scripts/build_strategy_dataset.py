"""Build one canonical cohort containing both temporal representations."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.data_loading import load_dataset
from src.features import compute_trial_features
from src.preprocessing import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--out", type=Path, default=config.DATA_PROCESSED_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "canonical_trials.pkl"
    if cache.exists() and not args.force:
        with cache.open("rb") as handle:
            trials = pickle.load(handle)
        drop_reasons = {}
    else:
        raw = load_dataset(args.raw, condition=config.CONDITION_FREE_EYE)
        trials, drop_reasons = preprocess_dataset(raw, return_audit=True)
        with cache.open("wb") as handle:
            pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)

    required = {"pos_movement_norm", "pos_go_to_arrival_norm"}
    missing = [t["metadata"]["trial_id"] for t in trials if not required.issubset(t)]
    if missing:
        raise RuntimeError(f"{len(missing)} trials lack comparison windows; rebuild with --force")

    rows = []
    for trial in trials:
        row = compute_trial_features(trial)
        row.update({
            "target_speed_screen_s": trial["metadata"].get("target_speed_screen_s"),
            "go_to_arrival_time_s": (
                trial["move_end_idx"] - trial["go_signal_idx"]
            ) / config.RECORDING_HZ,
            "pre_movement_wait_s": (
                trial["move_start_idx"] - trial["go_signal_idx"]
            ) / config.RECORDING_HZ,
        })
        rows.append(row)
    metadata = pd.DataFrame(rows)
    metadata.to_csv(args.out / "canonical_trial_metadata.csv", index=False)

    label_counts = metadata.groupby(["responseText", "successful"], dropna=False).size()
    protocol = {
        "condition": config.CONDITION_FREE_EYE,
        "n_trials": len(trials),
        "n_subjects": int(metadata.subject.nunique()),
        "window_modes": list(config.WINDOW_MODES),
        "normalised_length": config.NORMALISED_LENGTH,
        "filter_hz": config.LOWPASS_CUTOFF_HZ,
        "filter_order": config.LOWPASS_ORDER,
        "position_axes_cached": ["x", "y", "z"],
        "model_axes": ["x", "y"],
        "completion_rule": "non-empty MAT pressedTime/arrival",
        "too_early_retained": True,
        "late_arrival_cutoff_s_after_window": config.LATE_ARRIVAL_CUTOFF_S,
        "drop_reasons": drop_reasons,
        "label_counts": {
            f"{label}|{code}": int(count) for (label, code), count in label_counts.items()
        },
    }
    (args.out / "canonical_dataset_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
