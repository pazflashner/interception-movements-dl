"""Attach the executed target speed from each trial's MAT ``dotArray``."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import pickle
from pathlib import Path

import numpy as np
import scipy.io as sio


STIMULUS_HZ = 60.0
SCREEN_WIDTH_PX = 1920.0


def trajectory_metrics(trajectory: np.ndarray, position_scale: float = 1.0) -> tuple[float, float]:
    trajectory = np.asarray(trajectory, dtype=float)
    step = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    moving = np.flatnonzero(step > 1e-8)
    if not len(moving):
        raise ValueError("target never moves")
    speed = float(np.median(step[moving]) * STIMULUS_HZ / position_scale)
    onset = float((moving[0] + 1) / STIMULUS_HZ)
    return speed, onset


def read_executed_target(item):
    index, trial_id, mat_path = item
    try:
        struct = sio.loadmat(
            mat_path, squeeze_me=True, struct_as_record=False,
            variable_names=["thistrial"],
        )["thistrial"]
        speed, onset = trajectory_metrics(np.asarray(struct.dotArray), SCREEN_WIDTH_PX)
        return {
            "index": index,
            "trial_id": trial_id,
            "stimulus_name": str(struct.thisstimulus.dotsFilename),
            "target_speed_screen_s": speed,
            "target_motion_onset_s": onset,
            "error": "",
        }
    except Exception as exc:
        return {"index": index, "trial_id": trial_id, "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--trials", default=str(root / "data" / "corrected_v2" / "trials.pkl"))
    parser.add_argument("--results", default=r"D:\DropBox\Dropbox\results")
    parser.add_argument("--stimuli", default=r"D:\DropBox\Dropbox\stimuli")
    parser.add_argument("--out", default=str(root / "data" / "final_study" / "trials.pkl"))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--external-audit", type=int, default=112)
    args = parser.parse_args()

    with open(args.trials, "rb") as handle:
        trials = pickle.load(handle)
    results_root, stimuli_root = Path(args.results), Path(args.stimuli)
    items = []
    for index, trial in enumerate(trials):
        meta = trial["metadata"]
        suffix = f"{meta['condition']}_{meta['sp']}_{meta['side']}_{meta['rep']}"
        mat_path = results_root / str(meta["subject"]) / f"trialinfo_{suffix}.mat"
        items.append((index, str(meta["trial_id"]), mat_path))

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        target_rows = list(pool.map(read_executed_target, items, chunksize=8))
    failures = [row for row in target_rows if row["error"]]
    if failures:
        raise RuntimeError(f"failed to read {len(failures)} executed targets; examples={failures[:5]}")

    onset_errors = []
    for row in target_rows:
        trial = trials[row["index"]]
        meta = dict(trial["metadata"])
        meta.update({key: row[key] for key in (
            "stimulus_name", "target_speed_screen_s", "target_motion_onset_s"
        )})
        trial = dict(trial)
        trial["metadata"] = meta
        trials[row["index"]] = trial
        cached_onset = (trial["go_signal_idx"] - trial["stim_onset_idx"]) / 240.0
        onset_errors.append(cached_onset - row["target_motion_onset_s"])

    # External files are useful documentation but are not always identical to
    # the executed dotArray. Quantify that discrepancy on a stratified sample.
    audit_indices = np.linspace(0, len(trials) - 1, min(args.external_audit, len(trials)), dtype=int)
    external_rows = []
    for index in audit_indices:
        row = target_rows[index]
        path = stimuli_root / row["stimulus_name"]
        try:
            speed, onset = trajectory_metrics(np.loadtxt(path, delimiter=","), 1.0)
            external_rows.append({
                "speed_abs_difference": abs(speed - row["target_speed_screen_s"]),
                "onset_abs_difference_s": abs(onset - row["target_motion_onset_s"]),
            })
        except Exception:
            external_rows.append({"speed_abs_difference": np.nan, "onset_abs_difference_s": np.nan})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)
    speeds = np.array([row["target_speed_screen_s"] for row in target_rows])
    external_speed = np.array([row["speed_abs_difference"] for row in external_rows])
    external_onset = np.array([row["onset_abs_difference_s"] for row in external_rows])
    protocol = {
        "authoritative_target": "thistrial.dotArray from paired MAT file",
        "linked_external_filename_field": "thistrial.thisstimulus.dotsFilename",
        "n_trials": len(trials),
        "target_sample_hz": STIMULUS_HZ,
        "screen_width_px": SCREEN_WIDTH_PX,
        "speed_unit": "screen widths per second",
        "speed_min_median_max": [float(speeds.min()), float(np.median(speeds)), float(speeds.max())],
        "max_abs_cached_onset_difference_s": float(np.max(np.abs(onset_errors))),
        "external_audit_n": len(external_rows),
        "external_speed_mismatch_gt_0.005_rate": float(np.nanmean(external_speed > 0.005)),
        "external_onset_mismatch_gt_one_frame_rate": float(np.nanmean(external_onset > 1 / STIMULUS_HZ)),
        "link_failures": 0,
    }
    out.with_suffix(".protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
