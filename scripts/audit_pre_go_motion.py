"""Audit cached pre-go evidence without changing labels, exclusions, or inputs."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

OUT = ROOT / "studies" / "review_corrected_evaluation" / "results" / "event_audit"


def preceding_run(above: np.ndarray, go: int) -> int:
    count = 0
    for value in above[:go][::-1]:
        if not value:
            break
        count += 1
    return count


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (config.DATA_PROCESSED_DIR / "canonical_trials.pkl").open("rb") as handle:
        trials = pickle.load(handle)
    rows = []
    for trial in trials:
        go = trial["go_signal_idx"]
        end = trial["move_end_idx"]
        meta = trial["metadata"]
        row = {"subject": meta["subject"], "trial_id": meta["trial_id"],
               "responseText": meta["responseText"],
               "go_delay_s": (go - trial["stim_onset_idx"]) / config.RECORDING_HZ,
               "initiation_s": (trial["move_start_idx"] - go) / config.RECORDING_HZ,
               "csv_minus_mat_arrival_s": (end - trial["stim_onset_idx"]) / config.RECORDING_HZ - meta["arrival_s"]}
        for mode, key, dims in (("filtered3d", "pos_filtered", 3),
                                ("filtered2d", "pos_filtered", 2), ("raw2d", "pos_raw", 2)):
            pos = trial[key][:, :dims]
            speed = np.linalg.norm(np.gradient(pos, 1 / config.RECORDING_HZ, axis=0), axis=1)
            above = speed > config.ONSET_SPEED_THRESHOLD
            row[f"{mode}_speed_at_go"] = speed[go]
            row[f"{mode}_above_go_sustained"] = bool(above[go:go + config.ONSET_SUSTAIN_FRAMES].all())
            row[f"{mode}_continuous_pre_go_s"] = preceding_run(above, go) / config.RECORDING_HZ
            start = max(trial["stim_onset_idx"], go - round(0.1 * config.RECORDING_HZ))
            row[f"{mode}_last100ms_displacement"] = float(np.linalg.norm(pos[go] - pos[start]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pre_go_trial_audit.csv", index=False)
    zero = frame.initiation_s == 0
    report = {
        "n_trials": len(frame), "zero_onset_trials": int(zero.sum()),
        "zero_onset_sustained_above_go_3d": int((zero & frame.filtered3d_above_go_sustained).sum()),
        "zero_onset_sustained_above_go_2d": int((zero & frame.filtered2d_above_go_sustained).sum()),
        "zero_onset_pre_go_at_least_50ms_2d": int((zero & (frame.filtered2d_continuous_pre_go_s >= .05)).sum()),
        "zero_onset_pre_go_at_least_50ms_raw2d": int((zero & (frame.raw2d_continuous_pre_go_s >= .05)).sum()),
        "go_delay_s_min_median_max": [float(frame.go_delay_s.min()), float(frame.go_delay_s.median()), float(frame.go_delay_s.max())],
        "csv_minus_mat_arrival_mean_s": float(frame.csv_minus_mat_arrival_s.mean()),
        "csv_minus_mat_arrival_sd_s": float(frame.csv_minus_mat_arrival_s.std()),
        "zero_onset_by_outcome": frame.loc[zero, "responseText"].value_counts().to_dict(),
        "decision": "No trials removed or relabelled. Threshold evidence is not a diagnosis of anticipation; zero-phase filtering can smear motion backward in time.",
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Deterministic examples: closest to the median pre-go duration in each group.
    groups = [("Zero onset, sustained pre-go", zero & (frame.filtered2d_continuous_pre_go_s >= .05)),
              ("Zero onset, shorter pre-go evidence", zero & (frame.filtered2d_continuous_pre_go_s < .05)),
              ("Positive initiation", ~zero)]
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for ax, (label, mask) in zip(axes, groups):
        group = frame[mask]
        if group.empty:
            continue
        idx = (group.filtered2d_continuous_pre_go_s - group.filtered2d_continuous_pre_go_s.median()).abs().idxmin()
        trial = trials[idx]
        go = trial["go_signal_idx"]
        time = (np.arange(len(trial["pos_raw"])) - go) / config.RECORDING_HZ
        for key, name, color in (("pos_raw", "Raw planar speed", "0.65"), ("pos_filtered", "Filtered planar speed", "0.1")):
            speed = np.linalg.norm(np.gradient(trial[key][:, :2], 1 / config.RECORDING_HZ, axis=0), axis=1)
            ax.plot(time, speed, label=name, color=color, linewidth=1)
        ax.axvline(0, color="0.3", linestyle="--")
        ax.axhline(config.ONSET_SPEED_THRESHOLD, color="0.4", linestyle=":")
        ax.set_title(f"{label}: {trial['metadata']['trial_id']}", fontsize=9)
        ax.set_ylabel("Speed (tracker units/s)", fontsize=8)
        ax.set_xlim(-.2, .25)
        ax.set_ylim(0, min(150, ax.get_ylim()[1]))
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Time relative to target motion (s)")
    fig.tight_layout()
    fig.savefig(OUT / "pre_go_examples.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
