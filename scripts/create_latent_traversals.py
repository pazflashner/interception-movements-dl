"""Decode controlled n=2/3 latent traversals for interpretation figures."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_corrected_study import load_per_trial_checkpoint
from src.evaluate import encode_trials
from src.features import features_from_arrays
from src.submovements import SubmovementConfig, decompose_normalized_trajectory
from src.train import split_subjects
from src.trajectory_view import project_trials_to_table_plane
from src.vae_model import encode_trial_condition


def decode_points(model, norm, points, condition, device):
    tm, ts, _, _ = norm.torch(device)
    cond = np.repeat(condition[None, :], len(points), axis=0).astype(np.float32)
    with torch.no_grad():
        traj, timing = model.decode(
            torch.as_tensor(points, dtype=torch.float32, device=device),
            torch.as_tensor(cond, dtype=torch.float32, device=device),
        )
        channels = model.input_dim // 100
        trajectories = ((traj * ts + tm).cpu().numpy()).reshape(len(points), 100, channels)
        timing = norm.denormalise_timing(timing.cpu().numpy())
    return trajectories, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=str(ROOT / "data" / "final_study" / "trials.pkl"))
    parser.add_argument("--models", default=str(ROOT / "results" / "final_study" / "core_models"))
    parser.add_argument("--out", default=str(ROOT / "results" / "final_study" / "latent_traversals"))
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(args.trials, "rb") as handle:
        trials = pickle.load(handle)
    trials = project_trials_to_table_plane(trials)
    train, _, _ = split_subjects(trials, 17, 4, 7, 42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SubmovementConfig(restarts=2, max_nfev=500)

    for n in (2, 3):
        run = Path(args.models) / f"trajectory_only_z{n}_seed42"
        model, norm = load_per_trial_checkpoint(run / "checkpoint.pt", device)
        mu, _, _, _ = encode_trials(model, train, norm, device)
        centre, scale = mu.mean(axis=0), mu.std(axis=0)
        levels = np.array([-2, -1, 0, 1, 2], dtype=float)
        points, labels = [], []
        if n == 2:
            for first in levels:
                for second in levels:
                    points.append(centre + scale * np.array([first, second]))
                    labels.append({"z1_level": first, "z2_level": second})
        else:
            for dim in range(n):
                for level in levels:
                    point = centre.copy(); point[dim] += scale[dim] * level
                    points.append(point)
                    labels.append({"varied_dimension": dim + 1, "level": level})
        points = np.asarray(points, dtype=np.float32)
        trajectories, timing = decode_points(
            model, norm, points,
            encode_trial_condition({"sp": 2, "side": 1, "target_speed_screen_s": 0.635}, model.condition_dim),
            device,
        )
        rows = []
        for i, label in enumerate(labels):
            move_time = max(float(timing[i, 0]), 1e-3)
            init_time = max(float(timing[i, 1]), 0.0)
            basic = features_from_arrays(trajectories[i], move_time, init_time)
            sub = decompose_normalized_trajectory(trajectories[i], move_time, cfg, f"traversal-z{n}-{i}").summary()
            rows.append({"index": i, **label, **{f"z{j+1}": points[i, j] for j in range(n)}, **basic, **sub})
        pd.DataFrame(rows).to_csv(out / f"latent_z{n}.csv", index=False)
        np.savez_compressed(out / f"latent_z{n}_trajectories.npz", trajectories=trajectories, timing=timing, points=points)
        print(f"saved traversal n={n}")


if __name__ == "__main__":
    main()
