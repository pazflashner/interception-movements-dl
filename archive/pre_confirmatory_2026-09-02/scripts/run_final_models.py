"""Train the pre-specified core deep-learning models across three seeds."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_corrected_study import (
    evaluate_per_trial_run,
    load_per_trial_checkpoint,
)
from src.run_config import RunConfig
from src.train import split_subjects, train_vae
from src.trajectory_view import MODEL_AXES, project_trials_to_table_plane
from src.vae_model import NormStats
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=str(ROOT / "data" / "final_study" / "trials.pkl"))
    parser.add_argument("--out", default=str(ROOT / "results" / "final_study" / "core_models"))
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 3, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.trials, "rb") as handle:
        trials = pickle.load(handle)
    trials = project_trials_to_table_plane(trials)
    train_trials, val_trials, test_trials = split_subjects(trials, 17, 4, 7, 42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    dims = args.dims[:1] if args.smoke else args.dims
    seeds = args.seeds[:1] if args.smoke else args.seeds
    epochs = min(3, args.epochs) if args.smoke else args.epochs

    for n in dims:
        for seed in seeds:
            run = out / f"trajectory_only_z{n}_seed{seed}"
            if (run / "checkpoint.pt").exists():
                model, norm = load_per_trial_checkpoint(run / "checkpoint.pt", device)
            else:
                cfg = RunConfig(
                    seed=seed,
                    latent_dim=n,
                    epochs=epochs,
                    patience=25,
                    encoder_uses_timing=False,
                    balance_subjects=True,
                    timing_weight=20.0,
                )
                model, _, _ = train_vae(train_trials, val_trials, cfg, run, device)
                ckpt = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
                norm = NormStats.from_checkpoint(ckpt)
            summary = evaluate_per_trial_run(
                model, norm, train_trials, val_trials, test_trials,
                run, config.CONTEXT_QUERY_SEED, device,
            )
            rows.append({"latent_dim": n, "seed": seed, **summary})
            pd.DataFrame(rows).to_csv(out / "seed_results.csv", index=False)
            print(pd.DataFrame([rows[-1]]).to_string(index=False), flush=True)

    protocol = {
        "model": "trajectory-only conditional variational autoencoder",
        "subject_split_seed": 42,
        "train_val_test_subjects": [17, 4, 7],
        "latent_dims": args.dims,
        "initialization_seeds": args.seeds,
        "encoder_inputs": "phase-normalized trajectory and task condition; timing withheld",
        "trajectory_axes": list(MODEL_AXES),
        "trajectory_axis_names": ["x_lateral", "y_forward"],
        "task_condition": "sp category, starting side, and exact 60 Hz target speed",
        "decoder_outputs": "trajectory, log movement duration, log initiation time",
        "epochs": args.epochs,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
