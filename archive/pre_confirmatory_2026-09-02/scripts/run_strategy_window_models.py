"""Train and evaluate both temporal-window CVAEs on the same cohort."""
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

import config
from scripts.run_corrected_study import evaluate_per_trial_run, load_per_trial_checkpoint
from src.run_config import RunConfig
from src.train import split_subjects, train_vae
from src.trajectory_view import project_trials_to_table_plane, select_trials_window
from src.vae_model import NormStats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=Path,
        default=config.DATA_PROCESSED_DIR / "canonical_trials.pkl",
    )
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--windows", nargs="+", choices=config.WINDOW_MODES,
                        default=list(config.WINDOW_MODES))
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 3, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    with args.trials.open("rb") as handle:
        canonical = pickle.load(handle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    windows = args.windows[:1] if args.smoke else args.windows
    dims = args.dims[:1] if args.smoke else args.dims
    seeds = args.seeds[:1] if args.smoke else args.seeds
    epochs = min(args.epochs, 3) if args.smoke else args.epochs

    for window_mode in windows:
        trials = project_trials_to_table_plane(select_trials_window(canonical, window_mode))
        train, val, test = split_subjects(trials, 17, 4, 7, 42)
        model_root = args.out / window_mode / "models"
        model_root.mkdir(parents=True, exist_ok=True)
        for latent_dim in dims:
            for seed in seeds:
                run = model_root / f"cvae_{window_mode}_z{latent_dim}_seed{seed}"
                if (run / "checkpoint.pt").exists():
                    model, norm = load_per_trial_checkpoint(run / "checkpoint.pt", device)
                else:
                    cfg = RunConfig(
                        seed=seed,
                        latent_dim=latent_dim,
                        window_mode=window_mode,
                        epochs=epochs,
                        patience=25,
                        encoder_uses_timing=False,
                        balance_subjects=True,
                        timing_weight=20.0,
                    )
                    model, _, _ = train_vae(train, val, cfg, run, device)
                    checkpoint = torch.load(
                        run / "checkpoint.pt", map_location=device, weights_only=False
                    )
                    norm = NormStats.from_checkpoint(checkpoint)
                summary = evaluate_per_trial_run(
                    model, norm, train, val, test, run,
                    config.CONTEXT_QUERY_SEED, device,
                )
                row = {
                    "window_mode": window_mode,
                    "latent_dim": latent_dim,
                    "seed": seed,
                    **summary,
                }
                rows.append(row)
                combined = args.out / "model_seed_results.csv"
                old = pd.read_csv(combined) if combined.exists() else pd.DataFrame()
                frame = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
                frame = frame.drop_duplicates(
                    ["window_mode", "latent_dim", "seed"], keep="last"
                ).sort_values(["window_mode", "latent_dim", "seed"])
                frame.to_csv(combined, index=False)
                print(pd.DataFrame([row]).to_string(index=False), flush=True)

    protocol = {
        "windows": args.windows,
        "latent_dims": args.dims,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "subject_split": {"seed": 42, "train": 17, "validation": 4, "test": 7},
        "encoder_inputs": "selected x-y trajectory window and task condition; timing withheld",
        "decoder_outputs": "selected x-y trajectory window, movement time, initiation time",
        "primary_interpretable_dims": [2, 3],
        "capacity_comparator": 8,
    }
    (args.out / "model_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
