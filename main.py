"""
Main pipeline: Interception Movements Deep Learning

Runs the full pipeline end-to-end:
  1. Load raw data
  2. Preprocess (filter, segment, normalise)
  3. Extract kinematic features
  4. Phase 1: K-Means clustering baseline
  5. Phase 2: Polynomial spline baseline
  6. Phase 3: Train Conditional VAE
  7. Evaluate on held-out test subjects

Usage
-----
    python main.py                        # Full pipeline
    python main.py --phase 1              # Only K-Means baseline
    python main.py --phase 2              # Only spline baseline
    python main.py --phase 3              # Only VAE training
    python main.py --phase eval           # Only evaluation (requires trained model)
    python main.py --sweep                # Latent-dim hyperparameter sweep
    python main.py --data-dir path/to/raw # Custom data directory

Reproducibility
---------------
Each training run is fully described by a ``RunConfig`` and gets its own
directory under ``results/runs/``, containing ``config.yaml``, ``checkpoint.pt``
(with the config embedded) and ``history.json``.

    python main.py --phase 3 --seed 7           # one seeded run
    python main.py --phase 3 --seeds 0 1 2 3 4  # repeat across seeds
    python main.py --phase 3 --config results/runs/z3_seed7_.../config.yaml
    python main.py --summarise                  # table of all runs so far
    python main.py --phase eval --run results/runs/z3_seed7_...

With only 28 subjects, results are noisy: prefer the mean ± sd across several
seeds over any single run.

Data Setup
----------
Download data from Dropbox and place subject folders under data/raw/:
    data/raw/subject01/li_2_1_1_1.csv
    data/raw/subject01/li_2_2_2_3.csv
    ...
    data/raw/subject02/...
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

import config
from src.data_loading import load_dataset
from src.preprocessing import preprocess_dataset
from src.features import extract_features_dataframe
from src.baseline_kmeans import cluster_trajectories, sweep_seeds
from src.baseline_spline import evaluate_spline_baseline
from src.run_config import RunConfig, find_runs, set_seed, summarise_runs
from src.train import (
    split_subjects_for,
    train_across_seeds,
    train_vae,
    sweep_latent_dims,
)
from src.evaluate import run_full_evaluation

import torch


def save_processed(trials: list[dict], path: Path):
    """Persist preprocessed trials to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(trials, f)
    print(f"Saved {len(trials)} preprocessed trials to {path}")


def load_processed(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def print_run_summary():
    """Print every run's config axes and best val loss; save as a CSV."""
    import pandas as pd

    rows = summarise_runs()
    if not rows:
        print(f"No runs found in {config.RUNS_DIR}.")
        return

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "runs_summary.csv"
    df.to_csv(out, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {out}")

    # 28 subjects → report spread, not a single run.
    grouped = df.groupby("latent_dim")["best_val_loss"]
    print("\n--- Best val loss by latent dim (across seeds) ---")
    for latent_dim, vals in grouped:
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"  z={latent_dim}: {vals.mean():.6f} +/- {sd:.6f}  (n={len(vals)} runs)")


def main():
    parser = argparse.ArgumentParser(description="Interception Movements DL Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw data directory")
    parser.add_argument("--phase", type=str, default="all", help="Phase to run: 1, 2, 3, eval, or all")
    parser.add_argument("--sweep", action="store_true", help="Run latent-dim sweep")
    parser.add_argument("--latent-dim", type=int, default=config.DEFAULT_LATENT_DIM, help="Latent dim for VAE")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--kl-weight", type=float, default=config.KL_WEIGHT)
    # Reproducibility / repeated runs
    parser.add_argument("--seed", type=int, default=config.SEED, help="Seed for this run")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Repeat training across these seeds, one run directory each",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Load run settings from an existing config.yaml (CLI flags are ignored)",
    )
    parser.add_argument("--run", type=str, default=None, help="Run directory to evaluate")
    parser.add_argument("--summarise", action="store_true", help="Print a table of all runs and exit")
    args = parser.parse_args()

    # ── Run configuration ─────────────────────────────────────────────────
    if args.config:
        cfg = RunConfig.load(Path(args.config))
        print(f"Loaded run config from {args.config}")
    else:
        cfg = RunConfig(
            seed=args.seed,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            kl_weight=args.kl_weight,
        )

    processed_path = config.DATA_PROCESSED_DIR / "trials.pkl"

    # Ensure output directories exist
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.summarise:
        print_run_summary()
        return

    # Seed everything before any data-dependent randomness (baselines,
    # evaluation sampling). train_vae re-seeds per run.
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    # ── Load & preprocess ─────────────────────────────────────────────────
    if processed_path.exists():
        print(f"Loading preprocessed data from {processed_path}")
        trials = load_processed(processed_path)
    else:
        print("Loading raw data...")
        raw_dir = Path(args.data_dir) if args.data_dir else config.DATA_RAW_DIR
        dataset = load_dataset(raw_dir=raw_dir)
        print("Preprocessing...")
        trials = preprocess_dataset(dataset)
        save_processed(trials, processed_path)

    # Extract features
    print("Extracting kinematic features...")
    feature_df = extract_features_dataframe(trials)
    feature_df.to_csv(config.RESULTS_DIR / "features.csv", index=False)
    print(f"Features saved: {len(feature_df)} trials, {len(feature_df.columns)} features")

    # Split data (subject split is seeded from the run config)
    train_trials, val_trials, test_trials = split_subjects_for(trials, cfg)

    # ── Phase 1: K-Means ──────────────────────────────────────────────────
    if args.phase in ("all", "1"):
        print("\n" + "=" * 60)
        print("PHASE 1: K-Means Clustering Baseline")
        print("=" * 60)

        if args.seeds:
            # Repeat the baseline across seeds; results land in one tidy CSV.
            km_sweep = sweep_seeds(
                trials, args.seeds, feature_df=feature_df,
                out_csv=config.RESULTS_DIR / "kmeans_seed_sweep.csv",
            )
        else:
            print("\n--- Clustering on trajectories ---")
            km_traj = cluster_trajectories(trials, use_features=False, seed=cfg.seed)

            print("\n--- Clustering on features ---")
            km_feat = cluster_trajectories(
                trials, use_features=True, feature_df=feature_df, seed=cfg.seed
            )

    # ── Phase 2: Spline baseline ──────────────────────────────────────────
    if args.phase in ("all", "2"):
        print("\n" + "=" * 60)
        print("PHASE 2: Polynomial Spline Baseline")
        print("=" * 60)
        spline_result = evaluate_spline_baseline(trials)

    # ── Phase 3: VAE training ─────────────────────────────────────────────
    if args.phase in ("all", "3"):
        print("\n" + "=" * 60)
        print("PHASE 3: Conditional VAE Training")
        print("=" * 60)

        if args.seeds:
            # Repeated runs: each seed re-splits subjects and retrains.
            seed_results = train_across_seeds(trials, args.seeds, cfg=cfg)
        elif args.sweep:
            sweep_results = sweep_latent_dims(train_trials, val_trials, cfg=cfg)
        else:
            model, history, run_dir = train_vae(train_trials, val_trials, cfg=cfg)

    # ── Evaluation ────────────────────────────────────────────────────────
    if args.phase in ("all", "eval"):
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        # Resolve the run to evaluate: explicit --run, else the newest run.
        if args.run:
            run_dir = Path(args.run)
        else:
            runs = find_runs()
            if not runs:
                print(f"No runs found in {config.RUNS_DIR}. Train first with --phase 3")
                return
            run_dir = runs[-1]
            print(f"No --run given; using most recent: {run_dir.name}")

        model_path = run_dir / "checkpoint.pt"
        if not model_path.exists():
            print(f"Checkpoint not found: {model_path}. Train first with --phase 3")
            return

        from src.vae_model import ConditionalVAE

        # The run's own config governs evaluation, including the seed that
        # determines which subjects are held out.
        run_cfg = RunConfig.load(run_dir)
        print(f"Evaluating {run_dir.name} (seed={run_cfg.seed}, z={run_cfg.latent_dim})")
        set_seed(run_cfg.seed, deterministic=run_cfg.deterministic)
        _, _, test_trials = split_subjects_for(trials, run_cfg)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = ConditionalVAE(
            latent_dim=ckpt["latent_dim"], hidden_dim=run_cfg.hidden_dim
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        train_mean = np.array(ckpt["train_mean"])
        train_std = np.array(ckpt["train_std"])

        # Spline baseline for comparison
        spline_result = evaluate_spline_baseline(test_trials)
        results = run_full_evaluation(
            model, test_trials,
            train_mean, train_std,
            spline_mse=spline_result["mean_mse"],
            device=device,
        )

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
