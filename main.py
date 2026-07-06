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
from src.baseline_kmeans import cluster_trajectories
from src.baseline_spline import evaluate_spline_baseline
from src.train import split_subjects, train_vae, sweep_latent_dims
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


def main():
    parser = argparse.ArgumentParser(description="Interception Movements DL Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw data directory")
    parser.add_argument("--phase", type=str, default="all", help="Phase to run: 1, 2, 3, eval, or all")
    parser.add_argument("--sweep", action="store_true", help="Run latent-dim sweep")
    parser.add_argument("--latent-dim", type=int, default=config.DEFAULT_LATENT_DIM, help="Latent dim for VAE")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    processed_path = config.DATA_PROCESSED_DIR / "trials.pkl"

    # Ensure output directories exist
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Split data
    train_trials, val_trials, test_trials = split_subjects(trials)

    # ── Phase 1: K-Means ──────────────────────────────────────────────────
    if args.phase in ("all", "1"):
        print("\n" + "=" * 60)
        print("PHASE 1: K-Means Clustering Baseline")
        print("=" * 60)

        print("\n--- Clustering on trajectories ---")
        km_traj = cluster_trajectories(trials, use_features=False)

        print("\n--- Clustering on features ---")
        km_feat = cluster_trajectories(trials, use_features=True, feature_df=feature_df)

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

        if args.sweep:
            sweep_results = sweep_latent_dims(
                train_trials, val_trials,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            )
        else:
            model, history = train_vae(
                train_trials, val_trials,
                latent_dim=args.latent_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )

    # ── Evaluation ────────────────────────────────────────────────────────
    if args.phase in ("all", "eval"):
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        # Load best model
        model_path = config.MODELS_DIR / f"cvae_z{args.latent_dim}_best.pt"
        if not model_path.exists():
            print(f"Model not found: {model_path}. Train first with --phase 3")
            return

        from src.vae_model import ConditionalVAE

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = ConditionalVAE(latent_dim=ckpt["latent_dim"]).to(device)
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
