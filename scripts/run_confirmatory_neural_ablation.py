"""Run neural ablations on the frozen strategy-window participant folds.

The deterministic conditional autoencoder isolates variational regularisation.
The unconditional VAE isolates task conditioning. Everything else matches the
primary CVAE protocol, including data, folds, architecture, losses and metrics.
"""
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
from scripts.run_confirmatory_cvae import DEFAULT_STUDY
from scripts.run_corrected_study import evaluate_per_trial_run, load_per_trial_checkpoint
from src.confirmatory_evaluation import prediction_tables
from src.confirmatory_protocol import PROTOCOL_VERSION, partition_trials, write_or_verify_manifest
from src.run_config import RunConfig
from src.train import train_vae
from src.trajectory_view import project_trials_to_table_plane, select_trials_window
from src.vae_model import NormStats


FAMILIES = {
    "conditional_ae": {"variational": False, "use_condition": True, "kl_weight": 0.0},
    "unconditional_vae": {"variational": True, "use_condition": False, "kl_weight": 1.0},
}


def collect_results(run_root: Path, family: str) -> pd.DataFrame:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_root.glob(f"fold*/{family}_z*_seed*/result.json"))
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["outer_fold", "latent_dim", "training_seed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--trials", type=Path, default=config.DATA_PROCESSED_DIR / "canonical_trials.pkl")
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--dims", nargs="+", type=int, default=[3, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-evaluate", action="store_true")
    args = parser.parse_args()

    with args.trials.open("rb") as handle:
        canonical = pickle.load(handle)
    trials = project_trials_to_table_plane(
        select_trials_window(canonical, config.WINDOW_GO_TO_ARRIVAL)
    )
    folds = write_or_verify_manifest(args.study / "protocol" / "participant_folds.json", trials)
    family_cfg = FAMILIES[args.family]
    requested_folds = args.folds[:1] if args.smoke else args.folds
    dims = args.dims[:1] if args.smoke else args.dims
    seeds = args.seeds[:1] if args.smoke else args.seeds
    epochs = min(args.epochs, 3) if args.smoke else args.epochs
    run_root = args.study / "runs" / args.family
    results_dir = args.study / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "model_family": args.family,
        "comparison_purpose": (
            "effect of variational regularisation"
            if args.family == "conditional_ae"
            else "effect of task conditioning"
        ),
        "matched_to_primary_cvae": True,
        "latent_dims": args.dims,
        "training_seeds": args.seeds,
        **family_cfg,
    }
    protocol_path = args.study / "protocol" / f"{args.family}_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    for fold_index in requested_folds:
        fold = folds[fold_index]
        train, validation, test = partition_trials(trials, fold)
        print(
            f"{args.family}, outer fold {fold_index}: {len(train)} train / "
            f"{len(validation)} validation / {len(test)} test trials",
            flush=True,
        )
        for latent_dim in dims:
            for training_seed in seeds:
                run_dir = run_root / f"fold{fold_index}" / f"{args.family}_z{latent_dim}_seed{training_seed}"
                result_path = run_dir / "result.json"
                if result_path.exists() and not args.force_evaluate:
                    print(f"Skip completed: {run_dir}", flush=True)
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "split.json").write_text(
                    json.dumps({**fold.to_dict(), "protocol_version": PROTOCOL_VERSION}, indent=2),
                    encoding="utf-8",
                )
                checkpoint_path = run_dir / "checkpoint.pt"
                if checkpoint_path.exists():
                    model, norm = load_per_trial_checkpoint(checkpoint_path, device)
                else:
                    cfg = RunConfig(
                        seed=training_seed,
                        latent_dim=latent_dim,
                        window_mode=config.WINDOW_GO_TO_ARRIVAL,
                        epochs=epochs,
                        patience=25,
                        encoder_uses_timing=False,
                        balance_subjects=True,
                        timing_weight=20.0,
                        variational=family_cfg["variational"],
                        use_condition=family_cfg["use_condition"],
                        kl_weight=family_cfg["kl_weight"],
                    )
                    model, _, _ = train_vae(train, validation, cfg, run_dir, device)
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    norm = NormStats.from_checkpoint(checkpoint)

                summary = evaluate_per_trial_run(
                    model, norm, train, validation, test, run_dir, config.CONTEXT_QUERY_SEED, device
                )
                timing_predictions, reconstruction_predictions, detailed = prediction_tables(
                    model, test, norm, device
                )
                timing_predictions.to_csv(run_dir / "timing_predictions.csv", index=False)
                reconstruction_predictions.to_csv(
                    run_dir / "reconstruction_predictions.csv", index=False
                )
                result = {
                    "protocol_version": PROTOCOL_VERSION,
                    "model_family": args.family,
                    "window_mode": config.WINDOW_GO_TO_ARRIVAL,
                    "outer_fold": fold_index,
                    "latent_dim": latent_dim,
                    "training_seed": training_seed,
                    "n_train_subjects": len(fold.train_subjects),
                    "n_validation_subjects": len(fold.validation_subjects),
                    "n_test_subjects": len(fold.test_subjects),
                    "n_train_trials": len(train),
                    "n_validation_trials": len(validation),
                    "n_test_trials": len(test),
                    "test_subjects": list(fold.test_subjects),
                    **summary,
                    **detailed,
                }
                result_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
                collect_results(run_root, args.family).to_csv(
                    results_dir / f"{args.family}_all_runs.csv", index=False
                )
                print(pd.DataFrame([result]).to_string(index=False), flush=True)

    frame = collect_results(run_root, args.family)
    frame.to_csv(results_dir / f"{args.family}_all_runs.csv", index=False)
    expected = len(requested_folds) * len(dims) * len(seeds)
    print(f"Completed rows available: {len(frame)}; requested in this invocation: {expected}")


if __name__ == "__main__":
    main()
