"""Minimum-jerk fidelity on every held-out participant fold and neural seed."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.generate_final_samples import evaluate_run, fit_generated, generate_run
from scripts.run_confirmatory_cvae import DEFAULT_STUDY
from scripts.run_corrected_study import (
    load_per_trial_checkpoint,
    training_latent_noise_covariance,
)
from src.confirmatory_protocol import partition_trials, write_or_verify_manifest
from src.features import compute_trial_features
from src.submovements import SubmovementConfig
from src.trajectory_view import project_trials_to_table_plane, select_trials_window


def _checkpoint(study: Path, fold: int, latent_dim: int, seed: int) -> Path:
    return (
        study
        / "runs"
        / "cvae"
        / f"fold{fold}"
        / f"cvae_z{latent_dim}_seed{seed}"
        / "checkpoint.pt"
    )


def _empirical_table(path: Path, trials: list[dict]) -> pd.DataFrame:
    empirical = pd.read_csv(path)
    empirical = empirical[empirical.mj_fit_success == True].copy()
    basic = pd.DataFrame([compute_trial_features(trial) for trial in trials])
    columns = [
        "trial_id",
        "movement_time_s",
        "initiation_time_s",
        "peak_speed_tracker_units_s",
        "path_length",
        "curvature_index",
        "max_lateral_deviation",
    ]
    return empirical.merge(basic[columns], on="trial_id", how="inner", validate="one_to_one")


def _success_mask(frame: pd.DataFrame) -> pd.Series:
    values = frame.mj_fit_success
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, default=config.DATA_PROCESSED_DIR / "canonical_trials.pkl")
    parser.add_argument(
        "--empirical",
        type=Path,
        default=ROOT / "studies" / "strategy_window_comparison" / "results" / "submovements_real.csv",
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--dims", nargs="+", type=int, default=[3, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--samples-per-subject", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--backend", choices=("thread", "process"), default="thread")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with args.trials.open("rb") as handle:
        canonical = pickle.load(handle)
    all_trials = project_trials_to_table_plane(
        select_trials_window(canonical, config.WINDOW_GO_TO_ARRIVAL)
    )
    folds = write_or_verify_manifest(args.study / "protocol" / "participant_folds.json", all_trials)
    empirical = _empirical_table(args.empirical, all_trials)
    out = args.study / "results" / "minimum_jerk"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit_cfg = SubmovementConfig(restarts=1, max_nfev=300)
    summaries = []

    for fold_index in args.folds:
        train, _, test = partition_trials(all_trials, folds[fold_index])
        for latent_dim in args.dims:
            for seed in args.seeds:
                run_name = f"cvae_z{latent_dim}_fold{fold_index}_seed{seed}"
                run_dir = out / run_name
                generated_path = run_dir / "generated_submovements.csv"
                fidelity_path = run_dir / "participant_fidelity.csv"
                run_dir.mkdir(parents=True, exist_ok=True)
                run_protocol_path = run_dir / "run_protocol.json"
                run_protocol = {
                    "latent_dim": latent_dim,
                    "outer_fold": fold_index,
                    "training_seed": seed,
                    "samples_per_subject": args.samples_per_subject,
                    "submovement_restarts": fit_cfg.restarts,
                    "submovement_max_nfev": fit_cfg.max_nfev,
                }
                if run_protocol_path.exists():
                    existing = json.loads(run_protocol_path.read_text(encoding="utf-8"))
                    if existing != run_protocol:
                        raise ValueError(
                            f"cached run protocol differs from request: {run_dir}"
                        )
                else:
                    run_protocol_path.write_text(
                        json.dumps(run_protocol, indent=2) + "\n", encoding="utf-8"
                    )
                if generated_path.exists() and not args.force:
                    generated = pd.read_csv(generated_path)
                else:
                    model, norm = load_per_trial_checkpoint(
                        _checkpoint(args.study, fold_index, latent_dim, seed), device
                    )
                    covariance = training_latent_noise_covariance(
                        model, train, norm, device
                    )
                    items = generate_run(
                        model,
                        norm,
                        test,
                        args.samples_per_subject,
                        device,
                        seed,
                        run_name,
                        covariance,
                    )
                    executor = (
                        ThreadPoolExecutor
                        if args.backend == "thread"
                        else ProcessPoolExecutor
                    )
                    with executor(max_workers=args.jobs) as pool:
                        rows = list(pool.map(fit_generated, items, repeat(fit_cfg)))
                    generated = pd.DataFrame(rows)
                    generated.to_csv(generated_path, index=False)
                if fidelity_path.exists() and not args.force:
                    fidelity = pd.read_csv(fidelity_path)
                else:
                    fidelity = evaluate_run(
                        generated[_success_mask(generated)],
                        empirical,
                        test,
                    )
                    fidelity.to_csv(fidelity_path, index=False)
                numeric = fidelity.select_dtypes(include=[np.number])
                summary = {
                    "model_family": "cvae",
                    "latent_dim": latent_dim,
                    "outer_fold": fold_index,
                    "training_seed": seed,
                    "n_test_subjects": len(folds[fold_index].test_subjects),
                    "samples_per_subject": args.samples_per_subject,
                    "generated_fit_success_rate": float(
                        _success_mask(generated).mean()
                    ),
                    **{
                        f"mean_{column}": float(numeric[column].mean())
                        for column in numeric.columns
                        if column not in {"n_empirical", "n_generated"}
                    },
                }
                summaries.append(summary)
                pd.DataFrame(summaries).to_csv(out / "run_summary.csv", index=False)
                print(
                    f"complete n={latent_dim} fold={fold_index} seed={seed} "
                    f"fit={summary['generated_fit_success_rate']:.3f}",
                    flush=True,
                )

    protocol = {
        "protocol_version": "strategy-confirmatory-v1",
        "source_repository": "https://github.com/JasonFriedman/submovements",
        "source_commit": "9c2f40ccc922d542242329c46cfd524c21188b4a",
        "empirical_fits": "all retained movement intervals; two optimizer restarts",
        "generated_fits": {
            "samples_per_held_out_subject": args.samples_per_subject,
            "optimizer_restarts": fit_cfg.restarts,
            "max_nfev": fit_cfg.max_nfev,
        },
        "primary_count_selection": "smallest k with normalized error <= 0.05; fallback <= 0.10; else minimum error",
        "sensitivity_count_selection": "minimum BIC",
        "latent_dims": args.dims,
        "folds": args.folds,
        "seeds": args.seeds,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
