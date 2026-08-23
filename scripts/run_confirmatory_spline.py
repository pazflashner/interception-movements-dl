"""Run the matched shape-only spline+PCA baseline on frozen folds."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.run_confirmatory_cvae import DEFAULT_STUDY
from src.baseline_spline import SplinePCARepresentation
from src.confirmatory_protocol import (
    PROTOCOL_VERSION,
    partition_trials,
    write_or_verify_manifest,
    write_or_verify_protocol,
)
from src.confirmatory_spline import TimingRidge, evaluate_spline_run
from src.trajectory_view import project_trials_to_table_plane, select_trials_window


def collect_results(run_root: Path) -> pd.DataFrame:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_root.glob("fold*/spline_pca_z*/result.json"))
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["outer_fold", "latent_dim"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, default=config.DATA_PROCESSED_DIR / "canonical_trials.pkl")
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 3, 4, 8])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with args.trials.open("rb") as handle:
        canonical = pickle.load(handle)
    trials = project_trials_to_table_plane(
        select_trials_window(canonical, config.WINDOW_GO_TO_ARRIVAL)
    )
    folds = write_or_verify_manifest(args.study / "protocol" / "participant_folds.json", trials)
    run_root = args.study / "runs" / "spline_pca"
    results_dir = args.study / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_or_verify_protocol(
        args.study / "protocol" / "spline_pca_protocol.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_family": "spline_pca",
            "encoder_inputs": "shape-only cubic spline coefficients",
            "timing_withheld_from_encoder": True,
            "timing_predictor": "ridge from latent code plus task condition",
            "pca_fit": "training participants only",
            "latent_dims": args.dims,
            "deterministic": True,
        },
    )

    for fold_index in args.folds:
        fold = folds[fold_index]
        train, validation, test = partition_trials(trials, fold)
        for latent_dim in args.dims:
            run_dir = run_root / f"fold{fold_index}" / f"spline_pca_z{latent_dim}"
            result_path = run_dir / "result.json"
            if result_path.exists() and not args.force:
                print(f"Skip completed: {run_dir}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            candidates = [
                SplinePCARepresentation(
                    n_components=latent_dim,
                    include_timing=False,
                    standardize_coefficients=standardize,
                ).fit(train)
                for standardize in (False, True)
            ]
            validation_truth = np.stack([trial["pos_norm"] for trial in validation])
            validation_mse = []
            for candidate in candidates:
                validation_reconstruction, _ = candidate.decode(candidate.encode(validation))
                validation_mse.append(
                    float(np.mean((validation_truth - validation_reconstruction) ** 2))
                )
            selected_index = int(np.argmin(validation_mse))
            representation = candidates[selected_index]
            timing_model = TimingRidge.fit(representation, train, validation)
            summary = evaluate_spline_run(
                representation,
                timing_model,
                train,
                validation,
                test,
                run_dir,
                config.CONTEXT_QUERY_SEED,
            )
            result = {
                "protocol_version": PROTOCOL_VERSION,
                "model_family": "spline_pca",
                "window_mode": config.WINDOW_GO_TO_ARRIVAL,
                "outer_fold": fold_index,
                "latent_dim": latent_dim,
                "training_seed": None,
                "n_train_subjects": len(fold.train_subjects),
                "n_validation_subjects": len(fold.validation_subjects),
                "n_test_subjects": len(fold.test_subjects),
                "n_train_trials": len(train),
                "n_validation_trials": len(validation),
                "n_test_trials": len(test),
                "test_subjects": list(fold.test_subjects),
                "pca_explained_variance": float(
                    representation.pca_.explained_variance_ratio_.sum()
                ),
                "standardize_spline_coefficients": representation.standardize_coefficients,
                "validation_mse_raw_coefficients": validation_mse[0],
                "validation_mse_standardized_coefficients": validation_mse[1],
                **summary,
            }
            result_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
            collect_results(run_root).to_csv(results_dir / "spline_pca_all_runs.csv", index=False)
            print(pd.DataFrame([result]).to_string(index=False))

    frame = collect_results(run_root)
    frame.to_csv(results_dir / "spline_pca_all_runs.csv", index=False)
    print(f"Completed rows available: {len(frame)}")


if __name__ == "__main__":
    main()
