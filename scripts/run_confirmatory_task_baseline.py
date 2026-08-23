"""Run the condition-only Ridge/residual baseline on frozen participant folds."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.run_confirmatory_cvae import DEFAULT_STUDY
from src.confirmatory_controls import ConditionRidge, evaluate_condition_ridge
from src.confirmatory_protocol import (
    PROTOCOL_VERSION,
    partition_trials,
    write_or_verify_manifest,
    write_or_verify_protocol,
)
from src.trajectory_view import project_trials_to_table_plane, select_trials_window


def collect_results(run_root: Path) -> pd.DataFrame:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_root.glob("fold*/condition_ridge_seed*/result.json"))
    ]
    return pd.DataFrame(rows).sort_values(["outer_fold", "generation_seed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, default=config.DATA_PROCESSED_DIR / "canonical_trials.pkl")
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with args.trials.open("rb") as handle:
        canonical = pickle.load(handle)
    trials = project_trials_to_table_plane(
        select_trials_window(canonical, config.WINDOW_GO_TO_ARRIVAL)
    )
    folds = write_or_verify_manifest(args.study / "protocol" / "participant_folds.json", trials)
    run_root = args.study / "runs" / "condition_ridge"
    results_dir = args.study / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_or_verify_protocol(
        args.study / "protocol" / "condition_ridge_protocol.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_family": "condition_ridge",
            "purpose": "task-metadata-only lower benchmark without participant fingerprint",
            "inputs": "sp category, starting side, exact executed target speed",
            "outputs": "trajectory, movement time, initiation time",
            "generation": "condition prediction plus residual sampled from matching training stratum",
            "outer_folds": 4,
            "generation_seeds": args.seeds,
        },
    )

    for fold_index in args.folds:
        train, validation, test = partition_trials(trials, folds[fold_index])
        model = ConditionRidge.fit(train, validation)
        for seed in args.seeds:
            run_dir = run_root / f"fold{fold_index}" / f"condition_ridge_seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists() and not args.force:
                print(f"Skip completed: {run_dir}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            summary = evaluate_condition_ridge(model, test, run_dir, seed)
            result = {
                "protocol_version": PROTOCOL_VERSION,
                "model_family": "condition_ridge",
                "window_mode": config.WINDOW_GO_TO_ARRIVAL,
                "outer_fold": fold_index,
                "latent_dim": None,
                "training_seed": None,
                "generation_seed": seed,
                "n_train_subjects": len(folds[fold_index].train_subjects),
                "n_validation_subjects": len(folds[fold_index].validation_subjects),
                "n_test_subjects": len(folds[fold_index].test_subjects),
                "n_train_trials": len(train),
                "n_validation_trials": len(validation),
                "n_test_trials": len(test),
                "test_subjects": list(folds[fold_index].test_subjects),
                **summary,
            }
            result_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
            print(pd.DataFrame([result]).to_string(index=False), flush=True)

    frame = collect_results(run_root)
    frame.to_csv(results_dir / "condition_ridge_all_runs.csv", index=False)
    print(f"Completed rows available: {len(frame)}")


if __name__ == "__main__":
    main()
