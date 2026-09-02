"""Re-evaluate frozen models into a separate study; never overwrite training runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from scripts.run_corrected_study import evaluate_per_trial_run, load_per_trial_checkpoint
from src.baseline_spline import SplinePCARepresentation
from src.confirmatory_controls import ConditionRidge, evaluate_condition_ridge
from src.confirmatory_evaluation import prediction_tables
from src.confirmatory_protocol import partition_trials, write_or_verify_manifest
from src.confirmatory_spline import TimingRidge, evaluate_spline_run
from src.context_query import DistanceReference
from src.features import compute_trial_features, kinematic_features_for_dim
from src.trajectory_view import project_trials_to_table_plane, select_trials_window

SOURCE = ROOT / "studies" / "final_strategy_evaluation"
DEST = ROOT / "studies" / "review_corrected_evaluation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["cvae", "conditional_ae", "unconditional_vae", "spline_pca", "condition_ridge"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    with (config.DATA_PROCESSED_DIR / "canonical_trials.pkl").open("rb") as handle:
        trials = project_trials_to_table_plane(select_trials_window(pickle.load(handle), config.WINDOW_GO_TO_ARRIVAL))
    folds = write_or_verify_manifest(DEST / "protocol" / "participant_folds.json", trials)
    (DEST / "results").mkdir(parents=True, exist_ok=True)
    for source in (SOURCE / "protocol").glob("*.json"):
        if source.name != "participant_folds.json":
            shutil.copy2(source, DEST / "protocol" / source.name)
    for family in args.families:
        for fold_index, fold in enumerate(folds):
            train, validation, test = partition_trials(trials, fold)
            reference = DistanceReference.fit(pd.DataFrame([compute_trial_features(t) for t in train]), kinematic_features_for_dim(2))
            ridge = None
            for source_result in sorted((SOURCE / "runs" / family / f"fold{fold_index}").glob("*/result.json")):
                source_run = source_result.parent
                target = DEST / "runs" / family / f"fold{fold_index}" / source_run.name
                target.mkdir(parents=True, exist_ok=True)
                if (target / "result.json").exists():
                    continue
                original = json.loads(source_result.read_text(encoding="utf-8"))
                if family in {"cvae", "conditional_ae", "unconditional_vae"}:
                    model, norm = load_per_trial_checkpoint(source_run / "checkpoint.pt", "cpu")
                    summary = evaluate_per_trial_run(model, norm, train, validation, test, target, config.CONTEXT_QUERY_SEED, "cpu")
                    timing, reconstruction, detailed = prediction_tables(model, test, norm, "cpu")
                    timing.to_csv(target / "timing_predictions.csv", index=False)
                    reconstruction.to_csv(target / "reconstruction_predictions.csv", index=False)
                    summary.update(detailed)
                elif family == "spline_pca":
                    representation = SplinePCARepresentation(n_components=original["latent_dim"], include_timing=False,
                        standardize_coefficients=original["standardize_spline_coefficients"]).fit(train)
                    timing_model = TimingRidge.fit(representation, train, validation)
                    summary = evaluate_spline_run(representation, timing_model, train, validation, test, target, config.CONTEXT_QUERY_SEED)
                else:
                    ridge = ridge or ConditionRidge.fit(train, validation)
                    summary = evaluate_condition_ridge(ridge, test, target, original["generation_seed"], reference)
                for name in ("split.json", "config.yaml", "history.json"):
                    if (source_run / name).exists():
                        shutil.copy2(source_run / name, target / name)
                provenance = {"source_run": source_run.relative_to(ROOT).as_posix(),
                              "source_result_sha256": hashlib.sha256(source_result.read_bytes()).hexdigest(),
                              "neural_retraining": False, "labels_changed": False,
                              "evaluation_version": "post-review-training-reference-v1"}
                (target / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
                (target / "result.json").write_text(json.dumps({**original, **summary, **provenance}, indent=2), encoding="utf-8")
                print(f"Corrected {family}/{fold_index}/{source_run.name}", flush=True)
        records = [json.loads(p.read_text()) for p in (DEST / "runs" / family).glob("fold*/*/result.json")]
        pd.DataFrame(records).to_csv(DEST / "results" / f"{family}_all_runs.csv", index=False)
        print(f"{family}: {len(records)} results", flush=True)


if __name__ == "__main__":
    main()
