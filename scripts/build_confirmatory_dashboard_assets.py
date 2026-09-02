"""Build compact assets for the final strategy-window dashboard.

Live generation uses the fold-0, seed-42 CVAE checkpoints.  Validation tabs
use out-of-fold participants from all four rotations and never consume the
live participant's query trials when computing a context fingerprint.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.run_corrected_study import (
    context_query_for_trials,
    load_per_trial_checkpoint,
    training_latent_noise_covariance,
)
from src.confirmatory_protocol import make_participant_folds, partition_trials
from src.evaluate import encode_trials
from src.features import compute_trial_features
from src.trajectory_view import project_trials_to_table_plane, select_trials_window


TRAINING_STUDY = ROOT / "studies" / "final_strategy_evaluation"
STUDY = ROOT / "studies" / "review_corrected_evaluation"
OUT = STUDY / "results" / "dashboard"
RUNS = TRAINING_STUDY / "runs"
ANALYSIS = STUDY / "results" / "analysis"
MIN_JERK = TRAINING_STUDY / "results" / "minimum_jerk"
REAL_SUBMOVEMENTS = (
    ROOT
    / "studies"
    / "strategy_window_comparison"
    / "results"
    / "submovements_real.csv"
)
LIVE_DIMS = (2, 3, 4, 8)
LIVE_FOLD = 0
LIVE_SEED = 42


def result_fingerprint_accuracy() -> pd.DataFrame:
    patterns = {
        "cvae": "fold*/cvae_z*_seed*/result.json",
        "conditional_ae": "fold*/conditional_ae_z*_seed*/result.json",
        "unconditional_vae": "fold*/unconditional_vae_z*_seed*/result.json",
        "spline_pca": "fold*/spline_pca_z*/result.json",
    }
    rows = []
    for family, pattern in patterns.items():
        for path in sorted((RUNS / family).glob(pattern)):
            result = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model_family": family,
                    "latent_dim": int(result["latent_dim"]),
                    "outer_fold": int(result["outer_fold"]),
                    "training_seed": result.get("training_seed"),
                    "fingerprint_balanced_accuracy": float(
                        result["fingerprint_balanced_accuracy"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (config.DATA_PROCESSED_DIR / "canonical_trials.pkl").open("rb") as handle:
        canonical = pickle.load(handle)
    trials = project_trials_to_table_plane(
        select_trials_window(canonical, config.WINDOW_GO_TO_ARRIVAL)
    )
    subjects = [trial["metadata"]["subject"] for trial in trials]
    folds = make_participant_folds(subjects)

    all_query_ids: set[str] = set()
    fold0_train = fold0_test = None
    for fold in folds:
        train, _, test = partition_trials(trials, fold)
        if fold.fold == LIVE_FOLD:
            fold0_train, fold0_test = train, test
        for split in context_query_for_trials(test, config.CONTEXT_QUERY_SEED):
            all_query_ids.update(
                test[index]["metadata"]["trial_id"] for index in split.query_indices
            )
    if fold0_train is None or fold0_test is None:
        raise RuntimeError("live fold was not found")

    basic = pd.DataFrame([compute_trial_features(trial) for trial in trials])
    empirical = basic[basic.trial_id.isin(all_query_ids)].copy()
    real_sub = pd.read_csv(REAL_SUBMOVEMENTS)
    sub_columns = [
        "trial_id",
        "mj_n_components",
        "mj_n_components_bic",
        "mj_fit_error",
        "mj_first_duration_s",
        "mj_first_amplitude",
        "mj_secondary_amplitude_fraction",
        "mj_mean_overlap_pct",
        "mj_fit_success",
    ]
    empirical = empirical.merge(
        real_sub[sub_columns], on="trial_id", how="left", validate="one_to_one"
    )
    empirical.to_csv(OUT / "empirical_query_features.csv", index=False)

    fingerprint_rows: list[dict] = []
    latent_stats: dict[str, dict] = {}
    for latent_dim in LIVE_DIMS:
        run_name = f"cvae_z{latent_dim}_seed{LIVE_SEED}"
        checkpoint = RUNS / "cvae" / f"fold{LIVE_FOLD}" / run_name / "checkpoint.pt"
        model, norm = load_per_trial_checkpoint(checkpoint, "cpu")
        train_mu, _, _, _ = encode_trials(model, fold0_train, norm, "cpu")
        test_mu, _, _, _ = encode_trials(model, fold0_test, norm, "cpu")
        covariance = training_latent_noise_covariance(
            model, fold0_train, norm, "cpu"
        )
        latent_stats[run_name] = {
            "training_center": train_mu.mean(axis=0).tolist(),
            "training_scale": (train_mu.std(axis=0) + 1e-8).tolist(),
            "shared_covariance": covariance.tolist(),
            "outer_fold": LIVE_FOLD,
            "training_seed": LIVE_SEED,
            "interpretation": "run-specific coordinates; no cross-seed axis identity",
        }
        for split in context_query_for_trials(fold0_test, config.CONTEXT_QUERY_SEED):
            row = {
                "run": run_name,
                "latent_dim": latent_dim,
                "subject": split.subject,
                "n_context": len(split.context_indices),
                "n_query": len(split.query_indices),
            }
            row.update(
                {
                    f"z{index + 1}": float(value)
                    for index, value in enumerate(
                        test_mu[split.context_indices].mean(axis=0)
                    )
                }
            )
            fingerprint_rows.append(row)
    pd.DataFrame(fingerprint_rows).to_csv(
        OUT / "subject_fingerprints.csv", index=False
    )
    (OUT / "latent_stats.json").write_text(
        json.dumps(latent_stats, indent=2) + "\n", encoding="utf-8"
    )

    generated_rows = []
    for latent_dim in (3, 8):
        for fold in range(4):
            path = (
                MIN_JERK
                / f"cvae_z{latent_dim}_fold{fold}_seed{LIVE_SEED}"
                / "generated_submovements.csv"
            )
            frame = pd.read_csv(path)
            frame["latent_dim"] = latent_dim
            frame["outer_fold"] = fold
            frame["training_seed"] = LIVE_SEED
            generated_rows.append(frame)
    generated = pd.concat(generated_rows, ignore_index=True)
    generated.to_csv(OUT / "generated_validation.csv", index=False)

    oof = pd.read_csv(ANALYSIS / "oof_metrics_summary.csv")
    accuracy = result_fingerprint_accuracy()
    accuracy_summary = (
        accuracy.groupby(["model_family", "latent_dim"], as_index=False)
        .agg(
            fingerprint_balanced_accuracy_mean=("fingerprint_balanced_accuracy", "mean"),
            fingerprint_balanced_accuracy_sd=("fingerprint_balanced_accuracy", "std"),
        )
    )
    comparison = oof.merge(
        accuracy_summary, on=["model_family", "latent_dim"], how="left"
    )
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    pd.read_csv(ANALYSIS / "timing_outlier_audit.csv").to_csv(
        OUT / "timing_outlier_audit.csv", index=False
    )
    pd.read_csv(
        TRAINING_STUDY / "results" / "condition_effects" / "condition_trajectory_summary.csv"
    ).to_csv(OUT / "condition_trajectory_summary.csv", index=False)
    pd.read_csv(MIN_JERK / "analysis" / "oof_summary.csv").to_csv(
        OUT / "minimum_jerk_summary.csv", index=False
    )
    pd.read_csv(MIN_JERK / "assumption_sensitivity_summary.csv").to_csv(
        OUT / "minimum_jerk_sensitivity.csv", index=False
    )
    for name, source in {
        "behavioral_probe_summary.csv": "behavioral_probes/summary.csv",
        "timing_fairness_summary.csv": "timing_fairness/summary.csv",
        "timing_fairness_paired.csv": "timing_fairness/paired_comparisons.csv",
        "submovement_sampling_reference.csv": "sampling_reference/summary.csv",
    }.items():
        pd.read_csv(STUDY / "results" / source).to_csv(OUT / name,index=False)
    (OUT / "event_audit.json").write_text(
        (STUDY / "results/event_audit/summary.json").read_text(),encoding="utf-8")

    speed_rows = []
    for trial in trials:
        speed_rows.append(
            {
                "sp": int(trial["metadata"]["sp"]),
                "target_speed": float(
                    trial["metadata"].get("target_speed_screen_s", np.nan)
                ),
            }
        )
    speed = pd.DataFrame(speed_rows).dropna()
    speed.groupby("sp", as_index=False).target_speed.agg(
        speed_min="min", speed_median="median", speed_max="max"
    ).to_csv(OUT / "condition_speed_ranges.csv", index=False)

    manifest = {
        "protocol_version": "strategy-confirmatory-v1",
        "evaluation_version": "post-review-training-reference-v1",
        "event_anchor": "MAT target motion; marker 5 is appearance; recording end is an arrival proxy",
        "window_mode": config.WINDOW_GO_TO_ARRIVAL,
        "n_trials": len(trials),
        "n_subjects": len(set(subjects)),
        "latent_dimensions": list(LIVE_DIMS),
        "live_fold": LIVE_FOLD,
        "live_seed": LIVE_SEED,
        "live_test_subjects": list(folds[LIVE_FOLD].test_subjects),
        "all_validation_subjects": sorted(set(empirical.subject)),
        "validation_dimensions": [3, 8],
        "position_dimensions": ["lateral x", "forward y"],
        "normalised_length": config.NORMALISED_LENGTH,
        "condition_controls_validated": False,
        "condition_note": (
            "Condition sliders are exploratory: no CVAE-vs-unconditional "
            "condition diagnostic survived corrected inference."
        ),
        "minimum_jerk_note": (
            "Component count is kinematic and model-order sensitive; it is not "
            "a validated cognitive-strategy label."
        ),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
