"""Build compact, leakage-safe assets for the interactive final-study dashboard."""
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
    load_per_trial_checkpoint,
    training_latent_noise_covariance,
)
from src.context_query import split_context_query
from src.evaluate import encode_trials
from src.features import compute_trial_features
from src.train import split_subjects
from src.trajectory_view import project_trials_to_table_plane


def run_parts(name: str) -> tuple[int, int]:
    left, right = name.split("_seed")
    return int(left.rsplit("z", 1)[1]), int(right)


def main() -> None:
    trials_path = ROOT / "data" / "final_study" / "trials.pkl"
    models_dir = ROOT / "results" / "final_study" / "core_models"
    submovements_path = ROOT / "results" / "final_study" / "submovements_real.csv"
    out = ROOT / "results" / "final_study" / "dashboard"
    out.mkdir(parents=True, exist_ok=True)

    with trials_path.open("rb") as handle:
        trials = project_trials_to_table_plane(pickle.load(handle))
    train_trials, _, test_trials = split_subjects(trials, 17, 4, 7, 42)

    test_subjects = [trial["metadata"]["subject"] for trial in test_trials]
    test_sp = [trial["metadata"]["sp"] for trial in test_trials]
    test_side = [trial["metadata"]["side"] for trial in test_trials]
    cq_splits = split_context_query(
        test_subjects, test_sp, test_side, seed=config.CONTEXT_QUERY_SEED
    )

    fingerprint_rows: list[dict] = []
    latent_stats: dict[str, dict] = {}
    for run in sorted(models_dir.glob("trajectory_only_z*_seed*")):
        checkpoint = run / "checkpoint.pt"
        if not checkpoint.exists():
            continue
        latent_dim, seed = run_parts(run.name)
        model, norm = load_per_trial_checkpoint(checkpoint, "cpu")
        model.eval()
        train_mu, _, _, _ = encode_trials(model, train_trials, norm, "cpu")
        test_mu, _, _, _ = encode_trials(model, test_trials, norm, "cpu")
        center = train_mu.mean(axis=0)
        scale = train_mu.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        covariance = training_latent_noise_covariance(
            model, train_trials, norm, "cpu"
        )
        latent_stats[run.name] = {
            "latent_dim": latent_dim,
            "seed": seed,
            "training_center": center.tolist(),
            "training_scale": scale.tolist(),
            "shared_covariance": covariance.tolist(),
        }
        for split in cq_splits:
            mean = test_mu[split.context_indices].mean(axis=0)
            row = {
                "run": run.name,
                "latent_dim": latent_dim,
                "seed": seed,
                "subject": split.subject,
                "n_context": len(split.context_indices),
                "n_query": len(split.query_indices),
            }
            row.update({f"z{i + 1}": float(value) for i, value in enumerate(mean)})
            fingerprint_rows.append(row)

    pd.DataFrame(fingerprint_rows).to_csv(out / "subject_fingerprints.csv", index=False)
    (out / "latent_stats.json").write_text(
        json.dumps(latent_stats, indent=2), encoding="utf-8"
    )

    query_ids: set[str] = set()
    for split in cq_splits:
        query_ids.update(
            test_trials[index]["metadata"]["trial_id"]
            for index in split.query_indices
        )
    basic = pd.DataFrame([compute_trial_features(trial) for trial in test_trials])
    speed_by_trial = {
        trial["metadata"]["trial_id"]: trial["metadata"].get("target_speed_screen_s")
        for trial in test_trials
    }
    basic["target_speed_screen_s"] = basic.trial_id.map(speed_by_trial)
    submovements = pd.read_csv(submovements_path)
    sub_columns = [
        "trial_id", "mj_n_components", "mj_fit_error", "mj_first_duration_s",
        "mj_first_amplitude", "mj_secondary_amplitude_fraction",
        "mj_mean_overlap_pct", "mj_pattern", "mj_fit_success",
    ]
    empirical = basic[basic.trial_id.isin(query_ids)].merge(
        submovements[sub_columns], on="trial_id", how="left", validate="one_to_one"
    )
    empirical.to_csv(out / "empirical_query_features.csv", index=False)

    condition_rows = []
    for trial in trials:
        meta = trial["metadata"]
        speed = meta.get("target_speed_screen_s")
        if speed is not None and np.isfinite(float(speed)):
            condition_rows.append({"sp": int(meta["sp"]), "speed": float(speed)})
    condition_df = pd.DataFrame(condition_rows)
    condition_summary = condition_df.groupby("sp").speed.agg(
        speed_min="min", speed_median="median", speed_max="max", n="size"
    ).reset_index()
    condition_summary.to_csv(out / "condition_speed_ranges.csv", index=False)

    manifest = {
        "protocol": "final_study",
        "condition": 2,
        "n_trials": len(trials),
        "n_subjects": len({trial["metadata"]["subject"] for trial in trials}),
        "test_subjects": sorted(set(test_subjects)),
        "context_query_seed": config.CONTEXT_QUERY_SEED,
        "latent_dimensions": sorted({row["latent_dim"] for row in fingerprint_rows}),
        "model_seeds": sorted({row["seed"] for row in fingerprint_rows}),
        "trajectory_axes": ["x_lateral", "y_forward"],
        "trajectory_points": config.NORMALISED_LENGTH,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dashboard assets written to {out}")


if __name__ == "__main__":
    main()
