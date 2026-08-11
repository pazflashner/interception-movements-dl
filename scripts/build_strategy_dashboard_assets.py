"""Build compact assets for the strategy-window dashboard."""
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
from src.trajectory_view import project_trials_to_table_plane, select_trials_window


def parse_run(name: str) -> tuple[int, int]:
    return (
        int(name.split("_z", 1)[1].split("_seed", 1)[0]),
        int(name.rsplit("seed", 1)[1]),
    )


def main() -> None:
    results = config.RESULTS_DIR
    out = results / "dashboard"
    out.mkdir(parents=True, exist_ok=True)

    with (config.DATA_PROCESSED_DIR / "canonical_trials.pkl").open("rb") as handle:
        canonical = pickle.load(handle)

    fingerprint_rows: list[dict] = []
    latent_stats: dict[str, dict] = {}
    query_ids: set[str] = set()
    test_subjects: list[str] = []

    for window_mode in config.WINDOW_MODES:
        trials = project_trials_to_table_plane(select_trials_window(canonical, window_mode))
        train_trials, _, test_trials = split_subjects(trials, 17, 4, 7, 42)
        subjects = [trial["metadata"]["subject"] for trial in test_trials]
        sp = [trial["metadata"]["sp"] for trial in test_trials]
        side = [trial["metadata"]["side"] for trial in test_trials]
        splits = split_context_query(subjects, sp, side, seed=config.CONTEXT_QUERY_SEED)
        test_subjects = sorted(set(subjects))
        if not query_ids:
            for split in splits:
                query_ids.update(
                    test_trials[index]["metadata"]["trial_id"]
                    for index in split.query_indices
                )

        for run in sorted((results / window_mode / "models").glob("cvae_*_z*_seed*")):
            checkpoint = run / "checkpoint.pt"
            if not checkpoint.exists():
                continue
            latent_dim, seed = parse_run(run.name)
            model, norm = load_per_trial_checkpoint(checkpoint, "cpu")
            train_mu, _, _, _ = encode_trials(model, train_trials, norm, "cpu")
            test_mu, _, _, _ = encode_trials(model, test_trials, norm, "cpu")
            center = train_mu.mean(axis=0)
            scale = np.where(train_mu.std(axis=0) > 1e-6, train_mu.std(axis=0), 1.0)
            covariance = training_latent_noise_covariance(model, train_trials, norm, "cpu")
            latent_stats[run.name] = {
                "window_mode": window_mode,
                "latent_dim": latent_dim,
                "seed": seed,
                "training_center": center.tolist(),
                "training_scale": scale.tolist(),
                "shared_covariance": covariance.tolist(),
            }
            for split in splits:
                mean = test_mu[split.context_indices].mean(axis=0)
                row = {
                    "run": run.name,
                    "window_mode": window_mode,
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

    physical = project_trials_to_table_plane(
        select_trials_window(canonical, config.WINDOW_MOVEMENT_ONLY)
    )
    _, _, test_trials = split_subjects(physical, 17, 4, 7, 42)
    basic = pd.DataFrame([compute_trial_features(trial) for trial in test_trials])
    sub = pd.read_csv(results / "submovements_real.csv")
    sub_columns = [
        "trial_id", "mj_n_components", "mj_fit_error", "mj_first_duration_s",
        "mj_first_amplitude", "mj_secondary_amplitude_fraction",
        "mj_mean_overlap_pct", "mj_pattern", "mj_fit_success",
    ]
    empirical = basic[basic.trial_id.isin(query_ids)].merge(
        sub[sub_columns], on="trial_id", how="left", validate="one_to_one"
    )
    empirical.to_csv(out / "empirical_query_features.csv", index=False)

    condition_rows = []
    for trial in canonical:
        meta = trial["metadata"]
        speed = meta.get("target_speed_screen_s")
        if speed is not None and np.isfinite(float(speed)):
            condition_rows.append({"sp": int(meta["sp"]), "speed": float(speed)})
    speed_summary = pd.DataFrame(condition_rows).groupby("sp").speed.agg(
        speed_min="min", speed_median="median", speed_max="max", n="size"
    ).reset_index()
    speed_summary.to_csv(out / "condition_speed_ranges.csv", index=False)

    manifest = {
        "study": "strategy_window_comparison",
        "condition": 2,
        "n_trials": len(canonical),
        "n_subjects": len({trial["metadata"]["subject"] for trial in canonical}),
        "test_subjects": test_subjects,
        "context_query_seed": config.CONTEXT_QUERY_SEED,
        "windows": list(config.WINDOW_MODES),
        "latent_dimensions": [2, 3, 4, 8],
        "model_seeds": [42, 43, 44],
        "trajectory_axes": ["x_lateral", "y_forward"],
        "trajectory_points": config.NORMALISED_LENGTH,
        "timing_withheld_from_encoder": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dashboard assets written to {out}")


if __name__ == "__main__":
    main()
