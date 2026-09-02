"""Fail fast when the strategy-window release artifacts are inconsistent."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


EXPECTED_TRIALS = 4732
EXPECTED_SUBJECTS = 28
EXPECTED_RUNS = {
    (window, dim, seed)
    for window in config.WINDOW_MODES
    for dim in (2, 3, 4, 8)
    for seed in (42, 43, 44)
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-release", action="store_true")
    args = parser.parse_args()

    with (config.DATA_PROCESSED_DIR / "canonical_trials.pkl").open("rb") as handle:
        trials = pickle.load(handle)
    trial_ids = [trial["metadata"]["trial_id"] for trial in trials]
    subjects = {trial["metadata"]["subject"] for trial in trials}
    require(len(trials) == EXPECTED_TRIALS, f"canonical cohort has {EXPECTED_TRIALS} trials")
    require(len(set(trial_ids)) == EXPECTED_TRIALS, "canonical trial IDs are unique")
    require(len(subjects) == EXPECTED_SUBJECTS, f"canonical cohort has {EXPECTED_SUBJECTS} participants")
    require(all("pos_movement_norm" in trial and "pos_go_to_arrival_norm" in trial for trial in trials), "both windows exist for every retained trial")

    model_results = pd.read_csv(config.RESULTS_DIR / "model_seed_results.csv")
    observed_runs = set(zip(model_results.window_mode, model_results.latent_dim, model_results.seed))
    require(len(model_results) == len(EXPECTED_RUNS), "model table has exactly 24 repeated-seed runs")
    require(observed_runs == EXPECTED_RUNS, "model table contains every pre-specified window/dimension/seed combination")
    for window, dim, seed in EXPECTED_RUNS:
        checkpoint = config.RESULTS_DIR / window / "models" / f"cvae_{window}_z{dim}_seed{seed}" / "checkpoint.pt"
        require(checkpoint.exists(), f"checkpoint exists for {window}, n={dim}, seed={seed}")

    sub = pd.read_csv(config.RESULTS_DIR / "submovements_real.csv")
    require(len(sub) == EXPECTED_TRIALS, "submovement table has one row per canonical trial")
    require(sub.trial_id.nunique() == EXPECTED_TRIALS, "submovement trial IDs are unique")
    require(set(sub.trial_id) == set(trial_ids), "submovement and canonical trial IDs match exactly")

    for window in config.WINDOW_MODES:
        baseline = config.RESULTS_DIR / window / "baselines"
        require((baseline / "baselines.json").exists(), f"{window} baseline summary exists")
        require((baseline / "kmeans_selection_corrected.csv").exists(), f"{window} permutation-tested K-Means exists")

    associations_path = config.RESULTS_DIR / "latent_associations" / "latent_submovement_associations.csv"
    require(associations_path.exists(), "latent association table exists")
    associations = pd.read_csv(associations_path)
    require(len(associations) == 1632, "latent association table has all models, levels, targets, and axes")

    assets = config.RESULTS_DIR / "dashboard"
    manifest_path = assets / "manifest.json"
    require(manifest_path.exists(), "dashboard manifest exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["n_trials"] == EXPECTED_TRIALS, "dashboard manifest uses canonical trial count")
    latent_stats = json.loads((assets / "latent_stats.json").read_text(encoding="utf-8"))
    require(len(latent_stats) == 24, "dashboard contains latent controls for all 24 models")

    if args.require_release:
        report = config.STUDY_ROOT / "output" / "pdf" / "Interception_Strategy_Window_Comparison.pdf"
        bundle = config.STUDY_ROOT / "output" / "share" / "Interception_Strategy_Dashboard.zip"
        require(report.exists() and report.stat().st_size > 100_000, "final PDF exists and is nontrivial")
        require(bundle.exists() and bundle.stat().st_size > 1_000_000, "dashboard ZIP exists and is nontrivial")
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
        forbidden = [name for name in names if "canonical_trials" in name or name.lower().endswith((".mat", ".pkl"))]
        require(not forbidden, "dashboard bundle contains no canonical/raw trial cache or MAT files")
    print("Strategy-window release validation completed successfully.")


if __name__ == "__main__":
    main()
