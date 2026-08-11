"""Run the non-neural reference analyses on the final x-y representation."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_corrected_study import kmeans_with_selection_null
from src.baseline_spline import evaluate_spline_baseline, evaluate_spline_pca_baseline
from src.features import compute_trial_features
from src.train import split_subjects
from src.trajectory_view import MODEL_AXES, MODEL_AXIS_NAMES, project_trials_to_table_plane


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=str(ROOT / "data" / "final_study" / "trials.pkl"))
    parser.add_argument("--out", default=str(ROOT / "results" / "final_study" / "baselines"))
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 3, 8])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.trials, "rb") as handle:
        trials = project_trials_to_table_plane(pickle.load(handle))
    train, _, test = split_subjects(trials, 17, 4, 7, 42)
    features = pd.DataFrame([compute_trial_features(trial) for trial in trials])
    kmeans = kmeans_with_selection_null(
        trials, features, out, seed=42, permutations=args.permutations, k_values=[28]
    )
    spline = evaluate_spline_baseline(test)
    spline_pca = {
        n: evaluate_spline_pca_baseline(train, test, n_components=n)["mean_mse"]
        for n in args.dims
    }
    payload = {
        "representation": "x-y table plane",
        "axes": list(MODEL_AXES),
        "axis_names": list(MODEL_AXIS_NAMES),
        "kmeans": kmeans,
        "spline_per_trial": spline["mean_mse"],
        "spline_pca": spline_pca,
    }
    (out / "baselines.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
