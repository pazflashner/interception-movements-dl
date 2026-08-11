"""Build machine-readable summary tables for the final PDF."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def ci(values, seed=2026, n_boot=5000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root)
    final = root / "results" / "final_study"
    out = final / "summary"
    out.mkdir(parents=True, exist_ok=True)

    with open(root / "data" / "final_study" / "trials.pkl", "rb") as handle:
        trials = pickle.load(handle)
    sub_all = pd.read_csv(final / "submovements_real.csv")
    fit_success_rate = float(sub_all.mj_fit_success.mean())
    sub = sub_all[sub_all.mj_fit_success == True].copy()
    sub["recorded_success"] = (sub.successful == 1).astype(float)

    subject = sub.groupby("subject").agg(
        n_trials=("trial_id", "size"),
        mean_components=("mj_n_components", "mean"),
        single_component_rate=("mj_n_components", lambda x: float((x == 1).mean())),
        overlapping_rate=("mj_pattern", lambda x: float((x == "overlapping_components").mean())),
        sequential_rate=("mj_pattern", lambda x: float((x == "sequential_components").mean())),
        recorded_success_rate=("recorded_success", "mean"),
        mean_fit_error=("mj_fit_error", "mean"),
        mean_secondary_fraction=("mj_secondary_amplitude_fraction", "mean"),
        mean_overlap_pct=("mj_mean_overlap_pct", "mean"),
    ).reset_index()
    subject.to_csv(out / "subject_submovement_summary.csv", index=False)

    threshold_rows = []
    errors = sub[[f"mj_error_k{k}" for k in range(1, 5)]].to_numpy()
    for threshold in (0.03, 0.05, 0.10):
        selected = []
        for row in errors:
            passing = np.flatnonzero(np.isfinite(row) & (row <= threshold))
            selected.append(int(passing[0] + 1) if len(passing) else int(np.nanargmin(row) + 1))
        counts = pd.Series(selected).value_counts(normalize=True)
        threshold_rows.append({"threshold": threshold, **{f"rate_k{k}": float(counts.get(k, 0)) for k in range(1, 5)}})
    pd.DataFrame(threshold_rows).to_csv(out / "order_threshold_sensitivity.csv", index=False)

    model_seed = pd.read_csv(final / "core_models" / "seed_results.csv")
    model_numeric = model_seed.select_dtypes(include=[np.number])
    model_summary = model_numeric.groupby("latent_dim").agg(["mean", "std"])
    model_summary.columns = [f"{a}_{b}" for a, b in model_summary.columns]
    model_summary.reset_index().to_csv(out / "model_seed_summary.csv", index=False)

    fp = pd.read_csv(final / "fingerprint_evaluation" / "subject_distribution_probes.csv")
    fp.groupby(["latent_dim", "target"]).agg(
        r2_mean=("r2_test", "mean"), r2_sd=("r2_test", "std"),
        mae_mean=("mae_test", "mean"),
    ).reset_index().to_csv(out / "subject_probe_summary.csv", index=False)

    trial_probe = pd.read_csv(final / "fingerprint_evaluation" / "trial_behavior_probes.csv")
    trial_probe.groupby(["latent_dim", "target", "metric_type"]).mean(numeric_only=True).reset_index().to_csv(
        out / "trial_probe_summary.csv", index=False
    )

    generation = pd.read_csv(final / "generation" / "generation_summary.csv")
    generation["latent_dim"] = generation.run.str.extract(r"_z(\d+)_").astype(int)
    generation.groupby("latent_dim").mean(numeric_only=True).reset_index().to_csv(
        out / "generation_seed_summary.csv", index=False
    )

    stability_path = final / "submovement_stability.csv"
    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    results = {
        "dataset": {
            "retained_trials": len(trials),
            "subjects": len({t["metadata"]["subject"] for t in trials}),
            "outcome_counts": sub.responseText.value_counts().to_dict(),
        },
        "submovements": {
            "fit_success_rate": fit_success_rate,
            "component_count_rates": {str(int(k)): float(v) for k, v in sub.mj_n_components.value_counts(normalize=True).sort_index().items()},
            "pattern_rates": {str(k): float(v) for k, v in sub.mj_pattern.value_counts(normalize=True).items()},
            "median_error": float(sub.mj_fit_error.median()),
            "subject_single_rate_ci95": ci(subject.single_component_rate),
            "high_restart_same_count": float(stability.same_selected_count.mean()) if len(stability) else None,
        },
        "models": json.loads(model_seed.to_json(orient="records")),
    }
    (out / "final_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["submovements"], indent=2))


if __name__ == "__main__":
    main()
