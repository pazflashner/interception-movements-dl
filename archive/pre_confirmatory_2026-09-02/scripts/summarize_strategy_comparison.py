"""Aggregate the strategy-window study into machine-readable tables and figures."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


METRICS = [
    "window_reconstruction_mse_tracker_units2",
    "movement_reconstruction_mse_tracker_units2",
    "movement_time_s_r2",
    "initiation_time_s_r2",
    "fingerprint_balanced_accuracy",
    "mean_ks",
    "mean_ks_rejected_fdr",
    "median_energy_distance",
    "mean_mmd_rbf",
    "probe_positive_r2",
]


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, path: Path, lower_better=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {"movement_only": "#2563EB", "go_to_arrival": "#DC2626"}
    for window_mode, group in summary.groupby("window_mode"):
        group = group.sort_values("latent_dim")
        ax.errorbar(
            group.latent_dim,
            group[f"{metric}_mean"],
            yerr=group[f"{metric}_std"].fillna(0),
            marker="o", linewidth=2.2, capsize=4,
            color=colors[window_mode], label=window_mode.replace("_", " "),
        )
    if metric.endswith("_r2"):
        ax.axhline(0, color="#64748B", linewidth=1, linestyle="--")
    if metric == "fingerprint_balanced_accuracy":
        ax.axhline(1 / 7, color="#64748B", linewidth=1, linestyle="--", label="chance")
    ax.set_xticks(sorted(summary.latent_dim.unique()))
    ax.set_xlabel("Latent dimensions")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + (" (lower is better)" if lower_better else ""), loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = config.RESULTS_DIR
    out = root / "summary"
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    models = pd.read_csv(root / "model_seed_results.csv")
    grouped = models.groupby(["window_mode", "latent_dim"])[METRICS].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index()
    summary.to_csv(out / "model_comparison_by_seed.csv", index=False)

    plot_metric(summary, "initiation_time_s_r2", "Held-out initiation-time R2",
                figures / "initiation_r2.png")
    plot_metric(summary, "movement_time_s_r2", "Held-out movement-time R2",
                figures / "movement_time_r2.png")
    plot_metric(summary, "fingerprint_balanced_accuracy", "Held-out subject identification",
                figures / "fingerprint_identification.png")
    plot_metric(summary, "mean_ks", "Mean generated-distribution KS statistic",
                figures / "distribution_ks.png", lower_better=True)
    plot_metric(summary, "movement_reconstruction_mse_tracker_units2",
                "Movement-region reconstruction MSE",
                figures / "movement_reconstruction.png", lower_better=True)

    payload = {
        "n_trials": 4732,
        "n_subjects": 28,
        "test_subjects": 7,
        "chance_identification": 1 / 7,
        "windows": list(config.WINDOW_MODES),
        "latent_dims": [2, 3, 4, 8],
        "seeds": [42, 43, 44],
        "model_summary": summary.to_dict("records"),
        "interpretation": {
            "primary_low_dimensional": [2, 3],
            "capacity_comparator": 8,
            "window_mse_cross_protocol_warning": (
                "Do not compare raw window MSE across protocols because the target intervals differ."
            ),
            "latent_axis_warning": (
                "Individual VAE axes may rotate, reflect, or swap across seeds."
            ),
        },
    }
    association_path = root / "latent_associations" / "latent_submovement_associations.csv"
    if association_path.exists():
        associations = pd.read_csv(association_path)
        payload["latent_associations"] = {
            "n_rows": len(associations),
            "fdr_significant": int(associations.fdr_reject_0_05.sum()),
        }
    for window_mode in config.WINDOW_MODES:
        generation = root / window_mode / "generation" / "generation_summary.csv"
        if generation.exists():
            payload.setdefault("generated_submovements", {})[window_mode] = pd.read_csv(
                generation
            ).to_dict("records")
    (out / "strategy_comparison_summary.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
