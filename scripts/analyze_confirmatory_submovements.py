"""Aggregate minimum-jerk fidelity with participant as the inferential unit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SOURCE = (
    ROOT
    / "studies"
    / "final_strategy_evaluation"
    / "results"
    / "minimum_jerk"
)
OUT = SOURCE / "analysis"
PRIMARY_METRICS = (
    "count_total_variation",
    "count_jsd",
    "ks_mj_fit_error",
    "ks_mj_first_duration_s",
    "ks_mj_first_amplitude",
    "ks_mj_secondary_amplitude_fraction",
    "ks_mj_mean_overlap_pct",
)
SENSITIVITY_METRICS = ("count_total_variation_bic", "count_jsd_bic")


def _run_metadata(run_dir: Path) -> dict:
    protocol = json.loads((run_dir / "run_protocol.json").read_text(encoding="utf-8"))
    return {
        "model_family": "cvae",
        "latent_dim": int(protocol["latent_dim"]),
        "outer_fold": int(protocol["outer_fold"]),
        "training_seed": int(protocol["training_seed"]),
        "samples_per_subject": int(protocol["samples_per_subject"]),
    }


def load_participant_fidelity() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(SOURCE.glob("cvae_z*_fold*_seed*")):
        path = run_dir / "participant_fidelity.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"subject", *PRIMARY_METRICS, *SENSITIVITY_METRICS}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        rows.append(frame.assign(**_run_metadata(run_dir)))
    if not rows:
        raise ValueError("no completed minimum-jerk runs found")
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(OUT / "participant_fidelity_raw.csv", index=False)
    return combined


def build_oof_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = PRIMARY_METRICS + SENSITIVITY_METRICS
    for (latent_dim, seed), group in frame.groupby(["latent_dim", "training_seed"]):
        if len(group) != 28 or group.subject.nunique() != 28:
            raise ValueError(
                f"n={latent_dim}, seed={seed} does not cover 28 participants exactly once"
            )
        row = {
            "latent_dim": latent_dim,
            "training_seed": seed,
            "n_participants": 28,
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"median_{metric}"] = float(group[metric].median())
        rows.append(row)
    by_seed = pd.DataFrame(rows).sort_values(["latent_dim", "training_seed"])
    by_seed.to_csv(OUT / "oof_by_seed.csv", index=False)
    summary_rows = []
    for latent_dim, group in by_seed.groupby("latent_dim"):
        row = {"latent_dim": latent_dim, "n_seeds": len(group)}
        for column in [c for c in by_seed if c.startswith("mean_")]:
            row[f"{column}_across_seeds"] = float(group[column].mean())
            row[f"{column}_sd_across_seeds"] = float(group[column].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "oof_summary.csv", index=False)
    return by_seed


def build_paired_dimension_test(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = PRIMARY_METRICS + SENSITIVITY_METRICS
    participant = (
        frame.groupby(["latent_dim", "subject"], as_index=False)[list(metrics)].mean()
    )
    n3 = participant[participant.latent_dim == 3].set_index("subject")
    n8 = participant[participant.latent_dim == 8].set_index("subject")
    common = n3.index.intersection(n8.index)
    if len(common) != 28:
        raise ValueError("n=3 versus n=8 comparison lacks 28 paired participants")
    rows = []
    for metric in metrics:
        low = n3.loc[common, metric].to_numpy(dtype=float)
        high = n8.loc[common, metric].to_numpy(dtype=float)
        difference = low - high  # positive means n=8 has lower loss
        if np.allclose(difference, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(difference, zero_method="pratt")
        rows.append(
            {
                "metric": metric,
                "n_participants": len(common),
                "n3_median": float(np.median(low)),
                "n8_median": float(np.median(high)),
                "n3_minus_n8_median": float(np.median(difference)),
                "n8_better_participants": int(np.sum(difference > 0)),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p_uncorrected": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    p = result.wilcoxon_p_uncorrected.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    corrected_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    corrected = np.empty_like(corrected_ranked)
    corrected[order] = np.clip(corrected_ranked, 0.0, 1.0)
    result["wilcoxon_p_fdr_bh"] = corrected
    result.to_csv(OUT / "n3_vs_n8_paired.csv", index=False)
    return result


def build_fit_success() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(SOURCE.glob("cvae_z*_fold*_seed*")):
        generated_path = run_dir / "generated_submovements.csv"
        if not generated_path.exists():
            continue
        frame = pd.read_csv(generated_path)
        success = frame.mj_fit_success
        if success.dtype != bool:
            success = success.astype(str).str.lower().eq("true")
        rows.append(
            {
                **_run_metadata(run_dir),
                "n_generated": len(frame),
                "fit_success_rate": float(success.mean()),
                "median_fit_error": float(
                    frame.loc[success, "mj_fit_error"].median()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "generated_fit_success.csv", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    participant = load_participant_fidelity()
    oof = build_oof_summary(participant)
    paired = build_paired_dimension_test(participant)
    fit = build_fit_success()
    manifest = {
        "participant_rows": len(participant),
        "oof_seed_rows": len(oof),
        "paired_metrics": len(paired),
        "completed_runs": len(fit),
        "inferential_unit": "held-out participant after averaging training seeds",
        "primary_count_rule": "error-threshold selection",
        "sensitivity_count_rule": "minimum BIC",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
