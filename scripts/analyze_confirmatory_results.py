"""Aggregate confirmatory results without selecting favorable folds or seeds."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.context_query import benjamini_hochberg
from src.confirmatory_evaluation import summarise_prediction_tables
from src.features import kinematic_features_for_dim


STUDY = ROOT / "studies" / "final_strategy_evaluation"
RUNS = STUDY / "runs"
OUT = STUDY / "results" / "analysis"
FAMILIES = (
    "cvae",
    "conditional_ae",
    "unconditional_vae",
    "spline_pca",
    "condition_ridge",
)
LOSS_METRICS = (
    "trajectory_mse",
    "movement_time_mae_ms",
    "initiation_time_mae_ms",
    "mean_ks",
    "ks_n_submovements",
    "energy_distance",
    "mmd_rbf",
)
FIDELITY_FEATURES = tuple(kinematic_features_for_dim(2))


def _fidelity_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    ks_columns = [f"ks_{feature}" for feature in FIDELITY_FEATURES]
    p_columns = [f"ks_p_{feature}" for feature in FIDELITY_FEATURES]
    missing = [column for column in ks_columns + p_columns if column not in frame]
    if missing:
        raise ValueError(f"fidelity table is missing required 2-D columns: {missing}")
    return ks_columns, p_columns


def _active_fdr_rejections(frame: pd.DataFrame, p_columns: list[str]) -> np.ndarray:
    return np.asarray(
        [
            int(benjamini_hochberg(row[p_columns].to_numpy(dtype=float)).sum())
            for _, row in frame.iterrows()
        ],
        dtype=int,
    )


def _run_directories(family: str) -> list[Path]:
    if family == "spline_pca":
        pattern = "fold*/*_z*"
    elif family == "condition_ridge":
        pattern = "fold*/condition_ridge_seed*"
    else:
        pattern = "fold*/*_z*_seed*"
    return sorted((RUNS / family).glob(pattern))


def _metadata(run_dir: Path) -> dict:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    return {
        "model_family": result["model_family"],
        "latent_dim": int(result["latent_dim"] or 0),
        "outer_fold": int(result["outer_fold"]),
        "training_seed": result.get("training_seed") or result.get("generation_seed"),
    }


def _participant_rows(run_dir: Path) -> list[dict]:
    meta = _metadata(run_dir)
    timing = pd.read_csv(run_dir / "timing_predictions.csv")
    reconstruction = pd.read_csv(run_dir / "reconstruction_predictions.csv")
    fidelity = pd.read_csv(run_dir / "context_query_fidelity.csv").set_index("subject")
    timing["movement_abs_error_ms"] = (
        timing.movement_time_true_s - timing.movement_time_pred_s
    ).abs() * 1000
    timing["initiation_abs_error_ms"] = (
        timing.initiation_time_true_s - timing.initiation_time_pred_s
    ).abs() * 1000
    timing_subject = timing.groupby("subject").agg(
        movement_time_mae_ms=("movement_abs_error_ms", "mean"),
        initiation_time_mae_ms=("initiation_abs_error_ms", "mean"),
    )
    reconstruction_subject = reconstruction.groupby("subject").agg(
        trajectory_mse=("trajectory_mse_tracker_units2", "mean")
    )
    ks_columns, p_columns = _fidelity_columns(fidelity)
    fidelity = fidelity.assign(
        mean_ks=fidelity[ks_columns].mean(axis=1),
        active_ks_rejected_fdr=_active_fdr_rejections(fidelity, p_columns),
    )
    joined = reconstruction_subject.join(timing_subject).join(
        fidelity[[
            "mean_ks",
            "ks_n_submovements",
            "energy_distance",
            "mmd_rbf",
            "active_ks_rejected_fdr",
        ]]
    )
    joined = joined.rename(columns={"active_ks_rejected_fdr": "ks_rejected_fdr"})
    return [{**meta, "subject": subject, **row.to_dict()} for subject, row in joined.iterrows()]


def build_participant_metrics() -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for run_dir in _run_directories(family):
            rows.extend(_participant_rows(run_dir))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "participant_metrics_raw.csv", index=False)
    averaged = (
        frame.groupby(["model_family", "latent_dim", "outer_fold", "subject"], as_index=False)
        .agg({metric: "mean" for metric in LOSS_METRICS + ("ks_rejected_fdr",)})
    )
    averaged.to_csv(OUT / "participant_metrics_seed_averaged.csv", index=False)
    return frame


def _oof_for_runs(run_dirs: list[Path], metadata: dict) -> dict:
    timing = pd.concat(
        [pd.read_csv(run_dir / "timing_predictions.csv") for run_dir in run_dirs],
        ignore_index=True,
    )
    reconstruction = pd.concat(
        [pd.read_csv(run_dir / "reconstruction_predictions.csv") for run_dir in run_dirs],
        ignore_index=True,
    )
    duplicated = timing.duplicated(["subject", "trial_id"]).any()
    if duplicated:
        raise ValueError(f"OOF prediction contains duplicated trials: {metadata}")
    summary = summarise_prediction_tables(timing, reconstruction)
    fidelity = pd.concat(
        [pd.read_csv(run_dir / "context_query_fidelity.csv") for run_dir in run_dirs],
        ignore_index=True,
    )
    if fidelity.subject.nunique() != 28 or len(fidelity) != 28:
        raise ValueError(f"OOF fidelity does not cover 28 participants once: {metadata}")
    ks_columns, p_columns = _fidelity_columns(fidelity)
    active_rejections = _active_fdr_rejections(fidelity, p_columns)
    return {
        **metadata,
        **summary,
        "n_oof_trials": len(timing),
        "n_oof_subjects": timing.subject.nunique(),
        "mean_ks": float(fidelity[ks_columns].mean().mean()),
        "mean_ks_n_submovements": float(fidelity.ks_n_submovements.mean()),
        "mean_ks_rejected_fdr": float(active_rejections.mean()),
        "ks_features_tested": len(ks_columns),
        "median_energy_distance": float(fidelity.energy_distance.median()),
        "mean_mmd_rbf": float(fidelity.mmd_rbf.mean()),
    }


def build_oof_metrics() -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        directories = _run_directories(family)
        metadata = pd.DataFrame([_metadata(path) | {"path": path} for path in directories])
        group_columns = ["model_family", "latent_dim"]
        if family != "spline_pca":
            group_columns.append("training_seed")
        for keys, group in metadata.groupby(group_columns, dropna=False):
            if len(group) != 4:
                raise ValueError(f"expected four outer folds for {keys}; found {len(group)}")
            values = keys if isinstance(keys, tuple) else (keys,)
            meta = dict(zip(group_columns, values))
            rows.append(_oof_for_runs(group.path.tolist(), meta))
    frame = pd.DataFrame(rows).sort_values(
        ["model_family", "latent_dim", "training_seed"], na_position="last"
    )
    frame.to_csv(OUT / "oof_metrics_by_seed.csv", index=False)
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"latent_dim", "training_seed", "n_oof_trials", "n_oof_subjects"}
    ]
    summary_rows = []
    for (family, latent_dim), group in frame.groupby(["model_family", "latent_dim"]):
        row = {"model_family": family, "latent_dim": latent_dim, "n_seeds": len(group)}
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd_across_seeds"] = float(
                group[column].std(ddof=1) if len(group) > 1 else 0.0
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "oof_metrics_summary.csv", index=False)
    return frame


def build_feature_fidelity() -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for run_dir in _run_directories(family):
            meta = _metadata(run_dir)
            fidelity = pd.read_csv(run_dir / "context_query_fidelity.csv")
            ks_columns, _ = _fidelity_columns(fidelity)
            for column in ks_columns:
                    rows.append(
                        {
                            **meta,
                            "feature": column.removeprefix("ks_"),
                            "mean_ks": float(fidelity[column].mean()),
                            "median_ks": float(fidelity[column].median()),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "feature_fidelity_raw.csv", index=False)
    summary = (
        frame.groupby(["model_family", "latent_dim", "feature"], as_index=False)
        .agg(mean_ks=("mean_ks", "mean"), sd_run_ks=("mean_ks", "std"))
    )
    summary.to_csv(OUT / "feature_fidelity_summary.csv", index=False)
    return frame


def build_timing_outlier_audit() -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        directories = _run_directories(family)
        metadata = pd.DataFrame([_metadata(path) | {"path": path} for path in directories])
        group_columns = ["model_family", "latent_dim", "training_seed"]
        for keys, group in metadata.groupby(group_columns, dropna=False):
            timing = pd.concat(
                [pd.read_csv(path / "timing_predictions.csv") for path in group.path],
                ignore_index=True,
            )
            if timing.duplicated(["subject", "trial_id"]).any():
                raise ValueError(f"duplicated OOF timing predictions for {keys}")
            row = dict(zip(group_columns, keys))
            for stem, plausibility_limit in (
                ("movement_time", config.MAX_MOVEMENT_TIME_S),
                ("initiation_time", config.MAX_INITIATION_TIME_S),
            ):
                true = timing[f"{stem}_true_s"].to_numpy(dtype=float)
                predicted = timing[f"{stem}_pred_s"].to_numpy(dtype=float)
                error_ms = np.abs(predicted - true) * 1000
                row.update(
                    {
                        f"{stem}_median_abs_error_ms": float(np.median(error_ms)),
                        f"{stem}_p95_abs_error_ms": float(np.quantile(error_ms, 0.95)),
                        f"{stem}_max_abs_error_ms": float(np.max(error_ms)),
                        f"{stem}_predictions_above_plausibility_limit": int(
                            np.sum(predicted > plausibility_limit)
                        ),
                        f"{stem}_plausibility_limit_s": plausibility_limit,
                    }
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "timing_outlier_audit.csv", index=False)
    return frame


def build_variance_decomposition() -> pd.DataFrame:
    frames = []
    metrics = [
        "trajectory_mse_trial_pooled",
        "movement_time_r2_trial_pooled",
        "initiation_time_r2_trial_pooled",
        "fingerprint_balanced_accuracy",
        "mean_ks",
        "median_energy_distance",
    ]
    for family in ("cvae", "conditional_ae", "unconditional_vae"):
        source = pd.read_csv(STUDY / "results" / f"{family}_all_runs.csv")
        for latent_dim, group in source.groupby("latent_dim"):
            for metric in metrics:
                fold_means = group.groupby("outer_fold")[metric].mean()
                seed_sds = group.groupby("outer_fold")[metric].std(ddof=1)
                frames.append(
                    {
                        "model_family": family,
                        "latent_dim": latent_dim,
                        "metric": metric,
                        "grand_mean": float(group[metric].mean()),
                        "sd_across_fold_means": float(fold_means.std(ddof=1)),
                        "mean_within_fold_seed_sd": float(seed_sds.mean()),
                    }
                )
    frame = pd.DataFrame(frames)
    frame.to_csv(OUT / "fold_vs_seed_variability.csv", index=False)
    return frame


def build_paired_comparisons(participant_raw: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        participant_raw.groupby(
            ["model_family", "latent_dim", "outer_fold", "subject"], as_index=False
        )
        .agg({metric: "mean" for metric in LOSS_METRICS})
    )
    rows = []
    for latent_dim in (3, 8):
        primary = averaged[
            (averaged.model_family == "cvae") & (averaged.latent_dim == latent_dim)
        ].set_index("subject")
        for comparator in (
            "conditional_ae",
            "unconditional_vae",
            "spline_pca",
            "condition_ridge",
        ):
            comparator_dim = 0 if comparator == "condition_ridge" else latent_dim
            other = averaged[
                (averaged.model_family == comparator)
                & (averaged.latent_dim == comparator_dim)
            ].set_index("subject")
            common = primary.index.intersection(other.index)
            if len(common) != 28:
                raise ValueError(
                    f"paired comparison lacks 28 participants: z={latent_dim}, {comparator}"
                )
            for metric in LOSS_METRICS:
                cvae_values = primary.loc[common, metric].to_numpy(dtype=float)
                comparator_values = other.loc[common, metric].to_numpy(dtype=float)
                difference = comparator_values - cvae_values
                try:
                    statistic, p_value = wilcoxon(difference, zero_method="pratt")
                except ValueError:
                    statistic, p_value = np.nan, 1.0
                rows.append(
                    {
                        "latent_dim": latent_dim,
                        "comparator": comparator,
                        "metric": metric,
                        "n_participants": len(common),
                        "cvae_mean": float(cvae_values.mean()),
                        "comparator_mean": float(comparator_values.mean()),
                        "cvae_median": float(np.median(cvae_values)),
                        "comparator_median": float(np.median(comparator_values)),
                        "comparator_minus_cvae_mean": float(difference.mean()),
                        "comparator_minus_cvae_median": float(np.median(difference)),
                        "cvae_better_participants": int(np.sum(difference > 0)),
                        "wilcoxon_statistic": float(statistic),
                        "wilcoxon_p_uncorrected": float(p_value),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["wilcoxon_p_fdr_bh"] = np.nan
    valid = frame.wilcoxon_p_uncorrected.notna()
    pvalues = frame.loc[valid, "wilcoxon_p_uncorrected"].to_numpy(dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    frame.loc[valid, "wilcoxon_p_fdr_bh"] = adjusted
    frame["significant_fdr_0_05"] = frame.wilcoxon_p_fdr_bh < 0.05
    frame.to_csv(OUT / "paired_model_comparisons.csv", index=False)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    participant = build_participant_metrics()
    oof = build_oof_metrics()
    feature = build_feature_fidelity()
    timing_outliers = build_timing_outlier_audit()
    variability = build_variance_decomposition()
    paired = build_paired_comparisons(participant)
    manifest = {
        "participant_rows": len(participant),
        "oof_model_seed_rows": len(oof),
        "feature_rows": len(feature),
        "timing_outlier_rows": len(timing_outliers),
        "variability_rows": len(variability),
        "paired_rows": len(paired),
        "principle": "all folds and seeds retained; neural seeds averaged only after participant pairing",
    }
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
