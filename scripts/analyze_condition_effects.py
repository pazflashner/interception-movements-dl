"""Run held-out condition-value diagnostics for CVAE and unconditional VAE."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.run_confirmatory_cvae import DEFAULT_STUDY
from scripts.run_corrected_study import load_per_trial_checkpoint
from src.condition_effects import build_condition_contrasts, condition_stratified_records
from src.confirmatory_protocol import partition_trials, write_or_verify_manifest
from src.trajectory_view import project_trials_to_table_plane, select_trials_window


FAMILIES = ("cvae", "unconditional_vae")


def _checkpoint(study: Path, family: str, fold: int, n: int, seed: int) -> Path:
    return (
        study
        / "runs"
        / family
        / f"fold{fold}"
        / f"{family}_z{n}_seed{seed}"
        / "checkpoint.pt"
    )


def _paired_summary(feature_frame: pd.DataFrame) -> pd.DataFrame:
    # Participant is the inferential unit. Conditions are repeated measures and
    # seeds are repeated fits, not 18 independent samples per participant.
    participant = (
        feature_frame.groupby(
            ["model_family", "latent_dim", "subject", "feature"], as_index=False
        )
        .agg(
            standardized_median_abs_error=(
                "standardized_median_abs_error",
                "mean",
            ),
            ks_statistic=("ks_statistic", "mean"),
        )
    )
    keys = ["latent_dim", "subject", "feature"]
    cvae = participant[participant.model_family == "cvae"].set_index(keys)
    unconditional = participant[
        participant.model_family == "unconditional_vae"
    ].set_index(keys)
    common = cvae.index.intersection(unconditional.index)
    rows = []
    for latent_dim in sorted(feature_frame.latent_dim.unique()):
        for feature in sorted(feature_frame.feature.unique()):
            index = [
                item for item in common if item[0] == latent_dim and item[-1] == feature
            ]
            for metric in ("standardized_median_abs_error", "ks_statistic"):
                cvae_values = cvae.loc[index, metric].to_numpy(dtype=float)
                other_values = unconditional.loc[index, metric].to_numpy(dtype=float)
                difference = other_values - cvae_values
                if np.allclose(difference, 0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = wilcoxon(difference, zero_method="pratt")
                rows.append(
                    {
                        "latent_dim": latent_dim,
                        "feature": feature,
                        "metric": metric,
                        "n_paired_participants": len(index),
                        "cvae_median": float(np.median(cvae_values)),
                        "unconditional_vae_median": float(np.median(other_values)),
                        "unconditional_minus_cvae_median": float(np.median(difference)),
                        "cvae_better_participants": int(np.sum(difference > 0)),
                        "wilcoxon_statistic": float(statistic),
                        "wilcoxon_p_uncorrected": float(p_value),
                    }
                )
    frame = pd.DataFrame(rows)
    p = frame.wilcoxon_p_uncorrected.to_numpy(dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("paired condition tests produced non-finite p-values")
    order = np.argsort(p)
    ranked = p[order]
    corrected_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    corrected = np.empty_like(corrected_ranked)
    corrected[order] = np.clip(corrected_ranked, 0.0, 1.0)
    frame["wilcoxon_p_fdr_bh"] = corrected
    return frame


def _contrast_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    # Average repeated network seeds first. A subject-condition contrast, not a
    # seed, is the descriptive unit for association with the observed effect.
    contrasts = (
        contrasts.groupby(
            [
                "model_family",
                "latent_dim",
                "contrast",
                "feature",
                "subject",
                "stratum",
            ],
            as_index=False,
        )
        .agg(
            empirical_contrast=("empirical_contrast", "first"),
            generated_contrast=("generated_contrast", "mean"),
        )
    )
    rows = []
    for keys, group in contrasts.groupby(
        ["model_family", "latent_dim", "contrast", "feature"]
    ):
        empirical = group.empirical_contrast.to_numpy(dtype=float)
        generated = group.generated_contrast.to_numpy(dtype=float)
        correlation = (
            np.nan
            if np.std(empirical) < 1e-12 or np.std(generated) < 1e-12
            else spearmanr(empirical, generated).statistic
        )
        sign_mask = np.abs(group.empirical_contrast.to_numpy()) > 1e-10
        sign_agreement = np.mean(
            np.sign(group.loc[sign_mask, "empirical_contrast"])
            == np.sign(group.loc[sign_mask, "generated_contrast"])
        ) if sign_mask.any() else np.nan
        rows.append(
            {
                "model_family": keys[0],
                "latent_dim": keys[1],
                "contrast": keys[2],
                "feature": keys[3],
                "n_contrasts": len(group),
                "spearman_r": float(correlation),
                "sign_agreement": float(sign_agreement),
                "median_absolute_contrast_error": float(
                    np.median(
                        np.abs(
                            group.generated_contrast - group.empirical_contrast
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _trajectory_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    participant = (
        trajectories.groupby(
            ["model_family", "latent_dim", "subject"], as_index=False
        )
        .mean_trajectory_mse.mean()
    )
    rows = []
    for latent_dim in sorted(participant.latent_dim.unique()):
        cvae = participant[
            (participant.model_family == "cvae")
            & (participant.latent_dim == latent_dim)
        ].set_index("subject")
        unconditional = participant[
            (participant.model_family == "unconditional_vae")
            & (participant.latent_dim == latent_dim)
        ].set_index("subject")
        common = cvae.index.intersection(unconditional.index)
        cvae_values = cvae.loc[common, "mean_trajectory_mse"].to_numpy(dtype=float)
        other_values = unconditional.loc[
            common, "mean_trajectory_mse"
        ].to_numpy(dtype=float)
        difference = other_values - cvae_values
        statistic, p_value = wilcoxon(difference, zero_method="pratt")
        rows.append(
            {
                "latent_dim": latent_dim,
                "n_paired_participants": len(common),
                "cvae_median": float(np.median(cvae_values)),
                "unconditional_vae_median": float(np.median(other_values)),
                "unconditional_minus_cvae_median": float(np.median(difference)),
                "cvae_better_participants": int(np.sum(difference > 0)),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p_uncorrected": float(p_value),
            }
        )
    frame = pd.DataFrame(rows)
    frame["wilcoxon_p_holm"] = np.minimum(
        1.0,
        frame.wilcoxon_p_uncorrected * len(frame),
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, default=config.DATA_PROCESSED_DIR / "canonical_trials.pkl")
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--dims", nargs="+", type=int, default=[3, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    out = args.study / "results" / "condition_effects"
    out.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        features = pd.read_csv(out / "condition_strata_features.csv")
        trajectories = pd.read_csv(out / "condition_strata_trajectories.csv")
        contrasts = pd.read_csv(out / "condition_contrasts.csv")
    else:
        with args.trials.open("rb") as handle:
            canonical = pickle.load(handle)
        trials = project_trials_to_table_plane(
            select_trials_window(canonical, config.WINDOW_GO_TO_ARRIVAL)
        )
        folds = write_or_verify_manifest(
            args.study / "protocol" / "participant_folds.json", trials
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feature_frames = []
        trajectory_frames = []
        contrast_frames = []

        for family in FAMILIES:
            for latent_dim in args.dims:
                for fold_index, fold in enumerate(folds):
                    train, _, test = partition_trials(trials, fold)
                    for seed in args.seeds:
                        checkpoint = _checkpoint(
                            args.study, family, fold_index, latent_dim, seed
                        )
                        model, norm = load_per_trial_checkpoint(checkpoint, device)
                        features_run, trajectories_run = condition_stratified_records(
                            model,
                            norm,
                            train,
                            test,
                            device,
                            generation_seed=seed,
                            n_generated=args.samples,
                        )
                        meta = {
                            "model_family": family,
                            "latent_dim": latent_dim,
                            "outer_fold": fold_index,
                            "training_seed": seed,
                        }
                        features_run = features_run.assign(**meta)
                        trajectories_run = trajectories_run.assign(**meta)
                        contrasts_run = build_condition_contrasts(features_run).assign(
                            **meta
                        )
                        feature_frames.append(features_run)
                        trajectory_frames.append(trajectories_run)
                        contrast_frames.append(contrasts_run)
                        print(
                            f"complete {family} n={latent_dim} fold={fold_index} seed={seed}",
                            flush=True,
                        )

        features = pd.concat(feature_frames, ignore_index=True)
        trajectories = pd.concat(trajectory_frames, ignore_index=True)
        contrasts = pd.concat(contrast_frames, ignore_index=True)
    paired = _paired_summary(features)
    contrast_summary = _contrast_summary(contrasts)
    trajectory_summary = _trajectory_summary(trajectories)
    features.to_csv(out / "condition_strata_features.csv", index=False)
    trajectories.to_csv(out / "condition_strata_trajectories.csv", index=False)
    contrasts.to_csv(out / "condition_contrasts.csv", index=False)
    paired.to_csv(out / "cvae_vs_unconditional_paired.csv", index=False)
    contrast_summary.to_csv(out / "condition_contrast_summary.csv", index=False)
    trajectory_summary.to_csv(out / "condition_trajectory_summary.csv", index=False)
    manifest = {
        "protocol_version": "strategy-confirmatory-v1",
        "models": list(FAMILIES),
        "latent_dims": args.dims,
        "seeds": args.seeds,
        "n_generated_per_stratum": args.samples,
        "feature_rows": len(features),
        "trajectory_rows": len(trajectories),
        "contrast_rows": len(contrasts),
        "principle": "fingerprint and latent draws fixed while task condition varies",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
