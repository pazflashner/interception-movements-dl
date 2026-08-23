"""Condition-stratified generation diagnostics for held-out participants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

import config
from scripts.run_corrected_study import context_query_for_trials, training_latent_noise_covariance
from src.evaluate import encode_trials
from src.features import (
    compute_trial_features,
    features_from_generated_window,
    kinematic_features_for_dim,
)
from src.vae_model import encode_trial_condition


def decode_generated(
    model,
    norm,
    z: np.ndarray,
    metadata: list[dict],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode latent samples under explicit task metadata."""
    conditions = np.stack(
        [encode_trial_condition(item, model.condition_dim) for item in metadata]
    ).astype(np.float32)
    tm, ts, _, _ = norm.torch(device)
    with torch.no_grad():
        trajectory_z, timing_z = model.decode(
            torch.as_tensor(z, dtype=torch.float32, device=device),
            torch.as_tensor(conditions, dtype=torch.float32, device=device),
        )
        trajectories = ((trajectory_z * ts + tm).cpu().numpy()).reshape(
            len(z), config.NORMALISED_LENGTH, model.input_dim // config.NORMALISED_LENGTH
        )
        timing = norm.denormalise_timing(timing_z.cpu().numpy())
    return trajectories, timing


def training_feature_scales(train_trials: list[dict], features: list[str]) -> dict[str, float]:
    frame = pd.DataFrame([compute_trial_features(trial) for trial in train_trials])
    return {
        feature: max(float(frame[feature].std(ddof=0)), 1e-8)
        for feature in features
    }


def condition_stratified_records(
    model,
    norm,
    train_trials: list[dict],
    test_trials: list[dict],
    device: str,
    generation_seed: int,
    n_generated: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare generated and empirical query distributions within sp x side."""
    mu, _, _, _ = encode_trials(model, test_trials, norm, device)
    covariance = training_latent_noise_covariance(model, train_trials, norm, device)
    position_dim = model.input_dim // config.NORMALISED_LENGTH
    features = kinematic_features_for_dim(position_dim)
    scales = training_feature_scales(train_trials, features)
    feature_rows: list[dict] = []
    trajectory_rows: list[dict] = []

    for split in context_query_for_trials(test_trials, config.CONTEXT_QUERY_SEED):
        query_trials = [test_trials[i] for i in split.query_indices]
        context_mean = mu[split.context_indices].mean(axis=0)
        rng = np.random.default_rng(
            generation_seed + sum(map(ord, split.subject))
        )
        # The same latent draws are reused for every task stratum. Therefore
        # differences across strata are caused by condition inputs, not by a
        # different Monte Carlo sample.
        z = rng.multivariate_normal(context_mean, covariance, size=n_generated).astype(
            np.float32
        )
        strata: dict[tuple[int, int], list[dict]] = {}
        for trial in query_trials:
            key = (int(trial["metadata"]["sp"]), int(trial["metadata"]["side"]))
            strata.setdefault(key, []).append(trial)

        for (sp, side), empirical_trials in sorted(strata.items()):
            selected = rng.integers(0, len(empirical_trials), size=n_generated)
            metadata = [empirical_trials[index]["metadata"] for index in selected]
            generated_trajectories, generated_timing = decode_generated(
                model, norm, z, metadata, device
            )
            generated_features = pd.DataFrame(
                [
                    features_from_generated_window(
                        generated_trajectories[i],
                        max(float(generated_timing[i, 0]), 1e-3),
                        max(float(generated_timing[i, 1]), 0.0),
                        config.WINDOW_GO_TO_ARRIVAL,
                    )
                    for i in range(n_generated)
                ]
            )
            empirical_features = pd.DataFrame(
                [compute_trial_features(trial) for trial in empirical_trials]
            )
            for feature in features:
                empirical_values = empirical_features[feature].to_numpy(dtype=float)
                generated_values = generated_features[feature].to_numpy(dtype=float)
                empirical_median = float(np.median(empirical_values))
                generated_median = float(np.median(generated_values))
                feature_rows.append(
                    {
                        "subject": split.subject,
                        "sp": sp,
                        "side": side,
                        "n_query": len(empirical_trials),
                        "feature": feature,
                        "empirical_mean": float(np.mean(empirical_values)),
                        "generated_mean": float(np.mean(generated_values)),
                        "empirical_median": empirical_median,
                        "generated_median": generated_median,
                        "median_abs_error": abs(generated_median - empirical_median),
                        "standardized_median_abs_error": abs(
                            generated_median - empirical_median
                        )
                        / scales[feature],
                        "ks_statistic": float(
                            ks_2samp(empirical_values, generated_values).statistic
                        ),
                    }
                )

            empirical_trajectories = np.stack(
                [trial["pos_norm"] for trial in empirical_trials]
            )
            trajectory_rows.append(
                {
                    "subject": split.subject,
                    "sp": sp,
                    "side": side,
                    "n_query": len(empirical_trials),
                    "mean_trajectory_mse": float(
                        np.mean(
                            (
                                empirical_trajectories.mean(axis=0)
                                - generated_trajectories.mean(axis=0)
                            )
                            ** 2
                        )
                    ),
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(trajectory_rows)


def build_condition_contrasts(feature_records: pd.DataFrame) -> pd.DataFrame:
    """Observed and generated sp3-sp1 / right-left median contrasts."""
    index = ["subject", "feature"]
    rows: list[dict] = []
    for (subject, feature), group in feature_records.groupby(index):
        for side, side_group in group.groupby("side"):
            by_sp = side_group.set_index("sp")
            if 1 in by_sp.index and 3 in by_sp.index:
                rows.append(
                    {
                        "subject": subject,
                        "feature": feature,
                        "contrast": "sp3_minus_sp1",
                        "stratum": int(side),
                        "empirical_contrast": float(
                            by_sp.loc[3, "empirical_median"]
                            - by_sp.loc[1, "empirical_median"]
                        ),
                        "generated_contrast": float(
                            by_sp.loc[3, "generated_median"]
                            - by_sp.loc[1, "generated_median"]
                        ),
                    }
                )
        for sp, sp_group in group.groupby("sp"):
            by_side = sp_group.set_index("side")
            if 1 in by_side.index and 2 in by_side.index:
                rows.append(
                    {
                        "subject": subject,
                        "feature": feature,
                        "contrast": "right_minus_left",
                        "stratum": int(sp),
                        "empirical_contrast": float(
                            by_side.loc[2, "empirical_median"]
                            - by_side.loc[1, "empirical_median"]
                        ),
                        "generated_contrast": float(
                            by_side.loc[2, "generated_median"]
                            - by_side.loc[1, "generated_median"]
                        ),
                    }
                )
    return pd.DataFrame(rows)
