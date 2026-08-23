"""Low-capacity controls for the frozen strategy-window evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import config
from scripts.run_corrected_study import (
    context_query_for_trials,
    finish_fidelity_table,
    mean_ks_statistic,
)
from src.confirmatory_evaluation import summarise_prediction_tables
from src.context_query import distribution_distances
from src.features import (
    compute_trial_features,
    features_from_generated_window,
    kinematic_features_for_dim,
)
from src.vae_model import encode_timing, encode_trial_condition, inverse_timing, transform_timing


RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)


def condition_matrix(trials: list[dict]) -> np.ndarray:
    """Task metadata available to every conditional model."""
    return np.stack([encode_trial_condition(trial["metadata"]) for trial in trials])


def trajectory_matrix(trials: list[dict]) -> np.ndarray:
    return np.stack([trial["pos_norm"].reshape(-1) for trial in trials])


def timing_matrix(trials: list[dict]) -> np.ndarray:
    return np.stack([encode_timing(trial) for trial in trials])


@dataclass
class ConditionRidge:
    """Predict trajectory and timing from task condition, without a fingerprint."""

    x_scaler: StandardScaler
    trajectory_model: Ridge
    timing_model: Ridge
    timing_mean: np.ndarray
    timing_std: np.ndarray
    trajectory_alpha: float
    timing_alpha: float
    train_conditions: np.ndarray
    trajectory_residuals: np.ndarray
    timing_residuals_z: np.ndarray
    position_dim: int

    @classmethod
    def fit(
        cls,
        train_trials: list[dict],
        validation_trials: list[dict],
    ) -> "ConditionRidge":
        x_train_raw = condition_matrix(train_trials)
        x_validation_raw = condition_matrix(validation_trials)
        x_scaler = StandardScaler().fit(x_train_raw)
        x_train = x_scaler.transform(x_train_raw)
        x_validation = x_scaler.transform(x_validation_raw)

        trajectory_train = trajectory_matrix(train_trials)
        trajectory_validation = trajectory_matrix(validation_trials)
        trajectory_alpha = min(
            RIDGE_ALPHAS,
            key=lambda alpha: float(
                np.mean(
                    (
                        Ridge(alpha=alpha).fit(x_train, trajectory_train).predict(x_validation)
                        - trajectory_validation
                    )
                    ** 2
                )
            ),
        )
        trajectory_model = Ridge(alpha=trajectory_alpha).fit(x_train, trajectory_train)

        timing_train = transform_timing(timing_matrix(train_trials), "log")
        timing_validation = transform_timing(timing_matrix(validation_trials), "log")
        timing_mean = timing_train.mean(axis=0)
        timing_std = timing_train.std(axis=0) + 1e-8
        timing_train_z = (timing_train - timing_mean) / timing_std
        timing_validation_z = (timing_validation - timing_mean) / timing_std
        timing_alpha = min(
            RIDGE_ALPHAS,
            key=lambda alpha: float(
                np.mean(
                    (
                        Ridge(alpha=alpha).fit(x_train, timing_train_z).predict(x_validation)
                        - timing_validation_z
                    )
                    ** 2
                )
            ),
        )
        timing_model = Ridge(alpha=timing_alpha).fit(x_train, timing_train_z)

        trajectory_residuals = trajectory_train - trajectory_model.predict(x_train)
        timing_residuals_z = timing_train_z - timing_model.predict(x_train)
        position_dim = train_trials[0]["pos_norm"].shape[1]
        return cls(
            x_scaler=x_scaler,
            trajectory_model=trajectory_model,
            timing_model=timing_model,
            timing_mean=timing_mean,
            timing_std=timing_std,
            trajectory_alpha=trajectory_alpha,
            timing_alpha=timing_alpha,
            train_conditions=x_train_raw,
            trajectory_residuals=trajectory_residuals,
            timing_residuals_z=timing_residuals_z,
            position_dim=position_dim,
        )

    def predict(self, trials: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        conditions = self.x_scaler.transform(condition_matrix(trials))
        trajectories = self.trajectory_model.predict(conditions).reshape(
            len(trials), config.NORMALISED_LENGTH, self.position_dim
        )
        timing_z = self.timing_model.predict(conditions)
        timing = inverse_timing(timing_z * self.timing_std + self.timing_mean, "log")
        return trajectories, timing

    def sample(
        self,
        metadata: list[dict],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Population generation by condition plus a matched training residual."""
        raw_conditions = np.stack([encode_trial_condition(item) for item in metadata])
        conditions = self.x_scaler.transform(raw_conditions)
        trajectories = self.trajectory_model.predict(conditions)
        timing_z = self.timing_model.predict(conditions)

        residual_indices = []
        for condition in raw_conditions:
            # Match the categorical sp one-hot and side. Exact speed remains in
            # the Ridge prediction; requiring an exact continuous match would
            # leave no residual pool.
            matches = np.flatnonzero(
                np.all(self.train_conditions[:, :4] == condition[:4], axis=1)
            )
            pool = matches if len(matches) else np.arange(len(self.train_conditions))
            residual_indices.append(int(rng.choice(pool)))
        residual_indices = np.asarray(residual_indices)
        trajectories = trajectories + self.trajectory_residuals[residual_indices]
        timing_z = timing_z + self.timing_residuals_z[residual_indices]
        timing = inverse_timing(timing_z * self.timing_std + self.timing_mean, "log")
        return (
            trajectories.reshape(len(metadata), config.NORMALISED_LENGTH, self.position_dim),
            timing,
        )


def evaluate_condition_ridge(
    model: ConditionRidge,
    test_trials: list[dict],
    out_dir,
    generation_seed: int,
) -> dict:
    reconstructed, predicted_timing = model.predict(test_trials)
    recorded = np.stack([trial["pos_norm"] for trial in test_trials])
    recorded_timing = timing_matrix(test_trials)
    subjects = [trial["metadata"]["subject"] for trial in test_trials]
    trial_ids = [trial["metadata"].get("trial_id", str(i)) for i, trial in enumerate(test_trials)]
    reconstruction = pd.DataFrame(
        {
            "subject": subjects,
            "trial_id": trial_ids,
            "trajectory_mse_tracker_units2": np.mean(
                (reconstructed - recorded) ** 2, axis=(1, 2)
            ),
        }
    )
    timing = pd.DataFrame(
        {
            "subject": subjects,
            "trial_id": trial_ids,
            "movement_time_true_s": recorded_timing[:, 0],
            "movement_time_pred_s": predicted_timing[:, 0],
            "initiation_time_true_s": recorded_timing[:, 1],
            "initiation_time_pred_s": predicted_timing[:, 1],
        }
    )
    reconstruction.to_csv(out_dir / "reconstruction_predictions.csv", index=False)
    timing.to_csv(out_dir / "timing_predictions.csv", index=False)

    fidelity_rows = []
    features = kinematic_features_for_dim(model.position_dim)
    for split in context_query_for_trials(test_trials, config.CONTEXT_QUERY_SEED):
        query_trials = [test_trials[i] for i in split.query_indices]
        rng = np.random.default_rng(
            generation_seed + sum(map(ord, split.subject))
        )
        selected = rng.integers(0, len(query_trials), size=120)
        metadata = [query_trials[index]["metadata"] for index in selected]
        generated_trajectories, generated_timing = model.sample(metadata, rng)
        generated = pd.DataFrame(
            [
                features_from_generated_window(
                    generated_trajectories[i],
                    max(float(generated_timing[i, 0]), 1e-3),
                    max(float(generated_timing[i, 1]), 0.0),
                    config.WINDOW_GO_TO_ARRIVAL,
                )
                for i in range(len(metadata))
            ]
        )
        empirical = pd.DataFrame([compute_trial_features(trial) for trial in query_trials])
        fidelity_rows.append(
            {
                "subject": split.subject,
                **distribution_distances(empirical, generated, features),
            }
        )
    fidelity = finish_fidelity_table(pd.DataFrame(fidelity_rows))
    fidelity.to_csv(out_dir / "context_query_fidelity.csv", index=False)
    return {
        **summarise_prediction_tables(timing, reconstruction),
        "mean_ks": mean_ks_statistic(fidelity),
        "mean_ks_rejected_fdr": float(fidelity.ks_rejected_fdr.mean()),
        "median_energy_distance": float(fidelity.energy_distance.median()),
        "mean_mmd_rbf": float(fidelity.mmd_rbf.mean()),
        "trajectory_ridge_alpha": model.trajectory_alpha,
        "timing_ridge_alpha": model.timing_alpha,
        "fingerprint_available": False,
    }
