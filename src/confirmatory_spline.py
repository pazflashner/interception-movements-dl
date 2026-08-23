"""Leakage-resistant spline+PCA comparison for the confirmatory protocol."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import config
from scripts.run_corrected_study import context_query_for_trials, finish_fidelity_table, mean_ks_statistic
from src.baseline_spline import SplinePCARepresentation
from src.confirmatory_evaluation import summarise_prediction_tables
from src.context_query import distribution_distances, fingerprint_identification, subject_summary, tune_and_test_ridge
from src.features import compute_trial_features, features_from_generated_window, kinematic_features_for_dim, movement_from_generated_window
from src.vae_model import encode_timing, encode_trial_condition, inverse_timing, transform_timing


RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _conditions(trials: list[dict]) -> np.ndarray:
    return np.stack([encode_trial_condition(t["metadata"]) for t in trials])


@dataclass
class TimingRidge:
    x_scaler: StandardScaler
    y_mean: np.ndarray
    y_std: np.ndarray
    model: Ridge
    alpha: float

    @classmethod
    def fit(
        cls,
        representation: SplinePCARepresentation,
        train_trials: list[dict],
        validation_trials: list[dict],
    ) -> "TimingRidge":
        train_codes = representation.encode(train_trials)
        validation_codes = representation.encode(validation_trials)
        x_train = np.hstack([train_codes, _conditions(train_trials)])
        x_validation = np.hstack([validation_codes, _conditions(validation_trials)])
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_validation = scaler.transform(x_validation)
        y_train = transform_timing(np.stack([encode_timing(t) for t in train_trials]), "log")
        y_validation = transform_timing(
            np.stack([encode_timing(t) for t in validation_trials]), "log"
        )
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0) + 1e-8
        y_train_z = (y_train - y_mean) / y_std
        y_validation_z = (y_validation - y_mean) / y_std
        best_alpha = min(
            RIDGE_ALPHAS,
            key=lambda alpha: np.mean(
                (
                    Ridge(alpha=alpha).fit(x_train, y_train_z).predict(x_validation)
                    - y_validation_z
                )
                ** 2
            ),
        )
        model = Ridge(alpha=best_alpha).fit(x_train, y_train_z)
        return cls(scaler, y_mean, y_std, model, best_alpha)

    def predict(self, codes: np.ndarray, conditions: np.ndarray) -> np.ndarray:
        x = self.x_scaler.transform(np.hstack([codes, conditions]))
        transformed = self.model.predict(x) * self.y_std + self.y_mean
        return inverse_timing(transformed, "log")


def fingerprint_tables(
    representation: SplinePCARepresentation,
    trials: list[dict],
    seed: int,
):
    codes = representation.encode(trials)
    rows: list[dict] = []
    query_features: list[dict] = []
    context_codes: dict[str, np.ndarray] = {}
    query_codes: dict[str, np.ndarray] = {}
    for split in context_query_for_trials(trials, seed):
        context = codes[split.context_indices]
        query = codes[split.query_indices]
        rows.append(
            {
                "subject": split.subject,
                **{f"z{i}_mean": value for i, value in enumerate(context.mean(axis=0))},
            }
        )
        context_codes[split.subject] = context
        query_codes[split.subject] = query
        query_features.extend(compute_trial_features(trials[i]) for i in split.query_indices)
    fingerprints = pd.DataFrame(rows).set_index("subject").sort_index()
    summaries = subject_summary(pd.DataFrame(query_features)).sort_index()
    return fingerprints, summaries, context_codes, query_codes


def _training_noise_covariance(codes: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    residuals = np.empty_like(codes)
    for subject in np.unique(subjects):
        mask = subjects == subject
        residuals[mask] = codes[mask] - codes[mask].mean(axis=0)
    covariance = np.atleast_2d(np.cov(residuals, rowvar=False))
    return covariance + np.eye(codes.shape[1]) * 1e-6


def evaluate_spline_run(
    representation: SplinePCARepresentation,
    timing_model: TimingRidge,
    train_trials: list[dict],
    validation_trials: list[dict],
    test_trials: list[dict],
    out_dir,
    seed: int,
) -> dict:
    codes = representation.encode(test_trials)
    reconstructed, _ = representation.decode(codes)
    recorded = np.stack([t["pos_norm"] for t in test_trials])
    predicted_timing = timing_model.predict(codes, _conditions(test_trials))
    recorded_timing = np.stack([encode_timing(t) for t in test_trials])
    subjects = [t["metadata"]["subject"] for t in test_trials]
    trial_ids = [t["metadata"].get("trial_id", str(i)) for i, t in enumerate(test_trials)]
    trajectory_mse = np.mean((reconstructed - recorded) ** 2, axis=(1, 2))
    timing_frame = pd.DataFrame(
        {
            "subject": subjects,
            "trial_id": trial_ids,
            "movement_time_true_s": recorded_timing[:, 0],
            "movement_time_pred_s": predicted_timing[:, 0],
            "initiation_time_true_s": recorded_timing[:, 1],
            "initiation_time_pred_s": predicted_timing[:, 1],
        }
    )
    reconstruction_frame = pd.DataFrame(
        {
            "subject": subjects,
            "trial_id": trial_ids,
            "trajectory_mse_tracker_units2": trajectory_mse,
        }
    )
    timing_frame.to_csv(out_dir / "timing_predictions.csv", index=False)
    reconstruction_frame.to_csv(out_dir / "reconstruction_predictions.csv", index=False)
    detailed = summarise_prediction_tables(timing_frame, reconstruction_frame)

    movement_errors = []
    for i, trajectory in enumerate(reconstructed):
        recovered = movement_from_generated_window(
            trajectory,
            recorded_timing[i, 0],
            recorded_timing[i, 1],
            config.WINDOW_GO_TO_ARRIVAL,
        )
        movement_errors.append(np.mean((recovered - test_trials[i]["pos_movement_norm"]) ** 2))

    tables = {
        name: fingerprint_tables(representation, trials, seed)
        for name, trials in (
            ("train", train_trials),
            ("validation", validation_trials),
            ("test", test_trials),
        )
    }
    probes = tune_and_test_ridge(
        tables["train"][0],
        tables["train"][1],
        tables["validation"][0],
        tables["validation"][1],
        tables["test"][0],
        tables["test"][1],
    )
    probes.to_csv(out_dir / "behavioral_probe.csv", index=False)
    identification = fingerprint_identification(tables["test"][2], tables["test"][3])

    train_codes = representation.encode(train_trials)
    train_subjects = np.asarray([t["metadata"]["subject"] for t in train_trials])
    shared_covariance = _training_noise_covariance(train_codes, train_subjects)
    fidelity_rows = []
    for split in context_query_for_trials(test_trials, seed):
        context_codes = codes[split.context_indices]
        query_trials = [test_trials[i] for i in split.query_indices]
        rng = np.random.default_rng(seed + sum(map(ord, split.subject)))
        generated_codes = rng.multivariate_normal(
            context_codes.mean(axis=0), shared_covariance, size=120
        )
        chosen = rng.integers(0, len(query_trials), size=120)
        generated_conditions = np.stack(
            [encode_trial_condition(query_trials[i]["metadata"]) for i in chosen]
        )
        generated_trajectories, _ = representation.decode(generated_codes)
        generated_timing = timing_model.predict(generated_codes, generated_conditions)
        generated_features = pd.DataFrame(
            [
                features_from_generated_window(
                    generated_trajectories[i],
                    max(float(generated_timing[i, 0]), 1e-3),
                    max(float(generated_timing[i, 1]), 0.0),
                    config.WINDOW_GO_TO_ARRIVAL,
                )
                for i in range(120)
            ]
        )
        empirical = pd.DataFrame([compute_trial_features(t) for t in query_trials])
        position_dim = generated_trajectories.shape[-1]
        fidelity_rows.append(
            {
                "subject": split.subject,
                **distribution_distances(
                    empirical,
                    generated_features,
                    kinematic_features_for_dim(position_dim),
                ),
            }
        )
    fidelity = finish_fidelity_table(pd.DataFrame(fidelity_rows))
    fidelity.to_csv(out_dir / "context_query_fidelity.csv", index=False)

    common = {
        "reconstruction_mse_tracker_units2": float(trajectory_mse.mean()),
        "window_reconstruction_mse_tracker_units2": float(trajectory_mse.mean()),
        "movement_reconstruction_mse_tracker_units2": float(np.mean(movement_errors)),
        "movement_time_s_r2": float(r2_score(recorded_timing[:, 0], predicted_timing[:, 0])),
        "movement_time_s_mae_ms": float(
            np.mean(np.abs(recorded_timing[:, 0] - predicted_timing[:, 0])) * 1000
        ),
        "initiation_time_s_r2": float(
            r2_score(recorded_timing[:, 1], predicted_timing[:, 1])
        ),
        "initiation_time_s_mae_ms": float(
            np.mean(np.abs(recorded_timing[:, 1] - predicted_timing[:, 1])) * 1000
        ),
        "timing_metric_type": "ridge_prediction_from_spline_shape_and_condition",
        **{f"fingerprint_{key}": value for key, value in identification.items()},
        "probe_positive_r2": int((probes.r2_test > 0).sum()),
        "probe_targets": int(len(probes)),
        "mean_ks": mean_ks_statistic(fidelity),
        "mean_ks_rejected_fdr": float(fidelity.ks_rejected_fdr.mean()),
        "median_energy_distance": float(fidelity.energy_distance.median()),
        "mean_mmd_rbf": float(fidelity.mmd_rbf.mean()),
        "timing_ridge_alpha": timing_model.alpha,
    }
    return {**common, **detailed}
