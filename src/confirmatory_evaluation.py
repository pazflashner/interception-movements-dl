"""Detailed prediction records for confirmatory participant-fold evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.evaluate import reconstruct


def prediction_tables(model, trials: list[dict], norm, device: str):
    """Return per-trial timing and reconstruction records plus explicit metrics."""
    reconstructed, recorded, predicted_timing, recorded_timing = reconstruct(
        model, trials, norm, device
    )
    if predicted_timing.shape[1] != 2:
        raise ValueError("confirmatory model must predict movement and initiation time")

    subjects = [t["metadata"]["subject"] for t in trials]
    trial_ids = [t["metadata"].get("trial_id", str(i)) for i, t in enumerate(trials)]
    trajectory_mse = np.mean((reconstructed - recorded) ** 2, axis=1)
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
    reconstruction = pd.DataFrame(
        {
            "subject": subjects,
            "trial_id": trial_ids,
            "trajectory_mse_tracker_units2": trajectory_mse,
        }
    )
    return timing, reconstruction, summarise_prediction_tables(timing, reconstruction)


def summarise_prediction_tables(
    timing: pd.DataFrame, reconstruction: pd.DataFrame
) -> dict[str, float]:
    """Compute trial-pooled and participant-balanced summaries explicitly."""
    result: dict[str, float] = {
        "trajectory_mse_trial_pooled": float(
            reconstruction["trajectory_mse_tracker_units2"].mean()
        ),
        "trajectory_mse_subject_balanced": float(
            reconstruction.groupby("subject")["trajectory_mse_tracker_units2"].mean().mean()
        ),
    }
    for stem in ("movement_time", "initiation_time"):
        true_col = f"{stem}_true_s"
        pred_col = f"{stem}_pred_s"
        absolute_error = (timing[true_col] - timing[pred_col]).abs()
        result[f"{stem}_r2_trial_pooled"] = float(
            r2_score(timing[true_col], timing[pred_col])
        )
        result[f"{stem}_mae_ms_trial_pooled"] = float(absolute_error.mean() * 1000)

        by_subject = timing.assign(absolute_error=absolute_error).groupby("subject")
        result[f"{stem}_mae_ms_subject_balanced"] = float(
            by_subject.absolute_error.mean().mean() * 1000
        )
        subject_means = by_subject[[true_col, pred_col]].mean()
        result[f"{stem}_r2_subject_means"] = float(
            r2_score(subject_means[true_col], subject_means[pred_col])
        )
    return result
