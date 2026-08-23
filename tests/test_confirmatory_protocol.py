from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.confirmatory_protocol import (
    make_participant_folds,
    partition_trials,
    validate_participant_folds,
)
from src.confirmatory_evaluation import summarise_prediction_tables


def test_outer_folds_hold_out_every_participant_once():
    subjects = [f"subject{i:02d}" for i in range(28)]
    folds = make_participant_folds(subjects)
    validate_participant_folds(folds, subjects)
    appearances = Counter(s for fold in folds for s in fold.test_subjects)
    assert appearances == Counter({s: 1 for s in subjects})
    assert all(
        (len(fold.train_subjects), len(fold.validation_subjects), len(fold.test_subjects))
        == (17, 4, 7)
        for fold in folds
    )


def test_outer_folds_are_deterministic():
    subjects = [f"subject{i:02d}" for i in range(28)]
    assert make_participant_folds(subjects) == make_participant_folds(list(reversed(subjects)))


def test_partition_trials_uses_explicit_ids_without_leakage():
    subjects = [f"subject{i:02d}" for i in range(28)]
    folds = make_participant_folds(subjects)
    trials = [
        {"metadata": {"subject": subject, "trial_id": f"{subject}-{trial}"}}
        for subject in subjects
        for trial in range(3)
    ]
    train, validation, test = partition_trials(trials, folds[0])
    split_subjects = [
        {t["metadata"]["subject"] for t in split}
        for split in (train, validation, test)
    ]
    assert not split_subjects[0] & split_subjects[1]
    assert not split_subjects[0] & split_subjects[2]
    assert not split_subjects[1] & split_subjects[2]
    assert [len(split) for split in (train, validation, test)] == [51, 12, 21]


def test_prediction_summary_distinguishes_trial_and_subject_weighting():
    timing = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "b"],
            "movement_time_true_s": [1.0, 1.0, 1.0, 2.0],
            "movement_time_pred_s": [1.0, 1.0, 1.0, 1.0],
            "initiation_time_true_s": [0.1, 0.1, 0.1, 0.5],
            "initiation_time_pred_s": [0.1, 0.1, 0.1, 0.1],
        }
    )
    reconstruction = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "b"],
            "trajectory_mse_tracker_units2": [0.0, 0.0, 0.0, 4.0],
        }
    )
    result = summarise_prediction_tables(timing, reconstruction)
    assert np.isclose(result["trajectory_mse_trial_pooled"], 1.0)
    assert np.isclose(result["trajectory_mse_subject_balanced"], 2.0)
    assert np.isclose(result["movement_time_mae_ms_trial_pooled"], 250.0)
    assert np.isclose(result["movement_time_mae_ms_subject_balanced"], 500.0)
