from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.features import compute_trial_features, movement_from_generated_window
from src.preprocessing import preprocess_trial
from src.trajectory_view import select_trial_window


def synthetic_trial(response_text="Too early"):
    n = 145
    frame = np.arange(n)
    go = 48
    move = 82
    y = np.zeros(n)
    phase = np.linspace(0, 1, n - move)
    y[move:] = 13 * (3 * phase**2 - 2 * phase**3)
    marker = np.zeros(n); marker[0] = 5
    return pd.DataFrame({
        "frame": frame, "x": np.zeros(n), "y": y, "z": np.zeros(n),
        "marker": marker, "responseText": response_text, "successful": -3,
        "go_signal_s": go / 240, "arrival_s": (n - 1) / 240,
        "arrival_window_end_s": 0.8, "subject": "subjectX", "condition": 2,
        "sp": 1, "side": 1, "rep": 1, "starting_position_mm": 120,
        "starting_side": "left", "trial_id": "subjectX_li_2_1_1_1",
        "target_speed_screen_s": 0.5,
    })


def test_too_early_is_retained_and_both_windows_exist():
    trial = preprocess_trial(synthetic_trial())
    assert trial["valid"]
    assert trial["pos_movement_norm"].shape == (100, 3)
    assert trial["pos_go_to_arrival_norm"].shape == (100, 3)
    # Waiting is represented as a long near-zero prefix only in go-to-arrival.
    assert np.abs(trial["pos_go_to_arrival_norm"][:20, 1]).max() < 0.1
    assert trial["pos_movement_norm"][20, 1] > 0.1


def test_behavioral_truth_does_not_change_with_encoder_window():
    trial = preprocess_trial(synthetic_trial("Success"))
    movement = select_trial_window(trial, config.WINDOW_MOVEMENT_ONLY)
    full = select_trial_window(trial, config.WINDOW_GO_TO_ARRIVAL)
    a = compute_trial_features(movement)
    b = compute_trial_features(full)
    for key in ("initiation_time_s", "movement_time_s", "path_length", "curvature_index"):
        assert np.isclose(a[key], b[key])


def test_generated_full_window_crops_at_predicted_initiation_fraction():
    t = np.linspace(0, 1, 100)
    onset = 0.4
    y = np.where(t < onset, 0.0, 13 * ((t - onset) / (1 - onset)) ** 2)
    full = np.column_stack([np.zeros(100), y])
    movement = movement_from_generated_window(full, 0.6, 0.4, config.WINDOW_GO_TO_ARRIVAL)
    assert movement.shape == (100, 2)
    assert np.allclose(movement[0], 0.0)
    assert movement[-1, 1] > 12.9
