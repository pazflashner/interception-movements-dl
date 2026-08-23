from copy import deepcopy

import numpy as np

from src.confirmatory_controls import ConditionRidge, condition_matrix


def _trial(subject: str, sp: int, side: int, offset: float) -> dict:
    phase = np.linspace(0.0, 1.0, 100)
    pos = np.column_stack([0.05 * side * phase, (1.0 + 0.2 * sp) * phase + offset])
    return {
        "pos_norm": pos,
        "pos_movement_norm": pos,
        "move_start_idx": 20,
        "move_end_idx": 100 + sp,
        "go_signal_idx": 10,
        "stim_onset_idx": 0,
        "metadata": {
            "subject": subject,
            "trial_id": f"{subject}_{sp}_{side}_{offset}",
            "sp": sp,
            "side": side,
            "target_speed_screen_s": 0.45 + 0.1 * sp,
        },
    }


def test_condition_ridge_uses_no_subject_identifier():
    first = _trial("a", 2, 1, 0.0)
    second = deepcopy(first)
    second["metadata"]["subject"] = "different"

    assert np.array_equal(condition_matrix([first]), condition_matrix([second]))


def test_condition_ridge_fit_predict_and_sample_shapes():
    train = [
        _trial(f"s{i}", sp, side, 0.01 * i)
        for i in range(8)
        for sp in (1, 2, 3)
        for side in (1, 2)
    ]
    validation = [
        _trial(f"v{i}", sp, side, -0.01 * i)
        for i in range(2)
        for sp in (1, 2, 3)
        for side in (1, 2)
    ]
    model = ConditionRidge.fit(train, validation)
    trajectory, timing = model.predict(validation)
    sampled_trajectory, sampled_timing = model.sample(
        [trial["metadata"] for trial in validation], np.random.default_rng(7)
    )

    assert trajectory.shape == sampled_trajectory.shape == (12, 100, 2)
    assert timing.shape == sampled_timing.shape == (12, 2)
    assert np.isfinite(sampled_trajectory).all()
    assert (sampled_timing >= 0).all()
