import numpy as np
import pytest

from scripts.run_review_controls import prepare_matched_movement, component_job


def test_common_operator_preserves_physical_duration():
    window = np.column_stack((np.linspace(0, 1, 100), np.linspace(0, 2, 100)))
    movement, hz = prepare_matched_movement(window, 6.0, 1.0)
    assert movement.shape == (100, 2)
    assert (len(movement) - 1) / hz == 6.0
    np.testing.assert_allclose(movement[0], 0)
    np.testing.assert_allclose(movement[-1], [6/7, 12/7])
    with pytest.raises(ValueError):
        prepare_matched_movement(window, 0.0, 1.0)


def test_identical_recorded_and_generated_inputs_have_identical_fits():
    phase = np.linspace(0, 1, 100)
    shape = 10*phase**3 - 15*phase**4 + 6*phase**5
    window = np.column_stack((shape*.3, shape*12))
    common = {"fit_seed_id": "identical", "movement_time_s": .4, "initiation_time_s": .0}
    a = component_job(({**common, "kind": "recorded"}, window, 1, 100))
    b = component_job(({**common, "kind": "generated"}, window, 1, 100))
    assert a["mj_fit_completed"] and b["mj_fit_completed"]
    for key in ("mj_parameters_json", "mj_n_components", "mj_fit_error", "mj_selected_optimizer_converged"):
        assert a[key] == b[key]
