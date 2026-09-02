import numpy as np
import pandas as pd
import pytest

from scripts.run_corrected_study import mean_ks_statistic
from src.context_query import DistanceReference, distribution_distances
from src.features import kinematic_features_for_dim


def test_constant_count_discrepancy_is_finite_and_not_dropped():
    real = pd.DataFrame({"n_submovements": np.ones(20)})
    fake = pd.DataFrame({"n_submovements": np.full(20, 2)})
    reference = DistanceReference.fit(real, ["n_submovements"])
    out = distribution_distances(real, fake, ["n_submovements"], reference)
    assert reference.scale[0] == 1
    assert out["energy_distance"] == pytest.approx(2)
    assert out["mmd_rbf"] > 0
    assert out["ks_n_submovements"] == 1


def test_reference_geometry_does_not_depend_on_generated_values():
    train = pd.DataFrame({"x": np.arange(20), "n_submovements": np.ones(20)})
    reference = DistanceReference.fit(train, list(train))
    before = reference.to_dict()
    distribution_distances(train, train * 100, list(train), reference)
    assert reference.to_dict() == before
    with pytest.raises(ValueError, match="feature order"):
        distribution_distances(train, train, list(reversed(list(train))), reference)


def test_mean_ks_ignores_inactive_z_and_bookkeeping():
    frame = pd.DataFrame({f"ks_{f}": [0.5] for f in kinematic_features_for_dim(2)})
    frame["ks_end_z"] = 0.0
    frame["ks_features_tested"] = 12
    frame["ks_rejected_fdr"] = 4
    assert mean_ks_statistic(frame) == pytest.approx(0.5)


def test_invalid_distance_samples_fail_explicitly():
    data = pd.DataFrame({"x": [np.nan, np.inf]})
    with pytest.raises(ValueError, match="finite"):
        distribution_distances(data, data, ["x"])


def test_calibration_solves_weighted_absolute_error_not_unweighted_ratio():
    from scripts.analyze_timing_fairness import calibration_factor
    true = np.array([1., 1., 100.])
    predicted = np.array([1., 1., 10.])
    w = np.ones(3)
    factor = calibration_factor(true, predicted, w)
    candidates = np.linspace(0, 15, 301)
    errors = np.array([np.sum(w * np.abs(true-c*predicted)) for c in candidates])
    assert np.sum(w * np.abs(true-factor*predicted)) == pytest.approx(errors.min())
