from src.features import KINEMATIC_FEATURES, kinematic_features_for_dim


def test_two_dimensional_fidelity_omits_constant_z_endpoint():
    features = kinematic_features_for_dim(2)

    assert "end_x" in features
    assert "end_y" in features
    assert "end_z" not in features
    assert len(features) == len(KINEMATIC_FEATURES) - 1


def test_three_dimensional_fidelity_keeps_all_endpoints():
    assert kinematic_features_for_dim(3) == KINEMATIC_FEATURES
