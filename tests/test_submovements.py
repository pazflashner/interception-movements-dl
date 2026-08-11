"""Deterministic checks for the minimum-jerk decomposition."""
from pathlib import Path
import importlib.util
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submovements import (
    SubmovementConfig,
    decompose_position,
    minimum_jerk_velocity,
)


def test_profile_matches_jason_repository():
    source = ROOT / "external" / "jason-submovements" / "python" / "movement_decompose_2d.py"
    spec = importlib.util.spec_from_file_location("jason_submovements", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    t = np.linspace(0, 0.8, 193)
    ours = minimum_jerk_velocity(t, 0.12, 0.31, np.array([-0.4, 2.7]))
    vx, vy, _ = module._minimum_jerk_velocity_2D(0.12, 0.31, -0.4, 2.7, t)
    np.testing.assert_allclose(ours[:, 0], vx, atol=1e-10)
    np.testing.assert_allclose(ours[:, 1], vy, atol=1e-10)


def test_recovers_single_component():
    fs = 240.0
    t = np.arange(0, 0.55, 1 / fs)
    vel = minimum_jerk_velocity(t, 0.04, 0.32, np.array([0.3, 12.0]))
    pos = np.cumsum(vel, axis=0) / fs
    cfg = SubmovementConfig(restarts=2, max_nfev=500)
    result = decompose_position(pos, cfg, "synthetic-single")
    assert result.selected.n_components == 1
    assert result.selected.normalized_error < 0.05


def test_recovers_two_overlapping_components():
    fs = 240.0
    t = np.arange(0, 0.72, 1 / fs)
    vel = (
        minimum_jerk_velocity(t, 0.02, 0.30, np.array([0.1, 7.0]))
        + minimum_jerk_velocity(t, 0.18, 0.28, np.array([-0.2, 5.0]))
    )
    pos = np.cumsum(vel, axis=0) / fs
    cfg = SubmovementConfig(restarts=3, max_nfev=700)
    result = decompose_position(pos, cfg, "synthetic-double")
    fit = result.fits[2]
    assert fit.normalized_error < 0.001
    np.testing.assert_allclose(fit.parameters[:, 0], [0.02, 0.18], atol=0.02)
    np.testing.assert_allclose(fit.parameters[:, 1], [0.30, 0.28], atol=0.03)


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)


if __name__ == "__main__":
    main()
