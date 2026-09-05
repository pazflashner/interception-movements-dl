from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
# Exercise the current shipped dashboard. The retired two-window explorer's
# private assets are not part of a clone or the current release.
from src.confirmatory_dashboard import ASSETS, decode, load_model, read_json


def test_dashboard_assets_are_consistent():
    manifest = read_json(str(ASSETS / "manifest.json"))
    fingerprints = pd.read_csv(ASSETS / "subject_fingerprints.csv")
    empirical = pd.read_csv(ASSETS / "empirical_query_features.csv")
    assert manifest["latent_dimensions"] == [2, 3, 4, 8]
    assert manifest["window_mode"] == config.WINDOW_GO_TO_ARRIVAL
    assert manifest["n_trials"] == 4732
    assert len(manifest["live_test_subjects"]) == 7
    assert set(empirical.subject) == set(manifest["all_validation_subjects"])
    assert set(fingerprints.latent_dim) == {2, 3, 4, 8}
    for dim, frame in fingerprints.groupby("latent_dim"):
        assert set(frame.subject) == set(manifest["live_test_subjects"])
        assert frame.n_context.gt(0).all() and frame.n_query.gt(0).all()
        assert np.isfinite(frame[[f"z{i+1}" for i in range(dim)]]).all().all()


def test_all_dashboard_dimensions_decode_finite_2d_outputs():
    fingerprints = pd.read_csv(ASSETS / "subject_fingerprints.csv")
    for latent_dim in [2, 3, 4, 8]:
        model, norm = load_model(latent_dim)
        for _, row in fingerprints[fingerprints.latent_dim == latent_dim].iterrows():
            center = row[[f"z{i+1}" for i in range(latent_dim)]].to_numpy(dtype=np.float32)
            trajectory, timing = decode(model, norm, center, 2, 1, 0.636)
            assert trajectory.shape == (1, 100, 2)
            assert timing.shape == (1, 2)
            assert np.isfinite(trajectory).all()
            assert np.isfinite(timing).all()
            assert (timing >= 0).all()


if __name__ == "__main__":
    tests = [
        test_dashboard_assets_are_consistent,
        test_all_dashboard_dimensions_decode_finite_2d_outputs,
    ]
    for test in tests:
        test()
        print("ok", test.__name__)
