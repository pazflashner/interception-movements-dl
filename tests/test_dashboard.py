from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.strategy_dashboard import ASSETS, decode, load_model, read_json, run_name


def test_dashboard_assets_are_consistent():
    manifest = read_json(str(ASSETS / "manifest.json"))
    fingerprints = pd.read_csv(ASSETS / "subject_fingerprints.csv")
    empirical = pd.read_csv(ASSETS / "empirical_query_features.csv")
    assert manifest["latent_dimensions"] == [2, 3, 4, 8]
    assert manifest["windows"] == list(config.WINDOW_MODES)
    assert manifest["n_trials"] == 4732
    assert len(manifest["test_subjects"]) == 7
    assert set(empirical.subject) == set(manifest["test_subjects"])
    assert set(fingerprints.latent_dim) == {2, 3, 4, 8}
    assert set(fingerprints.window_mode) == set(config.WINDOW_MODES)


def test_all_dashboard_dimensions_decode_finite_2d_outputs():
    stats = read_json(str(ASSETS / "latent_stats.json"))
    for window_mode in config.WINDOW_MODES:
        for latent_dim in [2, 3, 4, 8]:
            name = run_name(window_mode, latent_dim, 42)
            model, norm = load_model(window_mode, name)
            center = np.asarray(stats[name]["training_center"], dtype=np.float32)
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
