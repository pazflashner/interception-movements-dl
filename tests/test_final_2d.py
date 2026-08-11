"""Checks that the final-study neural path consistently uses the x-y plane."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import features_from_arrays
from src.trajectory_view import project_trials_to_table_plane
from src.vae_model import ConditionalVAE, TrajectoryDataset


def example_trial():
    t = np.linspace(0, 1, 100)
    pos = np.column_stack([0.1 * np.sin(np.pi * t), 13 * t, 0.01 * np.cos(np.pi * t)])
    return {
        "pos_norm": pos,
        "move_start_idx": 20,
        "move_end_idx": 80,
        "go_signal_idx": 10,
        "stim_onset_idx": 0,
        "metadata": {
            "sp": 2, "side": 1, "subject": "s", "trial_id": "t",
            "target_speed_screen_s": 0.635,
        },
    }


def test_projection_and_model_shapes():
    trials = project_trials_to_table_plane([example_trial()])
    assert trials[0]["pos_norm"].shape == (100, 2)
    ds = TrajectoryDataset(trials)
    assert ds.trajectories.shape == (1, 200)
    assert ds.conditions.shape == (1, 5)
    model = ConditionalVAE(
        input_dim=200, condition_dim=5, latent_dim=3, hidden_dim=16,
        timing_dim=2, encoder_uses_timing=False,
    )
    x = torch.from_numpy(ds.trajectories)
    c = torch.from_numpy(ds.conditions)
    recon, timing, mu, _, _ = model(x, c)
    assert recon.shape == (1, 200)
    assert timing.shape == (1, 2)
    assert mu.shape == (1, 3)


def test_features_accept_two_dimensions():
    trial = project_trials_to_table_plane([example_trial()])[0]
    features = features_from_arrays(trial["pos_norm"], 0.3, 0.1)
    assert np.isfinite(features["path_length"])
    assert features["end_z"] == 0.0


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)


if __name__ == "__main__":
    main()
