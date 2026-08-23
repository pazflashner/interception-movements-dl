from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline_spline import SplinePCARepresentation
from src.vae_model import ConditionalVAE


def test_deterministic_autoencoder_has_no_sampling_noise():
    model = ConditionalVAE(
        input_dim=12,
        condition_dim=4,
        hidden_dim=8,
        latent_dim=3,
        timing_dim=2,
        encoder_uses_timing=False,
        variational=False,
    )
    x = torch.randn(5, 12)
    c = torch.randn(5, 4)
    mu, logvar = model.encode(x, c)
    assert torch.equal(model.reparameterize(mu, logvar), mu)
    assert torch.all(logvar == -30.0)


def test_unconditional_vae_ignores_condition_in_encoder_and_decoder():
    model = ConditionalVAE(
        input_dim=12,
        condition_dim=4,
        hidden_dim=8,
        latent_dim=3,
        timing_dim=2,
        encoder_uses_timing=False,
        use_condition=False,
    )
    model.eval()
    x = torch.randn(5, 12)
    c1 = torch.randn(5, 4)
    c2 = torch.randn(5, 4)
    mu1, logvar1 = model.encode(x, c1)
    mu2, logvar2 = model.encode(x, c2)
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(logvar1, logvar2)
    y1, t1 = model.decode(mu1, c1)
    y2, t2 = model.decode(mu1, c2)
    assert torch.allclose(y1, y2)
    assert torch.allclose(t1, t2)


def _trial(offset: float, movement_frames: int) -> dict:
    t = np.linspace(0.0, 1.0, 100)
    pos = np.stack([0.05 * np.sin(np.pi * t), offset + t], axis=1)
    return {
        "pos_norm": pos,
        "move_start_idx": 10,
        "move_end_idx": 10 + movement_frames,
        "go_signal_idx": 0,
        "stim_onset_idx": 0,
        "metadata": {"subject": "s", "sp": 1, "side": 1},
    }


def test_shape_only_spline_code_does_not_change_when_only_timing_changes():
    train = [_trial(float(i), 80 + i) for i in range(6)]
    model = SplinePCARepresentation(n_components=2, include_timing=False).fit(train)
    original = _trial(0.25, 100)
    timing_changed = deepcopy(original)
    timing_changed["move_end_idx"] = 220
    assert np.allclose(model.encode([original]), model.encode([timing_changed]))
    _, decoded_timing = model.decode(model.encode([original]))
    assert decoded_timing.shape == (1, 0)


def test_raw_coefficient_spline_round_trip_has_expected_shape():
    train = [_trial(float(i), 80 + i) for i in range(6)]
    model = SplinePCARepresentation(
        n_components=2,
        include_timing=False,
        standardize_coefficients=False,
    ).fit(train)
    codes = model.encode(train[:2])
    trajectories, timing = model.decode(codes)
    assert codes.shape == (2, 2)
    assert trajectories.shape == (2, 100, 2)
    assert timing.shape == (2, 0)
