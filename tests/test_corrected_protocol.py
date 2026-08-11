from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.context_query import moment_matched_posterior, split_context_query
from src.data_loading import object_motion_onset_s
from src.hierarchical_vae import HierarchicalCVAE, hierarchical_loss
from src.features import features_from_arrays
from src.preprocessing import regularise_frame_grid
from src.vae_model import ConditionalVAE, encode_condition, inverse_timing, transform_timing


def test_frame_counter_regularisation_handles_duplicate_and_gap():
    df = pd.DataFrame({
        "frame": [10, 11, 11, 13],
        "x": [0.0, 1.0, 3.0, 6.0],
        "y": [0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0],
        "marker": [5, 0, 0, 0],
    })
    pos, marker, frames, quality = regularise_frame_grid(df)
    assert frames.tolist() == [10, 11, 12, 13]
    assert np.allclose(pos[:, 0], [0.0, 2.0, 4.0, 6.0])
    assert marker.tolist() == [5.0, 0.0, 0.0, 0.0]
    assert quality["n_duplicate_rows"] == 1
    assert quality["n_interpolated_frames"] == 1


def test_target_motion_onset_uses_post_difference_sample():
    target = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    assert np.isclose(object_motion_onset_s(target, stimulus_hz=60), 2 / 60)


def test_log_timing_roundtrip_is_nonnegative():
    values = np.array([[0.0, 0.2], [0.3, 1.0]], dtype=np.float32)
    restored = inverse_timing(transform_timing(values, "log"), "log")
    assert np.all(restored >= 0)
    assert np.allclose(restored, values, atol=1e-6)


def test_submovement_count_ignores_high_frequency_decoder_wiggle():
    t = np.linspace(0, 1, 100)
    forward = 10 * (3 * t ** 2 - 2 * t ** 3)
    wiggle = 0.02 * np.sin(2 * np.pi * 30 * t)
    pos = np.column_stack([wiggle, forward, np.zeros_like(t)])
    features = features_from_arrays(pos, movement_time_s=0.4, initiation_time_s=0.2)
    assert features["n_submovements"] <= 2


def test_context_query_is_disjoint_and_stratified():
    subjects = ["a"] * 12 + ["b"] * 12
    sp = ([1] * 4 + [2] * 4 + [3] * 4) * 2
    side = ([1, 1, 2, 2] * 3) * 2
    splits = split_context_query(subjects, sp, side, seed=7)
    for split in splits:
        assert not set(split.context_indices) & set(split.query_indices)
        combined = np.r_[split.context_indices, split.query_indices]
        assert len(np.unique(combined)) == 12


def test_moment_matching_preserves_cross_dimension_covariance():
    mu = np.array([[-1.0, -1.0], [1.0, 1.0]])
    logvar = np.log(np.full_like(mu, 0.25))
    mean, covariance = moment_matched_posterior(mu, logvar)
    assert np.allclose(mean, 0.0)
    assert covariance[0, 1] > 0.9
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_trajectory_only_encoder_does_not_receive_timing():
    model = ConditionalVAE(latent_dim=3, hidden_dim=16, timing_dim=2, encoder_uses_timing=False)
    x = torch.randn(5, 300)
    c = torch.tensor(np.tile(encode_condition(1, 1), (5, 1)))
    mu_a, _ = model.encode(x, c, torch.zeros(5, 2))
    mu_b, _ = model.encode(x, c, torch.ones(5, 2) * 100)
    assert torch.allclose(mu_a, mu_b)
    _, timing = model.decode(mu_a, c)
    assert timing.shape == (5, 2)


def test_hierarchical_model_shapes_and_gradients():
    model = HierarchicalCVAE(subject_dim=2, trial_dim=3, hidden_dim=24)
    context_x, context_c = torch.randn(6, 300), torch.randn(6, 4)
    target_x, target_t, target_c = torch.randn(7, 300), torch.randn(7, 2), torch.randn(7, 4)
    smu, slog = model.encode_subject(context_x, context_c)
    sz = model.sample(smu, slog)
    tmu, tlog = model.encode_trial(target_x, target_c, sz)
    tz = model.sample(tmu, tlog)
    recon, timing = model.decode(sz, tz, target_c)
    loss = hierarchical_loss(recon, target_x, timing, target_t, smu, slog, tmu, tlog)[0]
    loss.backward()
    assert smu.shape == (1, 2)
    assert tmu.shape == (7, 3)
    assert recon.shape == (7, 300) and timing.shape == (7, 2)
    assert all(p.grad is not None for p in model.parameters())
