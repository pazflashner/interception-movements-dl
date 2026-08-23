import numpy as np
import torch

from src.condition_effects import decode_generated
from src.vae_model import ConditionalVAE, NormStats


def _norm(input_dim: int) -> NormStats:
    return NormStats(
        train_mean=np.zeros(input_dim, dtype=np.float32),
        train_std=np.ones(input_dim, dtype=np.float32),
        timing_mean=np.zeros(2, dtype=np.float32),
        timing_std=np.ones(2, dtype=np.float32),
        timing_transform="identity",
    )


def test_unconditional_decode_is_invariant_to_metadata():
    torch.manual_seed(2)
    model = ConditionalVAE(
        input_dim=200,
        condition_dim=5,
        latent_dim=3,
        hidden_dim=16,
        timing_dim=2,
        encoder_uses_timing=False,
        use_condition=False,
    )
    z = np.ones((2, 3), dtype=np.float32)
    metadata = [
        {"sp": 1, "side": 1, "target_speed_screen_s": 0.5},
        {"sp": 3, "side": 2, "target_speed_screen_s": 0.8},
    ]
    trajectory, timing = decode_generated(model, _norm(200), z, metadata, "cpu")

    assert np.allclose(trajectory[0], trajectory[1])
    assert np.allclose(timing[0], timing[1])
