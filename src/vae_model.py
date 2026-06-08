"""
Phase 3 – Conditional Variational Autoencoder (CVAE) for interception
movement trajectories.

Architecture
------------
Encoder:  trajectory (T*3) + condition_vec → hidden → (μ, log σ²)
Decoder:  z + condition_vec → hidden → reconstructed trajectory (T*3)

Condition vector includes starting position and speed configuration
so the latent space encodes intrinsic movement style.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Condition encoding ────────────────────────────────────────────────────────
def encode_condition(sp: int, side: int) -> np.ndarray:
    """
    Create a condition vector from trial metadata.

    sp  : starting position index (1, 2, 3)
    side: starting side (1=left, 2=right)

    Returns a 4-dim vector: [one-hot sp (3)] + [side binary (1)]
    """
    vec = np.zeros(4, dtype=np.float32)
    if 1 <= sp <= 3:
        vec[sp - 1] = 1.0
    vec[3] = 1.0 if side == 2 else 0.0
    return vec


CONDITION_DIM = 4  # length of condition vector


# ── VAE Model ─────────────────────────────────────────────────────────────────
class ConditionalVAE(nn.Module):
    """Conditional VAE for trajectory reconstruction."""

    def __init__(
        self,
        input_dim: int = config.NORMALISED_LENGTH * 3,
        condition_dim: int = CONDITION_DIM,
        hidden_dim: int = 256,
        latent_dim: int = config.DEFAULT_LATENT_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        enc_input = input_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        dec_input = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar, z


# ── Loss function ─────────────────────────────────────────────────────────────
def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = config.KL_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE ELBO loss = reconstruction MSE + β · KL divergence.

    Returns (total_loss, recon_loss, kl_loss).
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


# ── Dataset ───────────────────────────────────────────────────────────────────
class TrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapping preprocessed trials."""

    def __init__(self, trials: list[dict]):
        self.trajectories = []
        self.conditions = []
        self.subjects = []
        self.trial_ids = []

        for t in trials:
            traj = t["pos_norm"].flatten().astype(np.float32)
            meta = t["metadata"]
            cond = encode_condition(
                sp=meta.get("sp", 1),
                side=meta.get("side", 1),
            )
            self.trajectories.append(traj)
            self.conditions.append(cond)
            self.subjects.append(meta.get("subject", ""))
            self.trial_ids.append(meta.get("trial_id", ""))

        self.trajectories = np.stack(self.trajectories)
        self.conditions = np.stack(self.conditions)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.trajectories[idx]),
            torch.from_numpy(self.conditions[idx]),
            idx,
        )
