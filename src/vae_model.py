"""
Phase 3 – Conditional Variational Autoencoder (CVAE) for interception
movement trajectories.

Architecture
------------
Encoder:  trajectory (T*3) + timing (2) + condition_vec → hidden → (μ, log σ²)
Decoder:  z + condition_vec → hidden → trajectory (T*3)  [head 1]
                                    → timing (2)         [head 2]

Condition vector includes starting position and speed configuration
so the latent space encodes intrinsic movement style.

Why a timing head
-----------------
Temporal normalisation resamples every trial to ``NORMALISED_LENGTH`` frames,
which makes the input dimension uniform but throws away *how long* the movement
took. Without that, the model can only ever describe trajectory shape: two
movements along the same path at half the speed are identical inputs, velocity
is recoverable only up to an unknown time scale, and a generated sample has no
duration to play it back over.

The timing channels (movement time and initiation time, both in seconds) are
therefore treated as part of what the model reconstructs: they are appended to
the encoder input so the latent must carry them, and predicted by a separate
decoder head so sampling ``z`` produces a full movement — shape *and* timing.
Set ``timing_dim=0`` to recover the shape-only model.
"""
from __future__ import annotations

from dataclasses import dataclass

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

    Start config and target speed
    -----------------------------
    ``sp`` indexes *both* the stimulus starting position
    (``config.STARTING_POSITIONS``: 120/140/160 mm) and the target speed range
    (``config.SPEED_RANGES``: 255-300 / 298-350 / 340-400 mm/s). They are
    confounded by the experimental design — the filename carries one index for
    the pair — so the one-hot over sp *is* the joint (start config, target
    speed) encoding. Adding separate start-position and speed columns would be
    exactly collinear with this one-hot and buy nothing.

    The exact speed within each range was randomised per trial and is not
    recoverable from the filenames; ``data/stimuli/`` is empty in this
    checkout, so the range index is the finest speed information available. If
    per-trial stimulus speeds are recovered later, append them here as a
    normalised continuous column and bump ``CONDITION_DIM``.

    This vector is concatenated into the encoder input *and* the decoder input
    (see ``ConditionalVAE.encode`` / ``.decode``), which is what lets the latent
    model movement style rather than task-driven variance: the decoder is given
    the task, so z does not need to encode it.
    """
    vec = np.zeros(4, dtype=np.float32)
    if 1 <= sp <= 3:
        vec[sp - 1] = 1.0
    vec[3] = 1.0 if side == 2 else 0.0
    return vec


CONDITION_DIM = 4  # length of condition vector
TIMING_DIM = config.TIMING_DIM  # movement_time_s, initiation_time_s


def encode_timing(trial: dict, fs: float = config.RECORDING_HZ) -> np.ndarray:
    """
    Extract the timing channels (in seconds) from a preprocessed trial.

    Derived from the raw-frame indices rather than stored separately, so trial
    dicts cached before the timing head existed still work.

        movement_time_s   : movement onset → finger arrival (movement duration)
        initiation_time_s : go-signal → movement onset (reaction / wait time)

    Reaction time is measured from the **go-signal** (``go_signal_idx``, the
    moment the object starts moving), not from object appearance, so the
    randomised foreperiod does not contaminate it. Falls back to the stimulus
    marker for trial dicts built before the go-signal was tracked.
    """
    ref = trial.get("go_signal_idx", trial["stim_onset_idx"])
    move_time = (trial["move_end_idx"] - trial["move_start_idx"]) / fs
    init_time = (trial["move_start_idx"] - ref) / fs
    return np.array([move_time, init_time], dtype=np.float32)


# ── VAE Model ─────────────────────────────────────────────────────────────────
class ConditionalVAE(nn.Module):
    """Conditional VAE for trajectory reconstruction with a timing head."""

    def __init__(
        self,
        input_dim: int = config.NORMALISED_LENGTH * 3,
        condition_dim: int = CONDITION_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        latent_dim: int = config.DEFAULT_LATENT_DIM,
        timing_dim: int = TIMING_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.timing_dim = timing_dim

        # Encoder — sees the trajectory, its timing, and the task condition
        enc_input = input_dim + timing_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder — shared trunk, one head per output modality
        dec_input = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.traj_head = nn.Linear(hidden_dim, input_dim)
        self.timing_head = nn.Linear(hidden_dim, timing_dim) if timing_dim else None

    def encode(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        timing: torch.Tensor | None = None,
    ):
        parts = [x]
        if self.timing_dim:
            if timing is None:
                raise ValueError(
                    f"model has timing_dim={self.timing_dim}; encode() needs a timing tensor"
                )
            parts.append(timing)
        parts.append(c)
        h = self.encoder(torch.cat(parts, dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        """Returns (trajectory, timing); timing is None when timing_dim == 0."""
        h = self.decoder(torch.cat([z, c], dim=-1))
        timing = self.timing_head(h) if self.timing_head is not None else None
        return self.traj_head(h), timing

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        timing: torch.Tensor | None = None,
    ):
        """Returns (recon_traj, recon_timing, mu, logvar, z)."""
        mu, logvar = self.encode(x, c, timing)
        z = self.reparameterize(mu, logvar)
        recon, recon_timing = self.decode(z, c)
        return recon, recon_timing, mu, logvar, z


# ── KL annealing ──────────────────────────────────────────────────────────────
def kl_weight_at(
    epoch: int,
    target: float = config.KL_WEIGHT,
    schedule: str = config.KL_ANNEAL,
    anneal_epochs: int = config.KL_ANNEAL_EPOCHS,
    cycles: int = config.KL_ANNEAL_CYCLES,
    ratio: float = config.KL_ANNEAL_RATIO,
) -> float:
    """
    β for a given 1-indexed epoch.

    Annealing exists to avoid posterior collapse: at full β from the start, the
    cheapest way to cut the KL term is to make q(z|x) equal the prior and ignore
    z, and a decoder that has learned to work without the latent gets no
    gradient pulling it back. Ramping β from 0 lets reconstruction establish a
    use for the latent first.

    See ``config.KL_ANNEAL`` for the schedules.
    """
    if schedule == "none":
        return target
    if anneal_epochs <= 0:
        return target

    e = max(epoch - 1, 0)  # epochs are 1-indexed

    if schedule == "linear":
        return target * min(1.0, e / anneal_epochs)

    if schedule == "cyclical":
        # Each cycle ramps over `ratio` of its length, then holds at full β.
        cycle_len = max(anneal_epochs, 1)
        pos = (e % cycle_len) / cycle_len
        if cycles > 0 and e >= cycle_len * cycles:
            return target  # past the last cycle, stay at the true objective
        return target * min(1.0, pos / ratio) if ratio > 0 else target

    raise ValueError(f"unknown KL schedule: {schedule!r}")


# ── Loss function ─────────────────────────────────────────────────────────────
def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = config.KL_WEIGHT,
    recon_timing: torch.Tensor | None = None,
    target_timing: torch.Tensor | None = None,
    timing_weight: float = config.TIMING_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE ELBO loss = trajectory MSE + λ · timing MSE + β · KL divergence.

    The two reconstruction terms are averaged over their own dimensions, so
    ``timing_weight`` sets the trade-off directly rather than being implicitly
    scaled by the 300:2 dimension ratio.

    Returns (total_loss, recon_loss, kl_loss, timing_loss).

    Reduction
    ---------
    Every term is **summed over its own dimensions and averaged over the
    batch** — the standard ELBO. This matters more than it looks: averaging the
    reconstruction over 300 trajectory dims while averaging the KL over
    ``latent_dim`` dims silently inflates the effective β by
    ``input_dim / latent_dim`` (100x at z=3), which crushes the latent and
    caps reconstruction quality regardless of how long the model trains.
    """
    recon_loss = F.mse_loss(recon, target, reduction="none").sum(dim=-1).mean()
    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()

    if recon_timing is not None and target_timing is not None:
        timing_loss = (
            F.mse_loss(recon_timing, target_timing, reduction="none").sum(dim=-1).mean()
        )
    else:
        timing_loss = torch.zeros((), device=recon.device, dtype=recon.dtype)

    total = recon_loss + timing_weight * timing_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss, timing_loss


# ── Normalisation statistics ──────────────────────────────────────────────────
@dataclass
class NormStats:
    """
    Train-set standardisation constants, kept together so trajectory and timing
    stats cannot drift apart between training, evaluation and the dashboard.
    """

    train_mean: np.ndarray
    train_std: np.ndarray
    timing_mean: np.ndarray
    timing_std: np.ndarray

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "NormStats":
        """Read the stats out of a checkpoint dict saved by ``train_vae``."""
        as_arr = lambda v: np.asarray(v, dtype=np.float32)
        # Checkpoints written before the timing head lack the timing stats;
        # identity constants keep them loadable (they also have timing_dim=0).
        return cls(
            train_mean=as_arr(ckpt["train_mean"]),
            train_std=as_arr(ckpt["train_std"]),
            timing_mean=as_arr(ckpt.get("timing_mean", np.zeros(TIMING_DIM))),
            timing_std=as_arr(ckpt.get("timing_std", np.ones(TIMING_DIM))),
        )

    def torch(self, device: str = "cpu"):
        """Return the four constants as tensors on *device*."""
        t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
        return t(self.train_mean), t(self.train_std), t(self.timing_mean), t(self.timing_std)

    def denormalise_timing(self, timing_z: np.ndarray) -> np.ndarray:
        """Map standardised timing predictions back to seconds."""
        return np.asarray(timing_z) * self.timing_std + self.timing_mean


# ── Dataset ───────────────────────────────────────────────────────────────────
class TrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapping preprocessed trials."""

    def __init__(self, trials: list[dict]):
        self.trajectories = []
        self.timings = []
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
            self.timings.append(encode_timing(t))
            self.conditions.append(cond)
            self.subjects.append(meta.get("subject", ""))
            self.trial_ids.append(meta.get("trial_id", ""))

        self.trajectories = np.stack(self.trajectories)
        self.timings = np.stack(self.timings)
        self.conditions = np.stack(self.conditions)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.trajectories[idx]),
            torch.from_numpy(self.timings[idx]),
            torch.from_numpy(self.conditions[idx]),
            idx,
        )
