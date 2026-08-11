"""Hierarchical conditional VAE with explicit subject and trial latents."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalCVAE(nn.Module):
    """Deep-Sets subject encoder plus a per-trial residual latent."""

    def __init__(self, input_dim=300, condition_dim=4, timing_dim=2,
                 subject_dim=3, trial_dim=4, hidden_dim=192):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.timing_dim = timing_dim
        self.subject_dim = subject_dim
        self.trial_dim = trial_dim
        self.hidden_dim = hidden_dim
        self.trial_embedding = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.subject_mu = nn.Linear(hidden_dim, subject_dim)
        self.subject_logvar = nn.Linear(hidden_dim, subject_dim)
        self.trial_encoder = nn.Sequential(
            nn.Linear(hidden_dim + subject_dim, hidden_dim), nn.ReLU()
        )
        self.trial_mu = nn.Linear(hidden_dim, trial_dim)
        self.trial_logvar = nn.Linear(hidden_dim, trial_dim)
        self.decoder = nn.Sequential(
            nn.Linear(subject_dim + trial_dim + condition_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.traj_head = nn.Linear(hidden_dim, input_dim)
        self.timing_head = nn.Linear(hidden_dim, timing_dim)

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def embed_trials(self, trajectory, condition):
        return self.trial_embedding(torch.cat([trajectory, condition], dim=-1))

    def encode_subject(self, context_trajectory, context_condition):
        embedded = self.embed_trials(context_trajectory, context_condition)
        pooled = embedded.mean(dim=0, keepdim=True)
        return self.subject_mu(pooled), self.subject_logvar(pooled)

    def encode_trial(self, trajectory, condition, subject_z):
        embedded = self.embed_trials(trajectory, condition)
        if subject_z.shape[0] == 1:
            subject_z = subject_z.expand(len(trajectory), -1)
        h = self.trial_encoder(torch.cat([embedded, subject_z], dim=-1))
        return self.trial_mu(h), self.trial_logvar(h)

    def decode(self, subject_z, trial_z, condition):
        if subject_z.shape[0] == 1:
            subject_z = subject_z.expand(len(trial_z), -1)
        h = self.decoder(torch.cat([subject_z, trial_z, condition], dim=-1))
        return self.traj_head(h), self.timing_head(h)


@dataclass
class HierarchicalNorm:
    trajectory_mean: np.ndarray
    trajectory_std: np.ndarray
    timing_mean: np.ndarray
    timing_std: np.ndarray
    timing_transform: str = "log"

    def tensors(self, device):
        return tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in (
            self.trajectory_mean, self.trajectory_std, self.timing_mean, self.timing_std
        ))

    def denormalise_timing(self, timing_z):
        from src.vae_model import inverse_timing
        transformed = np.asarray(timing_z) * self.timing_std + self.timing_mean
        return inverse_timing(transformed, self.timing_transform)


def hierarchical_loss(recon, target, recon_timing, target_timing,
                      subject_mu, subject_logvar, trial_mu, trial_logvar,
                      beta_subject=0.1, beta_trial=0.1, timing_weight=20.0):
    trajectory = F.mse_loss(recon, target, reduction="none").sum(-1).mean()
    timing = F.mse_loss(recon_timing, target_timing, reduction="none").sum(-1).mean()
    subject_kl = -0.5 * (1 + subject_logvar - subject_mu.square() - subject_logvar.exp()).sum(-1).mean()
    trial_kl = -0.5 * (1 + trial_logvar - trial_mu.square() - trial_logvar.exp()).sum(-1).mean()
    total = trajectory + timing_weight * timing + beta_subject * subject_kl + beta_trial * trial_kl
    return total, trajectory, timing, subject_kl, trial_kl


def _subject_arrays(trials, norm, device):
    from src.vae_model import encode_condition, encode_timing, transform_timing
    trajectory = np.stack([t["pos_norm"].reshape(-1) for t in trials]).astype(np.float32)
    timing = np.stack([encode_timing(t) for t in trials]).astype(np.float32)
    condition = np.stack([encode_condition(t["metadata"]["sp"], t["metadata"]["side"]) for t in trials])
    tm, ts, tim_m, tim_s = norm.tensors(device)
    return (
        (torch.as_tensor(trajectory, device=device) - tm) / ts,
        (transform_timing(torch.as_tensor(timing, device=device), norm.timing_transform) - tim_m) / tim_s,
        torch.as_tensor(condition, dtype=torch.float32, device=device),
    )


def train_hierarchical(
    train_trials, val_trials, run_dir: Path, subject_dim=3, trial_dim=4,
    epochs=150, subjects_per_epoch=17, context_size=16, target_size=16,
    lr=1e-3, seed=42, device=None, patience=25,
):
    """Train with equal numbers of episodes per subject to avoid trial imbalance."""
    from src.run_config import set_seed
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    all_train = np.stack([t["pos_norm"].reshape(-1) for t in train_trials]).astype(np.float32)
    vae_module = __import__("src.vae_model", fromlist=["encode_timing", "transform_timing"])
    all_timing = np.stack([vae_module.encode_timing(t) for t in train_trials])
    all_timing_transformed = vae_module.transform_timing(all_timing, "log")
    norm = HierarchicalNorm(all_train.mean(0), all_train.std(0) + 1e-6,
                            all_timing_transformed.mean(0), all_timing_transformed.std(0) + 1e-6,
                            timing_transform="log")
    model = HierarchicalCVAE(subject_dim=subject_dim, trial_dim=trial_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_by_subject = {s: [t for t in train_trials if t["metadata"]["subject"] == s]
                        for s in sorted({t["metadata"]["subject"] for t in train_trials})}
    val_by_subject = {s: [t for t in val_trials if t["metadata"]["subject"] == s]
                      for s in sorted({t["metadata"]["subject"] for t in val_trials})}
    rng = np.random.default_rng(seed)
    history = []
    best, stale = np.inf, 0

    def episode(subject_trials, train_mode, epoch_seed):
        local = np.random.default_rng(epoch_seed)
        order = local.permutation(len(subject_trials))
        nc = min(context_size, max(2, len(order) // 2))
        nt = min(target_size, len(order) - nc)
        context = [subject_trials[i] for i in order[:nc]]
        target = [subject_trials[i] for i in order[nc:nc + nt]]
        cx, _, cc = _subject_arrays(context, norm, device)
        tx, tt, tc = _subject_arrays(target, norm, device)
        smu, slog = model.encode_subject(cx, cc)
        sz = model.sample(smu, slog) if train_mode else smu
        tmu, tlog = model.encode_trial(tx, tc, sz)
        tz = model.sample(tmu, tlog) if train_mode else tmu
        recon, recon_timing = model.decode(sz, tz, tc)
        beta = 0.1
        return hierarchical_loss(recon, tx, recon_timing, tt, smu, slog, tmu, tlog,
                                 beta_subject=beta, beta_trial=beta)

    for epoch in range(1, epochs + 1):
        model.train()
        train_values = []
        subjects = list(train_by_subject)
        rng.shuffle(subjects)
        for j, subject in enumerate(subjects[:subjects_per_epoch]):
            losses = episode(train_by_subject[subject], True, seed * 100000 + epoch * 100 + j)
            optimizer.zero_grad()
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_values.append([float(x.detach()) for x in losses])
        model.eval()
        with torch.no_grad():
            # Validation context/query episodes are fixed across epochs so
            # early stopping compares the same data rather than episode noise.
            val_values = [[float(x) for x in episode(v, False, seed * 100000 + j)]
                          for j, v in enumerate(val_by_subject.values())]
        train_mean = np.mean(train_values, axis=0)
        val_mean = np.mean(val_values, axis=0)
        history.append({"epoch": epoch, "train_loss": train_mean[0], "val_loss": val_mean[0],
                        "val_trajectory": val_mean[1], "val_timing": val_mean[2],
                        "val_subject_kl": val_mean[3], "val_trial_kl": val_mean[4]})
        if val_mean[0] < best:
            best, stale = val_mean[0], 0
            torch.save({"model_state": model.state_dict(), "subject_dim": subject_dim,
                        "trial_dim": trial_dim, "norm": norm.__dict__, "epoch": epoch,
                        "val_loss": best, "seed": seed}, run_dir / "checkpoint.pt")
        else:
            stale += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"hier z_s={subject_dim} epoch={epoch} train={train_mean[0]:.3f} val={val_mean[0]:.3f}")
        if stale >= patience and epoch >= 50:
            break
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, norm, history
