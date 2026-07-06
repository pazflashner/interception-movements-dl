"""
Training script for the Conditional VAE.

Supports:
- Leave-N-subjects-out data splitting
- Hyperparameter sweep over latent dimensions
- Early stopping on validation loss
- Model checkpointing
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.vae_model import ConditionalVAE, TrajectoryDataset, vae_loss


# ── Data splitting ────────────────────────────────────────────────────────────
def split_subjects(
    trials: list[dict],
    n_train: int = config.N_TRAIN,
    n_val: int = config.N_VAL,
    n_test: int = config.N_TEST,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split trials by subject into train / val / test."""
    subjects = sorted(set(t["metadata"]["subject"] for t in trials))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train : n_train + n_val])
    test_subj = set(subjects[n_train + n_val : n_train + n_val + n_test])

    train = [t for t in trials if t["metadata"]["subject"] in train_subj]
    val = [t for t in trials if t["metadata"]["subject"] in val_subj]
    test = [t for t in trials if t["metadata"]["subject"] in test_subj]

    print(
        f"Split: {len(train)} train ({len(train_subj)} subj), "
        f"{len(val)} val ({len(val_subj)} subj), "
        f"{len(test)} test ({len(test_subj)} subj)"
    )
    return train, val, test


# ── Training loop ─────────────────────────────────────────────────────────────
def train_vae(
    train_trials: list[dict],
    val_trials: list[dict],
    latent_dim: int = config.DEFAULT_LATENT_DIM,
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    kl_weight: float = config.KL_WEIGHT,
    save_dir: Path | None = None,
    device: str | None = None,
) -> tuple[ConditionalVAE, dict]:
    """
    Train the CVAE and return the trained model + training history.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}, latent_dim={latent_dim}")

    save_dir = Path(save_dir or config.MODELS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datasets & loaders
    train_ds = TrajectoryDataset(train_trials)
    val_ds = TrajectoryDataset(val_trials)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Normalisation statistics (fit on train)
    train_mean = torch.from_numpy(train_ds.trajectories.mean(axis=0)).to(device)
    train_std = torch.from_numpy(train_ds.trajectories.std(axis=0) + 1e-8).to(device)

    # Model
    model = ConditionalVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [], "train_kl": [], "val_kl": []}
    best_val = float("inf")
    patience_counter = 0
    patience_limit = 30

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss, epoch_recon, epoch_kl, n = 0, 0, 0, 0
        for traj, cond, _ in train_loader:
            traj, cond = traj.to(device), cond.to(device)
            traj_z = (traj - train_mean) / train_std  # z-score

            recon, mu, logvar, _ = model(traj_z, cond)
            loss, rl, kl = vae_loss(recon, traj_z, mu, logvar, kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = traj.size(0)
            epoch_loss += loss.item() * bs
            epoch_recon += rl.item() * bs
            epoch_kl += kl.item() * bs
            n += bs

        history["train_loss"].append(epoch_loss / n)
        history["train_recon"].append(epoch_recon / n)
        history["train_kl"].append(epoch_kl / n)

        # ── Validate ──
        model.eval()
        val_loss, val_recon, val_kl, nv = 0, 0, 0, 0
        with torch.no_grad():
            for traj, cond, _ in val_loader:
                traj, cond = traj.to(device), cond.to(device)
                traj_z = (traj - train_mean) / train_std

                recon, mu, logvar, _ = model(traj_z, cond)
                loss, rl, kl = vae_loss(recon, traj_z, mu, logvar, kl_weight)

                bs = traj.size(0)
                val_loss += loss.item() * bs
                val_recon += rl.item() * bs
                val_kl += kl.item() * bs
                nv += bs

        vl = val_loss / nv
        history["val_loss"].append(vl)
        history["val_recon"].append(val_recon / nv)
        history["val_kl"].append(val_kl / nv)

        scheduler.step(vl)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train {history['train_loss'][-1]:.5f} | "
                f"Val {vl:.5f}"
            )

        # Early stopping
        if vl < best_val:
            best_val = vl
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "latent_dim": latent_dim,
                    "train_mean": train_mean.cpu().numpy().tolist(),
                    "train_std": train_std.cpu().numpy().tolist(),
                },
                save_dir / f"cvae_z{latent_dim}_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save history
    with open(save_dir / f"history_z{latent_dim}.json", "w") as f:
        json.dump(history, f)

    # Load best model
    ckpt = torch.load(save_dir / f"cvae_z{latent_dim}_best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    return model, history


# ── Latent-dim sweep ──────────────────────────────────────────────────────────
def sweep_latent_dims(
    train_trials: list[dict],
    val_trials: list[dict],
    dims: list[int] = config.LATENT_DIMS_SWEEP,
    **kwargs,
) -> dict:
    """Train VAE for each latent dim and report best validation loss."""
    results = {}
    for d in dims:
        print(f"\n{'='*60}\nLatent dim = {d}\n{'='*60}")
        model, hist = train_vae(train_trials, val_trials, latent_dim=d, **kwargs)
        best_val = min(hist["val_loss"])
        results[d] = {"best_val_loss": best_val, "epochs_trained": len(hist["val_loss"])}
        print(f"  → Best val loss: {best_val:.6f}")

    print("\n── Sweep summary ──")
    for d, r in sorted(results.items()):
        print(f"  z={d}: val_loss={r['best_val_loss']:.6f} ({r['epochs_trained']} epochs)")
    return results
