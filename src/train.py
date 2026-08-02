"""
Training script for the Conditional VAE.

Every run is driven by a single ``RunConfig`` (see ``src/run_config.py``) and
writes into its own directory:

    results/runs/<run_name>/config.yaml     # the run's dedicated config
    results/runs/<run_name>/checkpoint.pt   # best-val model, config embedded
    results/runs/<run_name>/history.json    # per-epoch losses

Supports:
- Leave-N-subjects-out data splitting (seeded from the run config)
- Hyperparameter sweep over latent dimensions (one run directory each)
- Early stopping on validation loss
- Model checkpointing alongside the config that produced it
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.run_config import RunConfig, make_run_dir, seed_worker, set_seed
from src.vae_model import ConditionalVAE, TrajectoryDataset, vae_loss


# ── Data splitting ────────────────────────────────────────────────────────────
def split_subject_ids(
    subjects: list[str],
    n_train: int = config.N_TRAIN,
    n_val: int = config.N_VAL,
    n_test: int = config.N_TEST,
    seed: int = config.SEED,
) -> tuple[list[str], list[str], list[str]]:
    """
    Assign subject IDs to train / val / test (leave-N-subjects-out).

    The single source of truth for the split: ``split_subjects`` and
    ``scripts/make_dataset.py`` both go through here, so the ``splits.json``
    written next to the dataset always matches what training uses.

    Sorts before shuffling so the result depends only on the set of subjects and
    the seed, not on the order they were loaded in, and uses a dedicated
    ``RandomState`` so it does not depend on how much global RNG state has
    already been consumed.
    """
    subjects = sorted(set(subjects))
    requested = n_train + n_val + n_test
    if len(subjects) < requested:
        print(
            f"  WARNING: split asks for {requested} subjects but only "
            f"{len(subjects)} are available; later splits will be short."
        )

    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    return (
        subjects[:n_train],
        subjects[n_train : n_train + n_val],
        subjects[n_train + n_val : n_train + n_val + n_test],
    )


def split_subjects(
    trials: list[dict],
    n_train: int = config.N_TRAIN,
    n_val: int = config.N_VAL,
    n_test: int = config.N_TEST,
    seed: int = config.SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split trials by subject into train / val / test."""
    all_subjects = [t["metadata"]["subject"] for t in trials]
    train_ids, val_ids, test_ids = split_subject_ids(
        all_subjects, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed
    )
    train_subj, val_subj, test_subj = set(train_ids), set(val_ids), set(test_ids)

    train = [t for t in trials if t["metadata"]["subject"] in train_subj]
    val = [t for t in trials if t["metadata"]["subject"] in val_subj]
    test = [t for t in trials if t["metadata"]["subject"] in test_subj]

    print(
        f"Split: {len(train)} train ({len(train_subj)} subj), "
        f"{len(val)} val ({len(val_subj)} subj), "
        f"{len(test)} test ({len(test_subj)} subj)"
    )
    return train, val, test


def split_subjects_for(trials: list[dict], cfg: RunConfig):
    """``split_subjects`` driven by a run config."""
    return split_subjects(
        trials,
        n_train=cfg.n_train,
        n_val=cfg.n_val,
        n_test=cfg.n_test,
        seed=cfg.seed,
    )


# ── Training loop ─────────────────────────────────────────────────────────────
def train_vae(
    train_trials: list[dict],
    val_trials: list[dict],
    cfg: RunConfig | None = None,
    run_dir: Path | None = None,
    device: str | None = None,
) -> tuple[ConditionalVAE, dict, Path]:
    """
    Train the CVAE for one run.

    Creates ``run_dir`` if not given, writes the run's ``config.yaml`` before
    training starts, and saves the best-val checkpoint into the same directory
    with the config embedded in it.

    Returns (model, history, run_dir).
    """
    cfg = cfg or RunConfig()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed everything for this run; the generator drives DataLoader shuffling.
    generator = set_seed(cfg.seed, deterministic=cfg.deterministic)

    # Write the dedicated config up front, so an interrupted run is still
    # identifiable and reproducible.
    cfg.stamp_environment(device)
    run_dir = Path(run_dir) if run_dir is not None else make_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir)

    print(f"Run: {run_dir.name}")
    print(
        f"Training on {device}, latent_dim={cfg.latent_dim}, seed={cfg.seed}, "
        f"timing_dim={cfg.timing_dim}"
    )

    # Datasets & loaders
    train_ds = TrajectoryDataset(train_trials)
    val_ds = TrajectoryDataset(val_trials)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Normalisation statistics (fit on train). Timing is standardised
    # separately: seconds and millimetres are not on a comparable scale.
    train_mean = torch.from_numpy(train_ds.trajectories.mean(axis=0)).to(device)
    train_std = torch.from_numpy(train_ds.trajectories.std(axis=0) + 1e-8).to(device)
    timing_mean = torch.from_numpy(train_ds.timings.mean(axis=0)).to(device)
    timing_std = torch.from_numpy(train_ds.timings.std(axis=0) + 1e-8).to(device)

    # Model
    model = ConditionalVAE(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        timing_dim=cfg.timing_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "val_recon": [],
        "train_kl": [], "val_kl": [],
        "train_timing": [], "val_timing": [],
    }
    best_val = float("inf")
    patience_counter = 0
    patience_limit = cfg.patience

    def run_batch(traj, timing, cond):
        """Standardise a batch, run the model, return (loss, recon, kl, timing)."""
        traj = traj.to(device)
        timing = timing.to(device)
        cond = cond.to(device)
        traj_z = (traj - train_mean) / train_std          # z-score
        timing_z = (timing - timing_mean) / timing_std
        if cfg.timing_dim == 0:
            timing_z = None

        recon, recon_timing, mu, logvar, _ = model(traj_z, cond, timing_z)
        return vae_loss(
            recon, traj_z, mu, logvar, cfg.kl_weight,
            recon_timing=recon_timing,
            target_timing=timing_z,
            timing_weight=cfg.timing_weight,
        )

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss, epoch_recon, epoch_kl, epoch_timing, n = 0, 0, 0, 0, 0
        for traj, timing, cond, _ in train_loader:
            loss, rl, kl, tl = run_batch(traj, timing, cond)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = traj.size(0)
            epoch_loss += loss.item() * bs
            epoch_recon += rl.item() * bs
            epoch_kl += kl.item() * bs
            epoch_timing += tl.item() * bs
            n += bs

        history["train_loss"].append(epoch_loss / n)
        history["train_recon"].append(epoch_recon / n)
        history["train_kl"].append(epoch_kl / n)
        history["train_timing"].append(epoch_timing / n)

        # ── Validate ──
        model.eval()
        val_loss, val_recon, val_kl, val_timing, nv = 0, 0, 0, 0, 0
        with torch.no_grad():
            for traj, timing, cond, _ in val_loader:
                loss, rl, kl, tl = run_batch(traj, timing, cond)

                bs = traj.size(0)
                val_loss += loss.item() * bs
                val_recon += rl.item() * bs
                val_kl += kl.item() * bs
                val_timing += tl.item() * bs
                nv += bs

        vl = val_loss / nv
        history["val_loss"].append(vl)
        history["val_recon"].append(val_recon / nv)
        history["val_kl"].append(val_kl / nv)
        history["val_timing"].append(val_timing / nv)

        scheduler.step(vl)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train {history['train_loss'][-1]:.5f} | "
                f"Val {vl:.5f} | "
                f"recon {history['val_recon'][-1]:.5f} | "
                f"timing {history['val_timing'][-1]:.5f} | "
                f"KL {history['val_kl'][-1]:.5f}"
            )

        # Early stopping
        if vl < best_val:
            best_val = vl
            patience_counter = 0
            # The checkpoint carries its own config, so it stays interpretable
            # even if moved away from its run directory.
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "latent_dim": cfg.latent_dim,
                    "timing_dim": cfg.timing_dim,
                    "train_mean": train_mean.cpu().numpy().tolist(),
                    "train_std": train_std.cpu().numpy().tolist(),
                    "timing_mean": timing_mean.cpu().numpy().tolist(),
                    "timing_std": timing_std.cpu().numpy().tolist(),
                    "timing_features": config.TIMING_FEATURES,
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "val_loss": vl,
                },
                run_dir / "checkpoint.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save history next to the config and checkpoint
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    # Load best model
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    # ASCII only: Windows consoles default to cp1252, which cannot encode
    # arrows or box-drawing characters.
    print(f"  -> Best val loss: {best_val:.6f} (epoch {ckpt['epoch']}) -> {run_dir}")

    return model, history, run_dir


# ── Latent-dim sweep ──────────────────────────────────────────────────────────
def sweep_latent_dims(
    train_trials: list[dict],
    val_trials: list[dict],
    dims: list[int] = config.LATENT_DIMS_SWEEP,
    cfg: RunConfig | None = None,
    device: str | None = None,
) -> dict:
    """
    Train the VAE for each latent dim, one run directory per dim.

    Each run re-seeds from ``cfg.seed``, so dims differ only in latent size.
    """
    base = cfg or RunConfig()
    results = {}
    for d in dims:
        print(f"\n{'='*60}\nLatent dim = {d}\n{'='*60}")
        run_cfg = replace(base, latent_dim=d)
        model, hist, run_dir = train_vae(train_trials, val_trials, cfg=run_cfg, device=device)
        best_val = min(hist["val_loss"])
        results[d] = {
            "best_val_loss": best_val,
            "epochs_trained": len(hist["val_loss"]),
            "run_dir": str(run_dir),
        }

    print("\n--- Sweep summary ---")
    for d, r in sorted(results.items()):
        print(f"  z={d}: val_loss={r['best_val_loss']:.6f} ({r['epochs_trained']} epochs)")
    return results


# ── Repeated runs ─────────────────────────────────────────────────────────────
def train_across_seeds(
    trials: list[dict],
    seeds: list[int],
    cfg: RunConfig | None = None,
    device: str | None = None,
) -> dict:
    """
    Repeat a run across seeds, re-splitting subjects each time.

    With 28 subjects both the split and the optimisation contribute noise, so
    each seed gets its own subject split as well as its own initialisation.
    Returns {seed: {best_val_loss, epochs_trained, run_dir}}.
    """
    base = cfg or RunConfig()
    results = {}
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        run_cfg = replace(base, seed=seed)
        train_trials, val_trials, _ = split_subjects_for(trials, run_cfg)
        _, hist, run_dir = train_vae(train_trials, val_trials, cfg=run_cfg, device=device)
        results[seed] = {
            "best_val_loss": min(hist["val_loss"]),
            "epochs_trained": len(hist["val_loss"]),
            "run_dir": str(run_dir),
        }

    vals = np.array([r["best_val_loss"] for r in results.values()])
    print(f"\n--- Across {len(seeds)} seeds ---")
    for seed, r in results.items():
        print(f"  seed {seed}: val_loss={r['best_val_loss']:.6f}")
    print(f"  mean={vals.mean():.6f}  sd={vals.std(ddof=1) if len(vals) > 1 else 0.0:.6f}  "
          f"min={vals.min():.6f}  max={vals.max():.6f}")
    return results
