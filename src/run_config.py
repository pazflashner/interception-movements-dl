"""
Per-run configuration and seeding.

Every training run is fully described by a single ``RunConfig``, which is
written to ``config.yaml`` inside that run's directory. Checkpoints are saved
next to their own config (and embed a copy of it), so a checkpoint can never be
separated from the settings that produced it.

Layout of a run directory::

    results/runs/z3_seed42_20260725-143012/
        config.yaml       # everything needed to reproduce the run
        checkpoint.pt     # best-val model state + embedded config
        history.json      # per-epoch losses

With only 28 subjects, single-run numbers are noisy; the intended workflow is
many runs differing only in ``seed``, aggregated afterwards.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
import config


# ── Seeding ───────────────────────────────────────────────────────────────────
def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """
    Seed every RNG the pipeline touches and return a torch ``Generator``.

    Seeds ``random``, ``numpy``, ``torch`` (CPU + all CUDA devices) and
    ``PYTHONHASHSEED``. The returned generator should be handed to any
    ``DataLoader(shuffle=True)`` so that shuffling is reproducible regardless of
    how much global RNG state other code has consumed in between.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int):
    """``worker_init_fn`` for DataLoader workers (num_workers > 0)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── Provenance ────────────────────────────────────────────────────────────────
def _git_state() -> tuple[str, bool]:
    """Return (commit sha, dirty flag); ('unknown', False) outside a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=config.PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", False


# ── Run configuration ─────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    """Complete, serialisable description of one training run."""

    # Reproducibility
    seed: int = config.SEED
    deterministic: bool = True

    # Data split (leave-N-subjects-out over 28 subjects)
    n_train: int = config.N_TRAIN
    n_val: int = config.N_VAL
    n_test: int = config.N_TEST
    normalised_length: int = config.NORMALISED_LENGTH

    # Model
    latent_dim: int = config.DEFAULT_LATENT_DIM
    hidden_dim: int = config.HIDDEN_DIM
    architecture: str = config.ARCHITECTURE   # "mlp" (default) or "cnn"
    # Baseline=True reconstructs timing after giving it to the encoder. False
    # makes the timing head predict timing from trajectory shape + condition.
    encoder_uses_timing: bool = True
    timing_transform: str = "log"
    # Reconstruct movement/initiation time alongside the trajectory, so the
    # latent keeps the temporal axis that resampling removes.
    predict_timing: bool = config.PREDICT_TIMING

    # Optimisation
    epochs: int = config.NUM_EPOCHS
    batch_size: int = config.BATCH_SIZE
    balance_subjects: bool = False
    lr: float = config.LEARNING_RATE
    kl_weight: float = config.KL_WEIGHT
    timing_weight: float = config.TIMING_WEIGHT
    patience: int = config.EARLY_STOPPING_PATIENCE
    # KL annealing: ramp beta from 0 so the latent is used before it is
    # regularised. See config.KL_ANNEAL.
    kl_anneal: str = config.KL_ANNEAL
    kl_anneal_epochs: int = config.KL_ANNEAL_EPOCHS
    kl_anneal_cycles: int = config.KL_ANNEAL_CYCLES
    grad_clip: float = config.GRAD_CLIP_NORM

    @property
    def timing_dim(self) -> int:
        return config.TIMING_DIM if self.predict_timing else 0

    # Environment (filled in at save time; informational only)
    device: str = ""
    created_at: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    versions: dict = field(default_factory=dict)

    # ── Naming ────────────────────────────────────────────────────────────
    @property
    def run_name(self) -> str:
        """Directory name: identifies the axes that vary across repeated runs."""
        stamp = (self.created_at or datetime.now().isoformat(timespec="seconds"))
        stamp = stamp.replace("-", "").replace(":", "").replace("T", "-")[:15]
        return f"z{self.latent_dim}_seed{self.seed}_{stamp}"

    # ── (De)serialisation ─────────────────────────────────────────────────
    def stamp_environment(self, device: str) -> "RunConfig":
        """Record device, timestamp and git state. Call once, before saving."""
        sha, dirty = _git_state()
        self.device = device
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.git_commit = sha
        self.git_dirty = dirty
        # str() matters: torch.__version__ is a TorchVersion (a str subclass),
        # which yaml.safe_dump refuses to represent.
        self.versions = {
            "python": str(platform.python_version()),
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
        }
        return self

    def to_dict(self) -> dict:
        """Nested dict, grouped for readability in YAML."""
        d = asdict(self)
        return {
            "seed": d["seed"],
            "deterministic": d["deterministic"],
            "data": {
                "n_train": d["n_train"],
                "n_val": d["n_val"],
                "n_test": d["n_test"],
                "normalised_length": d["normalised_length"],
            },
            "model": {
                "latent_dim": d["latent_dim"],
                "hidden_dim": d["hidden_dim"],
                "architecture": d["architecture"],
                "encoder_uses_timing": d["encoder_uses_timing"],
                "timing_transform": d["timing_transform"],
                "predict_timing": d["predict_timing"],
            },
            "train": {
                "epochs": d["epochs"],
                "batch_size": d["batch_size"],
                "balance_subjects": d["balance_subjects"],
                "lr": d["lr"],
                "kl_weight": d["kl_weight"],
                "timing_weight": d["timing_weight"],
                "patience": d["patience"],
                "kl_anneal": d["kl_anneal"],
                "kl_anneal_epochs": d["kl_anneal_epochs"],
                "kl_anneal_cycles": d["kl_anneal_cycles"],
                "grad_clip": d["grad_clip"],
            },
            "env": {
                "device": d["device"],
                "created_at": d["created_at"],
                "git_commit": d["git_commit"],
                "git_dirty": d["git_dirty"],
                "versions": d["versions"],
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        env = d.get("env", {})
        return cls(
            seed=d["seed"],
            deterministic=d.get("deterministic", True),
            **d.get("data", {}),
            **d.get("model", {}),
            **d.get("train", {}),
            device=env.get("device", ""),
            created_at=env.get("created_at", ""),
            git_commit=env.get("git_commit", ""),
            git_dirty=env.get("git_dirty", False),
            versions=env.get("versions", {}),
        )

    def save(self, run_dir: Path) -> Path:
        """Write ``config.yaml`` into ``run_dir`` and return its path."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "config.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
        return path

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        """Load from a ``config.yaml`` (or a run directory containing one)."""
        path = Path(path)
        if path.is_dir():
            path = path / "config.yaml"
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


# ── Run directories ───────────────────────────────────────────────────────────
def make_run_dir(cfg: RunConfig, root: Path | None = None, name: str | None = None) -> Path:
    """Create and return a fresh directory for this run."""
    root = Path(root or config.RUNS_DIR)
    run_dir = root / (name or cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_runs(root: Path | None = None) -> list[Path]:
    """
    All run directories holding both a config and a checkpoint.

    Sorted oldest-first by checkpoint mtime, so ``find_runs()[-1]`` is the most
    recently trained run regardless of naming.
    """
    root = Path(root or config.RUNS_DIR)
    if not root.exists():
        return []
    runs = [
        d for d in root.iterdir()
        if d.is_dir() and (d / "config.yaml").exists() and (d / "checkpoint.pt").exists()
    ]
    return sorted(runs, key=lambda d: (d / "checkpoint.pt").stat().st_mtime)


def summarise_runs(root: Path | None = None) -> list[dict]:
    """
    Collect one row per run: config axes + best validation loss.

    With 28 subjects the run-to-run spread matters more than any single value,
    so this is the intended way to read results.
    """
    rows = []
    for run_dir in find_runs(root):
        cfg = RunConfig.load(run_dir)
        history_path = run_dir / "history.json"
        best_val, n_epochs = float("nan"), 0
        if history_path.exists():
            with open(history_path) as f:
                hist = json.load(f)
            # Prefer val_objective: with KL annealing, val_loss is measured at
            # a beta that changes across epochs, so its minimum is not
            # comparable between runs. Older runs have only val_loss.
            curve = hist.get("val_objective") or hist.get("val_loss")
            if curve:
                best_val = min(curve)
                n_epochs = len(curve)
        rows.append({
            "run": run_dir.name,
            "seed": cfg.seed,
            "latent_dim": cfg.latent_dim,
            "lr": cfg.lr,
            "kl_weight": cfg.kl_weight,
            "predict_timing": cfg.predict_timing,
            "timing_weight": cfg.timing_weight,
            "best_val_loss": best_val,
            "epochs_trained": n_epochs,
        })
    return rows
