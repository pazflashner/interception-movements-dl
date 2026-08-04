"""
Phase 2/3 baseline report: CVAE reconstruction against the spline baselines.

For each seed the subject split, the CVAE and both spline references are all
recomputed, so every row is a self-contained comparison on that seed's held-out
subjects. Writes ``results/vae_vs_baseline.csv`` and prints a paired summary.

Why paired, and why several seeds
---------------------------------
With 28 subjects (17/4/7) a single run is noisy: changing the seed changes which
7 subjects are held out, which moves both the CVAE and the baseline. Comparing
them within a seed cancels most of that, so the paired difference is far more
informative than either column on its own.

What is being compared
----------------------
    CVAE          latent_dim numbers per trial, decoder shared across subjects,
                  test subjects never seen during training
    Spline+PCA    the same latent_dim numbers per trial, PCA basis fitted on the
                  training subjects only — the capacity-matched reference
    Spline        27 free parameters per trial, fitted to the trial it
                  reconstructs; an interpolation ceiling, not a representation

Usage
-----
    python scripts/baseline_report.py
    python scripts/baseline_report.py --seeds 0 1 2 3 4 --latent-dim 3
    python scripts/baseline_report.py --epochs 60        # quicker smoke run
"""
from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.baseline_spline import evaluate_spline_baseline, evaluate_spline_pca_baseline
from src.evaluate import compute_reconstruction_mse, timing_reconstruction_error
from src.run_config import RunConfig
from src.train import split_subjects, train_vae
from src.vae_model import NormStats


def run_seed(
    trials: list[dict],
    seed: int,
    cfg: RunConfig,
    device: str,
    pca_components: int | None = None,
) -> dict:
    """
    Train and score one seed; returns a single result row.

    ``pca_components`` fixes the baseline's dimensionality independently of the
    CVAE's. That matters when the CVAE also reconstructs timing: a z=5 model
    spending ~2 dims on timing has the same *shape* budget as a 3-component PCA,
    so comparing it against PCA at 5 components would understate it.
    """
    train_trials, val_trials, test_trials = split_subjects(trials, seed=seed)
    run_cfg = replace(cfg, seed=seed)

    model, history, run_dir = train_vae(
        train_trials, val_trials, cfg=run_cfg, device=device
    )
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    norm = NormStats.from_checkpoint(ckpt)

    vae_mse = compute_reconstruction_mse(model, test_trials, norm, device)
    pca = evaluate_spline_pca_baseline(
        train_trials, test_trials, n_components=pca_components or run_cfg.latent_dim
    )
    per_trial = evaluate_spline_baseline(test_trials)
    timing = timing_reconstruction_error(model, test_trials, norm, device)

    row = {
        "seed": seed,
        "latent_dim": run_cfg.latent_dim,
        "predict_timing": run_cfg.predict_timing,
        "pca_components": pca_components or run_cfg.latent_dim,
        "n_test_subjects": len({t["metadata"]["subject"] for t in test_trials}),
        "n_test_trials": len(test_trials),
        "vae_mse": vae_mse,
        "spline_pca_mse": pca["mean_mse"],
        "spline_per_trial_mse": per_trial["mean_mse"],
        "vae_beats_pca": bool(vae_mse < pca["mean_mse"]),
        "improvement_pct": 100 * (pca["mean_mse"] - vae_mse) / pca["mean_mse"],
        "epochs_trained": len(history["val_loss"]),
        "best_epoch": ckpt["epoch"],
        "run_dir": run_dir.name,
    }
    for _, t in timing.iterrows():
        row[f"{t['timing_feature']}_r2"] = t["r2"]
        row[f"{t['timing_feature']}_mae_ms"] = t["mae_ms"]
    return row


def main():
    ap = argparse.ArgumentParser(description="CVAE vs spline baselines across seeds")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--latent-dim", type=int, default=config.DEFAULT_LATENT_DIM)
    ap.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--no-timing", action="store_true",
        help="Shape-only CVAE (predict_timing=False), isolating trajectory reconstruction",
    )
    ap.add_argument(
        "--pca-components", type=int, default=None,
        help="Fix the baseline's dimensionality (default: match latent_dim)",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed = config.DATA_PROCESSED_DIR / "trials.pkl"
    if not processed.exists():
        raise SystemExit(f"{processed} not found. Run: python scripts/make_dataset.py")
    with open(processed, "rb") as f:
        trials = pickle.load(f)

    cfg = RunConfig(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        predict_timing=not args.no_timing,
    )
    rows = [
        run_seed(trials, s, cfg, device, pca_components=args.pca_components)
        for s in args.seeds
    ]
    df = pd.DataFrame(rows)

    out = Path(args.out) if args.out else config.RESULTS_DIR / "vae_vs_baseline.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # ── Summary ──
    diff = df["spline_pca_mse"] - df["vae_mse"]     # positive = CVAE better
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0

    print("\n" + "=" * 74)
    print(f"RECONSTRUCTION MSE (mm^2) on held-out subjects, z={args.latent_dim}, "
          f"{len(args.seeds)} seeds")
    print("=" * 74)
    cols = ["seed", "vae_mse", "spline_pca_mse", "spline_per_trial_mse", "improvement_pct"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    print(f"\n  CVAE            {df['vae_mse'].mean():.4f} +/- {df['vae_mse'].std(ddof=1):.4f}")
    print(f"  Spline+PCA      {df['spline_pca_mse'].mean():.4f} +/- {df['spline_pca_mse'].std(ddof=1):.4f}   (capacity-matched)")
    print(f"  Spline per-trial{df['spline_per_trial_mse'].mean():9.4f}                (interpolation ceiling)")
    print(f"\n  Paired difference (PCA - CVAE): {diff.mean():+.4f} +/- {sd:.4f}")
    print(f"  CVAE wins on {int(df['vae_beats_pca'].sum())} of {len(df)} seeds "
          f"(mean improvement {df['improvement_pct'].mean():.1f}%)")

    if len(diff) >= 5:
        from scipy import stats
        w = stats.wilcoxon(df["vae_mse"], df["spline_pca_mse"])
        print(f"  Wilcoxon signed-rank over {len(diff)} paired seeds: "
              f"W = {w.statistic:.1f}, p = {w.pvalue:.4g}")

    tcols = [c for c in df.columns if c.endswith("_r2")]
    if tcols:
        print("\n  Timing reconstruction R2 (test subjects):")
        for c in tcols:
            print(f"    {c[:-3]:20s} {df[c].mean():.3f} +/- {df[c].std(ddof=1):.3f}")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
