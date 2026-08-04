"""
Run the proposal's §4 evaluation plan against a trained CVAE.

Everything is computed on the **held-out test subjects** of the run's own seed,
which were unseen during training and validation.

    §4.1  Reconstruction MSE vs the spline baselines
    §4.2  Latent interpretability: traversal + Spearman vs kinematics
    §4.3  Behavioural probing R², leave-one-subject-out
    §4.4  Generative fidelity: KS per feature, MMD, energy distance
    §3.3  Per-subject fingerprints (latent mean and spread)

Not covered: the §4.4 benchmark against Prof. Friedman's submovement
decomposition pipeline, which is an external dependency not present in this
repository. It is reported as missing rather than approximated.

Usage
-----
    python scripts/evaluate_report.py                       # newest run
    python scripts/evaluate_report.py --run results/runs/z5_seed3_...
    python scripts/evaluate_report.py --sweep 2 3 4 8 16    # §3.3 latent sweep
"""
from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.baseline_spline import evaluate_spline_baseline, evaluate_spline_pca_baseline
from src.evaluate import (
    behavioural_probing,
    compute_fingerprints,
    compute_reconstruction_mse,
    encode_trials,
    generative_fidelity_ks,
    latent_feature_correlations,
    latent_traversal,
    timing_reconstruction_error,
    traversal_summary,
)
from src.run_config import RunConfig, find_runs, set_seed
from src.train import split_subjects_for, train_vae
from src.vae_model import ConditionalVAE, NormStats


def load_run(run_dir: Path, device: str):
    """Rebuild the model and its normalisation from a run directory."""
    cfg = RunConfig.load(run_dir)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model = ConditionalVAE(
        latent_dim=ckpt["latent_dim"],
        hidden_dim=cfg.hidden_dim,
        timing_dim=ckpt.get("timing_dim", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, NormStats.from_checkpoint(ckpt), cfg


def figure_traversal(traversal: dict, out_path: Path) -> Path:
    """One row per latent dimension: XY path, and the speed profile in real time."""
    dims = sorted(traversal["dims"])
    steps = traversal["steps"]
    fig, axes = plt.subplots(len(dims), 2, figsize=(9, 2.6 * len(dims)), dpi=170, squeeze=False)
    cmap = plt.get_cmap("viridis")

    for r, dim in enumerate(dims):
        d = traversal["dims"][dim]
        for i, traj in enumerate(d["trajectories"]):
            colour = cmap(i / max(len(steps) - 1, 1))
            axes[r][0].plot(traj[:, 0], traj[:, 1], color=colour, linewidth=1.4)
            mt = float(d["timing"][i, 0]) if d["timing"] is not None else 1.0
            mt = max(mt, 1e-3)
            speed = np.linalg.norm(np.gradient(traj, axis=0), axis=1) / (mt / (len(traj) - 1))
            axes[r][1].plot(np.linspace(0, mt, len(traj)) * 1000, speed, color=colour, linewidth=1.4)

        axes[r][0].set_title(f"z{dim}: path (x-y)", fontsize=9)
        axes[r][0].set_xlabel("x (mm)"); axes[r][0].set_ylabel("y (mm)")
        axes[r][1].set_title(f"z{dim}: speed profile", fontsize=9)
        axes[r][1].set_xlabel("time (ms)"); axes[r][1].set_ylabel("speed (mm/s)")
        for ax in axes[r]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(
        f"Latent traversal, {steps[0]:+.0f} to {steps[-1]:+.0f} prior SD "
        f"(sp={traversal['sp']}, side={traversal['side']})",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def figure_fingerprints(fp: pd.DataFrame, out_path: Path) -> Path:
    """Per-subject fingerprints in the first two latent dimensions, with spread."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=170)
    for _, r in fp.iterrows():
        ax.errorbar(
            r["z0_mean"], r["z1_mean"],
            xerr=r.get("z0_std", 0), yerr=r.get("z1_std", 0),
            fmt="o", markersize=7, capsize=3, alpha=0.85, label=r["subject"],
        )
    ax.set_xlabel("z0 (mean +/- SD across trials)")
    ax.set_ylabel("z1 (mean +/- SD across trials)")
    ax.set_title("Subject fingerprints, held-out test subjects")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def evaluate_run(run_dir: Path, trials: list[dict], device: str, out_dir: Path) -> dict:
    model, norm, cfg = load_run(run_dir, device)
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    train_trials, _, test_trials = split_subjects_for(trials, cfg)
    subj = sorted({t["metadata"]["subject"] for t in test_trials})

    print("\n" + "=" * 76)
    print(f"EVALUATION — {run_dir.name}")
    print(f"  z={model.latent_dim}, timing_dim={model.timing_dim}, seed={cfg.seed}")
    print(f"  {len(test_trials)} trials from {len(subj)} held-out subjects: {', '.join(subj)}")
    print("=" * 76)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"run": run_dir.name, "latent_dim": model.latent_dim, "seed": cfg.seed}

    # ── §4.1 Reconstruction ──────────────────────────────────────────────
    print("\n[4.1] Reconstruction MSE (mm^2)")
    vae_mse = compute_reconstruction_mse(model, test_trials, norm, device)
    pca = evaluate_spline_pca_baseline(train_trials, test_trials, n_components=3)
    per_trial = evaluate_spline_baseline(test_trials)
    print(f"  CVAE (z={model.latent_dim}, {sum(p.numel() for p in model.parameters()):,} params) : {vae_mse:.4f}")
    print(f"  Spline+PCA (3 comp, 108 params)      : {pca['mean_mse']:.4f}")
    print(f"  Spline per-trial (interp. ceiling)   : {per_trial['mean_mse']:.4f}")
    results |= {
        "vae_mse": vae_mse,
        "spline_pca_mse": pca["mean_mse"],
        "spline_per_trial_mse": per_trial["mean_mse"],
    }

    timing_df = timing_reconstruction_error(model, test_trials, norm, device)
    if not timing_df.empty:
        timing_df.to_csv(out_dir / "timing_reconstruction.csv", index=False)
        print("\n  Timing reconstruction:")
        print("   " + timing_df.to_string(index=False).replace("\n", "\n   "))
        for _, r in timing_df.iterrows():
            results[f"{r['timing_feature']}_r2"] = r["r2"]

    # ── §4.2 Latent interpretability ─────────────────────────────────────
    print("\n[4.2] Latent interpretability")
    mus, logvars, zs, subjects = encode_trials(model, test_trials, norm, device)

    trav = latent_traversal(model, norm, device=device)
    summary = traversal_summary(trav)
    summary.to_csv(out_dir / "latent_traversal_summary.csv", index=False)
    fig1 = figure_traversal(trav, out_dir / "figures" / f"traversal_{run_dir.name}.png")
    print(f"  traversal figure -> {fig1}")
    print("  what each dimension moves (range over the sweep):")
    for _, r in summary.iterrows():
        print(f"    {r['latent_dim']}: path_len {r['path_length_range']:7.2f} mm | "
              f"move_time {r['movement_time_s_range']*1000:6.1f} ms | "
              f"curvature {r['curvature_index_range']:6.3f} | "
              f"relative span {r['relative_span']:.3f}"
              + ("   <- effectively unused" if r["relative_span"] < 0.05 else ""))

    corr = latent_feature_correlations(mus, test_trials)
    corr.to_csv(out_dir / "latent_correlations.csv", index=False)
    strong = corr[(corr.p_value < 0.05) & (corr.spearman_rho.abs() > 0.3)]
    print(f"  Spearman |rho|>0.3 and p<0.05: {len(strong)} of {len(corr)} pairs")
    for _, r in strong.sort_values("spearman_rho", key=abs, ascending=False).head(8).iterrows():
        print(f"    {r['latent_dim']} ~ {r['feature']:22s} rho={r['spearman_rho']:+.3f}")

    # ── §3.3 Fingerprints ────────────────────────────────────────────────
    fp = compute_fingerprints(mus, subjects)
    fp.to_csv(out_dir / "fingerprints.csv", index=False)
    fig2 = figure_fingerprints(fp, out_dir / "figures" / f"fingerprints_{run_dir.name}.png")
    # Between- vs within-subject spread: >1 means subjects separate.
    between = fp[[c for c in fp if c.endswith("_mean")]].std().mean()
    within = fp[[c for c in fp if c.endswith("_std")]].mean().mean()
    print(f"\n[3.3] Fingerprints -> {fig2}")
    print(f"  between-subject SD {between:.3f} vs within-subject SD {within:.3f} "
          f"(ratio {between/max(within,1e-9):.2f}; >1 means subjects separate)")
    results |= {"fingerprint_between_sd": between, "fingerprint_within_sd": within,
                "fingerprint_ratio": between / max(within, 1e-9)}

    # ── §4.3 Behavioural probing ─────────────────────────────────────────
    print("\n[4.3] Behavioural probing R^2 (leave-one-subject-out)")
    probe = behavioural_probing(mus, test_trials, subjects)
    probe.to_csv(out_dir / "probing_r2.csv", index=False)
    print("   " + probe.to_string(index=False).replace("\n", "\n   "))
    print(f"  R^2 < 0 means worse than predicting the mean of the other "
          f"{probe['n_subjects'].iloc[0]-1} subjects.")
    for _, r in probe.iterrows():
        results[f"probe_{r['target']}_r2_linear"] = r["r2_linear_loso"]

    # ── §4.4 Generative fidelity ─────────────────────────────────────────
    print("\n[4.4] Generative fidelity (per test subject)")
    ks = generative_fidelity_ks(model, test_trials, norm, device=device)
    gf = pd.DataFrame(ks).T.reset_index().rename(columns={"index": "subject"})
    gf.to_csv(out_dir / "generative_fidelity.csv", index=False)
    print(f"  {'subject':<12} {'KS rejected':>12} {'MMD':>9} {'p':>7} {'energy':>9} {'p':>7}")
    for _, r in gf.iterrows():
        print(f"  {r['subject']:<12} {int(r['n_features_rejected']):>5}/{int(r['n_features']):<6} "
              f"{r['mmd_rbf']:>9.4f} {r['mmd_pvalue']:>7.3f} "
              f"{r['energy_distance']:>9.4f} {r['energy_pvalue']:>7.3f}")
    n_match = int((gf["mmd_pvalue"] > 0.05).sum())
    print(f"  MMD cannot distinguish generated from empirical for {n_match}/{len(gf)} subjects")
    print(f"  (mean {gf['n_features_rejected'].mean():.1f} of {int(gf['n_features'].iloc[0])} "
          f"features rejected by KS at p<0.05)")
    results |= {
        "mmd_mean": gf["mmd_rbf"].mean(),
        "energy_mean": gf["energy_distance"].mean(),
        "subjects_mmd_indistinguishable": n_match,
        "n_test_subjects": len(gf),
        "ks_features_rejected_mean": gf["n_features_rejected"].mean(),
    }

    print("\n[4.4] Submovement-decomposition benchmark: NOT RUN "
          "(external dependency, not in this repository)")
    return results


def main():
    ap = argparse.ArgumentParser(description="Run the proposal's §4 evaluation plan")
    ap.add_argument("--run", type=str, default=None, help="Run directory to evaluate")
    ap.add_argument("--sweep", type=int, nargs="+", default=None,
                    help="Train and evaluate across these latent dims (§3.3)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out) if args.out else config.RESULTS_DIR / "evaluation"

    processed = config.DATA_PROCESSED_DIR / "trials.pkl"
    if not processed.exists():
        raise SystemExit(f"{processed} not found. Run: python scripts/make_dataset.py")
    with open(processed, "rb") as f:
        trials = pickle.load(f)

    if args.sweep:
        # §3.3: sweep latent dimensionality, evaluating each with the full plan.
        base = RunConfig(seed=args.seed, epochs=args.epochs)
        rows = []
        for d in args.sweep:
            cfg = replace(base, latent_dim=d)
            tr, va, _ = split_subjects_for(trials, cfg)
            _, _, run_dir = train_vae(tr, va, cfg=cfg, device=device)
            rows.append(evaluate_run(run_dir, trials, device, out_dir / f"z{d}"))
        df = pd.DataFrame(rows)
        path = out_dir / "latent_dim_sweep.csv"
        df.to_csv(path, index=False)
        print("\n" + "=" * 76)
        print("LATENT-DIM SWEEP")
        print("=" * 76)
        cols = ["latent_dim", "vae_mse", "spline_pca_mse", "fingerprint_ratio",
                "ks_features_rejected_mean", "subjects_mmd_indistinguishable"]
        print(df[[c for c in cols if c in df]].to_string(index=False))
        print(f"\nSaved to {path}")
        return

    run_dir = Path(args.run) if args.run else (find_runs()[-1] if find_runs() else None)
    if run_dir is None:
        raise SystemExit("No runs found. Train first: python main.py --phase 3")
    res = evaluate_run(run_dir, trials, device, out_dir)
    pd.DataFrame([res]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
