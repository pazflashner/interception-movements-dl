"""
Run the proposal's §4 evaluation on the CVAE *and* on the spline representation.

The proposal evaluates the CVAE alone. That answers "is the CVAE any good?" but
not "is the CVAE worth it?" — for which the same battery has to be applied to a
baseline representation of the same code length. Every metric below is computed
by shared code, on the same held-out subjects, from codes of the same width:

    CVAE                latent_dim numbers/trial, ~2.9e5 shared parameters,
                        non-linear, stochastic, KL-regularised
    Spline+PCA          latent_dim numbers/trial, ~1e2 shared parameters,
                        linear, deterministic, no prior

Both encode shape *and* timing, so neither is solving an easier problem.

Usage
-----
    python scripts/compare_representations.py --seeds 0 1 2 3 4
    python scripts/compare_representations.py --seeds 0 --latent-dim 3
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.baseline_spline import SplinePCARepresentation
from src.evaluate import (
    behavioural_probing,
    compare_feature_distributions,
    compute_fingerprints,
    encode_trials,
    latent_feature_correlations,
    reconstruct,
)
from src.features import features_from_arrays
from src.run_config import RunConfig
from src.train import split_subjects, train_vae
from src.vae_model import ConditionalVAE, NormStats
from sklearn.metrics import r2_score


def _fingerprint_ratio(codes: np.ndarray, subjects: list[str]) -> float:
    """Between-subject spread / within-subject spread. >1 means subjects separate."""
    fp = compute_fingerprints(codes, subjects)
    between = fp[[c for c in fp if c.endswith("_mean")]].std().mean()
    within = fp[[c for c in fp if c.endswith("_std")]].mean().mean()
    return float(between / max(within, 1e-9))


def _generative_scores(
    codes: np.ndarray,
    subjects: list[str],
    trials: list[dict],
    sample_fn,
    decode_fn,
    n_samples: int,
    seed: int,
) -> dict:
    """Mean KS rejections / MMD / energy over subjects, for either representation."""
    from src.features import compute_trial_features

    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects)
    rows = []
    for subj in np.unique(subjects):
        mask = subjects == subj
        subj_trials = [t for t, m in zip(trials, mask) if m]
        emp = pd.DataFrame([compute_trial_features(t) for t in subj_trials])

        drawn = sample_fn(codes[mask], n_samples, rng)
        trajs, timing = decode_fn(drawn, subj_trials, rng)
        gen = pd.DataFrame([
            features_from_arrays(trajs[i], max(float(timing[i, 0]), 1e-3), float(timing[i, 1]))
            for i in range(len(trajs))
        ])
        rows.append(compare_feature_distributions(emp, gen, n_permutations=200, seed=seed))

    df = pd.DataFrame(rows)
    return {
        "ks_rejected": df["n_features_rejected"].mean(),
        "n_features": int(df["n_features"].iloc[0]),
        "mmd": df["mmd_rbf"].mean(),
        "energy": df["energy_distance"].mean(),
        "n_indistinguishable": int((df["mmd_pvalue"] > 0.05).sum()),
        "n_subjects": len(df),
    }


def evaluate_representation(
    name: str,
    codes: np.ndarray,
    recon_traj: np.ndarray,
    true_traj: np.ndarray,
    recon_timing: np.ndarray,
    true_timing: np.ndarray,
    trials: list[dict],
    subjects: list[str],
    gen: dict,
    n_params: int,
) -> dict:
    """Assemble one row of the comparison from already-computed pieces."""
    row = {
        "representation": name,
        "code_dim": codes.shape[1],
        "shared_params": n_params,
        "recon_mse": float(np.mean((recon_traj - true_traj) ** 2)),
        "movement_time_r2": r2_score(true_timing[:, 0], recon_timing[:, 0]),
        "initiation_time_r2": r2_score(true_timing[:, 1], recon_timing[:, 1]),
        "fingerprint_ratio": _fingerprint_ratio(codes, subjects),
    }

    probe = behavioural_probing(codes, trials, subjects)
    for _, r in probe.iterrows():
        # Keep the better of the two probe families per target: the proposal
        # asks whether the fingerprint *can* predict the behaviour, not which
        # regressor happens to suit 6 training points.
        row[f"probe_{r['target']}"] = max(r["r2_linear_loso"], r["r2_svr_loso"])
        row[f"probe_{r['target']}_linear"] = r["r2_linear_loso"]
    row["probe_mean_r2"] = probe[["r2_linear_loso", "r2_svr_loso"]].max(axis=1).mean()
    # R2 is unbounded below, so a single collapsed fold can dominate a mean.
    # The count of targets predicted better than the subject mean is the robust
    # summary of how much behaviour the fingerprint actually carries.
    row["probe_n_positive"] = int(
        (probe[["r2_linear_loso", "r2_svr_loso"]].max(axis=1) > 0).sum()
    )
    row["probe_n_targets"] = len(probe)

    corr = latent_feature_correlations(codes, trials)
    row["max_abs_spearman"] = corr["spearman_rho"].abs().max()
    row["n_strong_corr"] = int(((corr.p_value < 0.05) & (corr.spearman_rho.abs() > 0.3)).sum())

    row |= {
        "ks_rejected": gen["ks_rejected"],
        "mmd": gen["mmd"],
        "energy": gen["energy"],
        "subjects_indistinguishable": gen["n_indistinguishable"],
        "n_test_subjects": gen["n_subjects"],
    }
    return row


def run_seed(trials: list[dict], seed: int, latent_dim: int, epochs: int, device: str) -> list[dict]:
    train_trials, val_trials, test_trials = split_subjects(trials, seed=seed)
    subjects = [t["metadata"]["subject"] for t in test_trials]
    n_samples = 100

    # ── CVAE ─────────────────────────────────────────────────────────────
    cfg = RunConfig(seed=seed, latent_dim=latent_dim, epochs=epochs)
    model, _, run_dir = train_vae(train_trials, val_trials, cfg=cfg, device=device)
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    norm = NormStats.from_checkpoint(ckpt)

    rec, true, rec_t, true_t = reconstruct(model, test_trials, norm, device)
    T = config.NORMALISED_LENGTH
    mus, logvars, _, _ = encode_trials(model, test_trials, norm, device)

    def vae_sample(subj_codes, n, rng):
        mean = subj_codes.mean(0)
        sd = np.sqrt(subj_codes.var(0) + 1e-9)
        return rng.normal(mean, sd, size=(n, subj_codes.shape[1]))

    def vae_decode(drawn, subj_trials, rng):
        from src.vae_model import encode_condition
        conds = np.stack([
            encode_condition(t["metadata"].get("sp", 1), t["metadata"].get("side", 1))
            for t in subj_trials
        ])
        c = torch.tensor(conds[rng.integers(0, len(conds), len(drawn))],
                         dtype=torch.float32, device=device)
        tm, ts, tim_m, tim_s = norm.torch(device)
        with torch.no_grad():
            rz, tz = model.decode(torch.tensor(drawn, dtype=torch.float32, device=device), c)
            trajs = (rz * ts + tm).cpu().numpy().reshape(-1, T, 3)
            timing = (tz * tim_s + tim_m).cpu().numpy()
        return trajs, timing

    vae_gen = _generative_scores(mus, subjects, test_trials, vae_sample, vae_decode, n_samples, seed)
    rows = [evaluate_representation(
        "CVAE", mus, rec.reshape(-1, T, 3), true.reshape(-1, T, 3), rec_t, true_t,
        test_trials, subjects, vae_gen, sum(p.numel() for p in model.parameters()),
    )]

    # ── Spline + PCA ─────────────────────────────────────────────────────
    rep = SplinePCARepresentation(n_components=latent_dim).fit(train_trials)
    codes = rep.encode(test_trials)
    strajs, stiming = rep.decode(codes)
    n_params = rep.pca_.components_.size + rep.pca_.mean_.size

    def pca_sample(subj_codes, n, rng):
        return rep.sample_subject(subj_codes, n, rng)

    def pca_decode(drawn, subj_trials, rng):
        return rep.decode(drawn)

    pca_gen = _generative_scores(codes, subjects, test_trials, pca_sample, pca_decode, n_samples, seed)
    rows.append(evaluate_representation(
        "Spline+PCA", codes, strajs, true.reshape(-1, T, 3), stiming, true_t,
        test_trials, subjects, pca_gen, n_params,
    ))

    # ── Random-code control ──────────────────────────────────────────────
    # A floor for the probing and fingerprint metrics. R2 is unbounded below and
    # LOSO over 7 subjects is volatile, so "R2 = -4.6" means nothing without
    # knowing what pure noise scores on the same 7 subjects. Only the metrics
    # that do not need a decoder are defined here.
    rng = np.random.default_rng(seed)
    rand_codes = rng.normal(size=(len(test_trials), latent_dim))
    rand_probe = behavioural_probing(rand_codes, test_trials, subjects)
    rand_best = rand_probe[["r2_linear_loso", "r2_svr_loso"]].max(axis=1)
    rand_row = {
        "representation": "Random codes",
        "code_dim": latent_dim,
        "shared_params": 0,
        "fingerprint_ratio": _fingerprint_ratio(rand_codes, subjects),
        "probe_mean_r2": rand_best.mean(),
        "probe_n_positive": int((rand_best > 0).sum()),
        "probe_n_targets": len(rand_probe),
        "max_abs_spearman": latent_feature_correlations(rand_codes, test_trials)["spearman_rho"].abs().max(),
    }
    for t, v in zip(rand_probe["target"], rand_best):
        rand_row[f"probe_{t}"] = v
    rows.append(rand_row)

    for r in rows:
        r["seed"] = seed
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate CVAE and spline on the same §4 battery")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--latent-dim", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(config.DATA_PROCESSED_DIR / "trials.pkl", "rb") as f:
        trials = pickle.load(f)

    rows = []
    for s in args.seeds:
        rows += run_seed(trials, s, args.latent_dim, args.epochs, device)
    df = pd.DataFrame(rows)

    out = Path(args.out) if args.out else config.RESULTS_DIR / "representation_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    metrics = [
        ("recon_mse", "Recon MSE (mm^2)", "lower"),
        ("movement_time_r2", "Movement time R^2", "higher"),
        ("initiation_time_r2", "Initiation time R^2", "higher"),
        ("fingerprint_ratio", "Fingerprint between/within", "higher"),
        ("probe_n_positive", "Targets with R^2 > 0 (of 11)", "higher"),
        ("probe_initiation_time_s", "  R^2 initiation time", "higher"),
        ("probe_movement_time_s", "  R^2 movement time", "higher"),
        ("probe_curvature_index", "  R^2 curvature", "higher"),
        ("probe_peak_speed_mm_s", "  R^2 peak speed", "higher"),
        ("probe_time_to_peak_speed", "  R^2 time-to-peak", "higher"),
        ("probe_path_length", "  R^2 path length", "higher"),
        ("probe_max_lateral_deviation", "  R^2 lateral deviation", "higher"),
        ("probe_movement_time_s_sd", "  R^2 movement-time SD", "higher"),
        ("probe_curvature_index_sd", "  R^2 curvature SD", "higher"),
        ("max_abs_spearman", "Max |Spearman| vs kinematics", "higher"),
        ("ks_rejected", "KS features rejected (of 11)", "lower"),
        ("mmd", "MMD", "lower"),
        ("energy", "Energy distance", "lower"),
        ("subjects_indistinguishable", "Subjects MMD-matched (of 7)", "higher"),
    ]
    g = df.groupby("representation")

    from scipy import stats as _st
    piv = df.pivot(index="seed", columns="representation")

    print("\n" + "=" * 96)
    print(f"§4 EVALUATION — CVAE vs SPLINE, {len(args.seeds)} seeds, "
          f"code width {args.latent_dim}, held-out subjects")
    print("=" * 96)
    print(f"  {'metric':<32} {'CVAE':>15} {'Spline+PCA':>15} {'Random':>9} {'wins':>6} {'p':>8}")
    print("  " + "-" * 92)
    for key, label, direction in metrics:
        if key not in df:
            continue
        v, sd = g[key].mean(), g[key].std(ddof=1).fillna(0.0)
        cv, pv = v.get("CVAE", np.nan), v.get("Spline+PCA", np.nan)
        rv = v.get("Random codes", np.nan)

        wins, pstr = "", ""
        if key in piv and "CVAE" in piv[key] and "Spline+PCA" in piv[key]:
            a, b = piv[key]["CVAE"].values, piv[key]["Spline+PCA"].values
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 3:
                a, b = a[ok], b[ok]
                n_win = int((a > b).sum() if direction == "higher" else (a < b).sum())
                wins = f"{n_win}/{len(a)}"
                try:
                    pstr = f"{_st.wilcoxon(a, b).pvalue:.4f}"
                except ValueError:
                    pstr = "-"

        rand_s = f"{rv:>9.3f}" if np.isfinite(rv) else f"{'-':>9}"
        print(f"  {label:<32} {cv:>8.3f}+/-{sd.get('CVAE',0):<5.3f} "
              f"{pv:>8.3f}+/-{sd.get('Spline+PCA',0):<5.3f} {rand_s} {wins:>6} {pstr:>8}")

    print(f"\n  shared parameters: CVAE {int(g['shared_params'].first()['CVAE']):,} "
          f"vs Spline+PCA {int(g['shared_params'].first()['Spline+PCA']):,} vs Random 0")
    print("  'Random' = codes drawn from N(0,I) of the same width: the floor for the")
    print("  probing and fingerprint metrics, which are volatile over 7 subjects.")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
