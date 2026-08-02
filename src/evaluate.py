"""
Evaluation & behavioural analysis.

Runs on held-out test subjects using the trained CVAE encoder:
- Reconstruction MSE (vs spline baseline)
- Timing reconstruction (movement / initiation time, in ms)
- Latent space interpretability (correlations with kinematics)
- Behavioural probing (R² via linear / SVR)
- Generative fidelity (KS test on shape *and* on movement time)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.vae_model import (
    ConditionalVAE,
    NormStats,
    TrajectoryDataset,
    encode_condition,
)
from src.features import compute_trial_features


# ── Encode trials ─────────────────────────────────────────────────────────────
def encode_trials(
    model: ConditionalVAE,
    trials: list[dict],
    norm: NormStats,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Encode trials through the CVAE encoder.

    Returns (mu, logvar, z, subjects).
    """
    model.eval()
    ds = TrajectoryDataset(trials)
    tm, ts, tim_m, tim_s = norm.torch(device)

    traj = torch.from_numpy(ds.trajectories).to(device)
    timing = torch.from_numpy(ds.timings).to(device)
    cond = torch.from_numpy(ds.conditions).to(device)

    with torch.no_grad():
        traj_z = (traj - tm) / ts
        timing_z = (timing - tim_m) / tim_s if model.timing_dim else None
        mu, logvar = model.encode(traj_z, cond, timing_z)
        z = model.reparameterize(mu, logvar)

    return mu.cpu().numpy(), logvar.cpu().numpy(), z.cpu().numpy(), ds.subjects


# ── Subject fingerprints ─────────────────────────────────────────────────────
def compute_fingerprints(
    mus: np.ndarray, subjects: list[str]
) -> pd.DataFrame:
    """Aggregate per-subject latent distributions (mean & std of mu)."""
    df = pd.DataFrame(mus, columns=[f"z{i}" for i in range(mus.shape[1])])
    df["subject"] = subjects
    grouped = df.groupby("subject")
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std().add_suffix("_std")
    counts = grouped.size().rename("n_trials")
    return pd.concat([means, stds, counts], axis=1).reset_index()


# ── Reconstruction ───────────────────────────────────────────────────────────
def reconstruct(
    model: ConditionalVAE,
    trials: list[dict],
    norm: NormStats,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pass trials through the full model.

    Returns (recon_traj, true_traj, recon_timing_s, true_timing_s), all in
    original units — millimetres for the trajectory, seconds for the timing.
    ``recon_timing_s`` is empty when the model has no timing head.
    """
    model.eval()
    ds = TrajectoryDataset(trials)
    tm, ts, tim_m, tim_s = norm.torch(device)

    traj = torch.from_numpy(ds.trajectories).to(device)
    timing = torch.from_numpy(ds.timings).to(device)
    cond = torch.from_numpy(ds.conditions).to(device)

    with torch.no_grad():
        traj_z = (traj - tm) / ts
        timing_z = (timing - tim_m) / tim_s if model.timing_dim else None
        recon_z, recon_timing_z, _, _, _ = model(traj_z, cond, timing_z)
        recon = recon_z * ts + tm
        recon_timing = (
            (recon_timing_z * tim_s + tim_m).cpu().numpy()
            if recon_timing_z is not None
            else np.empty((len(ds), 0), dtype=np.float32)
        )

    return recon.cpu().numpy(), ds.trajectories, recon_timing, ds.timings


def compute_reconstruction_mse(
    model: ConditionalVAE,
    trials: list[dict],
    norm: NormStats,
    device: str = "cpu",
) -> float:
    """Mean per-trial trajectory reconstruction MSE, in original (mm²) scale."""
    recon, true, _, _ = reconstruct(model, trials, norm, device)
    return float(np.mean((recon - true) ** 2, axis=1).mean())


def timing_reconstruction_error(
    model: ConditionalVAE,
    trials: list[dict],
    norm: NormStats,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    How well the timing head recovers the temporal axis that resampling removes.

    Reported in milliseconds, next to the spread of the true values: an MAE
    comparable to the SD means the head is predicting little more than the mean.
    """
    _, _, recon_timing, true_timing = reconstruct(model, trials, norm, device)
    if recon_timing.shape[1] == 0:
        return pd.DataFrame()

    rows = []
    for i, name in enumerate(config.TIMING_FEATURES):
        pred, true = recon_timing[:, i], true_timing[:, i]
        rows.append({
            "timing_feature": name,
            "mae_ms": float(np.mean(np.abs(pred - true)) * 1000),
            "rmse_ms": float(np.sqrt(np.mean((pred - true) ** 2)) * 1000),
            "true_mean_ms": float(np.mean(true) * 1000),
            "true_sd_ms": float(np.std(true) * 1000),
            "r2": float(r2_score(true, pred)),
        })
    return pd.DataFrame(rows)


# ── Latent-kinematics correlation ─────────────────────────────────────────────
def latent_feature_correlations(
    mus: np.ndarray,
    trials: list[dict],
) -> pd.DataFrame:
    """
    Compute Spearman correlations between each latent dimension and
    kinematic features.
    """
    features_list = [compute_trial_features(t) for t in trials]
    feat_df = pd.DataFrame(features_list)

    numeric_feats = [
        "initiation_time_s", "movement_time_s",
        "peak_speed_mm_s", "time_to_peak_speed",
        "path_length", "curvature_index", "max_lateral_deviation",
    ]
    corr_rows = []
    for zi in range(mus.shape[1]):
        for feat in numeric_feats:
            rho, pval = stats.spearmanr(mus[:, zi], feat_df[feat].values)
            corr_rows.append({
                "latent_dim": f"z{zi}",
                "feature": feat,
                "spearman_rho": rho,
                "p_value": pval,
            })
    return pd.DataFrame(corr_rows)


# ── Behavioural probing (R²) ─────────────────────────────────────────────────
def behavioural_probing(
    mus: np.ndarray,
    trials: list[dict],
    subjects: list[str],
) -> pd.DataFrame:
    """
    Fit linear and SVR probes from subject fingerprints to predict
    macro-level behavioural metrics.
    """
    feat_list = [compute_trial_features(t) for t in trials]
    feat_df = pd.DataFrame(feat_list)
    feat_df["subject"] = subjects

    targets = [
        "initiation_time_s", "movement_time_s",
        "peak_speed_mm_s", "curvature_index",
    ]

    # Per-subject averages
    subj_feats = feat_df.groupby("subject")[targets].mean()

    # Subject fingerprints
    fp = compute_fingerprints(mus, subjects)
    fp = fp.set_index("subject")
    z_cols = [c for c in fp.columns if c.endswith("_mean")]

    common = subj_feats.index.intersection(fp.index)
    X = fp.loc[common, z_cols].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    results = []
    for target in targets:
        y = subj_feats.loc[common, target].values
        # Linear
        lr = LinearRegression().fit(X_sc, y)
        r2_lin = r2_score(y, lr.predict(X_sc))
        # SVR
        svr = SVR(kernel="rbf").fit(X_sc, y)
        r2_svr = r2_score(y, svr.predict(X_sc))
        results.append({
            "target": target,
            "r2_linear": r2_lin,
            "r2_svr": r2_svr,
        })

    return pd.DataFrame(results)


# ── Generative fidelity ──────────────────────────────────────────────────────
def generative_fidelity_ks(
    model: ConditionalVAE,
    trials: list[dict],
    norm: NormStats,
    n_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """
    For each test subject, sample from the learned latent distribution and
    compare generated against empirical distributions with a KS test — on path
    length (shape) and, when the model has a timing head, on movement time.

    The movement-time test is the one that says whether the model reproduces
    the *temporal* spread of a subject's movements, not just their geometry.
    """
    model.eval()
    mus, logvars, _, subjects = encode_trials(model, trials, norm, device)
    _, _, tim_m, tim_s = norm.torch(device)

    unique_subjects = np.unique(subjects)
    ks_results = {}

    for subj in unique_subjects:
        mask = np.array(subjects) == subj
        subj_mus = mus[mask]
        subj_logvars = logvars[mask]

        # Empirical features
        subj_trials = [t for t, s in zip(trials, subjects) if s == subj]
        emp_feats = pd.DataFrame([compute_trial_features(t) for t in subj_trials])

        # Generate samples from aggregated posterior
        agg_mu = subj_mus.mean(axis=0)
        agg_std = np.sqrt(np.exp(subj_logvars).mean(axis=0) + subj_mus.var(axis=0))

        # Use first trial's condition for generation
        meta0 = subj_trials[0]["metadata"]
        cond = torch.tensor(
            encode_condition(meta0.get("sp", 1), meta0.get("side", 1)),
            dtype=torch.float32
        ).to(device).unsqueeze(0).repeat(n_samples, 1)

        z = torch.tensor(
            np.random.randn(n_samples, *agg_mu.shape) * agg_std + agg_mu,
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            recon_z, recon_timing_z = model.decode(z, cond)
            tm, ts, _, _ = norm.torch(device)
            gen = (recon_z * ts + tm).cpu().numpy()
            gen_trajs = gen.reshape(n_samples, config.NORMALISED_LENGTH, 3)
            gen_timing = (
                (recon_timing_z * tim_s + tim_m).cpu().numpy()
                if recon_timing_z is not None
                else None
            )

        # KS test on path length (shape)
        emp_pl = emp_feats["path_length"].values
        gen_pl = np.array([np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)) for t in gen_trajs])
        ks_stat, ks_p = stats.ks_2samp(emp_pl, gen_pl)
        entry = {"ks_stat": ks_stat, "ks_pvalue": ks_p}

        # KS test on movement time (temporal)
        if gen_timing is not None:
            mt_idx = config.TIMING_FEATURES.index("movement_time_s")
            ks_t, ks_tp = stats.ks_2samp(
                emp_feats["movement_time_s"].values, gen_timing[:, mt_idx]
            )
            entry["ks_stat_movement_time"] = ks_t
            entry["ks_pvalue_movement_time"] = ks_tp

        ks_results[subj] = entry

    return ks_results


# ── Full evaluation ──────────────────────────────────────────────────────────
def run_full_evaluation(
    model: ConditionalVAE,
    test_trials: list[dict],
    norm: NormStats,
    spline_mse: float,
    device: str = "cpu",
    save_dir: Path | None = None,
) -> dict:
    """Run all evaluation metrics and print summary."""
    save_dir = Path(save_dir or config.RESULTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    # 1. Reconstruction MSE
    vae_mse = compute_reconstruction_mse(model, test_trials, norm, device)
    print(f"\nReconstruction MSE - VAE: {vae_mse:.6f} | Spline: {spline_mse:.6f}")

    # 2. Timing reconstruction — the axis temporal normalisation removes
    timing_df = timing_reconstruction_error(model, test_trials, norm, device)
    if not timing_df.empty:
        timing_df.to_csv(save_dir / "timing_reconstruction.csv", index=False)
        print("\nTiming reconstruction (test subjects):")
        print(timing_df.to_string(index=False))
    else:
        print("\nTiming reconstruction: model has no timing head (shape only).")

    # 3. Encode test trials
    mus, logvars, zs, subjects = encode_trials(model, test_trials, norm, device)

    # 4. Fingerprints
    fp = compute_fingerprints(mus, subjects)
    fp.to_csv(save_dir / "fingerprints.csv", index=False)
    print(f"\nSubject fingerprints saved ({len(fp)} subjects)")

    # 5. Latent-kinematics correlations
    corr_df = latent_feature_correlations(mus, test_trials)
    corr_df.to_csv(save_dir / "latent_correlations.csv", index=False)
    sig = corr_df[corr_df["p_value"] < 0.05]
    print(f"\nSignificant latent-feature correlations: {len(sig)} / {len(corr_df)}")
    if len(sig) > 0:
        print(sig.to_string(index=False))

    # 6. Behavioural probing
    probe_df = behavioural_probing(mus, test_trials, subjects)
    probe_df.to_csv(save_dir / "probing_r2.csv", index=False)
    print("\nBehavioural probing R2:")
    print(probe_df.to_string(index=False))

    # 7. Generative fidelity
    ks = generative_fidelity_ks(model, test_trials, norm, device=device)
    print("\nGenerative fidelity (KS test):")
    for subj, r in ks.items():
        line = f"  {subj}: path_length KS={r['ks_stat']:.4f}, p={r['ks_pvalue']:.4f}"
        if "ks_stat_movement_time" in r:
            line += (
                f" | movement_time KS={r['ks_stat_movement_time']:.4f}, "
                f"p={r['ks_pvalue_movement_time']:.4f}"
            )
        print(line)

    return {
        "vae_mse": vae_mse,
        "spline_mse": spline_mse,
        "timing": timing_df,
        "fingerprints": fp,
        "correlations": corr_df,
        "probing": probe_df,
        "generative_ks": ks,
    }
