"""
Evaluation & behavioural analysis.

Runs on held-out test subjects using the trained CVAE encoder:
- Reconstruction MSE (vs spline baseline)
- Latent space interpretability (correlations with kinematics)
- Behavioural probing (R² via linear / SVR)
- Generative fidelity (KS test, MMD)
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
from src.vae_model import ConditionalVAE, TrajectoryDataset, encode_condition
from src.features import compute_trial_features


# ── Encode trials ─────────────────────────────────────────────────────────────
def encode_trials(
    model: ConditionalVAE,
    trials: list[dict],
    train_mean: np.ndarray,
    train_std: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Encode trials through the CVAE encoder.

    Returns (mu, logvar, z, subjects).
    """
    model.eval()
    ds = TrajectoryDataset(trials)
    tm = torch.tensor(train_mean, dtype=torch.float32).to(device)
    ts = torch.tensor(train_std, dtype=torch.float32).to(device)

    mus, logvars, zs = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            traj, cond, _ = ds[i]
            traj = traj.to(device).unsqueeze(0)
            cond = cond.to(device).unsqueeze(0)
            traj_z = (traj - tm) / ts
            mu, logvar = model.encode(traj_z, cond)
            z = model.reparameterize(mu, logvar)
            mus.append(mu.cpu().numpy()[0])
            logvars.append(logvar.cpu().numpy()[0])
            zs.append(z.cpu().numpy()[0])

    return np.array(mus), np.array(logvars), np.array(zs), ds.subjects


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


# ── Reconstruction MSE ───────────────────────────────────────────────────────
def compute_reconstruction_mse(
    model: ConditionalVAE,
    trials: list[dict],
    train_mean: np.ndarray,
    train_std: np.ndarray,
    device: str = "cpu",
) -> float:
    """Compute mean reconstruction MSE on trials (in original scale)."""
    model.eval()
    ds = TrajectoryDataset(trials)
    tm = torch.tensor(train_mean, dtype=torch.float32).to(device)
    ts = torch.tensor(train_std, dtype=torch.float32).to(device)

    mses = []
    with torch.no_grad():
        for i in range(len(ds)):
            traj, cond, _ = ds[i]
            traj = traj.to(device).unsqueeze(0)
            cond = cond.to(device).unsqueeze(0)
            traj_z = (traj - tm) / ts
            recon_z, _, _, _ = model(traj_z, cond)
            recon = recon_z * ts + tm
            mse = float(torch.mean((recon - traj) ** 2).item())
            mses.append(mse)

    return float(np.mean(mses))


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
        "initiation_time_frames", "movement_time_frames",
        "peak_speed", "time_to_peak_speed",
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
        "initiation_time_frames", "movement_time_frames",
        "peak_speed", "curvature_index",
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
    train_mean: np.ndarray,
    train_std: np.ndarray,
    n_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """
    For each test subject, sample from the learned latent distribution
    and compare generated trajectory features against empirical ones
    using the Kolmogorov-Smirnov test.
    """
    model.eval()
    mus, logvars, _, subjects = encode_trials(model, trials, train_mean, train_std, device)
    tm = torch.tensor(train_mean, dtype=torch.float32).to(device)
    ts = torch.tensor(train_std, dtype=torch.float32).to(device)

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
        ).to(device).unsqueeze(0)

        gen_trajs = []
        with torch.no_grad():
            for _ in range(n_samples):
                z = torch.tensor(
                    np.random.randn(*agg_mu.shape) * agg_std + agg_mu,
                    dtype=torch.float32
                ).to(device).unsqueeze(0)
                recon_z = model.decode(z, cond)
                recon = (recon_z * ts + tm).cpu().numpy()[0]
                gen_trajs.append(recon.reshape(config.NORMALISED_LENGTH, 3))

        # KS test on path length
        emp_pl = emp_feats["path_length"].values
        gen_pl = np.array([np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)) for t in gen_trajs])
        ks_stat, ks_p = stats.ks_2samp(emp_pl, gen_pl)

        ks_results[subj] = {"ks_stat": ks_stat, "ks_pvalue": ks_p}

    return ks_results


# ── Full evaluation ──────────────────────────────────────────────────────────
def run_full_evaluation(
    model: ConditionalVAE,
    test_trials: list[dict],
    train_mean: np.ndarray,
    train_std: np.ndarray,
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
    vae_mse = compute_reconstruction_mse(model, test_trials, train_mean, train_std, device)
    print(f"\nReconstruction MSE — VAE: {vae_mse:.6f} | Spline: {spline_mse:.6f}")

    # 2. Encode test trials
    mus, logvars, zs, subjects = encode_trials(model, test_trials, train_mean, train_std, device)

    # 3. Fingerprints
    fp = compute_fingerprints(mus, subjects)
    fp.to_csv(save_dir / "fingerprints.csv", index=False)
    print(f"\nSubject fingerprints saved ({len(fp)} subjects)")

    # 4. Latent-kinematics correlations
    corr_df = latent_feature_correlations(mus, test_trials)
    corr_df.to_csv(save_dir / "latent_correlations.csv", index=False)
    sig = corr_df[corr_df["p_value"] < 0.05]
    print(f"\nSignificant latent-feature correlations: {len(sig)} / {len(corr_df)}")
    if len(sig) > 0:
        print(sig.to_string(index=False))

    # 5. Behavioural probing
    probe_df = behavioural_probing(mus, test_trials, subjects)
    probe_df.to_csv(save_dir / "probing_r2.csv", index=False)
    print(f"\nBehavioural probing R²:")
    print(probe_df.to_string(index=False))

    # 6. Generative fidelity
    ks = generative_fidelity_ks(model, test_trials, train_mean, train_std, device=device)
    print(f"\nGenerative fidelity (KS test on path length):")
    for subj, r in ks.items():
        print(f"  {subj}: KS={r['ks_stat']:.4f}, p={r['ks_pvalue']:.4f}")

    return {
        "vae_mse": vae_mse,
        "spline_mse": spline_mse,
        "fingerprints": fp,
        "correlations": corr_df,
        "probing": probe_df,
        "generative_ks": ks,
    }
