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
from src.features import (
    KINEMATIC_FEATURES,
    compute_trial_features,
    features_from_arrays,
)


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
    sample: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pass trials through the full model.

    Returns (recon_traj, true_traj, recon_timing_s, true_timing_s), all in
    original units — millimetres for the trajectory, seconds for the timing.
    ``recon_timing_s`` is empty when the model has no timing head.

    ``sample=False`` decodes from the posterior mean μ rather than a draw from
    q(z|x). That is the standard way to measure VAE reconstruction: sampling
    adds noise that has nothing to do with how well the model represents the
    trial, and would handicap it against a deterministic baseline like PCA.
    Set ``sample=True`` to include the stochasticity.
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
        z = model.reparameterize(mu, logvar) if sample else mu
        recon_z, recon_timing_z = model.decode(z, cond)
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


# ── Latent traversal ─────────────────────────────────────────────────────────
def latent_traversal(
    model: ConditionalVAE,
    norm: NormStats,
    sp: int = 2,
    side: int = 1,
    n_steps: int = 7,
    span: float = 2.0,
    device: str = "cpu",
) -> dict:
    """
    Decode a sweep of each latent dimension with the others held at the prior mean.

    Proposal §4 asks whether traversing a latent dimension produces "distinct,
    meaningful structural changes". This returns, per dimension, the decoded
    trajectories and their timing across ``±span`` prior SDs, which the report
    turns into figures and into a per-dimension summary of *what changes* — so
    the reading is anchored to kinematics rather than to eyeballing a curve.

    Held at a single task condition, since the decoder is conditional: the
    traversal shows movement style at a fixed task, which is exactly the
    quantity the conditioning is meant to isolate.
    """
    model.eval()
    tm, ts, tim_m, tim_s = norm.torch(device)
    cond = torch.tensor(encode_condition(sp, side), dtype=torch.float32, device=device)
    steps = np.linspace(-span, span, n_steps)

    out = {"steps": steps, "sp": sp, "side": side, "dims": {}}
    for dim in range(model.latent_dim):
        z = torch.zeros(n_steps, model.latent_dim, device=device)
        z[:, dim] = torch.tensor(steps, dtype=torch.float32, device=device)
        with torch.no_grad():
            recon_z, timing_z = model.decode(z, cond.expand(n_steps, -1))
            trajs = ((recon_z * ts + tm).cpu().numpy()
                     .reshape(n_steps, config.NORMALISED_LENGTH, 3))
            timing = (
                (timing_z * tim_s + tim_m).cpu().numpy()
                if timing_z is not None else None
            )

        rows = []
        for i in range(n_steps):
            mt = float(timing[i, 0]) if timing is not None else 0.5
            it = float(timing[i, 1]) if timing is not None else 0.0
            rows.append(features_from_arrays(trajs[i], max(mt, 1e-3), it))
        out["dims"][dim] = {
            "trajectories": trajs,
            "timing": timing,
            "features": pd.DataFrame(rows),
        }
    return out


def traversal_summary(traversal: dict) -> pd.DataFrame:
    """
    What each latent dimension actually changes, as a range over the sweep.

    Turns the qualitative traversal into numbers: for every dimension, how much
    each kinematic feature moves from one end of the sweep to the other. A
    dimension whose features barely move is an unused (collapsed) dimension.
    """
    rows = []
    for dim, d in traversal["dims"].items():
        f = d["features"]
        row = {"latent_dim": f"z{dim}"}
        for feat in KINEMATIC_FEATURES:
            v = f[feat].values
            row[f"{feat}_range"] = float(np.max(v) - np.min(v))
        # Scale-free measure of whether this dimension does anything at all.
        row["relative_span"] = float(
            np.mean([
                (np.max(f[c].values) - np.min(f[c].values)) / (abs(np.mean(f[c].values)) + 1e-9)
                for c in ("path_length", "movement_time_s", "curvature_index")
            ])
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ── Behavioural probing (R²) ─────────────────────────────────────────────────
#: Macro-level behaviour a subject fingerprint should predict for someone the
#: model never saw. The first three are named in proposal §4.3; the rest are the
#: other standard descriptors of an interception movement — how symmetric the
#: velocity profile is, how far and how straight the hand travels.
BEHAVIOURAL_TARGETS_MEAN = [
    "initiation_time_s",
    "movement_time_s",
    "curvature_index",
    "peak_speed_mm_s",
    "time_to_peak_speed",
    "path_length",
    "max_lateral_deviation",
    "n_submovements",
]

#: Within-subject variability of the same quantities: motor control treats a
#: person's *consistency* as part of their signature, and §1 of the proposal
#: asks the model to reproduce distributions rather than single movements.
BEHAVIOURAL_TARGETS_SD = [
    "movement_time_s",
    "initiation_time_s",
    "curvature_index",
    "max_lateral_deviation",
]


def behavioural_probing(
    mus: np.ndarray,
    trials: list[dict],
    subjects: list[str],
    use_variance: bool = True,
) -> pd.DataFrame:
    """
    Predict macro-level behaviour from subject fingerprints, scored by
    leave-one-subject-out cross-validation.

    Proposal §4 asks for R² "for unseen subjects". Fitting and scoring on the
    same 7 subjects — as this did previously — measures how well 7 points can be
    interpolated by a model with as many free parameters, not generalisation;
    it reported R² ≈ 0.96 for probes that carry no predictive content. Each
    subject is now held out in turn, the scaler and probe are fitted on the
    remaining ones only, and R² is computed across the held-out predictions.

    With only 7 test subjects this is a 7-point estimate and will be unstable —
    negative R² simply means the probe does worse than predicting the mean,
    which is a meaningful (and common) outcome at this sample size. ``n_subjects``
    is reported so the number is never read without it.
    """
    feat_df = pd.DataFrame([compute_trial_features(t) for t in trials])
    feat_df["subject"] = subjects

    grouped = feat_df.groupby("subject")
    means = grouped[BEHAVIOURAL_TARGETS_MEAN].mean()
    # Intra-subject spread is a behavioural signature in its own right — how
    # *consistent* a mover someone is. It is also the only thing that can test
    # whether the variance half of the fingerprint (§3.3) carries information:
    # a latent mean cannot predict variability unless the code encodes it.
    sds = grouped[BEHAVIOURAL_TARGETS_SD].std().add_suffix("_sd")
    subj_feats = pd.concat([means, sds], axis=1)
    targets = list(subj_feats.columns)

    # Fingerprint = per-subject latent mean, optionally with the spread, which
    # is the other half of the "distribution" the proposal asks fingerprints to
    # capture (§3.3).
    fp = compute_fingerprints(mus, subjects).set_index("subject")
    z_cols = [c for c in fp.columns if c.endswith("_mean")]
    if use_variance:
        z_cols += [c for c in fp.columns if c.endswith("_std")]

    common = subj_feats.index.intersection(fp.index)
    X = np.nan_to_num(fp.loc[common, z_cols].values)
    n = len(common)

    results = []
    for target in targets:
        y = subj_feats.loc[common, target].values
        preds = {"linear": np.empty(n), "svr": np.empty(n)}

        for i in range(n):  # leave-one-subject-out
            tr = np.ones(n, dtype=bool)
            tr[i] = False
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[~tr])
            preds["linear"][i] = LinearRegression().fit(Xtr, y[tr]).predict(Xte)[0]
            preds["svr"][i] = SVR(kernel="rbf").fit(Xtr, y[tr]).predict(Xte)[0]

        results.append({
            "target": target,
            "r2_linear_loso": r2_score(y, preds["linear"]),
            "r2_svr_loso": r2_score(y, preds["svr"]),
            # Predicting the training mean scores R2 = 0 by construction; a
            # probe below that is worse than useless.
            "baseline_r2": 0.0,
            "n_subjects": n,
            "n_features": X.shape[1],
        })

    return pd.DataFrame(results)


# ── Two-sample comparison of feature distributions ───────────────────────────
def compare_feature_distributions(
    emp_feats: pd.DataFrame,
    gen_feats: pd.DataFrame,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """
    Univariate (KS per feature) and multivariate (MMD, energy) comparison.

    Shared by the CVAE and the spline/PCA representation so that a difference in
    the numbers is a difference between the models, never between two
    implementations of the same test.
    """
    entry = {"n_empirical": len(emp_feats), "n_generated": len(gen_feats)}
    for feat in KINEMATIC_FEATURES:
        ks_s, ks_p = stats.ks_2samp(emp_feats[feat].values, gen_feats[feat].values)
        entry[f"ks_{feat}"] = ks_s
        entry[f"ks_p_{feat}"] = ks_p
    ks_cols = [c for c in entry if c.startswith("ks_p_")]
    entry["n_features_rejected"] = int(sum(entry[c] < 0.05 for c in ks_cols))
    entry["n_features"] = len(ks_cols)

    E = emp_feats[KINEMATIC_FEATURES].values
    G = gen_feats[KINEMATIC_FEATURES].values
    scaler = StandardScaler().fit(E)
    Es, Gs = scaler.transform(E), np.nan_to_num(scaler.transform(G))
    entry["mmd_rbf"] = mmd_rbf(Es, Gs)
    entry["energy_distance"] = energy_distance(Es, Gs)
    entry["mmd_pvalue"] = permutation_pvalue(Es, Gs, mmd_rbf, n_permutations, seed)
    entry["energy_pvalue"] = permutation_pvalue(Es, Gs, energy_distance, n_permutations, seed)
    return entry


# ── Multivariate distribution distances ──────────────────────────────────────
def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """
    Unbiased squared Maximum Mean Discrepancy with an RBF kernel.

    ``gamma`` defaults to the median heuristic (1 / median pairwise squared
    distance over the pooled sample), which avoids hand-tuning a bandwidth per
    subject. 0 means the two samples are indistinguishable to the kernel.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if gamma is None:
        pooled = np.vstack([X, Y])
        d2 = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / med

    def k(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)

    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(
        Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean()
    )


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Multivariate energy distance: 2·E|X−Y| − E|X−X'| − E|Y−Y'|.

    Zero exactly when the two distributions coincide, and unlike MMD it needs no
    bandwidth choice — the two are reported side by side so a conclusion never
    rests on one kernel setting.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    d = lambda A, B: np.sqrt(np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1))
    return float(2 * d(X, Y).mean() - d(X, X).mean() - d(Y, Y).mean())


def permutation_pvalue(
    X: np.ndarray, Y: np.ndarray, statistic, n_permutations: int = 200, seed: int = 0
) -> float:
    """
    Permutation p-value for a two-sample statistic.

    MMD and energy distance have no usable null distribution in closed form at
    these sample sizes, so significance is obtained by shuffling the pooled
    labels — the only way to say whether a non-zero distance is more than noise.
    """
    rng = np.random.default_rng(seed)
    observed = statistic(X, Y)
    pooled = np.vstack([X, Y])
    n = len(X)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        if statistic(pooled[:n], pooled[n:]) >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


# ── Generative fidelity ──────────────────────────────────────────────────────
def generative_fidelity_ks(
    model: ConditionalVAE,
    trials: list[dict],
    norm: NormStats,
    n_samples: int = 100,
    device: str = "cpu",
    n_permutations: int = 200,
    seed: int = 0,
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

        # Draw conditions from the subject's *own* trial mix rather than reusing
        # the first trial's. The empirical sample spans every (sp, side) the
        # subject performed, so generating everything under one condition would
        # compare a single-condition sample against a multi-condition one and
        # charge the difference to the model.
        rng = np.random.default_rng(seed)
        subj_conds = np.stack([
            encode_condition(t["metadata"].get("sp", 1), t["metadata"].get("side", 1))
            for t in subj_trials
        ])
        cond = torch.tensor(
            subj_conds[rng.integers(0, len(subj_conds), size=n_samples)],
            dtype=torch.float32,
        ).to(device)

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

        # Describe generated samples with the *same* feature code as recorded
        # trials, so the comparison is not partly measuring an implementation
        # difference. Generated timing comes from the timing head; without it
        # the empirical mean stands in, and only shape features are meaningful.
        mt_idx = config.TIMING_FEATURES.index("movement_time_s")
        it_idx = config.TIMING_FEATURES.index("initiation_time_s")
        gen_rows = []
        for i, traj in enumerate(gen_trajs):
            if gen_timing is not None:
                mt, it = float(gen_timing[i, mt_idx]), float(gen_timing[i, it_idx])
            else:
                mt = float(emp_feats["movement_time_s"].mean())
                it = float(emp_feats["initiation_time_s"].mean())
            gen_rows.append(features_from_arrays(traj, max(mt, 1e-3), it))
        gen_feats = pd.DataFrame(gen_rows)

        # Univariate: KS per feature
        entry = {"n_empirical": len(emp_feats), "n_generated": len(gen_feats)}
        for feat in KINEMATIC_FEATURES:
            ks_s, ks_p = stats.ks_2samp(emp_feats[feat].values, gen_feats[feat].values)
            entry[f"ks_{feat}"] = ks_s
            entry[f"ks_p_{feat}"] = ks_p
        ks_cols = [c for c in entry if c.startswith("ks_p_")]
        entry["n_features_rejected"] = int(sum(entry[c] < 0.05 for c in ks_cols))
        entry["n_features"] = len(ks_cols)

        # Multivariate: MMD and energy distance over the standardised feature
        # space, scaled on the empirical sample so both are on one footing.
        E = emp_feats[KINEMATIC_FEATURES].values
        G = gen_feats[KINEMATIC_FEATURES].values
        scaler = StandardScaler().fit(E)
        Es, Gs = scaler.transform(E), np.nan_to_num(scaler.transform(G))
        entry["mmd_rbf"] = mmd_rbf(Es, Gs)
        entry["energy_distance"] = energy_distance(Es, Gs)
        entry["mmd_pvalue"] = permutation_pvalue(Es, Gs, mmd_rbf, n_permutations, seed)
        entry["energy_pvalue"] = permutation_pvalue(
            Es, Gs, energy_distance, n_permutations, seed
        )

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
    spline_pca_mse: float | None = None,
) -> dict:
    """Run all evaluation metrics and print summary."""
    save_dir = Path(save_dir or config.RESULTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    # 1. Reconstruction MSE. Two reference points, which measure different
    # things — see src/baseline_spline.py.
    vae_mse = compute_reconstruction_mse(model, test_trials, norm, device)
    print(f"\nReconstruction MSE (mm^2), held-out test subjects:")
    print(f"  CVAE (z={model.latent_dim}, generalises to unseen subjects) : {vae_mse:.6f}")
    if spline_pca_mse is not None:
        verdict = "BETTER" if vae_mse < spline_pca_mse else "worse"
        print(f"  Spline+PCA (same {model.latent_dim} dims, fitted on train)      : "
              f"{spline_pca_mse:.6f}   <- CVAE is {verdict}")
    print(f"  Spline per-trial fit (27 params/trial, sees the trial)  : {spline_mse:.6f}")
    print("  The per-trial spline is an interpolation ceiling, not a competing")
    print("  representation; the capacity-matched comparison is Spline+PCA.")

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
    print("\nGenerative fidelity (KS test + multivariate distance):")
    for subj, r in ks.items():
        nan = float("nan")
        line = f"  {subj}: "
        if "ks_path_length" in r:
            line += f"path_length KS={r['ks_path_length']:.3f} (p={r.get('ks_p_path_length', nan):.3f})"
        if "ks_movement_time_s" in r:
            line += (f" | movement_time KS={r['ks_movement_time_s']:.3f} "
                     f"(p={r.get('ks_p_movement_time_s', nan):.3f})")
        line += (f" | features rejected {r.get('n_features_rejected', 0)}/{r.get('n_features', 0)}"
                 f" | energy_dist={r.get('energy_distance', nan):.2f}"
                 f" (p={r.get('energy_pvalue', nan):.3f})")
        print(line)

    return {
        "vae_mse": vae_mse,
        "spline_mse": spline_mse,
        "spline_pca_mse": spline_pca_mse,
        "timing": timing_df,
        "fingerprints": fp,
        "correlations": corr_df,
        "probing": probe_df,
        "generative_ks": ks,
    }
