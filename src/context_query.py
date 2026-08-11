"""Leakage-resistant subject fingerprint evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class SubjectTrialSplit:
    subject: str
    context_indices: np.ndarray
    query_indices: np.ndarray


def split_context_query(
    subjects: Iterable[str],
    sp: Iterable[int],
    side: Iterable[int],
    fraction: float = 0.5,
    seed: int = 2026,
) -> list[SubjectTrialSplit]:
    """Split every subject into disjoint context/query sets by task stratum."""
    subjects = np.asarray(list(subjects))
    sp = np.asarray(list(sp))
    side = np.asarray(list(side))
    out: list[SubjectTrialSplit] = []
    for subject in sorted(np.unique(subjects)):
        rng = np.random.default_rng(seed + sum(map(ord, str(subject))))
        subject_idx = np.flatnonzero(subjects == subject)
        context: list[int] = []
        query: list[int] = []
        strata = np.array([f"{a}:{b}" for a, b in zip(sp[subject_idx], side[subject_idx])])
        for stratum in sorted(np.unique(strata)):
            idx = subject_idx[strata == stratum].copy()
            rng.shuffle(idx)
            if len(idx) == 1:
                (context if len(context) <= len(query) else query).append(int(idx[0]))
                continue
            n_context = int(np.clip(round(len(idx) * fraction), 1, len(idx) - 1))
            context.extend(idx[:n_context].tolist())
            query.extend(idx[n_context:].tolist())
        if not context or not query:
            idx = subject_idx.copy()
            rng.shuffle(idx)
            cut = max(1, min(len(idx) - 1, round(len(idx) * fraction)))
            context, query = idx[:cut].tolist(), idx[cut:].tolist()
        out.append(SubjectTrialSplit(subject, np.sort(context), np.sort(query)))
    return out


def moment_matched_posterior(mu: np.ndarray, logvar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and full covariance of an equally weighted posterior mixture."""
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    mean = mu.mean(axis=0)
    centred = mu - mean
    between = centred.T @ centred / max(len(mu), 1)
    within = np.diag(np.exp(logvar).mean(axis=0))
    covariance = between + within
    covariance += np.eye(covariance.shape[0]) * 1e-8
    return mean, covariance


def fingerprint_identification(
    context_mu: dict[str, np.ndarray], query_mu: dict[str, np.ndarray]
) -> dict:
    """Closed-set identification after enrolling each held-out subject."""
    labels = sorted(set(context_mu) & set(query_mu))
    centres = np.stack([context_mu[s].mean(axis=0) for s in labels])
    scale = np.std(np.vstack([context_mu[s] for s in labels]), axis=0) + 1e-8
    truth: list[str] = []
    pred: list[str] = []
    for subject in labels:
        distances = np.linalg.norm((query_mu[subject][:, None, :] - centres[None, :, :]) / scale, axis=2)
        pred.extend([labels[i] for i in distances.argmin(axis=1)])
        truth.extend([subject] * len(query_mu[subject]))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "chance": 1.0 / len(labels),
        "n_subjects": len(labels),
        "n_query_trials": len(truth),
    }


def subject_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Distribution summaries used as behavioral targets, never model inputs."""
    targets = [
        "initiation_time_s", "movement_time_s", "curvature_index",
        "peak_speed_tracker_units_s", "path_length", "max_lateral_deviation",
        "n_submovements",
    ]
    grouped = features.groupby("subject")[targets]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std().add_suffix("_sd")
    return pd.concat([means, stds], axis=1)


def tune_and_test_ridge(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    val_x: pd.DataFrame,
    val_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> pd.DataFrame:
    """Tune Ridge on validation subjects and score test subjects exactly once."""
    rows = []
    for target in train_y.columns:
        scaler = StandardScaler().fit(train_x.values)
        xtr = scaler.transform(train_x.values)
        xval = scaler.transform(val_x.values)
        xte = scaler.transform(test_x.values)
        ytr = train_y[target].values
        yval = val_y[target].values
        yte = test_y[target].values
        best_alpha, best_mse = None, np.inf
        for alpha in alphas:
            pred = Ridge(alpha=alpha).fit(xtr, ytr).predict(xval)
            mse = float(np.mean((pred - yval) ** 2))
            if mse < best_mse:
                best_mse, best_alpha = mse, alpha
        model = Ridge(alpha=best_alpha).fit(np.vstack([xtr, xval]), np.r_[ytr, yval])
        pred = model.predict(xte)
        rows.append({
            "target": target,
            "alpha": best_alpha,
            "r2_test": float(r2_score(yte, pred)),
            "mae_test": float(np.mean(np.abs(pred - yte))),
            "n_train_subjects": len(train_x),
            "n_val_subjects": len(val_x),
            "n_test_subjects": len(test_x),
        })
    return pd.DataFrame(rows)


def distribution_distances(empirical: pd.DataFrame, generated: pd.DataFrame, features: list[str]) -> dict:
    """Effect sizes and FDR-ready p-values for one subject's query sample."""
    row: dict[str, float] = {}
    for feature in features:
        e = empirical[feature].dropna().to_numpy()
        g = generated[feature].dropna().to_numpy()
        ks = stats.ks_2samp(e, g)
        row[f"ks_{feature}"] = float(ks.statistic)
        row[f"ks_p_{feature}"] = float(ks.pvalue)
        row[f"wasserstein_{feature}"] = float(stats.wasserstein_distance(e, g))
    e = empirical[features].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    g = generated[features].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    centre = e.mean(axis=0)
    scale = e.std(axis=0) + 1e-8
    e, g = (e - centre) / scale, (g - centre) / scale
    dxy = np.linalg.norm(e[:, None, :] - g[None, :, :], axis=-1)
    dxx = np.linalg.norm(e[:, None, :] - e[None, :, :], axis=-1)
    dyy = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)
    row["energy_distance"] = float(2 * dxy.mean() - dxx.mean() - dyy.mean())
    pooled = np.vstack([e, g])
    d2 = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=-1)
    median = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / median
    kernel = lambda a, b: np.exp(-gamma * np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))
    kxx, kyy, kxy = kernel(e, e), kernel(g, g), kernel(e, g)
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    row["mmd_rbf"] = float(kxx.sum() / (len(e) * (len(e) - 1)) +
                           kyy.sum() / (len(g) * (len(g) - 1)) - 2 * kxy.mean())
    return row


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg rejection mask."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    passed = ranked <= alpha * np.arange(1, len(p) + 1) / max(len(p), 1)
    reject = np.zeros(len(p), dtype=bool)
    if passed.any():
        reject[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return reject
