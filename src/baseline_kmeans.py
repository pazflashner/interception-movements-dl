"""
Phase 1 – K-Means clustering baseline.

Tests whether individual trajectories are inherently separable by
clustering either the full normalised trajectories or the extracted
kinematic features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def select_feature_matrix(
    feature_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the clustering matrix from an explicit column allowlist.

    Selecting every numeric column instead quietly includes the trial counter
    and the task labels. Those carry no subject identity, and since
    ``StandardScaler`` weights every column equally they act as pure noise
    dimensions that dilute the signal. The allowlist lives in
    ``config.KMEANS_FEATURE_COLUMNS``; see the note there for the measured
    effect and what is excluded.
    """
    feature_columns = list(feature_columns or config.KMEANS_FEATURE_COLUMNS)

    missing = [c for c in feature_columns if c not in feature_df.columns]
    if missing:
        raise KeyError(
            f"feature_df is missing required columns: {missing}. "
            f"Available: {sorted(feature_df.columns)}"
        )

    if verbose:
        numeric = set(feature_df.select_dtypes(include=[np.number]).columns)
        excluded = sorted(numeric - set(feature_columns))
        print(f"  Clustering on {len(feature_columns)} kinematic features; "
              f"excluding {len(excluded)} other numeric columns: {excluded}")

    return feature_df[feature_columns].values, feature_columns


def cluster_trajectories(
    trials: list[dict],
    n_clusters_range: range = config.KMEANS_N_CLUSTERS_RANGE,
    use_features: bool = False,
    feature_df: pd.DataFrame | None = None,
    seed: int = config.SEED,
    feature_columns: list[str] | None = None,
) -> dict:
    """
    Cluster trials using K-Means and evaluate against true subject labels.

    Parameters
    ----------
    trials : list of preprocessed trial dicts
    n_clusters_range : range of K values to try
    use_features : if True, cluster on extracted features instead of raw trajectories
    feature_df : required if use_features is True
    seed : passed to KMeans as random_state, so runs are reproducible
    feature_columns : override for ``config.KMEANS_FEATURE_COLUMNS``

    Returns
    -------
    dict with best_k, best_ari, best_nmi, results_per_k
    """
    # Build data matrix
    if use_features and feature_df is not None:
        X, feature_columns = select_feature_matrix(feature_df, feature_columns)
        subjects = feature_df["subject"].values
    else:
        # Flatten normalised trajectories: (n_trials, T*3)
        X = np.array([t["pos_norm"].flatten() for t in trials])
        subjects = np.array([t["metadata"]["subject"] for t in trials])
        feature_columns = None

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode subject labels as integers
    unique_subjects = np.unique(subjects)
    label_map = {s: i for i, s in enumerate(unique_subjects)}
    true_labels = np.array([label_map[s] for s in subjects])

    results = []
    best_ari = -1
    best_k = n_clusters_range.start

    for k in n_clusters_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        pred = km.fit_predict(X_scaled)
        ari = adjusted_rand_score(true_labels, pred)
        nmi = normalized_mutual_info_score(true_labels, pred)
        results.append({"k": k, "ari": ari, "nmi": nmi})
        if ari > best_ari:
            best_ari = ari
            best_k = k

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["ari"].idxmax()]

    # Chance level matters here: with 28 subjects ARI is near 0 for a random
    # partition, so a small positive ARI is not self-evidently meaningful.
    print(f"K-Means best K={int(best_row['k'])}: ARI={best_row['ari']:.4f}, NMI={best_row['nmi']:.4f} "
          f"({len(unique_subjects)} subjects, {X.shape[1]} dims)")
    return {
        "best_k": int(best_row["k"]),
        "best_ari": float(best_row["ari"]),
        "best_nmi": float(best_row["nmi"]),
        "results": results_df,
        "seed": seed,
        "feature_columns": feature_columns,
        "n_features": int(X.shape[1]),
    }


# ── Repeated runs across seeds ────────────────────────────────────────────────
def sweep_seeds(
    trials: list[dict],
    seeds: list[int],
    feature_df: pd.DataFrame | None = None,
    n_clusters_range: range = config.KMEANS_N_CLUSTERS_RANGE,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Repeat the K-Means baseline across seeds, for both representations.

    Returns a tidy frame with one row per (seed, representation, k). Note that
    the seed only re-initialises K-Means here: unlike the VAE phase there is no
    train/test split, so every seed clusters the identical 4763 trials. The
    resulting spread is algorithmic variance only and is a floor on, not an
    estimate of, the run-to-run noise of the baseline.
    """
    rows = []
    for seed in seeds:
        for name, use_features in (("trajectories", False), ("features", True)):
            if use_features and feature_df is None:
                continue
            print(f"\n--- seed {seed} | {name} ---")
            res = cluster_trajectories(
                trials,
                n_clusters_range=n_clusters_range,
                use_features=use_features,
                feature_df=feature_df,
                seed=seed,
            )
            df = res["results"].copy()
            df.insert(0, "representation", name)
            df.insert(0, "seed", seed)
            rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(f"\nSaved {len(out)} rows to {out_csv}")
    return out
