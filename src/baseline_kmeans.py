"""
Phase 1 – K-Means clustering baseline.

Tests whether individual trajectories are inherently separable by
clustering either the full normalised trajectories or the extracted
kinematic features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def cluster_trajectories(
    trials: list[dict],
    n_clusters_range: range = config.KMEANS_N_CLUSTERS_RANGE,
    use_features: bool = False,
    feature_df: pd.DataFrame | None = None,
) -> dict:
    """
    Cluster trials using K-Means and evaluate against true subject labels.

    Parameters
    ----------
    trials : list of preprocessed trial dicts
    n_clusters_range : range of K values to try
    use_features : if True, cluster on extracted features instead of raw trajectories
    feature_df : required if use_features is True

    Returns
    -------
    dict with best_k, best_ari, best_nmi, results_per_k
    """
    # Build data matrix
    if use_features and feature_df is not None:
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        X = feature_df[numeric_cols].values
        subjects = feature_df["subject"].values
    else:
        # Flatten normalised trajectories: (n_trials, T*3)
        X = np.array([t["pos_norm"].flatten() for t in trials])
        subjects = np.array([t["metadata"]["subject"] for t in trials])

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
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        pred = km.fit_predict(X_scaled)
        ari = adjusted_rand_score(true_labels, pred)
        nmi = normalized_mutual_info_score(true_labels, pred)
        results.append({"k": k, "ari": ari, "nmi": nmi})
        if ari > best_ari:
            best_ari = ari
            best_k = k

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["ari"].idxmax()]

    print(f"K-Means best K={int(best_row['k'])}: ARI={best_row['ari']:.4f}, NMI={best_row['nmi']:.4f}")
    return {
        "best_k": int(best_row["k"]),
        "best_ari": float(best_row["ari"]),
        "best_nmi": float(best_row["nmi"]),
        "results": results_df,
    }
