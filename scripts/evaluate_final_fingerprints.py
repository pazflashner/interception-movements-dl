"""Evaluate whether CVAE latents encode validated submovement behavior."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_corrected_study import load_per_trial_checkpoint
from src.context_query import split_context_query, tune_and_test_ridge
from src.evaluate import encode_trials
from src.train import split_subjects
from src.trajectory_view import project_trials_to_table_plane, select_trials_window
from src.vae_model import encode_trial_condition
import config


CONTINUOUS_TRIAL_TARGETS = [
    "mj_fit_error",
    "mj_first_duration_s",
    "mj_first_amplitude",
    "mj_secondary_amplitude_fraction",
    "mj_mean_overlap_pct",
]


def subject_behavior_summary(frame: pd.DataFrame) -> pd.DataFrame:
    continuous = ["mj_n_components", *CONTINUOUS_TRIAL_TARGETS]
    grouped = frame.groupby("subject")[continuous]
    summary = pd.concat([grouped.mean().add_suffix("_mean"), grouped.std().add_suffix("_sd")], axis=1)
    summary["recorded_success_rate"] = frame.assign(
        recorded_success=(frame.successful == 1).astype(float)
    ).groupby("subject").recorded_success.mean()
    summary["single_component_rate"] = frame.assign(
        single=(frame.mj_n_components == 1).astype(float)
    ).groupby("subject").single.mean()
    return summary.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fingerprint_tables(model, norm, trials, features, device, seed):
    mu, _, _, _ = encode_trials(model, trials, norm, device)
    subjects = np.array([t["metadata"]["subject"] for t in trials])
    sp = np.array([t["metadata"]["sp"] for t in trials])
    side = np.array([t["metadata"]["side"] for t in trials])
    fp_rows, query_rows = [], []
    for split in split_context_query(subjects, sp, side, seed=seed):
        mean = mu[split.context_indices].mean(axis=0)
        fp_rows.append({"subject": split.subject, **{f"z{i}_mean": value for i, value in enumerate(mean)}})
        query_rows.append(features.iloc[split.query_indices])
    fp = pd.DataFrame(fp_rows).set_index("subject").sort_index()
    query = pd.concat(query_rows, ignore_index=True)
    return fp, subject_behavior_summary(query).sort_index(), mu


def tune_ridge_trial(train_x, train_y, val_x, val_y, test_x, test_y):
    scaler = StandardScaler().fit(train_x)
    xtr, xv, xte = scaler.transform(train_x), scaler.transform(val_x), scaler.transform(test_x)
    best_alpha, best = None, np.inf
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        pred = Ridge(alpha=alpha).fit(xtr, train_y).predict(xv)
        score = np.mean((pred - val_y) ** 2)
        if score < best:
            best, best_alpha = score, alpha
    model = Ridge(alpha=best_alpha).fit(np.vstack([xtr, xv]), np.r_[train_y, val_y])
    pred = model.predict(xte)
    return float(r2_score(test_y, pred)), float(np.mean(np.abs(test_y - pred))), best_alpha


def tune_classifier(train_x, train_y, val_x, val_y, test_x, test_y, binary=False):
    scaler = StandardScaler().fit(train_x)
    xtr, xv, xte = scaler.transform(train_x), scaler.transform(val_x), scaler.transform(test_x)
    best_c, best = None, -np.inf
    for c in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=2000, class_weight="balanced").fit(xtr, train_y)
        score = balanced_accuracy_score(val_y, model.predict(xv))
        if score > best:
            best, best_c = score, c
    model = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(
        np.vstack([xtr, xv]), np.r_[train_y, val_y]
    )
    pred = model.predict(xte)
    row = {
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro")),
        "C": best_c,
    }
    if binary:
        prob = model.predict_proba(xte)[:, list(model.classes_).index(1)]
        row["auc"] = float(roc_auc_score(test_y, prob))
        row["brier"] = float(brier_score_loss(test_y, prob))
    return row


def add_model_inputs(mu, trials):
    condition_dim = 5 if "target_speed_screen_s" in trials[0]["metadata"] else 4
    cond = np.stack([encode_trial_condition(t["metadata"], condition_dim) for t in trials])
    return np.column_stack([mu, cond])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=str(config.DATA_PROCESSED_DIR / "canonical_trials.pkl"))
    parser.add_argument("--submovements", default=str(config.RESULTS_DIR / "submovements_real.csv"))
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-mode", choices=config.WINDOW_MODES, required=True)
    args = parser.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(args.trials, "rb") as handle:
        trials = pickle.load(handle)
    trials = project_trials_to_table_plane(select_trials_window(trials, args.window_mode))
    sub = pd.read_csv(args.submovements)
    sub = sub[sub.mj_fit_success == True].copy()
    feature_by_id = sub.set_index("trial_id")
    kept = [(trial, feature_by_id.loc[trial["metadata"]["trial_id"]]) for trial in trials
            if trial["metadata"]["trial_id"] in feature_by_id.index]
    trials = [item[0] for item in kept]
    features = pd.DataFrame([item[1] for item in kept]).reset_index(drop=True)

    train, val, test = split_subjects(trials, 17, 4, 7, 42)
    sets = {"train": set(t["metadata"]["trial_id"] for t in train),
            "val": set(t["metadata"]["trial_id"] for t in val),
            "test": set(t["metadata"]["trial_id"] for t in test)}
    indices = {name: np.array([i for i, t in enumerate(trials) if t["metadata"]["trial_id"] in ids])
               for name, ids in sets.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    subject_rows, trial_rows = [], []
    for run in sorted(Path(args.models).glob("cvae_*_z*_seed*")):
        if not (run / "checkpoint.pt").exists():
            continue
        model, norm = load_per_trial_checkpoint(run / "checkpoint.pt", device)
        tables = {}
        all_mu = np.zeros((len(trials), model.latent_dim), dtype=float)
        for name, idx in indices.items():
            subset_trials = [trials[i] for i in idx]
            subset_features = features.iloc[idx].reset_index(drop=True)
            fp, behavior, mu = fingerprint_tables(
                model, norm, subset_trials, subset_features, device, config.CONTEXT_QUERY_SEED
            )
            tables[name] = (fp, behavior)
            all_mu[idx] = mu
        probe = tune_and_test_ridge(
            tables["train"][0], tables["train"][1],
            tables["val"][0], tables["val"][1],
            tables["test"][0], tables["test"][1],
        )
        dim = model.latent_dim
        seed = int(run.name.rsplit("seed", 1)[1])
        probe.insert(0, "window_mode", args.window_mode)
        probe.insert(1, "seed", seed); probe.insert(1, "latent_dim", dim)
        subject_rows.extend(probe.to_dict("records"))

        x = add_model_inputs(all_mu, trials)
        count = features.mj_n_components.astype(int).to_numpy()
        success = (features.successful == 1).astype(int).to_numpy()
        count_result = tune_classifier(
            x[indices["train"]], count[indices["train"]],
            x[indices["val"]], count[indices["val"]],
            x[indices["test"]], count[indices["test"]],
        )
        trial_rows.append({"window_mode": args.window_mode, "latent_dim": dim, "seed": seed, "target": "mj_n_components",
                           "metric_type": "classification", **count_result})
        success_result = tune_classifier(
            x[indices["train"]], success[indices["train"]],
            x[indices["val"]], success[indices["val"]],
            x[indices["test"]], success[indices["test"]], binary=True,
        )
        trial_rows.append({"window_mode": args.window_mode, "latent_dim": dim, "seed": seed, "target": "recorded_success",
                           "metric_type": "classification", **success_result})
        for target in CONTINUOUS_TRIAL_TARGETS:
            y = features[target].fillna(0.0).to_numpy(dtype=float)
            r2, mae, alpha = tune_ridge_trial(
                x[indices["train"]], y[indices["train"]],
                x[indices["val"]], y[indices["val"]],
                x[indices["test"]], y[indices["test"]],
            )
            trial_rows.append({"window_mode": args.window_mode, "latent_dim": dim, "seed": seed, "target": target,
                               "metric_type": "regression", "r2": r2, "mae": mae, "alpha": alpha})

    pd.DataFrame(subject_rows).to_csv(out / "subject_distribution_probes.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(out / "trial_behavior_probes.csv", index=False)
    print(pd.DataFrame(trial_rows).groupby(["latent_dim", "target"]).mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
