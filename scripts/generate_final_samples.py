"""Generate held-out subject movements and evaluate submovement distributions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_corrected_study import load_per_trial_checkpoint, training_latent_noise_covariance
from src.context_query import benjamini_hochberg, moment_matched_posterior, split_context_query
from src.evaluate import encode_trials
from src.features import (
    compute_trial_features,
    features_from_generated_window,
    movement_from_generated_window,
)
from src.submovements import SubmovementConfig, decompose_normalized_trajectory
from src.train import split_subjects
from src.trajectory_view import project_trials_to_table_plane, select_trials_window
from src.vae_model import encode_trial_condition
import config


CONTINUOUS_FEATURES = [
    "movement_time_s", "initiation_time_s", "peak_speed_tracker_units_s",
    "path_length", "curvature_index", "max_lateral_deviation",
    "mj_fit_error", "mj_first_duration_s", "mj_first_amplitude",
    "mj_secondary_amplitude_fraction", "mj_mean_overlap_pct",
]


def fit_generated(item, cfg):
    row, trajectory = item
    try:
        movement = movement_from_generated_window(
            trajectory,
            row["movement_time_s"],
            row["initiation_time_s"],
            row["window_mode"],
        )
        result = decompose_normalized_trajectory(
            movement, row["movement_time_s"], cfg,
            f"{row['run']}-{row['subject']}-{row['sample_id']}",
        )
        row.update(result.summary())
        row["mj_fit_success"] = True
        row["mj_failure"] = ""
    except Exception as exc:
        row["mj_fit_success"] = False
        row["mj_failure"] = f"{type(exc).__name__}: {exc}"
    return row


def generate_run(model, norm, test_trials, n_per_subject, device, seed, run_name, shared_covariance):
    mu, logvar, _, _ = encode_trials(model, test_trials, norm, device)
    subjects = [t["metadata"]["subject"] for t in test_trials]
    sp = [t["metadata"]["sp"] for t in test_trials]
    side = [t["metadata"]["side"] for t in test_trials]
    tm, ts, _, _ = norm.torch(device)
    items = []
    for split in split_context_query(subjects, sp, side, seed=config.CONTEXT_QUERY_SEED):
        mean = mu[split.context_indices].mean(axis=0)
        rng = np.random.default_rng(seed + sum(map(ord, split.subject)))
        z = rng.multivariate_normal(mean, shared_covariance, size=n_per_subject).astype(np.float32)
        query = [test_trials[i] for i in split.query_indices]
        chosen = rng.integers(0, len(query), size=n_per_subject)
        cond = np.stack([encode_trial_condition(query[i]["metadata"], model.condition_dim)
                         for i in chosen]).astype(np.float32)
        with torch.no_grad():
            rz, rtz = model.decode(torch.as_tensor(z, device=device), torch.as_tensor(cond, device=device))
            channels = model.input_dim // config.NORMALISED_LENGTH
            trajectories = ((rz * ts + tm).cpu().numpy()).reshape(n_per_subject, config.NORMALISED_LENGTH, channels)
            timing = norm.denormalise_timing(rtz.cpu().numpy())
        for i in range(n_per_subject):
            move_time = max(float(timing[i, 0]), 1e-3)
            init_time = max(float(timing[i, 1]), 0.0)
            window_mode = query[chosen[i]].get("window_mode", config.WINDOW_MOVEMENT_ONLY)
            basic = features_from_generated_window(
                trajectories[i], move_time, init_time, window_mode
            )
            row = {
                "run": run_name,
                "subject": split.subject,
                "sample_id": i,
                "sp": int(query[chosen[i]]["metadata"]["sp"]),
                "side": int(query[chosen[i]]["metadata"]["side"]),
                "window_mode": window_mode,
                **basic,
            }
            items.append((row, trajectories[i]))
    return items


def categorical_distances(empirical, generated):
    support = np.arange(1, 5)
    pe = np.array([(empirical == value).mean() for value in support], dtype=float)
    pg = np.array([(generated == value).mean() for value in support], dtype=float)
    return float(0.5 * np.abs(pe - pg).sum()), float(jensenshannon(pe, pg, base=2.0) ** 2)


def evaluate_run(generated, empirical, test_trials):
    subjects = [t["metadata"]["subject"] for t in test_trials]
    sp = [t["metadata"]["sp"] for t in test_trials]
    side = [t["metadata"]["side"] for t in test_trials]
    trial_ids = [t["metadata"]["trial_id"] for t in test_trials]
    rows = []
    for split in split_context_query(subjects, sp, side, seed=config.CONTEXT_QUERY_SEED):
        ids = [trial_ids[i] for i in split.query_indices]
        e = empirical[empirical.trial_id.isin(ids)]
        g = generated[generated.subject == split.subject]
        row = {"subject": split.subject, "n_empirical": len(e), "n_generated": len(g)}
        for feature in CONTINUOUS_FEATURES:
            ev = e[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            gv = g[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(ev) and len(gv):
                ks = stats.ks_2samp(ev, gv)
                row[f"ks_{feature}"] = float(ks.statistic)
                row[f"ks_p_{feature}"] = float(ks.pvalue)
                row[f"wasserstein_{feature}"] = float(stats.wasserstein_distance(ev, gv))
        tv, jsd = categorical_distances(e.mj_n_components.to_numpy(), g.mj_n_components.to_numpy())
        row["count_total_variation"] = tv
        row["count_jsd"] = jsd
        p_columns = [f"ks_p_{feature}" for feature in CONTINUOUS_FEATURES if f"ks_p_{feature}" in row]
        row["ks_rejected_fdr"] = int(benjamini_hochberg(np.array([row[column] for column in p_columns])).sum())
        row["ks_features_tested"] = len(p_columns)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=str(config.DATA_PROCESSED_DIR / "canonical_trials.pkl"))
    parser.add_argument("--submovements", default=str(config.RESULTS_DIR / "submovements_real.csv"))
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-mode", choices=config.WINDOW_MODES, required=True)
    parser.add_argument("--samples-per-subject", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--recompute-fidelity", action="store_true")
    args = parser.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(args.trials, "rb") as handle:
        trials = pickle.load(handle)
    trials = project_trials_to_table_plane(select_trials_window(trials, args.window_mode))
    train_trials, _, test_trials = split_subjects(trials, 17, 4, 7, 42)
    empirical = pd.read_csv(args.submovements)
    empirical = empirical[empirical.mj_fit_success == True].copy()
    basic = pd.DataFrame([compute_trial_features(trial) for trial in trials])
    basic_columns = [
        "trial_id", "movement_time_s", "initiation_time_s",
        "peak_speed_tracker_units_s", "path_length", "curvature_index",
        "max_lateral_deviation",
    ]
    empirical = empirical.merge(basic[basic_columns], on="trial_id", how="inner", validate="one_to_one")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SubmovementConfig(restarts=1, max_nfev=300)
    summaries = []

    for run in sorted(Path(args.models).glob("cvae_*_z*_seed*")):
        if not (run / "checkpoint.pt").exists():
            continue
        generated_path = out / f"{run.name}_generated.csv"
        fidelity_path = out / f"{run.name}_fidelity.csv"
        if generated_path.exists():
            generated_all = pd.read_csv(generated_path)
        else:
            model, norm = load_per_trial_checkpoint(run / "checkpoint.pt", device)
            shared_covariance = training_latent_noise_covariance(model, train_trials, norm, device)
            items = generate_run(
                model, norm, test_trials, args.samples_per_subject, device,
                int(run.name.rsplit("seed", 1)[1]), run.name, shared_covariance,
            )
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                rows = list(pool.map(fit_generated, items, repeat(cfg)))
            generated_all = pd.DataFrame(rows)
            generated_all.to_csv(generated_path, index=False)
        if fidelity_path.exists() and not args.recompute_fidelity:
            fidelity = pd.read_csv(fidelity_path)
        else:
            generated_valid = generated_all[generated_all.mj_fit_success == True]
            fidelity = evaluate_run(generated_valid, empirical, test_trials)
            fidelity.to_csv(fidelity_path, index=False)
        numeric = fidelity.select_dtypes(include=[np.number])
        summary = {
            "run": run.name,
            "generated_fit_success_rate": float(generated_all.mj_fit_success.mean()),
            **{f"mean_{column}": float(numeric[column].mean())
                                        for column in numeric.columns if column not in {"n_empirical", "n_generated"}}}
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(out / "generation_summary.csv", index=False)
        print(run.name, summary, flush=True)

    protocol = {
        "samples_per_test_subject": args.samples_per_subject,
        "fingerprint_source": "context-half mean latent only",
        "latent_noise": "one shared covariance estimated from training subjects",
        "condition_source": "empirical query condition mixture",
        "continuous_metrics": ["KS statistic", "Wasserstein distance"],
        "count_metrics": ["total variation", "Jensen-Shannon divergence"],
        "submovement_restarts_generated": cfg.restarts,
        "window_mode": args.window_mode,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
