"""Frozen-model, post-review fingerprint controls and matched component fits.

New results only: never replace historical model predictions or component fits.
The matched component endpoint describes the model's filtered, 100-point target
representation, not unprocessed biological submovements.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import false_discovery_control, ks_2samp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from scripts.generate_final_samples import generate_run, categorical_distances
from scripts.run_corrected_study import (
    context_query_for_trials, load_per_trial_checkpoint,
    training_latent_noise_covariance,
)
from src.confirmatory_protocol import make_participant_folds, partition_trials
from src.context_query import DistanceReference, distribution_distances, subject_summary
from src.evaluate import encode_trials
from src.features import (compute_trial_features, features_from_generated_window,
                          movement_from_generated_window, kinematic_features_for_dim)
from src.statistical_tests import paired_wilcoxon
from src.submovements import SubmovementConfig, decompose_position
from src.trajectory_view import project_trials_to_table_plane, select_trials_window
from src.vae_model import encode_trial_condition

OUT = ROOT / "studies/review_corrected_evaluation/results/review_controls"
TRAINED = ROOT / "studies/final_strategy_evaluation"
CORRECTED = ROOT / "studies/review_corrected_evaluation"
METRICS = ["mean_ks", "energy_distance", "mmd_rbf"]
MJ_FEATURES = ["mj_fit_error", "mj_first_duration_s", "mj_first_amplitude",
               "mj_secondary_amplitude_fraction", "mj_mean_overlap_pct"]


def save_protocol():
    protocol = {
        "version": "post-review-controls-v1", "neural_retraining": False,
        "dims": [3, 8], "folds": [0, 1, 2, 3], "seeds": [42, 43, 44],
        "context_query_seed": config.CONTEXT_QUERY_SEED,
        "fingerprint": {"n_generated": 120,
            "own": "mean encoder posterior mean over this subject's context",
            "population": "equal-person mean of the 17 training-context centroids",
            "wrong": "mean of distances for all six other test-context centroids; no pooling their samples",
            "fixed": "decoder, target task draws, latent noise, train-only covariance and distance reference",
            "tests": "12 paired two-sided Pratt Wilcoxon tests: 2 dimensions x 2 controls x 3 distances, BH together; seeds averaged within each of 28 subjects",
            "inference_limit": "post-review exploratory control; folds have overlapping training sets; within-session context enrollment, no causal/stable-strategy claim"},
        "minimum_jerk": {"n_generated": 10, "restarts": 2, "max_nfev": 400,
            "input": "recorded canonical filtered go-window target (100 x 2) and generated go-window (100 x 2)",
            "common_operator": "same phase crop by own measured/predicted timing to 100 movement points; physical sample_hz=99/movement_seconds; same lowpass and decomposition",
            "interpretation": "matched downstream evaluation of the model target representation; upstream acquisition filtering is not undone; true vs predicted timing remains an intended difference",
            "selection": "historical error-based count selection retained regardless of convergence; all statuses reported, no favorable exclusions",
            "starts": "deterministic trial-specific seed; real trial ID / original generated run-subject-sample ID",
            "sensitivity": "first two query trials and first two generated samples per participant, n=3/8 seed42; same input, four restarts and max_nfev700",
            "sample_reference": "500 empirical plugin resamples at observed query and n=10 sizes; 3 seed-like draws averaged per subject, no formal null p"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("existing protocol differs; choose a new version")
    path.write_text(json.dumps(protocol, indent=2))


def load_trials():
    path = ROOT / "studies/strategy_window_comparison/data/canonical_trials.pkl"
    with path.open("rb") as handle:
        return project_trials_to_table_plane(select_trials_window(pickle.load(handle), config.WINDOW_GO_TO_ARRIVAL))


def load_run(train, fold, dim, seed):
    path = TRAINED / f"runs/cvae/fold{fold}/cvae_z{dim}_seed{seed}/checkpoint.pt"
    model, norm = load_per_trial_checkpoint(path, "cpu")
    model.eval()
    covariance = training_latent_noise_covariance(model, train, norm, "cpu")
    return model, norm, covariance, hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint_run(train, test, fold, dim, seed, model, norm, covariance):
    destination = OUT / "fingerprint" / f"z{dim}_fold{fold}_seed{seed}.csv"
    if destination.exists():
        return
    destination.parent.mkdir(exist_ok=True)
    features = kinematic_features_for_dim(2)
    ref_data = json.loads((CORRECTED / f"runs/cvae/fold{fold}/cvae_z{dim}_seed{seed}/distance_reference.json").read_text())
    reference = DistanceReference(tuple(ref_data["features"]), np.array(ref_data["centre"]), np.array(ref_data["scale"]), ref_data["rbf_gamma"])
    test_mu = encode_trials(model, test, norm, "cpu")[0]
    train_mu = encode_trials(model, train, norm, "cpu")[0]
    splits = context_query_for_trials(test, config.CONTEXT_QUERY_SEED)
    centroids = {s.subject: test_mu[s.context_indices].mean(axis=0) for s in splits}
    train_centroids = [train_mu[s.context_indices].mean(axis=0) for s in context_query_for_trials(train, config.CONTEXT_QUERY_SEED)]
    population = np.mean(train_centroids, axis=0)
    tm, ts, _, _ = norm.torch("cpu")
    rows = []
    for split in splits:
        assert not set(split.context_indices) & set(split.query_indices)
        query = [test[i] for i in split.query_indices]
        empirical = pd.DataFrame([compute_trial_features(t) for t in query])
        mean = centroids[split.subject]
        rng = np.random.default_rng(config.CONTEXT_QUERY_SEED + sum(map(ord, split.subject)))
        own_z = rng.multivariate_normal(mean, covariance, size=120)
        noise = own_z - mean
        chosen = rng.integers(0, len(query), size=120)
        cond = np.stack([encode_trial_condition(query[i]["metadata"], model.condition_dim) for i in chosen]).astype(np.float32)
        centres = [("own", split.subject, mean), ("population", "training_mean", population)]
        centres += [("wrong", subject, centre) for subject, centre in centroids.items() if subject != split.subject]
        for arm, donor, centre in centres:
            z = (own_z if arm == "own" else noise + centre).astype(np.float32)
            with torch.no_grad():
                x, timing = model.decode(torch.from_numpy(z), torch.from_numpy(cond))
                x = ((x * ts + tm).numpy()).reshape(120, 100, 2)
                timing = norm.denormalise_timing(timing.numpy())
            generated = pd.DataFrame([features_from_generated_window(p, max(float(t[0]), .001), max(float(t[1]), 0), config.WINDOW_GO_TO_ARRIVAL) for p, t in zip(x, timing)])
            assert np.isfinite(generated[features].to_numpy()).all()
            distances = distribution_distances(empirical, generated, features, reference)
            rows.append({"latent_dim": dim, "outer_fold": fold, "seed": seed, "subject": split.subject,
                         "arm": arm, "donor": donor, "n_context": len(split.context_indices), "n_query": len(query),
                         "mean_ks": float(np.mean([distances[f"ks_{f}"] for f in features])), **distances})
    frame = pd.DataFrame(rows)
    saved = pd.read_csv(CORRECTED / f"runs/cvae/fold{fold}/cvae_z{dim}_seed{seed}/context_query_fidelity.csv").set_index("subject")
    own = frame[frame.arm == "own"].set_index("subject")
    cols = [f"ks_{f}" for f in features] + ["energy_distance", "mmd_rbf"]
    np.testing.assert_allclose(own[cols], saved.loc[own.index, cols], atol=1e-6, rtol=1e-6)
    frame.to_csv(destination, index=False)


def summarize_fingerprints():
    frame = pd.concat([pd.read_csv(p) for p in sorted((OUT / "fingerprint").glob("z*.csv"))], ignore_index=True)
    assert len(frame) == 24 * 7 * 8
    by_seed = frame.groupby(["latent_dim", "outer_fold", "seed", "subject", "arm"], as_index=False)[METRICS].mean()
    by_person = by_seed.groupby(["latent_dim", "outer_fold", "subject", "arm"], as_index=False)[METRICS].mean()
    by_person.to_csv(OUT / "fingerprint_participant.csv", index=False)
    by_person.groupby(["latent_dim", "arm"], as_index=False)[METRICS].mean().to_csv(OUT / "fingerprint_summary.csv", index=False)
    comparisons = []
    for dim in (3, 8):
        own = by_person[(by_person.latent_dim == dim) & (by_person.arm == "own")].set_index("subject")
        for control in ("population", "wrong"):
            other = by_person[(by_person.latent_dim == dim) & (by_person.arm == control)].set_index("subject").loc[own.index]
            assert len(own) == 28
            for metric in METRICS:
                d = other[metric].to_numpy() - own[metric].to_numpy()
                w = paired_wilcoxon(d)
                comparisons.append({"latent_dim": dim, "control": control, "metric": metric,
                    "own_mean": own[metric].mean(), "control_mean": other[metric].mean(),
                    "control_minus_own": d.mean(), "own_better_n": int((d > 0).sum()),
                    "n_subjects": 28, "wilcoxon_p": w.pvalue})
    comparisons = pd.DataFrame(comparisons)
    comparisons["p_fdr_bh"] = false_discovery_control(comparisons.wilcoxon_p)
    comparisons.to_csv(OUT / "fingerprint_paired.csv", index=False)


def prepare_matched_movement(window, movement_time_s, initiation_time_s):
    """One downstream representation operator for both recorded and generated."""
    if not np.isfinite(movement_time_s) or movement_time_s <= 0:
        raise ValueError("movement duration must be finite and positive")
    movement = movement_from_generated_window(window, movement_time_s, initiation_time_s, config.WINDOW_GO_TO_ARRIVAL)
    return movement, (len(movement) - 1) / movement_time_s


def component_job(item):
    # Child processes on Windows start fresh; inherit no parent BLAS limits.
    # Prevent each worker from also starting a full numerical thread pool.
    threadpool_limits(1)
    row, window, restarts, max_nfev = item
    row = dict(row)
    try:
        movement, sample_hz = prepare_matched_movement(window, row["movement_time_s"], row["initiation_time_s"])
        cfg = SubmovementConfig(restarts=restarts, max_nfev=max_nfev, sample_hz=sample_hz)
        result = decompose_position(movement, cfg, row["fit_seed_id"])
        row.update(result.summary())
        row["mj_parameters_json"] = json.dumps(result.selected.parameters.tolist())
        row["fit_failure"] = ""
        row["sample_hz"] = sample_hz
    except Exception as exc:
        row.update(mj_fit_completed=False, fit_failure=f"{type(exc).__name__}: {exc}")
    return row


def prepare_components(trials, folds):
    items, manifest = [], []
    for fold, split in enumerate(folds):
        train, _, test = partition_trials(trials, split)
        for s in context_query_for_trials(test, config.CONTEXT_QUERY_SEED):
            for ordinal, i in enumerate(s.query_indices):
                t = test[i]
                row = {"kind": "recorded", "outer_fold": fold, "subject": s.subject,
                       "trial_id": t["metadata"]["trial_id"], "fit_seed_id": t["metadata"]["trial_id"],
                       "query_ordinal": ordinal, "movement_time_s": (t["move_end_idx"]-t["move_start_idx"])/240,
                       "initiation_time_s": (t["move_start_idx"]-t["go_signal_idx"])/240}
                row["job_id"] = row["trial_id"]
                items.append((row, t["pos_norm"], 2, 400))
        for dim in (3, 8):
            for seed in (42, 43, 44):
                model, norm, cov, digest = load_run(train, fold, dim, seed)
                run_name = f"cvae_z{dim}_fold{fold}_seed{seed}"
                generated = generate_run(model, norm, test, 10, "cpu", seed, run_name, cov)
                for row, window in generated:
                    row.update(kind="generated", outer_fold=fold, latent_dim=dim, seed=seed,
                               fit_seed_id=f"{row['run']}-{row['subject']}-{row['sample_id']}")
                    row["job_id"] = row["fit_seed_id"]
                    items.append((row, window, 2, 400))
                manifest.append({"fold": fold, "dim": dim, "seed": seed, "checkpoint_sha256": digest})
                print("prepared", run_name, flush=True)
    (OUT / "checkpoints.json").write_text(json.dumps(manifest, indent=2))
    return items


def run_component_jobs(items, path, jobs, backend="process"):
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["job_id"] for r in rows}
    pending = [item for item in items if item[0]["job_id"] not in done]
    started = time.monotonic()
    executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    with executor(max_workers=jobs) as pool:
        futures = [pool.submit(component_job, item) for item in pending]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 50 == 0 or i == len(pending):
                pd.DataFrame(rows).sort_values("job_id").to_csv(path, index=False)
                print(f"{path.stem}: {len(rows)}/{len(items)}, elapsed {time.monotonic()-started:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fingerprint", "components"])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--backend", choices=["process", "thread"], default="process")
    args = parser.parse_args()
    torch.set_num_threads(1)
    threadpool_limits(1)
    save_protocol()
    trials = load_trials()
    folds = make_participant_folds([t["metadata"]["subject"] for t in trials])
    if args.stage == "fingerprint":
        for fold, split in enumerate(folds):
            train, _, test = partition_trials(trials, split)
            for dim in (3, 8):
                for seed in (42, 43, 44):
                    model, norm, cov, _ = load_run(train, fold, dim, seed)
                    fingerprint_run(train, test, fold, dim, seed, model, norm, cov)
                    print(f"fingerprint done z{dim} fold{fold} seed{seed}", flush=True)
        summarize_fingerprints()
    else:
        cache = OUT / "component_inputs.pkl"
        if cache.exists():
            with cache.open("rb") as f: items = pickle.load(f)
        else:
            items = prepare_components(trials, folds)
            with cache.open("wb") as f: pickle.dump(items, f)
        run_component_jobs(items, OUT / "matched_components.csv", args.jobs, args.backend)
        sensitivity = []
        for row, window, _, _ in items:
            if (row["kind"] == "recorded" and row["query_ordinal"] < 2) or (row["kind"] == "generated" and row["seed"] == 42 and row["sample_id"] < 2):
                sensitivity.append((row, window, 4, 700))
        run_component_jobs(sensitivity, OUT / "matched_component_sensitivity.csv", args.jobs, args.backend)


if __name__ == "__main__":
    main()
