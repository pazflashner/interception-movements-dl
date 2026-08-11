"""Run the audited interception-movement study end to end.

This script intentionally writes only under ``results/corrected_v2``. Legacy
results remain available for comparison but are never mixed into this report.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.baseline_spline import evaluate_spline_baseline, evaluate_spline_pca_baseline
from src.context_query import (
    benjamini_hochberg,
    distribution_distances,
    fingerprint_identification,
    moment_matched_posterior,
    split_context_query,
    subject_summary,
    tune_and_test_ridge,
)
from src.evaluate import encode_trials, reconstruct
from src.features import KINEMATIC_FEATURES, compute_trial_features, features_from_arrays
from src.hierarchical_vae import HierarchicalCVAE, HierarchicalNorm, train_hierarchical
from src.run_config import RunConfig, set_seed
from src.train import split_subjects, train_vae
from src.vae_model import ConditionalVAE, NormStats, encode_condition, encode_trial_condition


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def finish_fidelity_table(frame: pd.DataFrame) -> pd.DataFrame:
    pcols = [c for c in frame if c.startswith("ks_p_")]
    frame = frame.copy()
    frame["ks_rejected_fdr"] = [int(benjamini_hochberg(row[pcols].to_numpy()).sum())
                                 for _, row in frame.iterrows()]
    frame["ks_features_tested"] = len(pcols)
    return frame


def mean_ks_statistic(frame: pd.DataFrame) -> float:
    columns = [c for c in frame if c.startswith("ks_") and not c.startswith("ks_p_")
               and c not in {"ks_rejected_fdr", "ks_features_tested"}]
    return float(frame[columns].mean().mean())


def context_query_for_trials(trials, seed):
    return split_context_query(
        [t["metadata"]["subject"] for t in trials],
        [t["metadata"]["sp"] for t in trials],
        [t["metadata"]["side"] for t in trials],
        seed=seed,
    )


def encode_split(model, trials, norm, device):
    mu, logvar, _, subjects = encode_trials(model, trials, norm, device)
    return mu, logvar, np.asarray(subjects)


def training_latent_noise_covariance(model, trials, norm, device):
    """Estimate one shared latent-noise covariance from training subjects."""
    mu, logvar, _, subjects = encode_trials(model, trials, norm, device)
    subjects = np.asarray(subjects)
    residuals = np.empty_like(mu)
    for subject in np.unique(subjects):
        mask = subjects == subject
        residuals[mask] = mu[mask] - mu[mask].mean(axis=0)
    between_trial = np.atleast_2d(np.cov(residuals, rowvar=False))
    if model.latent_dim == 1:
        between_trial = between_trial.reshape(1, 1)
    posterior_noise = np.diag(np.exp(logvar).mean(axis=0))
    return between_trial + posterior_noise + np.eye(model.latent_dim) * 1e-6


def fingerprint_tables(model, split_trials, norm, device, cq_seed, include_std=False):
    """Context fingerprints and query-only behavioral summaries."""
    mu, logvar, subjects = encode_split(model, split_trials, norm, device)
    rows, query_features = [], []
    context_codes, query_codes = {}, {}
    for split in context_query_for_trials(split_trials, cq_seed):
        c_mu, c_log = mu[split.context_indices], logvar[split.context_indices]
        q_mu = mu[split.query_indices]
        mean, covariance = moment_matched_posterior(c_mu, c_log)
        row = {"subject": split.subject, **{f"z{i}_mean": v for i, v in enumerate(mean)}}
        if include_std:
            row.update({f"z{i}_sd": v for i, v in enumerate(np.sqrt(np.diag(covariance)))})
        rows.append(row)
        context_codes[split.subject] = c_mu
        query_codes[split.subject] = q_mu
        query_features.extend(compute_trial_features(split_trials[i]) for i in split.query_indices)
    fp = pd.DataFrame(rows).set_index("subject").sort_index()
    qf = pd.DataFrame(query_features)
    return fp, subject_summary(qf).sort_index(), context_codes, query_codes


def evaluate_timing_and_reconstruction(model, trials, norm, device):
    recon, truth, predicted_timing, true_timing = reconstruct(model, trials, norm, device)
    row = {"reconstruction_mse_tracker_units2": float(np.mean((recon - truth) ** 2))}
    for i, name in enumerate(config.TIMING_FEATURES):
        row[f"{name}_r2"] = float(r2_score(true_timing[:, i], predicted_timing[:, i]))
        row[f"{name}_mae_ms"] = float(np.mean(np.abs(true_timing[:, i] - predicted_timing[:, i])) * 1000)
    row["timing_metric_type"] = "reconstruction" if model.encoder_uses_timing else "prediction_from_shape"
    return row


def generate_per_trial_model(model, trials, norm, n_samples, device, seed, shared_covariance):
    mu, logvar, _, subjects = encode_trials(model, trials, norm, device)
    tm, ts, tim_m, tim_s = norm.torch(device)
    rows = []
    for split in context_query_for_trials(trials, seed):
        mean = mu[split.context_indices].mean(axis=0)
        rng = np.random.default_rng(seed + sum(map(ord, split.subject)))
        z = rng.multivariate_normal(mean, shared_covariance, size=n_samples).astype(np.float32)
        query_trials = [trials[i] for i in split.query_indices]
        chosen = rng.integers(0, len(query_trials), size=n_samples)
        cond = np.stack([encode_trial_condition(query_trials[i]["metadata"], model.condition_dim)
                         for i in chosen]).astype(np.float32)
        with torch.no_grad():
            rz, rtz = model.decode(torch.as_tensor(z, device=device), torch.as_tensor(cond, device=device))
            channels = model.input_dim // config.NORMALISED_LENGTH
            trajectories = ((rz * ts + tm).cpu().numpy()).reshape(n_samples, config.NORMALISED_LENGTH, channels)
            timing = norm.denormalise_timing(rtz.cpu().numpy())
        generated = pd.DataFrame([
            features_from_arrays(trajectories[i], max(float(timing[i, 0]), 1e-3), float(timing[i, 1]))
            for i in range(n_samples)
        ])
        empirical = pd.DataFrame([compute_trial_features(t) for t in query_trials])
        row = {"subject": split.subject, **distribution_distances(empirical, generated, KINEMATIC_FEATURES)}
        rows.append(row)
    return finish_fidelity_table(pd.DataFrame(rows))


def load_per_trial_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg.get("model", cfg)
    input_dim = int(ckpt.get("input_dim", len(ckpt["train_mean"])))
    condition_dim = int(ckpt.get("condition_dim", 4))
    model = ConditionalVAE(
        input_dim=input_dim, condition_dim=condition_dim, latent_dim=ckpt["latent_dim"], hidden_dim=model_cfg["hidden_dim"], timing_dim=ckpt["timing_dim"],
        encoder_uses_timing=ckpt.get("encoder_uses_timing", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, NormStats.from_checkpoint(ckpt)


def evaluate_per_trial_run(model, norm, train_trials, val_trials, test_trials, out_dir, cq_seed, device):
    tables = {}
    for name, trials in (("train", train_trials), ("val", val_trials), ("test", test_trials)):
        tables[name] = fingerprint_tables(model, trials, norm, device, cq_seed, include_std=False)
    probes = tune_and_test_ridge(
        tables["train"][0], tables["train"][1], tables["val"][0], tables["val"][1],
        tables["test"][0], tables["test"][1],
    )
    probes.to_csv(out_dir / "behavioral_probe.csv", index=False)
    if model.latent_dim <= 3:
        std_tables = {name: fingerprint_tables(model, trials, norm, device, cq_seed, include_std=True)
                      for name, trials in (("train", train_trials), ("val", val_trials), ("test", test_trials))}
        std_probes = tune_and_test_ridge(
            std_tables["train"][0], std_tables["train"][1], std_tables["val"][0], std_tables["val"][1],
            std_tables["test"][0], std_tables["test"][1],
        )
        std_probes.to_csv(out_dir / "behavioral_probe_mean_plus_sd_ablation.csv", index=False)
    ident = fingerprint_identification(tables["test"][2], tables["test"][3])
    timing = evaluate_timing_and_reconstruction(model, test_trials, norm, device)
    shared_covariance = training_latent_noise_covariance(model, train_trials, norm, device)
    fidelity = generate_per_trial_model(
        model, test_trials, norm, 120, device, cq_seed, shared_covariance
    )
    fidelity.to_csv(out_dir / "context_query_fidelity.csv", index=False)
    return {**timing, **{f"fingerprint_{k}": v for k, v in ident.items()},
            "probe_positive_r2": int((probes.r2_test > 0).sum()),
            "probe_targets": int(len(probes)),
            "mean_ks": mean_ks_statistic(fidelity),
            "mean_ks_rejected_fdr": float(fidelity.ks_rejected_fdr.mean()),
            "median_energy_distance": float(fidelity.energy_distance.median()),
            "mean_mmd_rbf": float(fidelity.mmd_rbf.mean())}


def hierarchical_fingerprint_and_query(model, norm, trials, device, seed):
    rows, query_features, context = [], [], {}
    for split in context_query_for_trials(trials, seed):
        c = [trials[i] for i in split.context_indices]
        from src.hierarchical_vae import _subject_arrays
        cx, _, cc = _subject_arrays(c, norm, device)
        with torch.no_grad():
            mu, logvar = model.encode_subject(cx, cc)
        mean = mu.cpu().numpy()[0]
        row = {"subject": split.subject, **{f"z{i}_mean": v for i, v in enumerate(mean)}}
        rows.append(row)
        context[split.subject] = (mean, np.exp(logvar.cpu().numpy()[0]))
        query_features.extend(compute_trial_features(trials[i]) for i in split.query_indices)
    return pd.DataFrame(rows).set_index("subject").sort_index(), subject_summary(pd.DataFrame(query_features)).sort_index(), context


def generate_hierarchical(model, norm, trials, n_samples, device, seed):
    from src.hierarchical_vae import _subject_arrays
    tm, ts, tim_m, tim_s = norm.tensors(device)
    rows = []
    for split in context_query_for_trials(trials, seed):
        context = [trials[i] for i in split.context_indices]
        query = [trials[i] for i in split.query_indices]
        cx, _, cc = _subject_arrays(context, norm, device)
        with torch.no_grad():
            smu, _ = model.encode_subject(cx, cc)
        rng = np.random.default_rng(seed + sum(map(ord, split.subject)))
        chosen = rng.integers(0, len(query), size=n_samples)
        cond = torch.as_tensor(np.stack([
            encode_condition(query[i]["metadata"]["sp"], query[i]["metadata"]["side"]) for i in chosen
        ]), dtype=torch.float32, device=device)
        trial_z = torch.randn(n_samples, model.trial_dim, device=device)
        with torch.no_grad():
            rz, rtz = model.decode(smu, trial_z, cond)
            channels = model.input_dim // config.NORMALISED_LENGTH
            trajectories = ((rz * ts + tm).cpu().numpy()).reshape(n_samples, config.NORMALISED_LENGTH, channels)
            timing = norm.denormalise_timing(rtz.cpu().numpy())
        generated = pd.DataFrame([features_from_arrays(trajectories[i], max(float(timing[i, 0]), 1e-3),
                                                               float(timing[i, 1])) for i in range(n_samples)])
        empirical = pd.DataFrame([compute_trial_features(t) for t in query])
        rows.append({"subject": split.subject, **distribution_distances(empirical, generated, KINEMATIC_FEATURES)})
    return finish_fidelity_table(pd.DataFrame(rows))


def hierarchical_reconstruction_and_id(model, norm, trials, device, seed):
    from src.hierarchical_vae import _subject_arrays
    tm, ts, tim_m, tim_s = norm.tensors(device)
    truth_traj, pred_traj, truth_timing, pred_timing = [], [], [], []
    centres, chunk_codes, chunk_labels = {}, [], []
    for split in context_query_for_trials(trials, seed):
        context = [trials[i] for i in split.context_indices]
        query = [trials[i] for i in split.query_indices]
        cx, _, cc = _subject_arrays(context, norm, device)
        qx, qt, qc = _subject_arrays(query, norm, device)
        with torch.no_grad():
            smu, _ = model.encode_subject(cx, cc)
            tmu, _ = model.encode_trial(qx, qc, smu)
            rz, rtz = model.decode(smu, tmu, qc)
        centres[split.subject] = smu.cpu().numpy()[0]
        truth_traj.append((qx * ts + tm).cpu().numpy())
        pred_traj.append((rz * ts + tm).cpu().numpy())
        truth_timing.append(norm.denormalise_timing(qt.cpu().numpy()))
        pred_timing.append(norm.denormalise_timing(rtz.cpu().numpy()))
        for start in range(0, len(query), 12):
            chunk = query[start:start + 12]
            if len(chunk) < 4:
                continue
            xx, _, cc2 = _subject_arrays(chunk, norm, device)
            with torch.no_grad():
                qmu, _ = model.encode_subject(xx, cc2)
            chunk_codes.append(qmu.cpu().numpy()[0]); chunk_labels.append(split.subject)
    truth_traj = np.vstack(truth_traj); pred_traj = np.vstack(pred_traj)
    truth_timing = np.vstack(truth_timing); pred_timing = np.vstack(pred_timing)
    labels = sorted(centres); centre_array = np.stack([centres[s] for s in labels])
    scale = centre_array.std(0) + 1e-8
    predicted = [labels[np.linalg.norm((code - centre_array) / scale, axis=1).argmin()] for code in chunk_codes]
    from sklearn.metrics import balanced_accuracy_score
    return {
        "reconstruction_mse_tracker_units2": float(np.mean((pred_traj - truth_traj) ** 2)),
        "movement_time_s_r2": float(r2_score(truth_timing[:, 0], pred_timing[:, 0])),
        "initiation_time_s_r2": float(r2_score(truth_timing[:, 1], pred_timing[:, 1])),
        "movement_time_s_mae_ms": float(np.mean(np.abs(pred_timing[:, 0] - truth_timing[:, 0])) * 1000),
        "initiation_time_s_mae_ms": float(np.mean(np.abs(pred_timing[:, 1] - truth_timing[:, 1])) * 1000),
        "timing_metric_type": "prediction_from_shape",
        "fingerprint_balanced_accuracy": float(balanced_accuracy_score(chunk_labels, predicted)),
        "fingerprint_chance": 1 / len(labels),
        "fingerprint_n_subjects": len(labels),
        "fingerprint_n_query_chunks": len(chunk_labels),
    }


def evaluate_hierarchical(model, norm, train_trials, val_trials, test_trials, out_dir, seed, device):
    tables = {name: hierarchical_fingerprint_and_query(model, norm, trials, device, seed)
              for name, trials in (("train", train_trials), ("val", val_trials), ("test", test_trials))}
    probes = tune_and_test_ridge(tables["train"][0], tables["train"][1], tables["val"][0], tables["val"][1],
                                 tables["test"][0], tables["test"][1])
    probes.to_csv(out_dir / "behavioral_probe.csv", index=False)
    fidelity = generate_hierarchical(model, norm, test_trials, 120, device, seed)
    fidelity.to_csv(out_dir / "context_query_fidelity.csv", index=False)
    reconstruction = hierarchical_reconstruction_and_id(model, norm, test_trials, device, seed)
    return {**reconstruction, "probe_positive_r2": int((probes.r2_test > 0).sum()), "probe_targets": len(probes),
            "mean_ks": mean_ks_statistic(fidelity),
            "mean_ks_rejected_fdr": float(fidelity.ks_rejected_fdr.mean()),
            "median_energy_distance": float(fidelity.energy_distance.median()),
            "mean_mmd_rbf": float(fidelity.mmd_rbf.mean()),
            "mean_subject_kl": float(pd.read_json(out_dir / "history.json").val_subject_kl.tail(10).mean())}


def kmeans_with_selection_null(trials, features, out_dir, seed=42, permutations=200, k_values=None):
    labels = np.array([t["metadata"]["subject"] for t in trials])
    representations = {
        "trajectory": np.stack([t["pos_norm"].reshape(-1) for t in trials]),
        "kinematic_features": features[config.KMEANS_FEATURE_COLUMNS].to_numpy(),
    }
    rng = np.random.default_rng(seed)
    rows = []
    for name, raw in representations.items():
        x = StandardScaler().fit_transform(np.nan_to_num(raw))
        clusterings = []
        # k=28 is pre-specified by the known number of subjects. Passing more
        # values is allowed only for explicitly labelled sensitivity analysis.
        for k in (k_values or [len(np.unique(labels))]):
            pred = KMeans(k, n_init=10, random_state=seed).fit_predict(x)
            clusterings.append((k, pred, adjusted_rand_score(labels, pred), normalized_mutual_info_score(labels, pred)))
        best = max(clusterings, key=lambda item: item[2])
        null = []
        for _ in range(permutations):
            shuffled = rng.permutation(labels)
            null.append(max(adjusted_rand_score(shuffled, pred) for _, pred, _, _ in clusterings))
        rows.append({"representation": name, "k": best[0], "ari": best[2], "nmi": best[3],
                     "selection": "pre_specified_subject_count" if len(k_values or [0]) == 1 else "sensitivity_best_k",
                     "permutation_p": (1 + np.sum(np.asarray(null) >= best[2])) / (permutations + 1),
                     "null_95pct": float(np.percentile(null, 95))})
    pd.DataFrame(rows).to_csv(out_dir / "kmeans_selection_corrected.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=str(ROOT / "data" / "corrected_v2" / "trials.pkl"))
    parser.add_argument("--out", default=str(ROOT / "results" / "corrected_v2"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 3, 4, 8, 16])
    parser.add_argument("--hier-dims", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing-weight", type=float, default=20.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(args.trials, "rb") as handle:
        trials = pickle.load(handle)
    features = pd.DataFrame([compute_trial_features(t) for t in trials])
    features.to_csv(out / "features_corrected.csv", index=False)
    train_trials, val_trials, test_trials = split_subjects(trials, 17, 4, 7, args.seed)
    save_json(out / "protocol.json", {"subject_split_seed": args.seed, "context_query_seed": config.CONTEXT_QUERY_SEED,
              "n_trials": len(trials), "n_subjects": len({t['metadata']['subject'] for t in trials}),
              "position_unit": config.POSITION_UNIT, "dims": args.dims, "hierarchical_dims": args.hier_dims,
              "timing_weight": args.timing_weight})
    kmeans = kmeans_with_selection_null(
        trials, features, out, args.seed, 20 if args.smoke else 200,
        k_values=[28],
    )
    spline = evaluate_spline_baseline(test_trials)
    spline_pca = {n: evaluate_spline_pca_baseline(train_trials, test_trials, n_components=n)["mean_mse"]
                  for n in args.dims}
    save_json(out / "baselines.json", {"kmeans": kmeans, "spline_per_trial": spline["mean_mse"],
                                        "spline_pca": spline_pca})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summaries = []
    epochs = min(args.epochs, 3) if args.smoke else args.epochs
    dims = args.dims[:1] if args.smoke else args.dims
    for encoder_uses_timing in (True, False):
        variant = "joint_reconstruction" if encoder_uses_timing else "trajectory_only_timing_prediction"
        for n in dims:
            run = out / "runs" / f"per_trial_{variant}_z{n}"
            cfg = RunConfig(seed=args.seed, latent_dim=n, epochs=epochs, patience=25,
                            encoder_uses_timing=encoder_uses_timing, balance_subjects=True,
                            timing_weight=args.timing_weight)
            if (run / "checkpoint.pt").exists() and (run / "history.json").exists():
                print(f"Resuming from completed run: {run}")
                model, norm = load_per_trial_checkpoint(run / "checkpoint.pt", device)
            else:
                model, _, run = train_vae(train_trials, val_trials, cfg, run, device)
                ckpt = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
                norm = NormStats.from_checkpoint(ckpt)
            summary = evaluate_per_trial_run(model, norm, train_trials, val_trials, test_trials,
                                             run, config.CONTEXT_QUERY_SEED, device)
            summaries.append({"model": "per_trial_cvae", "variant": variant, "latent_dim": n, **summary})
    hdims = args.hier_dims[:1] if args.smoke else args.hier_dims
    for n in hdims:
        run = out / "runs" / f"hierarchical_v2_subject_z{n}_trial_z4"
        if (run / "checkpoint.pt").exists() and (run / "history.json").exists():
            print(f"Resuming from completed run: {run}")
            ckpt = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
            model = HierarchicalCVAE(subject_dim=ckpt["subject_dim"], trial_dim=ckpt["trial_dim"]).to(device)
            model.load_state_dict(ckpt["model_state"])
            norm_values = dict(ckpt["norm"])
            for key in ("trajectory_mean", "trajectory_std", "timing_mean", "timing_std"):
                norm_values[key] = np.asarray(norm_values[key])
            norm = HierarchicalNorm(**norm_values)
        else:
            model, norm, _ = train_hierarchical(train_trials, val_trials, run, subject_dim=n, trial_dim=4,
                                                 epochs=epochs, seed=args.seed, device=device,
                                                 patience=25)
        summary = evaluate_hierarchical(model, norm, train_trials, val_trials, test_trials,
                                        run, config.CONTEXT_QUERY_SEED, device)
        summaries.append({"model": "hierarchical_cvae", "variant": "subject_trial", "latent_dim": n, **summary})
    pd.DataFrame(summaries).to_csv(out / "model_summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
