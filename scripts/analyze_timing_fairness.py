"""Validation-only calibration and common-head timing comparison on frozen codes."""
from __future__ import annotations

import copy
import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon, false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from scripts.run_corrected_study import load_per_trial_checkpoint
from src.baseline_spline import SplinePCARepresentation
from src.confirmatory_protocol import make_participant_folds, partition_trials
from src.evaluate import encode_trials, reconstruct
from src.trajectory_view import project_trials_to_table_plane, select_trials_window
from src.vae_model import encode_timing, encode_trial_condition, inverse_timing, transform_timing

SOURCE = ROOT / "studies" / "final_strategy_evaluation"
OUT = ROOT / "studies" / "review_corrected_evaluation" / "results" / "timing_fairness"


def weights(trials):
    ids = np.array([t["metadata"]["subject"] for t in trials])
    count = {s: (ids == s).sum() for s in np.unique(ids)}
    w = np.array([1/count[s] for s in ids], float)
    return w / w.mean()


def weighted_median(values, w):
    order = np.argsort(values); values, w = values[order], w[order]
    return float(values[np.searchsorted(np.cumsum(w), w.sum()/2)])


def calibration_factor(true, predicted, w):
    valid = predicted > 1e-9
    return weighted_median(true[valid] / predicted[valid], w[valid] * predicted[valid]) if valid.any() else 1.0


def fit_common_head(xtr, ytr, train_trials, xval, yval, validation_trials, seed):
    xs, ys = StandardScaler().fit(xtr), StandardScaler().fit(transform_timing(ytr, "log"))
    a, b = xs.transform(xtr), ys.transform(transform_timing(ytr, "log"))
    av, bv = xs.transform(xval), ys.transform(transform_timing(yval, "log"))
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
                         learning_rate_init=1e-3, alpha=1e-4, batch_size=64,
                         max_iter=1, warm_start=True, random_state=seed, shuffle=True)
    wt, wv = weights(train_trials), weights(validation_trials)
    best, best_loss, stale, best_epoch = None, np.inf, 0, 0
    for epoch in range(1, 401):
        model.partial_fit(a, b, sample_weight=wt)
        loss = float(np.average(np.mean((model.predict(av)-bv)**2, axis=1), weights=wv))
        if loss < best_loss - 1e-7:
            best, best_loss, stale, best_epoch = copy.deepcopy(model), loss, 0, epoch
        else:
            stale += 1
        if stale >= 40:
            break
    return xs, ys, best, best_epoch, best_loss


def codes(family, dim, seed, fold, train, validation, test):
    if family == "cvae":
        run = SOURCE / "runs" / family / f"fold{fold}" / f"cvae_z{dim}_seed{seed}"
        model, norm = load_per_trial_checkpoint(run / "checkpoint.pt", "cpu")
        return [np.hstack([encode_trials(model, part, norm, "cpu")[0],
                           np.stack([encode_trial_condition(t["metadata"]) for t in part])])
                for part in (train, validation, test)]
    old = json.loads((SOURCE / "runs" / family / f"fold{fold}" / f"spline_pca_z{dim}" / "result.json").read_text())
    rep = SplinePCARepresentation(dim, include_timing=False,
             standardize_coefficients=old["standardize_spline_coefficients"]).fit(train)
    return [np.hstack([rep.encode(part), np.stack([encode_trial_condition(t["metadata"]) for t in part])])
            for part in (train, validation, test)]


def summarize(raw):
    detail=[]
    participant=[]
    for keys,g in raw.groupby(["comparison","model_family","latent_dim","seed"],dropna=False):
        for endpoint in ("movement","initiation"):
            errors = g.assign(err=(g[f"{endpoint}_true_s"]-g[f"{endpoint}_pred_s"]).abs()).groupby("subject").err.mean()*1000
            metadata = dict(zip(["comparison","model_family","latent_dim","seed"],keys)) | {"endpoint": endpoint}
            detail.append(metadata | {"mae_ms_subject_balanced": errors.mean(),
                "r2_trial_pooled":r2_score(g[f"{endpoint}_true_s"],g[f"{endpoint}_pred_s"])})
            participant.extend(metadata | {"subject": s, "mae_ms": value} for s, value in errors.items())
    detail=pd.DataFrame(detail); detail.to_csv(OUT/"summary_by_seed.csv",index=False)
    detail.groupby(["comparison","model_family","latent_dim","endpoint"],dropna=False).agg(
      mae_ms_mean=("mae_ms_subject_balanced","mean"),mae_ms_sd=("mae_ms_subject_balanced","std"),
      r2_mean=("r2_trial_pooled","mean")).reset_index().to_csv(OUT/"summary.csv",index=False)
    participant=pd.DataFrame(participant).groupby(
        ["comparison","model_family","latent_dim","endpoint","subject"],as_index=False).mae_ms.mean()
    participant.to_csv(OUT/"participant_seed_averaged.csv",index=False)
    paired=[]
    for keys,g in participant.groupby(["comparison","latent_dim","endpoint"]):
        wide=g.pivot(index="subject",columns="model_family",values="mae_ms")
        if len(wide)!=28 or wide.isna().any().any():
            raise ValueError("Timing comparison requires 28 paired participants")
        d=(wide.cvae-wide.spline_pca).to_numpy()
        paired.append(dict(zip(["comparison","latent_dim","endpoint"],keys)) | {
            "n_participants":len(d),"cvae_mean_ms":wide.cvae.mean(),"spline_mean_ms":wide.spline_pca.mean(),
            "mean_difference_ms":d.mean(),"cvae_better_participants":int((d<0).sum()),
            "wilcoxon_p":float(wilcoxon(d,zero_method="pratt").pvalue) if np.any(d) else 1.0})
    paired=pd.DataFrame(paired)
    paired["p_fdr_bh"]=false_discovery_control(paired.wilcoxon_p.to_numpy())
    paired.to_csv(OUT/"paired_comparisons.csv",index=False)
    print(pd.read_csv(OUT/"summary.csv").to_string(index=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    if args.summarize_only:
        summarize(pd.read_csv(OUT/"predictions.csv"))
        return
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    OUT.mkdir(parents=True, exist_ok=True)
    with (config.DATA_PROCESSED_DIR / "canonical_trials.pkl").open("rb") as handle:
        all_trials = project_trials_to_table_plane(select_trials_window(pickle.load(handle), config.WINDOW_GO_TO_ARRIVAL))
    folds = make_participant_folds([t["metadata"]["subject"] for t in all_trials])
    rows=[]
    for fold, split in enumerate(folds):
        train, validation, test = partition_trials(all_trials, split)
        timing = [np.stack([encode_timing(t) for t in part]) for part in (train, validation, test)]
        test_ids = [t["metadata"]["subject"] for t in test]
        for dim in (3, 8):
            for family in ("cvae", "spline_pca"):
                for seed in (42,43,44):
                    encoded = codes(family, dim, seed, fold, train, validation, test)
                    xs, ys, head, epoch, val_loss = fit_common_head(encoded[0], timing[0], train, encoded[1], timing[1], validation, seed)
                    print(f"Common timing head: {family} n={dim} fold={fold} seed={seed}, epoch={epoch}", flush=True)
                    pred = inverse_timing(ys.inverse_transform(head.predict(xs.transform(encoded[2]))), "log")
                    for i, subject in enumerate(test_ids):
                        rows.append({"comparison":"common_mlp", "model_family":family,"latent_dim":dim,"outer_fold":fold,"seed":seed,
                                     "subject":subject,"trial_id":test[i]["metadata"]["trial_id"],
                                     "movement_true_s":timing[2][i,0],"movement_pred_s":pred[i,0],
                                     "initiation_true_s":timing[2][i,1],"initiation_pred_s":pred[i,1],
                                     "best_epoch":epoch,"validation_log_mse":val_loss})
        # Validate-only multiplicative calibration of the original heads.
        for family, dims, seeds in (("cvae",(3,8),(42,43,44)),("spline_pca",(3,8),(None,))):
            for dim in dims:
                for seed in seeds:
                    if family == "cvae":
                        run=SOURCE/"runs"/family/f"fold{fold}"/f"cvae_z{dim}_seed{seed}"
                        model,norm=load_per_trial_checkpoint(run/"checkpoint.pt","cpu")
                        val_true=np.stack([encode_timing(t) for t in validation]); val_pred=reconstruct(model,validation,norm,"cpu")[2]
                    else:
                        run=SOURCE/"runs"/family/f"fold{fold}"/f"spline_pca_z{dim}"
                        old=json.loads((run/"result.json").read_text()); rep=SplinePCARepresentation(
                            n_components=dim,include_timing=False,
                            standardize_coefficients=old["standardize_spline_coefficients"]).fit(train)
                        from src.confirmatory_spline import TimingRidge, _conditions
                        tm=TimingRidge.fit(rep,train,validation); val_true=timing[1]; val_pred=tm.predict(rep.encode(validation),_conditions(validation))
                    factors=[calibration_factor(val_true[:,j],val_pred[:,j],weights(validation)) for j in range(2)]
                    test_frame=pd.read_csv(run/"timing_predictions.csv")
                    for _, r in test_frame.iterrows():
                        rows.append({"comparison":"validation_calibrated_original","model_family":family,"latent_dim":dim,"outer_fold":fold,"seed":seed,
                          "subject":r.subject,"trial_id":r.trial_id,"movement_true_s":r.movement_time_true_s,
                          "movement_pred_s":r.movement_time_pred_s*factors[0],"initiation_true_s":r.initiation_time_true_s,
                          "initiation_pred_s":r.initiation_time_pred_s*factors[1],"movement_factor":factors[0],"initiation_factor":factors[1]})
    raw=pd.DataFrame(rows); raw.to_csv(OUT/"predictions.csv",index=False)
    summarize(raw)


if __name__ == "__main__":
    main()
