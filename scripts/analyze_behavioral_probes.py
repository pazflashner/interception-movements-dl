"""Aggregate every saved participant-level probe without hiding negative R2."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "review_corrected_evaluation"
OUT = STUDY / "results" / "behavioral_probes"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    score_rows, pred_rows = [], []
    for result_path in STUDY.glob("runs/*/fold*/*/result.json"):
        run = result_path.parent
        meta = json.loads(result_path.read_text())
        family = meta["model_family"]
        if family == "condition_ridge":
            continue
        for suffix, kind in (("", "mean"), ("_mean_plus_sd", "mean_plus_sd")):
            score_path = run / ("behavioral_probe.csv" if not suffix else "behavioral_probe_mean_plus_sd_ablation.csv")
            pred_path = run / f"behavioral_probe{suffix}_predictions.csv"
            if not score_path.exists() or not pred_path.exists():
                continue
            common = {"model_family": family, "latent_dim": int(meta["latent_dim"]),
                      "outer_fold": int(meta["outer_fold"]),
                      "training_seed": meta.get("training_seed"), "fingerprint": kind}
            score_rows.extend({**common, **row} for row in pd.read_csv(score_path).to_dict("records"))
            pred_rows.extend({**common, **row} for row in pd.read_csv(pred_path).to_dict("records"))
    scores, predictions = pd.DataFrame(score_rows), pd.DataFrame(pred_rows)
    scores.to_csv(OUT / "per_fold_scores.csv", index=False)
    predictions.to_csv(OUT / "out_of_fold_predictions.csv", index=False)
    by_seed = []
    for keys, group in predictions.groupby(["model_family", "latent_dim", "training_seed", "fingerprint", "target"], dropna=False):
        true, pred, base = group.true.to_numpy(), group.predicted.to_numpy(), group.train_mean_baseline.to_numpy()
        by_seed.append(dict(zip(["model_family", "latent_dim", "training_seed", "fingerprint", "target"], keys)) | {
            "n_participants": len(group), "r2_oof_pooled": r2_score(true, pred),
            "mae_model": np.mean(np.abs(true-pred)), "mae_train_mean_baseline": np.mean(np.abs(true-base)),
            "model_minus_baseline_mae": np.mean(np.abs(true-pred))-np.mean(np.abs(true-base))})
    by_seed = pd.DataFrame(by_seed)
    by_seed.to_csv(OUT / "pooled_by_seed.csv", index=False)
    summary = by_seed.groupby(["model_family", "latent_dim", "fingerprint", "target"], dropna=False).agg(
        n_seed_evaluations=("r2_oof_pooled", "size"), r2_oof_mean=("r2_oof_pooled", "mean"),
        r2_oof_sd=("r2_oof_pooled", "std"), mae_model_mean=("mae_model", "mean"),
        mae_baseline_mean=("mae_train_mean_baseline", "mean"),
        model_minus_baseline_mae_mean=("model_minus_baseline_mae", "mean")).reset_index()
    fold_summary = scores.groupby(["model_family", "latent_dim", "fingerprint", "target"], dropna=False).agg(
        r2_fold_mean=("r2_test", "mean"), r2_fold_median=("r2_test", "median"),
        positive_folds=("r2_test", lambda value: int((value > 0).sum())), n_folds=("r2_test", "size")).reset_index()
    summary.merge(fold_summary, on=["model_family", "latent_dim", "fingerprint", "target"], how="left").to_csv(
        OUT / "summary.csv", index=False)
    print(f"Probe records: {len(scores)} scores, {len(predictions)} participant predictions")


if __name__ == "__main__":
    main()
