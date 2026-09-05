"""Summaries for separately versioned post-review controls; no model fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, false_discovery_control
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_review_controls import OUT, TRAINED, CORRECTED, MJ_FEATURES, load_trials
from scripts.run_corrected_study import context_query_for_trials
from scripts.calibrate_submovement_sampling import categorical_distance
from scripts.generate_final_samples import categorical_distances
from src.confirmatory_protocol import make_participant_folds, partition_trials
from src.features import compute_trial_features
from src.context_query import subject_summary
from src.statistical_tests import paired_wilcoxon
import config


def direct_context_baseline():
    trials = load_trials()
    folds = make_participant_folds([t["metadata"]["subject"] for t in trials])
    rows = []
    for fold, partition in enumerate(folds):
        train, val, test = partition_trials(trials, partition)
        development = pd.DataFrame([compute_trial_features(t) for t in train + val])
        # Match probe refitting access: constant target calculated on development
        # query summaries, not on test participants or on all development trials.
        dev_queries = []
        for group in (train, val):
            for split in context_query_for_trials(group, config.CONTEXT_QUERY_SEED):
                dev_queries.extend(compute_trial_features(group[i]) for i in split.query_indices)
        constant = subject_summary(pd.DataFrame(dev_queries)).mean()
        for split in context_query_for_trials(test, config.CONTEXT_QUERY_SEED):
            context = subject_summary(pd.DataFrame([compute_trial_features(test[i]) for i in split.context_indices])).loc[split.subject]
            query = subject_summary(pd.DataFrame([compute_trial_features(test[i]) for i in split.query_indices])).loc[split.subject]
            for target in query.index:
                rows.append({"outer_fold": fold, "subject": split.subject, "target": target,
                    "true": query[target], "context_prediction": context[target], "development_mean": constant[target]})
    baseline = pd.DataFrame(rows)
    baseline.to_csv(OUT / "direct_context_predictions.csv", index=False)
    probe = pd.read_csv(CORRECTED / "results/behavioral_probes/out_of_fold_predictions.csv")
    probe = probe[(probe.model_family == "cvae") & (probe.fingerprint == "mean") & probe.latent_dim.isin([3, 8])]
    joined = probe.merge(baseline, on=["outer_fold", "subject", "target"], validate="many_to_one", suffixes=("", "_checked"))
    np.testing.assert_allclose(joined.true, joined.true_checked, atol=1e-10)
    scores = []
    for (dim, seed, target), group in joined.groupby(["latent_dim", "training_seed", "target"]):
        for arm, column in [("cvae_probe", "predicted"), ("direct_context", "context_prediction"), ("development_constant", "development_mean")]:
            scores.append({"latent_dim": dim, "seed": seed, "target": target, "arm": arm,
                "pooled_r2": r2_score(group.true, group[column]), "mae": np.abs(group.true-group[column]).mean(),
                "n_subjects": len(group)})
    scores = pd.DataFrame(scores)
    scores.groupby(["latent_dim", "target", "arm"], as_index=False)[["pooled_r2", "mae"]].mean().to_csv(OUT / "direct_context_summary.csv", index=False)
    print("direct-context summary complete", flush=True)


def component_metrics(e, g):
    row = {}
    for feature in MJ_FEATURES:
        row["ks_"+feature] = ks_2samp(e[feature], g[feature]).statistic
    for column, suffix in [("mj_n_components", ""), ("mj_n_components_bic", "_bic")]:
        row["count_total_variation"+suffix], row["count_jsd"+suffix] = categorical_distances(e[column].to_numpy(), g[column].to_numpy())
    return row


def component_summary():
    frame = pd.read_csv(OUT / "matched_components.csv")
    assert frame.job_id.is_unique
    assert frame.mj_fit_completed.eq(True).all(), "report execution failures before computing complete-case distances"
    real = frame[frame.kind == "recorded"].copy()
    generated = frame[frame.kind == "generated"].copy()
    assert len(generated) == 1680
    trials = load_trials()
    expected = {trials[i]["metadata"]["trial_id"] for s in context_query_for_trials(trials, config.CONTEXT_QUERY_SEED) for i in s.query_indices}
    assert set(real.trial_id) == expected
    rows = []
    for (dim, fold, seed, subject), g in generated.groupby(["latent_dim", "outer_fold", "seed", "subject"]):
        e = real[real.subject == subject]
        assert len(g) == 10 and len(e) > 0
        rows.append({"latent_dim": int(dim), "outer_fold": int(fold), "seed": int(seed), "subject": subject,
                     "n_empirical": len(e), "n_generated": len(g), **component_metrics(e, g)})
    metrics = [*['ks_'+f for f in MJ_FEATURES], "count_total_variation", "count_jsd", "count_total_variation_bic", "count_jsd_bic"]
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "matched_component_participant_raw.csv", index=False)
    averaged = result.groupby(["latent_dim", "outer_fold", "subject"], as_index=False)[metrics].mean()
    averaged.to_csv(OUT / "matched_component_participant.csv", index=False)
    averaged.groupby("latent_dim", as_index=False)[metrics].mean().to_csv(OUT / "matched_component_summary.csv", index=False)
    a, b = [averaged[averaged.latent_dim == d].set_index("subject").sort_index() for d in [3, 8]]
    paired = []
    for metric in metrics:
        d = a[metric] - b[metric]
        paired.append({"metric": metric, "n3_mean": a[metric].mean(), "n8_mean": b[metric].mean(),
                       "n8_better_n": int((d > 0).sum()), "p": paired_wilcoxon(d).pvalue})
    paired = pd.DataFrame(paired)
    paired["p_fdr_bh"] = false_discovery_control(paired.p)
    paired.to_csv(OUT / "matched_component_paired.csv", index=False)
    old_real = pd.read_csv(ROOT / "studies/strategy_window_comparison/results/submovements_real.csv").set_index("trial_id")
    old_generated = pd.concat([pd.read_csv(p) for p in sorted((TRAINED / "results/minimum_jerk").glob("cvae_z*_fold*_seed*/generated_submovements.csv"))])
    old_generated["job_id"] = old_generated.run + "-" + old_generated.subject + "-" + old_generated.sample_id.astype(str)
    old_generated = old_generated.set_index("job_id")
    changes = []
    for kind, group in frame.groupby("kind"):
        old = old_real.loc[group.trial_id] if kind == "recorded" else old_generated.loc[group.job_id]
        changes.append({"kind": kind, "n": len(group),
            "selected_converged_n": int(group.mj_selected_optimizer_converged.sum()),
            "all_candidates_converged_n": int(group.mj_all_candidates_converged.sum()),
            "count_agreement_with_legacy": float(np.mean(group.mj_n_components.to_numpy() == old.mj_n_components.to_numpy())),
            "median_error": group.mj_fit_error.median()})
    pd.DataFrame(changes).to_csv(OUT / "matched_component_diagnostics.csv", index=False)
    sensitivity = pd.read_csv(OUT / "matched_component_sensitivity.csv")
    assert len(sensitivity) == 168 and sensitivity.mj_fit_completed.eq(True).all()
    sensitivity_rows = []
    for kind, group in sensitivity.groupby("kind"):
        base = frame.set_index("job_id").loc[group.job_id]
        sensitivity_rows.append({"kind": kind, "n": len(group),
            "count_agreement": float(np.mean(group.mj_n_components.to_numpy() == base.mj_n_components.to_numpy())),
            "base_selected_converged_n": int(base.mj_selected_optimizer_converged.sum()),
            "high_budget_selected_converged_n": int(group.mj_selected_optimizer_converged.sum()),
            "mean_error_change": float(np.mean(group.mj_fit_error.to_numpy()-base.mj_fit_error.to_numpy()))})
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "matched_component_sensitivity_summary.csv", index=False)
    print("component summaries complete", flush=True)


def sampling_reference():
    frame = pd.read_csv(OUT / "matched_components.csv")
    real = frame[frame.kind == "recorded"]
    observed = pd.read_csv(OUT / "matched_component_summary.csv").set_index("latent_dim")
    rng = np.random.default_rng(20260905)
    vectors, individual = {}, []
    for subject, e in real.groupby("subject"):
        for feature in MJ_FEATURES + ["mj_n_components"]:
            values = e[feature].to_numpy()
            left = rng.choice(values, (1500, len(values)))
            right = rng.choice(values, (1500, 10))
            if feature == "mj_n_components":
                tv, jsd = categorical_distance(left, right)
                stats = {"count_total_variation": tv, "count_jsd": jsd}
            else:
                stats = {"ks_"+feature: ks_2samp(left, right, axis=1, method="asymp").statistic}
            for metric, values_drawn in stats.items():
                averaged = values_drawn.reshape(500, 3).mean(axis=1)
                vectors.setdefault(metric, []).append(averaged)
                individual.append({"subject": subject, "metric": metric, "reference_mean": averaged.mean(), "n_query": len(e)})
        print("matched sampling reference", subject, flush=True)
    pd.DataFrame(individual).to_csv(OUT / "matched_component_sampling_participant.csv", index=False)
    rows = []
    for metric, draws in vectors.items():
        distribution = np.stack(draws).mean(axis=0)
        for dim in (3, 8):
            rows.append({"latent_dim": dim, "metric": metric, "observed": observed.loc[dim, metric],
                         "reference_mean": distribution.mean(), "reference_95": np.quantile(distribution, .95)})
    pd.DataFrame(rows).to_csv(OUT / "matched_component_sampling.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["context", "components"])
    args = parser.parse_args()
    if args.stage == "context": direct_context_baseline()
    else:
        component_summary()
        sampling_reference()
