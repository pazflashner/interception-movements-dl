"""Empirical-bootstrap reference for finite-sample minimum-jerk distances."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from scripts.run_corrected_study import context_query_for_trials

SOURCE = ROOT / "studies" / "final_strategy_evaluation" / "results" / "minimum_jerk"
OUT = ROOT / "studies" / "review_corrected_evaluation" / "results" / "sampling_reference"
FEATURES = ("mj_fit_error", "mj_first_duration_s", "mj_first_amplitude",
            "mj_secondary_amplitude_fraction", "mj_mean_overlap_pct")


def categorical_distance(a, b):
    pa=np.stack([(a==k).mean(axis=1) for k in range(1,5)],axis=1)
    pb=np.stack([(b==k).mean(axis=1) for k in range(1,5)],axis=1)
    midpoint=(pa+pb)/2
    with np.errstate(divide="ignore",invalid="ignore"):
        terms=np.where(pa>0,pa*np.log2(pa/np.maximum(midpoint,1e-30)),0)
        terms+=np.where(pb>0,pb*np.log2(pb/np.maximum(midpoint,1e-30)),0)
    return np.abs(pa-pb).sum(axis=1)/2,terms.sum(axis=1)/2


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (config.DATA_PROCESSED_DIR/"canonical_trials.pkl").open("rb") as handle:
        trials=pickle.load(handle)
    real=pd.read_csv(config.RESULTS_DIR/"submovements_real.csv").set_index("trial_id")
    observed=pd.read_csv(SOURCE/"analysis"/"participant_fidelity_raw.csv")
    rng=np.random.default_rng(20260902); repeats=500; seed_replicates=3
    rows=[]; null_vectors={}
    for split in context_query_for_trials(trials,config.CONTEXT_QUERY_SEED):
        ids=[trials[i]["metadata"]["trial_id"] for i in split.query_indices]
        empirical=real.loc[ids]
        for m in (10,120):
            for feature in (*FEATURES,"mj_n_components"):
                values=empirical[feature].dropna().to_numpy()
                left=rng.choice(values,(repeats*seed_replicates,len(values)))
                right=rng.choice(values,(repeats*seed_replicates,m))
                if feature=="mj_n_components":
                    tv,jsd=categorical_distance(left,right)
                    stats={"count_total_variation":tv,"count_jsd":jsd}
                else:
                    stats={f"ks_{feature}":ks_2samp(left,right,axis=1,method="asymp").statistic}
                for metric,values_drawn in stats.items():
                    averaged=values_drawn.reshape(repeats,seed_replicates).mean(axis=1)
                    null_vectors.setdefault((m,metric),[]).append(averaged)
                    rows.append({"subject":split.subject,"metric":metric,"n_empirical":len(values),
                                 "n_generated":m,"null_mean":averaged.mean(),"null_95":np.quantile(averaged,.95)})
        print(f"Sampling reference: {split.subject}",flush=True)
    pd.DataFrame(rows).to_csv(OUT/"participant_reference.csv",index=False)
    aggregate=[]
    for (m,metric),vectors in null_vectors.items():
        distribution=np.stack(vectors).mean(axis=0)
        for dim in (3,8):
            value=observed[observed.latent_dim==dim].groupby("subject")[metric].mean().mean()
            aggregate.append({"latent_dim":dim,"metric":metric,"n_generated":m,
               "observed_at_n10":value,"reference_mean":distribution.mean(),
               "reference_95":np.quantile(distribution,.95),
               "observed_minus_reference":value-distribution.mean()})
    pd.DataFrame(aggregate).to_csv(OUT/"summary.csv",index=False)
    (OUT/"protocol.json").write_text(json.dumps({"bootstrap_repeats":repeats,"seed":20260902,
      "model_seed_replicates_averaged":seed_replicates,"n_generated":[10,120],
      "scope":"Empirical plug-in reference, not a formal goodness-of-fit test or a hard KS floor. Each resample is independent conditional on the observed query distribution; fitting and training uncertainty are not included. No new model samples were fitted."},indent=2))


if __name__=="__main__":
    main()
