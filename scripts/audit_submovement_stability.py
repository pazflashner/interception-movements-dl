"""Re-fit a stratified subset with more restarts to audit optimizer stability."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_submovements import fit_one
from src.submovements import SubmovementConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=str(ROOT / "data" / "corrected_v2" / "trials.pkl"))
    parser.add_argument("--base", default=str(ROOT / "results" / "final_study" / "submovements_real.csv"))
    parser.add_argument("--out", default=str(ROOT / "results" / "final_study" / "submovement_stability.csv"))
    parser.add_argument("--sample", type=int, default=112)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    with open(args.trials, "rb") as handle:
        trials = pickle.load(handle)
    by_id = {trial["metadata"]["trial_id"]: trial for trial in trials}
    rng = np.random.default_rng(2026)
    chosen = []
    per_subject = max(1, args.sample // base.subject.nunique())
    for _, group in base.groupby("subject"):
        take = min(per_subject, len(group))
        chosen.extend(rng.choice(group.trial_id.to_numpy(), size=take, replace=False).tolist())
    if len(chosen) < args.sample:
        remaining = base[~base.trial_id.isin(chosen)].trial_id.to_numpy()
        chosen.extend(rng.choice(remaining, size=min(args.sample - len(chosen), len(remaining)), replace=False).tolist())
    selected_trials = [by_id[trial_id] for trial_id in chosen]
    cfg = SubmovementConfig(restarts=8, max_nfev=1000)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        high_rows = list(pool.map(fit_one, selected_trials, repeat(cfg)))
    high = pd.DataFrame(high_rows)
    compare = base[base.trial_id.isin(chosen)].merge(high, on="trial_id", suffixes=("_base", "_high"))
    compare["same_selected_count"] = compare.mj_n_components_base == compare.mj_n_components_high
    compare["relative_error_change"] = (
        compare.mj_fit_error_high - compare.mj_fit_error_base
    ) / np.maximum(compare.mj_fit_error_base, 1e-8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    compare.to_csv(args.out, index=False)
    print("same count", compare.same_selected_count.mean())
    print("median relative error change", compare.relative_error_change.median())


if __name__ == "__main__":
    main()
