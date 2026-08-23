"""Sensitivity of component count to optimizer, temporal bounds, and filter."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.extract_submovements import fit_one
from src.submovements import SubmovementConfig


CONFIGS = {
    "optimizer_4_restarts": SubmovementConfig(restarts=4, max_nfev=700),
    "jason_167ms_constraints": SubmovementConfig(
        min_duration_s=0.167,
        min_onset_spacing_s=0.167,
        restarts=2,
        max_nfev=400,
    ),
    "five_hz_filter": SubmovementConfig(
        cutoff_hz=5.0,
        restarts=2,
        max_nfev=400,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, default=config.DATA_PROCESSED_DIR / "canonical_trials.pkl")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "studies" / "strategy_window_comparison" / "results" / "submovements_real.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "studies" / "final_strategy_evaluation" / "results" / "minimum_jerk" / "assumption_sensitivity.csv",
    )
    parser.add_argument("--per-subject", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=min(2, os.cpu_count() or 1))
    args = parser.parse_args()

    with args.trials.open("rb") as handle:
        trials = pickle.load(handle)
    base = pd.read_csv(args.base)
    by_id = {trial["metadata"]["trial_id"]: trial for trial in trials}
    rng = np.random.default_rng(20260824)
    selected_ids = []
    for subject, group in base.groupby("subject"):
        selected_ids.extend(
            rng.choice(
                group.trial_id.to_numpy(),
                size=min(args.per_subject, len(group)),
                replace=False,
            ).tolist()
        )
    selected_trials = [by_id[trial_id] for trial_id in selected_ids]
    base_columns = [
        "trial_id",
        "subject",
        "mj_n_components",
        "mj_n_components_bic",
        "mj_fit_error",
    ]
    base_selected = base[base.trial_id.isin(selected_ids)][base_columns].copy()
    rows = []
    for name, fit_cfg in CONFIGS.items():
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            fitted = list(pool.map(fit_one, selected_trials, repeat(fit_cfg)))
        alternative = pd.DataFrame(fitted)[
            [
                "trial_id",
                "mj_n_components",
                "mj_n_components_bic",
                "mj_fit_error",
                "mj_fit_success",
            ]
        ]
        comparison = base_selected.merge(
            alternative,
            on="trial_id",
            how="inner",
            suffixes=("_base", "_alternative"),
            validate="one_to_one",
        )
        comparison["sensitivity"] = name
        comparison["same_threshold_count"] = (
            comparison.mj_n_components_base
            == comparison.mj_n_components_alternative
        )
        comparison["same_bic_count"] = (
            comparison.mj_n_components_bic_base
            == comparison.mj_n_components_bic_alternative
        )
        rows.append(comparison)
        print(
            f"{name}: same threshold count="
            f"{comparison.same_threshold_count.mean():.3f}",
            flush=True,
        )

    result = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    summary = (
        result.groupby("sensitivity", as_index=False)
        .agg(
            n_trials=("trial_id", "size"),
            same_threshold_count_rate=("same_threshold_count", "mean"),
            same_bic_count_rate=("same_bic_count", "mean"),
            median_base_error=("mj_fit_error_base", "median"),
            median_alternative_error=("mj_fit_error_alternative", "median"),
        )
    )
    summary.to_csv(args.out.with_name("assumption_sensitivity_summary.csv"), index=False)
    protocol = {
        "sample_seed": 20260824,
        "participants": int(base.subject.nunique()),
        "trials_per_participant": args.per_subject,
        "n_trials": len(selected_trials),
        "base": "10 Hz, 100 ms minimum duration, 50 ms onset spacing, 2 restarts",
        "sensitivities": {name: cfg.__dict__ for name, cfg in CONFIGS.items()},
    }
    args.out.with_name("assumption_sensitivity_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
