"""Extract Jason-compatible minimum-jerk features for every retained trial."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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

from src.submovements import SubmovementConfig, decompose_trial


def fit_one(trial: dict, cfg: SubmovementConfig) -> dict:
    meta = trial["metadata"]
    row = {
        "trial_id": meta.get("trial_id", ""),
        "subject": meta.get("subject", ""),
        "sp": meta.get("sp"),
        "side": meta.get("side"),
        "rep": meta.get("rep"),
        "responseText": meta.get("responseText", ""),
        "successful": meta.get("successful", np.nan),
    }
    try:
        result = decompose_trial(trial, cfg)
        row.update(result.summary())
        row["mj_parameters_json"] = json.dumps(result.selected.parameters.tolist())
        row["mj_fit_success"] = True
        row["mj_failure"] = ""
        for n in range(1, cfg.max_components + 1):
            fit = result.fits.get(n)
            row[f"mj_error_k{n}"] = fit.normalized_error if fit else np.nan
            row[f"mj_bic_k{n}"] = fit.bic if fit else np.nan
            row[f"mj_nfev_k{n}"] = fit.nfev if fit else np.nan
    except Exception as exc:
        row.update({
            "mj_fit_success": False,
            "mj_failure": f"{type(exc).__name__}: {exc}",
            "mj_n_components": np.nan,
            "mj_n_components_bic": np.nan,
            "mj_fit_error": np.nan,
        })
    return row


def read_completed(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=str(ROOT / "studies" / "strategy_window_comparison" / "data" / "canonical_trials.pkl"))
    parser.add_argument("--out", default=str(ROOT / "studies" / "strategy_window_comparison" / "results" / "submovements_real.csv"))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--backend", choices=("process", "thread"), default="process")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.trials, "rb") as handle:
        trials = pickle.load(handle)
    cfg = SubmovementConfig(restarts=args.restarts, max_nfev=args.max_nfev)
    completed = read_completed(out)
    done = set(completed.trial_id.astype(str)) if "trial_id" in completed else set()
    pending = [trial for trial in trials if trial["metadata"].get("trial_id", "") not in done]
    if args.limit:
        pending = pending[: args.limit]

    rows = completed.to_dict("records")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        executor = ProcessPoolExecutor if args.backend == "process" else ThreadPoolExecutor
        with executor(max_workers=args.jobs) as pool:
            fitted = list(pool.map(fit_one, batch, repeat(cfg)))
        rows.extend(fitted)
        frame = pd.DataFrame(rows).drop_duplicates("trial_id", keep="last").sort_values("trial_id")
        frame.to_csv(out, index=False)
        print(f"saved {len(frame)}/{len(trials)} trials to {out}", flush=True)

    protocol = {
        "source_repository": "https://github.com/JasonFriedman/submovements",
        "source_commit": "9c2f40ccc922d542242329c46cfd524c21188b4a",
        "dimensions": "x-y table plane",
        "filter_hz": cfg.cutoff_hz,
        "sample_hz": cfg.sample_hz,
        "max_components": cfg.max_components,
        "min_duration_s": cfg.min_duration_s,
        "min_onset_spacing_s": cfg.min_onset_spacing_s,
        "error_threshold": cfg.error_threshold,
        "fallback_error_threshold": cfg.fallback_error_threshold,
        "restarts": cfg.restarts,
        "max_nfev": cfg.max_nfev,
        "selection": "smallest k with error<=0.05; else smallest k with error<0.10; else minimum error",
    }
    out.with_suffix(".protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
