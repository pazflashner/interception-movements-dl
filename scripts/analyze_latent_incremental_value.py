"""Does the latent code add predictive information beyond direct context measurement?

The delivered comparison races the CVAE latent probe against a direct-context
baseline and the probe loses on all 14 behavioural targets. A race does not test
whether the latent carries information the context average lacks: a weaker
predictor can still be complementary. This script runs the nested test instead.

Two nested models predict the same held-out query summary for one participant:

    A:  true ~ 1 + context_prediction
    B:  true ~ 1 + context_prediction + probe_prediction

B contains A, so B can only add. Every score is leave-one-participant-out, so a
gain must generalise across people rather than reflect the extra free parameter.
Participants are the unit of replication; the paired test uses the project's
frozen Wilcoxon settings and BH correction across the reported family.

This is the *forecast encompassing* form of the nested test. It asks whether the
latent-based prediction adds to the direct-context prediction. It does not refit
a regression on raw latent coordinates, which would need per-participant
fingerprints for all four folds; only seven test participants of fold 0 are
published, so that variant is provided separately and requires the full run
archive. Both answer the same question; this one is runnable from the delivered
snapshot and avoids fitting eight extra coordinates on seventeen participants.

A positive result licenses "the latent contributes information beyond direct
measurement". A null licenses "direct measurement is sufficient for these
targets". Neither licenses a claim about generation, which is a separate task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.statistical_tests import paired_wilcoxon

STUDY = ROOT / "studies" / "review_corrected_evaluation"
PROBES = STUDY / "results" / "behavioral_probes" / "out_of_fold_predictions.csv"
DIRECT = STUDY / "results" / "review_controls" / "direct_context_predictions.csv"
OUT = STUDY / "results" / "latent_incremental_value"


def load_pairs(model_family: str, fingerprint: str) -> pd.DataFrame:
    """Join seed-averaged probe predictions to direct-context predictions."""
    for path in (PROBES, DIRECT):
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path.relative_to(ROOT).as_posix()}; "
                "run scripts/prepare_review_workspace.py first"
            )
    probe = pd.read_csv(PROBES)
    probe = probe[(probe.model_family == model_family) & (probe.fingerprint == fingerprint)]
    if probe.empty:
        raise ValueError(f"no probe rows for family={model_family} fingerprint={fingerprint}")
    # Average the three training seeds per participant, matching how the
    # delivered participant-level tables aggregate optimisation noise.
    probe = (
        probe.groupby(["latent_dim", "outer_fold", "subject", "target"], as_index=False)
        .agg(probe_prediction=("predicted", "mean"),
             true_probe=("true", "mean"),
             n_seeds=("training_seed", "size"))
    )
    direct = pd.read_csv(DIRECT)
    joined = probe.merge(
        direct, on=["outer_fold", "subject", "target"], how="inner", validate="many_to_one"
    )
    # The two pipelines recomputed the query summary independently; if they
    # disagree the join is wrong and every downstream number is meaningless.
    np.testing.assert_allclose(joined.true_probe, joined.true, atol=1e-10)
    return joined.drop(columns=["true_probe"])


def _loso_predictions(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Leave-one-out OLS predictions for design ``x`` (no intercept column)."""
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    out = np.empty(n, dtype=float)
    for i in range(n):
        keep = np.arange(n) != i
        coef, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
        out[i] = design[i] @ coef
    return out


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def nested_test(pairs: pd.DataFrame, dims: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-participant-out A-vs-B comparison for every dimension/target."""
    rows, per_subject = [], []
    for dim in dims:
        block = pairs[pairs.latent_dim == dim]
        for target, group in block.groupby("target", sort=True):
            group = group.sort_values("subject").reset_index(drop=True)
            y = group.true.to_numpy(float)
            ctx = group.context_prediction.to_numpy(float)
            probe = group.probe_prediction.to_numpy(float)
            if len(group) < 8 or np.ptp(probe) == 0 or np.ptp(ctx) == 0:
                continue
            pred_a = _loso_predictions(y, ctx[:, None])
            pred_b = _loso_predictions(y, np.column_stack([ctx, probe]))
            err_a, err_b = np.abs(y - pred_a), np.abs(y - pred_b)
            statistic, p_value = paired_wilcoxon(err_a - err_b)
            # Sign convention: positive means adding the latent helped.
            rows.append({
                "latent_dim": dim, "target": target, "n_participants": len(group),
                "r2_context_only": _r2(y, pred_a),
                "r2_context_plus_latent": _r2(y, pred_b),
                "delta_r2": _r2(y, pred_b) - _r2(y, pred_a),
                "mae_context_only": float(err_a.mean()),
                "mae_context_plus_latent": float(err_b.mean()),
                "mae_reduction": float(err_a.mean() - err_b.mean()),
                "participants_improved": int((err_b < err_a).sum()),
                "residual_corr_with_probe": float(
                    np.corrcoef(y - pred_a, probe)[0, 1]
                ) if np.ptp(y - pred_a) > 0 else np.nan,
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p_uncorrected": float(p_value),
            })
            for i, subject in enumerate(group.subject):
                per_subject.append({
                    "latent_dim": dim, "target": target, "subject": subject,
                    "true": y[i], "pred_context_only": pred_a[i],
                    "pred_context_plus_latent": pred_b[i],
                    "abs_error_context_only": err_a[i],
                    "abs_error_context_plus_latent": err_b[i],
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no comparable target/dimension cells were produced")
    frame["wilcoxon_p_fdr_bh"] = false_discovery_control(
        frame.wilcoxon_p_uncorrected.to_numpy(float)
    )
    frame["significant_fdr_0_05"] = frame.wilcoxon_p_fdr_bh < 0.05
    return frame.sort_values(["latent_dim", "target"]).reset_index(drop=True), pd.DataFrame(per_subject)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", default="cvae")
    parser.add_argument("--fingerprint", default="mean", choices=["mean", "mean_plus_sd"])
    parser.add_argument("--dims", type=int, nargs="+", default=[3, 8])
    args = parser.parse_args()

    pairs = load_pairs(args.model_family, args.fingerprint)
    summary, per_subject = nested_test(pairs, tuple(args.dims))
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "nested_model_comparison.csv", index=False)
    per_subject.to_csv(OUT / "nested_participant_predictions.csv", index=False)
    protocol = {
        "question": "does the latent probe add predictive information beyond direct context?",
        "form": "forecast encompassing; nested leave-one-participant-out OLS",
        "model_a": "true ~ 1 + context_prediction",
        "model_b": "true ~ 1 + context_prediction + probe_prediction",
        "model_family": args.model_family,
        "fingerprint": args.fingerprint,
        "dims": list(args.dims),
        "seed_handling": "probe predictions averaged over training seeds per participant",
        "unit_of_replication": "participant",
        "test": "two-sided Pratt Wilcoxon on paired absolute errors, 12-decimal rounding",
        "multiplicity": "Benjamini-Hochberg across all reported dimension/target cells",
        "limits": [
            "encompassing form; does not refit on raw latent coordinates",
            "folds share training participants, so cells are not fully independent",
            "a null result bounds these 14 summaries only, not generation",
        ],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    show = summary[[
        "latent_dim", "target", "r2_context_only", "r2_context_plus_latent",
        "delta_r2", "mae_reduction", "participants_improved",
        "wilcoxon_p_fdr_bh", "significant_fdr_0_05",
    ]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    wins = int((summary.delta_r2 > 0).sum())
    print(f"\ncells where adding the latent improved out-of-sample R2: {wins}/{len(summary)}")
    print(f"cells significant after BH: {int(summary.significant_fdr_0_05.sum())}/{len(summary)}")
    print(f"written to {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
