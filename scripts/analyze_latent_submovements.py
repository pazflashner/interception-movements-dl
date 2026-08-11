"""Associate CVAE coordinates with timing and minimum-jerk behavior."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.run_corrected_study import load_per_trial_checkpoint
from src.context_query import benjamini_hochberg, split_context_query
from src.evaluate import encode_trials
from src.features import compute_trial_features
from src.trajectory_view import project_trials_to_table_plane, select_trials_window


TARGETS = [
    "initiation_time_s",
    "movement_time_s",
    "recorded_success",
    "mj_n_components",
    "mj_first_duration_s",
    "mj_first_amplitude",
    "mj_secondary_amplitude_fraction",
    "mj_mean_overlap_pct",
]


def residualise(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values) & np.isfinite(design).all(axis=1)
    out = np.full(len(values), np.nan)
    if finite.sum() > design.shape[1] + 5:
        fit = LinearRegression().fit(design[finite], values[finite])
        out[finite] = values[finite] - fit.predict(design[finite])
    return out


def correlation_rows(frame: pd.DataFrame, z_columns: list[str], design: np.ndarray) -> list[dict]:
    rows = []
    residual_z = {z: residualise(frame[z].to_numpy(), design) for z in z_columns}
    for target in TARGETS:
        residual_target = residualise(frame[target].to_numpy(), design)
        for z in z_columns:
            finite = np.isfinite(residual_z[z]) & np.isfinite(residual_target)
            rho, p = spearmanr(residual_z[z][finite], residual_target[finite])
            rows.append({
                "level": "trial_within_subject_partial",
                "latent": z,
                "target": target,
                "rho": float(rho),
                "p_value": float(p),
                "n": int(finite.sum()),
            })
    return rows


def subject_rows(frame: pd.DataFrame, z_columns: list[str]) -> list[dict]:
    subjects = frame.subject.to_numpy()
    sp = frame.sp.to_numpy()
    side = frame.side.to_numpy()
    fingerprints, query = [], []
    for split in split_context_query(subjects, sp, side, seed=config.CONTEXT_QUERY_SEED):
        fp = {"subject": split.subject}
        fp.update(frame.iloc[split.context_indices][z_columns].mean().to_dict())
        fingerprints.append(fp)
        q = {"subject": split.subject}
        q.update(frame.iloc[split.query_indices][TARGETS].mean().to_dict())
        query.append(q)
    fp = pd.DataFrame(fingerprints).set_index("subject")
    q = pd.DataFrame(query).set_index("subject")
    rows = []
    for target in TARGETS:
        for z in z_columns:
            finite = np.isfinite(fp[z]) & np.isfinite(q[target])
            rho, p = spearmanr(fp.loc[finite, z], q.loc[finite, target])
            rows.append({
                "level": "subject_context_to_query",
                "latent": z,
                "target": target,
                "rho": float(rho),
                "p_value": float(p),
                "n": int(finite.sum()),
            })
    return rows


def add_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["fdr_reject_0_05"] = False
    for _, indices in frame.groupby(["window_mode", "latent_dim", "seed", "level"]).groups.items():
        idx = list(indices)
        frame.loc[idx, "fdr_reject_0_05"] = benjamini_hochberg(
            frame.loc[idx, "p_value"].to_numpy()
        )
    return frame


def plot_heatmap(frame: pd.DataFrame, path: Path, title: str) -> None:
    matrix = frame.pivot(index="target", columns="latent", values="rho").reindex(TARGETS)
    significant = frame.pivot(
        index="target", columns="latent", values="fdr_reject_0_05"
    ).reindex(TARGETS)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.65 * matrix.shape[1] + 3.5), 6.2))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(matrix.shape[0]), [name.replace("_", " ") for name in matrix.index])
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix.iloc[y, x]
            mark = "*" if bool(significant.iloc[y, x]) else ""
            ax.text(x, y, f"{value:+.2f}{mark}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Spearman rho")
    ax.set_xlabel("Model-specific latent coordinate")
    ax.set_ylabel("")
    fig.text(0.01, 0.01, "* Benjamini-Hochberg FDR < 0.05; association is not causation.", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, default=config.DATA_PROCESSED_DIR / "canonical_trials.pkl")
    parser.add_argument("--submovements", type=Path, default=config.RESULTS_DIR / "submovements_real.csv")
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "latent_associations")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with args.trials.open("rb") as handle:
        canonical = pickle.load(handle)
    sub = pd.read_csv(args.submovements)
    sub = sub[sub.mj_fit_success == True].copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []

    for window_mode in config.WINDOW_MODES:
        trials = project_trials_to_table_plane(select_trials_window(canonical, window_mode))
        real = pd.DataFrame([compute_trial_features(trial) for trial in trials])
        real["recorded_success"] = (real.successful == 1).astype(float)
        real["target_speed_screen_s"] = [
            trial["metadata"].get("target_speed_screen_s", np.nan) for trial in trials
        ]
        real = real.merge(sub[["trial_id", *[t for t in TARGETS if t.startswith("mj_")]]],
                          on="trial_id", how="inner", validate="one_to_one")
        trial_index = {trial["metadata"]["trial_id"]: i for i, trial in enumerate(trials)}
        kept_indices = np.array([trial_index[trial_id] for trial_id in real.trial_id], dtype=int)
        kept_trials = [trials[i] for i in kept_indices]

        design_frame = pd.get_dummies(
            real[["subject", "sp", "side"]].astype(str), drop_first=True, dtype=float
        )
        design_frame["target_speed_screen_s"] = real.target_speed_screen_s.to_numpy(float)
        design = design_frame.to_numpy(float)

        model_root = args.results / window_mode / "models"
        for run in sorted(model_root.glob("cvae_*_z*_seed*")):
            checkpoint = run / "checkpoint.pt"
            if not checkpoint.exists():
                continue
            model, norm = load_per_trial_checkpoint(checkpoint, device)
            mu, _, _, _ = encode_trials(model, kept_trials, norm, device)
            z_columns = [f"z{i + 1}" for i in range(model.latent_dim)]
            frame = real.copy()
            frame[z_columns] = mu
            seed = int(run.name.rsplit("seed", 1)[1])
            rows = correlation_rows(frame, z_columns, design) + subject_rows(frame, z_columns)
            for row in rows:
                row.update({
                    "window_mode": window_mode,
                    "latent_dim": model.latent_dim,
                    "seed": seed,
                    "run": run.name,
                })
            all_rows.extend(rows)

    associations = add_fdr(pd.DataFrame(all_rows))
    associations.to_csv(args.out / "latent_submovement_associations.csv", index=False)
    for (window_mode, latent_dim, seed, level), frame in associations.groupby(
        ["window_mode", "latent_dim", "seed", "level"]
    ):
        if seed != 42:
            continue
        plot_heatmap(
            frame,
            args.out / "heatmaps" / f"{window_mode}_z{latent_dim}_{level}.png",
            f"{window_mode.replace('_', ' ').title()} | n={latent_dim} | {level.replace('_', ' ')}",
        )
    print(associations.groupby(["window_mode", "latent_dim", "level"]).rho
          .apply(lambda s: float(np.mean(np.abs(s)))).to_string())


if __name__ == "__main__":
    main()
