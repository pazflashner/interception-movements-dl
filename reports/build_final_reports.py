"""Build the final scientific report and the matching student results guide.

Both documents consume only frozen confirmatory outputs.  Figure and table
identifiers are intentionally shared so the explanatory guide can be used as
an index into the advisor-facing report without changing the scientific copy.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from scripts.run_corrected_study import (
    context_query_for_trials,
    load_per_trial_checkpoint,
    training_latent_noise_covariance,
)
from src.confirmatory_protocol import make_participant_folds, partition_trials
from src.evaluate import encode_trials, reconstruct
from src.trajectory_view import project_trials_to_table_plane, select_trials_window
from src.vae_model import encode_trial_condition


TRAINING_STUDY = ROOT / "studies" / "final_strategy_evaluation"
STUDY = ROOT / "studies" / "review_corrected_evaluation"
RESULTS = STUDY / "results"
ANALYSIS = RESULTS / "analysis"
MIN_JERK = TRAINING_STUDY / "results" / "minimum_jerk"
OLD_RESULTS = ROOT / "studies" / "strategy_window_comparison" / "results"

NAVY = "#17324D"
BLUE = "#246A8D"
TEAL = "#2A9D8F"
ORANGE = "#D97706"
RED = "#B23A48"
PURPLE = "#6B5CA5"
GRAY = "#5D6872"
LIGHT = "#EDF2F5"
GRID = "#D5DEE5"
INK = "#18212A"

MODEL_LABELS = {
    "condition_ridge": "Condition Ridge",
    "spline_pca": "Spline+PCA",
    "conditional_ae": "Conditional AE",
    "unconditional_vae": "Unconditional VAE",
    "cvae": "CVAE",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#7C8994",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "oof": pd.read_csv(ANALYSIS / "oof_metrics_summary.csv"),
        "participant": pd.read_csv(ANALYSIS / "participant_metrics_seed_averaged.csv"),
        "paired": pd.read_csv(ANALYSIS / "paired_model_comparisons.csv"),
        "timing": pd.read_csv(ANALYSIS / "timing_outlier_audit.csv"),
        "condition": pd.read_csv(TRAINING_STUDY / "results" / "condition_effects" / "condition_trajectory_summary.csv"),
        "condition_features": pd.read_csv(
            TRAINING_STUDY / "results" / "condition_effects" / "cvae_vs_unconditional_paired.csv"
        ),
        "min_jerk": pd.read_csv(MIN_JERK / "analysis" / "oof_summary.csv"),
        "min_jerk_paired": pd.read_csv(MIN_JERK / "analysis" / "n3_vs_n8_paired.csv"),
        "sensitivity": pd.read_csv(MIN_JERK / "assumption_sensitivity_summary.csv"),
        "real_submovements": pd.read_csv(OLD_RESULTS / "submovements_real.csv"),
        "probes": pd.read_csv(RESULTS / "behavioral_probes" / "summary.csv"),
        "timing_fairness": pd.read_csv(RESULTS / "timing_fairness" / "summary.csv"),
        "timing_fairness_paired": pd.read_csv(RESULTS / "timing_fairness" / "paired_comparisons.csv"),
        "sampling": pd.read_csv(RESULTS / "sampling_reference" / "summary.csv"),
        "kmeans": pd.read_csv(
            OLD_RESULTS / "go_to_arrival" / "baselines" / "kmeans_selection_corrected.csv"
        ),
    }


def fingerprint_accuracy() -> pd.DataFrame:
    patterns = {
        "cvae": "fold*/cvae_z*_seed*/result.json",
        "conditional_ae": "fold*/conditional_ae_z*_seed*/result.json",
        "unconditional_vae": "fold*/unconditional_vae_z*_seed*/result.json",
        "spline_pca": "fold*/spline_pca_z*/result.json",
    }
    rows: list[dict] = []
    for family, pattern in patterns.items():
        for path in sorted((STUDY / "runs" / family).glob(pattern)):
            result = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model_family": family,
                    "latent_dim": int(result["latent_dim"]),
                    "outer_fold": int(result["outer_fold"]),
                    "training_seed": result.get("training_seed"),
                    "balanced_accuracy": float(result["fingerprint_balanced_accuracy"]),
                    "chance": float(result["fingerprint_chance"]),
                }
            )
    return pd.DataFrame(rows)


def model_row(oof: pd.DataFrame, family: str, latent_dim: int) -> pd.Series:
    row = oof[(oof.model_family == family) & (oof.latent_dim == latent_dim)]
    if len(row) != 1:
        raise ValueError(f"expected one result row for {family}, n={latent_dim}; found {len(row)}")
    return row.iloc[0]


def make_pipeline_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.15))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.15)
    ax.axis("off")
    boxes = [
        (0.15, "Target motion\nto recorded end", NAVY),
        (2.15, "10 Hz; x-y;\n100 phase points", BLUE),
        (4.15, "Model fit on\n17/4/7 subjects", TEAL),
        (6.15, "Context trials\nenroll fingerprint", ORANGE),
        (8.15, "Query distributions\nheld out", PURPLE),
    ]
    for index, (x, text, color) in enumerate(boxes):
        ax.add_patch(
            plt.Rectangle((x, 0.55), 1.65, 1.0, facecolor="white", edgecolor=GRAY, linewidth=.7)
        )
        ax.text(x + 0.825, 1.05, text, color=INK, ha="center", va="center")
        if index < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[index + 1][0] - 0.08, 1.05),
                xytext=(x + 1.72, 1.05),
                arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.4},
            )
    ax.text(
        5.25,
        0.15,
        "Four outer participant folds; each participant tested once; neural seeds 42/43/44",
        ha="center",
        va="center",
        color=INK,
        fontsize=9,
    )
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_compact_benchmark_figure(oof: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.8))
    x = np.arange(2)
    for ax, column, title in zip(axes,
        ["trajectory_mse_subject_balanced_mean", "mean_ks_mean"],
        ["Reconstruction MSE (tracker units squared)", "Generated distribution: mean KS"]):
        for offset, family, label, color in [(-.18, "spline_pca", "Spline+PCA", BLUE), (.18, "cvae", "CVAE", TEAL)]:
            values = [float(model_row(oof, family, n)[column]) for n in (3, 8)]
            ax.bar(x + offset, values, .36, color=color, label=label)
        ax.set_xticks(x, ["n=3", "n=8"])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Lower is better", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_benchmark_figure(oof: pd.DataFrame, path: Path) -> None:
    order = [
        ("condition_ridge", 0),
        ("spline_pca", 3),
        ("cvae", 3),
        ("unconditional_vae", 3),
        ("spline_pca", 8),
        ("conditional_ae", 8),
        ("cvae", 8),
        ("unconditional_vae", 8),
    ]
    labels = [f"{MODEL_LABELS[f]}\nn={n}" if n else MODEL_LABELS[f] for f, n in order]
    rows = [model_row(oof, family, n) for family, n in order]
    metrics = [
        ("trajectory_mse_subject_balanced_mean", "Trajectory MSE", None),
        ("initiation_time_mae_ms_subject_balanced_mean", "Initiation MAE (ms)", None),
        ("mean_ks_mean", "Mean feature KS", None),
        ("mean_mmd_rbf_mean", "Mean MMD squared", None),
    ]
    colors_for_models = [GRAY, BLUE, TEAL, PURPLE, BLUE, ORANGE, TEAL, PURPLE]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2))
    for ax, (column, title, _) in zip(axes.ravel(), metrics):
        values = [float(row[column]) for row in rows]
        ax.bar(np.arange(len(values)), values, color=colors_for_models, width=0.75)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("lower is better")
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Out-of-fold benchmark endpoints (28 held-out participants)",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_capacity_fingerprint_figure(
    oof: pd.DataFrame, fingerprints: pd.DataFrame, path: Path
) -> None:
    cvae = oof[oof.model_family == "cvae"].sort_values("latent_dim")
    acc = (
        fingerprints[fingerprints.model_family == "cvae"]
        .groupby("latent_dim", as_index=False)
        .balanced_accuracy.agg(["mean", "std"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.25))
    axes[0].plot(cvae.latent_dim, cvae.trajectory_mse_subject_balanced_mean, "o-", color=BLUE, lw=2)
    axes[0].set_title("A  Reconstruction", loc="left", weight="bold")
    axes[0].set_ylabel("trajectory MSE")
    axes[1].plot(cvae.latent_dim, cvae.mean_ks_mean, "o-", color=TEAL, lw=2)
    axes[1].set_title("B  Generated distributions", loc="left", weight="bold")
    axes[1].set_ylabel("mean KS")
    axes[2].errorbar(
        acc.latent_dim,
        acc["mean"],
        yerr=acc["std"],
        marker="o",
        color=PURPLE,
        capsize=3,
        lw=2,
    )
    axes[2].axhline(1 / 7, color=GRAY, ls="--", lw=1.2, label="7-way chance")
    axes[2].set_title("C  Closed-set enrollment", loc="left", weight="bold")
    axes[2].set_ylabel("balanced accuracy")
    axes[2].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.set_xlabel("latent dimension n")
        ax.set_xticks([2, 3, 4, 8])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "CVAE capacity and fingerprint enrollment",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_condition_figure(condition: pd.DataFrame, path: Path) -> None:
    condition = condition.sort_values("latent_dim")
    x = np.arange(len(condition))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.bar(x - width / 2, condition.cvae_median, width, label="CVAE", color=TEAL)
    ax.bar(
        x + width / 2,
        condition.unconditional_vae_median,
        width,
        label="Unconditional VAE",
        color=PURPLE,
    )
    for i, row in condition.reset_index(drop=True).iterrows():
        ax.text(
            i,
            max(row.cvae_median, row.unconditional_vae_median) + 0.006,
            f"Holm p={row.wilcoxon_p_holm:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 1.22 * max(condition.cvae_median.max(), condition.unconditional_vae_median.max()))
    ax.set_xticks(x, [f"n={int(n)}" for n in condition.latent_dim])
    ax.set_ylabel("condition-specific trajectory error")
    ax.set_title(
        "Condition-stratum trajectory comparison",
        loc="left",
        weight="bold",
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_submovement_figure(
    real: pd.DataFrame,
    min_jerk: pd.DataFrame,
    sensitivity: pd.DataFrame,
    path: Path,
) -> None:
    real = real[real.mj_fit_success == True].copy()
    support = np.arange(1, 5)
    threshold_dist = np.array([(real.mj_n_components == k).mean() for k in support])
    bic_dist = np.array([(real.mj_n_components_bic == k).mean() for k in support])
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.4))
    width = 0.34
    axes[0].bar(support - width / 2, threshold_dist, width, color=TEAL, label="threshold rule")
    axes[0].bar(support + width / 2, bic_dist, width, color=ORANGE, label="BIC")
    axes[0].set_xticks(support)
    axes[0].set_xlabel("selected components")
    axes[0].set_ylabel("recorded-trial proportion")
    axes[0].set_title("A  Recorded model order", loc="left", weight="bold")
    axes[0].legend(fontsize=7)

    axes[1].bar(
        [0, 1],
        min_jerk.sort_values("latent_dim").mean_count_total_variation_across_seeds,
        color=[TEAL, BLUE],
    )
    axes[1].set_xticks([0, 1], ["CVAE n=3", "CVAE n=8"])
    axes[1].set_ylabel("component-count total variation")
    axes[1].set_title("B  Generated count mismatch", loc="left", weight="bold")

    sens = sensitivity.set_index("sensitivity").loc[
        ["optimizer_4_restarts", "jason_167ms_constraints", "five_hz_filter"]
    ]
    axes[2].barh(
        ["4 restarts", "167 ms constraints", "5 Hz filter"],
        sens.same_threshold_count_rate,
        color=[TEAL, ORANGE, RED],
    )
    axes[2].set_xlim(0, 1.02)
    axes[2].set_xlabel("same selected count as primary")
    axes[2].set_title("C  Assumption sensitivity", loc="left", weight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Historical minimum-jerk component analysis",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_reconstruction_generation_figure(path: Path) -> dict[str, str]:
    with (config.DATA_PROCESSED_DIR / "canonical_trials.pkl").open("rb") as handle:
        canonical = pickle.load(handle)
    trials = project_trials_to_table_plane(
        select_trials_window(canonical, config.WINDOW_GO_TO_ARRIVAL)
    )
    folds = make_participant_folds([trial["metadata"]["subject"] for trial in trials])
    train, _, test = partition_trials(trials, folds[0])
    checkpoint = TRAINING_STUDY / "runs" / "cvae" / "fold0" / "cvae_z8_seed42" / "checkpoint.pt"
    model, norm = load_per_trial_checkpoint(checkpoint, "cpu")
    reconstructed, recorded, _, _ = reconstruct(model, test, norm, "cpu")
    reconstructed = reconstructed.reshape(len(test), config.NORMALISED_LENGTH, 2)
    recorded = recorded.reshape(len(test), config.NORMALISED_LENGTH, 2)
    errors = np.mean((reconstructed - recorded) ** 2, axis=(1, 2))
    typical_index = int(np.argsort(errors)[len(errors) // 2])
    difficult_index = int(np.argsort(errors)[int(0.9 * (len(errors) - 1))])

    mu, _, _, _ = encode_trials(model, test, norm, "cpu")
    split = context_query_for_trials(test, config.CONTEXT_QUERY_SEED)[0]
    subject = split.subject
    context_mean = mu[split.context_indices].mean(axis=0)
    covariance = training_latent_noise_covariance(model, train, norm, "cpu")
    rng = np.random.default_rng(20260824)
    n_samples = 24
    z = rng.multivariate_normal(context_mean, covariance, size=n_samples).astype(np.float32)
    query = [test[i] for i in split.query_indices]
    chosen = rng.integers(0, len(query), size=n_samples)
    cond = np.stack(
        [encode_trial_condition(query[i]["metadata"], model.condition_dim) for i in chosen]
    ).astype(np.float32)
    tm, ts, _, _ = norm.torch("cpu")
    with torch.no_grad():
        decoded, _ = model.decode(torch.from_numpy(z), torch.from_numpy(cond))
        generated = ((decoded * ts + tm).numpy()).reshape(n_samples, config.NORMALISED_LENGTH, 2)

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.35))
    examples = [(typical_index, "A  Median-error reconstruction"), (difficult_index, "B  90th-percentile error")]
    for ax, (index, title) in zip(axes[:2], examples):
        ax.plot(recorded[index, :, 0], recorded[index, :, 1], color=NAVY, lw=2, label="recorded")
        ax.plot(
            reconstructed[index, :, 0],
            reconstructed[index, :, 1],
            color=TEAL,
            lw=2,
            ls="--",
            label="posterior-mean reconstruction",
        )
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("lateral x")
        ax.set_ylabel("forward y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7)
    query_subject = [trial for trial in query if trial["metadata"]["subject"] == subject]
    for trial in query_subject[:18]:
        arr = trial["pos_norm"]
        axes[2].plot(arr[:, 0], arr[:, 1], color="#B7C2CA", lw=0.7, alpha=0.55)
    for arr in generated:
        axes[2].plot(arr[:, 0], arr[:, 1], color=PURPLE, lw=0.8, alpha=0.45)
    axes[2].set_title(f"C  Fingerprint-conditioned samples ({subject})", loc="left", weight="bold")
    axes[2].set_xlabel("lateral x")
    axes[2].set_ylabel("forward y")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].legend(
        handles=[
            Line2D([0], [0], color="#B7C2CA", lw=2, label="recorded query"),
            Line2D([0], [0], color=PURPLE, lw=2, label="generated"),
        ],
        fontsize=7,
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Held-out reconstruction and context generation (fold 0, seed 42)",
        x=0.02,
        ha="left",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "typical_trial": test[typical_index]["metadata"]["trial_id"],
        "difficult_trial": test[difficult_index]["metadata"]["trial_id"],
        "generation_subject": subject,
    }


def make_timing_outlier_figure(timing: pd.DataFrame, path: Path) -> None:
    frame = timing[
        (timing.model_family == "cvae") & (timing.latent_dim.isin([3, 8]))
    ].copy()
    frame["label"] = [f"n={int(n)}, seed {int(s)}" for n, s in zip(frame.latent_dim, frame.training_seed)]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    x = np.arange(len(frame))
    ax.semilogy(x, frame.initiation_time_median_abs_error_ms, "o", color=TEAL, label="median")
    ax.semilogy(x, frame.initiation_time_p95_abs_error_ms, "s", color=ORANGE, label="95th percentile")
    ax.semilogy(x, frame.initiation_time_max_abs_error_ms, "^", color=RED, label="maximum")
    ax.set_xticks(x, frame.label, rotation=25, ha="right")
    ax.set_ylabel("absolute initiation-time error (ms, log scale)")
    ax.set_title("CVAE initiation-time absolute-error quantiles", loc="left")
    ax.legend(ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)




def make_association_figure(path: Path) -> None:
    from scipy.stats import spearmanr
    from src.features import features_from_generated_window
    run = "cvae_z3_seed42"
    model, norm = load_per_trial_checkpoint(TRAINING_STUDY / "runs/cvae/fold0" / run / "checkpoint.pt", "cpu")
    stats = json.loads((TRAINING_STUDY / "results/dashboard/latent_stats.json").read_text())[run]
    z = np.random.default_rng(1147).multivariate_normal(stats["training_center"], stats["shared_covariance"], 500).astype(np.float32)
    c = encode_trial_condition({"sp": 2, "side": 1, "target_speed_screen_s": .635}, model.condition_dim)
    mean, scale, _, _ = norm.torch("cpu")
    with torch.no_grad():
        positions, times = model.decode(torch.from_numpy(z), torch.from_numpy(np.repeat(c[None,:],len(z),axis=0)))
        positions = ((positions*scale+mean).numpy()).reshape(-1,100,2)
        times = norm.denormalise_timing(times.numpy())
    f = pd.DataFrame([features_from_generated_window(x,max(float(t[0]),.001),max(float(t[1]),0),config.WINDOW_GO_TO_ARRIVAL) for x,t in zip(positions,times)])
    columns = ["initiation_time_s","movement_time_s","peak_speed_tracker_units_s","path_length","curvature_index","max_lateral_deviation"]
    rho = np.array([[spearmanr(z[:,i],f[col]).statistic for col in columns] for i in range(3)])
    pd.DataFrame(rho,index=["z1","z2","z3"],columns=columns).to_csv(path.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(9.5,3.3))
    im=ax.imshow(rho,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(6),["Initiation","Movement","Peak speed","Path length","Curvature","Lateral dev."],rotation=20,ha="right")
    ax.set_yticks(range(3),["z1","z2","z3"])
    for i in range(3):
        for j in range(6):
            ax.text(j,i,f"{rho[i,j]:.2f}",ha="center",va="center",color="white" if abs(rho[i,j])>.55 else "black")
    fig.colorbar(im,ax=ax,label="Spearman correlation")
    fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)


from reports.article_layout import guide_report
from reports.concise_article import scientific_report, appendix_report


def make_review_control_figures(figures):
    controls = RESULTS / "review_controls"
    frame = pd.read_csv(controls / "fingerprint_participant.csv")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    rng = np.random.default_rng(512)
    labels = []
    for index, (dim, arm) in enumerate([(3,"population"),(3,"wrong"),(8,"population"),(8,"wrong")]):
        own = frame[(frame.latent_dim==dim)&(frame.arm=="own")].set_index("subject")
        other = frame[(frame.latent_dim==dim)&(frame.arm==arm)].set_index("subject").loc[own.index]
        delta = other.mean_ks.to_numpy()-own.mean_ks.to_numpy()
        color = BLUE if arm=="population" else TEAL
        ax.scatter(index+rng.uniform(-.13,.13,len(delta)),delta,s=23,color=color,alpha=.8,zorder=3)
        ax.plot([index-.2,index+.2],[delta.mean()]*2,color=INK,lw=2.2,zorder=4)
        labels.append(f"n={dim}\n"+("Training average" if arm=="population" else "Other participants"))
    ax.axhline(0,color=GRAY,lw=1,ls="--")
    ax.set_xticks(range(4),labels)
    ax.set_ylabel("Control mean KS minus own-fingerprint KS")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(figures["fingerprint_control"],dpi=220);plt.close(fig)

    matched = pd.read_csv(controls / "matched_component_summary.csv").set_index("latent_dim")
    legacy = pd.read_csv(MIN_JERK / "analysis/oof_summary.csv").set_index("latent_dim")
    fits = pd.read_csv(controls / "matched_components.csv")
    fig, axes = plt.subplots(1,2,figsize=(9.5,3.4))
    x=np.arange(2);width=.34
    axes[0].bar(x-width/2,[legacy.loc[d,"mean_count_total_variation_across_seeds"] for d in [3,8]],width,color=GRAY,label="Historical procedure")
    axes[0].bar(x+width/2,[matched.loc[d,"count_total_variation"] for d in [3,8]],width,color=TEAL,label="Matched procedure")
    axes[0].set_xticks(x,["n=3","n=8"]);axes[0].set_ylabel("Count total variation");axes[0].legend(fontsize=8)
    real=fits[fits.kind=="recorded"]
    probs=[]
    for subject,e in real.groupby("subject"):
        probs.append([(e.mj_n_components==k).mean() for k in range(1,5)])
    x=np.arange(4)
    axes[1].bar(x-.25,np.mean(probs,axis=0),.25,label="Recorded query",color=GRAY)
    for offset,dim,color in [(0,3,BLUE),(.25,8,TEAL)]:
        g=fits[(fits.kind=="generated")&(fits.latent_dim==dim)]
        probs=[[(e.mj_n_components==k).mean() for k in range(1,5)] for _,e in g.groupby(["subject","seed"])]
        axes[1].bar(x+offset,np.mean(probs,axis=0),.25,label=f"Generated n={dim}",color=color)
    axes[1].set_xticks(x,["1","2","3","4"]);axes[1].set_xlabel("Selected component count");axes[1].set_ylabel("Mean participant proportion");axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(figures["matched_components"],dpi=220);plt.close(fig)

    with (ROOT / "studies/strategy_window_comparison/data/canonical_trials.pkl").open("rb") as f:
        trials=pickle.load(f)
    trial=next(t for t in trials if t["metadata"]["trial_id"]=="subject01_li_2_2_1_20")
    times=(np.arange(len(trial["pos_raw"]))-trial["go_signal_idx"])/240
    mask=(times>=-.2)&(times<=.25)
    fig,ax=plt.subplots(figsize=(9,2.4))
    for key,label,color in [("pos_raw","Raw planar speed",GRAY),("pos_filtered","Filtered planar speed",INK)]:
        speed=np.linalg.norm(np.gradient(trial[key][:,:2],1/240,axis=0),axis=1)
        ax.plot(times[mask],speed[mask],label=label,color=color,lw=1.3)
    ax.axvline(0,color=GRAY,ls="--",lw=1);ax.axhline(5,color=GRAY,ls=":",lw=1)
    ax.set_xlabel("Time relative to target motion (s)");ax.set_ylabel("Speed (tracker units/s)")
    ax.set_title("subject01_li_2_2_1_20",loc="left",fontsize=9);ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(figures["event_excerpt"],dpi=220);plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "pdf")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    tables = load_tables()
    fingerprints = fingerprint_accuracy()

    figures = {
        "pipeline": figure_dir / "figure_1_pipeline.png",
        "benchmark": figure_dir / "figure_2_benchmarks.png",
        "capacity": figure_dir / "figure_3_capacity.png",
        "condition": figure_dir / "figure_4_conditioning.png",
        "submovement": figure_dir / "figure_5_submovements.png",
        "examples": figure_dir / "figure_a1_examples.png",
        "timing": figure_dir / "figure_a2_timing_outliers.png",
        "associations": figure_dir / "figure_a4_associations.png",
        "fingerprint_control": figure_dir / "fingerprint_controls.png",
        "matched_components": figure_dir / "matched_components.png",
        "event_excerpt": figure_dir / "event_excerpt.png",
    }
    make_pipeline_figure(figures["pipeline"])
    make_compact_benchmark_figure(tables["oof"], figures["benchmark"])
    make_capacity_fingerprint_figure(tables["oof"], fingerprints, figures["capacity"])
    make_condition_figure(tables["condition"], figures["condition"])
    make_submovement_figure(
        tables["real_submovements"], tables["min_jerk"], tables["sensitivity"], figures["submovement"]
    )
    example_meta = make_reconstruction_generation_figure(figures["examples"])
    make_timing_outlier_figure(tables["timing"], figures["timing"])
    make_association_figure(figures["associations"])
    make_review_control_figures(figures)

    scientific = args.out / "Interception_Movements_Final_Scientific_Report.pdf"
    guide = args.out / "Interception_Movements_Results_Guide.pdf"
    appendix = args.out / "Interception_Movements_Supplementary_Appendix.pdf"
    scientific_report(scientific, tables, fingerprints, figures, example_meta)
    appendix_report(appendix, tables, fingerprints, figures, example_meta)
    guide_report(guide, tables, fingerprints, figures)
    print(scientific)
    print(guide)
    print(appendix)


if __name__ == "__main__":
    main()
