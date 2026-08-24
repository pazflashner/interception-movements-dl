"""Build the final scientific report and the matching student results guide.

Both documents consume only frozen confirmatory outputs.  Figure and table
identifiers are intentionally shared so the explanatory guide can be used as
an index into the advisor-facing report without changing the scientific copy.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

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


STUDY = ROOT / "studies" / "final_strategy_evaluation"
RESULTS = STUDY / "results"
ANALYSIS = RESULTS / "analysis"
MIN_JERK = RESULTS / "minimum_jerk"
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
            "font.family": "DejaVu Sans",
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
        "condition": pd.read_csv(RESULTS / "condition_effects" / "condition_trajectory_summary.csv"),
        "condition_features": pd.read_csv(
            RESULTS / "condition_effects" / "cvae_vs_unconditional_paired.csv"
        ),
        "min_jerk": pd.read_csv(MIN_JERK / "analysis" / "oof_summary.csv"),
        "min_jerk_paired": pd.read_csv(MIN_JERK / "analysis" / "n3_vs_n8_paired.csv"),
        "sensitivity": pd.read_csv(MIN_JERK / "assumption_sensitivity_summary.csv"),
        "real_submovements": pd.read_csv(OLD_RESULTS / "submovements_real.csv"),
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
        (0.15, "Target onset\nto arrival", NAVY),
        (2.15, "10 Hz; x-y;\n100 phase points", BLUE),
        (4.15, "Model fit on\n17/4/7 subjects", TEAL),
        (6.15, "Context trials\nenroll fingerprint", ORANGE),
        (8.15, "Query distributions\nheld out", PURPLE),
    ]
    for index, (x, text, color) in enumerate(boxes):
        ax.add_patch(
            plt.Rectangle((x, 0.55), 1.65, 1.0, facecolor=color, edgecolor="none", alpha=0.96)
        )
        ax.text(x + 0.825, 1.05, text, color="white", ha="center", va="center", weight="bold")
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
        ("mean_mmd_rbf_mean", "Mean MMD", None),
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
        "Figure 2. Out-of-fold benchmark endpoints (28 held-out participants)",
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
        "Figure 3. CVAE capacity and fingerprint enrollment",
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
        "Figure 4. Explicit task conditioning did not improve the matched diagnostic",
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
        "Figure 5. Minimum-jerk component analysis is model-order sensitive",
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
    checkpoint = STUDY / "runs" / "cvae" / "fold0" / "cvae_z8_seed42" / "checkpoint.pt"
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
        "Figure A1. Held-out reconstruction and context-fingerprint generation (fold 0, seed 42)",
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
    ax.set_title("Figure A2. Timing heads are accurate typically but have rare extrapolation failures", loc="left", weight="bold")
    ax.legend(ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


class ManualPDF:
    def __init__(self, path: Path, title: str, authors: str, compact: bool = False):
        self.path = path
        self.canvas = canvas.Canvas(str(path), pagesize=A4)
        self.canvas.setTitle(title)
        self.canvas.setAuthor(authors)
        self.width, self.height = A4
        self.page = 0
        self.title = title
        self.compact = compact
        base = 8.7 if compact else 10.2
        self.styles = {
            "body": ParagraphStyle(
                "body",
                fontName="Helvetica",
                fontSize=base,
                leading=base * 1.28,
                textColor=colors.HexColor(INK),
                alignment=TA_JUSTIFY,
                spaceAfter=2.5 * mm,
            ),
            "small": ParagraphStyle(
                "small",
                fontName="Helvetica",
                fontSize=7.2 if compact else 8.4,
                leading=9.0 if compact else 10.5,
                textColor=colors.HexColor(INK),
                alignment=TA_LEFT,
            ),
            "caption": ParagraphStyle(
                "caption",
                fontName="Helvetica-Oblique",
                fontSize=7.4 if compact else 8.4,
                leading=9.2 if compact else 10.5,
                textColor=colors.HexColor(GRAY),
                alignment=TA_LEFT,
            ),
            "h1": ParagraphStyle(
                "h1",
                fontName="Helvetica-Bold",
                fontSize=15.2 if compact else 17.0,
                leading=18.2 if compact else 20.5,
                textColor=colors.HexColor(NAVY),
                alignment=TA_LEFT,
            ),
            "h2": ParagraphStyle(
                "h2",
                fontName="Helvetica-Bold",
                fontSize=10.6 if compact else 12.3,
                leading=13.0 if compact else 15.0,
                textColor=colors.HexColor(BLUE),
                alignment=TA_LEFT,
            ),
            "callout": ParagraphStyle(
                "callout",
                fontName="Helvetica-Bold",
                fontSize=8.8 if compact else 10.0,
                leading=11.3 if compact else 13.0,
                textColor=colors.HexColor(NAVY),
                alignment=TA_LEFT,
            ),
        }

    def new_page(self, section: str = "") -> float:
        if self.page:
            self._footer()
            self.canvas.showPage()
        self.page += 1
        self.canvas.setFillColor(colors.HexColor(NAVY))
        self.canvas.rect(0, self.height - 10 * mm, self.width, 10 * mm, stroke=0, fill=1)
        self.canvas.setFillColor(colors.white)
        self.canvas.setFont("Helvetica-Bold", 7.5)
        self.canvas.drawString(15 * mm, self.height - 6.4 * mm, "WORKSHOP ON DEEP LEARNING | INTERCEPTION MOVEMENTS")
        if section:
            self.canvas.setFont("Helvetica", 7.5)
            self.canvas.drawRightString(self.width - 15 * mm, self.height - 6.4 * mm, section.upper())
        return self.height - 18 * mm

    def _footer(self) -> None:
        self.canvas.setStrokeColor(colors.HexColor(GRID))
        self.canvas.line(15 * mm, 12 * mm, self.width - 15 * mm, 12 * mm)
        self.canvas.setFillColor(colors.HexColor(GRAY))
        self.canvas.setFont("Helvetica", 7.3)
        self.canvas.drawString(15 * mm, 7.7 * mm, "Simaan Libbiss and Paz Flashner | Prof. Jason Friedman | Advisor: Moni Shahar")
        self.canvas.drawRightString(self.width - 15 * mm, 7.7 * mm, f"Page {self.page}")

    def heading(self, text: str, y: float, level: int = 1, x: float = 15 * mm, width: float | None = None) -> float:
        width = width or self.width - 30 * mm
        style = self.styles["h1" if level == 1 else "h2"]
        p = Paragraph(text, style)
        _, h = p.wrap(width, y - 15 * mm)
        p.drawOn(self.canvas, x, y - h)
        return y - h - (2.2 * mm if level == 1 else 1.2 * mm)

    def paragraph(
        self,
        text: str,
        y: float,
        style: str = "body",
        x: float = 15 * mm,
        width: float | None = None,
        gap: float | None = None,
    ) -> float:
        width = width or self.width - 30 * mm
        p = Paragraph(text, self.styles[style])
        _, h = p.wrap(width, y - 15 * mm)
        p.drawOn(self.canvas, x, y - h)
        return y - h - (gap if gap is not None else 1.5 * mm)

    def callout(self, text: str, y: float, height: float | None = None) -> float:
        width = self.width - 30 * mm
        p = Paragraph(text, self.styles["callout"])
        _, h = p.wrap(width - 8 * mm, 40 * mm)
        box_h = height or h + 7 * mm
        self.canvas.setFillColor(colors.HexColor(LIGHT))
        self.canvas.setStrokeColor(colors.HexColor(BLUE))
        self.canvas.roundRect(15 * mm, y - box_h, width, box_h, 2 * mm, stroke=1, fill=1)
        p.drawOn(self.canvas, 19 * mm, y - box_h + (box_h - h) / 2)
        return y - box_h - 3 * mm

    def image(self, path: Path, y: float, width: float, height: float, x: float = 15 * mm) -> float:
        self.canvas.drawImage(str(path), x, y - height, width=width, height=height, preserveAspectRatio=True, anchor="c")
        return y - height - 1.5 * mm

    def table(
        self,
        data: list[list],
        y: float,
        col_widths: list[float],
        x: float = 15 * mm,
        font_size: float | None = None,
        repeat_rows: int = 1,
    ) -> float:
        fs = font_size or (7.0 if self.compact else 8.1)
        header_style = ParagraphStyle(
            f"table_header_{self.page}_{fs}",
            fontName="Helvetica-Bold",
            fontSize=fs,
            leading=fs * 1.22,
            textColor=colors.white,
            alignment=TA_LEFT,
        )
        body_left = ParagraphStyle(
            f"table_body_left_{self.page}_{fs}",
            fontName="Helvetica",
            fontSize=fs,
            leading=fs * 1.22,
            textColor=colors.HexColor(INK),
            alignment=TA_LEFT,
        )
        body_center = ParagraphStyle(
            f"table_body_center_{self.page}_{fs}",
            parent=body_left,
            alignment=TA_CENTER,
        )
        wrapped = []
        for row_index, row in enumerate(data):
            wrapped.append(
                [
                    cell
                    if isinstance(cell, Paragraph)
                    else Paragraph(
                        str(cell),
                        header_style if row_index == 0 else (body_left if col_index == 0 else body_center),
                    )
                    for col_index, cell in enumerate(row)
                ]
            )
        table = Table(wrapped, colWidths=col_widths, repeatRows=repeat_rows)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), fs),
                    ("LEADING", (0, 0), (-1, -1), fs * 1.25),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F9")]),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(GRID)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ]
            )
        )
        _, h = table.wrap(sum(col_widths), y - 15 * mm)
        table.drawOn(self.canvas, x, y - h)
        return y - h - 2 * mm

    def assert_space(self, y: float, label: str) -> None:
        if y < 17 * mm:
            raise RuntimeError(f"page {self.page} overflow after {label}: y={y / mm:.1f} mm")

    def finish(self) -> None:
        self._footer()
        self.canvas.save()


def fmt(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "NA"
    if value != 0 and abs(value) < 0.001:
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def benchmark_table(oof: pd.DataFrame, fingerprints: pd.DataFrame) -> list[list[str]]:
    acc_summary = (
        fingerprints.groupby(["model_family", "latent_dim"]).balanced_accuracy.mean().to_dict()
    )
    selected = [
        ("condition_ridge", 0),
        ("spline_pca", 3),
        ("cvae", 3),
        ("unconditional_vae", 3),
        ("spline_pca", 8),
        ("conditional_ae", 8),
        ("cvae", 8),
        ("unconditional_vae", 8),
    ]
    data = [["Model", "n", "Traj. MSE", "Init. MAE ms", "Move MAE ms", "Mean KS", "MMD", "Enroll. acc."]]
    for family, n in selected:
        row = model_row(oof, family, n)
        acc = acc_summary.get((family, n), np.nan)
        data.append(
            [
                MODEL_LABELS[family],
                "-" if n == 0 else str(n),
                fmt(row.trajectory_mse_subject_balanced_mean, 3),
                fmt(row.initiation_time_mae_ms_subject_balanced_mean, 1),
                fmt(row.movement_time_mae_ms_subject_balanced_mean, 1),
                fmt(row.mean_ks_mean, 3),
                fmt(row.mean_mmd_rbf_mean, 3),
                "-" if not np.isfinite(acc) else f"{100 * acc:.1f}%",
            ]
        )
    return data


def paired_table(paired: pd.DataFrame, latent_dim: int, comparator: str) -> list[list[str]]:
    frame = paired[(paired.latent_dim == latent_dim) & (paired.comparator == comparator)].set_index("metric")
    metrics = [
        ("trajectory_mse", "Trajectory MSE"),
        ("initiation_time_mae_ms", "Initiation MAE"),
        ("movement_time_mae_ms", "Movement MAE"),
        ("mean_ks", "Mean KS"),
        ("mmd_rbf", "MMD"),
    ]
    data = [["Endpoint", "CVAE", MODEL_LABELS[comparator], "CVAE better", "BH-FDR p"]]
    for metric, label in metrics:
        row = frame.loc[metric]
        data.append(
            [
                label,
                fmt(row.cvae_mean, 3 if "mae" not in metric else 1),
                fmt(row.comparator_mean, 3 if "mae" not in metric else 1),
                f"{int(row.cvae_better_participants)}/28",
                fmt(row.wilcoxon_p_fdr_bh, 4),
            ]
        )
    return data


def scientific_report(
    output: Path,
    tables: dict[str, pd.DataFrame],
    fingerprints: pd.DataFrame,
    figures: dict[str, Path],
    example_meta: dict[str, str],
) -> None:
    pdf = ManualPDF(
        output,
        "Conditional Generative Modeling of Interception-Movement Fingerprints",
        "Simaan Libbiss and Paz Flashner",
        compact=True,
    )
    usable = pdf.width - 30 * mm

    # Page 1
    y = pdf.new_page("Abstract and motivation")
    y = pdf.heading("Conditional Generative Modeling of Interception-Movement Fingerprints", y)
    y = pdf.paragraph(
        "Simaan Libbiss and Paz Flashner | 0368-3538 Workshop on Deep Learning<br/>"
        "Research supervision: Prof. Jason Friedman, Dept. Physical Therapy and Sagol School of Neuroscience | Course advisor: Moni Shahar",
        y,
        style="small",
        gap=4 * mm,
    )
    y = pdf.heading("Abstract", y, level=2)
    y = pdf.paragraph(
        "We study whether fast planar interception movements admit compact participant-specific latent representations that can generate held-out distributions of trajectory shape and timing. The confirmatory analysis retains the interval from target-motion onset to finger arrival, filters the table-plane trajectory at 10 Hz, and resamples position to 100 phase points while predicting physical initiation and movement time separately. A conditional variational autoencoder (CVAE; n=2,3,4,8) is evaluated against spline+PCA, a deterministic conditional autoencoder, an unconditional VAE, and a task-condition-only stochastic Ridge baseline. Four participant folds provide out-of-fold predictions for all 28 participants; three neural seeds assess optimization variability. Spline+PCA is strongest for deterministic reconstruction, whereas CVAE n=3 and n=8 improve timing and generated-distribution fidelity. The n=8 model is the stronger generator; n=3 remains the compact interpretation target. Unconditional VAE results are statistically comparable, so participant-specific effects of explicit task conditioning are not established. Minimum-jerk component fidelity is modest and sensitive to model-order assumptions.",
        y,
    )
    y = pdf.callout(
        "Primary conclusion: the evidence supports a capacity-interpretability tradeoff, not a claim that the CVAE is uniformly more accurate than classical or neural comparators.",
        y,
    )
    y = pdf.heading("1. Motivation and research question", y, level=2)
    y = pdf.paragraph(
        "Interception combines feedforward prediction with online correction [1]. Individual motor signatures have been demonstrated in other movement paradigms [2], but an interception model must additionally reproduce stochastic within-person variation. We therefore ask: <b>can a low-dimensional generative fingerprint, enrolled from a subset of a previously unseen participant's trials, reproduce the distribution of that participant's remaining movements?</b> This formulation separates participant enrollment from zero-shot prediction and treats full movement distributions, rather than single trajectories, as the target.",
        y,
    )
    y = pdf.heading("Contributions", y, level=2)
    y = pdf.paragraph(
        "(i) leakage-resistant participant rotation and disjoint context/query evaluation; (ii) matched deterministic, unconditional, condition-only, and spline baselines; (iii) explicit separation of phase-normalized shape from physical timing; and (iv) a secondary minimum-jerk decomposition derived from Prof. Friedman's public implementation.",
        y,
    )
    pdf.assert_space(y, "page 1")

    # Page 2
    y = pdf.new_page("Data and protocol")
    y = pdf.heading("2. Dataset and preprocessing", y)
    y = pdf.paragraph(
        "The source dataset contains 240 Hz electromagnetic finger-position recordings. We use 4,732 retained free-eye-movement trials (condition 2) from 28 participants. Filename metadata provides starting-position/speed category, stimulus side, and repetition; MAT metadata supplies outcome labels and exact target speed. The unreliable receive-time column is ignored and timing is derived from the frame counter. The primary analysis uses lateral x and forward y because the task is performed on a tabletop; out-of-plane z is excluded.",
        y,
    )
    y = pdf.paragraph(
        "Each trajectory begins at target-motion onset (marker 5) and ends at recorded finger arrival. This preserves the pre-movement waiting interval. Position is low-pass filtered with a fourth-order 10 Hz Butterworth filter, translated to the trial origin, and resampled to 100 phase points. Resampling removes physical duration; therefore initiation time and movement time are withheld from the encoder and predicted by separate decoder heads. Trials missing required events or not reaching the interception site are excluded by the canonical audit. The late-trial rule and the condition-2 fixation message remain task-specific assumptions for Prof. Friedman's review.",
        y,
    )
    y = pdf.image(figures["pipeline"], y, usable, 39 * mm)
    y = pdf.paragraph(
        "<b>Figure 1.</b> Frozen strategy-window protocol. Outer folds contain 17 train, 4 validation, and 7 test participants; every participant is tested exactly once. Within each test participant, context trials estimate the fingerprint and disjoint query trials define empirical target distributions.",
        y,
        style="caption",
    )
    y = pdf.heading("2.1 Inputs and targets", y, level=2)
    data = [
        ["Element", "Representation", "Role"],
        ["Trajectory", "100 x 2 phase-normalized x-y positions", "encoder input and decoder output"],
        ["Task condition", "sp category, side, exact executed target speed", "CVAE/CAE condition vector"],
        ["Physical timing", "initiation time; movement time", "decoder targets only"],
        ["Fingerprint", "mean context posterior coordinate", "participant enrollment for generation"],
    ]
    y = pdf.table(data, y, [34 * mm, 72 * mm, 70 * mm])
    y = pdf.heading("2.2 Feasibility check", y, level=2)
    kmeans = tables["kmeans"].set_index("representation")
    y = pdf.paragraph(
        f"A pre-specified 28-cluster K-means probe produced low absolute agreement (trajectory ARI={kmeans.loc['trajectory','ari']:.3f}; feature ARI={kmeans.loc['kinematic_features','ari']:.3f}) but exceeded its permutation null in both representations (p={kmeans.loc['trajectory','permutation_p']:.4f}). This is evidence of weak non-random participant structure, not evidence of 28 clean clusters.",
        y,
    )
    pdf.assert_space(y, "page 2")

    # Page 3
    y = pdf.new_page("Models and evaluation")
    y = pdf.heading("3. Models and training", y)
    y = pdf.paragraph(
        "The CVAE follows the stochastic latent-variable formulation of Kingma and Welling [3] and the conditional structured-output extension of Sohn et al. [4]. The encoder maps trajectory and task condition to diagonal-Gaussian posterior parameters; the decoder reconstructs trajectory and log-transformed timing. Training minimizes trajectory reconstruction loss, weighted timing loss, and KL divergence to the standard-normal prior. Participant-balanced sampling prevents high-trial-count participants from dominating batches. Early stopping is selected on validation participants only.",
        y,
    )
    model_data = [
        ["Comparator", "Question isolated"],
        ["Spline+PCA (n=2,3,4,8)", "classical low-rank shape reconstruction and stochastic coefficient generation"],
        ["Conditional AE (n=3,8)", "effect of KL regularization/stochastic bottleneck"],
        ["Unconditional VAE (n=3,8)", "incremental value of explicit task conditioning"],
        ["Condition-only Ridge", "performance available from task metadata without a participant fingerprint"],
    ]
    y = pdf.table(model_data, y, [55 * mm, 121 * mm])
    y = pdf.heading("4. Evaluation", y)
    y = pdf.paragraph(
        "Observed-trial reconstruction uses posterior means. Participant-balanced trajectory MSE and timing MAE summarize out-of-fold prediction. Closed-set enrollment assigns query trials to the nearest context fingerprint among the seven participants in that fold; nominal chance is 1/7. Generative fidelity compares context-conditioned samples with query trials using per-feature two-sample KS statistics across 11 active 2-D kinematic features, RBF-MMD, and energy distance. Lower values indicate closer empirical/generated distributions. Model contrasts use two-sided paired Wilcoxon tests on 28 seed-averaged held-out participants with Benjamini-Hochberg correction. Fold effects and optimization seeds are retained separately.",
        y,
    )
    y = pdf.heading("4.1 Minimum-jerk secondary analysis", y, level=2)
    y = pdf.paragraph(
        "Recorded and generated movement velocities are decomposed into one to four overlapping 2-D minimum-jerk components [5] using Prof. Friedman's implementation (repository commit 9c2f40c). Each component has onset, duration, lateral displacement, and forward displacement. Our primary order rule selects the smallest model with normalized fit error <=0.05, then <=0.10, otherwise the minimum-error candidate. Because the repository fits candidate orders but does not prescribe order selection, BIC and parameter sensitivities are reported explicitly.",
        y,
    )
    y = pdf.heading("4.2 Reproducibility", y, level=2)
    y = pdf.paragraph(
        "The confirmatory protocol is frozen as strategy-confirmatory-v1. The complete matrix contains 48 CVAE, 24 conditional-AE, 24 unconditional-VAE, 16 spline+PCA, and 12 condition-only runs. Neural seeds 42/43/44 vary initialization and training randomness; four outer rotations vary the held-out cohort. All metrics are derived from out-of-fold predictions only.",
        y,
    )
    pdf.assert_space(y, "page 3")

    # Page 4
    y = pdf.new_page("Benchmark results")
    y = pdf.heading("5. Benchmark results", y)
    y = pdf.image(figures["benchmark"], y, usable, 88 * mm)
    y = pdf.paragraph(
        "<b>Figure 2.</b> Participant-balanced out-of-fold endpoints. Spline+PCA minimizes deterministic trajectory MSE. Neural models provide lower timing error and, at n=8, lower generated-distribution distances. Metrics are not interchangeable; no single row is globally dominant.",
        y,
        style="caption",
    )
    table = benchmark_table(tables["oof"], fingerprints)
    y = pdf.table(
        table,
        y,
        [34 * mm, 8 * mm, 20 * mm, 23 * mm, 23 * mm, 19 * mm, 18 * mm, 23 * mm],
        font_size=6.4,
    )
    y = pdf.paragraph(
        "<b>Table 1.</b> Aggregate out-of-fold metrics. Enrollment accuracy is unavailable for the condition-only baseline. The n=8 CVAE reaches 50.7% closed-set enrollment versus 14.3% nominal chance, but this setting assumes context trials from each candidate test participant and is not zero-shot identification.",
        y,
        style="caption",
    )
    pdf.assert_space(y, "page 4")

    # Page 5
    y = pdf.new_page("Paired comparisons")
    y = pdf.heading("5.1 CVAE versus the spline baseline", y)
    y = pdf.paragraph(
        "The central comparison is a tradeoff. At n=3 and n=8, spline+PCA reconstructs observed trajectories more accurately, while the CVAE predicts timing and reproduces held-out feature distributions more closely. These statements are paired at the participant level and survive correction.",
        y,
    )
    left_w = 84 * mm
    right_x = 15 * mm + 91 * mm
    y_left = pdf.table(paired_table(tables["paired"], 3, "spline_pca"), y, [30*mm,13*mm,17*mm,18*mm,17*mm], x=15*mm, font_size=6.9)
    y_right = pdf.table(paired_table(tables["paired"], 8, "spline_pca"), y, [30*mm,13*mm,17*mm,18*mm,17*mm], x=right_x, font_size=6.9)
    y = min(y_left, y_right)
    pdf.canvas.setFillColor(colors.HexColor(GRAY))
    pdf.canvas.setFont("Helvetica-Oblique", 7.2)
    pdf.canvas.drawString(15 * mm, y - 0.5 * mm, "Table 2a. n=3 paired comparison")
    pdf.canvas.drawString(right_x, y - 0.5 * mm, "Table 2b. n=8 paired comparison")
    y -= 5.5 * mm
    y = pdf.image(figures["capacity"], y, usable, 58 * mm)
    y = pdf.paragraph(
        "<b>Figure 3.</b> Increasing latent capacity improves CVAE reconstruction, feature-distribution fidelity, and descriptive enrollment accuracy. The n=3 model preserves the requested low-dimensional interface; n=8 is the stronger capacity benchmark. The current data do not support claiming that n=3 is sufficient for the complete generative objective.",
        y,
        style="caption",
    )
    y = pdf.heading("5.2 What the condition-only baseline establishes", y, level=2)
    y = pdf.paragraph(
        "CVAE n=3 improves trajectory and timing prediction over condition-only Ridge for every participant, but not mean KS or MMD. CVAE n=8 improves all primary endpoints for most participants (BH-FDR p<0.001 for MSE, timing, mean KS, and MMD). Thus low-dimensional latent codes carry participant-specific predictive information, while strong distribution-level benefit over task metadata appears only at higher capacity.",
        y,
    )
    pdf.assert_space(y, "page 5")

    # Page 6
    y = pdf.new_page("Conditioning and submovements")
    y = pdf.heading("6. Conditioning and latent interpretation", y)
    y = pdf.image(figures["condition"], y, 112 * mm, 50 * mm, x=47 * mm)
    y = pdf.paragraph(
        "<b>Figure 4.</b> With the context fingerprint and latent draws fixed, CVAE and unconditional-VAE condition-stratum trajectory errors do not differ significantly (n=3 Holm p=1.000; n=8 p=0.132). Across 44 feature-level tests, no conditioning advantage survives BH-FDR. Task controls are therefore architecturally available but not validated as participant-specific causal controls. Axis-specific latent correlations are shown only as run-specific dashboard diagnostics because independent VAE fits are sign-, permutation-, and rotation-indeterminate.",
        y,
        style="caption",
    )
    y = pdf.heading("7. Minimum-jerk results", y)
    y = pdf.image(figures["submovement"], y, usable, 55 * mm)
    y = pdf.paragraph(
        "<b>Figure 5.</b> Recorded component counts depend strongly on model-order criterion. Generated count mismatch remains substantial (total variation 0.353 for n=3; 0.355 for n=8), and none of nine n=3 versus n=8 component-fidelity contrasts survives BH-FDR. Additional optimizer restarts preserve the threshold count on 100% of 56 sampled trials, whereas 167 ms constraints and a 5 Hz filter preserve it on only 55.4% and 51.8%. The decomposition supports kinematic description but not a cognitive-strategy label without Prof. Friedman's validation of filter and order rules.",
        y,
        style="caption",
    )
    pdf.assert_space(y, "page 6")

    # Page 7
    y = pdf.new_page("Discussion")
    y = pdf.heading("8. Discussion and lessons learned", y)
    discussion = [
        ["Supported", "Boundary"],
        ["Participant structure is non-random and context enrollment is above nominal chance.", "Enrollment uses trials from the test participant; it is not zero-shot identity recognition."],
        ["Spline+PCA is the best deterministic shape compressor at matched n.", "This does not make it the best stochastic generator or timing model."],
        ["CVAE n=8 improves timing and distribution fidelity over spline and task-only Ridge.", "Unconditional VAE is statistically comparable; explicit condition value is unproven."],
        ["CVAE n=3 is a usable compact fingerprint interface.", "Its distribution advantage over the condition-only baseline is not established."],
        ["Minimum-jerk parameters offer interpretable kinematic summaries.", "Component count is analysis-dependent and cannot currently be called strategy."],
    ]
    y = pdf.table(discussion, y, [88 * mm, 88 * mm], font_size=7.0)
    y = pdf.heading("8.1 Timing-head failure mode", y, level=2)
    y = pdf.paragraph(
        "Log-timing heads yield median initiation errors near 22-27 ms and 95th-percentile errors near 99-117 ms, but isolated extrapolations reach 11.5 s for CVAE n=8. These rare failures can make R^2 strongly negative despite moderate MAE. Participant-balanced MAE is therefore the primary timing endpoint; the unclipped outliers are disclosed in Figure A2 rather than hidden by post-hoc bounds.",
        y,
    )
    y = pdf.heading("8.2 Limitations", y, level=2)
    y = pdf.paragraph(
        "The cohort contains 28 participants and no independent acquisition cohort. Context enrollment is required for unseen participants. Phase normalization intentionally separates shape from time but can remove local velocity information. Generated distributions are evaluated on selected kinematic summaries and cannot certify perceptual realism. Latent axes are not identifiable across independent fits. The minimum-jerk order rule, 10 Hz filter, short-task temporal bounds, late-trial threshold, and condition-2 fixation flag require domain confirmation.",
        y,
    )
    y = pdf.heading("8.3 End product and future work", y, level=2)
    y = pdf.paragraph(
        "The Streamlit dashboard exposes the strategy-window generator, context fingerprints, n=3/n=8 models, held-out metrics, and run-specific latent diagnostics. Condition sliders are labeled exploratory. After Prof. Friedman's review, the priority is to freeze task-specific exclusions and minimum-jerk settings, rerun only affected stages, calibrate or bound timing outputs in a pre-specified manner, and evaluate an independent cohort if available. A hierarchical subject-level VAE is a publication-oriented extension, not required for the workshop result.",
        y,
    )
    pdf.assert_space(y, "page 7")

    # Page 8
    y = pdf.new_page("References and appendix")
    y = pdf.heading("References", y)
    refs = (
        "[1] Brenner, E. and Smeets, J.B.J. Continuously updating one's predictions underlies successful interception. <i>J Neurophysiol</i> 120, 3257-3274 (2018). doi:10.1152/jn.00517.2018.<br/>"
        "[2] Slowinski, P. et al. Dynamic similarity promotes interpersonal coordination in joint action. <i>J R Soc Interface</i> 13, 20151093 (2016). doi:10.1098/rsif.2015.1093.<br/>"
        "[3] Kingma, D.P. and Welling, M. Auto-Encoding Variational Bayes. ICLR (2014). arXiv:1312.6114.<br/>"
        "[4] Sohn, K., Lee, H. and Yan, X. Learning Structured Output Representation using Deep Conditional Generative Models. NeurIPS 28 (2015).<br/>"
        "[5] Flash, T. and Hogan, N. The coordination of arm movements: an experimentally confirmed mathematical model. <i>J Neurosci</i> 5, 1688-1703 (1985). doi:10.1523/JNEUROSCI.05-07-01688.1985.<br/>"
        "[6] Friedman, J. <i>submovements</i> repository, commit 9c2f40c. github.com/JasonFriedman/submovements."
    )
    y = pdf.paragraph(refs, y, style="small")
    y = pdf.heading("Appendix A. Representative held-out outputs", y)
    y = pdf.image(figures["examples"], y, usable, 63 * mm)
    y = pdf.paragraph(
        f"<b>Figure A1.</b> Posterior-mean reconstructions and fingerprint-conditioned generation for a held-out participant. Reference run: fold 0, CVAE n=8, seed 42; typical trial {example_meta['typical_trial']}; high-error trial {example_meta['difficult_trial']}. This figure is illustrative; inferential results use every fold and seed.",
        y,
        style="caption",
    )
    y = pdf.image(figures["timing"], y, 145 * mm, 48 * mm, x=31 * mm)
    y = pdf.paragraph(
        "<b>Figure A2.</b> CVAE initiation-time error distribution by seed. Log scale makes the separation between typical and worst-case behavior visible. Full raw and seed-averaged tables, feature-wise fidelity files, condition diagnostics, and minimum-jerk sensitivity outputs accompany the executable repository.",
        y,
        style="caption",
    )
    y = pdf.heading("Questions for Prof. Friedman", y, level=2)
    y = pdf.paragraph(
        "Confirm (i) whether condition-2 'Not fixating on the dot enough!!!' trials should remain; (ii) the intended late-trial reference event and threshold; (iii) whether 10 Hz and our short-task component bounds are appropriate; and (iv) the preferred minimum-jerk model-order criterion before strategy-level interpretation.",
        y,
    )
    pdf.assert_space(y, "page 8")
    pdf.finish()
    if pdf.page != 8:
        raise RuntimeError(f"scientific report must be 8 pages, generated {pdf.page}")


def guide_report(
    output: Path,
    tables: dict[str, pd.DataFrame],
    fingerprints: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    pdf = ManualPDF(
        output,
        "Results Guide: Interception-Movement Fingerprints",
        "Simaan Libbiss and Paz Flashner",
        compact=False,
    )
    usable = pdf.width - 30 * mm

    y = pdf.new_page("How to use this guide")
    y = pdf.heading("Results Guide for Simaan and Paz", y)
    y = pdf.paragraph(
        "This document explains the exact analysis behind the scientific report. Figure and table numbers match the report. Use it to check the model's unit of analysis, the direction of each metric, and which conclusions are justified before presenting the work.",
        y,
    )
    y = pdf.callout(
        "The sentence to remember: spline+PCA reconstructs observed shape better; the CVAE predicts timing and generated distributions better; n=8 is the stronger generator, while n=3 is the requested compact fingerprint. Explicit conditioning and cognitive strategy labels are not yet validated.",
        y,
    )
    y = pdf.heading("The project question", y, level=2)
    y = pdf.paragraph(
        "We are not asking whether one trial identifies a person with no prior data. We enroll an unseen participant from a context half of their trials, obtain one fingerprint center, and ask whether that fingerprint can generate the distribution of the participant's disjoint query trials. This is participant-level adaptation followed by held-out within-participant prediction.",
        y,
    )
    y = pdf.heading("What the CVAE sees", y, level=2)
    y = pdf.paragraph(
        "Encoder input: 100 x 2 filtered position samples plus task metadata. It does not receive participant identity, physical initiation time, physical movement time, or trial number. Decoder outputs: a phase-normalized x-y trajectory plus initiation and movement time. The fingerprint is the mean posterior coordinate over the participant's context trials. Averaging is meaningful within one trained VAE coordinate system, but latent axes cannot be compared directly across independently trained seeds without alignment.",
        y,
    )
    y = pdf.image(figures["pipeline"], y, usable, 39 * mm)
    y = pdf.paragraph("<b>Figure 1.</b> The test participant is unseen during model fitting, but their context trials are intentionally used for enrollment. Their query trials remain untouched until evaluation.", y, style="caption")
    pdf.assert_space(y, "guide page 1")

    y = pdf.new_page("Metric reference")
    y = pdf.heading("Metric reference", y)
    metric_data = [
        ["Metric", "Formula / comparison", "Better", "What it measures"],
        ["Trajectory MSE", "mean squared x-y error over 100 phases", "lower", "posterior-mean reconstruction of an observed trial"],
        ["Timing MAE", "mean |predicted - observed|, milliseconds", "lower", "typical absolute timing error; primary timing endpoint"],
        ["R^2", "1 - sum(y-yhat)^2 / sum(y-ybar)^2", "higher", "variance explained; can be negative and is outlier-sensitive"],
        ["KS statistic", "maximum separation between two empirical CDFs", "lower", "one feature's generated-versus-query distribution mismatch"],
        ["MMD", "kernel distance between multivariate samples", "lower", "joint distribution mismatch after feature scaling"],
        ["Energy distance", "distance-based multivariate discrepancy", "lower", "joint mismatch; raw values are scale/outlier sensitive"],
        ["Balanced accuracy", "mean recall across seven enrolled subjects", "higher", "closed-set fingerprint enrollment, not zero-shot identification"],
        ["TV / JSD", "categorical distribution distance", "lower", "component-count distribution mismatch"],
        ["Wilcoxon + BH-FDR", "paired participant ranks + multiple-test correction", "p<.05", "whether a model contrast is consistent across participants"],
    ]
    y = pdf.table(metric_data, y, [31*mm,54*mm,18*mm,73*mm], font_size=7.4)
    y = pdf.heading("The inference unit", y, level=2)
    y = pdf.paragraph(
        "Neural seeds are repeated measurements of the same participant split. We first average seeds within each held-out participant, then run paired tests across 28 participants. Treating 3 seeds x 6 condition strata x 28 participants as independent would artificially inflate the sample size; that earlier aggregation was corrected before these reports.",
        y,
    )
    y = pdf.heading("Why MAE is primary for timing", y, level=2)
    y = pdf.paragraph(
        "Most timing predictions are close, but a few log-space decoder outputs extrapolate to impossible durations. MAE, median error, and the 95th percentile describe typical performance more honestly than a single pooled R^2. We still retain the unclipped R^2 and maximum error in raw outputs so failure cases are visible.",
        y,
    )
    pdf.assert_space(y, "guide page 2")

    y = pdf.new_page("Reading the benchmark")
    y = pdf.heading("Reading Figure 2 and Table 1", y)
    y = pdf.image(figures["benchmark"], y, usable, 90 * mm)
    y = pdf.paragraph(
        "Each panel answers a different question. Trajectory MSE uses an observed test trial as encoder input, so it measures compression/reconstruction. KS and MMD use only a context fingerprint plus sampled latent noise and compare generated samples with separate query trials, so they measure distribution generation. A model can win one and lose the other without contradiction.",
        y,
    )
    table = benchmark_table(tables["oof"], fingerprints)
    y = pdf.table(table, y, [34*mm,8*mm,20*mm,23*mm,23*mm,19*mm,18*mm,23*mm], font_size=6.4)
    y = pdf.heading("What the rows say", y, level=2)
    y = pdf.paragraph(
        "Spline+PCA n=8 has the lowest trajectory MSE (0.023 versus CVAE 0.032). CVAE n=8 has better initiation MAE (34.8 versus 55.6 ms), movement MAE (44.2 versus 50.2 ms), mean KS (0.219 versus 0.255), and MMD (0.074 versus 0.132). Therefore 'CVAE is better' is too broad; 'CVAE is better for timing and generated distributions, while spline+PCA is better for deterministic shape reconstruction' is accurate.",
        y,
    )
    pdf.assert_space(y, "guide page 3")

    y = pdf.new_page("Paired evidence")
    y = pdf.heading("What makes the comparison scientific", y)
    y = pdf.paragraph(
        "Table 2 compares the two methods participant by participant after seed averaging. 'CVAE better 23/28' means the CVAE had lower error for 23 of 28 held-out participants. The Wilcoxon test uses the signed ranks of all 28 paired differences; BH-FDR adjusts p-values across the family of model/metric comparisons.",
        y,
    )
    y1 = pdf.table(paired_table(tables["paired"], 3, "spline_pca"), y, [33*mm,18*mm,24*mm,25*mm,23*mm], x=15*mm, font_size=8.0)
    y = pdf.paragraph("<b>Table 2a.</b> CVAE n=3 versus spline+PCA n=3.", y1, style="caption")
    y2 = pdf.table(paired_table(tables["paired"], 8, "spline_pca"), y, [33*mm,18*mm,24*mm,25*mm,23*mm], x=15*mm, font_size=8.0)
    y = pdf.paragraph("<b>Table 2b.</b> CVAE n=8 versus spline+PCA n=8.", y2, style="caption")
    y = pdf.heading("Why the condition-only baseline matters", y, level=2)
    y = pdf.paragraph(
        "Without this baseline, good generated distributions might simply reflect the six task strata and population variability. At n=3, CVAE clearly improves reconstruction and timing but not mean KS/MMD over condition-only Ridge. At n=8, it improves all primary endpoints for most participants. This is the strongest evidence that the higher-capacity fingerprint carries useful participant information beyond task metadata.",
        y,
    )
    y = pdf.heading("What not to say", y, level=2)
    y = pdf.paragraph(
        "Do not say that n=3 fully predicts each person's distributions, that the CVAE is universally superior, or that negative R^2 means every timing prediction is bad. Use the endpoint-specific statements above.",
        y,
    )
    pdf.assert_space(y, "guide page 4")

    y = pdf.new_page("Latent dimension")
    y = pdf.heading("Why n=3 and n=8 tell different parts of the story", y)
    y = pdf.image(figures["capacity"], y, usable, 62 * mm)
    y = pdf.paragraph(
        "<b>Figure 3A-B.</b> More latent capacity lowers reconstruction and distribution error. <b>Figure 3C.</b> Enrollment also improves descriptively. This does not prove that every new coordinate has a separate human meaning; a VAE may rotate or distribute information across coordinates.",
        y,
    )
    y = pdf.heading("n=3", y, level=2)
    y = pdf.paragraph(
        "This matches Jason's desired 2-3 controls and is easy to visualize. It beats spline+PCA for timing and distribution fidelity, but it does not beat the condition-only baseline on the main distribution metrics. Treat it as the compact interpretation model, not the strongest final generator.",
        y,
    )
    y = pdf.heading("n=8", y, level=2)
    y = pdf.paragraph(
        "This model has enough capacity to improve generated distributions beyond the task-only baseline and gives the best CVAE fidelity. It is harder to interpret directly. Its result tells us the architecture can represent useful participant information, while n=3 may be too restrictive for complete fidelity.",
        y,
    )
    y = pdf.heading("Why two latent coordinates may correlate", y, level=2)
    y = pdf.paragraph(
        "The KL term encourages a standard-normal aggregate prior but does not guarantee independent semantic factors. Coordinate correlations within one fit can arise from the data manifold and from imperfect posterior factorization. Across seeds, signs and axis order can change. This is why the dashboard's heatmap is a run-specific exploratory explanation, not a cross-seed causal map of what each slider controls.",
        y,
    )
    pdf.assert_space(y, "guide page 5")

    y = pdf.new_page("Conditioning")
    y = pdf.heading("Did the task condition help?", y)
    y = pdf.image(figures["condition"], y, 145 * mm, 62 * mm, x=31 * mm)
    y = pdf.paragraph(
        "<b>Figure 4.</b> The CVAE receives task condition and the unconditional VAE does not. We hold the fingerprint and random latent draws fixed, change only the condition, and compare the resulting six sp x side strata with observed strata. If conditioning gave reliable participant-specific control, CVAE errors should be consistently lower.",
        y,
    )
    y = pdf.paragraph(
        "They are not significantly lower: n=3 Holm p=1.000 and n=8 p=0.132 for the trajectory diagnostic, and none of 44 feature tests survives BH-FDR. Therefore the dashboard can expose the controls because the architecture uses them, but we cannot claim that moving a speed or side control reproduces the correct participant-specific effect.",
        y,
    )
    y = pdf.callout(
        "Correct answer if asked: conditioning was motivated by task structure, but the matched unconditional ablation shows that its incremental value is not established in this dataset.",
        y,
    )
    y = pdf.heading("Why we keep the CVAE at all", y, level=2)
    y = pdf.paragraph(
        "The research interface requested by Jason is conditional: choose a fingerprint and task setting, then sample plausible movement distributions. The CVAE is the direct architecture for that interface. The ablation result means its conditioning claims must stay provisional, not that the model or experiment is invalid.",
        y,
    )
    pdf.assert_space(y, "guide page 6")

    y = pdf.new_page("Minimum-jerk analysis")
    y = pdf.heading("What a minimum-jerk component is", y)
    y = pdf.paragraph(
        "A component is one smooth 2-D velocity pulse described by onset time, duration, lateral displacement, and forward displacement. Several components can overlap so their summed velocity approximates a recorded movement. Component count is chosen by us after fitting 1, 2, 3, and 4 candidate models; Jason's repository does not choose the winning count automatically.",
        y,
    )
    y = pdf.image(figures["submovement"], y, usable, 61 * mm)
    y = pdf.paragraph(
        "<b>Figure 5A.</b> Threshold selection and BIC produce very different recorded count distributions. <b>5B.</b> Total variation around 0.35 means the generated categorical count distribution still differs substantially from recorded query trials. <b>5C.</b> Extra optimizer restarts do not change the primary count, but filter and temporal constraints change almost half of counts.",
        y,
    )
    y = pdf.heading("Can count be called strategy?", y, level=2)
    y = pdf.paragraph(
        "Not yet. A one-component movement can be compatible with a wait-then-go interpretation and multiple components can be compatible with correction, but the decomposition is kinematic. The sensitivity result shows that count also depends on analysis settings. We need Jason to approve the filter, 167 ms question, and model-order criterion before mapping count to a cognitive strategy.",
        y,
    )
    y = pdf.heading("n=3 versus n=8", y, level=2)
    y = pdf.paragraph(
        "n=8 trends better for several continuous component parameters, but none of nine paired contrasts survives BH-FDR. It does not improve component-count TV or JSD. Therefore minimum-jerk analysis does not provide evidence for choosing n=8 over n=3.",
        y,
    )
    pdf.assert_space(y, "guide page 7")

    y = pdf.new_page("Examples and timing failures")
    y = pdf.heading("Representative outputs are illustrations, not the test", y)
    y = pdf.image(figures["examples"], y, usable, 64 * mm)
    y = pdf.paragraph(
        "<b>Figure A1.</b> Panels A-B encode the observed trial and decode its posterior mean. Panel C does not encode the query trials; it samples around the participant's context fingerprint and compares the resulting family of trajectories with recorded query examples. One attractive plot cannot establish fidelity, which is why Tables 1-2 use every held-out participant.",
        y,
    )
    y = pdf.image(figures["timing"], y, 150 * mm, 53 * mm, x=28 * mm)
    y = pdf.paragraph(
        "<b>Figure A2.</b> The median and 95th percentile are reasonable while maxima are orders of magnitude larger. This explains the apparently contradictory timing story: MAE can be around 35-45 ms while pooled R^2 is poor or negative. The model is usually close but occasionally catastrophically wrong.",
        y,
    )
    pdf.assert_space(y, "guide page 8")

    y = pdf.new_page("Presentation checklist")
    y = pdf.heading("Questions you should be able to answer", y)
    qa = [
        ["Question", "Short accurate answer"],
        ["Why target onset rather than finger onset?", "It preserves waiting and initiation strategy; physical initiation time remains an explicit output."],
        ["Why resample to 100 phases?", "The network needs fixed length. Because phase removes duration, timing is predicted separately."],
        ["Why average latent codes?", "Context trials are repeated noisy observations in one learned coordinate system; their mean estimates a participant center."],
        ["Why is spline reconstruction better?", "A low-rank linear shape basis is highly efficient for these smooth trajectories; CVAE also learns stochastic generation and timing."],
        ["Why use CVAE if U-VAE is similar?", "It matches the desired task-conditional interface, but its incremental conditioning value remains unproven and is reported honestly."],
        ["Do components prove strategy?", "No. They are model-dependent kinematic primitives that may support later strategy interpretation."],
        ["What is the main positive result?", "Participant fingerprints support above-chance enrollment; n=8 generates timing/features better than spline and task-only Ridge."],
        ["What is the main limitation?", "The requested n=3 code does not establish full distribution benefit over task metadata, and conditioning is not validated."],
    ]
    y = pdf.table(qa, y, [58 * mm, 118 * mm], font_size=8.0)
    y = pdf.heading("Claims to avoid", y, level=2)
    y = pdf.paragraph(
        "Avoid: 'CVAE is more accurate'; 'n=3 predicts the subject'; 'latent z1 controls peak count'; 'component count is the participant's strategy'; 'the data are insufficient' without a learning-curve test; or 'R^2=negative means the model learned nothing.' Replace each with endpoint-specific, test-backed wording from this guide.",
        y,
    )
    y = pdf.heading("Final interpretation", y, level=2)
    y = pdf.paragraph(
        "The project succeeds as a deep-learning benchmark and analysis: it establishes a leakage-resistant participant-fingerprint task, shows where a CVAE adds value, identifies where a classical spline remains stronger, and exposes two unresolved scientific issues - conditioning validity and submovement model order. It does not yet prove a fully controllable 2-3D biological strategy space.",
        y,
    )
    pdf.assert_space(y, "guide page 9")
    pdf.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "pdf")
    args = parser.parse_args()
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
    }
    make_pipeline_figure(figures["pipeline"])
    make_benchmark_figure(tables["oof"], figures["benchmark"])
    make_capacity_fingerprint_figure(tables["oof"], fingerprints, figures["capacity"])
    make_condition_figure(tables["condition"], figures["condition"])
    make_submovement_figure(
        tables["real_submovements"], tables["min_jerk"], tables["sensitivity"], figures["submovement"]
    )
    example_meta = make_reconstruction_generation_figure(figures["examples"])
    make_timing_outlier_figure(tables["timing"], figures["timing"])

    scientific = args.out / "Interception_Movements_Final_Scientific_Report.pdf"
    guide = args.out / "Interception_Movements_Results_Guide.pdf"
    scientific_report(scientific, tables, fingerprints, figures, example_meta)
    guide_report(guide, tables, fingerprints, figures)
    print(scientific)
    print(guide)


if __name__ == "__main__":
    main()
