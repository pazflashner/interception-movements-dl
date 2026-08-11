"""Build the advisor-facing final study PDF from final-study artifacts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_corrected_study import load_per_trial_checkpoint
from src.context_query import split_context_query
from src.preprocessing import lowpass_filter
from src.submovements import minimum_jerk_velocity, reconstruct_velocity
from src.train import split_subjects
from src.vae_model import encode_condition

FINAL = ROOT / "results" / "final_study"
OUT_DIR = ROOT / "output" / "final_report"
FIG_DIR = OUT_DIR / "figures"
OUT = OUT_DIR / "Interception_Movement_Fingerprints_Final.pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#17324D")
BLUE = "#176B87"
CYAN = "#25A7B8"
ORANGE = "#D97732"
GREEN = "#2E7D5B"
RED = "#B54747"
LIGHT = colors.HexColor("#EEF3F6")
MID = colors.HexColor("#CCD8DF")
TEXT = colors.HexColor("#26343E")


def load_data():
    with open(ROOT / "data" / "final_study" / "trials.pkl", "rb") as handle:
        trials = pickle.load(handle)
    sub = pd.read_csv(FINAL / "submovements_real.csv")
    seeds = pd.read_csv(FINAL / "core_models" / "seed_results.csv")
    subject_probe = pd.read_csv(FINAL / "fingerprint_evaluation" / "subject_distribution_probes.csv")
    trial_probe = pd.read_csv(FINAL / "fingerprint_evaluation" / "trial_behavior_probes.csv")
    generation = pd.read_csv(FINAL / "generation" / "generation_summary.csv")
    generation["latent_dim"] = generation.run.str.extract(r"_z(\d+)_").astype(int)
    baseline_dir = FINAL / "baselines"
    kmeans = pd.read_csv(baseline_dir / "kmeans_selection_corrected.csv")
    baselines = json.loads((baseline_dir / "baselines.json").read_text())
    stability_path = FINAL / "submovement_stability.csv"
    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    return trials, sub, seeds, subject_probe, trial_probe, generation, kmeans, baselines, stability


def savefig(name):
    plt.tight_layout()
    path = FIG_DIR / name
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()
    return path


def make_figures(trials, sub, seeds, subject_probe, trial_probe, generation, kmeans, baselines):
    # Data geometry and event timing.
    variations = np.array([np.abs(np.diff(t["pos_norm"], axis=0)).sum(axis=0) for t in trials])
    fractions = np.median(variations / variations.sum(axis=1, keepdims=True), axis=0)
    sample = trials[len(trials) // 3]
    time = np.arange(len(sample["pos_filtered"])) / 240.0
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.5))
    axes[0].bar(["x lateral", "y forward", "z vertical"], fractions * 100, color=[CYAN, BLUE, ORANGE])
    axes[0].set_ylabel("Median share of path variation (%)"); axes[0].set_ylim(0, 100); axes[0].grid(axis="y", alpha=.2)
    for axis, colour in zip(range(3), [CYAN, BLUE, ORANGE]):
        axes[1].plot(time, sample["pos_filtered"][:, axis] - sample["pos_filtered"][0, axis], color=colour, label="xyz"[axis])
    for idx, label, style in ((sample["go_signal_idx"], "go", "--"), (sample["move_start_idx"], "onset", ":"),
                              (sample["move_end_idx"], "arrival", "-.")):
        axes[1].axvline(idx / 240.0, color="black", ls=style, lw=1, label=label)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Position relative to start (tracker units)")
    axes[1].legend(frameon=False, ncol=2, fontsize=8); axes[1].grid(alpha=.2)
    fig.suptitle("Table-plane movement and preserved timing variables")
    fig.tight_layout(); fig.savefig(FIG_DIR / "data_geometry.png", dpi=190); plt.close(fig)

    # Pipeline diagram.
    fig, ax = plt.subplots(figsize=(8.2, 2.7)); ax.axis("off")
    blocks = [
        (0.01, "Filtered x-y\ntable-plane trajectory"), (0.21, "Phase-normalized\nmovement shape"),
        (0.41, "Conditional VAE\nlatent z"), (0.61, "Generated shape\n+ generated timing"),
        (0.81, "Minimum-jerk\nbehavior distributions"),
    ]
    for x, text in blocks:
        ax.add_patch(plt.Rectangle((x, .30), .16, .40, facecolor="#EEF3F6", edgecolor=BLUE, lw=1.5))
        ax.text(x + .08, .50, text, ha="center", va="center", fontsize=9)
    for x, _ in blocks[:-1]: ax.annotate("", xy=(x + .20, .50), xytext=(x + .16, .50), arrowprops={"arrowstyle": "->", "color": "#17324D"})
    ax.text(.5, .12, "Task condition enters the encoder/decoder; true timing is withheld from the encoder", ha="center", fontsize=9, color=BLUE)
    savefig("pipeline.png")

    # Baselines.
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    axes[0].bar(kmeans.representation.str.replace("_", " "), kmeans.ari, color=[BLUE, CYAN])
    axes[0].axhline(kmeans.null_95pct.max(), color="black", ls="--", label="95% permutation null")
    axes[0].set_ylabel("Adjusted Rand Index"); axes[0].set_title("Fixed k=28 K-Means")
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(axis="y", alpha=.2)
    spline = pd.Series(baselines["spline_pca"], dtype=float)
    model = seeds.groupby("latent_dim").reconstruction_mse_tracker_units2.agg(["mean", "std"])
    axes[1].plot(spline.index.astype(int), spline.values, "o--", label="Spline + PCA")
    axes[1].errorbar(model.index, model["mean"], yerr=model["std"], fmt="o-", capsize=3, label="Trajectory-only CVAE")
    axes[1].set_yscale("log"); axes[1].set_xlabel("Latent dimension"); axes[1].set_ylabel("Held-out trajectory MSE")
    axes[1].set_title("Compression baselines"); axes[1].legend(frameon=False, fontsize=8); axes[1].grid(alpha=.2)
    savefig("baselines.png")

    # Core model performance.
    grouped = seeds.groupby("latent_dim")
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.3))
    metrics = [
        ("movement_time_s_r2", "Movement-time R2", BLUE),
        ("initiation_time_s_r2", "Initiation-time R2", ORANGE),
        ("fingerprint_balanced_accuracy", "Fingerprint ID accuracy", GREEN),
    ]
    for ax, (column, title, colour) in zip(axes, metrics):
        agg = grouped[column].agg(["mean", "std"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], fmt="o-", capsize=3, color=colour)
        ax.axhline(0, color="black", lw=.8)
        if column == "fingerprint_balanced_accuracy": ax.axhline(1 / 7, color="black", ls="--", lw=.8)
        ax.set_title(title); ax.set_xlabel("Latent n"); ax.grid(alpha=.2)
    savefig("deep_results.png")

    # Minimum-jerk examples.
    valid = sub[sub.mj_fit_success == True].copy()
    by_id = {t["metadata"]["trial_id"]: t for t in trials}
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.25), sharey=True)
    for ax, count in zip(axes, (1, 2, 3)):
        candidates = valid[valid.mj_n_components == count]
        target_error = candidates.mj_fit_error.median()
        row = candidates.iloc[(candidates.mj_fit_error - target_error).abs().argmin()]
        trial = by_id[row.trial_id]
        start, end = trial["move_start_idx"], trial["move_end_idx"]
        pos = trial["pos_filtered"][start : end + 1, :2]
        filtered = lowpass_filter(pos, cutoff=10, fs=240)
        vel = np.gradient(filtered, 1 / 240, axis=0); t = np.arange(len(vel)) / 240
        params = np.asarray(json.loads(row.mj_parameters_json))
        summed = reconstruct_velocity(t, params)
        ax.plot(t, np.linalg.norm(vel, axis=1), color="black", lw=1.6, label="recorded")
        ax.plot(t, np.linalg.norm(summed, axis=1), color=RED, lw=1.4, label="sum")
        for i, p in enumerate(params):
            comp = minimum_jerk_velocity(t, p[0], p[1], p[2:4])
            ax.plot(t, np.linalg.norm(comp, axis=1), ls="--", lw=1, label=f"component {i+1}")
        ax.set_title(f"{count} component{'s' if count > 1 else ''}\nerror={row.mj_fit_error:.3f}")
        ax.set_xlabel("Movement time (s)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Speed (tracker units/s)"); axes[-1].legend(frameon=False, fontsize=7)
    savefig("submovement_examples.png")

    # Population submovement patterns.
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.3))
    counts = valid.mj_n_components.value_counts(normalize=True).sort_index()
    axes[0].bar(counts.index.astype(int), counts.values * 100, color=BLUE)
    axes[0].set_xticks([1, 2, 3, 4]); axes[0].set_ylabel("Trials (%)"); axes[0].set_title("Effective component count")
    subject_rate = valid.assign(single=valid.mj_n_components == 1).groupby("subject").single.mean()
    axes[1].hist(subject_rate * 100, bins=10, color=CYAN, edgecolor="white")
    axes[1].set_xlabel("Single-component trials (%)"); axes[1].set_title("Between-subject variation")
    outcome = pd.crosstab(valid.responseText, valid.mj_n_components, normalize="index") * 100
    bottom = np.zeros(len(outcome))
    for count in sorted(outcome.columns):
        axes[2].bar(np.arange(len(outcome)), outcome[count], bottom=bottom, label=f"k={int(count)}")
        bottom += outcome[count].to_numpy()
    axes[2].set_xticks(np.arange(len(outcome)), [x.replace("Not fixating on the dot enough!!!", "Not-fixating flag") for x in outcome.index], rotation=20, ha="right")
    axes[2].set_ylabel("Within-label trials (%)"); axes[2].set_title("Recorded outcome labels"); axes[2].legend(frameon=False, fontsize=7)
    savefig("submovement_population.png")

    # Subject and trial probes.
    targets = [
        "mj_n_components_mean", "single_component_rate", "mj_first_duration_s_mean",
        "mj_secondary_amplitude_fraction_mean", "mj_mean_overlap_pct_mean", "recorded_success_rate",
    ]
    probe = subject_probe[subject_probe.target.isin(targets)].groupby(["target", "latent_dim"]).r2_test.mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.7))
    im = axes[0].imshow(probe.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(probe.columns)), probe.columns); axes[0].set_yticks(range(len(probe.index)), [x.replace("mj_", "").replace("_", " ") for x in probe.index], fontsize=7)
    axes[0].set_xlabel("Latent n"); axes[0].set_title("Subject query-summary R2")
    fig.colorbar(im, ax=axes[0], fraction=.046)
    trial_class = trial_probe[trial_probe.metric_type == "classification"].groupby(["target", "latent_dim"]).balanced_accuracy.mean().unstack()
    x = np.arange(len(trial_class.columns)); width = .34
    for i, (target, row) in enumerate(trial_class.iterrows()):
        axes[1].bar(x + (i - .5) * width, row.values, width, label=target.replace("mj_", ""))
    axes[1].axhline(.5, color="black", ls="--", lw=.8, label="binary chance")
    axes[1].axhline(.25, color="black", ls=":", lw=.8, label="4-class chance")
    axes[1].set_xticks(x, trial_class.columns); axes[1].set_xlabel("Latent n"); axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Trial-level behavioral probes"); axes[1].legend(frameon=False, fontsize=7)
    savefig("behavior_probes.png")

    # Generation fidelity.
    gen = generation.groupby("latent_dim").mean(numeric_only=True)
    feature_names = ["movement_time_s", "initiation_time_s", "path_length", "curvature_index",
                     "mj_first_duration_s", "mj_first_amplitude", "mj_secondary_amplitude_fraction", "mj_mean_overlap_pct"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for n in gen.index:
        values = [gen.loc[n, f"mean_ks_{feature}"] for feature in feature_names]
        axes[0].plot(range(len(feature_names)), values, "o-", label=f"n={n}")
    axes[0].set_xticks(range(len(feature_names)), [x.replace("mj_", "").replace("_", " ") for x in feature_names], rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("Mean KS statistic"); axes[0].set_title("Generated continuous distributions"); axes[0].legend(frameon=False, fontsize=7); axes[0].grid(alpha=.2)
    axes[1].bar(gen.index.astype(str), gen.mean_count_jsd, color=[BLUE, CYAN, ORANGE][:len(gen)])
    axes[1].set_xlabel("Latent n"); axes[1].set_ylabel("Mean count JSD (0=match)"); axes[1].set_title("Generated component-count distribution")
    axes[1].grid(axis="y", alpha=.2)
    savefig("generation_fidelity.png")

    # Controlled n=2 traversals.
    traversal = pd.read_csv(FINAL / "latent_traversals" / "latent_z2.csv")
    arrays = np.load(FINAL / "latent_traversals" / "latent_z2_trajectories.npz")
    trajectories = arrays["trajectories"]
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.2))
    for idx, row in traversal.iterrows():
        if row.z2_level == 0:
            axes[0].plot(trajectories[idx, :, 0], trajectories[idx, :, 1], label=f"z1={int(row.z1_level)}")
        if row.z1_level == 0:
            axes[1].plot(trajectories[idx, :, 0], trajectories[idx, :, 1], label=f"z2={int(row.z2_level)}")
    for ax, title in zip(axes[:2], ("Vary latent 1", "Vary latent 2")):
        ax.set_xlabel("Lateral x"); ax.set_ylabel("Forward y"); ax.set_title(title); ax.legend(frameon=False, fontsize=6); ax.grid(alpha=.2)
    pivot = traversal.pivot(index="z2_level", columns="z1_level", values="movement_time_s")
    im = axes[2].imshow(pivot.values, origin="lower", cmap="viridis")
    axes[2].set_xticks(range(5), pivot.columns.astype(int)); axes[2].set_yticks(range(5), pivot.index.astype(int))
    axes[2].set_xlabel("Latent 1 level"); axes[2].set_ylabel("Latent 2 level"); axes[2].set_title("Generated movement time")
    fig.colorbar(im, ax=axes[2], fraction=.046, label="seconds")
    savefig("latent_traversal.png")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("titlex", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8),
        "subtitle": ParagraphStyle("subx", parent=base["Normal"], fontSize=11, leading=15, textColor=colors.HexColor("#566B78"), spaceAfter=12),
        "h1": ParagraphStyle("h1x", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=4, spaceAfter=7),
        "h2": ParagraphStyle("h2x", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor(BLUE), spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("bodyx", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=TEXT, spaceAfter=5),
        "small": ParagraphStyle("smallx", parent=base["BodyText"], fontSize=7.6, leading=10, textColor=colors.HexColor("#5D707B"), spaceAfter=3),
        "callout": ParagraphStyle("calloutx", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10.2, leading=14, textColor=NAVY, backColor=LIGHT, borderPadding=8, spaceAfter=8),
    }


def table(rows, widths, font=8):
    wrapped = [[cell if isinstance(cell, Paragraph) else Paragraph(str(cell), ParagraphStyle("cell", fontName="Helvetica", fontSize=font, leading=font + 2)) for cell in row] for row in rows]
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .35, MID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build(trials, sub, seeds, subject_probe, trial_probe, generation, kmeans, baselines, stability):
    s = styles(); P = lambda text, style="body": Paragraph(text, s[style]); story = []
    stimulus_protocol = json.loads(
        (ROOT / "data" / "final_study" / "trials.protocol.json").read_text(encoding="utf-8")
    )
    model = seeds.groupby("latent_dim").agg({"movement_time_s_r2": "mean", "initiation_time_s_r2": "mean", "fingerprint_balanced_accuracy": "mean", "reconstruction_mse_tracker_units2": "mean"})
    best_move = model.movement_time_s_r2.idxmax(); best_id = model.fingerprint_balanced_accuracy.idxmax()
    valid = sub[sub.mj_fit_success == True]
    component_rates = valid.mj_n_components.value_counts(normalize=True).sort_index()
    bic_four_rate = float((valid.mj_n_components_bic == 4).mean())
    gen = generation.groupby("latent_dim").mean(numeric_only=True)
    best_count = gen.mean_count_jsd.idxmin()
    fdr_rejections = gen.mean_ks_rejected_fdr
    subject_probe_mean = subject_probe.groupby(["latent_dim", "target"]).r2_test.mean()
    low_dim_positive = int((subject_probe_mean.loc[[2, 3]] > 0).sum())
    low_dim_total = int(len(subject_probe_mean.loc[[2, 3]]))
    stability_text = f"{100*stability.same_selected_count.mean():.1f}%" if len(stability) else "not available"

    story += [Spacer(1, 16 * mm), P("Learning Low-Dimensional Movement Fingerprints", "title"),
              P("A conditional variational autoencoder study of fast interception movements", "subtitle"),
              P("Seman Libbiss and Paz Flashner | Deep Learning Workshop | Prof. Jason Friedman | Advisor: Moni", "subtitle"),
              Spacer(1, 4 * mm), P("Research objective", "h1"),
              P("Can two or three continuous latent parameters learned directly from movement trajectories represent an individual and reproduce that person's held-out distributions of movement shape, timing, and minimum-jerk submovement structure?", "callout"),
              P("The main contribution is the conditional variational autoencoder and its held-out subject evaluation. K-Means, spline/PCA, and minimum-jerk decomposition are used as baselines and interpretable measurement tools."),
              P("Headline findings", "h2"),
              table([
                  ["Question", "Result"],
                  ["Is subject information present?", f"Yes, but natural clustering is weak. Enrolled-subject identification is best at n={best_id}, averaging {100*model.loc[best_id,'fingerprint_balanced_accuracy']:.1f}% versus 14.3% chance."],
                  ["Does movement shape contain timing?", f"Movement duration is partly predictable, best at n={best_move} with mean R2={model.loc[best_move,'movement_time_s_r2']:.2f}; initiation time is weaker."],
                  ["Are interpretable patterns present?", f"The minimum-jerk rule assigns {100*component_rates.get(1,0):.1f}% single-component and {100*(1-component_rates.get(1,0)):.1f}% multi-component trials."],
                  ["Do fingerprints predict unseen-subject summaries?", f"Only {low_dim_positive}/{low_dim_total} low-dimensional probe/model combinations have positive test R2; most summary targets do not transfer reliably."],
                  ["Do 2-3 controls reproduce distributions?", f"They capture some structure, but not every distribution. n=8 has the best continuous fidelity; component-count JSD is similarly low at n=2 ({gen.loc[2,'mean_count_jsd']:.3f}) and n={best_count} ({gen.loc[best_count,'mean_count_jsd']:.3f})."],
              ], [45 * mm, 125 * mm], 8.0), PageBreak()]

    story += [P("1. Experiment and data", "h1"),
              P("Participants moved a tracked finger forward on a table to intercept a visual target entering a goal region. The task required rapid movement after the target began moving. We analyze condition 2 only and condition the model on the start-position/speed category, starting side, and exact target speed derived from the executed 60 Hz MAT dotArray."),
              table([
                  ["Item", "Study value"], ["Retained data", f"{len(trials):,} trials from {len(set(t['metadata']['subject'] for t in trials))} participants"],
                  ["Sampling", "Finger position at 240 Hz; target trajectory at 60 Hz"], ["Trajectory", "x lateral, y forward, z vertical; tracker coordinate unit not assumed to be mm"],
                  ["Timing", "Frame counter used; receipt-time column ignored"], ["Filtering", "Fourth-order Butterworth low-pass, 10 Hz"],
                  ["Events", "Target motion (go), finger movement onset, recorded arrival"],
              ], [45 * mm, 125 * mm]),
              Image(str(FIG_DIR / "data_geometry.png"), width=165 * mm, height=71 * mm),
              P(f"The x-y plane is used by the final baseline, CVAE, and submovement model. The y coordinate carries the dominant forward reach; z is treated as off-plane variation. In a {stimulus_protocol['external_audit_n']}-trial audit, {100*stimulus_protocol['external_speed_mismatch_gt_0.005_rate']:.1f}% of external stimulus files differed materially from the executed MAT target, so MAT dotArray is the authoritative condition source."), PageBreak()]

    story += [P("2. Deep-learning formulation", "h1"),
              P("Every movement-onset-to-arrival trajectory is translated to start at zero and resampled to 100 phase points. Temporal normalization makes input dimensions equal but removes physical duration. The decoder therefore generates movement duration and initiation time as separate positive log-time outputs."),
              Image(str(FIG_DIR / "pipeline.png"), width=166 * mm, height=55 * mm),
              P("Primary model", "h2"),
              P("A trajectory-only conditional VAE receives 200 table-plane coordinates (100 phase points x 2 axes) and the task condition, compresses them to n=2, n=3, or n=8, and reconstructs trajectory and timing. True timing is withheld from the encoder, so timing performance measures inference from trajectory shape rather than copying an input."),
              P("Held-out protocol", "h2"),
              table([
                  ["Stage", "Information allowed"], ["Training", "17 participants"], ["Hyperparameter validation", "4 separate participants"],
                  ["Final test", "7 untouched participants"], ["Fingerprint enrollment", "Context half of each test participant's trials"],
                  ["Evaluation", "Disjoint query half; stratified by task condition"], ["Repetition", "Three neural-network initialization seeds"],
              ], [48 * mm, 122 * mm]), PageBreak()]

    story += [P("3. Baselines", "h1"),
              Image(str(FIG_DIR / "baselines.png"), width=165 * mm, height=72 * mm),
              P(f"With k fixed to the known 28 participants, K-Means gives trajectory ARI={kmeans.loc[kmeans.representation=='trajectory','ari'].iloc[0]:.3f} and feature ARI={kmeans.loc[kmeans.representation=='kinematic_features','ari'].iloc[0]:.3f}. Both exceed a 200-permutation null, but the small ARI shows that trials do not form clean natural subject clusters."),
              P("Spline/PCA is a strong deterministic compression baseline. It is more accurate at high dimension, whereas the CVAE is competitive at low dimension while also learning a continuous probabilistic latent and generating timing. Therefore reconstruction MSE is one criterion, not the complete research objective."), PageBreak()]

    story += [P("4. CVAE performance", "h1"),
              Image(str(FIG_DIR / "deep_results.png"), width=166 * mm, height=68 * mm),
              P("Movement duration is recoverable from trajectory shape to a useful but incomplete degree. Initiation time occurs before the modeled movement and is consequently harder to infer. This difference is expected and demonstrates why the two timing variables are reported separately."),
              table([["n", "Trajectory MSE", "Movement-time R2", "Initiation-time R2", "Fingerprint ID"], *[
                  [int(n), f"{row.reconstruction_mse_tracker_units2:.3f}", f"{row.movement_time_s_r2:.2f}", f"{row.initiation_time_s_r2:.2f}", f"{100*row.fingerprint_balanced_accuracy:.1f}%"]
                  for n, row in model.iterrows()
              ]], [18 * mm, 38 * mm, 38 * mm, 38 * mm, 38 * mm]),
              P("Fingerprint identification is closed-set enrollment: the model sees context trials from each new participant and identifies their query trials. It demonstrates stable transferable identity information, not zero-shot identification of a person with no recorded movement."), PageBreak()]

    story += [P("5. Minimum-jerk behavioral interpretation", "h1"),
              P("Recorded and generated x-y velocity profiles are decomposed into a sum of one to four minimum-jerk components following Prof. Friedman's repository. Each component has onset, duration, lateral displacement, and forward displacement. The count is the smallest model reaching normalized error <=0.05, then <0.10, otherwise the minimum-error model."),
              P(f"A BIC sensitivity check selected the maximum four-component model for {100*bic_four_rate:.1f}% of trials. Because filtered velocity residuals are temporally correlated and violate the independent-error assumption behind that penalty, BIC is not used as the primary order rule; this remains an explicit validation question for Prof. Friedman."),
              Image(str(FIG_DIR / "submovement_examples.png"), width=166 * mm, height=65 * mm),
              table([
                  ["Derived variable", "Interpretation"], ["Component count", "Effective number of minimum-jerk primitives needed for the velocity profile"],
                  ["Secondary amplitude fraction", "Share of fitted displacement assigned beyond the first component"],
                  ["Overlap", "Temporal overlap between consecutive components"], ["Relative onset", "Next onset relative to the previous component duration"],
                  ["Fit error", "Normalized residual velocity energy"],
              ], [50 * mm, 120 * mm]),
              P("These are kinematic patterns consistent with single or corrective movement organization. They are not direct measurements of a cognitive decision strategy."), PageBreak()]

    story += [P("6. Submovement structure in recorded trials", "h1"),
              Image(str(FIG_DIR / "submovement_population.png"), width=166 * mm, height=67 * mm),
              P(f"Median normalized fit error is {valid.mj_fit_error.median():.3f}. The eight-restart stability audit reproduces the selected component count in {stability_text} of audited trials. Most multi-component profiles overlap, which is compatible with online correction but does not by itself identify feedback control."),
              P("The recorded outcome label 'Not fixating on the dot enough!!!' is kept as its own category. It is not silently combined with ordinary failure because its meaning in free-eye condition 2 requires confirmation."), PageBreak()]

    story += [P("7. What the learned latent represents", "h1"),
              Image(str(FIG_DIR / "behavior_probes.png"), width=166 * mm, height=74 * mm),
              P("Subject-level probes use only context latent means and predict distribution summaries calculated from disjoint query trials. Trial-level probes add the known task condition and ask whether individual latent codes contain component-count or recorded-outcome information."),
              P(f"Positive R2 indicates transfer to previously unseen participants; negative R2 means the probe is worse than a training/validation mean prediction. Across n=2 and n=3, only {low_dim_positive} of {low_dim_total} probe/model combinations are positive. The latent clearly carries identity and some trial behavior, but a general two- or three-number map to every subject distribution is not established."), PageBreak()]

    story += [P("8. Generated query distributions", "h1"),
              P("For each test participant, the context-mean latent is the only personal fingerprint. Samples add one shared latent-noise covariance estimated from training subjects, combine it with the query condition mixture, and decode. This preserves the requested n personal controls without quietly adding subject-specific variance parameters."),
              Image(str(FIG_DIR / "generation_fidelity.png"), width=166 * mm, height=72 * mm),
              P("Continuous variables are compared with KS statistics, KS p-values corrected within each subject by false-discovery rate (FDR), and Wasserstein distances. Component count is discrete, so total-variation distance and Jensen-Shannon divergence are used instead of applying a continuous KS p-value."),
              P(f"Across the 11 continuous features, the generated and recorded distributions are rejected as exactly equal for a mean of {fdr_rejections.loc[2]:.1f} features at n=2, {fdr_rejections.loc[3]:.1f} at n=3, and {fdr_rejections.loc[8]:.1f} at n=8. The model therefore achieves partial similarity, not exact distributional reproduction."),
              P("A value near zero indicates closer distributional agreement. The purpose is not to select a model from the test set; n=2/3 are the scientific target and n=8 is the pre-specified capacity comparison."), PageBreak()]

    story += [P("9. Controlling the latent variables", "h1"),
              P("The pre-specified seed-42 n=2 decoder is evaluated on a grid spanning -2 to +2 training-set standard deviations around the population latent center while holding task condition fixed. This directly tests the requested use case: change two controls and observe the generated movement."),
              Image(str(FIG_DIR / "latent_traversal.png"), width=166 * mm, height=65 * mm),
              P("A valid within-model interpretation requires smooth changes and plausible trajectories. VAE axes may rotate, reflect, or permute across initialization seeds, so this plot is descriptive for one pre-specified decoder; it does not assign a universal or causal human trait to latent 1 or latent 2."), PageBreak()]

    story += [P("10. Conclusions", "h1"),
              P("What the study supports", "h2"),
              P("- Interception trajectories carry participant-specific information that transfers after enrollment.<br/>- A CVAE can jointly represent phase-normalized shape and generate the timing removed by normalization.<br/>- Minimum-jerk decomposition provides interpretable measurements for both recorded and generated movements.<br/>- The context/query design allows a subject fingerprint to be evaluated without reusing the same trials as targets."),
              P("What remains limited", "h2"),
              P("- Natural unsupervised subject clusters are weak.<br/>- Initiation timing is substantially harder to infer than movement duration.<br/>- Two or three latent controls do not reproduce every behavioral distribution equally well.<br/>- Outcome interpretation is limited by the condition-2 not-fixating label.<br/>- The data contain only 28 participants with condition-2 trials, so subject-level generalization has wide uncertainty."),
              P("Overall interpretation", "h2"),
              P("The project demonstrates a complete deep generative pipeline from raw trajectories to low-dimensional fingerprints, held-out subject enrollment, trajectory/timing generation, and behavioral distribution validation. The strongest conclusion is that individual information is learnable; the stronger claim of a universally sufficient two-parameter human model is not yet established."), PageBreak()]

    story += [P("11. Assumptions for Prof. Friedman", "h1"),
              P("The analysis was completed under explicit, rerunnable assumptions. Feedback on any item can be applied through configuration followed by a full reproducible rerun."),
              table([
                  ["Assumption", "Decision used"], ["Task plane", "x-y table plane; z treated as off-plane variation"],
                  ["Submovement timing", "100 ms minimum component duration; 50 ms minimum onset spacing"],
                  ["Model order", "Smallest k reaching 0.05 normalized error, 0.10 fallback"],
                  ["Eye-fixation label", "Retained and reported separately in condition 2"],
                  ["Position unit", "Tracker units; no unverified conversion to millimetres"],
                  ["Executed target", "Paired MAT dotArray used when the external stimulus CSV differs; speed scaled by a 1920-pixel screen width"],
                  ["Late arrivals", "Arrival more than one second after the target window excluded"],
                  ["Participant count", "28 participants have retained condition-2 trials"],
              ], [52 * mm, 118 * mm]),
              P("References", "h2"),
              P("Brenner, E. & Smeets, J. B. J Neurophysiol (2018), doi:10.1152/jn.00517.2018.<br/>Flash, T. & Hogan, N. J Neurosci (1985), minimum-jerk movement model.<br/>Friedman, J. submovements repository, commit 9c2f40c, github.com/JasonFriedman/submovements.<br/>Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes, ICLR (2014).<br/>Sohn, K., Lee, H. & Yan, X. Conditional deep generative models, NeurIPS (2015).<br/>Slowinski, P. et al. Dynamic similarity and individual motor signatures, J R Soc Interface (2016), doi:10.1098/rsif.2015.1093."),
              P("All numerical results and figures are generated by the repository scripts from saved final-study artifacts. The PDF contains no manually entered result values.", "small")]

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#607582"))
        canvas.drawString(20 * mm, 11 * mm, "Low-dimensional interception movement fingerprints")
        canvas.drawRightString(190 * mm, 11 * mm, f"Page {doc.page}"); canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=17 * mm, bottomMargin=18 * mm,
                            title="Learning Low-Dimensional Movement Fingerprints")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    data = load_data()
    make_figures(*data[:-1])
    build(*data)
    print(OUT)


if __name__ == "__main__":
    main()
