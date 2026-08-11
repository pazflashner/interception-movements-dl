"""Build the advisor-facing PDF for the two-window CVAE study."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.preprocessing import lowpass_filter
from src.submovements import minimum_jerk_velocity, reconstruct_velocity


OUT_DIR = config.STUDY_ROOT / "output" / "pdf"
FIG_DIR = OUT_DIR / "figures"
OUT = OUT_DIR / "Interception_Strategy_Window_Comparison.pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#17324D")
BLUE = "#2563EB"
RED = "#DC2626"
TEAL = "#0F766E"
ORANGE = "#D97706"
LIGHT = colors.HexColor("#EEF3F6")
MID = colors.HexColor("#CCD8DF")
TEXT = colors.HexColor("#26343E")
WINDOW_LABEL = {
    "movement_only": "Movement onset -> arrival",
    "go_to_arrival": "Target motion onset -> arrival",
}


def load_data() -> dict:
    with (config.DATA_PROCESSED_DIR / "canonical_trials.pkl").open("rb") as handle:
        trials = pickle.load(handle)
    results = pd.read_csv(config.RESULTS_DIR / "model_seed_results.csv")
    sub = pd.read_csv(config.RESULTS_DIR / "submovements_real.csv")
    audit = pd.read_csv(config.STUDY_ROOT / "data_audit" / "condition2_trial_completion_audit.csv")
    associations_path = config.RESULTS_DIR / "latent_associations" / "latent_submovement_associations.csv"
    associations = pd.read_csv(associations_path) if associations_path.exists() else pd.DataFrame()
    baselines = {}
    for mode in config.WINDOW_MODES:
        base = config.RESULTS_DIR / mode / "baselines"
        baselines[mode] = {
            "json": json.loads((base / "baselines.json").read_text(encoding="utf-8")),
            "kmeans": pd.read_csv(base / "kmeans_selection_corrected.csv"),
        }
    generation = {}
    for mode in config.WINDOW_MODES:
        path = config.RESULTS_DIR / mode / "generation" / "generation_summary.csv"
        generation[mode] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return {
        "trials": trials,
        "results": results,
        "sub": sub,
        "audit": audit,
        "associations": associations,
        "baselines": baselines,
        "generation": generation,
    }


def savefig(name: str) -> Path:
    plt.tight_layout()
    path = FIG_DIR / name
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()
    return path


def model_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "movement_reconstruction_mse_tracker_units2",
        "movement_time_s_r2",
        "initiation_time_s_r2",
        "fingerprint_balanced_accuracy",
        "mean_ks",
    ]
    return results.groupby(["window_mode", "latent_dim"])[metrics].agg(["mean", "std"])


def make_figures(data: dict) -> dict[str, Path]:
    trials = data["trials"]
    results = data["results"]
    audit = data["audit"]
    sub = data["sub"]
    figures: dict[str, Path] = {}

    # Temporal-window illustration from one representative retained trial.
    def initiation_seconds(trial: dict) -> float:
        frames = np.asarray(trial["frame_values"], dtype=float)
        return float(
            (frames[trial["move_start_idx"]] - frames[trial["go_signal_idx"]])
            / config.RECORDING_HZ
        )

    sample = min(trials, key=lambda trial: abs(initiation_seconds(trial) - 0.20))
    go = np.asarray(sample["pos_go_to_arrival_norm"])[:, :2]
    movement = np.asarray(sample["pos_movement_norm"])[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    axes[0].plot(go[:, 0], go[:, 1], color=RED, lw=2.1)
    frames = np.asarray(sample["frame_values"], dtype=float)
    initiation = initiation_seconds(sample)
    arrival_after_go = float(
        (frames[sample["move_end_idx"]] - frames[sample["go_signal_idx"]])
        / config.RECORDING_HZ
    )
    onset = int(round(initiation / max(arrival_after_go, 1e-6) * (len(go) - 1)))
    onset = int(np.clip(onset, 0, len(go) - 1))
    axes[0].plot(go[: onset + 1, 0], go[: onset + 1, 1], color="#94A3B8", lw=3, label="wait")
    axes[0].scatter(*go[0], color=BLUE, s=45, label="go")
    axes[0].scatter(*go[onset], color=ORANGE, s=45, label="movement onset")
    axes[0].scatter(*go[-1], color="#991B1B", s=45, label="arrival")
    axes[0].set_title("Strategy-inclusive window")
    axes[1].plot(movement[:, 0], movement[:, 1], color=BLUE, lw=2.1)
    axes[1].scatter(*movement[0], color=ORANGE, s=45, label="movement onset")
    axes[1].scatter(*movement[-1], color="#991B1B", s=45, label="arrival")
    axes[1].set_title("Execution-only window")
    for ax in axes:
        ax.set_xlabel("Lateral x")
        ax.set_ylabel("Forward y")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=.2)
    figures["windows"] = savefig("temporal_windows.png")

    # Trial audit and retained cohort.
    label_counts = audit.responseText.fillna("missing").value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35))
    labels = [label.replace("Not fixating on the dot enough!!!", "Not-fixating flag") for label in label_counts.index]
    axes[0].barh(labels[::-1], label_counts.values[::-1], color=[TEAL, ORANGE, RED, BLUE, "#64748B"][:len(labels)][::-1])
    axes[0].set_xlabel("Condition-2 trials")
    axes[0].set_title("Recorded outcome labels")
    axes[0].grid(axis="x", alpha=.2)
    exclusions = pd.Series({"Retained": len(trials), "Late >1 s": 27, "No arrival": 4})
    axes[1].bar(exclusions.index, exclusions.values, color=[TEAL, ORANGE, RED])
    axes[1].set_ylabel("Trials")
    axes[1].set_title("Canonical cohort decision")
    axes[1].grid(axis="y", alpha=.2)
    figures["audit"] = savefig("trial_audit.png")

    # Baselines: K-Means and fair learned compression.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    x = np.arange(2)
    width = .34
    for offset, mode in zip((-.17, .17), config.WINDOW_MODES):
        km = data["baselines"][mode]["kmeans"].set_index("representation")
        axes[0].bar(x + offset, [km.loc["trajectory", "ari"], km.loc["kinematic_features", "ari"]], width, label=WINDOW_LABEL[mode])
    axes[0].set_xticks(x, ["Trajectory", "Kinematic features"])
    axes[0].set_ylabel("Adjusted Rand Index")
    axes[0].set_title("K-Means, k fixed to 28")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(axis="y", alpha=.2)
    for mode, color in (("movement_only", BLUE), ("go_to_arrival", RED)):
        spline = pd.Series(data["baselines"][mode]["json"]["spline_pca"], dtype=float)
        neural = results[results.window_mode == mode].groupby("latent_dim").window_reconstruction_mse_tracker_units2.mean()
        axes[1].plot(spline.index.astype(int), spline.values, "o--", color=color, alpha=.65, label=f"Spline+PCA, {WINDOW_LABEL[mode]}")
        axes[1].plot(neural.index, neural.values, "o-", color=color, label=f"CVAE, {WINDOW_LABEL[mode]}")
    axes[1].set_yscale("log")
    axes[1].set_xticks([2, 3, 4, 8])
    axes[1].set_xlabel("Latent dimensions")
    axes[1].set_ylabel("Held-out window MSE")
    axes[1].set_title("Learned compression comparison")
    axes[1].legend(frameon=False, fontsize=6)
    axes[1].grid(alpha=.2)
    figures["baselines"] = savefig("baselines.png")

    # Repeated-seed model comparison.
    metrics = [
        ("initiation_time_s_r2", "Initiation-time R2"),
        ("movement_time_s_r2", "Movement-time R2"),
        ("fingerprint_balanced_accuracy", "Fingerprint balanced accuracy"),
        ("mean_ks", "Mean generated-distribution KS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for mode, color in (("movement_only", BLUE), ("go_to_arrival", RED)):
            group = results[results.window_mode == mode].groupby("latent_dim")[metric].agg(["mean", "std"])
            ax.errorbar(group.index, group["mean"], yerr=group["std"], marker="o", lw=2, capsize=3, color=color, label=WINDOW_LABEL[mode])
        if metric.endswith("_r2"):
            ax.axhline(0, color="#64748B", ls="--", lw=1)
        if metric == "fingerprint_balanced_accuracy":
            ax.axhline(1 / 7, color="#64748B", ls="--", lw=1, label="chance")
        ax.set_xticks([2, 3, 4, 8])
        ax.set_xlabel("Latent dimensions")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    figures["models"] = savefig("model_comparison.png")

    # Recorded minimum-jerk distributions.
    valid = sub[sub.mj_fit_success == True].copy()  # noqa: E712
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.35))
    counts = valid.mj_n_components.value_counts(normalize=True).sort_index()
    axes[0].bar(counts.index, 100 * counts.values, color=BLUE)
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_ylabel("Trials (%)")
    axes[0].set_title("Component count")
    subject_single = valid.assign(single=valid.mj_n_components == 1).groupby("subject").single.mean()
    axes[1].hist(100 * subject_single, bins=10, color=TEAL, edgecolor="white")
    axes[1].set_xlabel("Single-component trials (%)")
    axes[1].set_title("Between-participant variation")
    axes[2].scatter(valid.mj_secondary_amplitude_fraction, valid.mj_mean_overlap_pct, s=5, alpha=.16, color=ORANGE)
    axes[2].set_xlabel("Secondary amplitude fraction")
    axes[2].set_ylabel("Mean overlap (%)")
    axes[2].set_title("Multi-component organization")
    figures["submovements"] = savefig("submovement_population.png")

    # One physical decomposition example for each selected order.
    by_id = {trial["metadata"]["trial_id"]: trial for trial in trials}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), sharey=True)
    for ax, count in zip(axes, (1, 2, 3)):
        candidates = valid[valid.mj_n_components == count]
        row = candidates.iloc[(candidates.mj_fit_error - candidates.mj_fit_error.median()).abs().argmin()]
        trial = by_id[row.trial_id]
        position = np.asarray(trial["pos_filtered"])[trial["move_start_idx"] : trial["move_end_idx"] + 1, :2]
        position = lowpass_filter(position, cutoff=10, fs=240)
        time = np.arange(len(position)) / 240
        velocity = np.gradient(position, 1 / 240, axis=0)
        params = np.asarray(json.loads(row.mj_parameters_json))
        reconstructed = reconstruct_velocity(time, params)
        ax.plot(time, np.linalg.norm(velocity, axis=1), color="#111827", lw=1.8, label="recorded")
        ax.plot(time, np.linalg.norm(reconstructed, axis=1), color=RED, lw=1.5, label="sum")
        for index, component in enumerate(params):
            profile = minimum_jerk_velocity(time, component[0], component[1], component[2:4])
            ax.plot(time, np.linalg.norm(profile, axis=1), ls="--", lw=1, label=f"component {index + 1}")
        ax.set_title(f"{count} component{'s' if count > 1 else ''}")
        ax.set_xlabel("Movement time (s)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Speed (tracker units/s)")
    axes[-1].legend(frameon=False, fontsize=6)
    figures["sub_examples"] = savefig("submovement_examples.png")
    return figures


def styles() -> dict[str, ParagraphStyle]:
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


def report_table(rows: list[list[object]], widths: list[float], font: float = 8) -> Table:
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=font, leading=font + 2)
    wrapped = [[cell if isinstance(cell, Paragraph) else Paragraph(str(cell), cell_style) for cell in row] for row in rows]
    table = Table(wrapped, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .35, MID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build(data: dict, figures: dict[str, Path]) -> None:
    s = styles()
    P = lambda text, style="body": Paragraph(text, s[style])
    story: list = []
    results = data["results"]
    grouped = results.groupby(["window_mode", "latent_dim"]).mean(numeric_only=True)
    valid = data["sub"][data["sub"].mj_fit_success == True]  # noqa: E712
    counts = valid.mj_n_components.value_counts(normalize=True)
    associations = data["associations"]
    significant = int(associations.fdr_reject_0_05.sum()) if len(associations) else 0

    go3 = grouped.loc[("go_to_arrival", 3)]
    go8 = grouped.loc[("go_to_arrival", 8)]
    movement8 = grouped.loc[("movement_only", 8)]
    story += [
        Spacer(1, 15 * mm),
        P("Learning Interception-Movement Fingerprints", "title"),
        P("A conditional variational autoencoder comparison of execution-only and strategy-inclusive trajectories", "subtitle"),
        P("Seman Libbiss and Paz Flashner | Deep Learning Workshop | Prof. Jason Friedman | Advisor: Moni", "subtitle"),
        Spacer(1, 3 * mm),
        P("Research objective", "h1"),
        P("Can a small continuous latent code learned from finger trajectories represent participant-specific interception behavior and generate held-out distributions of trajectory shape, initiation, movement time, and submovement structure?", "callout"),
        P("Main result", "h2"),
        P(f"The strategy-inclusive n=3 CVAE predicts held-out initiation time with mean R2={go3.initiation_time_s_r2:.2f} and movement time with R2={go3.movement_time_s_r2:.2f}. n=8 improves distribution fidelity and timing, but is treated as a capacity comparator rather than an interpretable fingerprint."),
        report_table([
            ["Question", "Finding"],
            ["Does waiting belong in the modeled trajectory?", "Yes for strategy timing: the target-motion window makes initiation time recoverable at n=3/4/8. It does not improve every metric."],
            ["Is participant information learnable?", f"Yes after enrollment. Best held-out balanced accuracy is {100 * movement8.fingerprint_balanced_accuracy:.1f}% versus 14.3% chance; unsupervised K-Means remains weak."],
            ["Do 2-3 controls solve the full goal?", f"Partly. n=3 is viable for strategy timing, but n=8 has stronger overall fidelity (go-window mean KS={go8.mean_ks:.3f})."],
            ["Are submovements cognitive strategies?", "No. They are interpretable kinematic components that can support, but not prove, a strategy interpretation."],
        ], [48 * mm, 122 * mm]),
        PageBreak(),
    ]

    story += [
        P("1. Experiment, events, and trial audit", "h1"),
        P("Participants moved a tracked finger forward on a table to intercept a visual target entering a goal region. We analyze condition 2 (free eye movements). Finger position is sampled at 240 Hz; the executed target trajectory is stored at 60 Hz. Timing uses the frame counter rather than the receipt-time column."),
        Image(str(figures["audit"]), width=165 * mm, height=67 * mm),
        report_table([
            ["Audit item", "Verified result"],
            ["Raw condition-2 CSV/MAT pairs", f"{len(data['audit']):,} trials from {data['audit'].subject.nunique()} participants"],
            ["Arrival at/before target motion onset", "0"],
            ["Recording ended at/before target motion onset", "0"],
            ["Too early labels", "48; every one arrived after target motion began and is retained"],
            ["Final canonical cohort", f"{len(data['trials']):,}; 27 late arrivals (>1 s after target window) and 4 no-arrival timeouts excluded"],
        ], [58 * mm, 112 * mm]),
        P("The recorded label Too early means arrival before the interception window, not movement before the target began. Treating those trials as premature departures would remove valid strategy information."),
        PageBreak(),
    ]

    story += [
        P("2. Two temporal representations", "h1"),
        P("The scientific disagreement about normalization is tested directly rather than decided by wording. Both protocols use exactly the same retained trials, filtering, axes, conditions, participant split, latent dimensions, training seeds, and evaluation."),
        Image(str(figures["windows"]), width=165 * mm, height=65 * mm),
        report_table([
            ["Protocol", "Information retained", "Primary interpretation"],
            [WINDOW_LABEL["movement_only"], "Execution shape and corrections after detected movement onset", "Execution fingerprint"],
            [WINDOW_LABEL["go_to_arrival"], "Waiting, initiation, execution, and corrections", "Strategy timing plus execution"],
        ], [48 * mm, 72 * mm, 50 * mm]),
        P("Each 2-D trajectory is translated to start at zero and resampled to 100 phase points. This removes physical time from the encoder input, so the decoder separately predicts movement time and initiation time. For the strategy window, predicted initiation time is also used to locate movement onset inside a generated trajectory before physical movement features are computed."),
        PageBreak(),
    ]

    story += [
        P("3. Deep-learning formulation and held-out protocol", "h1"),
        P("A conditional variational autoencoder receives 200 trajectory values (100 phase points x 2 axes) plus the start/speed category, starting side, and exact executed target speed. The encoder produces a Gaussian latent distribution; the decoder reconstructs the trajectory and predicts the two positive timing variables. KL divergence regularizes the latent space so nearby coordinates support continuous sampling."),
        report_table([
            ["Stage", "Design"],
            ["Training", "17 participants; subject-balanced batches"],
            ["Hyperparameter validation", "4 separate participants"],
            ["Final test", "7 untouched participants"],
            ["Fingerprint enrollment", "Mean latent code from a context half of each held-out participant's trials"],
            ["Fingerprint test", "Disjoint, condition-stratified query half"],
            ["Latent sweep", "n=2 and n=3 are interpretable targets; n=4 transition; n=8 capacity comparator"],
            ["Repetition", "Seeds 42, 43, and 44 for every protocol/dimension combination"],
        ], [52 * mm, 118 * mm]),
        P("A participant fingerprint is therefore not the code of one trial. It is an enrollment estimate from repeated trials. Averaging is meaningful inside one trained VAE because the KL-regularized latent space is continuous; axes are not compared directly across independently trained seeds because they may rotate, reflect, or swap."),
        PageBreak(),
    ]

    story += [
        P("4. Classical baselines", "h1"),
        Image(str(figures["baselines"]), width=165 * mm, height=69 * mm),
        P("K-Means uses the pre-specified 28-participant count. Both trajectory and kinematic-feature ARI values exceed a 200-permutation null, but ARI near 0.05-0.07 is weak in absolute terms. This supports participant signal without claiming clean natural clusters."),
        P("The very small error of a spline fitted directly to each test trial is descriptive, not a generalization result: it uses the test trajectory itself to choose its coefficients. The fair compression baseline is spline coefficients learned per trial followed by PCA fitted only on training participants. CVAE and spline+PCA answer different questions; reconstruction alone does not measure probabilistic generation, timing prediction, or participant enrollment."),
        PageBreak(),
    ]

    story += [
        P("5. Repeated-seed CVAE results", "h1"),
        Image(str(figures["models"]), width=165 * mm, height=122 * mm),
        report_table([
            ["Conclusion", "Evidence"],
            ["n=2 is too narrow", "Initiation-time prediction is unstable and strongly negative in both protocols."],
            ["n=3 is the lowest stable strategy model", f"Go-window initiation R2={go3.initiation_time_s_r2:.2f}, movement-time R2={go3.movement_time_s_r2:.2f}, fingerprint ID={100*go3.fingerprint_balanced_accuracy:.1f}%."],
            ["n=8 has strongest capacity", f"Go-window mean KS={go8.mean_ks:.3f}; movement-only fingerprint ID={100*movement8.fingerprint_balanced_accuracy:.1f}%."],
            ["Neither window dominates", "Strategy-inclusive improves initiation timing and usually distribution KS; movement-only improves execution reconstruction and participant identification."],
        ], [55 * mm, 115 * mm]),
        PageBreak(),
    ]

    story += [
        P("6. What fingerprint identification means", "h1"),
        P("For each held-out participant, the model averages latent codes from context trials, then assigns each disjoint query trial to the closest enrolled participant fingerprint. Chance is 1/7=14.3%. This is closed-set enrollment, not zero-shot recognition of a participant whose movements were never observed."),
        report_table([
            ["Latent n", "Movement-only ID", "Strategy-window ID"],
            *[[n, f"{100*grouped.loc[('movement_only', n)].fingerprint_balanced_accuracy:.1f}%", f"{100*grouped.loc[('go_to_arrival', n)].fingerprint_balanced_accuracy:.1f}%"] for n in [2, 3, 4, 8]],
        ], [35 * mm, 67 * mm, 68 * mm]),
        P("The result shows repeatable participant information in trajectory latents, while weak K-Means shows that the full trial population does not separate into 28 clean unsupervised clusters. Both statements can be true: enrollment uses known context examples for each test participant; K-Means does not."),
        PageBreak(),
    ]

    story += [
        P("7. Minimum-jerk submovement interpretation", "h1"),
        P("Following Prof. Friedman's repository, each physical movement velocity is approximated by one to four overlapping 2-D minimum-jerk components. Each component has onset time, duration, lateral displacement, and forward displacement. The selected count is the smallest model reaching normalized error <=0.05, then <=0.10, otherwise the minimum-error candidate."),
        Image(str(figures["sub_examples"]), width=165 * mm, height=64 * mm),
        report_table([
            ["Measure", "Physical meaning"],
            ["Component count", "Number of smooth velocity primitives needed by the selected approximation"],
            ["First duration", "Duration of the first fitted primitive"],
            ["First amplitude", "Magnitude of its 2-D displacement vector"],
            ["Secondary amplitude fraction", "Share of total fitted displacement assigned after the first component"],
            ["Mean overlap", "Temporal overlap between consecutive fitted components"],
            ["Fit error", "Residual velocity energy normalized by recorded velocity energy"],
        ], [52 * mm, 118 * mm]),
        P("These variables can describe single, overlapping, or sequential movement organization. They do not by themselves identify hesitation, regret, feedforward control, or feedback control."),
        PageBreak(),
    ]

    story += [
        P("8. Recorded submovement structure", "h1"),
        Image(str(figures["submovements"]), width=165 * mm, height=67 * mm),
        P(f"The deterministic fit succeeds for {100*data['sub'].mj_fit_success.mean():.1f}% of retained trials. The selected representation contains {100*counts.get(1, 0):.1f}% single-component and {100*(1-counts.get(1, 0)):.1f}% multi-component movements; median normalized fit error is {valid.mj_fit_error.median():.3f}. Participant-level variation motivates asking whether latent fingerprints capture these distributions."),
        P("Behavioral truth is always computed from detected physical movement onset to arrival, even when the CVAE input begins at target motion onset. The waiting interval therefore cannot be mistaken for a movement component."),
        PageBreak(),
    ]

    trial_heatmap = config.RESULTS_DIR / "latent_associations" / "heatmaps" / "go_to_arrival_z3_trial_within_subject_partial.png"
    subject_heatmap = config.RESULTS_DIR / "latent_associations" / "heatmaps" / "go_to_arrival_z3_subject_context_to_query.png"
    story += [
        P("9. Latent associations and interpretability", "h1"),
        P("Trial-level associations residualize participant, start category, side, and exact speed before computing Spearman correlations. Subject-level associations correlate context fingerprints with behavioral means from disjoint query trials. Benjamini-Hochberg false-discovery correction is applied within each model and level."),
    ]
    if trial_heatmap.exists():
        story.append(Image(str(trial_heatmap), width=158 * mm, height=90 * mm))
    if subject_heatmap.exists():
        story.append(Image(str(subject_heatmap), width=158 * mm, height=90 * mm))
    story += [
        P(f"Across all models, seeds, levels, and tested targets, {significant} associations survive the specified within-model FDR correction. These are associations, not causal effects. A slider changes a decoder coordinate, but a correlation does not prove that the coordinate is a universal human trait."),
        PageBreak(),
    ]

    story += [
        P("10. Generated distributions", "h1"),
        P("For each held-out participant, the context-mean latent is the only personal center. Generation samples from one covariance estimated only from training participants and combines it with the empirical query-condition mixture. Continuous variables use two-sample KS and Wasserstein distance; discrete component count uses total variation and Jensen-Shannon divergence."),
        report_table([
            ["Model", "Mean KS from full repeated-seed evaluation"],
            *[[f"{WINDOW_LABEL[mode]}, n={n}", f"{grouped.loc[(mode, n)].mean_ks:.3f}"] for mode in config.WINDOW_MODES for n in [3, 8]],
        ], [105 * mm, 65 * mm]),
        P("Lower KS indicates closer empirical and generated distributions. These values show partial reproduction rather than equality. The n=3 strategy model is the low-dimensional candidate; n=8 establishes how much fidelity can improve when the bottleneck is relaxed."),
        PageBreak(),
    ]

    story += [
        P("11. Conclusions and questions for Prof. Friedman", "h1"),
        P("Supported", "h2"),
        P("- Participant-specific information transfers after enrollment.<br/>- Waiting time should not be discarded when initiation strategy is a target.<br/>- A strategy-inclusive n=3 CVAE is a defensible low-dimensional model, while n=8 provides stronger capacity.<br/>- The decoder can generate trajectory and the timing removed by phase normalization.<br/>- Minimum-jerk decomposition provides interpretable measurements for recorded and generated movements."),
        P("Not established", "h2"),
        P("- Two coordinates do not capture the full behavioral distribution reliably.<br/>- Trial clusters are not clean natural participant groups.<br/>- Minimum-jerk component count is not a direct cognitive-strategy label.<br/>- Latent axes are not universal across independent training seeds.<br/>- The current data do not support strong zero-shot claims for unseen participants."),
        P("Validation questions", "h2"),
        report_table([
            ["Question", "Current rerunnable decision"],
            ["Task plane", "x-y table plane; z treated as off-plane noise"],
            ["Completion", "Non-empty MAT pressedTime indicates arrival at the interception site"],
            ["Too early", "Retained because all arrivals occur after target motion onset"],
            ["Late threshold", "Exclude arrival more than 1 s after target window"],
            ["Submovement constraints", "100 ms minimum duration; 50 ms minimum onset spacing"],
            ["Model order", "Smallest adequate normalized-error model; BIC retained as sensitivity output"],
            ["Units", "Tracker units until physical conversion is confirmed"],
        ], [55 * mm, 115 * mm]),
        P("References", "h2"),
        P("Brenner, E. and Smeets, J. B. J Neurophysiol (2018), doi:10.1152/jn.00517.2018.<br/>Flash, T. and Hogan, N. J Neurosci (1985), minimum-jerk movement model.<br/>Friedman, J. submovements repository, commit 9c2f40c, github.com/JasonFriedman/submovements.<br/>Kingma, D. P. and Welling, M. Auto-Encoding Variational Bayes, ICLR (2014).<br/>Sohn, K., Lee, H. and Yan, X. Conditional deep generative models, NeurIPS (2015).<br/>Slowinski, P. et al. Dynamic similarity and individual motor signatures, J R Soc Interface (2016), doi:10.1098/rsif.2015.1093."),
        P("All numerical claims and figures are generated from the current study artifacts. No raw Dropbox data are included in the PDF or dashboard bundle.", "small"),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#607582"))
        canvas.drawString(20 * mm, 11 * mm, "Interception movement strategy fingerprints")
        canvas.drawRightString(190 * mm, 11 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Learning Interception-Movement Fingerprints",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    data = load_data()
    figures = make_figures(data)
    build(data, figures)
    print(OUT)


if __name__ == "__main__":
    main()
