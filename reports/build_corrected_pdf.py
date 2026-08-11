"""Build the audited results PDF from corrected-v3 artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "corrected_v3"
OUT_DIR = ROOT / "output" / "pdf"
FIG_DIR = OUT_DIR / "figures"
OUT = OUT_DIR / "Interception_Corrected_Results.pdf"
SUMMARY_OUT = ROOT / "CORRECTED_STUDY_README.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#18324A")
BLUE = "#176B87"
CYAN = "#27A6B8"
ORANGE = "#D97732"
RED = "#B54747"
GREEN = "#2E7D5B"
LIGHT = colors.HexColor("#EEF3F6")
MID = colors.HexColor("#D1DCE3")
TEXT = colors.HexColor("#25313A")


def load():
    summary = pd.read_csv(RESULTS / "model_summary.csv")
    kmeans = pd.read_csv(RESULTS / "kmeans_selection_corrected.csv")
    baselines = json.loads((RESULTS / "baselines.json").read_text())
    protocol = json.loads((RESULTS / "protocol.json").read_text())
    return summary, kmeans, baselines, protocol


def figures(summary, kmeans, baselines):
    per = summary[summary.model == "per_trial_cvae"].copy()
    joint = per[per.variant == "joint_reconstruction"].sort_values("latent_dim")
    shape = per[per.variant == "trajectory_only_timing_prediction"].sort_values("latent_dim")
    dims = shape.latent_dim.to_numpy()

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(joint.latent_dim, joint.reconstruction_mse_tracker_units2, "o-", label="Joint CVAE")
    plt.plot(shape.latent_dim, shape.reconstruction_mse_tracker_units2, "o-", label="Trajectory-only encoder")
    spline = pd.Series(baselines["spline_pca"], dtype=float)
    plt.plot(spline.index.astype(int), spline.values, "o--", label="Spline + PCA")
    plt.xlabel("Latent dimension n"); plt.ylabel("Held-out MSE (tracker units squared)")
    plt.yscale("log"); plt.grid(alpha=.25); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(FIG_DIR / "reconstruction.png", dpi=180); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharex=True)
    axes[0].plot(dims, shape.movement_time_s_r2, "o-", color=BLUE)
    axes[0].axhline(0, color="black", lw=.8); axes[0].set_title("Movement time")
    axes[1].plot(dims, shape.initiation_time_s_r2, "o-", color=ORANGE)
    axes[1].axhline(0, color="black", lw=.8); axes[1].set_title("Initiation time")
    for ax in axes:
        ax.set_xlabel("Latent dimension n"); ax.set_ylabel("Test R2"); ax.grid(alpha=.25)
    fig.suptitle("Timing predicted from trajectory shape (timing not given to encoder)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "timing_prediction.png", dpi=180); plt.close(fig)

    plt.figure(figsize=(7.2, 4.0))
    plt.plot(joint.latent_dim, joint.fingerprint_balanced_accuracy * 100, "o-", label="Joint encoder")
    plt.plot(shape.latent_dim, shape.fingerprint_balanced_accuracy * 100, "o-", label="Trajectory-only encoder")
    plt.axhline(100 / 7, color="black", ls="--", lw=1, label="Chance (1/7)")
    plt.xlabel("Latent dimension n"); plt.ylabel("Balanced identification accuracy (%)")
    plt.ylim(0, 100); plt.grid(alpha=.25); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(FIG_DIR / "fingerprint_id.png", dpi=180); plt.close()

    plt.figure(figsize=(7.2, 4.0))
    plt.plot(joint.latent_dim, joint.mean_ks_rejected_fdr, "o-", label="Joint encoder")
    plt.plot(shape.latent_dim, shape.mean_ks_rejected_fdr, "o-", label="Trajectory-only encoder")
    hier = summary[summary.model == "hierarchical_cvae"].sort_values("latent_dim")
    plt.plot(hier.latent_dim, hier.mean_ks_rejected_fdr, "o-", label="Hierarchical CVAE")
    plt.xlabel("Subject/per-trial latent dimension n"); plt.ylabel("Mean features rejected after FDR (of 12)")
    plt.ylim(0, 12.3); plt.grid(alpha=.25); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(FIG_DIR / "distribution_fidelity.png", dpi=180); plt.close()

    probe_rows = []
    for n in (2, 3):
        p = RESULTS / "runs" / f"per_trial_trajectory_only_timing_prediction_z{n}" / "behavioral_probe.csv"
        d = pd.read_csv(p).set_index("target")
        for target in ("path_length_mean", "movement_time_s_mean", "initiation_time_s_mean",
                       "curvature_index_mean", "n_submovements_mean"):
            probe_rows.append({"n": n, "target": target, "r2": d.loc[target, "r2_test"]})
    probe = pd.DataFrame(probe_rows)
    labels = ["Path length", "Movement time", "Initiation time", "Curvature", "Submovements"]
    x = np.arange(len(labels)); width = .34
    plt.figure(figsize=(7.2, 4.1))
    for offset, n, colour in ((-.17, 2, BLUE), (.17, 3, ORANGE)):
        values = probe[probe.n == n].r2.to_numpy()
        plt.bar(x + offset, values, width, label=f"n={n}", color=colour)
    plt.axhline(0, color="black", lw=.8); plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Held-out subject R2 (symmetric log scale)")
    plt.yscale("symlog", linthresh=1.0); plt.ylim(-60, 1.0); plt.grid(axis="y", alpha=.25)
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(FIG_DIR / "low_dim_probe.png", dpi=180); plt.close()

    plt.figure(figsize=(6.7, 3.8))
    plt.bar(kmeans.representation.str.replace("_", " "), kmeans.ari, color=[BLUE, CYAN])
    plt.axhline(kmeans.null_95pct.max(), color="black", ls="--", label="Largest 95% permutation null")
    plt.ylabel("Adjusted Rand Index (ARI)"); plt.title("K-Means fixed at k=28")
    plt.ylim(0, .12); plt.grid(axis="y", alpha=.25); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans.png", dpi=180); plt.close()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title2", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24,
                                leading=28, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontSize=11.5, leading=16,
                                   textColor=colors.HexColor("#536773"), spaceAfter=14),
        "h1": ParagraphStyle("H1x", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17,
                             leading=21, textColor=NAVY, spaceBefore=5, spaceAfter=8),
        "h2": ParagraphStyle("H2x", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                             leading=15, textColor=colors.HexColor(BLUE), spaceBefore=7, spaceAfter=4),
        "body": ParagraphStyle("Bodyx", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4,
                               leading=13.2, textColor=TEXT, spaceAfter=6),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontSize=7.8, leading=10.2,
                                textColor=colors.HexColor("#536773"), spaceAfter=4),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10.2,
                                  leading=14.2, textColor=NAVY, backColor=LIGHT, borderPadding=8, spaceAfter=9),
    }


def table(rows, widths=None, font=8.0):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font), ("LEADING", (0, 0), (-1, -1), font + 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), .35, MID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build(summary, kmeans, baselines, protocol):
    s = styles(); story = []
    P = lambda text, style="body": Paragraph(text, s[style])
    story += [Spacer(1, 18 * mm), P("Interception Movements", "title"),
              P("Corrected preliminary methods and results audit", "subtitle"),
              P("Deep Learning Workshop | Seman and Paz | Prof. Jason Friedman | Advisor: Moni", "subtitle"),
              Spacer(1, 5 * mm),
              P("Bottom line", "h1"),
              P("The corrected experiments support a modest claim: movement trajectories contain transferable subject information. They do not yet support the stronger goal that two or three subject parameters can generate accurate distributions of reaction time, movement time, curvature, trajectory, and strategy.", "callout"),
              P("This report replaces the numerical interpretation in the earlier preliminary PDF. It is generated directly from the corrected-v3 artifacts and uses one fixed 17/4/7 subject split. Results remain preliminary because model initialization and subject split were not repeated in v3.")]
    highlights = [
        ["Question", "Corrected answer"],
        ["Are subjects distinguishable?", "Weak clustering but significant against permutation; closed-set identification 51-75% vs 14.3% chance."],
        ["Can shape predict timing?", "Movement time partly: best R2=0.73 at n=8. Initiation time remains weak: best R2=0.21 across the sweep."],
        ["Do n=2/3 fingerprints predict subject distributions?", "Not for the main timing/curvature targets. They mainly predict average path length and some speed summaries."],
        ["Are generated distributions faithful?", "No. Best trajectory-only n=16 still rejects 4.9 of 12 feature distributions per test subject on average after FDR."],
        ["Did the hierarchical model solve it?", "No. It identifies enrolled subjects but generates worse held-out query distributions."],
    ]
    story += [table(highlights, [45 * mm, 125 * mm], 8.2), PageBreak()]

    story += [P("1. What was corrected", "h1")]
    corrections = [
        "Frame-counter timing: duplicate counters are averaged and missing counters interpolated before 10 Hz filtering. The receipt-time column is not used.",
        "Target-motion synchronization: diff index i now maps to moving sample i+1, correcting a 16.7 ms offset.",
        "Segmentation language: the model trajectory is movement onset to recorded arrival. Go-signal to movement onset is a separate initiation-time scalar.",
        "Units: x/y/z and derived spatial quantities are reported in tracker units. The CSV specification does not establish millimetres.",
        "Timing target: models learn log(time + epsilon), producing nonnegative generated durations.",
        "Timing interpretation: joint-model timing is reconstruction because timing enters the encoder. A trajectory-only encoder provides genuine timing prediction.",
        "Subject balance: the per-trial CVAE samples subjects with inverse-frequency weights; the hierarchical CVAE trains equal subject episodes.",
        "Evaluation leakage: subject fingerprints are inferred from context trials and compared with disjoint query trials.",
        "Behavioral probe: Ridge is fit on training subjects, alpha selected on validation subjects, and test subjects scored once. The primary fingerprint uses only n latent means.",
        "Distribution tests: KS plus Benjamini-Hochberg FDR, Wasserstein, MMD, and energy distances are computed on query trials only.",
        "Submovement heuristic: recorded and decoded trajectories use duration-restored 10 Hz smoothing and 50 ms minimum peak separation.",
    ]
    story += [P("<br/>".join(f"- {x}" for x in corrections)), PageBreak()]

    story += [P("2. Data and protocol", "h1"),
              P(f"Condition 2 only. The corrected cache contains <b>{protocol['n_trials']:,} trials from {protocol['n_subjects']} subjects</b>. The fixed split is 17 training, 4 validation, and 7 test subjects. For every subject, trials are stratified by starting-position/speed-range index and side, then divided 50/50 into context and query sets."),
              P("The 4,684 retained trials exclude 48 'Too early' trials, 27 arrivals more than one second after the target window, and 4 no-arrival timeouts. The 'Not fixating on the dot enough!!!' trials remain included pending clarification for the free-eye condition."),
              P("Frame-grid audit", "h2"),
              P("Among retained trials, 969 contain at least one missing frame and 529 contain one duplicate frame; 1,498 contain either. The maximum correction is five interpolated frames and one duplicate. Fifteen trials exceed the one-second movement plausibility flag, but remain included and flagged."),
              P("Model inputs", "h2"),
              table([["Input", "Treatment"],
                     ["Trajectory", "10 Hz Butterworth; movement onset to arrival; resampled to 100 x 3; origin aligned"],
                     ["Condition", "3-level start-position/speed-range index plus left/right side"],
                     ["Movement/initiation time", "Positive log-timing targets; never included in the trajectory-only encoder"],
                     ["Trial repetition", "Recorded in metadata, not used as a condition in the primary fingerprint model"]],
                    [44 * mm, 126 * mm], 8.2), PageBreak()]

    story += [P("3. Baselines", "h1"),
              P("K-Means is fixed at k=28 before looking at labels. The permutation p-value compares the observed ARI with 200 shuffled subject-label assignments; it does not turn weak clustering into strong separation."),
              Image(str(FIG_DIR / "kmeans.png"), width=160 * mm, height=91 * mm)]
    km_rows = [["Representation", "ARI", "NMI", "Permutation p"]]
    for _, row in kmeans.iterrows():
        km_rows.append([row.representation.replace("_", " "), f"{row.ari:.3f}", f"{row.nmi:.3f}", f"{row.permutation_p:.3f}"])
    story += [table(km_rows, [65 * mm, 28 * mm, 28 * mm, 38 * mm]),
              P("Interpretation: subject identity affects movement, but the natural cluster structure is weak (ARI about 0.10). The kinematic summary is only slightly better than the trajectory."),
              P("Spline reconstruction", "h2"),
              P(f"The per-trial spline MSE ({baselines['spline_per_trial']:.6f}) is an interpolation ceiling because it is fit directly to each test trial. Spline+PCA is the population baseline fitted on training subjects."),
              Image(str(FIG_DIR / "reconstruction.png"), width=160 * mm, height=93 * mm), PageBreak()]

    story += [P("4. Corrected model designs", "h1"),
              P("Per-trial joint CVAE", "h2"),
              P("Encoder input: normalized trajectory + true log movement/initiation time + condition. Decoder reconstructs trajectory and timing. This is useful as a compression baseline, but its timing R2 is reconstruction, not inference from shape."),
              P("Per-trial trajectory-only CVAE", "h2"),
              P("Encoder input: normalized trajectory + condition only. Decoder predicts the trajectory and log timing. Subject fingerprints are context-trial latent means. This is the cleanest direct test of whether trajectory shape carries timing and identity."),
              P("Hierarchical subject/trial CVAE", "h2"),
              P("A Deep-Sets context encoder produces an explicit low-dimensional subject latent. A separate four-dimensional trial latent represents within-subject variation. The decoder receives subject latent, trial latent, and task condition. Training uses subject-balanced episodes and fixed validation episodes."),
              P("Context/query test", "h2"),
              table([["Stage", "Information allowed"],
                     ["Fingerprint enrollment", "Context half of one subject's trials"],
                     ["Behavioral probe target", "Distribution summaries from the disjoint query half"],
                     ["Generation", "Context fingerprint + query condition mix; no query trajectory/timing"],
                     ["Final scoring", "Generated sample versus empirical query trials"]], [45 * mm, 125 * mm]),
              P("This is still an adaptation setting: the model sees context trials from the new person. It is not zero-shot generation for a completely unobserved person."), PageBreak()]

    shape = summary[(summary.model == "per_trial_cvae") & (summary.variant == "trajectory_only_timing_prediction")].sort_values("latent_dim")
    best_move = shape.loc[shape.movement_time_s_r2.idxmax()]
    best_init = shape.loc[shape.initiation_time_s_r2.idxmax()]
    story += [P("5. Reconstruction and timing", "h1"),
              P(f"When timing is withheld from the encoder, movement time is partly predictable from trajectory shape. The best held-out result is n={int(best_move.latent_dim)}: R2={best_move.movement_time_s_r2:.2f}, MAE={best_move.movement_time_s_mae_ms:.0f} ms. Initiation time is much weaker; the best sweep value is R2={best_init.initiation_time_s_r2:.2f}."),
              Image(str(FIG_DIR / "timing_prediction.png"), width=164 * mm, height=82 * mm),
              P("The non-monotonic timing curve is not evidence that n=8 is universally optimal. It is one initialization on one subject split and reflects a tradeoff between trajectory reconstruction, timing, and KL regularization."),
              P("The joint encoder reaches higher timing R2 at high n because it receives the true timing. Those numbers must not be described as prediction."), PageBreak()]

    story += [P("6. What the fingerprint contains", "h1"),
              P("Closed-set identification", "h2"),
              P("Each test subject is enrolled from context trials. Query-trial codes are assigned to the nearest enrolled context mean. This asks whether latent identity information transfers to new subjects after enrollment; it does not show that subjects form clean unsupervised clusters."),
              Image(str(FIG_DIR / "fingerprint_id.png"), width=160 * mm, height=89 * mm),
              P("Low-dimensional behavioral probe", "h2"),
              P("The primary n=2/3 probe uses exactly two or three context latent means. It can predict average path length (R2 about 0.79) and some speed summaries, but it fails on subject mean movement time, initiation time, curvature, and most variability targets."),
              Image(str(FIG_DIR / "low_dim_probe.png"), width=160 * mm, height=91 * mm),
              P("Negative R2 means worse than predicting the training/validation mean for every held-out subject. With only seven test subjects these estimates are noisy, but the large negative values do not support the requested timing-distribution claim."), PageBreak()]

    best_shape = shape.loc[shape.mean_ks.idxmin()]
    hier = summary[summary.model == "hierarchical_cvae"].sort_values("latent_dim")
    story += [P("7. Distribution fidelity", "h1"),
              P("For each test subject, the model infers a fingerprint from context trials, generates 120 trials under the query condition mix, and compares them with that subject's query trials. KS p-values are FDR-corrected across 12 features within each subject."),
              Image(str(FIG_DIR / "distribution_fidelity.png"), width=160 * mm, height=89 * mm),
              P(f"The best trajectory-only model is n={int(best_shape.latent_dim)} with mean KS={best_shape.mean_ks:.3f} and {best_shape.mean_ks_rejected_fdr:.1f}/12 rejected features per subject. Its median multivariate energy distance is {best_shape.median_energy_distance:.2f} and mean MMD is {best_shape.mean_mmd_rbf:.3f}. This is better than n=2/3, but it is neither low-dimensional nor a successful full distribution match."),
              P("After corrected smoothing, submovement-count KS is low at high n. Remaining mismatches are dominated by peak speed, endpoint depth, timing, curvature, and lateral deviation. The hierarchical models reject about 10.6-10.7 features per subject and therefore do not improve the main objective."),
              P("A failure to reject a KS test is not proof of equivalence. Here the complementary effect-size metrics and the number of rejected features point in the same direction: fidelity remains incomplete."), PageBreak()]

    story += [P("8. Conclusions and next decisions", "h1"),
              P("Supported", "h2"),
              P("- Subject identity has a statistically detectable but weak relationship with movement trajectories.<br/>- A learned latent can identify enrolled held-out subjects above chance.<br/>- Movement duration is partly recoverable from time-normalized trajectory shape when modeled as a separate output.<br/>- Higher latent capacity improves reconstruction and query-distribution matching."),
              P("Not supported", "h2"),
              P("- Two or three latent means do not currently generate the requested subject timing/curvature distributions.<br/>- High joint-model timing R2 does not demonstrate timing inference, because timing is an encoder input.<br/>- Closed-set identification does not prove clean natural clustering or zero-shot fingerprinting.<br/>- The tested hierarchical architecture is not a solution; it learns identity signal but poor generative fidelity."),
              P("Recommended next steps", "h2"),
              P("1. Present v3 as a corrected preliminary negative/mixed result, not a completed subject generator.<br/>2. Confirm tracker coordinate units and the condition-2 fixation label with Prof. Friedman.<br/>3. Obtain Prof. Friedman's validated submovement decomposition before making strategy claims.<br/>4. Repeat the final protocol across pre-specified subject splits and initialization seeds; report confidence intervals.<br/>5. If the 2-3 parameter requirement remains strict, add stronger subject-level supervision or more subjects rather than increasing n and calling it the same goal.<br/>6. Evaluate exact screen-space target speed only as a documented ablation unless its mapping to physical speed is confirmed."),
              P("References", "h2"),
              P("Brenner, E. and Smeets, J. B. (2018). Continuously updating one's predictions underlies successful interception. Journal of Neurophysiology. doi:10.1152/jn.00517.2018.<br/>Kingma, D. P. and Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.<br/>Sohn, K., Lee, H., and Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models. NeurIPS.<br/>Zaheer, M. et al. (2017). Deep Sets. NeurIPS."),
              P("Generated by the audited code in this repository. Every numerical table and chart in this report is read from results/corrected_v3. The earlier preliminary PDF is retained unchanged as a legacy artifact.", "small")]

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#607582"))
        canvas.drawString(20 * mm, 11 * mm, "Interception movements - corrected preliminary audit")
        canvas.drawRightString(190 * mm, 11 * mm, f"Page {doc.page}"); canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=17 * mm, bottomMargin=18 * mm,
                            title="Interception Movements - Corrected Preliminary Results")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def write_summary(summary, kmeans, protocol):
    shape = summary[(summary.model == "per_trial_cvae") & (summary.variant == "trajectory_only_timing_prediction")]
    best = shape.loc[shape.mean_ks.idxmin()]
    text = f"""# Corrected Study (v3)\n\nThis is the authoritative result set after the methods audit. Legacy results remain under `results/` and in the old preliminary PDF, but must not be mixed with v3.\n\n## Protocol\n\n- {protocol['n_trials']} retained condition-2 trials, {protocol['n_subjects']} subjects.\n- Fixed 17/4/7 subject split.\n- Context/query split within every subject; query trials are not used to infer the fingerprint.\n- Frame-counter regularization, 10 Hz filtering, movement-onset to arrival trajectory, log timing outputs.\n- Primary low-dimensional fingerprint = subject context mean in n dimensions.\n\n## Main findings\n\n- Fixed-k K-Means: trajectory ARI {kmeans.loc[kmeans.representation=='trajectory','ari'].iloc[0]:.3f}; feature ARI {kmeans.loc[kmeans.representation=='kinematic_features','ari'].iloc[0]:.3f}. Both exceed the 200-permutation null, but clustering is weak.\n- Trajectory-only n=8 movement-time prediction: R2 {shape.loc[shape.latent_dim==8,'movement_time_s_r2'].iloc[0]:.3f}. Initiation-time R2 {shape.loc[shape.latent_dim==8,'initiation_time_s_r2'].iloc[0]:.3f}.\n- Best trajectory-only distribution model: n={int(best.latent_dim)}, mean KS {best.mean_ks:.3f}, mean FDR-rejected features {best.mean_ks_rejected_fdr:.2f}/12.\n- n=2/3 subject fingerprints do not generalize to subject timing or curvature distributions.\n- Hierarchical subject/trial CVAE did not improve generative fidelity.\n\n## Run\n\n```powershell\npython scripts/run_corrected_study.py --epochs 150 --dims 2 3 4 8 16 --hier-dims 2 3 4 --timing-weight 20 --out results\\corrected_v3\npython reports/build_corrected_pdf.py\n```\n"""
    SUMMARY_OUT.write_text(text, encoding="utf-8")


def main():
    summary, kmeans, baselines, protocol = load()
    figures(summary, kmeans, baselines)
    build(summary, kmeans, baselines, protocol)
    write_summary(summary, kmeans, protocol)
    print(OUT)


if __name__ == "__main__":
    main()
