"""Build the concise advisor-facing story plus a complete figure appendix."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from reports.build_strategy_comparison_pdf import (
    BLUE,
    FIG_DIR as TECHNICAL_FIG_DIR,
    LIGHT,
    NAVY,
    ORANGE,
    RED,
    WINDOW_LABEL,
    build as _unused_build,
    load_data,
    make_figures,
    report_table,
    styles,
)


OUT_DIR = config.STUDY_ROOT / "output" / "advisor_brief"
FIG_DIR = OUT_DIR / "figures"
OUT = OUT_DIR / "Interception_Movement_Advisor_Brief.pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def build_extra_figures(data: dict) -> dict[str, Path]:
    results = data["results"]
    generation = data["generation"]
    figures: dict[str, Path] = {}

    # Main decision figure: only the measures needed to select a primary model.
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.15))
    metrics = [
        ("initiation_time_s_r2", "Initiation-time R2", False),
        ("fingerprint_balanced_accuracy", "Enrollment accuracy", False),
        ("mean_ks", "Mean distribution KS", True),
    ]
    for ax, (metric, title, lower_better) in zip(axes, metrics):
        for mode, color in (("movement_only", BLUE), ("go_to_arrival", RED)):
            group = results[results.window_mode == mode].groupby("latent_dim")[metric].agg(["mean", "std"])
            ax.errorbar(
                group.index, group["mean"], yerr=group["std"], marker="o",
                linewidth=2, capsize=3, color=color, label=WINDOW_LABEL[mode],
            )
        if metric.endswith("_r2"):
            ax.axhline(0, color="#64748B", linewidth=1, linestyle="--")
        if metric == "fingerprint_balanced_accuracy":
            ax.axhline(1 / 7, color="#64748B", linewidth=1, linestyle="--", label="chance")
        ax.set_xticks([2, 3, 4, 8])
        ax.set_xlabel("Latent dimensions")
        ax.set_title(title + (" (lower better)" if lower_better else ""), fontsize=10)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=6.5)
    figures["decision"] = savefig("primary_model_decision.png")

    # Representative generated-distribution tradeoff.
    rows = []
    for mode in config.WINDOW_MODES:
        frame = generation[mode].copy()
        frame["latent_dim"] = frame.run.str.extract(r"_z(\d+)_").astype(int)
        rows.append(frame)
    generated = pd.concat(rows, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.15))
    x = np.arange(2)
    width = .18
    offsets = [-1.5, -.5, .5, 1.5]
    palette = ["#93C5FD", BLUE, "#FCA5A5", RED]
    labels = []
    for offset, color, (_, row) in zip(offsets, palette, generated.iterrows()):
        mode = "Strategy" if "go_to_arrival" in row.run else "Execution"
        label = f"{mode} n={int(row.latent_dim)}"
        labels.append(label)
        axes[0].bar(x[0] + offset * width, row.mean_ks_initiation_time_s, width, color=color, label=label)
        axes[1].bar(x[1] + offset * width, row.mean_count_jsd, width, color=color, label=label)
    axes[0].set_xticks([0], ["Initiation time"])
    axes[0].set_ylabel("KS statistic")
    axes[0].set_title("Timing fidelity (lower better)")
    axes[1].set_xticks([1], ["Component count"])
    axes[1].set_ylabel("Jensen-Shannon divergence")
    axes[1].set_title("Execution-structure fidelity (lower better)")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    figures["generation"] = savefig("generated_distribution_tradeoff.png")

    # Dashboard map, kept conceptual so it remains valid across UI revisions.
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.axis("off")
    blocks = [
        (.01, "Protocol\nexecution / strategy"),
        (.205, "Model\nn + seed"),
        (.40, "Task condition\nsp + side + speed"),
        (.595, "Latent controls\ncenter / participant"),
        (.79, "Inspect\ntrajectory / validation"),
    ]
    for x0, label in blocks:
        ax.add_patch(plt.Rectangle((x0, .30), .17, .42, facecolor="#EEF3F6", edgecolor="#176B87", lw=1.5))
        ax.text(x0 + .085, .51, label, ha="center", va="center", fontsize=8.2)
    for index, (x0, _) in enumerate(blocks[:-1]):
        ax.annotate("", xy=(blocks[index + 1][0], .51), xytext=(x0 + .17, .51), arrowprops={"arrowstyle": "->", "color": "#17324D"})
    ax.text(.5, .12, "One dashboard contains both temporal protocols and every tested latent width", ha="center", fontsize=9, color="#176B87")
    figures["dashboard"] = savefig("dashboard_workflow.png")
    return figures


def image_grid(paths: list[Path], width: float, height: float, columns: int = 2) -> Table:
    cells = [Image(str(path), width=width, height=height) for path in paths]
    rows = [cells[index:index + columns] for index in range(0, len(cells), columns)]
    while len(rows[-1]) < columns:
        rows[-1].append("")
    return Table(rows, colWidths=[width + 3 * mm] * columns)


def build(data: dict, technical: dict[str, Path], extra: dict[str, Path]) -> None:
    s = styles()
    P = lambda text, style="body": Paragraph(text, s[style])
    results = data["results"]
    grouped = results.groupby(["window_mode", "latent_dim"]).mean(numeric_only=True)
    go3 = grouped.loc[("go_to_arrival", 3)]
    go8 = grouped.loc[("go_to_arrival", 8)]
    movement8 = grouped.loc[("movement_only", 8)]
    movement_generation = data["generation"]["movement_only"].set_index("run")
    strategy_generation = data["generation"]["go_to_arrival"].set_index("run")
    associations = data["associations"]
    go_n3_seed42 = associations[
        (associations.window_mode == "go_to_arrival")
        & (associations.latent_dim == 3)
        & (associations.seed == 42)
    ]
    go_trial = go_n3_seed42[go_n3_seed42.level == "trial_within_subject_partial"]
    go_subject = go_n3_seed42[go_n3_seed42.level == "subject_context_to_query"]

    story: list = []
    story += [
        Spacer(1, 13 * mm),
        P("Interception-Movement Fingerprints", "title"),
        P("Advisor brief: temporal representation, CVAE findings, latent interpretation, and interactive review", "subtitle"),
        P("Seman Libbiss and Paz Flashner | Deep Learning Workshop | Prof. Jason Friedman | Advisor: Moni", "subtitle"),
        Spacer(1, 3 * mm),
        P("Question addressed after the submitted proposal", "h1"),
        P("Can a low-dimensional conditional VAE preserve participant-specific interception behavior while generating held-out distributions of movement shape, initiation, duration, and submovement structure?", "callout"),
        P("Recommendation", "h2"),
        P("Use the target-motion-onset -> arrival CVAE with n=3 as the primary low-dimensional model. It is the smallest tested representation with stable held-out initiation and movement-time prediction, and it retains the waiting interval required for strategy analysis. Use n=8 as a capacity reference. Keep movement-onset -> arrival as an execution-focused control, not as a second equal final model."),
        report_table([
            ["Primary conclusion", "Evidence"],
            ["n=3 is the interpretable strategy model", f"Initiation R2={go3.initiation_time_s_r2:.2f}; movement-time R2={go3.movement_time_s_r2:.2f}; all values are held-out participant means across three seeds."],
            ["n=8 is the capacity bound", f"Strategy-window mean KS improves to {go8.mean_ks:.3f}; enrollment reaches {100*go8.fingerprint_balanced_accuracy:.1f}%."],
            ["Execution-only remains informative", f"Its n=8 enrollment reaches {100*movement8.fingerprint_balanced_accuracy:.1f}% versus 14.3% chance and its generated component-count distribution is more faithful."],
            ["The full goal is not solved", "No tested low-dimensional model exactly reproduces every held-out behavioral distribution."],
        ], [48 * mm, 122 * mm]),
        PageBreak(),
    ]

    story += [
        P("1. Why two temporal protocols were compared", "h1"),
        P("Phase resampling is needed to provide a fixed-size neural input, but it removes physical duration. Following Jason's concern, physical timing is withheld from the encoder and predicted separately by the decoder. A second issue is the start event: beginning at finger movement onset removes the participant's decision to wait."),
        Image(str(technical["windows"]), width=165 * mm, height=65 * mm),
        report_table([
            ["Protocol", "Scientific content", "Role in final study"],
            ["Movement onset -> arrival", "Physical execution and corrections", "Execution control"],
            ["Target motion onset -> arrival", "Waiting, initiation, execution, corrections", "Primary strategy model"],
        ], [47 * mm, 73 * mm, 50 * mm]),
        P("Both protocols use the same 4,732 retained condition-2 trials, x-y table plane, 10 Hz filter, task conditions, participant split, dimensions, seeds, and evaluation. Movement time and initiation time are withheld from the encoder and decoded separately."),
        P("Trial audit", "h2"),
        P("No recording ended before target motion began. All 48 Too early labels contain a completed arrival after the target started and are retained. Four no-arrival timeouts and 27 arrivals more than one second after the target window are excluded under the current rerunnable rule."),
        PageBreak(),
    ]

    story += [
        P("2. Model and evaluation", "h1"),
        P("The CVAE encodes 100 x-y phase samples plus task condition (start/speed category, starting side, and exact executed target speed) into a Gaussian latent distribution. The decoder reconstructs trajectory and predicts positive initiation and movement times. KL regularization supports continuous latent interpolation and participant-level averaging within one trained model."),
        report_table([
            ["Stage", "Protocol"],
            ["Training", "17 participants; subject-balanced batches"],
            ["Hyperparameter selection", "4 separate validation participants"],
            ["Final evaluation", "7 untouched participants"],
            ["Fingerprint enrollment", "Context-half latent mean for each held-out participant"],
            ["Fingerprint query", "Disjoint, condition-stratified trial half"],
            ["Model sweep", "Both temporal protocols x n=2/3/4/8 x seeds 42/43/44 = 24 runs"],
        ], [52 * mm, 118 * mm]),
        P("The fingerprint-identification result is closed-set enrollment: context movements are available for each test participant. It is not zero-shot identification of a person with no recorded movement."),
        P("Baselines", "h2"),
        P("K-Means with k fixed to 28 is above a 200-permutation null but weak in absolute terms (trajectory ARI about 0.05-0.06). A spline fitted directly to each test trial is descriptive, not predictive. The fair learned compression baseline is spline coefficients followed by PCA fitted only on training participants."),
        PageBreak(),
    ]

    story += [
        P("3. Main comparison and model choice", "h1"),
        Image(str(extra["decision"]), width=165 * mm, height=64 * mm),
        report_table([
            ["Model", "Initiation R2", "Movement R2", "Enrollment", "Mean KS"],
            ["Strategy n=3", f"{go3.initiation_time_s_r2:.2f}", f"{go3.movement_time_s_r2:.2f}", f"{100*go3.fingerprint_balanced_accuracy:.1f}%", f"{go3.mean_ks:.3f}"],
            ["Strategy n=8", f"{go8.initiation_time_s_r2:.2f}", f"{go8.movement_time_s_r2:.2f}", f"{100*go8.fingerprint_balanced_accuracy:.1f}%", f"{go8.mean_ks:.3f}"],
            ["Execution n=8", f"{movement8.initiation_time_s_r2:.2f}", f"{movement8.movement_time_s_r2:.2f}", f"{100*movement8.fingerprint_balanced_accuracy:.1f}%", f"{movement8.mean_ks:.3f}"],
        ], [42 * mm, 30 * mm, 31 * mm, 34 * mm, 33 * mm]),
        P("n=2 is too restrictive: initiation-time R2 is unstable and strongly negative in both protocols. n=4 is a useful transition point but adds little to the scientific story. n=3 is therefore the low-dimensional recommendation, while n=8 shows the attainable capacity with this architecture and dataset."),
        P("There is room for both temporal protocols only because they answer different questions. The strategy protocol is primary. The execution protocol remains a control that reveals what is gained or lost by including the waiting interval."),
        PageBreak(),
    ]

    story += [
        P("4. Generated distributions: the two protocols are complementary", "h1"),
        Image(str(extra["generation"]), width=160 * mm, height=64 * mm),
        report_table([
            ["Representative seed-42 model", "Initiation KS", "Component-count JSD"],
            ["Execution n=3", f"{movement_generation.loc['cvae_movement_only_z3_seed42'].mean_ks_initiation_time_s:.3f}", f"{movement_generation.loc['cvae_movement_only_z3_seed42'].mean_count_jsd:.3f}"],
            ["Execution n=8", f"{movement_generation.loc['cvae_movement_only_z8_seed42'].mean_ks_initiation_time_s:.3f}", f"{movement_generation.loc['cvae_movement_only_z8_seed42'].mean_count_jsd:.3f}"],
            ["Strategy n=3", f"{strategy_generation.loc['cvae_go_to_arrival_z3_seed42'].mean_ks_initiation_time_s:.3f}", f"{strategy_generation.loc['cvae_go_to_arrival_z3_seed42'].mean_count_jsd:.3f}"],
            ["Strategy n=8", f"{strategy_generation.loc['cvae_go_to_arrival_z8_seed42'].mean_ks_initiation_time_s:.3f}", f"{strategy_generation.loc['cvae_go_to_arrival_z8_seed42'].mean_count_jsd:.3f}"],
        ], [76 * mm, 47 * mm, 47 * mm]),
        P("The strategy window is substantially better for initiation-time distributions because waiting is represented. The execution window is better for generated component-count distributions because all model capacity is focused on physical movement. Continuous outputs use KS/Wasserstein; discrete component count uses JSD/total variation."),
        P("Minimum-jerk components", "h2"),
        Spacer(1, 2 * mm),
        P("Each component is a fitted smooth velocity primitive with onset, duration, and 2-D displacement. Component count, secondary amplitude fraction, and overlap describe kinematic organization. They are compatible with single or corrective movement organization but do not by themselves prove feedforward, feedback, hesitation, or regret."),
        PageBreak(),
    ]

    trial_heatmap = config.RESULTS_DIR / "latent_associations" / "heatmaps" / "go_to_arrival_z3_trial_within_subject_partial.png"
    subject_heatmap = config.RESULTS_DIR / "latent_associations" / "heatmaps" / "go_to_arrival_z3_subject_context_to_query.png"
    story += [
        P("5. What the n=3 strategy latent represents", "h1"),
        P("Trial-level correlations residualize participant, start category, side, and exact speed. Subject-level correlations compare context fingerprints with means from disjoint query trials. The heatmaps below use the pre-specified seed-42 n=3 strategy model."),
        image_grid([trial_heatmap, subject_heatmap], 78 * mm, 59 * mm),
        report_table([
            ["Strong association", "Spearman rho", "Interpretation boundary"],
            ["Trial z3 vs initiation time", "+0.57", "Within-participant, condition-adjusted"],
            ["Trial z1 vs initiation time", "+0.55", "Within-participant, condition-adjusted"],
            ["Trial z1 vs movement time", "-0.49", "Within-participant, condition-adjusted"],
            ["Subject z3 vs query initiation mean", "+0.61", "N=28 context-to-query association"],
        ], [67 * mm, 34 * mm, 69 * mm]),
        P(f"Within this model, {int(go_trial.fdr_reject_0_05.sum())}/24 trial-level and {int(go_subject.fdr_reject_0_05.sum())}/24 subject-level associations survive Benjamini-Hochberg correction. With thousands of trials, effect size matters more than significance count. Latent axes can rotate, reflect, or swap across seeds, so z1/z2/z3 are model-specific coordinates, not universal psychological variables."),
        PageBreak(),
    ]

    story += [
        P("6. One dashboard contains both protocols", "h1"),
        Image(str(extra["dashboard"]), width=165 * mm, height=56 * mm),
        report_table([
            ["Control", "Available choices"],
            ["Temporal protocol", "Movement onset -> arrival; target motion onset -> arrival"],
            ["Latent width", "n=2, 3, 4, 8"],
            ["Training seed", "42, 43, 44 in the full dashboard"],
            ["Task condition", "Start/speed category, side, exact executed speed"],
            ["Fingerprint source", "Training-population center or an enrolled held-out participant"],
            ["Generated output", "Trajectory, speed, minimum-jerk decomposition, sampled distributions"],
            ["Evidence tabs", "Held-out validation, repeated-seed comparison, latent heatmaps, protocol"],
        ], [48 * mm, 122 * mm]),
        P("Use", "h2"),
        P("1. Extract the dashboard ZIP and run setup_and_launch.bat once.<br/>2. Select the strategy protocol and n=3 for the primary scientific view.<br/>3. Choose a participant context fingerprint or the population center.<br/>4. Change one latent slider while holding task condition fixed.<br/>5. Inspect trajectory and timing, then use Held-out validation before interpreting the generated distribution.<br/>6. Use n=8 to inspect the capacity bound, not as an eight-trait explanation."),
        P("The full dashboard includes every seed. The email-sized dashboard uses seed 42 for live generation but retains all-seed aggregate comparison tables."),
        PageBreak(),
    ]

    story += [
        P("7. Conclusions, assumptions, and requested feedback", "h1"),
        P("What the current evidence supports", "h2"),
        P("- A low-dimensional strategy-inclusive CVAE can jointly represent trajectory shape and predict withheld timing.<br/>- n=3 is the smallest stable candidate for the requested controllable fingerprint.<br/>- Participant information transfers after context enrollment.<br/>- Waiting is informative for initiation strategy; execution-only remains useful for movement organization.<br/>- Generated distributions are partially, not exactly, reproduced."),
        P("Assumptions made for the current analysis", "h2"),
        report_table([
            ["Item", "Current decision to confirm"],
            ["Completion event", "A non-empty MAT pressedTime indicates arrival at the interception site"],
            ["Too early", "Retain: all 48 arrivals occur after target motion begins"],
            ["Late threshold", "Exclude arrival more than 1 s after the target window"],
            ["Go signal", "Executed MAT dotArray target-motion onset defines time zero"],
            ["Task plane", "x-y table plane; z treated as off-plane variation"],
            ["Submovement constraints", "100 ms minimum duration; 50 ms minimum onset spacing"],
            ["Component order", "Smallest model reaching normalized error <=0.05, then <=0.10; BIC retained as sensitivity output"],
            ["Units and target scaling", "Report tracker units until physical conversion and screen scaling are confirmed"],
            ["Not-fixating label", "Keep separate in condition 2 until its interpretation is confirmed"],
        ], [51 * mm, 119 * mm], 7.6),
        P("Feedback requested", "h2"),
        P("We would appreciate Jason's confirmation or correction of the task-specific assumptions above, and comments from both Jason and Moni on the temporal comparison, CVAE design, evaluation protocol, and interpretation boundaries. If an event or component rule changes, the corresponding preprocessing or submovement stage can be rerun without changing the overall study design."),
        PageBreak(),
    ]

    # Appendix cover and complete secondary figures.
    story += [
        P("Appendix", "title"),
        P("Complete secondary figures and numerical sweeps", "subtitle"),
        P("The main text deliberately reports only the models needed for the scientific decision. The appendix preserves the complete audit trail, baseline comparison, all latent widths, and every seed-42 association heatmap."),
        PageBreak(),
        P("A1. Data audit and temporal windows", "h1"),
        Image(str(technical["audit"]), width=165 * mm, height=67 * mm),
        Image(str(technical["windows"]), width=165 * mm, height=65 * mm),
        PageBreak(),
        P("A2. Baselines and full repeated-seed result panels", "h1"),
        Image(str(technical["baselines"]), width=165 * mm, height=69 * mm),
        Image(str(technical["models"]), width=165 * mm, height=122 * mm),
        PageBreak(),
        P("A3. Recorded minimum-jerk structure", "h1"),
        Image(str(technical["sub_examples"]), width=165 * mm, height=64 * mm),
        Image(str(technical["submovements"]), width=165 * mm, height=67 * mm),
        PageBreak(),
        P("A4. Complete repeated-seed numerical sweep", "h1"),
        report_table([
            ["Protocol", "n", "Move MSE", "Move R2", "Initiation R2", "Enrollment", "Mean KS"],
            *[
                [
                    "Strategy" if mode == "go_to_arrival" else "Execution",
                    n,
                    f"{grouped.loc[(mode, n)].movement_reconstruction_mse_tracker_units2:.3f}",
                    f"{grouped.loc[(mode, n)].movement_time_s_r2:.2f}",
                    f"{grouped.loc[(mode, n)].initiation_time_s_r2:.2f}",
                    f"{100*grouped.loc[(mode, n)].fingerprint_balanced_accuracy:.1f}%",
                    f"{grouped.loc[(mode, n)].mean_ks:.3f}",
                ]
                for mode in config.WINDOW_MODES for n in [2, 3, 4, 8]
            ],
        ], [34 * mm, 13 * mm, 27 * mm, 25 * mm, 29 * mm, 23 * mm, 19 * mm], 7.2),
        P("Values are means across seeds 42, 43, and 44 on held-out participants. Raw full-window MSE is not compared across temporal protocols because the target intervals differ; the table reports movement-region MSE."),
        PageBreak(),
    ]

    heatmap_dir = config.RESULTS_DIR / "latent_associations" / "heatmaps"
    for page_index, (mode, level, title) in enumerate([
        ("go_to_arrival", "trial_within_subject_partial", "A5. Strategy protocol: trial-level partial associations"),
        ("go_to_arrival", "subject_context_to_query", "A6. Strategy protocol: subject context-to-query associations"),
        ("movement_only", "trial_within_subject_partial", "A7. Execution protocol: trial-level partial associations"),
        ("movement_only", "subject_context_to_query", "A8. Execution protocol: subject context-to-query associations"),
    ]):
        paths = [heatmap_dir / f"{mode}_z{n}_{level}.png" for n in [2, 3, 4, 8]]
        story += [
            P(title, "h1"),
            image_grid(paths, 80 * mm, 57 * mm),
            P("Seed 42. Asterisks survive within-model Benjamini-Hochberg correction. Compare association patterns, not latent-axis labels, across independently trained models.", "small"),
            PageBreak(),
        ]

    story += [
        P("A9. References and reproducibility", "h1"),
        P("Brenner, E. and Smeets, J. B. J Neurophysiol (2018), doi:10.1152/jn.00517.2018.<br/>Flash, T. and Hogan, N. J Neurosci (1985), minimum-jerk movement model.<br/>Friedman, J. submovements repository, commit 9c2f40c, github.com/JasonFriedman/submovements.<br/>Kingma, D. P. and Welling, M. Auto-Encoding Variational Bayes, ICLR (2014).<br/>Sohn, K., Lee, H. and Yan, X. Conditional deep generative models, NeurIPS (2015).<br/>Slowinski, P. et al. Dynamic similarity and individual motor signatures, J R Soc Interface (2016), doi:10.1098/rsif.2015.1093."),
        P("All numerical values and figures are generated from the current study artifacts. The code, split, seeds, retained-trial list, checkpoints, and validation tables are preserved in the project directory. No raw Dropbox trajectory or MAT file is included in the advisor package.", "small"),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#607582"))
        canvas.drawString(20 * mm, 11 * mm, "Interception-movement advisor brief")
        canvas.drawRightString(190 * mm, 11 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Interception-Movement Fingerprints - Advisor Brief",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    data = load_data()
    technical = make_figures(data)
    extra = build_extra_figures(data)
    build(data, technical, extra)
    print(OUT)


if __name__ == "__main__":
    main()
