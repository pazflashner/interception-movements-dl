"""Academic report and companion reading guide from versioned result tables."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, StyleSheet1
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    KeepTogether, PageBreak,
)
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "studies/review_corrected_evaluation/results"
WIDTH = 166 * mm
LABELS = {"cvae": "CVAE", "spline_pca": "Spline+PCA", "conditional_ae": "Conditional AE",
          "unconditional_vae": "Unconditional VAE", "condition_ridge": "Condition Ridge"}
FEATURE_LABELS = {
    "initiation_time_s": "Initiation time", "movement_time_s": "Movement time",
    "peak_speed_tracker_units_s": "Peak speed", "path_length": "Path length",
    "curvature_index": "Curvature index", "max_lateral_deviation": "Lateral deviation",
    "n_submovements": "Speed-peak count",
}


def number(x, digits=3):
    if not np.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def pvalue(x):
    return "&lt;0.0001" if x < 0.0001 else f"{x:.4f}"


def target_label(target):
    base, stat = target.rsplit("_", 1)
    return f"{FEATURE_LABELS.get(base, base)} ({stat})"


class Article:
    def __init__(self, output, title, subtitle):
        self.output = output
        self.title = title
        self.story = []
        self.styles = StyleSheet1()
        specs = {
            "body": dict(fontName="Times-Roman", fontSize=10.5, leading=13.4,
                         alignment=TA_JUSTIFY, spaceAfter=6, allowWidows=0, allowOrphans=0),
            "heading": dict(fontName="Times-Bold", fontSize=12, leading=14.5,
                            spaceBefore=12, spaceAfter=6, keepWithNext=True),
            "subheading": dict(fontName="Times-Bold", fontSize=10.5, leading=13,
                               spaceBefore=8, spaceAfter=4, keepWithNext=True),
            "caption": dict(fontName="Times-Roman", fontSize=9, leading=11.3, spaceAfter=9),
            "table": dict(fontName="Times-Roman", fontSize=8.2, leading=10),
            "title": dict(fontName="Times-Bold", fontSize=17, leading=20,
                          alignment=TA_CENTER, spaceAfter=12),
            "authors": dict(fontName="Times-Roman", fontSize=10, leading=13,
                            alignment=TA_CENTER, spaceAfter=12),
        }
        for name, spec in specs.items():
            self.styles.add(ParagraphStyle(name, textColor=colors.black, **spec))
        self.p(title, "title")
        self.p(subtitle, "authors")

    def p(self, text, style="body"):
        self.story.append(Paragraph(text, self.styles[style]))

    def h(self, text, sub=False):
        self.p(text, "subheading" if sub else "heading")

    def block(self, items):
        headings = []
        while self.story and isinstance(self.story[-1], Paragraph) and getattr(self.story[-1].style, "keepWithNext", False):
            headings.insert(0, self.story.pop())
        self.story.append(KeepTogether(headings + items))

    def table(self, rows, caption, widths=None):
        widths = widths or [WIDTH / len(rows[0])] * len(rows[0])
        if abs(sum(widths)-WIDTH) > .1:
            raise ValueError("Table must fit the text block")
        cells = [[Paragraph(("<b>"+str(x)+"</b>") if i == 0 else str(x), self.styles["table"])
                  for x in row] for i, row in enumerate(rows)]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("LINEABOVE", (0,0), (-1,0), .8, colors.black),
            ("LINEBELOW", (0,0), (-1,0), .5, colors.black),
            ("LINEBELOW", (0,-1), (-1,-1), .8, colors.black),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        self.block([Paragraph(caption, self.styles["caption"]), t, Spacer(1,8)])

    def figure(self, path, caption, max_height=100*mm):
        with PILImage.open(path) as im:
            w, h = im.size
        scale = min(WIDTH/w, max_height/h)
        self.block([Image(str(path), w*scale, h*scale),
                    Spacer(1,4), Paragraph(caption, self.styles["caption"])])

    def page(self):
        self.story.append(PageBreak())

    def finish(self):
        def footer(c, doc):
            c.saveState()
            c.setFont("Times-Roman", 9)
            c.drawCentredString(105*mm, 12*mm, str(doc.page))
            c.restoreState()
        SimpleDocTemplate(str(self.output), pagesize=(210*mm,297*mm),
            leftMargin=22*mm, rightMargin=22*mm, topMargin=20*mm, bottomMargin=20*mm,
            title=self.title, author="Simaan Libbiss and Paz Flashner",
            subject="Interception movement modeling; participant-held-out evaluation").build(
                self.story, onFirstPage=footer, onLaterPages=footer)


def benchmark_rows(oof, fingerprints, all_dims=False):
    acc = fingerprints.groupby(["model_family","latent_dim"]).balanced_accuracy.mean()
    rows = [["Model", "n", "MSE", "Init. MAE (ms)", "Move MAE (ms)", "Mean KS", "MMD<super>2</super>", "Enroll. (%)"]]
    for _, r in oof.iterrows():
        family, n = r.model_family, int(r.latent_dim)
        if not all_dims and n not in (0,3,8):
            continue
        rows.append([LABELS[family], str(n) if n else "-",
            number(r.trajectory_mse_subject_balanced_mean),
            number(r.initiation_time_mae_ms_subject_balanced_mean,1),
            number(r.movement_time_mae_ms_subject_balanced_mean,1),
            number(r.mean_ks_mean),number(r.mean_mmd_rbf_mean),
            number(100*acc.get((family,n),np.nan),1)])
    return rows


def probe_rows(probes):
    rows = [["Query-summary target", "n=3 pooled R<super>2</super>", "n=3 fold R<super>2</super>",
             "n=8 pooled R<super>2</super>", "n=8 fold R<super>2</super>"]]
    subset = probes[(probes.model_family=="cvae") & (probes.fingerprint=="mean")]
    for name in subset.target.unique():
        a = subset[(subset.latent_dim==3)&(subset.target==name)].iloc[0]
        b = subset[(subset.latent_dim==8)&(subset.target==name)].iloc[0]
        rows.append([target_label(name),number(a.r2_oof_mean),number(a.r2_fold_mean),
                     number(b.r2_oof_mean),number(b.r2_fold_mean)])
    return rows


def guide_report(output, tables, fingerprints, figures):
    a=Article(output,"Reading the Interception-Movement Results",
        "Companion guide for Simaan and Paz<br/>Figure and table numbers refer to the scientific report")
    a.h("1 The question we actually tested")
    a.p("Can recordings from part of a new participant's session provide a small fingerprint that generates the distribution of the rest of that session? 'New participant' means excluded from training, not a person for whom we have no recordings. We still need context trials to enroll that person. This is not a disease predictor, a proof of psychological strategy, or an automatic explanation of each latent coordinate.")
    a.p("There are three separate tasks. Reconstruction receives the actual trial and reproduces its shape. Timing prediction also receives that trial's shape, but not its physical timing labels. Generation receives only the participant's context fingerprint plus the task conditions, not the query trajectory. Always identify which task a result belongs to before saying a model is better.")
    a.h("2 What enters the encoder (Appendix Figure S1)")
    a.p("Marker 5 is appearance. The object then waits 0.20-0.50 seconds before it moves. Our time zero is that later motion event. The 100 x 2 input contains lateral and forward finger position from target motion to the last recorded sample, including any waiting after the go signal. It is not 100 raw 240-Hz frames: resampling gives 100 phase points spanning a trial of variable duration. Filtering reduces high-frequency noise; it is not a guarantee that everything left is intentional movement.")
    a.p("The encoder also gets start category, side and exact target speed. Start category couples distance and a speed range; side tells which side the target starts on. Neither is participant identity. Outcome, repetition, initiation seconds and movement seconds are not encoder inputs. A separate decoder head predicts those two physical times so the path can be assigned a time scale.")
    a.p("The final tracker sample is an arrival proxy, about 26 ms earlier than the MAT arrival timestamp on average. Movement onset is a sustained speed threshold, not an observed decision. Some people are already above that threshold at the go event: 357 labels are zero. We retain these cases for Jason to assess rather than silently calling them noise or removing them. The encoder still starts at target motion.")
    a.h("3 What a fingerprint means")
    a.p("Each context trajectory produces a posterior mean vector. We average those vectors within one participant to get theta. Averaging is a chosen summary, not something a VAE automatically proves to be meaningful. It can discard variability. The VAE's KL penalty encourages a regular latent representation but does not give every coordinate a named physical meaning or guarantee that the mean is a typical trial.")
    a.p("To generate, we add training-derived latent variation around theta and decode with a task condition. All participants share the latent noise covariance in this design. The nonlinear decoder can still give different output spreads at different centres. We do not learn a separate full latent probability distribution for every unseen participant. The requested two or three controls are the centre coordinates; n=8 is the more flexible comparison.")
    a.p("The new same-decoder control supports the value of the correct person's context. At n=8, mean KS is 0.219 with that person's fingerprint, 0.290 with the training-participant average and 0.317 with other people's fingerprints. Task draws, random latent variation and the decoder are fixed. All 12 corrected comparisons favour the correct fingerprint. This establishes useful personal information for this generator; it does not validate a psychological interpretation of its coordinates. See main Table 3 and Figure 2.")
    a.h("4 Folds, seeds, context and query")
    a.table([
        ["Term","Exactly what we did","What it does not establish"],
        ["Outer fold","17 training, 4 validation, 7 test participants. Four folds test each of the 28 once.","Not every possible split; not an independent new cohort."],
        ["Training seed","42,43,44 change neural initialization and training randomness.","Three seeds are not three new sets of people."],
        ["Context/query","About half of each test person's trials build the fingerprint; the other half measures distributions.","Not zero-shot prediction; task-condition mixture of query trials is known."],
        ["Model selection","Training fits parameters; validation chooses early stopping and Ridge regularization.","We must not select a correction by whichever test result looks best."],
    ],"Protocol terms",[29*mm,68*mm,69*mm])
    a.p("The model matrix is 48 CVAEs, 24 conditional AEs, 24 unconditional VAEs, 16 spline evaluations and 12 condition-Ridge evaluations. A later fairness check trained 48 small timing MLPs on frozen codes. It did not retrain the main CVAE. The corrected metrics reuse all 124 main runs; they are not selected examples.")
    a.h("5 Why spline can win without settling the whole project")
    a.figure(figures["benchmark"],"Scientific report Figure 1: separate tasks, separate metrics.")
    a.p("At n=3, spline's trajectory MSE is 0.097 and CVAE's is 0.177: spline wins reconstruction. At n=8, it is 0.023 versus 0.032: spline still wins. We should say that plainly. The CVAE has a timing objective and a regularized latent bottleneck, whereas spline+PCA directly compresses shape. That difference is part of the design, but we have not proved it is the cause of every error difference.")
    a.p("For generated distributions, n=8 CVAE mean KS is 0.219 versus spline 0.255 and condition Ridge 0.277. These are lower discrepancies, not 78.1% or 74.5% accuracy. At n=3, CVAE is better than spline on mean KS but not better than condition Ridge. So three coordinates do not yet meet the full personal-distribution objective just because the interface is compact.")
    a.h("6 How each metric is computed")
    a.table([
        ["Metric","Definition and interpretation"],
        ["Trajectory MSE","Square each coordinate error; average across axes, points and trials within each person, then average people. Units are tracker units squared. Lower is better. It measures reconstruction of an input, not generation of an unseen query path."],
        ["Timing MAE","Average absolute predicted-minus-actual time. We average within person first, then people; values are milliseconds. A 40 ms MAE is an average absolute error, not a worst-case bound."],
        ["R-squared","1 - sum of squared prediction errors / sum of squared deviations from the observed mean. Zero equals the test-set constant mean under squared error; negative is worse. It is not classification accuracy. State whether the rows are trials or participant summaries."],
        ["Two-sample KS","Largest vertical gap between empirical cumulative distributions. Zero means identical sampled cumulative curves; higher is worse. Mean KS averages 11 feature statistics, not p-values."],
        ["Squared MMD","Kernel comparison of the joint feature vectors. Our unbiased estimator can be slightly negative from finite-sample variation. The kernel bandwidth and feature scales are fitted on training trials and shared across models."],
        ["Energy discrepancy","2 times the average between-sample distance, minus the two within-sample averages. We use the non-square-root V-statistic. Constant count features retain a one-count scale; they are not divided by an almost-zero test SD."],
        ["Count total variation","Half the sum of absolute differences in count probabilities. Zero is a match; one is disjoint probability mass. Here it concerns fitted minimum-jerk component counts."],
        ["JSD","Symmetric Jensen-Shannon divergence of count histograms, using base-2 logarithms. Zero is a match; higher is worse. It is not the same metric as total variation or KS."],
    ],"Keep the unit, observational level and preferred direction attached to every number.",[34*mm,132*mm])
    a.h("7 R-squared: the question Moni asked")
    a.p("Suppose actual values are 1,2,3 and predictions are 1,2,2. The squared prediction error sums to 1. The actual mean is 2 and squared deviations from that mean sum to 2. R-squared is 1-1/2=0.5. This means half that test-set squared-error baseline remains, not that 50% of predictions are correct. A constant-target test set makes the denominator zero, which needs separate handling.")
    a.p("Our participant probe has one query-summary value per held-out person. A fold contains seven such values. A very small between-person variance within one fold makes its R-squared unstable. Pooling all 28 out-of-fold predictions changes the denominator. This explains why Appendix Table S4 includes both pooled R-squared and the average seven-person fold score; they cannot be substituted for each other.")
    a.p("Mean initiation time is a comparatively successful n=3 probe: pooled R-squared 0.723, average fold R-squared 0.661, MAE 14.8 ms versus 32.2 ms for the training-mean constant. But ten of fourteen pooled scores are negative. The honest conclusion is selective predictive information, not that the fingerprint explains all visible distributions. MAE may improve even when R-squared is negative because it penalizes errors differently.")
    a.p("There is also a simpler baseline: use the initiation-time mean actually measured in the context trials as the prediction for the query mean. Its MAE is 9.28 ms, versus 14.78 ms for the n=3 latent probe. For mean peak speed, direct context gives 1.73 versus 7.74 tracker units/s. Direct context has lower MAE on all 14 summaries at both n=3 and n=8. It retains target-specific measured summaries rather than one shared compact code. This limits the compression claim without contradicting the positive generation control.")
    a.h("8 Fair comparisons and significance (Table 4; Appendix Table S2)")
    a.p("Each paired test has 28 rows: one seed-averaged error per held-out participant for each model. A two-sided Wilcoxon test asks whether paired differences systematically favour one direction, under its assumptions. BH correction controls the false discovery rate across the stated family under the procedure's assumptions; it does not say that a particular scientific claim has a 95% probability of being true. No significant difference is not proof of equivalence.")
    a.p("The original timing comparison gave CVAE a nonlinear head and spline a linear head. We added a common small MLP for both frozen representations. At n=3, the initiation gap shrank to 41.4 versus 45.8 ms; corrected p=0.0076. At n=8 it is 42.7 versus 47.9 ms; p=0.0423. Movement-time differences under the common head were not significant. CVAE's code had already been trained with timing labels, so this remains a comparison of complete representation-learning choices.")
    a.p("We also fitted a multiplicative timing calibration using validation participants only. This is not an oracle chosen on the test group. It helps spline initiation MAE, especially at n=3, but does not remove the CVAE advantage in that sensitivity. Neither check proves a specific percentage of the original gap was caused by retransformation bias.")
    a.h("9 Enrollment, clustering and conditioning")
    a.figure(figures["capacity"],"Appendix Figure S2: descriptive capacity trend; error bars are not a confidence interval for a new cohort.",80*mm)
    a.p("Enrollment classifies query trial codes against seven candidate context means. About 40.9% at n=3 and 50.7% at n=8 exceed nominal 14.3% chance, but this is not classification among all 28 people or people with no enrollment recordings. K-means is different: it clusters the available representations without those labels, then ARI/NMI compare assignments with identity. The small ARI says the clusters are not clean subject groups, even if the permutation result is non-random.")
    a.p("The unconditional VAE often performs similarly to CVAE, with some endpoint-specific differences. The dedicated condition-stratum test found no advantage after its corrections. Do not say that condition sliders are proven to reproduce causal human responses. The decoder may respond to the sliders, but that response still needs validation against task-specific changes in people.")
    a.h("10 Submovements and heatmaps")
    a.p("A smooth minimum-jerk component is a fitted elementary movement pulse with a start time, duration and planar displacement. Several overlapping pulses can approximate a velocity curve. This is not a direct measurement of separate decisions. The speed-peak count in the main feature table is a simpler peak-detection heuristic and must never be renamed minimum-jerk count.")
    a.figure(figures["matched_components"],"Scientific report Figure 3: historical and matched-procedure results are separate evaluations.",85*mm)
    a.p("The new component comparison refits all 2,376 query recordings and 1,680 generated movements using the same downstream representation, filtering and optimizer budget. There are still only ten generated examples per person per seed. Main Table 6 reports the resulting distances and a newly calculated empirical sampling reference. Read those values rather than the historical 0.353/0.355 count-TV result. The reference is not a mathematical floor or formal hypothesis-test p-value. A completed fit, a converged optimizer and a biologically correct decomposition are three different claims.")
    a.p("A latent-feature heatmap reports Spearman rank correlations for one model and sampling setup. A high correlation suggests an association worth inspecting; it does not prove that changing only that coordinate controls a biological parameter. Coordinates can be correlated, and axes can change sign or rotate across trained models. The heatmap is an exploration aid, not an independently validated strategy theory.")
    a.h("11 Limitations we should volunteer")
    for text in [
        "The arrival endpoint is a recording proxy. Jason must confirm timing alignment, pre-go movement and the fixation warning before we treat those assumptions as settled.",
        "Typical timing error and extreme failure are both relevant. A rare 11.5-second prediction can damage R-squared even if MAE is tens of milliseconds. We did not clip these failures out of the main score.",
        "The 74.6% zero-onset figure refers to centered log-target variance. It is not measured model loss and does not prove why outliers occur.",
        "We have no learning-curve evidence that too few people caused the failures. We can propose more data, but cannot present that explanation as a result.",
        "All people came from one dataset. Four rotations test each once, but training sets overlap and this is not a replication on a new cohort.",
        "Our present two- or three-coordinate fingerprint does not reliably predict every person's distribution. This is a meaningful limitation of the tested approach, not something to omit.",
    ]: a.p(text)
    a.h("12 Reading files and preparing to present")
    a.p("Start with scientific Tables 2-6 and Figures 1-4, then Appendix Figures S5-S7 for trajectories and event examples. Use the dashboard's Generate section for an illustrative n and participant; use Benchmarks for all-fold numbers. Live trajectories use fold 0, seed 42 only. The corrected study's results folder contains exact rows behind every comparison, including unsuccessful probes. Appendix S8 lists their locations and the ZIP launch command.")
    a.p("Before presenting any graph, be able to state: what entered the model, what was withheld, what each plotted point represents, what the comparator is, the formula/unit, whether lower or higher is preferable, and what the result does not establish. For a number such as R-squared, also identify its denominator and observational unit. That is a stronger explanation than memorizing whether the number looked good.")
    a.finish()
