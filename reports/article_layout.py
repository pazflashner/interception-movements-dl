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


def scientific_report(output, tables, fingerprints, figures, example_meta):
    a=Article(output,"Low-Dimensional Generative Models of Human Interception Movements",
        "Simaan Libbiss and Paz Flashner<br/>Workshop on Deep Learning, Tel Aviv University<br/>"
        "Research supervision: Prof. Jason Friedman, Dept. Physical Therapy &amp; Sagol School of Neuroscience<br/>"
        "Course advisor: Moni Shahar")
    a.h("Abstract")
    a.p("We investigate whether a compact representation inferred from a participant's interception trials can generate the distribution of that participant's remaining movements. We evaluate a conditional variational autoencoder (CVAE) and four comparator families on 4,732 trials from 28 participants, using four participant-held-out folds and three neural training seeds. The input preserves the interval from target motion to the end of the recorded reach; physical timing is predicted separately from the phase-normalized trajectory. Spline+PCA provides lower reconstruction error at matched latent dimension, whereas CVAE provides lower timing error and several lower generated-distribution distances. A common-head sensitivity analysis reduces, but does not eliminate, the CVAE's initiation-time advantage. Participant-level behavioural probes recover some between-participant differences but fail on many targets. Explicit task conditioning and minimum-jerk component fidelity remain limited. The findings support task-dependent benefits of a learned representation, not a uniformly superior model or a validated two- or three-parameter account of individual strategy.")
    a.h("1 Introduction")
    a.p("Interception requires predicting a moving target while allowing online corrections [1]. Compact models of individual movement variation could support future motor-assessment research, but clinical inference is outside this study. Following work on individual motor signatures [2], our question is whether a low-dimensional representation of recorded trajectories can support participant-specific generation, rather than merely identify participants. The requested compact interface is two or three latent coordinates; n=8 serves as a higher-capacity comparison.")
    a.p("We distinguish three tasks: reconstruction of an observed trajectory; prediction of its physical timing from shape; and generation of new trials after enrollment from a subset of a held-out participant's recordings. A successful reconstruction does not establish the third task. Our machine-learning objective is therefore evaluated against both classical compression and non-personalized stochastic baselines.")
    a.h("2 Data and temporal representation")
    a.p("The experiment records three-dimensional finger position at 240 Hz using an electromagnetic tracker while a participant reaches toward an interception region on a tabletop. Target trajectories run at 60 Hz. We analyze condition 2 (free eye movements). Of 4,763 available condition-2 recordings, the canonical rules remove four timeouts and 27 excessively late arrivals, retaining 4,732 trials from 28 participants. The source project description lists 29 participants; the analyzed cohort is the 28 with available condition-2 recordings. Trial filenames encode condition, start-position/speed category, side and repetition. MAT metadata supplies exact target speed and event times. Repetition and outcome labels are not encoder inputs.")
    a.p("Marker 5 identifies target <i>appearance</i>, not motion. Time zero is the MAT-derived target-motion event, which follows appearance by 0.20-0.50 s (median 0.35 s). Tracker timing uses frame-counter intervals at 240 Hz. The retained window ends at the last tracker sample, an arrival proxy rather than the MAT pressedTime itself. This proxy precedes the MAT event by 26.2 ms on average (SD 3.6 ms). Figure A3 documents the event audit; synchronization requires confirmation from Prof. Friedman.")
    a.p("Position is fourth-order, zero-phase Butterworth filtered at 10 Hz, translated to the window origin and resampled to 100 phase points. Lateral x and forward y are modeled; z is omitted from the trajectory output. The canonical movement-onset label is the first post-go 3-D speed crossing above 5 tracker units/s sustained for three frames. Initiation time is go-to-onset; movement time is onset-to-recording-end. Both physical times are withheld from the encoder. The primary go-to-end input retains post-go waiting; an earlier movement-only representation was explored but is not part of the present model comparison.")
    a.p("The late-arrival cutoff is 1 s beyond the target's interception window. A 'Too early' outcome is not treated as premature departure: recorded early arrivals after the go event remain included. The retained cohort includes 326 'Too late' and 1,352 'Not fixating on the dot enough!!!' outcomes. Other quality flags are audits, not additional exclusions. Tracker coordinates are reported in tracker units because their physical calibration is unconfirmed.")
    a.figure(figures["pipeline"],"<b>Figure 1.</b> Participant-held-out protocol. Context and query trials are disjoint within each test participant; no query trajectory determines the enrollment fingerprint.",40*mm)
    a.h("3 Models and enrollment")
    a.p("The CVAE [3,4] maps a 100 x 2 trajectory and a five-element condition vector (three start-category indicators, side and exact target speed) through two 256-unit ReLU layers to a diagonal-Gaussian posterior. A corresponding two-layer decoder takes a latent draw and the condition vector, and outputs position and two timing channels. Training uses participant-balanced sampling, Adam (learning rate 0.001; batch size 64), gradient clipping at 5, validation early stopping (patience 25) and a 150-epoch limit. Five of the 96 neural runs reached this limit. All normalization statistics are fitted to training data.")
    a.p("The objective is a summed standardized position squared error plus 20 times the summed standardized log-timing squared error and a KL penalty to N(0,I). The KL coefficient increases linearly from 0 to 1 over 50 epochs. Timing is transformed as log(t+0.001) and inverted by max(exp(u)-0.001,0). These are weighted reconstruction objectives, not calibrated likelihoods for the reported behavioural distributions. KL regularization does not guarantee semantic independence, identifiable latent axes or meaningful averages for every participant.")
    a.table([
        ["Model", "Representation and comparison"],
        ["CVAE; n=2,3,4,8", "Variational trajectory code with task condition; joint shape and timing objective."],
        ["Spline+PCA; n=2,3,4,8", "Cubic B-spline fit (five knots), training-fitted PCA and validation-tuned Ridge log-timing head; classical compression comparator."],
        ["Conditional AE; n=3,8", "Same conditional architecture without stochastic bottleneck or KL term."],
        ["Unconditional VAE; n=3,8", "Same variational architecture with task condition suppressed."],
        ["Condition Ridge", "Task metadata alone predicts shape and timing; training-stratum residual resampling supplies stochastic variation without a personal fingerprint."],
    ],"<b>Table 1.</b> Model families. Equal latent dimension does not equalize parameter count, objective or timing-head capacity.",[46*mm,120*mm])
    a.p("For a held-out participant s, the fingerprint is the average posterior mean over context trials: theta_s = mean{mu(x,c): (x,c) in context_s}. Generation samples z from N(theta_s, Sigma_train), using a shared covariance estimated from within-participant training variability and posterior uncertainty. The decoder generates 120 trials per participant. Query task conditions are sampled to match the evaluated condition mixture; query trajectories and timings are not used for enrollment. Thus generation is conditional on a known task mixture, not zero-shot prediction of a new participant. The spline generator similarly uses context-mean coefficients with training-derived variation. A shared latent covariance does not imply equal output variance under a nonlinear decoder.")
    a.h("4 Evaluation")
    a.p("Four fixed outer folds each contain 17 training, four validation and seven test participants. Every participant is tested once. Context/query allocation is approximately 50/50, stratified by task where possible. Neural seeds 42, 43 and 44 vary training randomness; the fold allocation is fixed. The matrix contains 48 CVAE, 24 conditional-AE, 24 unconditional-VAE, 16 spline and 12 condition-Ridge evaluations. This covers all specified folds and seeds, not all possible participant splits. Training sets overlap across folds; there is no independent acquisition cohort.")
    a.p("For participant s, trajectory MSE averages squared raw-coordinate reconstruction error over that participant's trials, phase points and two axes. Timing MAE averages absolute error over trials, then participants receive equal weight. Reconstruction uses posterior means, not sampled posterior noise. R<super>2</super> = 1 - sum(y-yhat)<super>2</super>/sum(y-mean(y))<super>2</super>. Trial-pooled timing R<super>2</super>, R<super>2</super> of participant means, and participant-summary probe R<super>2</super> have different observational units and are reported separately in the accompanying tables.")
    a.p("Distribution fidelity uses two-sample KS statistics on 11 features: initiation and movement time, peak speed, relative time to peak speed, path length, straight-line distance, curvature, lateral deviation, speed-peak count and x/y endpoints. The reported mean KS weights features and participants equally; it is an effect-size summary, not a combined hypothesis test. Speed peaks use a 10% prominence and 50 ms separation heuristic and are distinct from fitted minimum-jerk components. Movement-derived features are computed on the extracted movement interval using the same feature routine for recorded and generated paths. Generated intervals are recovered from predicted initiation and movement time.")
    a.p("Multivariate distances use one training-only feature reference per fold shared by every model. Continuous coordinates are scaled by training SD; count scales are at least one count and constant features use scale one. RBF bandwidth is fixed from up to 512 training examples. We report the unbiased squared MMD estimator (which can be slightly negative at finite sample size) and energy E = 2 mean||X-Y|| - mean||X-X'|| - mean||Y-Y'||, including within-sample diagonal terms. Lower discrepancy is preferable. KS testing for discrete counts is not used as a continuous-null proof of agreement.")
    a.p("Paired comparisons average seeds within each participant before two-sided Wilcoxon tests. Benjamini-Hochberg correction covers the 56 model-endpoint comparisons as one family; the eight added timing-head comparisons form a separate sensitivity family. Cross-fold participant dependence and the small cohort limit population-level inference. Closed-set enrollment assigns each query code to the closest of seven context centroids; nominal chance is 1/7. K-means is an in-sample descriptive analysis, not held-out classification.")
    a.h("5 Results")
    a.h("5.1 Reconstruction and generated distributions",True)
    a.table(benchmark_rows(tables["oof"],fingerprints),
        "<b>Table 2.</b> Out-of-fold participant-balanced means; neural results average three seeds. MSE is in tracker units squared. Timing uses the original heads; Table 4 examines head capacity and calibration. Enrollment is a closed-set task, not zero-shot identity prediction.",
        [33*mm,8*mm,17*mm,23*mm,23*mm,18*mm,21*mm,23*mm])
    a.p("Spline+PCA has lower observed-trajectory MSE than CVAE at n=3 (0.097 versus 0.177) and n=8 (0.023 versus 0.032). CVAE has lower original-head timing MAE, mean KS and multivariate discrepancy at those dimensions (Table 3). These comparisons establish an endpoint-dependent tradeoff; they do not show that VAE is intrinsically superior to spline approximation. Representative reconstructions and generated paths appear in Figure A1.")
    a.figure(figures["benchmark"],"<b>Figure 2.</b> Selected benchmark endpoints. All panels are errors or discrepancies; smaller values indicate better performance for that particular task.")
    paired=tables["paired"]
    rows=[["n","Endpoint","CVAE","Spline","CVAE lower /28","Adjusted p"]]
    metric_labels={"trajectory_mse":"Trajectory MSE","initiation_time_mae_ms":"Initiation MAE","movement_time_mae_ms":"Movement MAE","mean_ks":"Mean KS","mmd_rbf":"MMD squared"}
    for n in (3,8):
        for metric,label in metric_labels.items():
            r=paired[(paired.latent_dim==n)&(paired.comparator=="spline_pca")&(paired.metric==metric)].iloc[0]
            digits=1 if "mae" in metric else 3
            rows.append([n,label,number(r.cvae_mean,digits),number(r.comparator_mean,digits),int(r.cvae_better_participants),pvalue(r.wilcoxon_p_fdr_bh)])
    a.table(rows,"<b>Table 3.</b> Paired CVAE-spline contrasts, adjusted within the complete 56-comparison family. The lower MSE belongs to spline; timing is in ms.",[8*mm,44*mm,25*mm,25*mm,32*mm,32*mm])
    a.p("At n=3, CVAE does not improve mean KS, MMD or energy over the condition-only baseline after correction. At n=8, all three distribution distances are lower than that baseline. Relative to unconditional VAE, the results are mixed: at n=3 unconditional VAE has lower MSE while CVAE has lower mean KS; at n=8 unconditional VAE has lower energy distance. Most other endpoints do not distinguish these two families. Hence explicit conditioning is not uniformly beneficial, and a useful personal representation cannot be inferred from reconstruction alone.")
    a.figure(figures["capacity"],"<b>Figure 3.</b> CVAE capacity and closed-set enrollment. Accuracy error bars describe fold/seed spread, not a population confidence interval. Increasing n improves mean reconstruction and distribution fidelity, but does not prove interpretable coordinate semantics.",78*mm)
    a.h("5.2 Timing-head sensitivity",True)
    a.p("The original CVAE timing head is nonlinear, whereas spline uses linear Ridge. Direct exponentiation of a squared-log-error prediction also need not optimize physical-time MAE. We therefore add two post-review sensitivities, without selecting on test errors: (i) one positive multiplicative factor per timing endpoint fitted to validation MAE; and (ii) the same two-layer 64-unit MLP timing head trained on frozen CVAE or spline codes plus conditions. The latter uses train-fitted input/log-target scaling, participant weights and validation early stopping (up to 400 epochs, patience 40). It comprises 48 additional small head fits, not retraining of the generative models.")
    rows=[["Head comparison","n","Endpoint","CVAE ms","Spline ms","Adjusted p"]]
    for _,r in tables["timing_fairness_paired"].iterrows():
        rows.append(["Common MLP" if r.comparison=="common_mlp" else "Validation calibration",
                     r.latent_dim,r.endpoint,number(r.cvae_mean_ms,1),number(r.spline_mean_ms,1),pvalue(r.p_fdr_bh)])
    a.table(rows,"<b>Table 4.</b> Timing sensitivities. Paired participants are seed-averaged; correction covers these eight tests. Calibration changes only timing predictions for this sensitivity, not the main generator.",[46*mm,8*mm,29*mm,26*mm,26*mm,31*mm])
    a.p("With the common MLP, initiation MAE is 41.4 versus 45.8 ms at n=3 and 42.7 versus 47.9 ms at n=8, favouring CVAE; movement-time differences are not significant. The smaller gaps show that the original timing comparison was partly sensitive to head choice. CVAE codes were also learned with timing supervision whereas spline codes were not, so this does not isolate representation geometry alone. Validation calibration is an MAE sensitivity, not a test-set oracle, a smearing correction or evidence assigning a percentage of the advantage to a specific cause.")
    a.h("5.3 Participant-level behavioural prediction",True)
    a.p("A separate validation-tuned Ridge probe predicts 14 query-summary targets (mean and SD of seven behaviours) from context fingerprints. It is trained on training participants, selects regularization on validation participants and refits on both before testing. The comparison constant is the original 17-training-participant mean. Table 5 includes negative results. Pooled out-of-fold R<super>2</super> uses all 28 held-out summary predictions per seed; mean fold R<super>2</super> averages seven-participant scores and is not the same statistic.")
    a.table(probe_rows(tables["probes"]),"<b>Table 5.</b> CVAE mean-fingerprint probes, averaging each stated R-squared statistic across available seeds. Speed-peak count is the heuristic feature, not minimum-jerk order.",[58*mm,27*mm,27*mm,27*mm,27*mm])
    a.p("For n=3, pooled R<super>2</super> is 0.723 for mean initiation time and 0.356 for mean peak speed. Their MAEs are 14.8 ms versus a 32.2 ms training-mean baseline, and 7.74 versus 12.08 tracker units/s, respectively. Ten of 14 pooled R<super>2</super> values are negative; 12 of 14 mean fold scores are negative. Thus broad recovery of a participant's behavioural distribution parameters is not established. All model-family probes and available mean-plus-SD ablations are included in the downloadable tables; adding SD increases fingerprint dimension and is not equivalent to a two- or three-number interface.")
    a.h("5.4 Conditioning, clustering and latent associations",True)
    k=tables["kmeans"].set_index("representation")
    a.p(f"The in-sample trajectory K-means analysis gives ARI={k.loc['trajectory','ari']:.3f}, with permutation p={k.loc['trajectory','permutation_p']:.5f} (200 permutations; minimum resolution 1/201). This indicates weak non-random structure, not clean subject clusters. Context enrollment is a distinct held-out task (Table 2). Neither result guarantees that latent means preserve all participant-specific variability.")
    a.figure(figures["condition"],"<b>Figure 4.</b> Matched condition-stratum trajectory diagnostic with fixed fingerprints and latent draws. Holm-adjusted p=1.000 (n=3) and 0.132 (n=8); none of 44 feature-level comparisons survives BH correction. This diagnostic does not prove that the decoder ignores its condition input.",75*mm)
    a.p("The dashboard's latent-feature heatmap is a within-run Spearman association under a specified sampling distribution and fixed task condition. Such correlations are descriptive, can reflect correlated latent draws, and do not establish causal control or coordinate identity across seeds. Figure A4 provides a reproducible example. We do not equate an axis with a cognitive strategy.")
    a.h("6 Minimum-jerk analysis")
    a.p("As a secondary kinematic analysis, recorded and generated movement velocities are fitted with one to four overlapping planar minimum-jerk components [5]. Each component has onset, duration and x/y displacement. This is an adaptation of Prof. Friedman's repository [6], not an unchanged execution: optimization, bounds and the tangential-speed objective differ. The primary model-order rule selects the smallest order with normalized error at most 0.05, then at most 0.10, otherwise the minimum-error candidate. BIC is an alternative. Component number is not a validated classification into wait-then-go versus corrective strategy.")
    a.figure(figures["submovement"],"<b>Figure 5.</b> Component counts, generated count discrepancy and sensitivity to fitting assumptions. The generated analysis uses only 10 samples per participant per seed at n=3 and n=8 (1,680 generated fits overall).",78*mm)
    a.p("Generated count total variation is 0.353 (n=3) and 0.355 (n=8); none of nine paired dimension comparisons survives BH correction. On 56 sampled recorded trials, additional optimizer restarts preserve threshold-selected counts in 100%, but the 167 ms constraint and 5 Hz filter preserve them in only 55.4% and 51.8%. Counts therefore depend materially on analysis settings.")
    sample=tables["sampling"].query("n_generated == 10")
    rows=[["Discrepancy","Observed n=3","Observed n=8","Sampling reference mean"]]
    names={"ks_mj_fit_error":"Fit error KS","ks_mj_first_duration_s":"First duration KS","ks_mj_first_amplitude":"First amplitude KS",
           "ks_mj_secondary_amplitude_fraction":"Secondary amplitude fraction KS","ks_mj_mean_overlap_pct":"Overlap KS","count_total_variation":"Count total variation","count_jsd":"Count JSD"}
    for key,label in names.items():
        s=sample[sample.metric==key].set_index("latent_dim")
        rows.append([label,number(s.loc[3,'observed_at_n10']),number(s.loc[8,'observed_at_n10']),number(s.loc[3,'reference_mean'])])
    a.table(rows,"<b>Table 6.</b> Empirical finite-sample reference at the actual generated sample size. Each of 500 replicates draws two independent samples from each participant's empirical query distribution, matches sample sizes, and averages three seed-like draws before pooling participants. This is a plug-in diagnostic, not a formal null p-value or hard KS floor.",[67*mm,30*mm,30*mm,39*mm])
    a.p("The empirical reference accounts for a substantial part of finite-sample KS discrepancy, but observed component-duration, amplitude, overlap and count differences remain above it. This supports a restricted conclusion of incomplete component fidelity. An n=120 resampling reference is supplied for planning; it is not an evaluation of 120 generated minimum-jerk fits and is not substituted for the n=10 comparison.")
    a.h("7 Limitations and conclusions")
    a.p("The strategy-inclusive CVAE is a useful experimental model, with lower distribution distances than spline at matched n and stronger capacity at n=8. Spline remains the stronger deterministic shape compressor. A common-head comparison retains a smaller initiation-time advantage, but does not establish superiority on movement timing or isolate the latent representation from its training objective. Broad prediction of participant-level distribution parameters, a uniquely beneficial task condition, and stable interpretable strategy coordinates are not established.")
    a.p("The 357 zero-initiation labels (7.5%) occur because the post-go threshold is already satisfied; none is a missing-onset fallback. Among these, 109 have at least 50 ms of continuous pre-go threshold exceedance in filtered planar speed, and 100 do in raw planar speed. These observations cannot be dismissed as noise, but do not establish anticipation because event alignment and zero-phase filtering require review. The 74.6% share sometimes associated with these trials is a share of centered log-target variance, not measured trained-model loss or proof of the cause of timing extrapolation. Figure A2 discloses the original timing tails, including CVAE initiation predictions reaching about 11.5 s; no post-hoc clipping is used to improve the benchmark.")
    a.p("No learning-curve experiment in this report establishes that sample size caused any model failure. More participants may help, but this remains a hypothesis. The four-fold evaluation measures performance within the available cohort, with repeated fitting rather than a fully independent replication. The report's distance reevaluation, event audit, behavioural-probe aggregation, common timing heads and sampling-reference analysis are post-review analyses. They reuse frozen main checkpoints and do not alter the canonical labels or exclusions. The source training records include dirty-worktree flags for two model families; checkpoint hashes and per-run source provenance are retained, rather than claiming every fit arose from a clean commit.")
    a.h("7.1 Points requiring domain confirmation",True)
    for text in [
        "1. Confirm that the target-motion event, rather than appearance, is the permitted movement cue, and confirm its alignment with tracker frames and MAT pressedTime. Should the observed pre-go motion change trial validity or onset measurement?",
        "2. Confirm whether the condition-2 fixation warning should affect inclusion, and whether the 1 s late-arrival cutoff is appropriate. These trials have not been silently removed.",
        "3. Confirm tracker units, the x-y plane, onset threshold and 10 Hz filtering. Advise on minimum-jerk duration bounds, candidate orders and order criterion before any cognitive interpretation.",
    ]: a.p(text)
    a.p("The immediate next step is to resolve these acquisition and analysis assumptions, then rerun affected stages. A different latent architecture or clinical interpretation is not required for the current workshop objective. We welcome comments on the modeling question, evaluation and scope of the conclusions.")
    a.h("Acknowledgments and reproducibility")
    a.p("We thank Moni Shahar for guidance on the project design and Prof. Jason Friedman for the experiment, data and methodological advice. AI assistance was used in software development and report preparation. Responsibility for verification and interpretation remains with the authors. Original outputs are preserved separately from the versioned corrected evaluation; Appendix B identifies the current artifacts.")
    a.h("References")
    refs=[
        "[1] Brenner E, Smeets JBJ. Continuously updating one's predictions underlies successful interception. <i>J Neurophysiol.</i> 2018;120:3257-3274. doi:10.1152/jn.00517.2018.",
        "[2] Slowinski P et al. Dynamic similarity promotes interpersonal coordination in joint action. <i>J R Soc Interface.</i> 2016;13:20151093. doi:10.1098/rsif.2015.1093.",
        "[3] Kingma DP, Welling M. Auto-Encoding Variational Bayes. ICLR, 2014. arXiv:1312.6114.",
        "[4] Sohn K, Lee H, Yan X. Learning Structured Output Representation using Deep Conditional Generative Models. NeurIPS 28, 2015.",
        "[5] Flash T, Hogan N. The coordination of arm movements: an experimentally confirmed mathematical model. <i>J Neurosci.</i> 1985;5:1688-1703. doi:10.1523/JNEUROSCI.05-07-01688.1985.",
        "[6] Friedman J. submovements. github.com/JasonFriedman/submovements, repository revision 9c2f40c. Adaptation details are recorded in the accompanying code.",
    ]
    for ref in refs: a.p(ref,"caption")
    a.page(); a.h("Appendix A: Representative outputs and diagnostic figures")
    a.figure(figures["examples"],f"<b>Figure A1.</b> Held-out posterior-mean reconstructions and context-fingerprint generation. CVAE n=8, fold 0, seed 42; typical trial {escape(example_meta['typical_trial'])}; high-error trial {escape(example_meta['difficult_trial'])}. Examples were chosen using reconstruction-error quantiles and do not replace all-participant evaluation.",100*mm)
    a.figure(figures["timing"],"<b>Figure A2.</b> Original CVAE timing-error quantiles and maxima. Small median absolute errors can coexist with negative R-squared when a few errors are very large.",75*mm)
    a.figure(RESULTS/"event_audit/pre_go_examples.png","<b>Figure A3.</b> Event-audit examples with raw and filtered motion around the go event. Selection is deterministic within the audit groups; threshold exceedance is not a diagnosis of premature movement.",145*mm)
    if "associations" in figures:
        a.figure(figures["associations"],"<b>Figure A4.</b> Decoder association heatmap, CVAE n=3, fold 0, seed 42. 500 draws from the training-centred shared latent covariance; fixed sp=2, side=1 and target speed 0.635 screen units/s. Spearman correlations are within this generated sample, not independent tests of causal control or human strategy.",90*mm)
    a.h("Appendix B: Complete comparison and accompanying artifacts")
    a.table(benchmark_rows(tables["oof"],fingerprints,True),"<b>Table B1.</b> Complete dimension sweep. See Table 2 for metric definitions.",[33*mm,8*mm,17*mm,23*mm,23*mm,18*mm,21*mm,23*mm])
    a.p("The ZIP contains the scientific report, a separate student results guide, one Streamlit dashboard and machine-readable tables. The dashboard uses strategy-window CVAE checkpoints at n=2,3,4,8 (fold 0, seed 42); benchmark tables use all four folds and all saved seeds. Live generation is an illustrative checkpoint, not an ensemble. Minimum-jerk validation is available only for n=3 and n=8.")
    a.p("Extract the ZIP, install requirements_dashboard.txt in a Python environment, then run <font name='Courier'>python -m streamlit run src/confirmatory_dashboard.py</font> from the extracted folder. In Generate, select n and a context-enrolled participant, then adjust latent values and task condition. Held-out validation shows empirical/query comparisons; Benchmarks separates reconstruction and distribution metrics; Latent associations shows the current decoder's correlations; Diagnostics exposes timing, conditioning and secondary fitting limits. Condition controls are exploratory, not validated causal interventions.")
    a.p("Current corrected tables are under studies/review_corrected_evaluation/results: analysis (full 56 paired comparisons, feature-wise metrics, seed variation and timing tails); behavioral_probes (all available mean and mean-plus-SD scores and predictions); timing_fairness (validation factors, common-head predictions and paired tests); event_audit (all 4,732 trial rows); and sampling_reference (matched-size reference distributions). Unchanged training checkpoints and minimum-jerk fits remain under studies/final_strategy_evaluation. Small review tables and the dashboard assets accompany the compact ZIP; raw recordings and the full training matrix are not included.")
    a.finish()


def guide_report(output, tables, fingerprints, figures):
    a=Article(output,"Reading the Interception-Movement Results",
        "Companion guide for Simaan and Paz<br/>Figure and table numbers refer to the scientific report")
    a.h("1 The question we actually tested")
    a.p("Can recordings from part of a new participant's session provide a small fingerprint that generates the distribution of the rest of that session? 'New participant' means excluded from training, not a person for whom we have no recordings. We still need context trials to enroll that person. This is not a disease predictor, a proof of psychological strategy, or an automatic explanation of each latent coordinate.")
    a.p("There are three separate tasks. Reconstruction receives the actual trial and reproduces its shape. Timing prediction also receives that trial's shape, but not its physical timing labels. Generation receives only the participant's context fingerprint plus the task conditions, not the query trajectory. Always identify which task a result belongs to before saying a model is better.")
    a.h("2 What enters the encoder (Figure 1)")
    a.p("Marker 5 is appearance. The object then waits 0.20-0.50 seconds before it moves. Our time zero is that later motion event. The 100 x 2 input contains lateral and forward finger position from target motion to the last recorded sample, including any waiting after the go signal. It is not 100 raw 240-Hz frames: resampling gives 100 phase points spanning a trial of variable duration. Filtering reduces high-frequency noise; it is not a guarantee that everything left is intentional movement.")
    a.p("The encoder also gets start category, side and exact target speed. Start category couples distance and a speed range; side tells which side the target starts on. Neither is participant identity. Outcome, repetition, initiation seconds and movement seconds are not encoder inputs. A separate decoder head predicts those two physical times so the path can be assigned a time scale.")
    a.p("The final tracker sample is an arrival proxy, about 26 ms earlier than the MAT arrival timestamp on average. Movement onset is a sustained speed threshold, not an observed decision. Some people are already above that threshold at the go event: 357 labels are zero. We retain these cases for Jason to assess rather than silently calling them noise or removing them. The encoder still starts at target motion.")
    a.h("3 What a fingerprint means")
    a.p("Each context trajectory produces a posterior mean vector. We average those vectors within one participant to get theta. Averaging is a chosen summary, not something a VAE automatically proves to be meaningful. It can discard variability. The VAE's KL penalty encourages a regular latent representation but does not give every coordinate a named physical meaning or guarantee that the mean is a typical trial.")
    a.p("To generate, we add training-derived latent variation around theta and decode with a task condition. All participants share the latent noise covariance in this design. The nonlinear decoder can still give different output spreads at different centres. We do not learn a separate full latent probability distribution for every unseen participant. The requested two or three controls are the centre coordinates; n=8 is the more flexible comparison.")
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
    a.figure(figures["benchmark"],"Scientific report Figure 2: separate tasks, separate metrics.")
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
    a.p("Our participant probe has one query-summary value per held-out person. A fold contains seven such values. A very small between-person variance within one fold makes its R-squared unstable. Pooling all 28 out-of-fold predictions changes the denominator. This explains why Table 5 includes both pooled R-squared and the average seven-person fold score; they cannot be substituted for each other.")
    a.p("Mean initiation time is a comparatively successful n=3 probe: pooled R-squared 0.723, average fold R-squared 0.661, MAE 14.8 ms versus 32.2 ms for the training-mean constant. But ten of fourteen pooled scores are negative. The honest conclusion is selective predictive information, not that the fingerprint explains all visible distributions. MAE may improve even when R-squared is negative because it penalizes errors differently.")
    a.h("8 Fair comparisons and significance (Tables 3 and 4)")
    a.p("Each paired test has 28 rows: one seed-averaged error per held-out participant for each model. A two-sided Wilcoxon test asks whether paired differences systematically favour one direction. BH correction controls the false discovery rate across the stated family of comparisons; it does not say that a particular scientific claim has a 95% probability of being true. No significant difference is not proof of equivalence.")
    a.p("The original timing comparison gave CVAE a nonlinear head and spline a linear head. We added a common small MLP for both frozen representations. At n=3, the initiation gap shrank to 41.4 versus 45.8 ms; corrected p=0.0076. At n=8 it is 42.7 versus 47.9 ms; p=0.0423. Movement-time differences under the common head were not significant. CVAE's code had already been trained with timing labels, so this remains a comparison of complete representation-learning choices.")
    a.p("We also fitted a multiplicative timing calibration using validation participants only. This is not an oracle chosen on the test group. It helps spline initiation MAE, especially at n=3, but does not remove the CVAE advantage in that sensitivity. Neither check proves a specific percentage of the original gap was caused by retransformation bias.")
    a.h("9 Enrollment, clustering and conditioning")
    a.figure(figures["capacity"],"Scientific report Figure 3: descriptive capacity trend; error bars are not a confidence interval for a new cohort.",80*mm)
    a.p("Enrollment classifies query trial codes against seven candidate context means. About 40.9% at n=3 and 50.7% at n=8 exceed nominal 14.3% chance, but this is not classification among all 28 people or people with no enrollment recordings. K-means is different: it clusters the available representations without those labels, then ARI/NMI compare assignments with identity. The small ARI says the clusters are not clean subject groups, even if the permutation result is non-random.")
    a.p("The unconditional VAE often performs similarly to CVAE, with some endpoint-specific differences. The dedicated condition-stratum test found no advantage after its corrections. Do not say that condition sliders are proven to reproduce causal human responses. The decoder may respond to the sliders, but that response still needs validation against task-specific changes in people.")
    a.h("10 Submovements and heatmaps")
    a.p("A smooth minimum-jerk component is a fitted elementary movement pulse with a start time, duration and planar displacement. Several overlapping pulses can approximate a velocity curve. This is not a direct measurement of separate decisions. The speed-peak count in the main feature table is a simpler peak-detection heuristic and must never be renamed minimum-jerk count.")
    a.figure(figures["submovement"],"Scientific report Figure 5: the fitting rule materially changes component counts.",85*mm)
    a.p("We fitted only ten generated examples per person per seed for the expensive decomposition. Even two samples from the same underlying distribution can have a sizable KS at that size. Our empirical resampling reference is about 0.267 for first-component duration, compared with observed 0.450 at n=3 and 0.401 at n=8. Count total variation is about 0.151 in the reference but 0.353/0.355 observed. Therefore sampling noise matters, but does not explain all the mismatch. The reference is not a mathematical floor or formal hypothesis-test p-value.")
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
    a.p("Start with scientific Tables 2-5, then Figures A1-A4. Use the dashboard's Generate section for an illustrative n and participant; use Benchmarks for all-fold numbers. Live trajectories use fold 0, seed 42 only. The corrected study's results folder contains exact rows behind every comparison, including unsuccessful probes. The report's Appendix B lists their locations and the ZIP launch command.")
    a.p("Before presenting any graph, be able to state: what entered the model, what was withheld, what each plotted point represents, what the comparator is, the formula/unit, whether lower or higher is preferable, and what the result does not establish. For a number such as R-squared, also identify its denominator and observational unit. That is a stronger explanation than memorizing whether the number looked good.")
    a.finish()
