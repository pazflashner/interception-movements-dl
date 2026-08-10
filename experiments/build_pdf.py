import sys, os
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, ListFlowable, ListItem)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

FIG = os.path.join(REPO, "figures")
OUT = os.path.join(REPO, "Interception_Preliminary_Results.pdf")

NAVY = colors.HexColor("#1A2B4A"); BLUE = colors.HexColor("#0277BD")
GREY = colors.HexColor("#607D8B")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=15, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=BLUE, fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
CAP = ParagraphStyle("CAP", parent=ss["BodyText"], fontSize=8.5, textColor=GREY, alignment=TA_CENTER, spaceAfter=10)
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], textColor=NAVY, fontSize=20, spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["BodyText"], textColor=BLUE, fontSize=12, alignment=TA_CENTER, spaceAfter=2)
META = ParagraphStyle("META", parent=ss["BodyText"], textColor=GREY, fontSize=9, alignment=TA_CENTER, spaceAfter=1)
BULLET = ParagraphStyle("BULLET", parent=BODY, spaceAfter=2)

def P(t): return Paragraph(t, BODY)
def bullets(items): return ListFlowable([ListItem(Paragraph(i, BULLET), leftIndent=6) for i in items],
                                        bulletType="bullet", start="•", leftIndent=12)
def fig(name, caption, width=6.6*inch):
    path = os.path.join(FIG, name)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    im = Image(path, width=width, height=width*h/w)
    return [im, Paragraph(caption, CAP)]

story = []

# ── Title ──
story += [Spacer(1, 0.3*inch),
          Paragraph("Models of Interception Movements", TITLE),
          Paragraph("From Raw Trajectories to Individual Behavioural Signatures", SUB),
          Spacer(1, 6),
          Paragraph("Preliminary Results", ParagraphStyle("x", parent=SUB, textColor=GREY, fontSize=11)),
          Spacer(1, 4),
          Paragraph("Seman Libbiss &amp; Paz Flashner  ·  Workshop on Deep Learning", META),
          Paragraph("In collaboration with Prof. Jason Friedman, Tel Aviv University", META),
          Spacer(1, 12)]

# ── 0. Overview ──
story += [Paragraph("1. Overview and goal", H1),
    P("The goal of this project is to compress each interception movement into a small set of "
      "numbers &mdash; a low-dimensional latent code &mdash; that can (a) reconstruct the movement and "
      "(b) act as an individual &ldquo;fingerprint&rdquo; whose aggregate over a subject&rsquo;s trials predicts "
      "their behaviour. We build up through three phases: a K-Means clustering check (is any subject "
      "structure present at all?), a polynomial-spline baseline (a non-machine-learning reference), and "
      "a Conditional Variational Autoencoder (CVAE), the core model. Learned-model results are evaluated "
      "on <b>7 subjects held out entirely from training</b> and averaged over 10 random train/test splits "
      "(seeds); the K-Means check and the single-model latent-interpretation analysis are single-run and "
      "flagged as such."),
    P("<b>Headline.</b> The CVAE compresses phase-normalised interception trajectories together with "
      "explicit timing variables and reconstructs them on held-out subjects. Subject information is present "
      "(unseen-subject identification ~64% vs 14% chance), but the current per-trial latent plus post-hoc "
      "averaging do not yet provide a validated low-dimensional fingerprint that reproduces unseen "
      "behavioural distributions. That is a legitimate preliminary result &mdash; see the caveats in "
      "Section 12.")]

# ── 2. Data ──
story += [Paragraph("2. Data and trial segmentation", H1),
    P("We analyse the free eye-movement condition (condition 2): 3D finger trajectories at 240 Hz. "
      "The target appears, holds still for a randomised foreperiod (0.18&ndash;0.48 s), then "
      "<b>starts moving</b> (the go-signal); the participant may only move after that. The trajectory fed "
      "to the model runs from <b>movement onset to arrival</b>: movement onset is found by a "
      "<b>velocity threshold</b> after the go-signal, and the endpoint is the recorded interception (an "
      "event, not a threshold). The pre-movement waiting period is therefore <b>not</b> inside the "
      "trajectory &mdash; it is stored separately as the reaction-time scalar (movement onset minus "
      "go-signal), which keeps the randomised foreperiod out of that measure. Each movement is low-pass "
      "filtered (10 Hz), resampled to 100 frames, and origin-aligned. (Timing is currently taken from the "
      "frame index; frame-counter gaps affect ~1/3 of trials by one step, &lt;1.5%, and will be switched to "
      "frame-counter deltas.)"),
    P("From 4,763 condition-2 trials we retain <b>4,684 (98.3%)</b>, dropping 48 &ldquo;too early&rdquo; "
      "(moved before the go-signal), 27 that arrive &gt;1 s after the target&rsquo;s window (disengagement), "
      "and 4 timeouts. Fixing the movement-end to arrival also removed an artefact in which late sensor "
      "jitter had stretched a 0.5 s reach to as long as 9.7 s; the maximum movement time is now 1.36 s. "
      "Retained trials per subject are ~175&ndash;180 for most, but only 57 and 28 for two subjects "
      "(subjects 02 and 41); training is not yet subject-balanced.")]

# ── 3. K-Means ──
story += [Paragraph("3. Phase 1 &mdash; K-Means baseline: is the signal even there?", H1),
    P("K-Means is a classical clustering algorithm: it groups trials by geometric similarity, with no "
      "notion of &ldquo;subject.&rdquo; We check whether the clusters line up with the true subjects "
      "(Adjusted Rand Index, ARI: 1.0 = perfect match, 0 = random)."),
    Table([["Representation", "ARI", "NMI"],
           ["Raw trajectories (300-dim)", "0.093", "0.292"],
           ["Kinematic features (11-dim)", "0.104", "0.294"]],
          colWidths=[3.2*inch, 1.4*inch, 1.4*inch]),
    Spacer(1, 6),
    P("<b>Interpretation.</b> Both scores are weak but above chance (0) &mdash; weak, above-chance subject "
      "structure that a simple algorithm barely recovers. We read this as <i>motivating</i> a learned "
      "model, not as formal proof of a fingerprint: k is chosen against the true labels with no permutation "
      "test, and K-Means uses no train/test split. It says subject structure exists but is hard to extract "
      "with a simple method.")]

# ── 4. Spline ──
story += [Paragraph("4. Phase 2 &mdash; Polynomial spline baseline", H1),
    P("The spline is pure mathematics (no learning): we fit a smooth cubic curve to each trajectory. "
      "It copies the <i>shape</i> almost perfectly &mdash; a per-trial fit reaches a reconstruction MSE "
      "of ~0.0002, an interpolation ceiling &mdash; and even a capacity-matched version (Spline+PCA, "
      "reduced to the same number of dimensions) reconstructs geometry <b>better</b> than the CVAE at low n "
      "(~0.24 vs ~0.33 at n=3, 10 seeds). So the spline is the stronger low-dimensional geometric "
      "reconstructor. The CVAE&rsquo;s value is different in kind: it provides a structured, continuous latent "
      "space (needed to average trials into a fingerprint) and compresses trajectory and timing jointly. "
      "Crucially, <b>neither</b> representation yet generates realistic subject-level distributions "
      "(Section 6). The takeaway is that reconstruction MSE alone is the wrong single scoreboard for a "
      "fingerprinting project &mdash; not that the CVAE is uniformly more accurate.")]

# ── 5. CVAE ──
story += [Paragraph("5. Phase 3 &mdash; the Conditional VAE (our model)", H1),
    Paragraph("What it is and what it outputs", H2),
    P("The CVAE has an encoder that squeezes a trajectory into n latent numbers, and a decoder that "
      "rebuilds it. We chose a <b>variational</b> autoencoder (not a plain one) because its latent space "
      "is smooth and continuous &mdash; which is what makes averaging a subject&rsquo;s trials into one "
      "meaningful fingerprint valid. It is <b>conditional</b>: we feed the task facts (start position, "
      "side) separately, so the latent is free to encode personal style rather than the task. The decoder "
      "has two outputs: <b>the full reconstructed trajectory</b> (its main job) <b>and a timing head</b>. "
      "<b>Important nuance:</b> the encoder is <i>given</i> the movement and reaction times as inputs and the "
      "decoder reconstructs them, so the R&sup2; below is <b>timing reconstruction</b> (how efficiently the "
      "latent compresses the two timing numbers) &mdash; <b>not</b> prediction of timing from trajectory "
      "shape. The timing term is up-weighted to per-dimension parity with the 300 trajectory values, so its "
      "prominence in the latent is partly imposed by the objective. Carrying timing this way follows Prof. "
      "Friedman&rsquo;s suggestion, so a generated code yields a shape <i>and</i> its durations."),
    Paragraph("Latent-dimension sweep", H2),
    P("We swept the fingerprint size n over {2, 3, 4, 8, 16}, 10 seeds each.")]
story += fig("latent_sweep.png",
             "Figure 1. As n grows, reconstruction improves, timing reconstruction saturates around the n=8 "
             "elbow (R&sup2; ~ 0.98), and the behavioural fingerprint peaks around n=8.")
story += [Table([["n", "Recon MSE", "Timing R\u00b2 (move)", "Timing R\u00b2 (react)", "Behav. features > chance (of 11)"],
                 ["2", "0.51", "0.71", "0.81", "3.0"],
                 ["3", "0.33", "0.83", "0.90", "4.2"],
                 ["4", "0.20", "0.88", "0.95", "4.5"],
                 ["8", "0.09", "0.98", "0.99", "5.2"],
                 ["16", "0.07", "0.98", "0.99", "4.8"]],
                colWidths=[0.5*inch, 1.1*inch, 1.4*inch, 1.4*inch, 2.2*inch]),
    Spacer(1, 6),
    P("<b>What we chose and why.</b> We report all n. <b>n = 8 is the sweet spot</b>: reconstruction and "
      "timing saturate there (R&sup2; ~ 0.98) and the behavioural fingerprint also peaks (~5 of 11 "
      "features), with no gain at n = 16. We nonetheless take <b>n = 3 as the headline fingerprint</b> "
      "for presentation &mdash; the smallest size that is directly visualizable, already capturing timing "
      "at R&sup2; ~ 0.83&ndash;0.90 (reconstruction). What the CVAE offers over the spline is a structured, "
      "continuous latent space (needed to average trials into a fingerprint) and joint trajectory+timing "
      "compression &mdash; not uniformly better accuracy (see Sections 4 and 6).")]

story += [Paragraph("What each latent variable controls", H2),
    P("Because n=3 is small, we can ask what each latent number does by correlating it with movement "
      "features (Figure 2). All three axes turn out to encode <b>timing and speed</b> (reaction time, "
      "movement time, peak speed) in different combinations; none cleanly isolates spatial shape such as "
      "curvature. So the latent is interpretable but timing-centric &mdash; though note this is partly "
      "<i>imposed</i>: timing is fed to the encoder and up-weighted, so its dominance is not purely "
      "discovered from the data. (Single n=3 model.) It still points to spatial / strategy features "
      "(e.g. number of sub-movements) as something a future version could encode more explicitly.")]
story += fig("latent_interpretation.png",
             "Figure 2. Correlation of each latent (z0&ndash;z2) with movement features. Red = positive, "
             "blue = negative. The axes are timing/speed-dominated.")

# ── 6. Fingerprint ──
story += [Paragraph("6. Subject fingerprints", H1),
    P("A subject <b>fingerprint</b> is the aggregate of that subject&rsquo;s trial latent codes &mdash; the "
      "mean (who they are on average) plus the spread (trial-to-trial variability). The most robust test of "
      "it: build a fingerprint from half of an <i>unseen</i> subject&rsquo;s trials and classify the other "
      "half by nearest fingerprint. By this measure the fingerprint identifies held-out subjects at "
      "<b>~64% at n=8 (chance 1/7 ~ 14%)</b> &mdash; so it carries real, transferable subject signal.")]
story += fig("dashboard_inference.png",
             "Figure 3. One subject&rsquo;s 180 trials (blue) and their fingerprint (red X). The wide cloud "
             "&mdash; within-subject variability larger than between-subject differences &mdash; is why clean "
             "separation is hard.")
story += [P("<b>But the signal is limited, and we are careful not to over-claim.</b> A regression probe from "
      "the fingerprint to behavioural features is only <i>indicative</i> here: it is trained "
      "leave-one-subject-out on the 7 test subjects (6 per fold, up to 32 predictors), so it is "
      "underpowered; a properly-powered probe (fit on the 17 training subjects, evaluated once on 7) is a "
      "pending refinement. Subjects also do not separate cleanly (Figure 3)."),
    P("We do <b>not</b> claim this is proven to be a pure data limitation. Weak separation could stem from "
      "several causes we have not disentangled: the small subject count (28); the <b>per-trial</b> training "
      "objective with post-hoc averaging (the model is never trained on a subject-level variable); timing "
      "dominating the latent; unmodelled trial-order / session effects (trial number is not given to the "
      "model); or the architecture. Varying latent size, architecture (MLP/CNN), conditioning, and loss "
      "(Sections 10&ndash;11) did not overcome it &mdash; but a <b>hierarchical VAE</b> with explicit "
      "subject- and trial-level latents, the design that directly targets this goal, remains untested and is "
      "the most promising next step.")]

# ── 7. Dashboard ──
story += [Paragraph("7. The dashboard (deliverable)", H1),
    P("We built an interactive tool on the trained model with two modes. <b>Exploration</b> lets a "
      "researcher move the latent sliders and watch a movement (and its timing) be generated &mdash; a way "
      "to see what the fingerprint space contains. <b>Inference</b> takes a subject&rsquo;s trials and reads "
      "out their fingerprint (mean + spread), reproducing Figure 3 for any uploaded subject.")]
story += fig("dashboard_exploration.png",
             "Figure 4. Exploration mode: four latent settings produce four different generated movements, "
             "each with its own movement and wait time.")

# ── 8. Proposal vs. actual ──
story += [PageBreak(), Paragraph("8. Where we followed the proposal, and where we deviated", H1),
    P("For transparency, this compares what the submitted proposal said we would do against what we did.")]
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.5, leading=10.5, alignment=0, spaceAfter=0)
CELLH = ParagraphStyle("CELLH", parent=CELL, textColor=colors.white)
def C(t, hdr=False): return Paragraph(t, CELLH if hdr else CELL)
dev = [[C("Proposal item", True), C("Status", True), C("Note", True)],
       [C("K-Means baseline (2 representations)"), C("Done"), C("Raw trajectory + kinematic features.")],
       [C("Spline baseline (cubic, few knots)"), C("Done"), C("As specified.")],
       [C("CVAE + timing head"), C("Done"), C("Encoder receives, and decoder reconstructs, movement &amp; reaction time.")],
       [C("Latent sweep n = 2,3,4,8,16"), C("Done"), C("Full sweep; headline n=3, elbow n=8.")],
       [C("Eval: MSE, Spearman, probing R\u00b2, KS, MMD/energy"), C("Done"), C("All implemented (probe is underpowered &mdash; see Section 12).")],
       [C("Dashboard (Inference + Exploration)"), C("Done +"), C("Extended to multi-trial subject fingerprints.")],
       [C("Segmentation of the movement window"), C("Refined"), C("Trajectory window is movement onset "
        "(velocity threshold) to arrival; the reaction time (go-signal to onset) is stored separately. "
        "Refined beyond the proposal after inspecting the raw data.")],
       [C("Trial filtering"), C("Added"), C("Drop too-early / &gt;1 s-late / timeout via the .mat labels "
        "(not in the proposal). &lsquo;Not fixating&rsquo; trials retained (see Q1).")],
       [C("Benchmark vs Friedman sub-movement pipeline"), C("Not done"), C("External dependency not yet available; pending (see questions).")],
       [C("Fallback to a supervised predictor"), C("Not needed"), C("The VAE converged; fallback not triggered.")]]
t = Table(dev, colWidths=[2.1*inch, 0.7*inch, 3.8*inch])
t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), NAVY),
                       ("VALIGN", (0,0), (-1,-1), "TOP"),
                       ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CCCCCC")),
                       ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F0F4F8")]),
                       ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
story += [t]

# ── 9. Methods ──
story += [Paragraph("9. Methods summary", H1),
    bullets([
      "<b>Data:</b> condition 2, 4,684 trials, 28 subjects; 3D finger position at 240 Hz.",
      "<b>Preprocessing:</b> velocity-threshold movement onset to arrival; reaction time (go-signal to "
      "onset) stored separately; 10 Hz Butterworth low-pass; resample to 100 frames (cubic spline); "
      "origin normalisation.",
      "<b>Split:</b> leave-N-subjects-out, 17 train / 4 validation / 7 test.",
      "<b>Model:</b> Conditional VAE, encoder/decoder MLP (hidden 256), condition = start position + side; "
      "the encoder receives movement &amp; reaction time and the decoder reconstructs them (timing head); "
      "KL annealing; standard-ELBO loss. A selectable 1-D CNN variant exists (Section 10).",
      "<b>Baselines:</b> K-Means (trajectory &amp; feature), polynomial spline (per-trial &amp; Spline+PCA).",
      "<b>Evaluation (7 held-out subjects):</b> reconstruction MSE; timing R&sup2;; latent&ndash;feature Spearman "
      "correlations; behavioural probing (linear + SVR, leave-one-subject-out); generative fidelity "
      "(KS, MMD, energy distance).",
      "<b>Tools:</b> Python, PyTorch, scikit-learn, NumPy/Pandas, Matplotlib, Streamlit.",
    ])]

# ── 10. Architecture & feature comparison ──
story += [Paragraph("10. Architecture &amp; feature comparison (results)", H1),
    P("We implemented and tested the three natural extensions against the baseline, across all latent "
      "sizes and 10 seeds (Figure 5): a <b>1-D convolutional</b> encoder/decoder (which reads the "
      "trajectory as a time-ordered signal rather than an unordered vector), conditioning on the "
      "<b>exact target speed</b> (rather than only its range), and adding the <b>sub-movement count</b> "
      "(peaks in the speed profile) as a feature and probing target.")]
story += fig("variant_comparison.png",
             "Figure 5. Four variants across latent size (mean ± sd, 10 seeds). The CNN (orange/red) "
             "leads on timing; reconstruction and the fingerprint are essentially unchanged.")
story += [bullets([
      "<b>The CNN improves timing reconstruction</b> consistently (movement-time R² up by 0.03&ndash;0.08; "
      "it reaches ~0.99 by n=4&ndash;8, where the MLP needs n=8+). Respecting frame order helps compress "
      "the temporal axis.",
      "<b>The CNN does not reconstruct trajectories better</b> &mdash; slightly worse at small n, equal at "
      "large n &mdash; and does not improve the fingerprint.",
      "<b>Exact target speed has no measurable effect</b> on any metric: it duplicates the speed range "
      "already carried by the condition. (Note: this value is screen pixels/s, not physical mm/s; used "
      "standardised.)",
      "<b>No variant&rsquo;s fingerprint predicts our sub-movement count</b> (a <i>find_peaks</i> heuristic "
      "on the speed profile, not Prof. Friedman&rsquo;s decomposition): near-zero / negative R² for all "
      "variants on unseen subjects.",
      "<b>The fingerprint does not improve under any architecture or feature</b> tried (Figure 5, right).",
    ]),
    P("<b>Conclusion.</b> We adopt the CNN as an optional, principled architecture for the "
      "timing / kinematic side; exact-speed conditioning and explicit sub-movement modelling are not "
      "worth their added complexity here. None of these changes improved the fingerprint &mdash; consistent "
      "with the other sections, though (as noted in Section 6) this does not by itself prove the cause is "
      "the data alone. More subjects, or a hierarchical subject-level model, are the most promising "
      "directions.")]

# ── 11. Loss-function experiments ──
story += [Paragraph("11. Loss-function experiments", H1),
    P("A natural question (raised by Paz) is whether a different training loss could produce "
      "better-separated fingerprints. We tested two changes against the standard ELBO, at n=3 and n=8, "
      "10 seeds (Figure 6): a <b>beta-VAE</b> (4x the KL weight, for a more structured latent) and a "
      "<b>discriminative</b> term that explicitly pulls a subject&rsquo;s trials together and pushes "
      "different subjects apart. We score them on <b>held-out fingerprint identification accuracy</b> &mdash; "
      "split each unseen subject&rsquo;s trials, build a fingerprint from one half, and check whether the "
      "other half lands nearest its own subject (chance = 1/7 ~ 14%).")]
story += fig("loss_comparison.png",
             "Figure 6. Loss comparison at n=8. The discriminative term inflates the training-set "
             "separation ratio but lowers held-out identification and collapses reconstruction and timing.")
story += [bullets([
      "<b>The baseline is already strong</b> by this metric: it identifies unseen subjects at "
      "<b>~64% at n=8</b> (vs 14% chance) &mdash; the fingerprint carries real, transferable subject signal.",
      "<b>beta-VAE: no change</b> (64.2% vs 63.7%).",
      "<b>The discriminative loss backfires:</b> it raises the training-set separation ratio "
      "(0.36 to 0.92) but that does not transfer &mdash; held-out identification <i>drops</i> to 51%, and "
      "reconstruction (MSE 5x) and timing (R&sup2; 0.97 to 0.40) collapse. It memorises the 17 training "
      "subjects rather than learning a transferable signature.",
    ]),
    P("<b>Conclusion.</b> Changing the loss did not improve the held-out fingerprint here. Forcing more "
      "separation only overfit the training subjects at the cost of the behavioural information. Together "
      "with Sections 5&ndash;6 and 10, no approach we tried lifted the fingerprint &mdash; but, as stated in "
      "Section 6, that does not prove the cause is the data alone; a hierarchical subject-level model is the "
      "untested lever most likely to help.")]

# ── 12. Caveats & known limitations ──
story += [Paragraph("12. Caveats and known limitations", H1),
    P("For an honest reading of the results above:"),
    bullets([
      "<b>Timing is reconstructed, not predicted.</b> The encoder is given movement and reaction time; the "
      "R² measures how well the latent compresses them, not that they can be inferred from trajectory shape.",
      "<b>The behavioural probe is underpowered.</b> It is fit leave-one-subject-out on the 7 test subjects "
      "(6 per fold, up to 32 predictors), so &ldquo;features above chance&rdquo; counts are indicative only; "
      "a probe fit on the 17 training subjects and evaluated once on 7 is the correct protocol and is pending.",
      "<b>Generative fidelity is in-sample.</b> Each test subject&rsquo;s own trials are observed to build "
      "the distribution being matched; it is not a from-a-limited-sample prediction test.",
      "<b>&ldquo;Data-limited&rdquo; is not proven.</b> The per-trial objective + post-hoc averaging, timing "
      "dominance, unmodelled trial-order effects (trial number is not a model input), and the untested "
      "hierarchical architecture are all candidate causes alongside the 28-subject count.",
      "<b>Exact speed is in screen pixels/s</b>, not the physical mm/s ranges; used only after standardisation.",
      "<b>Sub-movement count is a heuristic</b> (find_peaks on speed), not Prof. Friedman&rsquo;s decomposition.",
      "<b>Subject imbalance:</b> two subjects contribute 57 and 28 trials vs ~175 for the rest.",
      "<b>Timing uses the frame index;</b> frame-counter gaps affect ~1/3 of trials by one step (&lt;1.5%) "
      "and will move to frame-counter deltas.",
      "<b>Significance values are exploratory</b> (overlapping random splits, no multiplicity correction).",
    ])]

# ── 13. Questions ──
story += [Paragraph("13. Questions for Prof. Friedman", H1),
    bullets([
      "<b>The &lsquo;not fixating&rsquo; flag.</b> It appears on ~29% of condition-2 (free-eye) trials &mdash; more "
      "than in condition 1 &mdash; and those trials&rsquo; hand movements look normal. Was there an eye tracker, "
      "and is that check meaningful in condition 2? We are currently keeping these trials.",
      "<b>Segmentation zero-time.</b> Do you agree the correct reference is the target&rsquo;s motion onset "
      "(the go-signal), and that predicting movement + reaction time addresses the resampling concern?",
      "<b>Late-arrival cutoff.</b> We drop trials arriving &gt;1 s after the target&rsquo;s window as "
      "disengagement (there is no clean gap in the data). Is 1 s reasonable?",
      "<b>&lsquo;Too early&rsquo; trials</b> (moved before the go-signal) &mdash; we exclude these as invalid; please confirm.",
      "<b>Sub-movement benchmark.</b> Could we get access to your sub-movement decomposition pipeline to "
      "benchmark generative fidelity against it?",
    ])]

# ── 14. References ──
story += [Paragraph("14. References", H1),
    bullets([
      "Kingma, D. P., &amp; Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.",
      "Brenner, E., &amp; Smeets, J. B. (2018). Continuously updating one&rsquo;s predictions underlies "
      "successful interception. J. Neurophysiology, 120(6), 3257&ndash;3274.",
      "Higgins, I., et al. (2017). beta-VAE: Learning basic visual concepts with a constrained variational "
      "framework. ICLR. (structured-latent / disentanglement reference for Section 11.)",
      "Sohn, K., Lee, H., &amp; Yan, X. (2015). Learning structured output representation using deep "
      "conditional generative models. NeurIPS. (conditional VAE.)",
    ])]

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=0.7*inch, bottomMargin=0.7*inch,
                        leftMargin=0.8*inch, rightMargin=0.8*inch,
                        title="Interception Movements — Preliminary Results")
doc.build(story)
print("saved", OUT)
