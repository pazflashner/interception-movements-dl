"""Build the 8-page course-submission PDF.

Run:  python production/build_submission_report.py

WHY THIS EXISTS
    The 31-page laboratory draft is the source of record. The course brief asks
    for a PDF of at most 8 pages covering every proposal item in greater detail,
    plus lessons learned, benchmarking and potential extensions. This script
    produces that submission from the same numbers, keeping the draft's
    one-section-and-one-table-per-prediction-task layout.

HOW TO EDIT
    All prose lives in the SECTIONS list below as plain Python strings. Edit a
    string, re-run, get a new PDF. Tables are literal row lists taken from the
    laboratory draft; edit them in place. Figures are referenced by filename
    from production/figures_extracted/ (regenerate with the snippet in
    FIGURE_NOTES if that folder is missing).

    Keep it at 8 pages: the script prints the page count and warns if over.
    Trim prose first, then figures, then table rows.

EVERY NUMBER HERE COMES FROM THE LABORATORY DRAFT OR ITS SOURCE CSVs.
Do not edit a value without changing it in the study output as well.
"""
from __future__ import annotations

import pathlib
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.article_layout import Article, WIDTH


def cols(*fractions: float) -> list[float]:
    """Column widths in proportion, scaled to fill the text block exactly."""
    total = sum(fractions)
    return [WIDTH * f / total for f in fractions]

FIGURES = ROOT / "production" / "figures_extracted"
OUTPUT = ROOT / "production" / "8_pages_draft.pdf"
SOURCE_DRAFT = ROOT / "production" / "Interception_Movements_Methods_Reconstruction_Review.pdf"

FIGURE_NOTES = """If production/figures_extracted/ is missing, recreate it with:
    import pymupdf, pathlib
    d = pymupdf.open('production/Interception_Movements_Methods_Reconstruction_Review.pdf')
    out = pathlib.Path('production/figures_extracted'); out.mkdir(exist_ok=True)
    for i, pg in enumerate(d, 1):
        for j, img in enumerate(pg.get_images(full=True)):
            info = d.extract_image(img[0])
            (out / f'p{i:02d}_{j}.{info["ext"]}').write_bytes(info['image'])
"""

TITLE = "Compact Representations of Human Interception Movements"
SUBTITLE = (
    "Seman Libbiss and Paz Flashner<br/>"
    "0368-3538 Workshop on Deep Learning, Tel Aviv University<br/>"
    "Research supervision: Prof. Jason Friedman | Course advisor: Moni Shahar"
)

# ── Tables, transcribed from the laboratory draft ────────────────────────────
# One table per prediction task, as agreed with the course advisor.

T_RECONSTRUCTION = [
    ["Model", "n=3 MSE", "n=8 MSE"],
    ["Spline + PCA", "0.097255", "0.022772"],
    ["VAE", "0.148299", "0.033487"],
    ["Conditional AE", "0.169661", "0.025968"],
    ["CVAE", "0.176981", "0.032481"],
    ["Condition-only Ridge", "1.233795", "1.233795"],
]

T_GENERATION = [
    ["Model", "n", "Mean KS", "Energy", "MMD squared"],
    ["Spline + PCA", "3", "0.325126", "0.977433", "0.181467"],
    ["CVAE", "3", "0.277795", "0.745431", "0.130935"],
    ["Conditional AE", "3", "0.287578", "0.801489", "0.141410"],
    ["VAE", "3", "0.284587", "0.747566", "0.130298"],
    ["Spline + PCA", "8", "0.254794", "0.734715", "0.115291"],
    ["CVAE", "8", "0.219404", "0.430451", "0.067493"],
    ["Conditional AE", "8", "0.258636", "0.620200", "0.111100"],
    ["VAE", "8", "0.215100", "0.411000", "0.063900"],
]

T_FINGERPRINT = [
    ["Model", "Fingerprint center", "Mean KS", "Energy", "MMD squared"],
    ["CVAE n=3", "Own participant", "0.278", "0.745", "0.131"],
    ["CVAE n=3", "Training-population mean", "0.311", "1.016", "0.175"],
    ["CVAE n=3", "Other test participants", "0.338", "1.273", "0.220"],
    ["CVAE n=8", "Own participant", "0.219", "0.430", "0.067"],
    ["CVAE n=8", "Training-population mean", "0.290", "0.919", "0.149"],
    ["CVAE n=8", "Other test participants", "0.317", "1.196", "0.199"],
]

T_TIMING_ORIGINAL = [
    ["Model", "n", "Initiation MAE (ms)", "Movement MAE (ms)"],
    ["Spline + PCA", "3", "56.16", "51.16"],
    ["CVAE", "3", "37.83", "43.45"],
    ["Conditional AE", "3", "37.75", "44.09"],
    ["VAE", "3", "37.92", "42.91"],
    ["Spline + PCA", "8", "55.58", "50.21"],
    ["CVAE", "8", "34.84", "44.16"],
    ["Conditional AE", "8", "39.87", "42.67"],
    ["VAE", "8", "35.98", "42.21"],
]

T_TIMING_MATCHED = [
    ["Head", "n", "Duration", "CVAE MAE", "Spline MAE", "BH-adjusted p"],
    ["Common MLP", "3", "initiation", "41.43", "45.75", "0.0076"],
    ["Common MLP", "3", "movement", "43.89", "46.12", "0.1872"],
    ["Common MLP", "8", "initiation", "42.72", "47.87", "0.0423"],
    ["Common MLP", "8", "movement", "44.84", "44.81", "0.8314"],
]

T_IDENTIFICATION = [
    ["Model", "n=3 balanced accuracy", "n=8 balanced accuracy"],
    ["Spline + PCA", "36.4%", "48.6%"],
    ["CVAE", "40.9%", "50.7%"],
    ["Conditional AE", "39.0%", "46.5%"],
    ["VAE", "40.0%", "50.8%"],
]

T_PROBES = [
    ["Query-summary target", "CVAE n=3", "VAE n=8", "Direct context", "Dev. constant"],
    ["Initiation time (mean) [ms]", "14.778", "17.173", "9.275", "30.381"],
    ["Initiation time (sd) [ms]", "13.717", "14.602", "10.928", "14.028"],
    ["Movement time (mean) [ms]", "36.444", "35.394", "11.049", "48.762"],
    ["Peak speed (mean) [units/s]", "7.744", "8.760", "1.734", "11.876"],
    ["[EDIT] remaining 10 targets", "-", "-", "-", "-"],
]

T_KMEANS = [
    ["Input", "ARI", "Normalized mutual information", "Permutation p"],
    ["trajectory", "0.0518", "0.1944", "0.00498"],
    ["kinematic features", "0.0703", "0.2197", "0.00498"],
]

T_CONDITION = [
    ["Model", "n", "Correct conditions", "Decoder shuffled", "Correct better"],
    ["CVAE", "3", "0.176981", "0.180578", "25/28"],
    ["CVAE", "8", "0.032481", "0.035105", "28/28"],
    ["Conditional AE", "3", "0.169661", "0.171582", "26/28"],
]

T_COMPONENTS = [
    ["Discrepancy", "CVAE n=3", "CVAE n=8", "VAE n=8", "Reference 95th pct."],
    ["Fit error KS", "0.339", "0.308", "0.328", "0.282"],
    ["First duration KS", "0.455", "0.409", "0.406", "0.283"],
    ["First amplitude KS", "0.377", "0.351", "0.362", "0.283"],
    ["Count total variation", "0.214", "0.232", "[EDIT]", "0.146"],
]



FOOTNOTE = (
    "<super>1</super> Σ<sub>train</sub> = Cov<sub>train</sub>(μ<sub>i</sub> − "
    "μ<sub>s(i)</sub>) + diag(E<sub>train</sub>[σ<sub>i</sub><super>2</super>]) "
    "+ 10<super>−6</super>I"
)

_footnote_page = {}


class FootnoteMarker(Flowable):
    """Zero-height marker recording which page the footnote belongs on."""

    def __init__(self):
        super().__init__()
        self.width = self.height = 0

    def draw(self):
        _footnote_page["sigma"] = self.canv.getPageNumber()


def _render(article, footnote_page=None):
    """Build the PDF, drawing FOOTNOTE at the foot of *footnote_page*."""
    style = ParagraphStyle("footnote", fontName="Times-Roman", fontSize=8,
                           leading=9.6, textColor=colors.black)

    def page_furniture(c, doc):
        c.saveState()
        c.setFont("Times-Roman", 9)
        c.drawCentredString(105 * mm, 10 * mm, str(doc.page))
        if footnote_page is not None and doc.page == footnote_page:
            c.setLineWidth(0.4)
            c.line(22 * mm, 19 * mm, 72 * mm, 19 * mm)
            para = Paragraph(FOOTNOTE, style)
            _, h = para.wrap(166 * mm, 20 * mm)
            para.drawOn(c, 22 * mm, 19 * mm - h - 1.5 * mm)
        c.restoreState()

    SimpleDocTemplate(str(article.output), pagesize=(210 * mm, 297 * mm),
                      leftMargin=22 * mm, rightMargin=22 * mm,
                      topMargin=20 * mm, bottomMargin=22 * mm,
                      title=article.title,
                      author="Seman Libbiss and Paz Flashner",
                      subject="Interception movement modeling").build(
        list(article.story), onFirstPage=page_furniture, onLaterPages=page_furniture)


REFERENCES = [
    "<a name=\"ref1\"/>[1] Brenner E, Smeets JBJ. Continuously updating one's predictions underlies "
    'successful interception. <i>J Neurophysiol.</i> 2018;120:3257-3274. '
    '<link href="https://doi.org/10.1152/jn.00517.2018" color="#1A4C8B">doi</link>',
    '<a name="ref2"/>[2] Kingma DP, Welling M. Auto-Encoding Variational Bayes. <i>ICLR</i>, 2014. '
    '<link href="https://arxiv.org/abs/1312.6114" color="#1A4C8B">arXiv:1312.6114</link>',
    '<a name="ref3"/>[3] Sohn K, Lee H, Yan X. Learning Structured Output Representation using Deep '
    'Conditional Generative Models. <i>NeurIPS.</i> 2015;28.',
    '<a name="ref4"/>[4] Benjamini Y, Hochberg Y. Controlling the false discovery rate. <i>J R Stat Soc B</i> 1995;57. '
    '<link href="https://doi.org/10.1111/j.2517-6161.1995.tb02031.x" color="#1A4C8B">doi</link>',
    '<a name="ref5"/>[5] Gretton A, Borgwardt KM, Rasch MJ, Scholkopf B, Smola A. A Kernel Two-Sample '
    'Test. <i>JMLR</i> 2012;13. '
    '<link href="https://www.jmlr.org/papers/v13/gretton12a.html" color="#1A4C8B">pdf</link>',
    '<a name="ref6"/>[6] Szekely GJ, Rizzo ML. Energy statistics based on distances. <i>J Stat Plan Inference</i> 2013;143. '
    '<link href="https://doi.org/10.1016/j.jspi.2013.03.018" color="#1A4C8B">doi</link>',
    '<a name="ref7"/>[7] Flash T, Hogan N. The coordination of arm movements. <i>J Neurosci</i> 1985;5. '
    '<link href="https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985" color="#1A4C8B">doi</link>',
    '<a name="ref8"/>[8] Friedman J. <i>submovements.</i> '
    '<link href="https://github.com/JasonFriedman/submovements" color="#1A4C8B">github</link>, '
    'rev. 9c2f40c; adapted, GPL-3.0.',
    '<a name="ref9"/>[9] Rohrer B, Hogan N. Avoiding spurious submovement decompositions II. <i>Biol Cybern</i> 2006;94. '
    '<link href="https://doi.org/10.1007/s00422-006-0055-y" color="#1A4C8B">doi</link>',
    '<a name="ref10"/>[10] Bengio Y, Grandvalet Y. No unbiased estimator of the variance of K-fold cross-validation. <i>JMLR</i> 2004;5. '
    '<link href="https://www.jmlr.org/papers/v5/grandvalet04a.html" color="#1A4C8B">pdf</link>',
]


def _compose() -> Article:
    """Build a FRESH story. Reportlab flowables are single-use, so each render
    pass needs its own objects; reusing them raises LayoutError."""
    a = Article(OUTPUT, TITLE, SUBTITLE)

    # ── 1 ────────────────────────────────────────────────────────────────────
    a.h("1 Introduction")
    a.p(
        "Interception movements are made constantly, yet motor control lacks robust models of their "
        "detailed kinematics. Existing models of interception continuously update predictions using feedforward or feedback strategies <link href='#ref1' color='#1A4C8B'>[1]</link>. "
        "However, they lack a compact parametrization for stable individual differences (motor signatures). "
        "Because perception and movement are stochastic, the modelling target is a distribution of "
        "movement features rather than one deterministic trajectory."
    )
    a.p(
        "We asked whether a participant's trials can be compressed into a small continuous latent code "
        "that regenerates the distribution of that participant's remaining movements. We evaluate five "
        "model families on identical participant-held-out folds, report each prediction task and its "
        "metric separately, and add a control in which only the supplied fingerprint varies. The "
        "contribution is a matched benchmark with an explicit account of what the representation "
        "fails to do."
    )

    # ── 2 ────────────────────────────────────────────────────────────────────
    a.h("2 Data and task")
    a.p(
        "Participants slid a finger across a tabletop to control a visual cursor on a display, aiming to reach a target square as a circle moved horizontally into it, requiring rapid interception movements under one second. Finger position was recorded in 3D at 240 Hz with a Polhemus Liberty tracker; target trajectories were sampled at 60 Hz. Only condition 2, free eye movements, was analyzed. Of 4,763 available recordings, four missing-arrival timeouts and 27 trials arriving more than 1 s after the target window were excluded, leaving 4,732 trials from 28 participants. "
    )
    a.p(
            "One event definition is critical. Marker 5 records target appearance, after which the target holds still for a randomized 0.20–0.50 s foreperiod before moving. Behavioral zero-time is target motion onset, recovered from the target trajectory rather than the marker; anchoring to the marker would inject the randomized foreperiod into every initiation-time measurement. The window ends at the last tracker sample, serving as an arrival proxy that precedes the recorded arrival event by an average of 26.2 ms. "
        )

    # ── 3 ────────────────────────────────────────────────────────────────────
    a.h("3 Methods")
    a.h("3.1 Representation and models", True)
    a.p(
        "Positions are fourth-order zero-phase Butterworth filtered at 10 Hz, translated to the initial "
        "window position and resampled to 100 phase points. The final models use lateral x and forward "
        "y. Resampling removes physical duration, so encoders see shape only; movement time and "
        "initiation time are withheld from every encoder and predicted separately."
    )
    a.p(
        "The CVAE encodes 100 × 2 positions and five condition values through two 256-unit ReLU layers to a diagonal Gaussian posterior <link href='#ref2' color='#1A4C8B'>[2,3]</link>; its decoder maps a latent draw and conditions to trajectory shape and two log-duration outputs (initiation and movement times). "
    )
    a.p(
            "Spline + PCA is the capacity-matched comparator: a cubic B-spline with five fixed interior knots gives 18 coefficients, compressed by PCA fitted on training participants only to the same latent size, with a separate Ridge regression predicting log-durations. "
        )
    a.p(
                "Two ablations isolate design choices: a conditional autoencoder without the stochastic bottleneck, and an unconditional VAE with conditions zeroed. A condition-only Ridge model provides the floor. The sweep covered n ∈ {2, 3, 4, 8}; the planned n=16 was not evaluated. "
            )
    a.h("3.2 Participant-held-out protocol", True)
    a.p(
        "Four fixed folds use 17 training, 4 validation and 7 test participants; every participant is tested once. For each participant fold, neural models are trained across seeds 42, 43, and 44 to evaluate different weight initializations and optimization randomness under identical data splits. " 
        "Each test participant's trials are split into disjoint context and query halves, stratified by start category and side. The fingerprint is the mean encoder posterior over context trials. Around this fingerprint, 120 trajectories are generated using a shared training covariance Σ<sub>train</sub><super>1</super>, which combines within-subject trial dispersion and mean encoder uncertainty. These trajectories are conditioned on task parameters drawn from query trials and compared against the query feature distributions. Query shapes and timings never enter the fingerprint. "
        "Paired two-sided Wilcoxon tests are conducted using one seed-averaged value per participant with Pratt zero handling and Benjamini-Hochberg correction <link href='#ref4' color='#1A4C8B'>[4]</link> within each declared family."
    )
    a.story.append(FootnoteMarker())

    # ── 4 ────────────────────────────────────────────────────────────────────
    a.h("4 Task 1: trajectory reconstruction")
    a.p(
        "Reconstruction measures how well an observed trial is reproduced from its individual latent code, using posterior means and participant-balanced MSE (averaged across trials and seeds per subject). "
    )
    a.table(T_RECONSTRUCTION,
            "<b>Table 1.</b> Participant-balanced reconstruction error. Lower is better. The "
            "condition-only Ridge has no latent dimension; its result is repeated for reference.",
            cols(55, 35, 35))
    a.figure(FIGURES / "p08_0.png",
             "<b>Figure 1.</b> Recorded inputs and decoded outputs at neural seed 42; blue is VAE at "
             "n=3 and conditional AE at n=8. All panels share axis limits.",
             68 * mm)
    a.p(
        "Spline + PCA achieved the lowest error at both capacities. This is expected: PCA is the "
        "optimal linear projection under squared error, while the variational models trade "
        "reconstruction accuracy for a samplable latent space. Among neural models VAE was best at n=3 "
        "and conditional AE at n=8, demonstrating that conditioning alone does not explain neural performance rankings. "
    )

    # ── 5 ────────────────────────────────────────────────────────────────────
    a.h("5 Task 2: generating participant feature distributions")
    a.p(
        "The generation task evaluates 120 sampled trajectories per participant against their held-out query trials across 11 kinematic features. Evaluations use the mean two-sample Kolmogorov–Smirnov (KS) statistic and two multivariate distance metrics: squared MMD <link href='#ref5' color='#1A4C8B'>[5]</link> and Energy distance <link href='#ref6' color='#1A4C8B'>[6]</link>. Crucially, none of these measures represents a point-to-point reconstruction error. "
    )
    a.table(T_GENERATION,
            "<b>Table 2.</b> Cohort-averaged generated feature discrepancies. Lower is better. All "
            "three metrics compare distributions.",
            cols(40, 12, 28, 28, 28))
    a.figure(FIGURES / "p11_0.png",
             "<b>Figure 2.</b> Generated-feature discrepancies at three and eight dimensions. Lines "
             "connect tested settings only and do not estimate untested capacities (draft Figure 7).",
             42 * mm)
    a.p(
        "At n=8, the unconditional VAE attains the lowest cohort-average values on all three metrics and serves as the dashboard default for that reason. Crucially, this reflects a numerical preference on the evaluated cohort rather than uniformly significant superiority: of seven CVAE-versus-VAE contrasts at n=8, exactly one reaches statistical significance (favoring the unconditional model). At n=3, no CVAE improvement over the condition-only Ridge survives FDR correction; at n=8, all three metric improvements survive. "
    )
    a.figure(FIGURES / "p12_0.png",
             "<b>Figure 3.</b> Recorded query and generated trajectories at n=3 for subject01; the "
             "first 30 available samples in fixed ordering (draft Figure 8).", 44 * mm)
    a.figure(FIGURES / "p13_0.png",
             "<b>Figure 4.</b> Recorded query and generated feature distributions for the same "
             "participant at n=8 (draft Figure 10).", 68 * mm)

    # ── 6 ────────────────────────────────────────────────────────────────────
    a.h("6 Control: contribution of the personal fingerprint")
    a.p(
        "A same-decoder control isolates the effect of the individual fingerprint. Across all 24 CVAE checkpoints (n ∈ {3, 8}, 4 folds, 3 seeds), the decoder, 120 condition draws, latent noise, shared covariance Σ<sub>train</sub>, and distance reference are held fixed; only the supplied center is replaced — either by the overall population mean or by another participant's fingerprint."
 )
    a.table(T_FINGERPRINT,
            "<b>Table 3.</b> Fingerprint replacement with the generative procedure held fixed. Lower "
            "is better. The other-person row averages donor-specific distances.",
            cols(24, 50, 22, 22, 26))
    a.p(
        "The correct fingerprint improves all three distance metrics against both controls across both dimensions (12 contrasts, all p<sub>BH</sub> &lt; 0.0004). At n=3, the mean KS statistic improves over the population center for 22 of 28 participants and over other-person centers for 27 of 28; at n=8, both counts reach 27 of 28. Because only a single input was varied, this performance gain is directly attributable to the fingerprint. Importantly, this establishes input-sensitivity rather than proving that the fingerprint captures a stable cognitive strategy across sessions."
    )

    # ── 7 ────────────────────────────────────────────────────────────────────
    a.h("7 Task 3: recovering physical timing from shape")
    a.p(
         "Physical durations are withheld from every encoder, so predicting them from normalized shape and task conditions constitutes a separate task, evaluated using participant-balanced MAE. "    )
    a.table(T_TIMING_ORIGINAL,
            "<b>Table 4.</b> Original timing-head errors. Lower is better.",
            cols(40, 12, 44, 44))
    a.p(
        "The unadjusted baseline comparison was strictly unmatched: the CVAE timing head used a nonlinear MLP, whereas the Spline + PCA baseline relied on a linear Ridge regression, meaning the contrast evaluated predictor capacity alongside representation quality. Fitting an identical two-layer MLP to frozen latent codes from both representations yields the matched comparison in Table 5. "
    )
    a.table(T_TIMING_MATCHED,
            "<b>Table 5.</b> Matched-head timing sensitivities in ms. BH adjustment covers the eight "
            "paired participant comparisons in the source draft.",
            cols(30, 10, 26, 24, 24, 30))
    a.figure(FIGURES / "p17_0.png",
             "<b>Figure 5.</b> Timing errors under original, common-MLP and validation-calibrated "
             "heads. Points are separate fitted procedures (draft Figure 12).", 46 * mm)
    a.p(
        "Under a matched head, the n=8 initiation time advantage decreases from 20.7 ms to 5.1 ms yet remains statistically detectable, whereas the movement-time advantage disappears. Crucially, this remains an imperfect geometry contrast, as CVAE latent codes were trained with joint timing supervision, whereas Spline + PCA codes were not. " )

    # ── 8 ────────────────────────────────────────────────────────────────────
    a.h("8 Task 4: participant identification after enrollment")
    a.p(
        "Each test participant is enrolled using their fingerprint (the mean latent code over context trials); query trials are then assigned to the nearest of the seven enrolled participant centers via Euclidean distance. "    )
    a.table(T_IDENTIFICATION,
            "<b>Table 6.</b> Identification among seven enrolled test participants, chance 14.3%. "
            "Descriptive closed-set results, not identification of unseen identities without enrolment.",
            cols(45, 40, 40))
    a.p(
        "Classification accuracy exceeds chance level (1/7 ≈ 14.3%) across all settings. At n=8, both variational models reach approximately 51% accuracy, indicating partial separation of participant-specific motor features accompanied by considerable inter-individual overlap."
    )

    # ── 9 ────────────────────────────────────────────────────────────────────
    a.h("9 Task 5: predicting behavioural summaries from the code")
    a.p(
        "Validation-tuned Ridge probes predict the means and standard deviations of seven query behaviors directly from context fingerprints. Two baseline controls are evaluated: a population-mean baseline and a direct average of the participant's own context trials. "
    )
    a.table(T_PROBES,
            "<b>Table 7.</b> MAE for query-summary targets. Timing in ms; other errors use each "
            "feature's units. [EDIT: paste the remaining 10 rows from draft Table 10, or state that "
            "the full 14-row table is in the laboratory draft.]",
            cols(52, 22, 22, 26, 26))
    a.p(
        "Direct trial measurement outperforms the latent probe on all 14 targets across both latent capacities (n ∈ {3, 8}). A nested leave-one-participant-out test evaluated whether the fingerprint adds residual predictive power beyond direct context averaging: across all 28 evaluation cells, none remained significant after correction (mean ΔR<super>2</super> = −0.017). Crucially, a positive control verified statistical sensitivity at the same sample size (N=28). Thus, given a direct sample of a participant's trials, the latent fingerprint provides no additional information regarding these summary statistics."
    )

    # ── 10 ───────────────────────────────────────────────────────────────────
    a.h("10 Feasibility baseline: K-means clustering")
    a.p(
        "An initial feasibility baseline evaluated whether trials are inherently separable by participant. Unsupervised K-means clustering (K=28) was fitted on time-normalized trajectories and normalized kinematic features without labels, which were used ex post facto solely to evaluate cluster alignment." )
    a.p(
        "The Adjusted Rand Index (ARI) reached 0.0518 on spatial trajectories and 0.0703 on kinematic features, with corresponding Normalized Mutual Information (NMI) values of 0.1944 and 0.2197. Both metrics significantly exceeded all 200 shuffled-label permutations (empirical p = 0.00498), yet absolute ARI values remained near zero. This confirms that participant identity is statistically detectable yet weak—sufficient to motivate a non-linear generative model, but insufficient to form clean spatial clusters. Notably, this descriptive analysis includes all trials, unlike the held-out evaluation framework used above." )

    # ── 11 ───────────────────────────────────────────────────────────────────
    a.h("11 Control: does explicit task conditioning matter?")
    a.p(
        "Shuffling decoder conditions within each test participant—while holding original latent codes fixed and without retraining—tests whether conditional models actively utilize their conditional inputs." )
    a.table(T_CONDITION,
            "<b>Table 9.</b> Frozen condition-shuffle diagnostic; the last column counts participants "
            "with lower MSE under correct decoder conditions.",
            cols(30, 10, 34, 32, 32))
    a.p(
        "Supplying correct conditions yields superior generation accuracy for 25 of 28 participants at n=3 and 28 of 28 at n=8, confirming that conditional information is genuinely utilized by the decoder. An architectural audit further revealed that zeroing a conditional network's condition weights recovers the unconditional model exactly. Because the conditional model class subsumes the unconditional solution, incorporating extra inputs does not guarantee empirical superiority under finite training iterations and validation selection." )

    # ── 12 ───────────────────────────────────────────────────────────────────
    a.h("12 Secondary analysis: minimum-jerk components")
    a.p(
        "Following the proposal, generated movements were benchmarked against Prof. Friedman's submovement pipeline <link href='#ref8' color='#1A4C8B'>[8]</link> by fitting one to four overlapping minimum-jerk pulses <link href='#ref7' color='#1A4C8B'>[7]</link>; this is an adapted implementation, not the published scattershot algorithm <link href='#ref9' color='#1A4C8B'>[9]</link>. Recorded and generated trajectories were originally decomposed with different inputs and optimizer budgets, so part of the measured mismatch was procedural; one identical operator across 4,056 fits reduced the selected-count total variation from 0.353/0.355 to 0.214/0.232 at n=3/8, so the earlier values overstated the mismatch."
    )
    a.table(T_COMPONENTS,
            "<b>Table 10.</b> Participant-balanced matched component discrepancies. The reference is "
            "the 95th percentile of an empirical resampling distribution, not a formal threshold.",
            cols(50, 24, 24, 24, 30))
    a.p(
        "All discrepancies remain above the sampling reference and all 4,056 selected fits converged, but no n=3 versus n=8 contrast survives correction, and the selected count is far more sensitive to the timing crop and filter cutoff than to optimizer budget. The analysis is dominated by fitting assumptions and remains secondary."
    )

    # ── 14 ───────────────────────────────────────────────────────────────────
    a.h("13 Limitations")
    a.p(
        "While the study cohort (N=28) provided sufficient statistical sensitivity for within-subject controls, relying on a single dataset limits generalizability across broader demographic or clinical populations. Furthermore, cross-validation folds share training participants; consequently, evaluation errors are not strictly independent, rendering p-values somewhat optimistic <link href='#ref10' color='#1A4C8B'>[10]</link>. Additionally, context and query trials originate from the same recording session, meaning latent fingerprints may partly reflect session-specific factors—such as posture or tracker calibration—rather than temporally stable individual traits." 
    )

    # ── 15 ───────────────────────────────────────────────────────────────────
    a.h("14 Lessons learned")
    a.p(
        "<b>Audit evaluation code as strictly as model code: </b> Our evaluation metric was initially normalized using evaluation data rather than a fixed training baseline. This caused zero-variance features to divide by near-zero values, creating an adaptive yardstick. Resolving this required recomputing all 124 evaluation checkpoints. ")
    a.p(
        "<b>Ensure fair comparisons before drawing conclusions: </b> Apparent model advantages in Sections 7 and 12 actually stemmed from differences in classifier capacity. Standardizing evaluation heads revealed that original metrics reflected procedural asymmetry rather than true representation quality. ")
    
    a.p(
        "<b>Record explicit execution settings over implicit defaults: </b> Key experimental constraints—such as withholding physical timing from both encoders—were managed only in execution scripts rather than class definitions. All effective parameters are now explicitly logged per run." 
    )

    # ── 16 ───────────────────────────────────────────────────────────────────
    a.h("15 Potential extensions")
    a.p(
        "The most principled extension is an architecture matching the research question. The current model has no person-level variable: the fingerprint is a post-hoc average of trial codes, and nothing in training encourages a participant's trials to cluster. A hierarchical model with an explicit subject latent and a per-trial residual would express that story directly."
    )
    a.p(
        "Three cheaper extensions follow. The timing weight of 20 was never tuned yet sits on the "
        "reconstruction-versus-timing trade-off that decides the comparison. Retraining with timing "
        "supervision disabled would isolate representation geometry from the training objective. And "
        "the encoder flattens 100x2 into 200 numbers, discarding temporal ordering; a convolutional "
        "encoder would exploit the smoothness the spline basis exploits by construction."
    )
    a.p(
        "For any future recording the decisive experiment is cross-session enrollment: splitting context "
        "and query trials across different sessions would test whether the fingerprint is a stable "
        "personal trait, the one claim this dataset cannot support."
    )

    # ── 17 ───────────────────────────────────────────────────────────────────
    a.h("16 Deliverable: interactive dashboard")
    a.p(
        "The deliverable is a Streamlit dashboard supporting CVAE and spline + PCA at n=2,3,4,8 and unconditional VAE and conditional AE at n=3,8. It exposes generation from a chosen fingerprint, held-out validation examples, benchmark tables, latent associations and diagnostics, and runs without the private recordings. Live generation uses fold-0 seed-42 references while benchmark tables aggregate repeated runs, as labelled in the interface. A planned second mode accepting new CSV recordings was not implemented."
    )

    a.h("17 Conclusion")
    a.p(
        "A participant's context-derived fingerprint measurably improves conditional trajectory generation under strict control where all other components remain fixed. However, a capacity-matched Spline + PCA baseline reconstructs trajectories with higher fidelity, an unconditional model ablation matches or exceeds conditional generation accuracy, and direct averaging of a participant's context trials predicts behavioral summaries more effectively. Ultimately, the learned latent representation functions as a useful conditioning signal for generative modeling, rather than a validated low-dimensional account of individual interception strategy." )

    a.h("References")
    # One paragraph with line breaks: avoids 9pt of inter-paragraph spacing per entry.
    a.styles.add(ParagraphStyle("refs", fontName="Times-Roman", fontSize=8, leading=9.4,
                                textColor=colors.black, spaceAfter=0))
    a.p("<br/>".join(REFERENCES), "refs")

    return a


def build() -> Path:
    _render(_compose())                                  # pass 1: locate the footnote page
    _render(_compose(), _footnote_page.get("sigma"))     # pass 2: draw it there
    return OUTPUT


def ensure_figures() -> bool:
    """Extract the laboratory draft's figures if they are not already present.

    They are derived artifacts, so they are git-ignored rather than committed;
    this keeps a fresh clone runnable without carrying 4 MB of duplicated PNGs.
    """
    if FIGURES.exists() and any(FIGURES.glob("*.png")):
        return True
    if not SOURCE_DRAFT.exists():
        return False
    FIGURES.mkdir(parents=True, exist_ok=True)
    try:
        import pymupdf
        doc = pymupdf.open(SOURCE_DRAFT)
        for page_number, page in enumerate(doc, 1):
            for index, image in enumerate(page.get_images(full=True)):
                info = doc.extract_image(image[0])
                (FIGURES / f"p{page_number:02d}_{index}.{info['ext']}").write_bytes(info["image"])
    except ImportError:
        from pypdf import PdfReader                      # already required for the page check
        for page_number, page in enumerate(PdfReader(SOURCE_DRAFT).pages, 1):
            for index, img in enumerate(page.images):
                suffix = pathlib.Path(img.name).suffix or ".png"
                (FIGURES / f"p{page_number:02d}_{index}{suffix}").write_bytes(img.data)
    print(f"extracted figures from {SOURCE_DRAFT.name} into {FIGURES.name}/")
    return True


def main() -> None:
    ensure_figures()
    if not FIGURES.exists():
        print(f"warning: {FIGURES} missing; figures will be skipped.\n{FIGURE_NOTES}")
    path = build()
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(path).pages)
        flag = "OK" if pages <= 8 else f"OVER BY {pages - 8} - trim prose, then figures, then rows"
        print(f"{path}\n  {pages} pages [{flag}]")
    except Exception:
        print(path)


if __name__ == "__main__":
    main()
